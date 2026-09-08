"""Exact repricing provider: one engine call per distinct state (spec 7.3).

This is the reference procedure the approximate providers are gated
against, and the mode the conformance oracle runs in: for a given day and
path it makes exactly the calls ``ReplayBacktestEngine`` would make, with
the same aged product and the same environment, so the two agree to the
last bit.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import ImpliedBasisYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from ..config import GateConfig
from .base import DayStates, GateReport, StateKey, float_key
from .cache import StateCache

#: Engine settings that can change a price for a given state.
_ENGINE_FIELDS = (
    "pricing_engine_type", "method", "vol_model_mc_method", "pde_params", "mc_params",
    "quad_params", "vol_source", "surface_vol_mode", "vol_model", "vol_model_solver",
    "vol_model_engine_options", "dividend_source", "futures_curve_extrapolation",
    "futures_curve_min_tenor_days",
)

#: Attributes a product carries that are lifecycle STATE, not contract terms.
_STATE_PREFIXES = ("_otc_",)


def engine_fingerprint(
    engine_config: Any, delta_bump_size: Optional[float], gamma_bump_size: Optional[float]
) -> str:
    """Identity of everything that can change a price for a given state."""
    h = hashlib.blake2b(digest_size=16)
    for name in _ENGINE_FIELDS:
        h.update(repr(getattr(engine_config, name, None)).encode())
        h.update(b"\x1f")
    h.update(repr((delta_bump_size, gamma_bump_size)).encode())
    return h.hexdigest()


def _canonical(obj: Any, seen: set, depth: int = 0) -> str:
    """A structural, address-free rendering of a product's terms.

    ``repr`` is not enough: ``SnowballOption.__repr__`` is a rounded
    summary that omits the observation schedule, so two contracts that
    differ only in their observation days would print alike.  This walks
    the object graph instead -- dataclass fields, ``__dict__`` entries,
    sequences, arrays, enums, dates -- and skips lifecycle state.
    """
    if depth > 12:
        raise ValidationError("product terms nest too deeply to fingerprint")
    if obj is None or isinstance(obj, (bool, int, str, bytes)):
        return repr(obj)
    if isinstance(obj, float):
        return repr(float(obj))
    if isinstance(obj, enum.Enum):
        return f"{type(obj).__name__}.{obj.name}"
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return f"{type(obj).__name__}:{obj.isoformat()}"
    if isinstance(obj, np.ndarray):
        return f"ndarray:{obj.dtype}:{obj.shape}:{np.ascontiguousarray(obj).tobytes().hex()}"
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return repr(obj.item())
    if isinstance(obj, (list, tuple)):
        inner = ",".join(_canonical(x, seen, depth + 1) for x in obj)
        return f"{type(obj).__name__}[{inner}]"
    if isinstance(obj, (set, frozenset)):
        inner = ",".join(sorted(_canonical(x, seen, depth + 1) for x in obj))
        return f"{type(obj).__name__}{{{inner}}}"
    if isinstance(obj, dict):
        inner = ",".join(
            f"{_canonical(k, seen, depth + 1)}:{_canonical(v, seen, depth + 1)}"
            for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))
        )
        return f"dict{{{inner}}}"
    if callable(obj) and not hasattr(obj, "__dict__"):
        return f"callable:{getattr(obj, '__qualname__', type(obj).__name__)}"
    marker = id(obj)
    if marker in seen:
        return f"<cycle:{type(obj).__name__}>"
    seen.add(marker)
    if dataclasses.is_dataclass(obj):
        items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    elif hasattr(obj, "__dict__"):
        items = sorted(vars(obj).items())
    else:
        return f"{type(obj).__name__}:{obj!r}"
    inner = ",".join(
        f"{name}={_canonical(value, seen, depth + 1)}"
        for name, value in items
        if not name.startswith(_STATE_PREFIXES) and not callable(value)
    )
    seen.discard(marker)
    return f"{type(obj).__name__}({inner})"


def product_fingerprint(product: Any) -> str:
    """Identity of the contract as priced (terms only, no lifecycle state)."""
    return hashlib.blake2b(_canonical(product, set()).encode(), digest_size=16).hexdigest()


class RepricingPricer:
    """Prices each distinct ``(spot, vol, env, knocked_in)`` state once, exactly.

    One engine per product, shared by every path, exactly as
    ``ReplayBacktestEngine`` shares one engine across a path's days.  The
    PDE solver keeps per-instance caches (critical points by spot, banded
    factorisations by step, grid layouts), so a shared engine was checked
    rather than assumed: sixty single-path oracle comparisons, discrete and
    continuous knock-in, matched the replay to the bit through one engine.
    One engine per path was tried too and bought nothing but ~0.5 MB a path.
    """

    def __init__(
        self,
        product: Any,
        *,
        engine_config: Any,
        start_date: pd.Timestamp,
        underlying: str,
        cache: StateCache,
        delta_bump_size: Optional[float] = None,
        gamma_bump_size: Optional[float] = None,
    ) -> None:
        self.product = product
        self.engine_config = engine_config
        self.start_date = pd.Timestamp(start_date).normalize()
        self.underlying = underlying
        self.cache = cache
        self.delta_bump_size = delta_bump_size
        self.gamma_bump_size = gamma_bump_size
        self._engine_fp = engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size)
        self._product_fp = product_fingerprint(product)
        # Built from the ORIGINAL contract, exactly as
        # ``ReplayBacktestEngine.__init__`` does (the factory reads only the
        # product's type); the aged copy is passed to every call.
        self._engine = create_pricing_engine(
            product, engine_config,
            delta_bump_size=delta_bump_size, gamma_bump_size=gamma_bump_size,
        )
        self._aged: Dict[Tuple[pd.Timestamp, bool], Any] = {}
        self._engine_calls = 0

    # -- aging ---------------------------------------------------------

    def reset_aging_memo(self) -> None:
        """Drop the per-day product memo (tests and long runs)."""
        self._aged.clear()

    def aged_product(self, date: pd.Timestamp, *, knocked_in: bool) -> Any:
        """The tracker's time-decayed copy for ``date`` and the KI flag.

        One copy serves every path: ``product_for_pricing`` reads no market
        data, it only subtracts elapsed time and shifts the observation
        schedule (``barrier_config.time_shift`` writes ``valuation_date``
        on the environment it is handed and reads nothing from it), so the
        environment below is a throwaway.
        """
        key = (pd.Timestamp(date).normalize(), bool(knocked_in))
        product = self._aged.get(key)
        if product is None:
            tracker = AutocallableLifecycleTracker(
                product=self.product, quantity=1.0, has_lifecycle=True,
                lifecycle=AutocallableLifecycleState(knocked_in=bool(knocked_in)),
                start_date=self.start_date,
            )
            product = tracker.product_for_pricing(key[0], self._aging_env(key[0]))
            self._aged[key] = product
        return product

    def _aging_env(self, date: pd.Timestamp) -> PricingEnvironment:
        """A throwaway environment for schedule shifting; never used to price."""
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=1.0, asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=0.2),
            rate_curve=FlatRateCurve(rate=0.0),
            valuation_date=pd.Timestamp(date).to_pydatetime(),
        )

    # -- pricing -------------------------------------------------------

    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pv, delta, gamma)`` per alive state, per unit product."""
        date = pd.Timestamp(states.date).normalize()
        m = len(states)
        pv = np.empty(m)
        delta = np.empty(m)
        gamma = np.empty(m)
        if m == 0:
            return pv, delta, gamma
        keys = self.state_keys(states)
        hit, c_pv, c_delta, c_gamma = self.cache.get_many(keys)
        pv[hit], delta[hit], gamma[hit] = c_pv[hit], c_delta[hit], c_gamma[hit]

        # One engine call per DISTINCT missing key, scattered back to every
        # state that shares it.
        pending: Dict[StateKey, List[int]] = {}
        for n in np.flatnonzero(~hit):
            pending.setdefault(keys[int(n)], []).append(int(n))
        for key, positions in pending.items():
            values = self._price_one(states, positions[0], date, key)
            self.cache.put(key, *values)
            for pos in positions:
                pv[pos], delta[pos], gamma[pos] = values
        return pv, delta, gamma

    def _price_one(
        self, states: DayStates, n: int, date: pd.Timestamp, key: StateKey
    ) -> Tuple[float, float, float]:
        product = self.aged_product(date, knocked_in=bool(states.knocked_in[n]))
        engine = self._engine
        self._seed_engine(engine, key)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=float(states.spot[n]), asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=float(states.vol[n])),
            rate_curve=FlatRateCurve(rate=float(states.rate[n])),
            div_yield=states.div_yield[n],
            basis_yield=ImpliedBasisYield(float(states.basis_yield[n])),
            valuation_date=pd.Timestamp(date).to_pydatetime(),
        )
        try:
            price = float(engine.price(product, env))
            greeks = engine.calculate_greeks(product, env)
        except Exception as exc:  # fail closed with the state in the message
            raise ValidationError(
                f"pricing failed on day {states.day_index} at spot={states.spot[n]!r}, "
                f"vol={states.vol[n]!r}, knocked_in={bool(states.knocked_in[n])}: {exc}"
            ) from exc
        self._engine_calls += 1
        # The replay marks with price() and takes greeks from
        # calculate_greeks(); the two are separate calls there, so they are
        # separate calls here.
        return price, float(greeks["delta"]), float(greeks["gamma"])

    def _seed_engine(self, engine: Any, key: StateKey) -> None:
        """Give an MC engine this state's own seed (spec 7.3).

        A recomputed state then matches its cached value bit for bit, and
        the seed is a function of the state, not of iteration order.
        """
        params = getattr(engine, "params", None)
        if params is not None and hasattr(params, "random_seed"):
            params.random_seed = key.seed()

    def state_keys(self, states: DayStates) -> List[StateKey]:
        """One ``StateKey`` per state, in order."""
        spot_keys = float_key(states.spot)
        vol_keys = float_key(states.vol)
        return [
            StateKey(
                product_fingerprint=self._product_fp, day_index=int(states.day_index),
                knocked_in=bool(states.knocked_in[n]), spot_key=int(spot_keys[n]),
                vol_key=int(vol_keys[n]), env_key=int(states.env_key[n]),
                engine_fingerprint=self._engine_fp,
            )
            for n in range(len(states))
        ]

    # -- reporting -----------------------------------------------------

    def verify(self, states: DayStates, gate: GateConfig) -> GateReport:
        """Exact mode prices the states themselves: the gap is zero by construction."""
        return GateReport(mode="exact", sampled=0, max_pv_gap_bp=0.0,
                          max_delta_gap_hands=0.0, passed=True)

    def fingerprint(self) -> str:
        return self._engine_fp

    @property
    def product_fingerprint(self) -> str:
        """Identity of the contract this pricer prices."""
        return self._product_fp

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": "repricing", "mode": "exact", "engine_calls": self._engine_calls,
            "cache": self.cache.stats().as_dict(),
        }
