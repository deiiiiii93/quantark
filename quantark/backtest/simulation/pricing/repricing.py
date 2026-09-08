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
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import ImpliedBasisYield, SignedDividendYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from ..config import GateConfig
from .base import DayStates, GateReport, StateKey, bucket_centre, bucket_key, float_key, row_keys
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
    engine_config: Any, delta_bump_size: Optional[float], gamma_bump_size: Optional[float],
    *, spot_step: Optional[float] = None, vol_step: Optional[float] = None, q_step: Optional[float] = None,
) -> str:
    """Identity of everything that can change a price for a given state, the mode included."""
    h = hashlib.blake2b(digest_size=16)
    for name in _ENGINE_FIELDS:
        h.update(repr(getattr(engine_config, name, None)).encode())
        h.update(b"\x1f")
    h.update(repr((delta_bump_size, gamma_bump_size)).encode())
    h.update(repr(("ladder" if spot_step is not None else "exact", spot_step, vol_step, q_step)).encode())
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
        spot_step: Optional[float] = None,
        vol_step: Optional[float] = None,
        q_step: Optional[float] = None,
    ) -> None:
        self.product = product
        self.engine_config = engine_config
        self.start_date = pd.Timestamp(start_date).normalize()
        self.underlying = underlying
        self.cache = cache
        self.delta_bump_size = delta_bump_size
        self.gamma_bump_size = gamma_bump_size
        self.spot_step = None if spot_step is None else float(spot_step)
        self.vol_step = vol_step
        self.q_step = q_step
        if self.spot_step is not None and self.spot_step <= 0.0:
            raise ValidationError("spot_step must be positive or None")
        self._x_ref = math.log(float(product.initial_price))
        self._engine_fp = engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size,
                                             spot_step=self.spot_step, vol_step=vol_step, q_step=q_step)
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

    @property
    def mode(self) -> str:
        """``exact`` or ``ladder``."""
        return "exact" if self.spot_step is None else "ladder"

    def ladder_nodes(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray]:
        """Lower node index and interpolation weight per state (ladder mode)."""
        if self.spot_step is None:
            raise ValidationError("ladder_nodes is only defined in ladder mode")
        u = (np.log(np.asarray(states.spot, dtype=float)) - self._x_ref) / self.spot_step
        j = np.floor(u).astype(np.int64)
        return j, u - j

    def node_spot(self, j: np.ndarray) -> np.ndarray:
        """``S_ref * exp(j * spot_step)``: the same node on every day and in every cell."""
        return np.exp(self._x_ref + np.asarray(j, dtype=float) * self.spot_step)

    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pv, delta, gamma)`` per alive state, per unit product."""
        if len(states) == 0:
            empty = np.empty(0)
            return empty, empty.copy(), empty.copy()
        if self.spot_step is None:
            return self._price_exact(states)
        return self._price_ladder(states)

    def _price_exact(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        date = pd.Timestamp(states.date).normalize()
        m = len(states)
        pv = np.empty(m)
        delta = np.empty(m)
        gamma = np.empty(m)
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
        self._sample(states)
        return pv, delta, gamma

    def _price_ladder(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Price the two bracketing nodes of every state, then interpolate in log-spot."""
        date = pd.Timestamp(states.date).normalize()
        keys, rows, node_j, w, vol_c, q_c = self._ladder_layout(states)
        hit, pv_n, delta_n, gamma_n = self.cache.get_many(keys)
        # A state exactly on a node (w == 0) takes that node's own value;
        # its upper node would be multiplied by zero, so it is not priced.
        needed = np.ones(len(keys), dtype=bool)
        needed[1::2] = w != 0.0
        pending: Dict[StateKey, List[int]] = {}
        for r in np.flatnonzero(~hit & needed):
            pending.setdefault(keys[int(r)], []).append(int(r))
        node_spot = self.node_spot(node_j)
        for key, positions in pending.items():
            r = positions[0]
            n = int(rows[r])
            values = self._price_env(
                date, knocked_in=bool(states.knocked_in[n]), key=key, spot=float(node_spot[r]),
                vol=float(vol_c[n]), rate=float(states.rate[n]),
                div_yield=SignedDividendYield(float(q_c[n])), basis_yield=None,
                label=f"ladder node {int(node_j[r])}",
            )
            self.cache.put(key, *values)
            for pos in positions:
                pv_n[pos], delta_n[pos], gamma_n[pos] = values
        self._sample(states)
        lo, hi = slice(0, None, 2), slice(1, None, 2)
        on_node = w == 0.0

        def blend(v: np.ndarray) -> np.ndarray:
            return np.where(on_node, v[lo], (1.0 - w) * v[lo] + w * v[hi])

        return blend(pv_n), blend(delta_n), blend(gamma_n)

    def _ladder_layout(self, states: DayStates):
        """The 2m node states a day needs: keys, owning state, node index, weight, bucket centres.

        Row ``2n`` is the lower node of state ``n`` and ``2n + 1`` the upper.
        The environment key is (rate, q centre): a node carries no basis,
        because nothing under ``asset/equity`` reads one.
        """
        j, w = self.ladder_nodes(states)
        vol_k, vol_c = bucket_key(states.vol, self.vol_step), bucket_centre(states.vol, self.vol_step)
        q_c = bucket_centre(states.q_T, self.q_step)
        env_key = row_keys(np.column_stack([states.rate, np.asarray(q_c, dtype=float)]))
        m = len(states)
        node_j = np.repeat(j, 2) + np.tile([0, 1], m)
        rows = np.repeat(np.arange(m), 2)
        keys = [
            StateKey(
                product_fingerprint=self._product_fp, day_index=int(states.day_index),
                knocked_in=bool(states.knocked_in[n]), spot_key=int(node_j[r]), vol_key=int(vol_k[n]),
                env_key=int(env_key[n]), engine_fingerprint=self._engine_fp,
            )
            for r, n in enumerate(rows)
        ]
        return keys, rows, node_j, w, vol_c, q_c

    def _ladder_keys(self, states: DayStates) -> List[StateKey]:
        return self._ladder_layout(states)[0]

    def _sample(self, states: DayStates) -> None:
        """Hook for the gate's reservoir; nothing to record until the gate lands."""
        return None

    def _price_one(
        self, states: DayStates, n: int, date: pd.Timestamp, key: StateKey
    ) -> Tuple[float, float, float]:
        return self._price_env(
            date, knocked_in=bool(states.knocked_in[n]), key=key, spot=float(states.spot[n]),
            vol=float(states.vol[n]), rate=float(states.rate[n]), div_yield=states.div_yield[n],
            basis_yield=ImpliedBasisYield(float(states.basis_yield[n])),
            label=f"day {states.day_index} path {int(states.path_index[n])}",
        )

    def _price_env(
        self, date: pd.Timestamp, *, knocked_in: bool, key: StateKey, spot: float, vol: float,
        rate: float, div_yield: Any, basis_yield: Any, label: str,
    ) -> Tuple[float, float, float]:
        """One engine call: mark with ``price``, greeks from ``calculate_greeks``.

        The replay makes the two calls separately, so they are separate
        calls here; both modes come through this one method.
        """
        product = self.aged_product(date, knocked_in=knocked_in)
        self._seed_engine(self._engine, key)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=spot, asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=vol), rate_curve=FlatRateCurve(rate=rate),
            div_yield=div_yield, basis_yield=basis_yield,
            valuation_date=pd.Timestamp(date).to_pydatetime(),
        )
        try:
            price = float(self._engine.price(product, env))
            greeks = self._engine.calculate_greeks(product, env)
        except Exception as exc:  # fail closed with the state in the message
            raise ValidationError(
                f"pricing failed at {label}: spot={spot!r}, vol={vol!r}, knocked_in={knocked_in}: {exc}"
            ) from exc
        self._engine_calls += 1
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
        """One ``StateKey`` per state in exact mode; two per state (lower, upper node) in ladder mode."""
        if self.spot_step is not None:
            return self._ladder_keys(states)
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
            "provider": "repricing", "mode": self.mode, "engine_calls": self._engine_calls,
            "cache": self.cache.stats().as_dict(),
        }
