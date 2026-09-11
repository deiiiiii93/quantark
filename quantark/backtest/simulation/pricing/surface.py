"""The PDE life surface: one solve per bucket, read along every path (spec 7.2).

A snowball's remaining life on day ``d`` is the start-date solve's slab at
the node for ``d``.  Solving once with a node on every simulation day and
interpolating in log-spot replaces one engine call per path per day with
one solve per (vol, q, rate) bucket; the gate measures what that costs.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import SignedDividendYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from ..config import CacheConfig, GateConfig
from .base import (DayStates, GateReport, GateScale, StressCase, bucket_centre, bucket_key,
                   float_key, state_row)
from .cache import StateCache
from .repricing import RepricingPricer, engine_fingerprint


def _node_greeks(x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delta and gamma on every node of every column.

    The solver's non-uniform three-point stencil (``_calculate_delta_gamma``)
    at each interior node, converted at that node's spot; the two edge
    nodes copy their neighbours.  ``v`` is ``(n_x, n_t)``.
    """
    h_m = (x[1:-1] - x[:-2])[:, None]
    h_p = (x[2:] - x[1:-1])[:, None]
    h_sum = h_m + h_p
    v_m, v_0, v_p = v[:-2], v[1:-1], v[2:]
    dv = -h_p / (h_m * h_sum) * v_m + (h_p - h_m) / (h_m * h_p) * v_0 + h_m / (h_p * h_sum) * v_p
    d2v = 2.0 * (v_m / (h_m * h_sum) - v_0 / (h_m * h_p) + v_p / (h_p * h_sum))
    s = np.exp(x[1:-1])[:, None]
    delta = np.empty_like(v)
    gamma = np.empty_like(v)
    delta[1:-1] = dv / s
    gamma[1:-1] = (d2v - dv) / (s * s)
    delta[0], delta[-1] = delta[1], delta[-2]
    gamma[0], gamma[-1] = gamma[1], gamma[-2]
    return delta, gamma


@dataclass(frozen=True)
class LifeSurface:
    """Both branch slabs of one solve with their node greeks.

    ``v0`` / ``v1`` are the solver's BRANCH columns: each is the value after
    that node's diffusion and before its event transforms, which is what a
    path alive (or knocked in) on that day is worth.  Reading the projected
    column instead would blend the two branches within a cell of a barrier
    and differentiate across the observation's value jump.
    """

    t: np.ndarray
    x: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    d0: np.ndarray
    g0: np.ndarray
    d1: np.ndarray
    g1: np.ndarray
    step_of: Dict[float, int]

    @property
    def nbytes(self) -> int:
        arrays = (self.t, self.x, self.v0, self.v1, self.d0, self.g0, self.d1, self.g1)
        return int(sum(a.nbytes for a in arrays))

    def node(self, elapsed: float) -> int:
        """The column for an elapsed time: exact float first, then ``is_close``.

        The maturity itself is the terminal column (the payoff after the
        terminal transforms); a time past it has no value on this surface.
        """
        hit = self.step_of.get(elapsed)
        if hit is not None:
            return hit
        for key, k in self.step_of.items():
            if is_close(key, elapsed):
                return k
        tau = float(self.t[-1])
        if is_close(elapsed, tau):
            return int(self.t.size - 1)
        raise ValidationError(f"no surface node at elapsed time {elapsed!r} (maturity {tau!r})")

    def readout(
        self, x_spot: np.ndarray, k: int, knocked_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PV, delta and gamma at log-spots ``x_spot`` from column ``k``, per KI flag."""
        x_spot = np.asarray(x_spot, dtype=float)
        outside = (x_spot < self.x[0]) | (x_spot > self.x[-1])
        if outside.any():
            bad = x_spot[outside]
            raise ValidationError(
                f"spot {np.exp(bad[0])!r} lies outside the life-surface grid "
                f"[{np.exp(self.x[0])!r}, {np.exp(self.x[-1])!r}]; widen the grid or use repricing"
            )
        ki = np.asarray(knocked_in, dtype=bool)
        v_alive, d_alive, g_alive = self.v0[:, k], self.d0[:, k], self.g0[:, k]
        pv = np.where(ki, np.interp(x_spot, self.x, self.v1[:, k]), np.interp(x_spot, self.x, v_alive))
        delta = np.where(ki, np.interp(x_spot, self.x, self.d1[:, k]), np.interp(x_spot, self.x, d_alive))
        gamma = np.where(ki, np.interp(x_spot, self.x, self.g1[:, k]), np.interp(x_spot, self.x, g_alive))
        return pv, delta, gamma

    def select(self, columns: Sequence[int]) -> "LifeSurface":
        """This surface restricted to ``columns`` (sorted, unique), ``step_of`` remapped.

        Column data is copied, so a readout of a kept column is bit-identical
        to the readout of the original.  Keep the terminal column: ``node``
        maps the maturity to it.
        """
        keep = np.array(sorted({int(k) for k in columns}), dtype=np.int64)
        if keep.size == 0 or keep[0] < 0 or keep[-1] >= self.t.size:
            raise ValidationError(f"columns must index the surface's {self.t.size} time nodes")
        position = {int(k): n for n, k in enumerate(keep)}
        return LifeSurface(
            t=np.array(self.t[keep]), x=self.x,
            v0=np.ascontiguousarray(self.v0[:, keep]), v1=np.ascontiguousarray(self.v1[:, keep]),
            d0=np.ascontiguousarray(self.d0[:, keep]), g0=np.ascontiguousarray(self.g0[:, keep]),
            d1=np.ascontiguousarray(self.d1[:, keep]), g1=np.ascontiguousarray(self.g1[:, keep]),
            step_of={t: position[k] for t, k in self.step_of.items() if k in position},
        )


class SurfaceCache:
    """LRU of ``LifeSurface`` objects under a byte budget."""

    def __init__(self, max_bytes: int) -> None:
        if int(max_bytes) <= 0:
            raise ValidationError("surface_cache_bytes must be positive")
        self.max_bytes = int(max_bytes)
        self._store: "OrderedDict[tuple, LifeSurface]" = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._evictions = 0

    def get(self, key: tuple) -> Optional[LifeSurface]:
        hit = self._store.get(key)
        if hit is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return hit

    def put(self, key: tuple, surface: LifeSurface) -> None:
        if surface.nbytes > self.max_bytes:
            raise ValidationError(
                f"one life surface needs {surface.nbytes} bytes, more than surface_cache_bytes={self.max_bytes}"
            )
        self._store[key] = surface
        self._bytes += surface.nbytes
        while self._bytes > self.max_bytes:
            _, old = self._store.popitem(last=False)
            self._bytes -= old.nbytes
            self._evictions += 1

    def stats(self) -> Dict[str, Any]:
        return {"hits": self._hits, "misses": self._misses, "evictions": self._evictions,
                "entries": len(self._store), "bytes_used": self._bytes}


def _replaced(row: DayStates, **changes: Any) -> DayStates:
    """``DayStates`` with some fields swapped.

    ``NamedTuple._replace`` cannot be used here: ``DayStates`` overrides
    ``__len__`` to count states, and ``_make`` checks that same ``len``
    against the field count, so ``_replace`` fails with a field-count
    error that has nothing to do with the caller.
    """
    fields = {name: getattr(row, name) for name in DayStates._fields}
    fields.update(changes)
    return DayStates(**fields)


class LifeSurfacePricer:
    """One PDE solve per (vol, q, rate) bucket; readout on every day and path.

    The rate is bucketed with ``q_step`` -- both are yields, and the
    surface needs one flat rate for its whole life.
    """

    def __init__(
        self, product: Any, *, engine_config: Any, start_date: pd.Timestamp, dates: pd.DatetimeIndex,
        underlying: str, vol_step: float, q_step: float, surface_cache_bytes: int, gate: GateConfig,
        delta_bump_size: Optional[float] = None, gamma_bump_size: Optional[float] = None,
        cache: Optional[StateCache] = None, compact: bool = True,
    ) -> None:
        self.product = product
        self.compact = bool(compact)
        self.engine_config = engine_config
        self.start_date = pd.Timestamp(start_date).normalize()
        self.dates = pd.DatetimeIndex(dates)
        self.underlying = underlying
        self.vol_step = float(vol_step)
        self.q_step = float(q_step)
        self.gate = gate
        self._engine = create_pricing_engine(product, engine_config, delta_bump_size=delta_bump_size,
                                             gamma_bump_size=gamma_bump_size)
        if not hasattr(self._engine, "solve_life_surface"):
            raise ValidationError("the life_surface provider needs a PDE engine")
        # The exact side of the gate, and the aged-product memo the engine's
        # helpers use.  Given the run's cache, its exact repricings are keyed
        # like an exact-mode run's and survive on the disk tier.
        self._exact = RepricingPricer(
            product, engine_config=engine_config, start_date=start_date, underlying=underlying,
            cache=cache if cache is not None else StateCache(CacheConfig(memory_bytes=8_000_000)),
            delta_bump_size=delta_bump_size, gamma_bump_size=gamma_bump_size,
        )
        self._surfaces = SurfaceCache(surface_cache_bytes)
        self._fp = hashlib.blake2b(
            (engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size)
             + repr(("life_surface", self.vol_step, self.q_step))).encode(), digest_size=16,
        ).hexdigest()
        self._maturity = float(product.maturity)
        self._extra_times = tuple(
            t for t in (self.elapsed(d) for d in range(1, len(self.dates)))
            if 0.0 < t < self._maturity
        )
        self._solves = 0
        self._readouts = 0
        self._reservoir: List[DayStates] = []
        self._seen = 0
        seed = int.from_bytes(hashlib.blake2b((self._exact.product_fingerprint + self._fp).encode(),
                                              digest_size=4).digest(), "big")
        self._rng = np.random.default_rng(seed)

    # -- identity -------------------------------------------------------

    @property
    def mode(self) -> str:
        return "life_surface"

    @property
    def product_fingerprint(self) -> str:
        return self._exact.product_fingerprint

    def fingerprint(self) -> str:
        return self._fp

    def aged_product(self, date, *, knocked_in: bool):
        """The tracker's time-decayed copy (the exact pricer's memo)."""
        return self._exact.aged_product(date, knocked_in=knocked_in)

    def elapsed(self, d: int) -> float:
        """The tracker's elapsed-time convention: ``(date - start).days / 365``."""
        return max(0.0, (self.dates[d].normalize() - self.start_date).days / 365.0)

    # -- surfaces -------------------------------------------------------

    def _surface(self, vol: float, q: float, rate: float) -> LifeSurface:
        key = (float(vol), float(q), float(rate))
        hit = self._surfaces.get(key)
        if hit is not None:
            return hit
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=float(self.product.initial_price), asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=float(vol)), rate_curve=FlatRateCurve(rate=float(rate)),
            div_yield=SignedDividendYield(float(q)), valuation_date=self.start_date.to_pydatetime(),
        )
        try:
            sol = self._engine.solve_life_surface(self.product, env, extra_times=self._extra_times)
        except Exception as exc:
            raise ValidationError(
                f"life-surface solve failed at vol={vol!r}, q={q!r}, rate={rate!r}: {exc}"
            ) from exc
        self._solves += 1
        # Every column is a branch column now (LifeSurfaceSolution), column 0
        # included, so the valuation-date readout needs no separate vector:
        # ``sol.t0_readout`` is ``sol.v0[:, 0]`` to the bit whenever it exists.
        d0, g0 = _node_greeks(sol.x, sol.v0)
        d1, g1 = _node_greeks(sol.x, sol.v1)
        surface = LifeSurface(t=sol.t, x=sol.x, v0=sol.v0, v1=sol.v1, d0=d0, g0=g0, d1=d1, g1=g1,
                              step_of=sol.step_of)
        if self.compact:
            surface = surface.select(self._read_columns(surface))
        self._surfaces.put(key, surface)
        return surface

    def _read_columns(self, surface: LifeSurface) -> List[int]:
        """Column 0, the terminal column, and the node of every calendar day up to maturity."""
        columns = {0, int(surface.t.size - 1)}
        tau = float(surface.t[-1])
        for d in range(1, len(self.dates)):
            elapsed = self.elapsed(d)
            if elapsed > tau and not is_close(elapsed, tau):
                break
            columns.add(surface.node(elapsed))
        return sorted(columns)

    # -- pricing --------------------------------------------------------

    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pv, delta, gamma)`` per alive state, per unit product, off the bucket's surface."""
        m = len(states)
        pv, delta, gamma = np.empty(m), np.empty(m), np.empty(m)
        if m == 0:
            return pv, delta, gamma
        elapsed = self.elapsed(int(states.day_index))
        vol_c = bucket_centre(states.vol, self.vol_step)
        q_c = bucket_centre(states.q_T, self.q_step)
        rate_c = bucket_centre(states.rate, self.q_step)
        buckets = np.column_stack([bucket_key(states.vol, self.vol_step), bucket_key(states.q_T, self.q_step),
                                   bucket_key(states.rate, self.q_step)])
        x_spot = np.log(np.asarray(states.spot, dtype=float))
        for first in np.unique(buckets, axis=0, return_index=True)[1]:
            i = int(first)
            same = np.all(buckets == buckets[i], axis=1)
            surface = self._surface(float(vol_c[i]), float(q_c[i]), float(rate_c[i]))
            k = 0 if elapsed == 0.0 else surface.node(elapsed)
            pv[same], delta[same], gamma[same] = surface.readout(x_spot[same], k, states.knocked_in[same])
        self._readouts += m
        self._sample(states)
        return pv, delta, gamma

    def _sample(self, states: DayStates) -> None:
        """Algorithm R over every state read off a surface."""
        capacity = int(self.gate.sample_states)
        if capacity <= 0:
            return
        for n in range(len(states)):
            self._seen += 1
            if len(self._reservoir) < capacity:
                self._reservoir.append(state_row(states, n))
                continue
            slot = int(self._rng.integers(0, self._seen))
            if slot < capacity:
                self._reservoir[slot] = state_row(states, n)

    def sample_visited(self) -> List[DayStates]:
        return list(self._reservoir)

    def verify(self, samples: Sequence[DayStates], gate: GateConfig, scale: GateScale) -> GateReport:
        """Each sample exactly through the engine at its own spot, vol and dividend object, then the readout."""
        worst_pv = worst_delta = 0.0
        count = 0
        for row in samples:
            if len(row) != 1:
                raise ValidationError("verify takes one-row DayStates (see state_row)")
            pv_e, delta_e, _ = self._exact.price_exact(row)
            pv_s, delta_s, _ = self._readout_only(row)
            worst_pv = max(worst_pv, abs(float(pv_s[0]) - pv_e) / float(scale.unit_notional) * 1e4)
            worst_delta = max(worst_delta, abs(float(delta_s[0]) - delta_e) * float(scale.hands_per_unit_delta))
            count += 1
        return GateReport(
            mode="life_surface", sampled=count, max_pv_gap_bp=worst_pv, max_delta_gap_hands=worst_delta,
            passed=worst_pv <= float(gate.pv_tolerance_bp) and worst_delta <= float(gate.delta_tolerance_hands),
        )

    # -- the designed barrier stress set --------------------------------

    def barrier_stress_cases(self, *, vol: float, rate: float, q: float) -> List[StressCase]:
        """Designed states either side of the knock-in, on its observation dates.

        The reservoir samples where a run happened to go.  This says where
        a risk tool has to be right whether the run went there or not.

        Both lifecycle states appear on both sides of the barrier, and
        that is also how the observation convention is made explicit:
        below the barrier on an observation date, the alive state is the
        one BEFORE the observation is applied and the knocked-in state is
        the one after it.  Above the barrier the pair is simply a note
        that knocked in earlier in life against one that never did.

        Parameters sit just inside the far edge of their buckets, which is
        the worst bucketing error the surface can be asked to carry; a
        state at a bucket centre would report none at all.  The dividend
        is a flat object matching ``q_T`` exactly, so the exact side and
        the surface agree on what q means and the bucketing term measures
        bucketing rather than a term-carry mismatch.  Term carry is what
        the visited-state sampler covers.
        """
        if not self.gate.barrier_offsets or not getattr(self.product, "has_ki_barrier", False):
            return []
        days = self._stress_days()
        if not days:
            return []
        vol_s = self._bucket_corner(vol, self.vol_step)
        q_s = self._bucket_corner(q, self.q_step)
        rate_s = self._bucket_corner(rate, self.q_step)
        # A spot the surface's grid does not reach is one the run cannot
        # reach either: the readout refuses it. Designing such a state
        # would gate nothing, so the set leaves it out.
        grid = self._surface(float(bucket_centre(np.array([vol_s]), self.vol_step)[0]),
                             float(bucket_centre(np.array([q_s]), self.q_step)[0]),
                             float(bucket_centre(np.array([rate_s]), self.q_step)[0]))
        low, high = float(grid.x[0]), float(grid.x[-1])
        cases: List[StressCase] = []
        for day in days:
            level = self._ki_barrier_on(day)
            if level is None:
                continue
            for offset in self.gate.barrier_offsets:
                spot = level * (1.0 + float(offset))
                if not (low <= float(np.log(spot)) <= high):
                    continue
                for knocked_in in (False, True):
                    # The run steps the lifecycle BEFORE it prices, so on a
                    # knock-in observation date a path below the barrier is
                    # already marked. Alive-and-below is not a state the run
                    # can produce, and gating on one the run cannot reach
                    # measures nothing. Between observations it is reachable,
                    # and there both lifecycle states are designed.
                    if offset < 0.0 and not knocked_in:
                        continue
                    if offset < 0.0:
                        when = "knocked-in"
                    else:
                        when = "knocked-in" if knocked_in else "alive"
                    label = (f"d{day}:{'above' if offset > 0.0 else 'below'}"
                             f":{abs(offset):.2%}:{when}")
                    cases.append(StressCase(
                        label=label, distance=float(offset),
                        states=self._stress_row(day, spot, knocked_in, vol_s, rate_s, q_s)))
        return cases

    def _bucket_corner(self, value: float, step: float) -> float:
        """Just inside the far edge of the bucket ``value`` falls in."""
        if float(step) <= 0.0:
            return float(value)
        centre = float(bucket_centre(np.array([float(value)]), float(step))[0])
        return centre + float(step) * (0.5 - 1e-9)

    def _ki_observation_times(self) -> List[float]:
        """The knock-in observation times, or every day when it is continuous."""
        config = self.product.barrier_config
        if getattr(config, "ki_continuous", False):
            return [self.elapsed(d) for d in range(1, len(self.dates))]
        return [float(t) for t in (config.ki_observation_dates or ())]

    def _stress_days(self) -> List[int]:
        """The knock-in observation dates this run's calendar actually reaches."""
        wanted = int(self.gate.barrier_dates)
        if wanted <= 0:
            return []
        times = self._ki_observation_times()
        days = [d for d in range(len(self.dates))
                if 0.0 < self.elapsed(d) < self._maturity
                and any(is_close(self.elapsed(d), t) for t in times)]
        if len(days) <= wanted:
            return days
        # Evenly spaced through the observation schedule, endpoints kept:
        # the first date is the least decayed and the last the most, and
        # a barrier is usually met somewhere between them.
        picks = np.linspace(0.0, len(days) - 1.0, wanted)
        return sorted({days[int(round(float(i)))] for i in picks})

    def _ki_barrier_on(self, day: int) -> Optional[float]:
        """The knock-in level in force at ``day``, stepping schedules included."""
        config = self.product.barrier_config
        elapsed = self.elapsed(day)
        times = [float(t) for t in (config.ki_observation_dates or ())]
        for idx, t in enumerate(times):
            if is_close(elapsed, t):
                level = self.product.get_ki_barrier_at(idx)
                if level is not None:
                    return float(level)
                break
        level = getattr(config, "ki_barrier", None)
        return None if level is None else float(level)

    def _stress_row(self, day: int, spot: float, knocked_in: bool,
                    vol: float, rate: float, q: float) -> DayStates:
        """One designed state, shaped like any other row the pricers take."""
        return DayStates(
            day_index=int(day), date=self.dates[int(day)],
            path_index=np.zeros(1, dtype=np.int64), spot=np.array([float(spot)]),
            vol=np.array([float(vol)]), rate=np.array([float(rate)]), q_T=np.array([float(q)]),
            div_yield=(SignedDividendYield(float(q)),), basis_yield=np.zeros(1),
            env_key=float_key(np.array([float(rate)])),
            knocked_in=np.array([bool(knocked_in)]),
        )

    def verify_stress(self, cases: Sequence[StressCase], gate: GateConfig, scale: GateScale,
                      *, attribute: bool = False) -> GateReport:
        """Every designed state against a fresh exact solve, worst case reported.

        ``attribute`` splits each gap three ways at the cost of two more
        exact solves per state, which is why it is off by default: the
        gate only needs the total, and the split is a diagnostic.
        """
        worst_pv = worst_delta = 0.0
        rows: List[Dict[str, Any]] = []
        for case in cases:
            row = case.states
            pv_e, delta_e, _ = self._exact.price_exact(row)
            pv_s, delta_s, _ = self._readout_only(row)
            pv_bp = (float(pv_s[0]) - pv_e) / float(scale.unit_notional) * 1e4
            delta_hands = (float(delta_s[0]) - delta_e) * float(scale.hands_per_unit_delta)
            worst_pv = max(worst_pv, abs(pv_bp))
            worst_delta = max(worst_delta, abs(delta_hands))
            # Absolute hands alone misread a state whose delta is an order
            # of magnitude above a typical one, which is what happens beside
            # a barrier near expiry, so the relative gap travels with it.
            entry: Dict[str, Any] = {
                "label": case.label, "distance": float(case.distance),
                "day_index": int(row.day_index), "spot": float(row.spot[0]),
                "knocked_in": bool(row.knocked_in[0]),
                "pv_gap_bp": pv_bp, "delta_gap_hands": delta_hands,
                "exact_delta_hands": delta_e * float(scale.hands_per_unit_delta),
                "delta_gap_rel": (delta_hands / abs(delta_e * float(scale.hands_per_unit_delta))
                                  if delta_e != 0.0 else float("nan")),
            }
            if attribute:
                entry.update(self._attribute(row, pv_bp, delta_hands, scale))
            rows.append(entry)
        return GateReport(
            mode="life_surface_barrier", sampled=len(rows), max_pv_gap_bp=worst_pv,
            max_delta_gap_hands=worst_delta, attribution=tuple(rows),
            passed=(worst_pv <= float(gate.pv_tolerance_bp)
                    and worst_delta <= float(gate.delta_tolerance_hands)),
        )

    def _attribute(self, row: DayStates, pv_bp: float, delta_hands: float,
                   scale: GateScale) -> Dict[str, float]:
        """Split one gap into bucketing, propagation and readout.

        bucketing    the surface was solved at the bucket centre, not at
                     this state's own vol, q and rate
        propagation  the surface's own column against a fresh solve, read
                     at a grid NODE so no spot interpolation is in the way
        readout      whatever moving off that node adds

        The three are defined so they sum to the total exactly, and the
        readout term is the remainder rather than a fourth measurement.
        """
        centred = self._centred_row(row)
        pv_c, delta_c, _ = self._exact.price_exact(centred)
        pv_e, delta_e, _ = self._exact.price_exact(row)
        bucket_pv = (pv_c - pv_e) / float(scale.unit_notional) * 1e4
        bucket_delta = (delta_c - delta_e) * float(scale.hands_per_unit_delta)

        node = self._node_row(centred)
        pv_ne, delta_ne, _ = self._exact.price_exact(node)
        pv_ns, delta_ns, _ = self._readout_only(node)
        prop_pv = (float(pv_ns[0]) - pv_ne) / float(scale.unit_notional) * 1e4
        prop_delta = (float(delta_ns[0]) - delta_ne) * float(scale.hands_per_unit_delta)
        return {
            "bucket_pv_bp": bucket_pv, "bucket_delta_hands": bucket_delta,
            "propagation_pv_bp": prop_pv, "propagation_delta_hands": prop_delta,
            "readout_pv_bp": pv_bp - bucket_pv - prop_pv,
            "readout_delta_hands": delta_hands - bucket_delta - prop_delta,
        }

    def _centred_row(self, row: DayStates) -> DayStates:
        """The same state with its parameters moved to the bucket the surface solved."""
        vol = bucket_centre(row.vol, self.vol_step)
        q = bucket_centre(row.q_T, self.q_step)
        rate = bucket_centre(row.rate, self.q_step)
        return _replaced(row, vol=vol, q_T=q, rate=rate, env_key=float_key(rate),
                         div_yield=(SignedDividendYield(float(q[0])),))

    def _node_row(self, row: DayStates) -> DayStates:
        """The same state moved to the surface's nearest grid node in spot."""
        surface = self._surface(float(row.vol[0]), float(row.q_T[0]), float(row.rate[0]))
        node = int(np.argmin(np.abs(surface.x - float(np.log(float(row.spot[0]))))))
        return _replaced(row, spot=np.array([float(np.exp(surface.x[node]))]))

    def _readout_only(self, states: DayStates):
        """``price_day`` without feeding the reservoir (the gate must not sample itself)."""
        seen, reservoir = self._seen, list(self._reservoir)
        try:
            return self.price_day(states)
        finally:
            self._seen, self._reservoir = seen, reservoir

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": "life_surface", "mode": "life_surface", "solves": self._solves,
            "readouts": self._readouts, "engine_calls": self._exact.stats()["engine_calls"],
            "surface_cache": self._surfaces.stats(), "cache": self._exact.stats()["cache"],
        }
