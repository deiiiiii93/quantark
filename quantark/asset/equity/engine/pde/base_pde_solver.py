"""
Base PDE solver providing common infrastructure for finite difference pricing.

Implements the Crank-Nicolson scheme for solving the Black-Scholes PDE
backward in time, with support for Rannacher smoothing.
"""

from abc import abstractmethod
import functools
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Tuple, List, NamedTuple, Sequence
import numpy as np

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.settlement_support import (
    resolve_terminal_timing,
)
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementRequest,
    SettlementResolver,
)
from quantark.asset.equity.engine.pde.core.backends import BANDED_CACHE_MAX_ENTRIES
from quantark.asset.equity.engine.pde.core.coefficients import StepCoefficients
from quantark.asset.equity.engine.pde.core.components import (
    CentralDifference,
    delta_gamma,
    flat_exact,
    interpolate_price,
    scalar_stencil,
)
from quantark.asset.equity.engine.pde.core.identity import market_fingerprints
from quantark.asset.equity.engine.pde.core.samples import sample_bsm
from quantark.asset.equity.engine.pde.core.adapters.base import FamilyAdapter
from quantark.asset.equity.engine.pde.core.prepare import (
    Operation,
    OutputRequest,
    SessionArtefacts,
    bind_layout,
    prepare,
    run,
)
from quantark.asset.equity.engine.pde.grid import (
    EventSchedule,
    GridBinder,
    GridRequest,
    Layout,
    DiscreteEvent,
    MarketSnapshot,
    Nodes,
    TimeRefinement,
    apply_discrete_event,
    fold_decision,
    fold_nodal,
    validate_external_layout,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError, NumericalError
from quantark.util.numerical import is_close, safe_divide
from quantark.util.enum.option_enums import ExerciseType, ObservationType
from quantark.util.enum import ObservationAggregation
from quantark.util.enum.engine_enums import EngineType, EventProjectionMode



# Memoization of curve lookups and step coefficients, keyed on the VALUE
# fingerprints of every market input the memoized function reads (spec 2
# decisions 14, 18, 19). The market parameter objects are plain mutable
# dataclasses, so an identity-only key served stale values after an in-place
# mutation (``env.rate_curve.rate = 0.07``). A market object without a value
# fingerprint gets no memo: it is sampled fresh on every call.
# - Step coefficients: one value-keyed store; its key holds the exact grid
#   bytes, so a hit returns what a fresh build would, from any solve.
# - Discount factors: scoped per curve OBJECT (weakref-evicted) and keyed by
#   the curve's value inside that scope. The (t, T) inner keys are rounded,
#   first build wins, so the sharing domain must stay what it always was.
# Each memo is size-bounded so long-lived processes cannot grow without bound.
_ENV_DF_MEMO: dict = {}
_ENV_STEP_COEFF_MEMO: "OrderedDict[tuple, StepCoefficients]" = OrderedDict()
_DF_MEMO_MAX_ENTRIES = 200_000

#: The base family's treatment for ``event_projection=None`` (spec 3A §5.4). NODAL through stage 1
#: (bitwise); flip (c) makes it CELL_AVERAGE, the treatment of every other core route.
_BASE_FAMILY_DEFAULT_PROJECTION = EventProjectionMode.CELL_AVERAGE
_STEP_COEFF_MEMO_MAX_ENTRIES = 64


def _per_object_memo(market_obj, store: dict) -> Optional[dict]:
    """The memo dict of one market object (id key, weakref-evicted), or None if not weakref-able."""
    if market_obj is None:
        return None
    key = id(market_obj)
    entry = store.get(key)
    if entry is not None and entry[0]() is market_obj:
        return entry[1]
    try:
        ref = weakref.ref(market_obj, lambda _r, _k=key: store.pop(_k, None))
    except TypeError:
        return None
    memo: dict = {}
    store[key] = (ref, memo)
    return memo

#: Operational cache sizes (spec §4.3): implementation details, not knobs.
#: (BANDED_CACHE_MAX_ENTRIES lives with the factorization cache, core.backends.)
GRID_CACHE_MAX_ENTRIES = 128
#: Product cache tokens always use the former "standard" keying.
_CACHE_KEYING = "standard"


def _market_scoped(method):
    """Fingerprint the environment once for the duration of a solve.

    A fingerprint costs a sha256 per market object and the discount memo is
    read many times per step, so a solve computes them once and every memo
    lookup inside it reads them (``BasePDESolver._market_fps``). Nested
    scopes restore the outer one.
    """

    @functools.wraps(method)
    def scoped(self, product, pricing_env, *args, **kwargs):
        outer = getattr(self, "_market_fp", None)
        self._market_fp = (pricing_env, market_fingerprints(pricing_env), {})
        try:
            return method(self, product, pricing_env, *args, **kwargs)
        finally:
            self._market_fp = outer

    return scoped


class PDESolutionResult(NamedTuple):
    """
    Result from PDE solving containing solution and grid data.

    This type allows both price() and calculate_greeks() to share the
    common solving logic via _solve(), eliminating code duplication.

    Attributes:
        solution_vec: Solution values at t=0 (present time), AFTER any
            valuation-date event application (per-node exact triggers) —
            the right column for per-spot scenario curves.
        x_vec: Log-price grid points
        s_vec: Price grid points (for convenience)
        spot_log: Log of spot price for interpolation
        readout_vec: Smooth 0+ branch column for the actual-spot readout.
            A valuation-date event makes solution_vec discontinuous at its
            barrier; interpolating across that jump blends the branches, so
            price()/greeks read the pre-event branch and apply today's
            (deterministic) transitions pointwise at spot. None = use
            solution_vec (no valuation-date event).
        readout_override: Fully-resolved pointwise price at the actual spot
            (set when a valuation-date transition adds a cash amount at
            spot, e.g. a triggered phoenix coupon). None = interpolate.
    """

    solution_vec: np.ndarray
    x_vec: np.ndarray
    s_vec: np.ndarray
    spot_log: float
    readout_vec: Optional[np.ndarray] = None
    readout_override: Optional[float] = None
    #: the valuation instant's event, folded at each level of a projection (spec 3C decision 26)
    decision: Optional[DiscreteEvent] = None
    #: a spot-independent receivable added to each projected price (spec 3C decision 29)
    cash: float = 0.0


def _project_rows(result, spot_levels):
    """(price, delta, gamma) per level, interpolated from one solved surface (no decision, no cash)."""
    prices = np.asarray(result.solution_vec, dtype=float)
    deltas = np.gradient(prices, result.s_vec, edge_order=2)
    gammas = np.gradient(deltas, result.s_vec, edge_order=2)
    return [(float(np.interp(s, result.s_vec, prices)), float(np.interp(s, result.s_vec, deltas)),
             float(np.interp(s, result.s_vec, gammas))) for s in spot_levels]


class CompositeSolution(NamedTuple):
    """A knock-in's solved parts (spec 3C decision 26): each projected on its own grid, the rows composed level by
    level by the facade's rule, the knock-in's own decision deciding whether a level is a hit."""

    vanilla: PDESolutionResult
    leg: PDESolutionResult
    participation: Optional[float]
    constant: float
    decision: Optional[DiscreteEvent]

    def rows(self, spot_levels=None):
        levels = [float(s) for s in (self.vanilla.s_vec if spot_levels is None else spot_levels)]
        vanilla, leg = _project_rows(self.vanilla, levels), _project_rows(self.leg, levels)
        out = []
        for spot, (pv, dv, gv), (pl, dl, gl) in zip(levels, vanilla, leg):
            if self.participation is not None:
                pv, dv, gv = self.participation * pv, self.participation * dv, self.participation * gv
            if fold_decision(self.decision, spot, 0.0)[3]:          # a hit: participation x vanilla
                price, delta, gamma = pv, dv, gv
            else:
                price, delta, gamma = pv + self.constant - pl, dv - dl, gv - gl
            out.append({"spot": spot, "price": float(price), "delta": float(delta), "gamma": float(gamma),
                        "calculation_mode": "engine_grid"})
        return out


class LifeSurfaceSolution(NamedTuple):
    """Both value surfaces of one solve on every time node (life-surface readout).

    ``v0`` / ``v1`` are the alive and knocked-in BRANCH surfaces, shape
    ``(n_x, n_t)``; column ``k`` is the value at ``t[k]`` after that node's
    diffusion and BEFORE its event transforms.  That is what a pointwise
    readout needs: a discrete observation writes a value JUMP onto the grid
    (the surviving branch above the barrier, the knocked-in branch at or
    below it), so within a cell of the barrier the projected column is the
    value of neither branch, and differentiating it is O(J/h) wrong.  The
    march still propagates the projected values; only the readout differs.
    The terminal column is the payoff, which no event follows.

    ``step_of`` maps every event and extra time to its column.
    ``t0_readout`` is the smooth valuation-date column for the alive surface
    when the valuation date itself carries events (the same column
    ``calculate_greeks`` reads), which is now ``v0[:, 0]``; else ``None``.

    ``requested_steps`` / ``actual_steps`` are the time layout's own.
    """

    t: np.ndarray
    x: np.ndarray
    s: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    step_of: Dict[float, int]
    t0_readout: Optional[np.ndarray]
    knocked_in_at_valuation: bool
    requested_steps: int
    actual_steps: int


class PDESessionOutputs(NamedTuple):
    """One-solve session output bundle (spec sections 8/9.3 native seam).

    ``solution`` is populated only when the caller asked for grid outputs and
    the request did not short-circuit (expired / knocked-out). Event fields
    are populated only by autocallable solvers.
    """

    npv: float
    solution: Optional[PDESolutionResult]
    event_stats: object
    event_distribution: object


class BasePDESolver(BaseEngine):
    """
    Abstract base class for PDE-based option pricing.

    Solves the Black-Scholes PDE using finite difference methods in log-price space (x = ln(S)):
        dV/dt + (r-q-0.5*sigma^2)*dV/dx + 0.5*sigma^2*d2V/dx^2 - r*V = 0

    The PDE is solved backward in time from maturity to valuation date.

    NOTE: numerical Black engine — a smile surface is collapsed to a single
    strike-selected constant vol (sigma = get_vol(K, T)). Not smile-aware; for
    smile-consistent dynamics use LV / SLV / Heston Monte-Carlo or PDE.
    """

    engine_type = EngineType.PDE
    #: (env, market fingerprints, in-scope memos) while a solve's market scope is open
    _market_fp = None

    def __init__(self, params: Optional[PDEParams] = None):
        """Initialize the PDE solver with configuration parameters."""
        super().__init__(params if params is not None else PDEParams())
        self._cache_enabled = bool(getattr(self.params, "cache_enabled", True))
        # A prepared execution session's artefacts (adapter-owned clones only;
        # spec 2 §5.10): the bound Layout, the pre-flat-exact step sets and the
        # warmed factorizations. Pack misses build in the march, identically,
        # so correctness never depends on pack completeness.
        self._session_artefacts: Optional[SessionArtefacts] = None
        # Declarative grid layer (grid redesign spec §4.6) — built lazily.
        self._grid_binder = None
        # Frozen base-market layout held by create_bump_context instances: every
        # bumped re-solve reuses it (theta rolls rebind time only, §4.8).
        self._frozen_base_layout = None
        # Discrete-monitoring state shared by the barrier-family solvers
        # (populated by _setup_observation_indices).
        self._observation_indices: set = set()
        self._schedule_records: Dict[int, List] = {}
        self._schedule_aggregation: "ObservationAggregation" = (
            ObservationAggregation.STOP_FIRST_HIT
        )
        self._terminal_schedule_records: List = []
        self._has_terminal_observation: bool = False
        self._total_tau: float = 0.0

    def _setup_observation_indices(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        tau: float,
        t_vec: np.ndarray,
        resolve_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Resolve a product's discrete observation schedule onto the time grid.

        Routing convention (shared by all barrier-family solvers):
        - an observation at t=0 maps to time index 0,
        - an observation at t=tau is recorded as a TERMINAL observation
          (``_has_terminal_observation`` / ``_terminal_schedule_records``) and
          must be applied in ``set_terminal_condition`` — NOT silently dropped
          and NOT double-applied as an interior step,
        - interior observations snap to the nearest time-grid node.

        Args:
            product: Product carrying observation_schedule / observation_dates
            pricing_env: Pricing environment (for schedule resolution)
            tau: Total time to maturity
            t_vec: Time grid nodes
            resolve_kwargs: Product-specific defaults forwarded to
                ``ObservationSchedule.resolve`` (e.g. default_barrier /
                default_upper / default_lower / default_payoff)
        """
        self._total_tau = tau
        self._observation_indices.clear()
        self._schedule_records.clear()
        self._schedule_aggregation = ObservationAggregation.STOP_FIRST_HIT
        self._terminal_schedule_records = []
        self._has_terminal_observation = False

        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            schedule_kwargs = dict(resolve_kwargs or {})
            schedule_kwargs.setdefault("product", product)
            resolved_records = schedule.resolve(
                pricing_env=pricing_env, **schedule_kwargs
            )
            self._schedule_aggregation = schedule.aggregation_mode
            if self._schedule_aggregation in (
                ObservationAggregation.BEST,
                ObservationAggregation.WORST,
            ):
                raise PricingError(
                    f"PDE solver does not support aggregation mode "
                    f"{self._schedule_aggregation.value}"
                )
            for rec in resolved_records:
                if is_close(rec.observation_time, 0.0):
                    self._observation_indices.add(0)
                    self._schedule_records.setdefault(0, []).append(rec)
                elif is_close(rec.observation_time, tau):
                    self._terminal_schedule_records.append(rec)
                    self._has_terminal_observation = True
                elif 0.0 < rec.observation_time < tau:
                    idx = int(np.argmin(np.abs(t_vec - rec.observation_time)))
                    self._observation_indices.add(idx)
                    self._schedule_records.setdefault(idx, []).append(rec)
        elif (
            getattr(product, "observation_type", None) == ObservationType.DISCRETE
            and getattr(product, "observation_dates", None) is not None
        ):
            for obs_time in product.observation_dates:
                if is_close(obs_time, 0.0):
                    self._observation_indices.add(0)
                elif is_close(obs_time, tau):
                    self._has_terminal_observation = True
                elif 0.0 < obs_time < tau:
                    idx = int(np.argmin(np.abs(t_vec - obs_time)))
                    self._observation_indices.add(idx)

    def _setup_maturity_observation(self, product, pricing_env, resolve_kwargs) -> None:
        """The observation state at maturity (tau <= 0; spec 3C decision 30): the records observed at the valuation
        instant ARE the maturity records, filed as ``_setup_observation_indices`` files a terminal observation, so
        ``_terminal_event`` builds them exactly as the march's last node does."""
        self._setup_observation_indices(product, pricing_env, 0.0, np.array([0.0]), resolve_kwargs)
        records = list(self._schedule_records.pop(0, []))
        dated = 0 in self._observation_indices
        self._observation_indices.discard(0)
        self._terminal_schedule_records = records
        self._has_terminal_observation = bool(records) or dated
        self._total_tau = 0.0

    def _valuation_instant_records(self, product, pricing_env, resolve_kwargs):
        """The records observed at the valuation instant, exactly those ``_setup_observation_indices`` files under
        time index 0 (spec 3C decision 25, P5), found without a time grid: None when nothing is observed then;
        ``[]`` for a bare observation date at 0 (the product's default record); else the resolved records."""
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            if schedule.aggregation_mode in (ObservationAggregation.BEST, ObservationAggregation.WORST):
                raise PricingError(f"PDE solver does not support aggregation mode {schedule.aggregation_mode.value}")
            kwargs = dict(resolve_kwargs or {})
            kwargs.setdefault("product", product)
            records = [rec for rec in schedule.resolve(pricing_env=pricing_env, **kwargs)
                       if is_close(rec.observation_time, 0.0)]
            return records or None
        if (getattr(product, "observation_type", None) == ObservationType.DISCRETE
                and getattr(product, "observation_dates", None) is not None):
            return [] if any(is_close(t, 0.0) for t in product.observation_dates) else None
        return None

    def _resolved_terminal_payoffs(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        default_payoff: float,
        *,
        event_payment: bool = True,
    ) -> List[Tuple]:
        """
        Terminal observation records paired with settlement-discounted payoffs.

        Returns ``[(record_or_None, payoff), ...]``: the terminal schedule
        records when the product is discretely monitored and the schedule
        observes at t=T, else a single ``(None, default_payoff)`` entry (the
        product-level default for continuous monitoring or date-list
        schedules). Event-paid records use their resolved payment time;
        expiry-paid records use the product's terminal payment timing.
        """
        records = (
            self._terminal_schedule_records
            if (
                getattr(product, "observation_type", None)
                == ObservationType.DISCRETE
                and self._terminal_schedule_records
            )
            else [None]
        )
        out = []
        for rec in records:
            payoff = rec.payoff if rec is not None else default_payoff
            settlement_time = self._terminal_payment_time(
                product, pricing_env
            )
            if rec is not None and event_payment:
                settlement_time = rec.settlement_time
            payoff = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=payoff,
                current_time=self._total_tau,
                settlement_time=settlement_time,
            )
            out.append((rec, payoff))
        return out

    def _is_cache_enabled(self) -> bool:
        """``PDEParams.cache_enabled`` -- the one cache control (spec §5.2)."""
        return self._cache_enabled

    def create_bump_context(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> "BasePDESolver":
        """A fresh instance with the parent's constructor arguments, holding the
        frozen base layout (spec 2 §5.10).

        Spot/vol/rate/div bumps reuse the layout by identity; calendar (theta)
        bumps rebuild only the time layout (``rebind_time``). A request with a
        terminal outcome freezes nothing: bumping re-prices the short-circuit.
        """
        if not self._uses_grid_layer():
            raise NotImplementedError(
                "legacy params-freeze bump contexts were removed with the "
                "declarative grid layer (0.4.0)"
            )
        layout = bind_layout(self, product, pricing_env, OutputRequest(Operation.PRICE))
        if layout is None:
            return self
        clone = type(self)(**self._constructor_kwargs())
        clone._frozen_base_layout = layout
        return clone

    def _constructor_kwargs(self) -> dict:
        """The arguments a bump context is constructed with (subclasses add theirs)."""
        return {"params": self.params}

    # ------------------------------------------------------------------
    # Declarative grid layer seam (grid redesign spec §4.6)
    # ------------------------------------------------------------------

    def _uses_grid_layer(self) -> bool:
        """Whether this solver has migrated to the declarative grid layer.

        Migrated solvers return True and implement ``grid_request`` /
        ``event_schedule``; every 1D solver has (the legacy grid path is gone).
        """
        return False

    def grid_request(
        self, product: BaseEquityProduct, market: MarketSnapshot, tau: float
    ) -> GridRequest:
        """Declare this product's grid geometry (migrated solvers only).

        ``tau`` is passed explicitly (product maturity against the pricing
        env's valuation date) — MarketSnapshot stays a pure market object.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not migrated to the grid layer"
        )

    def representative_vol(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """sigma_ref for MarketSnapshot — strike-selected vol by default."""
        tau = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", pricing_env.spot)
        return float(pricing_env.get_vol(strike, tau))

    def event_schedule(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        layout: Layout,
    ) -> EventSchedule:
        """Per-solve event semantics (migrated solvers override)."""
        return EventSchedule()

    def market_snapshot(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> MarketSnapshot:
        """The market inputs the grid builders consume (spec §4.2)."""
        tau = product.get_maturity(pricing_env)
        return MarketSnapshot(
            spot=float(pricing_env.spot),
            sigma_ref=self.representative_vol(product, pricing_env),
            r_ref=float(pricing_env.get_rate(tau)),
            q_ref=float(pricing_env.get_div_yield(tau)),
            bus_days_in_year=int(pricing_env.bus_days_in_year),
        )

    @property
    def grid_binder(self) -> GridBinder:
        """Engine-owned binder; cache behavior follows the engine's cache
        strategy (``disable`` -> no layout cache)."""
        if self._grid_binder is None:
            self._grid_binder = GridBinder(
                getattr(self.params, "accuracy", "standard"),
                getattr(self.params, "grid", None),
                cache_enabled=self._is_cache_enabled(),
                cache_max_entries=GRID_CACHE_MAX_ENTRIES,
            )
        return self._grid_binder

    def _populate_observation_maps(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        layout: Layout,
        tau: float,
    ) -> None:
        """Layer-path observation bookkeeping hook (no-op in the base;
        autocallable solvers derive their index maps from layout.step_of)."""

    def _get_event_times(
        self, product: BaseEquityProduct, tau: float
    ) -> Optional[List[float]]:
        """Helper to collect observation/event times."""
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None and getattr(schedule, "times", None):
            return [t for t in schedule.times if 0 < t < tau]
        for attr in ("observation_dates", "obs_times", "event_times"):
            if hasattr(product, attr):
                times = getattr(product, attr)
                if times:
                    return [t for t in times if 0 < t < tau]
        return None
    # -- session factorization packs (spec section 9.2) --------------------

    def _generic_grid_request(
        self, product: BaseEquityProduct, market: MarketSnapshot, tau: float
    ) -> GridRequest:
        """Shared declaration for the simple 1D solvers (spec §4.6 notes).

        - anchors: spot + strike;
        - criticals: spot, strike, and any product barriers;
        - hard bounds: CONTINUOUS absorbing barriers — knock-out barriers
          (ported verbatim from the legacy `_resolve_spatial_bounds`
          clamping) and touch barriers — are domain edges (single barriers
          set exactly one side). DISCRETE barriers are NOT: between two
          observations the spot may cross the barrier and come back
          unobserved, so the domain has to carry that overshoot;
        - event times: the generic observation schedule (discretely observed
          barrier/touch variants), interior only.
        """
        spot = market.spot
        strike = float(getattr(product, "strike", spot) or spot)
        barriers = [b for b in self._get_barriers(product) if b and b > 0]

        hard_lower = hard_upper = None
        obs_type = getattr(product, "observation_type", None)
        # Absorbing under continuous observation: once the barrier is touched the claim's value
        # there is known for the rest of its life (KO: the rebate/zero; touch: the rebate or
        # nothing), so no state beyond it is ever solved. Every TouchType is absorbing.
        absorbing = (
            getattr(product, "is_knock_out", False)
            or getattr(product, "touch_type", None) is not None
        )
        if obs_type == ObservationType.CONTINUOUS and absorbing:
            if hasattr(product, "lower_barrier") and hasattr(product, "upper_barrier"):
                lb = getattr(product, "lower_barrier", 0) or 0
                ub = getattr(product, "upper_barrier", 0) or 0
                hard_lower = float(lb) if lb > 0 else None
                hard_upper = float(ub) if ub > 0 else None
            elif hasattr(product, "barrier"):
                b = getattr(product, "barrier", 0) or 0
                if b > 0:
                    if getattr(product, "is_up_barrier", False):
                        hard_upper = float(b)
                    else:
                        hard_lower = float(b)

        events = [
            float(t)
            for t in (self._get_event_times(product, tau) or [])
            if 0.0 < float(t) < tau
        ]
        criticals = [spot, strike] + barriers
        return GridRequest(
            tau=tau,
            bound_anchors=(spot, strike),
            critical_prices=tuple(p for p in criticals if p and p > 0),
            hard_lower=hard_lower,
            hard_upper=hard_upper,
            event_times=tuple(sorted(set(events))),
            refinement=self._time_refinement(product),
            day_count=market.bus_days_in_year,
        )

    def _external_layout_check(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        layout: Layout,
    ) -> None:
        """Validate an externally supplied layout before solving on it."""
        market = self.market_snapshot(product, pricing_env)
        tau = product.get_maturity(pricing_env)
        validate_external_layout(
            layout, self.grid_request(product, market, tau), market
        )

    def _freeze_cache_value(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_cache_value(v) for v in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (self._freeze_cache_value(k), self._freeze_cache_value(v))
                    for k, v in value.items()
                )
            )
        return repr(value)

    def _cache_value_is_simple(
        self, value, depth: int, max_depth: int, max_len: int
    ) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if depth >= max_depth:
            return False
        if isinstance(value, (list, tuple)):
            if len(value) > max_len:
                return False
            return all(
                self._cache_value_is_simple(v, depth + 1, max_depth, max_len)
                for v in value
            )
        if isinstance(value, dict):
            if len(value) > max_len:
                return False
            return all(
                self._cache_value_is_simple(k, depth + 1, max_depth, max_len)
                and self._cache_value_is_simple(v, depth + 1, max_depth, max_len)
                for k, v in value.items()
            )
        return False

    def _cache_dict_is_reasonable(
        self, attrs: Dict, max_depth: int = 2, max_len: int = 64
    ) -> bool:
        if len(attrs) > max_len:
            return False
        for value in attrs.values():
            if not self._cache_value_is_simple(value, 0, max_depth, max_len):
                return False
        return True

    def _product_cache_token(self, product: BaseEquityProduct, strategy: str) -> Tuple[str, object]:
        if strategy == "strict":
            return ("id", id(product))
        key_fn = getattr(product, "cache_key", None)
        if callable(key_fn):
            return ("key", self._freeze_cache_value(key_fn()))
        if strategy in ("standard", "aggressive"):
            attrs = getattr(product, "__dict__", None)
            if attrs is not None and self._cache_dict_is_reasonable(attrs):
                return ("dict", self._freeze_cache_value(attrs))
            if strategy == "aggressive":
                return ("repr", repr(product))
        return ("id", id(product))

    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set the terminal condition (payoff at maturity)."""
        pass

    @abstractmethod
    def set_boundary_conditions(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set boundary conditions at the spatial edges for a given time step."""
        pass

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """Return a list of critical prices (e.g., strikes) for grid concentration."""
        points = []
        if hasattr(product, "strike") and product.strike > 0:
            points.append(product.strike)
        return points

    def _prepare_solve_state(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ):
        """_solve's state preamble, shared with session preparation.

        The base family carries no pre-grid solve state; the two-surface
        autocallable solvers override this (KI regime, valuation flags,
        barrier level) and BOTH their _solve and the session adapter's
        prepare() call it, so grid-key evaluation on a fresh clone sees the
        same state a direct solve would.
        """
        return None

    #: The family seam (spec 2 §5.1): each solver names its adapter class.
    _adapter_class = FamilyAdapter

    def _time_refinement(self, product) -> TimeRefinement:
        """The refinement this solve's stepper asks of the time grid (spec 3A §5.3, decision 6).

        A function of ``(params, product)`` alone, so every path that builds this product's
        request (prepare, bump contexts, sessions, external-layout checks) builds the same one.
        A custom stepper without ``time_refinement`` asks for none.
        """
        stepper = self.adapter.stepper(self.params, product)
        ask = getattr(stepper, "time_refinement", None)
        return TimeRefinement() if ask is None else ask()

    def _base_family_projection(self) -> EventProjectionMode:
        """The projection a base-family discrete observation uses (spec 3A §5.4)."""
        mode = self.params.event_projection
        return _BASE_FAMILY_DEFAULT_PROJECTION if mode is None else mode

    def _apply_discrete_event(self, column, x_vec, s_vec, event: DiscreteEvent, *, nodal: bool) -> np.ndarray:
        """One observation's records on ``column``. ``nodal``: continuous monitoring (the barrier is
        the domain edge) and the valuation-date observation (a terminal outcome, ``prepare``
        step 0) keep the nodal mask in every mode."""
        mode = EventProjectionMode.NODAL if nodal else self._base_family_projection()
        return apply_discrete_event(Nodes(x_vec, s_vec), column, event, mode)

    def _apply_discrete_event_at(self, grid, t_idx, x_vec, s_vec, event: DiscreteEvent, *, nodal: bool) -> None:
        """``_apply_discrete_event`` on ``grid[:, t_idx]``, written back. The nodal fold runs in place
        on the column, as the 0.7.0 step hooks did: no column copy on the per-step path."""
        if nodal or self._base_family_projection() is EventProjectionMode.NODAL:
            fold_nodal(grid[:, t_idx], s_vec, event)
        else:
            grid[:, t_idx] = self._apply_discrete_event(grid[:, t_idx], x_vec, s_vec, event, nodal=False)

    @property
    def adapter(self) -> FamilyAdapter:
        """This solver's family adapter, built once (a copy of the solver rebuilds its own)."""
        adapter = self.__dict__.get("_adapter")
        if adapter is None or adapter.solver is not self:
            adapter = self._adapter_class(self)
            self._adapter = adapter
        return adapter

    def explain(self, product, pricing_env, *, lifecycle_state=None):
        """The resolved plan of ``price``'s preparation, without the march (spec 2 §4.7).

        Raises where ``price`` would; its ``identity["problem"]`` is the one
        ``price`` marches with.
        """
        from quantark.asset.equity.engine.pde.core.plan import build_plan

        problem = prepare(self, product, pricing_env, OutputRequest(Operation.PRICE),
                          lifecycle_state=lifecycle_state, frozen=self._frozen_base_layout,
                          session=self._session_artefacts)
        return build_plan(problem, self, product, pricing_env, lifecycle_state)

    def _run(self, product, pricing_env, request, *, lifecycle_state=None):
        """``run(prepare(...))`` with this solver's frozen layout and session artefacts."""
        return run(prepare(self, product, pricing_env, request, lifecycle_state=lifecycle_state,
                           frozen=self._frozen_base_layout, session=self._session_artefacts))

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """Price the option using the PDE finite difference method."""
        return self._run(product, pricing_env, OutputRequest(Operation.PRICE))

    def _session_outputs(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        want_events: bool = False,
        want_grid: bool = False,
        streams: Optional[frozenset] = None,
    ) -> PDESessionOutputs:
        """One value solve serving PV (+ grid projection). Event fields are
        autocallable-only; the base family always returns them as None."""
        return self._run(product, pricing_env,
                         OutputRequest(Operation.SESSION_OUTPUTS, grid=want_grid))

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """Calculate Delta and Gamma directly from the PDE solution surface."""
        return self._run(product, pricing_env, OutputRequest(Operation.GREEKS, greeks=True))

    def calculate_spot_greeks_curve(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot_levels: Sequence[float],
    ) -> list[dict[str, float | str]]:
        """Interpolate a spot Greeks curve from one PDE solve."""
        spots = np.asarray([float(spot) for spot in spot_levels], dtype=float)
        if spots.size == 0:
            return []
        if not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
            raise ValueError("spot levels must be positive and finite")
        if product.get_maturity(pricing_env) <= 0:
            return super().calculate_spot_greeks_curve(product, pricing_env, spots)

        result = self._spot_curve_solution(product, pricing_env)
        if result is None:          # decided at the base spot (spec 3C §5.2 (g)): point Greeks, ``reprice`` rows
            return super().calculate_spot_greeks_curve(product, pricing_env, spots)
        return self._grid_projection_from_solution(result, spots)

    def _spot_curve_solution(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        """The solved surface a spot Greeks curve reads: the value march's final column."""
        return self._run(product, pricing_env,
                         OutputRequest(Operation.SESSION_OUTPUTS, grid=True)).solution

    def _grid_projection_from_solution(
        self,
        result: PDESolutionResult,
        spot_levels: Optional[Sequence[float]] = None,
    ) -> list[dict[str, float | str]]:
        """Project price/delta/gamma from a solve's carrier (the session's GRID and the public curve, spec 3C
        decision 26): a composite or branch carrier projects itself; a surface is read at each level within the
        branch alive there, then the valuation instant's decision is folded at that level and the receivable added.
        ``spot_levels`` None projects at the grid nodes themselves."""
        if hasattr(result, "rows"):
            return result.rows(spot_levels)
        spots = [float(s) for s in np.asarray(result.s_vec if spot_levels is None else spot_levels, dtype=float)]
        out = []
        for spot, (price, delta, gamma) in zip(spots, _project_rows(result, spots)):
            if result.decision is not None:
                price, delta, gamma, _ = fold_decision(result.decision, spot, price, delta, gamma)
            if result.cash:
                price = price + result.cash
            out.append({"spot": spot, "price": price, "delta": delta, "gamma": gamma,
                        "calculation_mode": "engine_grid"})
        return out

    def _build_step_coefficients(
        self,
        pricing_env: PricingEnvironment,
        ref_strike: float,
        t_vec: np.ndarray,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        """Sample forward (r, q, sigma) per time step and build operator sets.

        The per-step sets BEFORE the flat-exact replacement, which callers apply
        (``_flat_exact_step_coefficients``): ``sample_bsm`` then
        ``CentralDifference.assemble_sets``.
        """
        # Keyed on every input the sampler reads: the three market objects by
        # value, the spot (TermCoefficients.from_env passes it to every
        # get_vol), the reference strike and the grid. No fingerprint, no memo.
        fps = self._market_fps(pricing_env)
        market_key = (fps["rate"], fps["div"], fps["vol"], fps["spot"])
        mkey = None
        if None not in market_key:
            mkey = market_key + (
                float(ref_strike),
                np.asarray(t_vec, dtype=float).tobytes(),
                np.asarray(dx_vec, dtype=float).tobytes(),
                int(num_x),
            )
            cached = _ENV_STEP_COEFF_MEMO.get(mkey)
            if cached is not None:
                return cached

        samples = sample_bsm(pricing_env, t_vec, ref_strike=ref_strike, provenance=fps)
        result = CentralDifference().assemble_sets(samples, dx_vec, num_x)
        if mkey is not None:
            if len(_ENV_STEP_COEFF_MEMO) >= _STEP_COEFF_MEMO_MAX_ENTRIES:
                _ENV_STEP_COEFF_MEMO.popitem(last=False)  # FIFO eviction
            _ENV_STEP_COEFF_MEMO[mkey] = result
        return result

    def _flat_exact_step_coefficients(
        self,
        sc: StepCoefficients,
        r: float,
        q: float,
        sigma: float,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        """Rebuild a single unique set from the exact cumulative scalars.

        One unique set means the curves are constant on the grid, so the
        cumulative scalars equal the forwards mathematically; substituting is
        exact (not an approximation) and makes flat envs bit-identical to the
        pre-term code path.
        """
        if sc.n_unique != 1:
            return sc
        return flat_exact(sc, self._calculate_coefficients(r, q, sigma, dx_vec, num_x))

    def _calculate_coefficients(
        self, r: float, q: float, sigma: float, dx_vec: np.ndarray, num_x: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate coefficients (l, c, u) for the tridiagonal FD system."""
        return scalar_stencil(r, q, sigma, dx_vec, num_x)

    def _apply_step_modifications(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Apply product-specific modifications after each time step.

        Override this method for American options (early exercise)
        or barrier options (barrier checks).

        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: The option product
            pricing_env: Pricing environment
        """
        pass  # Default: no modifications

    def _interpolate_price(
        self, v_vec: np.ndarray, x_vec: np.ndarray, x_target: float
    ) -> float:
        """
        Interpolate option value at target log-price.

        Uses linear interpolation between nearest grid points.

        Args:
            v_vec: Option values at grid points
            x_vec: Log-price grid points
            x_target: Target log-price

        Returns:
            Interpolated option value
        """
        return interpolate_price(v_vec, x_vec, x_target)

    def _calculate_delta_gamma(
        self, v_vec: np.ndarray, x_vec: np.ndarray, x_target: float, spot: float
    ) -> Tuple[float, float]:
        """
        Calculate delta and gamma from the solution vector.

        In log-space:
            dV/dS = (1/S) * dV/dx
            d2V/dS2 = (1/S^2) * (d2V/dx^2 - dV/dx)

        Args:
            v_vec: Option values at grid points
            x_vec: Log-price grid points
            x_target: Target log-price (ln(spot))
            spot: Current spot price

        Returns:
            Tuple of (delta, gamma)
        """
        return delta_gamma(v_vec, x_vec, x_target, spot)

    @staticmethod
    def _current_time(total_tau: float, tau_remaining: float) -> float:
        """Elapsed time (from valuation) at a backward-induction step."""
        return max(total_tau - tau_remaining, 0.0)

    def _market_fps(self, pricing_env: PricingEnvironment) -> Dict[str, Optional[str]]:
        """The environment's market fingerprints: the solve's, else computed now."""
        scope = getattr(self, "_market_fp", None)
        if scope is not None and scope[0] is pricing_env:
            return scope[1]
        return market_fingerprints(pricing_env)

    def _curve_df_memo(self, leg: str, pricing_env: PricingEnvironment) -> dict:
        """The discount memo of one curve: scoped to the curve object, keyed by its value.

        The inner keys round (t, T) to 12 decimals and the first build wins, so
        the memo's sharing domain decides the last ULP of every discount factor.
        That domain stays the curve object's lifetime, as every banked number
        was computed; a value-keyed global store would let one solve's times
        seed another's (measured: a vega bump moved in its last digit depending
        on what the process priced before). The value fingerprint inside the
        scope is what makes an in-place mutation re-sample (decision 14).
        """
        curve = pricing_env.rate_curve if leg == "r" else pricing_env.div_yield
        scope = getattr(self, "_market_fp", None)
        # Within one market scope (one prepare's fingerprints) a leg's memo is fixed:
        # resolve it once per scope, not twice per step (timing gate §9.5). The
        # resolved memos live in the scope tuple, so they end with it.
        resolved = scope[2] if scope is not None and scope[0] is pricing_env else None
        if resolved is not None:
            hit = resolved.get(leg)
            if hit is not None and hit[0] is curve:
                return hit[1]
        memo = self._resolve_curve_df_memo(leg, pricing_env, curve, scope)
        if resolved is not None:
            resolved[leg] = (curve, memo)
        return memo

    def _resolve_curve_df_memo(self, leg, pricing_env, curve, scope) -> dict:
        if scope is not None and scope[0] is pricing_env:
            curve_fp = scope[1]["rate" if leg == "r" else "div"]
        else:
            from quantark.execution.cache.fingerprint import try_fingerprint
            curve_fp = try_fingerprint(curve)
        if curve_fp is None:
            return {}
        per_object = _per_object_memo(curve, _ENV_DF_MEMO)
        if per_object is None:
            return {}
        key = (leg, curve_fp)
        memo = per_object.get(key)
        if memo is None:
            per_object.clear()  # the curve's earlier values are unreachable now
            memo = per_object[key] = {}
        return memo

    def _df_between_times(
        self, pricing_env: PricingEnvironment, start_time: float, end_time: float
    ) -> float:
        """
        Forward discount factor DF(start_time, end_time), both measured from
        the valuation date. Term-structure consistent: DF(0,T)/DF(0,t), NOT
        exp(-r(tau)*tau) with tau = remaining time (they coincide only under
        a flat curve).

        Memoized per rate curve VALUE (its fingerprint; spec 2 decision 14):
        the boundary-condition path evaluates the same (t, T) pairs many
        times per step, and term-structure curves make each curve lookup a
        np.interp call — the memo keeps the term/flat cost ratio inside the
        spec's 20% budget.
        """
        if end_time <= start_time:
            return 1.0
        # _curve_df_memo's in-scope hit, inline: this runs twice per column group per step
        scope = self._market_fp
        hit = scope[2].get("r") if scope is not None and scope[0] is pricing_env else None
        if hit is not None and hit[0] is pricing_env.rate_curve:
            memo = hit[1]
        else:
            memo = self._curve_df_memo("r", pricing_env)
        key = (round(float(start_time), 12), round(float(end_time), 12))
        v = memo.get(key)
        if v is None:
            df_end = pricing_env.get_discount_factor(end_time)
            df_start = pricing_env.get_discount_factor(start_time)
            v = float(safe_divide(df_end, df_start, fallback=1.0))
            if len(memo) >= _DF_MEMO_MAX_ENTRIES:
                memo.clear()
            memo[key] = v
        return v

    def _carry_df_between_times(
        self, pricing_env: PricingEnvironment, start_time: float, end_time: float
    ) -> float:
        """Forward dividend/carry discount factor exp(-(q(T)T - q(t)t)).

        Term-structure consistent analog of _df_between_times for the carry
        leg of asymptotic boundary conditions; equals exp(-q*(T-t)) only
        under a flat carry curve.
        """
        if end_time <= start_time:
            return 1.0
        scope = self._market_fp
        hit = scope[2].get("q") if scope is not None and scope[0] is pricing_env else None
        if hit is not None and hit[0] is pricing_env.div_yield:
            memo = hit[1]
        else:
            memo = self._curve_df_memo("q", pricing_env)
        key = ("q", round(float(start_time), 12), round(float(end_time), 12))
        v = memo.get(key)
        if v is None:
            w_end = float(pricing_env.get_div_yield(end_time)) * float(end_time)
            w_start = (
                float(pricing_env.get_div_yield(start_time)) * float(start_time)
                if start_time > 0.0
                else 0.0
            )
            v = float(np.exp(-(w_end - w_start)))
            if len(memo) >= _DF_MEMO_MAX_ENTRIES:
                memo.clear()
            memo[key] = v
        return v

    def _cashflow_value_at_time(
        self,
        pricing_env: PricingEnvironment,
        cashflow: float,
        current_time: float,
        settlement_time: Optional[float],
    ) -> float:
        """Discount a cashflow from its settlement time back to current_time."""
        if settlement_time is None or settlement_time <= current_time:
            return float(cashflow)
        df = self._df_between_times(pricing_env, current_time, settlement_time)
        return float(cashflow) * df

    def _terminal_payment_time(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> float:
        """Return the payment time for the terminal determination."""
        return float(self._terminal_timing(product, pricing_env).payment_time)

    @staticmethod
    def _multiplier(product) -> float:
        """Underlying units per contract (spec 3C decision 24): the base family's solves are per contract."""
        return float(getattr(product, "contract_multiplier", 1.0))

    def _terminal_delay_df(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> float:
        """Return DF(payment) / DF(terminal determination)."""
        return float(self._terminal_timing(product, pricing_env).delay_df)

    def _terminal_timing(self, product: BaseEquityProduct, pricing_env: PricingEnvironment):
        """``resolve_terminal_timing``, once per market scope and product: the boundary and event
        hooks read it every step, and it is fixed for the scope's product and environment. The
        memo lives in the scope tuple, so it ends with the scope (as ``_curve_df_memo``)."""
        scope = self._market_fp
        resolved = scope[2] if scope is not None and scope[0] is pricing_env else None
        if resolved is None:
            return resolve_terminal_timing(product, pricing_env)
        hit = resolved.get("terminal_timing")
        if hit is not None and hit[0] is product:
            return hit[1]
        timing = resolve_terminal_timing(product, pricing_env)
        resolved["terminal_timing"] = (product, timing)
        return timing

    def _event_payment_time(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        determination_time: float,
        *,
        kind: CashflowKind = CashflowKind.HIT,
        cashflow_id: str = "barrier_event",
    ) -> float:
        """Resolve a node-determined event to its contractual payment time."""
        timing = SettlementResolver.resolve_contingent(
            product,
            SettlementRequest(
                kind=kind,
                determination_time=float(determination_time),
                cashflow_id=cashflow_id,
            ),
            pricing_env,
        )
        return float(timing.payment_time)

    def _calculate_intrinsic(self, product: BaseEquityProduct, spot: float) -> float:
        """
        Calculate intrinsic value of the option.

        Args:
            product: The option product
            spot: Current spot price

        Returns:
            Intrinsic value
        """
        if hasattr(product, "get_payoff"):
            return product.get_payoff(spot)
        return 0.0

    def _intrinsic_delta(self, product: BaseEquityProduct, spot: float) -> float:
        """
        Calculate delta of intrinsic value.

        Args:
            product: The option product
            spot: Current spot price

        Returns:
            Intrinsic delta
        """
        multiplier = getattr(product, "contract_multiplier", 1.0)
        if hasattr(product, "is_call") and hasattr(product, "strike"):
            if product.is_call():
                delta = 1.0 if spot > product.strike else 0.0
            else:
                delta = -1.0 if spot < product.strike else 0.0
            return delta * multiplier
        return 0.0 * multiplier

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Helper to collect barrier levels from known product attributes."""
        barriers = []
        for attr in ("barrier", "upper_barrier", "lower_barrier"):
            if hasattr(product, attr):
                val = getattr(product, attr)
                if val is not None and val > 0:
                    barriers.append(val)
        return barriers
