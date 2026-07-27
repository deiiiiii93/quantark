"""
Base PDE solver providing common infrastructure for finite difference pricing.

Implements the Crank-Nicolson scheme for solving the Black-Scholes PDE
backward in time, with support for Rannacher smoothing.
"""

from abc import abstractmethod
import weakref
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import math
import threading
from typing import Dict, Optional, Tuple, List, NamedTuple, Sequence
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.pde.grid import (
    EventSchedule,
    GridBinder,
    GridRequest,
    Layout,
    MarketSnapshot,
    validate_external_layout,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError, NumericalError
from quantark.util.numerical import is_close, safe_divide
from quantark.util.enum.option_enums import ExerciseType, ObservationType
from quantark.util.enum import ObservationAggregation
from quantark.util.enum.engine_enums import EngineType

from .backward_operator import BackwardOperator


# Per-market-object memoization for curve lookups and step coefficients.
# Keyed on the CURVE OBJECTS (rate curve / dividend yield), not the mutable
# PricingEnvironment: the established market-data update pattern is attribute
# REPLACEMENT (env.rate_curve = new_curve), which yields a new object and
# therefore a fresh memo — a bumped or refreshed environment can never reuse
# stale values (codex code-review finding). Curve/dataclass objects are
# unhashable (eq=True), so stores key on id() with a weakref eviction
# callback plus an identity re-check; id reuse cannot alias because the
# stored weakref must still point at the SAME object. Each memo is size-
# bounded so long-lived processes cannot grow without bound.
_ENV_DF_MEMO: dict = {}
_ENV_STEP_COEFF_MEMO: dict = {}
_DF_MEMO_MAX_ENTRIES = 200_000
_STEP_COEFF_MEMO_MAX_ENTRIES = 64


def _per_object_memo(market_obj, store: dict) -> Optional[dict]:
    """Return the memo dict for a market object, or None if not weakref-able."""
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


class StepCoefficients(NamedTuple):
    """Per-step (l, c, u) operator coefficient sets, deduped by unique triple.

    ``lcu_sets[set_index[j]]`` is the operator for backward step ``j`` (the
    interval [t_vec[j], t_vec[j+1]]). Flat market inputs produce exactly one
    set, preserving single-operator factorization reuse; term inputs pay the
    designed per-step rebuild (spec Component 4).
    """

    lcu_sets: list
    set_index: np.ndarray
    n_unique: int


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

    _grid_cache_max_entries: int = 128
    _global_cache_enabled: bool = True
    _global_cache_strategy: Optional[str] = None
    _cache_lock = threading.Lock()

    def __init__(self, params: Optional[PDEParams] = None):
        """Initialize the PDE solver with configuration parameters."""
        super().__init__(params if params is not None else PDEParams())
        self._matrix_cache: Dict[Tuple[float, float], Tuple] = {}
        self._cache_enabled = bool(getattr(self.params, "cache_enabled", True))
        self._cache_strategy = getattr(self.params, "cache_strategy", "standard")
        cache_size = getattr(self.params, "grid_cache_max_entries", None)
        if cache_size is not None:
            self.set_grid_cache_max_entries(cache_size)
        self._critical_points_cache: "OrderedDict[Tuple, Tuple[float, ...]]" = (
            OrderedDict()
        )
        # Session-injected preparation (adapter-owned clones only; spec
        # section 9.2). Grids/coefficients short-circuit their builders;
        # the matrix pack is a READ-ONLY mapping consulted before the
        # per-solve _matrix_cache — misses fall through and build locally,
        # so correctness never depends on pack completeness.
        self._session_grids = None
        self._session_step_coefficients = None
        self._session_matrix_pack = None
        # Declarative grid layer (grid redesign spec §4.6) — built lazily.
        self._grid_binder = None
        # Frozen base-market layout set by create_bump_context clones: every
        # bumped re-solve reuses it (theta rolls rebind time only, §4.8).
        self._frozen_base_layout = None
        # Layout bound for the CURRENT solve (None on the legacy path).
        self._active_layout = None
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
            resolved_records = schedule.resolve(
                pricing_env=pricing_env, **(resolve_kwargs or {})
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

    def _resolved_terminal_payoffs(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        default_payoff: float,
    ) -> List[Tuple]:
        """
        Terminal observation records paired with settlement-discounted payoffs.

        Returns ``[(record_or_None, payoff), ...]``: the terminal schedule
        records when the product is discretely monitored and the schedule
        observes at t=T, else a single ``(None, default_payoff)`` entry (the
        product-level default for continuous monitoring or date-list
        schedules). Record payoffs with a settlement_time are discounted back
        to maturity via the forward discount factor.
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
            if rec is not None and rec.settlement_time is not None:
                payoff = self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=payoff,
                    current_time=self._total_tau,
                    settlement_time=rec.settlement_time,
                )
            out.append((rec, payoff))
        return out

    @classmethod
    def clear_grid_cache(cls) -> None:
        """No-op since 0.4.0: layouts are cached per engine instance by the
        GridBinder (kept for source compatibility with older call sites)."""

    @classmethod
    def set_cache_enabled(cls, enabled: bool, clear: bool = False) -> None:
        """Enable or disable cache usage for this solver class."""
        cls._global_cache_enabled = bool(enabled)
        if clear:
            cls.clear_grid_cache()

    @classmethod
    def set_cache_strategy(cls, strategy: str, clear: bool = False) -> None:
        """Set the cache strategy for this solver class."""
        if strategy not in ("disable", "strict", "standard", "aggressive"):
            raise PricingError(
                "cache_strategy must be one of disable, strict, standard, aggressive, got "
                f"{strategy}"
            )
        cls._global_cache_strategy = strategy
        if clear:
            cls.clear_grid_cache()

    @classmethod
    def set_grid_cache_max_entries(cls, max_entries: int) -> None:
        """Set maximum number of grid entries to keep in cache."""
        if max_entries <= 0:
            raise PricingError(
                f"Grid cache size must be positive, got {max_entries}"
            )
        with cls._cache_lock:
            cls._grid_cache_max_entries = max_entries

    def _is_cache_enabled(self) -> bool:
        return self._resolve_cache_strategy() != "disable"

    def _resolve_cache_strategy(self) -> str:
        if not self.__class__._global_cache_enabled or not self._cache_enabled:
            return "disable"
        strategy = self.__class__._global_cache_strategy
        if strategy is None:
            strategy = self._cache_strategy
        return strategy

    def create_bump_context(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> "BasePDESolver":
        """
        Return a PDE solver clone with the base spatial domain frozen.

        Migrated solvers freeze the whole base-market ``Layout`` (spec §4.8):
        spot/vol/rate/div bumps reuse it by identity; calendar (theta) bumps
        rebuild only the time layout via ``rebind_time``. Legacy solvers keep
        the historical params-freeze.
        """
        if self._uses_grid_layer():
            tau = product.get_maturity(pricing_env)
            if tau <= 0:
                return self
            clone = type(self)(params=deepcopy(self.params))
            prep = getattr(clone, "_prepare_for_request", None)
            if prep is not None:
                prep(product, pricing_env)
            market = clone.market_snapshot(product, pricing_env)
            request = clone.grid_request(product, market, tau)
            clone._frozen_base_layout = clone.grid_binder.bind(request, market)
            return clone

        raise NotImplementedError(
            "legacy params-freeze bump contexts were removed with the "
            "declarative grid layer (0.4.0)"
        )

    # ------------------------------------------------------------------
    # Declarative grid layer seam (grid redesign spec §4.6)
    # ------------------------------------------------------------------

    def _uses_grid_layer(self) -> bool:
        """Whether this solver has migrated to the declarative grid layer.

        Migrated solvers return True and implement ``grid_request`` /
        ``event_schedule``; the legacy ``_build_grids`` path serves the rest
        until their phase lands (the flag disappears at Phase 4).
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
                cache_max_entries=int(
                    getattr(self.params, "grid_cache_max_entries", 128) or 128
                ),
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
        - hard bounds: CONTINUOUS knock-out barriers are absorbing domain
          edges (single barriers set exactly one side) — ported verbatim
          from the legacy `_resolve_spatial_bounds` clamping;
        - event times: the generic observation schedule (discretely observed
          barrier/touch variants), interior only.
        """
        spot = market.spot
        strike = float(getattr(product, "strike", spot) or spot)
        barriers = [b for b in self._get_barriers(product) if b and b > 0]

        hard_lower = hard_upper = None
        obs_type = getattr(product, "observation_type", None)
        is_ko = getattr(product, "is_knock_out", False)
        if obs_type == ObservationType.CONTINUOUS and is_ko:
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
        )

    def _grids_via_layer(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bind (or reuse the frozen) layout and expose it as the 5-tuple."""
        configure_bgk = getattr(self, "_configure_bgk", None)
        if configure_bgk is not None:
            configure_bgk(product, pricing_env, sigma, tau)
        market = self.market_snapshot(product, pricing_env)
        request = self.grid_request(product, market, tau)
        if self._session_grids is not None:
            layout = self._layout_from_session_grids(request)
        else:
            layout = self._bound_layout_for_solve(request, market)
        self._populate_observation_maps(product, pricing_env, layout, tau)
        self._active_layout = layout
        return (
            layout.spatial.x,
            layout.spatial.s,
            layout.spatial.dx,
            layout.time.t,
            layout.time.dt,
        )

    def _layout_from_session_grids(self, request: GridRequest) -> Layout:
        """Assemble the layout AROUND injected session grids (no spatial
        rebuild — the expensive node placement was done at preparation).

        The cheap TimeLayout is rebuilt from the request (damping schedules,
        step_of) and verified node-for-node against the injected time grid —
        a stale pack raises instead of silently changing the discretization.
        """
        from quantark.asset.equity.engine.pde.grid.config import resolve_config
        from quantark.asset.equity.engine.pde.grid.space import SpatialLayout
        from quantark.asset.equity.engine.pde.grid.time import build_time

        x_vec, s_vec, dx_vec, t_vec, _ = self._session_grids
        time_layout = build_time(request, self.grid_binder.config)
        if not np.array_equal(time_layout.t, t_vec):
            raise PricingError(
                "injected session grids diverge from the declarative layer "
                "time geometry; re-prepare the session"
            )
        spatial = SpatialLayout(
            s=s_vec,
            x=x_vec,
            dx=dx_vec,
            bounds=(float(s_vec[0]), float(s_vec[-1])),
            achieved_eps=float("nan"),
        )
        return Layout(
            spatial=spatial,
            time=time_layout,
            request=request,
            config_key=self.grid_binder.config.key,
        )

    def _theta_schedule_from_layout(self, layout: Layout) -> np.ndarray:
        """Per-step theta from the layout's damping frozensets (spec §4.5).

        theta = 1.0 on terminal-damped steps (wins on overlap),
        event_theta on event-damped steps, params.theta elsewhere.
        """
        params = self.params
        n = layout.time.actual_steps
        theta = np.full(n, float(params.theta))
        for k in layout.time.event_damping_steps:
            theta[k] = float(getattr(params, "event_theta", 1.0))
        for k in layout.time.terminal_damping_steps:
            theta[k] = 1.0
        return theta


    def _bound_layout_for_solve(
        self, request: GridRequest, market: MarketSnapshot
    ) -> Layout:
        """The layout for this solve: frozen (bump clone) > rebind > bind.

        Alignment-identical requests reuse the frozen layout by identity
        (coverage validated); a changed schedule/tau (calendar roll) rebinds
        the time layout on the SAME spatial object; otherwise a plain bind.
        """
        frozen = self._frozen_base_layout
        if frozen is None:
            return self.grid_binder.bind(request, market)
        fr = frozen.request
        if (
            request.tau == fr.tau
            and request.event_times == fr.event_times
            and request.hard_lower == fr.hard_lower
            and request.hard_upper == fr.hard_upper
        ):
            validate_external_layout(frozen, request, market)
            return frozen
        return self.grid_binder.rebind_time(frozen, request)

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

    def _params_cache_key(self) -> Tuple:
        params: PDEParams = self.params
        # Post-0.4.0: grid geometry is fingerprinted by the resolved
        # GridConfig key; surviving scheme/engine knobs listed explicitly.
        return (
            self.grid_binder.config.key,
            params.grid_size,
            params.time_steps,
            params.cache_strategy,
            params.use_banded_solver,
            params.event_projection,
            params.theta,
            params.event_theta,
            params.use_rannacher,
            params.rannacher_steps,
            params.rannacher_at_events,
            params.event_rannacher_steps,
            params.event_steps_per_day,
            params.boundary_mode,
            params.bus_days_in_year,
            getattr(params, "s_min", 0.0),
            getattr(params, "s_max", 0.0),
        )

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

    def _solve(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        """
        Core PDE solving logic shared by price() and calculate_greeks().

        This method contains all the common grid building, terminal condition setup,
        and time stepping logic. Subclasses can override this method to implement
        custom solving strategies (e.g., two-surface approach for Snowball).

        Args:
            product: The equity product to price
            pricing_env: Pricing environment with market data

        Returns:
            PDESolutionResult containing solution vector and grid data

        Note:
            This method assumes tau > 0 (not expired). Callers should handle
            the expired case before calling _solve().
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        strike = getattr(product, "strike", spot)
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # This PDE solver diffuses with a single strike-selected constant vol;
        # a smile surface is collapsed (skew ignored). Make that explicit.
        from quantark.param.vol.collapse_guard import guard_constant_vol
        guard_constant_vol(
            pricing_env.vol_surface, type(self).__name__,
            strict=getattr(self.params, "strict_smile", False),
        )

        # 1. Discretization
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )

        # 2. Setup System
        grid = np.zeros((len(x_vec), len(t_vec)))
        self.set_terminal_condition(grid, x_vec, s_vec, product, pricing_env)

        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, len(x_vec))
        A = self._build_operator_matrix(l, c, u, len(x_vec))

        # Term-structure step coefficients: one set for flat inputs (exact
        # scalar substitution -> zero behavior change), one per unique
        # forward triple otherwise (designed per-step rebuild).
        sc = self._step_coefficients_for_solve(
            pricing_env, strike, t_vec, dx_vec, len(x_vec)
        )
        sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, len(x_vec))
        step_coeffs = None if sc.n_unique == 1 else sc

        # 3. Solve
        self._time_stepping(
            grid,
            A,
            l,
            u,
            x_vec,
            s_vec,
            t_vec,
            dt_vec,
            product,
            pricing_env,
            r,
            q,
            sigma,
            tau,
            step_coeffs=step_coeffs,
        )

        return PDESolutionResult(
            solution_vec=grid[:, 0],
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=np.log(spot),
        )

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

    def _price_with_solution(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Tuple[float, Optional[PDESolutionResult]]:
        """price()'s preamble + one solve; None solution = short-circuit.

        Native session seam: valid only where ``price`` delegates to it (the
        2D ADI autocallable solvers override ``price`` entirely and never use
        this).
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0:
            return self._calculate_intrinsic(product, spot), None

        result = self._solve(product, pricing_env)
        return (
            self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log),
            result,
        )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """Price the option using the PDE finite difference method."""
        return self._price_with_solution(product, pricing_env)[0]

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
        npv, solution = self._price_with_solution(product, pricing_env)
        return PDESessionOutputs(
            npv=float(npv),
            solution=solution if want_grid else None,
            event_stats=None,
            event_distribution=None,
        )

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """Calculate Delta and Gamma directly from the PDE solution surface."""
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0:
            return {
                "price": self._calculate_intrinsic(product, spot),
                "delta": self._intrinsic_delta(product, spot),
                "gamma": 0.0,
            }

        result = self._solve(product, pricing_env)
        # Valuation-date events: price from the pointwise-exact readout,
        # delta/gamma from the smooth 0+ branch column (a cash transition at
        # the known spot is a constant shift with zero delta/gamma).
        readout_vec = (
            result.readout_vec if result.readout_vec is not None else result.solution_vec
        )
        if result.readout_override is not None:
            price = float(result.readout_override)
        else:
            price = self._interpolate_price(readout_vec, result.x_vec, result.spot_log)
        delta, gamma = self._calculate_delta_gamma(
            readout_vec, result.x_vec, result.spot_log, spot
        )

        return {"price": price, "delta": delta, "gamma": gamma}

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

        result = self._solve(product, pricing_env)
        return self._grid_projection_from_solution(result, spots)

    def _grid_projection_from_solution(
        self,
        result: PDESolutionResult,
        spot_levels: Optional[Sequence[float]] = None,
    ) -> list[dict[str, float | str]]:
        """Project price/delta/gamma from one solved surface (native session
        seam shared with ``calculate_spot_greeks_curve``). ``spot_levels``
        None projects at the grid nodes themselves."""
        prices = np.asarray(result.solution_vec, dtype=float)
        deltas = np.gradient(prices, result.s_vec, edge_order=2)
        gammas = np.gradient(deltas, result.s_vec, edge_order=2)
        spots = np.asarray(
            result.s_vec if spot_levels is None else spot_levels, dtype=float
        )
        return [
            {
                "spot": float(spot),
                "price": float(np.interp(spot, result.s_vec, prices)),
                "delta": float(np.interp(spot, result.s_vec, deltas)),
                "gamma": float(np.interp(spot, result.s_vec, gammas)),
                "calculation_mode": "engine_grid",
            }
            for spot in spots
        ]

    def _build_grids(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the spatial and temporal grids for solving the PDE.

        ONE construction site for every path (value solve, event-stats sweep,
        session preparation and injection): migrated solvers route through the
        declarative layer; injected session grids are verified against the
        layer geometry rather than trusted (a stale pack cannot silently
        change the discretization).
        """
        if self._uses_grid_layer():
            return self._grids_via_layer(
                product, pricing_env, spot, sigma, tau, r, q
            )
        raise PricingError(
            "the legacy grid path was removed with the declarative grid "
            "layer (0.4.0); every PDE solver now binds through GridBinder"
        )

    def _step_coefficients_for_solve(
        self,
        pricing_env: PricingEnvironment,
        ref_strike: float,
        t_vec: np.ndarray,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        """Session-injection dispatch: an injected artifact (PRE-flat-exact;
        the caller still applies _flat_exact_step_coefficients, which is
        deterministic and value-identical) short-circuits the build."""
        if self._session_step_coefficients is not None:
            return self._session_step_coefficients
        return self._build_step_coefficients(
            pricing_env, ref_strike, t_vec, dx_vec, num_x
        )

    def _build_step_coefficients(
        self,
        pricing_env: PricingEnvironment,
        ref_strike: float,
        t_vec: np.ndarray,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        """Sample forward (r, q, sigma) per time step and build operator sets."""
        from quantark.priceenv.term_sampling import TermCoefficients

        memo = _per_object_memo(pricing_env.rate_curve, _ENV_STEP_COEFF_MEMO)
        if memo is not None:
            mkey = (
                float(ref_strike),
                np.asarray(t_vec, dtype=float).tobytes(),
                np.asarray(dx_vec, dtype=float).tobytes(),
                int(num_x),
            )
            cached = memo.get(mkey)
            # identity re-check: replacing div_yield / vol_surface on the env
            # must invalidate (the memo key is only the rate curve)
            if cached is not None and (
                cached[0] is pricing_env.div_yield
                and cached[1] is pricing_env.vol_surface
            ):
                return cached[2]

        tc = TermCoefficients.from_env(
            pricing_env, np.asarray(t_vec, dtype=float), ref_strike=float(ref_strike)
        )
        n_steps = len(t_vec) - 1
        # 12 decimals (same precision as the dt cache keys) absorbs DF
        # round-trip ulp noise so flat curves dedupe to ONE set
        triples = np.column_stack(
            (
                np.round(tc.fwd_rates, 12),
                np.round(tc.fwd_carry, 12),
                np.round(tc.step_vols, 12),
            )
        )
        uniq, set_index = np.unique(triples, axis=0, return_inverse=True)
        n_unique = uniq.shape[0]

        # Vectorized _calculate_coefficients over all unique triples at once
        r_u, q_u, sig_u = uniq[:, 0:1], uniq[:, 1:2], uniq[:, 2:3]  # (n_unique, 1)
        mu = r_u - q_u - 0.5 * sig_u * sig_u
        D = 0.5 * sig_u * sig_u
        dx_vec = np.asarray(dx_vec, dtype=float)
        l = np.zeros((n_unique, num_x))
        c = np.zeros((n_unique, num_x))
        u = np.zeros((n_unique, num_x))
        if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
            dx = dx_vec[0]
            l[:, :] = D / (dx * dx) - mu / (2.0 * dx)
            c[:, :] = -2.0 * D / (dx * dx) - r_u
            u[:, :] = D / (dx * dx) + mu / (2.0 * dx)
        else:
            h_m, h_p = dx_vec[:-1], dx_vec[1:]
            h_sum, h_prod = h_m + h_p, h_m * h_p
            l[:, 1:-1] = 2.0 * D / (h_m * h_sum) - mu * h_p / (h_m * h_sum)
            c[:, 1:-1] = -2.0 * D / h_prod + mu * (h_p - h_m) / h_prod - r_u
            u[:, 1:-1] = 2.0 * D / (h_p * h_sum) + mu * h_m / (h_p * h_sum)
            l[:, 0], c[:, 0], u[:, 0] = l[:, 1], c[:, 1], u[:, 1]
            l[:, -1], c[:, -1], u[:, -1] = l[:, -2], c[:, -2], u[:, -2]

        lcu_sets = [(l[k], c[k], u[k]) for k in range(n_unique)]
        result = StepCoefficients(
            lcu_sets=lcu_sets,
            set_index=np.asarray(set_index, dtype=int).reshape(n_steps),
            n_unique=n_unique,
        )
        if memo is not None:
            if len(memo) >= _STEP_COEFF_MEMO_MAX_ENTRIES:
                memo.pop(next(iter(memo)))  # FIFO eviction
            memo[mkey] = (pricing_env.div_yield, pricing_env.vol_surface, result)
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
        lcu = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        return StepCoefficients(
            lcu_sets=[lcu], set_index=sc.set_index, n_unique=1
        )

    def _calculate_coefficients(
        self, r: float, q: float, sigma: float, dx_vec: np.ndarray, num_x: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate coefficients (l, c, u) for the tridiagonal FD system."""
        mu = r - q - 0.5 * sigma * sigma
        D = 0.5 * sigma * sigma

        if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
            dx = dx_vec[0]
            l = np.full(num_x, D / (dx * dx) - mu / (2.0 * dx))
            c = np.full(num_x, -2.0 * D / (dx * dx) - r)
            u = np.full(num_x, D / (dx * dx) + mu / (2.0 * dx))
        else:
            h_m, h_p = dx_vec[:-1], dx_vec[1:]
            h_sum, h_prod = h_m + h_p, h_m * h_p

            l_diff, c_diff, u_diff = (
                2.0 * D / (h_m * h_sum),
                -2.0 * D / h_prod,
                2.0 * D / (h_p * h_sum),
            )
            l_drift, c_drift, u_drift = (
                -mu * h_p / (h_m * h_sum),
                mu * (h_p - h_m) / h_prod,
                mu * h_m / (h_p * h_sum),
            )

            l, c, u = np.zeros(num_x), np.zeros(num_x), np.zeros(num_x)
            l[1:-1], c[1:-1], u[1:-1] = (
                l_diff + l_drift,
                c_diff + c_drift - r,
                u_diff + u_drift,
            )
            l[0], c[0], u[0] = l[1], c[1], u[1]
            l[-1], c[-1], u[-1] = l[-2], c[-2], u[-2]

        return l, c, u

    def _build_operator_matrix(
        self, l: np.ndarray, c: np.ndarray, u: np.ndarray, num_x: int
    ) -> sp.csc_matrix:
        """Construct the sparse spatial operator matrix A."""
        diagonals = [l[2:-1], c[1:-1], u[1:-2]]
        return sp.diags(
            diagonals, offsets=[-1, 0, 1], shape=(num_x - 2, num_x - 2), format="csc"
        )

    def _time_stepping(
        self,
        grid: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_vec: np.ndarray,
        dt_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        r: float,
        q: float,
        sigma: float,
        tau: float,
        step_coeffs: Optional[StepCoefficients] = None,
    ) -> None:
        """Backward time stepping using the Crank-Nicolson scheme."""
        params: PDEParams = self.params
        num_t, num_x = len(t_vec), len(x_vec)
        I_int = sp.eye(num_x - 2, format="csc")
        self._matrix_cache.clear()
        self._term_A_cache: dict = {}

        # Canonical damping schedule (terminal Rannacher + event smoothing),
        # shared with all other sweeps via BackwardOperator.theta_by_step.
        theta_by_step = self._theta_schedule_from_layout(self._active_layout)

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            theta = float(theta_by_step[j])

            if step_coeffs is not None:
                k = int(step_coeffs.set_index[j])
                l, _c_j, u = step_coeffs.lcu_sets[k]
                A = self._operator_matrix_for_set(step_coeffs, k, num_x)
            else:
                k = 0
            M1, M2_lu = self._get_matrices(I_int, A, dt, theta, coeff_key=k)
            self.set_boundary_conditions(
                grid, x_vec, s_vec, j, tau - t_vec[j], product, pricing_env
            )

            # Solve system for interior points
            rhs = M1 @ grid[1:-1, j + 1]
            self._inject_boundary_contributions(rhs, grid, l, u, j, dt, theta)
            grid[1:-1, j] = M2_lu.solve(rhs)

            self._apply_step_modifications(
                grid, x_vec, s_vec, j, tau - t_vec[j], product, pricing_env
            )

    def _inject_boundary_contributions(self, rhs, grid, l, u, j, dt, theta) -> None:
        """Add Dirichlet boundary terms to the RHS of the reduced interior system."""
        if len(grid) > 2:
            # Terms from V[0]
            rhs[0] += dt * (
                (1.0 - theta) * l[1] * grid[0, j + 1] + theta * l[1] * grid[0, j]
            )
            # Terms from V[-1]
            rhs[-1] += dt * (
                (1.0 - theta) * u[-2] * grid[-1, j + 1] + theta * u[-2] * grid[-1, j]
            )

    def _operator_matrix_for_set(
        self, step_coeffs: StepCoefficients, k: int, num_x: int
    ) -> sp.csc_matrix:
        """Lazy per-unique-set sparse operator (cleared per _time_stepping)."""
        A = self._term_A_cache.get(k)
        if A is None:
            l, c, u = step_coeffs.lcu_sets[k]
            A = self._build_operator_matrix(l, c, u, num_x)
            self._term_A_cache[k] = A
        return A

    def _get_matrices(
        self, I: sp.csc_matrix, A: sp.csc_matrix, dt: float, theta: float,
        coeff_key: int = 0,
    ) -> Tuple[sp.csc_matrix, spla.SuperLU]:
        """
        Get or compute matrices for time stepping.

        Caches LU factorizations for efficiency when dt and theta repeat.

        Args:
            I: Identity matrix
            A: Spatial operator matrix
            dt: Time step size
            theta: Scheme parameter (0.5 = CN, 1.0 = BE)

        Returns:
            Tuple of (M1, M2_lu) where:
                M1 = I + (1-theta)*dt*A (for RHS)
                M2_lu = LU factorization of I - theta*dt*A (for LHS)
        """
        # Round dt to avoid floating point comparison issues
        key = (coeff_key, round(dt, 12), round(theta, 6))

        pack = self._session_matrix_pack
        if pack is not None:
            hit = pack.get(key)
            if hit is not None:
                return hit

        if self._is_cache_enabled() and key in self._matrix_cache:
            return self._matrix_cache[key]

        # Build matrices
        M1 = I + (1.0 - theta) * dt * A
        M2 = I - theta * dt * A

        # LU factorization of M2
        try:
            M2_lu = spla.splu(M2)
        except Exception as e:
            raise NumericalError(f"Failed to factorize matrix: {e}")

        if self._is_cache_enabled():
            self._matrix_cache[key] = (M1, M2_lu)
        return M1, M2_lu

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
        return float(np.interp(x_target, x_vec, v_vec))

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
        # Snap to the grid node nearest x_target (interior only)
        idx = int(np.searchsorted(x_vec, x_target))
        idx = max(1, min(idx, len(x_vec) - 2))
        if idx > 1 and abs(x_vec[idx - 1] - x_target) < abs(x_vec[idx] - x_target):
            idx -= 1

        # Non-uniform three-point stencil (exact for quadratics on ANY local
        # spacing). The symmetric formulas are only valid when h_m == h_p; on
        # the adaptive grid the asymmetry error is proportional to gamma.
        h_m = x_vec[idx] - x_vec[idx - 1]
        h_p = x_vec[idx + 1] - x_vec[idx]
        h_sum = h_m + h_p

        v_m, v_0, v_p = v_vec[idx - 1], v_vec[idx], v_vec[idx + 1]
        dv_dx = (
            -h_p / (h_m * h_sum) * v_m
            + (h_p - h_m) / (h_m * h_p) * v_0
            + h_m / (h_p * h_sum) * v_p
        )
        d2v_dx2 = 2.0 * (
            v_m / (h_m * h_sum) - v_0 / (h_m * h_p) + v_p / (h_p * h_sum)
        )

        # Convert to price-space derivatives AT THE NODE where the stencil
        # was evaluated (identical to `spot` when the spot is a grid node,
        # which is the default; consistent when x_target falls between nodes).
        s_node = float(np.exp(x_vec[idx]))
        delta = dv_dx / s_node
        gamma = (d2v_dx2 - dv_dx) / (s_node**2)

        return delta, gamma

    @staticmethod
    def _current_time(total_tau: float, tau_remaining: float) -> float:
        """Elapsed time (from valuation) at a backward-induction step."""
        return max(total_tau - tau_remaining, 0.0)

    def _df_between_times(
        self, pricing_env: PricingEnvironment, start_time: float, end_time: float
    ) -> float:
        """
        Forward discount factor DF(start_time, end_time), both measured from
        the valuation date. Term-structure consistent: DF(0,T)/DF(0,t), NOT
        exp(-r(tau)*tau) with tau = remaining time (they coincide only under
        a flat curve).

        Memoized per pricing environment (identity-checked, strong ref held):
        the boundary-condition path evaluates the same (t, T) pairs many
        times per step, and term-structure curves make each curve lookup a
        np.interp call — the memo keeps the term/flat cost ratio inside the
        spec's 20% budget.
        """
        if end_time <= start_time:
            return 1.0
        memo = _per_object_memo(pricing_env.rate_curve, _ENV_DF_MEMO)
        if memo is None:
            memo = {}
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
        memo = _per_object_memo(pricing_env.div_yield, _ENV_DF_MEMO)
        if memo is None:
            memo = {}
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

    def _pack_uses_banded(self, num_x: int) -> bool:
        """Whether the backward march for this configuration uses the banded
        solver path (only the two-surface family does)."""
        return False

    def _pack_banded_entry(
        self, banded: dict, step_coeffs, l, c, u, dt, theta, coeff_key
    ) -> None:
        raise NotImplementedError(
            "banded packs exist only where _pack_uses_banded is True"
        )

    def _session_factorization_packs(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment,
        grids, max_entries: Optional[int] = None,
    ) -> Tuple[dict, dict]:
        """Eagerly build the (coeff_key, dt, theta) -> matrices maps one solve
        would lazily build, through the SAME builders the solve uses.

        Numeric factorization reuse is legal only when coefficients, time
        step, theta schedule, and scheme parameters are identical (spec
        section 9.2); the caller keys the published pack by exactly those
        inputs. Returns (matrix_pack, banded_pack) as plain dicts.

        ``max_entries`` bounds the eager build (code-gate finding
        2026-07-16): once the packs hold that many entries, enumeration
        stops and the remaining keys are built lazily per solve — bitwise
        identically, since the march's own construction is the same code.
        The enumeration runs in march order, so the bounded pack holds the
        keys the march requests first.
        """
        x_vec, s_vec, dx_vec, t_vec, dt_vec = grids
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)
        strike = getattr(product, "strike", spot)
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)
        num_x = len(x_vec)

        sc = self._step_coefficients_for_solve(
            pricing_env, strike, t_vec, dx_vec, num_x
        )
        sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, num_x)
        step_coeffs = None if sc.n_unique == 1 else sc
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        A = self._build_operator_matrix(l, c, u, num_x)

        theta_by_step = self._theta_schedule_from_layout(self._active_layout)
        use_banded = self._pack_uses_banded(num_x)
        banded: dict = {}
        saved_matrix, self._matrix_cache = self._matrix_cache, {}
        saved_term_A = getattr(self, "_term_A_cache", {})
        self._term_A_cache = {}
        try:
            I_int = sp.eye(num_x - 2, format="csc")
            for j in range(len(t_vec) - 2, -1, -1):
                if (
                    max_entries is not None
                    and len(self._matrix_cache) + len(banded) >= max_entries
                ):
                    break
                dt = dt_vec[j]
                theta = float(theta_by_step[j])
                if step_coeffs is not None:
                    k = int(step_coeffs.set_index[j])
                    l_j, c_j, u_j = step_coeffs.lcu_sets[k]
                else:
                    k = 0
                    l_j, c_j, u_j = l, c, u
                if use_banded:
                    self._pack_banded_entry(
                        banded, step_coeffs, l_j, c_j, u_j, dt, theta, k
                    )
                else:
                    A_j = (
                        self._operator_matrix_for_set(step_coeffs, k, num_x)
                        if step_coeffs is not None
                        else A
                    )
                    self._get_matrices(I_int, A_j, dt, theta, coeff_key=k)
            matrix_pack = dict(self._matrix_cache)
        finally:
            self._matrix_cache = saved_matrix
            self._term_A_cache = saved_term_A
        return matrix_pack, banded

    def _critical_points_cache_key(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
    ) -> Tuple:
        strategy = self._resolve_cache_strategy()
        return (
            strategy,
            f"{product.__class__.__module__}.{product.__class__.__qualname__}",
            self._product_cache_token(product, strategy),
            round(spot, 12),
            pricing_env.valuation_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
        )

    def _get_cached_critical_points(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
    ) -> Tuple[float, ...]:
        if not self._is_cache_enabled():
            return tuple(
                sorted(
                    [
                        round(p, 12)
                        for p in self.get_critical_points(product, pricing_env)
                        if p is not None
                    ]
                )
            )
        key = self._critical_points_cache_key(product, pricing_env, spot)
        cached = self._critical_points_cache.get(key)
        if cached is not None:
            self._critical_points_cache.move_to_end(key)
            return cached
        points = tuple(
            sorted(
                [
                    round(p, 12)
                    for p in self.get_critical_points(product, pricing_env)
                    if p is not None
                ]
            )
        )
        self._critical_points_cache[key] = points
        self._critical_points_cache.move_to_end(key)
        max_entries = max(1, self._grid_cache_max_entries)
        if len(self._critical_points_cache) > max_entries:
            self._critical_points_cache.popitem(last=False)
        return points
