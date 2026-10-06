"""
PDE solver for Snowball (autocallable) options using the Two-Surface method.

This solver maintains two value surfaces:
- V0: Value when knock-in (KI) has NOT occurred
- V1: Value when knock-in (KI) HAS occurred

The surfaces interact at barrier observation times:
- KO barrier hit: Both surfaces jump to KO payoff (product terminates)
- KI barrier hit: V0 transitions to V1 (V0 <- V1)

For detailed design, see: asset/equity/engine/docs/snowball_pde_engine.md
"""

from collections import OrderedDict
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from scipy.special import ndtr

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))

from quantark.asset.equity.engine.pde.core.adapters.base import continuous_knock_in
from quantark.asset.equity.engine.pde.core.adapters.snowball import SnowballAdapter
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest
from quantark.asset.equity.engine.pde.core.statistics.rows import EdgeRow, RowTerm
from quantark.asset.equity.engine.pde.base_pde_solver import (
    GRID_CACHE_MAX_ENTRIES,
    _CACHE_KEYING,
    BasePDESolver,
    LifeSurfaceSolution,
    PDESessionOutputs,
    PDESolutionResult,
)
from quantark.asset.equity.engine.pde.grid.events import project_event_values
from quantark.asset.equity.engine.pde.grid import (
    EventSchedule,
    GridRequest,
    Layout,
    MarketSnapshot,
    project_between,
)
from quantark.asset.equity.engine.event_stats import AutocallableEventStats, StatisticsProvenance
from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    is_greater_than_or_close,
)



class _ContinuousKIFirstPassage:
    """Per-solve state for the FIRST_PASSAGE continuous-KI correction.

    Per-step nodal application of the KI regime jump monitors the continuous
    barrier only at step boundaries: paths that touch the barrier inside a
    step yet end on the live side are never knocked in, and the live surface
    is biased high by O(sqrt(dt)). ``step_correction`` returns the missing
    value transfer

        C(x) = integral over live y of  K(x,y) p_hit(x,y) [V1 - V0](y) dy

    in closed form under two exact-to-leading-order reductions: the step
    transition K and Brownian-bridge crossing probability p_hit use the SAME
    per-step-constant GBM coefficients as the stepping operator, and
    [V1 - V0] is locally linear at the barrier, lambda * (y - b) in log
    space, with lambda read off the current column at distance ~ sigma
    sqrt(dt) from the barrier. The y-integral is then a Gaussian partial
    moment (reflection principle):

        C(x) = lambda * exp(-2 mu_e a / sig^2) * [m Phi(m/s) + s phi(m/s)]

    with a the log distance to the barrier on the live side, m = mu_e dt - a,
    s = sigma sqrt(dt), and mu_e the drift signed toward the barrier
    (mu for a down barrier, -mu for a reverse/up barrier). No fitted
    constants. Paths ending on the breached side are already captured by the
    nodal mask at the neighbouring column, so only touch-and-return mass is
    added here.
    """

    #: Beyond this many step-widths from the barrier the crossing mass is
    #: below double-precision noise (phi(10) ~ 1e-22); nodes there take zero.
    _BAND_STDS = 10.0

    def __init__(
        self, dt: np.ndarray, mu: np.ndarray, sig2: np.ndarray, is_reverse: bool
    ):
        self._dt = np.asarray(dt, dtype=float)
        self._mu = np.asarray(mu, dtype=float)
        self._sig2 = np.asarray(sig2, dtype=float)
        self._is_reverse = bool(is_reverse)
        self._cache: Dict[tuple, tuple] = {}
        # (n_steps, n_cols) coefficients drive the columnar branch: the 2-D
        # solvers monitor the barrier under the spot dynamics CONDITIONAL on
        # each variance column, so every column carries its own (mu, sigma^2).
        if self._mu.ndim != self._sig2.ndim:
            raise ValidationError(
                "first-passage mu and sigma^2 must share a shape"
            )
        if self._sig2.ndim not in (1, 2):
            raise ValidationError(
                "first-passage coefficients must be per-step or per-step-per-column"
            )
        self._columnar = self._sig2.ndim == 2

    def _geometry(
        self, dt: float, mu: float, sig2: float, s_vec: np.ndarray, barrier: float
    ):
        """Cached (G, k_star): the lambda multiplier profile and the node
        the slope is sampled at. G is zero off the live band."""
        key = (
            round(dt, 12), round(mu, 12), round(sig2, 12),
            round(float(barrier), 12), len(s_vec),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = (None, None, None)
        if dt > 0.0 and sig2 > 0.0 and barrier > 0.0:
            log_s = np.log(np.asarray(s_vec, dtype=float))
            log_b = np.log(float(barrier))
            if self._is_reverse:
                a = log_b - log_s
                mu_e = -mu
            else:
                a = log_s - log_b
                mu_e = mu
            s_step = np.sqrt(sig2 * dt)
            band = np.flatnonzero((a > 0.0) & (a < self._BAND_STDS * s_step))
            if band.size:
                ab = a[band]
                m = mu_e * dt - ab
                t = m / s_step
                bracket = m * ndtr(t) + s_step * np.exp(-0.5 * t * t) / _SQRT_2PI
                G = np.zeros(len(s_vec), dtype=float)
                G[band] = np.exp(-2.0 * mu_e * ab / sig2) * bracket
                live = np.flatnonzero(a > 0.0)
                # lambda is a SLOPE, and the grid has to resolve it. The nodal
                # mask has just set d to zero on the breached side, so on the
                # grid d steps from 0 to its full size across the barrier: a
                # node sitting a fraction of a cell above it reports that step,
                # not the slope, and d/a diverges as the node approaches the
                # barrier. Sample no closer than half a cell -- a resolution
                # floor, not a tuned constant. Barrier-aligned grids (every 1-D
                # layout) already clear it, so their sample is unchanged.
                if live.size > 1:
                    resolved = live[a[live] >= 0.5 * (a[live[1]] - a[live[0]])]
                    if resolved.size:
                        live = resolved
                k_star = int(live[np.argmin(np.abs(a[live] - s_step))])
                result = (G, k_star, float(a[k_star]))
        self._cache[key] = result
        return result

    def step_correction(
        self, t_idx: int, s_vec: np.ndarray, barrier: float, d: np.ndarray
    ) -> np.ndarray:
        """Correction to ADD to the live surface at the current column.

        ``d`` is V1 - V0 at the current column AFTER the nodal mask (so it is
        zero on the breached side); shape (n_x,) or (n_x, n_cols) -- the
        event-stats sweep passes every indicator column at once, since the
        indicator expectations follow the same regime dynamic programming.
        """
        dt = float(self._dt[t_idx])
        if self._columnar:
            return self._columnar_correction(dt, t_idx, s_vec, barrier, d)
        G, k_star, dist = self._geometry(
            dt, float(self._mu[t_idx]), float(self._sig2[t_idx]), s_vec, barrier
        )
        if G is None:
            return np.zeros_like(d)
        lam = d[k_star] / dist
        if d.ndim == 2:
            return G[:, None] * np.asarray(lam)[None, :]
        return G * lam

    def add_correction(self, t_idx: int, s_vec: np.ndarray, barrier: float, v0: np.ndarray,
                       v1: np.ndarray) -> None:
        """``v0 += step_correction(t_idx, s_vec, barrier, v1 - v0)`` in place. On a statistics block
        with per-step coefficients only the ``k_star`` row of ``v1 - v0`` is read, so only that row
        is formed (the same subtraction, so the same bits)."""
        if self._columnar or v0.ndim != 2:
            v0 += self.step_correction(t_idx, s_vec, barrier, v1 - v0)
            return
        G, k_star, dist = self._geometry(
            float(self._dt[t_idx]), float(self._mu[t_idx]), float(self._sig2[t_idx]), s_vec, barrier
        )
        if G is None:
            v0 += 0.0                       # step_correction's zeros_like(d)
            return
        lam = (v1[k_star] - v0[k_star]) / dist
        v0 += G[:, None] * np.asarray(lam)[None, :]

    def _columnar_correction(
        self,
        dt: float,
        t_idx: int,
        s_vec: np.ndarray,
        barrier: float,
        d: np.ndarray,
    ) -> np.ndarray:
        """One (mu, sigma^2) per column, evaluated through the SAME closed form.

        A 2-D (log-spot, variance) solver knows the spot's instantaneous
        variance exactly -- it is the column's own variance node, scaled by the
        leverage at the barrier for SLV -- so the crossing geometry is rebuilt
        per column rather than shared. Freezing the variance across the step
        neglects its own diffusion, an O(dt) error in sigma^2 and hence
        O(dt^1.5) in a correction that repairs O(sqrt(dt)); it is the same
        conditional-Gaussian reduction the Monte Carlo bridge makes. Columns
        are looped rather than broadcast so the arithmetic is bit-for-bit the
        scalar path's, which is what keeps the certified 1-D engines inert.
        """
        if d.ndim != 2:
            raise ValidationError(
                "columnar first-passage coefficients need a (n_x, n_cols) surface"
            )
        mu_row = np.asarray(self._mu[t_idx], dtype=float)
        sig2_row = np.asarray(self._sig2[t_idx], dtype=float)
        if mu_row.size != d.shape[1]:
            raise ValidationError(
                "first-passage columns must match the surface's column count"
            )
        out = np.zeros_like(d)
        for col in range(d.shape[1]):
            G, k_star, dist = self._geometry(
                dt, float(mu_row[col]), float(sig2_row[col]), s_vec, barrier
            )
            if G is None:
                continue
            out[:, col] = G * (d[k_star, col] / dist)
        return out


class SnowballPDESolver(BasePDESolver):
    """
    PDE solver for Snowball (autocallable) options using the Two-Surface method.

    Maintains two price grids to track the knock-in state:
        - grid_v0: Value surface for "not knocked-in" state (receives rebate at maturity)
        - grid_v1: Value surface for "knocked-in" state (has downside exposure)

    The solver handles:
        - Discrete KO observations with time-varying barriers and rates
        - Continuous or discrete KI monitoring
        - INSTANT or EXPIRY coupon payment timing
        - Standard and reverse snowball structures
        - Airbag and protection features (via product payoff methods)

    Algorithm:
        1. Initialize both grids with terminal conditions at maturity
        2. Step backward in time using Crank-Nicolson
        3. At KO observation times: apply KO payoff to breached regions
        4. At KI observation times (or every step for continuous): V0 <- V1
           in the breached region; for continuous monitoring the live region
           additionally receives the closed-form intra-step crossing
           correction (first passage, always on) so the
           scheme prices the continuously monitored barrier rather than
           step-width discrete monitoring
        5. Interpolate final price from V0 (or V1 if already knocked-in)
    """

    # Subclasses can override this to specify their supported product type
    _supported_product_type: type = SnowballOption
    _solver_name: str = "SnowballPDESolver"
    #: FIRST_PASSAGE continuous-KI correction eligibility. The closed form
    #: reduces the crossing to a per-step-constant GBM, so a solver is
    #: eligible once it can report those coefficients AT THE BARRIER --
    #: which every solver can, since the correction never looks further than
    #: a few step-widths from it (see _first_passage_step_coefficients).
    #: Kept as a hook for solvers whose barrier treatment must stay pinned.
    _first_passage_ki_supported: bool = True
    settlement_support = SettlementSupport.EVENT_AND_TERMINAL
    _adapter_class = SnowballAdapter

    def __init__(
        self, params: Optional[PDEParams] = None, enable_profiling: bool = False
    ):
        """
        Initialize Snowball PDE solver.

        Args:
            params: PDE engine configuration parameters
            enable_profiling: Enable timing breakdown for matrix, RHS, solve, barrier
        """
        super().__init__(params)

        # Two-surface grids
        self._grid_v0: Optional[np.ndarray] = None
        self._grid_v1: Optional[np.ndarray] = None

        # KO observation tracking
        self._ko_observation_indices: Dict[int, ResolvedObservationRecord] = {}
        self._ko_terminal_record: Optional[ResolvedObservationRecord] = None
        self._has_terminal_ko: bool = False

        # KI observation tracking
        self._ki_observation_indices: Set[int] = set()
        self._ki_barrier_by_tidx: Dict[int, float] = {}
        self._ki_continuous: bool = False
        self._ki_barrier: float = 0.0
        # FIRST_PASSAGE continuous-KI correction state, rebuilt per solve.
        self._ki_fp: Optional[_ContinuousKIFirstPassage] = None
        self._is_reverse: bool = False

        self._active_schedule: Optional[EventSchedule] = None

        # Time tracking
        self._total_tau: float = 0.0
        self._structured_terminal_delay_df: float = 1.0
        self._profile_enabled = enable_profiling
        self._profile_stats: Dict[str, float] = {}
        self._ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = OrderedDict()
        self._ki_profile_cache: "OrderedDict[Tuple, Dict[str, List[Optional[float]]]]" = OrderedDict()
        # Environment for the NEXT grid_request. grid_request takes only (product, market,
        # tau), but a date-based observation schedule can only be turned into year fractions
        # against a pricing environment, so the prepare step stashes it here.
        self._grid_request_env: Optional[PricingEnvironment] = None
        # Per-solve memo of _product_cache_token: the token is a pure function
        # of the product's serialized state, which cannot change inside one
        # solve, but its construction re-serializes every KO/KI schedule record
        # and dominated the stepping loop (~2 rebuilds per time step via the
        # boundary-condition path). Cleared at each solve entry so a product
        # mutated (or GC'd and id-reused) between solves can never hit a stale
        # token.
        self._product_token_memo: Dict[Tuple[int, str], Tuple[str, object]] = {}

    def _constructor_kwargs(self) -> dict:
        return {**super()._constructor_kwargs(), "enable_profiling": self._profile_enabled}

    def enable_profiling(self, enabled: bool = True) -> None:
        """Toggle internal timing breakdown collection."""
        self._profile_enabled = enabled

    def get_profile_stats(self) -> Dict[str, float]:
        """Return timing breakdown from the most recent solve."""
        return dict(self._profile_stats)

    def _reset_profile_stats(self) -> None:
        self._profile_stats = {
            "grid_build": 0.0,
            "boundary": 0.0,
            "matrix_build": 0.0,
            "rhs": 0.0,
            "solve": 0.0,
            "barrier": 0.0,
        }

    def _prepare_solve_state(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> bool:
        """_solve's state preamble (KI regime, valuation flags, KI barrier),
        shared with session preparation so grid-key evaluation on a fresh
        clone sees the same state a direct solve would. Returns
        knocked_in_at_valuation."""
        spot = pricing_env.spot
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )
        # Store the state for potential use in calculate_greeks
        self._knocked_in_at_valuation = knocked_in_at_valuation
        self._is_reverse = product.is_reverse
        self._ki_continuous = ki_continuous
        self._ki_fp = None  # per-solve; the time-stepping loop rebuilds it
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                self._ki_barrier = ki_barrier[0]
            else:
                self._ki_barrier = ki_barrier
        self._structured_terminal_delay_df = self._terminal_delay_df(
            product, pricing_env
        )
        return knocked_in_at_valuation

    def _prepare_continuous_ki_correction(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        t_vec: np.ndarray,
    ) -> None:
        """Build the per-solve FIRST_PASSAGE state consumed by _apply_ki_jump,
        the schedule's continuous stage, and the event-stats sweep.

        The per-step (dt, mu, sigma^2) come from the same forward-coefficient
        sampling the stepping operator uses, so the correction is exactly
        consistent with the discretized dynamics. Stays ``None`` (inert) for
        discrete KI, for products without a KI barrier, and under the NONE
        legacy opt-out.
        """
        self._ki_fp = None
        if not self._first_passage_ki_supported:
            return
        if not product.has_ki_barrier:
            return
        if not self._ki_continuous:
            return
        t = np.asarray(t_vec, dtype=float)
        if t.size < 3:
            return
        barrier = float(self._ki_barrier)
        mu, sig2 = self._first_passage_step_coefficients(
            product, pricing_env, t, barrier
        )
        self._ki_fp = _ContinuousKIFirstPassage(
            dt=np.diff(t),
            mu=mu,
            sig2=sig2,
            is_reverse=bool(product.is_reverse),
        )

    def _first_passage_step_coefficients(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        t_vec: np.ndarray,
        barrier: float,
    ):
        """Per-step ``(mu, sigma^2)`` governing the barrier crossing.

        Sampled from the same forward coefficients the stepping operator uses,
        so the correction is exactly consistent with the discretized dynamics.
        The closed form only ever looks in a band a few step-widths either side
        of ``barrier``, so solvers whose diffusion varies in space or state
        override this to report the dynamics THERE rather than disabling the
        correction outright.
        """
        from quantark.priceenv.term_sampling import TermCoefficients

        tc = TermCoefficients.from_env(
            pricing_env, t_vec, ref_strike=float(product.strike)
        )
        sig2 = tc.step_vols * tc.step_vols
        return tc.fwd_rates - tc.fwd_carry - 0.5 * sig2, sig2

    def _check_product_type(self, product: BaseEquityProduct) -> None:
        """
        Check that the product is of the supported type for this solver.

        Subclasses can override _supported_product_type and _solver_name class
        attributes to customize the type check.

        Raises:
            PricingError: If product is not of the supported type
        """
        if not isinstance(product, self._supported_product_type):
            raise PricingError(
                f"{self._solver_name} only supports {self._supported_product_type.__name__}, "
                f"got {type(product).__name__}"
            )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a Snowball option using the Two-Surface PDE method.

        Args:
            product: SnowballOption to price
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a SnowballOption
            ValidationError: If product configuration is incompatible with PDE
        """
        return self._run(product, pricing_env, OutputRequest(Operation.PRICE))

    def _spot_curve_solution(self, product, pricing_env):
        # the solved surface alone: no valuation-date short-circuit (as _solve had none)
        return self._run(product, pricing_env,
                         OutputRequest(Operation.SESSION_OUTPUTS, grid=True, short_circuit=False)).solution

    def _session_outputs(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        want_events: bool = False,
        want_grid: bool = False,
        streams: Optional[frozenset] = None,
    ) -> PDESessionOutputs:
        """One solve serving PV + event stats + grid projection.

        The event-stat indicator columns march as the statistics group of the
        same march as the value surfaces (spec 2 §5.4); a family whose statistics
        are not a PDE group (KO-reset) runs the value march, then its own
        ``calculate_event_stats`` with the npv supplied.
        """
        from quantark.cashleg.event_distribution import EventDistribution

        in_march = want_events and self.adapter.supports_statistics_group
        value = self._run(product, pricing_env,
                          OutputRequest(Operation.SESSION_OUTPUTS, grid=True,
                                        event_stats=in_march, streams=streams))
        npv, solution, stats = value.npv, value.solution, value.event_stats
        if want_events and not in_march and solution is not None:
            stats = self.calculate_event_stats(
                product, pricing_env, npv=float(npv), streams=streams
            )
        dist = None if stats is None else EventDistribution.from_autocallable_stats(stats)
        return PDESessionOutputs(
            npv=float(npv),
            solution=solution if want_grid else None,
            event_stats=stats,
            event_distribution=dist,
        )

    def calculate_event_stats(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        npv: Optional[float] = None,
        streams: Optional[frozenset] = None,
    ) -> Optional[AutocallableEventStats]:
        """Provide per-observation KO probabilities and expected discounted cashflows.

        ``npv`` / ``streams`` support the single-pass ``price_with_events`` path
        (skip the internal value solve; prune unrequested indicator columns).
        """
        if not isinstance(product, self._event_stats_product_type()):
            return None
        if pricing_env is None:
            return None
        # the value group marches alongside unless the caller supplied the npv; a product knocked
        # out at valuation is the adapter's terminal outcome (spec 3B §5.5, decision 15)
        return self._run(product, pricing_env, OutputRequest(
            Operation.EVENT_STATS, price=npv is None, event_stats=True,
            external_pv=None if npv is None else float(npv), streams=streams))

    def price_with_events(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        emit_distribution: bool = True,
        streams: Optional[frozenset] = None,
    ) -> "PricingResult":
        """Single-pass NPV + event distribution [§11.2, §11.3].

        Replicates ``price()``'s expired / immediate-KO short-circuits before any
        combined solve, then runs the value sweep once and reuses its NPV for the
        event-distribution residual (no internal re-price), pruning indicator
        columns to ``streams``.
        """
        from quantark.cashleg.event_distribution import EventDistribution, PricingResult

        out = self._session_outputs(
            product, pricing_env, want_events=emit_distribution, streams=streams
        )
        if out.event_distribution is not None:
            return PricingResult(
                npv=out.npv, event_distribution=out.event_distribution
            )
        # [§11.3] degenerate (maturity-only) distribution: expired /
        # immediate-KO short-circuits, emit_distribution=False, or a product
        # type without event stats.
        tau = product.get_maturity(pricing_env)
        return PricingResult(
            npv=out.npv,
            event_distribution=EventDistribution.trivial(max(float(tau), 0.0)),
        )

    def _event_stats_product_type(self) -> type:
        """Product type accepted by ``calculate_event_stats`` (overridable)."""
        return SnowballOption

    def _make_event_stats(self, **fields) -> AutocallableEventStats:
        """Construct the event-stats dataclass (overridable by subclasses), with the native
        implementation provenance (spec 3B §5.6) unless the caller set one."""
        fields.setdefault("provenance", self._native_provenance())
        return AutocallableEventStats(**fields)

    def _native_provenance(self) -> StatisticsProvenance:
        """The family's own statistics model, priced by the same model (spec 3B §5.6)."""
        model = self.adapter.statistics_model
        return StatisticsProvenance(model, model)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a Snowball option using Two-Surface PDE method.

        Price from the pointwise-exact readout price() uses; delta/gamma from
        the smooth 0+ branch column (a valuation-date observation writes a
        value JUMP onto the grid, so differentiating the post-event vector
        across it is O(J/h) wrong and worsens under refinement).

        Raises:
            PricingError: If product is not a SnowballOption
            ValidationError: If product configuration is incompatible with PDE
        """
        return self._run(product, pricing_env, OutputRequest(Operation.GREEKS, greeks=True))

    def solve_life_surface(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment, *, extra_times: Sequence[float]
    ) -> LifeSurfaceSolution:
        """One two-surface solve with ``extra_times`` as grid nodes; both slabs returned.

        The solve is the ordinary one (same grid layer, same schedule, same
        stepping); the extra nodes carry no event and no damping. The nodes
        travel in the request, so a later ``price`` is byte-identical to one on
        a fresh solver.
        """
        nodes = tuple(sorted(float(t) for t in extra_times))
        return self._run(product, pricing_env,
                         OutputRequest(Operation.LIFE_SURFACE, life_surface=True, extra_times=nodes))

    def _validate_product(self, product: SnowballOption) -> None:
        """
        Validate that product configuration is compatible with PDE solver.

        Args:
            product: SnowballOption to validate

        Raises:
            ValidationError: If configuration is incompatible
        """
        # Check for continuous KI with non-scalar barrier
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if ki_continuous and product.has_ki_barrier:
            if isinstance(product.barrier_config.ki_barrier, list):
                raise ValidationError(
                    "Continuous KI monitoring requires scalar ki_barrier. "
                    "Use discrete monitoring for time-varying KI barriers."
                )

    @staticmethod
    def _filter_observations_by_tau(
        records: List[ResolvedObservationRecord], tau: float
    ) -> List[ResolvedObservationRecord]:
        """
        Filter and sort observation records within [0, tau] range.

        This utility consolidates the repeated observation filtering pattern.

        Args:
            records: List of resolved observation records
            tau: Time to maturity (upper bound for filtering)

        Returns:
            Sorted list of records with observation_time in [0, tau]
        """
        filtered = [
            rec for rec in records
            if rec.observation_time is not None and 0.0 <= rec.observation_time <= tau
        ]
        filtered.sort(key=lambda rec: float(rec.observation_time))
        return filtered

    @staticmethod
    def _get_barrier_mask(
        s_vec: np.ndarray, barrier: float, is_reverse: bool, is_up_barrier: bool = True
    ) -> np.ndarray:
        """
        Get boolean mask for grid points that breach a barrier.

        This helper consolidates the barrier mask logic used throughout the solver.

        Args:
            s_vec: Array of spot prices on the grid
            barrier: Barrier level
            is_reverse: True for reverse snowball (inverts barrier direction)
            is_up_barrier: True for UP barrier (KO), False for DOWN barrier (KI)

        Returns:
            Boolean mask where True indicates barrier is breached

        Logic:
            - Standard UP KO: mask = s_vec >= barrier
            - Reverse UP KO: mask = s_vec <= barrier (inverted)
            - Standard DOWN KI: mask = s_vec <= barrier
            - Reverse DOWN KI: mask = s_vec >= barrier (inverted)
        """
        if is_up_barrier:
            # UP barrier (typically KO)
            if is_reverse:
                return s_vec <= barrier  # DOWN for reverse
            else:
                return s_vec >= barrier  # UP for standard
        else:
            # DOWN barrier (typically KI)
            if is_reverse:
                return s_vec >= barrier  # UP for reverse
            else:
                return s_vec <= barrier  # DOWN for standard

    def _use_cell_average_events(self) -> bool:
        """True when discrete event jumps use the dual-cell projection.

        ``None`` is this route's own treatment: the dual-cell projection.
        """
        return self.params.event_projection in (None, EventProjectionMode.CELL_AVERAGE)

    def _event_uses_projection(self, t_idx: int) -> bool:
        """True when the discrete event at ``t_idx`` uses the projection.

        Observations falling exactly on the valuation date (``t_idx == 0``)
        are deterministic — today's spot is known — so their trigger is
        applied with the product's exact inclusive comparison instead of a
        cell average [2026-07-23 review, finding 2].
        """
        return t_idx != 0 and self._use_cell_average_events()

    def _reset_t0_readout_state(self) -> None:
        """Per-solve valuation-date readout state (see PDESolutionResult).

        ``_t0_pre_event_cols``: smooth 0+ branch columns (v0, v1) captured
        before the t=0 events are written onto the grid. ``_t0_readout_cols``
        / ``_t0_readout_values`` refine them when a t=0 cash transition
        resolves at the actual spot (phoenix coupons).
        """
        self._t0_pre_event_cols = None
        self._t0_readout_cols = None
        self._t0_readout_values = None

    def _t0_has_events(self, product) -> bool:
        """Discrete events registered on the valuation date (t_idx == 0).

        Continuous KI is a continuous-barrier treatment whose value
        function is continuous at the barrier — there is no t=0 jump to
        shield the readout from — so it does not count. Legacy nodal mode
        is the pinned characterization discretization and keeps the raw
        post-event interpolation bitwise.
        """
        if not self._use_cell_average_events():
            return False
        if 0 in self._ko_observation_indices:
            return True
        if (
            product.has_ki_barrier
            and not self._ki_continuous
            and 0 in self._ki_observation_indices
        ):
            return True
        return False

    def _compose_t0_readout(self, sel: int):
        """(readout_vec, readout_override) for the selected surface (0=v0, 1=v1)."""
        readout_vec = None
        readout_override = None
        if self._t0_readout_cols is not None:
            readout_vec = self._t0_readout_cols[sel]
        elif self._t0_pre_event_cols is not None:
            readout_vec = self._t0_pre_event_cols[sel]
        if self._t0_readout_values is not None:
            readout_override = float(self._t0_readout_values[sel])
        return readout_vec, readout_override

    def _event_nodal_mask(
        self,
        s_vec: np.ndarray,
        barrier,
        is_reverse: bool,
        is_up_barrier: bool,
        at_valuation: bool = False,
    ) -> np.ndarray:
        """Breach mask for a nodal event application.

        For a valuation-date observation in cell-average mode the mask
        additionally owns nodes within ``is_close`` tolerance of the barrier:
        the observation is deterministic and inclusive, and grid nodes are
        ``exp(log(.))`` round-trips, so a raw ``>=``/``<=`` would let 1-ULP
        noise decide ownership. Legacy nodal mode keeps the raw comparison
        (it is the pinned characterization discretization).
        """
        mask = self._get_barrier_mask(s_vec, barrier, is_reverse, is_up_barrier)
        if at_valuation and self._use_cell_average_events():
            b = float(barrier) if barrier is not None else 0.0
            if b > 0.0:
                mask = mask | np.isclose(
                    np.asarray(s_vec, dtype=float),
                    b,
                    rtol=Tolerance.RELATIVE,
                    atol=0.0,
                )
        return mask

    def _project_event_values(
        self,
        s_vec: np.ndarray,
        barrier: Optional[float],
        is_reverse: bool,
        is_up_barrier: bool,
        v_survive,
        v_breach,
    ) -> np.ndarray:
        """Post-event nodal values for a discrete binary event.

        Conservative dual-cell average of the complete event function in
        log-price space; the breach direction matches ``_get_barrier_mask``
        exactly (including reverse products and the barrier<=0 degenerate
        cases). ``v_survive``/``v_breach`` may be scalars or arrays of shape
        (n,) or (n, k).
        """
        s_vec = np.asarray(s_vec, dtype=float)
        x_vec = np.log(s_vec)
        b = float(barrier) if barrier is not None else 0.0
        b_x = np.log(b) if b > 0.0 else -np.inf
        breach_up = bool(is_up_barrier) != bool(is_reverse)
        return project_event_values(x_vec, b_x, v_survive, v_breach, breach_up)

    @staticmethod
    def _record_is_non_negative_time(record: ResolvedObservationRecord) -> bool:
        """Return True if record's time is >= 0 (within numerical tolerance)."""
        return is_greater_than_or_close(record.observation_time, 0.0)

    @staticmethod
    def _find_record_at_time(
        records: List[ResolvedObservationRecord], target_time: float
    ) -> Optional[ResolvedObservationRecord]:
        """Find an observation record at a specific time (within tolerance)."""
        for rec in records:
            if is_close(rec.observation_time, target_time):
                return rec
        return None

    def _is_already_knocked_in(self, product: SnowballOption, spot: float) -> bool:
        """Check if spot is in the knocked-in region (spot-only proxy)."""
        if not product.has_ki_barrier:
            return False

        ki_barrier = product.barrier_config.ki_barrier
        if isinstance(ki_barrier, list):
            ki_barrier = ki_barrier[0]

        if product.is_reverse:
            return spot >= ki_barrier  # UP KI for reverse
        else:
            return spot <= ki_barrier  # DOWN KI for standard

    def _is_knocked_in_at_valuation(
        self,
        product: SnowballOption,
        spot: float,
        pricing_env: PricingEnvironment,
        ki_continuous: bool,
    ) -> bool:
        """
        Determine KI state at valuation date (t=0).

        For continuous monitoring, a barrier breach at valuation implies immediate KI.
        For discrete monitoring, a barrier breach only matters if there is a KI observation at t=0.
        """
        if getattr(product, "_otc_lifecycle_knocked_in", False):
            return True
        if not product.has_ki_barrier:
            return False

        if ki_continuous:
            return self._is_already_knocked_in(product, spot)

        ki_records = product.resolve_ki_observations(pricing_env)
        ki_record_0 = self._find_record_at_time(ki_records, 0.0)
        if ki_record_0 is None:
            return False
        if ki_record_0.barrier is None:
            raise ValidationError(
                "KI observation at valuation requires a barrier level."
            )

        if product.is_reverse:
            return spot >= ki_record_0.barrier
        return spot <= ki_record_0.barrier

    def _valuation_decision(self, product, pricing_env: PricingEnvironment):
        """The family's own decision at the valuation instant (spec 3B decision 15): the observations
        scheduled at time zero applied to the known spot, from the carried knock-in state."""
        carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        return product.decide_observations_at_valuation(pricing_env.spot, pricing_env, knocked_in=carried)

    def _knocked_out_value(self, product, pricing_env: PricingEnvironment, decision) -> float:
        """The value of a knock-out the valuation-date decision fired (the immediate knock-out payoff)."""
        return self._get_immediate_ko_payoff(product, pricing_env)

    def _get_immediate_ko_payoff(
        self, product: SnowballOption, pricing_env: PricingEnvironment
    ) -> float:
        """Get KO payoff when valuation date is a KO observation and KO is triggered."""
        ko_records = product.resolve_ko_observations(pricing_env)
        ko_record_0 = self._find_record_at_time(ko_records, 0.0)
        if ko_record_0 is None:
            raise ValidationError(
                "Immediate KO payoff requested but no KO observation exists at valuation date."
            )

        payoff = ko_record_0.payoff if ko_record_0.payoff is not None else 0.0
        settlement_time = ko_record_0.settlement_time
        if settlement_time is not None and settlement_time > 0.0:
            df = pricing_env.get_discount_factor(settlement_time)
            return float(payoff) * float(df)
        return float(payoff)

    def _calculate_terminal_value(
        self, product: SnowballOption, spot: float, pricing_env: PricingEnvironment
    ) -> float:
        """Value at zero time to maturity: the observations still pending at this instant are decided on the spot.

        The shortcut used to go straight to the terminal payoff, so a knock-out decided at the last observation
        was paid as a survival (the rebate instead of the knock-out coupon). Everywhere else the solver already
        decides an observation at the valuation instant on the known spot; this is the same decision.

        The knock-in state is the decision's: the carried one, or a knock-in observed now (a continuously monitored
        level is observed now by the decision itself). A spot-only level test knocked in a claim whose discrete
        knock-in schedule had already ended. The redemption is cash at its payment time, not at this instant.
        """
        carried = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        if type(product) is SnowballOption:
            decision = product.decide_observations_at_valuation(spot, pricing_env, knocked_in=carried)
            if decision.knocked_out:
                return self._get_immediate_ko_payoff(product, pricing_env)
            carried = decision.knocked_in
        else:
            decided, carried = self._decide_variant_at_valuation(product, spot, pricing_env, carried)
            if decided is not None:
                return decided
        redemption = product.get_payoff(spot, pricing_env, knocked_in=carried)
        return float(redemption) * self._terminal_delay_df(product, pricing_env)

    def _decide_variant_at_valuation(
        self, product, spot: float, pricing_env: PricingEnvironment, carried: bool
    ) -> Tuple[Optional[float], bool]:
        """Hook for a snowball variant whose pending observations need their own decision at zero
        maturity: (the value of a knock-out decided now, else ``None``; the knock-in state after the
        instant). The base keeps the carried state."""
        return None, carried

    @staticmethod
    def _ko_survives_ki(product: SnowballOption) -> bool:
        """Whether a knock-out can still fire once the trade has knocked in.

        ``disable_ko_after_ki`` says it cannot, and the two-surface solver
        expresses "knocked in" as the V1 surface -- so honouring the flag means
        writing the KO payoff to V0 alone and leaving V1 to run on unbarriered.
        The flag is meaningless without a KI barrier, which is the guard the
        Monte Carlo engine applies too.
        """
        return not (
            product.barrier_config.disable_ko_after_ki and product.has_ki_barrier
        )

    def _resolve_ki_barrier_at_tidx(self, t_idx: int) -> float:
        """Resolve KI barrier for a specific PDE time index."""
        if not self._ki_continuous:
            mapped = self._ki_barrier_by_tidx.get(t_idx)
            if mapped is not None:
                return float(mapped)
        return float(self._ki_barrier)

    def _stash_grid_request_env(self, pricing_env: PricingEnvironment) -> None:
        """Record the environment ``grid_request`` resolves date schedules against.

        The last hook the base solver calls before ``grid_request`` (which
        takes no environment), so a reused solver never resolves one solve's
        observation times against the previous solve's environment.
        """
        self._grid_request_env = pricing_env

    def _ki_nodes_in_grid(self, product: BaseEquityProduct) -> bool:
        """Whether the time grid carries the interior daily-KI nodes.

        SINGLE source of truth for a decision two places must agree on:
        ``grid_request`` declares the grid, and the observation maps demand
        that every KI observation land on a node.  A regime that drops the nodes
        but still demands them raises ``ValidationError`` at pricing time, so
        both read this predicate rather than re-deriving the condition.

        The nodes are dropped when the product is already knocked in —
        monitoring is moot. The readout comes off the V1 surface, and V1
        never sees a KI jump (the jump writes V1 values INTO V0), so the KI
        indices cannot affect the answer.
        """
        if getattr(product, "_otc_lifecycle_knocked_in", False):
            return False
        return True

    def _aligned_time_index(
        self, t_vec: np.ndarray, obs_time: float, label: str
    ) -> int:
        for idx, t_val in enumerate(t_vec):
            if is_close(float(t_val), float(obs_time), abs_tol=Tolerance.PRECISION):
                return int(idx)
        nearest = int(np.argmin(np.abs(t_vec - obs_time)))
        nearest_time = float(t_vec[nearest])
        raise ValidationError(
            f"{label} time {obs_time} does not align with PDE time grid "
            f"(nearest {nearest_time}). Use event-aligned time grid or increase time steps."
        )

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.

        For Snowball options, includes:
        - Strike price
        - All KO barriers
        - KI barrier

        Args:
            product: SnowballOption
            pricing_env: Pricing environment

        Returns:
            List of critical price levels
        """
        points = []

        if hasattr(product, "strike") and product.strike > 0:
            points.append(product.strike)

        if hasattr(product, "initial_price") and product.initial_price > 0:
            points.append(product.initial_price)

        # Add KO barriers
        ko_barrier = product.barrier_config.ko_barrier
        if isinstance(ko_barrier, list):
            points.extend([b for b in ko_barrier if b > 0])
        elif ko_barrier > 0:
            points.append(ko_barrier)

        # Add KI barrier
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                points.extend([b for b in ki_barrier if b > 0])
            elif ki_barrier > 0:
                points.append(ki_barrier)

        # Add airbag barrier if present
        if product.airbag_config.airbag_barrier is not None:
            points.append(product.airbag_config.airbag_barrier)

        return sorted(set([p for p in points if p > 0]))

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Collect all barrier levels for spatial grid construction."""
        barriers = []

        if hasattr(product, "barrier_config"):
            # KO barriers
            ko_barrier = product.barrier_config.ko_barrier
            if isinstance(ko_barrier, list):
                barriers.extend(ko_barrier)
            elif ko_barrier > 0:
                barriers.append(ko_barrier)

            # KI barrier
            if product.barrier_config.ki_barrier is not None:
                ki_barrier = product.barrier_config.ki_barrier
                if isinstance(ki_barrier, list):
                    barriers.extend(ki_barrier)
                elif ki_barrier > 0:
                    barriers.append(ki_barrier)

        return barriers

    def _ko_coupon_align_times(
        self, product: BaseEquityProduct, tau: float
    ) -> List[float]:
        """KO (and Phoenix coupon, same dates) observation times.

        These MUST be grid nodes exactly — they drive the value KO jumps and the
        event-distribution resets, so a misalignment here is a correctness bug.

        A schedule stated in DATES carries no year fraction of its own: only the
        pricing environment can supply one. Those schedules are therefore resolved
        through the same cached resolver ``_populate_observation_maps`` uses, so
        grid geometry and event indices come from one source. Reading the raw
        config alone silently yielded no nodes for a date-based schedule, and the
        map then looked up a time the grid had never been told about.
        """
        out = []
        cfg = getattr(product, "barrier_config", None)
        if cfg is not None:
            sched = cfg.ko_observation_schedule
            if sched is not None and sched.uses_dates():
                out += [
                    rec.observation_time
                    for rec in self._resolved_ko_records_for_grid(product, "knock-out")
                ]
            elif sched is not None:
                out += [
                    rec.observation_time
                    for rec in sched.records
                    if rec.observation_time is not None
                ]
            elif cfg.ko_observation_dates is not None:
                out += list(cfg.ko_observation_dates)
        return sorted({float(t) for t in out if t is not None and 0.0 < float(t) < tau})

    def _grid_request_environment(self, what: str):
        """The environment stashed by the prepare step, or a loud failure.

        ``grid_request`` cannot take one (it is a frozen geometry declaration
        shared by every solver), so the caller must have run
        ``_stash_grid_request_env`` first; the adapter's solve state does
        (``core.prepare.request_geometry`` for a caller outside a solve).
        """
        env = self._grid_request_env
        if env is None:
            raise ValidationError(
                f"a date-based {what} observation schedule needs a pricing environment to "
                "resolve its observation times, but the grid was requested without one: "
                "resolve it through core.prepare.request_geometry before grid_request"
            )
        return env

    def _resolved_ko_records_for_grid(self, product, what: str):
        return self._get_cached_ko_records(self._grid_request_environment(what), product)

    def _ki_monitor_times(
        self, product: BaseEquityProduct, tau: float
    ) -> List[float]:
        """Interior KI monitoring dates — only for daily-discrete KI.

        Empty for every other regime (spec §4 table): European (maturity-only
        => no interior dates), continuous, no-KI, and already-knocked-in
        (monitoring moot).  ``ki_continuous`` here is derived identically to the
        solver's ``self._ki_continuous``. A DATE-based schedule is resolved
        against the prepared environment, for the reason given on
        ``_ko_coupon_align_times``.
        """
        cfg = getattr(product, "barrier_config", None)
        if cfg is None or not getattr(product, "has_ki_barrier", False):
            return []
        if getattr(product, "_otc_lifecycle_knocked_in", False):
            return []
        ki_continuous = (
            cfg.ki_continuous
            or cfg.ki_observation_type == ObservationType.CONTINUOUS
        )
        if ki_continuous:
            return []
        out = []
        sched = cfg.ki_observation_schedule
        if sched is not None and sched.uses_dates():
            profile = self._get_cached_ki_profile(
                self._grid_request_environment("knock-in"), product
            )
            out += list(profile.get("observation_times") or [])
        elif sched is not None:
            out += [
                rec.observation_time
                for rec in sched.records
                if rec.observation_time is not None
            ]
        elif cfg.ki_observation_dates is not None:
            out += list(cfg.ki_observation_dates)
        # Interior only: European KI (obs at maturity only) => empty (correct).
        return sorted({float(t) for t in out if t is not None and 0.0 < float(t) < tau})

    # ------------------------------------------------------------------
    # Declarative grid layer (grid redesign spec §4.6) — migrated solver
    # ------------------------------------------------------------------

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(
        self, product: BaseEquityProduct, market: MarketSnapshot, tau: float,
        extra_times: Optional[Tuple[float, ...]] = None,
    ) -> GridRequest:
        """Consolidated geometry declaration — ALL regime gating in one place.

        Reuses the certified helpers verbatim: KO/coupon dates from
        ``_ko_coupon_align_times`` (must-be-nodes), daily-KI dates from
        ``_ki_monitor_times`` gated by ``_ki_nodes_in_grid`` (dropped for
        already-knocked-in products). Requires the solve state and the
        environment first — the adapter's solve state guarantees the order;
        external callers go through ``core.prepare.request_geometry``.
        """
        strike = float(product.strike)
        align = self._ko_coupon_align_times(product, tau)
        monitor = (
            self._ki_monitor_times(product, tau)
            if self._ki_nodes_in_grid(product)
            else []
        )
        criticals = [market.spot, strike]
        criticals += [b for b in self._get_barriers(product) if b and b > 0]
        return GridRequest(
            tau=tau,
            bound_anchors=(market.spot, strike),
            critical_prices=tuple(p for p in criticals if p and p > 0),
            hard_lower=None,
            hard_upper=None,
            event_times=tuple(sorted(set(align) | set(monitor))),
            extra_times=() if extra_times is None else tuple(extra_times),
            refinement=self._time_refinement(product),
            day_count=market.bus_days_in_year,
        )

    def _populate_observation_maps(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        layout: Layout,
        tau: float,
    ) -> None:
        """Derive the observation-index maps FROM the layout (no searching).

        These maps remain the read model for boundary payoffs, terminal KO
        handling and event stats; they are a view of ``layout.time.step_of``
        — one authoritative construction, so grid geometry and event indices
        cannot disagree. Fully retired when those consumers migrate (Phase 4).
        """
        self._total_tau = tau
        self._ko_observation_indices.clear()
        self._ki_observation_indices.clear()
        self._ki_barrier_by_tidx.clear()
        self._ko_terminal_record = None
        self._has_terminal_ko = False
        step_of = layout.time.step_at  # tolerance-aware lookup (bound method)

        for rec in self._get_cached_ko_records(pricing_env, product):
            obs_time = rec.observation_time
            if obs_time is None:
                continue
            if is_close(obs_time, 0.0):
                self._ko_observation_indices[0] = rec
            elif is_close(obs_time, tau):
                self._ko_terminal_record = rec
                self._has_terminal_ko = True
            elif 0.0 < obs_time < tau:
                self._ko_observation_indices[step_of(obs_time)] = rec

        if (
            product.has_ki_barrier
            and not self._ki_continuous
            and self._ki_nodes_in_grid(product)
        ):
            ki_profile = self._get_cached_ki_profile(pricing_env, product)
            ki_times = ki_profile["observation_times"]
            ki_barriers = ki_profile.get("barriers") or []
            for obs_idx, obs_time in enumerate(ki_times):
                if obs_time is None:
                    continue
                barrier = None
                if obs_idx < len(ki_barriers):
                    barrier = ki_barriers[obs_idx]
                if barrier is None:
                    barrier = self._ki_barrier
                if is_close(obs_time, 0.0):
                    self._ki_observation_indices.add(0)
                    self._ki_barrier_by_tidx[0] = float(barrier)
                elif is_close(obs_time, tau):
                    idx = layout.time.actual_steps
                    self._ki_observation_indices.add(idx)
                    self._ki_barrier_by_tidx[idx] = float(barrier)
                elif 0.0 < obs_time < tau:
                    idx = step_of(obs_time)
                    self._ki_observation_indices.add(idx)
                    self._ki_barrier_by_tidx[idx] = float(barrier)

    def _set_terminal_condition_v0(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set terminal condition for the V0 (not knocked-in) surface.

        V0 payoff at maturity = Principal + Rebate (fixed or call-style)
        """
        payoffs = np.array(
            [product.get_maturity_payoff_v0(s, pricing_env) for s in s_vec]
        )
        grid[:, -1] = payoffs * self._structured_terminal_delay_df

    def _set_terminal_condition_v1(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set terminal condition for the V1 (knocked-in) surface.

        V1 payoff at maturity = Principal + Participation × downside
        (with protection floor applied)
        """
        payoffs = np.array(
            [product.get_maturity_payoff_v1(s, pricing_env) for s in s_vec]
        )
        grid[:, -1] = payoffs * self._structured_terminal_delay_df

    def _apply_terminal_ko(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        s_vec: np.ndarray,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
    ) -> None:
        """Apply KO payoff at terminal time for grid points in breached region."""
        grids = (grid_v0, grid_v1) if self._ko_survives_ki(product) else (grid_v0,)
        names = ("alive", "ki")[:len(grids)]
        self._terminal_ko_columns([g[:, -1] for g in grids], s_vec, ko_record, product, pricing_env,
                                  surfaces=names)

    def _terminal_ko_columns(self, cols, s_vec, ko_record, product, pricing_env, breach=None,
                             surfaces=("alive", "ki")):
        """The terminal knock-out on columns (spec 3B §5.1): ``_ko_columns`` at maturity, where the
        projection follows ``_use_cell_average_events`` and the nodal stage is the plain barrier mask."""
        barrier = ko_record.barrier
        if breach is None:
            payoff = ko_record.payoff if ko_record.payoff is not None else 0.0
            # Discount payoff if settlement is different from observation
            cashflow_value = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=payoff,
                current_time=self._total_tau,
                settlement_time=ko_record.settlement_time,
            )
            targets = [cashflow_value] * len(cols)
        else:
            targets = [breach.ko(ko_record, col, name) for col, name in zip(cols, surfaces)]

        if self._use_cell_average_events():
            for col, target in zip(cols, targets):
                col[...] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    col, target,
                )
            return

        # Apply to breached region (KO is an UP barrier)
        mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

        for col, target in zip(cols, targets):
            col[mask] = target if np.ndim(target) == 0 else target[mask]

    def _set_boundary_conditions_v0(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set boundary conditions for V0 surface."""
        lower, upper = self._boundary_rows_v0(t_idx, tau, product, pricing_env, s_vec)
        self._rows_v0_stash = (t_idx, tau, product, pricing_env, lower, upper)   # for the statistics group
        grid[0, t_idx] = lower.value
        grid[-1, t_idx] = upper.value

    def _boundary_rows_v0(self, t_idx, tau, product, pricing_env, s_vec):
        """V0's Dirichlet rows at ``t_idx`` with their allocation (spec 3B §5.1): ``(lower, upper)``."""
        current_time = self._total_tau - tau
        df_to_maturity = self._df_between_times(
            pricing_env, current_time, self._total_tau
        ) * self._structured_terminal_delay_df

        principal_per_contract = product.initial_price * product.contract_multiplier
        principal = (
            principal_per_contract if product.payoff_config.include_principal else 0.0
        )
        rebate = product.payoff_config.rebate_rate * principal_per_contract

        # Lower boundary (S -> 0)
        # For V0, if continuous KI, it will transition to V1
        # Otherwise, discounted principal + rebate
        if self._ki_continuous and product.has_ki_barrier:
            # Will be overwritten by KI jump
            lower = EdgeRow(
                self._grid_v1[0, t_idx] if self._grid_v1 is not None else 0.0,
                "v0_lower_copy_ki", copy_from_ki=True,
            )
        else:
            v = (principal + rebate) * df_to_maturity
            lower = EdgeRow(v, "v0_lower_principal_rebate", (RowTerm("M", v),))

        # Upper boundary (S -> ∞)
        # Check if above all KO barriers - if so, value is KO payoff
        max_ko_barrier = self._get_max_ko_barrier(product)
        if s_vec[-1] >= max_ko_barrier:
            # Use current KO payoff, allocated to the knock-out unit of the record it pays
            rec = self._next_ko_record_at_time(product, pricing_env, current_time, t_idx)
            payoff = float(rec.payoff or 0.0) if rec is not None else 0.0
            unit = (self._cashflow_value_at_time(pricing_env=pricing_env, cashflow=1.0, current_time=current_time,
                                                 settlement_time=rec.settlement_time) if rec is not None else 0.0)
            # _get_ko_payoff_at_time's cash from the record and unit already found: _cashflow_value_at_time
            # returns float(cash) * df (or float(cash)) and the unit is that df (or 1.0), so the product is
            # the same float -- one record scan and one discount per row
            ko_payoff = payoff * unit if rec is not None else 0.0
            upper = EdgeRow(ko_payoff, "v0_upper_ko",
                            (RowTerm("KO", unit, None if rec is None else float(rec.observation_time), payoff),))
        else:
            if (
                product.payoff_config.call_rebate_enabled
                and product.payoff_config.call_strike is not None
            ):
                df, df_div = self._get_asymptotic_discount_factors(pricing_env, tau)
                participation = (
                    product.payoff_config.call_participation_rate
                    * product.contract_multiplier
                )
                tenor_factor = (
                    product.get_contract_tenor(pricing_env)
                    if product.accrual_config.is_annualized_rebate
                    else 1.0
                )
                participation *= tenor_factor
                strike = product.payoff_config.call_strike
                v = (
                    principal * df
                    + participation * (s_vec[-1] * df_div - strike * df)
                )
                upper = EdgeRow(v, "v0_upper_call_rebate", (RowTerm("M", v),))
            else:
                # Deep OTM: principal + rebate (discounted)
                v = (principal + rebate) * df_to_maturity
                upper = EdgeRow(v, "v0_upper_principal_rebate", (RowTerm("M", v),))
        return self._knock_in_side_row(t_idx, tau, product, pricing_env, s_vec, lower, upper)

    def _knock_in_side_row(self, t_idx, tau, product, pricing_env, s_vec, lower, upper):
        """Under discretely observed knock-in (European included), keyed on the contract, the not-knocked-in row on
        the knock-in side takes the knocked-in surface's row and its allocation (spec 3C §5.5, decision 15): the
        lower row of a standard product, the upper row of a reverse one, where that edge lies beyond every knock-in
        level - and only while a knock-in observation remains at or after the row's time, read from the product's
        resolved observation times (never a solver's index set, which the 2D solvers never fill)."""
        if not product.has_ki_barrier or continuous_knock_in(product):
            return lower, upper
        now = self._total_tau - tau
        times = self._get_cached_ki_profile(pricing_env, product)["observation_times"]
        if not any(t is not None and is_greater_than_or_close(float(t), now) for t in times):
            return lower, upper
        barrier = product.barrier_config.ki_barrier
        levels = [float(b) for b in barrier] if isinstance(barrier, (list, tuple)) else [float(barrier)]
        if not product.is_reverse and float(s_vec[0]) < min(levels):
            v1_lower, _ = self._boundary_rows_v1(t_idx, tau, product, pricing_env, s_vec)
            lower = EdgeRow(v1_lower.value, "v0_lower_certain_ki", v1_lower.terms)
        elif product.is_reverse and float(s_vec[-1]) > max(levels):
            _, v1_upper = self._boundary_rows_v1(t_idx, tau, product, pricing_env, s_vec)
            upper = EdgeRow(v1_upper.value, "v0_upper_certain_ki", v1_upper.terms)
        return lower, upper

    def _set_boundary_conditions_v1(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Set boundary conditions for V1 surface."""
        lower, upper = self._boundary_rows_v1(t_idx, tau, product, pricing_env, s_vec)
        self._rows_v1_stash = (t_idx, tau, product, pricing_env, lower, upper)   # for the statistics group
        grid[0, t_idx] = lower.value
        grid[-1, t_idx] = upper.value

    def _boundary_rows_v1(self, t_idx, tau, product, pricing_env, s_vec):
        """V1's Dirichlet rows at ``t_idx`` with their allocation (spec 3B §5.1): ``(lower, upper)``."""
        current_time = self._total_tau - tau
        df_to_maturity = self._df_between_times(
            pricing_env, current_time, self._total_tau
        ) * self._structured_terminal_delay_df

        principal_per_contract = product.initial_price * product.contract_multiplier
        principal = (
            principal_per_contract if product.payoff_config.include_principal else 0.0
        )
        strike = product.strike
        initial_price = product.initial_price
        participation = product.payoff_config.participation_rate

        # Lower boundary (S -> 0)
        # Deep ITM put for standard snowball: maximum loss
        if product.is_reverse:
            # Reverse: embedded call, S=0 means no loss
            v = principal * df_to_maturity
            lower = EdgeRow(v, "v1_lower_reverse", (RowTerm("M", v),))
        else:
            if (
                product.payoff_config.protection_type == ProtectionType.NONE
            ):
                df, df_div = self._get_asymptotic_discount_factors(pricing_env, tau)
                effective_strike = strike
                effective_participation = participation
                airbag = product.airbag_config
                if airbag.airbag_barrier is not None and s_vec[0] < airbag.airbag_barrier:
                    effective_participation = airbag.airbag_participation_rate
                    if airbag.airbag_strike is not None:
                        effective_strike = airbag.airbag_strike

                slope = effective_participation * product.contract_multiplier
                v = (
                    principal * df
                    + slope * (s_vec[0] * df_div - effective_strike * df)
                )
                lower = EdgeRow(v, "v1_lower_none_slope", (RowTerm("M", v),))
            else:
                # Standard: embedded put, S=0 means maximum loss
                # Loss = participation × (-K/S0) × N
                max_loss = participation * (-strike / initial_price) * principal_per_contract
                # Apply protection floor if applicable
                if product.payoff_config.protection_type.name == "FULL":
                    max_loss = 0.0
                elif product.payoff_config.protection_type.name == "PARTIAL":
                    floor = -product.payoff_config.protection_rate * principal_per_contract
                    max_loss = max(max_loss, floor)
                v = (principal + max_loss) * df_to_maturity
                lower = EdgeRow(v, "v1_lower_max_loss", (RowTerm("M", v),))

        # Upper boundary (S -> ∞)
        # For standard: no loss (put is worthless)
        # For reverse: maximum loss (call is deep ITM)
        if product.is_reverse:
            # For very high S, reverse payoff depends on protection type.
            protection = product.payoff_config.protection_type
            if protection == ProtectionType.NONE:
                df, df_div = self._get_asymptotic_discount_factors(pricing_env, tau)
                participation = product.payoff_config.participation_rate
                effective_strike = strike
                airbag = product.airbag_config
                if airbag.airbag_barrier is not None and s_vec[-1] > airbag.airbag_barrier:
                    participation = airbag.airbag_participation_rate
                    if airbag.airbag_strike is not None:
                        effective_strike = airbag.airbag_strike

                slope = participation * product.contract_multiplier
                v = (
                    (principal + slope * effective_strike) * df
                    - slope * s_vec[-1] * df_div
                )
                upper = EdgeRow(v, "v1_upper_reverse_none", (RowTerm("M", v),))
            else:
                if protection == ProtectionType.PARTIAL:
                    floor = (
                        product.payoff_config.protection_rate
                        * product.initial_price
                        * product.contract_multiplier
                    )
                    v = (principal - floor) * df_to_maturity
                    upper = EdgeRow(v, "v1_upper_reverse_partial", (RowTerm("M", v),))
                else:
                    v = principal * df_to_maturity
                    upper = EdgeRow(v, "v1_upper_reverse_full", (RowTerm("M", v),))
        else:
            # Put is worthless at high S
            v = principal * df_to_maturity
            upper = EdgeRow(v, "v1_upper_principal", (RowTerm("M", v),))
        return lower, upper

    def _apply_step_modifications_two_surface(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Apply barrier modifications to both surfaces at a time step.

        Order of operations:
        1. Apply KO jump to both surfaces (KO takes precedence)
        2. Apply KI jump: V0 <- V1 in breached region

        The knock-out wins a tie with a knock-in OBSERVED at the same instant. While the knock-out
        survives a knock-in that order delivers it: V1 carries the knock-out too, so copying V1 into
        V0 keeps it. Under ``disable_ko_after_ki`` V1 does not, and the copy would overwrite the
        knock-out wherever both levels are breached; the knock-out is then applied last
        (``_ko_jump_goes_last``).
        """
        self._two_surface_events(grid_v0[:, t_idx], grid_v1[:, t_idx], s_vec, t_idx, grid_v0.shape[1],
                                 self._total_tau - tau, product, pricing_env)

    def _two_surface_events(self, v0, v1, s_vec, t_idx, num_t, current_time, product, pricing_env,
                            breach=None) -> None:
        """The column form of ``_apply_step_modifications_two_surface`` (spec 3B §5.1): ``v0``/``v1`` are
        the value columns at ``t_idx`` or statistics blocks, modified in place; ``breach`` as
        ``_ko_columns``."""
        ko_record = self._ko_observation_indices.get(t_idx)
        should_apply_ki = product.has_ki_barrier and (
            self._ki_continuous
            or t_idx in self._ki_observation_indices
        )
        ko_last = ko_record is not None and should_apply_ki and self._ko_jump_goes_last(product)
        # V1 is the knocked-in surface; disable_ko_after_ki removes it from the knock-out.
        survives = self._ko_survives_ki(product)
        ko_cols = [v0, v1] if survives else [v0]
        ko_names = ("alive", "ki") if survives else ("alive",)

        # 1. Apply KO jump if this is a KO observation time
        if ko_record is not None and not ko_last:
            self._ko_columns(ko_cols, s_vec, t_idx, ko_record, product, pricing_env, current_time,
                             breach=breach, surfaces=ko_names)

        # 2. Apply KI jump
        # For continuous KI: apply at every time step
        # For discrete KI: apply only at observation times
        if should_apply_ki:
            self._ki_columns(v0, v1, s_vec, t_idx, num_t, product)

        if ko_last:
            self._ko_columns(ko_cols, s_vec, t_idx, ko_record, product, pricing_env, current_time,
                             breach=breach, surfaces=ko_names)

    def _ko_jump_goes_last(self, product: SnowballOption) -> bool:
        """Whether a knock-out sharing an observation with a knock-in must be applied after it.

        Only under ``disable_ko_after_ki`` (see ``_apply_step_modifications_two_surface``), and only
        for a DISCRETE knock-in: a continuously monitored one breached at this node was touched
        before the observation, so it did come first and the knock-out is already disabled. The two
        jumps write disjoint nodes unless the knock-out level is at or below the knock-in level, so
        the order changes nothing for any other contract.
        """
        return not self._ko_survives_ki(product) and not self._ki_continuous

    def _apply_ko_jump(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        current_time: float,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        ko_record: ResolvedObservationRecord,
    ) -> None:
        """Apply KO payoff to both surfaces in the breached region."""
        # V1 is the knocked-in surface; disable_ko_after_ki removes it here.
        grids = (grid_v0, grid_v1) if self._ko_survives_ki(product) else (grid_v0,)
        names = ("alive", "ki")[:len(grids)]
        self._ko_columns([g[:, t_idx] for g in grids], s_vec, t_idx, ko_record, product, pricing_env,
                         current_time, surfaces=names)

    def _ko_columns(self, cols, s_vec, t_idx, ko_record, product, pricing_env, current_time,
                    breach=None, surfaces=("alive", "ki")):
        """The knock-out stage on columns (spec 3B §5.1): each array in ``cols`` -- a value column
        ``grid[:, t_idx]`` or a statistics block -- takes the breach target where the barrier is breached,
        in place. ``breach=None`` is the value's scalar cash: the same calls on the same arrays as before."""
        barrier = ko_record.barrier
        if breach is None:
            payoff = ko_record.payoff if ko_record.payoff is not None else 0.0
            # Discount payoff based on settlement time
            cashflow_value = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=payoff,
                current_time=current_time,
                settlement_time=ko_record.settlement_time,
            )
            targets = [cashflow_value] * len(cols)
        else:
            targets = [breach.ko(ko_record, col, name) for col, name in zip(cols, surfaces)]

        if self._event_uses_projection(t_idx):
            for col, target in zip(cols, targets):
                col[...] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    col, target,
                )
            return

        # Determine breached region (KO is an UP barrier)
        mask = self._event_nodal_mask(
            s_vec, barrier, product.is_reverse, True, at_valuation=(t_idx == 0)
        )

        for col, target in zip(cols, targets):
            col[mask] = target if np.ndim(target) == 0 else target[mask]

    def _apply_ki_jump(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        product: SnowballOption,
    ) -> None:
        """
        Apply KI jump: V0 <- V1 in the breached region.

        When the KI barrier is hit, the "not knocked-in" value becomes
        the "knocked-in" value at that spot.
        """
        self._ki_columns(grid_v0[:, t_idx], grid_v1[:, t_idx], s_vec, t_idx, grid_v0.shape[1], product)

    def _ki_columns(self, v0, v1, s_vec, t_idx, num_t, product) -> None:
        """The knock-in stage on columns (spec 3B §5.1): ``v0 <- v1`` where the barrier is breached, in
        place. ``v0``/``v1`` are value columns ``grid[:, t_idx]`` or statistics blocks: a knock-in copies
        every column."""
        ki_barrier = self._resolve_ki_barrier_at_tidx(t_idx)

        # Continuous KI monitoring is a continuous-barrier
        # treatment applied every step: it stays a nodal mask regardless of the
        # event_projection setting. Only discretely observed KI events project.
        ki_discrete = not self._ki_continuous
        if self._event_uses_projection(t_idx) and ki_discrete:
            v0[...] = self._project_event_values(
                s_vec, ki_barrier, product.is_reverse, False,
                v0, v1,
            )
            return

        # Determine breached region (KI is a DOWN barrier)
        mask = self._event_nodal_mask(
            s_vec, ki_barrier, product.is_reverse, False,
            at_valuation=(t_idx == 0 and ki_discrete),
        )

        # V0 transitions to V1 in breached region (a contiguous mask -- every one-sided barrier on a
        # sorted grid -- copies a slice: the same values, without the boolean gather on a block)
        idx = np.flatnonzero(mask)
        if idx.size and idx[-1] - idx[0] + 1 == idx.size:
            v0[idx[0]:idx[-1] + 1] = v1[idx[0]:idx[-1] + 1]
        elif idx.size:
            v0[mask] = v1[mask]

        # FIRST_PASSAGE correction [2026-08-18]: the jump above monitors the
        # continuous barrier only at step boundaries, so crossings INSIDE the
        # step are invisible and the live surface is biased high by
        # O(sqrt(dt)). Mix the live region toward V1 with the exact
        # touch-and-return probability for the step (paths ending breached
        # are already captured by the mask at the neighbouring column).
        # Interior steps only: the terminal column has no step interior, and
        # the valuation column's events are owned by the t0 readout.
        fp = self._ki_fp
        if fp is not None and 0 < t_idx < num_t - 1:
            fp.add_correction(t_idx, s_vec, ki_barrier, v0, v1)

    def _get_max_ko_barrier(self, product: SnowballOption) -> float:
        """Get the maximum KO barrier level."""
        ko_barrier = product.barrier_config.ko_barrier
        if isinstance(ko_barrier, list):
            return max(ko_barrier)
        return ko_barrier

    def _get_ko_payoff_at_time(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        current_time: float,
        t_idx: int,
    ) -> float:
        """Get KO payoff for current time/index: the record observed there, else the next scheduled one
        (past observations ignored), discounted from its settlement."""
        return self._ko_payoff_of_record(
            self._next_ko_record_at_time(product, pricing_env, current_time, t_idx), pricing_env, current_time)

    def _ko_payoff_of_record(self, record, pricing_env: PricingEnvironment, current_time: float) -> float:
        """A knock-out record's payoff (0.0 when unset or when there is no record) at ``current_time``."""
        if record is None:
            return 0.0
        return self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=record.payoff if record.payoff is not None else 0.0,
            current_time=current_time,
            settlement_time=record.settlement_time,
        )

    def _next_ko_record_at_time(
        self,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        current_time: float,
        t_idx: int,
    ) -> Optional[ResolvedObservationRecord]:
        """The knock-out record ``_get_ko_payoff_at_time`` pays at ``t_idx``: the record observed there,
        else the next scheduled one (the last when none is ahead), else ``None``.

        The scan is memoised per resolved record list (the cache's own object, so a new product or
        market starts a new memo) and time: the boundary rows ask it every step, twice under the
        value-mode statistics."""
        ko_record = self._ko_observation_indices.get(t_idx)
        if ko_record is not None:
            return ko_record
        ko_records = self._get_cached_ko_records(pricing_env, product)
        memo = getattr(self, "_next_ko_memo", None)
        if memo is None or memo[0] is not ko_records:
            memo = self._next_ko_memo = (ko_records, {})
        key = float(current_time)
        if key in memo[1]:
            return memo[1][key]
        future_records = [
            rec for rec in ko_records if self._record_is_non_negative_time(rec)
        ]
        next_rec: Optional[ResolvedObservationRecord] = None
        if future_records:
            for rec in future_records:
                if is_greater_than_or_close(rec.observation_time, current_time):
                    next_rec = rec
                    break
            if next_rec is None:
                next_rec = future_records[-1]
        memo[1][key] = next_rec
        return next_rec

    # _df_between_times / _cashflow_value_at_time are inherited from
    # BasePDESolver.

    def _get_asymptotic_discount_factors(
        self, pricing_env: PricingEnvironment, tau_to_maturity: float
    ) -> Tuple[float, float]:
        """
        Get risk-free and dividend discount factors for asymptotic boundary conditions.

        This helper consolidates the repeated pattern of computing discount factors
        for boundary conditions when using asymptotic mode.

        Args:
            pricing_env: Pricing environment with rate curves
            tau_to_maturity: Time to maturity

        Returns:
            Tuple of (risk_free_df, dividend_df)
        """
        if tau_to_maturity <= 0:
            return 1.0, 1.0
        total_tau = self._total_tau if self._total_tau > 0 else tau_to_maturity
        current_time = max(total_tau - tau_to_maturity, 0.0)
        df = self._df_between_times(pricing_env, current_time, total_tau)
        df_div = self._carry_df_between_times(pricing_env, current_time, total_tau)
        delay_df = self._structured_terminal_delay_df
        return float(df) * delay_df, float(df_div) * delay_df

    # Override abstract methods from base class (not used for two-surface)
    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Not used - two-surface solver has separate terminal conditions."""
        pass

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
        """Not used - two-surface solver has separate boundary conditions."""
        pass

    def __repr__(self) -> str:
        return "SnowballPDESolver()"

    def _product_cache_token(self, product: BaseEquityProduct, strategy: str) -> Tuple[str, object]:
        # Memoized per solve (see __init__): the boundary-condition path asks
        # for this token on every time step, and the base implementation
        # re-serializes the whole product each call. "strict" stays unmemoized
        # — it is already an O(1) id() lookup.
        if strategy == "strict":
            return super()._product_cache_token(product, strategy)
        memo_key = (id(product), strategy)
        token = self._product_token_memo.get(memo_key)
        if token is None:
            token = super()._product_cache_token(product, strategy)
            self._product_token_memo[memo_key] = token
        return token

    def _observation_cache_key(
        self, pricing_env: PricingEnvironment, product: SnowballOption, kind: str
    ) -> Tuple:
        # The resolvers roll settlement dates on the effective settlement
        # calendar, and the product token names a calendar by name alone, so
        # the key carries that calendar's value identity (spec 2 decision 19).
        from quantark.asset.equity.settlement import (
            calendar_identity,
            effective_settlement_calendar,
        )

        # Read once per step by the boundary and event paths: inside one market
        # scope (a fixed environment, one call's product) the key is computed once.
        scope = getattr(self, "_market_fp", None)
        memo = scope[2] if scope is not None and len(scope) > 2 and scope[0] is pricing_env else None
        memo_key = ("observation_cache_key", kind, id(product))
        if memo is not None:
            hit = memo.get(memo_key)
            if hit is not None and hit[0] is product:
                return hit[1]
        strategy = _CACHE_KEYING
        key = (
            kind,
            strategy,
            f"{product.__class__.__module__}.{product.__class__.__qualname__}",
            self._product_cache_token(product, strategy),
            pricing_env.valuation_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
            calendar_identity(effective_settlement_calendar(product, pricing_env)),
        )
        if memo is not None:
            memo[memo_key] = (product, key)
        return key

    def _get_cached_ko_records(
        self, pricing_env: PricingEnvironment, product: SnowballOption
    ) -> List[ResolvedObservationRecord]:
        if not self._is_cache_enabled():
            return product.resolve_ko_observations(pricing_env)
        key = self._observation_cache_key(pricing_env, product, "ko")
        cached = self._ko_records_cache.get(key)
        if cached is not None:
            self._ko_records_cache.move_to_end(key)
            return cached
        records = product.resolve_ko_observations(pricing_env)
        self._ko_records_cache[key] = records
        self._ko_records_cache.move_to_end(key)
        if len(self._ko_records_cache) > GRID_CACHE_MAX_ENTRIES:
            self._ko_records_cache.popitem(last=False)
        return records

    def _get_cached_ki_profile(
        self, pricing_env: PricingEnvironment, product: SnowballOption
    ) -> Dict[str, List[Optional[float]]]:
        if not self._is_cache_enabled():
            return product.get_ki_observation_profile(pricing_env)
        key = self._observation_cache_key(pricing_env, product, "ki")
        cached = self._ki_profile_cache.get(key)
        if cached is not None:
            self._ki_profile_cache.move_to_end(key)
            return cached
        profile = product.get_ki_observation_profile(pricing_env)
        self._ki_profile_cache[key] = profile
        self._ki_profile_cache.move_to_end(key)
        if len(self._ki_profile_cache) > GRID_CACHE_MAX_ENTRIES:
            self._ki_profile_cache.popitem(last=False)
        return profile
