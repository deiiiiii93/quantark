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

import logging
from collections import OrderedDict
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

from scipy.linalg import solve_banded
from scipy.special import ndtr

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))

from quantark.asset.equity.engine.pde.base_pde_solver import (
    BasePDESolver,
    PDESessionOutputs,
    PDESolutionResult,
)
from quantark.asset.equity.engine.pde.backward_operator import BackwardOperator
from quantark.asset.equity.engine.pde.grid.events import project_event_values
from quantark.asset.equity.engine.pde.grid import (
    EventSchedule,
    GridRequest,
    Layout,
    MarketSnapshot,
    project_between,
)
from quantark.asset.equity.engine.event_stats import AutocallableEventStats
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType
from quantark.util.enum.engine_enums import (
    ContinuousKICorrection,
    EventProjectionMode,
    KnockInMonitoringMode,
)
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    is_greater_than_or_close,
    is_zero,
    safe_exp,
    safe_sqrt,
)

# BGK continuity-correction constant beta = -zeta(1/2)/sqrt(2*pi). Defined here
# with the identical high-precision value used by SnowballQuadEngine so the PDE
# and quad BGK shifted barriers coincide (kept as a local literal rather than a
# cross-engine import to avoid a PDE->quad module dependency) [§11.6].
_BGK_BETA = 0.5825971579390107



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
           correction (ContinuousKICorrection.FIRST_PASSAGE, default) so the
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

        # BGK opt-in continuous-KI state [§11.6]. Set per-solve by _configure_bgk:
        # when active, KI is applied on every step against the shifted barrier and
        # the interior daily-KI nodes are dropped from the time grid.
        self._bgk_active: bool = False
        self._bgk_ki_barrier: float = 0.0
        # Declarative grid layer state for the current solve (None when the
        # legacy path runs — subclasses not yet migrated, session grids).
        self._active_layout: Optional[Layout] = None
        self._active_schedule: Optional[EventSchedule] = None
        self._stats_layout: Optional[Layout] = None

        # Time tracking
        self._total_tau: float = 0.0
        self._banded_cache: "OrderedDict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
        self._banded_cache_max_entries = self.params.banded_cache_max_entries
        # Session-injected banded factorization pack (read-only; see
        # BasePDESolver._session_matrix_pack for the contract).
        self._session_banded_pack = None
        self._profile_enabled = enable_profiling
        self._profile_stats: Dict[str, float] = {}
        self._ko_records_cache: "OrderedDict[Tuple, List[ResolvedObservationRecord]]" = OrderedDict()
        self._ki_profile_cache: "OrderedDict[Tuple, Dict[str, List[Optional[float]]]]" = OrderedDict()

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

    def _solve(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> PDESolutionResult:
        """
        Core Two-Surface PDE solving logic for Snowball options.

        This method contains the common solving logic shared by price() and
        calculate_greeks(). It handles the two-surface approach with V0/V1
        state transitions.

        Args:
            product: SnowballOption to price (already validated)
            pricing_env: Pricing environment with market data

        Returns:
            PDESolutionResult with appropriate surface (V0 or V1) at t=0

        Note:
            This method assumes the product has been validated and is not
            expired or knocked out at valuation. Callers should check these
            conditions before calling _solve().
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        # State preamble (KI regime, valuation flags, barrier level) shared
        # with session preparation via the _prepare_solve_state seam.
        knocked_in_at_valuation = self._prepare_solve_state(product, pricing_env)
        self._reset_t0_readout_state()

        # Extract market data
        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # BGK state is resolved at the top of _build_grids so the time grid
        # drops interior KI nodes (and subclasses cannot skip it).

        if self._profile_enabled:
            self._reset_profile_stats()

        # Build grids. The declarative layer serves migrated solvers unless
        # a prepared session injected grids (adapter clones keep the legacy
        # path until the execution seams re-point at Phase 4). Phoenix/KO-
        # reset inherit this _solve and take the legacy branch until their
        # own migration flips _uses_grid_layer.
        if self._profile_enabled:
            t0 = perf_counter()
        self._active_layout = None
        self._active_schedule = None
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        if self._active_layout is not None:
            self._active_schedule = self.event_schedule(
                product, pricing_env, self._active_layout
            )
        if self._profile_enabled:
            self._profile_stats["grid_build"] += perf_counter() - t0

        # Initialize both grids
        num_x, num_t = len(x_vec), len(t_vec)
        self._grid_v0 = np.zeros((num_x, num_t))
        self._grid_v1 = np.zeros((num_x, num_t))

        # Set terminal conditions
        self._set_terminal_condition_v0(
            self._grid_v0, x_vec, s_vec, product, pricing_env
        )
        self._set_terminal_condition_v1(
            self._grid_v1, x_vec, s_vec, product, pricing_env
        )

        # Apply terminal KO if at maturity observation
        if self._has_terminal_ko and self._ko_terminal_record is not None:
            self._apply_terminal_ko(
                self._grid_v0,
                self._grid_v1,
                s_vec,
                product,
                pricing_env,
                self._ko_terminal_record,
            )

        # Apply terminal KI if at maturity observation (European KI fix)
        if product.has_ki_barrier:
            is_terminal_ki = self._ki_continuous or self._bgk_active
            if not is_terminal_ki:
                if (num_t - 1) in self._ki_observation_indices:
                    is_terminal_ki = True
            if is_terminal_ki:
                self._apply_ki_jump(
                    self._grid_v0, self._grid_v1, s_vec, num_t - 1, product
                )

        # Build operator matrices
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        A = self._build_operator_matrix(l, c, u, num_x)

        # Term-structure step coefficients (one set for flat inputs)
        sc = self._step_coefficients_for_solve(
            pricing_env, product.strike, t_vec, dx_vec, num_x
        )
        sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, num_x)
        step_coeffs = None if sc.n_unique == 1 else sc

        # Time stepping for both surfaces
        self._time_stepping_two_surface(
            self._grid_v0,
            self._grid_v1,
            A,
            l,
            c,
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

        # Return appropriate surface based on knocked-in state
        spot_log = np.log(spot)
        if knocked_in_at_valuation:
            solution_vec = self._grid_v1[:, 0]
        else:
            solution_vec = self._grid_v0[:, 0]

        readout_vec, readout_override = self._compose_t0_readout(
            1 if knocked_in_at_valuation else 0
        )
        return PDESolutionResult(
            solution_vec=solution_vec,
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=spot_log,
            readout_vec=readout_vec,
            readout_override=readout_override,
        )

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
        if not (self._ki_continuous or self._bgk_active):
            return
        if self.params.continuous_ki_correction is not (
            ContinuousKICorrection.FIRST_PASSAGE
        ):
            return
        t = np.asarray(t_vec, dtype=float)
        if t.size < 3:
            return
        barrier = (
            float(self._bgk_ki_barrier)
            if self._bgk_active
            else float(self._ki_barrier)
        )
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
        return self._price_with_solution(product, pricing_env)[0]

    def _price_with_solution(self, product, pricing_env):
        """price()'s preamble + one solve; None solution = short-circuit
        (expired or knocked-out at valuation). Native session seam."""
        self._check_product_type(product)

        if pricing_env is None:
            raise ValidationError(
                f"PricingEnvironment is required for {self._solver_name}"
            )

        # Validate PDE compatibility
        self._validate_product(product)

        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0 or is_zero(tau):
            # Expired: return terminal payoff
            return self._calculate_terminal_value(product, spot, pricing_env), None

        # Check if knocked out at valuation
        knocked_out_at_valuation = self._is_knocked_out_at_valuation(
            product, spot, pricing_env
        )
        if knocked_out_at_valuation:
            return self._get_immediate_ko_payoff(product, pricing_env), None

        # Solve PDE and interpolate price. Valuation-date events (an
        # observation falling exactly on t=0) are deterministic at the known
        # spot: the readout interpolates the smooth 0+ branch column
        # (readout_vec) and applies today's transitions pointwise
        # (readout_override), never blending across the t=0 jump.
        result = self._solve(product, pricing_env)
        if result.readout_override is not None:
            return float(result.readout_override), result
        readout_vec = (
            result.readout_vec if result.readout_vec is not None else result.solution_vec
        )
        return (
            self._interpolate_price(readout_vec, result.x_vec, result.spot_log),
            result,
        )

    def _session_outputs(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        want_events: bool = False,
        want_grid: bool = False,
        streams: Optional[frozenset] = None,
    ) -> PDESessionOutputs:
        """One VALUE solve serving PV + event stats + grid projection.

        The event-stat indicator sweep remains the engine's designed separate
        pass (``calculate_event_stats`` with the npv supplied never re-runs
        the value solve) — the session performs exactly as many backward
        marches as the direct ``price_with_events`` path.
        """
        from quantark.cashleg.event_distribution import EventDistribution

        npv, solution = self._price_with_solution(product, pricing_env)
        stats = None
        dist = None
        if want_events and solution is not None:
            stats = self.calculate_event_stats(
                product, pricing_env, npv=float(npv), streams=streams
            )
            if stats is not None:
                dist = EventDistribution.from_autocallable_stats(stats)
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
        return self._compute_event_stats(product, pricing_env, npv=npv, streams=streams)

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
        """Construct the event-stats dataclass (overridable by subclasses)."""
        return AutocallableEventStats(**fields)

    # --- Extra indicator-surface hooks (overridden by Phoenix for coupons) ---

    def _n_extra_event_cols(self, n_ko: int) -> int:
        """Extra stacked indicator columns beyond ``[KO_0..KO_{n-1}]``."""
        return 0

    def _set_extra_event_indicators(
        self, v0, v1, s_vec, n_ko, ko_idx, rec, product, pricing_env, t_vec, t_idx
    ) -> None:
        """Set extra indicator columns at an observation (no-op for Snowball)."""
        return None

    def _extract_extra_event_stats(
        self,
        initial_grid,
        x_vec,
        spot_log,
        n_ko,
        ko_records,
        pricing_env,
        product,
        col_overrides=None,
    ) -> dict:
        """Extra event-stats fields from the extra columns (none for Snowball)."""
        return {}

    def _t0_extra_indicator_overrides(
        self,
        product,
        pricing_env,
        spot,
        n_ko,
        rec0_pos,
        rec0,
        t_vec,
        ko_triggered,
        df_delay0,
    ) -> dict:
        """Exact valuation-date outcomes for extra indicator columns
        (none for Snowball)."""
        return {}

    def _compute_event_stats(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        npv: Optional[float] = None,
        streams: Optional[frozenset] = None,
    ) -> Optional[AutocallableEventStats]:
        """
        Native PDE implementation:
        - Propagates stacked indicator surfaces through the same backward PDE stepping.
        - Applies KO/KI jumps to all indicator surfaces at observation times.
        - Returns KO per-observation probabilities (by dividing discounted indicators by
          discount factors) and expected discounted KO cashflows.

        ``npv``: the product value; when provided, the internal ``self.price()``
        solve is skipped and this value is used for the maturity residual
        (single-pass path — the caller already ran the value sweep).

        ``streams``: the ``EventType`` set the caller needs [§11.1]. ``None`` ⇒
        the full distribution (KO + coupon + KI, unchanged behaviour). Pruning
        the KI indicator columns and/or the Phoenix coupon columns when they are
        not requested leaves ``ko_probability`` / ``survival`` / ``pv`` bit-
        identical — the KI *regime jump* still runs, only the auxiliary
        indicator columns are dropped.
        """
        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)
        if tau <= 0 or is_zero(tau):
            return None

        # Validate PDE compatibility
        self._validate_product(product)

        # Determine knocked-in state at valuation
        already_knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )

        # Extract market data
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(product.strike, tau)

        # Store product properties needed by _build_grids
        self._is_reverse = product.is_reverse
        self._ki_continuous = ki_continuous
        self._ki_fp = None  # per-solve; the time-stepping loop rebuilds it
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                self._ki_barrier = ki_barrier[0]
            else:
                self._ki_barrier = ki_barrier

        # SAME geometry as the value solve (single construction site:
        # _build_grids layer-routes for migrated solvers).
        self._stats_layout = None
        self._active_layout = None
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
        self._stats_layout = self._active_layout
        num_x, num_t = len(x_vec), len(t_vec)

        ko_records = self._filter_observations_by_tau(
            product.resolve_ko_observations(pricing_env), tau
        )
        if not ko_records:
            return None
        n_ko = len(ko_records)

        # Map time index -> ko record index.
        ko_index_by_tidx: Dict[int, int] = {}
        for k, rec in enumerate(ko_records):
            obs_time = float(rec.observation_time)
            if is_close(obs_time, 0.0):
                t_idx = 0
            elif is_close(obs_time, tau):
                t_idx = num_t - 1
            else:
                t_idx = int(np.argmin(np.abs(t_vec - obs_time)))
            if not is_close(float(t_vec[t_idx]), obs_time):
                raise ValidationError(
                    "Time grid must align with KO observation times for event stats."
                )
            ko_index_by_tidx[t_idx] = k

        # Surface columns: [KO_0..KO_{n-1}, <extra coupon cols>, KI_indicator,
        # KI_ever_indicator]. The KI_indicator carries the "settles knocked-in"
        # semantics (absorbed to 0 on any KO). The KI_ever_indicator tracks
        # P(the underlying breaches the KI barrier at any point in [0, T]),
        # independent of KO/autocall — it is a pure first-passage statistic and is
        # therefore EXEMPT from the KO absorption below (matching the QUAD and MC
        # ki_ever definition).
        # Stream selection [§11.1]: prune auxiliary indicator columns the caller
        # does not need. KO columns are always present; the KI regime jump always
        # runs (it drives the KO columns), only the KI *indicator* columns and
        # the Phoenix coupon columns are optional.
        if streams is None:
            want_ki = True
            want_coupon = True
        else:
            from quantark.cashleg.event_distribution import EventType

            want_ki = bool(
                streams & {EventType.KI, EventType.MATURITY_WITH_KI}
            )
            want_coupon = EventType.COUPON in streams

        n_extra = self._n_extra_event_cols(n_ko) if want_coupon else 0
        if want_ki:
            ki_col = n_ko + n_extra
            ki_ever_col = n_ko + n_extra + 1
            n_cols = n_ko + n_extra + 2
        else:
            ki_col = -1
            ki_ever_col = -1
            n_cols = n_ko + n_extra

        # Terminal conditions at maturity (t = T):
        # - KO indicators are zero at maturity (KO only at discrete observations via jumps)
        # - Both KI indicators are 1 on the KI surface and 0 on the no-KI surface
        v0_next = np.zeros((num_x, n_cols), dtype=float)
        v1_next = np.zeros((num_x, n_cols), dtype=float)
        if want_ki:
            v1_next[:, ki_col] = 1.0
            v1_next[:, ki_ever_col] = 1.0

        # Apply terminal KO/KI events at maturity if observation schedules include t=T.
        terminal_tidx = num_t - 1
        terminal_ko_idx = ko_index_by_tidx.get(terminal_tidx)
        if terminal_ko_idx is not None:
            rec = ko_records[terminal_ko_idx]
            barrier = float(rec.barrier) if rec.barrier is not None else 0.0
            df_delay = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=1.0,
                current_time=float(t_vec[terminal_tidx]),
                settlement_time=rec.settlement_time,
            )
            if self._use_cell_average_events():
                # KO absorbs every surface to 0 (KO_i to df_delay); KI-ever is
                # exempt (pure first-passage statistic) so its breach target
                # is its own current value.
                for v in (v0_next, v1_next):
                    target = np.zeros_like(v)
                    target[:, terminal_ko_idx] = df_delay
                    if want_ki:
                        target[:, ki_ever_col] = v[:, ki_ever_col]
                    v[:] = self._project_event_values(
                        s_vec, barrier, product.is_reverse, True, v, target
                    )
            else:
                mask_ko = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

                # KI-ever is exempt from KO absorption (pure first-passage statistic).
                if want_ki:
                    ever0 = v0_next[mask_ko, ki_ever_col].copy()
                    ever1 = v1_next[mask_ko, ki_ever_col].copy()
                v0_next[mask_ko, :] = 0.0
                v1_next[mask_ko, :] = 0.0
                v0_next[mask_ko, terminal_ko_idx] = df_delay
                v1_next[mask_ko, terminal_ko_idx] = df_delay
                if want_ki:
                    v0_next[mask_ko, ki_ever_col] = ever0
                    v1_next[mask_ko, ki_ever_col] = ever1
            if want_coupon:
                self._set_extra_event_indicators(
                    v0_next, v1_next, s_vec, n_ko, terminal_ko_idx, rec,
                    product, pricing_env, t_vec, terminal_tidx,
                )

        is_terminal_ki = product.has_ki_barrier and (
            self._ki_continuous
            or self._bgk_active
            or terminal_tidx in self._ki_observation_indices
        )
        if is_terminal_ki:
            ki_barrier = self._resolve_ki_barrier_at_tidx(terminal_tidx)
            if self._use_cell_average_events() and not (
                self._ki_continuous or self._bgk_active
            ):
                v0_next[:] = self._project_event_values(
                    s_vec, ki_barrier, product.is_reverse, False, v0_next, v1_next
                )
            else:
                mask_ki = self._get_barrier_mask(s_vec, ki_barrier, product.is_reverse, is_up_barrier=False)
                v0_next[mask_ki, :] = v1_next[mask_ki, :]

        # Operator coefficients and banded solver setup
        params: PDEParams = self.params
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        sc_ev = self._build_step_coefficients(
            pricing_env, product.strike, t_vec, dx_vec, num_x
        )
        # FIRST_PASSAGE continuous-KI state for the stats sweep's KI stage:
        # the indicator columns follow the same regime dynamic programming as
        # the value surfaces, so they take the same crossing correction.
        self._prepare_continuous_ki_correction(product, pricing_env, t_vec)
        sc_ev = self._flat_exact_step_coefficients(sc_ev, r, q, sigma, dx_vec, num_x)
        ev_step_coeffs = None if sc_ev.n_unique == 1 else sc_ev
        use_banded = params.use_banded_solver and (num_x - 2) > 2
        if not use_banded:
            raise ValidationError("Event stats PDE currently requires banded solver path.")

        # Canonical damping schedule shared with the pricing sweep — the
        # event-distribution pass MUST run the identical discretization for
        # the KO-probability / NPV decomposition to reconcile.
        theta_by_step = self._theta_schedule_from_layout(self._stats_layout)

        n_int = num_x - 2
        rhs = np.empty((n_int, 2 * n_cols), dtype=float)

        # Valuation-date events are deterministic at the known spot: capture
        # the smooth 0+ indicator columns before the j==0 events land, and
        # overlay the exact t=0 outcomes on the spot readout below.
        t0_pre_grids = None
        t0_capture = self._use_cell_average_events() and (
            0 in ko_index_by_tidx
            or (
                product.has_ki_barrier
                and not (self._ki_continuous or self._bgk_active)
                and 0 in self._ki_observation_indices
            )
        )

        for j in range(num_t - 2, -1, -1):
            dt = float(dt_vec[j])
            theta = float(theta_by_step[j])

            if ev_step_coeffs is not None:
                ev_key = int(ev_step_coeffs.set_index[j])
                l, c, u = ev_step_coeffs.lcu_sets[ev_key]
            else:
                ev_key = 0
            banded, lower1, main1, upper1 = self._get_banded_system(
                l, c, u, dt, theta, coeff_key=ev_key
            )

            # Initialize "current" with next boundaries (approximation); interior will be solved.
            v0_cur = v0_next.copy()
            v1_cur = v1_next.copy()

            # Build RHS for all columns.
            v0n = v0_next[1:-1, :]
            v1n = v1_next[1:-1, :]

            rhs_v0 = rhs[:, :n_cols]
            rhs_v1 = rhs[:, n_cols:]

            rhs_v0[:] = main1[:, None] * v0n
            rhs_v0[1:, :] += lower1[:, None] * v0n[:-1, :]
            rhs_v0[:-1, :] += upper1[:, None] * v0n[1:, :]

            rhs_v1[:] = main1[:, None] * v1n
            rhs_v1[1:, :] += lower1[:, None] * v1n[:-1, :]
            rhs_v1[:-1, :] += upper1[:, None] * v1n[1:, :]

            # Boundary contributions (Dirichlet terms).
            if num_x > 2:
                lhs_l = float(l[1])
                lhs_u = float(u[-2])
                rhs_v0[0, :] += dt * (
                    (1.0 - theta) * lhs_l * v0_next[0, :] + theta * lhs_l * v0_cur[0, :]
                )
                rhs_v0[-1, :] += dt * (
                    (1.0 - theta) * lhs_u * v0_next[-1, :] + theta * lhs_u * v0_cur[-1, :]
                )
                rhs_v1[0, :] += dt * (
                    (1.0 - theta) * lhs_l * v1_next[0, :] + theta * lhs_l * v1_cur[0, :]
                )
                rhs_v1[-1, :] += dt * (
                    (1.0 - theta) * lhs_u * v1_next[-1, :] + theta * lhs_u * v1_cur[-1, :]
                )

            sol = solve_banded(
                (1, 1),
                banded,
                rhs,
                overwrite_b=False,
                check_finite=False,
            )
            v0_cur[1:-1, :] = sol[:, :n_cols]
            v1_cur[1:-1, :] = sol[:, n_cols:]

            if j == 0 and t0_capture:
                t0_pre_grids = (v0_cur.copy(), v1_cur.copy())

            # Apply KO jump (if observation time).
            ko_idx = ko_index_by_tidx.get(j)
            if ko_idx is not None:
                rec = ko_records[ko_idx]
                barrier = float(rec.barrier) if rec.barrier is not None else 0.0
                df_delay = self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=1.0,
                    current_time=float(t_vec[j]),
                    settlement_time=rec.settlement_time,
                )
                ko_surfaces = (
                    (v0_cur, v1_cur) if self._ko_survives_ki(product) else (v0_cur,)
                )
                if self._event_uses_projection(j):
                    # Zero all event surfaces in the KO region (KO_i indicator
                    # to df_delay); KI-ever is exempt (first-passage statistic).
                    for v in ko_surfaces:
                        target = np.zeros_like(v)
                        target[:, ko_idx] = df_delay
                        if want_ki:
                            target[:, ki_ever_col] = v[:, ki_ever_col]
                        v[:] = self._project_event_values(
                            s_vec, barrier, product.is_reverse, True, v, target
                        )
                else:
                    mask_ko = self._event_nodal_mask(
                        s_vec, barrier, product.is_reverse, True,
                        at_valuation=(j == 0),
                    )

                    # Zero all event surfaces in KO region, then set the KO_i indicator.
                    # KI-ever is exempt (pure first-passage statistic, no KO absorption).
                    for v in ko_surfaces:
                        if want_ki:
                            ever = v[mask_ko, ki_ever_col].copy()
                        v[mask_ko, :] = 0.0
                        v[mask_ko, ko_idx] = df_delay
                        if want_ki:
                            v[mask_ko, ki_ever_col] = ever
                if want_coupon:
                    self._set_extra_event_indicators(
                        v0_cur, v1_cur, s_vec, n_ko, ko_idx, rec,
                        product, pricing_env, t_vec, j,
                    )

            # Apply KI jump (continuous / BGK every step, or discrete at obs indices).
            if product.has_ki_barrier:
                should_apply_ki = (
                    self._ki_continuous
                    or self._bgk_active
                    or j in self._ki_observation_indices
                )
                if should_apply_ki:
                    ki_barrier = self._resolve_ki_barrier_at_tidx(j)
                    ki_discrete = not (self._ki_continuous or self._bgk_active)
                    if self._event_uses_projection(j) and ki_discrete:
                        v0_cur[:] = self._project_event_values(
                            s_vec, ki_barrier, product.is_reverse, False,
                            v0_cur, v1_cur,
                        )
                    else:
                        mask_ki = self._event_nodal_mask(
                            s_vec, ki_barrier, product.is_reverse, False,
                            at_valuation=(j == 0 and ki_discrete),
                        )
                        v0_cur[mask_ki, :] = v1_cur[mask_ki, :]
                        fp = self._ki_fp
                        if fp is not None and 0 < j < len(t_vec) - 1:
                            d = v1_cur - v0_cur
                            v0_cur += fp.step_correction(j, s_vec, ki_barrier, d)

            # Enforce simple Neumann-like boundary (zero slope) for stability.
            v0_cur[0, :] = v0_cur[1, :]
            v0_cur[-1, :] = v0_cur[-2, :]
            v1_cur[0, :] = v1_cur[1, :]
            v1_cur[-1, :] = v1_cur[-2, :]

            v0_next = v0_cur
            v1_next = v1_cur

        # Select initial regime based on knocked-in at valuation. When the
        # valuation date carries events, read the smooth 0+ branch columns
        # (mirroring the loop's boundary enforcement on the captured copies)
        # and overlay the exact t=0 outcomes at the known spot.
        if t0_pre_grids is not None:
            for g in t0_pre_grids:
                g[0, :] = g[1, :]
                g[-1, :] = g[-2, :]
            initial_grid = (
                t0_pre_grids[1] if knocked_in_at_valuation else t0_pre_grids[0]
            )
        else:
            initial_grid = v1_next if knocked_in_at_valuation else v0_next
        spot_log = float(np.log(spot))

        # Exact valuation-date outcomes at the known spot. Untriggered t=0
        # streams are exactly zero on the pre-event columns already; a
        # triggered t=0 KO resolves every stream deterministically.
        t0_overrides: Dict[int, float] = {}
        rec0_pos = ko_index_by_tidx.get(0)
        if t0_pre_grids is not None and rec0_pos is not None:
            rec0 = ko_records[rec0_pos]
            b0 = float(rec0.barrier) if rec0.barrier is not None else 0.0
            t0_ko_triggered = bool(
                self._event_nodal_mask(
                    np.asarray([spot], dtype=float),
                    b0,
                    product.is_reverse,
                    True,
                    at_valuation=True,
                )[0]
            )
            df_delay0 = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=1.0,
                current_time=float(t_vec[0]),
                settlement_time=rec0.settlement_time,
            )
            if t0_ko_triggered:
                for i in range(n_ko):
                    t0_overrides[i] = 0.0
                t0_overrides[rec0_pos] = df_delay0
                if want_ki:
                    t0_overrides[ki_col] = 0.0
                    t0_overrides[ki_ever_col] = 0.0
            else:
                t0_overrides[rec0_pos] = 0.0
            if want_coupon:
                t0_overrides.update(
                    self._t0_extra_indicator_overrides(
                        product,
                        pricing_env,
                        spot,
                        n_ko,
                        rec0_pos,
                        rec0,
                        t_vec,
                        t0_ko_triggered,
                        df_delay0,
                    )
                )

        def _read_col(col: int) -> float:
            if col in t0_overrides:
                return t0_overrides[col]
            return float(np.interp(spot_log, x_vec, initial_grid[:, col]))

        ed_unit = np.array([_read_col(i) for i in range(n_ko)], dtype=float)
        ko_times = np.array([float(rec.observation_time) for rec in ko_records], dtype=float)
        ko_probability = np.zeros(n_ko, dtype=float)
        ed_ko_cf = np.zeros(n_ko, dtype=float)

        for i, rec in enumerate(ko_records):
            obs_time = float(rec.observation_time)
            settle = rec.settlement_time if rec.settlement_time is not None else obs_time
            settle = float(settle)
            df0 = pricing_env.get_discount_factor(settle)
            if df0 > 0.0:
                ko_probability[i] = float(ed_unit[i] / df0)
            payoff = float(rec.payoff) if rec.payoff is not None else 0.0
            ed_ko_cf[i] = float(ed_unit[i] * payoff)

        survival_probability = np.ones(n_ko, dtype=float)
        cumulative = 0.0
        for i in range(n_ko):
            cumulative += ko_probability[i]
            survival_probability[i] = max(0.0, 1.0 - cumulative)

        ki_times = np.array([], dtype=float)
        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        if already_knocked_in:
            ki_probability = 1.0
            ki_ever_probability = 1.0
            ki_times = np.array([0.0], dtype=float)
            ki_event_probability = np.array([1.0], dtype=float)
            ki_survival_probability = np.array([0.0], dtype=float)
        elif want_ki:
            df_T = pricing_env.get_discount_factor(float(tau))
            ed_ki = _read_col(ki_col)
            ki_probability = float(ed_ki / df_T) if df_T > 0.0 else 0.0
            ed_ki_ever = _read_col(ki_ever_col)
            ki_ever_probability = float(ed_ki_ever / df_T) if df_T > 0.0 else 0.0
        else:
            # KI columns were pruned (no leg reads KI): report 0, KO/pv are intact.
            ki_probability = 0.0
            ki_ever_probability = 0.0

        # Single-pass: use the value-sweep npv when the caller supplied one,
        # else fall back to an internal price() solve (standalone event-stats).
        pv = float(npv) if npv is not None else float(self.price(product, pricing_env))
        expected_discounted_maturity_cf = float(pv - float(np.sum(ed_ko_cf)))

        if want_coupon:
            extra_fields = self._extract_extra_event_stats(
                initial_grid,
                x_vec,
                spot_log,
                n_ko,
                ko_records,
                pricing_env,
                product,
                col_overrides=t0_overrides,
            )
        else:
            extra_fields = {}
        # The maturity cashflow is pv minus KO cashflows; for products with extra
        # cashflow streams (Phoenix coupons) also remove those so the decomposition
        # pv = sum(ko) + sum(coupon) + maturity stays correctly classified.
        coupon_cf = extra_fields.get("expected_discounted_coupon_cashflow")
        if coupon_cf is not None:
            expected_discounted_maturity_cf -= float(np.sum(coupon_cf))

        return self._make_event_stats(
            pv=pv,
            ko_times=ko_times,
            ko_probability=ko_probability,
            survival_probability=survival_probability,
            expected_discounted_ko_cashflow=ed_ko_cf,
            ki_probability=ki_probability,
            expected_discounted_maturity_cashflow=expected_discounted_maturity_cf,
            reconciliation_error=0.0,
            ki_times=ki_times,
            ki_event_probability=ki_event_probability,
            ki_survival_probability=ki_survival_probability,
            # Two unambiguous, cross-engine-consistent KI fields. The legacy
            # `ki_probability` keeps the PDE's historical "settles knocked-in"
            # meaning (KI indicator absorbed to 0 on any KO), which equals
            # `ki_survive_knocked_in_probability`. `ki_ever_probability` comes from
            # the dedicated KI-ever column that carries no KO absorption.
            ki_ever_probability=ki_ever_probability,
            ki_survive_knocked_in_probability=ki_probability,
            **extra_fields,
        )

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a Snowball option using Two-Surface PDE method.

        Args:
            product: SnowballOption
            pricing_env: Pricing environment with market data

        Returns:
            Dictionary with price, delta, gamma

        Raises:
            PricingError: If product is not a SnowballOption
            ValidationError: If product configuration is incompatible with PDE
        """
        self._check_product_type(product)

        if pricing_env is None:
            raise ValidationError(
                f"PricingEnvironment is required for {self._solver_name}"
            )

        # Validate PDE compatibility
        self._validate_product(product)

        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0 or is_zero(tau):
            # Expired: return terminal value with zero Greeks
            return {
                "price": self._calculate_terminal_value(product, spot, pricing_env),
                "delta": 0.0,
                "gamma": 0.0,
            }

        # Check if knocked out at valuation
        knocked_out_at_valuation = self._is_knocked_out_at_valuation(
            product, spot, pricing_env
        )
        if knocked_out_at_valuation:
            # KO payoff is fixed, so delta and gamma are zero
            return {
                "price": self._get_immediate_ko_payoff(product, pricing_env),
                "delta": 0.0,
                "gamma": 0.0,
            }

        # Solve PDE
        result = self._solve(product, pricing_env)

        # Extract price and Greeks from appropriate surface
        price = self._interpolate_price(
            result.solution_vec, result.x_vec, result.spot_log
        )
        delta, gamma = self._calculate_delta_gamma(
            result.solution_vec, result.x_vec, result.spot_log, spot
        )

        return {"price": price, "delta": delta, "gamma": gamma}

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
        """True when discrete event jumps use the dual-cell projection."""
        return (
            getattr(self.params, "event_projection", EventProjectionMode.NODAL)
            is EventProjectionMode.CELL_AVERAGE
        )

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

        Continuous/BGK KI is a continuous-barrier treatment whose value
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
            and not (self._ki_continuous or self._bgk_active)
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

    def _is_knocked_out_at_valuation(
        self, product: SnowballOption, spot: float, pricing_env: PricingEnvironment
    ) -> bool:
        """
        Determine KO state at valuation date (t=0).

        KO for SnowballOption is modeled as discrete in this solver; a KO breach at valuation
        only matters if there is a KO observation scheduled at t=0.
        """
        if product.barrier_config.ko_observation_type != ObservationType.DISCRETE:
            return False

        ko_records = product.resolve_ko_observations(pricing_env)
        ko_record_0 = self._find_record_at_time(ko_records, 0.0)
        if ko_record_0 is None:
            return False
        if ko_record_0.barrier is None:
            raise ValidationError(
                "KO observation at valuation requires a barrier level."
            )

        if product.is_reverse:
            return spot <= ko_record_0.barrier
        return spot >= ko_record_0.barrier

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
        """Calculate terminal payoff when already expired."""
        # Determine if knocked-in based on current spot
        knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        if not knocked_in:
            knocked_in = self._is_already_knocked_in(product, spot)
        return product.get_payoff(spot, pricing_env, knocked_in=knocked_in)

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
        """
        Build spatial and temporal grids for the Two-Surface PDE solver.

        Extends base class to:
        - Include all KO barriers and KI barrier in spatial grid
        - Align time grid with all observation times
        - Track observation indices for discrete barrier checks
        """
        # Resolve BGK state FIRST: the event/KI-monitor time selection below
        # (and in super()._build_grids) depends on self._bgk_active. Living
        # here — not in _solve — guarantees subclasses that override _solve
        # (e.g. KOResetSnowballPDESolver) cannot silently skip it.
        self._configure_bgk(product, pricing_env, sigma, tau)

        result = super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        x_vec, s_vec, dx_vec, t_vec, dt_vec = result

        # Store total time to maturity
        self._total_tau = tau

        # Clear previous observation tracking
        self._ko_observation_indices.clear()
        self._ki_observation_indices.clear()
        self._ki_barrier_by_tidx.clear()
        self._ko_terminal_record = None
        self._has_terminal_ko = False

        # Setup KO observation indices
        ko_records = self._get_cached_ko_records(pricing_env, product)
        for rec in ko_records:
            obs_time = rec.observation_time
            if is_close(obs_time, 0.0):
                self._ko_observation_indices[0] = rec
            elif is_close(obs_time, tau):
                self._ko_terminal_record = rec
                self._has_terminal_ko = True
            elif 0.0 < obs_time < tau:
                idx = self._aligned_time_index(t_vec, obs_time, "KO observation")
                self._ko_observation_indices[idx] = rec

        # Setup KI observation indices (if discrete).  ``_ki_nodes_in_grid``
        # gates this: when the grid was deliberately built WITHOUT interior KI
        # nodes (active BGK, or an already-knocked-in product) alignment is
        # neither possible nor needed, and demanding it would raise on a grid
        # that is correct by construction.
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
                elif 0.0 < obs_time <= tau:
                    idx = self._aligned_time_index(t_vec, obs_time, "KI observation")
                    self._ki_observation_indices.add(idx)
                    self._ki_barrier_by_tidx[idx] = float(barrier)

        return result

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
        if self._bgk_active:
            return float(self._bgk_ki_barrier)
        if not self._ki_continuous:
            mapped = self._ki_barrier_by_tidx.get(t_idx)
            if mapped is not None:
                return float(mapped)
        return float(self._ki_barrier)

    def _bgk_shifted_ki_barrier(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        sigma: float,
        tau: float,
    ) -> float:
        """Broadie-Glasserman-Kou continuity-corrected KI barrier [§11.6].

        Shift the discrete KI barrier AWAY from spot by ``exp(±beta*sigma*sqrt(dt))``
        with ``dt`` = the ACTUAL spacing between consecutive KI monitoring
        dates (median interval; the BGK correction assumes equal spacing, so a
        warning is logged for materially irregular schedules). A standard
        down-in barrier shifts DOWN (``-``), a reverse up-in barrier shifts UP
        (``+``). ``sigma`` is the same strike-selected constant vol used to
        build the operator, so the shift is consistent with the diffusion the
        grid resolves.
        """
        ki_barrier = product.barrier_config.ki_barrier
        if isinstance(ki_barrier, list):
            ki_barrier = ki_barrier[0]
        ki_barrier = float(ki_barrier)

        # _bgk_applicable guarantees at least TWO interior monitor times, so
        # an inter-observation spacing exists. Use the spacing BETWEEN
        # observations; the valuation-to-first-observation stub is not a
        # monitoring interval (a late-starting monitoring window would
        # otherwise pollute the median).
        times = np.sort(np.asarray(self._ki_monitor_times(product, tau), dtype=float))
        intervals = np.diff(times)
        intervals = intervals[intervals > Tolerance.ZERO]
        dt = float(np.median(intervals))
        if intervals.size > 1 and float(np.max(intervals)) > 1.5 * float(
            np.min(intervals)
        ):
            logging.warning(
                "BGK continuity correction assumes equally-spaced KI monitoring; "
                "this schedule's intervals range from %.6f to %.6f years. Using "
                "the median interval %.6f for the barrier shift.",
                float(np.min(intervals)),
                float(np.max(intervals)),
                dt,
            )

        shift = _BGK_BETA * float(sigma) * safe_sqrt(dt)
        return ki_barrier * safe_exp(shift if product.is_reverse else -shift)

    def _bgk_requested(self) -> bool:
        """True iff the opt-in BGK monitoring mode is selected on the params."""
        return (
            getattr(self.params, "ki_monitoring_mode", None)
            is KnockInMonitoringMode.BGK_APPROXIMATION
        )

    def _bgk_applicable(self, product: BaseEquityProduct, tau: float) -> bool:
        """BGK engages only for discretely-monitored KI with a genuine
        monitoring FREQUENCY (at least two interior dates, so an
        inter-observation spacing exists).

        European (maturity-only), continuous, no-KI, already-knocked-in, and
        single-interior-date products are priced unchanged under the exact
        path — a continuity correction has no meaningful dt for them.
        """
        return len(self._ki_monitor_times(product, tau)) >= 2

    def _configure_bgk(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        sigma: float,
        tau: float,
    ) -> None:
        """Resolve per-solve BGK state; log an inert note when opted-in but N/A.

        Must run before ``_build_grids`` so ``_time_grid_spec`` /
        ``_get_event_times`` see ``self._bgk_active`` and drop the interior daily
        KI nodes.
        """
        self._bgk_active = False
        self._bgk_ki_barrier = 0.0
        if not self._bgk_requested():
            return
        if not self._bgk_applicable(product, tau):
            logging.warning(
                "PDEParams.ki_monitoring_mode=BGK_APPROXIMATION is inert for this "
                "product: the Broadie-Glasserman-Kou continuity correction engages "
                "only for discretely-monitored knock-in (European / continuous / "
                "no-KI / already-knocked-in are priced unchanged under the exact "
                "path)."
            )
            return
        self._bgk_active = True
        self._bgk_ki_barrier = self._bgk_shifted_ki_barrier(
            product, pricing_env, sigma, tau
        )

    def _ki_nodes_in_grid(self, product: BaseEquityProduct) -> bool:
        """Whether the time grid carries the interior daily-KI nodes.

        SINGLE source of truth for a decision two places must agree on:
        ``_time_grid_spec`` builds the grid, and ``_build_grids`` demands that
        every KI observation land on a node.  A regime that drops the nodes
        but still demands them raises ``ValidationError`` at pricing time, so
        both read this predicate rather than re-deriving the condition.

        The nodes are dropped in two regimes:

        - active BGK — continuous monitoring at a shifted barrier replaces the
          discrete dates entirely [§11.6];
        - already knocked in — monitoring is moot. The readout comes off the
          V1 surface, and V1 never sees a KI jump (the jump writes V1 values
          INTO V0), so the KI indices cannot affect the answer.
        """
        if self._bgk_active:
            return False
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

        # BGK opt-in mode: the shifted KI barrier must be a spatial node so the
        # continuous KI mask lands on it exactly [§11.6]. Computed statelessly
        # from the env vol so the frozen-bump critical points capture it too.
        if self._bgk_requested():
            tau = product.get_maturity(pricing_env)
            if tau > 0 and self._bgk_applicable(product, tau):
                sigma = pricing_env.get_vol(product.strike, tau)
                points.append(
                    self._bgk_shifted_ki_barrier(product, pricing_env, sigma, tau)
                )

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
        Reads the barrier config directly (no pricing env / instance state), so
        it is safe to call during grid construction.
        """
        out = []
        cfg = getattr(product, "barrier_config", None)
        if cfg is not None:
            sched = cfg.ko_observation_schedule
            if sched is not None:
                out += [
                    rec.observation_time
                    for rec in sched.records
                    if rec.observation_time is not None
                ]
            elif cfg.ko_observation_dates is not None:
                out += list(cfg.ko_observation_dates)
        return sorted({float(t) for t in out if t is not None and 0.0 < float(t) < tau})

    def _ki_monitor_times(
        self, product: BaseEquityProduct, tau: float
    ) -> List[float]:
        """Interior KI monitoring dates — only for daily-discrete KI.

        Empty for every other regime (spec §4 table): European (maturity-only
        => no interior dates), continuous, no-KI, and already-knocked-in
        (monitoring moot).  Reads config directly; ``ki_continuous`` here is
        derived identically to the solver's ``self._ki_continuous``.
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
        if sched is not None:
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
        self, product: BaseEquityProduct, market: MarketSnapshot, tau: float
    ) -> GridRequest:
        """Consolidated geometry declaration — ALL regime gating in one place.

        Reuses the certified helpers verbatim: KO/coupon dates from
        ``_ko_coupon_align_times`` (must-be-nodes), daily-KI dates from
        ``_ki_monitor_times`` gated by ``_ki_nodes_in_grid`` (dropped under
        active BGK and for already-knocked-in products). Requires BGK state
        to be resolved first (``_configure_bgk``) — ``_solve`` guarantees the
        order; external callers go through ``_prepare_for_request``.
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
        if self._bgk_active:
            criticals.append(float(self._bgk_ki_barrier))
        return GridRequest(
            tau=tau,
            bound_anchors=(market.spot, strike),
            critical_prices=tuple(p for p in criticals if p and p > 0),
            hard_lower=None,
            hard_upper=None,
            event_times=tuple(sorted(set(align) | set(monitor))),
        )

    def _prepare_for_request(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """Resolve the solve state grid_request depends on; returns tau."""
        tau = product.get_maturity(pricing_env)
        self._prepare_solve_state(product, pricing_env)
        sigma = pricing_env.get_vol(product.strike, tau)
        self._configure_bgk(product, pricing_env, sigma, tau)
        return tau

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

    def event_schedule(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        layout: Layout,
    ) -> EventSchedule:
        """Interior + continuous stages for the two-surface solve.

        Terminal (t = tau) and valuation-date (t = 0) events keep their
        certified inline handling in ``_solve`` / the t0 readout; interior
        discrete events and the continuous KI coupling live here as pure
        transforms over ``{"alive": V0_col, "ki": V1_col}``.
        """
        spatial = layout.spatial
        tau = layout.request.tau
        is_reverse = bool(getattr(product, "is_reverse", False))
        interior: dict = {}

        def _chain(prev, fn):
            if prev is None:
                return fn
            return lambda states: fn(prev(states))

        # KO: both surfaces projected against the discounted redemption
        # (breach_up = up-barrier XOR reverse — certified direction rule).
        for rec in self._get_cached_ko_records(pricing_env, product):
            t_obs = rec.observation_time
            if (
                t_obs is None
                or is_close(t_obs, 0.0)
                or is_close(t_obs, tau)
                or not (0.0 < t_obs < tau)
            ):
                continue
            step = layout.time.step_at(t_obs)
            cash = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=rec.payoff if rec.payoff is not None else 0.0,
                current_time=t_obs,
                settlement_time=rec.settlement_time,
            )
            barrier = float(rec.barrier)
            breach_up = not is_reverse
            ko_hits_ki = self._ko_survives_ki(product)

            def _ko(states, _b=barrier, _c=cash, _up=breach_up, _hits_ki=ko_hits_ki):
                cash_col = np.full_like(states["alive"], _c)
                return {
                    "alive": project_between(
                        spatial, _b, _up, cash_col, states["alive"]
                    ),
                    "ki": project_between(
                        spatial, _b, _up, cash_col, states["ki"]
                    )
                    if _hits_ki
                    else states["ki"],
                }

            interior[step] = _chain(interior.get(step), _ko)

        # Discrete KI: alive <- ki below the (possibly per-date) barrier.
        # Applied AFTER any coincident KO (certified order: KO first).
        if (
            product.has_ki_barrier
            and not self._ki_continuous
            and self._ki_nodes_in_grid(product)
        ):
            ki_profile = self._get_cached_ki_profile(pricing_env, product)
            ki_times = ki_profile["observation_times"]
            ki_barriers = ki_profile.get("barriers") or []
            for obs_idx, t_obs in enumerate(ki_times):
                if t_obs is None or not (0.0 < t_obs < tau):
                    continue
                if is_close(t_obs, 0.0) or is_close(t_obs, tau):
                    continue
                barrier = None
                if obs_idx < len(ki_barriers):
                    barrier = ki_barriers[obs_idx]
                if barrier is None:
                    barrier = self._ki_barrier
                step = layout.time.step_at(t_obs)
                breach_up = is_reverse  # KI is a down barrier (up if reverse)

                def _ki(states, _b=float(barrier), _up=breach_up):
                    return {
                        "alive": project_between(
                            spatial, _b, _up, states["ki"], states["alive"]
                        ),
                        "ki": states["ki"],
                    }

                interior[step] = _chain(interior.get(step), _ki)

        # Continuous (or BGK-shifted continuous) KI: nodal coupling per step.
        continuous = None
        if product.has_ki_barrier and (self._ki_continuous or self._bgk_active):
            kib = (
                float(self._bgk_ki_barrier)
                if self._bgk_active
                else float(self._ki_barrier)
            )
            s = spatial.s

            def continuous(step, states, _b=kib, _rev=is_reverse):
                mask = (s >= _b) if _rev else (s <= _b)
                alive = states["alive"].copy()
                alive[mask] = states["ki"][mask]
                # FIRST_PASSAGE correction: restore the intra-step crossing
                # mass the nodal jump cannot see (see _apply_ki_jump).
                fp = self._ki_fp
                if fp is not None and step > 0:
                    d = states["ki"] - alive
                    alive += fp.step_correction(step, s, _b, d)
                return {"alive": alive, "ki": states["ki"]}

        return EventSchedule(interior=interior, continuous=continuous)

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
        grid[:, -1] = payoffs

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
        grid[:, -1] = payoffs

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
        barrier = ko_record.barrier
        payoff = ko_record.payoff if ko_record.payoff is not None else 0.0

        # Discount payoff if settlement is different from observation
        cashflow_value = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=payoff,
            current_time=self._total_tau,
            settlement_time=ko_record.settlement_time,
        )

        grids = (grid_v0, grid_v1) if self._ko_survives_ki(product) else (grid_v0,)

        if self._use_cell_average_events():
            for grid in grids:
                grid[:, -1] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    grid[:, -1], cashflow_value,
                )
            return

        # Apply to breached region (KO is an UP barrier)
        mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

        for grid in grids:
            grid[mask, -1] = cashflow_value

    def _time_stepping_two_surface(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_vec: np.ndarray,
        dt_vec: np.ndarray,
        product: SnowballOption,
        pricing_env: PricingEnvironment,
        r: float,
        q: float,
        sigma: float,
        tau: float,
        step_coeffs=None,
    ) -> None:
        """
        Backward time stepping for both V0 and V1 surfaces.

        At each time step:
        1. Step both surfaces backward using Crank-Nicolson
        2. Apply boundary conditions
        3. Apply KO jump (if observation time)
        4. Apply KI jump (if observation time or continuous)
        """
        params: PDEParams = self.params
        num_t, num_x = len(t_vec), len(x_vec)
        # FIRST_PASSAGE continuous-KI state for _apply_ki_jump and the
        # schedule's continuous stage (inert unless continuous KI is live).
        self._prepare_continuous_ki_correction(product, pricing_env, t_vec)
        profile = self._profile_enabled
        timings = self._profile_stats
        use_banded = params.use_banded_solver
        n_int = num_x - 2
        I_int = sp.eye(n_int, format="csc")
        self._matrix_cache.clear()
        self._banded_cache.clear()
        self._term_A_cache = {}

        # Canonical damping schedule (terminal Rannacher + event smoothing):
        # from the layout's frozensets on the migrated path, else legacy.
        theta_schedule = self._theta_schedule_from_layout(self._active_layout)

        rhs = None
        rhs_v0 = None
        rhs_v1 = None
        if use_banded and n_int > 2:
            rhs = np.empty((n_int, 2))
            rhs_v0 = rhs[:, 0]
            rhs_v1 = rhs[:, 1]

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            current_time = t_vec[j]
            tau_remaining = tau - current_time
            theta = float(theta_schedule[j])
            if step_coeffs is not None:
                coeff_key = int(step_coeffs.set_index[j])
                l, c, u = step_coeffs.lcu_sets[coeff_key]
            else:
                coeff_key = 0

            # Set boundary conditions for both surfaces
            if profile:
                t0 = perf_counter()
            self._set_boundary_conditions_v0(
                grid_v0, x_vec, s_vec, j, tau_remaining, product, pricing_env
            )
            self._set_boundary_conditions_v1(
                grid_v1, x_vec, s_vec, j, tau_remaining, product, pricing_env
            )
            if profile:
                timings["boundary"] += perf_counter() - t0

            if use_banded and n_int > 2:
                if profile:
                    t0 = perf_counter()
                banded, lower1, main1, upper1 = self._get_banded_system(
                    l, c, u, dt, theta, coeff_key=coeff_key
                )
                if profile:
                    timings["matrix_build"] += perf_counter() - t0

                v0_next = grid_v0[1:-1, j + 1]
                v1_next = grid_v1[1:-1, j + 1]

                if profile:
                    t0 = perf_counter()
                np.multiply(main1, v0_next, out=rhs_v0)
                rhs_v0[1:] += lower1 * v0_next[:-1]
                rhs_v0[:-1] += upper1 * v0_next[1:]
                self._inject_boundary_contributions(rhs_v0, grid_v0, l, u, j, dt, theta)

                np.multiply(main1, v1_next, out=rhs_v1)
                rhs_v1[1:] += lower1 * v1_next[:-1]
                rhs_v1[:-1] += upper1 * v1_next[1:]
                self._inject_boundary_contributions(rhs_v1, grid_v1, l, u, j, dt, theta)
                if profile:
                    timings["rhs"] += perf_counter() - t0

                if profile:
                    t0 = perf_counter()
                sol = solve_banded(
                    (1, 1),
                    banded,
                    rhs,
                    overwrite_b=True,
                    check_finite=False,
                )
                if profile:
                    timings["solve"] += perf_counter() - t0
                grid_v0[1:-1, j] = sol[:, 0]
                grid_v1[1:-1, j] = sol[:, 1]
            else:
                if profile:
                    t0 = perf_counter()
                if step_coeffs is not None:
                    A = self._operator_matrix_for_set(step_coeffs, coeff_key, num_x)
                M1, M2_lu = self._get_matrices(I_int, A, dt, theta, coeff_key=coeff_key)
                if profile:
                    timings["matrix_build"] += perf_counter() - t0

                if profile:
                    t0 = perf_counter()
                rhs_v0 = M1 @ grid_v0[1:-1, j + 1]
                self._inject_boundary_contributions(rhs_v0, grid_v0, l, u, j, dt, theta)

                rhs_v1 = M1 @ grid_v1[1:-1, j + 1]
                self._inject_boundary_contributions(rhs_v1, grid_v1, l, u, j, dt, theta)
                if profile:
                    timings["rhs"] += perf_counter() - t0

                if profile:
                    t0 = perf_counter()
                grid_v0[1:-1, j] = M2_lu.solve(rhs_v0)
                grid_v1[1:-1, j] = M2_lu.solve(rhs_v1)
                if profile:
                    timings["solve"] += perf_counter() - t0

            # Apply barrier modifications. The valuation-date column is
            # captured BEFORE its events: t=0 events are deterministic at the
            # known spot, so the readout interpolates this smooth 0+ branch
            # and applies today's transitions pointwise instead of blending
            # across the nodal jump the event application writes.
            if profile:
                t0 = perf_counter()
            if j == 0 and self._t0_has_events(product):
                self._t0_pre_event_cols = (
                    grid_v0[:, 0].copy(),
                    grid_v1[:, 0].copy(),
                )
            schedule = self._active_schedule
            if schedule is not None and j > 0:
                # Migrated path: pure stage transforms on the named blocks.
                # (j == 0 keeps the certified valuation-date handling below —
                # inclusive t0 masks + readout are endpoint stages.)
                states = {"alive": grid_v0[:, j], "ki": grid_v1[:, j]}
                states = schedule.apply(j, states)
                states = schedule.continuous(j, states)
                grid_v0[:, j] = states["alive"]
                grid_v1[:, j] = states["ki"]
            else:
                self._apply_step_modifications_two_surface(
                    grid_v0, grid_v1, x_vec, s_vec, j, tau_remaining,
                    product, pricing_env,
                )
            if profile:
                timings["barrier"] += perf_counter() - t0

    def _get_banded_system(
        self,
        l: np.ndarray,
        c: np.ndarray,
        u: np.ndarray,
        dt: float,
        theta: float,
        coeff_key: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pack = self._session_banded_pack
        if pack is not None:
            hit = pack.get((coeff_key, round(dt, 12), round(theta, 12)))
            if hit is not None:
                return hit
        if not self._is_cache_enabled():
            lower = -theta * dt * l[2:-1]
            main = 1.0 - theta * dt * c[1:-1]
            upper = -theta * dt * u[1:-2]

            banded = np.zeros((3, len(main)))
            banded[0, 1:] = upper
            banded[1, :] = main
            banded[2, :-1] = lower

            lower1 = (1.0 - theta) * dt * l[2:-1]
            main1 = 1.0 + (1.0 - theta) * dt * c[1:-1]
            upper1 = (1.0 - theta) * dt * u[1:-2]
            return banded, lower1, main1, upper1

        key = (coeff_key, round(dt, 12), round(theta, 12))
        cached = self._banded_cache.get(key)
        if cached is not None:
            self._banded_cache.move_to_end(key)
            return cached

        lower = -theta * dt * l[2:-1]
        main = 1.0 - theta * dt * c[1:-1]
        upper = -theta * dt * u[1:-2]

        banded = np.zeros((3, len(main)))
        banded[0, 1:] = upper
        banded[1, :] = main
        banded[2, :-1] = lower

        lower1 = (1.0 - theta) * dt * l[2:-1]
        main1 = 1.0 + (1.0 - theta) * dt * c[1:-1]
        upper1 = (1.0 - theta) * dt * u[1:-2]

        self._banded_cache[key] = (banded, lower1, main1, upper1)
        self._banded_cache.move_to_end(key)
        if len(self._banded_cache) > self._banded_cache_max_entries:
            self._banded_cache.popitem(last=False)
        return banded, lower1, main1, upper1

    def _pack_uses_banded(self, num_x: int) -> bool:
        # Mirrors _time_stepping_two_surface: banded branch engages when the
        # params request it and the interior system is non-trivial.
        return bool(self.params.use_banded_solver) and (num_x - 2) > 2

    def _pack_banded_entry(
        self, banded: dict, step_coeffs, l, c, u, dt, theta, coeff_key
    ) -> None:
        # Route through _get_banded_system itself (single construction
        # implementation) against a throwaway cache. FIRST build wins,
        # mirroring the engine's own cache: distinct dt floats can round to
        # one 12-digit key, and the march reuses the first-built entry for
        # all of them — the pack must hold that same entry bitwise.
        key = (coeff_key, round(dt, 12), round(theta, 12))
        if key in banded:
            return
        saved, self._banded_cache = self._banded_cache, OrderedDict()
        try:
            entry = self._get_banded_system(l, c, u, dt, theta, coeff_key=coeff_key)
            banded[key] = entry
        finally:
            self._banded_cache = saved

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
        current_time = self._total_tau - tau
        df_to_maturity = self._df_between_times(
            pricing_env, current_time, self._total_tau
        )

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
            grid[0, t_idx] = (
                self._grid_v1[0, t_idx] if self._grid_v1 is not None else 0.0
            )
        else:
            grid[0, t_idx] = (principal + rebate) * df_to_maturity

        # Upper boundary (S -> ∞)
        # Check if above all KO barriers - if so, value is KO payoff
        max_ko_barrier = self._get_max_ko_barrier(product)
        if s_vec[-1] >= max_ko_barrier:
            # Use current KO payoff
            ko_payoff = self._get_ko_payoff_at_time(
                product, pricing_env, current_time, t_idx
            )
            grid[-1, t_idx] = ko_payoff
        else:
            if (
                self.params.boundary_mode == "asymptotic"
                and product.payoff_config.call_rebate_enabled
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
                grid[-1, t_idx] = (
                    principal * df
                    + participation * (s_vec[-1] * df_div - strike * df)
                )
            else:
                # Deep OTM: principal + rebate (discounted)
                grid[-1, t_idx] = (principal + rebate) * df_to_maturity

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
        current_time = self._total_tau - tau
        df_to_maturity = self._df_between_times(
            pricing_env, current_time, self._total_tau
        )

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
            grid[0, t_idx] = principal * df_to_maturity
        else:
            if (
                self.params.boundary_mode == "asymptotic"
                and product.payoff_config.protection_type == ProtectionType.NONE
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
                grid[0, t_idx] = (
                    principal * df
                    + slope * (s_vec[0] * df_div - effective_strike * df)
                )
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
                grid[0, t_idx] = (principal + max_loss) * df_to_maturity

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
                grid[-1, t_idx] = (
                    (principal + slope * effective_strike) * df
                    - slope * s_vec[-1] * df_div
                )
            else:
                if protection == ProtectionType.PARTIAL:
                    floor = (
                        product.payoff_config.protection_rate
                        * product.initial_price
                        * product.contract_multiplier
                    )
                    grid[-1, t_idx] = (principal - floor) * df_to_maturity
                else:
                    grid[-1, t_idx] = principal * df_to_maturity
        else:
            # Put is worthless at high S
            grid[-1, t_idx] = principal * df_to_maturity

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
        """
        current_time = self._total_tau - tau

        # 1. Apply KO jump if this is a KO observation time
        ko_record = self._ko_observation_indices.get(t_idx)
        if ko_record is not None:
            self._apply_ko_jump(
                grid_v0,
                grid_v1,
                s_vec,
                t_idx,
                current_time,
                product,
                pricing_env,
                ko_record,
            )

        # 2. Apply KI jump
        # For continuous KI: apply at every time step
        # For discrete KI: apply only at observation times
        if product.has_ki_barrier:
            should_apply_ki = (
                self._ki_continuous
                or self._bgk_active
                or t_idx in self._ki_observation_indices
            )
            if should_apply_ki:
                self._apply_ki_jump(grid_v0, grid_v1, s_vec, t_idx, product)

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
        barrier = ko_record.barrier
        payoff = ko_record.payoff if ko_record.payoff is not None else 0.0

        # Discount payoff based on settlement time
        cashflow_value = self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=payoff,
            current_time=current_time,
            settlement_time=ko_record.settlement_time,
        )

        # V1 is the knocked-in surface; disable_ko_after_ki removes it here.
        grids = (grid_v0, grid_v1) if self._ko_survives_ki(product) else (grid_v0,)

        if self._event_uses_projection(t_idx):
            for grid in grids:
                grid[:, t_idx] = self._project_event_values(
                    s_vec, barrier, product.is_reverse, True,
                    grid[:, t_idx], cashflow_value,
                )
            return

        # Determine breached region (KO is an UP barrier)
        mask = self._event_nodal_mask(
            s_vec, barrier, product.is_reverse, True, at_valuation=(t_idx == 0)
        )

        for grid in grids:
            grid[mask, t_idx] = cashflow_value

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
        ki_barrier = self._resolve_ki_barrier_at_tidx(t_idx)

        # Continuous (or BGK-continuous) KI monitoring is a continuous-barrier
        # treatment applied every step: it stays a nodal mask regardless of the
        # event_projection setting. Only discretely observed KI events project.
        ki_discrete = not (self._ki_continuous or self._bgk_active)
        if self._event_uses_projection(t_idx) and ki_discrete:
            grid_v0[:, t_idx] = self._project_event_values(
                s_vec, ki_barrier, product.is_reverse, False,
                grid_v0[:, t_idx], grid_v1[:, t_idx],
            )
            return

        # Determine breached region (KI is a DOWN barrier)
        mask = self._event_nodal_mask(
            s_vec, ki_barrier, product.is_reverse, False,
            at_valuation=(t_idx == 0 and ki_discrete),
        )

        # V0 transitions to V1 in breached region
        grid_v0[mask, t_idx] = grid_v1[mask, t_idx]

        # FIRST_PASSAGE correction [2026-08-18]: the jump above monitors the
        # continuous barrier only at step boundaries, so crossings INSIDE the
        # step are invisible and the live surface is biased high by
        # O(sqrt(dt)). Mix the live region toward V1 with the exact
        # touch-and-return probability for the step (paths ending breached
        # are already captured by the mask at the neighbouring column).
        # Interior steps only: the terminal column has no step interior, and
        # the valuation column's events are owned by the t0 readout.
        fp = self._ki_fp
        if fp is not None and 0 < t_idx < grid_v0.shape[1] - 1:
            d = grid_v1[:, t_idx] - grid_v0[:, t_idx]
            grid_v0[:, t_idx] += fp.step_correction(t_idx, s_vec, ki_barrier, d)

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
        """Get KO payoff for current time/index."""
        ko_record = self._ko_observation_indices.get(t_idx)
        if ko_record is not None:
            payoff = ko_record.payoff if ko_record.payoff is not None else 0.0
            return self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=payoff,
                current_time=current_time,
                settlement_time=ko_record.settlement_time,
            )

        # Boundary fallback: use next scheduled KO record (ignore past observations).
        ko_records = self._get_cached_ko_records(pricing_env, product)
        future_records = [
            rec for rec in ko_records if self._record_is_non_negative_time(rec)
        ]
        if not future_records:
            return 0.0

        next_rec: Optional[ResolvedObservationRecord] = None
        for rec in future_records:
            if is_greater_than_or_close(rec.observation_time, current_time):
                next_rec = rec
                break
        next_rec = next_rec if next_rec is not None else future_records[-1]
        return self._cashflow_value_at_time(
            pricing_env=pricing_env,
            cashflow=next_rec.payoff or 0.0,
            current_time=current_time,
            settlement_time=next_rec.settlement_time,
        )

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
        return float(df), float(df_div)

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

    def _observation_cache_key(
        self, pricing_env: PricingEnvironment, product: SnowballOption, kind: str
    ) -> Tuple:
        strategy = self._resolve_cache_strategy()
        return (
            kind,
            strategy,
            f"{product.__class__.__module__}.{product.__class__.__qualname__}",
            self._product_cache_token(product, strategy),
            pricing_env.valuation_date,
            pricing_env.day_count_convention,
            pricing_env.bus_days_in_year,
        )

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
        if len(self._ko_records_cache) > self.params.grid_cache_max_entries:
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
        if len(self._ki_profile_cache) > self.params.grid_cache_max_entries:
            self._ki_profile_cache.popitem(last=False)
        return profile
