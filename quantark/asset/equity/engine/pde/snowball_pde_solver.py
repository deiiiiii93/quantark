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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from quantark.asset.equity.engine.pde.base_pde_solver import (
    BasePDESolver,
    PDESolutionResult,
    TimeGridSpec,
)
from quantark.asset.equity.engine.event_stats import AutocallableEventStats
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ProtectionType
from quantark.util.exceptions import PricingError, ValidationError
from quantark.util.numerical import (
    Tolerance,
    is_close,
    is_greater_than_or_close,
    is_zero,
    safe_divide,
)


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
        5. Interpolate final price from V0 (or V1 if already knocked-in)
    """

    # Subclasses can override this to specify their supported product type
    _supported_product_type: type = SnowballOption
    _solver_name: str = "SnowballPDESolver"

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
        self._is_reverse: bool = False

        # Time tracking
        self._total_tau: float = 0.0
        self._banded_cache: "OrderedDict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
        self._banded_cache_max_entries = self.params.banded_cache_max_entries
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

        # Determine knocked-in state at valuation
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )

        # Store the state for potential use in calculate_greeks
        self._knocked_in_at_valuation = knocked_in_at_valuation

        # Extract market data
        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # Store product properties for later use
        self._is_reverse = product.is_reverse
        self._ki_continuous = ki_continuous
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                self._ki_barrier = ki_barrier[0]
            else:
                self._ki_barrier = ki_barrier

        if self._profile_enabled:
            self._reset_profile_stats()

        # Build grids
        if self._profile_enabled:
            t0 = perf_counter()
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
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
            is_terminal_ki = self._ki_continuous
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
        )

        # Return appropriate surface based on knocked-in state
        spot_log = np.log(spot)
        if knocked_in_at_valuation:
            solution_vec = self._grid_v1[:, 0]
        else:
            solution_vec = self._grid_v0[:, 0]

        return PDESolutionResult(
            solution_vec=solution_vec,
            x_vec=x_vec,
            s_vec=s_vec,
            spot_log=spot_log,
        )

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
            return self._calculate_terminal_value(product, spot, pricing_env)

        # Check if knocked out at valuation
        knocked_out_at_valuation = self._is_knocked_out_at_valuation(
            product, spot, pricing_env
        )
        if knocked_out_at_valuation:
            return self._get_immediate_ko_payoff(product, pricing_env)

        # Solve PDE and interpolate price
        result = self._solve(product, pricing_env)
        return self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log)

    def calculate_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """Provide per-observation KO probabilities and expected discounted cashflows."""
        if not isinstance(product, self._event_stats_product_type()):
            return None
        if pricing_env is None:
            return None
        return self._compute_event_stats(product, pricing_env)

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
        self, initial_grid, x_vec, spot_log, n_ko, ko_records, pricing_env, product
    ) -> dict:
        """Extra event-stats fields from the extra columns (none for Snowball)."""
        return {}

    def _compute_event_stats(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Optional[AutocallableEventStats]:
        """
        Native PDE implementation:
        - Propagates stacked indicator surfaces through the same backward PDE stepping.
        - Applies KO/KI jumps to all indicator surfaces at observation times.
        - Returns KO per-observation probabilities (by dividing discounted indicators by
          discount factors) and expected discounted KO cashflows.
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
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                self._ki_barrier = ki_barrier[0]
            else:
                self._ki_barrier = ki_barrier

        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )
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
        n_extra = self._n_extra_event_cols(n_ko)
        ki_col = n_ko + n_extra
        ki_ever_col = n_ko + n_extra + 1
        n_cols = n_ko + n_extra + 2

        # Terminal conditions at maturity (t = T):
        # - KO indicators are zero at maturity (KO only at discrete observations via jumps)
        # - Both KI indicators are 1 on the KI surface and 0 on the no-KI surface
        v0_next = np.zeros((num_x, n_cols), dtype=float)
        v1_next = np.zeros((num_x, n_cols), dtype=float)
        v1_next[:, ki_col] = 1.0
        v1_next[:, ki_ever_col] = 1.0

        # Apply terminal KO/KI events at maturity if observation schedules include t=T.
        terminal_tidx = num_t - 1
        terminal_ko_idx = ko_index_by_tidx.get(terminal_tidx)
        if terminal_ko_idx is not None:
            rec = ko_records[terminal_ko_idx]
            barrier = float(rec.barrier) if rec.barrier is not None else 0.0
            mask_ko = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

            # KI-ever is exempt from KO absorption (pure first-passage statistic).
            ever0 = v0_next[mask_ko, ki_ever_col].copy()
            ever1 = v1_next[mask_ko, ki_ever_col].copy()
            v0_next[mask_ko, :] = 0.0
            v1_next[mask_ko, :] = 0.0
            df_delay = self._cashflow_value_at_time(
                pricing_env=pricing_env,
                cashflow=1.0,
                current_time=float(t_vec[terminal_tidx]),
                settlement_time=rec.settlement_time,
            )
            v0_next[mask_ko, terminal_ko_idx] = df_delay
            v1_next[mask_ko, terminal_ko_idx] = df_delay
            v0_next[mask_ko, ki_ever_col] = ever0
            v1_next[mask_ko, ki_ever_col] = ever1
            self._set_extra_event_indicators(
                v0_next, v1_next, s_vec, n_ko, terminal_ko_idx, rec,
                product, pricing_env, t_vec, terminal_tidx,
            )

        is_terminal_ki = product.has_ki_barrier and (
            self._ki_continuous or terminal_tidx in self._ki_observation_indices
        )
        if is_terminal_ki:
            ki_barrier = self._resolve_ki_barrier_at_tidx(terminal_tidx)
            mask_ki = self._get_barrier_mask(s_vec, ki_barrier, product.is_reverse, is_up_barrier=False)
            v0_next[mask_ki, :] = v1_next[mask_ki, :]

        # Operator coefficients and banded solver setup
        params: PDEParams = self.params
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        use_banded = params.use_banded_solver and (num_x - 2) > 2
        if not use_banded:
            raise ValidationError("Event stats PDE currently requires banded solver path.")

        # Rannacher smoothing indices (reuse the same rule as the pricing solver).
        smooth_js: set[int] = set()
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            event_times = self._get_event_times(product, tau)
            if event_times:
                for et in event_times:
                    idx = int(np.argmin(np.abs(t_vec - et)))
                    if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                        for k in range(params.rannacher_steps):
                            smooth_idx = idx - 1 - k
                            if smooth_idx >= 0:
                                smooth_js.add(smooth_idx)

        n_int = num_x - 2
        rhs = np.empty((n_int, 2 * n_cols), dtype=float)

        for j in range(num_t - 2, -1, -1):
            dt = float(dt_vec[j])
            steps_from_end = num_t - 1 - j
            theta = (
                1.0
                if params.use_rannacher
                and (steps_from_end < params.rannacher_steps or j in smooth_js)
                else params.theta
            )

            banded, lower1, main1, upper1 = self._get_banded_system(l, c, u, dt, theta)

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

            # Apply KO jump (if observation time).
            ko_idx = ko_index_by_tidx.get(j)
            if ko_idx is not None:
                rec = ko_records[ko_idx]
                barrier = float(rec.barrier) if rec.barrier is not None else 0.0
                mask_ko = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

                # Zero all event surfaces in KO region, then set the KO_i indicator.
                # KI-ever is exempt (pure first-passage statistic, no KO absorption).
                ever0 = v0_cur[mask_ko, ki_ever_col].copy()
                ever1 = v1_cur[mask_ko, ki_ever_col].copy()
                v0_cur[mask_ko, :] = 0.0
                v1_cur[mask_ko, :] = 0.0
                df_delay = self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=1.0,
                    current_time=float(t_vec[j]),
                    settlement_time=rec.settlement_time,
                )
                v0_cur[mask_ko, ko_idx] = df_delay
                v1_cur[mask_ko, ko_idx] = df_delay
                v0_cur[mask_ko, ki_ever_col] = ever0
                v1_cur[mask_ko, ki_ever_col] = ever1
                self._set_extra_event_indicators(
                    v0_cur, v1_cur, s_vec, n_ko, ko_idx, rec,
                    product, pricing_env, t_vec, j,
                )

            # Apply KI jump (continuous or discrete at observation indices).
            if product.has_ki_barrier:
                should_apply_ki = self._ki_continuous or j in self._ki_observation_indices
                if should_apply_ki:
                    ki_barrier = self._resolve_ki_barrier_at_tidx(j)
                    mask_ki = self._get_barrier_mask(s_vec, ki_barrier, product.is_reverse, is_up_barrier=False)
                    v0_cur[mask_ki, :] = v1_cur[mask_ki, :]

            # Enforce simple Neumann-like boundary (zero slope) for stability.
            v0_cur[0, :] = v0_cur[1, :]
            v0_cur[-1, :] = v0_cur[-2, :]
            v1_cur[0, :] = v1_cur[1, :]
            v1_cur[-1, :] = v1_cur[-2, :]

            v0_next = v0_cur
            v1_next = v1_cur

        # Select initial regime based on knocked-in at valuation.
        initial_grid = v1_next if knocked_in_at_valuation else v0_next
        spot_log = float(np.log(spot))

        ed_unit = np.array(
            [float(np.interp(spot_log, x_vec, initial_grid[:, i])) for i in range(n_ko)],
            dtype=float,
        )
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
        else:
            df_T = pricing_env.get_discount_factor(float(tau))
            ed_ki = float(np.interp(spot_log, x_vec, initial_grid[:, ki_col]))
            ki_probability = float(ed_ki / df_T) if df_T > 0.0 else 0.0
            ed_ki_ever = float(np.interp(spot_log, x_vec, initial_grid[:, ki_ever_col]))
            ki_ever_probability = float(ed_ki_ever / df_T) if df_T > 0.0 else 0.0

        pv = float(self.price(product, pricing_env))
        expected_discounted_maturity_cf = float(pv - float(np.sum(ed_ko_cf)))

        extra_fields = self._extract_extra_event_stats(
            initial_grid, x_vec, spot_log, n_ko, ko_records, pricing_env, product
        )
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

        # Setup KI observation indices (if discrete)
        if product.has_ki_barrier and not self._ki_continuous:
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

    def _resolve_ki_barrier_at_tidx(self, t_idx: int) -> float:
        """Resolve KI barrier for a specific PDE time index."""
        if not self._ki_continuous:
            mapped = self._ki_barrier_by_tidx.get(t_idx)
            if mapped is not None:
                return float(mapped)
        return float(self._ki_barrier)

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

    def _time_grid_spec(self, product, tau) -> "TimeGridSpec":
        """Decoupled time-grid concerns for autocallables (spec §4 Component 1).

        align = KO/coupon dates (must be nodes); monitor = daily-discrete KI
        dates (resolution only); steps_per_day from params.
        """
        return TimeGridSpec(
            align_times=self._ko_coupon_align_times(product, tau),
            monitor_times=self._ki_monitor_times(product, tau),
            steps_per_day=float(self.params.event_steps_per_day),
        )

    def _get_event_times(
        self, product: BaseEquityProduct, tau: float
    ) -> Optional[List[float]]:
        """Back-compat union used by Rannacher damping and the grid cache key.

        Returns ``sorted(align ∪ monitor)`` so damping still fires at KO **and**
        discrete-KI dates independent of any downstream stream selection
        [§11.2].  Correctness-critical alignment lives in
        ``_ko_coupon_align_times``; ``_ki_monitor_times`` adds resolution.
        """
        union = sorted(
            set(self._ko_coupon_align_times(product, tau))
            | set(self._ki_monitor_times(product, tau))
        )
        return union or None

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

        # Apply to breached region (KO is an UP barrier)
        mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

        grid_v0[mask, -1] = cashflow_value
        grid_v1[mask, -1] = cashflow_value

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
        profile = self._profile_enabled
        timings = self._profile_stats
        use_banded = params.use_banded_solver
        n_int = num_x - 2
        I_int = sp.eye(n_int, format="csc")
        self._matrix_cache.clear()
        self._banded_cache.clear()

        # Rannacher smoothing indices
        smooth_js = set()
        event_theta = params.event_theta
        event_steps = params.event_rannacher_steps
        if params.use_rannacher and params.auto_grid and params.rannacher_at_events:
            event_times = self._get_event_times(product, tau)
            if event_times and event_steps > 0:
                for et in event_times:
                    idx = int(np.argmin(np.abs(t_vec - et)))
                    if 0 < idx < num_t - 1 and is_close(float(t_vec[idx]), float(et)):
                        for k in range(event_steps):
                            smooth_idx = idx - 1 - k
                            if smooth_idx >= 0:
                                smooth_js.add(smooth_idx)

        rhs = None
        rhs_v0 = None
        rhs_v1 = None
        if use_banded and n_int > 2:
            rhs = np.empty((n_int, 2))
            rhs_v0 = rhs[:, 0]
            rhs_v1 = rhs[:, 1]

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            steps_from_end = num_t - 1 - j
            current_time = t_vec[j]
            tau_remaining = tau - current_time

            # Determine theta (Rannacher smoothing uses backward Euler)
            theta = params.theta
            if params.use_rannacher and steps_from_end < params.rannacher_steps:
                theta = 1.0
            elif j in smooth_js:
                theta = event_theta

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
                    l, c, u, dt, theta
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
                M1, M2_lu = self._get_matrices(I_int, A, dt, theta)
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

            # Apply barrier modifications
            if profile:
                t0 = perf_counter()
            self._apply_step_modifications_two_surface(
                grid_v0, grid_v1, x_vec, s_vec, j, tau_remaining, product, pricing_env
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        key = (round(dt, 12), round(theta, 12))
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
                self._ki_continuous or t_idx in self._ki_observation_indices
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

        # Determine breached region (KO is an UP barrier)
        mask = self._get_barrier_mask(s_vec, barrier, product.is_reverse, is_up_barrier=True)

        # Apply to both surfaces
        grid_v0[mask, t_idx] = cashflow_value
        grid_v1[mask, t_idx] = cashflow_value

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

        # Determine breached region (KI is a DOWN barrier)
        mask = self._get_barrier_mask(s_vec, ki_barrier, product.is_reverse, is_up_barrier=False)

        # V0 transitions to V1 in breached region
        grid_v0[mask, t_idx] = grid_v1[mask, t_idx]

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

    @staticmethod
    def _df_between_times(
        pricing_env: PricingEnvironment, start_time: float, end_time: float
    ) -> float:
        """Calculate discount factor between two times."""
        if end_time <= start_time:
            return 1.0
        df_end = pricing_env.get_discount_factor(end_time)
        df_start = pricing_env.get_discount_factor(start_time)
        return float(safe_divide(df_end, df_start, fallback=1.0))

    @staticmethod
    def _get_asymptotic_discount_factors(
        pricing_env: PricingEnvironment, tau_to_maturity: float
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
        r = pricing_env.get_rate(tau_to_maturity)
        q = pricing_env.get_div_yield(tau_to_maturity)
        df = np.exp(-r * tau_to_maturity)
        df_div = np.exp(-q * tau_to_maturity)
        return float(df), float(df_div)

    def _cashflow_value_at_time(
        self,
        pricing_env: PricingEnvironment,
        cashflow: float,
        current_time: float,
        settlement_time: Optional[float],
    ) -> float:
        """Discount a cashflow from settlement time to current time."""
        if settlement_time is None or settlement_time <= current_time:
            return float(cashflow)
        df = self._df_between_times(pricing_env, current_time, settlement_time)
        return float(cashflow) * df

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

    def _grid_cache_key(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ) -> Tuple:
        base_key = super()._grid_cache_key(
            product, pricing_env, spot, sigma, tau, r, q
        )
        if not hasattr(product, "barrier_config"):
            return base_key

        ko_records = self._get_cached_ko_records(pricing_env, product)
        ko_key = tuple(
            sorted(
                (
                    round(rec.observation_time, 12),
                    round(rec.barrier if rec.barrier is not None else 0.0, 12),
                )
                for rec in ko_records
            )
        )

        ki_key = ()
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if product.has_ki_barrier:
            ki_profile = self._get_cached_ki_profile(pricing_env, product)
            ki_barriers = tuple(
                round(float(b), 12) for b in (ki_profile.get("barriers") or [])
            )
            ki_times = tuple(
                round(float(t), 12)
                for t in (ki_profile.get("observation_times") or [])
                if 0.0 <= float(t) <= tau
            )
            ki_key = (ki_continuous, ki_barriers, ki_times)

        return base_key + (ko_key, ki_key)

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
