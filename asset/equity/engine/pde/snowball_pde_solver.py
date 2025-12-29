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

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from asset.equity.engine.pde.base_pde_solver import BasePDESolver
from asset.equity.param import PDEParams
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.observation_schedule import ResolvedObservationRecord
from asset.equity.product.option.snowball_option import SnowballOption
from priceenv import PricingEnvironment
from util.enum import ObservationType
from util.exceptions import PricingError, ValidationError
from util.numerical import is_close, is_zero, safe_divide


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

    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize Snowball PDE solver.

        Args:
            params: PDE engine configuration parameters
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
        self._ki_continuous: bool = False
        self._ki_barrier: float = 0.0
        self._is_reverse: bool = False

        # Time tracking
        self._total_tau: float = 0.0

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
        if not isinstance(product, SnowballOption):
            raise PricingError(
                f"SnowballPDESolver only supports SnowballOption, "
                f"got {type(product).__name__}"
            )

        if pricing_env is None:
            raise ValidationError(
                "PricingEnvironment is required for SnowballPDESolver"
            )

        # Validate PDE compatibility
        self._validate_product(product)

        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        if tau <= 0 or is_zero(tau):
            # Expired: return terminal payoff
            return self._calculate_terminal_value(product, spot, pricing_env)

        # Determine the state at valuation date (t=0). For discrete monitoring,
        # barrier states can only change on observation times.
        ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        knocked_in_at_valuation = self._is_knocked_in_at_valuation(
            product, spot, pricing_env, ki_continuous=ki_continuous
        )
        knocked_out_at_valuation = self._is_knocked_out_at_valuation(
            product, spot, pricing_env
        )

        if knocked_out_at_valuation:
            return self._get_immediate_ko_payoff(product, pricing_env)

        # Extract market data
        strike = product.strike
        r = pricing_env.get_rate(tau)
        q = pricing_env.get_div_yield(tau)
        sigma = pricing_env.get_vol(strike, tau)

        # Store product properties for later use
        self._is_reverse = product.is_reverse
        self._ki_continuous = (
            product.barrier_config.ki_continuous
            or product.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if product.has_ki_barrier:
            ki_barrier = product.barrier_config.ki_barrier
            if isinstance(ki_barrier, list):
                self._ki_barrier = ki_barrier[0]  # Use first for continuous check
            else:
                self._ki_barrier = ki_barrier

        # Build grids
        x_vec, s_vec, dx_vec, t_vec, dt_vec = self._build_grids(
            product, pricing_env, spot, sigma, tau, r, q
        )

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

        # Build operator matrices
        l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        A = self._build_operator_matrix(l, c, u, num_x)

        # Time stepping for both surfaces
        self._time_stepping_two_surface(
            self._grid_v0,
            self._grid_v1,
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
        )

        # Interpolate final price from appropriate surface
        spot_log = np.log(spot)
        if knocked_in_at_valuation:
            return self._interpolate_price(self._grid_v1[:, 0], x_vec, spot_log)
        else:
            return self._interpolate_price(self._grid_v0[:, 0], x_vec, spot_log)

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
    def _record_is_non_negative_time(record: ResolvedObservationRecord) -> bool:
        """Return True if record's time is >= 0 (within numerical tolerance)."""
        return bool(
            record.observation_time > 0.0 or is_close(record.observation_time, 0.0)
        )

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
        self._ko_terminal_record = None
        self._has_terminal_ko = False

        # Setup KO observation indices
        ko_records = product.resolve_ko_observations(pricing_env)
        for rec in ko_records:
            obs_time = rec.observation_time
            if is_close(obs_time, 0.0):
                self._ko_observation_indices[0] = rec
            elif is_close(obs_time, tau):
                self._ko_terminal_record = rec
                self._has_terminal_ko = True
            elif 0.0 < obs_time < tau:
                idx = int(np.argmin(np.abs(t_vec - obs_time)))
                self._ko_observation_indices[idx] = rec

        # Setup KI observation indices (if discrete)
        if product.has_ki_barrier and not self._ki_continuous:
            ki_profile = product.get_ki_observation_profile(pricing_env)
            ki_times = ki_profile["observation_times"]
            for obs_time in ki_times:
                if is_close(obs_time, 0.0):
                    self._ki_observation_indices.add(0)
                elif 0.0 < obs_time <= tau:
                    idx = int(np.argmin(np.abs(t_vec - obs_time)))
                    self._ki_observation_indices.add(idx)

        return result

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

    def _get_event_times(
        self, product: BaseEquityProduct, tau: float
    ) -> Optional[List[float]]:
        """Collect all observation times for time grid alignment."""
        event_times = []

        if hasattr(product, "barrier_config"):
            # KO observation times
            ko_schedule = product.barrier_config.ko_observation_schedule
            if ko_schedule is not None:
                for rec in ko_schedule.records:
                    if rec.observation_time is not None:
                        t = rec.observation_time
                        if 0 < t < tau:
                            event_times.append(t)
            elif product.barrier_config.ko_observation_dates is not None:
                for t in product.barrier_config.ko_observation_dates:
                    if 0 < t < tau:
                        event_times.append(t)

            # KI observation times (if discrete)
            if not self._ki_continuous:
                ki_schedule = product.barrier_config.ki_observation_schedule
                if ki_schedule is not None:
                    for rec in ki_schedule.records:
                        if rec.observation_time is not None:
                            t = rec.observation_time
                            if 0 < t < tau:
                                event_times.append(t)
                elif product.barrier_config.ki_observation_dates is not None:
                    for t in product.barrier_config.ki_observation_dates:
                        if 0 < t < tau:
                            event_times.append(t)

        return sorted(set(event_times)) if event_times else None

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

        # Apply to breached region
        if product.is_reverse:
            # DOWN KO for reverse
            mask = s_vec <= barrier
        else:
            # UP KO for standard
            mask = s_vec >= barrier

        grid_v0[mask, -1] = cashflow_value
        grid_v1[mask, -1] = cashflow_value

    def _time_stepping_two_surface(
        self,
        grid_v0: np.ndarray,
        grid_v1: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
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
        I_int = sp.eye(num_x - 2, format="csc")
        self._matrix_cache.clear()

        # Rannacher smoothing indices
        smooth_js = set()
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

        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            steps_from_end = num_t - 1 - j
            current_time = t_vec[j]
            tau_remaining = tau - current_time

            # Determine theta (Rannacher smoothing uses backward Euler)
            theta = (
                1.0
                if params.use_rannacher
                and (steps_from_end < params.rannacher_steps or j in smooth_js)
                else params.theta
            )

            M1, M2_lu = self._get_matrices(I_int, A, dt, theta)

            # Set boundary conditions for both surfaces
            self._set_boundary_conditions_v0(
                grid_v0, x_vec, s_vec, j, tau_remaining, product, pricing_env
            )
            self._set_boundary_conditions_v1(
                grid_v1, x_vec, s_vec, j, tau_remaining, product, pricing_env
            )

            # Step V0 backward
            rhs_v0 = M1 @ grid_v0[1:-1, j + 1]
            self._inject_boundary_contributions(rhs_v0, grid_v0, l, u, j, dt, theta)
            grid_v0[1:-1, j] = M2_lu.solve(rhs_v0)

            # Step V1 backward
            rhs_v1 = M1 @ grid_v1[1:-1, j + 1]
            self._inject_boundary_contributions(rhs_v1, grid_v1, l, u, j, dt, theta)
            grid_v1[1:-1, j] = M2_lu.solve(rhs_v1)

            # Apply barrier modifications
            self._apply_step_modifications_two_surface(
                grid_v0, grid_v1, x_vec, s_vec, j, tau_remaining, product, pricing_env
            )

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

        principal = product.notional if product.payoff_config.include_principal else 0.0
        rebate = product.payoff_config.rebate_rate * product.notional

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

        principal = product.notional if product.payoff_config.include_principal else 0.0
        strike = product.strike
        initial_price = product.initial_price
        participation = product.payoff_config.participation_rate

        # Lower boundary (S -> 0)
        # Deep ITM put for standard snowball: maximum loss
        if product.is_reverse:
            # Reverse: embedded call, S=0 means no loss
            grid[0, t_idx] = principal * df_to_maturity
        else:
            # Standard: embedded put, S=0 means maximum loss
            # Loss = participation × (-K/S0) × N
            max_loss = participation * (-strike / initial_price) * product.notional
            # Apply protection floor if applicable
            if product.payoff_config.protection_type.name == "FULL":
                max_loss = 0.0
            elif product.payoff_config.protection_type.name == "PARTIAL":
                floor = -product.payoff_config.protection_rate * product.notional
                max_loss = max(max_loss, floor)
            grid[0, t_idx] = (principal + max_loss) * df_to_maturity

        # Upper boundary (S -> ∞)
        # For standard: no loss (put is worthless)
        # For reverse: maximum loss (call is deep ITM)
        if product.is_reverse:
            # For very high S, call loss could be large
            # But typically bounded by participation
            grid[-1, t_idx] = principal * df_to_maturity  # Simplified
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

        # Determine breached region
        if product.is_reverse:
            # DOWN KO for reverse
            mask = s_vec <= barrier
        else:
            # UP KO for standard
            mask = s_vec >= barrier

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
        ki_barrier = self._ki_barrier

        # Determine breached region
        if product.is_reverse:
            # UP KI for reverse
            mask = s_vec >= ki_barrier
        else:
            # DOWN KI for standard
            mask = s_vec <= ki_barrier

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
        ko_records = product.resolve_ko_observations(pricing_env)
        future_records = [
            rec for rec in ko_records if self._record_is_non_negative_time(rec)
        ]
        if not future_records:
            return 0.0

        next_rec: Optional[ResolvedObservationRecord] = None
        for rec in future_records:
            if rec.observation_time >= current_time or is_close(
                rec.observation_time, current_time
            ):
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
