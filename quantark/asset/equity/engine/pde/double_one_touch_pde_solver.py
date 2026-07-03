"""
PDE solver for double one-touch (and double no-touch) options.

Implements the finite difference method for digital barrier options
with two barriers (upper and lower).
"""

from typing import Dict, Optional, List, Set
import numpy as np

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ObservationAggregation, TouchType
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver
from .spatial_grid import SpatialGrid
from .time_grid import TimeGrid


class DoubleOneTouchPDESolver(BasePDESolver):
    """
    PDE solver for double one-touch and double no-touch options.

    Double one-touch options pay a fixed rebate if EITHER barrier is touched.
    Double no-touch options pay a fixed rebate if NEITHER barrier is touched.

    For double one-touch:
        - Boundary conditions: rebate at both barriers
        - Terminal condition: 0 (didn't touch either barrier yet)

    For double no-touch:
        - Boundary conditions: 0 at both barriers (touched = failed)
        - Terminal condition: rebate inside corridor
    """

    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize double one-touch option PDE solver.

        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
        self._observation_indices: Set[int] = set()
        self._schedule_records: Dict[int, List] = {}
        self._schedule_aggregation: ObservationAggregation = (
            ObservationAggregation.STOP_FIRST_HIT
        )
        self._total_tau: float = 0.0

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a double one-touch or double no-touch option.

        Args:
            product: Double one-touch option
            pricing_env: Pricing environment

        Returns:
            Option price

        Raises:
            PricingError: If product is not a double one-touch option
        """
        if not isinstance(product, DoubleOneTouchOption):
            raise PricingError(
                f"DoubleOneTouchPDESolver only supports DoubleOneTouchOption, "
                f"got {type(product).__name__}"
            )

        # Check if barrier already hit (outside corridor)
        spot = pricing_env.spot
        if product.is_barrier_hit(spot):
            if product.is_double_one_touch:
                # Already touched, immediate rebate
                return product.rebate
            else:
                # No-touch already failed
                return 0.0

        return super().price(product, pricing_env)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a double one-touch or double no-touch option.

        Args:
            product: Double one-touch option
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma

        Raises:
            PricingError: If product is not a double one-touch option
        """
        if not isinstance(product, DoubleOneTouchOption):
            raise PricingError(
                f"DoubleOneTouchPDESolver only supports DoubleOneTouchOption, "
                f"got {type(product).__name__}"
            )

        spot = pricing_env.spot
        tau = product.get_maturity(pricing_env)

        # Handle expired case
        if tau <= 0:
            return {
                "price": self._calculate_intrinsic(product, spot),
                "delta": self._intrinsic_delta(product, spot),
                "gamma": 0.0,
            }

        # Check if barrier already hit (outside corridor)
        if product.is_barrier_hit(spot):
            if product.is_double_one_touch:
                # Already touched, fixed rebate (delta=gamma=0)
                return {"price": product.rebate, "delta": 0.0, "gamma": 0.0}
            else:
                # No-touch already failed
                return {"price": 0.0, "delta": 0.0, "gamma": 0.0}

        return super().calculate_greeks(product, pricing_env)

    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set the terminal condition at maturity.

        For double one-touch:
            - At maturity, if neither barrier touched, value = 0
            - At barriers and beyond, value = rebate

        For double no-touch:
            - At maturity, if neither barrier touched, value = rebate
            - At barriers and beyond, value = 0

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: Double one-touch option
            pricing_env: Pricing environment
        """
        upper = product.upper_barrier
        lower = product.lower_barrier
        rebate = product.rebate

        # Use tolerance for floating-point comparisons
        tol = 1e-10
        at_or_above_upper = s_vec >= upper - tol * upper
        at_or_below_lower = s_vec <= lower + tol * lower

        if product.is_double_one_touch:
            # One-touch: at maturity, inside corridor = 0, at/beyond barriers = rebate
            grid[:, -1] = 0.0
            grid[at_or_above_upper, -1] = rebate
            grid[at_or_below_lower, -1] = rebate
        else:
            # No-touch: at maturity, inside corridor = rebate, at/beyond barriers = 0
            grid[:, -1] = rebate
            grid[at_or_above_upper, -1] = 0.0
            grid[at_or_below_lower, -1] = 0.0

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
        """
        Set boundary conditions at spatial edges.

        For double one-touch:
            - Both barriers: rebate (possibly discounted)

        For double no-touch:
            - Both barriers: 0

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Double one-touch option
            pricing_env: Pricing environment
        """
        rebate = product.rebate
        # Forward discount factor from the current step to maturity,
        # term-structure consistent (DF(t,T), not DF(0,tau)).
        current_time = self._current_time(self._total_tau, tau)
        df = self._df_between_times(pricing_env, current_time, self._total_tau)

        if product.is_double_one_touch:
            if product.payment_at_hit:
                barrier_value = rebate
            else:
                barrier_value = rebate * df

            # Both boundaries are barriers
            grid[0, t_idx] = barrier_value  # Lower barrier
            grid[-1, t_idx] = barrier_value  # Upper barrier
        else:
            # No-touch: barriers are absorbing at zero
            grid[0, t_idx] = 0.0
            grid[-1, t_idx] = 0.0

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
        Apply barrier checks at each time step.

        For discrete monitoring, only check at observation times.
        For continuous monitoring, check at every step.

        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Double one-touch option
            pricing_env: Pricing environment
        """
        # For discrete monitoring, only check at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return

        schedule_records = self._schedule_records.get(t_idx)
        upper = product.upper_barrier
        lower = product.lower_barrier
        rebate = product.rebate
        current_time = self._current_time(self._total_tau, tau)
        df = self._df_between_times(pricing_env, current_time, self._total_tau)

        if schedule_records:
            for rec in schedule_records:
                upper = (
                    rec.upper_barrier
                    if rec.upper_barrier is not None
                    else product.upper_barrier
                )
                lower = (
                    rec.lower_barrier
                    if rec.lower_barrier is not None
                    else product.lower_barrier
                )
                payoff = rec.payoff
                if product.is_double_one_touch:
                    barrier_value = payoff if product.payment_at_hit else payoff * df
                else:
                    barrier_value = 0.0
                at_or_above_upper = s_vec >= upper
                at_or_below_lower = s_vec <= lower
                outside_corridor = at_or_above_upper | at_or_below_lower
                if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                    grid[outside_corridor, t_idx] += barrier_value
                else:
                    grid[outside_corridor, t_idx] = barrier_value
                    return
            return

        if product.is_double_one_touch:
            if product.payment_at_hit:
                barrier_value = rebate
            else:
                barrier_value = rebate * df
        else:
            barrier_value = 0.0

        # Apply barrier values at/beyond barriers
        at_or_above_upper = s_vec >= upper
        at_or_below_lower = s_vec <= lower
        outside_corridor = at_or_above_upper | at_or_below_lower
        grid[outside_corridor, t_idx] = barrier_value

    def _get_barriers(self, product: BaseEquityProduct) -> List[float]:
        """Include schedule-specific barriers when building spatial bounds."""
        barriers = super()._get_barriers(product)
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.lower_barrier is not None:
                    barriers.append(rec.lower_barrier)
                if rec.upper_barrier is not None:
                    barriers.append(rec.upper_barrier)
        return barriers

    def _build_grids(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float,
    ):
        """Build grids for double barrier options.

        Uses a small buffer beyond the barriers to ensure proper PDE
        diffusion from barrier values into the interior. The barriers
        are included as interior points so their values propagate.
        """
        params: PDEParams = self.params

        lower = product.lower_barrier
        upper = product.upper_barrier

        s_min, s_max = SpatialGrid.calculate_auto_bounds(
            spot,
            sigma,
            tau,
            r,
            q,
            barriers=[lower, upper],
            num_std=5.0,  # Wider range for barrier options
        )

        # Critical points: both barriers and spot for grid concentration
        critical_points = [lower, upper]
        if s_min < spot < s_max:
            critical_points.append(spot)

        # Build spatial grid with concentration at barriers
        x_vec, s_vec, dx_vec = SpatialGrid.build(
            s_min,
            s_max,
            params.grid_size,
            critical_points=critical_points,
            use_adaptive=params.adaptive_grid,
        )

        # Get event times
        event_times = self._get_event_times(product, tau)

        # Build time grid
        t_vec, dt_vec = TimeGrid.build(
            tau,
            params.time_steps,
            method=params.time_grid_type,
            event_times=event_times,
            grade_exponent=params.grade_exponent,
        )

        # Setup observation indices for discrete monitoring
        self._total_tau = tau
        self._observation_indices.clear()
        self._schedule_records.clear()
        self._schedule_aggregation = ObservationAggregation.STOP_FIRST_HIT
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            resolved_records = schedule.resolve(
                pricing_env=pricing_env,
                default_upper=product.upper_barrier,
                default_lower=product.lower_barrier,
                default_payoff=product.rebate,
                require_double=True,
            )
            self._schedule_aggregation = schedule.aggregation_mode
            if self._schedule_aggregation in (
                ObservationAggregation.BEST,
                ObservationAggregation.WORST,
            ):
                raise PricingError(
                    f"PDE solver does not support aggregation mode {self._schedule_aggregation.value}"
                )
            for rec in resolved_records:
                if 0 < rec.observation_time < tau:
                    idx = np.argmin(np.abs(t_vec - rec.observation_time))
                    self._observation_indices.add(idx)
                    self._schedule_records.setdefault(idx, []).append(rec)
        elif (
            product.observation_type == ObservationType.DISCRETE
            and product.observation_dates is not None
        ):
            for obs_time in product.observation_dates:
                if 0 < obs_time < tau:
                    idx = np.argmin(np.abs(t_vec - obs_time))
                    self._observation_indices.add(idx)

        return x_vec, s_vec, dx_vec, t_vec, dt_vec

    def get_critical_points(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.

        For double one-touch, both barriers are critical.

        Args:
            product: Double one-touch option
            pricing_env: Pricing environment

        Returns:
            List containing both barriers
        """
        points = [product.lower_barrier, product.upper_barrier]
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.lower_barrier is not None:
                    points.append(rec.lower_barrier)
                if rec.upper_barrier is not None:
                    points.append(rec.upper_barrier)
        # sort and make unique before return
        points = sorted(set(points))
        return points

    def __repr__(self):
        return "DoubleOneTouchPDESolver()"
