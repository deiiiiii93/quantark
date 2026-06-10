"""
PDE solver for one-touch (and no-touch) options.

Implements the finite difference method for digital barrier options
that pay a fixed rebate on touching (or not touching) a barrier.
"""

from typing import Dict, Optional, List, Set
import numpy as np

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, ObservationAggregation, TouchType
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver
from .spatial_grid import SpatialGrid
from .time_grid import TimeGrid


class OneTouchPDESolver(BasePDESolver):
    """
    PDE solver for one-touch and no-touch options.

    One-touch options pay a fixed rebate if the barrier is touched.
    No-touch options pay a fixed rebate if the barrier is NOT touched.

    For one-touch (payment at expiry):
        Terminal condition: rebate at barrier side, 0 elsewhere
        Boundary at barrier: rebate (discounted if payment at expiry)

    For no-touch:
        Terminal condition: rebate inside (away from barrier), 0 at barrier
        Boundary at barrier: 0

    For one-touch with payment at hit:
        We solve for the present value of the expected rebate,
        with the barrier being an absorbing boundary with value = rebate.
    """

    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize one-touch option PDE solver.

        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
        self._observation_indices: Set[int] = set()
        self._schedule_records: Dict[int, List] = {}
        self._schedule_aggregation: ObservationAggregation = (
            ObservationAggregation.STOP_FIRST_HIT
        )

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a one-touch or no-touch option using PDE method.

        Args:
            product: One-touch option
            pricing_env: Pricing environment

        Returns:
            Option price

        Raises:
            PricingError: If product is not a one-touch option
        """
        if not isinstance(product, OneTouchOption):
            raise PricingError(
                f"OneTouchPDESolver only supports OneTouchOption, "
                f"got {type(product).__name__}"
            )

        # Check if barrier already hit
        spot = pricing_env.spot
        if product.is_barrier_hit(spot):
            if product.is_one_touch:
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
        Calculate Greeks for a one-touch or no-touch option.

        Args:
            product: One-touch option
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma

        Raises:
            PricingError: If product is not a one-touch option
        """
        if not isinstance(product, OneTouchOption):
            raise PricingError(
                f"OneTouchPDESolver only supports OneTouchOption, "
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

        # Check if barrier already hit
        if product.is_barrier_hit(spot):
            if product.is_one_touch:
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

        For one-touch (payment at expiry):
            - If barrier not touched by expiry, one-touch pays 0
            - This is handled by solving backwards

        For no-touch:
            - Pays rebate at expiry if barrier was never touched
            - So terminal is rebate away from barrier, 0 at barrier

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: One-touch option
            pricing_env: Pricing environment
        """
        barrier = product.barrier
        rebate = product.rebate

        if product.is_one_touch:
            # One-touch: at maturity, if we haven't touched, we get 0
            # But at barrier, we get rebate (this is set in boundary conditions)
            grid[:, -1] = 0.0

            # Set barrier boundary to rebate value
            if product.is_up_barrier:
                grid[s_vec >= barrier, -1] = rebate
            else:
                grid[s_vec <= barrier, -1] = rebate
        else:
            # No-touch: pays rebate at expiry if never touched
            # Terminal is rebate everywhere except at barrier
            grid[:, -1] = rebate

            # Zero at barrier (already touched)
            if product.is_up_barrier:
                grid[s_vec >= barrier, -1] = 0.0
            else:
                grid[s_vec <= barrier, -1] = 0.0

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

        For one-touch:
            - Barrier boundary: rebate (discounted if payment at expiry)
            - Far boundary: 0 (never reach barrier from there)

        For no-touch:
            - Barrier boundary: 0 (touched = failed)
            - Far boundary: discounted rebate

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: One-touch option
            pricing_env: Pricing environment
        """
        rebate = product.rebate
        r = pricing_env.get_rate(tau) if tau > 0 else 0.0

        # Discount factor
        df = np.exp(-r * tau) if tau > 0 else 1.0

        if product.is_one_touch:
            if product.payment_at_hit:
                # Pay immediately on hit - boundary is full rebate
                barrier_value = rebate
            else:
                # Pay at expiry - discount the rebate
                barrier_value = rebate * df

            far_value = 0.0  # Too far from barrier to ever reach it
        else:
            # No-touch
            barrier_value = 0.0  # Touched = no payout
            far_value = rebate * df  # Never touch = get discounted rebate

        if product.is_up_barrier:
            # Up barrier: upper boundary is barrier
            grid[0, t_idx] = far_value
            grid[-1, t_idx] = barrier_value
        else:
            # Down barrier: lower boundary is barrier
            grid[0, t_idx] = barrier_value
            grid[-1, t_idx] = far_value

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

        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: One-touch option
            pricing_env: Pricing environment
        """
        # For discrete monitoring, only check at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return

        barrier = product.barrier
        rebate = product.rebate
        r = pricing_env.get_rate(tau) if tau > 0 else 0.0
        df = np.exp(-r * tau) if tau > 0 else 1.0

        schedule_records = self._schedule_records.get(t_idx)
        if schedule_records:
            for rec in schedule_records:
                barrier = rec.barrier if rec.barrier is not None else product.barrier
                payoff = rec.payoff
                if product.is_one_touch:
                    barrier_value = payoff if product.payment_at_hit else payoff * df
                else:
                    barrier_value = 0.0
                if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                    if product.is_up_barrier:
                        grid[s_vec >= barrier, t_idx] += barrier_value
                    else:
                        grid[s_vec <= barrier, t_idx] += barrier_value
                else:
                    if product.is_up_barrier:
                        grid[s_vec >= barrier, t_idx] = barrier_value
                    else:
                        grid[s_vec <= barrier, t_idx] = barrier_value
                    return
            return

        if product.is_one_touch:
            if product.payment_at_hit:
                barrier_value = rebate
            else:
                barrier_value = rebate * df
        else:
            barrier_value = 0.0

        # Apply barrier value where barrier is hit
        if product.is_up_barrier:
            grid[s_vec >= barrier, t_idx] = barrier_value
        else:
            grid[s_vec <= barrier, t_idx] = barrier_value

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
        """Build grids for one-touch option."""
        params: PDEParams = self.params
        barrier = product.barrier

        # Use volatility-based bounds with wider range for barrier options
        # This ensures proper probability capture for touch/no-touch dynamics
        s_min, s_max = SpatialGrid.calculate_auto_bounds(
            spot,
            sigma,
            tau,
            r,
            q,
            barriers=[barrier],
            num_std=5.0,  # Wider range for barrier options
        )

        # Ensure barrier is at the appropriate boundary edge
        if product.is_up_barrier:
            # Up barrier: upper boundary should be at/beyond barrier
            s_max = max(s_max, barrier * 1.01)
        else:
            # Down barrier: lower boundary should be at/beyond barrier
            s_min = min(s_min, barrier * 0.99)

        # Override with user bounds if specified
        if params.s_min > 0:
            s_min = params.s_min
        if params.s_max > 0:
            s_max = params.s_max

        # Build spatial grid with concentration at barrier
        critical_points = [barrier]

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
        self._observation_indices.clear()
        self._schedule_records.clear()
        self._schedule_aggregation = ObservationAggregation.STOP_FIRST_HIT
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            resolved_records = schedule.resolve(
                pricing_env=pricing_env,
                default_barrier=product.barrier,
                default_payoff=product.rebate,
                require_single=True,
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

        For one-touch, only the barrier is critical.

        Args:
            product: One-touch option
            pricing_env: Pricing environment

        Returns:
            List containing the barrier
        """
        points = [product.barrier]
        schedule = getattr(product, "observation_schedule", None)
        if schedule is not None:
            for rec in schedule.records:
                if rec.barrier is not None:
                    points.append(rec.barrier)
        # sort and make unique before return
        points = sorted(set(points))
        return points

    def __repr__(self):
        return "OneTouchPDESolver()"
