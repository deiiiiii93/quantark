"""
PDE solver for double barrier options.

Implements the finite difference method for double knock-in and
knock-out barrier options (corridor options).
"""

from typing import Dict, Optional, List, Set
import numpy as np

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.double_barrier_option import DoubleBarrierOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DoubleBarrierType, ObservationType, ObservationAggregation
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver


class DoubleBarrierPDESolver(BasePDESolver):
    """
    PDE solver for double barrier (corridor) options.

    Double barrier options have both upper and lower barriers.
    The option knocks out (or in) if either barrier is hit.

    For knock-out corridor options:
        - Terminal condition: payoff inside corridor, zero outside
        - Boundary conditions: rebate at both barriers

    For knock-in corridor options, we use:
        Knock-in = Vanilla - Knock-out
    """

    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize double barrier option PDE solver.

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
        Price a double barrier option using PDE method.

        For knock-in options, uses: Knock-in = Vanilla - Knock-out

        Args:
            product: Double barrier option
            pricing_env: Pricing environment

        Returns:
            Option price

        Raises:
            PricingError: If product is not a double barrier option
        """
        if not isinstance(product, DoubleBarrierOption):
            raise PricingError(
                f"DoubleBarrierPDESolver only supports DoubleBarrierOption, "
                f"got {type(product).__name__}"
            )

        # Check if barrier is already hit (outside corridor)
        spot = pricing_env.spot
        if product.is_barrier_hit(spot):
            if product.is_knock_out:
                # Already knocked out
                return product.rebate
            else:
                # Knocked in, price as vanilla
                return self._price_vanilla(product, pricing_env)

        if product.is_knock_in:
            # Knock-in = Vanilla - Knock-out
            vanilla_price = self._price_vanilla(product, pricing_env)
            ko_price = self._price_knock_out(product, pricing_env)
            return vanilla_price - ko_price
        else:
            # Direct knock-out pricing
            return super().price(product, pricing_env)

    def _price_vanilla(
        self, product: DoubleBarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price the underlying vanilla option.
        """
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from .european_pde_solver import EuropeanPDESolver

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )

        solver = EuropeanPDESolver(self.params)
        return solver.price(vanilla, pricing_env)

    def _price_knock_out(
        self, product: DoubleBarrierOption, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price as knock-out (for knock-in decomposition).
        """
        ko_product = DoubleBarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            upper_barrier=product.upper_barrier,
            lower_barrier=product.lower_barrier,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
        )

        return super().price(ko_product, pricing_env)

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for a double barrier option using PDE method.

        For knock-in options, uses: Greeks_KI = Greeks_Vanilla - Greeks_KO

        Args:
            product: Double barrier option
            pricing_env: Pricing environment

        Returns:
            Dictionary with price, delta, gamma

        Raises:
            PricingError: If product is not a double barrier option
        """
        if not isinstance(product, DoubleBarrierOption):
            raise PricingError(
                f"DoubleBarrierPDESolver only supports DoubleBarrierOption, "
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

        # Check if barrier is already hit (outside corridor)
        if product.is_barrier_hit(spot):
            if product.is_knock_out:
                # Already knocked out: rebate is constant, so delta=gamma=0
                return {"price": product.rebate, "delta": 0.0, "gamma": 0.0}
            else:
                # Knocked in: return vanilla Greeks
                return self._calculate_greeks_vanilla(product, pricing_env)

        if product.is_knock_in:
            # Knock-in = Vanilla - Knock-out decomposition
            vanilla_greeks = self._calculate_greeks_vanilla(product, pricing_env)
            ko_greeks = self._calculate_greeks_knock_out(product, pricing_env)

            return {
                "price": vanilla_greeks["price"] - ko_greeks["price"],
                "delta": vanilla_greeks["delta"] - ko_greeks["delta"],
                "gamma": vanilla_greeks["gamma"] - ko_greeks["gamma"],
            }
        else:
            # Direct knock-out pricing
            return super().calculate_greeks(product, pricing_env)

    def _calculate_greeks_vanilla(
        self, product: DoubleBarrierOption, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks for the underlying vanilla option.
        """
        from quantark.asset.equity.product.option import EuropeanVanillaOption
        from .european_pde_solver import EuropeanPDESolver

        vanilla = EuropeanVanillaOption(
            strike=product.strike,
            option_type=product.option_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
        )

        solver = EuropeanPDESolver(self.params)
        return solver.calculate_greeks(vanilla, pricing_env)

    def _calculate_greeks_knock_out(
        self, product: DoubleBarrierOption, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks as knock-out (for knock-in decomposition).
        """
        ko_product = DoubleBarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            upper_barrier=product.upper_barrier,
            lower_barrier=product.lower_barrier,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
            observation_schedule=product.observation_schedule,
        )

        return super().calculate_greeks(ko_product, pricing_env)

    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """
        Set the terminal condition (payoff at maturity).

        For double knock-out:
            - Payoff inside corridor (between barriers)
            - Rebate outside corridor

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: Double barrier option
            pricing_env: Pricing environment
        """
        K = product.strike
        upper = product.upper_barrier
        lower = product.lower_barrier
        rebate = product.rebate

        # Calculate base payoff
        if product.is_call():
            payoff = np.maximum(s_vec - K, 0.0)
        else:
            payoff = np.maximum(K - s_vec, 0.0)

        # Zero (or rebate) payoff outside corridor
        outside_corridor = (s_vec >= upper) | (s_vec <= lower)
        payoff[outside_corridor] = rebate

        grid[:, -1] = payoff

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

        For double barrier:
            - Lower boundary: rebate (at/below lower barrier)
            - Upper boundary: rebate (at/above upper barrier)

        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Double barrier option
            pricing_env: Pricing environment
        """
        rebate = product.rebate
        # Rebate paid at maturity, discounted with the forward factor
        # DF(t, T) (term-structure consistent, not DF(0, tau)).
        current_time = self._current_time(self._total_tau, tau)
        discounted_rebate = rebate * self._df_between_times(
            pricing_env, current_time, self._total_tau
        )

        # Both boundaries are at the barriers, so both get rebate
        grid[0, t_idx] = discounted_rebate  # Lower barrier
        grid[-1, t_idx] = discounted_rebate  # Upper barrier

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
            product: Double barrier option
            pricing_env: Pricing environment
        """
        # For discrete monitoring, only check at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return

        schedule_records = self._schedule_records.get(t_idx)
        total_tau = (
            self._total_tau
            if self._total_tau > 0
            else product.get_maturity(pricing_env)
        )
        current_time = self._current_time(total_tau, tau)

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
                discounted_payoff = self._cashflow_value_at_time(
                    pricing_env=pricing_env,
                    cashflow=rec.payoff,
                    current_time=current_time,
                    settlement_time=(
                        rec.settlement_time
                        if rec.settlement_time is not None
                        else total_tau
                    ),
                )
                outside_corridor = (s_vec >= upper) | (s_vec <= lower)
                if self._schedule_aggregation == ObservationAggregation.ACCUMULATE:
                    grid[outside_corridor, t_idx] += discounted_payoff
                else:
                    grid[outside_corridor, t_idx] = discounted_payoff
                    return
            return

        upper = product.upper_barrier
        lower = product.lower_barrier
        rebate = product.rebate

        discounted_rebate = rebate * self._df_between_times(
            pricing_env, current_time, total_tau
        )

        # Apply knockout at both barriers
        outside_corridor = (s_vec >= upper) | (s_vec <= lower)
        grid[outside_corridor, t_idx] = discounted_rebate

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
        """
        Build grids with barriers as spatial boundaries.

        For double barrier options, the spatial grid is constrained
        to the corridor between the barriers.
        """
        params: PDEParams = self.params

        # Use barriers as spatial boundaries (with small buffer)
        lower = product.lower_barrier
        upper = product.upper_barrier

        # Small buffer to ensure barrier points are included
        buffer = 0.001
        s_min = lower * (1 - buffer)
        s_max = upper * (1 + buffer)

        # Get critical points
        critical_points = self.get_critical_points(product, pricing_env)

        # Build spatial grid
        from .spatial_grid import SpatialGrid

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
        from .time_grid import TimeGrid

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
            hasattr(product, "observation_type")
            and product.observation_type == ObservationType.DISCRETE
            and hasattr(product, "observation_dates")
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

        For double barrier: strike and both barriers are critical.

        Args:
            product: Double barrier option
            pricing_env: Pricing environment

        Returns:
            List containing strike and both barriers
        """
        points = [product.strike, product.lower_barrier, product.upper_barrier]
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
        return "DoubleBarrierPDESolver()"
