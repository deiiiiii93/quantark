"""
PDE solver for American options.

Implements the finite difference method with early exercise constraint
for pricing American calls and puts.
"""

from dataclasses import replace
from datetime import datetime
from typing import Optional, List
import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    AmericanExerciseTimings,
    american_exercise_requires_dates,
    build_american_exercise_date_grid,
    resolve_american_exercise_timings,
    resolve_terminal_timing,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option.american_option import AmericanOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError

from .base_pde_solver import BasePDESolver


class AmericanPDESolver(BasePDESolver):
    """
    PDE solver for American options with early exercise.
    
    Uses finite difference method to solve the Black-Scholes PDE,
    with the early exercise constraint applied at each time step:
        V(S, t) >= max(intrinsic_value, continuation_value)
    
    Terminal condition:
        Call: max(S - K, 0)
        Put:  max(K - S, 0)
    
    Boundary conditions (same as European):
        Call: V(0) = 0, V(Smax) ≈ Smax - K*exp(-r*tau)
        Put:  V(0) ≈ K, V(Smax) = 0
    
    Early exercise constraint:
        At each time step, option value is set to max(intrinsic, continuation)
    """

    settlement_support = SettlementSupport.AMERICAN_EXERCISE

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        request = self._generic_grid_request(product, market, tau)
        date_times = self._exercise_date_times
        if date_times is None:
            return request
        interior = tuple(
            float(time) for time in date_times if 0.0 < time < tau
        )
        return replace(
            request,
            event_times=(*request.event_times, *interior),
        )
    
    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize American option PDE solver.
        
        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
        self._exercise_timings: Optional[AmericanExerciseTimings] = None
        self._exercise_boundary_delay_dfs: Optional[np.ndarray] = None
        self._exercise_dates: Optional[tuple[datetime, ...]] = None
        self._exercise_date_times: Optional[np.ndarray] = None

    def _prepare_solve_state(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
    ) -> None:
        """Resolve authoritative exercise dates before binding the PDE grid."""
        self._exercise_timings = None
        self._exercise_boundary_delay_dfs = None
        self._exercise_dates = None
        self._exercise_date_times = None
        if american_exercise_requires_dates(product):
            dates, times = build_american_exercise_date_grid(
                product, pricing_env
            )
            self._exercise_dates = dates
            self._exercise_date_times = times
    
    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price an American option using PDE method with early exercise.
        
        Args:
            product: American option
            pricing_env: Pricing environment
            
        Returns:
            Option price
            
        Raises:
            PricingError: If product is not an American option
        """
        if not isinstance(product, AmericanOption):
            raise PricingError(
                f"AmericanPDESolver only supports AmericanOption, "
                f"got {type(product).__name__}"
            )

        self._prepare_solve_state(product, pricing_env)
        tau = product.get_maturity(pricing_env)
        if tau <= 0.0:
            timings = resolve_american_exercise_timings(
                product,
                pricing_env,
                np.array([0.0]),
                exercise_dates=self._exercise_dates,
            )
            self._exercise_timings = timings
            return product.get_payoff(pricing_env.spot) * timings.delay_dfs[0]

        value = super().price(product, pricing_env)
        timings = self._exercise_timings
        if timings is None:
            raise PricingError(
                "American exercise timings were not retained by the PDE solve"
            )
        valuation_obstacle = (
            product.intrinsic_value(pricing_env.spot)
            * timings.delay_dfs[0]
        )
        return max(value, valuation_obstacle)
    
    def set_terminal_condition(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> None:
        """
        Set the terminal condition (payoff at maturity).
        
        For American options (same as European):
            Call: max(S - K, 0)
            Put:  max(K - S, 0)
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: American option
            pricing_env: Pricing environment
        """
        K = product.strike
        multiplier = product.contract_multiplier
        delay_df = resolve_terminal_timing(product, pricing_env).delay_df
        
        if product.is_call():
            grid[:, -1] = (
                np.maximum(s_vec - K, 0.0) * multiplier * delay_df
            )
        else:  # put
            grid[:, -1] = (
                np.maximum(K - s_vec, 0.0) * multiplier * delay_df
            )
    
    def set_boundary_conditions(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> None:
        """
        Set boundary conditions at spatial edges.
        
        For American options:
            Lower boundary (S → 0):
                Call: V → 0
                Put:  V → K (immediate exercise value)
            
            Upper boundary (S → ∞):
                Call: V → S - K (immediate exercise value)
                Put:  V → 0
        
        Note: American boundary conditions use intrinsic value
        rather than discounted value since immediate exercise is possible.
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: American option
            pricing_env: Pricing environment
        """
        K = product.strike
        multiplier = product.contract_multiplier
        delay_df = 1.0
        if self._exercise_boundary_delay_dfs is not None:
            delay_df = float(self._exercise_boundary_delay_dfs[t_idx])
        
        if product.is_call():
            # Lower boundary: call worth 0 when S = 0
            grid[0, t_idx] = 0.0
            # Upper boundary: call worth at least intrinsic value
            grid[-1, t_idx] = (
                max(s_vec[-1] - K, 0.0) * multiplier * delay_df
            )
        else:  # put
            # Lower boundary: put worth K when S = 0 (immediate exercise)
            grid[0, t_idx] = K * multiplier * delay_df
            # Upper boundary: put worth 0 when S is very large
            grid[-1, t_idx] = 0.0
    
    def _apply_step_modifications(
        self,
        grid: np.ndarray,
        x_vec: np.ndarray,
        s_vec: np.ndarray,
        t_idx: int,
        tau: float,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> None:
        """
        Apply early exercise constraint after each time step.
        
        At each grid point, the option value must be at least
        the intrinsic value (what you get by exercising now).
        
        V(S, t) = max(continuation_value, intrinsic_value)
        
        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: American option
            pricing_env: Pricing environment
        """
        timings = self._exercise_timings
        if timings is None:
            raise PricingError(
                "American exercise timings must be resolved before stepping"
            )
        if not bool(timings.eligible[t_idx]):
            return

        K = product.strike
        multiplier = product.contract_multiplier
        
        # Calculate intrinsic value at all grid points
        if product.is_call():
            intrinsic = np.maximum(s_vec - K, 0.0)
        else:  # put
            intrinsic = np.maximum(K - s_vec, 0.0)
        
        # Apply early exercise constraint
        exercise_value = (
            intrinsic * multiplier * timings.delay_dfs[t_idx]
        )
        grid[:, t_idx] = np.maximum(grid[:, t_idx], exercise_value)

    def _time_stepping(
        self,
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
        step_coeffs=None,
    ) -> None:
        """Resolve exercise payment timing once, then run the PDE march."""
        node_dates = None
        if american_exercise_requires_dates(product):
            dates = self._exercise_dates
            date_times = self._exercise_date_times
            if dates is None or date_times is None:
                raise PricingError(
                    "American exercise dates must be resolved before "
                    "building the PDE grid"
                )
            mapped: list[Optional[datetime]] = [None] * len(t_vec)
            for date, date_time in zip(dates, date_times):
                index = int(np.searchsorted(t_vec, date_time))
                if (
                    index >= len(t_vec)
                    or float(t_vec[index]) != float(date_time)
                ):
                    raise PricingError(
                        "American exercise date is not an exact PDE time node"
                    )
                mapped[index] = date
            node_dates = tuple(mapped)

        timings = resolve_american_exercise_timings(
            product,
            pricing_env,
            t_vec,
            exercise_dates=node_dates,
        )
        self._exercise_timings = timings

        eligible_indices = np.flatnonzero(timings.eligible)
        boundary_delay_dfs = np.empty(len(t_vec), dtype=float)
        for index, node_time in enumerate(t_vec):
            positions = eligible_indices[eligible_indices >= index]
            next_index = int(
                positions[0] if positions.size else eligible_indices[-1]
            )
            boundary_delay_dfs[index] = (
                timings.payment_dfs[next_index]
                / pricing_env.get_discount_factor(float(node_time))
            )
        self._exercise_boundary_delay_dfs = boundary_delay_dfs

        super()._time_stepping(
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
    
    def get_critical_points(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.
        
        For American options, the strike is the critical point.
        The early exercise boundary is also important but varies with time.
        
        Args:
            product: American option
            pricing_env: Pricing environment
            
        Returns:
            List containing the strike price
        """
        return [product.strike]
    
    def __repr__(self):
        return "AmericanPDESolver()"
