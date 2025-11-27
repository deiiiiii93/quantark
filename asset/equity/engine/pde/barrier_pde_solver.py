"""
PDE solver for single barrier options.

Implements the finite difference method for knock-in and knock-out
barrier options with continuous or discrete monitoring.
"""

from typing import Optional, List, Set
import numpy as np

from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option.barrier_option import BarrierOption
from asset.equity.param import PDEParams
from priceenv import PricingEnvironment
from util.enum import ObservationType
from util.exceptions import PricingError

from .base_pde_solver import BasePDESolver


class BarrierPDESolver(BasePDESolver):
    """
    PDE solver for single barrier options.
    
    Handles all four barrier types:
        - DOWN_OUT: Knock-out when price goes below barrier
        - DOWN_IN:  Knock-in when price goes below barrier
        - UP_OUT:   Knock-out when price goes above barrier
        - UP_IN:    Knock-in when price goes above barrier
    
    For knock-out options, the grid boundary at the barrier is set to
    the rebate value (or 0 if no rebate).
    
    For knock-in options, we use the identity:
        Knock-in = Vanilla - Knock-out
    
    This solver also handles discrete observation by only checking
    the barrier at specified observation times.
    """
    
    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize barrier option PDE solver.
        
        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
        self._observation_indices: Set[int] = set()
    
    def price(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a barrier option using PDE method.
        
        For knock-in options, uses: Knock-in = Vanilla - Knock-out
        
        Args:
            product: Barrier option
            pricing_env: Pricing environment
            
        Returns:
            Option price
            
        Raises:
            PricingError: If product is not a barrier option
        """
        if not isinstance(product, BarrierOption):
            raise PricingError(
                f"BarrierPDESolver only supports BarrierOption, "
                f"got {type(product).__name__}"
            )
        
        # Check if barrier is already hit
        spot = pricing_env.spot
        if product.is_barrier_hit(spot):
            if product.is_knock_out:
                # Already knocked out, return rebate
                return product.rebate
            else:
                # Knocked in, price as vanilla
                return self._price_vanilla(product, pricing_env)
        
        if product.is_knock_in:
            # Knock-in = Vanilla - Knock-out (with no rebate)
            vanilla_price = self._price_vanilla(product, pricing_env)
            
            # Create a temporary knock-out version with no rebate
            ko_price = self._price_knock_out(product, pricing_env)
            
            return vanilla_price - ko_price
        else:
            # Direct knock-out pricing
            return super().price(product, pricing_env)
    
    def _price_vanilla(
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price the underlying vanilla option.
        
        Args:
            product: Barrier option (used for strike, type, maturity)
            pricing_env: Pricing environment
            
        Returns:
            Vanilla option price
        """
        from asset.equity.product.option import EuropeanVanillaOption
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
        self,
        product: BarrierOption,
        pricing_env: PricingEnvironment
    ) -> float:
        """
        Price as a knock-out option (for knock-in decomposition).
        
        Creates a temporary knock-out version and prices it.
        
        Args:
            product: Original barrier option
            pricing_env: Pricing environment
            
        Returns:
            Knock-out option price
        """
        # For knock-in decomposition, we need knock-out with zero rebate
        from util.enum import BarrierType
        
        # Convert knock-in type to knock-out
        if product.barrier_type == BarrierType.UP_IN:
            ko_type = BarrierType.UP_OUT
        elif product.barrier_type == BarrierType.DOWN_IN:
            ko_type = BarrierType.DOWN_OUT
        else:
            ko_type = product.barrier_type
        
        ko_product = BarrierOption(
            strike=product.strike,
            option_type=product.option_type,
            barrier=product.barrier,
            barrier_type=ko_type,
            maturity=product.maturity,
            exercise_date=product.exercise_date,
            settlement_date=product.settlement_date,
            rebate=0.0,  # Zero rebate for decomposition
            observation_type=product.observation_type,
            observation_dates=product.observation_dates,
        )
        
        return super().price(ko_product, pricing_env)
    
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
        
        For knock-out options:
            - Zero payoff if barrier already hit
            - Standard payoff otherwise
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: Barrier option
            pricing_env: Pricing environment
        """
        K = product.strike
        barrier = product.barrier
        
        # Calculate base payoff
        if product.is_call():
            payoff = np.maximum(s_vec - K, 0.0)
        else:
            payoff = np.maximum(K - s_vec, 0.0)
        
        # For knock-out, zero payoff where barrier is hit
        if product.is_up_barrier:
            # Up barrier: knockout at high prices
            payoff[s_vec >= barrier] = product.rebate
        else:
            # Down barrier: knockout at low prices
            payoff[s_vec <= barrier] = product.rebate
        
        grid[:, -1] = payoff
    
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
        
        For barrier options, the boundary at the barrier level
        is set to the rebate (discounted if paid at expiry).
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Barrier option
            pricing_env: Pricing environment
        """
        K = product.strike
        barrier = product.barrier
        rebate = product.rebate
        
        r = pricing_env.get_rate(tau) if tau > 0 else 0.0
        q = pricing_env.get_div_yield(tau) if tau > 0 else 0.0
        
        df = np.exp(-r * tau) if tau > 0 else 1.0
        df_div = np.exp(-q * tau) if tau > 0 else 1.0
        
        # Discounted rebate (assuming rebate paid at expiry)
        discounted_rebate = rebate * df
        
        if product.is_up_barrier:
            # Up barrier option
            # Lower boundary: standard European-style
            if product.is_call():
                grid[0, t_idx] = 0.0
            else:
                grid[0, t_idx] = K * df
            
            # Upper boundary: at/above barrier, value is rebate
            grid[-1, t_idx] = discounted_rebate
        else:
            # Down barrier option
            # Lower boundary: at/below barrier, value is rebate
            grid[0, t_idx] = discounted_rebate
            
            # Upper boundary: standard European-style
            if product.is_call():
                grid[-1, t_idx] = max(s_vec[-1] * df_div - K * df, 0.0)
            else:
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
        Apply barrier checks at each time step.
        
        For discrete monitoring, only check at observation times.
        For continuous monitoring, check at every time step.
        
        Args:
            grid: Solution grid
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: Barrier option
            pricing_env: Pricing environment
        """
        # For continuous monitoring, always apply barrier
        # For discrete, only at observation times
        if product.observation_type == ObservationType.DISCRETE:
            if t_idx not in self._observation_indices:
                return
        
        barrier = product.barrier
        rebate = product.rebate
        
        r = pricing_env.get_rate(tau) if tau > 0 else 0.0
        discounted_rebate = rebate * np.exp(-r * tau)
        
        # Apply barrier knockout
        if product.is_up_barrier:
            # Up barrier: knockout at high prices
            grid[s_vec >= barrier, t_idx] = discounted_rebate
        else:
            # Down barrier: knockout at low prices
            grid[s_vec <= barrier, t_idx] = discounted_rebate
    
    def _build_grids(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        spot: float,
        sigma: float,
        tau: float,
        r: float,
        q: float
    ):
        """
        Build grids and setup observation indices for discrete monitoring.
        """
        result = super()._build_grids(product, pricing_env, spot, sigma, tau, r, q)
        x_vec, s_vec, dx_vec, t_vec, dt_vec = result
        
        # Setup observation time indices for discrete monitoring
        self._observation_indices.clear()
        
        if (
            hasattr(product, 'observation_type') and
            product.observation_type == ObservationType.DISCRETE and
            hasattr(product, 'observation_dates') and
            product.observation_dates is not None
        ):
            for obs_time in product.observation_dates:
                if 0 < obs_time < tau:
                    # Find closest time index
                    idx = np.argmin(np.abs(t_vec - obs_time))
                    self._observation_indices.add(idx)
        
        return result
    
    def get_critical_points(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.
        
        For barrier options, both strike and barrier are critical.
        
        Args:
            product: Barrier option
            pricing_env: Pricing environment
            
        Returns:
            List containing strike and barrier
        """
        return [product.strike, product.barrier]
    
    def __repr__(self):
        return "BarrierPDESolver()"

