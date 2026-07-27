"""
PDE solver for American options.

Implements the finite difference method with early exercise constraint
for pricing American calls and puts.
"""

from typing import Optional, List
import numpy as np

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

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        return self._generic_grid_request(product, market, tau)
    
    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize American option PDE solver.
        
        Args:
            params: PDE engine configuration parameters
        """
        super().__init__(params)
    
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
        
        return super().price(product, pricing_env)
    
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
        
        if product.is_call():
            grid[:, -1] = np.maximum(s_vec - K, 0.0)
        else:  # put
            grid[:, -1] = np.maximum(K - s_vec, 0.0)
    
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
        
        if product.is_call():
            # Lower boundary: call worth 0 when S = 0
            grid[0, t_idx] = 0.0
            # Upper boundary: call worth at least intrinsic value
            grid[-1, t_idx] = max(s_vec[-1] - K, 0.0)
        else:  # put
            # Lower boundary: put worth K when S = 0 (immediate exercise)
            grid[0, t_idx] = K
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
        K = product.strike
        
        # Calculate intrinsic value at all grid points
        if product.is_call():
            intrinsic = np.maximum(s_vec - K, 0.0)
        else:  # put
            intrinsic = np.maximum(K - s_vec, 0.0)
        
        # Apply early exercise constraint
        grid[:, t_idx] = np.maximum(grid[:, t_idx], intrinsic)
    
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

