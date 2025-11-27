"""
PDE solver for European vanilla options.

Implements the simplest case of PDE pricing: European calls and puts
with standard Black-Scholes boundary conditions.
"""

from typing import Optional, List
import numpy as np

from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.param import PDEParams
from priceenv import PricingEnvironment
from util.exceptions import PricingError

from .base_pde_solver import BasePDESolver


class EuropeanPDESolver(BasePDESolver):
    """
    PDE solver for European vanilla options.
    
    Uses finite difference method to solve the Black-Scholes PDE
    for European call and put options.
    
    Terminal condition:
        Call: max(S - K, 0)
        Put:  max(K - S, 0)
    
    Boundary conditions:
        Call: V(0) = 0, V(Smax) ≈ Smax - K*exp(-r*tau)
        Put:  V(0) ≈ K*exp(-r*tau), V(Smax) = 0
    """
    
    def __init__(self, params: Optional[PDEParams] = None):
        """
        Initialize European option PDE solver.
        
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
        Price a European vanilla option using PDE method.
        
        Args:
            product: European vanilla option
            pricing_env: Pricing environment
            
        Returns:
            Option price
            
        Raises:
            PricingError: If product is not a European vanilla option
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"EuropeanPDESolver only supports EuropeanVanillaOption, "
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
        
        For European options:
            Call: max(S - K, 0)
            Put:  max(K - S, 0)
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            product: European vanilla option
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
        
        For European options:
            Lower boundary (S → 0):
                Call: V → 0
                Put:  V → K * exp(-r*tau)
            
            Upper boundary (S → ∞):
                Call: V → S - K * exp(-r*tau)
                Put:  V → 0
        
        Args:
            grid: Solution grid [num_x, num_t]
            x_vec: Log-price grid points
            s_vec: Price grid points
            t_idx: Current time index
            tau: Time remaining to maturity
            product: European vanilla option
            pricing_env: Pricing environment
        """
        K = product.strike
        r = pricing_env.get_rate(tau) if tau > 0 else 0.0
        q = pricing_env.get_div_yield(tau) if tau > 0 else 0.0
        
        df = np.exp(-r * tau) if tau > 0 else 1.0
        df_div = np.exp(-q * tau) if tau > 0 else 1.0
        
        if product.is_call():
            # Lower boundary: call worth 0 when S = 0
            grid[0, t_idx] = 0.0
            # Upper boundary: call worth approximately S*exp(-q*tau) - K*exp(-r*tau)
            grid[-1, t_idx] = max(s_vec[-1] * df_div - K * df, 0.0)
        else:  # put
            # Lower boundary: put worth K*exp(-r*tau) when S = 0
            grid[0, t_idx] = K * df
            # Upper boundary: put worth 0 when S is very large
            grid[-1, t_idx] = 0.0
    
    def get_critical_points(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment
    ) -> List[float]:
        """
        Get critical prices for grid concentration.
        
        For European options, the strike is the critical point.
        
        Args:
            product: European vanilla option
            pricing_env: Pricing environment
            
        Returns:
            List containing the strike price
        """
        return [product.strike]
    
    def __repr__(self):
        return "EuropeanPDESolver()"

