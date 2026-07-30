"""
PDE solver for European vanilla options.

Implements the simplest case of PDE pricing: European calls and puts
with standard Black-Scholes boundary conditions.
"""

from typing import Dict, List, Optional
import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.engine.settlement_support import (
    pending_receivable_pv,
    resolve_terminal_timing,
    terminal_lifecycle_pv,
    validate_settlement_capability,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import PricingError

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

    settlement_support = SettlementSupport.TERMINAL_ONLY
    supports_lifecycle_state = True

    def _uses_grid_layer(self) -> bool:
        return True

    def grid_request(self, product, market, tau):
        return self._generic_grid_request(product, market, tau)
    
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
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
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

        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return fixed_pv

        timing = resolve_terminal_timing(product, pricing_env)
        if product.get_maturity(pricing_env) <= 0.0:
            contingent = (
                self._calculate_intrinsic(product, pricing_env.spot)
                * timing.delay_df
            )
        else:
            contingent = super().price(product, pricing_env)
        return contingent + pending_receivable_pv(lifecycle_state, pricing_env)

    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> Dict[str, float]:
        """Return grid Greeks plus any spot-invariant pending receivables."""
        validate_settlement_capability(self, product, lifecycle_state)
        fixed_pv = terminal_lifecycle_pv(lifecycle_state, pricing_env)
        if fixed_pv is not None:
            return {"price": fixed_pv, "delta": 0.0, "gamma": 0.0}

        timing = resolve_terminal_timing(product, pricing_env)
        if product.get_maturity(pricing_env) <= 0.0:
            greeks = {
                "price": (
                    self._calculate_intrinsic(product, pricing_env.spot)
                    * timing.delay_df
                ),
                "delta": (
                    self._intrinsic_delta(product, pricing_env.spot)
                    * timing.delay_df
                ),
                "gamma": 0.0,
            }
        else:
            greeks = super().calculate_greeks(product, pricing_env)
        greeks["price"] += pending_receivable_pv(lifecycle_state, pricing_env)
        return greeks
    
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
        delay_df = resolve_terminal_timing(product, pricing_env).delay_df
        
        if product.is_call():
            grid[:, -1] = np.maximum(s_vec - K, 0.0) * delay_df
        else:  # put
            grid[:, -1] = np.maximum(K - s_vec, 0.0) * delay_df
    
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
        total_tau = product.get_maturity(pricing_env)
        current_time = max(total_tau - tau, 0.0)

        df = self._df_between_times(pricing_env, current_time, total_tau)
        df_div = self._carry_df_between_times(pricing_env, current_time, total_tau)
        delay_df = resolve_terminal_timing(product, pricing_env).delay_df
        
        if product.is_call():
            # Lower boundary: call worth 0 when S = 0
            grid[0, t_idx] = 0.0
            # Upper boundary: call worth approximately S*exp(-q*tau) - K*exp(-r*tau)
            grid[-1, t_idx] = (
                max(s_vec[-1] * df_div - K * df, 0.0) * delay_df
            )
        else:  # put
            # Lower boundary: put worth K*exp(-r*tau) when S = 0
            grid[0, t_idx] = K * df * delay_df
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
