"""
PDE solver for European vanilla options.

Implements the simplest case of PDE pricing: European calls and puts
with standard Black-Scholes boundary conditions.
"""

from typing import Dict, List, Optional
import numpy as np

from quantark.asset.equity.engine.capabilities import SettlementSupport
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment

from .base_pde_solver import BasePDESolver
from .core.adapters.european import EuropeanAdapter
from .core.prepare import Operation, OutputRequest


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
    _adapter_class = EuropeanAdapter

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
        return self._run(product, pricing_env, OutputRequest(Operation.PRICE),
                         lifecycle_state=lifecycle_state)

    def calculate_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        *,
        lifecycle_state=None,
    ) -> Dict[str, float]:
        """Return grid Greeks plus any spot-invariant pending receivables."""
        return self._run(product, pricing_env, OutputRequest(Operation.GREEKS, greeks=True),
                         lifecycle_state=lifecycle_state)
    
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
        grid[:, -1], _ = self._terminal_payoff(product, pricing_env, s_vec)

    def _terminal_payoff(self, product, pricing_env, s_vec):
        """The payoff the march starts from (spec 3C decision 30): per contract, with the terminal payment
        discount. ``(value, slope)`` on ``s_vec``."""
        K = product.strike
        delay_df = self._terminal_timing(product, pricing_env).delay_df
        m = self._multiplier(product)
        if product.is_call():
            return np.maximum(s_vec - K, 0.0) * delay_df * m, np.where(s_vec > K, 1.0, 0.0) * delay_df * m
        return np.maximum(K - s_vec, 0.0) * delay_df * m, np.where(s_vec < K, -1.0, 0.0) * delay_df * m

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
            Lower boundary (the grid's lower edge S_min):
                Call: V → 0
                Put:  V → K DF(t,T) - S_min CF(t,T), floored at 0
            
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
        delay_df = self._terminal_timing(product, pricing_env).delay_df
        m = self._multiplier(product)
        
        if product.is_call():
            # Lower boundary: call worth 0 when S = 0
            grid[0, t_idx] = 0.0
            # Upper boundary: call worth approximately S*exp(-q*tau) - K*exp(-r*tau)
            grid[-1, t_idx] = (
                max(s_vec[-1] * df_div - K * df, 0.0) * delay_df * m
            )
        else:  # put
            # Lower boundary: the put at the grid's finite lower edge, forward over [t, T]
            # (spec 3A §5.10): K DF(t,T) - S_min CF(t,T), not its S -> 0 limit K DF(t,T)
            grid[0, t_idx] = max(K * df - s_vec[0] * df_div, 0.0) * delay_df * m
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


def european_expiry_outcome(product, pricing_env, lifecycle_state=None, greeks: bool = False):
    """This route on a European at its exercise instant: its expiry outcome (the payoff at the spot x the terminal
    delay discount, plus any pending receivable; delta the intrinsic slope, gamma 0), decided before any grid or vol
    is read. The vol-model European PDE routes settle on it before their model runs (W9, ruling 2026-10-10)."""
    solver = EuropeanPDESolver(PDEParams())
    if greeks:
        return solver.calculate_greeks(product, pricing_env, lifecycle_state=lifecycle_state)
    return solver.price(product, pricing_env, lifecycle_state=lifecycle_state)
