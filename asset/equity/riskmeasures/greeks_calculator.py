"""
Greeks calculation for equity derivatives.
"""
import math
from typing import Dict, Optional
from scipy import stats
from copy import deepcopy
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.engine.base_engine import BaseEngine
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, NumericalError


class GreeksCalculator:
    """
    Calculator for option Greeks using both analytical and numerical methods.
    
    Supports:
    - Analytical Greeks: Using closed-form Black-Scholes formulas
    - Numerical Greeks: Using finite difference method (FDM)
    """
    
    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize Greeks calculator.
        
        Args:
            params: Engine parameters (for bump size in FDM)
        """
        self.params = params if params is not None else EngineParams()
    
    def calculate_analytical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks using analytical Black-Scholes formulas.
        
        Only works for European vanilla options under Black-Scholes model.
        
        Args:
            product: European vanilla option
            pricing_env: Pricing environment
            price: Pre-calculated price (optional, will calculate if not provided)
            
        Returns:
            Dictionary of Greeks: delta, gamma, vega, theta, rho
            
        Raises:
            ValidationError: If product is not a European vanilla option
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise ValidationError(
                f"Analytical Greeks only support EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )
        
        # Extract parameters
        S = pricing_env.spot
        K = product.strike
        T = product.maturity
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)
        
        # Handle edge case: option at expiry
        if T < 1e-10:
            return self._greeks_at_expiry(product, S)
        
        # Calculate d1 and d2
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Calculate discount factors
        discount_div = math.exp(-q * T)
        discount_rf = math.exp(-r * T)
        
        # Standard normal PDF and CDF
        n_d1 = stats.norm.pdf(d1)  # phi(d1)
        N_d1 = stats.norm.cdf(d1)  # Phi(d1)
        N_d2 = stats.norm.cdf(d2)  # Phi(d2)
        
        greeks = {}
        
        # Calculate price if not provided
        if price is None:
            if product.is_call():
                price = S * discount_div * N_d1 - K * discount_rf * N_d2
            else:
                price = K * discount_rf * stats.norm.cdf(-d2) - S * discount_div * stats.norm.cdf(-d1)
        greeks['price'] = price
        
        # Delta: ∂V/∂S
        if product.is_call():
            delta = discount_div * N_d1
        else:
            delta = -discount_div * stats.norm.cdf(-d1)
        greeks['delta'] = delta
        
        # Gamma: ∂²V/∂S²
        gamma = discount_div * n_d1 / (S * sigma * sqrt_T)
        greeks['gamma'] = gamma
        
        # Vega: ∂V/∂σ (divided by 100 for 1% change)
        vega = S * discount_div * n_d1 * sqrt_T / 100
        greeks['vega'] = vega
        
        # Theta: ∂V/∂t (per day, divided by 365)
        term1 = -S * discount_div * n_d1 * sigma / (2 * sqrt_T)
        if product.is_call():
            term2 = -r * K * discount_rf * N_d2
            term3 = q * S * discount_div * N_d1
            theta = (term1 + term2 + term3) / 365
        else:
            term2 = r * K * discount_rf * stats.norm.cdf(-d2)
            term3 = -q * S * discount_div * stats.norm.cdf(-d1)
            theta = (term1 + term2 + term3) / 365
        greeks['theta'] = theta
        
        # Rho: ∂V/∂r (divided by 100 for 1% change)
        if product.is_call():
            rho = K * T * discount_rf * N_d2 / 100
        else:
            rho = -K * T * discount_rf * stats.norm.cdf(-d2) / 100
        greeks['rho'] = rho
        
        return greeks
    
    def calculate_numerical_greeks(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        engine: BaseEngine,
        base_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method (FDM).
        
        Uses central differences for better accuracy.
        Works for any product and engine combination.
        
        Args:
            product: The derivative product
            pricing_env: Pricing environment
            engine: Pricing engine to use
            base_price: Pre-calculated base price (optional)
            
        Returns:
            Dictionary of Greeks: delta, gamma, vega, theta, rho
        """
        # Calculate base price if not provided
        if base_price is None:
            base_price = engine.price(product, pricing_env)
        
        greeks = {'price': base_price}
        
        bump = self.params.bump_size
        
        # Delta and Gamma: bump spot
        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= (1 + bump)
        price_up_spot = engine.price(product, env_up)
        
        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= (1 - bump)
        price_down_spot = engine.price(product, env_down)
        
        delta = (price_up_spot - price_down_spot) / (2 * pricing_env.spot * bump)
        gamma = (price_up_spot - 2 * base_price + price_down_spot) / \
                (pricing_env.spot * bump) ** 2
        
        greeks['delta'] = delta
        greeks['gamma'] = gamma
        
        # Vega: bump volatility (1% absolute)
        env_up_vol = deepcopy(pricing_env)
        current_vol = pricing_env.get_vol(product.strike, product.maturity)
        # For flat vol surface, we need to create a new surface
        from param.vol import FlatVolSurface
        env_up_vol.vol_surface = FlatVolSurface(current_vol + 0.01)
        price_up_vol = engine.price(product, env_up_vol)
        
        vega = price_up_vol - base_price  # Already per 1% change
        greeks['vega'] = vega
        
        # Theta: bump time (1 day = 1/365 year)
        product_theta = deepcopy(product)
        time_bump = 1 / 365
        if product_theta.maturity > time_bump:
            product_theta.maturity -= time_bump
            price_theta = engine.price(product_theta, pricing_env)
            theta = price_theta - base_price  # Price change as time passes (negative for long options)
        else:
            # Near expiry, theta is approximately -intrinsic_value_change
            theta = 0.0  # Simplified
        greeks['theta'] = theta
        
        # Rho: bump risk-free rate (1% = 0.01)
        env_up_rate = deepcopy(pricing_env)
        from param.rrf import FlatRateCurve
        current_rate = pricing_env.get_rate(product.maturity)
        env_up_rate.rate_curve = FlatRateCurve(current_rate + 0.01)
        price_up_rate = engine.price(product, env_up_rate)
        
        rho = price_up_rate - base_price  # Already per 1% change
        greeks['rho'] = rho
        
        return greeks
    
    def _greeks_at_expiry(self, product: EuropeanVanillaOption, spot: float) -> Dict[str, float]:
        """
        Calculate Greeks at expiry.
        
        At expiry:
        - Price = intrinsic value
        - Delta = 1 (ITM call), -1 (ITM put), 0 (OTM)
        - Gamma, Vega, Theta, Rho = 0
        
        Args:
            product: European vanilla option
            spot: Current spot price
            
        Returns:
            Dictionary of Greeks
        """
        price = product.get_payoff(spot)
        
        # Delta at expiry
        if product.is_call():
            delta = 1.0 if spot > product.strike else 0.0
        else:
            delta = -1.0 if spot < product.strike else 0.0
        
        return {
            'price': price,
            'delta': delta,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            'rho': 0.0
        }
    
    def compare_greeks(
        self,
        analytical: Dict[str, float],
        numerical: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare analytical and numerical Greeks.
        
        Args:
            analytical: Analytical Greeks
            numerical: Numerical Greeks
            
        Returns:
            Dictionary with 'analytical', 'numerical', and 'difference' sub-dictionaries
        """
        difference = {}
        for key in analytical:
            if key in numerical:
                diff = analytical[key] - numerical[key]
                rel_diff = diff / analytical[key] if abs(analytical[key]) > 1e-10 else 0
                difference[key] = {
                    'absolute': diff,
                    'relative': rel_diff
                }
        
        return {
            'analytical': analytical,
            'numerical': numerical,
            'difference': difference
        }

