"""
Greeks calculation for bond options.

Provides both analytical (closed-form Black '76) and numerical (finite difference)
methods for calculating option sensitivities.
"""
import math
from typing import Dict, Optional
from copy import deepcopy
from datetime import timedelta

from scipy import stats

from quantark.asset.bond.product.option.euro_short_term_bond_option import EuroShortTermBondOption
from quantark.asset.bond.engine.analytical.black_engine import BlackBondOptionEngine
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.util.exceptions import ValidationError, NumericalError


class BondGreeksCalculator:
    """
    Calculator for bond option Greeks using both analytical and numerical methods.
    
    Supports:
    - Analytical Greeks: Using closed-form Black '76 formulas
      - Delta, Gamma, Vega, Theta, Rho
    - Numerical Greeks: Using finite difference method (FDM)
    - Bond-specific measures:
      - Option DV01: Sensitivity to parallel rate shift
      - Option Duration: Effective duration through the option
    """
    
    def __init__(self, bump_size: float = 0.01):
        """
        Initialize Greeks calculator.
        
        Args:
            bump_size: Relative bump size for FDM (default: 1%)
        """
        self.bump_size = bump_size
        self.rate_bump = 0.0001  # 1 basis point for rate sensitivities
    
    def calculate_analytical_greeks(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        volatility: Optional[float] = None,
        price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks using analytical Black '76 formulas.
        
        Args:
            option: European bond option
            pricing_env: Pricing environment
            volatility: Volatility for pricing (optional)
            price: Pre-calculated price (optional)
            
        Returns:
            Dictionary of Greeks: price, delta, gamma, vega, theta, rho
            
        Raises:
            ValidationError: If option is invalid
        """
        engine = BlackBondOptionEngine(pricing_env)
        valuation_date = pricing_env.valuation_date
        
        # Get pricing details
        results = engine.price_with_details(option, volatility, valuation_date)
        
        F = results.forward_bond_price
        K = option.strike
        T = results.time_to_expiry
        sigma = results.volatility
        D = results.discount_factor
        d1 = results.d1
        d2 = results.d2
        
        greeks = {}
        
        # Price
        if price is None:
            price = results.price
        greeks["price"] = price
        
        # Handle near-expiry case
        if T < 1e-10:
            return self._greeks_at_expiry(option, F)
        
        sqrt_T = math.sqrt(T)
        n_d1 = stats.norm.pdf(d1)  # Standard normal PDF
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        
        # Delta: ∂V/∂F (sensitivity to forward bond price)
        # For call: D * N(d1)
        # For put: -D * N(-d1) = D * (N(d1) - 1)
        if option.is_call():
            delta = D * N_d1
        else:
            delta = D * (N_d1 - 1)
        greeks["delta"] = delta * option.notional
        
        # Gamma: ∂²V/∂F² (convexity in forward price)
        # Gamma = D * n(d1) / (F * σ * √T)
        gamma = D * n_d1 / (F * sigma * sqrt_T)
        greeks["gamma"] = gamma * option.notional
        
        # Vega: ∂V/∂σ (sensitivity to volatility, per 1% change)
        # Vega = D * F * √T * n(d1)
        vega = D * F * sqrt_T * n_d1 / 100  # Per 1% vol change
        greeks["vega"] = vega * option.notional
        
        # Theta: ∂V/∂t (time decay, per day)
        # Theta = -D * F * σ * n(d1) / (2 * √T) - r * V
        # where V is the undiscounted option value
        r = pricing_env.get_rate(T)
        
        term1 = -D * F * sigma * n_d1 / (2 * sqrt_T)
        term2 = -r * results.price / option.notional  # Discount effect
        
        theta = (term1 + term2) / 365  # Per day
        greeks["theta"] = theta * option.notional
        
        # Rho: ∂V/∂r (sensitivity to interest rate, per 1% change)
        # This captures both the discount effect and forward price effect
        # Rho ≈ -T * V (simplified, actual is more complex for bonds)
        rho = -T * results.price / 100  # Per 1% rate change
        greeks["rho"] = rho
        
        # Forward bond price
        greeks["forward_price"] = F
        
        # Moneyness
        greeks["moneyness"] = F / K
        
        return greeks
    
    def calculate_numerical_greeks(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        volatility: Optional[float] = None,
        base_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method (FDM).
        
        Uses central differences for better accuracy.
        
        Args:
            option: European bond option
            pricing_env: Pricing environment
            volatility: Volatility for pricing (optional)
            base_price: Pre-calculated base price (optional)
            
        Returns:
            Dictionary of Greeks: price, delta, gamma, vega, theta, rho, dv01
        """
        engine = BlackBondOptionEngine(pricing_env)
        valuation_date = pricing_env.valuation_date
        
        # Calculate base price if not provided
        if base_price is None:
            base_price = engine.price(option, volatility, valuation_date)
        
        greeks = {"price": base_price}
        
        # Get forward price for delta calculations
        results = engine.price_with_details(option, volatility, valuation_date)
        F = results.forward_bond_price
        T = results.time_to_expiry
        sigma = volatility if volatility is not None else results.volatility
        
        # Delta and Gamma: bump forward price via rate curve
        # We bump rates to change the forward price
        delta, gamma = self._calculate_delta_gamma_fdm(
            option, pricing_env, engine, base_price, F, volatility
        )
        greeks["delta"] = delta
        greeks["gamma"] = gamma
        
        # Vega: bump volatility (1% absolute)
        vega = self._calculate_vega_fdm(
            option, pricing_env, engine, base_price, sigma, volatility
        )
        greeks["vega"] = vega
        
        # Theta: advance time by 1 day
        theta = self._calculate_theta_fdm(
            option, pricing_env, engine, base_price, volatility
        )
        greeks["theta"] = theta
        
        # Rho: bump risk-free rate by 1%
        rho = self._calculate_rho_fdm(
            option, pricing_env, engine, base_price, volatility
        )
        greeks["rho"] = rho
        
        # DV01: bump rates by 1 basis point
        dv01 = self._calculate_dv01_fdm(
            option, pricing_env, engine, base_price, volatility
        )
        greeks["dv01"] = dv01
        
        greeks["forward_price"] = F
        
        return greeks
    
    def _calculate_delta_gamma_fdm(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        engine: BlackBondOptionEngine,
        base_price: float,
        F: float,
        volatility: Optional[float]
    ) -> tuple:
        """Calculate delta and gamma using central differences on rates."""
        # Bump rates to change forward price
        base_rate = pricing_env.rate_curve.get_rate(1.0)
        bump = self.bump_size * base_rate if base_rate > 0 else 0.001
        
        # Up bump
        env_up = deepcopy(pricing_env)
        env_up.rate_curve = FlatRateCurve(rate=base_rate + bump)
        engine_up = BlackBondOptionEngine(env_up)
        results_up = engine_up.price_with_details(option, volatility, pricing_env.valuation_date)
        price_up = results_up.price
        F_up = results_up.forward_bond_price
        
        # Down bump
        env_down = deepcopy(pricing_env)
        env_down.rate_curve = FlatRateCurve(rate=max(0.001, base_rate - bump))
        engine_down = BlackBondOptionEngine(env_down)
        results_down = engine_down.price_with_details(option, volatility, pricing_env.valuation_date)
        price_down = results_down.price
        F_down = results_down.forward_bond_price
        
        # Delta = dV/dF using chain rule through rate
        dF = F_up - F_down
        if abs(dF) > 1e-10:
            delta = (price_up - price_down) / dF
        else:
            delta = 0.0
        
        # Gamma = d²V/dF²
        if abs(dF) > 1e-10:
            gamma = (price_up - 2 * base_price + price_down) / ((dF / 2) ** 2)
        else:
            gamma = 0.0
        
        return delta, gamma
    
    def _calculate_vega_fdm(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        engine: BlackBondOptionEngine,
        base_price: float,
        sigma: float,
        volatility: Optional[float]
    ) -> float:
        """Calculate vega using forward difference (1% vol bump)."""
        # Bump volatility up by 1%
        vol_up = sigma + 0.01
        price_up = engine.price(option, vol_up, pricing_env.valuation_date)
        
        # Vega per 1% change
        return price_up - base_price
    
    def _calculate_theta_fdm(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        engine: BlackBondOptionEngine,
        base_price: float,
        volatility: Optional[float]
    ) -> float:
        """Calculate theta (1 day time decay)."""
        # Create pricing env for next day
        next_day = pricing_env.valuation_date + timedelta(days=1)
        
        # Check if option would be expired
        if option.is_expired(next_day):
            return -base_price  # Option value goes to zero
        
        env_next = deepcopy(pricing_env)
        env_next.valuation_date = next_day
        
        engine_next = BlackBondOptionEngine(env_next)
        
        try:
            price_next = engine_next.price(option, volatility, next_day)
            return price_next - base_price
        except Exception:
            return 0.0
    
    def _calculate_rho_fdm(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        engine: BlackBondOptionEngine,
        base_price: float,
        volatility: Optional[float]
    ) -> float:
        """Calculate rho (1% rate bump)."""
        base_rate = pricing_env.rate_curve.get_rate(1.0)
        
        # Bump rate up by 1%
        env_up = deepcopy(pricing_env)
        env_up.rate_curve = FlatRateCurve(rate=base_rate + 0.01)
        engine_up = BlackBondOptionEngine(env_up)
        price_up = engine_up.price(option, volatility, pricing_env.valuation_date)
        
        return price_up - base_price
    
    def _calculate_dv01_fdm(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        engine: BlackBondOptionEngine,
        base_price: float,
        volatility: Optional[float]
    ) -> float:
        """Calculate DV01 (1 basis point rate bump)."""
        base_rate = pricing_env.rate_curve.get_rate(1.0)
        
        # Bump rate up by 1bp
        env_up = deepcopy(pricing_env)
        env_up.rate_curve = FlatRateCurve(rate=base_rate + 0.0001)
        engine_up = BlackBondOptionEngine(env_up)
        price_up = engine_up.price(option, volatility, pricing_env.valuation_date)
        
        # DV01 is typically the price decrease for rate increase
        return base_price - price_up
    
    def _greeks_at_expiry(
        self,
        option: EuroShortTermBondOption,
        bond_price: float
    ) -> Dict[str, float]:
        """
        Calculate Greeks at expiry.
        
        At expiry:
        - Price = intrinsic value
        - Delta = 1 (ITM call), -1 (ITM put), 0 (OTM)
        - Gamma, Vega, Theta, Rho = 0
        
        Args:
            option: Bond option
            bond_price: Current bond price
            
        Returns:
            Dictionary of Greeks
        """
        price = option.get_payoff(bond_price)
        
        # Delta at expiry
        if option.is_call():
            delta = option.notional if bond_price > option.strike else 0.0
        else:
            delta = -option.notional if bond_price < option.strike else 0.0
        
        return {
            "price": price,
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
            "forward_price": bond_price,
            "moneyness": bond_price / option.strike,
        }
    
    def calculate_bond_sensitivities(
        self,
        option: EuroShortTermBondOption,
        pricing_env: PricingEnvironment,
        volatility: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate bond-specific sensitivities for the option.
        
        Includes:
        - option_dv01: DV01 of the option position
        - option_duration: Effective duration through the option
        - underlying_dv01: DV01 of the underlying bond
        - underlying_duration: Duration of the underlying bond
        
        Args:
            option: Bond option
            pricing_env: Pricing environment
            volatility: Volatility for pricing
            
        Returns:
            Dictionary of bond sensitivities
        """
        from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
        
        engine = BlackBondOptionEngine(pricing_env)
        bond_engine = BondDiscountEngine(pricing_env)
        valuation_date = pricing_env.valuation_date
        
        # Base calculations
        base_price = engine.price(option, volatility, valuation_date)
        base_bond_price = bond_engine.dirty_price(option.underlying, valuation_date, valuation_date)
        
        sensitivities = {
            "option_price": base_price,
            "underlying_price": base_bond_price,
        }
        
        # Underlying bond DV01 and duration
        underlying_dv01 = bond_engine.dv01(option.underlying, valuation_date, valuation_date)
        underlying_duration = bond_engine.modified_duration(option.underlying, valuation_date, valuation_date)
        
        sensitivities["underlying_dv01"] = underlying_dv01
        sensitivities["underlying_duration"] = underlying_duration
        
        # Option DV01 (via FDM)
        base_rate = pricing_env.rate_curve.get_rate(1.0)
        
        env_up = deepcopy(pricing_env)
        env_up.rate_curve = FlatRateCurve(rate=base_rate + 0.0001)
        engine_up = BlackBondOptionEngine(env_up)
        price_up = engine_up.price(option, volatility, valuation_date)
        
        option_dv01 = base_price - price_up
        sensitivities["option_dv01"] = option_dv01
        
        # Option effective duration
        # Duration = DV01 / (Price * 0.0001)
        if base_price > 1e-10:
            option_duration = option_dv01 / (base_price * 0.0001)
        else:
            option_duration = 0.0
        sensitivities["option_duration"] = option_duration
        
        # Delta-adjusted measures
        greeks = self.calculate_analytical_greeks(option, pricing_env, volatility)
        delta = greeks["delta"] / option.notional  # Per-unit delta
        
        # Delta-equivalent DV01
        sensitivities["delta_equivalent_dv01"] = abs(delta) * underlying_dv01
        
        return sensitivities
    
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
            Dictionary with 'analytical', 'numerical', and 'difference' sub-dicts
        """
        difference = {}
        
        for key in analytical:
            if key in numerical:
                diff = analytical[key] - numerical[key]
                if abs(analytical[key]) > 1e-10:
                    rel_diff = diff / analytical[key]
                else:
                    rel_diff = 0.0
                difference[key] = {"absolute": diff, "relative": rel_diff}
        
        return {
            "analytical": analytical,
            "numerical": numerical,
            "difference": difference,
        }

