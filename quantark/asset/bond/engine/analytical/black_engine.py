"""
Black '76 pricing engine for European bond options.

The Black model is the standard approach for pricing European options on
bonds and other fixed income instruments.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from scipy import stats

from asset.bond.product.option.euro_short_term_bond_option import EuroShortTermBondOption
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, NumericalError, PricingError


@dataclass
class BlackBondOptionResults:
    """
    Results from Black model bond option pricing.
    
    Attributes:
        price: Option price
        forward_bond_price: Forward price of the underlying bond
        d1: Black model d1 parameter
        d2: Black model d2 parameter
        discount_factor: Discount factor to option expiry
        time_to_expiry: Time to option expiry in years
        volatility: Volatility used in pricing
    """
    price: float
    forward_bond_price: float
    d1: float
    d2: float
    discount_factor: float
    time_to_expiry: float
    volatility: float


class BlackBondOptionEngine:
    """
    Black '76 pricing engine for European bond options.
    
    Uses the Black '76 model to price European options on bonds:
    
        Call = D(T) * [F * N(d1) - K * N(d2)]
        Put  = D(T) * [K * N(-d2) - F * N(-d1)]
    
    where:
        D(T) = discount factor to option expiry
        F = forward bond price
        K = strike price
        d1 = [ln(F/K) + σ²T/2] / (σ√T)
        d2 = d1 - σ√T
        σ = volatility of bond price
    
    The forward bond price is calculated by:
        F = (Dirty Price - PV of coupons before expiry) * exp(r*T)
    
    Attributes:
        pricing_env: Pricing environment with market data
        bond_engine: Engine for pricing the underlying bond
    """
    
    def __init__(self, pricing_env: PricingEnvironment):
        """
        Initialize Black bond option engine.
        
        Args:
            pricing_env: Pricing environment with rate curve and volatility
            
        Raises:
            ValidationError: If pricing environment is invalid
        """
        if pricing_env is None:
            raise ValidationError("Pricing environment is required")
        
        self.pricing_env = pricing_env
        self.bond_engine = BondDiscountEngine(pricing_env)
    
    def price(
        self,
        option: EuroShortTermBondOption,
        volatility: Optional[float] = None,
        valuation_date: Optional[datetime] = None
    ) -> float:
        """
        Price a European bond option using Black '76 model.
        
        Args:
            option: Bond option to price
            volatility: Price volatility (if None, uses vol surface from pricing_env)
            valuation_date: Valuation date (default: pricing env date)
            
        Returns:
            Option price
            
        Raises:
            PricingError: If option has expired or pricing fails
            NumericalError: If numerical issues occur
        """
        results = self.price_with_details(option, volatility, valuation_date)
        return results.price
    
    def price_with_details(
        self,
        option: EuroShortTermBondOption,
        volatility: Optional[float] = None,
        valuation_date: Optional[datetime] = None
    ) -> BlackBondOptionResults:
        """
        Price a European bond option with detailed results.
        
        Args:
            option: Bond option to price
            volatility: Price volatility (if None, uses vol surface from pricing_env)
            valuation_date: Valuation date (default: pricing env date)
            
        Returns:
            BlackBondOptionResults with pricing details
            
        Raises:
            PricingError: If option has expired or pricing fails
            NumericalError: If numerical issues occur
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        # Check if option has expired
        if option.is_expired(valuation_date):
            raise PricingError("Option has expired")
        
        # Calculate time to expiry
        T = option.get_time_to_expiry(valuation_date)
        
        # Handle near-expiry case
        if T < 1e-10:
            bond_price = self._get_bond_price(option, valuation_date)
            payoff = option.get_payoff(bond_price)
            return BlackBondOptionResults(
                price=payoff,
                forward_bond_price=bond_price,
                d1=0.0,
                d2=0.0,
                discount_factor=1.0,
                time_to_expiry=T,
                volatility=0.0
            )
        
        # Get forward bond price
        F = self._calculate_forward_bond_price(option, valuation_date)
        
        # Get strike
        K = option.strike
        
        # Get discount factor to option expiry
        discount_factor = self.pricing_env.get_discount_factor(T)
        
        # Get volatility
        if volatility is None:
            volatility = self._get_volatility(option, valuation_date, F, T)
        
        # Validate inputs
        self._validate_inputs(F, K, T, volatility)
        
        # Calculate d1 and d2
        d1, d2 = self._calculate_d1_d2(F, K, T, volatility)
        
        # Calculate option price
        if option.is_call():
            price = self._price_call(F, K, discount_factor, d1, d2)
        else:
            price = self._price_put(F, K, discount_factor, d1, d2)
        
        # Apply notional
        price *= option.notional
        
        # Sanity check
        if price < 0:
            raise NumericalError(f"Negative price computed: {price}")
        
        return BlackBondOptionResults(
            price=price,
            forward_bond_price=F,
            d1=d1,
            d2=d2,
            discount_factor=discount_factor,
            time_to_expiry=T,
            volatility=volatility
        )
    
    def _get_bond_price(
        self,
        option: EuroShortTermBondOption,
        valuation_date: datetime
    ) -> float:
        """
        Get current bond price (clean or dirty based on option specs).
        
        Args:
            option: Bond option
            valuation_date: Valuation date
            
        Returns:
            Bond price
        """
        if option.strike_is_clean:
            return self.bond_engine.clean_price(
                option.underlying,
                valuation_date,
                valuation_date
            )
        else:
            return self.bond_engine.dirty_price(
                option.underlying,
                valuation_date,
                valuation_date
            )
    
    def _calculate_forward_bond_price(
        self,
        option: EuroShortTermBondOption,
        valuation_date: datetime
    ) -> float:
        """
        Calculate the forward price of the bond to option expiry.
        
        Forward Price = (Spot Dirty Price - PV of coupons before expiry) * exp(r*T)
        
        For clean price options, we adjust by subtracting accrued interest at expiry.
        
        Args:
            option: Bond option
            valuation_date: Valuation date
            
        Returns:
            Forward bond price
        """
        underlying = option.underlying
        expiry_date = option.expiry_date
        
        # Get time to expiry
        T = option.get_time_to_expiry(valuation_date)
        
        # Get current dirty price
        spot_dirty = self.bond_engine.dirty_price(underlying, valuation_date, valuation_date)
        
        # Get cashflows between valuation and expiry
        all_cashflows = underlying.get_cashflows(valuation_date)
        
        # Calculate PV of coupons paid before expiry
        coupon_pv = 0.0
        for cf in all_cashflows:
            if cf.payment_date <= expiry_date:
                # Time from valuation to coupon payment
                time_to_coupon = (cf.payment_date - valuation_date).days / 365.0
                if time_to_coupon >= 0:
                    df = self.pricing_env.get_discount_factor(time_to_coupon)
                    coupon_pv += cf.amount * df
        
        # Forward dirty price
        r = self.pricing_env.get_rate(T)
        forward_dirty = (spot_dirty - coupon_pv) * math.exp(r * T)
        
        # Adjust for clean vs dirty
        if option.strike_is_clean:
            # Calculate accrued interest at expiry
            accrued_at_expiry = underlying.calculate_accrued_interest(expiry_date)
            forward_clean = forward_dirty - accrued_at_expiry
            return forward_clean
        else:
            return forward_dirty
    
    def _get_volatility(
        self,
        option: EuroShortTermBondOption,
        valuation_date: datetime,
        forward_price: float,
        time_to_expiry: float
    ) -> float:
        """
        Get volatility for pricing.
        
        First tries to use the vol surface from pricing environment.
        If not available, uses a default volatility.
        
        Args:
            option: Bond option
            valuation_date: Valuation date
            forward_price: Forward bond price
            time_to_expiry: Time to option expiry
            
        Returns:
            Volatility for pricing
        """
        try:
            # Try to use vol surface if available
            if self.pricing_env.vol_surface is not None:
                return self.pricing_env.get_vol(option.strike, time_to_expiry)
        except Exception:
            pass
        
        # Default volatility for bonds (typically lower than equities)
        # Around 5-15% for government bonds
        return 0.10  # 10% default
    
    def _validate_inputs(
        self,
        F: float,
        K: float,
        T: float,
        sigma: float
    ) -> None:
        """
        Validate inputs for numerical stability.
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            sigma: Volatility
            
        Raises:
            ValidationError: If inputs are invalid
        """
        if F <= 0:
            raise ValidationError(f"Forward price must be positive, got {F}")
        if K <= 0:
            raise ValidationError(f"Strike must be positive, got {K}")
        if T < 0:
            raise ValidationError(f"Time to expiry must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if sigma > 5.0:
            raise ValidationError(f"Volatility too high: {sigma}")
    
    def _calculate_d1_d2(
        self,
        F: float,
        K: float,
        T: float,
        sigma: float
    ) -> tuple:
        """
        Calculate Black model d1 and d2 parameters.
        
        d1 = [ln(F/K) + σ²T/2] / (σ√T)
        d2 = d1 - σ√T
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            sigma: Volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        try:
            sqrt_T = math.sqrt(T)
            log_moneyness = math.log(F / K)
            
            # Check for extreme values
            if abs(log_moneyness) > 100:
                raise NumericalError(f"Extreme moneyness: ln(F/K) = {log_moneyness}")
            
            numerator = log_moneyness + 0.5 * sigma * sigma * T
            denominator = sigma * sqrt_T
            
            if denominator <= 1e-10:
                raise NumericalError(f"Denominator too small: σ√T = {denominator}")
            
            d1 = numerator / denominator
            d2 = d1 - sigma * sqrt_T
            
            return d1, d2
            
        except (OverflowError, ValueError) as e:
            raise NumericalError(f"Numerical error in d1/d2 calculation: {e}")
    
    def _price_call(
        self,
        F: float,
        K: float,
        discount_factor: float,
        d1: float,
        d2: float
    ) -> float:
        """
        Calculate call option price using Black formula.
        
        Call = D(T) * [F * N(d1) - K * N(d2)]
        
        Args:
            F: Forward price
            K: Strike price
            discount_factor: Discount factor to expiry
            d1: d1 parameter
            d2: d2 parameter
            
        Returns:
            Call option price
        """
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        
        return discount_factor * (F * N_d1 - K * N_d2)
    
    def _price_put(
        self,
        F: float,
        K: float,
        discount_factor: float,
        d1: float,
        d2: float
    ) -> float:
        """
        Calculate put option price using Black formula.
        
        Put = D(T) * [K * N(-d2) - F * N(-d1)]
        
        Args:
            F: Forward price
            K: Strike price
            discount_factor: Discount factor to expiry
            d1: d1 parameter
            d2: d2 parameter
            
        Returns:
            Put option price
        """
        N_minus_d1 = stats.norm.cdf(-d1)
        N_minus_d2 = stats.norm.cdf(-d2)
        
        return discount_factor * (K * N_minus_d2 - F * N_minus_d1)
    
    def implied_volatility(
        self,
        option: EuroShortTermBondOption,
        market_price: float,
        valuation_date: Optional[datetime] = None,
        initial_guess: float = 0.10,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Calculate implied volatility from market price using Newton-Raphson.
        
        Args:
            option: Bond option
            market_price: Observed market price
            valuation_date: Valuation date
            initial_guess: Initial volatility guess (default: 10%)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
            
        Raises:
            NumericalError: If convergence fails
        """
        if valuation_date is None:
            valuation_date = self.pricing_env.valuation_date
        
        if market_price <= 0:
            raise ValidationError(f"Market price must be positive, got {market_price}")
        
        # Adjust for notional
        target_price = market_price / option.notional
        
        sigma = initial_guess
        
        for iteration in range(max_iterations):
            # Calculate price and vega at current volatility
            results = self.price_with_details(option, sigma, valuation_date)
            price = results.price / option.notional
            
            # Calculate raw vega (not per 1% change) for Newton-Raphson
            vega = self._calculate_raw_vega(
                results.forward_bond_price,
                option.strike,
                results.time_to_expiry,
                sigma,
                results.discount_factor,
                results.d1
            )
            
            # Check convergence
            price_diff = price - target_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            # Newton-Raphson update with fallback to bisection if vega too small
            if abs(vega) < 1e-10:
                # Fallback: try bisection step
                if price < target_price:
                    sigma = sigma * 1.5  # increase vol
                else:
                    sigma = sigma * 0.5  # decrease vol
            else:
                sigma = sigma - price_diff / vega
            
            # Bounds check
            sigma = max(0.001, min(5.0, sigma))
        
        raise NumericalError(
            f"Implied volatility did not converge after {max_iterations} iterations"
        )
    
    def _calculate_raw_vega(
        self,
        F: float,
        K: float,
        T: float,
        sigma: float,
        discount_factor: float,
        d1: float
    ) -> float:
        """
        Calculate raw vega (sensitivity to volatility) for Newton-Raphson.
        
        Vega = D(T) * F * √T * n(d1)
        
        where n(d1) is the standard normal PDF.
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            sigma: Volatility
            discount_factor: Discount factor
            d1: d1 parameter
            
        Returns:
            Raw vega (∂V/∂σ)
        """
        sqrt_T = math.sqrt(T)
        n_d1 = stats.norm.pdf(d1)
        
        # Raw vega without scaling
        return discount_factor * F * sqrt_T * n_d1
    
    def _calculate_vega(
        self,
        F: float,
        K: float,
        T: float,
        sigma: float,
        discount_factor: float,
        d1: float
    ) -> float:
        """
        Calculate vega (sensitivity to volatility).
        
        Vega = D(T) * F * √T * n(d1)
        
        where n(d1) is the standard normal PDF.
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiry
            sigma: Volatility
            discount_factor: Discount factor
            d1: d1 parameter
            
        Returns:
            Vega (per 1% vol change)
        """
        sqrt_T = math.sqrt(T)
        n_d1 = stats.norm.pdf(d1)
        
        # Vega per 1% change (divide by 100)
        return discount_factor * F * sqrt_T * n_d1 / 100
    
    def __repr__(self):
        return f"BlackBondOptionEngine(valuation_date={self.pricing_env.valuation_date.date()})"

