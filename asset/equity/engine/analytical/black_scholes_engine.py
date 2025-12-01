"""
Analytical Black-Scholes pricing engine for European options.
"""

import math
from typing import Optional
from scipy import stats
from asset.equity.engine.base_engine import BaseEngine
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.product.base_equity_product import BaseEquityProduct
from asset.equity.param import EngineParams
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, NumericalError, PricingError


class BlackScholesEngine(BaseEngine):
    """
    Analytical Black-Scholes engine for European vanilla options.

    Uses closed-form Black-Scholes-Merton formula with continuous dividends:
        Call: S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
        Put:  K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

    where:
        d1 = [ln(S/K) + (r - q + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T
    """

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize Black-Scholes engine.

        Args:
            params: Engine configuration parameters
        """
        super().__init__(params)

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a European vanilla option using Black-Scholes formula.

        Args:
            product: European vanilla option
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a European vanilla option
            NumericalError: If numerical issues occur during calculation
        """
        if not isinstance(product, EuropeanVanillaOption):
            raise PricingError(
                f"BlackScholesEngine only supports EuropeanVanillaOption, "
                f"got {type(product).__name__}"
            )

        # Extract parameters
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)

        # Validate inputs
        self._validate_inputs(S, K, T, r, q, sigma)

        # Handle edge cases
        if T < 1e-10:  # Option has expired
            return product.get_payoff(S)

        # Calculate d1 and d2 with numerical stability checks
        try:
            d1, d2 = self._calculate_d1_d2(S, K, T, r, q, sigma)
        except Exception as e:
            raise NumericalError(f"Error calculating d1/d2: {e}")

        # Calculate option price
        try:
            if product.is_call():
                price = self._price_call(S, K, T, r, q, d1, d2)
            else:
                price = self._price_put(S, K, T, r, q, d1, d2)
        except Exception as e:
            raise NumericalError(f"Error calculating option price: {e}")

        # Sanity checks on output
        if price < 0:
            raise NumericalError(f"Negative price computed: {price}")

        # Check against intrinsic value
        intrinsic = product.intrinsic_value(S)
        if price < intrinsic - 1e-6:  # Small tolerance for numerical errors
            raise NumericalError(
                f"Price ({price:.6f}) below intrinsic value ({intrinsic:.6f})"
            )

        return price

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> None:
        """
        Validate input parameters for numerical stability.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility

        Raises:
            ValidationError: If inputs are invalid
        """
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValidationError(f"Strike price must be positive, got {K}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if sigma > 5.0:
            raise ValidationError(
                f"Volatility too high for numerical stability: {sigma}"
            )
        if q < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {q}")

        # Check for extreme parameter combinations
        if abs(r) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {r}")
        if T > 100:
            raise ValidationError(
                f"Maturity too long for numerical stability: {T} years"
            )

    def _calculate_d1_d2(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> tuple:
        """
        Calculate d1 and d2 parameters with numerical stability.

        d1 = [ln(S/K) + (r - q + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility

        Returns:
            Tuple of (d1, d2)

        Raises:
            NumericalError: If calculation encounters numerical issues
        """
        try:
            # Calculate sqrt(T) once
            sqrt_T = math.sqrt(T)

            # Calculate log moneyness with error handling
            if S <= 0 or K <= 0:
                raise NumericalError(f"Invalid S ({S}) or K ({K}) for log calculation")

            log_moneyness = math.log(S / K)

            # Check for extreme values that could cause overflow
            if abs(log_moneyness) > 100:
                raise NumericalError(f"Extreme moneyness: ln(S/K) = {log_moneyness}")

            # Calculate variance term
            var_term = sigma * sigma / 2

            # Calculate drift-adjusted log moneyness
            drift_adjustment = (r - q + var_term) * T

            # Check for overflow before computing
            if abs(drift_adjustment) > 100:
                raise NumericalError(f"Extreme drift adjustment: {drift_adjustment}")

            numerator = log_moneyness + drift_adjustment
            denominator = sigma * sqrt_T

            if denominator <= 1e-10:
                raise NumericalError(f"Denominator too small: σ*√T = {denominator}")

            d1 = numerator / denominator
            d2 = d1 - sigma * sqrt_T

            # Check for extreme values in d1/d2
            if abs(d1) > 10 or abs(d2) > 10:
                # This is a warning but not necessarily an error
                # Options deep ITM or OTM can have large d1/d2
                pass

            return d1, d2

        except (OverflowError, ValueError) as e:
            raise NumericalError(f"Numerical overflow in d1/d2 calculation: {e}")

    def _price_call(
        self, S: float, K: float, T: float, r: float, q: float, d1: float, d2: float
    ) -> float:
        """
        Calculate call option price.

        Call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            d1: d1 parameter
            d2: d2 parameter

        Returns:
            Call option price
        """
        try:
            # Calculate discount factors with overflow protection
            if abs(q * T) > 100 or abs(r * T) > 100:
                raise NumericalError("Extreme discount factor exponent")

            discount_div = math.exp(-q * T)
            discount_rf = math.exp(-r * T)

            # Calculate normal CDFs
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)

            # Calculate price components
            forward_component = S * discount_div * N_d1
            strike_component = K * discount_rf * N_d2

            price = forward_component - strike_component

            return price

        except (OverflowError, ValueError) as e:
            raise NumericalError(f"Numerical error in call pricing: {e}")

    def _price_put(
        self, S: float, K: float, T: float, r: float, q: float, d1: float, d2: float
    ) -> float:
        """
        Calculate put option price.

        Put = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            q: Dividend yield
            d1: d1 parameter
            d2: d2 parameter

        Returns:
            Put option price
        """
        try:
            # Calculate discount factors with overflow protection
            if abs(q * T) > 100 or abs(r * T) > 100:
                raise NumericalError("Extreme discount factor exponent")

            discount_div = math.exp(-q * T)
            discount_rf = math.exp(-r * T)

            # Calculate normal CDFs for put
            N_minus_d1 = stats.norm.cdf(-d1)
            N_minus_d2 = stats.norm.cdf(-d2)

            # Calculate price components
            strike_component = K * discount_rf * N_minus_d2
            forward_component = S * discount_div * N_minus_d1

            price = strike_component - forward_component

            return price

        except (OverflowError, ValueError) as e:
            raise NumericalError(f"Numerical error in put pricing: {e}")

    def __repr__(self):
        return "BlackScholesEngine(analytical)"
