"""
Analytical pricing engine for European cash-or-nothing digital options.
"""

import math
from typing import Optional
from scipy import stats

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option import (
    CashOrNothingDigitalOption,
    EuropeanVanillaOption,
)
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.param import EngineParams
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError, NumericalError, PricingError


class DigitalOptionAnalyticalEngine(BaseEngine):
    """
    Closed-form Black-Scholes pricing for European cash-or-nothing digital options.

    Pricing formulas:
        Call = payout * exp(-rT) * N(d2)
        Put  = payout * exp(-rT) * N(-d2)
    where:
        d1 = [ln(S/K) + (r - q + 0.5σ²)T] / (σ√T)
        d2 = d1 - σ√T
    """

    engine_type = EngineType.ANALYTICAL

    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_MATURITY = 1e-10
    MAX_MATURITY = 30.0

    # Strike bump for the call-spread replication of the digital (-dC/dK).
    _H_REL = 1e-4
    _H_FLOOR = 1e-6

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize digital option analytical engine.

        Args:
            params: Engine configuration parameters
        """
        super().__init__(params)
        self._vanilla_engine = BlackScholesEngine()

    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Price a cash-or-nothing digital option using closed-form Black-Scholes.

        Args:
            product: CashOrNothingDigitalOption
            pricing_env: Pricing environment with market data

        Returns:
            Option price

        Raises:
            PricingError: If product is not a cash digital option
            ValidationError: If input parameters are invalid
            NumericalError: If numerical computation fails
        """
        if not isinstance(product, CashOrNothingDigitalOption):
            raise PricingError(
                f"DigitalOptionAnalyticalEngine only supports CashOrNothingDigitalOption, "
                f"got {type(product).__name__}"
            )

        # Extract parameters
        S = pricing_env.spot
        K = product.strike
        T = product.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        q = pricing_env.get_div_yield(T)
        sigma = pricing_env.get_vol(K, T)
        payout = product.payout

        # Validate inputs
        self._validate_inputs(S, K, T, r, q, sigma, payout)

        # Handle near-expiry
        if T < self.MIN_MATURITY:
            return product.get_payoff(S)

        # Under a smile surface the level-only N(d2) digital misses the skew
        # term (-vega * d-sigma/d-K). Price by static replication off the
        # smile-consistent vanilla call spread instead.
        if getattr(pricing_env.vol_surface, "is_smile", False):
            return self._replicated_digital(product, pricing_env) * product.contract_multiplier

        # Calculate d1 and d2 with numerical stability checks
        try:
            d1, d2 = self._calculate_d1_d2(S, K, T, r, q, sigma)
        except Exception as e:
            raise NumericalError(f"Error calculating d1/d2: {e}")

        # Calculate option price
        try:
            discount = math.exp(-r * T)
            if product.is_call():
                price = payout * discount * stats.norm.cdf(d2)
            else:
                price = payout * discount * stats.norm.cdf(-d2)
        except Exception as e:
            raise NumericalError(f"Error calculating digital option price: {e}")

        if price < 0:
            raise NumericalError(f"Negative price computed: {price}")

        return price * product.contract_multiplier

    def _replicated_digital(
        self, option: CashOrNothingDigitalOption, pricing_env: PricingEnvironment
    ) -> float:
        """Smile-consistent cash digital via the centred call spread (-dC/dK).

        cash_call(K) = -dC/dK ≈ (C(K-h) - C(K+h)) / (2h), each leg priced
        through the smile (BlackScholesEngine reads get_vol(K±h, T)). The put
        digital follows from parity: cash_call + cash_put = exp(-rT).
        """
        K = option.strike
        T = option.get_maturity(pricing_env)
        r = pricing_env.get_rate(T)
        h = max(self._H_FLOOR, self._H_REL * K)
        if K - h <= 0.0:
            raise PricingError(f"Replication strike bump h={h} too large for strike {K}")

        c_up = self._vanilla_call_price(pricing_env, K + h, T)
        c_dn = self._vanilla_call_price(pricing_env, K - h, T)
        cash_call = (c_dn - c_up) / (2.0 * h)  # = -dC/dK, per unit payout

        df = math.exp(-r * T)
        base = cash_call if option.is_call() else (df - cash_call)
        price = option.payout * base
        if price < 0:
            raise NumericalError(f"Negative replicated digital price: {price}")
        return price

    def _vanilla_call_price(
        self, pricing_env: PricingEnvironment, strike: float, maturity: float
    ) -> float:
        """Unit-payout European call price at `strike` priced through the smile."""
        leg = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )
        return self._vanilla_engine.price(leg, pricing_env)

    def _validate_inputs(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float, payout: float
    ) -> None:
        """
        Validate input parameters for numerical stability.
        """
        if S <= 0:
            raise ValidationError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValidationError(f"Strike price must be positive, got {K}")
        if payout <= 0:
            raise ValidationError(f"Payout must be positive, got {payout}")
        if T < 0:
            raise ValidationError(f"Time to maturity must be non-negative, got {T}")
        if sigma <= 0:
            raise ValidationError(f"Volatility must be positive, got {sigma}")
        if sigma < self.MIN_VOL or sigma > self.MAX_VOL:
            raise ValidationError(
                f"Volatility {sigma} outside supported range [{self.MIN_VOL}, {self.MAX_VOL}]"
            )
        if q < 0:
            raise ValidationError(f"Dividend yield must be non-negative, got {q}")
        if abs(r) > 1.0:
            raise ValidationError(f"Risk-free rate outside reasonable bounds: {r}")
        if T > self.MAX_MATURITY:
            raise ValidationError(
                f"Maturity too long for numerical stability: {T} years"
            )

    def _calculate_d1_d2(
        self, S: float, K: float, T: float, r: float, q: float, sigma: float
    ) -> tuple:
        """
        Calculate d1 and d2 parameters with numerical stability.
        """
        try:
            sqrt_T = math.sqrt(T)

            if S <= 0 or K <= 0:
                raise NumericalError(f"Invalid S ({S}) or K ({K}) for log calculation")

            log_moneyness = math.log(S / K)
            if abs(log_moneyness) > 100:
                raise NumericalError(f"Extreme moneyness: ln(S/K) = {log_moneyness}")

            var_term = sigma * sigma / 2
            drift_adjustment = (r - q + var_term) * T
            if abs(drift_adjustment) > 100:
                raise NumericalError(f"Extreme drift adjustment: {drift_adjustment}")

            numerator = log_moneyness + drift_adjustment
            denominator = sigma * sqrt_T

            if denominator <= 1e-10:
                raise NumericalError(f"Denominator too small: σ*√T = {denominator}")

            d1 = numerator / denominator
            d2 = d1 - sigma * sqrt_T

            return d1, d2
        except (OverflowError, ValueError) as e:
            raise NumericalError(f"Numerical overflow in d1/d2 calculation: {e}")
