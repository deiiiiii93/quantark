"""
Unit tests for cash-or-nothing digital option product and analytical pricing.
"""

import sys
import math
from pathlib import Path
from datetime import datetime
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.product.option import CashOrNothingDigitalOption, EuropeanVanillaOption  # noqa: E402
from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine  # noqa: E402
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import OptionType  # noqa: E402
from quantark.util.exceptions import ValidationError, PricingError  # noqa: E402


def _pricing_env(spot: float = 100.0, vol: float = 0.20, rate: float = 0.05, div: float = 0.02):
    """Helper to build pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def test_cash_digital_payoff():
    digital_call = CashOrNothingDigitalOption(
        strike=100.0, payout=10.0, option_type=OptionType.CALL, maturity=1.0
    )
    digital_put = CashOrNothingDigitalOption(
        strike=100.0, payout=10.0, option_type=OptionType.PUT, maturity=1.0
    )

    assert digital_call.get_payoff(110.0) == 10.0
    assert digital_call.get_payoff(100.0) == 0.0
    assert digital_put.get_payoff(90.0) == 10.0
    assert digital_put.get_payoff(100.0) == 0.0

    try:
        digital_call.get_payoff(-1.0)
        assert False, "Negative spot should raise ValidationError"
    except ValidationError:
        pass


def test_cash_digital_call_pricing():
    pricing_env = _pricing_env()
    payout = 10.0
    digital_call = CashOrNothingDigitalOption(
        strike=100.0, payout=payout, option_type=OptionType.CALL, maturity=1.0
    )

    engine = DigitalOptionAnalyticalEngine()
    price = engine.price(digital_call, pricing_env)

    # Expected from closed-form BSM digital: P * e^{-rT} * N(d2)
    S = pricing_env.spot
    K = digital_call.strike
    T = digital_call.get_maturity(pricing_env)
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)
    sigma = pricing_env.get_vol(K, T)
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    expected_price = payout * math.exp(-r * T) * stats.norm.cdf(d2)

    assert abs(price - expected_price) < 1e-6, f"Digital call price mismatch: {price} vs {expected_price}"


def test_cash_digital_put_pricing():
    pricing_env = _pricing_env()
    payout = 10.0
    digital_put = CashOrNothingDigitalOption(
        strike=100.0, payout=payout, option_type=OptionType.PUT, maturity=1.0
    )

    engine = DigitalOptionAnalyticalEngine()
    price = engine.price(digital_put, pricing_env)

    S = pricing_env.spot
    K = digital_put.strike
    T = digital_put.get_maturity(pricing_env)
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)
    sigma = pricing_env.get_vol(K, T)
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    expected_price = payout * math.exp(-r * T) * stats.norm.cdf(-d2)

    assert abs(price - expected_price) < 1e-6, f"Digital put price mismatch: {price} vs {expected_price}"


def test_near_expiry_returns_payoff():
    pricing_env = _pricing_env()
    digital_call = CashOrNothingDigitalOption(
        strike=100.0, payout=5.0, option_type=OptionType.CALL, maturity=1e-12
    )

    engine = DigitalOptionAnalyticalEngine()
    pricing_env.spot_quote.spot = 105.0
    price = engine.price(digital_call, pricing_env)

    assert price == 5.0, f"Near-expiry payoff mismatch, expected 5 got {price}"


def test_validation_and_type_errors():
    try:
        CashOrNothingDigitalOption(
            strike=100.0, payout=0.0, option_type=OptionType.CALL, maturity=1.0
        )
        assert False, "Zero payout should raise ValidationError"
    except ValidationError:
        pass

    pricing_env = _pricing_env()
    vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = DigitalOptionAnalyticalEngine()
    try:
        engine.price(vanilla, pricing_env)
        assert False, "Pricing non-digital product should raise PricingError"
    except PricingError:
        pass

