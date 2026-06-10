"""
Unit tests for European vanilla option pricing and Greeks.
"""

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import OptionType
from quantark.util.exceptions import NumericalError, ValidationError


def _discounted_european_lower_bound(
    option: EuropeanVanillaOption,
    spot: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
) -> float:
    spot_pv = spot * math.exp(-dividend_yield * maturity)
    strike_pv = option.strike * math.exp(-rate * maturity)
    if option.is_call():
        return max(spot_pv - strike_pv, 0.0) * option.contract_multiplier
    return max(strike_pv - spot_pv, 0.0) * option.contract_multiplier


def test_high_dividend_call_can_price_below_immediate_payoff():
    """European call lower bound discounts dividends and strike."""
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=8356.744),
        vol_surface=FlatVolSurface(volatility=0.2395),
        rate_curve=FlatRateCurve(rate=0.014),
        div_yield=ContinuousDividendYield(div_yield=0.0795),
        valuation_date=datetime(2026, 6, 6),
    )
    call = EuropeanVanillaOption(
        strike=98.77679014859717,
        option_type=OptionType.CALL,
        exercise_date=datetime(2026, 8, 12),
        contract_multiplier=1.0,
    )

    engine = BlackScholesEngine()
    price = engine.price(call, pricing_env)
    maturity = call.get_maturity(pricing_env)
    immediate_payoff = call.intrinsic_value(pricing_env.spot)
    lower_bound = _discounted_european_lower_bound(
        call, pricing_env.spot, 0.014, 0.0795, maturity
    )

    assert price == pytest.approx(8137.155016, abs=1e-6)
    assert price < immediate_payoff
    assert price >= lower_bound - 1e-6


def test_discounted_put_lower_bound_can_be_below_immediate_payoff():
    """European put lower bound discounts the strike, unlike immediate payoff."""
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=1.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.10),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )
    put = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )

    price = BlackScholesEngine().price(put, pricing_env)
    immediate_payoff = put.intrinsic_value(pricing_env.spot)
    lower_bound = _discounted_european_lower_bound(
        put, pricing_env.spot, 0.10, 0.0, 1.0
    )

    assert price < immediate_payoff
    assert price >= lower_bound - 1e-6


def test_black_scholes_rejects_price_below_discounted_european_lower_bound():
    """The sanity gate rejects prices below the European PV lower bound."""
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=120.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.01),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )
    call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    lower_bound = _discounted_european_lower_bound(call, 120.0, 0.01, 0.0, 1.0)

    engine._price_call = lambda *args: lower_bound - 1e-5

    with pytest.raises(NumericalError, match="discounted European lower bound"):
        engine.price(call, pricing_env)


def test_black_scholes_still_rejects_negative_prices():
    """The lower-bound fix must not weaken negative price validation."""
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.01),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )
    call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    engine._price_call = lambda *args: -1.0

    with pytest.raises(NumericalError, match="Negative price computed"):
        engine.price(call, pricing_env)


def test_call_option_pricing():
    """Test European call option pricing."""
    # Market data
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    # Create call option
    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    # Price the option
    engine = BlackScholesEngine()
    price = engine.price(call, pricing_env)

    # Expected price (pre-calculated)
    expected_price = 9.227006

    assert abs(price - expected_price) < 0.0001, (
        f"Call price mismatch: {price} vs {expected_price}"
    )
    print(f"✓ Call option pricing test passed: ${price:.6f}")


def test_put_option_pricing():
    """Test European put option pricing."""
    # Market data
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    # Create put option
    put = EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

    # Price the option
    engine = BlackScholesEngine()
    price = engine.price(put, pricing_env)

    # Expected price (pre-calculated)
    expected_price = 6.330081

    assert abs(price - expected_price) < 0.0001, (
        f"Put price mismatch: {price} vs {expected_price}"
    )
    print(f"✓ Put option pricing test passed: ${price:.6f}")


def test_put_call_parity():
    """Test put-call parity relationship."""
    # Market data
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02

    spot = SpotQuote(spot=S)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=r)
    div = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    # Create call and put
    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)

    # Price both
    engine = BlackScholesEngine()
    call_price = engine.price(call, pricing_env)
    put_price = engine.price(put, pricing_env)

    # Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)
    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    assert abs(lhs - rhs) < 1e-6, f"Put-call parity violated: {lhs} vs {rhs}"
    print(f"✓ Put-call parity test passed: difference = {abs(lhs - rhs):.10f}")


def test_greeks_call():
    """Test Greeks calculation for call option."""
    # Market data
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)

    engine = BlackScholesEngine()
    price = engine.price(call, pricing_env)

    # Calculate analytical Greeks
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.calculate_analytical_greeks(call, pricing_env, price)

    # Verify Greeks are in reasonable ranges
    assert 0 < greeks["delta"] < 1, f"Call delta out of range: {greeks['delta']}"
    assert greeks["gamma"] > 0, f"Gamma should be positive: {greeks['gamma']}"
    assert greeks["vega"] > 0, f"Vega should be positive: {greeks['vega']}"
    assert greeks["theta"] < 0, f"Long call theta should be negative: {greeks['theta']}"

    print(f"✓ Call Greeks test passed")
    print(f"  Delta: {greeks['delta']:.6f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega:  {greeks['vega']:.6f}")
    print(f"  Theta: {greeks['theta']:.6f}")
    print(f"  Rho:   {greeks['rho']:.6f}")


def test_greeks_put():
    """Test Greeks calculation for put option."""
    # Market data
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    put = EuropeanVanillaOption(100.0, OptionType.PUT, maturity=1.0)

    engine = BlackScholesEngine()
    price = engine.price(put, pricing_env)

    # Calculate analytical Greeks
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.calculate_analytical_greeks(put, pricing_env, price)

    # Verify Greeks are in reasonable ranges
    assert -1 < greeks["delta"] < 0, f"Put delta out of range: {greeks['delta']}"
    assert greeks["gamma"] > 0, f"Gamma should be positive: {greeks['gamma']}"
    assert greeks["vega"] > 0, f"Vega should be positive: {greeks['vega']}"
    assert greeks["theta"] < 0, f"Long put theta should be negative: {greeks['theta']}"

    print(f"✓ Put Greeks test passed")
    print(f"  Delta: {greeks['delta']:.6f}")
    print(f"  Gamma: {greeks['gamma']:.6f}")
    print(f"  Vega:  {greeks['vega']:.6f}")
    print(f"  Theta: {greeks['theta']:.6f}")
    print(f"  Rho:   {greeks['rho']:.6f}")


def test_validation_errors():
    """Test that validation errors are raised for invalid inputs."""
    # Test negative spot
    try:
        spot = SpotQuote(spot=-100.0)
        assert False, "Should have raised ValidationError for negative spot"
    except ValidationError:
        print("✓ Negative spot validation test passed")

    # Test zero spot
    try:
        spot = SpotQuote(spot=0.0)
        assert False, "Should have raised ValidationError for zero spot"
    except ValidationError:
        print("✓ Zero spot validation test passed")

    # Test negative volatility
    try:
        vol = FlatVolSurface(volatility=-0.20)
        assert False, "Should have raised ValidationError for negative vol"
    except ValidationError:
        print("✓ Negative volatility validation test passed")

    # Test zero volatility
    try:
        vol = FlatVolSurface(volatility=0.0)
        assert False, "Should have raised ValidationError for zero vol"
    except ValidationError:
        print("✓ Zero volatility validation test passed")

    # Test negative strike
    try:
        option = EuropeanVanillaOption(
            strike=-100.0, option_type=OptionType.CALL, maturity=1.0
        )
        assert False, "Should have raised ValidationError for negative strike"
    except ValidationError:
        print("✓ Negative strike validation test passed")

    # Test negative maturity
    try:
        option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=-1.0
        )
        assert False, "Should have raised ValidationError for negative maturity"
    except ValidationError:
        print("✓ Negative maturity validation test passed")


def test_itm_otm_options():
    """Test in-the-money and out-of-the-money options."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )

    engine = BlackScholesEngine()

    # Deep ITM call (strike = 80)
    itm_call = EuropeanVanillaOption(80.0, OptionType.CALL, maturity=1.0)
    itm_call_price = engine.price(itm_call, pricing_env)
    assert itm_call_price > 20.0, "Deep ITM call should have high value"
    print(f"✓ Deep ITM call test passed: ${itm_call_price:.6f}")

    # Deep OTM call (strike = 120)
    otm_call = EuropeanVanillaOption(120.0, OptionType.CALL, maturity=1.0)
    otm_call_price = engine.price(otm_call, pricing_env)
    assert otm_call_price < 5.0, "Deep OTM call should have low value"
    print(f"✓ Deep OTM call test passed: ${otm_call_price:.6f}")


def test_date_based_option_calendar_days():
    """Test date-based option with calendar day convention."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    valuation_date = datetime(2024, 1, 1)
    exercise_date = datetime(2025, 1, 1)  # Exactly 1 year (366 days - leap year)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.CALENDAR_DAYS,
    )

    # Create call option with dates
    call_dates = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, exercise_date=exercise_date
    )

    # Create call option with maturity
    call_maturity = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=366.0 / 365.0
    )

    engine = BlackScholesEngine()
    price_dates = engine.price(call_dates, pricing_env)
    price_maturity = engine.price(call_maturity, pricing_env)

    # Prices should be very close
    assert abs(price_dates - price_maturity) < 0.01, (
        f"Date-based and maturity-based prices should match: "
        f"{price_dates:.6f} vs {price_maturity:.6f}"
    )
    print(
        f"✓ Date-based option (calendar days) test passed: ${price_dates:.6f} vs ${price_maturity:.6f}"
    )


def test_date_based_option_business_days():
    """Test date-based option with business day convention."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    valuation_date = datetime(2024, 1, 1)
    exercise_date = datetime(2024, 7, 1)  # 6 months

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=252,
    )

    # Create call option with dates
    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, exercise_date=exercise_date
    )

    engine = BlackScholesEngine()
    price = engine.price(call, pricing_env)

    # Verify maturity calculation
    maturity = call.get_maturity(pricing_env)
    assert maturity > 0, f"Maturity should be positive: {maturity}"
    print(
        f"✓ Date-based option (business days) test passed: ${price:.6f}, maturity={maturity:.4f}"
    )


def test_date_based_settlement_date():
    """Test date-based option with settlement date."""
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.02)

    valuation_date = datetime(2024, 1, 1)
    exercise_date = datetime(2025, 1, 1)
    settlement_date = datetime(2025, 1, 3)  # T+2 settlement

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=valuation_date,
    )

    # Create call option with both dates
    call = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        exercise_date=exercise_date,
        settlement_date=settlement_date,
    )

    engine = BlackScholesEngine()
    price = engine.price(call, pricing_env)

    assert price > 0, f"Price should be positive: {price}"
    print(f"✓ Date-based option with settlement date test passed: ${price:.6f}")


def test_date_validation():
    """Test date validation errors."""
    valuation_date = datetime(2024, 1, 1)
    exercise_date = datetime(2023, 12, 31)  # Before valuation date

    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.20)
    rate = FlatRateCurve(rate=0.05)

    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        valuation_date=valuation_date,
    )

    # Create option with invalid dates
    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, exercise_date=exercise_date
    )

    engine = BlackScholesEngine()
    try:
        price = engine.price(call, pricing_env)
        assert False, (
            "Should have raised ValidationError for exercise date before valuation date"
        )
    except ValidationError:
        print("✓ Date validation test passed (exercise date before valuation date)")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("Running QuantArk Unit Tests")
    print("=" * 70 + "\n")

    tests = [
        ("Call Option Pricing", test_call_option_pricing),
        ("Put Option Pricing", test_put_option_pricing),
        ("Put-Call Parity", test_put_call_parity),
        ("Call Greeks", test_greeks_call),
        ("Put Greeks", test_greeks_put),
        ("Validation Errors", test_validation_errors),
        ("ITM/OTM Options", test_itm_otm_options),
        ("Date-Based Option (Calendar Days)", test_date_based_option_calendar_days),
        ("Date-Based Option (Business Days)", test_date_based_option_business_days),
        ("Date-Based Settlement Date", test_date_based_settlement_date),
        ("Date Validation", test_date_validation),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\nTest: {test_name}")
            print("-" * 70)
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
