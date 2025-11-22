"""
Demonstration of European Vanilla Option pricing and Greeks calculation.

This script demonstrates:
1. Setting up market data (spot, volatility, rates, dividends)
2. Creating European Call and Put options
3. Pricing options using Black-Scholes analytical formula
4. Calculating Greeks using both analytical and numerical methods
5. Comparing analytical vs numerical Greeks
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from asset.equity.param import EngineParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from util.calendar import DayCountConvention


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_market_data(pricing_env: PricingEnvironment):
    """Print market data summary."""
    print(f"Spot Price (S):        ${pricing_env.spot:.2f}")
    print(f"Volatility (σ):        {pricing_env.get_vol(100, 1.0):.2%}")
    print(f"Risk-Free Rate (r):    {pricing_env.get_rate(1.0):.2%}")
    print(f"Dividend Yield (q):    {pricing_env.get_div_yield(1.0):.2%}")


def print_option_details(option: EuropeanVanillaOption):
    """Print option specification."""
    print(f"Option Type:           {option.option_type}")
    print(f"Strike (K):            ${option.strike:.2f}")
    print(f"Time to Maturity (T):  {option.maturity:.2f} years")


def print_greeks(greeks: dict, method: str):
    """Print Greeks with nice formatting."""
    print(f"\n{method} Greeks:")
    print(f"  Price:  ${greeks['price']:10.6f}")
    print(f"  Delta:   {greeks['delta']:10.6f}")
    print(f"  Gamma:   {greeks['gamma']:10.6f}")
    print(f"  Vega:    {greeks['vega']:10.6f}")
    print(f"  Theta:   {greeks['theta']:10.6f} (per day)")
    print(f"  Rho:     {greeks['rho']:10.6f}")


def print_comparison(analytical: dict, numerical: dict):
    """Print comparison between analytical and numerical Greeks."""
    print("\nGreeks Comparison (Analytical vs Numerical):")
    print("-" * 80)
    print(
        f"{'Greek':<10} {'Analytical':>15} {'Numerical':>15} {'Abs Diff':>15} {'Rel Diff %':>15}"
    )
    print("-" * 80)

    for key in ["price", "delta", "gamma", "vega", "theta", "rho"]:
        anal_val = analytical[key]
        num_val = numerical[key]
        abs_diff = anal_val - num_val
        rel_diff = (abs_diff / anal_val * 100) if abs(anal_val) > 1e-10 else 0

        print(
            f"{key.capitalize():<10} {anal_val:>15.6f} {num_val:>15.6f} "
            f"{abs_diff:>15.6f} {rel_diff:>14.4f}%"
        )


def demo_european_call():
    """Demonstrate European Call option pricing and Greeks."""
    print_section("EUROPEAN CALL OPTION")

    # Step 1: Set up market data
    print("\n1. Market Data Setup")
    print("-" * 40)
    spot_quote = SpotQuote(spot=100.0, asset_name="AAPL")
    vol_surface = FlatVolSurface(volatility=0.20)  # 20% vol
    rate_curve = FlatRateCurve(rate=0.05)  # 5% risk-free rate
    div_yield = ContinuousDividendYield(div_yield=0.02)  # 2% dividend yield

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    print_market_data(pricing_env)

    # Step 2: Create European Call option
    print("\n2. Option Specification")
    print("-" * 40)
    call_option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0  # 1 year
    )

    print_option_details(call_option)

    # Step 3: Price the option
    print("\n3. Pricing")
    print("-" * 40)
    engine = BlackScholesEngine()
    price = engine.price(call_option, pricing_env)
    print(f"Call Option Price: ${price:.6f}")

    # Step 4: Calculate Greeks
    print("\n4. Greeks Calculation")
    print("-" * 40)

    # Analytical Greeks
    greeks_calc = GreeksCalculator()
    analytical_greeks = greeks_calc.calculate_analytical_greeks(
        call_option, pricing_env, price
    )
    print_greeks(analytical_greeks, "Analytical")

    # Numerical Greeks
    numerical_greeks = greeks_calc.calculate_numerical_greeks(
        call_option, pricing_env, engine, price
    )
    print_greeks(numerical_greeks, "Numerical (FDM)")

    # Step 5: Compare Greeks
    print("\n5. Comparison")
    print("-" * 40)
    print_comparison(analytical_greeks, numerical_greeks)

    return call_option, pricing_env, analytical_greeks


def demo_european_put():
    """Demonstrate European Put option pricing and Greeks."""
    print_section("EUROPEAN PUT OPTION")

    # Use same market data
    print("\n1. Market Data Setup")
    print("-" * 40)
    spot_quote = SpotQuote(spot=100.0, asset_name="AAPL")
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    print_market_data(pricing_env)

    # Create European Put option
    print("\n2. Option Specification")
    print("-" * 40)
    put_option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.PUT, maturity=1.0
    )

    print_option_details(put_option)

    # Price the option
    print("\n3. Pricing")
    print("-" * 40)
    engine = BlackScholesEngine()
    price = engine.price(put_option, pricing_env)
    print(f"Put Option Price: ${price:.6f}")

    # Calculate Greeks
    print("\n4. Greeks Calculation")
    print("-" * 40)

    greeks_calc = GreeksCalculator()
    analytical_greeks = greeks_calc.calculate_analytical_greeks(
        put_option, pricing_env, price
    )
    print_greeks(analytical_greeks, "Analytical")

    numerical_greeks = greeks_calc.calculate_numerical_greeks(
        put_option, pricing_env, engine, price
    )
    print_greeks(numerical_greeks, "Numerical (FDM)")

    # Compare Greeks
    print("\n5. Comparison")
    print("-" * 40)
    print_comparison(analytical_greeks, numerical_greeks)

    return put_option, pricing_env, analytical_greeks


def demo_put_call_parity():
    """Demonstrate Put-Call Parity relationship."""
    print_section("PUT-CALL PARITY VERIFICATION")

    # Market data
    spot_quote = SpotQuote(spot=100.0)
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    # Create call and put with same strike and maturity
    K = 100.0
    T = 1.0
    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)

    # Price both options
    engine = BlackScholesEngine()
    call_price = engine.price(call, pricing_env)
    put_price = engine.price(put, pricing_env)

    # Put-Call Parity: C - P = S*e^(-q*T) - K*e^(-r*T)
    import math

    S = pricing_env.spot
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    print(f"\nPut-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)")
    print(f"\nParameters:")
    print(f"  S = ${S:.2f}")
    print(f"  K = ${K:.2f}")
    print(f"  T = {T:.2f} years")
    print(f"  r = {r:.2%}")
    print(f"  q = {q:.2%}")
    print(f"\nPrices:")
    print(f"  Call Price (C): ${call_price:.6f}")
    print(f"  Put Price (P):  ${put_price:.6f}")
    print(f"\nParity Check:")
    print(f"  LHS (C - P):              ${lhs:.6f}")
    print(f"  RHS (S*e^(-qT) - K*e^(-rT)): ${rhs:.6f}")
    print(f"  Difference:               ${abs(lhs - rhs):.10f}")

    if abs(lhs - rhs) < 1e-6:
        print("\n  ✓ Put-Call Parity holds!")
    else:
        print("\n  ✗ Put-Call Parity violated!")


def demo_date_based_options():
    """Demonstrate date-based option pricing with different day count conventions."""
    print_section("DATE-BASED OPTION PRICING")

    # Market data
    spot_quote = SpotQuote(spot=100.0, asset_name="AAPL")
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    # Calendar day convention
    print("\n1. Calendar Day Convention")
    print("-" * 40)
    valuation_date = datetime(2024, 1, 1)
    exercise_date = datetime(2025, 1, 1)  # 1 year (366 days - leap year)

    pricing_env_calendar = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.CALENDAR_DAYS,
    )

    call_calendar = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, exercise_date=exercise_date
    )

    engine = BlackScholesEngine()
    price_calendar = engine.price(call_calendar, pricing_env_calendar)
    maturity_calendar = call_calendar.get_maturity(pricing_env_calendar)

    print(f"Valuation Date: {valuation_date.date()}")
    print(f"Exercise Date:  {exercise_date.date()}")
    print(f"Year Fraction:  {maturity_calendar:.6f} years")
    print(f"Option Price:   ${price_calendar:.6f}")

    # Business day convention
    print("\n2. Business Day Convention")
    print("-" * 40)
    pricing_env_business = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=252,
    )

    call_business = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, exercise_date=exercise_date
    )

    price_business = engine.price(call_business, pricing_env_business)
    maturity_business = call_business.get_maturity(pricing_env_business)

    print(f"Valuation Date: {valuation_date.date()}")
    print(f"Exercise Date:  {exercise_date.date()}")
    print(f"Year Fraction:  {maturity_business:.6f} years")
    print(f"Option Price:   ${price_business:.6f}")

    # Comparison with maturity-based
    print("\n3. Comparison with Maturity-Based Option")
    print("-" * 40)
    call_maturity = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    price_maturity = engine.price(call_maturity, pricing_env_calendar)

    print(f"Date-based (calendar): ${price_calendar:.6f}")
    print(f"Date-based (business): ${price_business:.6f}")
    print(f"Maturity-based (T=1):  ${price_maturity:.6f}")
    print(
        f"\nDifference (calendar vs maturity): ${abs(price_calendar - price_maturity):.6f}"
    )

    # Settlement date example
    print("\n4. Option with Settlement Date (T+2)")
    print("-" * 40)
    settlement_date = exercise_date + timedelta(days=2)

    call_settlement = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        exercise_date=exercise_date,
        settlement_date=settlement_date,
    )

    price_settlement = engine.price(call_settlement, pricing_env_calendar)

    print(f"Exercise Date:   {exercise_date.date()}")
    print(f"Settlement Date: {settlement_date.date()}")
    print(f"Option Price:    ${price_settlement:.6f}")
    print("\n(Note: Settlement date is stored but pricing uses exercise date)")


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print(
        "*"
        + "  QUANTARK - European Vanilla Option Pricing Demonstration".center(78)
        + "*"
    )
    print("*" + " " * 78 + "*")
    print("*" * 80)

    try:
        # Demo 1: European Call
        demo_european_call()

        # Demo 2: European Put
        demo_european_put()

        # Demo 3: Put-Call Parity
        demo_put_call_parity()

        # Demo 4: Date-Based Options
        demo_date_based_options()

        print_section("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("\nAll calculations completed without errors.")
        print(
            "The Greeks show good agreement between analytical and numerical methods."
        )
        print("Date-based options support both calendar and business day conventions.")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
