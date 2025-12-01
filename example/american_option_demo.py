"""
Demonstration of American option pricing using analytical approximation methods.

This script demonstrates:
1. American call and put option pricing with all three methods (BS93, BS02, BAW)
2. Comparison between American and European option prices
3. Method comparison and accuracy analysis
4. Edge case handling (zero dividend, negative rates, high volatility)
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import AmericanOption, EuropeanVanillaOption
from asset.equity.engine.analytical import AmericanOptionAnalyticalEngine, BlackScholesEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from util.enum.engine_enums import AmericanAnalyticalMethod


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


def demo_american_call():
    """Demonstrate American call option pricing with all three methods."""
    print_section("American Call Option Pricing")
    
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.25)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.03)
    
    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )
    
    print("\nMarket Data:")
    print_market_data(pricing_env)
    
    print("\nOption Specification:")
    call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    print(f"Type:                  American Call")
    print(f"Strike (K):            ${call.strike:.2f}")
    print(f"Time to Maturity (T):  {call.maturity:.2f} years")
    
    print("\nPricing with Different Methods:")
    print("-" * 80)
    
    engine_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price_bs93 = engine_bs93.price(call, pricing_env)
    print(f"BS93 (Bjerksund-Stensland 1993):    ${price_bs93:.6f}")
    
    engine_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02)
    price_bs02 = engine_bs02.price(call, pricing_env)
    print(f"BS02 (Bjerksund-Stensland 2002):    ${price_bs02:.6f}")
    
    engine_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW)
    price_baw = engine_baw.price(call, pricing_env)
    print(f"BAW (Barone-Adesi-Whaley 1987):     ${price_baw:.6f}")
    
    euro_call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    euro_engine = BlackScholesEngine()
    euro_price = euro_engine.price(euro_call, pricing_env)
    print(f"\nEuropean Call (Black-Scholes):      ${euro_price:.6f}")
    
    print(f"\nAmerican Premium (BS93):            ${price_bs93 - euro_price:.6f}")
    print(f"American Premium (BS02):            ${price_bs02 - euro_price:.6f}")
    print(f"American Premium (BAW):             ${price_baw - euro_price:.6f}")
    
    intrinsic = call.intrinsic_value(100.0)
    print(f"\nIntrinsic Value:                    ${intrinsic:.6f}")
    print(f"Time Value (BS93):                  ${price_bs93 - intrinsic:.6f}")


def demo_american_put():
    """Demonstrate American put option pricing with all three methods."""
    print_section("American Put Option Pricing")
    
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.25)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.03)
    
    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )
    
    print("\nMarket Data:")
    print_market_data(pricing_env)
    
    print("\nOption Specification:")
    put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)
    print(f"Type:                  American Put")
    print(f"Strike (K):            ${put.strike:.2f}")
    print(f"Time to Maturity (T):  {put.maturity:.2f} years")
    
    print("\nPricing with Different Methods:")
    print("-" * 80)
    
    engine_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price_bs93 = engine_bs93.price(put, pricing_env)
    print(f"BS93 (put-call transformation):     ${price_bs93:.6f}")
    
    engine_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02)
    price_bs02 = engine_bs02.price(put, pricing_env)
    print(f"BS02 (put-call transformation):     ${price_bs02:.6f}")
    
    engine_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW)
    price_baw = engine_baw.price(put, pricing_env)
    print(f"BAW (direct put pricing):           ${price_baw:.6f}")
    
    euro_put = EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)
    euro_engine = BlackScholesEngine()
    euro_price = euro_engine.price(euro_put, pricing_env)
    print(f"\nEuropean Put (Black-Scholes):       ${euro_price:.6f}")
    
    print(f"\nAmerican Premium (BS93):            ${price_bs93 - euro_price:.6f}")
    print(f"American Premium (BS02):            ${price_bs02 - euro_price:.6f}")
    print(f"American Premium (BAW):             ${price_baw - euro_price:.6f}")
    
    intrinsic = put.intrinsic_value(100.0)
    print(f"\nIntrinsic Value:                    ${intrinsic:.6f}")
    print(f"Time Value (BS93):                  ${price_bs93 - intrinsic:.6f}")


def demo_method_comparison():
    """Compare all three methods across different scenarios."""
    print_section("Method Comparison Across Scenarios")
    
    scenarios = [
        {"name": "ATM, normal vol", "spot": 100.0, "strike": 100.0, "vol": 0.20},
        {"name": "ITM, normal vol", "spot": 110.0, "strike": 100.0, "vol": 0.20},
        {"name": "OTM, normal vol", "spot": 90.0, "strike": 100.0, "vol": 0.20},
        {"name": "ATM, high vol", "spot": 100.0, "strike": 100.0, "vol": 0.50},
        {"name": "ATM, low vol", "spot": 100.0, "strike": 100.0, "vol": 0.10},
    ]
    
    print("\nAmerican Call Pricing Comparison:")
    print("-" * 80)
    print(f"{'Scenario':<20} {'BS93':>12} {'BS02':>12} {'BAW':>12} {'Max Diff':>12}")
    print("-" * 80)
    
    for scenario in scenarios:
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=scenario["spot"]),
            vol_surface=FlatVolSurface(volatility=scenario["vol"]),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=datetime(2024, 1, 1),
        )
        
        call = AmericanOption(strike=scenario["strike"], option_type=OptionType.CALL, maturity=1.0)
        
        p_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(call, pricing_env)
        p_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02).price(call, pricing_env)
        p_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW).price(call, pricing_env)
        
        max_diff = max(abs(p_bs93 - p_bs02), abs(p_bs93 - p_baw), abs(p_bs02 - p_baw))
        
        print(f"{scenario['name']:<20} ${p_bs93:>10.4f} ${p_bs02:>10.4f} ${p_baw:>10.4f} ${max_diff:>10.4f}")
    
    print("\nAmerican Put Pricing Comparison:")
    print("-" * 80)
    print(f"{'Scenario':<20} {'BS93':>12} {'BS02':>12} {'BAW':>12} {'Max Diff':>12}")
    print("-" * 80)
    
    for scenario in scenarios:
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=scenario["spot"]),
            vol_surface=FlatVolSurface(volatility=scenario["vol"]),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=datetime(2024, 1, 1),
        )
        
        put = AmericanOption(strike=scenario["strike"], option_type=OptionType.PUT, maturity=1.0)
        
        p_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(put, pricing_env)
        p_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02).price(put, pricing_env)
        p_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW).price(put, pricing_env)
        
        max_diff = max(abs(p_bs93 - p_bs02), abs(p_bs93 - p_baw), abs(p_bs02 - p_baw))
        
        print(f"{scenario['name']:<20} ${p_bs93:>10.4f} ${p_bs02:>10.4f} ${p_baw:>10.4f} ${max_diff:>10.4f}")


def demo_edge_cases():
    """Demonstrate edge cases and special scenarios."""
    print_section("Edge Cases and Special Scenarios")
    
    print("\n1. Zero Dividend Call (American = European):")
    print("-" * 80)
    
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )
    
    call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    euro_call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    
    am_price = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(call, pricing_env)
    euro_price = BlackScholesEngine().price(euro_call, pricing_env)
    
    print(f"American Call Price:   ${am_price:.6f}")
    print(f"European Call Price:   ${euro_price:.6f}")
    print(f"Difference:            ${abs(am_price - euro_price):.6f}")
    
    print("\n2. Negative Interest Rates:")
    print("-" * 80)
    
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=-0.01),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    
    put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)
    put_price = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(put, pricing_env)
    
    print(f"American Put (r=-1%):  ${put_price:.6f}")
    
    print("\n3. High Volatility (σ = 100%):")
    print("-" * 80)
    
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=1.0),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    
    call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    call_price = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(call, pricing_env)
    
    print(f"American Call (σ=100%): ${call_price:.6f}")
    
    print("\n4. Deep In-The-Money:")
    print("-" * 80)
    
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=150.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    
    call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    call_price = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(call, pricing_env)
    intrinsic = 50.0
    
    print(f"Spot Price:            $150.00")
    print(f"Strike Price:          $100.00")
    print(f"Intrinsic Value:       ${intrinsic:.2f}")
    print(f"American Call Price:   ${call_price:.6f}")
    print(f"Time Value:            ${call_price - intrinsic:.6f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  American Vanilla Option Pricing Demonstration")
    print("  Using Analytical Approximation Methods")
    print("=" * 80)
    
    demo_american_call()
    demo_american_put()
    demo_method_comparison()
    demo_edge_cases()
    
    print("\n" + "=" * 80)
    print("  Demonstration Complete")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. American options have early exercise premium over European options")
    print("2. BS93, BS02, and BAW methods produce similar but not identical prices")
    print("3. BS02 is generally more accurate than BS93, especially for longer maturities")
    print("4. BAW uses direct put pricing while BS93/BS02 use put-call transformation")
    print("5. For zero-dividend calls, American = European (no early exercise)")
    print()