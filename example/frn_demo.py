"""
Demonstration of Floating Rate Note (FRN) pricing with QuantArk.

This example shows:
1. Creating FRNs with different indices (SOFR, EURIBOR, SHIBOR)
2. Pricing with forward curves
3. Calculating Discount Margin and Simple Margin
4. Comparing in-advance vs in-arrears reset conventions
5. Displaying cashflow schedules
6. Working with rate caps and floors
"""

from datetime import datetime
from pathlib import Path
import sys


from quantark.asset.bond.product.couponbond.frn import (
    FloatingRateBond,
    FloatingCashFlow,
    create_simple_frn,
)
from quantark.asset.bond.engine.discount.frn_engine import FRNDiscountEngine, FRNPricingResults
from quantark.param.index import (
    RateIndex,
    IndexFixing,
    IndexFixingStore,
    SOFR,
    SOFR_3M,
    EURIBOR_3M,
    EURIBOR_6M,
    SHIBOR_3M,
    REPO_7D,
    create_index,
)
from quantark.param.rrf import FlatRateCurve
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention, BusinessDayConvention
from quantark.util.enum import PaymentFrequency, ResetConvention


def example_1_simple_sofr_frn():
    """Example 1: Price a simple SOFR-linked FRN."""
    print("=" * 80)
    print("Example 1: Simple SOFR-Linked Floating Rate Note")
    print("=" * 80)
    
    # Create a 5-year FRN linked to 3-month SOFR with 50bp spread
    issue_date = datetime(2024, 1, 1)
    maturity_date = datetime(2029, 1, 1)
    denominator = 1000000.0
    spread = 0.0050  # 50bp
    
    frn = create_simple_frn(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        index=SOFR_3M,
        spread=spread,
        payment_frequency=PaymentFrequency.QUARTERLY
    )
    
    print(f"FRN: {frn}")
    print(f"Index: {frn.index}")
    print(f"Spread: {frn.spread:.2%}")
    print(f"Day Count: {frn.day_count_convention.name}")
    
    # Create pricing environment with 5% flat curve
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    
    engine = FRNDiscountEngine(pricing_env)
    
    # Price the FRN
    dirty_price = engine.dirty_price(frn)
    clean_price = engine.clean_price(frn)
    accrued = frn.calculate_accrued_interest(valuation_date)
    
    print(f"\nPricing as of {valuation_date.date()} (issue date):")
    print(f"  Dirty Price: ${dirty_price:,.2f}")
    print(f"  Clean Price: ${clean_price:,.2f}")
    print(f"  Accrued Interest: ${accrued:,.2f}")
    print(f"  Price as % of par: {clean_price/denominator*100:.3f}%")
    
    # Risk metrics
    eff_dur = engine.effective_duration(frn)
    spread_dur = engine.spread_duration(frn)
    wal = engine.weighted_average_life(frn)
    dv01 = engine.dv01(frn)
    cs01 = engine.cs01(frn)
    
    print(f"\nRisk Metrics:")
    print(f"  Effective Duration: {eff_dur:.4f} years")
    print(f"  Spread Duration: {spread_dur:.4f} years")
    print(f"  Weighted Average Life: {wal:.2f} years")
    print(f"  DV01: ${dv01:,.2f}")
    print(f"  CS01: ${cs01:,.2f}")
    
    print()


def example_2_discount_margin_calculation():
    """Example 2: Calculate Discount Margin for FRN."""
    print("=" * 80)
    print("Example 2: Discount Margin Calculation")
    print("=" * 80)
    
    frn = create_simple_frn(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        denominator=1000000.0,
        index=SOFR_3M,
        spread=0.0050,  # 50bp quoted spread
        payment_frequency=PaymentFrequency.QUARTERLY
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = FRNDiscountEngine(pricing_env)
    
    print(f"FRN Quoted Spread: {frn.spread:.2%}")
    print(f"\nCalculating Discount Margin at different prices:\n")
    
    # Test at different prices
    prices = [980000, 990000, 1000000, 1010000, 1020000]
    
    print(f"{'Price':>12} {'Price %':>10} {'DM':>10} {'Simple Margin':>15}")
    print("-" * 50)
    
    for price in prices:
        dm = engine.discount_margin(frn, price, clean_price=True)
        sm = engine.simple_margin(frn, price, clean_price=True)
        
        print(f"${price:>10,} {price/frn.denominator*100:>9.2f}% {dm*10000:>9.1f}bp {sm*10000:>14.1f}bp")
    
    print("\nNote: At par, DM equals the quoted spread")
    print()


def example_3_different_indices():
    """Example 3: FRNs with different reference indices."""
    print("=" * 80)
    print("Example 3: FRNs with Different Reference Indices")
    print("=" * 80)
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = FRNDiscountEngine(pricing_env)
    
    # Define FRNs with different indices
    frns = [
        ("USD SOFR FRN", SOFR_3M, 0.0050, "USD", 1000000),
        ("EUR EURIBOR FRN", EURIBOR_3M, 0.0075, "EUR", 1000000),
        ("CNY SHIBOR FRN", SHIBOR_3M, 0.0080, "CNY", 10000000),
        ("CNY DR007 FRN", REPO_7D, 0.0100, "CNY", 50000000),
    ]
    
    print(f"Comparing FRNs with 5-year maturity:\n")
    print(f"{'Name':<20} {'Index':<12} {'Spread':>8} {'Price %':>10} {'Eff Dur':>10}")
    print("-" * 65)
    
    for name, index, spread, currency, denominator in frns:
        frn = create_simple_frn(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            denominator=denominator,
            index=index,
            spread=spread,
            payment_frequency=PaymentFrequency.QUARTERLY
        )
        
        price = engine.clean_price(frn)
        eff_dur = engine.effective_duration(frn)
        
        print(f"{name:<20} {index.name:<12} {spread*10000:>6.0f}bp {price/denominator*100:>9.2f}% {eff_dur:>9.4f}")
    
    print()


def example_4_reset_conventions():
    """Example 4: Compare in-advance vs in-arrears reset."""
    print("=" * 80)
    print("Example 4: Reset Conventions - In-Advance vs In-Arrears")
    print("=" * 80)
    
    base_params = {
        "issue_date": datetime(2024, 1, 1),
        "maturity_date": datetime(2027, 1, 1),
        "denominator": 1000000.0,
        "index": SOFR_3M,
        "spread": 0.0050,
        "payment_frequency": PaymentFrequency.QUARTERLY,
    }
    
    # In-advance (standard)
    frn_advance = FloatingRateBond(
        **base_params,
        reset_convention=ResetConvention.IN_ADVANCE
    )
    
    # In-arrears with lookback
    frn_arrears = FloatingRateBond(
        **base_params,
        reset_convention=ResetConvention.IN_ARREARS,
        lookback_days=5
    )
    
    print("In-Advance Reset:")
    print(f"  Rate is fixed at the start of each accrual period")
    print(f"  First cashflow fixing date: {frn_advance.get_all_floating_cashflows()[0].fixing_date.date()}")
    
    print("\nIn-Arrears Reset (with 5-day lookback):")
    print(f"  Rate is fixed near the end of each accrual period")
    print(f"  First cashflow fixing date: {frn_arrears.get_all_floating_cashflows()[0].fixing_date.date()}")
    
    print()


def example_5_cap_and_floor():
    """Example 5: FRN with rate caps and floors."""
    print("=" * 80)
    print("Example 5: FRN with Rate Cap and Floor")
    print("=" * 80)
    
    # Create FRN with collar (cap and floor)
    frn = FloatingRateBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        denominator=1000000.0,
        index=SOFR_3M,
        spread=0.0050,  # 50bp spread
        payment_frequency=PaymentFrequency.QUARTERLY,
        rate_cap=0.08,   # 8% cap
        rate_floor=0.03  # 3% floor
    )
    
    print(f"FRN with Collar:")
    print(f"  Index: {frn.index.name}")
    print(f"  Spread: {frn.spread:.2%}")
    print(f"  Rate Cap: {frn.rate_cap:.2%}")
    print(f"  Rate Floor: {frn.rate_floor:.2%}")
    
    # Simulate different rate scenarios
    print(f"\nEffective rate at different index levels:")
    print(f"{'Index Rate':>12} {'+ Spread':>12} {'Effective Rate':>15}")
    print("-" * 42)
    
    test_rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    
    for index_rate in test_rates:
        total = index_rate + frn.spread
        effective = min(frn.rate_cap, max(frn.rate_floor, total))
        
        status = ""
        if effective == frn.rate_cap:
            status = " (capped)"
        elif effective == frn.rate_floor:
            status = " (floored)"
        
        print(f"{index_rate:>11.2%} {total:>11.2%} {effective:>14.2%}{status}")
    
    print()


def example_6_cashflow_schedule():
    """Example 6: Display detailed cashflow schedule."""
    print("=" * 80)
    print("Example 6: FRN Cashflow Schedule")
    print("=" * 80)
    
    frn = create_simple_frn(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2026, 1, 1),  # 2-year for shorter display
        denominator=1000000.0,
        index=SOFR_3M,
        spread=0.0050,
        payment_frequency=PaymentFrequency.QUARTERLY
    )
    
    # Add some historical fixings
    cashflows = frn.get_all_floating_cashflows()
    
    # Simulate fixings for first two periods
    frn.add_fixing(cashflows[0].fixing_date, 0.0525)
    frn.add_fixing(cashflows[1].fixing_date, 0.0530)
    
    print(f"FRN: {frn}")
    print(f"\nCashflow Schedule:")
    print("-" * 100)
    print(f"{'#':<3} {'Payment Date':<14} {'Fixing Date':<14} {'Status':<10} {'Rate':>8} {'DCF':>8} {'Amount':>14}")
    print("-" * 100)
    
    cashflows = frn.get_all_floating_cashflows()
    
    for i, cf in enumerate(cashflows, 1):
        status = "Fixed" if not cf.is_projected else "Projected"
        rate = cf.effective_rate
        
        print(f"{i:<3} {cf.payment_date.date()!s:<14} {cf.fixing_date.date()!s:<14} "
              f"{status:<10} {rate:>7.2%} {cf.day_count_fraction:>7.4f} ${cf.amount:>12,.2f}")
    
    # Add principal to last cashflow for total
    total_coupons = sum(cf.amount for cf in cashflows)
    print("-" * 100)
    print(f"Total coupon payments: ${total_coupons:,.2f}")
    print(f"Principal at maturity: ${frn.denominator:,.2f}")
    print(f"Total cashflows: ${total_coupons + frn.denominator:,.2f}")
    
    print()


def example_7_full_analysis():
    """Example 7: Complete FRN analysis."""
    print("=" * 80)
    print("Example 7: Complete FRN Analysis")
    print("=" * 80)
    
    frn = create_simple_frn(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        denominator=1000000.0,
        index=SOFR_3M,
        spread=0.0060,  # 60bp spread
        payment_frequency=PaymentFrequency.QUARTERLY
    )
    
    valuation_date = datetime(2024, 1, 1)
    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = FRNDiscountEngine(pricing_env)
    
    # Market price at slight discount
    market_price = 985000.0
    
    results = engine.full_analysis(
        frn,
        market_price=market_price,
        clean_price=True
    )
    
    print(f"FRN Details:")
    print(f"  Index: {frn.index.name}")
    print(f"  Quoted Spread: {frn.spread:.2%}")
    print(f"  Maturity: {frn.maturity_date.date()}")
    print(f"  Notional: ${frn.denominator:,.2f}")
    
    print(f"\nMarket Data:")
    print(f"  Market Price: ${market_price:,.2f}")
    print(f"  Valuation Date: {valuation_date.date()}")
    
    print(f"\nPricing Results:")
    print(f"  Model Dirty Price: ${results.dirty_price:,.2f}")
    print(f"  Model Clean Price: ${results.clean_price:,.2f}")
    print(f"  Accrued Interest: ${results.accrued_interest:,.2f}")
    
    print(f"\nYield Measures:")
    print(f"  Yield to Maturity: {results.yield_to_maturity:.2%}" if results.yield_to_maturity else "  Yield to Maturity: N/A")
    print(f"  Discount Margin: {results.discount_margin*10000:.1f}bp" if results.discount_margin else "  Discount Margin: N/A")
    print(f"  Simple Margin: {results.simple_margin*10000:.1f}bp" if results.simple_margin else "  Simple Margin: N/A")
    print(f"  Current Coupon: {results.current_coupon:.2%}" if results.current_coupon else "  Current Coupon: N/A")
    print(f"  Assumed Index Rate: {results.assumed_index_rate:.2%}" if results.assumed_index_rate else "  Assumed Index Rate: N/A")
    
    print(f"\nRisk Measures:")
    print(f"  Effective Duration: {results.effective_duration:.4f} years")
    print(f"  Spread Duration: {results.spread_duration:.4f} years")
    print(f"  Weighted Average Life: {results.weighted_average_life:.2f} years")
    
    print()


def example_8_upward_sloping_curve():
    """Example 8: Pricing with upward-sloping forward curve."""
    print("=" * 80)
    print("Example 8: Pricing with Upward-Sloping Forward Curve")
    print("=" * 80)
    
    frn = create_simple_frn(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        denominator=1000000.0,
        index=SOFR_3M,
        spread=0.0050,
        payment_frequency=PaymentFrequency.QUARTERLY
    )
    
    # Define upward-sloping curve
    pillars = [
        (0.25, 0.045),  # 3 months: 4.5%
        (0.5, 0.047),   # 6 months: 4.7%
        (1.0, 0.050),   # 1 year: 5.0%
        (2.0, 0.053),   # 2 years: 5.3%
        (5.0, 0.055),   # 5 years: 5.5%
        (10.0, 0.057),  # 10 years: 5.7%
    ]
    
    valuation_date = datetime(2024, 1, 1)
    
    # Compare flat vs upward-sloping curve
    flat_curve = FlatRateCurve(rate=0.050)
    sloped_curve = LinearRateCurve(pillars)
    
    print("Rate Curve Comparison:\n")
    print(f"{'Tenor':>10} {'Flat':>10} {'Sloped':>10}")
    print("-" * 32)
    
    for tenor in [0.25, 0.5, 1.0, 2.0, 5.0]:
        print(f"{tenor:>10.2f}y {flat_curve.get_rate(tenor)*100:>9.2f}% {sloped_curve.get_rate(tenor)*100:>9.2f}%")
    
    print("\nFRN Pricing Comparison:\n")
    
    for curve_name, curve in [("Flat 5%", flat_curve), ("Upward Sloping", sloped_curve)]:
        pricing_env = PricingEnvironment(rate_curve=curve, valuation_date=valuation_date)
        engine = FRNDiscountEngine(pricing_env)
        
        price = engine.clean_price(frn)
        
        print(f"{curve_name:<20}: ${price:>12,.2f} ({price/frn.denominator*100:.3f}%)")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FLOATING RATE NOTE (FRN) PRICING DEMONSTRATION")
    print("=" * 80 + "\n")
    
    example_1_simple_sofr_frn()
    example_2_discount_margin_calculation()
    example_3_different_indices()
    example_4_reset_conventions()
    example_5_cap_and_floor()
    example_6_cashflow_schedule()
    example_7_full_analysis()
    example_8_upward_sloping_curve()
    
    print("=" * 80)
    print("All FRN examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

