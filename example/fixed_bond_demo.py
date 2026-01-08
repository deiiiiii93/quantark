"""
Demonstration of fixed bond pricing with QuantArk.

This example shows:
1. Creating a fixed bond with various payment frequencies
2. Using different day count conventions
3. Pricing bonds with interpolated rate curves
4. Calculating bond metrics (duration, convexity, YTM)
5. Computing accrued interest
"""

from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.bond.product.couponbond.fixed_bond import FixedBond, create_simple_fixed_bond
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from param.rrf.rate_curve import FlatRateCurve, LinearRateCurve, LogLinearRateCurve
from priceenv import PricingEnvironment
from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    CalendarType,
    create_calendar,
)
from util.enum import PaymentFrequency


def example_1_simple_bond():
    """Example 1: Price a simple semi-annual coupon bond."""
    print("=" * 80)
    print("Example 1: Simple Semi-Annual Coupon Bond")
    print("=" * 80)

    # Create a simple 5-year bond with 5% coupon, semi-annual payments
    issue_date = datetime(2023, 1, 1)
    maturity_date = datetime(2028, 1, 1)
    denominator = 1000.0
    coupon_rate = 0.05  # 5% annual coupon

    bond = create_simple_fixed_bond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA,
    )

    print(f"Bond: {bond}")
    print(f"Regular coupon payment: ${bond.get_coupon_payment():.2f}")

    # Create pricing environment with flat 4% rate curve
    valuation_date = datetime(2024, 2, 1)
    rate_curve = FlatRateCurve(rate=0.04)

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve, valuation_date=valuation_date
    )

    # Price the bond
    engine = BondDiscountEngine(pricing_env)

    dirty_price = engine.dirty_price(bond)
    clean_price = engine.clean_price(bond)
    accrued = engine.accrued_interest(bond)

    print(f"\nPricing as of {valuation_date.date()}:")
    print(f"  Dirty Price: ${dirty_price:.2f}")
    print(f"  Clean Price: ${clean_price:.2f}")
    print(f"  Accrued Interest: ${accrued:.2f}")
    print(f"  Price as % of par: {clean_price/denominator*100:.3f}%")

    # Calculate bond metrics
    mod_duration = engine.modified_duration(bond)
    convexity = engine.convexity(bond)
    dv01 = engine.dv01(bond)

    print(f"\nRisk Metrics:")
    print(f"  Modified Duration: {mod_duration:.4f} years")
    print(f"  Convexity: {convexity:.4f}")
    print(f"  DV01: ${dv01:.4f}")

    # Calculate yield to maturity
    ytm = engine.yield_to_maturity(bond, clean_price, clean_price=True)
    print(f"  Yield to Maturity: {ytm:.4%}")

    print()


def example_2_different_frequencies():
    """Example 2: Compare bonds with different payment frequencies."""
    print("=" * 80)
    print("Example 2: Bonds with Different Payment Frequencies")
    print("=" * 80)

    issue_date = datetime(2023, 1, 1)
    maturity_date = datetime(2028, 1, 1)
    denominator = 1000.0
    coupon_rate = 0.06
    valuation_date = datetime(2024, 1, 1)

    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve, valuation_date=valuation_date
    )
    engine = BondDiscountEngine(pricing_env)

    frequencies = [
        PaymentFrequency.ANNUAL,
        PaymentFrequency.SEMI_ANNUAL,
        PaymentFrequency.QUARTERLY,
        PaymentFrequency.MONTHLY,
    ]

    print(f"All bonds: 6% coupon, 5-year maturity, priced at 5% yield\n")

    for freq in frequencies:
        bond = create_simple_fixed_bond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            denominator=denominator,
            coupon_rate=coupon_rate,
            payment_frequency=freq,
        )

        price = engine.clean_price(bond)
        duration = engine.modified_duration(bond)

        print(f"{freq.name:15s}: Price=${price:8.2f}, Duration={duration:.4f} years")

    print()


def example_3_day_count_conventions():
    """Example 3: Compare different day count conventions."""
    print("=" * 80)
    print("Example 3: Different Day Count Conventions")
    print("=" * 80)

    issue_date = datetime(2023, 1, 1)
    maturity_date = datetime(2028, 1, 1)
    denominator = 1000.0
    coupon_rate = 0.05
    valuation_date = datetime(2024, 2, 1)

    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve, valuation_date=valuation_date
    )
    engine = BondDiscountEngine(pricing_env)

    conventions = [
        DayCountConvention.ACT_360,
        DayCountConvention.ACT_365,
        DayCountConvention.ACT_ACT_ISDA,
        DayCountConvention.THIRTY_360_US,
        DayCountConvention.THIRTY_360_EUROPEAN,
    ]

    print(f"All bonds: 5% coupon, semi-annual, priced at 5% yield\n")

    for convention in conventions:
        bond = create_simple_fixed_bond(
            issue_date=issue_date,
            maturity_date=maturity_date,
            denominator=denominator,
            coupon_rate=coupon_rate,
            day_count_convention=convention,
        )

        price = engine.clean_price(bond)
        accrued = engine.accrued_interest(bond)

        print(f"{convention.name:20s}: Price=${price:8.2f}, Accrued=${accrued:.2f}")

    print()


def example_4_interpolated_curves():
    """Example 4: Price bond with interpolated rate curves."""
    print("=" * 80)
    print("Example 4: Interpolated Rate Curves")
    print("=" * 80)

    issue_date = datetime(2023, 1, 1)
    maturity_date = datetime(2028, 1, 1)
    denominator = 1000.0
    coupon_rate = 0.05
    valuation_date = datetime(2024, 1, 1)

    bond = create_simple_fixed_bond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
    )

    # Define rate pillars (time, rate) - upward sloping curve
    pillars = [
        (0.5, 0.030),  # 6 months: 3.0%
        (1.0, 0.035),  # 1 year: 3.5%
        (2.0, 0.040),  # 2 years: 4.0%
        (5.0, 0.045),  # 5 years: 4.5%
        (10.0, 0.050),  # 10 years: 5.0%
    ]

    # Compare different interpolation methods
    curve_types = [
        ("Linear", LinearRateCurve(pillars)),
        ("Log-Linear", LogLinearRateCurve(pillars)),
    ]

    print(f"Bond: 5% coupon, 5-year maturity")
    print(f"Rate curve: Upward sloping from 3% to 5%\n")

    for name, curve in curve_types:
        pricing_env = PricingEnvironment(
            rate_curve=curve, valuation_date=valuation_date
        )
        engine = BondDiscountEngine(pricing_env)

        price = engine.clean_price(bond)
        duration = engine.modified_duration(bond)
        ytm = engine.yield_to_maturity(bond, price, clean_price=True)

        print(
            f"{name:15s}: Price=${price:8.2f}, Duration={duration:.4f}, YTM={ytm:.3%}"
        )

    print()


def example_5_yield_curve_sensitivity():
    """Example 5: Demonstrate yield curve sensitivity."""
    print("=" * 80)
    print("Example 5: Yield Curve Sensitivity Analysis")
    print("=" * 80)

    issue_date = datetime(2023, 1, 1)
    maturity_date = datetime(2033, 1, 1)  # 10-year bond
    denominator = 1000.0
    coupon_rate = 0.04
    valuation_date = datetime(2024, 1, 1)

    bond = create_simple_fixed_bond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
    )

    print(f"Bond: 4% coupon, 10-year maturity")
    print(f"Sensitivity to parallel yield curve shifts:\n")

    # Test parallel shifts
    yield_levels = [0.02, 0.03, 0.04, 0.05, 0.06]

    for yield_level in yield_levels:
        rate_curve = FlatRateCurve(rate=yield_level)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve, valuation_date=valuation_date
        )
        engine = BondDiscountEngine(pricing_env)

        price = engine.clean_price(bond)
        duration = engine.modified_duration(bond)

        print(
            f"Yield {yield_level:.1%}: Price=${price:8.2f} ({price/denominator*100:6.2f}%), "
            f"Duration={duration:.3f} years"
        )

    print()


def example_6_cashflow_schedule():
    """Example 6: Display cashflow schedule."""
    print("=" * 80)
    print("Example 6: Cashflow Schedule")
    print("=" * 80)

    issue_date = datetime(2024, 1, 1)
    maturity_date = datetime(2027, 1, 1)  # 3-year bond for shorter schedule
    denominator = 1000.0
    coupon_rate = 0.05

    bond = create_simple_fixed_bond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    )

    print(f"Bond: {bond}\n")
    print("Cashflow Schedule:")
    print("-" * 80)
    print(
        f"{'Payment Date':<15} {'Accrual Start':<15} {'Accrual End':<15} {'Amount':>12}"
    )
    print("-" * 80)

    cashflows = bond.get_all_cashflows()
    for cf in cashflows:
        print(
            f"{cf.payment_date.date()!s:<15} "
            f"{cf.accrual_start_date.date()!s:<15} "
            f"{cf.accrual_end_date.date()!s:<15} "
            f"${cf.amount:>10.2f}"
        )

    print("-" * 80)
    print(f"Total cashflows: {len(cashflows)}")
    print(f"Total payments: ${sum(cf.amount for cf in cashflows):.2f}")
    print()


def example_7_business_day_adjustment():
    """Example 7: Business day adjustments with calendar."""
    print("=" * 80)
    print("Example 7: Business Day Adjustments")
    print("=" * 80)

    issue_date = datetime(2024, 1, 1)
    maturity_date = datetime(2027, 1, 1)
    denominator = 1000.0
    coupon_rate = 0.05

    # Create bond with US calendar
    calendar = create_calendar(CalendarType.US, year_range=(2024, 2027))

    bond = FixedBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
        payment_frequency=PaymentFrequency.QUARTERLY,
        day_count_convention=DayCountConvention.ACT_360,
        calendar=calendar,
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
    )

    print(f"Bond with US calendar and Modified Following convention\n")
    print("First 8 payment dates (may be adjusted for holidays):")
    print("-" * 80)

    cashflows = bond.get_all_cashflows()[:8]
    for i, cf in enumerate(cashflows, 1):
        is_business_day = calendar.is_business_day(cf.payment_date)
        status = "✓ Business day" if is_business_day else "✗ Holiday/Weekend"
        print(f"{i}. {cf.payment_date.date()!s:<15} {status}")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FIXED BOND PRICING DEMONSTRATION")
    print("=" * 80 + "\n")

    example_1_simple_bond()
    example_2_different_frequencies()
    example_3_day_count_conventions()
    example_4_interpolated_curves()
    example_5_yield_curve_sensitivity()
    example_6_cashflow_schedule()
    example_7_business_day_adjustment()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
