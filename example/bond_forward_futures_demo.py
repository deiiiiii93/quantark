"""
Demonstration of Bond Forward and Bond Futures pricing with QuantArk.

This example shows:
1. Creating and pricing bond forwards with repo rate
2. Creating bond futures with deliverable basket
3. CTD (Cheapest to Deliver) analysis for bond futures
4. Conversion factor calculation
5. Basis analysis (gross basis, net basis, implied repo)
6. Greeks calculation (DV01, duration, convexity)
7. Hedge ratio calculation
"""

from datetime import datetime
from pathlib import Path
import sys


from quantark.asset.bond.product.couponbond.fixed_bond import FixedBond, create_simple_fixed_bond
from quantark.asset.bond.product.forward.bond_forward import BondForward
from quantark.asset.bond.product.futures.bond_futures import BondFutures, DeliverableBond
from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from quantark.asset.bond.engine.analytical.bond_forward_engine import BondForwardEngine
from quantark.asset.bond.engine.analytical.bond_futures_engine import BondFuturesEngine
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import PaymentFrequency


def example_1_simple_bond_forward():
    """Example 1: Price a simple bond forward contract."""
    print("=" * 80)
    print("Example 1: Simple Bond Forward Contract")
    print("=" * 80)

    # Create underlying bond: 10-year 5% coupon bond
    issue_date = datetime(2023, 1, 15)
    maturity_date = datetime(2033, 1, 15)
    denominator = 100.0
    coupon_rate = 0.05

    bond = create_simple_fixed_bond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA,
    )

    print(f"Underlying Bond: {bond}")

    # Create forward contract: 6-month forward delivery
    delivery_date = datetime(2024, 6, 15)
    repo_rate = 0.045  # 4.5% repo rate

    forward = BondForward(
        underlying=bond,
        delivery_date=delivery_date,
        repo_rate=repo_rate,
        is_long=True,
        contract_size=100_000,
    )

    print(f"Forward Contract: {forward}")
    print(f"  Delivery Date: {delivery_date.date()}")
    print(f"  Repo Rate: {repo_rate:.2%}")

    # Set up pricing environment
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.05)

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    # Price the forward
    engine = BondForwardEngine(pricing_env)
    results = engine.price(forward, valuation_date)

    print(f"\nPricing Results as of {valuation_date.date()}:")
    print(f"  Spot Dirty Price: ${results.spot_dirty_price:.4f}")
    print(f"  Spot Clean Price: ${results.spot_clean_price:.4f}")
    print(f"  Accrued at Spot: ${results.accrued_at_spot:.4f}")
    print(f"  Forward Dirty Price: ${results.forward_dirty_price:.4f}")
    print(f"  Forward Clean Price: ${results.forward_clean_price:.4f}")
    print(f"  Accrued at Delivery: ${results.accrued_at_delivery:.4f}")
    print(f"  Time to Delivery: {results.time_to_delivery:.4f} years")

    # Calculate Greeks
    greeks = engine.calculate_greeks(forward, valuation_date)

    print(f"\nRisk Measures (Greeks):")
    print(f"  DV01: ${greeks['dv01']:.6f}")
    print(f"  Modified Duration: {greeks['modified_duration']:.4f} years")
    print(f"  Convexity: {greeks['convexity']:.4f}")
    print(f"  Repo Sensitivity: ${greeks['repo_sensitivity']:.6f} per bp")
    print(f"  Daily Carry: ${greeks['carry']:.6f}")

    # Calculate basis point value
    bpv = engine.calculate_basis_point_value(forward, valuation_date)
    print(f"  Basis Point Value: ${bpv:.2f} per contract")

    print()


def example_2_forward_with_contracted_price():
    """Example 2: Forward with contracted price - mark-to-market value."""
    print("=" * 80)
    print("Example 2: Forward Contract with Contracted Price (MTM)")
    print("=" * 80)

    # Create underlying bond
    bond = create_simple_fixed_bond(
        issue_date=datetime(2023, 6, 15),
        maturity_date=datetime(2033, 6, 15),
        denominator=100.0,
        coupon_rate=0.04,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    )

    # Create forward with a contracted price
    delivery_date = datetime(2024, 12, 15)
    contracted_price = 98.50  # Agreed forward clean price

    forward = BondForward(
        underlying=bond,
        delivery_date=delivery_date,
        forward_price=contracted_price,
        repo_rate=0.05,
        is_long=True,
        contract_size=1_000_000,
    )

    print(f"Forward Contract with Contracted Price:")
    print(f"  Underlying: {bond}")
    print(f"  Delivery Date: {delivery_date.date()}")
    print(f"  Contracted Forward Price: ${contracted_price:.4f}")

    # Price at different market conditions
    valuation_date = datetime(2024, 6, 15)

    rate_levels = [0.04, 0.045, 0.05, 0.055, 0.06]

    print(f"\nMark-to-Market Value at Different Rate Levels:")
    print("-" * 70)
    print(f"{'Rate':<10} {'Spot Dirty':<15} {'Fwd Clean':<15} {'Contract Value':<15}")
    print("-" * 70)

    for rate in rate_levels:
        rate_curve = FlatRateCurve(rate=rate)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date,
        )

        engine = BondForwardEngine(pricing_env)
        results = engine.price(forward, valuation_date)

        print(
            f"{rate:.2%}      ${results.spot_dirty_price:>10.4f}    "
            f"${results.forward_clean_price:>10.4f}    ${results.forward_value:>12,.2f}"
        )

    print("-" * 70)
    print()


def example_3_implied_repo_rate():
    """Example 3: Calculate implied repo rate from forward price."""
    print("=" * 80)
    print("Example 3: Implied Repo Rate Calculation")
    print("=" * 80)

    # Create underlying bond
    bond = create_simple_fixed_bond(
        issue_date=datetime(2023, 1, 15),
        maturity_date=datetime(2033, 1, 15),
        denominator=100.0,
        coupon_rate=0.05,
    )

    # Different forward prices to analyze
    delivery_date = datetime(2024, 7, 15)
    forward_prices = [99.0, 99.5, 100.0, 100.5, 101.0]

    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.05)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    # Get spot price
    bond_engine = BondDiscountEngine(pricing_env)
    spot_dirty = bond_engine.dirty_price(bond, valuation_date)

    print(f"Bond: {bond}")
    print(f"Spot Dirty Price: ${spot_dirty:.4f}")
    print(f"Delivery Date: {delivery_date.date()}")
    print(f"\nImplied Repo Rates from Different Forward Prices:")
    print("-" * 50)
    print(f"{'Forward Clean':<15} {'Implied Repo':<15}")
    print("-" * 50)

    for fwd_price in forward_prices:
        forward = BondForward(
            underlying=bond,
            delivery_date=delivery_date,
            forward_price=fwd_price,
            repo_rate=0.0,  # Will calculate implied
            is_long=True,
        )

        engine = BondForwardEngine(pricing_env)
        results = engine.price(forward, valuation_date)

        print(f"${fwd_price:<14.4f} {results.implied_repo_rate:>12.4%}")

    print("-" * 50)
    print()


def example_4_simple_bond_futures():
    """Example 4: Price a bond futures contract with CTD analysis."""
    print("=" * 80)
    print("Example 4: Bond Futures with CTD Analysis")
    print("=" * 80)

    # Create deliverable basket of bonds
    # Bond 1: Higher coupon, shorter maturity
    bond1 = create_simple_fixed_bond(
        issue_date=datetime(2020, 3, 15),
        maturity_date=datetime(2030, 3, 15),
        denominator=100.0,
        coupon_rate=0.045,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    )

    # Bond 2: Medium coupon, medium maturity
    bond2 = create_simple_fixed_bond(
        issue_date=datetime(2021, 6, 15),
        maturity_date=datetime(2031, 6, 15),
        denominator=100.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    )

    # Bond 3: Lower coupon, longer maturity
    bond3 = create_simple_fixed_bond(
        issue_date=datetime(2022, 9, 15),
        maturity_date=datetime(2032, 9, 15),
        denominator=100.0,
        coupon_rate=0.035,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
    )

    print("Deliverable Basket:")
    print(f"  Bond 1: {bond1}")
    print(f"  Bond 2: {bond2}")
    print(f"  Bond 3: {bond3}")

    # Create futures contract
    delivery_date = datetime(2025, 3, 15)
    futures_price = 112.50

    futures = BondFutures(
        delivery_date=delivery_date,
        deliverable_basket=[
            DeliverableBond(bond1),
            DeliverableBond(bond2),
            DeliverableBond(bond3),
        ],
        futures_price=futures_price,
        contract_size=100_000,
    )

    print(f"\nFutures Contract:")
    print(f"  Delivery Date: {delivery_date.date()}")
    print(f"  Futures Price: ${futures_price:.4f}")
    print(f"  Contract Size: ${futures.contract_size:,.0f}")

    # Display conversion factors
    print(f"\nConversion Factors (auto-calculated):")
    for i in range(futures.get_basket_size()):
        cf = futures.get_conversion_factor(i)
        print(f"  Bond {i+1}: CF = {cf:.4f}")

    # Set up pricing environment
    valuation_date = datetime(2024, 11, 15)
    rate_curve = FlatRateCurve(rate=0.045)

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    # Price the futures
    engine = BondFuturesEngine(pricing_env, repo_rate=0.045)
    results = engine.price(futures, valuation_date)

    print(f"\nPricing Results as of {valuation_date.date()}:")
    print(f"  Theoretical Futures Price: ${results.theoretical_futures_price:.4f}")
    print(f"  CTD Bond: Bond {results.ctd_bond_index + 1}")
    print(f"  CTD Implied Repo: {results.ctd_implied_repo:.4%}")
    print(f"  Time to Delivery: {results.time_to_delivery:.4f} years")

    print(f"\nRisk Measures:")
    print(f"  DV01: ${results.dv01:.6f}")
    print(f"  Modified Duration: {results.modified_duration:.4f} years")
    print(f"  Convexity: {results.convexity:.4f}")

    print()


def example_5_detailed_basis_analysis():
    """Example 5: Detailed basis analysis for all deliverable bonds."""
    print("=" * 80)
    print("Example 5: Detailed Basis Analysis")
    print("=" * 80)

    # Create diverse deliverable basket
    bonds = [
        create_simple_fixed_bond(
            issue_date=datetime(2019, 1, 15),
            maturity_date=datetime(2029, 1, 15),
            denominator=100.0,
            coupon_rate=0.06,
        ),
        create_simple_fixed_bond(
            issue_date=datetime(2020, 7, 15),
            maturity_date=datetime(2030, 7, 15),
            denominator=100.0,
            coupon_rate=0.05,
        ),
        create_simple_fixed_bond(
            issue_date=datetime(2021, 3, 15),
            maturity_date=datetime(2031, 3, 15),
            denominator=100.0,
            coupon_rate=0.04,
        ),
        create_simple_fixed_bond(
            issue_date=datetime(2022, 9, 15),
            maturity_date=datetime(2032, 9, 15),
            denominator=100.0,
            coupon_rate=0.035,
        ),
    ]

    # Create futures
    delivery_date = datetime(2025, 6, 15)
    futures = BondFutures(
        delivery_date=delivery_date,
        deliverable_basket=[DeliverableBond(b) for b in bonds],
        futures_price=108.25,
        contract_size=100_000,
    )

    # Price and analyze
    valuation_date = datetime(2024, 12, 15)
    rate_curve = FlatRateCurve(rate=0.04)

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    engine = BondFuturesEngine(pricing_env, repo_rate=0.04)

    # Get basis analysis
    analyses = engine.analyze_basis(futures, valuation_date)

    print(
        f"Futures: Delivery {delivery_date.date()}, Price ${futures.futures_price:.4f}"
    )
    print(f"Valuation Date: {valuation_date.date()}")
    print(f"\nBasis Analysis (sorted by implied repo - highest = CTD):")
    print("-" * 100)
    print(
        f"{'Bond':<25} {'CF':<8} {'Dirty':<10} {'Gross Basis':<12} "
        f"{'Net Basis':<12} {'Implied Repo':<12} {'CTD':<5}"
    )
    print("-" * 100)

    for a in analyses:
        ctd_marker = "***" if a["is_ctd"] else ""
        print(
            f"{a['bond_description']:<25} {a['conversion_factor']:<8.4f} "
            f"${a['dirty_price']:<9.4f} ${a['gross_basis']:<11.4f} "
            f"${a['net_basis']:<11.4f} {a['implied_repo_rate']:<11.4%} {ctd_marker}"
        )

    print("-" * 100)
    print()


def example_6_conversion_factors():
    """Example 6: Demonstrate conversion factor calculation."""
    print("=" * 80)
    print("Example 6: Conversion Factor Calculation")
    print("=" * 80)

    delivery_date = datetime(2025, 3, 15)

    # Create bonds with different coupons and maturities
    test_cases = [
        (0.02, 7),  # 2% coupon, 7 years
        (0.04, 7),  # 4% coupon, 7 years
        (0.06, 7),  # 6% coupon (notional), 7 years
        (0.08, 7),  # 8% coupon, 7 years
        (0.06, 5),  # 6% coupon, 5 years
        (0.06, 10),  # 6% coupon, 10 years
        (0.06, 15),  # 6% coupon, 15 years
    ]

    print("Conversion factors for different bonds:")
    print("(Notional coupon = 6%, so CF = 1.0 for 6% coupon bond)")
    print("-" * 60)
    print(f"{'Coupon':<10} {'Maturity':<15} {'Conversion Factor':<20}")
    print("-" * 60)

    for coupon, years in test_cases:
        maturity = datetime(
            delivery_date.year + years, delivery_date.month, delivery_date.day
        )

        bond = create_simple_fixed_bond(
            issue_date=datetime(2020, 3, 15),
            maturity_date=maturity,
            denominator=100.0,
            coupon_rate=coupon,
        )

        # Create futures just to calculate CF
        futures = BondFutures(
            delivery_date=delivery_date,
            deliverable_basket=[DeliverableBond(bond)],
            futures_price=100.0,
        )

        cf = futures.get_conversion_factor(0)

        print(f"{coupon:.1%}       {years} years         {cf:.4f}")

    print("-" * 60)
    print("\nNote: CF < 1 for high-coupon bonds (premium bonds)")
    print("      CF > 1 for low-coupon bonds (discount bonds)")
    print()


def example_7_hedge_ratio():
    """Example 7: Calculate hedge ratio for bond position."""
    print("=" * 80)
    print("Example 7: Hedge Ratio Calculation")
    print("=" * 80)

    # Target bond to hedge
    target_bond = create_simple_fixed_bond(
        issue_date=datetime(2022, 6, 15),
        maturity_date=datetime(2032, 6, 15),
        denominator=100.0,
        coupon_rate=0.055,
    )

    # Position size
    position_notional = 10_000_000  # $10 million position

    # Futures for hedging
    futures = BondFutures(
        delivery_date=datetime(2025, 3, 15),
        deliverable_basket=[
            DeliverableBond(
                create_simple_fixed_bond(
                    issue_date=datetime(2021, 3, 15),
                    maturity_date=datetime(2031, 3, 15),
                    denominator=100.0,
                    coupon_rate=0.05,
                )
            ),
        ],
        futures_price=110.0,
        contract_size=100_000,
    )

    valuation_date = datetime(2024, 11, 15)
    rate_curve = FlatRateCurve(rate=0.045)

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    # Calculate target bond DV01
    bond_engine = BondDiscountEngine(pricing_env)
    target_dv01 = bond_engine.dv01(target_bond)

    # Scale to position size
    position_dv01 = target_dv01 * position_notional / 100.0

    print(f"Target Bond: {target_bond}")
    print(f"Position Notional: ${position_notional:,.0f}")
    print(f"Target Bond DV01 (per 100): ${target_dv01:.6f}")
    print(f"Position DV01: ${position_dv01:,.2f}")

    # Calculate hedge ratio
    futures_engine = BondFuturesEngine(pricing_env, repo_rate=0.045)
    hedge_ratio = futures_engine.calculate_hedge_ratio(
        futures, position_dv01, valuation_date
    )

    # Get futures Greeks
    fut_results = futures_engine.price(futures, valuation_date)
    futures_dv01_per_contract = fut_results.dv01 * futures.contract_size / 100.0

    print(f"\nFutures Contract:")
    print(f"  Theoretical Price: ${fut_results.theoretical_futures_price:.4f}")
    print(f"  DV01 per contract: ${futures_dv01_per_contract:,.2f}")

    print(f"\nHedge Calculation:")
    print(f"  Hedge Ratio: {hedge_ratio:.2f} contracts")
    print(f"  Rounded: {round(hedge_ratio)} contracts")

    # Verify hedge effectiveness
    hedged_dv01 = position_dv01 - round(hedge_ratio) * futures_dv01_per_contract
    hedge_effectiveness = 1 - abs(hedged_dv01) / position_dv01

    print(f"\nHedge Effectiveness:")
    print(f"  Residual DV01: ${hedged_dv01:,.2f}")
    print(f"  Hedge Effectiveness: {hedge_effectiveness:.2%}")
    print()


def example_8_delivery_option_value():
    """Example 8: Estimate delivery option value."""
    print("=" * 80)
    print("Example 8: Delivery Option Value Estimation")
    print("=" * 80)

    # Create diverse basket
    bonds = [
        create_simple_fixed_bond(
            issue_date=datetime(2018, 1, 15),
            maturity_date=datetime(2028, 1, 15),
            denominator=100.0,
            coupon_rate=0.07,  # High coupon
        ),
        create_simple_fixed_bond(
            issue_date=datetime(2020, 7, 15),
            maturity_date=datetime(2030, 7, 15),
            denominator=100.0,
            coupon_rate=0.05,  # Medium coupon
        ),
        create_simple_fixed_bond(
            issue_date=datetime(2022, 3, 15),
            maturity_date=datetime(2032, 3, 15),
            denominator=100.0,
            coupon_rate=0.03,  # Low coupon
        ),
    ]

    futures = BondFutures(
        delivery_date=datetime(2025, 6, 15),
        deliverable_basket=[DeliverableBond(b) for b in bonds],
        futures_price=105.0,
        contract_size=100_000,
    )

    valuation_date = datetime(2024, 12, 15)
    rate_curve = FlatRateCurve(rate=0.045)

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    engine = BondFuturesEngine(pricing_env, repo_rate=0.045)

    # Get delivery option value
    option_analysis = engine.find_delivery_option_value(futures, valuation_date)

    print("Delivery Option Analysis:")
    print("-" * 50)
    print(f"CTD Theoretical Price: ${option_analysis['ctd_theoretical_price']:.4f}")
    print(
        f"Average Theoretical Price: ${option_analysis['average_theoretical_price']:.4f}"
    )
    print(f"Max Theoretical Price: ${option_analysis['max_theoretical_price']:.4f}")
    print("-" * 50)
    print(f"Delivery Option Value: ${option_analysis['delivery_option_value']:.4f}")
    print(f"Option Value (% of CTD): {option_analysis['option_value_pct']:.2f}%")
    print()

    print("Interpretation:")
    print("  The delivery option value represents the advantage the seller has")
    print("  by being able to choose which bond to deliver. A positive value")
    print("  indicates the futures trades cheap relative to a single-bond forward.")
    print()


def example_9_rate_sensitivity():
    """Example 9: Rate sensitivity comparison - Forward vs Futures."""
    print("=" * 80)
    print("Example 9: Rate Sensitivity - Forward vs Futures")
    print("=" * 80)

    # Common underlying
    bond = create_simple_fixed_bond(
        issue_date=datetime(2022, 6, 15),
        maturity_date=datetime(2032, 6, 15),
        denominator=100.0,
        coupon_rate=0.05,
    )

    delivery_date = datetime(2025, 6, 15)

    # Create forward
    forward = BondForward(
        underlying=bond,
        delivery_date=delivery_date,
        repo_rate=0.045,
        is_long=True,
    )

    # Create futures with same underlying
    futures = BondFutures(
        delivery_date=delivery_date,
        deliverable_basket=[DeliverableBond(bond)],
        futures_price=100.0,  # Placeholder
        contract_size=100_000,
    )

    valuation_date = datetime(2024, 12, 15)

    print(f"Underlying: {bond}")
    print(f"Delivery: {delivery_date.date()}")
    print(f"\nRate Sensitivity Comparison:")
    print("-" * 80)
    print(
        f"{'Rate':<10} {'Forward Clean':<15} {'Fwd DV01':<12} {'Futures':<15} {'Fut DV01':<12}"
    )
    print("-" * 80)

    for rate in [0.03, 0.04, 0.05, 0.06, 0.07]:
        rate_curve = FlatRateCurve(rate=rate)
        pricing_env = PricingEnvironment(
            rate_curve=rate_curve,
            valuation_date=valuation_date,
        )

        # Forward pricing
        fwd_engine = BondForwardEngine(pricing_env)
        fwd_results = fwd_engine.price(forward, valuation_date)
        fwd_greeks = fwd_engine.calculate_greeks(forward, valuation_date)

        # Futures pricing
        fut_engine = BondFuturesEngine(pricing_env, repo_rate=rate)
        fut_results = fut_engine.price(futures, valuation_date)

        print(
            f"{rate:.1%}       ${fwd_results.forward_clean_price:>10.4f}    "
            f"${fwd_greeks['dv01']:>8.6f}    "
            f"${fut_results.theoretical_futures_price:>10.4f}    "
            f"${fut_results.dv01:>8.6f}"
        )

    print("-" * 80)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("BOND FORWARD AND FUTURES PRICING DEMONSTRATION")
    print("=" * 80 + "\n")

    example_1_simple_bond_forward()
    example_2_forward_with_contracted_price()
    example_3_implied_repo_rate()
    example_4_simple_bond_futures()
    example_5_detailed_basis_analysis()
    example_6_conversion_factors()
    example_7_hedge_ratio()
    example_8_delivery_option_value()
    example_9_rate_sensitivity()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
