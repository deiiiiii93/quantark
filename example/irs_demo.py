"""
Interest Rate Swap (IRS) Demo.

This script demonstrates the IRS module functionality including:
1. Vanilla IRS creation and pricing
2. Basis swap creation and pricing
3. Amortizing swap example
4. SOFR compounding swap
5. Risk metrics calculation
6. Par rate calculation

Run this script to see the IRS module in action.
"""

from datetime import datetime
import sys

from quantark.asset.rate.product.irs import (
    InterestRateSwap,
    BasisSwap,
    FixedLeg,
    FloatingLeg,
    NotionalSchedule,
    SwapDirection,
    create_vanilla_irs,
    create_basis_swap,
    create_amortizing_irs,
    create_compounding_irs,
)
from quantark.asset.rate.engine.irs_discount_engine import (
    IRSDiscountEngine,
    IRSPricingResults,
    BasisSwapPricingResults,
)
from quantark.asset.bond.schedule.cashflow import CompoundingMethod
from quantark.param.index import SOFR, SOFR_3M, EURIBOR_3M, SHIBOR_3M
from quantark.param.rrf import FlatRateCurve
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import PaymentFrequency


def print_separator(title: str):
    """Print a section separator."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_vanilla_irs():
    """Demonstrate vanilla IRS creation and pricing."""
    print_separator("1. VANILLA INTEREST RATE SWAP")
    
    # Market setup
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)  # 4.5% flat curve
    
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    # Create a 5-year payer swap (pay fixed, receive floating)
    print("\nCreating 5-year USD IRS:")
    print("  - Pay fixed 4.50%")
    print("  - Receive SOFR_3M + 0bp")
    print("  - Notional: $10,000,000")
    
    irs = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=0.045,  # 4.50%
        index=SOFR_3M,
        spread=0.0,
        direction=SwapDirection.PAYER,
        payment_frequency=PaymentFrequency.QUARTERLY,
    )
    
    print(f"\nSwap: {irs}")
    print(f"Effective Date: {irs.get_start_date().date()}")
    print(f"Maturity Date: {irs.get_end_date().date()}")
    print(f"Time to Maturity: {irs.time_to_maturity(valuation_date):.2f} years")
    
    # Pricing
    print("\n--- Pricing Results ---")
    npv = engine.npv(irs)
    par_rate = engine.par_rate(irs)
    
    print(f"NPV: ${npv:,.2f}")
    print(f"Par Rate: {par_rate:.4%}")
    print(f"Fixed Rate vs Par: {(irs.get_fixed_rate() - par_rate) * 10000:.1f} bps")
    
    # Full analysis
    results = engine.full_analysis(irs)
    print(f"\n--- Risk Metrics ---")
    print(f"DV01: ${results.dv01:,.2f}")
    print(f"Duration: {results.duration:.4f}")
    print(f"WAL: {results.weighted_average_life:.2f} years")
    
    # Cashflow schedule
    print(f"\n--- Cashflow Summary ---")
    fixed_cfs = irs.fixed_leg.get_fixed_cashflows()
    float_cfs = irs.floating_leg.get_floating_cashflows()
    
    print(f"Fixed leg: {len(fixed_cfs)} cashflows")
    print(f"Floating leg: {len(float_cfs)} cashflows")
    
    print("\nFirst 4 Fixed Leg Cashflows:")
    for cf in fixed_cfs[:4]:
        print(f"  {cf.payment_date.date()}: ${cf.amount:,.2f} "
              f"(rate={cf.fixed_rate:.4%}, dcf={cf.day_count_fraction:.4f})")
    
    print("\nFirst 4 Floating Leg Cashflows:")
    for cf in float_cfs[:4]:
        status = "projected" if cf.is_projected else "fixed"
        print(f"  {cf.payment_date.date()}: rate={cf.effective_rate:.4%} ({status})")
    
    return irs, engine


def demo_receiver_swap():
    """Demonstrate receiver swap (opposite direction)."""
    print_separator("2. RECEIVER SWAP (Receive Fixed)")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    # Receiver swap - receive fixed, pay floating
    print("\nCreating 5-year receiver swap:")
    print("  - Receive fixed 4.50%")
    print("  - Pay SOFR_3M + 0bp")
    
    receiver_swap = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=0.045,
        index=SOFR_3M,
        spread=0.0,
        direction=SwapDirection.RECEIVER,  # Receive fixed
    )
    
    print(f"\nSwap: {receiver_swap}")
    
    npv = engine.npv(receiver_swap)
    print(f"\nNPV: ${npv:,.2f}")
    print("(Note: NPV has opposite sign compared to payer swap)")
    
    return receiver_swap


def demo_basis_swap():
    """Demonstrate basis swap (floating vs floating)."""
    print_separator("3. BASIS SWAP (SOFR vs SOFR_3M)")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    print("\nCreating 5-year basis swap:")
    print("  - Pay SOFR (overnight)")
    print("  - Receive SOFR_3M + 10bp")
    
    basis_swap = create_basis_swap(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        index1=SOFR,       # Pay leg
        index2=SOFR_3M,    # Receive leg
        spread1=0.0,
        spread2=0.001,     # 10bp spread on receive leg
        payment_frequency1=PaymentFrequency.QUARTERLY,
        payment_frequency2=PaymentFrequency.QUARTERLY,
    )
    
    print(f"\nSwap: {basis_swap}")
    print(f"Basis Spread: {basis_swap.get_basis_spread():.4%}")
    
    npv = engine.npv(basis_swap)
    results = engine.full_basis_swap_analysis(basis_swap)
    
    print(f"\n--- Pricing Results ---")
    print(f"NPV: ${npv:,.2f}")
    print(f"Leg1 PV (Pay): ${results.leg1_pv:,.2f}")
    print(f"Leg2 PV (Receive): ${results.leg2_pv:,.2f}")
    print(f"DV01: ${results.dv01:,.2f}")
    
    return basis_swap


def demo_amortizing_swap():
    """Demonstrate amortizing denominator swap."""
    print_separator("4. AMORTIZING SWAP")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    print("\nCreating 5-year amortizing swap:")
    print("  - Initial denominator: $10,000,000")
    print("  - Annual amortization: $2,000,000")
    print("  - Final denominator: $2,000,000")
    
    amort_schedule = [
        (datetime(2025, 1, 15), 8_000_000),
        (datetime(2026, 1, 15), 6_000_000),
        (datetime(2027, 1, 15), 4_000_000),
        (datetime(2028, 1, 15), 2_000_000),
    ]
    
    amort_swap = create_amortizing_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        initial_notional=10_000_000,
        fixed_rate=0.045,
        index=SOFR_3M,
        amortization_schedule=amort_schedule,
        spread=0.0,
        direction=SwapDirection.PAYER,
    )
    
    print(f"\nSwap: {amort_swap}")
    
    # Show denominator schedule
    print("\nDenominator Schedule:")
    for date, amount in amort_schedule:
        print(f"  {date.date()}: ${amount:,.0f}")
    
    # Price it
    npv = engine.npv(amort_swap)
    results = engine.full_analysis(amort_swap)
    
    print(f"\n--- Pricing Results ---")
    print(f"NPV: ${npv:,.2f}")
    print(f"Par Rate: {results.par_rate:.4%}")
    print(f"WAL: {results.weighted_average_life:.2f} years")
    print("(Note: WAL is shorter due to amortization)")
    
    return amort_swap


def demo_compounding_swap():
    """Demonstrate SOFR compounding swap."""
    print_separator("5. SOFR COMPOUNDING SWAP")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    print("\nCreating 5-year SOFR compounding swap:")
    print("  - Pay fixed 4.50%")
    print("  - Receive compounded SOFR + 5bp")
    print("  - Compounding: Spread Exclusive (ISDA standard)")
    print("  - Lookback: 2 business days")
    
    sofr_swap = create_compounding_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=0.045,
        index=SOFR,  # Overnight SOFR
        spread=0.0005,  # 5bp
        direction=SwapDirection.PAYER,
        payment_frequency=PaymentFrequency.QUARTERLY,
        compounding_method=CompoundingMethod.SPREAD_EXCLUSIVE,
        lookback_days=2,
    )
    
    print(f"\nSwap: {sofr_swap}")
    print(f"Index: {sofr_swap.floating_leg.index}")
    print(f"Compounding Method: {sofr_swap.floating_leg.compounding_method}")
    
    npv = engine.npv(sofr_swap)
    results = engine.full_analysis(sofr_swap)
    
    print(f"\n--- Pricing Results ---")
    print(f"NPV: ${npv:,.2f}")
    print(f"Par Rate: {results.par_rate:.4%}")
    print(f"DV01: ${results.dv01:,.2f}")
    
    return sofr_swap


def demo_par_rate_calculation():
    """Demonstrate par rate calculation and at-market swap."""
    print_separator("6. PAR RATE AND AT-MARKET SWAP")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    # First, create a swap with arbitrary rate to find par rate
    test_swap = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=0.04,  # Arbitrary rate
        index=SOFR_3M,
        spread=0.0,
        direction=SwapDirection.PAYER,
    )
    
    par_rate = engine.par_rate(test_swap)
    print(f"\nCalculated Par Rate: {par_rate:.4%}")
    
    # Now create at-market swap with par rate
    print(f"\nCreating at-market swap with fixed rate = {par_rate:.4%}")
    
    at_market_swap = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=par_rate,
        index=SOFR_3M,
        spread=0.0,
        direction=SwapDirection.PAYER,
    )
    
    npv = engine.npv(at_market_swap)
    print(f"\nAt-market swap NPV: ${npv:,.2f}")
    print("(NPV should be approximately zero)")
    
    return at_market_swap


def demo_risk_metrics():
    """Demonstrate risk metric calculations."""
    print_separator("7. RISK METRICS ANALYSIS")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    # Create a swap for analysis
    swap = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=0.045,
        index=SOFR_3M,
        spread=0.0,
        direction=SwapDirection.PAYER,
    )
    
    print(f"\nSwap: {swap}")
    
    # Full analysis
    results = engine.full_analysis(swap)
    
    print("\n--- Valuation ---")
    print(f"NPV: ${results.npv:,.2f}")
    print(f"Receive Leg PV: ${results.receive_leg_pv:,.2f}")
    print(f"Pay Leg PV: ${results.pay_leg_pv:,.2f}")
    
    print("\n--- Par Rate ---")
    print(f"Par Rate: {results.par_rate:.4%}")
    print(f"Current Fixed Rate: {swap.get_fixed_rate():.4%}")
    
    print("\n--- Interest Rate Risk ---")
    print(f"DV01: ${results.dv01:,.2f}")
    print(f"Duration: {results.duration:.4f}")
    print(f"Convexity: {results.convexity:,.4f}")
    
    print("\n--- Other Metrics ---")
    print(f"WAL: {results.weighted_average_life:.2f} years")
    print(f"Net Accrued: ${results.net_accrued:,.2f}")
    
    return results


def demo_different_currencies():
    """Demonstrate swaps with different currency indices."""
    print_separator("8. MULTI-CURRENCY INDICES")
    
    valuation_date = datetime(2024, 1, 15)
    rate_curve = FlatRateCurve(rate=0.045)
    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date
    )
    engine = IRSDiscountEngine(pricing_env)
    
    # EUR swap with EURIBOR
    print("\n--- EUR IRS (EURIBOR 3M) ---")
    eur_swap = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=10_000_000,
        fixed_rate=0.035,  # 3.50%
        index=EURIBOR_3M,
        spread=0.0,
        direction=SwapDirection.PAYER,
    )
    print(f"Swap: {eur_swap}")
    print(f"Index: {eur_swap.floating_leg.index}")
    print(f"NPV: EUR {engine.npv(eur_swap):,.2f}")
    
    # CNY swap with SHIBOR
    print("\n--- CNY IRS (SHIBOR 3M) ---")
    cny_swap = create_vanilla_irs(
        effective_date=datetime(2024, 1, 15),
        maturity_date=datetime(2029, 1, 15),
        denominator=70_000_000,  # ~$10M equivalent
        fixed_rate=0.025,  # 2.50%
        index=SHIBOR_3M,
        spread=0.0,
        direction=SwapDirection.PAYER,
    )
    print(f"Swap: {cny_swap}")
    print(f"Index: {cny_swap.floating_leg.index}")
    print(f"NPV: CNY {engine.npv(cny_swap):,.2f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print(" INTEREST RATE SWAP MODULE DEMO")
    print("=" * 60)
    
    # Run all demonstrations
    demo_vanilla_irs()
    demo_receiver_swap()
    demo_basis_swap()
    demo_amortizing_swap()
    demo_compounding_swap()
    demo_par_rate_calculation()
    demo_risk_metrics()
    demo_different_currencies()
    
    print_separator("DEMO COMPLETE")
    print("\nAll demonstrations completed successfully!")
    print("The IRS module supports:")
    print("  - Vanilla IRS (fixed vs floating)")
    print("  - Basis Swaps (floating vs floating)")
    print("  - Amortizing denominator schedules")
    print("  - SOFR overnight compounding")
    print("  - Multiple rate indices (SOFR, EURIBOR, SHIBOR, etc.)")
    print("  - Full risk metrics (NPV, DV01, Duration, Par Rate)")


if __name__ == "__main__":
    main()
