"""
Phoenix Option Product Demo.

This script demonstrates the PhoenixOption product class and its features:
- Creating Phoenix options using helper functions
- Different variants: standard, reverse, step-down, memory/non-memory
- Coupon barrier triggering and payoff calculation
- Day count conventions (ACT_365, 30/360_US, 30/360_EUROPEAN)
- Coupon payment types (INSTANT vs EXPIRY)
- KO/KI barrier behavior

Note: This demo focuses on the product features. Engine pricing implementation
is not included in the current scope.

Usage:
    python example/phoenix_option_demo.py
"""

import sys
from pathlib import Path
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import (
    PhoenixOption,
    CouponBarrierConfig,
    create_standard_phoenix,
    create_stepdown_phoenix,
    create_reverse_phoenix,
    create_memory_phoenix,
    create_non_memory_phoenix,
)
from asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
)
from util.enum import ObservationType, CouponPayType, ProtectionType
from util.calendar.day_counter import DayCountConvention


def demo_basic_phoenix():
    """Demonstrate basic Phoenix option creation."""
    print("=" * 60)
    print("Demo 1: Basic Phoenix Option Creation")
    print("=" * 60)

    # Create a standard Phoenix using helper function
    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=12,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nStandard Phoenix (12 monthly observations):")
    print(f"  Initial Price: {phoenix.initial_price}")
    print(f"  Strike: {phoenix.strike}")
    print(f"  KO Barrier: {phoenix.barrier_config.ko_barrier} (UP, knocks out above)")
    print(f"  KI Barrier: {phoenix.barrier_config.ki_barrier} (DOWN, knocks in below)")
    print(f"  Coupon Barrier: {phoenix.coupon_config.coupon_barrier}")
    print(f"  KO Rate: {phoenix.barrier_config.ko_rate:.1%}")
    print(f"  Coupon Rate: {phoenix.coupon_config.coupon_rate:.1%} per period")
    print(f"  Memory Coupon: {phoenix.coupon_config.memory_coupon}")
    print(f"  Day Count Convention: {phoenix.coupon_config.day_count_convention.name}")
    print(f"  Coupon Pay Type: {phoenix.coupon_config.coupon_pay_type.name}")
    print(f"  Number of KO Observations: {phoenix.num_ko_observations}")
    print(f"  Maturity: {phoenix.maturity} years")
    print(f"  Notional: {phoenix.notional:,.0f}")
    print(f"  Is Reverse: {phoenix.is_reverse}")


def demo_coupon_triggering():
    """Demonstrate coupon barrier triggering logic."""
    print("\n" + "=" * 60)
    print("Demo 2: Coupon Barrier Triggering Logic")
    print("=" * 60)

    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nStandard Phoenix - Coupon triggers when spot >= coupon_barrier:")
    print(f"  Coupon Barrier: {phoenix.coupon_config.coupon_barrier}")
    print("-" * 50)

    test_spots = [90.0, 94.0, 95.0, 100.0, 105.0]
    for spot in test_spots:
        triggered = phoenix.is_coupon_triggered(spot, observation_idx=0)
        print(f"  Spot={spot:6.1f}: Coupon Triggered = {triggered}")

    # Reverse Phoenix
    reverse_phoenix = create_reverse_phoenix(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=97.0,
        ki_barrier=125.0,
        coupon_barrier=115.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nReverse Phoenix - Coupon triggers when spot <= coupon_barrier:")
    print(f"  Coupon Barrier: {reverse_phoenix.coupon_config.coupon_barrier:.1f}")
    print("-" * 50)

    test_spots_reverse = [120.0, 116.0, 115.0, 110.0, 105.0]
    for spot in test_spots_reverse:
        triggered = reverse_phoenix.is_coupon_triggered(spot, observation_idx=0)
        print(f"  Spot={spot:6.1f}: Coupon Triggered = {triggered}")


def demo_coupon_payoff():
    """Demonstrate coupon payoff calculation."""
    print("\n" + "=" * 60)
    print("Demo 3: Coupon Payoff Calculation")
    print("=" * 60)

    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.02,  # 2% per period
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nCoupon Payoff Calculation:")
    print(f"  Notional: {phoenix.notional:,.0f}")
    print(f"  Coupon Rate: {phoenix.coupon_config.coupon_rate:.1%} per period")
    print("-" * 50)

    # Basic coupon payoff (full period = 1.0 year fraction)
    basic_payoff = phoenix.get_coupon_payoff(observation_idx=0, year_fraction=1.0)
    print(f"\n  Full Year Fraction (1.0):")
    print(f"    Payoff = Notional x Rate x YearFraction")
    print(f"    Payoff = {phoenix.notional:,.0f} x {phoenix.coupon_config.coupon_rate} x 1.0")
    print(f"    Payoff = {basic_payoff:,.2f}")

    # Quarterly coupon
    quarterly_payoff = phoenix.get_coupon_payoff(observation_idx=0, year_fraction=0.25)
    print(f"\n  Quarterly (0.25 year fraction):")
    print(f"    Payoff = {phoenix.notional:,.0f} x {phoenix.coupon_config.coupon_rate} x 0.25")
    print(f"    Payoff = {quarterly_payoff:,.2f}")

    # Coupon payoff with date calculation
    print("\n  With Date-Based Year Fraction Calculation:")
    start = datetime(2024, 1, 1)
    end = datetime(2024, 4, 1)  # 91 days
    yf = phoenix.get_coupon_year_fraction(start, end)
    payoff_with_dates = phoenix.get_coupon_payoff(observation_idx=0, start_date=start, end_date=end)
    print(f"    Period: {start} to {end}")
    print(f"    Year Fraction (ACT_365): {yf:.6f}")
    print(f"    Payoff = {payoff_with_dates:,.2f}")


def demo_day_count_conventions():
    """Demonstrate different day count conventions."""
    print("\n" + "=" * 60)
    print("Demo 4: Day Count Conventions")
    print("=" * 60)

    conventions = [
        DayCountConvention.ACT_365,
        DayCountConvention.THIRTY_360_US,
        DayCountConvention.THIRTY_360_EUROPEAN,
    ]

    # Test period: Jan 1 to Apr 1 (91 actual days)
    start = datetime(2024, 1, 1)
    end = datetime(2024, 4, 1)

    print(f"\nPeriod: {start} to {end}")
    print(f"Actual days: 91")
    print("-" * 50)

    for convention in conventions:
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.15,
            coupon_rate=0.01,
            num_observations=4,
            maturity=1.0,
            notional=1_000_000.0,
            day_count_convention=convention,
        )

        yf = phoenix.get_coupon_year_fraction(start, end)
        payoff = phoenix.get_coupon_payoff(observation_idx=0, start_date=start, end_date=end)

        print(f"\n  {convention.name}:")
        print(f"    Year Fraction: {yf:.6f}")
        print(f"    Coupon Payoff: {payoff:,.2f}")

    # Special case: Feb end of month
    print("\n  Special Case: Jan 31 to Feb 28 (28 days)")
    start2 = datetime(2024, 1, 31)
    end2 = datetime(2024, 2, 28)

    for convention in conventions:
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.15,
            coupon_rate=0.01,
            num_observations=4,
            maturity=1.0,
            notional=1_000_000.0,
            day_count_convention=convention,
        )

        yf = phoenix.get_coupon_year_fraction(start2, end2)
        print(f"    {convention.name}: Year Fraction = {yf:.6f}")


def demo_phoenix_variants():
    """Demonstrate different Phoenix variants."""
    print("\n" + "=" * 60)
    print("Demo 5: Phoenix Variants")
    print("=" * 60)

    base_params = dict(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    # Standard Phoenix
    standard = create_standard_phoenix(**base_params)
    print("\n1. Standard Phoenix (Memory Coupon Enabled):")
    print(f"   Memory Coupon: {standard.coupon_config.memory_coupon}")
    print(f"   KO Direction: UP (knocks out above barrier)")
    print(f"   KI Direction: DOWN (knocks in below barrier)")

    # Memory vs Non-Memory Phoenix
    memory = create_memory_phoenix(**base_params)
    non_memory = create_non_memory_phoenix(**base_params)
    print("\n2. Memory vs Non-Memory Phoenix:")
    print(f"   Memory Phoenix - memory_coupon: {memory.coupon_config.memory_coupon}")
    print(f"   Non-Memory Phoenix - memory_coupon: {non_memory.coupon_config.memory_coupon}")
    print("   Memory: Missed coupons accumulate and are paid when barrier is hit later")
    print("   Non-Memory: Only current period coupon is paid")

    # Reverse Phoenix
    reverse_params = base_params.copy()
    reverse_params["ko_barrier"] = 97.0
    reverse_params["ki_barrier"] = 125.0
    reverse_params["coupon_barrier"] = 115.0
    reverse = create_reverse_phoenix(**reverse_params)
    print("\n3. Reverse Phoenix:")
    print(f"   Is Reverse: {reverse.is_reverse}")
    print(f"   KO Direction: DOWN (knocks out below barrier)")
    print(f"   KI Direction: UP (knocks in above barrier)")
    print(f"   Coupon triggers when spot <= coupon_barrier")


def demo_stepdown_phoenix():
    """Demonstrate step-down Phoenix with time-varying barriers."""
    print("\n" + "=" * 60)
    print("Demo 6: Step-Down Phoenix (Time-Varying Barriers)")
    print("=" * 60)

    # Step-down barriers decrease over time
    # The helper function generates barriers automatically from initial value and stepdown rate
    stepdown = create_stepdown_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=103.0,  # Starting KO barrier
        initial_coupon_barrier=95.0,  # Starting coupon barrier
        ko_stepdown_rate=0.01,  # Decrease by 1% of initial_price each period
        coupon_stepdown_rate=0.02,  # Decrease by 2% of initial_price each period
        ko_rate=0.15,
        ki_barrier=75.0,
        coupon_rate=0.01,
        num_observations=4,
    )

    print("\nStep-Down Structure (barriers decrease over time):")
    print("-" * 50)
    print(f"{'Quarter':>8s} {'KO Barrier':>12s} {'Coupon Barrier':>15s}")
    print("-" * 50)

    for i in range(4):
        ko_b = stepdown.barrier_config.ko_barrier[i]
        coupon_b = stepdown.coupon_config.coupon_barrier[i]
        print(f"{f'Q{i+1}':>8s} {ko_b:>12.1f} {coupon_b:>15.1f}")

    # Test time-varying coupon barrier
    print("\nCoupon Triggering with Time-Varying Barriers:")
    spot = 92.0
    for i in range(4):
        triggered = stepdown.is_coupon_triggered(spot, observation_idx=i)
        barrier = stepdown.get_coupon_barrier_at(i)
        print(f"  Q{i+1}: Spot={spot}, Barrier={barrier}, Triggered={triggered}")


def demo_ko_ki_triggering():
    """Demonstrate KO and KI barrier triggering."""
    print("\n" + "=" * 60)
    print("Demo 7: KO and KI Barrier Triggering")
    print("=" * 60)

    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nStandard Phoenix KO/KI Triggering:")
    print(f"  KO Barrier: {phoenix.barrier_config.ko_barrier} (UP)")
    print(f"  KI Barrier: {phoenix.barrier_config.ki_barrier} (DOWN)")
    print("-" * 50)

    test_spots = [70.0, 75.0, 80.0, 100.0, 103.0, 110.0]
    print(f"\n{'Spot':>8s} {'KO Triggered':>15s} {'KI Triggered':>15s}")
    print("-" * 40)

    for spot in test_spots:
        ko = phoenix.is_ko_triggered(spot, observation_idx=0)
        ki = phoenix.is_ki_triggered(spot)
        print(f"{spot:>8.1f} {str(ko):>15s} {str(ki):>15s}")

    print("\nKO: Triggers when spot >= KO barrier (early termination with bonus)")
    print("KI: Triggers when spot <= KI barrier (activates downside risk)")


def demo_payoff_calculations():
    """Demonstrate payoff calculations."""
    print("\n" + "=" * 60)
    print("Demo 8: Payoff Calculations")
    print("=" * 60)

    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nPhoenix Payoff Calculations:")
    print(f"  Notional: {phoenix.notional:,.0f}")
    print(f"  KO Rate: {phoenix.barrier_config.ko_rate:.1%}")
    print("-" * 50)

    # KO Payoff at different observation dates
    print("\n1. KO Payoff (early termination):")
    for obs_idx in range(4):
        ko_payoff = phoenix.get_ko_payoff(spot=105.0, observation_idx=obs_idx)
        obs_time = (obs_idx + 1) * 0.25
        print(f"   Quarter {obs_idx + 1} (t={obs_time}): KO Payoff = {ko_payoff:,.2f}")

    # Maturity payoff V0 (no KI)
    print("\n2. Maturity Payoff V0 (no knock-in occurred):")
    v0_payoff = phoenix.get_maturity_payoff_v0(spot=100.0)
    print(f"   V0 Payoff = Notional x Rebate Rate")
    print(f"   V0 Payoff = {phoenix.notional:,.0f} x {phoenix.payoff_config.rebate_rate}")
    print(f"   V0 Payoff = {v0_payoff:,.2f}")

    # Maturity payoff V1 (KI occurred)
    print("\n3. Maturity Payoff V1 (knock-in occurred, loss at maturity):")
    test_final_spots = [60.0, 75.0, 90.0, 100.0, 110.0]
    for spot in test_final_spots:
        v1_payoff = phoenix.get_maturity_payoff_v1(spot)
        print(f"   Final Spot = {spot:6.1f}: V1 Payoff = {v1_payoff:,.2f}")


def demo_coupon_pay_types():
    """Demonstrate INSTANT vs EXPIRY coupon payment timing."""
    print("\n" + "=" * 60)
    print("Demo 9: Coupon Payment Timing (INSTANT vs EXPIRY)")
    print("=" * 60)

    base_params = dict(
        initial_price=100.0,
        strike=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.15,
        coupon_rate=0.01,
        num_observations=4,
        maturity=1.0,
        notional=1_000_000.0,
    )

    print("\nCoupon Payment Timing Options:")
    print("-" * 50)

    for pay_type in [CouponPayType.INSTANT, CouponPayType.EXPIRY]:
        phoenix = create_standard_phoenix(**base_params, coupon_pay_type=pay_type)
        print(f"\n  {pay_type.name}:")
        print(f"    Coupon Pay Type: {phoenix.coupon_config.coupon_pay_type.name}")

        if pay_type == CouponPayType.INSTANT:
            print("    Behavior: Coupon paid immediately when barrier is hit")
            print("    Discounting: From observation date to valuation date")
        else:
            print("    Behavior: Coupons accumulated and paid at maturity")
            print("    Discounting: All coupons from maturity to valuation date")


def demo_phoenix_vs_snowball():
    """Compare Phoenix and Snowball option structures."""
    print("\n" + "=" * 60)
    print("Demo 10: Phoenix vs Snowball Comparison")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("Key Differences:")
    print("-" * 60)

    comparison = """
    | Feature         | Snowball                | Phoenix                    |
    |-----------------|-------------------------|----------------------------|
    | Coupon Payment  | Only on KO trigger      | Periodic when barrier hit  |
    | Coupon Barrier  | No (tied to KO)         | Yes (separate barrier)     |
    | KO Barrier      | Yes                     | Yes                        |
    | KI Barrier      | Optional                | Optional                   |
    | Memory Coupon   | No                      | Yes (accumulates missed)   |
    | Day Count Conv  | No (rate-based)         | Yes (year fraction calc)   |
    | Coupon Timing   | INSTANT/EXPIRY          | INSTANT/EXPIRY             |
    """
    print(comparison)

    print("Phoenix Advantage:")
    print("  - Periodic income even without knock-out")
    print("  - Memory coupon feature recovers missed coupons")
    print("  - Flexible day count conventions for coupon calculation")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Phoenix Option Product Demo")
    print("=" * 60)

    demo_basic_phoenix()
    demo_coupon_triggering()
    demo_coupon_payoff()
    demo_day_count_conventions()
    demo_phoenix_variants()
    demo_stepdown_phoenix()
    demo_ko_ki_triggering()
    demo_payoff_calculations()
    demo_coupon_pay_types()
    demo_phoenix_vs_snowball()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
