"""
Asian Option Product Demo.

This script demonstrates the creation and payoff calculation of Asian options.
Asian options are path-dependent options where the payoff depends on the average
price of the underlying asset over a specified observation period.

Types demonstrated:
1. Fixed strike (average price option) - payoff based on average vs strike
2. Floating strike (average strike option) - payoff based on final spot vs average
3. Arithmetic vs geometric averaging
"""

import numpy as np
from asset.equity.product.option import AsianOption
from util.enum import OptionType, AveragingType, AsianStrikeType


def demonstrate_fixed_strike_options():
    """Demonstrate fixed strike (average price) Asian options."""
    print("=" * 60)
    print("FIXED STRIKE (AVERAGE PRICE) ASIAN OPTIONS")
    print("=" * 60)
    print()

    # Create a fixed strike Asian call
    asian_call = AsianOption(
        strike=100,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
        num_observations=12,
    )
    print(f"Created: {asian_call}")
    print()

    # Simulate price path (monthly observations)
    price_path = [100, 102, 105, 103, 108, 110, 107, 112, 115, 113, 118, 120]
    avg_price = asian_call.get_average(price_path)

    print(f"Price path (monthly): {price_path}")
    print(f"Arithmetic average: {avg_price:.2f}")
    print()

    # Calculate payoff
    final_spot = 120
    payoff = asian_call.get_payoff(spot=final_spot, observed_prices=price_path)
    print(f"Final spot: {final_spot}")
    print(f"Call payoff: max({avg_price:.2f} - {asian_call.strike}, 0) = {payoff:.2f}")
    print()

    # Create a fixed strike Asian put
    asian_put = AsianOption(
        strike=110,
        option_type=OptionType.PUT,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
    )
    put_payoff = asian_put.get_payoff(spot=final_spot, observed_prices=price_path)
    print(f"Put (K=110) payoff: max({asian_put.strike} - {avg_price:.2f}, 0) = {put_payoff:.2f}")
    print()


def demonstrate_floating_strike_options():
    """Demonstrate floating strike (average strike) Asian options."""
    print("=" * 60)
    print("FLOATING STRIKE (AVERAGE STRIKE) ASIAN OPTIONS")
    print("=" * 60)
    print()

    # Create a floating strike Asian call
    floating_call = AsianOption(
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FLOATING,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=0.5,
    )
    print(f"Created: {floating_call}")
    print()

    # Uptrending price path
    uptrend_path = [100, 105, 110, 115, 120, 125]
    avg_uptrend = floating_call.get_average(uptrend_path)
    final_up = 130

    print("Scenario 1: Uptrending market")
    print(f"Price path: {uptrend_path}")
    print(f"Average: {avg_uptrend:.2f}, Final spot: {final_up}")
    call_payoff = floating_call.get_payoff(spot=final_up, observed_prices=uptrend_path)
    print(f"Call payoff: max({final_up} - {avg_uptrend:.2f}, 0) = {call_payoff:.2f}")
    print()

    # Downtrending price path
    downtrend_path = [100, 95, 90, 85, 80, 75]
    avg_downtrend = floating_call.get_average(downtrend_path)
    final_down = 70

    # Floating strike put
    floating_put = AsianOption(
        option_type=OptionType.PUT,
        asian_strike_type=AsianStrikeType.FLOATING,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=0.5,
    )

    print("Scenario 2: Downtrending market")
    print(f"Price path: {downtrend_path}")
    print(f"Average: {avg_downtrend:.2f}, Final spot: {final_down}")
    put_payoff = floating_put.get_payoff(spot=final_down, observed_prices=downtrend_path)
    print(f"Put payoff: max({avg_downtrend:.2f} - {final_down}, 0) = {put_payoff:.2f}")
    print()


def demonstrate_averaging_types():
    """Compare arithmetic vs geometric averaging."""
    print("=" * 60)
    print("ARITHMETIC VS GEOMETRIC AVERAGING")
    print("=" * 60)
    print()

    # Create options with different averaging types
    arithmetic_option = AsianOption(
        strike=100,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
    )

    geometric_option = AsianOption(
        strike=100,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.GEOMETRIC,
        maturity=1.0,
    )

    # Test with volatile price path
    volatile_path = [100, 120, 80, 130, 70, 140]
    print(f"Volatile price path: {volatile_path}")
    print()

    arith_avg = arithmetic_option.get_average(volatile_path)
    geom_avg = geometric_option.get_average(volatile_path)

    print(f"Arithmetic average: {arith_avg:.4f}")
    print(f"Geometric average:  {geom_avg:.4f}")
    print()

    # Payoff comparison
    final_spot = 110
    arith_payoff = arithmetic_option.get_payoff(spot=final_spot, observed_prices=volatile_path)
    geom_payoff = geometric_option.get_payoff(spot=final_spot, observed_prices=volatile_path)

    print(f"Final spot: {final_spot}")
    print(f"Arithmetic call payoff: {arith_payoff:.4f}")
    print(f"Geometric call payoff:  {geom_payoff:.4f}")
    print()

    # Note: Geometric average is always <= arithmetic average (AM-GM inequality)
    print("Note: Geometric average <= Arithmetic average (AM-GM inequality)")
    print(f"Difference: {arith_avg - geom_avg:.4f}")
    print()


def demonstrate_observation_schedule():
    """Demonstrate observation schedule configuration."""
    print("=" * 60)
    print("OBSERVATION SCHEDULE CONFIGURATION")
    print("=" * 60)
    print()

    # Default uniform schedule
    default_option = AsianOption(
        strike=100,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=4,  # Quarterly observations
    )
    default_times = default_option.get_observation_times()
    print(f"Default quarterly observations: {[f'{t:.3f}' for t in default_times]}")
    print()

    # Custom observation times
    custom_times = [0.1, 0.2, 0.5, 0.8, 1.0]  # Front-weighted observations
    custom_option = AsianOption(
        strike=100,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        observation_times=custom_times,
    )
    print(f"Custom observation times: {custom_option.get_observation_times()}")
    print()


def main():
    """Run all demonstrations."""
    print()
    print("*" * 60)
    print("      ASIAN OPTION PRODUCT DEMONSTRATION")
    print("*" * 60)
    print()

    demonstrate_fixed_strike_options()
    demonstrate_floating_strike_options()
    demonstrate_averaging_types()
    demonstrate_observation_schedule()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
