"""
Snowball Option Monte Carlo Pricing Demo.

This script demonstrates how to use the SnowballMCEngine to price
snowball (autocallable) options with different configurations:
- Standard vs reverse snowball structures
- Different Monte Carlo methods (PSEUDO, QUASI, RQMC)
- Time-varying barriers and rates
- INSTANT vs EXPIRY coupon payment timing
- disable_ko_after_ki behavior

Usage:
    python example/snowball_mc_demo.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
)
from asset.equity.param import MCParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import ObservationType, CouponPayType, ProtectionType
from util.enum.engine_enums import MonteCarloMethod, EngineType


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div_yield: float = 0.02,
) -> PricingEnvironment:
    """Create a basic pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def demo_basic_pricing():
    """Demonstrate basic snowball pricing."""
    print("=" * 60)
    print("Demo 1: Basic Snowball Pricing")
    print("=" * 60)

    env = create_pricing_env()

    # Create a standard snowball with monthly KO observations
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],  # Monthly
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        notional=1_000_000.0,
        maturity=1.0,
        is_reverse=False,
    )

    # Price with PSEUDO Monte Carlo
    engine = SnowballMCEngine(
        params=MCParams(num_paths=50000, seed=42),
        method=MonteCarloMethod.PSEUDO,
    )

    price = engine.price(snowball, env)
    result = engine.get_last_result()

    print(f"\nStandard Snowball (monthly KO, continuous KI):")
    print(f"  Initial Price: 100.0")
    print(f"  Strike: 100.0")
    print(f"  KO Barrier: 103.0 (UP)")
    print(f"  KI Barrier: 75.0 (DOWN)")
    print(f"  KO Rate: 15% annualized")
    print(f"  Maturity: 1 year")
    print(f"\nPricing Results:")
    print(f"  Price: {price:,.2f}")
    print(f"  Standard Error: {result.std_error:,.4f}")
    print(f"  KO Probability: {result.ko_probability:.2%}")
    print(f"  V0 Probability: {result.v0_probability:.2%}")
    print(f"  V1 Probability: {result.v1_probability:.2%}")
    if result.avg_ko_time is not None:
        print(f"  Avg KO Time: {result.avg_ko_time:.3f} years")


def demo_mc_methods():
    """Compare different Monte Carlo methods."""
    print("\n" + "=" * 60)
    print("Demo 2: Comparing Monte Carlo Methods")
    print("=" * 60)

    env = create_pricing_env()

    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],  # Quarterly
        ki_barrier=75.0,
        ki_continuous=True,
    )

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        notional=1_000_000.0,
        maturity=1.0,
    )

    print("\nComparing MC methods with 20,000 paths:")
    print("-" * 50)

    for method in [MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI]:
        engine = SnowballMCEngine(
            params=MCParams(num_paths=20000, seed=42),
            method=method,
        )
        price = engine.price(snowball, env)
        result = engine.get_last_result()
        print(f"{method.name:15s}: Price={price:12,.2f}, StdErr={result.std_error:8.2f}")

    # RQMC with smaller batch size
    engine = SnowballMCEngine(
        params=MCParams(num_paths=5000, seed=42),
        method=MonteCarloMethod.RANDOMIZED_QUASI,
    )
    price = engine.price(snowball, env)
    result = engine.get_last_result()
    print(f"{'RANDOMIZED_QUASI':15s}: Price={price:12,.2f}, StdErr={result.std_error:8.2f}, Batches={result.batches_used}")


def demo_convergence():
    """Demonstrate convergence behavior."""
    print("\n" + "=" * 60)
    print("Demo 3: Convergence Analysis")
    print("=" * 60)

    env = create_pricing_env()

    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_continuous=True,
    )

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        notional=1_000_000.0,
        maturity=1.0,
    )

    print("\nConvergence with increasing number of paths:")
    print("-" * 50)
    print(f"{'Num Paths':>12s} {'Price':>12s} {'StdErr':>10s} {'StdErr Ratio':>12s}")
    print("-" * 50)

    prev_std_err = None
    for num_paths in [1000, 5000, 10000, 50000, 100000]:
        engine = SnowballMCEngine(
            params=MCParams(num_paths=num_paths, seed=42),
            method=MonteCarloMethod.PSEUDO,
        )
        price = engine.price(snowball, env)
        result = engine.get_last_result()

        ratio_str = ""
        if prev_std_err is not None:
            ratio = prev_std_err / result.std_error
            ratio_str = f"{ratio:.2f}x"

        print(f"{num_paths:12,d} {price:12,.2f} {result.std_error:10.2f} {ratio_str:>12s}")
        prev_std_err = result.std_error

    print("\nNote: Standard error should decrease roughly as 1/sqrt(N)")


def demo_time_varying_barriers():
    """Demonstrate time-varying barriers and rates."""
    print("\n" + "=" * 60)
    print("Demo 4: Time-Varying Barriers and Rates")
    print("=" * 60)

    env = create_pricing_env()

    # Step-up barrier structure
    barrier_config = BarrierConfig(
        ko_barrier=[102.0, 103.0, 104.0, 105.0],  # Increasing barriers
        ko_rate=[0.12, 0.14, 0.16, 0.18],  # Increasing rates
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_continuous=True,
    )

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        notional=1_000_000.0,
        maturity=1.0,
    )

    engine = SnowballMCEngine(
        params=MCParams(num_paths=50000, seed=42),
        method=MonteCarloMethod.PSEUDO,
    )

    price = engine.price(snowball, env)
    result = engine.get_last_result()

    print("\nStep-Up Barrier Structure:")
    print("  Q1: KO=102.0, Rate=12%")
    print("  Q2: KO=103.0, Rate=14%")
    print("  Q3: KO=104.0, Rate=16%")
    print("  Q4: KO=105.0, Rate=18%")
    print(f"\nPrice: {price:,.2f}")
    print(f"KO Probability: {result.ko_probability:.2%}")


def demo_coupon_timing():
    """Compare INSTANT vs EXPIRY coupon payment."""
    print("\n" + "=" * 60)
    print("Demo 5: Coupon Payment Timing (INSTANT vs EXPIRY)")
    print("=" * 60)

    env = create_pricing_env()

    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_continuous=True,
    )

    print("\nComparing coupon payment timing:")
    print("-" * 50)

    for pay_type in [CouponPayType.INSTANT, CouponPayType.EXPIRY]:
        accrual_config = AccrualConfig(coupon_pay_type=pay_type)
        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            accrual_config=accrual_config,
            notional=1_000_000.0,
            maturity=1.0,
        )

        engine = SnowballMCEngine(
            params=MCParams(num_paths=50000, seed=42),
        )
        price = engine.price(snowball, env)
        result = engine.get_last_result()

        print(f"{pay_type.name:8s}: Price={price:12,.2f}")

    print("\nINSTANT: KO payoff discounted from observation date")
    print("EXPIRY:  KO payoff discounted from maturity")


def demo_disable_ko_after_ki():
    """Demonstrate disable_ko_after_ki behavior."""
    print("\n" + "=" * 60)
    print("Demo 6: disable_ko_after_ki Feature")
    print("=" * 60)

    env = create_pricing_env()

    # Use KI barrier close to spot for higher KI probability
    base_config = dict(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=95.0,  # Close to spot
        ki_continuous=True,
    )

    print("\nComparing disable_ko_after_ki settings:")
    print("-" * 50)

    for disable_ko in [False, True]:
        barrier_config = BarrierConfig(**base_config, disable_ko_after_ki=disable_ko)
        snowball = SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            notional=1_000_000.0,
            maturity=1.0,
        )

        engine = SnowballMCEngine(params=MCParams(num_paths=50000, seed=42))
        price = engine.price(snowball, env)
        result = engine.get_last_result()

        print(f"disable_ko_after_ki={str(disable_ko):5s}: "
              f"KO={result.ko_probability:5.2%}, "
              f"V0={result.v0_probability:5.2%}, "
              f"V1={result.v1_probability:5.2%}")

    print("\nWhen disable_ko_after_ki=True, KO events after KI are ignored")


def demo_reverse_snowball():
    """Demonstrate reverse snowball pricing."""
    print("\n" + "=" * 60)
    print("Demo 7: Reverse Snowball")
    print("=" * 60)

    env = create_pricing_env()

    # Reverse snowball: DOWN KO barrier, UP KI barrier
    barrier_config = BarrierConfig(
        ko_barrier=97.0,  # DOWN barrier
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=125.0,  # UP barrier
        ki_continuous=True,
    )

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        notional=1_000_000.0,
        maturity=1.0,
        is_reverse=True,
    )

    engine = SnowballMCEngine(
        params=MCParams(num_paths=50000, seed=42),
    )

    price = engine.price(snowball, env)
    result = engine.get_last_result()

    print(f"\nReverse Snowball:")
    print(f"  KO Barrier: 97.0 (DOWN)")
    print(f"  KI Barrier: 125.0 (UP)")
    print(f"\nPrice: {price:,.2f}")
    print(f"KO Probability: {result.ko_probability:.2%}")
    print(f"V0 Probability: {result.v0_probability:.2%}")
    print(f"V1 Probability: {result.v1_probability:.2%}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Snowball Option Monte Carlo Pricing Demo")
    print("=" * 60)

    demo_basic_pricing()
    demo_mc_methods()
    demo_convergence()
    demo_time_varying_barriers()
    demo_coupon_timing()
    demo_disable_ko_after_ki()
    demo_reverse_snowball()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
