#!/usr/bin/env python
"""
Airbag Snowball Option Pricing Demo

This example demonstrates how airbag snowball options work and how to price
them using the Monte Carlo engine.

Key concepts demonstrated:
1. Airbag structure: reduced participation rate when spot falls below airbag barrier
2. Comparison between standard and airbag snowball payoffs
3. Monte Carlo pricing with different airbag configurations
4. Impact of airbag parameters on option value

Airbag Snowball Structure:
- Standard snowball: 100% participation in downside when KI triggered
- Airbag snowball: reduced participation (e.g., 50%) when spot < airbag_barrier
- This provides additional protection in extreme market downturns

Usage:
    python example/airbag_snowball_demo.py
"""

from datetime import datetime

from quantark.asset.equity.product.option import (
    create_standard_snowball,
    create_airbag_snowball,
)
from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import MonteCarloMethod


def create_pricing_environment(
    spot: float = 100.0,
    volatility: float = 0.30,
    rate: float = 0.05,
) -> PricingEnvironment:
    """Create a standard pricing environment."""
    return PricingEnvironment(
        valuation_date=datetime(2024, 1, 1),
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=volatility),
        rate_curve=FlatRateCurve(rate=rate),
    )


def demo_payoff_comparison():
    """Compare V1 payoffs between standard and airbag snowballs."""
    print("=" * 70)
    print("AIRBAG VS STANDARD SNOWBALL: PAYOFF COMPARISON")
    print("=" * 70)

    # Create standard snowball (100% participation)
    standard = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        participation_rate=1.0,
        include_principal=True,
    )

    # Create airbag snowball (50% participation below airbag barrier)
    airbag = create_airbag_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,
        include_principal=True,
    )

    print("\nSnowball Parameters:")
    print(f"  Initial Price: 100.0")
    print(f"  Strike: 100.0")
    print(f"  KI Barrier: 75.0")
    print(f"  Notional: 1,000,000")
    print(f"  Airbag Barrier: 60.0 (airbag snowball only)")
    print(f"  Standard Participation: 100%")
    print(f"  Airbag Participation: 50%")

    print("\nV1 Payoff Comparison (KI triggered, no KO):")
    print("-" * 70)
    print(f"{'Spot':>10} {'Zone':>12} {'Standard':>15} {'Airbag':>15} {'Savings':>12}")
    print("-" * 70)

    spot_levels = [40, 50, 55, 60, 65, 70, 80, 90, 100]
    for spot in spot_levels:
        standard_payoff = standard.get_maturity_payoff_v1(float(spot))
        airbag_payoff = airbag.get_maturity_payoff_v1(float(spot))
        savings = airbag_payoff - standard_payoff

        zone = "Airbag" if spot < 60 else "Standard"
        print(
            f"{spot:>10} {zone:>12} {standard_payoff:>15,.0f} "
            f"{airbag_payoff:>15,.0f} {savings:>12,.0f}"
        )

    print("-" * 70)
    print("\nKey Insight: Airbag provides protection when spot falls below 60")
    print("             At spot=40, airbag saves 300,000 vs standard snowball")


def demo_mc_pricing():
    """Price airbag snowball with Monte Carlo engine."""
    print("\n" + "=" * 70)
    print("MONTE CARLO PRICING: AIRBAG VS STANDARD SNOWBALL")
    print("=" * 70)

    pricing_env = create_pricing_environment(spot=100.0, volatility=0.30)

    # Create snowballs
    standard = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        participation_rate=1.0,
        include_principal=True,
    )

    airbag = create_airbag_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,
        include_principal=True,
    )

    print("\nPricing with Monte Carlo (100,000 paths, Quasi-MC)...")
    engine = SnowballMCEngine(
        params=MCParams(num_paths=100000, seed=42),
        method=MonteCarloMethod.QUASI,
    )

    # Price standard snowball
    standard_price = engine.price(standard, pricing_env)
    standard_result = engine.get_last_result()

    # Price airbag snowball
    airbag_price = engine.price(airbag, pricing_env)
    airbag_result = engine.get_last_result()

    print("\nResults:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Standard':>20} {'Airbag':>20}")
    print("-" * 70)
    print(f"{'Price'::<25} {standard_price:>20,.2f} {airbag_price:>20,.2f}")
    print(f"{'Std Error'::<25} {standard_result.std_error:>20,.2f} {airbag_result.std_error:>20,.2f}")
    print(f"{'KO Probability'::<25} {standard_result.ko_probability:>19.2%} {airbag_result.ko_probability:>19.2%}")
    print(f"{'V0 Probability'::<25} {standard_result.v0_probability:>19.2%} {airbag_result.v0_probability:>19.2%}")
    print(f"{'V1 Probability'::<25} {standard_result.v1_probability:>19.2%} {airbag_result.v1_probability:>19.2%}")
    print("-" * 70)

    price_diff = airbag_price - standard_price
    print(f"\nAirbag Premium: {price_diff:,.2f} ({price_diff/standard_price*100:.2f}%)")
    print("\nNote: Airbag snowball is more valuable because it reduces V1 losses")


def demo_airbag_sensitivity():
    """Show sensitivity to airbag parameters."""
    print("\n" + "=" * 70)
    print("AIRBAG PARAMETER SENSITIVITY")
    print("=" * 70)

    pricing_env = create_pricing_environment(spot=100.0, volatility=0.30)
    engine = SnowballMCEngine(
        params=MCParams(num_paths=50000, seed=42),
        method=MonteCarloMethod.QUASI,
    )

    # Baseline standard snowball
    standard = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        participation_rate=1.0,
        include_principal=True,
    )
    standard_price = engine.price(standard, pricing_env)

    # Test different airbag barriers
    print("\n1. Airbag Barrier Sensitivity (participation=50%):")
    print("-" * 50)
    print(f"{'Airbag Barrier':>15} {'Price':>15} {'vs Standard':>15}")
    print("-" * 50)

    for barrier in [50, 55, 60, 65, 70]:
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            ko_barrier=103.0,
            ko_rate=0.15,
            airbag_barrier=float(barrier),
            participation_rate=1.0,
            airbag_participation_rate=0.5,
            include_principal=True,
        )
        price = engine.price(snowball, pricing_env)
        diff = price - standard_price
        print(f"{barrier:>15} {price:>15,.2f} {diff:>+15,.2f}")

    # Test different airbag participation rates
    print("\n2. Airbag Participation Rate Sensitivity (barrier=60):")
    print("-" * 50)
    print(f"{'Participation':>15} {'Price':>15} {'vs Standard':>15}")
    print("-" * 50)

    for part_rate in [0.3, 0.4, 0.5, 0.6, 0.7]:
        snowball = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            ko_barrier=103.0,
            ko_rate=0.15,
            airbag_barrier=60.0,
            participation_rate=1.0,
            airbag_participation_rate=part_rate,
            include_principal=True,
        )
        price = engine.price(snowball, pricing_env)
        diff = price - standard_price
        print(f"{part_rate:>14.0%} {price:>15,.2f} {diff:>+15,.2f}")

    print("\nKey Insights:")
    print("  - Higher airbag barrier -> more protection -> higher price")
    print("  - Lower airbag participation -> more protection -> higher price")


def demo_volatility_impact():
    """Show how volatility affects airbag value."""
    print("\n" + "=" * 70)
    print("VOLATILITY IMPACT ON AIRBAG VALUE")
    print("=" * 70)

    engine = SnowballMCEngine(
        params=MCParams(num_paths=50000, seed=42),
        method=MonteCarloMethod.QUASI,
    )

    print("\nPrice comparison at different volatility levels:")
    print("-" * 65)
    print(f"{'Volatility':>12} {'Standard':>15} {'Airbag':>15} {'Premium':>12} {'Premium %':>10}")
    print("-" * 65)

    for vol in [0.20, 0.25, 0.30, 0.35, 0.40]:
        pricing_env = create_pricing_environment(spot=100.0, volatility=vol)

        standard = create_standard_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            ko_barrier=103.0,
            ko_rate=0.15,
            participation_rate=1.0,
            include_principal=True,
        )

        airbag = create_airbag_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            ki_barrier=75.0,
            ko_barrier=103.0,
            ko_rate=0.15,
            airbag_barrier=60.0,
            participation_rate=1.0,
            airbag_participation_rate=0.5,
            include_principal=True,
        )

        standard_price = engine.price(standard, pricing_env)
        airbag_price = engine.price(airbag, pricing_env)
        premium = airbag_price - standard_price
        premium_pct = premium / standard_price * 100

        print(
            f"{vol:>11.0%} {standard_price:>15,.2f} {airbag_price:>15,.2f} "
            f"{premium:>+12,.2f} {premium_pct:>+9.2f}%"
        )

    print("\nKey Insight: Airbag premium increases with volatility")
    print("             Higher vol -> more chance of hitting airbag zone -> more value")


def demo_custom_airbag_strike():
    """Demonstrate airbag with custom strike."""
    print("\n" + "=" * 70)
    print("AIRBAG WITH CUSTOM STRIKE")
    print("=" * 70)

    print("\nAirbag can use a different strike when spot is in the airbag zone.")
    print("This provides even more protection by reducing the reference point.")

    # Standard airbag (same strike)
    airbag_same_strike = create_airbag_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        airbag_participation_rate=0.5,
        airbag_strike=None,  # Uses main strike (100)
        include_principal=True,
    )

    # Airbag with lower strike
    airbag_lower_strike = create_airbag_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        airbag_participation_rate=0.5,
        airbag_strike=80.0,  # Lower strike in airbag zone
        include_principal=True,
    )

    print("\nV1 Payoff Comparison (in airbag zone, spot < 60):")
    print("-" * 65)
    print(f"{'Spot':>10} {'Strike=100':>20} {'Strike=80':>20} {'Savings':>12}")
    print("-" * 65)

    for spot in [40, 45, 50, 55]:
        payoff_same = airbag_same_strike.get_maturity_payoff_v1(float(spot))
        payoff_lower = airbag_lower_strike.get_maturity_payoff_v1(float(spot))
        savings = payoff_lower - payoff_same
        print(f"{spot:>10} {payoff_same:>20,.0f} {payoff_lower:>20,.0f} {savings:>+12,.0f}")

    print("\nFormula comparison at spot=50:")
    print("  Strike=100: 0.5 * (50 - 100) * 1M / 100 = -250,000 loss")
    print("  Strike=80:  0.5 * (50 - 80) * 1M / 100 = -150,000 loss")
    print("  Custom airbag strike reduces the loss reference point")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("AIRBAG SNOWBALL OPTION DEMO")
    print("=" * 70)
    print("\nAirbag snowballs provide additional downside protection by reducing")
    print("participation rate when spot falls below the airbag barrier.")
    print("\nThis is particularly valuable in high-volatility markets where")
    print("extreme downside moves are more likely.")

    demo_payoff_comparison()
    demo_mc_pricing()
    demo_airbag_sensitivity()
    demo_volatility_impact()
    demo_custom_airbag_strike()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Airbag structure provides protection in extreme downside scenarios")
    print("2. Protection kicks in when spot falls below airbag barrier")
    print("3. Lower airbag participation rate = more protection = higher price")
    print("4. Higher airbag barrier = more scenarios protected = higher price")
    print("5. Airbag value increases with volatility (more tail risk protection)")
    print("6. Custom airbag strike can provide additional protection")


if __name__ == "__main__":
    main()
