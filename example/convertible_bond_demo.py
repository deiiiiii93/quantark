#!/usr/bin/env python
"""
Convertible Bond Pricing Demo

This example demonstrates how to use QuantArk's convertible bond pricing
capabilities, including:
- Creating convertible bond products with various features
- Pricing with different methods (tree and PDE based)
- Comparing pricing results across methods
- Using the two-level enum pattern for method selection

Convertible bonds are hybrid securities that combine:
- Fixed income features (coupons, principal repayment)
- Equity features (conversion option)
- Credit risk (default possibility)
- Embedded options (call/put provisions)

Note:
    Run from the project root or set PYTHONPATH=. to resolve local imports.
"""

from datetime import datetime

from quantark.asset.bond.product.convertible import (
    ConvertibleBond,
    CallScheduleEntry,
    PutScheduleEntry,
    DiscreteDividend,
)
from quantark.asset.bond.engine.convertible import ConvertibleBondEngine, ConvertibleBondResult
from quantark.asset.bond.engine.tree.convertible import ConvertibleBondTreeParams
from quantark.asset.bond.engine.pde.convertible import ConvertibleBondPDEParams
from quantark.param.quote import SpotQuote
from quantark.param.vol import FlatVolSurface
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import EngineType, ConvertibleBondMethod


def create_simple_convertible():
    """Create a simple convertible bond."""
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        face_value=100.0,
        coupon_rate=0.025,  # 2.5% coupon
        conversion_ratio=10.0,  # 10 shares per bond (conversion price = $10)
        credit_spread=0.02,  # 200bp credit spread
        hazard_rate=0.01,  # 1% annual default probability
        recovery_rate=0.4,  # 40% recovery on default
    )


def create_callable_convertible():
    """Create a callable convertible bond."""
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        face_value=100.0,
        coupon_rate=0.025,
        conversion_ratio=10.0,
        credit_spread=0.02,
        hazard_rate=0.01,
        recovery_rate=0.4,
        call_schedule=[
            # Issuer can call at 105 after year 2
            CallScheduleEntry(
                call_date=datetime(2026, 1, 1),
                call_price=105.0,
            ),
            # Call price drops to 102 after year 3
            CallScheduleEntry(
                call_date=datetime(2027, 1, 1),
                call_price=102.0,
            ),
        ],
    )


def create_puttable_convertible():
    """Create a puttable convertible bond."""
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),
        face_value=100.0,
        coupon_rate=0.025,
        conversion_ratio=10.0,
        credit_spread=0.02,
        hazard_rate=0.01,
        recovery_rate=0.4,
        put_schedule=[
            # Holder can put at par after year 2
            PutScheduleEntry(
                put_date=datetime(2026, 1, 1),
                put_price=100.0,
            ),
        ],
    )


def create_pricing_environment(stock_price: float = 12.0):
    """Create a pricing environment with given stock price."""
    return PricingEnvironment(
        valuation_date=datetime(2024, 6, 1),
        spot_quote=SpotQuote(spot=stock_price),
        vol_surface=FlatVolSurface(volatility=0.30),  # 30% vol
        rate_curve=FlatRateCurve(rate=0.05),  # 5% risk-free rate
    )


def demo_basic_pricing():
    """Demonstrate basic convertible bond pricing."""
    print("=" * 70)
    print("BASIC CONVERTIBLE BOND PRICING")
    print("=" * 70)

    cb = create_simple_convertible()
    env = create_pricing_environment(stock_price=12.0)

    print(f"\nBond: {cb}")
    print(f"Conversion price: ${cb.conversion_price:.2f}")
    print(f"Current stock price: ${env.spot:.2f}")
    print(f"Parity (conversion value): ${cb.parity(env.spot):.2f}")

    # Price with default method (binomial)
    engine = ConvertibleBondEngine(env)
    result = engine.price_with_details(cb)

    print(f"\nPricing Results (Binomial GS model):")
    print(f"  Clean Price: ${result.price:.4f}")
    print(f"  Dirty Price: ${result.dirty_price:.4f}")
    print(f"  Delta: {result.delta:.4f}")
    print(f"  Gamma: {result.gamma:.4f}")
    print(f"  Conversion Probability: {result.conversion_probability:.2%}")

    # Calculate conversion premium
    premium = cb.conversion_premium(env.spot, result.price)
    print(f"  Conversion Premium: {premium:.2%}")


def demo_method_comparison():
    """Compare different pricing methods."""
    print("\n" + "=" * 70)
    print("PRICING METHOD COMPARISON")
    print("=" * 70)

    cb = create_simple_convertible()
    env = create_pricing_environment(stock_price=12.0)

    methods = [
        ("Binomial GS", ConvertibleBondMethod.BINOMIAL_GS),
        ("Trinomial HW", ConvertibleBondMethod.TRINOMIAL_HW),
        ("Jump-Diffusion PDE", ConvertibleBondMethod.JUMP_DIFFUSION),
        ("Tsiveriotis-Fernandes PDE", ConvertibleBondMethod.TF),
    ]

    # Use higher resolution for better accuracy
    tree_params = ConvertibleBondTreeParams(num_steps=200)
    pde_params = ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=400)

    print(f"\n{'Method':<25} {'Price':>10} {'Delta':>10} {'Conv Prob':>12}")
    print("-" * 60)

    for name, method in methods:
        engine = ConvertibleBondEngine(
            env,
            method=method,
            tree_params=tree_params,
            pde_params=pde_params,
        )
        result = engine.price_with_details(cb)
        conv_prob = (
            f"{result.conversion_probability:.2%}"
            if result.conversion_probability > 0
            else "N/A"
        )
        print(f"{name:<25} {result.price:>10.4f} {result.delta:>10.4f} {conv_prob:>12}")


def demo_tf_decomposition():
    """Demonstrate TF model decomposition."""
    print("\n" + "=" * 70)
    print("TSIVERIOTIS-FERNANDES DECOMPOSITION")
    print("=" * 70)

    cb = create_simple_convertible()
    env = create_pricing_environment(stock_price=12.0)

    engine = ConvertibleBondEngine(
        env,
        method=EngineType.PDE(ConvertibleBondMethod.TF),
        pde_params=ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=400),
    )
    result = engine.price_with_details(cb)

    print(f"\nTF Model Decomposition:")
    print(f"  Total Dirty Price: ${result.dirty_price:.4f}")
    print(f"  Equity Component (u): ${result.equity_component:.4f}")
    print(f"  Bond Component (v - COCB): ${result.bond_component:.4f}")
    print(f"  Sum of Components: ${result.equity_component + result.bond_component:.4f}")

    # COCB can also be obtained directly
    cocb = engine.get_cocb(cb)
    print(f"\n  COCB (via method): ${cocb:.4f}")

    equity_pct = result.equity_component / result.dirty_price * 100
    bond_pct = result.bond_component / result.dirty_price * 100
    print(f"\n  Equity % of value: {equity_pct:.1f}%")
    print(f"  Bond % of value: {bond_pct:.1f}%")


def demo_callable_puttable():
    """Demonstrate callable and puttable bonds."""
    print("\n" + "=" * 70)
    print("CALLABLE AND PUTTABLE CONVERTIBLE BONDS")
    print("=" * 70)

    env = create_pricing_environment(stock_price=12.0)
    tree_params = ConvertibleBondTreeParams(num_steps=200)

    # Plain vanilla
    cb_plain = create_simple_convertible()
    engine_plain = ConvertibleBondEngine(env, tree_params=tree_params)
    price_plain = engine_plain.price(cb_plain)

    # Callable
    cb_callable = create_callable_convertible()
    engine_callable = ConvertibleBondEngine(env, tree_params=tree_params)
    price_callable = engine_callable.price(cb_callable)

    # Puttable
    cb_puttable = create_puttable_convertible()
    engine_puttable = ConvertibleBondEngine(env, tree_params=tree_params)
    price_puttable = engine_puttable.price(cb_puttable)

    print(f"\nPrice Comparison:")
    print(f"  Plain Convertible:    ${price_plain:.4f}")
    print(f"  Callable Convertible: ${price_callable:.4f}")
    print(f"  Puttable Convertible: ${price_puttable:.4f}")

    print(f"\n  Call feature value: ${price_plain - price_callable:.4f} (reduces value)")
    print(f"  Put feature value:  ${price_puttable - price_plain:.4f} (increases value)")


def demo_stock_price_sensitivity():
    """Demonstrate price sensitivity to stock price."""
    print("\n" + "=" * 70)
    print("STOCK PRICE SENSITIVITY")
    print("=" * 70)

    cb = create_simple_convertible()
    tree_params = ConvertibleBondTreeParams(num_steps=200)

    print(f"\nConversion price: ${cb.conversion_price:.2f}")
    print(f"\n{'Stock':>10} {'Parity':>10} {'CB Price':>10} {'Premium':>10} {'Delta':>8}")
    print("-" * 55)

    for stock_price in [5, 8, 10, 12, 15, 20]:
        env = create_pricing_environment(stock_price=stock_price)
        engine = ConvertibleBondEngine(env, tree_params=tree_params)
        result = engine.price_with_details(cb)

        parity = cb.parity(stock_price)
        premium = cb.conversion_premium(stock_price, result.price)
        premium_str = f"{premium:.1%}" if premium < 10 else ">1000%"

        print(
            f"{stock_price:>10.2f} {parity:>10.2f} {result.price:>10.2f} "
            f"{premium_str:>10} {result.delta:>8.4f}"
        )

    print(
        "\nNote: As stock rises, the bond approaches parity (delta -> conversion_ratio)"
    )
    print("      As stock falls, the bond is floor by bond value (delta -> 0)")


def demo_two_level_enum():
    """Demonstrate the two-level enum pattern."""
    print("\n" + "=" * 70)
    print("TWO-LEVEL ENUM PATTERN FOR METHOD SELECTION")
    print("=" * 70)

    cb = create_simple_convertible()
    env = create_pricing_environment()

    print("\nThree ways to specify the pricing method:")

    # Method 1: Two-level enum (recommended)
    print("\n1. Two-level enum pattern (recommended):")
    print('   engine = ConvertibleBondEngine(env, method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS))')
    engine1 = ConvertibleBondEngine(
        env, method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS)
    )
    print(f"   Price: ${engine1.price(cb):.4f}")

    # Method 2: Direct enum
    print("\n2. Direct method enum:")
    print('   engine = ConvertibleBondEngine(env, method=ConvertibleBondMethod.JUMP_DIFFUSION)')
    engine2 = ConvertibleBondEngine(
        env, method=ConvertibleBondMethod.JUMP_DIFFUSION
    )
    print(f"   Price: ${engine2.price(cb):.4f}")

    # Method 3: String (backward compatibility)
    print("\n3. String method (backward compatible):")
    print('   engine = ConvertibleBondEngine(env, method="tf")')
    engine3 = ConvertibleBondEngine(env, method="tf")
    print(f"   Price: ${engine3.price(cb):.4f}")


def demo_risk_metrics():
    """Demonstrate DV01, CS01, duration, convexity, and floor bond metrics."""
    print("\n" + "=" * 70)
    print("INTEREST RATE AND CREDIT RISK METRICS")
    print("=" * 70)

    cb = create_simple_convertible()
    env = create_pricing_environment(stock_price=12.0)

    # Use tree method for efficiency
    engine = ConvertibleBondEngine(
        env,
        tree_params=ConvertibleBondTreeParams(num_steps=100),
    )

    print(f"\nBond: {cb}")
    print(f"Stock Price: ${env.spot:.2f}")

    # Get full results including risk metrics
    result = engine.price_with_details(cb)

    print(f"\n--- Convertible Bond Metrics ---")
    print(f"  Price (clean):       ${result.price:.4f}")
    print(f"  Price (dirty):       ${result.dirty_price:.4f}")
    print(f"  DV01:                ${result.dv01:.6f}")
    print(f"  CS01:                ${result.cs01:.6f}")
    print(f"  Modified Duration:   {result.modified_duration:.4f} years")
    print(f"  Convexity:           {result.convexity:.2f}")

    print(f"\n--- Floor Bond Metrics (Straight Bond Value) ---")
    print(f"  Floor Bond Price:    ${result.floor_bond_price:.4f}")
    print(f"  Floor Bond DV01:     ${result.floor_bond_dv01:.6f}")
    print(f"  Floor Bond CS01:     ${result.floor_bond_cs01:.6f}")
    print(f"  Floor Bond Duration: {result.floor_bond_duration:.4f} years")
    print(f"  Floor Bond Convexity:{result.floor_bond_convexity:.2f}")

    # Calculate option value
    option_value = result.price - result.floor_bond_price
    print(f"\n--- Value Decomposition ---")
    print(f"  Floor Bond Value:    ${result.floor_bond_price:.4f} ({result.floor_bond_price/result.price*100:.1f}%)")
    print(f"  Option Value:        ${option_value:.4f} ({option_value/result.price*100:.1f}%)")

    # Compare DV01s
    print(f"\n--- Interest Rate Sensitivity Comparison ---")
    print(f"  Convertible DV01:    ${result.dv01:.6f}")
    print(f"  Floor Bond DV01:     ${result.floor_bond_dv01:.6f}")
    print(f"  Ratio (CB/Floor):    {result.dv01/result.floor_bond_dv01:.2%}")
    print(f"\n  Note: Convertible has lower DV01 because the equity component")
    print(f"        has less interest rate sensitivity than pure debt.")

    # Demonstrate direct method calls
    print(f"\n--- Direct Method Calls ---")
    print(f"  engine.dv01(cb):              ${engine.dv01(cb):.6f}")
    print(f"  engine.cs01(cb):              ${engine.cs01(cb):.6f}")
    print(f"  engine.modified_duration(cb): {engine.modified_duration(cb):.4f}")
    print(f"  engine.floor_bond_price(cb):  ${engine.floor_bond_price(cb):.4f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("QUANTARK CONVERTIBLE BOND PRICING DEMONSTRATION")
    print("=" * 70)

    demo_basic_pricing()
    demo_method_comparison()
    demo_tf_decomposition()
    demo_callable_puttable()
    demo_stock_price_sensitivity()
    demo_two_level_enum()
    demo_risk_metrics()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
