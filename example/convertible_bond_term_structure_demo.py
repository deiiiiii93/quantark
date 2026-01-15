#!/usr/bin/env python
"""
Convertible Bond Pricing with Term Structure Curves Demo

This example demonstrates how to price convertible bonds using different
term structure (yield curve) specifications with various pricing engines.

Key concepts demonstrated:
1. Different yield curve types:
   - FlatRateCurve: Constant rate across all maturities
   - LinearRateCurve: Linear interpolation between rate pillars
   - LogLinearRateCurve: Log-linear interpolation (market standard)
   - CubicSplineRateCurve: Smooth cubic spline interpolation

2. Different pricing engines:
   - Binomial GS (Goldman-Sachs binomial tree)
   - Trinomial HW (Hull-White trinomial tree)
   - Jump-Diffusion PDE
   - Tsiveriotis-Fernandes PDE

3. Impact of term structure shape on convertible bond pricing:
   - Upward sloping (normal)
   - Flat
   - Downward sloping (inverted)
   - Humped

Usage:
    python example/convertible_bond_term_structure_demo.py

Note:
    Run from the project root or set PYTHONPATH=. to resolve local imports.
"""

from datetime import datetime
from typing import List, Tuple

from asset.bond.product.convertible import ConvertibleBond
from asset.bond.engine.convertible import ConvertibleBondEngine
from asset.bond.engine.tree.convertible import ConvertibleBondTreeParams
from asset.bond.engine.pde.convertible import ConvertibleBondPDEParams
from param.quote import SpotQuote
from param.vol import FlatVolSurface
from param.rrf import (
    FlatRateCurve,
    LinearRateCurve,
    LogLinearRateCurve,
    CubicSplineRateCurve,
)
from priceenv import PricingEnvironment
from util.enum.engine_enums import ConvertibleBondMethod, ConvertibleBondTrinomialVolScheme


def create_convertible_bond() -> ConvertibleBond:
    """Create a standard convertible bond for testing."""
    return ConvertibleBond(
        issue_date=datetime(2024, 1, 1),
        maturity_date=datetime(2029, 1, 1),  # 5-year bond
        face_value=100.0,
        coupon_rate=0.03,  # 3% coupon
        conversion_ratio=8.0,  # Conversion price = $12.50
        credit_spread=0.015,  # 150bp credit spread
        hazard_rate=0.01,  # 1% default probability
        recovery_rate=0.4,  # 40% recovery
    )


def create_flat_curve(rate: float = 0.05) -> FlatRateCurve:
    """Create a flat rate curve."""
    return FlatRateCurve(rate=rate)


def create_upward_sloping_curve() -> LinearRateCurve:
    """
    Create an upward sloping (normal) yield curve.
    
    Typical shape during economic expansion:
    - Short rates lower than long rates
    - Reflects expectation of future rate increases
    """
    pillars = [
        (0.25, 0.030),  # 3-month: 3.0%
        (0.5, 0.035),   # 6-month: 3.5%
        (1.0, 0.040),   # 1-year: 4.0%
        (2.0, 0.045),   # 2-year: 4.5%
        (3.0, 0.048),   # 3-year: 4.8%
        (5.0, 0.050),   # 5-year: 5.0%
        (7.0, 0.052),   # 7-year: 5.2%
        (10.0, 0.055),  # 10-year: 5.5%
    ]
    return LinearRateCurve(pillars=pillars)


def create_inverted_curve() -> LinearRateCurve:
    """
    Create an inverted (downward sloping) yield curve.
    
    Typical shape before recessions:
    - Short rates higher than long rates
    - Often signals economic slowdown expectations
    """
    pillars = [
        (0.25, 0.055),  # 3-month: 5.5%
        (0.5, 0.053),   # 6-month: 5.3%
        (1.0, 0.050),   # 1-year: 5.0%
        (2.0, 0.045),   # 2-year: 4.5%
        (3.0, 0.042),   # 3-year: 4.2%
        (5.0, 0.040),   # 5-year: 4.0%
        (7.0, 0.038),   # 7-year: 3.8%
        (10.0, 0.035),  # 10-year: 3.5%
    ]
    return LinearRateCurve(pillars=pillars)


def create_humped_curve() -> CubicSplineRateCurve:
    """
    Create a humped yield curve.
    
    Rates peak at intermediate maturities:
    - Often seen during policy uncertainty
    - Market expects rate cuts after initial increases
    """
    pillars = [
        (0.25, 0.040),  # 3-month: 4.0%
        (0.5, 0.045),   # 6-month: 4.5%
        (1.0, 0.050),   # 1-year: 5.0%
        (2.0, 0.052),   # 2-year: 5.2% (peak)
        (3.0, 0.050),   # 3-year: 5.0%
        (5.0, 0.045),   # 5-year: 4.5%
        (7.0, 0.042),   # 7-year: 4.2%
        (10.0, 0.040),  # 10-year: 4.0%
    ]
    return CubicSplineRateCurve(pillars=pillars)


def create_log_linear_curve() -> LogLinearRateCurve:
    """
    Create a log-linear interpolated curve (market standard).
    
    Log-linear interpolation on discount factors ensures:
    - Smooth forward rates
    - No arbitrage in forward curve
    """
    pillars = [
        (0.25, 0.035),  # 3-month: 3.5%
        (0.5, 0.038),   # 6-month: 3.8%
        (1.0, 0.042),   # 1-year: 4.2%
        (2.0, 0.046),   # 2-year: 4.6%
        (3.0, 0.048),   # 3-year: 4.8%
        (5.0, 0.050),   # 5-year: 5.0%
        (7.0, 0.051),   # 7-year: 5.1%
        (10.0, 0.052),  # 10-year: 5.2%
    ]
    return LogLinearRateCurve(pillars=pillars)


def create_pricing_environment(
    rate_curve,
    stock_price: float = 12.0,
    volatility: float = 0.30,
) -> PricingEnvironment:
    """Create a pricing environment with the given rate curve."""
    return PricingEnvironment(
        valuation_date=datetime(2024, 6, 1),
        spot_quote=SpotQuote(spot=stock_price),
        vol_surface=FlatVolSurface(volatility=volatility),
        rate_curve=rate_curve,
    )


def demo_curve_types():
    """Demonstrate different yield curve types."""
    print("=" * 70)
    print("YIELD CURVE TYPES")
    print("=" * 70)

    curves = [
        ("Flat (5%)", create_flat_curve(0.05)),
        ("Upward Sloping", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
        ("Humped", create_humped_curve()),
        ("Log-Linear", create_log_linear_curve()),
    ]

    print(f"\n{'Maturity':<12}", end="")
    for name, _ in curves:
        print(f"{name:>14}", end="")
    print()
    print("-" * 82)

    maturities = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    for t in maturities:
        print(f"{t:<12.1f}", end="")
        for _, curve in curves:
            rate = curve.get_rate(t) * 100  # Convert to percentage
            print(f"{rate:>13.2f}%", end="")
        print()

    print("\nDiscount Factors:")
    print(f"{'Maturity':<12}", end="")
    for name, _ in curves:
        print(f"{name:>14}", end="")
    print()
    print("-" * 82)

    for t in maturities:
        print(f"{t:<12.1f}", end="")
        for _, curve in curves:
            df = curve.get_discount_factor(t)
            print(f"{df:>14.6f}", end="")
        print()


def demo_pricing_with_different_curves():
    """Price convertible bond with different yield curves."""
    print("\n" + "=" * 70)
    print("CONVERTIBLE BOND PRICING WITH DIFFERENT YIELD CURVES")
    print("=" * 70)

    cb = create_convertible_bond()
    print(f"\nConvertible Bond: {cb}")
    print(f"Conversion Price: ${cb.conversion_price:.2f}")

    curves = [
        ("Flat 5%", create_flat_curve(0.05)),
        ("Upward Sloping", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
        ("Humped", create_humped_curve()),
        ("Log-Linear", create_log_linear_curve()),
    ]

    tree_params = ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.CONSTANT_VOL)

    print(f"\nStock Price: $12.00 | Method: Trinomial HW | Steps: 200")
    print(f"\n{'Curve Type':<20} {'Price':>10} {'Delta':>10} {'Conv Prob':>12} {'DV01':>10}")
    print("-" * 65)

    for name, curve in curves:
        env = create_pricing_environment(curve, stock_price=12.0)
        engine = ConvertibleBondEngine(
            env,
            method=ConvertibleBondMethod.TRINOMIAL_HW,
            tree_params=tree_params,
        )
        result = engine.price_with_details(cb)

        conv_prob_str = (
            f"{result.conversion_probability:.2%}"
            if result.conversion_probability > 0
            else "N/A"
        )
        print(
            f"{name:<20} {result.price:>10.4f} {result.delta:>10.4f} "
            f"{conv_prob_str:>12} {result.dv01:>10.6f}"
        )


def demo_engine_comparison_with_term_structure():
    """Compare different engines using term structure curves."""
    print("\n" + "=" * 70)
    print("ENGINE COMPARISON WITH UPWARD SLOPING TERM STRUCTURE")
    print("=" * 70)

    cb = create_convertible_bond()
    curve = create_upward_sloping_curve()
    env = create_pricing_environment(curve, stock_price=12.0)

    print(f"\nCurve Type: Upward Sloping (Normal)")
    print(f"Stock Price: $12.00")

    methods = [
        ("Binomial GS", ConvertibleBondMethod.BINOMIAL_GS),
        ("Trinomial HW", ConvertibleBondMethod.TRINOMIAL_HW),
        ("Jump-Diffusion PDE", ConvertibleBondMethod.JUMP_DIFFUSION),
        ("TF PDE", ConvertibleBondMethod.TF),
    ]

    tree_params = ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.CONSTANT_VOL)
    pde_params = ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=400)

    print(f"\n{'Method':<20} {'Price':>10} {'Delta':>10} {'Gamma':>10} {'Conv Prob':>12}")
    print("-" * 65)

    for name, method in methods:
        engine = ConvertibleBondEngine(
            env,
            method=method,
            tree_params=tree_params,
            pde_params=pde_params,
        )
        result = engine.price_with_details(cb)

        conv_prob_str = (
            f"{result.conversion_probability:.2%}"
            if result.conversion_probability > 0
            else "N/A"
        )
        print(
            f"{name:<20} {result.price:>10.4f} {result.delta:>10.4f} "
            f"{result.gamma:>10.6f} {conv_prob_str:>12}"
        )


def demo_stock_price_sensitivity_by_curve():
    """Show how different curves affect price sensitivity to stock."""
    print("\n" + "=" * 70)
    print("STOCK PRICE SENSITIVITY ACROSS DIFFERENT CURVES")
    print("=" * 70)

    cb = create_convertible_bond()
    tree_params = ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.CONSTANT_VOL)

    curves = [
        ("Flat 5%", create_flat_curve(0.05)),
        ("Upward", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
    ]

    stock_prices = [8, 10, 12, 15, 18, 22]

    print(f"\nConversion Price: ${cb.conversion_price:.2f}")
    print(f"\n{'Stock':>8}", end="")
    for name, _ in curves:
        print(f"{name:>14}", end="")
    print()
    print("-" * 52)

    for stock in stock_prices:
        print(f"${stock:>6.0f}", end="")
        for _, curve in curves:
            env = create_pricing_environment(curve, stock_price=float(stock))
            engine = ConvertibleBondEngine(
                env,
                method=ConvertibleBondMethod.TRINOMIAL_HW,
                tree_params=tree_params,
            )
            price = engine.price(cb)
            print(f"{price:>14.4f}", end="")
        print()

    print("\nParity values (conversion value):")
    for stock in stock_prices:
        parity = cb.parity(float(stock))
        print(f"  Stock ${stock}: Parity ${parity:.2f}")


def demo_risk_metrics_by_curve():
    """Compare risk metrics across different yield curves."""
    print("\n" + "=" * 70)
    print("RISK METRICS ACROSS DIFFERENT YIELD CURVES")
    print("=" * 70)

    cb = create_convertible_bond()
    tree_params = ConvertibleBondTreeParams(num_steps=200)

    curves = [
        ("Flat 5%", create_flat_curve(0.05)),
        ("Upward Sloping", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
        ("Log-Linear", create_log_linear_curve()),
    ]

    print(f"\nStock Price: $12.00")
    print(
        f"\n{'Curve':<18} {'Price':>10} {'DV01':>10} {'CS01':>10} "
        f"{'Duration':>10} {'Convexity':>10}"
    )
    print("-" * 72)

    for name, curve in curves:
        env = create_pricing_environment(curve, stock_price=12.0)
        engine = ConvertibleBondEngine(
            env,
            method=ConvertibleBondMethod.TRINOMIAL_HW,
            tree_params=tree_params,
        )
        result = engine.price_with_details(cb)

        print(
            f"{name:<18} {result.price:>10.4f} {result.dv01:>10.6f} "
            f"{result.cs01:>10.6f} {result.modified_duration:>10.4f} "
            f"{result.convexity:>10.2f}"
        )


def demo_floor_bond_by_curve():
    """Compare floor bond values across different curves."""
    print("\n" + "=" * 70)
    print("FLOOR BOND (STRAIGHT BOND VALUE) BY CURVE TYPE")
    print("=" * 70)

    cb = create_convertible_bond()
    tree_params = ConvertibleBondTreeParams(num_steps=200, trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.CONSTANT_VOL)

    curves = [
        ("Flat 5%", create_flat_curve(0.05)),
        ("Upward Sloping", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
        ("Log-Linear", create_log_linear_curve()),
    ]

    print(f"\nStock Price: $12.00 (near-the-money)")
    print(
        f"\n{'Curve':<18} {'CB Price':>10} {'Floor Bond':>12} "
        f"{'Option Val':>12} {'Option %':>10}"
    )
    print("-" * 65)

    for name, curve in curves:
        env = create_pricing_environment(curve, stock_price=12.0)
        engine = ConvertibleBondEngine(
            env,
            method=ConvertibleBondMethod.TRINOMIAL_HW,
            tree_params=tree_params,
        )
        result = engine.price_with_details(cb)

        option_value = result.price - result.floor_bond_price
        option_pct = option_value / result.price * 100

        print(
            f"{name:<18} {result.price:>10.4f} {result.floor_bond_price:>12.4f} "
            f"{option_value:>12.4f} {option_pct:>9.1f}%"
        )

    print("\nNote: Higher rates -> Lower floor bond value -> Higher option % of total")


def demo_tf_decomposition_by_curve():
    """Show TF model decomposition across different curves."""
    print("\n" + "=" * 70)
    print("TSIVERIOTIS-FERNANDES DECOMPOSITION BY CURVE TYPE")
    print("=" * 70)

    cb = create_convertible_bond()
    pde_params = ConvertibleBondPDEParams(num_space_steps=200, num_time_steps=400)

    curves = [
        ("Flat 5%", create_flat_curve(0.05)),
        ("Upward Sloping", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
    ]

    print(f"\nStock Price: $12.00")
    print(
        f"\n{'Curve':<18} {'Total':>10} {'Equity (u)':>12} "
        f"{'Bond (v)':>12} {'Equity %':>10}"
    )
    print("-" * 65)

    for name, curve in curves:
        env = create_pricing_environment(curve, stock_price=12.0)
        engine = ConvertibleBondEngine(
            env,
            method=ConvertibleBondMethod.TF,
            pde_params=pde_params,
        )
        result = engine.price_with_details(cb)

        equity_pct = result.equity_component / result.dirty_price * 100

        print(
            f"{name:<18} {result.dirty_price:>10.4f} {result.equity_component:>12.4f} "
            f"{result.bond_component:>12.4f} {equity_pct:>9.1f}%"
        )

    print("\nNote: TF model decomposes convertible into equity and bond components")
    print("      Equity component reflects probability-weighted conversion value")


def demo_forward_rate_analysis():
    """Analyze forward rates implied by different curves."""
    print("\n" + "=" * 70)
    print("FORWARD RATE ANALYSIS")
    print("=" * 70)

    curves = [
        ("Upward Sloping", create_upward_sloping_curve()),
        ("Inverted", create_inverted_curve()),
        ("Log-Linear", create_log_linear_curve()),
    ]

    print("\nForward rates implied by term structure:")
    print(f"\n{'Period':<12}", end="")
    for name, _ in curves:
        print(f"{name:>18}", end="")
    print()
    print("-" * 68)

    periods = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10)]
    for t1, t2 in periods:
        print(f"{t1}Y -> {t2}Y", end="")
        print(f"{'':>4}", end="")
        for _, curve in curves:
            fwd = curve.get_forward_rate(float(t1) if t1 > 0 else 0.001, float(t2))
            print(f"{fwd*100:>17.2f}%", end="")
        print()

    print("\nNote: Forward rates show market expectations for future short rates")
    print("      Upward curve implies rising rates; inverted implies falling rates")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("CONVERTIBLE BOND PRICING WITH TERM STRUCTURE CURVES")
    print("=" * 70)
    print("\nThis demo shows how different yield curve shapes affect")
    print("convertible bond pricing across multiple pricing engines.")

    demo_curve_types()
    demo_pricing_with_different_curves()
    demo_engine_comparison_with_term_structure()
    demo_stock_price_sensitivity_by_curve()
    demo_risk_metrics_by_curve()
    demo_floor_bond_by_curve()
    demo_tf_decomposition_by_curve()
    demo_forward_rate_analysis()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Term structure shape significantly affects convertible bond pricing")
    print("2. Higher rates reduce floor bond value but may increase option value")
    print("3. Different engines converge with sufficient resolution")
    print("4. Log-linear interpolation is the market standard for discount curves")
    print("5. Forward rates reveal market expectations embedded in the curve")


if __name__ == "__main__":
    main()
