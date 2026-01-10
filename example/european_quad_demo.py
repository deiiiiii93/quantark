"""
Demonstration of European Vanilla Option pricing using Quadrature methods.

This script demonstrates:
1. Setting up market data (spot, volatility, rates, dividends)
2. Creating European Call and Put options
3. Pricing options using numerical quadrature (Simpson's rule, Gauss-Legendre)
4. Comparing quadrature results with analytical Black-Scholes
5. Demonstrating convergence with different grid sizes
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.engine.quad import EuropeanQuadEngine
from asset.equity.param import QuadParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from util.enum.engine_enums import EngineType, QuadratureMethod


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_market_data(pricing_env: PricingEnvironment):
    """Print market data summary."""
    print(f"Spot Price (S):        ${pricing_env.spot:.2f}")
    print(f"Volatility (σ):        {pricing_env.get_vol(100, 1.0):.2%}")
    print(f"Risk-Free Rate (r):    {pricing_env.get_rate(1.0):.2%}")
    print(f"Dividend Yield (q):    {pricing_env.get_div_yield(1.0):.2%}")


def print_option_details(option: EuropeanVanillaOption):
    """Print option specification."""
    print(f"Option Type:           {option.option_type}")
    print(f"Strike (K):            ${option.strike:.2f}")
    print(f"Time to Maturity (T):  {option.maturity:.2f} years")


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div: float = 0.02,
) -> PricingEnvironment:
    """Create a standard pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def demo_basic_quadrature_pricing():
    """Demonstrate basic quadrature pricing vs Black-Scholes."""
    print_section("BASIC QUADRATURE PRICING")

    # Step 1: Set up market data
    print("\n1. Market Data Setup")
    print("-" * 40)
    pricing_env = create_pricing_env()
    print_market_data(pricing_env)

    # Step 2: Create options
    print("\n2. Option Specifications")
    print("-" * 40)
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    put = EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

    print("Call Option:")
    print_option_details(call)
    print("\nPut Option:")
    print_option_details(put)

    # Step 3: Price with different engines
    print("\n3. Pricing Comparison")
    print("-" * 40)

    bs_engine = BlackScholesEngine()
    quad_engine = EuropeanQuadEngine()

    # Time the pricing
    start = time.perf_counter()
    bs_call = bs_engine.price(call, pricing_env)
    bs_time = time.perf_counter() - start

    start = time.perf_counter()
    quad_call = quad_engine.price(call, pricing_env)
    quad_time = time.perf_counter() - start

    bs_put = bs_engine.price(put, pricing_env)
    quad_put = quad_engine.price(put, pricing_env)

    print(f"\n{'Method':<25} {'Call Price':>15} {'Put Price':>15} {'Time (ms)':>12}")
    print("-" * 70)
    print(f"{'Black-Scholes (Analytical)':<25} ${bs_call:>13.6f} ${bs_put:>13.6f} {bs_time*1000:>10.3f}")
    print(f"{'Quadrature (Simpson)':<25} ${quad_call:>13.6f} ${quad_put:>13.6f} {quad_time*1000:>10.3f}")
    print("-" * 70)
    print(f"{'Difference':<25} ${abs(quad_call-bs_call):>13.6f} ${abs(quad_put-bs_put):>13.6f}")

    return pricing_env


def demo_quadrature_methods():
    """Demonstrate different quadrature methods."""
    print_section("QUADRATURE METHOD COMPARISON")

    pricing_env = create_pricing_env()
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)

    # Reference: Black-Scholes
    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    print(f"\nReference (Black-Scholes): ${bs_price:.6f}")
    print("\nQuadrature Methods:")
    print("-" * 70)
    print(f"{'Method':<30} {'Price':>15} {'Error':>15} {'Time (ms)':>12}")
    print("-" * 70)

    methods = [
        ("Simpson's Rule", QuadratureMethod.SIMPSON),
        ("Gauss-Legendre", QuadratureMethod.GAUSS_LEGENDRE),
    ]

    for name, method in methods:
        engine = EuropeanQuadEngine(method=method)

        start = time.perf_counter()
        price = engine.price(call, pricing_env)
        elapsed = time.perf_counter() - start

        error = abs(price - bs_price)
        print(f"{name:<30} ${price:>13.6f} {error:>15.6f} {elapsed*1000:>10.3f}")

    # Also demonstrate two-level enum pattern
    print("\n\nTwo-Level Enum Pattern:")
    print("-" * 40)
    engine = EuropeanQuadEngine(method=EngineType.QUADRATURE(QuadratureMethod.SIMPSON))
    price = engine.price(call, pricing_env)
    print(f"EngineType.QUADRATURE(QuadratureMethod.SIMPSON): ${price:.6f}")


def demo_convergence():
    """Demonstrate convergence with increasing grid points."""
    print_section("CONVERGENCE ANALYSIS")

    pricing_env = create_pricing_env()
    call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)

    # Reference: Black-Scholes
    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    print(f"\nReference (Black-Scholes): ${bs_price:.6f}")
    print("\nConvergence with Grid Size (Simpson's Rule):")
    print("-" * 70)
    print(f"{'Grid Points':>12} {'Price':>15} {'Error':>15} {'Time (ms)':>12}")
    print("-" * 70)

    grid_sizes = [101, 251, 501, 1001, 2001, 5001]

    for n in grid_sizes:
        params = QuadParams(grid_points=n)
        engine = EuropeanQuadEngine(params=params)

        start = time.perf_counter()
        price = engine.price(call, pricing_env)
        elapsed = time.perf_counter() - start

        error = abs(price - bs_price)
        print(f"{n:>12} ${price:>13.6f} {error:>15.8f} {elapsed*1000:>10.3f}")

    print("\nNote: Simpson's rule has O(1/N^4) convergence, so doubling grid points")
    print("      reduces error by approximately 16x.")


def demo_moneyness_scenarios():
    """Demonstrate pricing across different moneyness levels."""
    print_section("MONEYNESS SCENARIOS")

    pricing_env = create_pricing_env()

    bs_engine = BlackScholesEngine()
    quad_engine = EuropeanQuadEngine()

    print("\nCall Options at Different Strikes:")
    print("-" * 70)
    print(f"{'Strike':>10} {'Moneyness':>12} {'BS Price':>15} {'Quad Price':>15} {'Error':>12}")
    print("-" * 70)

    strikes = [80, 90, 95, 100, 105, 110, 120]
    spot = pricing_env.spot

    for K in strikes:
        call = EuropeanVanillaOption(strike=float(K), option_type=OptionType.CALL, maturity=1.0)

        bs_price = bs_engine.price(call, pricing_env)
        quad_price = quad_engine.price(call, pricing_env)
        error = abs(quad_price - bs_price)

        moneyness = "ITM" if K < spot else ("ATM" if K == spot else "OTM")
        print(f"${K:>9} {moneyness:>12} ${bs_price:>13.6f} ${quad_price:>13.6f} {error:>12.6f}")


def demo_maturity_scenarios():
    """Demonstrate pricing across different maturities."""
    print_section("MATURITY SCENARIOS")

    pricing_env = create_pricing_env()

    bs_engine = BlackScholesEngine()
    quad_engine = EuropeanQuadEngine()

    print("\nATM Call Options at Different Maturities:")
    print("-" * 70)
    print(f"{'Maturity':>10} {'BS Price':>15} {'Quad Price':>15} {'Error':>12}")
    print("-" * 70)

    maturities = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    for T in maturities:
        call = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=T)

        bs_price = bs_engine.price(call, pricing_env)
        quad_price = quad_engine.price(call, pricing_env)
        error = abs(quad_price - bs_price)

        print(f"{T:>10.2f}y ${bs_price:>13.6f} ${quad_price:>13.6f} {error:>12.6f}")


def demo_put_call_parity():
    """Verify put-call parity with quadrature pricing."""
    print_section("PUT-CALL PARITY VERIFICATION")

    import math

    pricing_env = create_pricing_env()
    S = pricing_env.spot
    K = 100.0
    T = 1.0
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)

    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)

    quad_engine = EuropeanQuadEngine()
    call_price = quad_engine.price(call, pricing_env)
    put_price = quad_engine.price(put, pricing_env)

    # Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)
    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    print(f"\nPut-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)")
    print(f"\nQuadrature Prices:")
    print(f"  Call Price (C): ${call_price:.6f}")
    print(f"  Put Price (P):  ${put_price:.6f}")
    print(f"\nParity Check:")
    print(f"  LHS (C - P):                 ${lhs:.6f}")
    print(f"  RHS (S*e^(-qT) - K*e^(-rT)): ${rhs:.6f}")
    print(f"  Difference:                  ${abs(lhs - rhs):.10f}")

    if abs(lhs - rhs) < 0.01:
        print("\n  Put-Call Parity holds with quadrature pricing!")


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print(
        "*"
        + "  QUANTARK - European Option Quadrature Pricing Demonstration".center(78)
        + "*"
    )
    print("*" + " " * 78 + "*")
    print("*" * 80)

    try:
        # Demo 1: Basic quadrature pricing
        demo_basic_quadrature_pricing()

        # Demo 2: Different quadrature methods
        demo_quadrature_methods()

        # Demo 3: Convergence analysis
        demo_convergence()

        # Demo 4: Moneyness scenarios
        demo_moneyness_scenarios()

        # Demo 5: Maturity scenarios
        demo_maturity_scenarios()

        # Demo 6: Put-call parity
        demo_put_call_parity()

        print_section("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("\nKey Takeaways:")
        print("  1. Quadrature pricing matches Black-Scholes to high precision")
        print("  2. Simpson's rule provides O(1/N^4) convergence")
        print("  3. 1000 grid points typically achieve <0.01 accuracy")
        print("  4. Quadrature is slower than analytical but provides a foundation")
        print("     for pricing complex products (barriers, autocallables)")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
