"""
Demonstration of PDEEngine for unified PDE pricing across multiple option types.

This script demonstrates:
1. Using PDEEngine as a unified interface for all PDE pricing
2. Pricing European, American, and Barrier options with automatic solver dispatch
3. Calculating Greeks using numerical methods via PDEEngine
4. Comparing PDE results with analytical methods
5. Method selection using the two-level enum pattern
"""

import sys
from pathlib import Path
from datetime import datetime


from quantark.asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
)
from quantark.asset.equity.engine import PDEEngine, BlackScholesEngine
from quantark.asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.param import PDEParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType, BarrierType
from quantark.util.enum.engine_enums import PDEMethod, EngineType, AmericanAnalyticalMethod


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


def demo_pde_engine_basics():
    """Demonstrate basic PDEEngine usage with automatic dispatch."""
    print_section("PART 1: PDEEngine Basics - Automatic Product Dispatch")

    # Market data setup
    print("\n1. Market Data Setup")
    print("-" * 40)
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="AAPL"),
        vol_surface=FlatVolSurface(0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )
    print_market_data(pricing_env)

    # Initialize PDEEngine with default method (Crank-Nicolson)
    print("\n2. Initialize Unified PDEEngine")
    print("-" * 40)
    pde_engine = PDEEngine(params=PDEParams(grid_size=400, time_steps=200))
    print(f"Engine: {pde_engine}")
    print(f"Method: {pde_engine.method.value}")
    print(
        f"Grid: {pde_engine.params.grid_size} space steps, {pde_engine.params.time_steps} time steps"
    )

    # Price European Option
    print("\n3. European Call Option (Auto-dispatch to EuropeanPDESolver)")
    print("-" * 40)
    euro_call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    price_euro = pde_engine.price(euro_call, pricing_env)
    print(f"European Call (K=100, T=1.0): ${price_euro:.6f}")

    # Price American Option
    print("\n4. American Put Option (Auto-dispatch to AmericanPDESolver)")
    print("-" * 40)
    amer_put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)
    price_amer = pde_engine.price(amer_put, pricing_env)
    print(f"American Put (K=100, T=1.0):  ${price_amer:.6f}")

    # Price Barrier Option
    print("\n5. Barrier Option (Auto-dispatch to BarrierPDESolver)")
    print("-" * 40)
    barrier_call = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
    )
    price_barrier = pde_engine.price(barrier_call, pricing_env)
    print(f"Up-and-Out Barrier Call (K=100, B=120, T=1.0): ${price_barrier:.6f}")

    print(
        "\n✓ PDEEngine automatically dispatched to correct solver for each product type!"
    )


def demo_method_selection():
    """Demonstrate different PDE method selection patterns."""
    print_section("PART 2: Method Selection Patterns")

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )

    euro_option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    params = PDEParams(grid_size=300, time_steps=150)

    print("\n1. Two-Level Enum Pattern (Recommended)")
    print("-" * 40)
    engine1 = PDEEngine(params=params, method=EngineType.PDE(PDEMethod.CRANK_NICOLSON))
    price1 = engine1.price(euro_option, pricing_env)
    print(f"EngineType.PDE(PDEMethod.CRANK_NICOLSON): ${price1:.6f}")

    print("\n2. Direct Enum Pattern")
    print("-" * 40)
    engine2 = PDEEngine(params=params, method=PDEMethod.IMPLICIT_EULER)
    price2 = engine2.price(euro_option, pricing_env)
    print(f"PDEMethod.IMPLICIT_EULER: ${price2:.6f}")

    print("\n3. String Pattern (Backward Compatible)")
    print("-" * 40)
    engine3 = PDEEngine(params=params, method="explicit_euler")
    price3 = engine3.price(euro_option, pricing_env)
    print(f"String 'explicit_euler': ${price3:.6f}")

    print("\n4. Default (No method specified = Crank-Nicolson)")
    print("-" * 40)
    engine4 = PDEEngine(params=params)
    price4 = engine4.price(euro_option, pricing_env)
    print(f"Default method: ${price4:.6f}")


def demo_greeks_calculation():
    """Demonstrate Greeks calculation using PDEEngine with GreeksCalculator."""
    print_section("PART 3: Greeks Calculation via PDEEngine")

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )

    calculator = GreeksCalculator()
    pde_engine = PDEEngine(params=PDEParams(grid_size=400, time_steps=200))

    print("\n1. American Put Option Greeks (PDE Numerical Greeks)")
    print("-" * 40)
    amer_put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

    greeks_pde = calculator.calculate_numerical_greeks(
        amer_put, pricing_env, pde_engine
    )

    print(f"  Price:  ${greeks_pde['price']:10.6f}")
    print(f"  Delta:   {greeks_pde['delta']:10.6f}")
    print(f"  Gamma:   {greeks_pde['gamma']:10.6f}")
    print(f"  Vega:    {greeks_pde['vega']:10.6f}")
    print(f"  Theta:   {greeks_pde['theta']:10.6f} (per day)")
    print(f"  Rho:     {greeks_pde['rho']:10.6f}")

    print("\n2. Barrier Option Greeks (PDE Numerical Greeks)")
    print("-" * 40)
    barrier = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
    )

    greeks_barrier = calculator.calculate_numerical_greeks(
        barrier, pricing_env, pde_engine
    )

    print(f"  Price:  ${greeks_barrier['price']:10.6f}")
    print(f"  Delta:   {greeks_barrier['delta']:10.6f}")
    print(f"  Gamma:   {greeks_barrier['gamma']:10.6f}")
    print(f"  Vega:    {greeks_barrier['vega']:10.6f}")

    print("\n✓ GreeksCalculator seamlessly works with PDEEngine for all option types!")


def demo_pde_vs_analytical():
    """Compare PDE pricing with analytical methods."""
    print_section("PART 4: PDE vs Analytical Pricing Comparison")

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )

    print("\n1. European Call Option")
    print("-" * 40)
    euro_call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    bs_engine = BlackScholesEngine()
    pde_engine = PDEEngine(params=PDEParams(grid_size=500, time_steps=250))

    price_bs = bs_engine.price(euro_call, pricing_env)
    price_pde = pde_engine.price(euro_call, pricing_env)

    print(f"Black-Scholes (Analytical): ${price_bs:.8f}")
    print(f"PDE (Numerical):            ${price_pde:.8f}")
    print(f"Absolute Difference:        ${abs(price_bs - price_pde):.8f}")
    print(
        f"Relative Error:             {abs(price_bs - price_pde) / price_bs * 100:.4f}%"
    )

    print("\n2. American Put Option")
    print("-" * 40)
    amer_put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

    analytical_engine = AmericanOptionAnalyticalEngine(
        method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
    )

    price_analytical = analytical_engine.price(amer_put, pricing_env)
    price_pde = pde_engine.price(amer_put, pricing_env)

    print(f"Bjerksund-Stensland (Analytical): ${price_analytical:.8f}")
    print(f"PDE (Numerical):                  ${price_pde:.8f}")
    print(f"Absolute Difference:              ${abs(price_analytical - price_pde):.8f}")
    print(
        f"Relative Error:                   {abs(price_analytical - price_pde) / price_analytical * 100:.4f}%"
    )

    print("\n✓ PDE results closely match analytical methods with fine grids!")


def demo_solver_caching():
    """Demonstrate solver caching for performance."""
    print_section("PART 5: Solver Caching for Performance")

    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )

    pde_engine = PDEEngine(params=PDEParams(grid_size=300, time_steps=150))

    print("\nPricing multiple European options with same engine:")
    print("-" * 40)

    strikes = [90, 95, 100, 105, 110]
    for strike in strikes:
        option = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=1.0
        )
        price = pde_engine.price(option, pricing_env)
        print(f"  K={strike:3d}: ${price:.6f}")

    print(f"\nSolver cache size: {len(pde_engine._solver_cache)}")
    print(
        f"Cached solver types: {[t.__name__ for t in pde_engine._solver_cache.keys()]}"
    )
    print("\n✓ Same solver instance reused for all European options (efficient!)")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print(
        "█"
        + "  PDEEngine Demonstration - Unified PDE Pricing Interface".center(78)
        + "█"
    )
    print("█" + " " * 78 + "█")
    print("█" * 80)

    demo_pde_engine_basics()
    demo_method_selection()
    demo_greeks_calculation()
    demo_pde_vs_analytical()
    demo_solver_caching()

    print_section("SUMMARY")
    print(
        """
PDEEngine Key Features:

✓ Unified Interface: Single engine for all PDE pricing (European, American, Barrier, etc.)
✓ Automatic Dispatch: Routes to correct solver based on product type
✓ Method Selection: Supports two-level enum, direct enum, and string patterns
✓ Greeks Integration: Seamlessly works with GreeksCalculator
✓ Solver Caching: Reuses solver instances for performance
✓ Consistent API: Follows same pattern as BlackScholesEngine, AmericanOptionAnalyticalEngine

Usage Pattern:
    engine = PDEEngine(params=PDEParams(grid_size=400, time_steps=200))
    price = engine.price(any_option_product, pricing_env)
    greeks = calculator.calculate_numerical_greeks(product, pricing_env, engine)
    """
    )

    print("\n" + "█" * 80 + "\n")


if __name__ == "__main__":
    main()
