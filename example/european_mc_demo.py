"""
Demonstration of European Option Monte Carlo Pricing.

This script demonstrates:
1. Pricing European options using different Monte Carlo methods
2. Comparing MC, QMC, and RQMC convergence properties
3. Validating against Black-Scholes analytical prices
4. Analyzing variance reduction techniques
5. Understanding standard error and convergence diagnostics
"""

import sys
from pathlib import Path
import time


from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.param import MCParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod, EngineType
from datetime import datetime


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


def demo_basic_mc_pricing():
    """Demonstrate basic Monte Carlo pricing."""
    print_section("BASIC MONTE CARLO PRICING")

    print("\n1. Market Data Setup")
    print("-" * 40)
    spot_quote = SpotQuote(spot=100.0, asset_name="AAPL")
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    print_market_data(pricing_env)

    print("\n2. Option Specification")
    print("-" * 40)
    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    print(f"Option Type:  {call.option_type}")
    print(f"Strike:       ${call.strike:.2f}")
    print(f"Maturity:     {call.maturity:.2f} years")

    print("\n3. Pricing with Different Methods")
    print("-" * 40)

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)
    print(f"\nBlack-Scholes Analytical Price: ${bs_price:.6f}")
    print("(This is the 'true' price we'll compare against)")

    params = MCParams(num_paths=50000, time_steps=252, seed=42)

    print(f"\nMonte Carlo Parameters:")
    print(f"  Number of paths:  {params.num_paths:,}")
    print(f"  Time steps:       {params.time_steps}")
    print(f"  Random seed:      {params.seed}")

    print("\n" + "-" * 80)

    methods = [
        (MonteCarloMethod.PSEUDO, "Pseudorandom Monte Carlo (MC)"),
        (MonteCarloMethod.QUASI, "Quasi-Monte Carlo with Sobol (QMC)"),
    ]

    for method, description in methods:
        print(f"\n{description}")
        print("  " + "-" * 76)

        engine = EuropeanMCEngine(params=params, method=method)

        start_time = time.time()
        mc_price = engine.price(call, pricing_env)
        elapsed = time.time() - start_time

        std_error = engine.get_last_std_error()
        error_vs_bs = abs(mc_price - bs_price)

        print(f"  Price:              ${mc_price:.6f}")
        print(f"  Standard Error:     ${std_error:.6f}")
        print(f"  95% Confidence:     [${mc_price - 1.96*std_error:.6f}, ${mc_price + 1.96*std_error:.6f}]")
        print(f"  Error vs BS:        ${error_vs_bs:.6f}")
        print(f"  Relative Error:     {error_vs_bs/bs_price*100:.4f}%")
        print(f"  Computation Time:   {elapsed:.3f} seconds")


def demo_rqmc_adaptive_batching():
    """Demonstrate RQMC with adaptive batching."""
    print_section("RANDOMIZED QMC WITH ADAPTIVE BATCHING")

    spot_quote = SpotQuote(spot=100.0)
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    print("\nRQMC runs multiple independent batches of scrambled Sobol sequences.")
    print("It adaptively stops when the target standard error is achieved.\n")

    params = MCParams(num_paths=8192, time_steps=252, seed=42)
    params.max_batches = 32
    params.target_std = 0.005
    params.min_batches = 4

    print(f"Configuration:")
    print(f"  Paths per batch:      {params.num_paths:,}")
    print(f"  Time steps:           {params.time_steps}")
    print(f"  Max batches:          {params.max_batches}")
    print(f"  Target std error:     ${params.target_std:.6f}")
    print(f"  Min batches:          {params.min_batches}")

    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI)

    start_time = time.time()
    rqmc_price = engine.price(call, pricing_env)
    elapsed = time.time() - start_time

    result = engine.get_last_rqmc_result()

    print(f"\nResults:")
    print(f"  Price:                ${rqmc_price:.6f}")
    print(f"  Standard Error:       ${result.std_error:.6f}")
    print(f"  95% Confidence:       [${rqmc_price - 1.96*result.std_error:.6f}, ${rqmc_price + 1.96*result.std_error:.6f}]")
    print(f"  Batches Used:         {result.batches_used}")
    print(f"  Total Paths:          {result.total_paths:,}")
    print(f"  Error vs BS:          ${abs(rqmc_price - bs_price):.6f}")
    print(f"  Relative Error:       {abs(rqmc_price - bs_price)/bs_price*100:.4f}%")
    print(f"  Computation Time:     {elapsed:.3f} seconds")

    print(f"\nBatch Statistics:")
    print(f"  Mean of batch means:  ${result.batch_means.mean():.6f}")
    print(f"  Std of batch means:   ${result.batch_means.std():.6f}")
    print(f"  Min batch mean:       ${result.batch_means.min():.6f}")
    print(f"  Max batch mean:       ${result.batch_means.max():.6f}")


def demo_convergence_comparison():
    """Compare convergence properties of MC vs QMC."""
    print_section("CONVERGENCE COMPARISON: MC vs QMC")

    spot_quote = SpotQuote(spot=100.0)
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    print(f"\nBlack-Scholes Reference Price: ${bs_price:.6f}\n")

    path_counts = [1024, 2048, 4096, 8192, 16384, 32768]

    print(f"{'Paths':>10} {'MC Price':>12} {'MC Error':>12} {'QMC Price':>12} {'QMC Error':>12}")
    print("-" * 60)

    for num_paths in path_counts:
        params_mc = MCParams(num_paths=num_paths, time_steps=252, seed=42)
        engine_mc = EuropeanMCEngine(params=params_mc, method=MonteCarloMethod.PSEUDO)
        mc_price = engine_mc.price(call, pricing_env)
        mc_error = abs(mc_price - bs_price)

        params_qmc = MCParams(num_paths=num_paths, time_steps=252, seed=42)
        engine_qmc = EuropeanMCEngine(params=params_qmc, method=MonteCarloMethod.QUASI)
        qmc_price = engine_qmc.price(call, pricing_env)
        qmc_error = abs(qmc_price - bs_price)

        print(f"{num_paths:>10,} ${mc_price:>11.6f} ${mc_error:>11.6f} ${qmc_price:>11.6f} ${qmc_error:>11.6f}")

    print("\nObservation:")
    print("  - MC converges at O(1/√N) rate")
    print("  - QMC converges at O(1/N) or faster for low-dimensional problems")
    print("  - QMC typically shows more consistent improvement with path count")


def demo_variance_reduction():
    """Demonstrate variance reduction with antithetic variates."""
    print_section("VARIANCE REDUCTION WITH ANTITHETIC VARIATES")

    spot_quote = SpotQuote(spot=100.0)
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    bs_engine = BlackScholesEngine()
    bs_price = bs_engine.price(call, pricing_env)

    print("\nAntithetic variates use negatively correlated paths: Z and -Z")
    print("This reduces variance, especially for smooth payoffs.\n")

    num_paths = 20000

    print(f"Configuration: {num_paths:,} paths\n")

    params_without = MCParams(num_paths=num_paths, time_steps=252, seed=42, use_antithetic=False)
    engine_without = EuropeanMCEngine(params=params_without, method=MonteCarloMethod.PSEUDO)

    print("1. Standard Monte Carlo (no variance reduction)")
    print("   " + "-" * 76)
    start = time.time()
    price_without = engine_without.price(call, pricing_env)
    time_without = time.time() - start
    std_error_without = engine_without.get_last_std_error()

    print(f"   Price:           ${price_without:.6f}")
    print(f"   Standard Error:  ${std_error_without:.6f}")
    print(f"   Error vs BS:     ${abs(price_without - bs_price):.6f}")
    print(f"   Time:            {time_without:.3f} seconds")

    params_with = MCParams(num_paths=num_paths, time_steps=252, seed=42, use_antithetic=True)
    engine_with = EuropeanMCEngine(params=params_with, method=MonteCarloMethod.PSEUDO)

    print("\n2. Monte Carlo with Antithetic Variates")
    print("   " + "-" * 76)
    start = time.time()
    price_with = engine_with.price(call, pricing_env)
    time_with = time.time() - start
    std_error_with = engine_with.get_last_std_error()

    print(f"   Price:           ${price_with:.6f}")
    print(f"   Standard Error:  ${std_error_with:.6f}")
    print(f"   Error vs BS:     ${abs(price_with - bs_price):.6f}")
    print(f"   Time:            {time_with:.3f} seconds")

    print("\n3. Variance Reduction Summary")
    print("   " + "-" * 76)
    variance_reduction = std_error_without / std_error_with
    efficiency_gain = (std_error_without / std_error_with) ** 2

    print(f"   Standard Error Ratio:     {variance_reduction:.4f}x")
    print(f"   Variance Reduction:       {variance_reduction**2:.4f}x")
    print(f"   Efficiency Gain:          {efficiency_gain:.4f}x")
    print(f"\n   (Efficiency gain measures the factor by which fewer paths are needed")
    print(f"    with antithetic variates to achieve the same accuracy)")


def demo_put_call_parity_mc():
    """Verify put-call parity with Monte Carlo pricing."""
    print_section("PUT-CALL PARITY WITH MONTE CARLO")

    import math

    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02

    spot_quote = SpotQuote(spot=S)
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=r)
    div_yield = ContinuousDividendYield(div_yield=q)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)

    print("\nPut-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)\n")

    params = MCParams(num_paths=100000, time_steps=252, seed=42)
    engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.PSEUDO)

    call_price = engine.price(call, pricing_env)
    put_price = engine.price(put, pricing_env)

    lhs = call_price - put_price
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

    print(f"Monte Carlo Pricing Results:")
    print(f"  Call Price (C):  ${call_price:.6f}")
    print(f"  Put Price (P):   ${put_price:.6f}")

    print(f"\nParity Check:")
    print(f"  LHS (C - P):              ${lhs:.6f}")
    print(f"  RHS (S*e^(-qT) - K*e^(-rT)): ${rhs:.6f}")
    print(f"  Difference:               ${abs(lhs - rhs):.6f}")

    if abs(lhs - rhs) < 0.05:
        print("\n  ✓ Put-Call Parity holds (within MC error tolerance)!")
    else:
        print("\n  ⚠ Large deviation - may need more paths")

    bs_engine = BlackScholesEngine()
    bs_call = bs_engine.price(call, pricing_env)
    bs_put = bs_engine.price(put, pricing_env)

    print(f"\nComparison with Black-Scholes:")
    print(f"  Call: MC=${call_price:.6f}, BS=${bs_call:.6f}, Diff=${abs(call_price - bs_call):.6f}")
    print(f"  Put:  MC=${put_price:.6f}, BS=${bs_put:.6f}, Diff=${abs(put_price - bs_put):.6f}")


def demo_two_level_enum_usage():
    """Demonstrate two-level enum pattern for method selection."""
    print_section("METHOD SELECTION PATTERNS")

    spot_quote = SpotQuote(spot=100.0)
    vol_surface = FlatVolSurface(volatility=0.20)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=datetime(2024, 1, 1),
    )

    call = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    params = MCParams(num_paths=10000, time_steps=100, seed=42)

    print("\nQuantArk supports three equivalent ways to specify the MC method:\n")

    print("1. Two-Level Enum Pattern (Preferred)")
    print("   " + "-" * 76)
    print("   engine = EuropeanMCEngine(")
    print("       params=params,")
    print("       method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)")
    print("   )")

    engine1 = EuropeanMCEngine(
        params=params,
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
    )
    price1 = engine1.price(call, pricing_env)
    print(f"   Price: ${price1:.6f}")

    print("\n2. Direct Method Enum")
    print("   " + "-" * 76)
    print("   engine = EuropeanMCEngine(")
    print("       params=params,")
    print("       method=MonteCarloMethod.QUASI")
    print("   )")

    engine2 = EuropeanMCEngine(params=params, method=MonteCarloMethod.QUASI)
    price2 = engine2.price(call, pricing_env)
    print(f"   Price: ${price2:.6f}")

    print("\n3. String (Backward Compatibility)")
    print("   " + "-" * 76)
    print("   engine = EuropeanMCEngine(")
    print("       params=params,")
    print("       method='quasi'")
    print("   )")

    engine3 = EuropeanMCEngine(params=params, method="quasi")
    price3 = engine3.price(call, pricing_env)
    print(f"   Price: ${price3:.6f}")

    print("\nAll three methods are equivalent and produce identical results.")


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  QUANTARK - European Option Monte Carlo Pricing Demo".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    try:
        demo_basic_mc_pricing()
        demo_rqmc_adaptive_batching()
        demo_convergence_comparison()
        demo_variance_reduction()
        demo_put_call_parity_mc()
        demo_two_level_enum_usage()

        print_section("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("\nKey Takeaways:")
        print("  • QMC (Sobol) converges faster than standard MC for European options")
        print("  • RQMC provides reliable error estimates via batch-to-batch variance")
        print("  • Antithetic variates reduce variance by 2x or more for smooth payoffs")
        print("  • All MC methods converge to the analytical Black-Scholes price")
        print("  • QuantArk supports flexible method selection via enums or strings")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
