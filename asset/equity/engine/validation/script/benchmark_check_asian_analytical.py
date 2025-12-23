"""
Benchmark Check Script for Asian Option Analytical Engine
Benchmark: Monte Carlo Engine
Generated: 2024-12-23
Default Tolerance: 5%
"""
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from asset.equity.product.option import AsianOption
from asset.equity.engine.analytical import AsianOptionAnalyticalEngine
from asset.equity.engine.mc import AsianOptionMCEngine
from asset.equity.param import MCParams
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType, AveragingType, AsianStrikeType
from util.enum.engine_enums import AsianAnalyticalMethod, MonteCarloMethod

TOLERANCE = 0.05  # 5% relative error
MC_PATHS = 100000  # Number of MC paths for benchmarking


class BenchmarkResults:
    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.results = []

    def add_result(self, case_name: str, analytical: float, mc: float, mc_se: float = 0.0):
        if mc != 0:
            rel_error = abs(analytical - mc) / abs(mc)
        else:
            rel_error = abs(analytical - mc)
        passed = rel_error <= self.tolerance
        self.results.append({
            'case': case_name,
            'analytical': analytical,
            'mc': mc,
            'mc_se': mc_se,
            'rel_error': rel_error,
            'passed': passed
        })

    def summary(self):
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)

        print(f"\n{'='*90}")
        print(f"BENCHMARK CHECK SUMMARY (Tolerance: {self.tolerance*100:.1f}%)")
        print(f"{'='*90}")
        print(f"{'Case':<35} {'Analytical':>12} {'MC':>12} {'SE':>10} {'Error':>10} {'Status':>8}")
        print(f"{'-'*90}")

        for r in self.results:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"{r['case']:<35} {r['analytical']:>12.4f} {r['mc']:>12.4f} "
                  f"{r['mc_se']:>10.4f} {r['rel_error']*100:>9.2f}% {status:>8}")

        print(f"{'-'*90}")
        print(f"Passed: {passed}/{total} ({100*passed/total if total > 0 else 0:.1f}%)")

        return passed == total


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    """Helper to create pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


# ============================================================
# TEST CASES
# ============================================================

def run_benchmark_tests(results: BenchmarkResults):
    """Run all benchmark test cases."""

    # Create engines
    mc_params = MCParams(num_paths=MC_PATHS, seed=42)
    mc_engine = AsianOptionMCEngine(params=mc_params, method=MonteCarloMethod.QUASI)

    # ============================================================
    # KEMNA_VORST vs MC (Geometric) - Using geometric MC for fair comparison
    # NOTE: KEMNA_VORST assumes CONTINUOUS geometric averaging, while MC uses
    # DISCRETE observations (num_observations=12). This creates a known
    # difference between the analytical formula and MC simulation.
    # ============================================================

    analytical_kv = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)

    # Geometric ATM call
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.GEOMETRIC,  # GEOMETRIC for fair comparison
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )
    analytical_price = analytical_kv.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("KV Geometric ATM Call", analytical_price, mc_price, mc_result.std_error)

    # Geometric OTM call
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    option = AsianOption(
        strike=110.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.GEOMETRIC,  # GEOMETRIC for fair comparison
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )
    analytical_price = analytical_kv.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("KV Geometric OTM Call", analytical_price, mc_price, mc_result.std_error)

    # Geometric ITM call (additional test)
    env = create_pricing_env(spot=110.0, rate=0.05, vol=0.20)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.GEOMETRIC,  # GEOMETRIC for fair comparison
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )
    analytical_price = analytical_kv.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("KV Geometric ITM Call", analytical_price, mc_price, mc_result.std_error)

    # Geometric ATM put
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.PUT,
        averaging_type=AveragingType.GEOMETRIC,  # GEOMETRIC for fair comparison
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=1.0,
        num_observations=12,
    )
    analytical_price = analytical_kv.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("KV Geometric ATM Put", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # TURNBULL_WAKEMAN vs MC (Arithmetic)
    # ============================================================

    analytical_tw = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)

    # Arithmetic ATM call
    env = create_pricing_env(spot=100.0, rate=0.08, vol=0.20, div=0.05)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=12,
    )
    analytical_price = analytical_tw.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("TW Arithmetic ATM Call", analytical_price, mc_price, mc_result.std_error)

    # Arithmetic ITM call
    env = create_pricing_env(spot=105.0, rate=0.08, vol=0.20, div=0.05)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=12,
    )
    analytical_price = analytical_tw.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("TW Arithmetic ITM Call", analytical_price, mc_price, mc_result.std_error)

    # Arithmetic OTM call
    env = create_pricing_env(spot=95.0, rate=0.08, vol=0.20, div=0.05)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=12,
    )
    analytical_price = analytical_tw.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("TW Arithmetic OTM Call", analytical_price, mc_price, mc_result.std_error)

    # Arithmetic ATM put
    env = create_pricing_env(spot=100.0, rate=0.08, vol=0.20, div=0.05)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.PUT,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=12,
    )
    analytical_price = analytical_tw.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("TW Arithmetic ATM Put", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # LEVY vs MC (Arithmetic)
    # ============================================================

    analytical_levy = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.LEVY)

    # Levy ATM call (b != 0 required)
    env = create_pricing_env(spot=100.0, rate=0.07, vol=0.20, div=0.09)  # b = -0.02
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=12,
    )
    analytical_price = analytical_levy.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("Levy Arithmetic ATM Call", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # CURRAN vs MC (Arithmetic)
    # ============================================================

    analytical_curran = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.CURRAN)

    # Curran ATM call
    env = create_pricing_env(spot=100.0, rate=0.08, vol=0.20, div=0.05)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=27,
    )
    analytical_price = analytical_curran.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("Curran Arithmetic ATM Call", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # DISCRETE_HHM vs MC (Discrete Arithmetic)
    # ============================================================

    analytical_hhm = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.DISCRETE_HHM)

    # HHM ATM call
    env = create_pricing_env(spot=100.0, rate=0.08, vol=0.20, div=0.05)
    option = AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FIXED,
        maturity=0.5,
        num_observations=27,
    )
    analytical_price = analytical_hhm.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("HHM Arithmetic ATM Call", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # Floating Strike Tests
    # ============================================================

    # Floating call
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    option = AsianOption(
        strike=0.0,
        option_type=OptionType.CALL,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FLOATING,
        maturity=1.0,
        num_observations=12,
    )
    analytical_price = analytical_tw.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("TW Floating Call", analytical_price, mc_price, mc_result.std_error)

    # Floating put
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
    option = AsianOption(
        strike=0.0,
        option_type=OptionType.PUT,
        averaging_type=AveragingType.ARITHMETIC,
        asian_strike_type=AsianStrikeType.FLOATING,
        maturity=1.0,
        num_observations=12,
    )
    analytical_price = analytical_tw.price(option, env)
    mc_price = mc_engine.price(option, env)
    mc_result = mc_engine.get_last_result()
    results.add_result("TW Floating Put", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # Different Maturities
    # ============================================================

    for T in [0.25, 0.5, 1.0, 2.0]:
        env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20)
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=T,
            num_observations=12,
        )
        analytical_price = analytical_tw.price(option, env)
        mc_price = mc_engine.price(option, env)
        mc_result = mc_engine.get_last_result()
        results.add_result(f"TW Maturity T={T}", analytical_price, mc_price, mc_result.std_error)

    # ============================================================
    # Different Volatilities
    # ============================================================

    for vol in [0.10, 0.20, 0.40]:
        env = create_pricing_env(spot=100.0, rate=0.05, vol=vol)
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        analytical_price = analytical_tw.price(option, env)
        mc_price = mc_engine.price(option, env)
        mc_result = mc_engine.get_last_result()
        results.add_result(f"TW Volatility σ={vol:.2f}", analytical_price, mc_price, mc_result.std_error)


if __name__ == "__main__":
    print(f"Benchmarking Asian Option Analytical Engine vs Monte Carlo")
    print(f"MC Paths: {MC_PATHS:,}")
    print(f"Tolerance: {TOLERANCE*100:.1f}%")
    print("="*90)

    results = BenchmarkResults(tolerance=TOLERANCE)
    run_benchmark_tests(results)

    success = results.summary()
    sys.exit(0 if success else 1)
