"""
Benchmark Check Script for Digital Option Analytical Engine
Benchmark: Monte Carlo Engine (DigitalOptionMCEngine)
Generated: 2024-12-25
Default Tolerance: 5%
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.engine.analytical.digital_option_engine import DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.mc.digital_option_mc_engine import DigitalOptionMCEngine
from quantark.asset.equity.param import MCParams
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from datetime import datetime

TOLERANCE = 0.05  # 5% relative error


class BenchmarkResults:
    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.results = []

    def add_result(self, case_name: str, analytical: float, mc: float, mc_se: float = None):
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

        print(f"\n{'='*80}")
        print(f"BENCHMARK CHECK SUMMARY (Tolerance: {self.tolerance*100:.1f}%)")
        print(f"{'='*80}")
        print(f"{'Case':<35} {'Analytical':>12} {'MC':>12} {'SE':>10} {'Error':>10} {'Status':>8}")
        print(f"{'-'*80}")

        for r in self.results:
            status = "PASS" if r['passed'] else "FAIL"
            se_str = f"{r['mc_se']:.4f}" if r['mc_se'] else "N/A"
            print(f"{r['case']:<35} {r['analytical']:>12.6f} {r['mc']:>12.6f} "
                  f"{se_str:>10} {r['rel_error']*100:>9.3f}% {status:>8}")

        print(f"{'-'*80}")
        print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")

        if passed < total:
            print(f"\nFailed cases:")
            for r in self.results:
                if not r['passed']:
                    print(f"  - {r['case']}: Analytical={r['analytical']:.6f}, "
                          f"MC={r['mc']:.6f}, Error={r['rel_error']*100:.2f}%")

        return passed == total


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    """Helper to create pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def create_digital_call(K=100.0, payout=10.0, T=1.0):
    """Helper to create digital call option."""
    return CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.CALL,
        maturity=T,
    )


def create_digital_put(K=100.0, payout=10.0, T=1.0):
    """Helper to create digital put option."""
    return CashOrNothingDigitalOption(
        strike=K,
        payout=payout,
        option_type=OptionType.PUT,
        maturity=T,
    )


# ============================================================
# TEST CASES
# ============================================================

def run_benchmark_tests(results: BenchmarkResults, use_qmc: bool = False):
    """Run all benchmark test cases."""

    analytical_engine = DigitalOptionAnalyticalEngine()

    mc_method = MonteCarloMethod.QUASI if use_qmc else MonteCarloMethod.PSEUDO
    num_paths = 100000 if use_qmc else 200000

    mc_params = MCParams(
        num_paths=num_paths,
        time_steps=252,
        seed=42
    )
    mc_engine = DigitalOptionMCEngine(params=mc_params, method=mc_method)

    method_suffix = " (QMC)" if use_qmc else " (MC)"

    # Base case: ATM Call
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call T=1Y{method_suffix}", analytical_price, mc_price, mc_se)

    # ATM Put
    product = create_digital_put(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Put T=1Y{method_suffix}", analytical_price, mc_price, mc_se)

    # ITM Call
    env = create_pricing_env(spot=110.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ITM Call (S=110, K=100){method_suffix}", analytical_price, mc_price, mc_se)

    # OTM Call
    env = create_pricing_env(spot=90.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"OTM Call (S=90, K=100){method_suffix}", analytical_price, mc_price, mc_se)

    # ITM Put
    env = create_pricing_env(spot=90.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_put(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ITM Put (S=90, K=100){method_suffix}", analytical_price, mc_price, mc_se)

    # OTM Put
    env = create_pricing_env(spot=110.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_put(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"OTM Put (S=110, K=100){method_suffix}", analytical_price, mc_price, mc_se)

    # Different maturity
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=100.0, payout=10.0, T=0.25)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call T=0.25Y{method_suffix}", analytical_price, mc_price, mc_se)

    # Longer maturity
    product = create_digital_call(K=100.0, payout=10.0, T=2.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call T=2Y{method_suffix}", analytical_price, mc_price, mc_se)

    # Different volatility
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.10, div=0.02)
    product = create_digital_call(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call Low Vol (10%){method_suffix}", analytical_price, mc_price, mc_se)

    # High volatility
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.40, div=0.02)
    product = create_digital_call(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call High Vol (40%){method_suffix}", analytical_price, mc_price, mc_se)

    # Zero rate
    env = create_pricing_env(spot=100.0, rate=0.0, vol=0.20, div=0.0)
    product = create_digital_call(K=100.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call Zero Rate{method_suffix}", analytical_price, mc_price, mc_se)

    # Different payout
    env = create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=100.0, payout=25.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"ATM Call Payout=25{method_suffix}", analytical_price, mc_price, mc_se)

    # Deep ITM
    env = create_pricing_env(spot=130.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=80.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"Deep ITM Call{method_suffix}", analytical_price, mc_price, mc_se)

    # Deep OTM
    env = create_pricing_env(spot=70.0, rate=0.05, vol=0.20, div=0.02)
    product = create_digital_call(K=130.0, payout=10.0, T=1.0)
    analytical_price = analytical_engine.price(product, env)
    mc_price = mc_engine.price(product, env)
    mc_se = mc_engine.get_last_std_error()
    results.add_result(f"Deep OTM Call{method_suffix}", analytical_price, mc_price, mc_se)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = BenchmarkResults(tolerance=TOLERANCE)

    print("\n" + "="*80)
    print("DIGITAL OPTION ANALYTICAL ENGINE - BENCHMARK CHECKS")
    print("="*80)

    print("\n[1/2] Running standard Monte Carlo benchmark...")
    run_benchmark_tests(results, use_qmc=False)

    print("\n[2/2] Running Quasi-Monte Carlo benchmark...")
    run_benchmark_tests(results, use_qmc=True)

    success = results.summary()
    sys.exit(0 if success else 1)
