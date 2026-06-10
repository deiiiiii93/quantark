"""
Benchmark Check Script for American Option PDE Solver
Benchmark: Monte Carlo Engine
Generated: 2025-02-14
Default Tolerance: 5%
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.product.option import AmericanOption
from quantark.asset.equity.engine.pde import AmericanPDESolver
from quantark.asset.equity.engine.mc import AmericanOptionMCEngine
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod

TOLERANCE = 0.05  # 5% relative error
MC_PATHS = 100000
MC_STEPS = 200


class BenchmarkResults:
    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.results = []

    def add_result(self, case_name: str, pde: float, mc: float, mc_se: float = 0.0):
        if mc != 0:
            rel_error = abs(pde - mc) / abs(mc)
        else:
            rel_error = abs(pde - mc)
        passed = rel_error <= self.tolerance
        self.results.append({
            "case": case_name,
            "pde": pde,
            "mc": mc,
            "mc_se": mc_se,
            "rel_error": rel_error,
            "passed": passed,
        })

    def summary(self):
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        print(f"\n{'='*90}")
        print(f"BENCHMARK CHECK SUMMARY (Tolerance: {self.tolerance*100:.1f}%)")
        print(f"{'='*90}")
        print(f"{'Case':<40} {'PDE':>12} {'MC':>12} {'SE':>10} {'Error':>10} {'Status':>8}")
        print(f"{'-'*90}")

        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"{r['case']:<40} {r['pde']:>12.4f} {r['mc']:>12.4f} "
                f"{r['mc_se']:>10.4f} {r['rel_error']*100:>9.2f}% {status:>8}"
            )

        print(f"{'-'*90}")
        print(f"Passed: {passed}/{total} ({100*passed/total if total > 0 else 0:.1f}%)")

        return passed == total


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def run_benchmark_tests(results: BenchmarkResults):
    mc_params = MCParams(num_paths=MC_PATHS, time_steps=MC_STEPS, seed=42)
    mc_engine = AmericanOptionMCEngine(params=mc_params, method=MonteCarloMethod.QUASI)

    pde_params = PDEParams(grid_size=250, time_steps=250)
    pde_engine = AmericanPDESolver(params=pde_params)

    test_cases = [
        ("ATM Call (q=2%)", 100.0, 100.0, 1.0, 0.05, 0.20, 0.02, OptionType.CALL),
        ("ATM Put (q=2%)", 100.0, 100.0, 1.0, 0.05, 0.20, 0.02, OptionType.PUT),
        ("ITM Put (q=2%)", 90.0, 110.0, 1.0, 0.05, 0.25, 0.02, OptionType.PUT),
        ("OTM Call (q=0%)", 90.0, 110.0, 1.0, 0.03, 0.20, 0.00, OptionType.CALL),
    ]

    for case in test_cases:
        name, spot, strike, maturity, rate, vol, div, opt_type = case
        env = create_pricing_env(spot=spot, rate=rate, vol=vol, div=div)
        option = AmericanOption(
            strike=strike,
            option_type=opt_type,
            maturity=maturity,
        )

        pde_price = pde_engine.price(option, env)
        mc_price = mc_engine.price(option, env)
        mc_result = mc_engine.get_last_result()
        mc_se = mc_result.std_error if mc_result is not None else 0.0
        results.add_result(name, pde_price, mc_price, mc_se)


if __name__ == "__main__":
    results = BenchmarkResults(tolerance=TOLERANCE)
    run_benchmark_tests(results)
    success = results.summary()
    sys.exit(0 if success else 1)
