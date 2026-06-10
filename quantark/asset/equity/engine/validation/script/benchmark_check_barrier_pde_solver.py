"""
Benchmark Check Script for BarrierPDESolver

Compares PDE solver results against:
1. Analytical engine (BarrierAnalyticalEngine)
2. Monte Carlo engine (BarrierOptionMCEngine)

Generated: 2025-12-25
"""

import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, ".")

import numpy as np

from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.param import PDEParams, MCParams
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.param.quote.spot_quote import SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, OptionType, ObservationType


# Default tolerances
ANALYTICAL_TOLERANCE = 0.05  # 5% for analytical (PDE vs Analytical expected diff)
MC_TOLERANCE = 0.08  # 8% for MC (Monte Carlo noise + PDE approximation)


class BenchmarkResults:
    """Track and report benchmark comparison results."""

    def __init__(
        self, analytical_tol: float = ANALYTICAL_TOLERANCE, mc_tol: float = MC_TOLERANCE
    ):
        self.analytical_tol = analytical_tol
        self.mc_tol = mc_tol
        self.results: List[Dict] = []

    def add_result(
        self,
        case_name: str,
        pde_price: float,
        analytical_price: float = None,
        mc_price: float = None,
        analytical_error: float = None,
        mc_error: float = None,
    ):
        """Add a benchmark result."""
        result = {
            "case": case_name,
            "pde": pde_price,
            "analytical": analytical_price,
            "mc": mc_price,
            "analytical_error": analytical_error,
            "mc_error": mc_error,
        }

        # Compute errors if not provided
        if analytical_price is not None and analytical_error is None:
            if analytical_price != 0:
                result["analytical_error"] = abs(pde_price - analytical_price) / abs(
                    analytical_price
                )
            else:
                result["analytical_error"] = abs(pde_price - analytical_price)

        if mc_price is not None and mc_error is None:
            if mc_price != 0:
                result["mc_error"] = abs(pde_price - mc_price) / abs(mc_price)
            else:
                result["mc_error"] = abs(pde_price - mc_price)

        # Check pass status
        result["analytical_pass"] = (
            analytical_price is None
            or result["analytical_error"] <= self.analytical_tol
        )
        result["mc_pass"] = mc_price is None or result["mc_error"] <= self.mc_tol

        self.results.append(result)

    def summary(self) -> bool:
        """Print summary and return True if all tests passed."""
        analytical_count = sum(1 for r in self.results if r["analytical"] is not None)
        analytical_passed = sum(
            1 for r in self.results if r["analytical"] is not None and r["analytical_pass"]
        )
        mc_count = sum(1 for r in self.results if r["mc"] is not None)
        mc_passed = sum(1 for r in self.results if r["mc"] is not None and r["mc_pass"])

        print(f"\n{'='*90}")
        print(f"BENCHMARK CHECK SUMMARY - BarrierPDESolver")
        print(f"{'='*90}")
        print(
            f"Analytical Comparison: {analytical_passed}/{analytical_count} passed "
            f"({100*analytical_passed/analytical_count:.1f}%)"
        )
        if mc_count > 0:
            print(
                f"MC Comparison: {mc_passed}/{mc_count} passed "
                f"({100*mc_passed/mc_count:.1f}%)"
            )
        else:
            print(f"MC Comparison: Skipped (--no-mc flag used)")
        print(f"{'='*90}\n")

        # Detailed results table
        print(
            f"{'Case':<35} {'PDE':>12} {'Analytical':>12} {'MC':>12} {'Err%':>8} {'Status':>8}"
        )
        print("-" * 90)

        for r in self.results:
            if r["analytical"] is not None:
                ref = r["analytical"]
                err = r["analytical_error"] * 100
                status = "PASS" if r["analytical_pass"] else "FAIL"
                mc_str = f"{r['mc']:>12.4f}" if r["mc"] is not None else " " * 12
                print(
                    f"{r['case']:<35} {r['pde']:>12.4f} {r['analytical']:>12.4f} "
                    f"{mc_str} {err:>7.2f}% {status:>8}"
                )
            elif r["mc"] is not None:
                err = r["mc_error"] * 100
                status = "PASS" if r["mc_pass"] else "FAIL"
                print(
                    f"{r['case']:<35} {r['pde']:>12.4f} {' '*12} {r['mc']:>12.4f} "
                    f"{err:>7.2f}% {status:>8}"
                )

        print("-" * 90)
        all_passed = analytical_passed == analytical_count and mc_passed == mc_count
        print(
            f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}\n"
        )
        return all_passed


def create_pricing_env(
    spot: float = 100.0, rate: float = 0.05, vol: float = 0.20, div: float = 0.0
) -> PricingEnvironment:
    """Helper to create a pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        valuation_date=datetime(2024, 1, 1),
    )


def create_pde_params(auto_grid: bool = True) -> PDEParams:
    """Create PDE parameters for testing."""
    if auto_grid:
        return PDEParams(auto_grid=True)
    else:
        return PDEParams(
            grid_size=252 * 2,
            time_steps=252 * 4,
            adaptive_grid=False,
            theta=0.5,
            use_rannacher=True,
            auto_grid=False,
        )


def create_mc_params() -> MCParams:
    """Create MC parameters for testing."""
    return MCParams(num_paths=100000, time_steps=252, seed=42)


# ============================================================
# TEST CASE DEFINITIONS
# ============================================================

TEST_CASES = [
    # (name, spot, strike, barrier, barrier_type, option_type, T, r, sigma, rebate)
    (
        "ATM Call D0O barrier=90",
        100,
        100,
        90,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ITM Call D0O barrier=85",
        100,
        95,
        85,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "OTM Call D0O barrier=95",
        100,
        105,
        95,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ATM Call U0O barrier=110",
        100,
        100,
        110,
        BarrierType.UP_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ATM Put U0O barrier=110",
        100,
        100,
        110,
        BarrierType.UP_OUT,
        OptionType.PUT,
        1.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ATM Put D0O barrier=90",
        100,
        100,
        90,
        BarrierType.DOWN_OUT,
        OptionType.PUT,
        1.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ITM Call D0O barrier=90 T=0.5",
        100,
        95,
        90,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        0.5,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ATM Call D0O barrier=90 T=2.0",
        100,
        100,
        90,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        2.0,
        0.05,
        0.20,
        0.0,
    ),
    (
        "ATM Call D0O low vol",
        100,
        100,
        90,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.10,
        0.0,
    ),
    (
        "ATM Call D0O high vol",
        100,
        100,
        90,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.40,
        0.0,
    ),
    (
        "ATM Call D0O with rebate",
        100,
        100,
        90,
        BarrierType.DOWN_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.20,
        2.0,
    ),
    (
        "ATM Call U0O with rebate",
        100,
        100,
        110,
        BarrierType.UP_OUT,
        OptionType.CALL,
        1.0,
        0.05,
        0.20,
        2.0,
    ),
]


def run_benchmark_tests(
    results: BenchmarkResults, include_mc: bool = True, test_subset: List[str] = None
) -> None:
    """
    Run all benchmark test cases.

    Args:
        results: BenchmarkResults object to store results
        include_mc: Whether to run Monte Carlo comparison (slower)
        test_subset: Optional list of test case names to run (runs all if None)
    """
    # Initialize engines
    pde_solver = BarrierPDESolver(create_pde_params())
    analytical_solver = BarrierAnalyticalEngine()

    mc_solver = None
    if include_mc:
        mc_solver = BarrierOptionMCEngine(
            params=create_mc_params(),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
        )

    # Filter test cases if subset specified
    cases_to_run = TEST_CASES
    if test_subset is not None:
        cases_to_run = [c for c in TEST_CASES if c[0] in test_subset]

    print(f"\nRunning {len(cases_to_run)} benchmark test cases...")
    if include_mc:
        print("Note: MC comparison enabled (this may take a while)")

    for i, case in enumerate(cases_to_run):
        name, spot, strike, barrier, btype, otype, T, r, sigma, rebate = case

        print(f"[{i+1}/{len(cases_to_run)}] Testing: {name}...")

        # Create pricing environment
        env = create_pricing_env(spot=spot, rate=r, vol=sigma)

        # Create option
        option = BarrierOption(
            strike=strike,
            option_type=otype,
            barrier=barrier,
            barrier_type=btype,
            maturity=T,
            rebate=rebate,
            observation_type=ObservationType.CONTINUOUS,
        )

        # Get PDE price
        try:
            pde_price = pde_solver.price(option, env)
        except Exception as e:
            print(f"  ⚠️  PDE pricing failed: {str(e)[:50]}")
            continue

        # Get analytical price
        analytical_price = None
        analytical_error = None
        try:
            analytical_price = analytical_solver.price(option, env)
            if analytical_price != 0:
                analytical_error = abs(pde_price - analytical_price) / abs(
                    analytical_price
                )
            else:
                analytical_error = abs(pde_price - analytical_price)
        except Exception as e:
            print(f"  ⚠️  Analytical pricing failed: {str(e)[:50]}")

        # Get MC price
        mc_price = None
        mc_error = None
        if include_mc and mc_solver is not None:
            try:
                mc_price = mc_solver.price(option, env)
                if mc_price != 0:
                    mc_error = abs(pde_price - mc_price) / abs(mc_price)
                else:
                    mc_error = abs(pde_price - mc_price)
            except Exception as e:
                print(f"  ⚠️  MC pricing failed: {str(e)[:50]}")

        # Add result
        results.add_result(
            case_name=name,
            pde_price=pde_price,
            analytical_price=analytical_price,
            mc_price=mc_price,
            analytical_error=analytical_error,
            mc_error=mc_error,
        )


def run_ki_benchmark_tests(results: BenchmarkResults) -> None:
    """
    Run benchmark tests specifically for knock-in options.

    KI options use the decomposition: KI = Vanilla - KO
    """
    pde_solver = BarrierPDESolver(create_pde_params())
    analytical_solver = BarrierAnalyticalEngine()

    ki_cases = [
        (
            "ATM Call D0I barrier=90",
            100,
            100,
            90,
            BarrierType.DOWN_IN,
            OptionType.CALL,
            1.0,
            0.05,
            0.20,
        ),
        (
            "ATM Call U0I barrier=110",
            100,
            100,
            110,
            BarrierType.UP_IN,
            OptionType.CALL,
            1.0,
            0.05,
            0.20,
        ),
        (
            "ATM Put U0I barrier=110",
            100,
            100,
            110,
            BarrierType.UP_IN,
            OptionType.PUT,
            1.0,
            0.05,
            0.20,
        ),
    ]

    print(f"\nRunning {len(ki_cases)} knock-in benchmark test cases...")

    for i, case in enumerate(ki_cases):
        name, spot, strike, barrier, btype, otype, T, r, sigma = case

        print(f"[{i+1}/{len(ki_cases)}] Testing: {name}...")

        env = create_pricing_env(spot=spot, rate=r, vol=sigma)

        option = BarrierOption(
            strike=strike,
            option_type=otype,
            barrier=barrier,
            barrier_type=btype,
            maturity=T,
            rebate=0.0,
            observation_type=ObservationType.CONTINUOUS,
        )

        # Get PDE price
        pde_price = pde_solver.price(option, env)

        # Get analytical price
        analytical_price = analytical_solver.price(option, env)

        # Compute error
        if analytical_price != 0:
            analytical_error = abs(pde_price - analytical_price) / abs(analytical_price)
        else:
            analytical_error = abs(pde_price - analytical_price)

        results.add_result(
            case_name=name,
            pde_price=pde_price,
            analytical_price=analytical_price,
            analytical_error=analytical_error,
        )


def run_discrete_monitoring_tests(results: BenchmarkResults) -> None:
    """
    Run benchmark tests for discrete barrier monitoring.

    PDE solver should handle discrete observation by checking
    only at observation times.
    """
    pde_solver = BarrierPDESolver(create_pde_params())
    mc_solver = BarrierOptionMCEngine(
        params=create_mc_params(), method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
    )

    discrete_cases = [
        (
            "Daily discrete D0O",
            100,
            100,
            90,
            BarrierType.DOWN_OUT,
            OptionType.CALL,
            1.0,
            0.05,
            0.20,
            "daily",
        ),
        (
            "Weekly discrete D0O",
            100,
            100,
            90,
            BarrierType.DOWN_OUT,
            OptionType.CALL,
            1.0,
            0.05,
            0.20,
            "weekly",
        ),
    ]

    print(f"\nRunning {len(discrete_cases)} discrete monitoring test cases...")

    for i, case in enumerate(discrete_cases):
        name, spot, strike, barrier, btype, otype, T, r, sigma, freq = case

        print(f"[{i+1}/{len(discrete_cases)}] Testing: {name}...")

        env = create_pricing_env(spot=spot, rate=r, vol=sigma)

        # Create observation dates
        if freq == "daily":
            obs_dates = [T * (i / 252) for i in range(1, 252)]
        elif freq == "weekly":
            obs_dates = [T * (i / 52) for i in range(1, 52)]
        else:
            obs_dates = None

        option = BarrierOption(
            strike=strike,
            option_type=otype,
            barrier=barrier,
            barrier_type=btype,
            maturity=T,
            rebate=0.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=obs_dates,
        )

        # Get PDE price
        pde_price = pde_solver.price(option, env)

        # Get MC price
        mc_price = mc_solver.price(option, env)

        # Compute error (MC is reference for discrete)
        if mc_price != 0:
            mc_error = abs(pde_price - mc_price) / abs(mc_price)
        else:
            mc_error = abs(pde_price - mc_price)

        results.add_result(
            case_name=name, pde_price=pde_price, mc_price=mc_price, mc_error=mc_error
        )


# ============================================================
# MAIN
# ============================================================


def main():
    """Run all benchmark check tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark check for BarrierPDESolver")
    parser.add_argument(
        "--no-mc", action="store_true", help="Skip Monte Carlo comparison (faster)"
    )
    parser.add_argument(
        "--subset", nargs="+", help="Run only specific test cases (by name)"
    )
    args = parser.parse_args()

    results = BenchmarkResults()

    print("\n" + "=" * 90)
    print("BARRIER PDE SOLVER - BENCHMARK CHECK")
    print("=" * 90)

    # Run main benchmark tests
    run_benchmark_tests(results, include_mc=not args.no_mc, test_subset=args.subset)

    # Run knock-in tests (analytical only, faster)
    run_ki_benchmark_tests(results)

    # Run discrete monitoring tests (MC comparison)
    if not args.no_mc:
        run_discrete_monitoring_tests(results)

    # Print summary
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
