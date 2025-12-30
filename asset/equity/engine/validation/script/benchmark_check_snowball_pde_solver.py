"""
Benchmark Check Script for Snowball PDE Solver
Engine: asset/equity/engine/pde/snowball_pde_solver.py
Benchmark: asset/equity/engine/mc/snowball_mc_engine.py
Generated: 2025-12-29
Default Tolerance: 5%

This script compares the PDE solver results against the Monte Carlo engine
to validate numerical accuracy across various market scenarios.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
)
from asset.equity.product.option.snowball_helpers import (
    create_standard_snowball as create_standard_helper,
    create_stepdown_snowball as create_stepdown_helper,
    create_european_ki_snowball as create_european_ki_helper,
    create_parachute_snowball as create_parachute_helper,
    create_airbag_snowball as create_airbag_helper,
)
from asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import PDEParams, MCParams
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from util.enum.engine_enums import MonteCarloMethod


TOLERANCE = 0.05  # 5% relative error default


class BenchmarkResults:
    """Collect and report benchmark comparison results."""

    def __init__(self, tolerance: float = TOLERANCE):
        self.tolerance = tolerance
        self.results: List[Dict] = []

    def add_result(
        self, case_name: str, pde_price: float, mc_price: float, mc_std_error: float = 0.0
    ):
        if mc_price != 0:
            rel_error = abs(pde_price - mc_price) / abs(mc_price)
        else:
            rel_error = abs(pde_price - mc_price)
        passed = rel_error <= self.tolerance
        self.results.append({
            "case": case_name,
            "pde": pde_price,
            "mc": mc_price,
            "mc_std_error": mc_std_error,
            "rel_error": rel_error,
            "passed": passed,
        })

    def summary(self):
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        print(f"\n{'='*100}")
        print(f"BENCHMARK CHECK SUMMARY: Snowball PDE Solver vs MC Engine (Tolerance: {self.tolerance*100:.1f}%)")
        print(f"{'='*100}")
        print(
            f"{'Case':<35} {'PDE':>14} {'MC':>14} {'MC SE':>10} {'Error':>10} {'Status':>8}"
        )
        print(f"{'-'*100}")

        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            mc_se_str = f"±{r['mc_std_error']:,.0f}" if r["mc_std_error"] > 0 else "N/A"
            print(
                f"{r['case']:<35} {r['pde']:>14,.2f} {r['mc']:>14,.2f} "
                f"{mc_se_str:>10} {r['rel_error']*100:>9.2f}% {status:>8}"
            )

        print(f"{'-'*100}")
        print(f"Passed: {passed}/{total} ({100*passed/max(total,1):.1f}%)")

        if passed < total:
            print("\nFailed Cases Analysis:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['case']}: Error {r['rel_error']*100:.2f}% > {self.tolerance*100:.1f}%")

        return passed == total


def create_pricing_env(
    spot: float, rate: float = 0.03, vol: float = 0.20, div: float = 0.0
) -> PricingEnvironment:
    """Create a pricing environment."""
    val_date = datetime(2024, 1, 15)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=val_date,
    )


def create_snowball(
    initial_price: float = 100.0,
    ko_barrier: float = 103.0,
    ki_barrier: float = 75.0,
    ko_rate: float = 0.15,
    maturity: float = 1.0,
    notional: float = 1_000_000.0,
    num_ko_obs: int = 12,
    ki_continuous: bool = True,
) -> SnowballOption:
    """Create a snowball option for testing."""
    ko_obs_dates = [maturity * (i + 1) / num_ko_obs for i in range(num_ko_obs)]

    barrier_config = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ko_observation_dates=ko_obs_dates,
        ki_barrier=ki_barrier,
        ki_continuous=ki_continuous,
    )

    return SnowballOption(
        initial_price=initial_price,
        strike=initial_price,
        barrier_config=barrier_config,
        notional=notional,
        maturity=maturity,
    )


def create_pde_solver(grid_size: int = 250, time_steps: int = 250) -> SnowballPDESolver:
    """Create PDE solver with good resolution."""
    params = PDEParams(
        grid_size=grid_size,
        time_steps=time_steps,
        theta=0.5,
        use_rannacher=True,
        rannacher_steps=2,
    )
    return SnowballPDESolver(params=params)


def create_mc_engine(
    num_paths: int = 200000, time_steps: int = 252, seed: int = 42
) -> SnowballMCEngine:
    """Create MC engine with high path count for accuracy."""
    params = MCParams(
        num_paths=num_paths,
        time_steps=time_steps,
        seed=seed,
    )
    return SnowballMCEngine(params=params, method=MonteCarloMethod.QUASI)


def run_comparison(
    case_name: str,
    snowball: SnowballOption,
    env: PricingEnvironment,
    pde_solver: SnowballPDESolver,
    mc_engine: SnowballMCEngine,
    results: BenchmarkResults,
) -> None:
    """Run a single comparison between PDE and MC."""
    try:
        pde_price = pde_solver.price(snowball, env)
    except Exception as e:
        print(f"  PDE Error for {case_name}: {e}")
        pde_price = float("nan")

    try:
        mc_price = mc_engine.price(snowball, env)
        mc_result = mc_engine.get_last_result()
        mc_std_error = mc_result.std_error if mc_result else 0.0
    except Exception as e:
        print(f"  MC Error for {case_name}: {e}")
        mc_price = float("nan")
        mc_std_error = 0.0

    if not np.isnan(pde_price) and not np.isnan(mc_price):
        results.add_result(case_name, pde_price, mc_price, mc_std_error)
    else:
        results.add_result(case_name, pde_price if not np.isnan(pde_price) else 0.0, 
                          mc_price if not np.isnan(mc_price) else 0.0, mc_std_error)


# ============================================================
# TEST CASES
# ============================================================


def run_benchmark_tests(results: BenchmarkResults):
    """Run all benchmark test cases."""
    pde_solver = create_pde_solver()
    mc_engine = create_mc_engine()

    # ─────────────────────────────────────────────────────────
    # BASE CASE: ATM Snowball
    # ─────────────────────────────────────────────────────────
    print("\n>>> Base Case: ATM Snowball")
    env = create_pricing_env(spot=100.0, rate=0.03, vol=0.20)
    snowball = create_snowball(
        initial_price=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        ko_rate=0.15,
        maturity=1.0,
    )
    run_comparison("ATM T=1Y σ=20%", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # VOLATILITY VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Volatility Variations")
    for vol in [0.15, 0.25, 0.35]:
        env = create_pricing_env(spot=100.0, vol=vol)
        snowball = create_snowball()
        run_comparison(f"ATM T=1Y σ={vol*100:.0f}%", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # SPOT VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Spot Variations")
    for spot in [95.0, 102.0, 85.0]:
        env = create_pricing_env(spot=spot, vol=0.20)
        snowball = create_snowball()
        run_comparison(f"Spot={spot:.0f} T=1Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # MATURITY VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Maturity Variations")
    for mat, num_obs in [(0.5, 6), (1.5, 18), (2.0, 24)]:
        env = create_pricing_env(spot=100.0, vol=0.20)
        snowball = create_snowball(maturity=mat, num_ko_obs=num_obs)
        run_comparison(f"ATM T={mat}Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # KO BARRIER VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> KO Barrier Variations")
    for ko in [102.0, 105.0, 110.0]:
        env = create_pricing_env(spot=100.0, vol=0.20)
        snowball = create_snowball(ko_barrier=ko)
        run_comparison(f"KO={ko:.0f}% T=1Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # KI BARRIER VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> KI Barrier Variations")
    for ki in [70.0, 80.0, 85.0]:
        env = create_pricing_env(spot=100.0, vol=0.20)
        snowball = create_snowball(ki_barrier=ki)
        run_comparison(f"KI={ki:.0f}% T=1Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # RATE VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Rate Variations")
    for rate in [0.01, 0.05, 0.08]:
        env = create_pricing_env(spot=100.0, vol=0.20, rate=rate)
        snowball = create_snowball()
        run_comparison(f"r={rate*100:.0f}% T=1Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # KO RATE VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> KO Rate (Coupon) Variations")
    for ko_rate in [0.10, 0.20, 0.30]:
        env = create_pricing_env(spot=100.0, vol=0.20)
        snowball = create_snowball(ko_rate=ko_rate)
        run_comparison(f"KO Rate={ko_rate*100:.0f}% T=1Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # DIVIDEND VARIATIONS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Dividend Yield Variations")
    for div in [0.02, 0.04]:
        env = create_pricing_env(spot=100.0, vol=0.20, div=div)
        snowball = create_snowball()
        run_comparison(f"q={div*100:.0f}% T=1Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # CONTINUOUS VS DISCRETE KI
    # ─────────────────────────────────────────────────────────
    print("\n>>> KI Monitoring Type")
    env = create_pricing_env(spot=100.0, vol=0.20)

    snowball_cont = create_snowball(ki_continuous=True)
    run_comparison("Continuous KI T=1Y", snowball_cont, env, pde_solver, mc_engine, results)

    # For discrete KI, use matching observation dates
    ko_obs_dates = [1.0 * (i + 1) / 12 for i in range(12)]
    barrier_config_discrete = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_dates=ko_obs_dates,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=ko_obs_dates,  # Monthly discrete KI
    )
    snowball_discrete = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config_discrete,
        notional=1_000_000.0,
        maturity=1.0,
    )
    run_comparison("Discrete KI T=1Y", snowball_discrete, env, pde_solver, mc_engine, results)


def run_stress_tests(results: BenchmarkResults):
    """Run stress test cases."""
    pde_solver = create_pde_solver(grid_size=300, time_steps=300)
    mc_engine = create_mc_engine(num_paths=500000)

    print("\n>>> Stress Tests (High Resolution)")

    # Near KO barrier
    env = create_pricing_env(spot=102.5, vol=0.20)
    snowball = create_snowball(ko_barrier=103.0)
    run_comparison("Spot near KO (102.5)", snowball, env, pde_solver, mc_engine, results)

    # Near KI barrier
    env = create_pricing_env(spot=76.0, vol=0.20)
    snowball = create_snowball(ki_barrier=75.0)
    run_comparison("Spot near KI (76.0)", snowball, env, pde_solver, mc_engine, results)

    # High volatility
    env = create_pricing_env(spot=100.0, vol=0.40)
    snowball = create_snowball()
    run_comparison("High Vol σ=40%", snowball, env, pde_solver, mc_engine, results)


# ============================================================
# SNOWBALL VARIANT TESTS
# ============================================================


def run_variant_tests(results: BenchmarkResults):
    """Run benchmark tests for snowball variants."""
    pde_solver = create_pde_solver()
    mc_engine = create_mc_engine()

    # ─────────────────────────────────────────────────────────
    # STEPDOWN SNOWBALL VARIANTS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Stepdown Snowball Variants")

    # Base stepdown
    env = create_pricing_env(spot=100.0, vol=0.20)
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,  # 0.5% per period
        ki_barrier=75.0,
    )
    run_comparison("Stepdown 0.5%/mo", snowball, env, pde_solver, mc_engine, results)

    # Aggressive stepdown
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=105.0,
        stepdown_rate=0.01,  # 1% per period
        ki_barrier=75.0,
    )
    run_comparison("Stepdown 1%/mo", snowball, env, pde_solver, mc_engine, results)

    # Stepdown with high volatility
    env_high_vol = create_pricing_env(spot=100.0, vol=0.30)
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
    )
    run_comparison("Stepdown σ=30%", snowball, env_high_vol, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # EUROPEAN KI SNOWBALL VARIANTS
    # ─────────────────────────────────────────────────────────
    print("\n>>> European KI Snowball Variants")

    # Base European KI
    env = create_pricing_env(spot=100.0, vol=0.20)
    snowball = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    run_comparison("European KI ATM", snowball, env, pde_solver, mc_engine, results)

    # European KI with low KI barrier
    snowball = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=60.0,  # Lower KI barrier
    )
    run_comparison("European KI Low KI", snowball, env, pde_solver, mc_engine, results)

    # European KI with 2-year maturity
    snowball = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=2.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        num_ko_observations=24,
    )
    run_comparison("European KI T=2Y", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # PARACHUTE SNOWBALL VARIANTS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Parachute Snowball Variants")

    # Base parachute
    env = create_pricing_env(spot=100.0, vol=0.20)
    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    run_comparison("Parachute ATM", snowball, env, pde_solver, mc_engine, results)

    # Parachute with higher KO barrier
    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=110.0,  # Higher KO barrier
        ki_barrier=75.0,
    )
    run_comparison("Parachute High KO", snowball, env, pde_solver, mc_engine, results)

    # Parachute with 6-month maturity
    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=0.5,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=80.0,
        num_observations=6,
    )
    run_comparison("Parachute T=6M", snowball, env, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # AIRBAG SNOWBALL VARIANTS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Airbag Snowball Variants")

    # Base airbag
    env = create_pricing_env(spot=100.0, vol=0.20)
    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,
    )
    run_comparison("Airbag 50% below 60%", snowball, env, pde_solver, mc_engine, results)

    # Airbag with higher airbag barrier
    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=70.0,  # Higher airbag barrier (more protection)
        participation_rate=1.0,
        airbag_participation_rate=0.5,
    )
    run_comparison("Airbag 50% below 70%", snowball, env, pde_solver, mc_engine, results)

    # Airbag with lower participation rate
    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.25,  # 25% participation below airbag
    )
    run_comparison("Airbag 25% below 60%", snowball, env, pde_solver, mc_engine, results)

    # Airbag with high volatility
    env_high_vol = create_pricing_env(spot=100.0, vol=0.35)
    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        airbag_barrier=60.0,
        airbag_participation_rate=0.5,
    )
    run_comparison("Airbag σ=35%", snowball, env_high_vol, pde_solver, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # VARIANT STRESS TESTS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Variant Stress Tests")

    pde_solver_fine = create_pde_solver(grid_size=300, time_steps=300)
    mc_engine_fine = create_mc_engine(num_paths=500000)

    # Stepdown near KO
    env = create_pricing_env(spot=102.0, vol=0.20)
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,
        ki_barrier=75.0,
    )
    run_comparison("Stepdown near KO", snowball, env, pde_solver_fine, mc_engine_fine, results)

    # Parachute with spot near KI barrier
    env = create_pricing_env(spot=77.0, vol=0.20)
    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    run_comparison("Parachute near KI", snowball, env, pde_solver_fine, mc_engine_fine, results)


# ============================================================
# NON-UNIFORM GRID TESTS
# ============================================================


def create_pde_solver_nonuniform(
    grid_size: int = 250, time_steps: int = 250, log_dx_target: float = 0.003
) -> SnowballPDESolver:
    """Create PDE solver with non-uniform (adaptive) grid."""
    params = PDEParams(
        grid_size=grid_size,
        time_steps=time_steps,
        theta=0.5,
        use_rannacher=True,
        rannacher_steps=2,
        adaptive_grid=True,  # Enable Tavella-Randall non-uniform grid
        log_dx_target=log_dx_target,  # Target spacing near critical points
    )
    return SnowballPDESolver(params=params)


def run_nonuniform_grid_tests(results: BenchmarkResults):
    """Run tests comparing non-uniform grid performance for snowball variants."""
    mc_engine = create_mc_engine()

    print("\n>>> Non-Uniform Grid Tests: Standard Snowball")
    # ─────────────────────────────────────────────────────────
    # STANDARD SNOWBALL WITH NON-UNIFORM GRID
    # ─────────────────────────────────────────────────────────
    env = create_pricing_env(spot=100.0, vol=0.20)

    # Coarse grid with non-uniform (should still be accurate due to concentration)
    pde_solver_nu_coarse = create_pde_solver_nonuniform(grid_size=150, time_steps=150)
    snowball = create_snowball()
    run_comparison("NonUniform Grid 150×150", snowball, env, pde_solver_nu_coarse, mc_engine, results)

    # Standard grid with non-uniform
    pde_solver_nu_std = create_pde_solver_nonuniform(grid_size=250, time_steps=250)
    run_comparison("NonUniform Grid 250×250", snowball, env, pde_solver_nu_std, mc_engine, results)

    # Fine grid with non-uniform (tighter epsilon)
    pde_solver_nu_fine = create_pde_solver_nonuniform(
        grid_size=200, time_steps=200, log_dx_target=0.002
    )
    run_comparison("NonUniform Grid eps=0.2%", snowball, env, pde_solver_nu_fine, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # STEPDOWN SNOWBALL (MULTIPLE CRITICAL POINTS)
    # ─────────────────────────────────────────────────────────
    print("\n>>> Non-Uniform Grid Tests: Stepdown (Multi-Critical)")

    # Stepdown has multiple KO barriers - good test for multi-critical-point grid
    pde_solver_nu = create_pde_solver_nonuniform(grid_size=250, time_steps=250)

    # Standard stepdown
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,
        ki_barrier=75.0,
    )
    run_comparison("Stepdown NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # Aggressive stepdown (more barrier levels)
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=110.0,
        stepdown_rate=0.015,  # 1.5% per period - creates wider spread of barriers
        ki_barrier=75.0,
    )
    run_comparison("Stepdown 1.5%/mo NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # EUROPEAN KI SNOWBALL
    # ─────────────────────────────────────────────────────────
    print("\n>>> Non-Uniform Grid Tests: European KI")

    snowball = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    run_comparison("European KI NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # European KI with low KI barrier (farther from strike)
    snowball = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=60.0,
    )
    run_comparison("European KI Low KI NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # PARACHUTE SNOWBALL
    # ─────────────────────────────────────────────────────────
    print("\n>>> Non-Uniform Grid Tests: Parachute")

    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    run_comparison("Parachute NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # Parachute with high KO barrier
    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=110.0,
        ki_barrier=75.0,
    )
    run_comparison("Parachute High KO NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # AIRBAG SNOWBALL (3 CRITICAL POINTS: KO, KI, AIRBAG)
    # ─────────────────────────────────────────────────────────
    print("\n>>> Non-Uniform Grid Tests: Airbag (Triple Critical Points)")

    # Airbag has 3 critical points: KO barrier, KI barrier, and airbag barrier
    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,
    )
    run_comparison("Airbag NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # Airbag with higher airbag barrier (closer to KI)
    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=70.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,
    )
    run_comparison("Airbag High AB NonUniform", snowball, env, pde_solver_nu, mc_engine, results)

    # ─────────────────────────────────────────────────────────
    # NON-UNIFORM GRID STRESS TESTS
    # ─────────────────────────────────────────────────────────
    print("\n>>> Non-Uniform Grid Stress Tests")

    mc_engine_fine = create_mc_engine(num_paths=500000)
    pde_solver_nu_fine = create_pde_solver_nonuniform(
        grid_size=300, time_steps=300, log_dx_target=0.002
    )

    # Spot near KO barrier
    env_near_ko = create_pricing_env(spot=102.5, vol=0.20)
    snowball = create_snowball(ko_barrier=103.0)
    run_comparison("Spot near KO NonUniform", snowball, env_near_ko, pde_solver_nu_fine, mc_engine_fine, results)

    # Spot near KI barrier
    env_near_ki = create_pricing_env(spot=76.0, vol=0.20)
    snowball = create_snowball(ki_barrier=75.0)
    run_comparison("Spot near KI NonUniform", snowball, env_near_ki, pde_solver_nu_fine, mc_engine_fine, results)

    # High volatility with non-uniform grid
    env_high_vol = create_pricing_env(spot=100.0, vol=0.40)
    snowball = create_snowball()
    run_comparison("High Vol σ=40% NonUniform", snowball, env_high_vol, pde_solver_nu_fine, mc_engine_fine, results)

    # Stepdown near KO with non-uniform
    env_near_ko = create_pricing_env(spot=102.0, vol=0.20)
    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        notional=1_000_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,
        ki_barrier=75.0,
    )
    run_comparison("Stepdown near KO NonUniform", snowball, env_near_ko, pde_solver_nu_fine, mc_engine_fine, results)


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Snowball PDE vs MC")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE, help="Error tolerance")
    parser.add_argument("--stress", action="store_true", help="Include stress tests")
    parser.add_argument("--variants", action="store_true", help="Include snowball variant tests")
    parser.add_argument("--nonuniform", action="store_true", help="Include non-uniform grid tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (stress + variants + nonuniform)")
    args = parser.parse_args()

    results = BenchmarkResults(tolerance=args.tolerance)

    print("=" * 100)
    print("SNOWBALL PDE SOLVER - BENCHMARK AGAINST MONTE CARLO")
    print("=" * 100)

    run_benchmark_tests(results)

    if args.stress or args.all:
        run_stress_tests(results)

    if args.variants or args.all:
        run_variant_tests(results)

    if args.nonuniform or args.all:
        run_nonuniform_grid_tests(results)

    success = results.summary()
    sys.exit(0 if success else 1)
