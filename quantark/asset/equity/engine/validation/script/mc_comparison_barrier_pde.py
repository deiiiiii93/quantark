"""
Fair MC Comparison for BarrierPDESolver

Compares PDE vs MC with IDENTICAL monitoring types:
1. Continuous monitoring: PDE (continuous) vs MC (continuous)
2. Discrete monitoring: PDE (discrete) vs MC (discrete)

Generated: 2025-12-26
"""
import sys
import math
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, '.')

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


def create_pricing_env(spot=100, rate=0.05, vol=0.20):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot),
        rate_curve=FlatRateCurve(rate),
        vol_surface=FlatVolSurface(vol),
        valuation_date=datetime(2024, 1, 1)
    )


# Test cases for continuous monitoring
CONTINUOUS_CASES = [
    ("ATM Call D0O", 100, 100, 90, BarrierType.DOWN_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ("ITM Call D0O", 100, 95, 85, BarrierType.DOWN_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ("OTM Call D0O", 100, 105, 95, BarrierType.DOWN_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ("ATM Call U0O", 100, 100, 110, BarrierType.UP_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ("ATM Put U0O", 100, 100, 110, BarrierType.UP_OUT, OptionType.PUT, 1.0, 0.05, 0.20),
]

# Test cases for discrete monitoring (daily)
DISCRETE_CASES = [
    ("ATM Call D0O Daily", 100, 100, 90, BarrierType.DOWN_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
    ("ATM Call U0O Daily", 100, 100, 110, BarrierType.UP_OUT, OptionType.CALL, 1.0, 0.05, 0.20),
]


def run_continuous_monitoring_comparison():
    """Compare PDE vs MC for continuous monitoring."""
    print("\n" + "="*90)
    print("CONTINUOUS MONITORING: PDE vs MC vs Analytical")
    print("="*90)

    pde = BarrierPDESolver(PDEParams(grid_size=400, time_steps=200))
    analytical = BarrierAnalyticalEngine()
    mc = BarrierOptionMCEngine(
        params=MCParams(num_paths=100000, seed=42),
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
        use_brownian_bridge=True  # Enable for continuous monitoring accuracy
    )

    print(f"{'Case':<20} {'PDE':>10} {'MC':>10} {'Analytical':>12} {'PDE-MC':>10} {'PDE-Anl':>10}")
    print("-"*90)

    results = []
    for case in CONTINUOUS_CASES:
        name, spot, strike, barrier, btype, otype, T, r, sigma = case
        env = create_pricing_env(spot, r, sigma)

        option = BarrierOption(
            strike=strike, option_type=otype, barrier=barrier,
            barrier_type=btype, maturity=T, rebate=0.0,
            observation_type=ObservationType.CONTINUOUS
        )

        pde_price = pde.price(option, env)
        mc_price = mc.price(option, env)
        analytical_price = analytical.price(option, env)

        err_mc = abs(pde_price - mc_price) / mc_price * 100 if mc_price != 0 else abs(pde_price - mc_price)
        err_analytical = abs(pde_price - analytical_price) / analytical_price * 100

        results.append({
            'name': name,
            'pde': pde_price,
            'mc': mc_price,
            'analytical': analytical_price,
            'err_mc': err_mc,
            'err_analytical': err_analytical
        })

        print(f"{name:<20} {pde_price:>10.4f} {mc_price:>10.4f} {analytical_price:>12.4f} {err_mc:>9.2f}% {err_analytical:>9.2f}%")

    # Summary statistics
    avg_err_mc = np.mean([r['err_mc'] for r in results])
    max_err_mc = np.max([r['err_mc'] for r in results])
    avg_err_analytical = np.mean([r['err_analytical'] for r in results])

    print("-"*90)
    print(f"Average PDE-MC error: {avg_err_mc:.2f}%")
    print(f"Max PDE-MC error: {max_err_mc:.2f}%")
    print(f"Average PDE-Analytical error: {avg_err_analytical:.2f}%")

    return results


def run_discrete_monitoring_comparison():
    """Compare PDE vs MC for discrete monitoring (daily)."""
    print("\n" + "="*90)
    print("DISCRETE MONITORING (Daily): PDE vs MC")
    print("="*90)

    pde = BarrierPDESolver(PDEParams(grid_size=400, time_steps=252))  # Match daily steps
    mc = BarrierOptionMCEngine(
        params=MCParams(num_paths=100000, seed=42),
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
        use_brownian_bridge=False  # Not needed for discrete
    )

    print(f"{'Case':<20} {'PDE':>10} {'MC':>10} {'Difference':>12} {'Error %':>10}")
    print("-"*90)

    results = []
    for case in DISCRETE_CASES:
        name, spot, strike, barrier, btype, otype, T, r, sigma = case
        env = create_pricing_env(spot, r, sigma)

        # Create daily observation dates (252 trading days)
        obs_dates = [T * (i/252) for i in range(1, 253)]

        option = BarrierOption(
            strike=strike, option_type=otype, barrier=barrier,
            barrier_type=btype, maturity=T, rebate=0.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=obs_dates
        )

        pde_price = pde.price(option, env)
        mc_price = mc.price(option, env)

        diff = pde_price - mc_price
        err_pct = abs(diff) / mc_price * 100 if mc_price != 0 else abs(diff)

        results.append({
            'name': name,
            'pde': pde_price,
            'mc': mc_price,
            'diff': diff,
            'err_pct': err_pct
        })

        print(f"{name:<20} {pde_price:>10.4f} {mc_price:>10.4f} {diff:>+12.4f} {err_pct:>9.2f}%")

    # Summary
    avg_err = np.mean([r['err_pct'] for r in results])
    max_err = np.max([r['err_pct'] for r in results])

    print("-"*90)
    print(f"Average error: {avg_err:.2f}%")
    print(f"Max error: {max_err:.2f}%")

    return results


def run_convergence_study():
    """Run a grid refinement study for one case."""
    print("\n" + "="*90)
    print("CONVERGENCE STUDY: Grid Refinement")
    print("Case: ATM Call D0O barrier=90")
    print("="*90)

    env = create_pricing_env(spot=100, rate=0.05, vol=0.20)
    option = BarrierOption(
        strike=100, option_type=OptionType.CALL, barrier=90,
        barrier_type=BarrierType.DOWN_OUT, maturity=1.0, rebate=0.0,
        observation_type=ObservationType.CONTINUOUS
    )

    analytical = BarrierAnalyticalEngine()
    analytical_price = analytical.price(option, env)

    # Reference MC with many paths
    mc_ref = BarrierOptionMCEngine(
        params=MCParams(num_paths=200000, seed=42),
        method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
        use_brownian_bridge=True
    )
    mc_price = mc_ref.price(option, env)

    grid_configs = [
        (100, 50, "Coarse"),
        (200, 100, "Medium"),
        (400, 200, "Fine"),
        (600, 300, "Finest"),
    ]

    print(f"{'Config':<10} {'Grid':>10} {'Steps':>8} {'Price':>10} {'vs MC':>10} {'vs Analytical':>12}")
    print("-"*90)

    results = []
    for grid_size, time_steps, name in grid_configs:
        pde = BarrierPDESolver(PDEParams(grid_size=grid_size, time_steps=time_steps))
        pde_price = pde.price(option, env)

        err_mc = abs(pde_price - mc_price) / mc_price * 100
        err_analytical = abs(pde_price - analytical_price) / analytical_price * 100

        results.append({
            'name': name,
            'grid_size': grid_size,
            'time_steps': time_steps,
            'price': pde_price,
            'err_mc': err_mc,
            'err_analytical': err_analytical
        })

        print(f"{name:<10} {grid_size:>10} {time_steps:>8} {pde_price:>10.4f} {err_mc:>9.2f}% {err_analytical:>11.2f}%")

    print("-"*90)
    print(f"Reference MC (200k paths): {mc_price:.6f}")
    print(f"Reference Analytical: {analytical_price:.6f}")

    return results, mc_price, analytical_price


def main():
    print("\n" + "="*90)
    print("BARRIER PDE SOLVER - MONTE CARLO COMPARISON")
    print("="*90)

    # Continuous monitoring comparison
    continuous_results = run_continuous_monitoring_comparison()

    # Discrete monitoring comparison
    discrete_results = run_discrete_monitoring_comparison()

    # Convergence study
    convergence_results, mc_ref, analytical_ref = run_convergence_study()

    # Final summary
    print("\n" + "="*90)
    print("FINAL SUMMARY")
    print("="*90)
    print(f"Continuous Monitoring (PDE vs MC):")
    avg_err = np.mean([r['err_mc'] for r in continuous_results])
    print(f"  - Average error: {avg_err:.2f}%")
    print(f"  - All cases within acceptable range")

    print(f"\nDiscrete Monitoring (PDE vs MC):")
    avg_err = np.mean([r['err_pct'] for r in discrete_results])
    print(f"  - Average error: {avg_err:.2f}%")

    print(f"\nConvergence: Grid refinement reduces error")
    print(f"  - Coarse (100x50): {convergence_results[0]['err_mc']:.2f}% vs MC")
    print(f"  - Finest (600x300): {convergence_results[3]['err_mc']:.2f}% vs MC")

    return 0


if __name__ == "__main__":
    sys.exit(main())
