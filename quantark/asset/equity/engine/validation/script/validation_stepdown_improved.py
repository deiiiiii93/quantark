"""
Validation script for Stepdown Snowball using the improved ODE-based SpatialGrid.

This script verifies that the Stepdown Snowball (which has multiple closely spaced
barriers) is priced accurately by the PDE solver using the improved adaptive grid
logic in SpatialGrid.build_tavella_randall_multi.

Target: Error < 2% vs 500k path Monte Carlo.
"""

import sys
import os
sys.path.insert(0, ".")

from datetime import datetime
import numpy as np

from quantark.asset.equity.product.option.snowball_helpers import create_stepdown_snowball
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum import MonteCarloMethod

def validate_stepdown_pricing():
    print("=" * 80)
    print("Validation: Stepdown Snowball with Improved Adaptive Grid")
    print("=" * 80)

    # 1. Setup Product
    # Stepdown Snowball: Barriers [103.0, 102.5, ..., 97.5]
    S = 100.0
    product = create_stepdown_snowball(
        initial_price=S,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,
        ki_barrier=75.0,
        ko_rate=0.15
    )
    
    env = PricingEnvironment(
        valuation_date=datetime(2025, 1, 1),
        spot_quote=SpotQuote(S),
        rate_curve=FlatRateCurve(0.03),
        vol_surface=FlatVolSurface(0.20),
        div_yield=ContinuousDividendYield(0.0)
    )

    # 2. Run Benchmark MC (500k paths)
    print("\n[1/2] Running Benchmark Monte Carlo (500k paths)...")
    mc_engine = SnowballMCEngine(
        params=MCParams(num_paths=500000, seed=42),
        method=MonteCarloMethod.QUASI
    )
    mc_price = mc_engine.price(product, env)
    print(f"      MC Price: {mc_price:,.2f}")

    # 3. Run PDE with Auto Grid
    print("\n[2/2] Running PDE with Auto Grid (Improved SpatialGrid)...")
    # Using default grid_size=400, auto_grid=True
    pde_solver = SnowballPDESolver(
        params=PDEParams()
    )
    pde_price = pde_solver.price(product, env)
    print(f"      PDE Price: {pde_price:,.2f}")

    # 4. Analyze Results
    diff = pde_price - mc_price
    rel_error = diff / mc_price
    
    print("\nResults Analysis:")
    print(f"  Difference: {diff:,.2f}")
    print(f"  Rel Error:  {rel_error:.2%}")
    
    # Check grid quality implicitly by inspecting the object
    # (Accessing protected members for validation report)
    x_vec, s_vec, _, _, _ = pde_solver._build_grids(product, env, S, 0.20, 1.0, 0.03, 0.0)
    barrier_min = min(product.barrier_config.ko_barrier)
    barrier_max = max(product.barrier_config.ko_barrier)
    points_in_barrier = len(s_vec[(s_vec >= barrier_min) & (s_vec <= barrier_max)])
    
    print(f"  Grid Points in Barrier Range [{barrier_min}, {barrier_max}]: {points_in_barrier}")
    
    if abs(rel_error) < 0.02:
        print("\n✅ PASSED: Error is within 2% tolerance.")
        return True
    else:
        print("\n❌ FAILED: Error exceeds 2% tolerance.")
        return False

if __name__ == "__main__":
    success = validate_stepdown_pricing()
    sys.exit(0 if success else 1)
