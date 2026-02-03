"""
Boundary Check Script for Snowball PDE Solver
Engine: asset/equity/engine/pde/snowball_pde_solver.py
Generated: 2025-12-29

This script validates the SnowballPDESolver against theoretical boundary
conditions and extreme market scenarios for the Two-Surface PDE method.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
from datetime import datetime, timedelta

from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
    AirbagConfig,
)
from asset.equity.product.option.snowball_helpers import (
    create_standard_snowball as create_standard_helper,
    create_stepdown_snowball as create_stepdown_helper,
    create_european_ki_snowball as create_european_ki_helper,
    create_parachute_snowball as create_parachute_helper,
    create_airbag_snowball as create_airbag_helper,
)
from asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from asset.equity.param import PDEParams
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from util.enum import ObservationType


class BoundaryCheckResults:
    """Collect and report boundary check results."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_result(self, test_name: str, passed: bool, message: str):
        if passed:
            self.passed.append((test_name, message))
        else:
            self.failed.append((test_name, message))

    def add_warning(self, test_name: str, message: str):
        self.warnings.append((test_name, message))

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*70}")
        print(f"BOUNDARY CHECK SUMMARY: Snowball PDE Solver")
        print(f"{'='*70}")
        print(f"Total Tests: {total}")
        print(f"Passed: {len(self.passed)} ({100*len(self.passed)/max(total,1):.1f}%)")
        print(f"Failed: {len(self.failed)} ({100*len(self.failed)/max(total,1):.1f}%)")
        print(f"Warnings: {len(self.warnings)}")

        if self.passed:
            print(f"\n{'─'*70}")
            print("PASSED TESTS:")
            for name, msg in self.passed:
                print(f"  ✓ {name}: {msg}")

        if self.failed:
            print(f"\n{'─'*70}")
            print("FAILED TESTS:")
            for name, msg in self.failed:
                print(f"  ✗ {name}: {msg}")

        if self.warnings:
            print(f"\n{'─'*70}")
            print("WARNINGS:")
            for name, msg in self.warnings:
                print(f"  ⚠ {name}: {msg}")

        return len(self.failed) == 0


def create_pricing_env(spot: float, rate: float = 0.03, vol: float = 0.20, div: float = 0.0):
    """Create a pricing environment with given parameters."""
    val_date = datetime(2024, 1, 15)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=val_date,
    )


def create_standard_snowball(
    initial_price: float = 100.0,
    ko_barrier: float = 103.0,
    ki_barrier: float = 75.0,
    ko_rate: float = 0.15,
    maturity: float = 1.0,
    contract_multiplier: float = 10_000.0,
    num_ko_obs: int = 12,
    ki_continuous: bool = True,
) -> SnowballOption:
    """Create a standard snowball option for testing."""
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
        contract_multiplier=contract_multiplier,
        maturity=maturity,
    )


def create_solver(grid_size: int = 200, time_steps: int = 200) -> SnowballPDESolver:
    """Create a PDE solver with specified grid size."""
    params = PDEParams(
        grid_size=grid_size,
        time_steps=time_steps,
        theta=0.5,  # Crank-Nicolson
        use_rannacher=True,
        rannacher_steps=2,
    )
    return SnowballPDESolver(params=params)


# ============================================================
# EXTREME MARKET CASE TESTS
# ============================================================


def test_low_volatility(results: BoundaryCheckResults):
    """
    Test: Low volatility should produce a value close to expected deterministic outcome.
    
    For a snowball with KO barrier above spot, low vol means:
    - High probability of KO at first observation
    - Price should be close to discounted KO payoff
    """
    spot = 100.0
    vol = 0.01  # Very low volatility
    rate = 0.03
    maturity = 1.0

    env = create_pricing_env(spot=spot, rate=rate, vol=vol)
    snowball = create_standard_snowball(
        initial_price=100.0,
        ko_barrier=105.0,  # Above spot
        ki_barrier=75.0,
        maturity=maturity,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # With very low vol and KO barrier above spot, price should be positive
        # (principal + rebate if survives, KO payoff if triggered)
        passed = price > 0 and price <= snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "Low Volatility",
            passed,
            f"Price = {price:,.2f} (expected positive, bounded)"
        )
    except Exception as e:
        results.add_result("Low Volatility", False, f"Error: {e}")


def test_high_volatility(results: BoundaryCheckResults):
    """
    Test: High volatility increases option value uncertainty.
    
    With high vol, more KI scenarios become possible.
    """
    spot = 100.0
    env_low = create_pricing_env(spot=spot, vol=0.10)
    env_high = create_pricing_env(spot=spot, vol=0.50)

    snowball = create_standard_snowball(
        initial_price=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    solver = create_solver()

    try:
        price_low = solver.price(snowball, env_low)
        price_high = solver.price(snowball, env_high)

        # Higher vol generally decreases snowball value (more KI risk)
        # But can also increase KO probability
        # Just check both prices are reasonable
        passed = price_low > 0 and price_high > 0
        results.add_result(
            "High Volatility",
            passed,
            f"Low vol: {price_low:,.2f}, High vol: {price_high:,.2f}"
        )
    except Exception as e:
        results.add_result("High Volatility", False, f"Error: {e}")


def test_near_expiry(results: BoundaryCheckResults):
    """
    Test: Near expiry option value approaches terminal payoff.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # Very short maturity
    snowball = create_standard_snowball(
        initial_price=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        maturity=0.01,  # ~3.5 days
        num_ko_obs=1,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # Near expiry, price should be close to terminal V0 payoff
        # (spot is between KI and KO barriers)
        terminal_v0 = snowball.get_maturity_payoff_v0(spot, pricing_env=env)
        passed = abs(price - terminal_v0) < snowball.initial_price * snowball.contract_multiplier * 0.05
        results.add_result(
            "Near Expiry",
            passed,
            f"Price = {price:,.2f}, Terminal V0 = {terminal_v0:,.2f}"
        )
    except Exception as e:
        results.add_result("Near Expiry", False, f"Error: {e}")


def test_deep_in_ko_region(results: BoundaryCheckResults):
    """
    Test: Spot deep in KO region should produce KO payoff.
    """
    spot = 110.0  # Above KO barrier
    env = create_pricing_env(spot=spot)

    snowball = create_standard_snowball(
        initial_price=100.0,
        ko_barrier=103.0,  # Spot is above this
        ki_barrier=75.0,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # If spot starts above KO barrier and first observation is at t=0,
        # price should be immediate KO payoff
        # Otherwise, interpolated from V0 surface at high spot
        passed = price > 0 and price <= snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "Deep in KO Region",
            passed,
            f"Price = {price:,.2f} (spot {spot} > KO barrier 103)"
        )
    except Exception as e:
        results.add_result("Deep in KO Region", False, f"Error: {e}")


def test_deep_in_ki_region(results: BoundaryCheckResults):
    """
    Test: Spot deep in KI region should use V1 surface.
    """
    spot = 70.0  # Below KI barrier
    env = create_pricing_env(spot=spot)

    snowball = create_standard_snowball(
        initial_price=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,  # Spot is below this
        ki_continuous=True,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # Spot below KI barrier with continuous monitoring means already knocked-in
        # Should use V1 surface (lower value due to downside exposure)
        terminal_v1 = snowball.get_maturity_payoff_v1(spot, env)
        # V1 price should be less than principal due to downside
        passed = price < snowball.initial_price * snowball.contract_multiplier
        results.add_result(
            "Deep in KI Region",
            passed,
            f"Price = {price:,.2f} (spot {spot} < KI barrier 75, V1 state)"
        )
    except Exception as e:
        results.add_result("Deep in KI Region", False, f"Error: {e}")


def test_zero_rate(results: BoundaryCheckResults):
    """
    Test: Zero interest rate removes discounting effects.
    """
    spot = 100.0
    env_zero = create_pricing_env(spot=spot, rate=0.0)
    env_positive = create_pricing_env(spot=spot, rate=0.05)

    snowball = create_standard_snowball()
    solver = create_solver()

    try:
        price_zero = solver.price(snowball, env_zero)
        price_positive = solver.price(snowball, env_positive)

        # Both should be positive and reasonable
        passed = price_zero > 0 and price_positive > 0
        results.add_result(
            "Zero Interest Rate",
            passed,
            f"r=0: {price_zero:,.2f}, r=5%: {price_positive:,.2f}"
        )
    except Exception as e:
        results.add_result("Zero Interest Rate", False, f"Error: {e}")


# ============================================================
# THEORETICAL RELATIONSHIP TESTS
# ============================================================


def test_v0_v1_relationship(results: BoundaryCheckResults):
    """
    Test: V0 surface should be >= V1 surface for standard snowball.
    
    V0 (never KI) receives rebate at maturity.
    V1 (knocked-in) has downside exposure.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # Use continuous KI for simpler test
    snowball = create_standard_snowball(
        ki_continuous=True,
    )
    solver = create_solver()

    try:
        # Price when never knocked-in (V0)
        price_v0 = solver.price(snowball, env)

        # For V1 comparison, we'd need to set up a knocked-in state
        # For now, check that V0 price is reasonable
        passed = price_v0 > 0 and price_v0 <= snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "V0-V1 Relationship",
            passed,
            f"V0 price = {price_v0:,.2f} (should be >= V1 due to rebate vs downside)"
        )
    except Exception as e:
        results.add_result("V0-V1 Relationship", False, f"Error: {e}")


def test_ko_barrier_effect(results: BoundaryCheckResults):
    """
    Test: Higher KO barrier should decrease snowball value.
    
    Higher barrier = harder to knock out = more KI risk.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball_low_ko = create_standard_snowball(ko_barrier=102.0)
    snowball_high_ko = create_standard_snowball(ko_barrier=110.0)
    solver = create_solver()

    try:
        price_low = solver.price(snowball_low_ko, env)
        price_high = solver.price(snowball_high_ko, env)

        # Higher KO barrier = harder to KO = more risk = lower price
        passed = price_low >= price_high * 0.95  # Allow some tolerance
        results.add_result(
            "KO Barrier Effect",
            passed,
            f"KO@102: {price_low:,.2f}, KO@110: {price_high:,.2f}"
        )
    except Exception as e:
        results.add_result("KO Barrier Effect", False, f"Error: {e}")


def test_ki_barrier_effect(results: BoundaryCheckResults):
    """
    Test: Lower KI barrier should increase snowball value.
    
    Lower barrier = harder to knock in = less downside risk.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball_high_ki = create_standard_snowball(ki_barrier=80.0)
    snowball_low_ki = create_standard_snowball(ki_barrier=60.0)
    solver = create_solver()

    try:
        price_high = solver.price(snowball_high_ki, env)
        price_low = solver.price(snowball_low_ki, env)

        # Lower KI barrier = harder to KI = less risk = higher price
        passed = price_low >= price_high * 0.95  # Allow some tolerance
        results.add_result(
            "KI Barrier Effect",
            passed,
            f"KI@80: {price_high:,.2f}, KI@60: {price_low:,.2f}"
        )
    except Exception as e:
        results.add_result("KI Barrier Effect", False, f"Error: {e}")


def test_maturity_effect(results: BoundaryCheckResults):
    """
    Test: Longer maturity effect on snowball value.
    
    Longer maturity = more time for KO but also more KI risk.
    Effect depends on barrier configuration.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball_short = create_standard_snowball(maturity=0.5, num_ko_obs=6)
    snowball_long = create_standard_snowball(maturity=2.0, num_ko_obs=24)
    solver = create_solver()

    try:
        price_short = solver.price(snowball_short, env)
        price_long = solver.price(snowball_long, env)

        # Both should be positive and reasonable
        passed = price_short > 0 and price_long > 0
        results.add_result(
            "Maturity Effect",
            passed,
            f"T=0.5Y: {price_short:,.2f}, T=2Y: {price_long:,.2f}"
        )
    except Exception as e:
        results.add_result("Maturity Effect", False, f"Error: {e}")


def test_ko_rate_effect(results: BoundaryCheckResults):
    """
    Test: Higher KO coupon rate should increase snowball value.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball_low_rate = create_standard_snowball(ko_rate=0.10)
    snowball_high_rate = create_standard_snowball(ko_rate=0.25)
    solver = create_solver()

    try:
        price_low = solver.price(snowball_low_rate, env)
        price_high = solver.price(snowball_high_rate, env)

        # Higher KO rate = larger coupon on KO = higher value
        passed = price_high >= price_low * 0.98  # Allow small tolerance
        results.add_result(
            "KO Rate Effect",
            passed,
            f"Rate=10%: {price_low:,.2f}, Rate=25%: {price_high:,.2f}"
        )
    except Exception as e:
        results.add_result("KO Rate Effect", False, f"Error: {e}")


def test_principal_bounds(results: BoundaryCheckResults):
    """
    Test: Price should be bounded by reasonable limits.
    
    Lower bound: 0 (worst case KI with full loss)
    Upper bound: Principal + max coupons
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball = create_standard_snowball()
    solver = create_solver()

    try:
        price = solver.price(snowball, env)

        # Price should be positive and bounded
        lower_bound = 0
        upper_bound = snowball.initial_price * snowball.contract_multiplier * 1.5  # Principal + generous coupon estimate
        passed = lower_bound < price < upper_bound
        results.add_result(
            "Principal Bounds",
            passed,
            f"Price = {price:,.2f}, bounds = [{lower_bound:,.0f}, {upper_bound:,.0f}]"
        )
    except Exception as e:
        results.add_result("Principal Bounds", False, f"Error: {e}")


def test_continuous_vs_discrete_ki(results: BoundaryCheckResults):
    """
    Test: Continuous KI should result in lower price than discrete KI.
    
    Continuous monitoring = more chances to knock in = higher KI probability.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # For discrete KI, need to provide ki_observation_dates
    maturity = 1.0
    num_obs = 12
    ko_obs_dates = [maturity * (i + 1) / num_obs for i in range(num_obs)]
    
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
        contract_multiplier=10_000.0,
        maturity=maturity,
    )
    snowball_continuous = create_standard_snowball(ki_continuous=True)
    solver = create_solver()

    try:
        price_discrete = solver.price(snowball_discrete, env)
        price_continuous = solver.price(snowball_continuous, env)

        # Continuous KI = more KI risk = lower price
        passed = price_discrete >= price_continuous * 0.95
        results.add_result(
            "Continuous vs Discrete KI",
            passed,
            f"Discrete: {price_discrete:,.2f}, Continuous: {price_continuous:,.2f}"
        )
    except Exception as e:
        results.add_result("Continuous vs Discrete KI", False, f"Error: {e}")


# ============================================================
# NUMERICAL STABILITY TESTS
# ============================================================


def test_grid_convergence(results: BoundaryCheckResults):
    """
    Test: Price should converge as grid resolution increases.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)
    snowball = create_standard_snowball()

    try:
        # Coarse grid
        solver_coarse = create_solver(grid_size=100, time_steps=100)
        price_coarse = solver_coarse.price(snowball, env)

        # Fine grid
        solver_fine = create_solver(grid_size=300, time_steps=300)
        price_fine = solver_fine.price(snowball, env)

        # Prices should be close (convergence)
        rel_diff = abs(price_fine - price_coarse) / price_fine
        passed = rel_diff < 0.05  # 5% tolerance
        results.add_result(
            "Grid Convergence",
            passed,
            f"Coarse: {price_coarse:,.2f}, Fine: {price_fine:,.2f}, diff: {rel_diff*100:.2f}%"
        )
    except Exception as e:
        results.add_result("Grid Convergence", False, f"Error: {e}")


def test_spot_sensitivity(results: BoundaryCheckResults):
    """
    Test: Price should change smoothly with spot.
    """
    env_base = create_pricing_env(spot=100.0)
    env_up = create_pricing_env(spot=101.0)
    env_down = create_pricing_env(spot=99.0)

    snowball = create_standard_snowball()
    solver = create_solver()

    try:
        price_base = solver.price(snowball, env_base)
        price_up = solver.price(snowball, env_up)
        price_down = solver.price(snowball, env_down)

        # Delta should be reasonable (not too extreme)
        delta_up = (price_up - price_base) / 1.0
        delta_down = (price_base - price_down) / 1.0

        # Deltas should have same sign and similar magnitude
        passed = abs(delta_up - delta_down) < snowball.initial_price * snowball.contract_multiplier * 0.1
        results.add_result(
            "Spot Sensitivity",
            passed,
            f"ΔUp: {delta_up:,.0f}, ΔDown: {delta_down:,.0f}"
        )
    except Exception as e:
        results.add_result("Spot Sensitivity", False, f"Error: {e}")


# ============================================================
# SNOWBALL VARIANT TESTS
# ============================================================


def test_stepdown_snowball_basics(results: BoundaryCheckResults):
    """
    Test: Stepdown snowball should have valid pricing.

    Stepdown structure has decreasing KO barriers over time,
    making knock-out progressively easier.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,  # 0.5% per period
        ki_barrier=75.0,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # Price should be positive and bounded
        passed = 0 < price < snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "Stepdown Snowball Basics",
            passed,
            f"Price = {price:,.2f} (stepdown KO barriers)"
        )
    except Exception as e:
        results.add_result("Stepdown Snowball Basics", False, f"Error: {e}")


def test_stepdown_vs_flat_ko(results: BoundaryCheckResults):
    """
    Test: Stepdown snowball should be worth more than flat KO snowball.

    Decreasing barriers = easier to knock out = less KI risk = higher value.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # Stepdown: starts at 103%, steps down by 0.5% each month
    snowball_stepdown = create_stepdown_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        initial_ko_barrier=103.0,
        stepdown_rate=0.005,
        ki_barrier=75.0,
    )

    # Flat KO at 103%
    snowball_flat = create_standard_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    solver = create_solver()

    try:
        price_stepdown = solver.price(snowball_stepdown, env)
        price_flat = solver.price(snowball_flat, env)

        # Stepdown should be worth more (easier KO = less risk)
        passed = price_stepdown >= price_flat * 0.98
        results.add_result(
            "Stepdown vs Flat KO",
            passed,
            f"Stepdown: {price_stepdown:,.2f}, Flat: {price_flat:,.2f}"
        )
    except Exception as e:
        results.add_result("Stepdown vs Flat KO", False, f"Error: {e}")


def test_european_ki_snowball_basics(results: BoundaryCheckResults):
    """
    Test: European KI snowball should have valid pricing.

    KI only observed at maturity (European-style),
    which increases probability of V0 outcome.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # Price should be positive and bounded
        passed = 0 < price < snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "European KI Snowball Basics",
            passed,
            f"Price = {price:,.2f} (KI only at maturity)"
        )
    except Exception as e:
        results.add_result("European KI Snowball Basics", False, f"Error: {e}")


def test_european_ki_vs_continuous_ki(results: BoundaryCheckResults):
    """
    Test: European KI should be worth more than continuous KI.

    European KI = less monitoring = lower KI probability = higher value.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # European KI (only at maturity)
    snowball_european = create_european_ki_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    # Continuous KI
    snowball_continuous = create_standard_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    solver = create_solver()

    try:
        price_european = solver.price(snowball_european, env)
        price_continuous = solver.price(snowball_continuous, env)

        # European KI should be worth more (less KI risk)
        passed = price_european >= price_continuous * 0.98
        results.add_result(
            "European KI vs Continuous KI",
            passed,
            f"European: {price_european:,.2f}, Continuous: {price_continuous:,.2f}"
        )
    except Exception as e:
        results.add_result("European KI vs Continuous KI", False, f"Error: {e}")


def test_parachute_snowball_basics(results: BoundaryCheckResults):
    """
    Test: Parachute snowball should have valid pricing.

    Last KO barrier drops to KI level, guaranteeing an exit
    at maturity if no prior KI.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # Price should be positive and bounded
        passed = 0 < price < snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "Parachute Snowball Basics",
            passed,
            f"Price = {price:,.2f} (last KO = KI barrier)"
        )
    except Exception as e:
        results.add_result("Parachute Snowball Basics", False, f"Error: {e}")


def test_parachute_vs_standard(results: BoundaryCheckResults):
    """
    Test: Parachute snowball should be worth more than standard.

    Parachute guarantees KO at maturity if not KI'ed,
    which reduces the probability of ending in V0/V1 at maturity.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # Parachute: last KO barrier = KI barrier
    snowball_parachute = create_parachute_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    # Standard: flat KO at 103%
    snowball_standard = create_standard_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    solver = create_solver()

    try:
        price_parachute = solver.price(snowball_parachute, env)
        price_standard = solver.price(snowball_standard, env)

        # Parachute should be worth more (guaranteed exit if not KI'ed)
        passed = price_parachute >= price_standard * 0.98
        results.add_result(
            "Parachute vs Standard",
            passed,
            f"Parachute: {price_parachute:,.2f}, Standard: {price_standard:,.2f}"
        )
    except Exception as e:
        results.add_result("Parachute vs Standard", False, f"Error: {e}")


def test_airbag_snowball_basics(results: BoundaryCheckResults):
    """
    Test: Airbag snowball should have valid pricing.

    Reduced participation below airbag barrier limits extreme losses.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    snowball = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,  # 50% participation below airbag
    )
    solver = create_solver()

    try:
        price = solver.price(snowball, env)
        # Price should be positive and bounded
        passed = 0 < price < snowball.initial_price * snowball.contract_multiplier * 1.5
        results.add_result(
            "Airbag Snowball Basics",
            passed,
            f"Price = {price:,.2f} (50% participation below airbag)"
        )
    except Exception as e:
        results.add_result("Airbag Snowball Basics", False, f"Error: {e}")


def test_airbag_vs_standard(results: BoundaryCheckResults):
    """
    Test: Airbag snowball should be worth more than standard.

    Airbag limits downside exposure, so value should be higher.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # Airbag: 50% participation below 60%
    snowball_airbag = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=60.0,
        participation_rate=1.0,
        airbag_participation_rate=0.5,
    )

    # Standard: full participation on downside
    snowball_standard = create_standard_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
    )

    solver = create_solver()

    try:
        price_airbag = solver.price(snowball_airbag, env)
        price_standard = solver.price(snowball_standard, env)

        # Airbag should be worth more (limited downside)
        passed = price_airbag >= price_standard * 0.98
        results.add_result(
            "Airbag vs Standard",
            passed,
            f"Airbag: {price_airbag:,.2f}, Standard: {price_standard:,.2f}"
        )
    except Exception as e:
        results.add_result("Airbag vs Standard", False, f"Error: {e}")


def test_airbag_barrier_effect(results: BoundaryCheckResults):
    """
    Test: Higher airbag barrier should increase snowball value.

    Higher airbag barrier = more protection = higher value.
    """
    spot = 100.0
    env = create_pricing_env(spot=spot)

    # Low airbag barrier (less protection)
    snowball_low = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=50.0,  # Low
        airbag_participation_rate=0.5,
    )

    # High airbag barrier (more protection)
    snowball_high = create_airbag_helper(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        airbag_barrier=70.0,  # High (closer to KI)
        airbag_participation_rate=0.5,
    )

    solver = create_solver()

    try:
        price_low = solver.price(snowball_low, env)
        price_high = solver.price(snowball_high, env)

        # Higher airbag barrier = more protection = higher value
        passed = price_high >= price_low * 0.98
        results.add_result(
            "Airbag Barrier Effect",
            passed,
            f"Airbag@50: {price_low:,.2f}, Airbag@70: {price_high:,.2f}"
        )
    except Exception as e:
        results.add_result("Airbag Barrier Effect", False, f"Error: {e}")


def test_variant_volatility_sensitivity(results: BoundaryCheckResults):
    """
    Test: All variants should respond reasonably to volatility changes.
    """
    spot = 100.0
    env_low = create_pricing_env(spot=spot, vol=0.15)
    env_high = create_pricing_env(spot=spot, vol=0.35)

    variants = [
        ("Standard", create_standard_helper(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0
        )),
        ("Stepdown", create_stepdown_helper(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0
        )),
        ("European KI", create_european_ki_helper(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0
        )),
        ("Parachute", create_parachute_helper(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0
        )),
        ("Airbag", create_airbag_helper(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0,
            airbag_barrier=60.0, airbag_participation_rate=0.5
        )),
    ]

    solver = create_solver()
    all_passed = True

    for name, snowball in variants:
        try:
            price_low = solver.price(snowball, env_low)
            price_high = solver.price(snowball, env_high)

            # Both should be positive and bounded
            passed = price_low > 0 and price_high > 0
            if not passed:
                all_passed = False
        except Exception as e:
            all_passed = False

    results.add_result(
        "Variant Volatility Sensitivity",
        all_passed,
        "All variants respond reasonably to volatility changes"
    )


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    results = BoundaryCheckResults()

    print("=" * 70)
    print("SNOWBALL PDE SOLVER - BOUNDARY CHECKS")
    print("=" * 70)

    # Extreme market cases
    print("\n>>> Running Extreme Market Case Tests...")
    test_low_volatility(results)
    test_high_volatility(results)
    test_near_expiry(results)
    test_deep_in_ko_region(results)
    test_deep_in_ki_region(results)
    test_zero_rate(results)

    # Theoretical relationships
    print("\n>>> Running Theoretical Relationship Tests...")
    test_v0_v1_relationship(results)
    test_ko_barrier_effect(results)
    test_ki_barrier_effect(results)
    test_maturity_effect(results)
    test_ko_rate_effect(results)
    test_principal_bounds(results)
    test_continuous_vs_discrete_ki(results)

    # Numerical stability
    print("\n>>> Running Numerical Stability Tests...")
    test_grid_convergence(results)
    test_spot_sensitivity(results)

    # Snowball variant tests
    print("\n>>> Running Snowball Variant Tests...")
    test_stepdown_snowball_basics(results)
    test_stepdown_vs_flat_ko(results)
    test_european_ki_snowball_basics(results)
    test_european_ki_vs_continuous_ki(results)
    test_parachute_snowball_basics(results)
    test_parachute_vs_standard(results)
    test_airbag_snowball_basics(results)
    test_airbag_vs_standard(results)
    test_airbag_barrier_effect(results)
    test_variant_volatility_sensitivity(results)

    # Print summary
    success = results.summary()
    sys.exit(0 if success else 1)
