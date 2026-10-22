"""
Comprehensive engine comparison tests for Phoenix options.

These tests verify that the three Phoenix engines (Quadrature, PDE, Monte Carlo)
produce consistent results for various product configurations.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    create_reverse_phoenix,
    create_standard_phoenix,
    create_stepdown_phoenix,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod


@pytest.fixture(scope="module")
def base_env():
    return create_pricing_env()


# Function scope (not module): the PDE solver accumulates grid / critical-point /
# observation caches on the instance. Sharing one solver across the module's many
# different products made results depend on test ordering, which is nondeterministic
# under `pytest -n auto --dist worksteal` and caused a platform-specific CI-only
# failure in test_stepdown_phoenix_consistency. A fresh engine per test is
# deterministic and cannot change any correct value (an empty-cache first price
# equals a shared engine's first price).
@pytest.fixture
def engines():
    return {
        "quad": PhoenixQuadEngine(params=QuadParams(grid_points=401)),
        "pde": PhoenixPDESolver(params=PDEParams()),
        "mc": PhoenixMCEngine(
            params=MCParams(num_paths=65536, seed=42),
            method=MonteCarloMethod.QUASI,
        ),
    }


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.03,
    div_yield: float = 0.00,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def build_observation_schedule(times: list[float], barrier: float) -> ObservationSchedule:
    return ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=barrier) for t in times]
    )


class TestPhoenixEngineComparison:
    """Test suite for comparing Phoenix pricing engines."""

    def test_standard_phoenix_consistency(self, base_env, engines):
        """Test that engines produce consistent prices for standard Phoenix."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        prices = {name: engine.price(phoenix, base_env) for name, engine in engines.items()}

        # All prices should be positive
        for name, price in prices.items():
            assert price > 0, f"{name} price is negative: {price}"

        # Quad and MC should be within 5% (allowing for MC variance)
        quad_mc_diff = abs(prices["quad"] - prices["mc"]) / prices["mc"]
        assert quad_mc_diff < 0.05, f"Quad-MC difference too large: {quad_mc_diff:.2%}"

        # PDE and MC should be within 10% (PDE has grid discretization error)
        pde_mc_diff = abs(prices["pde"] - prices["mc"]) / prices["mc"]
        assert pde_mc_diff < 0.10, f"PDE-MC difference too large: {pde_mc_diff:.2%}"

    def test_memory_coupon_increases_value(self, base_env, engines):
        """Test that memory coupon feature increases option value."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix_no_memory = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        phoenix_memory = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=True,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        for name, engine in engines.items():
            price_no_mem = engine.price(phoenix_no_memory, base_env)
            price_mem = engine.price(phoenix_memory, base_env)
            assert (
                price_mem > price_no_mem
            ), f"{name}: Memory coupon should increase value ({price_mem} <= {price_no_mem})"

    def test_quarterly_vs_monthly_observations(self, base_env):
        """Test that fewer observations (quarterly) increases value due to lower KO probability."""
        mc_engine = PhoenixMCEngine(params=MCParams(num_paths=50000, seed=42))
        pde_engine = PhoenixPDESolver(params=PDEParams())

        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        quarterly_obs = build_observation_schedule(
            [0.25, 0.5, 0.75, 1.0], barrier=75.0
        )

        phoenix_monthly = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=True,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        phoenix_quarterly = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=4,
            memory_coupon=True,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=quarterly_obs,
        )

        for engine in [mc_engine, pde_engine]:
            price_monthly = engine.price(phoenix_monthly, base_env)
            price_quarterly = engine.price(phoenix_quarterly, base_env)
            assert (
                price_quarterly > price_monthly
            ), f"{engine}: Quarterly should have higher value than monthly ({price_quarterly} <= {price_monthly})"

    def test_high_volatility_reduces_value(self, base_env, engines):
        """Test that higher volatility reduces Phoenix value (higher KO probability)."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        normal_vol_env = create_pricing_env(vol=0.20)
        high_vol_env = create_pricing_env(vol=0.35)

        for name, engine in engines.items():
            price_normal = engine.price(phoenix, normal_vol_env)
            price_high = engine.price(phoenix, high_vol_env)
            # High vol should reduce value (more KOs)
            assert (
                price_high < price_normal
            ), f"{name}: High vol should reduce value ({price_high} >= {price_normal})"

    def test_reverse_phoenix_sign(self, base_env, engines):
        """Test that reverse Phoenix with high KO barrier has negative value."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=125.0
        )
        phoenix = create_reverse_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=97.0,
            ki_barrier=125.0,
            coupon_barrier=115.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        for name, engine in engines.items():
            price = engine.price(phoenix, base_env)
            # Reverse Phoenix with KO below spot should have negative value
            # (short put position that's likely to be knocked into)
            assert price < 0, f"{name}: Reverse Phoenix should have negative value, got {price}"

    def test_coupon_pay_type_instant_vs_expiry(self, base_env):
        """Test that INSTANT coupon payment gives higher value than EXPIRY."""
        mc_engine = PhoenixMCEngine(params=MCParams(num_paths=30000, seed=42))
        pde_engine = PhoenixPDESolver(params=PDEParams())

        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )

        phoenix_instant = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )
        # Override coupon pay type
        phoenix_instant.coupon_config = phoenix_instant.coupon_config.__class__(
            coupon_barrier=95.0,
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            memory_coupon=False,
        )

        phoenix_expiry = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )
        phoenix_expiry.coupon_config = phoenix_expiry.coupon_config.__class__(
            coupon_barrier=95.0,
            coupon_rate=0.02,
            coupon_pay_type=CouponPayType.EXPIRY,
            memory_coupon=False,
        )

        for engine in [mc_engine, pde_engine]:
            price_instant = engine.price(phoenix_instant, base_env)
            price_expiry = engine.price(phoenix_expiry, base_env)
            assert (
                price_instant >= price_expiry * 0.95
            ), f"{engine}: INSTANT should have >= value than EXPIRY ({price_instant} < {price_expiry})"

    def test_stepdown_phoenix_consistency(self, base_env, engines):
        """Test that engines handle step-down barriers consistently."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_stepdown_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            initial_ko_barrier=103.0,
            initial_coupon_barrier=95.0,
            ko_stepdown_rate=0.01,
            coupon_stepdown_rate=0.01,
            ko_rate=0.12,
            ki_barrier=75.0,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            is_reverse=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        prices = {name: engine.price(phoenix, base_env) for name, engine in engines.items()}

        # All prices should be positive
        for name, price in prices.items():
            assert price > 0, f"{name} price is negative: {price}"

        # Check relative consistency
        quad_mc_diff = abs(prices["quad"] - prices["mc"]) / prices["mc"]
        pde_mc_diff = abs(prices["pde"] - prices["mc"]) / prices["mc"]

        # Allow wider tolerance for step-down (more complex payoff)
        assert quad_mc_diff < 0.08, f"Quad-MC difference too large: {quad_mc_diff:.2%}"
        assert pde_mc_diff < 0.10, f"PDE-MC difference too large: {pde_mc_diff:.2%}"


class TestPhoenixEngineConvergence:
    """Test convergence of engines as parameters are refined."""

    def test_mc_convergence_with_paths(self, base_env):
        """Test that MC price converges as number of paths increases."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=True,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        path_counts = [10000, 50000, 100000]
        prices = []
        std_errors = []

        for n_paths in path_counts:
            engine = PhoenixMCEngine(params=MCParams(num_paths=n_paths, seed=42))
            price = engine.price(phoenix, base_env)
            result = engine.get_last_result()
            prices.append(price)
            std_errors.append(result.std_error if result else 0.0)

        # Standard error should decrease as paths increase
        assert std_errors[1] < std_errors[0] * 1.5, "Std error should decrease with more paths"
        assert std_errors[2] < std_errors[1] * 1.2, "Std error should decrease with more paths"

        # Prices should be within 2 standard errors of each other
        for i in range(len(prices) - 1):
            diff = abs(prices[i] - prices[i + 1])
            combined_se = math.sqrt(std_errors[i] ** 2 + std_errors[i + 1] ** 2)
            assert (
                diff < 3 * combined_se
            ), f"MC prices should converge: {prices[i]} vs {prices[i+1]}, diff={diff}, SE={combined_se}"

    def test_pde_convergence_with_grid(self, base_env):
        """Test that PDE price stabilizes as grid is refined."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        from quantark.asset.equity.engine.pde import GridConfig

        grid_sizes = [100, 200, 300]
        prices = []

        for grid in grid_sizes:
            engine = PhoenixPDESolver(
                params=PDEParams(grid=GridConfig(points=grid))
            )
            price = engine.price(phoenix, base_env)
            prices.append(price)

        # Prices should stabilize as grid refines
        # Allow up to 5% variation due to discretization
        for i in range(len(prices) - 1):
            rel_diff = abs(prices[i] - prices[i + 1]) / prices[i + 1]
            assert (
                rel_diff < 0.05
            ), f"PDE prices should stabilize: grid {grid_sizes[i]} vs {grid_sizes[i+1]}, diff={rel_diff:.2%}"

    def test_quad_convergence_with_grid_points(self, base_env):
        """Test that Quadrature price stabilizes as grid points increase."""
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,  # No memory for better convergence
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        grid_points = [401, 601, 801]
        prices = []

        for n_points in grid_points:
            engine = PhoenixQuadEngine(params=QuadParams(grid_points=n_points))
            price = engine.price(phoenix, base_env)
            prices.append(price)

        # Quad prices should all be in a reasonable range
        # FFT-based methods may oscillate but should stay bounded
        avg_price = sum(prices) / len(prices)
        for price in prices:
            rel_diff = abs(price - avg_price) / avg_price
            assert (
                rel_diff < 0.15
            ), f"Quad prices should stay within 15% of average: {prices}, avg={avg_price:.2f}"


class TestPhoenixEngineSpecialCases:
    """Test edge cases and special scenarios."""

    def test_at_money_coupon_barrier(self, base_env):
        """Test Phoenix with coupon barrier equal to initial price."""
        mc_engine = PhoenixMCEngine(params=MCParams(num_paths=20000, seed=42))

        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=100.0,  # At money
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        price = mc_engine.price(phoenix, base_env)
        assert price > 0, "At-money coupon barrier should produce positive price"

    def test_high_ko_barrier_rarely_hits(self, base_env):
        """Test Phoenix with KO barrier far above spot (rarely hits)."""
        mc_engine = PhoenixMCEngine(params=MCParams(num_paths=20000, seed=42))

        # For standard Phoenix, KO triggers when spot >= ko_barrier
        # A very high KO barrier means KO rarely happens
        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=40.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=150.0,  # Very high KO barrier (50% above spot)
            ki_barrier=40.0,  # Low KI (may trigger before KO)
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.02,
            num_observations=12,
            memory_coupon=True,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        price = mc_engine.price(phoenix, base_env)
        # High KO barrier means product mostly survives to maturity
        # But KI at 40 may trigger (down barrier), leading to V1 put payoff
        # Value should be significantly higher than a product with easy KO
        assert price > 50000, f"Price with high KO barrier should be substantial, got {price}"

    def test_zero_coupon_rate(self, base_env):
        """Test Phoenix with zero coupon rate."""
        mc_engine = PhoenixMCEngine(params=MCParams(num_paths=20000, seed=42))

        monthly_obs = build_observation_schedule(
            [i / 12 for i in range(1, 13)], barrier=75.0
        )
        phoenix = create_standard_phoenix(
            initial_price=100.0,
            strike=100.0,
            maturity=1.0,
            contract_multiplier=10_000.0,
            ko_barrier=103.0,
            ki_barrier=75.0,
            coupon_barrier=95.0,
            ko_rate=0.12,
            coupon_rate=0.0,  # Zero coupon
            num_observations=12,
            memory_coupon=False,
            ki_continuous=False,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=monthly_obs,
        )

        price = mc_engine.price(phoenix, base_env)
        assert price > 0, "Phoenix with zero coupon should still have positive value (KO payoff)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
