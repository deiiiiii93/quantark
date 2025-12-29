"""
Unit tests for SnowballPDESolver.

Tests cover:
- Basic standard and reverse snowball pricing
- Two-Surface PDE method validation
- Convergence behavior (grid refinement)
- Edge cases (near-expiry, ATM barriers, already knocked-in/out)
- Comparison with Monte Carlo engine
- Different barrier monitoring types (continuous vs discrete KI)
- KO payoff timing (INSTANT vs EXPIRY)
- Time-varying barriers (step-down)
- PDEEngine facade dispatch
- Greeks calculation via numerical bumping
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.engine.pde import SnowballPDESolver
from asset.equity.engine.pde_engine import PDEEngine
from asset.equity.param import MCParams, PDEParams
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from asset.equity.product.option.snowball_option import SnowballOption
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import (
    CouponPayType,
    ObservationType,
)
from util.exceptions import PricingError, ValidationError
from util.numerical import is_close

# =============================================================================
# Fixtures - Common test configurations
# =============================================================================


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div_yield: float = 0.02,
) -> PricingEnvironment:
    """Create a basic pricing environment for testing."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def create_basic_barrier_config(
    ko_barrier: float = 103.0,
    ko_rate: float = 0.15,
    ki_barrier: float = 75.0,
    ko_observation_dates: List[float] = None,
    ki_observation_dates: List[float] = None,
    ki_continuous: bool = True,
    disable_ko_after_ki: bool = False,
) -> BarrierConfig:
    """Create a basic barrier configuration for testing."""
    if ko_observation_dates is None:
        ko_observation_dates = [0.25, 0.5, 0.75, 1.0]

    return BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_observation_dates,
        ki_barrier=ki_barrier,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_observation_dates=ki_observation_dates,
        ki_continuous=ki_continuous,
        disable_ko_after_ki=disable_ko_after_ki,
    )


def create_standard_snowball(
    initial_price: float = 100.0,
    strike: float = 100.0,
    notional: float = 1_000_000.0,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    payoff_config: PayoffConfig = None,
    accrual_config: AccrualConfig = None,
) -> SnowballOption:
    """Create a standard snowball option for testing."""
    if barrier_config is None:
        barrier_config = create_basic_barrier_config()

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        notional=notional,
        maturity=maturity,
        is_reverse=False,
    )


def create_reverse_snowball(
    initial_price: float = 100.0,
    strike: float = 100.0,
    notional: float = 1_000_000.0,
    maturity: float = 1.0,
    barrier_config: BarrierConfig = None,
    payoff_config: PayoffConfig = None,
    accrual_config: AccrualConfig = None,
) -> SnowballOption:
    """Create a reverse snowball option for testing."""
    if barrier_config is None:
        # Reverse snowball: DOWN KO, UP KI
        barrier_config = BarrierConfig(
            ko_barrier=97.0,  # DOWN barrier
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=125.0,  # UP barrier
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        )

    return SnowballOption(
        initial_price=initial_price,
        strike=strike,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        notional=notional,
        maturity=maturity,
        is_reverse=True,
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestSnowballPDESolverBasic:
    """Basic tests for SnowballPDESolver."""

    def test_solver_instantiation(self):
        """Test that solver can be instantiated with default params."""
        solver = SnowballPDESolver()
        assert solver is not None

    def test_solver_with_custom_params(self):
        """Test solver instantiation with custom PDE parameters."""
        params = PDEParams(grid_size=300, time_steps=150)
        solver = SnowballPDESolver(params=params)
        assert solver.params.grid_size == 300
        assert solver.params.time_steps == 150

    def test_standard_snowball_pricing(self):
        """Test basic standard snowball pricing returns a positive value."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)

        # Price should be positive (option has value)
        assert price > 0
        # Price should be less than notional (sanity check)
        assert price < snowball.notional

    def test_reverse_snowball_pricing(self):
        """Test basic reverse snowball pricing returns a positive value."""
        snowball = create_reverse_snowball()
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)

        assert price > 0
        assert price < snowball.notional

    def test_invalid_product_type(self):
        """Test that solver rejects non-SnowballOption products."""
        from asset.equity.product.option import EuropeanVanillaOption
        from util.enum import OptionType

        european = EuropeanVanillaOption(
            strike=100.0, maturity=1.0, option_type=OptionType.CALL
        )
        env = create_pricing_env()
        solver = SnowballPDESolver()

        with pytest.raises(PricingError):
            solver.price(european, env)


class TestSnowballPDEEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_near_expiry_pricing(self):
        """Test pricing when very close to expiry."""
        barrier_config = create_basic_barrier_config(
            ko_observation_dates=[0.01]  # Single observation near expiry
        )
        snowball = create_standard_snowball(
            maturity=0.01, barrier_config=barrier_config
        )
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=100, time_steps=50))

        # Should not raise and return a finite value
        price = solver.price(snowball, env)
        assert np.isfinite(price)

    def test_high_volatility(self):
        """Test pricing with high volatility."""
        snowball = create_standard_snowball()
        env = create_pricing_env(vol=0.50)  # 50% vol
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_low_volatility(self):
        """Test pricing with low volatility."""
        snowball = create_standard_snowball()
        env = create_pricing_env(vol=0.05)  # 5% vol
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_spot_at_ko_barrier(self):
        """Test pricing when spot is at the KO barrier."""
        barrier_config = create_basic_barrier_config(ko_barrier=103.0)
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env(spot=103.0)  # At KO barrier
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)

    def test_spot_at_ki_barrier(self):
        """Test pricing when spot is at the KI barrier."""
        barrier_config = create_basic_barrier_config(ki_barrier=75.0)
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env(spot=75.0)  # At KI barrier
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)

    def test_already_knocked_in(self):
        """Test pricing when spot is already below KI barrier."""
        barrier_config = create_basic_barrier_config(
            ki_barrier=75.0, ki_continuous=True
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env(spot=70.0)  # Below KI barrier
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        # With continuous KI, when spot starts below KI, the option starts knocked-in

    def test_discrete_ko_does_not_trigger_without_time0_observation(self):
        """Discrete KO should not be treated as triggered at valuation unless t=0 is an observation."""
        barrier_config = create_basic_barrier_config(
            ko_barrier=103.0,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env(spot=150.0)  # Above KO barrier but no t=0 observation
        solver = SnowballPDESolver(PDEParams(grid_size=150, time_steps=75))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert solver._grid_v0 is not None  # Should run PDE, not early KO return

    def test_immediate_ko_uses_time0_record_not_past(self):
        """Immediate KO payoff should use the t=0 observation record, not any past schedule records."""
        valuation_date = datetime(2024, 1, 1)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=150.0, timestamp=valuation_date),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=valuation_date,
        )

        ko_schedule = ObservationSchedule(
            records=[
                ObservationRecord(
                    observation_time=-0.25,
                    barrier=103.0,
                    return_rate=0.30,
                ),
                ObservationRecord(
                    observation_time=0.0,
                    barrier=103.0,
                    return_rate=0.00,
                ),
                ObservationRecord(
                    observation_time=0.25,
                    barrier=103.0,
                    return_rate=0.01,
                ),
            ]
        )
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.01,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ko_schedule,
            ki_barrier=None,
        )
        snowball = create_standard_snowball(
            barrier_config=barrier_config,
            accrual_config=AccrualConfig(is_annualized=False),
        )

        solver = SnowballPDESolver(PDEParams(grid_size=150, time_steps=75))
        price = solver.price(snowball, env)

        ko_records = snowball.resolve_ko_observations(env)
        record_0 = next(
            rec for rec in ko_records if is_close(rec.observation_time, 0.0)
        )
        assert price == pytest.approx(record_0.payoff, rel=0.0, abs=1e-10)


class TestSnowballPDEConvergence:
    """Tests for grid convergence behavior."""

    def test_price_converges_with_grid_refinement(self):
        """Test that price converges as grid is refined."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        # Price with different grid sizes (disable auto_grid for true convergence test)
        prices = []
        for grid_size in [100, 200, 400]:
            solver = SnowballPDESolver(
                PDEParams(
                    grid_size=grid_size, time_steps=grid_size // 2, auto_grid=False
                )
            )
            prices.append(solver.price(snowball, env))

        # Prices should be finite and positive
        for price in prices:
            assert np.isfinite(price)
            assert price > 0

        # Prices should generally converge (variation should decrease)
        # Allow some tolerance as PDE methods may have oscillations
        spread = max(prices) - min(prices)
        assert spread / np.mean(prices) < 0.10  # Within 10% spread

    def test_price_stability_with_time_refinement(self):
        """Test that price is stable with time step refinement."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        prices = []
        for time_steps in [50, 100, 200]:
            solver = SnowballPDESolver(
                PDEParams(grid_size=200, time_steps=time_steps, auto_grid=False)
            )
            prices.append(solver.price(snowball, env))

        # Prices should be relatively stable (within 10%)
        mean_price = np.mean(prices)
        for price in prices:
            assert abs(price - mean_price) / mean_price < 0.10


class TestSnowballPDEVsMC:
    """Tests comparing PDE solver with Monte Carlo engine."""

    @pytest.mark.slow
    def test_pde_vs_mc_standard_snowball(self):
        """Test that PDE and MC prices converge for standard snowball."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        # PDE price with fine grid
        pde_solver = SnowballPDESolver(PDEParams(grid_size=400, time_steps=200))
        pde_price = pde_solver.price(snowball, env)

        # MC price with many paths
        mc_engine = SnowballMCEngine(MCParams(num_paths=100000, num_steps=252))
        mc_price = mc_engine.price(snowball, env)

        # Should be within 2% of each other
        rel_diff = abs(pde_price - mc_price) / max(abs(pde_price), abs(mc_price))
        assert rel_diff < 0.02, (
            f"PDE={pde_price:.4f}, MC={mc_price:.4f}, diff={rel_diff:.4%}"
        )

    @pytest.mark.slow
    def test_pde_vs_mc_reverse_snowball(self):
        """Test that PDE and MC prices converge for reverse snowball."""
        snowball = create_reverse_snowball()
        env = create_pricing_env()

        pde_solver = SnowballPDESolver(PDEParams(grid_size=400, time_steps=200))
        pde_price = pde_solver.price(snowball, env)

        mc_engine = SnowballMCEngine(MCParams(num_paths=100000, num_steps=252))
        mc_price = mc_engine.price(snowball, env)

        rel_diff = abs(pde_price - mc_price) / max(abs(pde_price), abs(mc_price))
        assert rel_diff < 0.02, (
            f"PDE={pde_price:.4f}, MC={mc_price:.4f}, diff={rel_diff:.4%}"
        )


class TestSnowballPDEBarrierTypes:
    """Tests for different barrier monitoring types."""

    def test_continuous_ki_monitoring(self):
        """Test snowball with continuous KI monitoring."""
        barrier_config = create_basic_barrier_config(ki_continuous=True)
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_discrete_ki_monitoring(self):
        """Test snowball with discrete KI monitoring."""
        barrier_config = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_continuous=False,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_discrete_ki_higher_than_continuous(self):
        """Test that discrete KI monitoring gives higher price than continuous."""
        # Discrete KI is less likely to trigger, so option value should be higher
        barrier_config_continuous = create_basic_barrier_config(ki_continuous=True)
        barrier_config_discrete = BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_continuous=False,
        )

        snowball_continuous = create_standard_snowball(
            barrier_config=barrier_config_continuous
        )
        snowball_discrete = create_standard_snowball(
            barrier_config=barrier_config_discrete
        )
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=300, time_steps=150))

        price_continuous = solver.price(snowball_continuous, env)
        price_discrete = solver.price(snowball_discrete, env)

        # Discrete KI should have higher price (less KI risk)
        assert price_discrete > price_continuous


class TestSnowballPDEStepDown:
    """Tests for step-down KO barrier patterns."""

    def test_stepdown_ko_barrier(self):
        """Test snowball with step-down KO barriers."""
        # Barriers decrease each quarter
        barrier_config = BarrierConfig(
            ko_barrier=[103.0, 102.0, 101.0, 100.0],  # Step-down
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        )
        snowball = create_standard_snowball(barrier_config=barrier_config)
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_stepdown_higher_value_than_flat(self):
        """Test that step-down barriers give higher value than flat barriers."""
        # Step-down makes KO more likely later → higher value for seller
        barrier_config_flat = create_basic_barrier_config(ko_barrier=103.0)
        barrier_config_stepdown = BarrierConfig(
            ko_barrier=[103.0, 102.0, 101.0, 100.0],
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        )

        snowball_flat = create_standard_snowball(barrier_config=barrier_config_flat)
        snowball_stepdown = create_standard_snowball(
            barrier_config=barrier_config_stepdown
        )
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=300, time_steps=150))

        price_flat = solver.price(snowball_flat, env)
        price_stepdown = solver.price(snowball_stepdown, env)

        # Step-down should have higher price (easier KO later)
        assert price_stepdown > price_flat


class TestSnowballPDEKOPayoffTiming:
    """Tests for KO payoff timing (INSTANT vs EXPIRY)."""

    def test_instant_ko_payoff(self):
        """Test snowball with INSTANT KO payoff timing."""
        accrual_config = AccrualConfig(coupon_pay_type=CouponPayType.INSTANT)
        snowball = create_standard_snowball(accrual_config=accrual_config)
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_expiry_ko_payoff(self):
        """Test snowball with EXPIRY KO payoff timing."""
        accrual_config = AccrualConfig(coupon_pay_type=CouponPayType.EXPIRY)
        snowball = create_standard_snowball(accrual_config=accrual_config)
        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=200, time_steps=100))

        price = solver.price(snowball, env)
        assert np.isfinite(price)
        assert price > 0

    def test_instant_higher_value_than_expiry(self):
        """Test that INSTANT payment gives higher value than EXPIRY."""
        # Earlier payment is more valuable due to time value of money
        accrual_config_instant = AccrualConfig(coupon_pay_type=CouponPayType.INSTANT)
        accrual_config_expiry = AccrualConfig(coupon_pay_type=CouponPayType.EXPIRY)

        snowball_instant = create_standard_snowball(
            accrual_config=accrual_config_instant
        )
        snowball_expiry = create_standard_snowball(accrual_config=accrual_config_expiry)
        env = create_pricing_env(rate=0.10)  # Higher rate to amplify effect
        solver = SnowballPDESolver(PDEParams(grid_size=300, time_steps=150))

        price_instant = solver.price(snowball_instant, env)
        price_expiry = solver.price(snowball_expiry, env)

        # INSTANT should have higher price (time value of money)
        assert price_instant > price_expiry


class TestSnowballPDEEngineIntegration:
    """Tests for PDEEngine facade integration."""

    def test_pde_engine_dispatches_to_snowball_solver(self):
        """Test that PDEEngine correctly dispatches SnowballOption."""
        snowball = create_standard_snowball()
        env = create_pricing_env()

        engine = PDEEngine(PDEParams(grid_size=200, time_steps=100))
        price = engine.price(snowball, env)

        assert np.isfinite(price)
        assert price > 0

    def test_pde_engine_vs_direct_solver(self):
        """Test that PDEEngine gives same result as direct solver."""
        snowball = create_standard_snowball()
        env = create_pricing_env()
        params = PDEParams(grid_size=200, time_steps=100)

        engine = PDEEngine(params)
        solver = SnowballPDESolver(params)

        price_engine = engine.price(snowball, env)
        price_solver = solver.price(snowball, env)

        assert np.isclose(price_engine, price_solver, rtol=1e-10)


class TestSnowballPDEGreeks:
    """Tests for Greeks calculation using numerical bumping."""

    def test_delta_calculation(self):
        """Test delta calculation via spot bumping."""
        snowball = create_standard_snowball()
        solver = SnowballPDESolver(PDEParams(grid_size=300, time_steps=150))

        spot = 100.0
        bump = 0.01  # 1% bump

        env_base = create_pricing_env(spot=spot)
        env_up = create_pricing_env(spot=spot * (1 + bump))
        env_down = create_pricing_env(spot=spot * (1 - bump))

        price_base = solver.price(snowball, env_base)
        price_up = solver.price(snowball, env_up)
        price_down = solver.price(snowball, env_down)

        # Central difference delta
        delta = (price_up - price_down) / (2 * spot * bump)

        assert np.isfinite(delta)
        # Standard snowball has negative delta (short put)
        # With barriers, delta can vary but should be bounded

    def test_vega_calculation(self):
        """Test vega calculation via volatility bumping."""
        snowball = create_standard_snowball()
        solver = SnowballPDESolver(PDEParams(grid_size=300, time_steps=150))

        vol = 0.20
        bump = 0.01  # 1 vol point

        env_base = create_pricing_env(vol=vol)
        env_up = create_pricing_env(vol=vol + bump)

        price_base = solver.price(snowball, env_base)
        price_up = solver.price(snowball, env_up)

        # Forward difference vega
        vega = (price_up - price_base) / bump

        assert np.isfinite(vega)

    def test_theta_calculation(self):
        """Test theta calculation via maturity decay."""
        barrier_config = create_basic_barrier_config(
            ko_observation_dates=[0.5, 1.0]  # Reduced observations for shorter maturity
        )
        snowball_long = create_standard_snowball(
            maturity=1.0, barrier_config=barrier_config
        )

        barrier_config_short = create_basic_barrier_config(
            ko_observation_dates=[0.45, 0.95]  # Adjusted for 0.95 maturity
        )
        snowball_short = create_standard_snowball(
            maturity=0.95, barrier_config=barrier_config_short
        )

        env = create_pricing_env()
        solver = SnowballPDESolver(PDEParams(grid_size=300, time_steps=150))

        price_long = solver.price(snowball_long, env)
        price_short = solver.price(snowball_short, env)

        # Theta (price change per day)
        theta = (price_short - price_long) / 0.05  # 0.05 years ≈ 18 days

        assert np.isfinite(theta)


class TestSnowballPDERepr:
    """Tests for solver representation."""

    def test_repr(self):
        """Test string representation of solver."""
        solver = SnowballPDESolver()
        repr_str = repr(solver)

        assert "SnowballPDESolver" in repr_str
