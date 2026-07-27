"""
Tests for PDE pricing engine.

This module contains comprehensive tests for the PDE pricing framework including:
- PDEParams validation
- TimeGrid generation
- SpatialGrid generation
- All PDE solvers
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import products
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
    DoubleBarrierOption,
    OneTouchOption,
    DoubleOneTouchOption,
)

# Import engines
from quantark.asset.equity.engine import (
    BlackScholesEngine,
    BarrierAnalyticalEngine,
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
)

# Import parameters
from quantark.asset.equity.param import PDEParams

# Import market data
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

# Import enums
from quantark.util.enum import (
    OptionType,
    BarrierType,
    DoubleBarrierType,
    BarrierDirection,
    TouchType,
    ObservationType,
)
from quantark.util.exceptions import ValidationError, PricingError


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def pricing_env():
    """Standard pricing environment for tests."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime.now(),
    )


@pytest.fixture
def bs_engine():
    """Black-Scholes analytical engine."""
    return BlackScholesEngine()


@pytest.fixture
def pde_params():
    """Standard PDE parameters for tests."""
    return PDEParams(grid_size=200, time_steps=100)


# ============================================================================
# Test PDEParams
# ============================================================================


class TestPDEParams:
    """Tests for PDEParams configuration class."""

    def test_default_values(self):
        """Test default parameter values."""
        params = PDEParams()

        assert params.grid_size == 400
        assert params.time_steps == 200
        assert params.accuracy == "standard"
        assert params.grid is None
        assert params.bus_days_in_year == 252
        assert params.event_steps_per_day == 4
        assert params.max_time_steps == 5000
        assert params.max_grid_size == 2000
        assert params.rannacher_at_events == True
        assert params.theta == 0.5
        assert params.use_rannacher == True
        assert params.rannacher_steps == 1

    def test_custom_values(self):
        """Test custom parameter values."""
        params = PDEParams(
            grid_size=100,
            time_steps=50,
            theta=1.0,
            use_rannacher=False,
        )

        assert params.grid_size == 100
        assert params.time_steps == 50
        assert params.theta == 1.0
        assert params.use_rannacher == False

    def test_invalid_grid_size(self):
        """Test that invalid grid size raises error."""
        with pytest.raises(ValidationError):
            PDEParams(grid_size=0)

        with pytest.raises(ValidationError):
            PDEParams(grid_size=-10)

    def test_invalid_theta(self):
        """Test that invalid theta raises error."""
        with pytest.raises(ValidationError):
            PDEParams(theta=-0.1)

        with pytest.raises(ValidationError):
            PDEParams(theta=1.5)

    # time_grid_type tests retired with the knob (0.4.0): the grid layer
    # has ONE time algorithm; accuracy/GridConfig validation is covered in
    # test/pde_grid/test_grid_config.py.


# ============================================================================
# TimeGrid/SpatialGrid unit tests retired with the legacy modules (0.4.0):
# successors live in test/pde_grid/test_time_builder.py and test_space_builder.py.


class TestEuropeanPDESolver:
    """Tests for European option PDE solver."""

    def test_price_vs_black_scholes(self, pricing_env, bs_engine, pde_params):
        """Test PDE price matches Black-Scholes analytical."""
        strike = 100.0
        maturity = 1.0

        call = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )

        pde_solver = EuropeanPDESolver(pde_params)

        bs_price = bs_engine.price(call, pricing_env)
        pde_price = pde_solver.price(call, pricing_env)

        # Should be within 0.1% of analytical
        assert abs(pde_price - bs_price) / bs_price < 0.001

    def test_call_put_parity(self, pricing_env, pde_params):
        """Test call-put parity holds."""
        strike = 100.0
        maturity = 1.0

        call = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )
        put = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.PUT, maturity=maturity
        )

        pde_solver = EuropeanPDESolver(pde_params)

        call_price = pde_solver.price(call, pricing_env)
        put_price = pde_solver.price(put, pricing_env)

        spot = pricing_env.spot
        r = pricing_env.get_rate(maturity)
        q = pricing_env.get_div_yield(maturity)

        # Call - Put = S*exp(-qT) - K*exp(-rT)
        forward = spot * np.exp(-q * maturity) - strike * np.exp(-r * maturity)
        parity_diff = call_price - put_price - forward

        assert abs(parity_diff) < 0.01

    def test_convergence(self, pricing_env, bs_engine):
        """Test convergence in grid resolution."""
        strike = 100.0
        maturity = 1.0

        call = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )

        analytical = bs_engine.price(call, pricing_env)

        # Test with higher resolution
        params = PDEParams(grid_size=400, time_steps=400)
        solver = EuropeanPDESolver(params)
        pde_price = solver.price(call, pricing_env)
        error = abs(pde_price - analytical)

        # Final error should be small (within 0.1%)
        assert error / analytical < 0.001

    def test_invalid_product_type(self, pricing_env, pde_params):
        """Test that non-European option raises error."""
        american = AmericanOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        pde_solver = EuropeanPDESolver(pde_params)

        with pytest.raises(PricingError):
            pde_solver.price(american, pricing_env)


# ============================================================================
# Test AmericanPDESolver
# ============================================================================


class TestAmericanPDESolver:
    """Tests for American option PDE solver."""

    def test_american_geq_european(self, pricing_env, pde_params):
        """Test American option price >= European option price."""
        strike = 100.0
        maturity = 1.0

        euro_put = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.PUT, maturity=maturity
        )
        amer_put = AmericanOption(
            strike=strike, option_type=OptionType.PUT, maturity=maturity
        )

        euro_solver = EuropeanPDESolver(pde_params)
        amer_solver = AmericanPDESolver(pde_params)

        euro_price = euro_solver.price(euro_put, pricing_env)
        amer_price = amer_solver.price(amer_put, pricing_env)

        assert amer_price >= euro_price - 1e-6  # Small tolerance

    def test_early_exercise_premium_positive(self, pricing_env, pde_params):
        """Test that early exercise premium is positive for ITM puts."""
        strike = 110.0  # ITM put
        maturity = 1.0

        euro_put = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.PUT, maturity=maturity
        )
        amer_put = AmericanOption(
            strike=strike, option_type=OptionType.PUT, maturity=maturity
        )

        euro_solver = EuropeanPDESolver(pde_params)
        amer_solver = AmericanPDESolver(pde_params)

        euro_price = euro_solver.price(euro_put, pricing_env)
        amer_price = amer_solver.price(amer_put, pricing_env)

        # For ITM put, early exercise premium should be positive
        assert amer_price > euro_price


# ============================================================================
# Test BarrierPDESolver
# ============================================================================


class TestBarrierPDESolver:
    """Tests for barrier option PDE solver."""

    def test_continuous_up_and_out_converges_to_analytical(self, pricing_env):
        """Regression: continuous U0O near spot should match analytical closely."""
        option = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=110.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.CONTINUOUS,
        )

        analytical = BarrierAnalyticalEngine().price(option, pricing_env)
        pde = BarrierPDESolver(PDEParams(grid_size=400, time_steps=1008)).price(
            option, pricing_env
        )

        assert pde == pytest.approx(analytical, rel=5e-3, abs=1e-4)

    def test_continuous_up_and_out_rebate_matches_analytical(self, pricing_env):
        """Regression: continuous U0O rebate must be transmitted via boundary conditions."""
        option = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=110.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=2.0,
            observation_type=ObservationType.CONTINUOUS,
            pay_at_hit=False,
        )

        analytical = BarrierAnalyticalEngine().price(option, pricing_env)
        pde = BarrierPDESolver(PDEParams(grid_size=400, time_steps=1008)).price(
            option, pricing_env
        )

        assert pde == pytest.approx(analytical, rel=1e-2, abs=2e-3)

    def test_knockout_leq_vanilla(self, pricing_env, pde_params):
        """Test knock-out option price <= vanilla option price."""
        strike = 100.0
        barrier = 120.0
        maturity = 0.5

        vanilla = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )
        knockout = BarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            barrier=barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=maturity,
        )

        euro_solver = EuropeanPDESolver(pde_params)
        barrier_solver = BarrierPDESolver(pde_params)

        vanilla_price = euro_solver.price(vanilla, pricing_env)
        ko_price = barrier_solver.price(knockout, pricing_env)

        assert ko_price <= vanilla_price + 1e-6

    def test_knockin_plus_knockout_equals_vanilla(self, pricing_env, pde_params):
        """Test knock-in + knock-out = vanilla."""
        strike = 100.0
        barrier = 120.0
        maturity = 0.5

        vanilla = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )
        knockin = BarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            barrier=barrier,
            barrier_type=BarrierType.UP_IN,
            maturity=maturity,
            rebate=0.0,
        )
        knockout = BarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            barrier=barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=maturity,
            rebate=0.0,
        )

        euro_solver = EuropeanPDESolver(pde_params)
        barrier_solver = BarrierPDESolver(pde_params)

        vanilla_price = euro_solver.price(vanilla, pricing_env)
        ki_price = barrier_solver.price(knockin, pricing_env)
        ko_price = barrier_solver.price(knockout, pricing_env)

        # KI + KO = Vanilla (for zero rebate)
        assert abs(ki_price + ko_price - vanilla_price) < 0.01

    def test_barrier_hit_at_inception(self, pde_params):
        """Test pricing when spot is at barrier."""
        # Spot at 100, barrier at 100 (already hit)
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        solver = BarrierPDESolver(pde_params)

        knockout_hit = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=100.0,  # At spot
            barrier_type=BarrierType.UP_OUT,
            maturity=0.5,
            rebate=5.0,
            pay_at_hit=True,
        )
        price_hit = solver.price(knockout_hit, pricing_env)
        assert np.isclose(price_hit, 5.0)

        knockout_expiry = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=100.0,  # At spot
            barrier_type=BarrierType.UP_OUT,
            maturity=0.5,
            rebate=5.0,
            pay_at_hit=False,
        )
        price_expiry = solver.price(knockout_expiry, pricing_env)
        expected = 5.0 * np.exp(-pricing_env.get_rate(0.5) * 0.5)
        assert np.isclose(price_expiry, expected)

    def test_discrete_schedule_settlement_time_discounting(self, pde_params):
        """Test discrete monitoring discounts to observation settlement time (not maturity)."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.01),
            rate_curve=FlatRateCurve(rate=0.20),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        option = BarrierOption(
            strike=1_000_000.0,  # Negligible vanilla value
            option_type=OptionType.CALL,
            barrier=100.1,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=5.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.5],
        )

        solver = BarrierPDESolver(pde_params)
        price = solver.price(option, pricing_env)
        r = pricing_env.get_rate(1.0)
        pv_hit = option.rebate * np.exp(-r * 0.5)
        pv_expiry = option.rebate * np.exp(-r * 1.0)
        assert abs(price - pv_hit) < abs(price - pv_expiry)

    def test_discrete_without_maturity_observation_does_not_enforce_terminal_barrier(
        self, pde_params
    ):
        """Test that discrete monitoring without maturity observation doesn't knock out at T."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.01),
            rate_curve=FlatRateCurve(rate=0.01),
            div_yield=ContinuousDividendYield(div_yield=0.20),
            valuation_date=datetime.now(),
        )

        option = BarrierOption(
            strike=90.0,
            option_type=OptionType.PUT,
            barrier=85.0,
            barrier_type=BarrierType.DOWN_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.DISCRETE,
            observation_dates=[0.5],  # No maturity observation
        )

        bs_engine = BlackScholesEngine()
        vanilla_put = EuropeanVanillaOption(
            strike=90.0,
            option_type=OptionType.PUT,
            maturity=1.0,
        )
        vanilla_price = bs_engine.price(vanilla_put, pricing_env)

        solver = BarrierPDESolver(pde_params)
        price = solver.price(option, pricing_env)
        assert price == pytest.approx(vanilla_price, rel=0.10, abs=0.50)


# ============================================================================
# Test DoubleBarrierPDESolver
# ============================================================================


class TestDoubleBarrierPDESolver:
    """Tests for double barrier option PDE solver."""

    def test_double_ko_leq_vanilla(self, pricing_env, pde_params):
        """Test double knock-out <= vanilla."""
        strike = 100.0
        maturity = 0.5

        vanilla = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )
        double_ko = DoubleBarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            upper_barrier=120.0,
            lower_barrier=80.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=maturity,
        )

        euro_solver = EuropeanPDESolver(pde_params)
        double_solver = DoubleBarrierPDESolver(pde_params)

        vanilla_price = euro_solver.price(vanilla, pricing_env)
        double_ko_price = double_solver.price(double_ko, pricing_env)

        assert double_ko_price <= vanilla_price + 1e-6

    def test_outside_corridor_returns_rebate(self, pde_params):
        """Test that being outside corridor returns rebate."""
        # Spot at 130, outside upper barrier of 120
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=130.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        double_ko = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=120.0,
            lower_barrier=80.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.5,
            rebate=10.0,
        )

        solver = DoubleBarrierPDESolver(pde_params)
        price = solver.price(double_ko, pricing_env)

        assert np.isclose(price, 10.0)


# ============================================================================
# Test OneTouchPDESolver
# ============================================================================


class TestOneTouchPDESolver:
    """Tests for one-touch option PDE solver."""

    def test_one_touch_positive_price(self, pricing_env, pde_params):
        """Test that one-touch has positive price when barrier is reachable."""
        barrier = 110.0  # Closer barrier is more likely to be hit
        rebate = 100.0
        maturity = 1.0

        one_touch = OneTouchOption(
            barrier=barrier,
            barrier_direction=BarrierDirection.UP,
            maturity=maturity,
            rebate=rebate,
            payment_at_hit=True,
            touch_type=TouchType.ONE_TOUCH,
        )

        solver = OneTouchPDESolver(pde_params)
        price = solver.price(one_touch, pricing_env)

        # Should have positive value (some probability of touching)
        assert price > 0
        assert price < rebate  # Cannot be worth more than rebate

    def test_no_touch_positive_price(self, pricing_env, pde_params):
        """Test that no-touch has positive price."""
        barrier = 150.0  # Far barrier unlikely to be hit
        rebate = 100.0
        maturity = 0.5

        no_touch = OneTouchOption(
            barrier=barrier,
            barrier_direction=BarrierDirection.UP,
            maturity=maturity,
            rebate=rebate,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
        )

        solver = OneTouchPDESolver(pde_params)
        price = solver.price(no_touch, pricing_env)

        r = pricing_env.get_rate(maturity)
        discounted_rebate = rebate * np.exp(-r * maturity)

        # Should be close to discounted rebate since barrier is far
        assert price > 0
        assert price < discounted_rebate

    def test_barrier_hit_at_inception(self, pde_params):
        """Test pricing when spot is at barrier."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        one_touch = OneTouchOption(
            barrier=100.0,  # At spot
            barrier_direction=BarrierDirection.DOWN,
            maturity=0.5,
            rebate=100.0,
            touch_type=TouchType.ONE_TOUCH,
        )

        solver = OneTouchPDESolver(pde_params)
        price = solver.price(one_touch, pricing_env)

        # Already touched, should get rebate
        assert np.isclose(price, 100.0)


# ============================================================================
# Test DoubleOneTouchPDESolver
# ============================================================================


class TestDoubleOneTouchPDESolver:
    """Tests for double one-touch option PDE solver."""

    def test_double_one_touch_positive_price(self, pricing_env, pde_params):
        """Test that double one-touch has positive price."""
        upper_barrier = 110.0
        lower_barrier = 90.0
        rebate = 100.0
        maturity = 1.0

        double_one_touch = DoubleOneTouchOption(
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            maturity=maturity,
            rebate=rebate,
            payment_at_hit=True,
            touch_type=TouchType.DOUBLE_ONE_TOUCH,
        )

        solver = DoubleOneTouchPDESolver(pde_params)
        price = solver.price(double_one_touch, pricing_env)

        # Should have positive value
        assert price > 0
        assert price < rebate

    def test_double_no_touch_positive_price(self, pricing_env, pde_params):
        """Test that double no-touch has positive price."""
        upper_barrier = 150.0
        lower_barrier = 50.0
        rebate = 100.0
        maturity = 0.5

        double_no_touch = DoubleOneTouchOption(
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            maturity=maturity,
            rebate=rebate,
            payment_at_hit=False,
            touch_type=TouchType.DOUBLE_NO_TOUCH,
        )

        solver = DoubleOneTouchPDESolver(pde_params)
        price = solver.price(double_no_touch, pricing_env)

        r = pricing_env.get_rate(maturity)
        discounted_rebate = rebate * np.exp(-r * maturity)

        # Should be close to discounted rebate for wide corridor
        assert price > 0
        assert price < discounted_rebate

    def test_outside_corridor_at_inception(self, pde_params):
        """Test pricing when spot is outside corridor."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=130.0),  # Above upper barrier
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        double_one_touch = DoubleOneTouchOption(
            upper_barrier=120.0,
            lower_barrier=80.0,
            maturity=0.5,
            rebate=100.0,
            touch_type=TouchType.DOUBLE_ONE_TOUCH,
        )

        solver = DoubleOneTouchPDESolver(pde_params)
        price = solver.price(double_one_touch, pricing_env)

        # Already touched, should get rebate
        assert np.isclose(price, 100.0)


# ============================================================================
# Greeks Consistency Tests
# ============================================================================


class TestGreeksConsistency:
    """Tests that calculate_greeks() returns consistent prices with price()."""

    def test_european_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that EuropeanPDESolver Greeks price matches price()."""
        call = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        solver = EuropeanPDESolver(pde_params)
        price = solver.price(call, pricing_env)
        greeks = solver.calculate_greeks(call, pricing_env)

        # Price from calculate_greeks() should match price()
        assert abs(greeks["price"] - price) < 1e-10

    def test_american_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that AmericanPDESolver Greeks price matches price()."""
        put = AmericanOption(
            strike=100.0, option_type=OptionType.PUT, maturity=1.0
        )

        solver = AmericanPDESolver(pde_params)
        price = solver.price(put, pricing_env)
        greeks = solver.calculate_greeks(put, pricing_env)

        assert abs(greeks["price"] - price) < 1e-10

    def test_barrier_knockout_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that BarrierPDESolver Greeks price matches price() for knock-out."""
        knockout = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=0.5,
            rebate=0.0,
        )

        solver = BarrierPDESolver(pde_params)
        price = solver.price(knockout, pricing_env)
        greeks = solver.calculate_greeks(knockout, pricing_env)

        assert abs(greeks["price"] - price) < 1e-10

    def test_barrier_knockin_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that BarrierPDESolver Greeks price matches price() for knock-in."""
        knockin = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_IN,
            maturity=0.5,
            rebate=0.0,
        )

        solver = BarrierPDESolver(pde_params)
        price = solver.price(knockin, pricing_env)
        greeks = solver.calculate_greeks(knockin, pricing_env)

        # Price should match (both use knock-in decomposition)
        assert abs(greeks["price"] - price) < 1e-10

    def test_barrier_knockin_greeks_decomposition(self, pricing_env, pde_params):
        """Test that knock-in Greeks = vanilla Greeks - knock-out Greeks."""
        strike = 100.0
        barrier = 120.0
        maturity = 0.5

        vanilla = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=maturity
        )
        knockin = BarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            barrier=barrier,
            barrier_type=BarrierType.UP_IN,
            maturity=maturity,
            rebate=0.0,
        )
        knockout = BarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            barrier=barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=maturity,
            rebate=0.0,
        )

        euro_solver = EuropeanPDESolver(pde_params)
        barrier_solver = BarrierPDESolver(pde_params)

        vanilla_greeks = euro_solver.calculate_greeks(vanilla, pricing_env)
        ki_greeks = barrier_solver.calculate_greeks(knockin, pricing_env)
        ko_greeks = barrier_solver.calculate_greeks(knockout, pricing_env)

        # KI + KO = Vanilla for each Greek
        assert abs(ki_greeks["price"] + ko_greeks["price"] - vanilla_greeks["price"]) < 0.01
        assert abs(ki_greeks["delta"] + ko_greeks["delta"] - vanilla_greeks["delta"]) < 0.01
        assert abs(ki_greeks["gamma"] + ko_greeks["gamma"] - vanilla_greeks["gamma"]) < 0.01

    def test_double_barrier_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that DoubleBarrierPDESolver Greeks price matches price()."""
        double_ko = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=120.0,
            lower_barrier=80.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=0.5,
        )

        solver = DoubleBarrierPDESolver(pde_params)
        price = solver.price(double_ko, pricing_env)
        greeks = solver.calculate_greeks(double_ko, pricing_env)

        assert abs(greeks["price"] - price) < 1e-10

    def test_one_touch_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that OneTouchPDESolver Greeks price matches price()."""
        no_touch = OneTouchOption(
            barrier=150.0,
            barrier_direction=BarrierDirection.UP,
            maturity=0.5,
            rebate=100.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
        )

        solver = OneTouchPDESolver(pde_params)
        price = solver.price(no_touch, pricing_env)
        greeks = solver.calculate_greeks(no_touch, pricing_env)

        assert abs(greeks["price"] - price) < 1e-10

    def test_double_one_touch_greeks_price_consistency(self, pricing_env, pde_params):
        """Test that DoubleOneTouchPDESolver Greeks price matches price()."""
        double_no_touch = DoubleOneTouchOption(
            upper_barrier=150.0,
            lower_barrier=50.0,
            maturity=0.5,
            rebate=100.0,
            payment_at_hit=False,
            touch_type=TouchType.DOUBLE_NO_TOUCH,
        )

        solver = DoubleOneTouchPDESolver(pde_params)
        price = solver.price(double_no_touch, pricing_env)
        greeks = solver.calculate_greeks(double_no_touch, pricing_env)

        assert abs(greeks["price"] - price) < 1e-10

    def test_expired_option_greeks(self, pde_params):
        """Test Greeks for near-expired option converge to intrinsic values.

        Note: Gamma can be numerically unstable for very small tau, so we
        only verify price and delta converge to intrinsic values.
        """
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=110.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        # Near-expired deep ITM call (should approach intrinsic value)
        call = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1e-4  # ~1 hour
        )

        solver = EuropeanPDESolver(pde_params)
        greeks = solver.calculate_greeks(call, pricing_env)

        # Price should be very close to intrinsic (110 - 100 = 10)
        # Small time value remains for short maturity
        assert abs(greeks["price"] - 10.0) < 0.5
        # Delta should be close to 1 for deep ITM near expiry
        assert greeks["delta"] > 0.9

    def test_expired_option_greeks_contract_multiplier(self, pde_params):
        """Test expired-option delta scales with contract multiplier."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=110.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        class DummyCall(BaseEquityProduct):
            def __init__(self, strike: float, contract_multiplier: float) -> None:
                self.strike = strike
                self.contract_multiplier = contract_multiplier

            def is_call(self) -> bool:
                return True

            def get_payoff(self, spot: float) -> float:
                return max(spot - self.strike, 0.0) * self.contract_multiplier

            def get_maturity(self, pricing_env=None) -> float:
                return 0.0

            def validate(self) -> None:
                return None

        product = DummyCall(strike=100.0, contract_multiplier=100.0)
        solver = EuropeanPDESolver(pde_params)
        greeks = solver.calculate_greeks(product, pricing_env)

        intrinsic = (pricing_env.spot - product.strike) * product.contract_multiplier
        assert greeks["price"] == pytest.approx(intrinsic)
        assert greeks["delta"] == pytest.approx(product.contract_multiplier)
        assert greeks["gamma"] == 0.0

    def test_barrier_hit_greeks(self, pde_params):
        """Test Greeks when barrier is already hit."""
        # Spot above barrier (already knocked out)
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=130.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        knockout = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=0.5,
            rebate=5.0,
            pay_at_hit=True,
        )

        solver = BarrierPDESolver(pde_params)
        greeks = solver.calculate_greeks(knockout, pricing_env)

        # Price should be rebate
        assert abs(greeks["price"] - 5.0) < 0.01
        # Delta and gamma should be 0 (fixed rebate)
        assert abs(greeks["delta"]) < 0.01
        assert abs(greeks["gamma"]) < 0.01


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for PDE engine."""

    def test_greeks_calculation(self, pricing_env, pde_params):
        """Test that Greeks calculation works."""
        call = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        solver = EuropeanPDESolver(pde_params)
        greeks = solver.calculate_greeks(call, pricing_env)

        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks

        # Delta should be in [0, 1] for call
        assert 0 <= greeks["delta"] <= 1

        # Gamma should be positive
        assert greeks["gamma"] > 0

    def test_different_maturities(self, pricing_env, pde_params):
        """Test pricing across different maturities."""
        strike = 100.0

        solver = EuropeanPDESolver(pde_params)

        prices = []
        for maturity in [0.1, 0.5, 1.0, 2.0]:
            call = EuropeanVanillaOption(
                strike=strike, option_type=OptionType.CALL, maturity=maturity
            )
            price = solver.price(call, pricing_env)
            prices.append(price)

        # Longer maturity should mean higher price (for calls with positive rate-div)
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

    def test_expired_option(self, pde_params):
        """Test pricing of expired option."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime.now(),
        )

        # Option with zero maturity
        call = EuropeanVanillaOption(
            strike=90.0, option_type=OptionType.CALL, maturity=0.0001
        )

        solver = EuropeanPDESolver(pde_params)
        price = solver.price(call, pricing_env)

        # Should be approximately intrinsic value
        assert abs(price - 10.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
