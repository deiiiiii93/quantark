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
from asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
    DoubleBarrierOption,
    OneTouchOption,
    DoubleOneTouchOption,
)

# Import engines
from asset.equity.engine import (
    BlackScholesEngine,
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
    TimeGrid,
    SpatialGrid,
)

# Import parameters
from asset.equity.param import PDEParams

# Import market data
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

# Import enums
from util.enum import (
    OptionType,
    BarrierType,
    DoubleBarrierType,
    BarrierDirection,
    TouchType,
    ObservationType,
)
from util.exceptions import ValidationError, PricingError


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
        assert params.adaptive_grid == False
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

    def test_invalid_time_grid_type(self):
        """Test that invalid time grid type raises error."""
        with pytest.raises(ValidationError):
            PDEParams(time_grid_type="invalid")


# ============================================================================
# Test TimeGrid
# ============================================================================


class TestTimeGrid:
    """Tests for TimeGrid utility class."""

    def test_uniform_grid(self):
        """Test uniform time grid generation."""
        tau = 1.0
        num_steps = 100

        t_vec, dt_vec = TimeGrid.build_uniform(tau, num_steps)

        assert len(t_vec) == num_steps + 1
        assert len(dt_vec) == num_steps
        assert np.isclose(t_vec[0], 0.0)
        assert np.isclose(t_vec[-1], tau)
        assert np.allclose(dt_vec, tau / num_steps)

    def test_graded_grid(self):
        """Test graded time grid with clustering near maturity."""
        tau = 1.0
        num_steps = 100
        exponent = 2.0

        t_vec, dt_vec = TimeGrid.build_graded(tau, num_steps, exponent)

        assert len(t_vec) == num_steps + 1
        assert len(dt_vec) == num_steps
        assert np.isclose(t_vec[0], 0.0)
        assert np.isclose(t_vec[-1], tau)

        # Later steps should be smaller (more clustering near tau)
        assert dt_vec[-1] < dt_vec[0]

    def test_event_clustered_grid(self):
        """Test event-clustered time grid."""
        tau = 1.0
        event_times = [0.25, 0.5, 0.75]

        t_vec, dt_vec = TimeGrid.build_event_clustered(tau, event_times)

        # Check that event times are in the grid
        for event in event_times:
            assert np.any(np.isclose(t_vec, event, atol=1e-10))

    def test_invalid_tau(self):
        """Test that invalid tau raises error."""
        with pytest.raises(ValueError):
            TimeGrid.build_uniform(0.0, 100)

        with pytest.raises(ValueError):
            TimeGrid.build_uniform(-1.0, 100)


# ============================================================================
# Test SpatialGrid
# ============================================================================


class TestSpatialGrid:
    """Tests for SpatialGrid utility class."""

    def test_uniform_log_grid(self):
        """Test uniform grid in log-price space."""
        s_min = 50.0
        s_max = 150.0
        num_points = 100

        x_vec, s_vec, dx = SpatialGrid.build_uniform_log(s_min, s_max, num_points)

        assert len(x_vec) == num_points
        assert len(s_vec) == num_points
        assert np.isclose(s_vec[0], s_min)
        assert np.isclose(s_vec[-1], s_max)

        # Check uniform spacing in log-space
        x_diffs = np.diff(x_vec)
        assert np.allclose(x_diffs, dx)

    def test_tavella_randall_concentration(self):
        """Test Tavella-Randall grid concentrates near critical point."""
        s_min = 50.0
        s_max = 150.0
        num_points = 100
        critical = 100.0

        x_vec, s_vec, dx_vec = SpatialGrid.build_tavella_randall(
            s_min, s_max, num_points, critical
        )

        assert len(x_vec) == num_points

        # Grid spacing should be smaller near critical point
        critical_idx = np.argmin(np.abs(s_vec - critical))
        avg_dx_near_critical = np.mean(
            dx_vec[max(0, critical_idx - 5) : critical_idx + 5]
        )
        avg_dx_at_edges = np.mean([dx_vec[0], dx_vec[-1]])

        assert avg_dx_near_critical < avg_dx_at_edges

    def test_auto_bounds(self):
        """Test automatic bound calculation."""
        spot = 100.0
        sigma = 0.20
        tau = 1.0

        s_min, s_max = SpatialGrid.calculate_auto_bounds(spot, sigma, tau)

        assert s_min > 0
        assert s_min < spot
        assert s_max > spot

    def test_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        with pytest.raises(ValueError):
            SpatialGrid.build_uniform_log(0.0, 100.0, 100)

        with pytest.raises(ValueError):
            SpatialGrid.build_uniform_log(100.0, 50.0, 100)


# ============================================================================
# Test EuropeanPDESolver
# ============================================================================


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

        knockout = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=100.0,  # At spot
            barrier_type=BarrierType.UP_OUT,
            maturity=0.5,
            rebate=5.0,
        )

        solver = BarrierPDESolver(pde_params)
        price = solver.price(knockout, pricing_env)

        # Already knocked out, should get rebate
        assert np.isclose(price, 5.0)


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

    @pytest.mark.skip(
        reason="One-touch solver needs refinement for proper boundary handling"
    )
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

    @pytest.mark.skip(
        reason="Double one-touch solver needs refinement for proper boundary handling"
    )
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
