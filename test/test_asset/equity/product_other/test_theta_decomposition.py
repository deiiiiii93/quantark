"""
Tests for decomposed theta components (r_theta, q_theta, convexity_theta).

These tests verify the theta decomposition functionality in GreeksCalculator:
- Analytical theta components for European vanilla options
- Fast estimation of theta components from existing Greeks
- Repricing-based numerical theta decomposition
"""

import pytest
from datetime import datetime

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.param import EngineParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


@pytest.fixture
def pricing_env():
    """Standard pricing environment for testing."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


@pytest.fixture
def pricing_env_zero_div():
    """Pricing environment with zero dividend yield."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )


@pytest.fixture
def pricing_env_zero_rate():
    """Pricing environment with zero interest rate."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.0),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


class TestAnalyticalThetaDecomposition:
    """Test analytical theta decomposition for European vanilla options."""

    def test_theta_components_sum_to_theta_call(self, pricing_env):
        """Test that components sum to total theta for call option."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(call, pricing_env)

        total = (
            greeks["convexity_theta"]
            + greeks["r_theta"]
            + greeks["q_theta"]
        )
        assert total == pytest.approx(greeks["theta"], rel=1e-10)

    def test_theta_components_sum_to_theta_put(self, pricing_env):
        """Test that components sum to total theta for put option."""
        put = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.PUT,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(put, pricing_env)

        total = (
            greeks["convexity_theta"]
            + greeks["r_theta"]
            + greeks["q_theta"]
        )
        assert total == pytest.approx(greeks["theta"], rel=1e-10)

    def test_convexity_theta_always_negative(self, pricing_env):
        """Convexity theta should be negative (gamma decay) for both calls and puts."""
        calc = GreeksCalculator()

        for opt_type in [OptionType.CALL, OptionType.PUT]:
            option = EuropeanVanillaOption(
                strike=100.0,
                option_type=opt_type,
                maturity=1.0,
            )
            greeks = calc.calculate_analytical_greeks(option, pricing_env)
            assert greeks["convexity_theta"] < 0, f"convexity_theta should be negative for {opt_type}"

    def test_r_theta_sign_convention(self, pricing_env):
        """r_theta should be negative for calls, positive for puts."""
        calc = GreeksCalculator()

        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        call_greeks = calc.calculate_analytical_greeks(call, pricing_env)
        assert call_greeks["r_theta"] < 0, "r_theta should be negative for calls"

        put = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.PUT,
            maturity=1.0,
        )
        put_greeks = calc.calculate_analytical_greeks(put, pricing_env)
        assert put_greeks["r_theta"] > 0, "r_theta should be positive for puts"

    def test_q_theta_sign_convention(self, pricing_env):
        """q_theta should be positive for calls, negative for puts."""
        calc = GreeksCalculator()

        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        call_greeks = calc.calculate_analytical_greeks(call, pricing_env)
        assert call_greeks["q_theta"] > 0, "q_theta should be positive for calls"

        put = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.PUT,
            maturity=1.0,
        )
        put_greeks = calc.calculate_analytical_greeks(put, pricing_env)
        assert put_greeks["q_theta"] < 0, "q_theta should be negative for puts"

    def test_zero_dividend_q_theta_is_zero(self, pricing_env_zero_div):
        """q_theta should be zero when dividend yield is zero."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(call, pricing_env_zero_div)

        assert greeks["q_theta"] == pytest.approx(0.0, abs=1e-15)

    def test_zero_rate_r_theta_is_zero(self, pricing_env_zero_rate):
        """r_theta should be zero when interest rate is zero."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(call, pricing_env_zero_rate)

        assert greeks["r_theta"] == pytest.approx(0.0, abs=1e-15)

    def test_all_components_zero_at_expiry(self, pricing_env):
        """All theta components should be zero at expiry."""
        option = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1e-11,  # Essentially at expiry
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(option, pricing_env)

        assert greeks["theta"] == 0.0
        assert greeks["convexity_theta"] == 0.0
        assert greeks["r_theta"] == 0.0
        assert greeks["q_theta"] == 0.0

    def test_components_scale_with_multiplier(self, pricing_env):
        """Components should scale with contract multiplier."""
        option_1x = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
            contract_multiplier=1.0,
        )
        option_100x = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
            contract_multiplier=100.0,
        )

        calc = GreeksCalculator()
        greeks_1x = calc.calculate_analytical_greeks(option_1x, pricing_env)
        greeks_100x = calc.calculate_analytical_greeks(option_100x, pricing_env)

        assert greeks_100x["convexity_theta"] == pytest.approx(
            100 * greeks_1x["convexity_theta"], rel=1e-10
        )
        assert greeks_100x["r_theta"] == pytest.approx(
            100 * greeks_1x["r_theta"], rel=1e-10
        )
        assert greeks_100x["q_theta"] == pytest.approx(
            100 * greeks_1x["q_theta"], rel=1e-10
        )


class TestNumericalThetaDecomposition:
    """Test numerical theta decomposition using fast estimation."""

    def test_numerical_components_in_output(self, pricing_env):
        """Numerical Greeks should include theta components."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        engine = BlackScholesEngine()
        greeks = calc.calculate_numerical_greeks(call, pricing_env, engine)

        assert "convexity_theta" in greeks
        assert "r_theta" in greeks
        assert "q_theta" in greeks

    def test_fast_estimation_approximates_analytical(self, pricing_env):
        """Fast estimation should approximate analytical for European vanilla."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        engine = BlackScholesEngine()

        analytical = calc.calculate_analytical_greeks(call, pricing_env)
        numerical = calc.calculate_numerical_greeks(call, pricing_env, engine)

        # Fast estimation is an approximation using r_theta ≈ -r/T * rho
        # Allow 50% relative error since this is a first-order approximation
        assert numerical["convexity_theta"] == pytest.approx(
            analytical["convexity_theta"], rel=0.5
        )

    def test_numerical_components_sum_approximately(self, pricing_env):
        """Numerical components should sum approximately to total theta."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        engine = BlackScholesEngine()

        greeks = calc.calculate_numerical_greeks(call, pricing_env, engine)

        total = (
            greeks["convexity_theta"]
            + greeks["r_theta"]
            + greeks["q_theta"]
        )
        # Allow 1% relative error for numerical approximation
        assert total == pytest.approx(greeks["theta"], rel=0.01)


class TestFastEstimationMethod:
    """Test the estimate_theta_components() method directly."""

    def test_estimate_theta_components_basic(self, pricing_env):
        """Test basic estimation of theta components."""
        calc = GreeksCalculator()

        # Use some reasonable values
        theta = -0.05
        rho = 10.0
        dividend_rho = -5.0
        r = 0.05
        q = 0.02
        T = 1.0

        components = calc.estimate_theta_components(
            theta=theta,
            rho=rho,
            dividend_rho=dividend_rho,
            r=r,
            q=q,
            T=T,
        )

        assert "convexity_theta" in components
        assert "r_theta" in components
        assert "q_theta" in components

        # r_theta = -r / T * (rho * 100) / 365 = -0.05 / 1.0 * 1000 / 365 = -50 / 365
        assert components["r_theta"] == pytest.approx(-50 / 365, rel=1e-10)

        # q_theta = -q / T * (dividend_rho * 100) / 365 = -0.02 / 1.0 * (-500) / 365 = 10 / 365
        assert components["q_theta"] == pytest.approx(10 / 365, rel=1e-10)

        # convexity_theta = theta - r_theta - q_theta
        expected_convexity = theta - (-50 / 365) - (10 / 365)
        assert components["convexity_theta"] == pytest.approx(expected_convexity, rel=1e-10)

    def test_estimate_at_near_expiry(self):
        """Test estimation near expiry returns zeros."""
        calc = GreeksCalculator()

        components = calc.estimate_theta_components(
            theta=-0.05,
            rho=10.0,
            dividend_rho=-5.0,
            r=0.05,
            q=0.02,
            T=1e-11,  # Near expiry
        )

        assert components["convexity_theta"] == 0.0
        assert components["r_theta"] == 0.0
        assert components["q_theta"] == 0.0


class TestRepricingThetaDecomposition:
    """Test the _calculate_numerical_theta_components() repricing method."""

    def test_repricing_method_basic(self, pricing_env):
        """Test repricing-based theta decomposition."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        engine = BlackScholesEngine()

        components = calc._calculate_numerical_theta_components(
            call, pricing_env, engine
        )

        assert "convexity_theta" in components
        assert "r_theta" in components
        assert "q_theta" in components

    def test_repricing_approximates_analytical(self, pricing_env):
        """Repricing method should approximate analytical results."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        engine = BlackScholesEngine()

        analytical = calc.calculate_analytical_greeks(call, pricing_env)
        repricing = calc._calculate_numerical_theta_components(
            call, pricing_env, engine
        )

        # Repricing should be close to analytical (within 15%)
        # Some deviation is expected due to numerical differentiation
        assert repricing["convexity_theta"] == pytest.approx(
            analytical["convexity_theta"], rel=0.15
        )
        assert repricing["r_theta"] == pytest.approx(
            analytical["r_theta"], rel=0.15
        )
        assert repricing["q_theta"] == pytest.approx(
            analytical["q_theta"], rel=0.15
        )

    def test_repricing_components_sum_to_theta(self, pricing_env):
        """Repricing components should sum to total numerical theta."""
        call = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        engine = BlackScholesEngine()

        greeks = calc.calculate_numerical_greeks(call, pricing_env, engine)
        repricing = calc._calculate_numerical_theta_components(
            call, pricing_env, engine
        )

        total_repricing = (
            repricing["convexity_theta"]
            + repricing["r_theta"]
            + repricing["q_theta"]
        )

        # Should sum to theta (within 10% due to numerical differentiation)
        # The repricing method uses zeroed r/q environments which can diverge slightly
        assert total_repricing == pytest.approx(greeks["theta"], rel=0.10)


class TestThetaDecompositionForDifferentStrikes:
    """Test theta decomposition across different moneyness levels."""

    @pytest.mark.parametrize("strike,label", [
        (80.0, "ITM"),
        (100.0, "ATM"),
        (120.0, "OTM"),
    ])
    def test_components_sum_for_different_strikes(self, pricing_env, strike, label):
        """Components should sum to theta for different strike levels."""
        call = EuropeanVanillaOption(
            strike=strike,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(call, pricing_env)

        total = (
            greeks["convexity_theta"]
            + greeks["r_theta"]
            + greeks["q_theta"]
        )
        assert total == pytest.approx(greeks["theta"], rel=1e-10), f"Failed for {label}"

    @pytest.mark.parametrize("strike,label", [
        (80.0, "ITM"),
        (100.0, "ATM"),
        (120.0, "OTM"),
    ])
    def test_convexity_negative_for_different_strikes(self, pricing_env, strike, label):
        """Convexity theta should be negative regardless of moneyness."""
        call = EuropeanVanillaOption(
            strike=strike,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        calc = GreeksCalculator()
        greeks = calc.calculate_analytical_greeks(call, pricing_env)

        assert greeks["convexity_theta"] < 0, f"Failed for {label}"
