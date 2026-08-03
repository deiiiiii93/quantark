"""
Integration tests for Greeks calculation with BumpConfig.
"""

import math
import warnings

import pytest
from datetime import datetime
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.param import EngineParams, BumpConfig
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError


@pytest.fixture
def pricing_env():
    """Create a standard pricing environment for testing."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


@pytest.fixture
def call_option():
    """Create a call option for testing."""
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0
    )


@pytest.fixture
def put_option():
    """Create a put option for testing."""
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0
    )


@pytest.fixture
def bs_engine():
    """Create Black-Scholes engine for testing."""
    return BlackScholesEngine()


class TestEngineParamsBackwardCompatibility:
    """Test backward compatibility for bump_size."""

    def test_legacy_bump_size_creates_bump_config(self):
        """Test that legacy bump_size creates valid BumpConfig."""
        params = EngineParams(bump_size=0.001)
        assert params.bump_config is not None
        assert params.bump_config.spot_bump == 0.001

    def test_explicit_bump_config_overrides_default(self):
        """Test that explicit bump_config is used."""
        custom_config = BumpConfig(spot_bump=0.005)
        params = EngineParams(bump_size=0.001, bump_config=custom_config)
        assert params.get_effective_bump_config().spot_bump == 0.005

    def test_greeks_calculator_uses_bump_config(self):
        """Test that GreeksCalculator uses bump_config values."""
        config = BumpConfig(
            spot_bump=0.001,  # 0.1% instead of 1%
            vol_bump=0.005,   # 0.5 vol points
        )
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        assert calc._bump_config.spot_bump == 0.001
        assert calc._bump_config.vol_bump == 0.005

    def test_non_grid_engine_bump_context_falls_back_to_same_engine(
        self, call_option, pricing_env, bs_engine
    ):
        """Test that engines without numerical domains use the explicit fallback."""
        assert bs_engine.create_bump_context(call_option, pricing_env) is bs_engine


class TestNumericalGreeksWithBumpConfig:
    """Test numerical Greeks with custom bump configurations."""

    def test_greeks_with_default_bump_config(
        self, call_option, pricing_env, bs_engine
    ):
        """Test Greeks calculation with default bump config."""
        calc = GreeksCalculator()
        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)

        # Check all expected Greeks are present
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks
        assert "dividend_rho" in greeks

    def test_delta_with_custom_spot_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test delta calculation with custom spot bump."""
        config = BumpConfig(spot_bump=0.001)  # Smaller bump
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)
        # Delta should be in reasonable range for ATM call
        assert 0 < greeks["delta"] < 1

    def test_gamma_with_custom_spot_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test gamma calculation with custom spot bump."""
        config = BumpConfig(spot_bump=0.005)  # 0.5% bump
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)
        # Gamma should be positive
        assert greeks["gamma"] > 0

    def test_vega_with_custom_vol_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test vega with custom vol bump."""
        config = BumpConfig(vol_bump=0.005)  # 0.5 vol points
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)
        # Vega should be positive for call
        assert greeks["vega"] > 0

    def test_theta_with_custom_time_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test theta with 7-day bump instead of 1-day."""
        config = BumpConfig(time_bump_days=7)
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)
        # Theta should be negative for call (time decay)
        assert greeks["theta"] < 0

    def test_rho_with_custom_rate_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test rho with custom rate bump."""
        config = BumpConfig(rate_bump=0.0001)  # 1bp
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)
        # Rho should be positive for call
        assert greeks["rho"] > 0

    def test_dividend_rho_call_negative(
        self, call_option, pricing_env, bs_engine
    ):
        """Test that dividend_rho is negative for calls."""
        config = BumpConfig(div_bump=0.0001)
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)
        assert "dividend_rho" in greeks
        # Higher dividend yield -> lower call price
        assert greeks["dividend_rho"] < 0

    def test_dividend_rho_put_positive(
        self, put_option, pricing_env, bs_engine
    ):
        """Test that dividend_rho is positive for puts."""
        config = BumpConfig(div_bump=0.0001)
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(put_option, pricing_env, bs_engine)
        assert "dividend_rho" in greeks
        # Higher dividend yield -> higher put price
        assert greeks["dividend_rho"] > 0

    def test_all_greeks_with_full_custom_config(
        self, call_option, pricing_env, bs_engine
    ):
        """Test all Greeks with fully custom BumpConfig."""
        config = BumpConfig(
            spot_bump=0.005,   # 0.5%
            vol_bump=0.02,     # 2 vol points
            time_bump_days=7,  # 7-day theta
            rate_bump=0.001,   # 10bp
            div_bump=0.001,    # 10bp
        )
        params = EngineParams(bump_config=config)
        calc = GreeksCalculator(params=params)

        greeks = calc.calculate_numerical_greeks(call_option, pricing_env, bs_engine)

        # Verify all Greeks are calculated
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks
        assert "dividend_rho" in greeks

        # Verify signs for call option
        assert greeks["delta"] > 0
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0
        assert greeks["theta"] < 0  # Time decay
        assert greeks["rho"] > 0
        assert greeks["dividend_rho"] < 0  # Higher div -> lower call price


class TestIndividualGreekMethodOverrides:
    """Test individual Greek calculation methods with override parameters."""

    def test_delta_override_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test delta with explicit bump parameter override."""
        calc = GreeksCalculator()
        # Use 0.5% bump instead of default 1%
        delta = calc.calculate_numerical_delta(
            call_option, pricing_env, bs_engine, bump=0.005
        )
        assert 0 < delta < 1

    def test_gamma_override_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test gamma with explicit bump parameter override."""
        calc = GreeksCalculator()
        gamma = calc.calculate_numerical_gamma(
            call_option, pricing_env, bs_engine, bump=0.005
        )
        assert gamma > 0

    def test_vega_override_vol_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test vega with explicit vol_bump parameter override."""
        calc = GreeksCalculator()
        vega = calc.calculate_numerical_vega(
            call_option, pricing_env, bs_engine, vol_bump=0.02
        )
        assert vega > 0

    def test_theta_override_time_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test theta with explicit time_bump_days parameter override."""
        calc = GreeksCalculator()
        theta = calc.calculate_numerical_theta(
            call_option, pricing_env, bs_engine, time_bump_days=7
        )
        assert theta < 0

    def test_rho_override_rate_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test rho with explicit rate_bump parameter override."""
        calc = GreeksCalculator()
        rho = calc.calculate_numerical_rho(
            call_option, pricing_env, bs_engine, rate_bump=0.001
        )
        assert rho > 0

    def test_dividend_rho_override_div_bump(
        self, call_option, pricing_env, bs_engine
    ):
        """Test dividend_rho with explicit div_bump parameter override."""
        calc = GreeksCalculator()
        div_rho = calc.calculate_numerical_dividend_rho(
            call_option, pricing_env, bs_engine, div_bump=0.001
        )
        assert div_rho < 0  # Negative for call


class TestEdgeCases:
    """Test edge cases for bump configurations."""

    def test_zero_dividend_yield(self, call_option, pricing_env, bs_engine):
        """Test dividend_rho with zero dividend yield."""
        env_no_div = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),
            valuation_date=datetime(2024, 1, 1),
        )

        calc = GreeksCalculator()
        greeks = calc.calculate_numerical_greeks(call_option, env_no_div, bs_engine)
        assert "dividend_rho" in greeks

    def test_near_expiry_theta(self, pricing_env, bs_engine):
        """Test theta calculation when near expiry."""
        # Option with 0.5 days to maturity
        near_expiry_option = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=0.5 / 365
        )

        calc = GreeksCalculator()
        greeks = calc.calculate_numerical_greeks(
            near_expiry_option, pricing_env, bs_engine
        )
        # Theta should handle case where time_bump exceeds maturity
        assert "theta" in greeks

    def test_low_volatility_vega(self, call_option, bs_engine):
        """Test vega with very low volatility."""
        env_low_vol = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.01),  # 1% vol
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=datetime(2024, 1, 1),
        )

        calc = GreeksCalculator()
        greeks = calc.calculate_numerical_greeks(call_option, env_low_vol, bs_engine)
        assert "vega" in greeks
        assert greeks["vega"] > 0


class _MarketRecordingEngine(BaseEngine):
    """Smooth analytic stub that records every priced market state."""

    def __init__(self, params=None):
        super().__init__(params)
        self.spots = []
        self.vols = []
        self.rates = []

    def price(self, product, pricing_env):
        s = float(pricing_env.spot)
        self.spots.append(s)
        self.vols.append(
            float(getattr(pricing_env.vol_surface, "volatility", float("nan")))
        )
        self.rates.append(float(pricing_env.get_rate(1.0)))
        return (s / 100.0) ** 3


class _BumpContextRecordingEngine(_MarketRecordingEngine):
    """Engine whose finite-difference prices must run on a distinct context."""

    def __init__(self, params=None):
        super().__init__(params)
        self.bump_context_calls = 0
        self.bump_context = None

    def create_bump_context(self, product, pricing_env):
        self.bump_context_calls += 1
        self.bump_context = _MarketRecordingEngine(self.params)
        return self.bump_context


def _contains(values, target):
    return any(math.isclose(v, target, rel_tol=1e-12) for v in values)


class TestBumpSizeShimRetirement:
    """The deprecated ``EngineParams.bump_size`` shim must not override
    ``BumpConfig``'s documented defaults.

    Root cause (auto-grid round-2 review, F4): ``bump_size=1e-4`` silently
    fed ``BumpConfig(spot_bump=1e-4)``, making the documented 1% spot bump
    unreachable and turning default BUMP-mode gamma into kink noise on any
    piecewise-linear PDE readout.
    """

    def test_default_engine_params_uses_documented_spot_bump(self):
        assert EngineParams().bump_config.spot_bump == pytest.approx(0.01)

    def test_default_mc_and_pde_params_inherit_documented_spot_bump(self):
        from quantark.asset.equity.param import MCParams, PDEParams

        assert MCParams().bump_config.spot_bump == pytest.approx(0.01)
        assert PDEParams().bump_config.spot_bump == pytest.approx(0.01)

    def test_default_construction_emits_no_deprecation_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            EngineParams()

    def test_explicit_bump_size_warns_and_is_honored(self):
        with pytest.warns(DeprecationWarning, match="bump_size"):
            params = EngineParams(bump_size=0.001)
        assert params.bump_config.spot_bump == pytest.approx(0.001)

    def test_invalid_explicit_bump_size_still_rejected(self):
        with pytest.raises(ValidationError):
            EngineParams(bump_size=-1.0)
        with pytest.raises(ValidationError):
            EngineParams(bump_size=0.5)

    def test_base_engine_bump_greeks_use_bump_config(
        self, call_option, pricing_env
    ):
        engine = _MarketRecordingEngine()
        engine.calculate_greeks(call_option, pricing_env)
        assert _contains(engine.spots, 100.0 * 1.01)
        assert _contains(engine.spots, 100.0 * 0.99)

    def test_base_engine_bump_greeks_use_the_resolved_bump_context(
        self, call_option, pricing_env
    ):
        engine = _BumpContextRecordingEngine()

        greeks = engine.calculate_greeks(call_option, pricing_env)

        assert engine.bump_context_calls == 1
        assert engine.spots == []
        assert engine.bump_context is not None
        assert engine.bump_context.spots == pytest.approx([100.0, 101.0, 99.0])
        assert greeks["price"] == pytest.approx(1.0)
        assert greeks["delta"] == pytest.approx(0.03, rel=1e-4)

    def test_numerical_delta_default_bump_comes_from_bump_config(
        self, call_option, pricing_env
    ):
        params = EngineParams(bump_config=BumpConfig(spot_bump=0.004))
        calc = GreeksCalculator(params=params)
        engine = _MarketRecordingEngine(params)
        calc.calculate_numerical_delta(call_option, pricing_env, engine)
        assert _contains(engine.spots, 100.0 * 1.004)
        assert _contains(engine.spots, 100.0 * 0.996)

    def test_numerical_gamma_default_bump_comes_from_bump_config(
        self, call_option, pricing_env
    ):
        params = EngineParams(bump_config=BumpConfig(spot_bump=0.004))
        calc = GreeksCalculator(params=params)
        engine = _MarketRecordingEngine(params)
        calc.calculate_numerical_gamma(call_option, pricing_env, engine)
        assert _contains(engine.spots, 100.0 * 1.004)
        assert _contains(engine.spots, 100.0 * 0.996)

    def test_get_trade_greeks_uses_per_factor_bumps(
        self, call_option, pricing_env
    ):
        from quantark.portfolio.equity.position import EquityPosition

        config = BumpConfig(spot_bump=0.004, vol_bump=0.03, rate_bump=0.0007)
        engine = _MarketRecordingEngine(EngineParams(bump_config=config))
        pos = EquityPosition(
            product=call_option,
            quantity=1.0,
            entry_price=0.0,
            underlying="TEST",
            engine=engine,
            entry_timestamp=datetime(2026, 1, 1),
        )
        pos.get_trade_greeks(pricing_env, GreeksCalculator())
        assert _contains(engine.spots, 100.0 * 1.004)
        assert _contains(engine.spots, 100.0 * 0.996)
        assert _contains(engine.vols, 0.20 + 0.03)
        assert _contains(engine.vols, 0.20 - 0.03)
        assert _contains(engine.rates, 0.05 + 0.0007)
        assert _contains(engine.rates, 0.05 - 0.0007)
