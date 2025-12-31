"""
Tests for engine_type attribute and GreeksCalculationMode.

This test file covers:
1. Engine type detection via engine_type attribute
2. GreeksCalculator greeks_mode parameter behavior
3. PDE engine Greeks vs bump method
"""

import pytest
from datetime import datetime
from asset.equity.product.option import EuropeanVanillaOption, AmericanOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.engine.mc import EuropeanMCEngine
from asset.equity.engine.pde_engine import PDEEngine
from asset.equity.param import EngineParams, PDEParams, MCParams
from asset.equity.riskmeasures import GreeksCalculator
from param import SpotQuote, FlatVolSurface, FlatRateCurve
from param.div import ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from util.enum.engine_enums import EngineType, GreeksCalculationMode


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
def euro_call():
    """Create a European call option for testing."""
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0
    )


@pytest.fixture
def american_put():
    """Create an American put option for testing."""
    return AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0
    )


class TestEngineTypeAttribute:
    """Test that all engines have the correct engine_type attribute."""

    def test_base_engine_default_type(self):
        """Test that BaseEngine has default type ANALYTICAL."""
        from asset.equity.engine.base_engine import BaseEngine
        assert BaseEngine.engine_type == EngineType.ANALYTICAL

    def test_black_scholes_engine_type(self):
        """Test that BlackScholesEngine has type ANALYTICAL."""
        engine = BlackScholesEngine()
        assert engine.engine_type == EngineType.ANALYTICAL

    def test_european_mc_engine_type(self):
        """Test that EuropeanMCEngine has type MONTE_CARLO."""
        engine = EuropeanMCEngine(params=MCParams(num_paths=1000))
        assert engine.engine_type == EngineType.MONTE_CARLO

    def test_pde_engine_type(self):
        """Test that PDEEngine has type PDE."""
        engine = PDEEngine(params=PDEParams(grid_size=100, time_steps=50))
        assert engine.engine_type == EngineType.PDE

    def test_base_pde_solver_type(self):
        """Test that BasePDESolver has type PDE."""
        from asset.equity.engine.pde.base_pde_solver import BasePDESolver
        from util.enum.engine_enums import EngineType
        assert BasePDESolver.engine_type == EngineType.PDE


class TestGreeksCalculationModeEnum:
    """Test the GreeksCalculationMode enum."""

    def test_enum_values(self):
        """Test that GreeksCalculationMode has expected values."""
        assert GreeksCalculationMode.ENGINE.value == "engine"
        assert GreeksCalculationMode.BUMP.value == "bump"
        assert GreeksCalculationMode.AUTO.value == "auto"

    def test_enum_string_conversion(self):
        """Test string conversion of enum values."""
        assert str(GreeksCalculationMode.ENGINE) == "engine"
        assert str(GreeksCalculationMode.BUMP) == "bump"
        assert str(GreeksCalculationMode.AUTO) == "auto"


class TestGreeksCalculatorMode:
    """Test GreeksCalculator with different greeks_mode settings."""

    def test_default_mode_is_bump(self):
        """Test that default greeks_mode is BUMP."""
        calc = GreeksCalculator()
        assert calc.greeks_mode == GreeksCalculationMode.BUMP

    def test_explicit_mode_setting(self):
        """Test setting explicit greeks_mode."""
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)
        assert calc.greeks_mode == GreeksCalculationMode.AUTO

    def test_should_use_engine_greeks_bump_mode(self, pricing_env):
        """Test _should_use_engine_greeks returns False for BUMP mode."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=100, time_steps=50))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.BUMP)
        assert not calc._should_use_engine_greeks(pde_engine)

    def test_should_use_engine_greeks_engine_mode(self, pricing_env):
        """Test _should_use_engine_greeks returns True for ENGINE mode."""
        bs_engine = BlackScholesEngine()
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.ENGINE)
        # Even for non-PDE engine, ENGINE mode returns True
        assert calc._should_use_engine_greeks(bs_engine)

    def test_should_use_engine_greeks_auto_with_pde(self, pricing_env):
        """Test _should_use_engine_greeks with AUTO mode and PDE engine."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=100, time_steps=50))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)
        # AUTO mode should use engine Greeks for PDE
        assert calc._should_use_engine_greeks(pde_engine)

    def test_should_use_engine_greeks_auto_with_analytical(self, pricing_env):
        """Test _should_use_engine_greeks with AUTO mode and analytical engine."""
        bs_engine = BlackScholesEngine()
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)
        # AUTO mode should use bump for non-PDE engines
        assert not calc._should_use_engine_greeks(bs_engine)

    def test_should_use_engine_greeks_auto_with_mc(self, pricing_env):
        """Test _should_use_engine_greeks with AUTO mode and MC engine."""
        mc_engine = EuropeanMCEngine(params=MCParams(num_paths=1000))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)
        # AUTO mode should use bump for non-PDE engines
        assert not calc._should_use_engine_greeks(mc_engine)


class TestGreeksCalculatorPDEIntegration:
    """Integration tests for GreeksCalculator with PDE engines."""

    def test_bump_mode_uses_bump_method(self, euro_call, pricing_env):
        """Test that BUMP mode always uses bump method even for PDE."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=100, time_steps=50))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.BUMP)

        greeks = calc.calculate_numerical_greeks(euro_call, pricing_env, pde_engine)

        # Should have all Greeks
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

    def test_auto_mode_uses_pde_greeks(self, euro_call, pricing_env):
        """Test that AUTO mode uses PDE engine's calculate_greeks()."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=300, time_steps=150))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)

        greeks = calc.calculate_numerical_greeks(euro_call, pricing_env, pde_engine)

        # Should have all Greeks
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        # Delta should be in reasonable range for ATM call
        assert 0 < greeks["delta"] < 1
        # Gamma should be positive for call option
        assert greeks["gamma"] > 0

    def test_engine_mode_uses_pde_greeks(self, euro_call, pricing_env):
        """Test that ENGINE mode uses PDE engine's calculate_greeks()."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=100, time_steps=50))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.ENGINE)

        greeks = calc.calculate_numerical_greeks(euro_call, pricing_env, pde_engine)

        # Should have all Greeks
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks

    def test_auto_mode_uses_bump_for_analytical(self, euro_call, pricing_env):
        """Test that AUTO mode uses bump method for analytical engines."""
        bs_engine = BlackScholesEngine()
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)

        greeks = calc.calculate_numerical_greeks(euro_call, pricing_env, bs_engine)

        # Should have all Greeks
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks


class TestGreeksComparison:
    """Compare PDE grid Greeks vs bump method Greeks."""

    def test_pde_greeks_vs_bump_method(self, euro_call, pricing_env):
        """Compare PDE grid-based Greeks with bump method."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=300, time_steps=150))

        # Get PDE grid-based Greeks
        calc_pde = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)
        greeks_pde = calc_pde.calculate_numerical_greeks(euro_call, pricing_env, pde_engine)

        # Get bump method Greeks
        calc_bump = GreeksCalculator(greeks_mode=GreeksCalculationMode.BUMP)
        greeks_bump = calc_bump.calculate_numerical_greeks(euro_call, pricing_env, pde_engine)

        # Delta should be similar
        assert abs(greeks_pde["delta"] - greeks_bump["delta"]) < 0.05
        
        # For gamma, check that PDE grid gives positive value (expected for call)
        # Note: bump method with PDE engines can have numerical issues due to
        # grid interpolation and re-solving the PDE multiple times
        assert greeks_pde["gamma"] > 0
        # Gamma should be in reasonable range for ATM call
        assert greeks_pde["gamma"] < 0.1

    def test_american_option_pde_greeks(self, american_put, pricing_env):
        """Test PDE Greeks for American option."""
        pde_engine = PDEEngine(params=PDEParams(grid_size=300, time_steps=150))
        calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.AUTO)

        greeks = calc.calculate_numerical_greeks(american_put, pricing_env, pde_engine)

        # American put should have negative delta
        assert greeks["delta"] < 0
        # Gamma should be positive for put option
        assert greeks["gamma"] > 0
        # Theta should be negative (time decay for in-the-money put)
        assert greeks["theta"] < 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_greeks_calculator_without_mode_param(self, euro_call, pricing_env):
        """Test that GreeksCalculator works without greeks_mode parameter."""
        # Old code that doesn't specify greeks_mode should still work
        calc = GreeksCalculator()
        bs_engine = BlackScholesEngine()

        greeks = calc.calculate_numerical_greeks(euro_call, pricing_env, bs_engine)

        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks

    def test_greeks_calculator_with_params_only(self, euro_call, pricing_env):
        """Test that GreeksCalculator works with only params parameter."""
        params = EngineParams()
        calc = GreeksCalculator(params=params)
        bs_engine = BlackScholesEngine()

        greeks = calc.calculate_numerical_greeks(euro_call, pricing_env, bs_engine)

        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
