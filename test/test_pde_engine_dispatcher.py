"""
Additional tests for PDEEngine (unified PDE pricing engine with automatic dispatch).

This module extends the existing test_pde_engine.py with tests specific to the
unified PDEEngine dispatcher functionality.
"""

from pathlib import Path
import sys

# Add parent directory to path to import QuantArk modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from quantark.asset.equity.engine import PDEEngine
from quantark.asset.equity.engine.pde import (
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
)
from quantark.asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
    DoubleBarrierOption,
    OneTouchOption,
    DoubleOneTouchOption,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.util.enum import (
    OptionType,
    BarrierType,
    BarrierDirection,
    TouchType,
    DoubleBarrierType,
)
from quantark.util.enum.engine_enums import PDEMethod, EngineType
from quantark.util.exceptions import ValidationError


class TestPDEEngineBasics:
    """Test basic PDEEngine initialization and configuration."""

    def test_default_initialization(self):
        """Test PDEEngine with default parameters."""
        engine = PDEEngine()
        assert engine.method == PDEMethod.CRANK_NICOLSON
        assert engine.params is not None
        assert isinstance(engine.params, PDEParams)

    def test_initialization_with_params(self):
        """Test PDEEngine with custom PDEParams."""
        from quantark.asset.equity.engine.pde import GridConfig

        params = PDEParams(accuracy="high", grid=GridConfig(points=500))
        engine = PDEEngine(params=params)
        assert engine.params.accuracy == "high"
        assert engine.params.grid.points == 500

    def test_method_selection_enum(self):
        """Test method selection via PDEMethod enum."""
        engine = PDEEngine(method=PDEMethod.EXPLICIT_EULER)
        assert engine.method == PDEMethod.EXPLICIT_EULER
        assert engine.params.scheme == "explicit_euler"

    def test_method_selection_two_level_enum(self):
        """Test method selection via EngineType.PDE(PDEMethod) pattern."""
        engine = PDEEngine(method=EngineType.PDE(PDEMethod.IMPLICIT_EULER))
        assert engine.method == PDEMethod.IMPLICIT_EULER
        assert engine.params.scheme == "implicit_euler"

    def test_method_selection_string(self):
        """Test backward-compatible string method selection."""
        engine = PDEEngine(method="crank_nicolson")
        assert engine.method == PDEMethod.CRANK_NICOLSON

    def test_invalid_method_string(self):
        """Test error handling for invalid method string."""
        with pytest.raises(ValidationError, match="Invalid PDE method"):
            PDEEngine(method="invalid_method")

    def test_invalid_method_type(self):
        """Test error handling for invalid method type."""
        with pytest.raises(ValidationError, match="Invalid method type"):
            PDEEngine(method=123)

    def test_invalid_engine_type_in_tuple(self):
        """Test error handling when wrong EngineType used in tuple."""
        with pytest.raises(ValidationError, match="Expected EngineType.PDE"):
            PDEEngine(method=EngineType.ANALYTICAL(PDEMethod.CRANK_NICOLSON))


class TestPDEEngineDispatch:
    """Test automatic product-to-solver dispatch."""

    @pytest.fixture
    def pricing_env(self):
        """Create standard pricing environment."""
        return PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(0.2),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
        )

    @pytest.fixture
    def pde_engine(self):
        """Create PDEEngine with reasonable parameters."""
        return PDEEngine(params=PDEParams())

    def test_european_option_dispatch(self, pde_engine, pricing_env):
        """Test dispatch to EuropeanPDESolver."""
        product = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        price = pde_engine.price(product, pricing_env)

        assert isinstance(price, float)
        assert price > 0
        assert EuropeanVanillaOption in pde_engine._solver_cache
        assert isinstance(
            pde_engine._solver_cache[EuropeanVanillaOption], EuropeanPDESolver
        )

    def test_american_option_dispatch(self, pde_engine, pricing_env):
        """Test dispatch to AmericanPDESolver."""
        product = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

        price = pde_engine.price(product, pricing_env)

        assert isinstance(price, float)
        assert price > 0
        assert AmericanOption in pde_engine._solver_cache
        assert isinstance(pde_engine._solver_cache[AmericanOption], AmericanPDESolver)

    def test_barrier_option_dispatch(self, pde_engine, pricing_env):
        """Test dispatch to BarrierPDESolver."""
        product = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
        )

        price = pde_engine.price(product, pricing_env)

        assert isinstance(price, float)
        assert price >= 0
        assert BarrierOption in pde_engine._solver_cache
        assert isinstance(pde_engine._solver_cache[BarrierOption], BarrierPDESolver)

    def test_double_barrier_option_dispatch(self, pde_engine, pricing_env):
        """Test dispatch to DoubleBarrierPDESolver."""
        product = DoubleBarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            upper_barrier=120.0,
            lower_barrier=80.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            maturity=1.0,
        )

        price = pde_engine.price(product, pricing_env)

        assert isinstance(price, float)
        assert price >= 0
        assert DoubleBarrierOption in pde_engine._solver_cache
        assert isinstance(
            pde_engine._solver_cache[DoubleBarrierOption], DoubleBarrierPDESolver
        )

    def test_one_touch_option_dispatch(self, pde_engine, pricing_env):
        """Test dispatch to OneTouchPDESolver."""
        product = OneTouchOption(
            barrier=120.0,
            barrier_direction=BarrierDirection.UP,
            touch_type=TouchType.ONE_TOUCH,
            rebate=1.0,
            maturity=1.0,
        )

        price = pde_engine.price(product, pricing_env)

        assert isinstance(price, float)
        assert 0 <= price <= 1.0  # Binary option
        assert OneTouchOption in pde_engine._solver_cache
        assert isinstance(pde_engine._solver_cache[OneTouchOption], OneTouchPDESolver)

    def test_double_one_touch_option_dispatch(self, pde_engine, pricing_env):
        """Test dispatch to DoubleOneTouchPDESolver."""
        product = DoubleOneTouchOption(
            upper_barrier=120.0,
            lower_barrier=80.0,
            touch_type=TouchType.DOUBLE_ONE_TOUCH,
            rebate=1.0,
            maturity=1.0,
        )

        price = pde_engine.price(product, pricing_env)

        assert isinstance(price, float)
        assert 0 <= price <= 1.0  # Binary option
        assert DoubleOneTouchOption in pde_engine._solver_cache
        assert isinstance(
            pde_engine._solver_cache[DoubleOneTouchOption], DoubleOneTouchPDESolver
        )

    def test_solver_caching(self, pde_engine, pricing_env):
        """Test that solvers are cached and reused."""
        product1 = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        product2 = EuropeanVanillaOption(
            strike=110.0, option_type=OptionType.PUT, maturity=2.0
        )

        pde_engine.price(product1, pricing_env)
        solver1 = pde_engine._solver_cache[EuropeanVanillaOption]

        pde_engine.price(product2, pricing_env)
        solver2 = pde_engine._solver_cache[EuropeanVanillaOption]

        assert solver1 is solver2  # Same instance reused


class TestPDEEngineErrorHandling:
    """Test error handling and validation."""

    @pytest.fixture
    def pricing_env(self):
        """Create standard pricing environment."""
        return PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(0.2),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
        )

    def test_none_product(self, pricing_env):
        """Test error handling for None product."""
        engine = PDEEngine()
        with pytest.raises(ValidationError, match="Product cannot be None"):
            engine.price(None, pricing_env)

    def test_unsupported_product_type(self, pricing_env):
        """Test error handling for unsupported product type."""
        from quantark.asset.equity.product.base_equity_product import BaseEquityProduct

        class UnsupportedProduct(BaseEquityProduct):
            def get_maturity(self, pricing_env):
                return 1.0

            def validate(self):
                pass

            def get_payoff(self, spot):
                return 0.0

        engine = PDEEngine()
        product = UnsupportedProduct()

        with pytest.raises(
            ValidationError, match="PDEEngine does not support product type"
        ):
            engine.price(product, pricing_env)


class TestPDENumericalConsistency:
    """Test numerical consistency between PDEEngine and direct solver usage."""

    @pytest.fixture
    def pricing_env(self):
        """Create standard pricing environment."""
        return PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(0.2),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
        )

    @pytest.fixture
    def pde_params(self):
        """Create consistent PDE parameters."""
        return PDEParams()

    def test_european_option_consistency(self, pricing_env, pde_params):
        """Test European option price consistency."""
        product = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        engine = PDEEngine(params=pde_params)
        price_engine = engine.price(product, pricing_env)

        solver = EuropeanPDESolver(params=pde_params)
        price_solver = solver.price(product, pricing_env)

        assert abs(price_engine - price_solver) < 1e-10

    def test_american_option_consistency(self, pricing_env, pde_params):
        """Test American option price consistency."""
        product = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

        engine = PDEEngine(params=pde_params)
        price_engine = engine.price(product, pricing_env)

        solver = AmericanPDESolver(params=pde_params)
        price_solver = solver.price(product, pricing_env)

        assert abs(price_engine - price_solver) < 1e-10

    def test_barrier_option_consistency(self, pricing_env, pde_params):
        """Test barrier option price consistency."""
        product = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
        )

        engine = PDEEngine(params=pde_params)
        price_engine = engine.price(product, pricing_env)

        solver = BarrierPDESolver(params=pde_params)
        price_solver = solver.price(product, pricing_env)

        assert abs(price_engine - price_solver) < 1e-10


class TestGreeksCalculatorIntegration:
    """Test integration with GreeksCalculator for numerical Greeks."""

    @pytest.fixture
    def pricing_env(self):
        """Create standard pricing environment."""
        return PricingEnvironment(
            valuation_date=datetime(2024, 1, 1),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(0.2),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
        )

    @pytest.fixture
    def pde_engine(self):
        """Create PDEEngine with reasonable parameters."""
        return PDEEngine(params=PDEParams())

    @pytest.fixture
    def calculator(self):
        """Create GreeksCalculator with ENGINE mode for PDE grid-based Greeks."""
        from quantark.util.enum.engine_enums import GreeksCalculationMode
        return GreeksCalculator(greeks_mode=GreeksCalculationMode.ENGINE)

    def test_european_option_greeks(self, pde_engine, pricing_env, calculator):
        """Test Greeks calculation for European option via PDE."""
        product = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        greeks = calculator.calculate_numerical_greeks(product, pricing_env, pde_engine)

        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

        assert greeks["price"] > 0
        assert 0 < greeks["delta"] < 1  # Call delta
        assert 0 < greeks["gamma"] < 0.1  # Grid-based gamma, should be positive and small
        assert abs(greeks["vega"]) < 100  # Should be reasonable

    def test_american_option_greeks(self, pde_engine, pricing_env, calculator):
        """Test Greeks calculation for American option via PDE."""
        product = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)

        greeks = calculator.calculate_numerical_greeks(product, pricing_env, pde_engine)

        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks

        assert greeks["price"] > 0
        assert -1 < greeks["delta"] < 0  # Put delta
        assert greeks["gamma"] > 0

    def test_barrier_option_greeks(self, pde_engine, pricing_env, calculator):
        """Test Greeks calculation for barrier option via PDE."""
        product = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
        )

        greeks = calculator.calculate_numerical_greeks(product, pricing_env, pde_engine)

        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert greeks["price"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
