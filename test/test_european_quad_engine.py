"""
Unit tests for European vanilla option quadrature pricing engine.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.quad import EuropeanQuadEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.param import QuadParams
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType, QuadratureMethod
from quantark.util.exceptions import NumericalError, PricingError, ValidationError


def _discounted_european_lower_bound(
    option: EuropeanVanillaOption,
    spot: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
) -> float:
    spot_pv = spot * math.exp(-dividend_yield * maturity)
    strike_pv = option.strike * math.exp(-rate * maturity)
    if option.is_call():
        return max(spot_pv - strike_pv, 0.0) * option.contract_multiplier
    return max(strike_pv - spot_pv, 0.0) * option.contract_multiplier


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div: float = 0.02,
) -> PricingEnvironment:
    """Create a standard pricing environment for testing."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


class TestEuropeanQuadEngineBasic:
    """Basic functionality tests for EuropeanQuadEngine."""

    def test_rejects_price_below_discounted_european_lower_bound(self):
        """Quadrature sanity gate uses the discounted European lower bound."""
        pricing_env = create_pricing_env(spot=120.0, rate=0.01, div=0.0)
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
        engine = EuropeanQuadEngine()
        lower_bound = _discounted_european_lower_bound(call, 120.0, 0.01, 0.0, 1.0)
        engine._price_simpson = lambda *args, **kwargs: lower_bound - 1e-3

        with pytest.raises(NumericalError, match="discounted European lower bound"):
            engine.price(call, pricing_env)

    def test_still_rejects_negative_prices(self):
        """The lower-bound fix must not weaken negative price validation."""
        pricing_env = create_pricing_env(spot=100.0, rate=0.01, div=0.0)
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
        engine = EuropeanQuadEngine()
        engine._price_simpson = lambda *args, **kwargs: -1.0

        with pytest.raises(NumericalError, match="Negative price computed"):
            engine.price(call, pricing_env)

    def test_call_option_pricing_vs_black_scholes(self):
        """Test call option pricing matches Black-Scholes."""
        pricing_env = create_pricing_env()

        call = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(call, pricing_env)
        quad_price = quad_engine.price(call, pricing_env)

        # Quadrature should match BS within 0.01
        assert abs(quad_price - bs_price) < 0.01, (
            f"Quad price {quad_price:.6f} differs from BS {bs_price:.6f} "
            f"by {abs(quad_price - bs_price):.6f}"
        )

    def test_put_option_pricing_vs_black_scholes(self):
        """Test put option pricing matches Black-Scholes."""
        pricing_env = create_pricing_env()

        put = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.PUT, maturity=1.0
        )

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(put, pricing_env)
        quad_price = quad_engine.price(put, pricing_env)

        assert abs(quad_price - bs_price) < 0.01, (
            f"Quad price {quad_price:.6f} differs from BS {bs_price:.6f}"
        )

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        q = 0.02

        pricing_env = create_pricing_env(spot=S, rate=r, div=q)

        call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
        put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)

        engine = EuropeanQuadEngine()
        call_price = engine.price(call, pricing_env)
        put_price = engine.price(put, pricing_env)

        # Put-Call Parity: C - P = S*e^(-qT) - K*e^(-rT)
        lhs = call_price - put_price
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 0.01, f"Put-call parity violated: {lhs} vs {rhs}"


class TestEuropeanQuadEngineMoneyness:
    """Test different moneyness scenarios."""

    def test_deep_itm_call(self):
        """Test deep in-the-money call option."""
        pricing_env = create_pricing_env()

        itm_call = EuropeanVanillaOption(80.0, OptionType.CALL, maturity=1.0)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(itm_call, pricing_env)
        quad_price = quad_engine.price(itm_call, pricing_env)

        assert quad_price > 20.0, "Deep ITM call should have high value"
        assert abs(quad_price - bs_price) < 0.01

    def test_deep_otm_call(self):
        """Test deep out-of-the-money call option."""
        pricing_env = create_pricing_env()

        otm_call = EuropeanVanillaOption(120.0, OptionType.CALL, maturity=1.0)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(otm_call, pricing_env)
        quad_price = quad_engine.price(otm_call, pricing_env)

        assert quad_price < 5.0, "Deep OTM call should have low value"
        assert abs(quad_price - bs_price) < 0.01

    def test_deep_itm_put(self):
        """Test deep in-the-money put option."""
        # Use zero rate and dividend to avoid BS intrinsic value issues
        pricing_env = create_pricing_env(rate=0.0, div=0.0)

        itm_put = EuropeanVanillaOption(115.0, OptionType.PUT, maturity=1.0)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(itm_put, pricing_env)
        quad_price = quad_engine.price(itm_put, pricing_env)

        assert quad_price > 10.0, "Deep ITM put should have high value"
        assert abs(quad_price - bs_price) < 0.01

    def test_deep_otm_put(self):
        """Test deep out-of-the-money put option."""
        pricing_env = create_pricing_env()

        otm_put = EuropeanVanillaOption(80.0, OptionType.PUT, maturity=1.0)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(otm_put, pricing_env)
        quad_price = quad_engine.price(otm_put, pricing_env)

        assert quad_price < 3.0, "Deep OTM put should have low value"
        assert abs(quad_price - bs_price) < 0.01


class TestEuropeanQuadEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_near_expiry_call(self):
        """Test call option near expiry."""
        pricing_env = create_pricing_env()

        # Very short maturity
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=0.01)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(call, pricing_env)
        quad_price = quad_engine.price(call, pricing_env)

        assert abs(quad_price - bs_price) < 0.01

    def test_expired_option_returns_payoff(self):
        """Test that near-expired option returns close to intrinsic value."""
        pricing_env = create_pricing_env()

        # Very small maturity (nearly expired)
        call = EuropeanVanillaOption(90.0, OptionType.CALL, maturity=1e-11)

        engine = EuropeanQuadEngine()
        price = engine.price(call, pricing_env)

        # Should return intrinsic value: max(100 - 90, 0) = 10
        assert abs(price - 10.0) < 0.001

    def test_high_volatility(self):
        """Test with high volatility."""
        pricing_env = create_pricing_env(vol=0.80)

        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(call, pricing_env)
        quad_price = quad_engine.price(call, pricing_env)

        assert abs(quad_price - bs_price) < 0.05  # Slightly larger tolerance for high vol

    def test_long_maturity(self):
        """Test with long maturity."""
        pricing_env = create_pricing_env()

        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=5.0)

        bs_engine = BlackScholesEngine()
        quad_engine = EuropeanQuadEngine()

        bs_price = bs_engine.price(call, pricing_env)
        quad_price = quad_engine.price(call, pricing_env)

        assert abs(quad_price - bs_price) < 0.02


class TestEuropeanQuadEngineMethodSelection:
    """Test different quadrature methods and initialization patterns."""

    def test_simpson_method(self):
        """Test Simpson's rule method."""
        pricing_env = create_pricing_env()
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)

        engine = EuropeanQuadEngine(method=QuadratureMethod.SIMPSON)
        price = engine.price(call, pricing_env)

        assert price > 0

    def test_gauss_legendre_method(self):
        """Test Gauss-Legendre method."""
        pricing_env = create_pricing_env()
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)

        engine = EuropeanQuadEngine(method=QuadratureMethod.GAUSS_LEGENDRE)
        price = engine.price(call, pricing_env)

        bs_engine = BlackScholesEngine()
        bs_price = bs_engine.price(call, pricing_env)

        # Gauss-Legendre uses fewer points, so slightly larger tolerance
        assert abs(price - bs_price) < 0.05

    def test_string_method_selection(self):
        """Test method selection via string."""
        engine = EuropeanQuadEngine(method="simpson")
        assert engine.method == QuadratureMethod.SIMPSON

        engine = EuropeanQuadEngine(method="gauss_legendre")
        assert engine.method == QuadratureMethod.GAUSS_LEGENDRE

    def test_two_level_enum_pattern(self):
        """Test two-level enum pattern for method selection."""
        engine = EuropeanQuadEngine(
            method=EngineType.QUADRATURE(QuadratureMethod.SIMPSON)
        )
        assert engine.method == QuadratureMethod.SIMPSON

    def test_default_method(self):
        """Test default method is Simpson."""
        engine = EuropeanQuadEngine()
        assert engine.method == QuadratureMethod.SIMPSON


class TestEuropeanQuadEngineParams:
    """Test parameter configuration."""

    def test_custom_grid_points(self):
        """Test custom grid points parameter."""
        params = QuadParams(grid_points=2001)
        engine = EuropeanQuadEngine(params=params)

        pricing_env = create_pricing_env()
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)

        price = engine.price(call, pricing_env)
        assert price > 0

    def test_convergence_with_grid_size(self):
        """Test that higher grid points improve accuracy."""
        pricing_env = create_pricing_env()
        call = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)

        bs_engine = BlackScholesEngine()
        bs_price = bs_engine.price(call, pricing_env)

        # Lower grid points
        params_low = QuadParams(grid_points=501)
        engine_low = EuropeanQuadEngine(params=params_low)
        price_low = engine_low.price(call, pricing_env)
        error_low = abs(price_low - bs_price)

        # Higher grid points
        params_high = QuadParams(grid_points=2001)
        engine_high = EuropeanQuadEngine(params=params_high)
        price_high = engine_high.price(call, pricing_env)
        error_high = abs(price_high - bs_price)

        # Higher grid should be more accurate (or at least not worse)
        assert error_high <= error_low + 0.001  # Small tolerance for numerical noise


class TestEuropeanQuadEngineValidation:
    """Test validation and error handling."""

    def test_wrong_product_type_raises_error(self):
        """Test that non-European option raises PricingError."""
        from quantark.asset.equity.product.option import AmericanOption

        pricing_env = create_pricing_env()
        american = AmericanOption(100.0, OptionType.CALL, maturity=1.0)

        engine = EuropeanQuadEngine()

        with pytest.raises(PricingError):
            engine.price(american, pricing_env)

    def test_invalid_method_string_raises_error(self):
        """Test that invalid method string raises ValidationError."""
        with pytest.raises(ValidationError):
            EuropeanQuadEngine(method="invalid_method")

    def test_invalid_grid_points_raises_error(self):
        """Test that invalid grid points raises ValidationError."""
        with pytest.raises(ValidationError):
            QuadParams(grid_points=0)

        with pytest.raises(ValidationError):
            QuadParams(grid_points=50)  # Too few

    def test_invalid_num_std_devs_raises_error(self):
        """Test that invalid num_std_devs raises ValidationError."""
        with pytest.raises(ValidationError):
            QuadParams(num_std_devs=0)

        with pytest.raises(ValidationError):
            QuadParams(num_std_devs=2)  # Too few


class TestEuropeanQuadEngineContractMultiplier:
    """Test contract multiplier handling."""

    def test_contract_multiplier_applied(self):
        """Test that contract multiplier is applied to price."""
        pricing_env = create_pricing_env()

        call_no_mult = EuropeanVanillaOption(
            100.0, OptionType.CALL, maturity=1.0, contract_multiplier=1.0
        )
        call_with_mult = EuropeanVanillaOption(
            100.0, OptionType.CALL, maturity=1.0, contract_multiplier=100.0
        )

        engine = EuropeanQuadEngine()
        price_no_mult = engine.price(call_no_mult, pricing_env)
        price_with_mult = engine.price(call_with_mult, pricing_env)

        assert abs(price_with_mult - 100.0 * price_no_mult) < 0.01


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("Running EuropeanQuadEngine Unit Tests")
    print("=" * 70 + "\n")

    # Collect all test classes
    test_classes = [
        TestEuropeanQuadEngineBasic,
        TestEuropeanQuadEngineMoneyness,
        TestEuropeanQuadEngineEdgeCases,
        TestEuropeanQuadEngineMethodSelection,
        TestEuropeanQuadEngineParams,
        TestEuropeanQuadEngineValidation,
        TestEuropeanQuadEngineContractMultiplier,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
