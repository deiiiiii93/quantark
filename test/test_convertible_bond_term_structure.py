"""
Tests for convertible bond engines with term structure support.

These tests verify that PDE and Trinomial engines correctly handle
non-flat rate curves and volatility surfaces by producing different
prices for flat vs stepped term structures with the same average.
"""
import logging
import math
import pytest
from datetime import datetime, timedelta

from asset.bond.product.convertible.convertible_bond import ConvertibleBond
from asset.bond.engine.convertible.convertible_bond_engine import (
    ConvertibleBondEngine,
)
from asset.bond.engine.pde.convertible.jump_diffusion_engine import (
    ConvertibleBondJumpDiffusionEngine,
)
from asset.bond.engine.pde.convertible.tf_engine import ConvertibleBondTFEngine
from asset.bond.engine.pde.convertible.pde_params import ConvertibleBondPDEParams
from asset.bond.engine.tree.convertible.trinomial_engine import (
    ConvertibleBondTrinomialEngine,
)
from asset.bond.engine.tree.convertible.binomial_engine import (
    ConvertibleBondBinomialEngine,
)
from asset.bond.engine.tree.convertible.tree_params import ConvertibleBondTreeParams
from param.rrf.rate_curve import FlatRateCurve, RateCurve
from param.vol.vol_surface import FlatVolSurface, VolatilitySurface
from param.quote.spot_quote import SpotQuote
from priceenv import PricingEnvironment
from util.enum.engine_enums import ConvertibleBondTrinomialVolScheme
from util.enum.engine_enums import ConvertibleBondMethod
from util.numerical import safe_exp


class SteppedRateCurve(RateCurve):
    """
    Piecewise-constant (stepped) rate curve for testing.

    Returns rate1 for t < switch_time, rate2 for t >= switch_time.
    """

    def __init__(self, rate1: float, rate2: float, switch_time: float):
        """
        Initialize stepped rate curve.

        Args:
            rate1: Rate for t < switch_time
            rate2: Rate for t >= switch_time
            switch_time: Time at which rate switches (years)
        """
        self.rate1 = rate1
        self.rate2 = rate2
        self.switch_time = switch_time

    def get_rate(self, time_to_maturity: float) -> float:
        """Get spot rate at given maturity."""
        if time_to_maturity < self.switch_time:
            return self.rate1
        return self.rate2

    def get_discount_factor(self, time_to_maturity: float) -> float:
        """Get discount factor at given maturity."""
        if time_to_maturity <= 0:
            return 1.0
        if time_to_maturity <= self.switch_time:
            return safe_exp(-self.rate1 * time_to_maturity)
        # Discount first part at rate1, second part at rate2
        df1 = safe_exp(-self.rate1 * self.switch_time)
        df2 = safe_exp(-self.rate2 * (time_to_maturity - self.switch_time))
        return df1 * df2

    def __repr__(self):
        return f"SteppedRateCurve(rate1={self.rate1:.2%}, rate2={self.rate2:.2%}, switch={self.switch_time}y)"


class SteppedVolSurface(VolatilitySurface):
    """
    Piecewise total-variance volatility surface for testing.

    Uses vol1 for t <= switch_time and vol2 for t > switch_time, but keeps
    total variance continuous so forward volatility remains bounded.
    """

    def __init__(self, vol1: float, vol2: float, switch_time: float):
        """
        Initialize stepped volatility surface.

        Args:
            vol1: Volatility for t <= switch_time
            vol2: Volatility for t > switch_time
            switch_time: Time at which volatility switches (years)
        """
        self.vol1 = vol1
        self.vol2 = vol2
        self.switch_time = switch_time

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """Return implied volatility with continuous total variance."""
        if time_to_maturity <= 0:
            return self.vol1
        if time_to_maturity <= self.switch_time:
            return self.vol1

        total_var = (
            self.vol1 * self.vol1 * self.switch_time
            + self.vol2 * self.vol2 * (time_to_maturity - self.switch_time)
        )
        return math.sqrt(total_var / time_to_maturity)

    def __repr__(self):
        return (
            f"SteppedVolSurface(vol1={self.vol1:.2%}, "
            f"vol2={self.vol2:.2%}, switch={self.switch_time}y)"
        )


@pytest.fixture
def valuation_date():
    """Valuation date for tests."""
    return datetime(2024, 1, 1)


@pytest.fixture
def sample_convertible_bond(valuation_date):
    """
    Create a sample 4-year convertible bond for testing.

    The bond is set up with reasonable parameters for term structure testing.
    Use explicit dates aligned with payment schedules to avoid cashflow issues.
    """
    issue_date = datetime(2024, 1, 1)
    maturity_date = datetime(2028, 1, 1)  # 4-year maturity

    return ConvertibleBond(
        face_value=100.0,
        coupon_rate=0.02,  # 2% annual coupon
        issue_date=issue_date,
        maturity_date=maturity_date,
        conversion_ratio=1.0,  # 1 share per bond
        conversion_start_date=issue_date,
        conversion_end_date=maturity_date,
        credit_spread=0.02,  # 2% credit spread
        hazard_rate=0.01,  # 1% annual default intensity
        recovery_rate=0.4,  # 40% recovery
        stock_jump_on_default=0.5,  # 50% stock drop on default
        continuous_dividend_yield=0.01,  # 1% dividend yield
    )


@pytest.fixture
def flat_rate_curve():
    """Flat 5% rate curve."""
    return FlatRateCurve(rate=0.05)


@pytest.fixture
def stepped_rate_curve():
    """
    Stepped rate curve: 1% for years 0-2, 9% for years 2-4.

    This has approximately the same average rate as 5% flat,
    but the timing differs significantly.
    """
    return SteppedRateCurve(rate1=0.01, rate2=0.09, switch_time=2.0)


@pytest.fixture
def flat_vol_surface():
    """Flat 30% volatility surface."""
    return FlatVolSurface(volatility=0.30)


@pytest.fixture
def stepped_vol_surface():
    """
    Stepped volatility surface: 5% for years 0-2, 45% for years 2-4.

    This has a higher long-dated volatility with a lower short-dated volatility
    to exercise term-structure handling in tree engines.
    """
    return SteppedVolSurface(vol1=0.05, vol2=0.45, switch_time=2.0)


@pytest.fixture
def flat_vol_surface_high():
    """Flat 45% volatility surface for term-structure comparison."""
    return FlatVolSurface(volatility=0.45)


@pytest.fixture
def spot_quote():
    """Spot quote at $100."""
    return SpotQuote(spot=100.0)


@pytest.fixture
def pricing_env_flat(valuation_date, spot_quote, flat_rate_curve, flat_vol_surface):
    """Pricing environment with flat rate curve and vol surface."""
    return PricingEnvironment(
        valuation_date=valuation_date,
        spot_quote=spot_quote,
        rate_curve=flat_rate_curve,
        vol_surface=flat_vol_surface,
    )


@pytest.fixture
def pricing_env_stepped(valuation_date, spot_quote, stepped_rate_curve, flat_vol_surface):
    """Pricing environment with stepped rate curve."""
    return PricingEnvironment(
        valuation_date=valuation_date,
        spot_quote=spot_quote,
        rate_curve=stepped_rate_curve,
        vol_surface=flat_vol_surface,
    )


@pytest.fixture
def pricing_env_stepped_vol(
    valuation_date, spot_quote, flat_rate_curve, stepped_vol_surface
):
    """Pricing environment with stepped volatility surface."""
    return PricingEnvironment(
        valuation_date=valuation_date,
        spot_quote=spot_quote,
        rate_curve=flat_rate_curve,
        vol_surface=stepped_vol_surface,
    )


@pytest.fixture
def pricing_env_flat_vol_high(
    valuation_date, spot_quote, flat_rate_curve, flat_vol_surface_high
):
    """Pricing environment with high flat volatility surface."""
    return PricingEnvironment(
        valuation_date=valuation_date,
        spot_quote=spot_quote,
        rate_curve=flat_rate_curve,
        vol_surface=flat_vol_surface_high,
    )


class TestJumpDiffusionEngineTermStructure:
    """Tests for Jump-Diffusion PDE engine with term structure."""

    def test_flat_vs_stepped_produces_different_prices(
        self, sample_convertible_bond, pricing_env_flat, pricing_env_stepped
    ):
        """
        Flat vs stepped rate curves should produce different prices.

        The stepped curve has lower rates early and higher rates later,
        which should result in a different present value due to the
        timing of cash flows.
        """
        params = ConvertibleBondPDEParams(
            num_space_steps=50,
            num_time_steps=100,
        )

        engine_flat = ConvertibleBondJumpDiffusionEngine(pricing_env_flat, params)
        engine_stepped = ConvertibleBondJumpDiffusionEngine(pricing_env_stepped, params)

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        # Prices should differ by more than 0.1% of face value (0.1)
        price_diff = abs(price_stepped - price_flat)
        assert price_diff > 0.1, (
            f"Expected significant price difference for flat vs stepped curves. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, Diff: {price_diff:.4f}"
        )


class TestTFEngineTermStructure:
    """Tests for TF PDE engine with term structure."""

    def test_flat_vs_stepped_produces_different_prices(
        self, sample_convertible_bond, pricing_env_flat, pricing_env_stepped
    ):
        """
        Flat vs stepped rate curves should produce different prices.
        """
        params = ConvertibleBondPDEParams(
            num_space_steps=50,
            num_time_steps=100,
        )

        engine_flat = ConvertibleBondTFEngine(pricing_env_flat, params)
        engine_stepped = ConvertibleBondTFEngine(pricing_env_stepped, params)

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        # Prices should differ by more than 0.1% of face value
        price_diff = abs(price_stepped - price_flat)
        assert price_diff > 0.1, (
            f"Expected significant price difference for flat vs stepped curves. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, Diff: {price_diff:.4f}"
        )


class TestTrinomialEngineTermStructure:
    """Tests for Trinomial tree engine with term structure."""

    def test_flat_vs_stepped_produces_different_prices(
        self, sample_convertible_bond, pricing_env_flat, pricing_env_stepped
    ):
        """
        Flat vs stepped rate curves should produce different prices.
        """
        params = ConvertibleBondTreeParams(num_steps=100)

        engine_flat = ConvertibleBondTrinomialEngine(pricing_env_flat, params)
        engine_stepped = ConvertibleBondTrinomialEngine(pricing_env_stepped, params)

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        # Prices should differ by more than 0.1% of face value
        price_diff = abs(price_stepped - price_flat)
        assert price_diff > 0.1, (
            f"Expected significant price difference for flat vs stepped curves. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, Diff: {price_diff:.4f}"
        )


class TestTrinomialEngineVolTermStructure:
    """Tests for Trinomial tree engine with volatility term structure."""

    def test_invalid_trinomial_scheme_raises(self):
        """Invalid scheme names should fail parameter validation."""
        with pytest.raises(ValueError):
            ConvertibleBondTreeParams(
                num_steps=10, trinomial_vol_scheme="bad_scheme"
            )

    def test_constant_scheme_ignores_vol_term_structure(
        self,
        sample_convertible_bond,
        pricing_env_stepped_vol,
    ):
        """
        Constant-vol scheme should ignore volatility term structure.
        """
        params = ConvertibleBondTreeParams(
            num_steps=100,
            trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.CONSTANT_VOL,
        )

        engine_stepped = ConvertibleBondTrinomialEngine(
            pricing_env_stepped_vol, params
        )

        max_vol = engine_stepped._calculate_max_vol_for_grid(
            sample_convertible_bond
        )
        flat_env = PricingEnvironment(
            valuation_date=pricing_env_stepped_vol.valuation_date,
            spot_quote=pricing_env_stepped_vol.spot_quote,
            rate_curve=pricing_env_stepped_vol.rate_curve,
            vol_surface=FlatVolSurface(volatility=max_vol),
        )
        engine_flat = ConvertibleBondTrinomialEngine(flat_env, params)

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        price_diff = abs(price_stepped - price_flat)
        assert price_diff < 0.05, (
            "Expected constant-vol scheme to ignore term structure. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, "
            f"Diff: {price_diff:.4f}"
        )

    def test_fixed_dx_scheme_respects_vol_term_structure(
        self,
        sample_convertible_bond,
        pricing_env_flat_vol_high,
        pricing_env_stepped_vol,
    ):
        """
        Fixed-dx log-price scheme should reflect volatility term structure.
        """
        params = ConvertibleBondTreeParams(
            num_steps=100,
            trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.LOG_FIXED_DX,
        )

        engine_flat = ConvertibleBondTrinomialEngine(
            pricing_env_flat_vol_high, params
        )
        engine_stepped = ConvertibleBondTrinomialEngine(
            pricing_env_stepped_vol, params
        )

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        price_diff = abs(price_stepped - price_flat)
        assert price_diff > 0.05, (
            "Expected fixed-dx scheme to respond to term structure. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, "
            f"Diff: {price_diff:.4f}"
        )

    def test_variable_dx_scheme_respects_vol_term_structure(
        self,
        sample_convertible_bond,
        pricing_env_flat_vol_high,
        pricing_env_stepped_vol,
    ):
        """
        Variable-dx log-price scheme should reflect volatility term structure.
        """
        params = ConvertibleBondTreeParams(
            num_steps=100,
            trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.LOG_VARIABLE_DX,
        )

        engine_flat = ConvertibleBondTrinomialEngine(
            pricing_env_flat_vol_high, params
        )
        engine_stepped = ConvertibleBondTrinomialEngine(
            pricing_env_stepped_vol, params
        )

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        price_diff = abs(price_stepped - price_flat)
        assert price_diff > 0.05, (
            "Expected variable-dx scheme to respond to term structure. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, "
            f"Diff: {price_diff:.4f}"
        )

    def test_warning_logged_for_term_structure_under_constant_scheme(
        self, sample_convertible_bond, pricing_env_stepped_vol, caplog
    ):
        """
        Constant-vol scheme should warn when non-flat vol surface is used.
        """
        params = ConvertibleBondTreeParams(
            num_steps=50,
            trinomial_vol_scheme=ConvertibleBondTrinomialVolScheme.CONSTANT_VOL,
        )
        engine = ConvertibleBondTrinomialEngine(pricing_env_stepped_vol, params)

        with caplog.at_level(logging.WARNING):
            engine.price(sample_convertible_bond)

        assert any(
            "constant-vol scheme does not apply volatility term structure"
            in record.message
            for record in caplog.records
        ), "Expected warning about volatility term structure"

class TestBinomialEngineWarning:
    """Tests for Binomial engine non-flat curve warning."""

    def test_warning_logged_for_non_flat_rate_curve(
        self, sample_convertible_bond, pricing_env_stepped_vol, caplog
    ):
        """
        Binomial engine should log warning when non-flat vol surface is used.
        """
        params = ConvertibleBondTreeParams(num_steps=50)
        engine = ConvertibleBondBinomialEngine(pricing_env_stepped_vol, params)

        with caplog.at_level(logging.WARNING):
            engine.price(sample_convertible_bond)

        # Check that warning was logged
        assert any(
            "approximates volatility term structure" in record.message
            for record in caplog.records
        ), "Expected warning about piecewise curve approximation"

    def test_no_warning_for_flat_curves(
        self, sample_convertible_bond, pricing_env_flat, caplog
    ):
        """
        Binomial engine should NOT log warning when flat curves are used.
        """
        params = ConvertibleBondTreeParams(num_steps=50)
        engine = ConvertibleBondBinomialEngine(pricing_env_flat, params)

        with caplog.at_level(logging.WARNING):
            engine.price(sample_convertible_bond)

        # Check that no warning about piecewise curves was logged
        assert not any(
            "approximates volatility term structure" in record.message
            for record in caplog.records
        ), "Should not warn about piecewise curves when using flat curves"


class TestBinomialEnginePiecewiseRateCurve:
    """Tests for Binomial engine piecewise risk-free curve support."""

    def test_flat_vs_stepped_produces_different_prices(
        self, sample_convertible_bond, pricing_env_flat, pricing_env_stepped
    ):
        """
        Flat vs stepped rate curves should produce different prices.

        The binomial engine supports per-step forward rates under constant volatility.
        """
        params = ConvertibleBondTreeParams(num_steps=100)
        engine_flat = ConvertibleBondBinomialEngine(pricing_env_flat, params)
        engine_stepped = ConvertibleBondBinomialEngine(pricing_env_stepped, params)

        price_flat = engine_flat.price(sample_convertible_bond)
        price_stepped = engine_stepped.price(sample_convertible_bond)

        price_diff = abs(price_stepped - price_flat)
        assert price_diff > 0.1, (
            f"Expected significant price difference for flat vs stepped curves. "
            f"Flat: {price_flat:.4f}, Stepped: {price_stepped:.4f}, Diff: {price_diff:.4f}"
        )


class TestBackwardCompatibility:
    """Tests for backward compatibility with flat curves."""

    def test_engines_produce_consistent_results_with_flat_curves(
        self, sample_convertible_bond, pricing_env_flat
    ):
        """
        All engines should produce reasonable and consistent prices with flat curves.

        This verifies that the term structure changes don't break existing
        functionality for flat curve scenarios.
        """
        pde_params = ConvertibleBondPDEParams(
            num_space_steps=50,
            num_time_steps=100,
        )
        tree_params = ConvertibleBondTreeParams(num_steps=100)

        # Create engines
        jd_engine = ConvertibleBondJumpDiffusionEngine(pricing_env_flat, pde_params)
        tf_engine = ConvertibleBondTFEngine(pricing_env_flat, pde_params)
        tri_engine = ConvertibleBondTrinomialEngine(pricing_env_flat, tree_params)
        bin_engine = ConvertibleBondBinomialEngine(pricing_env_flat, tree_params)

        # Get prices
        jd_price = jd_engine.price(sample_convertible_bond)
        tf_price = tf_engine.price(sample_convertible_bond)
        tri_price = tri_engine.price(sample_convertible_bond)
        bin_price = bin_engine.price(sample_convertible_bond)

        # All prices should be positive and reasonable (between 80 and 150 for a 100 face value bond)
        for name, price in [
            ("Jump-Diffusion", jd_price),
            ("TF", tf_price),
            ("Trinomial", tri_price),
            ("Binomial", bin_price),
        ]:
            assert 80 <= price <= 150, f"{name} engine price {price:.4f} is outside reasonable range"

        # Prices from different engines should be within 5% of each other
        prices = [jd_price, tf_price, tri_price, bin_price]
        avg_price = sum(prices) / len(prices)
        for name, price in [
            ("Jump-Diffusion", jd_price),
            ("TF", tf_price),
            ("Trinomial", tri_price),
            ("Binomial", bin_price),
        ]:
            rel_diff = abs(price - avg_price) / avg_price
            assert rel_diff < 0.05, (
                f"{name} engine price {price:.4f} differs by {rel_diff:.1%} from average {avg_price:.4f}"
            )


class TestConvertibleRiskMetricsPiecewiseCurve:
    """Risk metrics should behave reasonably under piecewise curves."""

    def test_convexity_not_explosive_for_piecewise_curves(
        self, sample_convertible_bond, pricing_env_stepped
    ):
        """
        Convexity uses parallel curve bumps; it should not explode for piecewise curves.
        """
        tree_params = ConvertibleBondTreeParams(num_steps=50)
        engine = ConvertibleBondEngine(
            pricing_env_stepped,
            method=ConvertibleBondMethod.TRINOMIAL_HW,
            tree_params=tree_params,
        )
        result = engine.price_with_details(sample_convertible_bond)

        assert math.isfinite(result.convexity)
        assert abs(result.convexity) < 1e5, (
            f"Convexity magnitude seems wrong: {result.convexity}"
        )
        assert result.cs01 > 0.0

    def test_floor_bond_convexity_not_explosive_for_piecewise_curves(
        self, sample_convertible_bond, pricing_env_stepped
    ):
        """Floor bond convexity should be finite and not explode under piecewise curves."""
        tree_params = ConvertibleBondTreeParams(num_steps=50)
        engine = ConvertibleBondEngine(
            pricing_env_stepped,
            method=ConvertibleBondMethod.TRINOMIAL_HW,
            tree_params=tree_params,
        )
        result = engine.price_with_details(sample_convertible_bond)

        assert math.isfinite(result.floor_bond_convexity)
        assert abs(result.floor_bond_convexity) < 1e5, (
            f"Floor bond convexity magnitude seems wrong: {result.floor_bond_convexity}"
        )
