"""
Comprehensive tests for convertible bond pricing engines.
"""
import unittest
from datetime import datetime

from asset.bond.product.convertible.convertible_bond import (
    ConvertibleBond,
    CallScheduleEntry,
    PutScheduleEntry,
)
from asset.bond.engine.tree.convertible import (
    ConvertibleBondTreeParams,
    ConvertibleBondBinomialEngine,
    ConvertibleBondTrinomialEngine,
)
from asset.bond.engine.pde.convertible import (
    ConvertibleBondPDEParams,
    ConvertibleBondJumpDiffusionEngine,
    ConvertibleBondTFEngine,
)
from asset.bond.engine.convertible import (
    ConvertibleBondEngine,
    ConvertibleBondResult,
)
from param.quote import SpotQuote
from param.vol import FlatVolSurface
from param.rrf import FlatRateCurve
from priceenv import PricingEnvironment
from util.enum.engine_enums import EngineType, ConvertibleBondMethod
from util.exceptions import ValidationError, PricingError


class TestConvertibleBondEngineSetup(unittest.TestCase):
    """Base class with common setup for engine tests."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a standard convertible bond
        self.cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,  # conversion price = 10
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )

        # Standard pricing environment (stock at 12, above conversion price)
        self.pricing_env = PricingEnvironment(
            valuation_date=datetime(2024, 6, 1),
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )

        # Standard parameters
        self.tree_params = ConvertibleBondTreeParams(num_steps=50)
        self.pde_params = ConvertibleBondPDEParams(
            num_space_steps=50, num_time_steps=100
        )


class TestBinomialEngine(TestConvertibleBondEngineSetup):
    """Tests for ConvertibleBondBinomialEngine."""

    def test_basic_pricing(self):
        """Test basic pricing returns positive value."""
        engine = ConvertibleBondBinomialEngine(
            self.pricing_env, self.tree_params
        )
        price = engine.price(self.cb)
        self.assertGreater(price, 0)

    def test_price_with_details(self):
        """Test detailed pricing results."""
        engine = ConvertibleBondBinomialEngine(
            self.pricing_env, self.tree_params
        )
        result = engine.price_with_details(self.cb)

        # Check all fields are present and sensible
        self.assertGreater(result.price, 0)
        self.assertGreater(result.dirty_price, result.price)
        self.assertGreaterEqual(result.conversion_probability, 0)
        self.assertLessEqual(result.conversion_probability, 1)

    def test_in_the_money_bond(self):
        """Test bond that's deep in the money."""
        # Stock at 20, well above conversion price of 10
        env = PricingEnvironment(
            valuation_date=datetime(2024, 6, 1),
            spot_quote=SpotQuote(spot=20.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        engine = ConvertibleBondBinomialEngine(env, self.tree_params)
        result = engine.price_with_details(self.cb)

        # Should be close to parity
        parity = self.cb.parity(20.0)  # 200
        self.assertGreater(result.dirty_price, parity * 0.9)
        # Conversion probability should be high
        self.assertGreater(result.conversion_probability, 0.5)

    def test_out_of_the_money_bond(self):
        """Test bond that's deep out of the money."""
        # Stock at 5, well below conversion price of 10
        env = PricingEnvironment(
            valuation_date=datetime(2024, 6, 1),
            spot_quote=SpotQuote(spot=5.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        engine = ConvertibleBondBinomialEngine(env, self.tree_params)
        result = engine.price_with_details(self.cb)

        # Should be close to face value
        self.assertLess(result.dirty_price, 150)  # Not too high
        self.assertGreater(result.dirty_price, 50)  # Not too low
        # Conversion probability should be lower
        self.assertLess(result.conversion_probability, 0.7)

    def test_delta_is_positive(self):
        """Test delta is positive for convertible bond."""
        engine = ConvertibleBondBinomialEngine(
            self.pricing_env, self.tree_params
        )
        delta = engine.calculate_delta(self.cb)
        # Delta should be positive (bond value increases with stock)
        self.assertGreater(delta, 0)

    def test_expired_bond_raises_error(self):
        """Test that expired bond raises PricingError."""
        env = PricingEnvironment(
            valuation_date=datetime(2030, 1, 1),  # After maturity
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        engine = ConvertibleBondBinomialEngine(env, self.tree_params)
        with self.assertRaises(PricingError):
            engine.price(self.cb)


class TestTrinomialEngine(TestConvertibleBondEngineSetup):
    """Tests for ConvertibleBondTrinomialEngine."""

    def test_basic_pricing(self):
        """Test basic pricing returns positive value."""
        engine = ConvertibleBondTrinomialEngine(
            self.pricing_env, self.tree_params
        )
        price = engine.price(self.cb)
        self.assertGreater(price, 0)

    def test_default_probability_positive(self):
        """Test default probability is positive when hazard rate > 0."""
        engine = ConvertibleBondTrinomialEngine(
            self.pricing_env, self.tree_params
        )
        result = engine.price_with_details(self.cb)

        # With hazard rate of 0.01, should have some default probability
        self.assertGreater(result.default_probability, 0)
        self.assertLess(result.default_probability, 1)

    def test_recovery_component(self):
        """Test recovery component is present."""
        engine = ConvertibleBondTrinomialEngine(
            self.pricing_env, self.tree_params
        )
        result = engine.price_with_details(self.cb)

        # Recovery component should be non-negative
        self.assertGreaterEqual(result.recovery_component, 0)

    def test_zero_hazard_rate(self):
        """Test with zero hazard rate (no default risk)."""
        cb_no_default = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
            hazard_rate=0.0,  # No default
            credit_spread=0.0,
        )
        engine = ConvertibleBondTrinomialEngine(
            self.pricing_env, self.tree_params
        )
        result = engine.price_with_details(cb_no_default)

        # Default probability should be near zero
        self.assertLess(result.default_probability, 0.01)


class TestJumpDiffusionEngine(TestConvertibleBondEngineSetup):
    """Tests for ConvertibleBondJumpDiffusionEngine."""

    def test_basic_pricing(self):
        """Test basic pricing returns positive value."""
        engine = ConvertibleBondJumpDiffusionEngine(
            self.pricing_env, self.pde_params
        )
        price = engine.price(self.cb)
        self.assertGreater(price, 0)

    def test_price_with_details(self):
        """Test detailed pricing results."""
        engine = ConvertibleBondJumpDiffusionEngine(
            self.pricing_env, self.pde_params
        )
        result = engine.price_with_details(self.cb)

        self.assertGreater(result.price, 0)
        self.assertGreater(result.dirty_price, 0)

    def test_delta_calculation(self):
        """Test delta is calculated correctly."""
        engine = ConvertibleBondJumpDiffusionEngine(
            self.pricing_env, self.pde_params
        )
        result = engine.price_with_details(self.cb)

        # Delta should be positive
        self.assertGreater(result.delta, 0)

    def test_coupons_increase_value_when_not_convertible(self):
        """Test coupon cashflows are included in the PDE solve."""
        valuation_date = datetime(2024, 1, 1)
        env = PricingEnvironment(
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot=10.0),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.03),
        )
        params = ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
        engine = ConvertibleBondJumpDiffusionEngine(env, params)

        bond_common = dict(
            issue_date=valuation_date,
            maturity_date=datetime(2026, 1, 1),
            face_value=100.0,
            conversion_ratio=1.0,  # conversion price = 100 (deep OTM at spot=10)
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )
        cb_no_coupon = ConvertibleBond(coupon_rate=0.0, **bond_common)
        cb_with_coupon = ConvertibleBond(coupon_rate=0.05, **bond_common)

        no_coupon = engine.price_with_details(cb_no_coupon).dirty_price
        with_coupon = engine.price_with_details(cb_with_coupon).dirty_price

        self.assertGreater(with_coupon, no_coupon)

    def test_conversion_probability_bounds(self):
        """Test conversion probability is in [0, 1]."""
        engine = ConvertibleBondJumpDiffusionEngine(
            self.pricing_env, self.pde_params
        )
        result = engine.price_with_details(self.cb)
        self.assertGreaterEqual(result.conversion_probability, 0.0)
        self.assertLessEqual(result.conversion_probability, 1.0)

    def test_conversion_probability_zero_when_conversion_never_allowed(self):
        """Test eventual conversion probability is zero if conversion is disabled."""
        valuation_date = datetime(2024, 6, 1)
        env = PricingEnvironment(
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        params = ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
        engine = ConvertibleBondJumpDiffusionEngine(env, params)

        cb_no_convert = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
            conversion_start_date=datetime(2024, 1, 1),
            conversion_end_date=datetime(2024, 5, 1),  # before valuation
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )

        result = engine.price_with_details(cb_no_convert)
        self.assertAlmostEqual(result.conversion_probability, 0.0, delta=1e-8)

    def test_conversion_probability_one_when_immediate_conversion_optimal(self):
        """Test eventual conversion probability is one if conversion is optimal now."""
        valuation_date = datetime(2024, 6, 1)
        env = PricingEnvironment(
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.03),
        )
        params = ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
        engine = ConvertibleBondJumpDiffusionEngine(env, params)

        cb_convert_now = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=1.0,
            coupon_rate=0.0,
            conversion_ratio=10.0,
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )

        result = engine.price_with_details(cb_convert_now)
        self.assertAlmostEqual(result.conversion_probability, 1.0, delta=1e-6)


class TestTFEngine(TestConvertibleBondEngineSetup):
    """Tests for ConvertibleBondTFEngine."""

    def test_basic_pricing(self):
        """Test basic pricing returns positive value."""
        engine = ConvertibleBondTFEngine(self.pricing_env, self.pde_params)
        price = engine.price(self.cb)
        self.assertGreater(price, 0)

    def test_decomposition(self):
        """Test TF decomposition produces valid components."""
        engine = ConvertibleBondTFEngine(self.pricing_env, self.pde_params)
        result = engine.price_with_details(self.cb)

        # Components should be non-negative
        self.assertGreaterEqual(result.equity_component, 0)
        self.assertGreaterEqual(result.bond_component, 0)

        # Sum should equal total
        total_from_components = result.equity_component + result.bond_component
        self.assertAlmostEqual(
            total_from_components, result.dirty_price, delta=1.0
        )

    def test_cocb(self):
        """Test COCB method returns bond component."""
        engine = ConvertibleBondTFEngine(self.pricing_env, self.pde_params)
        cocb = engine.get_cocb(self.cb)

        # COCB should be positive
        self.assertGreater(cocb, 0)

        # COCB should match bond component from detailed results
        result = engine.price_with_details(self.cb)
        self.assertAlmostEqual(cocb, result.bond_component, delta=0.1)

    def test_coupons_increase_bond_component(self):
        """Test coupon cashflows are included in the bond-component PDE."""
        valuation_date = datetime(2024, 1, 1)
        env = PricingEnvironment(
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot=10.0),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.03),
        )
        params = ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
        engine = ConvertibleBondTFEngine(env, params)

        bond_common = dict(
            issue_date=valuation_date,
            maturity_date=datetime(2026, 1, 1),
            face_value=100.0,
            conversion_ratio=1.0,  # conversion price = 100 (deep OTM at spot=10)
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )
        cb_no_coupon = ConvertibleBond(coupon_rate=0.0, **bond_common)
        cb_with_coupon = ConvertibleBond(coupon_rate=0.05, **bond_common)

        no_coupon = engine.price_with_details(cb_no_coupon).bond_component
        with_coupon = engine.price_with_details(cb_with_coupon).bond_component

        self.assertGreater(with_coupon, no_coupon)

    def test_conversion_probability_bounds(self):
        """Test conversion probability is in [0, 1]."""
        engine = ConvertibleBondTFEngine(self.pricing_env, self.pde_params)
        result = engine.price_with_details(self.cb)
        self.assertGreaterEqual(result.conversion_probability, 0.0)
        self.assertLessEqual(result.conversion_probability, 1.0)

    def test_conversion_probability_zero_when_conversion_never_allowed(self):
        """Test eventual conversion probability is zero if conversion is disabled."""
        valuation_date = datetime(2024, 6, 1)
        env = PricingEnvironment(
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        params = ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
        engine = ConvertibleBondTFEngine(env, params)

        cb_no_convert = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
            conversion_start_date=datetime(2024, 1, 1),
            conversion_end_date=datetime(2024, 5, 1),  # before valuation
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )

        result = engine.price_with_details(cb_no_convert)
        self.assertAlmostEqual(result.conversion_probability, 0.0, delta=1e-8)

    def test_conversion_probability_one_when_immediate_conversion_optimal(self):
        """Test eventual conversion probability is one if conversion is optimal now."""
        valuation_date = datetime(2024, 6, 1)
        env = PricingEnvironment(
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.03),
        )
        params = ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
        engine = ConvertibleBondTFEngine(env, params)

        cb_convert_now = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=1.0,
            coupon_rate=0.0,
            conversion_ratio=10.0,
            credit_spread=0.02,
            hazard_rate=0.01,
            recovery_rate=0.4,
        )

        result = engine.price_with_details(cb_convert_now)
        self.assertAlmostEqual(result.conversion_probability, 1.0, delta=1e-6)


class TestFacadeEngine(TestConvertibleBondEngineSetup):
    """Tests for ConvertibleBondEngine (facade)."""

    def test_default_method(self):
        """Test default method is used when none specified."""
        engine = ConvertibleBondEngine(self.pricing_env)
        result = engine.price_with_details(self.cb)

        # Should use binomial_gs by default
        self.assertEqual(result.method, "binomial_gs")

    def test_two_level_enum_tree(self):
        """Test two-level enum pattern for tree methods."""
        engine = ConvertibleBondEngine(
            self.pricing_env,
            method=EngineType.TREE(ConvertibleBondMethod.BINOMIAL_GS),
        )
        result = engine.price_with_details(self.cb)
        self.assertEqual(result.method, "binomial_gs")

    def test_two_level_enum_pde(self):
        """Test two-level enum pattern for PDE methods."""
        engine = ConvertibleBondEngine(
            self.pricing_env,
            method=EngineType.PDE(ConvertibleBondMethod.TF),
        )
        result = engine.price_with_details(self.cb)
        self.assertEqual(result.method, "tf")

    def test_direct_method_enum(self):
        """Test using method enum directly."""
        engine = ConvertibleBondEngine(
            self.pricing_env,
            method=ConvertibleBondMethod.TRINOMIAL_HW,
        )
        result = engine.price_with_details(self.cb)
        self.assertEqual(result.method, "trinomial_hw")

    def test_string_method(self):
        """Test using string method."""
        engine = ConvertibleBondEngine(
            self.pricing_env,
            method="jump_diffusion",
        )
        result = engine.price_with_details(self.cb)
        self.assertEqual(result.method, "jump_diffusion")

    def test_invalid_method_string(self):
        """Test invalid string raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBondEngine(
                self.pricing_env,
                method="invalid_method",
            )

    def test_method_mismatch_raises_error(self):
        """Test wrong engine type for method raises error."""
        with self.assertRaises(ValidationError):
            ConvertibleBondEngine(
                self.pricing_env,
                method=EngineType.TREE(ConvertibleBondMethod.TF),  # TF is PDE
            )

    def test_cocb_only_tf(self):
        """Test COCB only available with TF method."""
        engine = ConvertibleBondEngine(
            self.pricing_env,
            method=ConvertibleBondMethod.BINOMIAL_GS,
        )
        with self.assertRaises(ValidationError):
            engine.get_cocb(self.cb)


class TestMethodComparison(TestConvertibleBondEngineSetup):
    """Tests comparing different pricing methods."""

    def test_methods_give_similar_results(self):
        """Test that different methods give similar results."""
        methods = [
            ConvertibleBondMethod.BINOMIAL_GS,
            ConvertibleBondMethod.TRINOMIAL_HW,
            ConvertibleBondMethod.JUMP_DIFFUSION,
            ConvertibleBondMethod.TF,
        ]

        prices = []
        for method in methods:
            engine = ConvertibleBondEngine(
                self.pricing_env,
                method=method,
                tree_params=ConvertibleBondTreeParams(num_steps=100),
                pde_params=ConvertibleBondPDEParams(
                    num_space_steps=100, num_time_steps=200
                ),
            )
            prices.append(engine.price(self.cb))

        # All prices should be in a reasonable range
        min_price = min(prices)
        max_price = max(prices)

        # Prices should all be positive
        self.assertTrue(all(p > 0 for p in prices))

        # Prices should be within 20% of each other
        # (different models have different assumptions)
        range_ratio = (max_price - min_price) / min_price
        self.assertLess(range_ratio, 0.20)


class TestCallableBond(TestConvertibleBondEngineSetup):
    """Tests for callable convertible bonds."""

    def test_callable_bond_pricing(self):
        """Test pricing callable convertible bond."""
        cb_callable = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
            credit_spread=0.02,
            call_schedule=[
                CallScheduleEntry(
                    call_date=datetime(2026, 1, 1),
                    call_price=105.0,
                )
            ],
        )

        engine = ConvertibleBondEngine(
            self.pricing_env,
            method=ConvertibleBondMethod.BINOMIAL_GS,
        )

        # Callable bond should have lower price than non-callable
        price_callable = engine.price(cb_callable)
        price_non_callable = engine.price(self.cb)

        # Callable should be <= non-callable (issuer has right)
        self.assertLessEqual(price_callable, price_non_callable + 5)


class TestPuttableBond(TestConvertibleBondEngineSetup):
    """Tests for puttable convertible bonds."""

    def test_puttable_bond_pricing(self):
        """Test pricing puttable convertible bond."""
        cb_puttable = ConvertibleBond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            face_value=100.0,
            coupon_rate=0.02,
            conversion_ratio=10.0,
            credit_spread=0.02,
            put_schedule=[
                PutScheduleEntry(
                    put_date=datetime(2026, 1, 1),
                    put_price=95.0,
                )
            ],
        )

        engine = ConvertibleBondEngine(
            self.pricing_env,
            method=ConvertibleBondMethod.BINOMIAL_GS,
        )

        # Puttable bond should have higher price than non-puttable
        price_puttable = engine.price(cb_puttable)
        price_non_puttable = engine.price(self.cb)

        # Puttable should be >= non-puttable (holder has right)
        self.assertGreaterEqual(price_puttable, price_non_puttable - 5)


if __name__ == "__main__":
    unittest.main()
