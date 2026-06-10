"""
Unit tests for basis stress functionality in QuantArk.

This module tests the new basis stress features including:
- ScenarioBuilder basis_stress method
- PathBuilder basis_stress method
- BasisDividendRelationshipMode enum and functionality
- Futures basis calculation methods
"""

import unittest
import math
from decimal import Decimal
from datetime import datetime

from quantark.stresstest.scenario.scenario_builder import ScenarioBuilder
from quantark.stresstest.scenario.scenario import Stress
from quantark.stresstest.stress.stress_types import (
    StressType,
    StressLevel,
    BasisDividendRelationshipMode,
)
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.dynamicscenario.path.path_builder import PathBuilder
from quantark.asset.equity.product.deltaone.futures import Futures
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.div.dividend_yield import (
    ContinuousDividendYield,
    TermStructureDividendYield,
)
from quantark.util.exceptions import ValidationError


class TestBasisStress(unittest.TestCase):
    """Test suite for basis stress functionality."""

    def test_scenario_builder_basis_stress_exists(self):
        """Test that ScenarioBuilder has basis_stress method."""
        builder = ScenarioBuilder()

        # Check that the method exists
        self.assertTrue(hasattr(builder, 'basis_stress'))

        # Test basic usage doesn't throw error
        try:
            result = builder.basis_stress(0.05)
            self.assertIsInstance(result, ScenarioBuilder)
        except Exception as e:
            self.fail(f"basis_stress method should not raise exception: {e}")

    def test_path_builder_basis_stress_exists(self):
        """Test that PathBuilder has basis_stress method."""
        builder = PathBuilder(num_days=3, name="TestPath")

        # Check that the method exists
        self.assertTrue(hasattr(builder, 'basis_stress'))

        # Test basic usage doesn't throw error
        try:
            result = builder.basis_stress(0.01, stress_type=StressType.ABSOLUTE, underlying="TEST")
            self.assertIsInstance(result, PathBuilder)
        except Exception as e:
            self.fail(f"basis_stress method should not raise exception: {e}")

    def test_path_builder_basis_values_exists(self):
        """Test that PathBuilder has basis_values method."""
        builder = PathBuilder(num_days=3, name="TestPath")

        # Check that the method exists
        self.assertTrue(hasattr(builder, 'basis_values'))

        # Test basic usage doesn't throw error
        try:
            result = builder.basis_values([0.01, 0.02, 0.03], underlying="TEST")
            self.assertIsInstance(result, PathBuilder)
        except Exception as e:
            self.fail(f"basis_values method should not raise exception: {e}")

    def test_stress_applicator_basis_adapter_registered(self):
        """Test that StressApplicator recognizes basis stress."""
        # Check that the basis stress adapter is registered
        adapters = StressApplicator._parameter_adapters
        self.assertIn("basis", adapters)

        # Check that it's callable
        adapter_func = adapters["basis"]
        self.assertTrue(callable(adapter_func))

    def test_basis_dividend_relationship_mode_enum(self):
        """Test the BasisDividendRelationshipMode enum."""
        # Test enum values
        self.assertEqual(BasisDividendRelationshipMode.INDEPENDENT.value, "independent")
        self.assertEqual(BasisDividendRelationshipMode.AUTO_ADJUST_DIVIDEND.value, "auto_adjust_dividend")
        self.assertEqual(BasisDividendRelationshipMode.AUTO_ADJUST_BASIS.value, "auto_adjust_basis")
        self.assertEqual(BasisDividendRelationshipMode.SYNCHRONIZED.value, "synchronized")

        # Test from_string method
        mode = BasisDividendRelationshipMode.from_string("auto_adjust_dividend")
        self.assertEqual(mode, BasisDividendRelationshipMode.AUTO_ADJUST_DIVIDEND)

        # Test invalid string raises error
        with self.assertRaises(ValueError):
            BasisDividendRelationshipMode.from_string("invalid_mode")

    def test_scenario_builder_basis_stress_with_enum_relationship_mode(self):
        """Test ScenarioBuilder basis_stress with enum relationship_mode."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")
        builder.basis_stress(
            0.05,
            relationship_mode=BasisDividendRelationshipMode.AUTO_ADJUST_DIVIDEND
        )
        builder.spot_stress(0.01)  # Add another stress so scenario is valid

        scenario = builder.build()

        # Check that we have at least the basis stress
        self.assertGreaterEqual(len(scenario.stresses), 1)

        # Check that basis stress has the relationship mode in metadata
        basis_stress = next((s for s in scenario.stresses if s.parameter == "basis"), None)
        self.assertIsNotNone(basis_stress)
        self.assertEqual(basis_stress.metadata.get("relationship_mode"), "auto_adjust_dividend")

    def test_scenario_builder_basis_stress_with_string_relationship_mode(self):
        """Test ScenarioBuilder basis_stress with string relationship_mode."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")
        builder.basis_stress(0.05, relationship_mode="auto_adjust_dividend")
        builder.spot_stress(0.01)  # Add another stress so scenario is valid

        scenario = builder.build()

        # Check that basis stress has the relationship mode in metadata
        basis_stress = next((s for s in scenario.stresses if s.parameter == "basis"), None)
        self.assertIsNotNone(basis_stress)
        self.assertEqual(basis_stress.metadata.get("relationship_mode"), "auto_adjust_dividend")

    def test_scenario_builder_invalid_relationship_mode_raises_error(self):
        """Test that invalid relationship_mode raises ValueError."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")

        # Invalid relationship mode should raise error
        with self.assertRaises(ValueError):
            builder.basis_stress(0.05, relationship_mode="invalid_mode")

    def test_scenario_builder_auto_adjust_dividend_no_auto_generation(self):
        """Test that auto_adjust_dividend mode does not auto-generate a dividend stress."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")
        builder.basis_stress(
            0.05,
            relationship_mode=BasisDividendRelationshipMode.AUTO_ADJUST_DIVIDEND
        )

        scenario = builder.build()

        # Check that we only have the basis stress
        basis_stress = next((s for s in scenario.stresses if s.parameter == "basis"), None)
        div_stress = next((s for s in scenario.stresses if s.parameter == "dividend_yield"), None)

        self.assertIsNotNone(basis_stress, "Basis stress should exist")
        self.assertIsNone(div_stress, "Dividend stress should not be auto-generated")

    def test_scenario_builder_synchronized_mode_no_auto_generation(self):
        """Test that synchronized mode does not auto-generate a dividend stress."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")
        builder.basis_stress(
            0.05,
            relationship_mode=BasisDividendRelationshipMode.SYNCHRONIZED
        )

        scenario = builder.build()

        # Check that we only have basis stress
        basis_stress = next((s for s in scenario.stresses if s.parameter == "basis"), None)
        div_stress = next((s for s in scenario.stresses if s.parameter == "dividend_yield"), None)

        self.assertIsNotNone(basis_stress, "Basis stress should exist")
        self.assertIsNone(div_stress, "Dividend stress should not be auto-generated")

    def test_scenario_builder_auto_adjust_basis_no_auto_generation(self):
        """Test that auto_adjust_basis mode on dividend stress does not auto-generate a basis stress."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")
        builder.div_yield_stress(
            0.03,
            relationship_mode=BasisDividendRelationshipMode.AUTO_ADJUST_BASIS
        )

        scenario = builder.build()

        # Check that we only have dividend stress
        div_stress = next((s for s in scenario.stresses if s.parameter == "dividend_yield"), None)
        basis_stress = next((s for s in scenario.stresses if s.parameter == "basis"), None)

        self.assertIsNotNone(div_stress, "Dividend stress should exist")
        self.assertIsNone(basis_stress, "Basis stress should not be auto-generated")

    def test_scenario_builder_independent_mode_no_auto_generation(self):
        """Test that independent mode does not auto-generate stresses."""
        builder = ScenarioBuilder()
        builder.name("Test Scenario")
        builder.basis_stress(
            0.05,
            relationship_mode=BasisDividendRelationshipMode.INDEPENDENT
        )

        scenario = builder.build()

        # Check that we only have basis stress
        basis_stress = next((s for s in scenario.stresses if s.parameter == "basis"), None)
        div_stress = next((s for s in scenario.stresses if s.parameter == "dividend_yield"), None)

        self.assertIsNotNone(basis_stress, "Basis stress should exist")
        self.assertIsNone(div_stress, "Dividend stress should NOT be auto-generated")


class TestDividendStressClamping(unittest.TestCase):
    """Test dividend yield clamping behavior in stress applicator."""

    def _make_env(self, div_yield):
        return PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.02),
            valuation_date=datetime(2025, 1, 1),
            div_yield=div_yield,
        )

    def test_flat_dividend_clamps_to_zero(self):
        env = self._make_env(ContinuousDividendYield(div_yield=0.01))
        stress = Stress(
            parameter="dividend_yield",
            stress_type=StressType.ABSOLUTE,
            stress_value=-0.02,
            level=StressLevel.PORTFOLIO,
        )

        StressApplicator._stress_dividend(env, stress)
        self.assertIsInstance(env.div_yield, ContinuousDividendYield)
        self.assertEqual(env.div_yield.div_yield, 0.0)

    def test_term_structure_dividend_clamps_to_zero(self):
        env = self._make_env(
            TermStructureDividendYield(times=[0.5, 1.0], yields=[0.01, 0.02])
        )
        stress = Stress(
            parameter="dividend_yield",
            stress_type=StressType.ABSOLUTE,
            stress_value=-0.05,
            level=StressLevel.PORTFOLIO,
        )

        StressApplicator._stress_dividend(env, stress)
        self.assertIsInstance(env.div_yield, TermStructureDividendYield)
        self.assertEqual(env.div_yield.yields, [0.0, 0.0])


class TestFuturesBasisCalculator(unittest.TestCase):
    """Test suite for Futures basis calculation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.future = Futures(
            underlying="ES",
            multiplier=50.0,
            maturity=0.25,  # 3 months
            basis=2.5,
        )

    def test_calculate_implied_basis(self):
        """Test implied basis calculation from observed futures price."""
        spot = 4480.0
        rate = 0.05
        div_yield = 0.015
        T = 0.25
        observed_futures = 4500.0

        # Expected theoretical forward
        theoretical_forward = spot * math.exp((rate - div_yield) * T)

        # Expected basis
        expected_basis = observed_futures - theoretical_forward

        implied_basis = self.future.calculate_implied_basis(
            spot, rate, div_yield, T, observed_futures
        )

        self.assertAlmostEqual(implied_basis, expected_basis, places=4)

    def test_calculate_annualized_basis_from_futures_price(self):
        """Test annualized basis calculation from futures price."""
        spot = 4480.0
        rate = 0.05
        div_yield = 0.015
        T = 0.25
        observed_futures = 4500.0

        # Calculate annualized basis: b = (1/T) * ln(F/S) - r + d
        expected_annualized = (math.log(observed_futures / spot) / T) - rate + div_yield

        annualized = self.future.calculate_annualized_basis(
            spot, rate, div_yield, T, observed_futures
        )

        self.assertAlmostEqual(annualized, expected_annualized, places=6)

    def test_calculate_annualized_basis_from_stored_basis(self):
        """Test annualized basis calculation from stored basis attribute."""
        spot = 4480.0
        rate = 0.05
        div_yield = 0.015
        T = 0.25

        # Use stored basis (2.5)
        # For small basis/S: b_annual ≈ basis / (spot * T)
        expected_annualized = 2.5 / (spot * T)

        annualized = self.future.calculate_annualized_basis(
            spot, rate, div_yield, T
        )

        self.assertAlmostEqual(annualized, expected_annualized, places=6)

    def test_calculate_basis_in_bps(self):
        """Test basis calculation in basis points."""
        spot = 4480.0
        rate = 0.05
        div_yield = 0.015
        T = 0.25
        observed_futures = 4500.0

        bps = self.future.calculate_basis_in_bps(
            spot, rate, div_yield, T, observed_futures
        )

        # Convert annualized basis to bps
        annualized = (math.log(observed_futures / spot) / T) - rate + div_yield
        expected_bps = annualized * 10000

        self.assertAlmostEqual(bps, expected_bps, places=2)

    def test_get_implied_dividend_from_basis(self):
        """Test implied dividend yield calculation."""
        spot = 4480.0
        rate = 0.05
        T = 0.25
        observed_futures = 4500.0

        # d = r - (1/T) * ln(F/S)
        expected_div = rate - (math.log(observed_futures / spot) / T)

        implied_div = self.future.get_implied_dividend_from_basis(
            spot, rate, T, observed_futures
        )

        self.assertAlmostEqual(implied_div, expected_div, places=6)

    def test_basis_calculation_validates_inputs(self):
        """Test that basis calculation validates inputs."""
        with self.assertRaises(ValidationError):
            self.future.calculate_implied_basis(
                spot=-100,  # Invalid: negative spot
                rate=0.05,
                div_yield=0.015,
                time_to_maturity=0.25,
                observed_futures_price=4500.0
            )

        with self.assertRaises(ValidationError):
            self.future.calculate_annualized_basis(
                spot=4480.0,
                rate=0.05,
                div_yield=0.015,
                time_to_maturity=0,  # Invalid: zero T
            )

        with self.assertRaises(ValidationError):
            self.future.get_implied_dividend_from_basis(
                spot=4480.0,
                rate=0.05,
                time_to_maturity=-0.25,  # Invalid: negative T
                observed_futures_price=4500.0
            )


if __name__ == '__main__':
    unittest.main()
