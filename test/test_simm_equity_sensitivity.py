"""
Tests for Equity sensitivity engine.
"""

import pytest
from unittest.mock import Mock, MagicMock

from simm.config import SIMMConfig
from simm.taxonomy import RiskClass
from simm.engines.risk_class.equity_engine import EquitySensitivityEngine
from simm.sensitivity import EquityDeltaSensitivity

# Mock EquityPosition
class MockEquityPosition:
    """Mock EquityPosition for testing."""

    def __init__(self, underlying="AAPL", quantity=100, position_id="pos1", delta=0.5, vega=0.3):
        self.underlying = underlying
        self.quantity = quantity
        self.position_id = position_id
        self._delta = delta
        self._vega = vega
        self.product = Mock()
        # Set default maturity
        self.product.get_maturity = Mock(return_value=1)

    def get_greeks(self, env, greeks_calculator):
        return {
            "delta": self._delta,
            "vega": self._vega,
            "gamma": 0.1,
            "theta": -0.05,
        }


class TestEquitySensitivityEngine:
    """Test the EquitySensitivityEngine class."""

    def test_init(self):
        """Test initialization."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)
        assert engine.config == config
        assert engine.risk_class == RiskClass.EQUITY
        assert engine.greeks_calculator is not None

    def test_init_with_custom_greeks_calculator(self):
        """Test initialization with custom Greeks calculator."""
        from asset.equity.riskmeasures import GreeksCalculator

        config = SIMMConfig()
        custom_calc = GreeksCalculator()
        engine = EquitySensitivityEngine(config, custom_calc)
        assert engine.greeks_calculator == custom_calc

    def test_risk_class(self):
        """Test risk class property."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)
        assert engine.risk_class == RiskClass.EQUITY

    def test_calculate_delta_sensitivities_basic(self):
        """Test basic delta sensitivity calculation."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        positions = [
            MockEquityPosition(
                underlying="AAPL",
                quantity=100,
                position_id="pos1",
                delta=0.5,
            ),
        ]

        env = Mock()
        pricing_environments = {"AAPL": env}

        sensitivities = engine.calculate_delta_sensitivities(positions, pricing_environments)

        # Check that sensitivities were created
        assert len(sensitivities) > 0

        # Check that all sensitivities have correct risk class
        for sens in sensitivities:
            assert sens.risk_class == RiskClass.EQUITY
            assert sens.margin_type.value == "Delta"
            # Should be scaled by quantity
            assert abs(sens.amount - 50.0) < 1e-10  # 0.5 * 100

    def test_calculate_delta_sensitivities_with_zero_delta(self):
        """Test that positions with zero delta are skipped."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        positions = [
            MockEquityPosition(
                underlying="AAPL",
                quantity=100,
                position_id="pos1",
                delta=0.0,
            ),
        ]

        env = Mock()
        pricing_environments = {"AAPL": env}

        sensitivities = engine.calculate_delta_sensitivities(positions, pricing_environments)

        # Should return empty list for zero delta
        assert len(sensitivities) == 0

    def test_calculate_vega_sensitivities_basic(self):
        """Test basic vega sensitivity calculation."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        positions = [
            MockEquityPosition(
                underlying="AAPL",
                quantity=100,
                position_id="pos1",
                vega=0.3,
            ),
        ]

        env = Mock()
        pricing_environments = {"AAPL": env}

        sensitivities = engine.calculate_vega_sensitivities(positions, pricing_environments)

        # Check that sensitivities were created
        assert len(sensitivities) > 0

        # Check that all sensitivities have correct risk class
        for sens in sensitivities:
            assert sens.risk_class == RiskClass.EQUITY
            assert sens.margin_type.value == "Vega"
            # Should be scaled by quantity
            assert abs(sens.amount - 30.0) < 1e-10  # 0.3 * 100

    def test_calculate_vega_sensitivities_with_zero_vega(self):
        """Test that positions with zero vega are skipped."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        positions = [
            MockEquityPosition(
                underlying="AAPL",
                quantity=100,
                position_id="pos1",
                vega=0.0,
            ),
        ]

        env = Mock()
        pricing_environments = {"AAPL": env}

        sensitivities = engine.calculate_vega_sensitivities(positions, pricing_environments)

        # Should return empty list for zero vega
        assert len(sensitivities) == 0

    def test_classify_equity_bucket(self):
        """Test equity bucket classification."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        # Test various equity types
        bucket_tech = engine._classify_equity_bucket("AAPL")
        assert bucket_tech == 5  # Technology bucket

        bucket_spy = engine._classify_equity_bucket("SPY")
        assert bucket_spy == 11  # ETF/Index bucket

        bucket_us = engine._classify_equity_bucket("US")
        assert bucket_us == 3  # North America bucket

        # Test unknown issuer (should default to bucket 8)
        bucket_unknown = engine._classify_equity_bucket("UNKNOWN_ISSUER")
        assert bucket_unknown == 8  # Default bucket

    def test_get_option_expiry_tenor(self):
        """Test option expiry tenor extraction."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        position = MockEquityPosition(underlying="AAPL")

        # Mock product with get_maturity
        position.product.get_maturity = Mock(return_value=2)
        tenor = engine._get_option_expiry_tenor(position)
        assert tenor == 2.0

        # Test default (product without get_maturity)
        del position.product.get_maturity
        tenor = engine._get_option_expiry_tenor(position)
        assert tenor == 1.0  # Default

    def test_classify_to_buckets(self):
        """Test bucket classification."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        positions = [
            MockEquityPosition(underlying="AAPL", position_id="pos1"),
            MockEquityPosition(underlying="SPY", position_id="pos2"),
        ]

        env = Mock()
        pricing_environments = {
            "AAPL": env,
            "SPY": env,
        }

        classification = engine.classify_to_buckets(positions, pricing_environments)

        assert "pos1" in classification
        assert "pos2" in classification
        assert classification["pos1"]["bucket"] == 5  # Technology
        assert classification["pos1"]["issuer"] == "AAPL"
        assert classification["pos2"]["bucket"] == 11  # ETF
        assert classification["pos2"]["issuer"] == "SPY"

    def test_is_equity_spot_or_futures(self):
        """Test detection of delta-one instruments."""
        config = SIMMConfig()
        engine = EquitySensitivityEngine(config)

        # Spot position
        position_spot = MockEquityPosition(underlying="AAPL")
        position_spot.product.__class__.__name__ = "SpotProduct"
        assert engine._is_equity_spot_or_futures(position_spot) is True

        # Futures position
        position_future = MockEquityPosition(underlying="ES")
        position_future.product.__class__.__name__ = "FuturesContract"
        assert engine._is_equity_spot_or_futures(position_future) is True

        # Option position (should not be delta-one)
        position_option = MockEquityPosition(underlying="AAPL")
        position_option.product.__class__.__name__ = "EuropeanVanillaOption"
        # Should be False since it has strike attribute
        assert engine._is_equity_spot_or_futures(position_option) is False

    def test_calculate_sensitivities_full(self):
        """Test full sensitivity calculation with config."""
        from simm.sensitivity import SensitivityCollection

        config = SIMMConfig(calculate_delta=True, calculate_vega=False)

        class MockEquitySensitivityEngine(EquitySensitivityEngine):
            def calculate_delta_sensitivities(self, positions, envs):
                return [
                    EquityDeltaSensitivity(
                        trade_id="test1",
                        amount=100.0,
                        issuer="AAPL",
                        bucket_number=5,
                    )
                ]

        engine = MockEquitySensitivityEngine(config)
        positions = []
        envs = {}

        result = engine.calculate_sensitivities(positions, envs)

        assert isinstance(result, SensitivityCollection)
        assert len(result.sensitivities) == 1
