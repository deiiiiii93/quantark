"""
Tests for Interest Rate sensitivity engine.
"""

import pytest
from unittest.mock import Mock, MagicMock

from simm.config import SIMMConfig
from simm.taxonomy import RiskClass, IRSubCurve, IR_TENORS
from simm.engines.risk_class.ir_engine import IRSensitivityEngine
from simm.sensitivity import IRDeltaSensitivity

# Mock FIPosition
class MockFIPosition:
    """Mock FIPosition for testing."""

    def __init__(self, underlying="USD", dv01=1000.0, position_id="pos1"):
        self.underlying = underlying
        self.dv01_value = dv01
        self.position_id = position_id

    def get_dv01(self, env):
        return self.dv01_value


class TestIRSensitivityEngine:
    """Test the IRSensitivityEngine class."""

    def test_init(self):
        """Test initialization."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)
        assert engine.config == config
        assert engine.risk_class == RiskClass.INTEREST_RATE

    def test_risk_class(self):
        """Test risk class property."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)
        assert engine.risk_class == RiskClass.INTEREST_RATE

    def test_calculate_delta_sensitivities_basic(self):
        """Test basic delta sensitivity calculation."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        positions = [
            MockFIPosition(underlying="USD", dv01=1000.0, position_id="pos1"),
        ]

        # Mock pricing environment
        env = Mock()
        pricing_environments = {"USD": env}

        sensitivities = engine.calculate_delta_sensitivities(positions, pricing_environments)

        # Check that sensitivities were created
        assert len(sensitivities) > 0

        # Check that all sensitivities have correct risk class
        for sens in sensitivities:
            assert sens.risk_class == RiskClass.INTEREST_RATE
            assert sens.margin_type.value == "Delta"

    def test_calculate_delta_sensitivities_with_zero_dv01(self):
        """Test that positions with zero DV01 are skipped."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        positions = [
            MockFIPosition(underlying="USD", dv01=0.0, position_id="pos1"),
        ]

        env = Mock()
        pricing_environments = {"USD": env}

        sensitivities = engine.calculate_delta_sensitivities(positions, pricing_environments)

        # Should return empty list for zero DV01
        assert len(sensitivities) == 0

    def test_determine_sub_curve(self):
        """Test sub-curve determination."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        position = MockFIPosition(underlying="USD")
        sub_curve = engine._determine_sub_curve(position)
        assert sub_curve == IRSubCurve.OIS

        # Test with EUR
        position_eur = MockFIPosition(underlying="EUR")
        sub_curve_eur = engine._determine_sub_curve(position_eur)
        assert sub_curve_eur == IRSubCurve.OIS

    def test_calculate_tenor_weights(self):
        """Test tenor weight calculation."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        position = MockFIPosition()
        weights = engine._calculate_tenor_weights(position)

        # Should return equal weights
        assert len(weights) == len(IR_TENORS)
        assert sum(weights) == pytest.approx(1.0, abs=1e-10)

        # All weights should be equal (simplified approach)
        for i in range(len(weights) - 1):
            assert weights[i] == pytest.approx(weights[i + 1], abs=1e-10)

    def test_classify_currency_volatility(self):
        """Test currency volatility classification."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        # Test low volatility currencies
        assert engine._classify_currency_volatility("USD") == "Low"
        assert engine._classify_currency_volatility("EUR") == "Low"
        assert engine._classify_currency_volatility("GBP") == "Low"
        assert engine._classify_currency_volatility("CHF") == "Low"
        assert engine._classify_currency_volatility("JPY") == "Low"

        # Test regular volatility currencies
        assert engine._classify_currency_volatility("CAD") == "Regular"
        assert engine._classify_currency_volatility("AUD") == "Regular"
        assert engine._classify_currency_volatility("NZD") == "Regular"

        # Test high volatility (emerging markets)
        assert engine._classify_currency_volatility("BRL") == "High"
        assert engine._classify_currency_volatility("TRY") == "High"
        assert engine._classify_currency_volatility("MXN") == "High"

    def test_classify_to_buckets(self):
        """Test bucket classification."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        positions = [
            MockFIPosition(underlying="USD", position_id="pos1"),
            MockFIPosition(underlying="EUR", position_id="pos2"),
        ]

        env = Mock()
        pricing_environments = {
            "USD": env,
            "EUR": env,
        }

        classification = engine.classify_to_buckets(positions, pricing_environments)

        assert "pos1" in classification
        assert "pos2" in classification
        assert classification["pos1"]["bucket"] == "USD"
        assert classification["pos1"]["volatility_class"] == "Low"
        assert classification["pos2"]["bucket"] == "EUR"
        assert classification["pos2"]["volatility_class"] == "Low"

    def test_calculate_vega_sensitivities(self):
        """Test vega sensitivity calculation."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        positions = [MockFIPosition(underlying="USD")]
        env = Mock()
        pricing_environments = {"USD": env}

        sensitivities = engine.calculate_vega_sensitivities(positions, pricing_environments)

        # Currently returns empty list (not implemented yet)
        assert len(sensitivities) == 0

    def test_calculate_curvature_sensitivities(self):
        """Test curvature sensitivity calculation."""
        config = SIMMConfig()
        engine = IRSensitivityEngine(config)

        positions = [MockFIPosition(underlying="USD")]
        env = Mock()
        pricing_environments = {"USD": env}

        sensitivities = engine.calculate_curvature_sensitivities(positions, pricing_environments)

        # Currently returns empty list (not implemented yet)
        assert len(sensitivities) == 0

    def test_calculate_sensitivities_full(self):
        """Test full sensitivity calculation with config."""
        from simm.sensitivity import SensitivityCollection

        config = SIMMConfig(calculate_delta=True, calculate_vega=False)

        class MockIRSensitivityEngine(IRSensitivityEngine):
            def calculate_delta_sensitivities(self, positions, envs):
                return [
                    IRDeltaSensitivity(
                        trade_id="test1",
                        amount=100.0,
                        currency="USD",
                        tenor=1.0,
                    )
                ]

        engine = MockIRSensitivityEngine(config)
        positions = []
        envs = {}

        result = engine.calculate_sensitivities(positions, envs)

        assert isinstance(result, SensitivityCollection)
        assert len(result.sensitivities) == 1
