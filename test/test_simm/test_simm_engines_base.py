"""
Tests for SIMM sensitivity engine base classes.
"""

import pytest
from quantark.simm.config import SIMMConfig
from quantark.simm.taxonomy import RiskClass, MarginType
from quantark.simm.engines.base import BaseSensitivityEngine, SensitivityEngine


class MockEngine(BaseSensitivityEngine):
    """Mock engine for testing."""

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.EQUITY


class TestBaseSensitivityEngine:
    """Test the BaseSensitivityEngine class."""

    def test_init(self):
        """Test initialization."""
        config = SIMMConfig()
        engine = MockEngine(config)
        assert engine.config == config
        assert engine.risk_class == RiskClass.EQUITY

    def test_calculate_delta_sensitivities_default(self):
        """Test default delta calculation returns empty list."""
        config = SIMMConfig()
        engine = MockEngine(config)
        positions = []
        envs = {}
        result = engine.calculate_delta_sensitivities(positions, envs)
        assert result == []

    def test_calculate_vega_sensitivities_default(self):
        """Test default vega calculation returns empty list."""
        config = SIMMConfig()
        engine = MockEngine(config)
        positions = []
        envs = {}
        result = engine.calculate_vega_sensitivities(positions, envs)
        assert result == []

    def test_calculate_curvature_sensitivities_default(self):
        """Test default curvature calculation returns empty list."""
        config = SIMMConfig()
        engine = MockEngine(config)
        positions = []
        envs = {}
        result = engine.calculate_curvature_sensitivities(positions, envs)
        assert result == []

    def test_calculate_base_corr_sensitivities_default(self):
        """Test default base correlation calculation returns empty list."""
        config = SIMMConfig()
        engine = MockEngine(config)
        positions = []
        envs = {}
        result = engine.calculate_base_corr_sensitivities(positions, envs)
        assert result == []

    def test_classify_to_buckets_default(self):
        """Test default bucket classification returns empty dict."""
        config = SIMMConfig()
        engine = MockEngine(config)
        positions = []
        envs = {}
        result = engine.classify_to_buckets(positions, envs)
        assert result == {}

    def test_calculate_sensitivities_with_delta_enabled(self):
        """Test full sensitivity calculation with delta enabled."""
        from quantark.simm.sensitivity import SensitivityCollection

        config = SIMMConfig(calculate_delta=True, calculate_vega=False)

        class TestEngineWithDelta(BaseSensitivityEngine):
            @property
            def risk_class(self) -> RiskClass:
                return RiskClass.EQUITY

            def calculate_delta_sensitivities(self, positions, envs):
                from quantark.simm.sensitivity import EquityDeltaSensitivity
                return [
                    EquityDeltaSensitivity(
                        trade_id="test1",
                        amount=100.0,
                        issuer="AAPL",
                        bucket_number=5,
                    )
                ]

        engine = TestEngineWithDelta(config)
        result = engine.calculate_sensitivities([], {})

        assert isinstance(result, SensitivityCollection)
        assert len(result.sensitivities) == 1
        assert result.sensitivities[0].trade_id == "test1"


class TestSensitivityEngineProtocol:
    """Test the SensitivityEngine protocol."""

    def test_protocol_exists(self):
        """Test that the protocol is defined."""
        assert hasattr(SensitivityEngine, "__call__")

    def test_mock_engine_implements_protocol(self):
        """Test that MockEngine implements the protocol."""
        engine = MockEngine(SIMMConfig())
        # Protocol check - should not raise an error
        assert isinstance(engine, SensitivityEngine)
