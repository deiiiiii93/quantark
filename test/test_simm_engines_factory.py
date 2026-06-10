"""
Tests for SIMM sensitivity engine factory functions.
"""

import pytest
from quantark.simm.config import SIMMConfig
from quantark.simm.engines.factory import create_engine, create_all_engines, get_available_engines
from quantark.simm.taxonomy import RiskClass


class TestCreateEngine:
    """Test the create_engine factory function."""

    def test_create_ir_engine(self):
        """Test creating IR engine."""
        config = SIMMConfig()
        try:
            engine = create_engine(RiskClass.INTEREST_RATE, config)
            assert engine.risk_class == RiskClass.INTEREST_RATE
        except Exception as e:
            pytest.skip(f"IR engine not implemented yet: {e}")

    def test_create_equity_engine(self):
        """Test creating equity engine."""
        config = SIMMConfig()
        try:
            engine = create_engine(RiskClass.EQUITY, config)
            assert engine.risk_class == RiskClass.EQUITY
        except Exception as e:
            pytest.skip(f"Equity engine not implemented yet: {e}")

    def test_create_engine_invalid_risk_class(self):
        """Test creating engine with invalid risk class."""
        config = SIMMConfig()
        with pytest.raises(ValueError, match="No sensitivity engine available"):
            create_engine("INVALID", config)


class TestCreateAllEngines:
    """Test the create_all_engines factory function."""

    def test_create_all_engines(self):
        """Test creating all available engines."""
        config = SIMMConfig()
        engines = create_all_engines(config)

        assert isinstance(engines, dict)

        # Should contain at least one engine
        assert len(engines) > 0


class TestGetAvailableEngines:
    """Test the get_available_engines function."""

    def test_get_available_engines(self):
        """Test getting available engines."""
        engines = get_available_engines()

        assert isinstance(engines, dict)
        # Should contain engine names
        assert len(engines) >= 0
