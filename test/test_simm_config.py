"""
Tests for SIMM Config Module.

Tests for SIMMConfig and SIMMVersion.
"""
import pytest

from quantark.simm.config import SIMMConfig, SIMMVersion
from quantark.util.exceptions import ValidationError


class TestSIMMVersion:
    """Tests for SIMMVersion enum."""
    
    def test_versions_defined(self):
        """SIMM versions should be defined."""
        assert SIMMVersion.V2_5.value == "2.5"
        assert SIMMVersion.V2_6.value == "2.6"
    
    def test_string_representation(self):
        """String representation should return version number."""
        assert str(SIMMVersion.V2_6) == "2.6"


class TestSIMMConfig:
    """Tests for SIMMConfig dataclass."""
    
    def test_default_values(self):
        """Default config should have sensible values."""
        config = SIMMConfig()
        
        assert config.version == SIMMVersion.V2_6
        assert config.calculation_currency == "USD"
        assert config.calculate_delta is True
        assert config.calculate_vega is True
        assert config.calculate_curvature is True
        assert config.calculate_base_corr is True
        assert config.ms_rates_fx == 1.0
        assert config.addon_fixed == 0.0
        assert config.include_attribution is True
    
    def test_custom_values(self):
        """Config should accept custom values."""
        config = SIMMConfig(
            calculation_currency="EUR",
            calculate_vega=False,
            ms_credit=1.5,
            addon_fixed=1_000_000,
        )
        
        assert config.version == SIMMVersion.V2_6
        assert config.calculation_currency == "EUR"
        assert config.calculate_vega is False
        assert config.ms_credit == 1.5
        assert config.addon_fixed == 1_000_000
    
    def test_invalid_currency(self):
        """Invalid currency should raise ValidationError."""
        with pytest.raises(ValidationError):
            SIMMConfig(calculation_currency="INVALID")
        
        with pytest.raises(ValidationError):
            SIMMConfig(calculation_currency="US")
    
    def test_invalid_multiplier(self):
        """Non-positive multiplier should raise ValidationError."""
        with pytest.raises(ValidationError):
            SIMMConfig(ms_rates_fx=0.0)
        
        with pytest.raises(ValidationError):
            SIMMConfig(ms_credit=-1.0)
    
    def test_invalid_addon(self):
        """Negative addon should raise ValidationError."""
        with pytest.raises(ValidationError):
            SIMMConfig(addon_fixed=-100)
        
        with pytest.raises(ValidationError):
            SIMMConfig(addon_factors={"factor1": -0.1})
    
    def test_get_product_class_multiplier(self):
        """get_product_class_multiplier should return correct values."""
        config = SIMMConfig(
            ms_rates_fx=1.0,
            ms_credit=1.2,
            ms_equity=0.9,
            ms_commodity=1.1,
        )
        
        assert config.get_product_class_multiplier("RatesFX") == 1.0
        assert config.get_product_class_multiplier("Credit") == 1.2
        assert config.get_product_class_multiplier("Equity") == 0.9
        assert config.get_product_class_multiplier("Commodity") == 1.1
    
    def test_get_product_class_multiplier_unknown(self):
        """Unknown product class should raise ValueError."""
        config = SIMMConfig()
        
        with pytest.raises(ValueError):
            config.get_product_class_multiplier("Unknown")
    
    def test_with_version(self):
        """with_version should create new config with different version."""
        original = SIMMConfig(
            calculation_currency="EUR",
            ms_credit=1.5,
        )
        
        with pytest.raises(ValidationError, match="only v2.6"):
            original.with_version(SIMMVersion.V2_5)
        # Original unchanged
        assert original.version == SIMMVersion.V2_6

    def test_rejects_v2_5(self):
        with pytest.raises(ValidationError, match="only v2.6"):
            SIMMConfig(version=SIMMVersion.V2_5)
    
    def test_addon_factors(self):
        """Addon factors should work correctly."""
        config = SIMMConfig(
            addon_factors={
                "regulatory": 0.05,
                "concentration": 0.02,
            }
        )
        
        assert config.addon_factors["regulatory"] == 0.05
        assert config.addon_factors["concentration"] == 0.02
