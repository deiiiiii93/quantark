"""
Unit tests for BumpConfig.
"""

import pytest
from quantark.asset.equity.param import BumpConfig
from quantark.util.exceptions import ValidationError


class TestBumpConfig:
    """Test BumpConfig validation and defaults."""

    def test_default_values(self):
        """Test that default values match industry standards."""
        config = BumpConfig()
        assert config.spot_bump == 0.01
        assert config.vol_bump == 0.01
        assert config.time_bump_days == 1
        assert config.time_bump_mode == "auto"
        assert config.rate_bump == 0.0001
        assert config.div_bump == 0.0001

    def test_from_tolerance(self):
        """Test creating from Tolerance constants."""
        config = BumpConfig.from_tolerance()
        assert config.spot_bump == 0.01  # Tolerance.BUMP_SPOT
        assert config.vol_bump == 0.01  # Tolerance.BUMP_VOL
        assert config.rate_bump == 0.0001  # Tolerance.BUMP_RATE
        assert config.div_bump == 0.0001  # Same as rate bump

    def test_validation_negative_spot_bump(self):
        """Test that negative spot_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="spot_bump must be positive"):
            BumpConfig(spot_bump=-0.01)

    def test_validation_large_spot_bump(self):
        """Test that too large spot_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="spot_bump seems too large"):
            BumpConfig(spot_bump=0.15)  # 15%

    def test_validation_zero_spot_bump(self):
        """Test that zero spot_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="spot_bump must be positive"):
            BumpConfig(spot_bump=0.0)

    def test_validation_negative_vol_bump(self):
        """Test that negative vol_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="vol_bump must be positive"):
            BumpConfig(vol_bump=-0.01)

    def test_validation_large_vol_bump(self):
        """Test that too large vol_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="vol_bump seems too large"):
            BumpConfig(vol_bump=0.15)  # 15 vol points

    def test_validation_zero_vol_bump(self):
        """Test that zero vol_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="vol_bump must be positive"):
            BumpConfig(vol_bump=0.0)

    def test_validation_negative_time_bump(self):
        """Test that negative time_bump_days raises ValidationError."""
        with pytest.raises(ValidationError, match="time_bump_days must be positive"):
            BumpConfig(time_bump_days=-1)

    def test_validation_zero_time_bump(self):
        """Test that zero time_bump_days raises ValidationError."""
        with pytest.raises(ValidationError, match="time_bump_days must be positive"):
            BumpConfig(time_bump_days=0)

    def test_validation_large_time_bump(self):
        """Test that too large time_bump_days raises ValidationError."""
        with pytest.raises(ValidationError, match="time_bump_days seems too large"):
            BumpConfig(time_bump_days=60)  # 2 months

    def test_validation_invalid_time_bump_mode(self):
        """Test that invalid time_bump_mode raises ValidationError."""
        with pytest.raises(ValidationError, match="time_bump_mode must be one of"):
            BumpConfig(time_bump_mode="trading_days")

    def test_time_bump_mode_normalized(self):
        """Test that time_bump_mode is normalized to lowercase."""
        config = BumpConfig(time_bump_mode="BUSINESS_DAYS")
        assert config.time_bump_mode == "business_days"

    def test_validation_negative_rate_bump(self):
        """Test that negative rate_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_bump must be positive"):
            BumpConfig(rate_bump=-0.0001)

    def test_validation_zero_rate_bump(self):
        """Test that zero rate_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_bump must be positive"):
            BumpConfig(rate_bump=0.0)

    def test_validation_large_rate_bump(self):
        """Test that too large rate_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="rate_bump seems too large"):
            BumpConfig(rate_bump=0.02)  # 200bp

    def test_validation_negative_div_bump(self):
        """Test that negative div_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="div_bump must be positive"):
            BumpConfig(div_bump=-0.0001)

    def test_validation_zero_div_bump(self):
        """Test that zero div_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="div_bump must be positive"):
            BumpConfig(div_bump=0.0)

    def test_validation_large_div_bump(self):
        """Test that too large div_bump raises ValidationError."""
        with pytest.raises(ValidationError, match="div_bump seems too large"):
            BumpConfig(div_bump=0.02)  # 200bp

    def test_get_bump_for_factor(self):
        """Test getting bump for specific factors."""
        config = BumpConfig(
            spot_bump=0.02,
            vol_bump=0.005,
        )
        assert config.get_bump_for_factor("spot") == 0.02
        assert config.get_bump_for_factor("vol") == 0.005
        assert config.get_bump_for_factor("time") == 1.0
        assert config.get_bump_for_factor("rate") == 0.0001
        assert config.get_bump_for_factor("div") == 0.0001

    def test_get_bump_for_invalid_factor(self):
        """Test that invalid factor raises ValidationError."""
        config = BumpConfig()
        with pytest.raises(ValidationError, match="Unknown risk factor"):
            config.get_bump_for_factor("invalid")

    def test_custom_spot_bump(self):
        """Test creating BumpConfig with custom spot bump."""
        config = BumpConfig(spot_bump=0.001)  # 0.1%
        assert config.spot_bump == 0.001
        assert config.vol_bump == 0.01  # Default
        assert config.time_bump_days == 1  # Default

    def test_custom_vol_bump(self):
        """Test creating BumpConfig with custom vol bump."""
        config = BumpConfig(vol_bump=0.02)  # 2 vol points
        assert config.spot_bump == 0.01  # Default
        assert config.vol_bump == 0.02
        assert config.time_bump_days == 1  # Default

    def test_custom_time_bump(self):
        """Test creating BumpConfig with custom time bump."""
        config = BumpConfig(time_bump_days=7)  # 7 days
        assert config.spot_bump == 0.01  # Default
        assert config.vol_bump == 0.01  # Default
        assert config.time_bump_days == 7

    def test_custom_rate_bump(self):
        """Test creating BumpConfig with custom rate bump."""
        config = BumpConfig(rate_bump=0.001)  # 10bp
        assert config.spot_bump == 0.01  # Default
        assert config.vol_bump == 0.01  # Default
        assert config.rate_bump == 0.001

    def test_custom_div_bump(self):
        """Test creating BumpConfig with custom div bump."""
        config = BumpConfig(div_bump=0.001)  # 10bp
        assert config.spot_bump == 0.01  # Default
        assert config.vol_bump == 0.01  # Default
        assert config.div_bump == 0.001

    def test_multiple_custom_bumps(self):
        """Test creating BumpConfig with multiple custom bumps."""
        config = BumpConfig(
            spot_bump=0.005,  # 0.5%
            vol_bump=0.02,    # 2 vol points
            rate_bump=0.001,  # 10bp
        )
        assert config.spot_bump == 0.005
        assert config.vol_bump == 0.02
        assert config.rate_bump == 0.001
        assert config.time_bump_days == 1  # Default
        assert config.div_bump == 0.0001  # Default

    def test_edge_case_minimum_spot_bump(self):
        """Test minimum valid spot bump."""
        config = BumpConfig(spot_bump=0.0001)  # 0.01%
        assert config.spot_bump == 0.0001

    def test_edge_case_maximum_spot_bump(self):
        """Test maximum valid spot bump."""
        config = BumpConfig(spot_bump=0.1)  # 10%
        assert config.spot_bump == 0.1

    def test_edge_case_minimum_vol_bump(self):
        """Test minimum valid vol bump."""
        config = BumpConfig(vol_bump=0.001)  # 0.1 vol point
        assert config.vol_bump == 0.001

    def test_edge_case_maximum_vol_bump(self):
        """Test maximum valid vol bump."""
        config = BumpConfig(vol_bump=0.1)  # 10 vol points
        assert config.vol_bump == 0.1

    def test_edge_case_minimum_rate_bump(self):
        """Test minimum valid rate bump."""
        config = BumpConfig(rate_bump=0.00001)  # 0.1bp
        assert config.rate_bump == 0.00001

    def test_edge_case_maximum_rate_bump(self):
        """Test maximum valid rate bump."""
        config = BumpConfig(rate_bump=0.01)  # 100bp
        assert config.rate_bump == 0.01

    def test_edge_case_minimum_div_bump(self):
        """Test minimum valid div bump."""
        config = BumpConfig(div_bump=0.00001)  # 0.1bp
        assert config.div_bump == 0.00001

    def test_edge_case_maximum_div_bump(self):
        """Test maximum valid div bump."""
        config = BumpConfig(div_bump=0.01)  # 100bp
        assert config.div_bump == 0.01

    def test_edge_case_minimum_time_bump(self):
        """Test minimum valid time bump."""
        config = BumpConfig(time_bump_days=1)
        assert config.time_bump_days == 1

    def test_edge_case_maximum_time_bump(self):
        """Test maximum valid time bump."""
        config = BumpConfig(time_bump_days=30)
        assert config.time_bump_days == 30
