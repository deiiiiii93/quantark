"""
Unit tests for VaR configuration classes.
"""

import pytest
from datetime import datetime

from var.config import VaRConfig, VaRMethod, EquityRiskFactorConfig, FIRiskFactorConfig
from util.exceptions import ValidationError


def test_var_config_defaults():
    """Test VaRConfig default values."""
    config = VaRConfig()
    
    assert config.confidence_level == 0.99
    assert config.holding_period == 1
    assert config.lookback_days == 252
    assert config.var_method == VaRMethod.PARAMETRIC
    assert config.scaling_method == "sqrt_t"
    assert config.mc_num_simulations == 10000
    assert config.mc_seed is None
    assert config.calculate_component_var is True
    assert config.calculate_marginal_var is True
    assert config.calculate_factor_var is True
    assert config.calculate_incremental_var is False
    assert config.calculate_stressed_var is False


def test_var_config_custom_confidence():
    """Test VaRConfig with custom confidence level."""
    config = VaRConfig(confidence_level=0.95)
    assert config.confidence_level == 0.95


def test_var_config_multiday():
    """Test VaRConfig for multi-day VaR."""
    config = VaRConfig(holding_period=10)
    assert config.holding_period == 10


def test_var_config_invalid_confidence():
    """Test VaRConfig validation for invalid confidence level."""
    with pytest.raises(ValidationError, match="confidence_level must be between 0 and 1"):
        VaRConfig(confidence_level=1.5)
    
    with pytest.raises(ValidationError, match="confidence_level must be between 0 and 1"):
        VaRConfig(confidence_level=0.0)
    
    with pytest.raises(ValidationError, match="confidence_level must be between 0 and 1"):
        VaRConfig(confidence_level=-0.5)


def test_var_config_invalid_holding_period():
    """Test VaRConfig validation for invalid holding period."""
    with pytest.raises(ValidationError, match="holding_period must be >= 1"):
        VaRConfig(holding_period=0)
    
    with pytest.raises(ValidationError, match="holding_period must be >= 1"):
        VaRConfig(holding_period=-1)


def test_var_config_invalid_lookback():
    """Test VaRConfig validation for invalid lookback days."""
    with pytest.raises(ValidationError, match="lookback_days must be >= 1"):
        VaRConfig(lookback_days=0)


def test_var_config_invalid_scaling_method():
    """Test VaRConfig validation for invalid scaling method."""
    with pytest.raises(ValidationError, match="scaling_method must be"):
        VaRConfig(scaling_method="invalid")


def test_var_config_invalid_mc_simulations():
    """Test VaRConfig validation for invalid MC simulations."""
    with pytest.raises(ValidationError, match="mc_num_simulations must be >= 100"):
        VaRConfig(mc_num_simulations=50)


def test_var_config_stressed_var_invalid_period():
    """Test VaRConfig validation for invalid stressed period."""
    start = datetime(2020, 1, 1)
    end = datetime(2019, 1, 1)
    
    with pytest.raises(ValidationError, match="stressed_period_start must be before stressed_period_end"):
        VaRConfig(
            calculate_stressed_var=True,
            stressed_period_start=start,
            stressed_period_end=end
        )


def test_equity_risk_factor_config_defaults():
    """Test EquityRiskFactorConfig defaults."""
    config = EquityRiskFactorConfig()
    
    assert config.include_spot is True
    assert config.include_vol is True
    assert config.include_rate is True
    assert config.include_div_yield is False


def test_equity_risk_factor_config_custom():
    """Test EquityRiskFactorConfig with custom settings."""
    config = EquityRiskFactorConfig(
        include_spot=True,
        include_vol=True,
        include_rate=False,
        include_div_yield=True
    )
    
    assert config.include_spot is True
    assert config.include_vol is True
    assert config.include_rate is False
    assert config.include_div_yield is True


def test_fi_risk_factor_config_defaults():
    """Test FIRiskFactorConfig defaults."""
    config = FIRiskFactorConfig()
    
    assert config.include_parallel_shift is True
    assert config.include_key_rates is False
    assert config.key_rate_tenors == [2.0, 5.0, 10.0, 30.0]


def test_fi_risk_factor_config_custom():
    """Test FIRiskFactorConfig with custom tenors."""
    config = FIRiskFactorConfig(
        include_key_rates=True,
        key_rate_tenors=[1.0, 3.0, 5.0, 7.0, 10.0]
    )
    
    assert config.include_key_rates is True
    assert config.key_rate_tenors == [1.0, 3.0, 5.0, 7.0, 10.0]


def test_var_method_enum():
    """Test VaRMethod enum values."""
    assert hasattr(VaRMethod, "PARAMETRIC")
    assert hasattr(VaRMethod, "HISTORICAL")
    assert hasattr(VaRMethod, "MONTE_CARLO")
    
    assert str(VaRMethod.PARAMETRIC) == "Parametric"
    assert str(VaRMethod.HISTORICAL) == "Historical"
    assert str(VaRMethod.MONTE_CARLO) == "Monte Carlo"


def test_var_config_with_risk_factors():
    """Test VaRConfig with risk factor configurations."""
    equity_factors = EquityRiskFactorConfig(include_spot=True, include_vol=True)
    fi_factors = FIRiskFactorConfig(include_key_rates=True)
    
    config = VaRConfig(
        equity_factors=equity_factors,
        fi_factors=fi_factors
    )
    
    assert config.equity_factors.include_spot is True
    assert config.equity_factors.include_vol is True
    assert config.fi_factors.include_key_rates is True
