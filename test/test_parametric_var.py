"""
Unit tests for parametric VaR engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from var import VaRConfig, VaRMethod, EquityRiskFactorConfig
from var.engines import ParametricVaREngine
from portfolio.equity.portfolio import EquityPortfolio
from portfolio.equity.position import EquityPosition
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from util.enum.option_enums import OptionType
from util.exceptions import ValidationError


@pytest.fixture
def sample_pricing_env():
    """Create a sample pricing environment."""
    valuation_date = datetime(2024, 1, 1)
    spot_quote = SpotQuote(spot=100.0, timestamp=valuation_date)

    vol_surface = FlatVolSurface(volatility=0.2)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    return PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date
    )


@pytest.fixture
def sample_portfolio(sample_pricing_env):
    """Create a sample equity portfolio."""
    portfolio = EquityPortfolio(
        portfolio_name="Test Portfolio",
        pricing_environments={"AAPL": sample_pricing_env}
    )
    
    call_option = EuropeanVanillaOption(
        strike=100.0,
        maturity=0.5,
        option_type=OptionType.CALL
    )
    
    engine = BlackScholesEngine()
    
    portfolio.add_position(
        product=call_option,
        quantity=10,
        entry_price=10.0,
        underlying="AAPL",
        engine=engine
    )
    
    return portfolio


@pytest.fixture
def sample_historical_data():
    """Create sample historical data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    data = pd.DataFrame({
        'spot_return': np.random.normal(0.0005, 0.02, 300),
        'vol_change': np.random.normal(0.0, 0.01, 300),
        'rate_shift': np.random.normal(0.0, 0.001, 300),
    }, index=dates)
    
    return data


def test_parametric_var_engine_initialization():
    """Test ParametricVaREngine initialization."""
    config = VaRConfig(
        confidence_level=0.95,
        var_method=VaRMethod.PARAMETRIC
    )
    
    engine = ParametricVaREngine(config=config)
    
    assert engine.config.confidence_level == 0.95
    assert engine.config.var_method == VaRMethod.PARAMETRIC


def test_parametric_var_calculation(sample_portfolio, sample_historical_data):
    """Test basic parametric VaR calculation."""
    config = VaRConfig(
        confidence_level=0.99,
        var_method=VaRMethod.PARAMETRIC,
        lookback_days=252,
        equity_factors=EquityRiskFactorConfig(
            include_spot=True,
            include_vol=True,
            include_rate=True
        )
    )
    
    engine = ParametricVaREngine(config=config)
    
    result = engine.calculate_var(sample_portfolio, sample_historical_data)
    
    assert result.var > 0
    assert result.cvar > 0
    assert result.cvar >= result.var
    assert result.confidence_level == 0.99
    assert result.holding_period == 1
    assert result.method == VaRMethod.PARAMETRIC
    assert result.portfolio_value > 0
    assert 0 <= result.var_as_pct <= 1


def test_parametric_var_empty_portfolio(sample_pricing_env, sample_historical_data):
    """Test VaR calculation with empty portfolio."""
    portfolio = EquityPortfolio(
        portfolio_name="Empty Portfolio",
        pricing_environments={"AAPL": sample_pricing_env}
    )
    
    engine = ParametricVaREngine()
    
    with pytest.raises(ValidationError, match="empty portfolio"):
        engine.calculate_var(portfolio, sample_historical_data)


def test_parametric_var_insufficient_data(sample_portfolio):
    """Test VaR with insufficient historical data."""
    short_data = pd.DataFrame({
        'spot_return': [0.01, -0.02, 0.015],
        'vol_change': [0.001, -0.002, 0.0],
        'rate_shift': [0.0001, 0.0, -0.0001],
    })
    
    config = VaRConfig(lookback_days=252)
    engine = ParametricVaREngine(config=config)
    
    with pytest.raises(Exception):
        engine.calculate_var(sample_portfolio, short_data)


def test_parametric_var_factor_attribution(sample_portfolio, sample_historical_data):
    """Test factor VaR attribution."""
    config = VaRConfig(
        confidence_level=0.99,
        calculate_factor_var=True,
        equity_factors=EquityRiskFactorConfig(
            include_spot=True,
            include_vol=True,
            include_rate=True
        )
    )
    
    engine = ParametricVaREngine(config=config)
    
    result = engine.calculate_var(sample_portfolio, sample_historical_data)
    
    assert result.factor_var is not None
    assert 'spot_return' in result.factor_var
    assert 'vol_change' in result.factor_var
    assert 'rate_shift' in result.factor_var
    
    assert all(v >= 0 for v in result.factor_var.values())


def test_parametric_var_multiday_scaling(sample_portfolio, sample_historical_data):
    """Test multi-day VaR scaling."""
    config_1day = VaRConfig(holding_period=1)
    engine_1day = ParametricVaREngine(config=config_1day)
    result_1day = engine_1day.calculate_var(sample_portfolio, sample_historical_data)
    
    config_10day = VaRConfig(holding_period=10, scaling_method="sqrt_t")
    engine_10day = ParametricVaREngine(config=config_10day)
    result_10day = engine_10day.calculate_var(sample_portfolio, sample_historical_data)
    
    expected_10day_var = result_1day.var * np.sqrt(10)
    
    assert np.isclose(result_10day.var, expected_10day_var, rtol=0.01)


def test_supports_portfolio(sample_portfolio):
    """Test portfolio type support check."""
    engine = ParametricVaREngine()
    
    assert engine.supports_portfolio(sample_portfolio) is True
    assert engine.supports_portfolio("not a portfolio") is False