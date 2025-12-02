"""
Unit tests for historical VaR engine.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from var import VaRConfig, VaRMethod
from var.engines import HistoricalVaREngine
from portfolio.equity.portfolio import EquityPortfolio
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from priceenv import PricingEnvironment
from param import SpotQuote, VolatilitySurface, RateCurve, DividendYield
from util.enum.option_enums import OptionType
from util.exceptions import ValidationError


@pytest.fixture
def sample_pricing_env():
    """Create a sample pricing environment."""
    valuation_date = datetime(2024, 1, 1)
    spot_quote = SpotQuote(spot=100.0, timestamp=valuation_date)
    
    vol_surface = VolatilitySurface(
        valuation_date=valuation_date,
        spot=100.0,
        flat_vol=0.2
    )
    
    rate_curve = RateCurve(
        valuation_date=valuation_date,
        flat_rate=0.05
    )
    
    div_yield = DividendYield(
        valuation_date=valuation_date,
        flat_yield=0.02
    )
    
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
        expiry=datetime(2024, 7, 1),
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


def test_historical_var_engine_initialization():
    """Test HistoricalVaREngine initialization."""
    config = VaRConfig(
        confidence_level=0.95,
        var_method=VaRMethod.HISTORICAL
    )
    
    engine = HistoricalVaREngine(config=config)
    
    assert engine.config.confidence_level == 0.95
    assert engine.config.var_method == VaRMethod.HISTORICAL


def test_historical_var_calculation(sample_portfolio, sample_historical_data):
    """Test basic historical VaR calculation."""
    config = VaRConfig(
        confidence_level=0.99,
        var_method=VaRMethod.HISTORICAL,
        lookback_days=252
    )
    
    engine = HistoricalVaREngine(config=config)
    
    result = engine.calculate_var(sample_portfolio, sample_historical_data)
    
    assert result.var > 0
    assert result.cvar > 0
    assert result.cvar >= result.var
    assert result.confidence_level == 0.99
    assert result.method == VaRMethod.HISTORICAL
    assert result.scenarios is not None
    assert len(result.scenarios) == 252
    assert result.worst_scenarios is not None
    assert len(result.worst_scenarios) == 10


def test_historical_var_empty_portfolio(sample_pricing_env, sample_historical_data):
    """Test historical VaR with empty portfolio."""
    portfolio = EquityPortfolio(
        portfolio_name="Empty Portfolio",
        pricing_environments={"AAPL": sample_pricing_env}
    )
    
    engine = HistoricalVaREngine()
    
    with pytest.raises(ValidationError, match="empty portfolio"):
        engine.calculate_var(portfolio, sample_historical_data)


def test_historical_var_insufficient_data(sample_portfolio):
    """Test historical VaR with insufficient data."""
    short_data = pd.DataFrame({
        'spot_return': [0.01, -0.02],
        'vol_change': [0.001, -0.002],
    })
    
    config = VaRConfig(lookback_days=252)
    engine = HistoricalVaREngine(config=config)
    
    with pytest.raises(Exception):
        engine.calculate_var(sample_portfolio, short_data)


def test_supports_portfolio(sample_portfolio):
    """Test portfolio type support check."""
    engine = HistoricalVaREngine()
    
    assert engine.supports_portfolio(sample_portfolio) is True
    assert engine.supports_portfolio("not a portfolio") is False