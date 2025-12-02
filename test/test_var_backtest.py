"""
Unit tests for VaR backtesting.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from var import VaRConfig
from var.backtest import VaRBacktester, VaRBacktestResult
from var.engines import ParametricVaREngine
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
def backtest_historical_data():
    """Create historical data with P&L for backtesting."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    data = pd.DataFrame({
        'spot_return': np.random.normal(0.0005, 0.02, 300),
        'vol_change': np.random.normal(0.0, 0.01, 300),
        'rate_shift': np.random.normal(0.0, 0.001, 300),
        'pnl': np.random.normal(50, 500, 300),
    }, index=dates)
    
    return data


def test_var_backtester_initialization():
    """Test VaRBacktester initialization."""
    backtester = VaRBacktester(confidence_level=0.99, holding_period=1)
    
    assert backtester.confidence_level == 0.99
    assert backtester.holding_period == 1


def test_kupiec_test():
    """Test Kupiec POF test calculation."""
    backtester = VaRBacktester(confidence_level=0.99)
    
    stat, pval = backtester._kupiec_test(
        num_exceptions=3,
        num_observations=250,
        confidence_level=0.99
    )
    
    assert stat >= 0
    assert 0 <= pval <= 1


def test_christoffersen_test():
    """Test Christoffersen conditional coverage test."""
    backtester = VaRBacktester(confidence_level=0.99)
    
    exception_flags = [False] * 245 + [True, False, True, False, True]
    
    stat, pval = backtester._christoffersen_test(
        exception_flags=exception_flags,
        confidence_level=0.99
    )
    
    assert stat >= 0
    assert 0 <= pval <= 1


def test_basel_traffic_light():
    """Test Basel traffic light zone classification."""
    backtester = VaRBacktester(confidence_level=0.99)
    
    green_zone = backtester._basel_traffic_light(
        num_exceptions=2,
        num_observations=250,
        confidence_level=0.99
    )
    assert green_zone == "green"
    
    yellow_zone = backtester._basel_traffic_light(
        num_exceptions=7,
        num_observations=250,
        confidence_level=0.99
    )
    assert yellow_zone == "yellow"
    
    red_zone = backtester._basel_traffic_light(
        num_exceptions=12,
        num_observations=250,
        confidence_level=0.99
    )
    assert red_zone == "red"


def test_backtest_empty_portfolio(sample_pricing_env, backtest_historical_data):
    """Test backtesting with empty portfolio."""
    portfolio = EquityPortfolio(
        portfolio_name="Empty Portfolio",
        pricing_environments={"AAPL": sample_pricing_env}
    )
    
    backtester = VaRBacktester(confidence_level=0.99)
    engine = ParametricVaREngine()
    
    with pytest.raises(ValidationError, match="empty portfolio"):
        backtester.run_backtest(portfolio, backtest_historical_data, engine)


def test_backtest_missing_pnl_column(sample_portfolio):
    """Test backtesting with data missing P&L column."""
    data = pd.DataFrame({
        'spot_return': [0.01, -0.02, 0.015],
        'vol_change': [0.001, -0.002, 0.0],
    })
    
    backtester = VaRBacktester()
    engine = ParametricVaREngine()
    
    with pytest.raises(ValidationError, match="'pnl' column"):
        backtester.run_backtest(sample_portfolio, data, engine)