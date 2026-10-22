"""Configurable trading-days-per-year for daily observation schedules (sub-project C)."""
from quantark.asset.equity.product.option import snowball_helpers as sh


def test_default_trading_days_constant():
    assert sh.DEFAULT_TRADING_DAYS_PER_YEAR == 252


def test_daily_uses_default_trading_days():
    dates = sh.generate_ko_observation_dates(1.0, "daily")
    assert len(dates) == 252


def test_daily_honors_configured_trading_days():
    dates = sh.generate_ko_observation_dates(1.0, "daily", trading_days_per_year=244)
    assert len(dates) == 244


def test_non_daily_unaffected_by_trading_days():
    dates = sh.generate_ko_observation_dates(1.0, "quarterly", trading_days_per_year=244)
    assert len(dates) == 4


def test_nonpositive_trading_days_rejected():
    import pytest
    from quantark.util.exceptions import ValidationError
    for bad in (0, -10):
        with pytest.raises(ValidationError, match="trading_days_per_year"):
            sh.generate_ko_observation_dates(1.0, "daily", trading_days_per_year=bad)
