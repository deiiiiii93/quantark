"""BUSINESS_DAYS without a calendar raises; one clock per env (spec §4.3/4.4)."""
from datetime import datetime

import pytest

from quantark.param import SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError


def test_business_days_without_calendar_raises():
    """The old fallback was EXACTLY act/365 (num_days*(D/365)/D) — a silent lie."""
    with pytest.raises(ValidationError, match="calendar"):
        calculate_year_fraction(
            datetime(2026, 1, 1), datetime(2026, 7, 1),
            DayCountConvention.BUSINESS_DAYS, 252,
        )


def test_env_clock_mismatch_raises():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(
        TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2027, 2, 9)
    )
    wrapped = TradingClockRateCurve(FlatRateCurve(0.02), m)
    with pytest.raises(ValidationError, match="clock"):
        PricingEnvironment(
            rate_curve=wrapped, valuation_date=datetime(2026, 2, 9),
            spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
            day_count_convention=DayCountConvention.BUSINESS_DAYS,
            bus_days_in_year=252,          # != wrapper's 244 -> must raise
            calendar=cal,
        )


def test_env_clock_match_passes():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(
        TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2027, 2, 9)
    )
    wrapped = TradingClockRateCurve(FlatRateCurve(0.02), m)
    PricingEnvironment(
        rate_curve=wrapped, valuation_date=datetime(2026, 2, 9),
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=244, calendar=cal,
    )
