"""The env date axis (DCN time-axis spec D9/D13/D14/D20/D21; AC1, AC2, AC2b).

``env_time(env, date, anchor=None)`` is the signed year fraction from ``anchor`` (default the env's valuation date) to
``date`` on the environment's clock. Under BUSINESS_DAYS a date that is an env-calendar holiday is placed by
``PricingEnvironment.holiday_time_adjustment`` (both ends rolled, D14).
"""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.param import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.priceenv import CurveQuotingBasis
from quantark.priceenv.date_axis import env_time, env_times
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.business_calendar import BusinessDayConvention as BDC
from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.exceptions import MarketDataError

US = create_calendar(CalendarType.US, year_range=(2024, 2027))
SSE = create_calendar(CalendarType.CHINA_SSE, year_range=(2024, 2027))
V = datetime(2025, 6, 30)


def _env(dc=DayCountConvention.CALENDAR_DAYS, calendar=None, adjust=None, v=V, days=252):
    kw = {} if adjust is None else {"holiday_time_adjustment": adjust}
    if dc == DayCountConvention.BUSINESS_DAYS:
        kw["rate_curve_quoting"] = CurveQuotingBasis.TRADING_YEAR
    return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=v, day_count_convention=dc,
                              calendar=calendar, bus_days_in_year=days, **kw)


DATES = [V + timedelta(days=k) for k in range(-40, 400, 3)]


# --- the field ------------------------------------------------------------------------------------------------------

def test_default_is_preceding():
    assert _env().holiday_time_adjustment is BDC.PRECEDING


@pytest.mark.parametrize("dc", [DayCountConvention.CALENDAR_DAYS, DayCountConvention.BUSINESS_DAYS])
def test_unadjusted_is_refused_under_every_convention(dc):
    with pytest.raises(MarketDataError, match="UNADJUSTED"):
        _env(dc, calendar=US, adjust=BDC.UNADJUSTED)


def test_non_convention_value_is_refused():
    with pytest.raises(MarketDataError):
        _env(adjust="following")


# --- AC1: calendar-day conventions ------------------------------------------------------------------------------------

@pytest.mark.parametrize("dc", [DayCountConvention.CALENDAR_DAYS, DayCountConvention.ACT_365])
def test_calendar_conventions_are_days_over_365_bitwise(dc):
    env = _env(dc)
    for d in DATES:
        t = env_time(env, d)
        expected = (d - V).days / 365.0
        assert t.hex() == expected.hex() if t != 0.0 else expected == 0.0, (d, t, expected)


def test_act_360_matches_calculate_year_fraction_signed():
    env = _env(DayCountConvention.ACT_360)
    for d in DATES:
        t = env_time(env, d)
        if d > V:
            assert t == calculate_year_fraction(V, d, DayCountConvention.ACT_360)
        elif d < V:
            assert t == -calculate_year_fraction(d, V, DayCountConvention.ACT_360)
        else:
            assert t == 0.0


def test_vector_form_equals_scalar_form():
    for env in (_env(), _env(DayCountConvention.ACT_360),
                _env(DayCountConvention.BUSINESS_DAYS, US, BDC.FOLLOWING)):
        np.testing.assert_array_equal(env_times(env, DATES), [env_time(env, d) for d in DATES])


def test_time_of_day_is_normalised():
    env = _env()
    assert env_time(env, datetime(2025, 7, 4, 15, 0)) == env_time(env, datetime(2025, 7, 4))
    late = _env(v=datetime(2025, 6, 30, 15, 0))
    assert env_time(late, datetime(2025, 6, 30)) == 0.0


# --- AC2: business days ----------------------------------------------------------------------------------------------

def test_business_days_preceding_equals_calculate_year_fraction():
    env = _env(DayCountConvention.BUSINESS_DAYS, SSE, days=244)
    for d in DATES:
        if d > V:
            assert env_time(env, d) == calculate_year_fraction(V, d, DayCountConvention.BUSINESS_DAYS, 244,
                                                               calendar=SSE)


def test_us_holiday_preceding_and_following():
    pre = _env(DayCountConvention.BUSINESS_DAYS, US, BDC.PRECEDING)
    fol = _env(DayCountConvention.BUSINESS_DAYS, US, BDC.FOLLOWING)
    jul = lambda day: datetime(2025, 7, day)  # noqa: E731
    assert env_time(pre, jul(3)) == env_time(pre, jul(4)) == 3 / 252
    assert env_time(pre, jul(7)) == 4 / 252
    assert env_time(fol, jul(4)) == env_time(fol, jul(7)) == 4 / 252
    assert env_time(fol, jul(3)) == 3 / 252


def test_modified_following_rolls_back_at_month_end():
    # 2025-08-30/31 is a weekend at a month end: MODIFIED_FOLLOWING keeps it in August (Fri 08-29)
    mf = _env(DayCountConvention.BUSINESS_DAYS, US, BDC.MODIFIED_FOLLOWING)
    fol = _env(DayCountConvention.BUSINESS_DAYS, US, BDC.FOLLOWING)
    assert env_time(mf, datetime(2025, 8, 31)) == env_time(mf, datetime(2025, 8, 29))
    assert env_time(fol, datetime(2025, 8, 31)) == env_time(fol, datetime(2025, 9, 2))  # Mon 09-01 Labor Day


@pytest.mark.parametrize("adjust", [BDC.PRECEDING, BDC.FOLLOWING, BDC.MODIFIED_FOLLOWING, BDC.MODIFIED_PRECEDING])
def test_holiday_anchor(adjust):
    """D14: both ends roll -- a date in the anchor's holiday run is at 0, a past date is never positive."""
    env = _env(DayCountConvention.BUSINESS_DAYS, US, adjust, v=datetime(2025, 7, 4))
    assert env_time(env, datetime(2025, 7, 5)) == 0.0
    assert env_time(env, datetime(2025, 7, 6)) == 0.0
    assert env_time(env, datetime(2025, 7, 3)) <= 0.0
    assert env_time(env, datetime(2025, 7, 7)) >= 0.0   # FOLLOWING rolls the anchor itself onto 07-07
    assert env_time(env, datetime(2025, 7, 8)) > 0.0


@pytest.mark.parametrize("adjust", [BDC.PRECEDING, BDC.FOLLOWING, BDC.MODIFIED_FOLLOWING, BDC.MODIFIED_PRECEDING])
def test_monotone_over_a_year(adjust):
    env = _env(DayCountConvention.BUSINESS_DAYS, US, adjust)
    times = env_times(env, [V + timedelta(days=k) for k in range(-60, 420)])
    assert np.all(np.diff(times) >= 0.0)


def test_adjustment_is_ignored_off_business_days():
    a = _env(DayCountConvention.ACT_360, US, BDC.PRECEDING)
    b = _env(DayCountConvention.ACT_360, US, BDC.FOLLOWING)
    np.testing.assert_array_equal(env_times(a, DATES), env_times(b, DATES))


# --- AC2b: anchor ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("env", [_env(), _env(DayCountConvention.ACT_360),
                                 _env(DayCountConvention.BUSINESS_DAYS, US, BDC.FOLLOWING)])
def test_explicit_anchor(env):
    anchor = datetime(2025, 7, 1)
    moved = _env(env.day_count_convention, env.calendar, env.holiday_time_adjustment, v=anchor)
    for d in DATES:
        assert env_time(env, d, anchor=anchor) == env_time(moved, d)
