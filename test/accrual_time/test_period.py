"""AccrualPeriod: the representations, the refusals and the math (spec §4.1, §5)."""
from datetime import datetime

import pytest

from quantark.asset.equity.product.option.accrual import (
    AccrualBasis, AccrualKind as K, AccrualPeriod, KnockInAnchor, coupon_accrual, cumulative_accrual,
)
from quantark.util.calendar import CalendarType, create_calendar, calculate_year_fraction
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.exceptions import ValidationError

A365 = AccrualBasis(DC.ACT_365)
T0, T1, T2 = datetime(2026, 1, 5), datetime(2026, 4, 6), datetime(2026, 7, 6)


def test_time_period_reads_the_quoted_fraction():
    p = AccrualPeriod(K.TIME, origin=0.0, start=1 / 12, end=2 / 12)
    assert cumulative_accrual(p) == 2 / 12
    assert coupon_accrual(p) == 2 / 12 - 1 / 12


def test_date_period_is_todays_year_fraction_call():
    p = AccrualPeriod(K.DATE, origin=T0, start=T1, end=T2, basis=A365)
    assert cumulative_accrual(p) == calculate_year_fraction(T0, T2, DC.ACT_365)
    assert coupon_accrual(p) == (T2 - T1).days / 365.0


def test_overrides_win_on_their_own_leg():
    p = AccrualPeriod(K.TIME, origin=0.0, start=0.0, end=0.25, cumulative_fraction=0.4, period_fraction=0.1)
    assert (cumulative_accrual(p), coupon_accrual(p)) == (0.4, 0.1)


def test_cumulative_fraction_only_serves_the_cumulative_leg():
    p = AccrualPeriod(K.NONE, cumulative_fraction=0.75)
    assert cumulative_accrual(p) == 0.75
    with pytest.raises(ValidationError, match="no accrual terms"):
        coupon_accrual(p)


def test_end_only_first_period_refuses_the_coupon_leg():
    p = AccrualPeriod(K.TIME, origin=0.0, end=0.3)
    assert cumulative_accrual(p) == 0.3
    with pytest.raises(ValidationError, match="period start unknown"):
        coupon_accrual(p)


def test_none_raises_when_asked():
    with pytest.raises(ValidationError, match="no accrual terms"):
        cumulative_accrual(AccrualPeriod(K.NONE))


@pytest.mark.parametrize("kw", [
    dict(kind=K.NONE, end=0.25),                                         # NONE carries no endpoints
    dict(kind=K.TIME, origin=0.0),                                       # no end
    dict(kind=K.TIME, origin=0.0, start=0.3, end=0.2),                   # end < start
    dict(kind=K.DATE, origin=T1, end=T0, basis=A365),                    # a DATE end before its origin
    dict(kind=K.TIME, origin=0.0, end=T1),                               # mixed TIME / DATE
    dict(kind=K.DATE, origin=T0, end=T1),                                # DATE without basis
    dict(kind=K.TIME, start=0.0, end=0.25, relative=True, origin=0.0),   # relative has no origin
    dict(kind=K.DATE, start=T0, end=T1, relative=True, basis=A365),      # relative is TIME
    dict(kind=K.TIME, origin=0.0, end=0.25, cumulative_fraction=-0.1),   # negative override
])
def test_refused_representations(kw):
    with pytest.raises(ValidationError):
        AccrualPeriod(**kw)


def test_zero_length_periods_are_zero_without_the_helper():
    p = AccrualPeriod(K.DATE, origin=T1, start=T1, end=T1, basis=A365)
    assert cumulative_accrual(p) == 0.0 and coupon_accrual(p) == 0.0


def test_relative_reads_the_anchor_plus_the_quoted_offset():
    anchor = KnockInAnchor(source_index=0, fixing=0.5, origin=0.0, basis=A365)
    p = AccrualPeriod(K.TIME, start=0.0, end=0.25, relative=True)
    assert cumulative_accrual(p, knock_in=anchor) == 0.75
    assert cumulative_accrual(AccrualPeriod(K.TIME, start=0.0, end=0.25, relative=True, anchor=anchor)) == 0.75
    with pytest.raises(ValidationError, match="anchor"):
        cumulative_accrual(p)


def test_dated_anchor_plus_quoted_offset_is_not_91_days():
    """Review round 3, P1-1: offset 0.25 on a dated knock-in adds 0.25, not YF(91 days)."""
    anchor = KnockInAnchor(source_index=0, fixing=T1, origin=T0, basis=A365)
    p = AccrualPeriod(K.TIME, start=0.0, end=0.25, relative=True, anchor=anchor)
    assert cumulative_accrual(p) == calculate_year_fraction(T0, T1, DC.ACT_365) + 0.25
    assert cumulative_accrual(p) != calculate_year_fraction(T0, datetime(2026, 7, 6), DC.ACT_365)


def test_business_days_reads_its_own_calendar_and_denominator():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    basis = AccrualBasis(DC.BUSINESS_DAYS, calendar=cal, bus_days_in_year=244.0)
    p = AccrualPeriod(K.DATE, origin=T0, end=T1, basis=basis)
    assert cumulative_accrual(p) == calculate_year_fraction(T0, T1, DC.BUSINESS_DAYS, 244.0, calendar=cal)
    with pytest.raises(ValidationError, match="BUSINESS_DAYS accrual needs its calendar and denominator"):
        cumulative_accrual(AccrualPeriod(K.DATE, origin=T0, end=T1, basis=AccrualBasis(DC.BUSINESS_DAYS)))


def test_a_past_numeric_record_keeps_its_place_but_not_its_cumulative_accrual():
    """Spec 3B decision 13: a record at a negative time is history kept in the schedule. Its time starts the next
    coupon period, as today; its own knock-out accrual (before the contract's time origin) is never paid: refused."""
    past = AccrualPeriod(K.TIME, origin=0.0, end=-0.25)
    nxt = AccrualPeriod(K.TIME, origin=0.0, start=-0.25, end=0.0)
    assert coupon_accrual(nxt) == 0.25
    with pytest.raises(ValidationError, match="before the contract's accrual origin"):
        cumulative_accrual(past)
