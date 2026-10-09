import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime

import numpy as np
import pytest

from quantark.cashleg.accrual_leg import AccrualLeg, KOBehavior, PaymentConvention, SurvivalBasis
from quantark.cashleg.base import LegDirection
from quantark.cashleg.base_amount import BaseAmount, BaseAmountMode
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import Calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.exceptions import ValidationError


def _env(rate=0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def _quarterly_schedule_1y():
    return LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )


def _leg(**overrides):
    kwargs = dict(
        rate=0.04,
        base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=_quarterly_schedule_1y(),
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    kwargs.update(overrides)
    return AccrualLeg(**kwargs)


def test_no_ko_reduces_to_deterministic_annuity():
    leg = _leg()
    pv = leg.value(EventDistribution.trivial(maturity=1.0), _env(rate=0.05), 0.0)
    expected = sum(
        0.04 * 1_000_000.0 * 0.25 * math.exp(-0.05 * t)
        for t in [0.25, 0.5, 0.75, 1.0]
    )
    assert pv == pytest.approx(expected, rel=1e-9)


def test_pay_full_schedule_ignores_ko():
    leg = _leg(ko_behavior=KOBehavior.PAY_FULL_SCHEDULE)
    dist_with_ko = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.3, 0.3, 0.0]),
            EventType.MATURITY_NO_KO: 0.1,
        },
        survival_probability=np.array([1.0, 0.7, 0.4, 0.1, 0.1]),
    )
    pv_with_ko = leg.value(dist_with_ko, _env(), 0.0)
    pv_trivial = leg.value(EventDistribution.trivial(1.0), _env(), 0.0)
    assert pv_with_ko == pytest.approx(pv_trivial, rel=1e-9)


def test_truncate_at_ko_reduces_pv_with_high_ko_probability():
    leg = _leg()
    dist_high_ko = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.5, 0.3, 0.1, 0.0]),
            EventType.MATURITY_NO_KO: 0.1,
        },
        survival_probability=np.array([1.0, 0.5, 0.2, 0.1, 0.1]),
    )
    pv_high = leg.value(dist_high_ko, _env(), 0.0)
    pv_no_ko = leg.value(EventDistribution.trivial(1.0), _env(), 0.0)
    assert pv_high < pv_no_ko


def test_notional_fraction_base_uses_position_notional():
    leg = _leg(
        base=BaseAmount(value=0.25, mode=BaseAmountMode.NOTIONAL_FRACTION),
        ko_behavior=KOBehavior.PAY_FULL_SCHEDULE,
    )
    pv_4m = leg.value(EventDistribution.trivial(1.0), _env(), 4_000_000.0)
    pv_8m = leg.value(EventDistribution.trivial(1.0), _env(), 8_000_000.0)
    assert pv_8m == pytest.approx(2 * pv_4m, rel=1e-9)


def test_complete_period_basis_uses_end_survival():
    dist = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.2, 0.2, 0.2, 0.0]),
            EventType.MATURITY_NO_KO: 0.4,
        },
        survival_probability=np.array([1.0, 0.8, 0.6, 0.4, 0.4]),
    )
    enter = _leg(survival_basis=SurvivalBasis.ENTER_PERIOD).value(dist, _env(), 0.0)
    complete = _leg(survival_basis=SurvivalBasis.COMPLETE_PERIOD).value(
        dist, _env(), 0.0
    )
    assert complete < enter


# ---------------------------------------------------------------------------
# day_count is read when the schedule carries period dates (W4a, findings §10)
# ---------------------------------------------------------------------------

_QUARTER_DATES = [
    datetime(2026, 1, 1),
    datetime(2026, 4, 1),
    datetime(2026, 7, 1),
    datetime(2026, 10, 1),
    datetime(2027, 1, 1),
]
_QUARTER_DAYS = [90, 91, 92, 92]
_PAY_TIMES = [0.25, 0.5, 0.75, 1.0]


def _dated_quarterly_schedule_1y():
    return LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array(_PAY_TIMES),
        period_start_dates=_QUARTER_DATES[:-1],
        period_end_dates=_QUARTER_DATES[1:],
    )


def _annuity(fractions, rate=0.05):
    return sum(
        0.04 * 1_000_000.0 * f * math.exp(-rate * t)
        for f, t in zip(fractions, _PAY_TIMES)
    )


def test_times_only_schedule_ignores_day_count():
    """No dates: the accrual stays ``end - start`` whatever the day count."""

    for dc in (
        DayCountConvention.ACT_365,
        DayCountConvention.ACT_360,
        DayCountConvention.THIRTY_360_US,
    ):
        pv = _leg(day_count=dc).value(EventDistribution.trivial(1.0), _env(), 0.0)
        assert pv == pytest.approx(38773.115547, abs=1e-6)


@pytest.mark.parametrize(
    "day_count, fractions",
    [
        (DayCountConvention.ACT_360, [d / 360.0 for d in _QUARTER_DAYS]),
        (DayCountConvention.ACT_365, [d / 365.0 for d in _QUARTER_DAYS]),
        (DayCountConvention.THIRTY_360_US, [0.25, 0.25, 0.25, 0.25]),
    ],
)
def test_dated_schedule_accrues_by_day_count(day_count, fractions):
    leg = _leg(schedule=_dated_quarterly_schedule_1y(), day_count=day_count)
    pv = leg.value(EventDistribution.trivial(1.0), _env(), 0.0)
    assert pv == pytest.approx(_annuity(fractions), rel=1e-12)


def test_dated_act_360_is_365_over_360_of_act_365():
    dist = EventDistribution.trivial(1.0)
    sched = _dated_quarterly_schedule_1y()
    pv_360 = _leg(schedule=sched, day_count=DayCountConvention.ACT_360).value(
        dist, _env(), 0.0
    )
    pv_365 = _leg(schedule=sched, day_count=DayCountConvention.ACT_365).value(
        dist, _env(), 0.0
    )
    assert pv_360 == pytest.approx(pv_365 * 365.0 / 360.0, rel=1e-12)


def test_dated_business_days_without_calendar_is_refused():
    with pytest.raises(ValidationError, match="BUSINESS_DAYS"):
        _leg(
            schedule=_dated_quarterly_schedule_1y(),
            day_count=DayCountConvention.BUSINESS_DAYS,
        )


def test_dated_business_days_without_year_length_is_refused():
    with pytest.raises(ValidationError, match="bus_days_in_year"):
        _leg(
            schedule=_dated_quarterly_schedule_1y(),
            day_count=DayCountConvention.BUSINESS_DAYS,
            calendar=Calendar(name="weekends"),
        )


def test_dated_business_days_counts_calendar_days_over_year_length():
    cal = Calendar(name="weekends")
    leg = _leg(
        schedule=_dated_quarterly_schedule_1y(),
        day_count=DayCountConvention.BUSINESS_DAYS,
        calendar=cal,
        bus_days_in_year=250,
    )
    counts = [
        cal.count_business_days(s, e, include_start=False, include_end=True)
        for s, e in zip(_QUARTER_DATES[:-1], _QUARTER_DATES[1:])
    ]
    pv = leg.value(EventDistribution.trivial(1.0), _env(), 0.0)
    assert pv == pytest.approx(_annuity([c / 250.0 for c in counts]), rel=1e-12)


def test_calendar_on_a_leg_that_never_reads_it_is_refused():
    with pytest.raises(ValidationError, match="calendar"):
        _leg(calendar=Calendar(name="weekends"), bus_days_in_year=252)
    with pytest.raises(ValidationError, match="calendar"):
        _leg(
            schedule=_dated_quarterly_schedule_1y(),
            day_count=DayCountConvention.ACT_360,
            calendar=Calendar(name="weekends"),
        )


@pytest.mark.parametrize(
    "day_count",
    [
        DayCountConvention.ACT_ACT_ISMA,
        DayCountConvention.ACT_ACT_BOND,
        DayCountConvention.ACT_ACT_CHINABOND,
    ],
)
def test_dated_coupon_frequency_conventions_are_refused(day_count):
    with pytest.raises(ValidationError, match="frequency"):
        _leg(schedule=_dated_quarterly_schedule_1y(), day_count=day_count)
