import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime

import numpy as np
import pytest

from cashleg.accrual_leg import AccrualLeg, KOBehavior, PaymentConvention, SurvivalBasis
from cashleg.base import LegDirection
from cashleg.base_amount import BaseAmount, BaseAmountMode
from cashleg.event_distribution import EventDistribution, EventType
from cashleg.leg_schedule import LegSchedule
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.calendar.day_counter import DayCountConvention


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
