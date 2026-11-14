"""The KO-reset tracker books a note that ends with schedule 1 on that note's payment (spec D30, D33).

A schedule-1 knock-out happens only on a path never knocked in, whose note ends at schedule 1's maturity. Under EXPIRY
pay the engines pay it with the never-knocked-in maturity (D30): on the explicit settlement date when one is booked,
else at schedule 1's maturity plus the settlement delay (D33). The tracker books it there -- it used to book the
record's own settlement date, as under INSTANT pay -- and the never-knocked-in maturity on the same date. The
simulated-path calendar schedule places the same settlement day and delay, and exact mode still equals the replay.
"""
from dataclasses import replace
from datetime import timedelta

import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, LifecycleEventType as E
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum import CouponPayType
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode

from .conftest import START, env, schedule

D = 1 / 365.0
PRE, POST = (91, 182), (273, 365)


def _dates(days, level):
    return ObservationSchedule(records=[
        ObservationRecord(observation_date=(START + pd.Timedelta(days=d)).to_pydatetime(), barrier=level) for d in days])


def ko_reset(pay, *, dated=False, settle_days=None, lag=0.0):
    """Schedule 1 (KO 103, 15%; KI 75) on days 91 and 182, schedule 2 (KO 95, 3%) on 273 and 365; 5% rebate, no
    principal. ``dated``: on dates from START (``settle_days``: an explicit settlement date that many days after
    START); else on the numeric clock (``lag``: a year-fraction settlement convention)."""
    sched = (lambda days, level: _dates(days, level)) if dated else (lambda days, level: schedule([d * D for d in days],
                                                                                                  level))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None if dated else POST[-1] * D,
        initial_date=START.to_pydatetime() if dated else None,
        exercise_date=(START + pd.Timedelta(days=POST[-1])).to_pydatetime() if dated else None,
        settlement_date=(START + pd.Timedelta(days=settle_days)).to_pydatetime() if settle_days else None,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=sched(PRE, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=sched(PRE, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=sched(POST, 95.0)),
        accrual_config=AccrualConfig(coupon_pay_type=pay),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        settlement_convention=(SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)
                               if lag else None),
    )


def _day(n):
    return START + pd.Timedelta(days=n)


def _book(product, closes):
    """{day: spot} observed as PortfolioLifecycleManager does; the terminating event."""
    tracker = KOResetLifecycleTracker(product=product, quantity=1.0, start_date=START)
    for n, spot in sorted(closes.items()):
        e = env(spot, _day(n))
        current = tracker.product_for_lifecycle()
        events = tracker.observe(_day(n), current, e, spot)
        maturity = tracker.settle_maturity_if_due(_day(n), current, e, spot)
        events += [maturity] if maturity is not None else []
        for event in events:
            if event.terminates_position:
                return event
    raise AssertionError("nothing terminated the note")


def test_numeric_expiry_schedule_one_knock_out_is_booked_at_schedule_one_maturity():
    event = _book(ko_reset(CouponPayType.EXPIRY), {91: 104.0})
    assert event.event_type is E.KNOCK_OUT
    assert event.realized_cashflow.payment_time == pytest.approx(PRE[-1] * D, abs=1e-12)


def test_numeric_expiry_schedule_one_knock_out_carries_the_settlement_delay():
    event = _book(ko_reset(CouponPayType.EXPIRY, lag=0.01), {91: 104.0})
    assert event.realized_cashflow.payment_time == pytest.approx(PRE[-1] * D + 0.01, abs=1e-12)


def test_instant_knock_outs_keep_their_own_settlement():
    event = _book(ko_reset(CouponPayType.INSTANT, lag=0.01), {91: 104.0})
    assert event.realized_cashflow.payment_time == pytest.approx(PRE[0] * D + 0.01, abs=1e-12)


def test_dated_expiry_schedule_one_knock_out_is_booked_on_schedule_one_maturity_date():
    event = _book(ko_reset(CouponPayType.EXPIRY, dated=True), {91: 104.0})
    assert pd.Timestamp(event.realized_cashflow.payment_date) == _day(PRE[-1])
    assert pd.Timestamp(event.metadata["settlement_date"]) == _day(PRE[-1])


@pytest.mark.parametrize("pay", [CouponPayType.EXPIRY, CouponPayType.INSTANT])
def test_a_settlement_date_pays_the_note_that_ends_with_schedule_one(pay):
    """D33: the never-knocked-in maturity (both pay types) and an EXPIRY schedule-1 knock-out are booked on the
    explicit settlement date; an INSTANT knock-out keeps its own record's settlement."""
    settle = POST[-1] + 5
    maturity = _book(ko_reset(pay, dated=True, settle_days=settle), {91: 100.0, 182: 100.0})
    assert maturity.event_type is E.MATURITY
    assert pd.Timestamp(maturity.realized_cashflow.payment_date) == _day(settle)
    knock_out = _book(ko_reset(pay, dated=True, settle_days=settle), {91: 104.0})
    expected = _day(settle) if pay is CouponPayType.EXPIRY else _day(PRE[0])
    assert pd.Timestamp(knock_out.realized_cashflow.payment_date) == expected


@pytest.mark.parametrize("dated, settle_days", [(False, None), (True, None), (True, POST[-1] + 5)])
def test_the_calendar_schedule_settles_an_expiry_schedule_one_knock_out_with_schedule_one(dated, settle_days):
    """The simulated paths book from the calendar schedule: under EXPIRY every schedule-1 knock-out settles on V0's day
    with V0's payment (delay measured from its own record)."""
    product = ko_reset(CouponPayType.EXPIRY, dated=dated, settle_days=settle_days)
    tracker = KOResetLifecycleTracker(product=product, quantity=1.0, start_date=START)
    dates = pd.date_range(START, START + pd.Timedelta(days=POST[-1] + 10), freq="D")
    cal = tracker.resolve_calendar_schedule(dates, env(100.0, START))
    assert list(cal.ko_settlement_day) == [cal.pre_settlement_day] * len(PRE)
    pre_payment = PRE[-1] * D + cal.pre_settlement_delay
    for i, day in enumerate(PRE):
        assert cal.ko_settlement_delay[i] == pytest.approx(pre_payment - day * D, abs=1e-12)
