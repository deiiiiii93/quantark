"""The run lifecycle tracker decides a DISCRETE barrier only on its schedule's observation dates (A15, user decision
2026-10-14: "knock out should only happen at the observation date, otherwise price it"; T17 concern 4).

``BarrierLifecycleTracker`` (the equity backtest and the dynamic scenario, via ``PortfolioLifecycleManager``) tested
``product.is_barrier_hit`` on every daily close whatever the monitoring. Now:

* DISCRETE: each schedule record is decided once, on its own date at that day's close, against the record's level and
  paying the record's payoff (W18c). A record date the run never visits is decided at the spot of the run instant
  before it, flagged ``gap=True`` (A7'(c), the signed W6 rule, as the DCN tracker). A record before the run start is
  history (signed rule 3: no barrier state is assumed). A numeric record falls on ``start + round(t * 365)`` days (the
  run clock every tracker uses). A spot beyond the level between records decides nothing; at expiry only a record on
  the expiry date can hit.
* CONTINUOUS (every close) and EXPIRY monitoring: unchanged.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import BarrierLifecycleTracker, LifecycleEventType
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleOneTouchOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierDirection, BarrierType, DoubleBarrierType, ObservationType, OptionType, TouchType

START = pd.Timestamp("2026-01-05")
#: monthly record dates (contract days 30, 60, ..., 360), the exercise date on the last one
RECORD_DAYS = [30 * k for k in range(1, 13)]
EXERCISE = (START + pd.Timedelta(days=360)).to_pydatetime()


def env(spot: float, day: int = 0) -> PricingEnvironment:
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.03),
                              valuation_date=(START + pd.Timedelta(days=day)).to_pydatetime())


def _on(day: int) -> pd.Timestamp:
    return START + pd.Timedelta(days=day)


def _records(level=None, payoff=None, upper=None, lower=None, days=RECORD_DAYS):
    kw = {}
    if payoff is not None:
        kw["payoff"] = payoff
    out = []
    for d in days:
        date = _on(d).to_pydatetime()
        if upper is not None:
            out.append(ObservationRecord(observation_date=date, upper_barrier=upper, lower_barrier=lower, **kw))
        else:
            out.append(ObservationRecord(observation_date=date, barrier=level, **kw))
    return ObservationSchedule(records=out)


def _up_out(level=120.0, record_level=120.0, payoff=None, observation_type=ObservationType.DISCRETE, quantity=1.0,
            days=RECORD_DAYS, barrier_type=BarrierType.UP_OUT):
    schedule = _records(record_level, payoff, days=days) if observation_type == ObservationType.DISCRETE else None
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=level, barrier_type=barrier_type,
                            rebate=5.0, exercise_date=EXERCISE, observation_type=observation_type,
                            observation_schedule=schedule)
    return BarrierLifecycleTracker(product=product, quantity=quantity, start_date=START)


def _run(tracker, path):
    """``path``: (contract day, spot) per run instant; the events of each close, in order."""
    out = []
    for day, spot in path:
        out.extend(tracker.observe(_on(day), env(spot, day), spot))
    return out


def test_crossing_between_records_and_back_does_not_knock_out():
    tracker = _up_out()
    events = _run(tracker, [(0, 100.0), (40, 125.0), (45, 130.0), (55, 118.0), (60, 119.0), (61, 125.0)])
    assert events == []
    assert tracker.state.alive and not tracker.state.knocked_out


def test_beyond_on_a_record_date_knocks_out_that_day_at_the_record_payoff():
    """Records at 115 paying 4.5 (``product.barrier`` 120, rebate 5): spot 117 on the day-60 record knocks out there,
    at the record's level and payoff (W18c), times the contract multiplier and the quantity."""
    tracker = _up_out(record_level=115.0, payoff=4.5, quantity=3.0)
    events = _run(tracker, [(0, 100.0), (59, 117.0), (60, 117.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
    ko = events[0]
    assert ko.date == _on(60)
    assert ko.barrier == 115.0
    assert ko.payoff == pytest.approx(4.5 * tracker.product.contract_multiplier, rel=1e-15)
    assert ko.cashflow == pytest.approx(3.0 * ko.payoff, rel=1e-15)
    assert ko.metadata["gap"] is False
    assert not tracker.state.alive and tracker.state.knocked_out


def test_a_record_the_run_skips_is_decided_at_the_previous_instants_spot():
    """The run visits days 55 and 65 only: the day-60 record is decided at day 55's close (``gap=True``)."""
    tracker = _up_out()
    events = _run(tracker, [(0, 100.0), (55, 121.0), (65, 110.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
    assert events[0].date == _on(60)
    assert events[0].spot == 121.0
    assert events[0].metadata["gap"] is True
    assert events[0].metadata["decided_on"] == _on(65)

    untouched = _up_out()
    assert _run(untouched, [(0, 100.0), (55, 110.0), (65, 125.0)]) == []    # day 60 at 110: not hit
    assert untouched.state.alive


def test_a_record_before_the_run_start_is_history():
    """Booked at day -30 with a record on day -10: a run starting on day 0 above the level decides nothing."""
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                            rebate=5.0, exercise_date=EXERCISE, observation_type=ObservationType.DISCRETE,
                            observation_schedule=_records(120.0, days=[-10] + RECORD_DAYS))
    tracker = BarrierLifecycleTracker(product=product, quantity=1.0, start_date=START)
    assert _run(tracker, [(0, 125.0), (10, 125.0)]) == []
    assert tracker.state.alive


def test_a_record_after_the_start_but_before_the_first_run_instant_is_history():
    """Ruling R31: "before the run" is before the FIRST RUN INSTANT. A run whose first market day (40) is after the
    tracker's start, with a record (30) in between, completes: that record is history (no state assumed, not hit),
    whatever the spot; the next record decides as usual."""
    tracker = _up_out()
    assert _run(tracker, [(40, 125.0), (45, 110.0)]) == []
    assert tracker.state.alive
    events = _run(tracker, [(60, 121.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT] and events[0].date == _on(60)


def test_a_record_skipped_after_the_first_run_instant_still_takes_the_previous_instants_spot():
    """The gap rule holds strictly after the first run instant (R31)."""
    tracker = _up_out()
    events = _run(tracker, [(40, 121.0), (65, 100.0)])           # day 60 skipped, decided at day 40's 121
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
    assert events[0].date == _on(60) and events[0].metadata["gap"] is True


@pytest.mark.parametrize("double", [False, True], ids=["one_touch", "double_one_touch"])
def test_a_one_touch_with_a_zero_rebate_books_the_record_payoff(double):
    """Product rebate 0, records paying 4.5: a touch on a record day books 4.5 (a one-touch's touch cash carries no
    contract multiplier, as ``get_payoff(touched=True)``), not 0."""
    if double:
        product = DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=75.0, rebate=0.0, exercise_date=EXERCISE,
                                       payment_at_hit=False, touch_type=TouchType.DOUBLE_ONE_TOUCH,
                                       observation_type=ObservationType.DISCRETE,
                                       observation_schedule=_records(upper=130.0, lower=75.0, payoff=4.5))
        spot = 131.0
    else:
        product = OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=0.0,
                                 exercise_date=EXERCISE, payment_at_hit=False, touch_type=TouchType.ONE_TOUCH,
                                 observation_type=ObservationType.DISCRETE,
                                 observation_schedule=_records(120.0, payoff=4.5))
        spot = 121.0
    tracker = BarrierLifecycleTracker(product=product, quantity=2.0, start_date=START)
    events = _run(tracker, [(0, 100.0), (30, spot)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]
    assert events[0].payoff == 4.5 and events[0].cashflow == 9.0


def test_numeric_records_fall_on_the_run_clock():
    """A numeric record at t lies on ``start + round(t * 365)`` days."""
    times = [d / 365.0 for d in RECORD_DAYS]
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                            rebate=5.0, maturity=360 / 365.0, observation_type=ObservationType.DISCRETE,
                            observation_schedule=ObservationSchedule(records=[
                                ObservationRecord(observation_time=t, barrier=120.0) for t in times]))
    tracker = BarrierLifecycleTracker(product=product, quantity=1.0, start_date=START)
    assert _run(tracker, [(0, 100.0), (29, 125.0)]) == []
    events = _run(tracker, [(30, 125.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT] and events[0].date == _on(30)


def test_beyond_at_expiry_without_a_record_there_pays_the_payoff():
    """Records end on day 330; the exercise date (day 360) holds none: a spot beyond the level there settles the
    vanilla payoff, not a knock-out."""
    tracker = _up_out(days=RECORD_DAYS[:-1])
    events = _run(tracker, [(0, 100.0), (330, 110.0), (360, 125.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.EXPIRY]
    assert events[0].payoff == pytest.approx(tracker.product.get_payoff(125.0), rel=1e-15)

    on_record = _up_out()
    events = _run(on_record, [(0, 100.0), (330, 110.0), (360, 125.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT]


def test_knock_in_only_on_a_record():
    tracker = _up_out(barrier_type=BarrierType.UP_IN)
    assert _run(tracker, [(0, 100.0), (40, 125.0), (59, 110.0), (60, 110.0)]) == []
    assert not tracker.state.knocked_in
    events = _run(tracker, [(89, 110.0), (90, 121.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_IN] and events[0].date == _on(90)
    assert tracker.state.knocked_in and tracker.state.alive


def test_continuous_monitoring_still_decides_every_close():
    tracker = _up_out(observation_type=ObservationType.CONTINUOUS)
    events = _run(tracker, [(0, 100.0), (40, 125.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT] and events[0].date == _on(40)


def _family():
    up, corridor = _records(120.0), _records(upper=130.0, lower=75.0)
    return {
        "double_barrier": DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0,
                                              lower_barrier=75.0, barrier_type=DoubleBarrierType.KNOCK_OUT,
                                              rebate=2.0, exercise_date=EXERCISE,
                                              observation_type=ObservationType.DISCRETE, observation_schedule=corridor),
        "single_sharkfin": SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                                knock_out_rebate=3.0, exercise_date=EXERCISE,
                                                observation_type=ObservationType.DISCRETE, observation_schedule=up),
        "double_sharkfin": DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0,
                                                lower_barrier=75.0, knock_out_rebate=3.0, exercise_date=EXERCISE,
                                                observation_type=ObservationType.DISCRETE,
                                                observation_schedule=_records(upper=130.0, lower=75.0)),
        "one_touch": OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=5.0,
                                    exercise_date=EXERCISE, payment_at_hit=False, touch_type=TouchType.ONE_TOUCH,
                                    observation_type=ObservationType.DISCRETE, observation_schedule=_records(120.0)),
        "double_one_touch": DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=75.0, rebate=5.0,
                                                 exercise_date=EXERCISE, payment_at_hit=False,
                                                 touch_type=TouchType.DOUBLE_ONE_TOUCH,
                                                 observation_type=ObservationType.DISCRETE,
                                                 observation_schedule=_records(upper=130.0, lower=75.0)),
    }


@pytest.mark.parametrize("name", list(_family()))
def test_every_family_member_is_decided_only_on_its_records(name):
    product = _family()[name]
    tracker = BarrierLifecycleTracker(product=product, quantity=1.0, start_date=START)
    assert _run(tracker, [(0, 100.0), (40, 135.0), (50, 70.0), (59, 100.0), (60, 100.0)]) == []
    assert tracker.state.alive
    events = _run(tracker, [(89, 100.0), (90, 135.0)])
    assert [e.event_type for e in events] == [LifecycleEventType.KNOCK_OUT] and events[0].date == _on(90)
