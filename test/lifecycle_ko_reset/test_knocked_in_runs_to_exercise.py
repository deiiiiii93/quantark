"""The tracker keeps a knocked-in KO-reset alive to the product's exercise date (owner ruling 2026-11-27, run 9, D5).

The owner's contract rule: a knocked-in KO-reset observes its schedule-2 knock-outs up to schedule 2's last observation;
if it has not knocked out it is a LIVE put to the PRODUCT'S EXERCISE DATE and pays put(S at exercise) at the exercise
settlement (``resolve_terminal_timing``: the explicit ``settlement_date`` when booked, else the exercise date plus the
settlement delay). The tracker follows the engines:

- a knocked-in note stays alive past schedule 2's last observation (no knock-out is tested after it) and books V1 on
  the exercise close's spot, determined at the exercise date and paid at the terminal payment;
- ``terminal_date`` / ``matured_before_run`` read that terminal: a carried knock-in retires only past the exercise;
- the run calendar (``resolve_calendar_schedule``) and the simulated paths place V1 there;
- a REBASED contract's ABSOLUTE copy (``with_absolute_schedule_2``) keeps the original exercise date and settlement
  date; a copy whose schedule 2 would end after the exercise (a knock-in on a close later than its record) is refused.
"""
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, LifecycleEventType as E
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env, schedule
from .test_dated_rebased_schedule import GENERATED, KI_DATES, LATEST_END, dated_rebased
from .test_tracker_rules import day, run, tracker

D = 1 / 365.0
#: schedule 2 (ABSOLUTE) ends on day 365; the exercise date is day 400
POST_END, EXERCISE = 365, 400


def on(n):
    return (START + pd.Timedelta(days=n)).to_pydatetime()


def note(mode=M.ABSOLUTE, *, exercise=EXERCISE, post=(273, POST_END), dated=False, annualized=False, disable=False,
         settlement_date=None, lag=None):
    """Schedule 1 (103, 15%) and the discrete KI (75) on days 91 and 182, schedule 2 (95, 3%) on ``post`` (days;
    offsets from the knock-in close under REBASED), exercise on day ``exercise``; 5% rebate, no principal."""
    if dated:
        sched = lambda days, level: ObservationSchedule(  # noqa: E731
            records=[ObservationRecord(observation_date=on(d), barrier=level) for d in days])
    else:
        sched = lambda days, level: schedule([d * D for d in days], level)  # noqa: E731
    maturity = exercise * D
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None if dated else maturity,
        tenor=maturity if annualized and not dated else None,
        initial_date=START.to_pydatetime() if dated else None, exercise_date=on(exercise) if dated else None,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=sched((91, 182), 103.0), ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=sched((91, 182), 75.0),
            disable_ko_after_ki=disable),
        post_barrier_config=BarrierConfig(
            ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=schedule([d * D for d in post], 95.0) if mode == M.REBASED else sched(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=annualized, is_annualized_ki=annualized,
                                     is_annualized_rebate=annualized),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)
    if settlement_date is not None:
        product.settlement_date = settlement_date
    if lag is not None:
        product.settlement_convention = SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    product.validate()
    return product


def carried(product, ki_day=91):
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", day(ki_day).to_pydatetime())
    return product


# --- the tracker ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dated", [False, True], ids=["numeric", "dated"])
def test_a_knocked_in_note_lives_past_schedule_two_and_books_the_put_at_exercise(dated):
    """Knocked in on day 91, not knocked out on schedule 2 (90 < 95 on days 273 and 365): alive past day 365, matures
    on day 400 on the exercise close's spot (60), V1 = min(60 - 100, 0); the day-365 spot (50) is not read."""
    t = tracker(note(dated=dated))
    closes = {91: 70.0, 273: 90.0, POST_END: 50.0, POST_END + 1: 80.0, EXERCISE - 1: 85.0, EXERCISE: 60.0}
    assert run(t, closes) == [(91, E.KNOCK_IN, None), (EXERCISE, E.MATURITY, "maturity")]
    cash = t.lifecycle.ledger.cashflows[-1]
    assert cash.amount == pytest.approx(-40.0, abs=1e-12)
    terminal = resolve_terminal_timing(t.contract(), env(60.0, START))
    if dated:
        assert pd.Timestamp(cash.determination_date).normalize() == day(EXERCISE)
        assert pd.Timestamp(cash.payment_date).normalize() == pd.Timestamp(terminal.payment_date).normalize()
    else:
        assert cash.determination_time == pytest.approx(EXERCISE * D, abs=1e-12)
        assert cash.payment_time == pytest.approx(float(terminal.payment_time), abs=1e-12)


def test_a_knocked_in_note_is_alive_between_schedule_twos_end_and_the_exercise():
    t = tracker(note())
    run(t, {91: 70.0, 273: 90.0, POST_END: 90.0, EXERCISE - 1: 90.0})
    assert t.lifecycle.alive and t.lifecycle.knocked_in and not t.lifecycle.ledger.cashflows
    assert t.terminal_date(env(90.0, day(EXERCISE - 1))) == day(EXERCISE)


def test_the_terminal_date_is_schedule_ones_end_until_a_knock_in_then_the_exercise():
    t = tracker(note())
    assert t.terminal_date(env()) == day(182)
    run(t, {91: 70.0})
    assert t.terminal_date(env(70.0, day(91))) == day(EXERCISE)


@pytest.mark.parametrize("carry", [False, True], ids=["fresh", "carried"])
def test_disable_ko_after_ki_books_the_put_at_exercise(carry):
    product = note(post=(91, POST_END), disable=True)
    if carry:
        assert run(tracker(carried(product)), {91: 99.0, POST_END: 99.0, EXERCISE: 70.0}) == [
            (EXERCISE, E.MATURITY, "maturity")]
    else:
        t = tracker(product)
        assert run(t, {91: 72.0, POST_END: 99.0, EXERCISE: 70.0}) == [
            (91, E.KNOCK_IN, None), (EXERCISE, E.MATURITY, "maturity")]
        assert t.lifecycle.ledger.cashflows[-1].amount == pytest.approx(-30.0, abs=1e-12)


def test_the_put_is_paid_on_the_booked_settlement_date():
    settlement = on(EXERCISE + 7)
    t = tracker(note(dated=True, settlement_date=settlement))
    run(t, {91: 70.0, POST_END: 90.0, EXERCISE: 60.0})
    cash = t.lifecycle.ledger.cashflows[-1]
    assert pd.Timestamp(cash.determination_date).normalize() == day(EXERCISE)
    assert pd.Timestamp(cash.payment_date).normalize() == pd.Timestamp(settlement)


def test_the_put_is_paid_after_the_settlement_lag():
    t = tracker(note(lag=2 * D))
    run(t, {91: 70.0, POST_END: 90.0, EXERCISE: 60.0})
    cash = t.lifecycle.ledger.cashflows[-1]
    assert cash.determination_time == pytest.approx(EXERCISE * D, abs=1e-12)
    assert cash.payment_time == pytest.approx((EXERCISE + 2) * D, abs=1e-12)


def test_an_annualized_put_accrues_to_the_exercise():
    """O2: the knocked-in amount accrues from the origin to the exercise date (400 days), not to schedule 2's end."""
    t = tracker(note(annualized=True))
    run(t, {91: 70.0, POST_END: 90.0, EXERCISE: 60.0})
    assert t.final_contract_tenor(env()) == pytest.approx(EXERCISE * D, abs=1e-15)
    assert t.lifecycle.ledger.cashflows[-1].amount == pytest.approx(-40.0 * EXERCISE * D, rel=1e-12)


# --- matured before the run ------------------------------------------------------------------------------------------

def _carried_tracker(start_day):
    return KOResetLifecycleTracker(product=carried(note(dated=True)), quantity=1.0, start_date=day(start_day))


def test_a_carried_knock_in_past_schedule_twos_end_has_not_matured_before_the_run():
    t = _carried_tracker(380)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not t.matured_before_run(env(90.0, day(380)))
        assert not t.retire_if_matured_before_run(env(90.0, day(380)))
    assert t.lifecycle.alive


def test_a_carried_knock_in_matures_on_a_run_starting_on_the_exercise():
    t = _carried_tracker(EXERCISE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run(t, {EXERCISE: 60.0}) == [(EXERCISE, E.MATURITY, "maturity")]
    assert t.lifecycle.ledger.cashflows[-1].amount == pytest.approx(-40.0, abs=1e-12)


def test_a_carried_knock_in_past_the_exercise_matured_before_the_run():
    t = _carried_tracker(EXERCISE + 1)
    assert t.matured_before_run(env(90.0, day(EXERCISE + 1)))
    with pytest.warns(MaturedBeforeRunWarning):
        assert t.retire_if_matured_before_run(env(90.0, day(EXERCISE + 1)))
    assert not t.lifecycle.alive and not t.lifecycle.ledger.cashflows



# --- REBASED: the ABSOLUTE copy keeps the exercise -------------------------------------------------------------------

def test_a_numeric_rebased_copy_keeps_the_exercise():
    """REBASED schedule 2 at offsets (91, 365) from the knock-in close; the exercise is day 547 (the latest end, a
    knock-in on day 182). Knocked in on day 91: schedule 2 on days 182 and 456, the copy still ends on day 547."""
    product = note(M.REBASED, exercise=547, post=(91, 365))
    t = tracker(product)
    run(t, {91: 70.0})
    copy = t.contract()
    assert copy.post_ko_mode == M.ABSOLUTE
    assert [r.observation_time for r in copy.post_barrier_config.ko_observation_schedule.records] == [
        pytest.approx(182 * D), pytest.approx(456 * D)]
    assert copy.maturity == product.maturity == pytest.approx(547 * D)
    aged = t.product_for_pricing(day(100), env(90.0, day(100)))
    assert aged.maturity == pytest.approx((547 - 100) * D)


@pytest.mark.parametrize("ki_day", [91, 182])
def test_a_knocked_in_rebased_note_books_the_put_at_exercise(ki_day):
    t = tracker(note(M.REBASED, exercise=547, post=(91, 365), annualized=True))
    end = ki_day + 365
    closes = {91: 90.0, ki_day: 70.0, end: 90.0, end + 1: 80.0, 546: 85.0, 547: 60.0}
    events = run(t, closes)
    assert events[0] == (ki_day, E.KNOCK_IN, None) and events[-1] == (547, E.MATURITY, "maturity")
    assert len(events) == 2
    assert t.lifecycle.ledger.cashflows[-1].amount == pytest.approx(-40.0 * 547 * D, rel=1e-12)


def test_a_dated_rebased_copy_keeps_the_exercise_and_the_settlement_date():
    settlement = datetime(2026, 10, 13)
    product = dated_rebased()
    product.settlement_date = settlement
    product.validate()
    anchor = knock_in_anchor_for(product.barrier_config.ki_observation_schedule.records[0], 0)
    converted = product.with_absolute_schedule_2(KI_DATES[0], anchor=anchor)
    assert [r.observation_date for r in converted.post_barrier_config.ko_observation_schedule.records] == list(
        GENERATED[0])
    assert converted.exercise_date == LATEST_END == product.exercise_date
    assert converted.settlement_date == settlement


def test_a_numeric_rebased_copy_keeps_the_settlement_convention():
    product = note(M.REBASED, exercise=547, post=(91, 365), lag=2 * D)
    t = tracker(product)
    run(t, {91: 70.0, 456: 90.0, 547: 60.0})
    cash = t.lifecycle.ledger.cashflows[-1]
    assert cash.determination_time == pytest.approx(547 * D, abs=1e-12)
    assert cash.payment_time == pytest.approx(549 * D, abs=1e-12)


def test_a_rebased_copy_whose_schedule_two_would_pass_the_exercise_is_refused():
    """A knock-in close later than its record (a record due on a non-trading day fires on the next close) places
    schedule 2 from that close: knocked in on day 184 against the day-182 record, schedule 2 would end on day 549,
    after the exercise (547). The contract does not say what such a note is (owner ruling pending): refused."""
    product = note(M.REBASED, exercise=547, post=(91, 365))
    with pytest.raises(ValidationError, match="after the exercise"):
        product.with_absolute_schedule_2(184 * D, anchor=knock_in_anchor_for(
            product.barrier_config.ki_observation_schedule.records[1], 1))


# --- the run calendar and the simulated paths ------------------------------------------------------------------------

DATES = pd.date_range(START, periods=600, freq="D")


def test_the_calendar_places_the_knocked_in_maturity_at_the_exercise():
    t = tracker(note())
    calendar = t.resolve_calendar_schedule(DATES, env())
    assert calendar.absolute.final_due_day == EXERCISE
    assert calendar.terminal_due_day == EXERCISE
    assert calendar.absolute.ko_due_day.tolist() == [273, POST_END]


@pytest.mark.parametrize("ki_day", [91, 182])
def test_the_calendar_places_every_rebased_knock_in_days_maturity_at_the_exercise(ki_day):
    t = tracker(note(M.REBASED, exercise=547, post=(91, 365)))
    calendar = t.resolve_calendar_schedule(DATES, env())
    placement = calendar.placement(ki_day)
    assert placement.final_due_day == 547
    assert placement.ko_due_day.tolist() == [ki_day + 91, ki_day + 365]
    assert calendar.terminal_due_day == 547


def test_the_calendar_flags_a_carried_knock_in_past_the_exercise_as_matured():
    def calendar(start_day):
        t = _carried_tracker(start_day)
        dates = pd.date_range(day(start_day), periods=40, freq="D")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MaturedBeforeRunWarning)
            return t.resolve_calendar_schedule(dates, env(90.0, day(start_day)))
    assert not calendar(380).matured_before_run
    assert calendar(EXERCISE + 1).matured_before_run


def test_the_simulated_paths_mature_a_knocked_in_path_at_the_exercise():
    from quantark.backtest.simulation.ko_reset_lifecycle import KOResetVectorLifecycle

    t = tracker(note())
    schedule_ = t.resolve_calendar_schedule(DATES[:420], env())
    lifecycle = KOResetVectorLifecycle([schedule_], np.array([1.0]), 1)
    path = {91: 70.0, 273: 90.0, POST_END: 50.0, EXERCISE: 60.0}
    booked = []

    def payoff(p, paths, knocked_in, tenors):
        booked.append(bool(knocked_in[0]))
        return np.minimum(np.array([path_spot] * paths.size) - 100.0, 0.0)

    records = []
    for d in range(420):
        path_spot = path.get(d, 90.0)
        records += lifecycle.step(d, np.array([path_spot]), payoff)
    maturity = [r for r in records if r.event == "maturity"]
    assert [r.day for r in maturity] == [EXERCISE] and booked == [True]
    assert maturity[0].cashflow == pytest.approx(-40.0, abs=1e-12)
