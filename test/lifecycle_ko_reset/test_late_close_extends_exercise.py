"""A late-close REBASED knock-in extends the copy's exercise (owner ruling 2026-11-28, "extend exercise date"; run 10).

The tracker hangs a REBASED schedule 2 from the knock-in CLOSE. A knock-in record due on a non-trading day fires on
the next close, so schedule 2 can end after the product's exercise date. Run 9 refused that copy. Now the copy's
exercise date (numeric: ``maturity``) moves to schedule 2's last observation (D1), and its settlement follows (D2):

- no explicit ``settlement_date``: the settlement convention applies to the new exercise by itself;
- an explicit ``settlement_date``: shifted by the booked exercise -> settlement lag in business days on the contract's
  settlement calendar (``settlement_convention.calendar``); refused when the contract names none.

The extended copy is a valid ABSOLUTE note (O1, D35), reads put(S at the extended exercise) and pays at its terminal.
The knocked-in tenor (O2) runs to the extended exercise (D3). The simulated-path calendar holds the latest placement
(D4). Copies whose schedule 2 ends on or before the exercise are unchanged.
"""
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, LifecycleEventType as E
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar import Calendar
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env
from .test_dated_rebased_schedule import dated_rebased
from .test_knocked_in_runs_to_exercise import D, note
from .test_tracker_rules import run, tracker

# --- the dated late-close note ---------------------------------------------------------------------------------------
# knock-in records Mon 2026-04-06 and Sat 2026-07-04 (schedule 1 on the same dates), schedule 2 at (1, 3) months from
# the knock-in, exercise Mon 2026-10-05 (the booked latest end is 07-04 + 3 months = 10-04). The Saturday record fires
# on the Monday 07-06 close, which places schedule 2 on 08-06 and 10-06: past the exercise by one day.
KI_DATES = [datetime(2026, 4, 6), datetime(2026, 7, 4)]
LATE_CLOSE = datetime(2026, 7, 6)
EXERCISE = datetime(2026, 10, 5)
EXTENDED = datetime(2026, 10, 6)
#: weekends only, plus Wed 2026-10-07: a business-day lag counted on it skips the holiday
CALENDAR = Calendar(holidays={datetime(2026, 10, 7)}, name="late-close")


def _day(date):
    return (pd.Timestamp(date) - START).days


def late_note(*, exercise=EXERCISE, **kw):
    return dated_rebased(ki_dates=KI_DATES, exercise_date=exercise, **kw)


def _anchor(product, index=1):
    return knock_in_anchor_for(product.barrier_config.ki_observation_schedule.records[index], index)


def _valid(copy):
    """The copy passes the product's own validation (O1, D35, date ordering) and keeps its dates doing so."""
    again = deepcopy(copy)
    again.validate()
    assert again.exercise_date == copy.exercise_date and again.settlement_date == copy.settlement_date
    assert again.maturity == copy.maturity
    return copy


# --- AC1: dated ------------------------------------------------------------------------------------------------------

def test_a_dated_late_close_copy_extends_its_exercise_to_schedule_twos_last_date():
    product = late_note()
    copy = _valid(product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product)))
    assert copy.post_ko_mode is M.ABSOLUTE
    assert [r.observation_date for r in copy.post_barrier_config.ko_observation_schedule.records] == [
        datetime(2026, 8, 6), EXTENDED]
    assert copy.exercise_date == EXTENDED
    assert product.exercise_date == EXERCISE                      # the contract itself is untouched


def test_a_dated_extension_keeps_the_exercise_time_of_day():
    product = late_note(exercise=datetime(2026, 10, 5, 15, 0))
    copy = _valid(product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product)))
    assert copy.exercise_date == datetime(2026, 10, 6, 15, 0)


def test_a_dated_extension_settles_by_the_convention():
    """No explicit settlement date: the convention (2 business days) applies to the new exercise, Tue 10-06 -> Fri
    10-09 (the holiday is Wed 10-07)."""
    convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=CALENDAR)
    product = late_note(settlement_convention=convention)
    copy = _valid(product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product)))
    kept = copy.settlement_convention                   # deep-copied with the contract (Calendar has no __eq__)
    assert copy.settlement_date is None and (kept.lag, kept.lag_unit) == (2.0, SettlementLagUnit.BUSINESS_DAYS)
    assert kept.calendar.identity() == CALENDAR.identity()
    terminal = resolve_terminal_timing(copy, env(100.0, START))
    assert terminal.determination_date == EXTENDED
    assert terminal.payment_date == datetime(2026, 10, 9)


def test_a_dated_extension_shifts_an_explicit_settlement_date_by_its_business_day_lag():
    """Booked: exercise Mon 10-05, settlement Fri 10-09 = 3 business days on the contract calendar (Wed 10-07 is a
    holiday). The extended exercise Tue 10-06 + 3 business days = Mon 10-12. The convention's own lag (2) is not the
    booked lag and is not read."""
    convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=CALENDAR)
    product = late_note(settlement_convention=convention, settlement_date=datetime(2026, 10, 9))
    copy = _valid(product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product)))
    assert copy.exercise_date == EXTENDED
    assert copy.settlement_date == datetime(2026, 10, 12)
    terminal = resolve_terminal_timing(copy, env(100.0, START))
    assert (terminal.determination_date, terminal.payment_date) == (EXTENDED, datetime(2026, 10, 12))


def test_a_dated_extension_keeps_the_settlement_time_of_day():
    convention = SettlementConvention(lag=0, calendar=CALENDAR)
    product = late_note(settlement_convention=convention, settlement_date=datetime(2026, 10, 6, 17, 30))
    copy = _valid(product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product)))
    assert copy.settlement_date == datetime(2026, 10, 8, 17, 30)        # 1 business day after Tue 10-06, past 10-07


def test_a_dated_extension_with_an_explicit_settlement_date_and_no_contract_calendar_is_refused():
    product = late_note(settlement_date=datetime(2026, 10, 9))
    with pytest.raises(ValidationError, match="calendar"):
        product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product))


def test_a_dated_extension_refuses_a_settlement_date_that_is_no_business_day_lag():
    """Sat 10-10 is not a whole business-day lag from the exercise on the contract calendar."""
    convention = SettlementConvention(lag=0, calendar=CALENDAR)
    product = late_note(settlement_convention=convention, settlement_date=datetime(2026, 10, 10))
    with pytest.raises(ValidationError, match="business-day lag"):
        product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product))


def test_a_dated_extension_past_a_booked_maturity_date_is_refused():
    """A maturity_date booking supplies the exercise (10-05); the extended exercise (10-06) would pass it, and moving
    the maturity date is not ruled."""
    product = late_note(exercise=None, maturity_date=EXERCISE)
    with pytest.raises(ValidationError, match="maturity_date"):
        product.with_absolute_schedule_2(LATE_CLOSE, anchor=_anchor(product))


def test_schedule_two_ending_later_on_the_exercise_day_is_refused():
    """A knock-in date carrying a time of day can end schedule 2 later on the exercise day itself; the ruling extends
    the exercise to a later DATE only."""
    product = late_note(exercise=datetime(2026, 10, 5, 15, 0))
    with pytest.raises(ValidationError, match="same day"):
        product.with_absolute_schedule_2(datetime(2026, 7, 5, 16, 0), anchor=_anchor(product))


def test_a_dated_copy_ending_on_or_before_the_exercise_is_unchanged():
    """AC5: a knock-in on its record's own date (Mon 04-06) ends schedule 2 before the exercise: nothing moves."""
    convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=CALENDAR)
    product = late_note(settlement_convention=convention, settlement_date=datetime(2026, 10, 9))
    copy = product.with_absolute_schedule_2(KI_DATES[0], anchor=_anchor(product, 0))
    assert copy.exercise_date == product.exercise_date and copy.settlement_date == product.settlement_date
    assert copy._terminal_accrual["post"] == product._terminal_accrual["post"]


# --- AC2: numeric ----------------------------------------------------------------------------------------------------

def _numeric(**kw):
    """Exercise day 547 = the last knock-in record (day 182) + schedule 2's last offset (365)."""
    return note(M.REBASED, exercise=547, post=(91, 365), **kw)


def test_a_numeric_late_close_copy_extends_its_maturity_to_schedule_twos_last_observation():
    product = _numeric()
    copy = _valid(product.with_absolute_schedule_2(184 * D, anchor=_anchor(product)))
    last = copy.post_barrier_config.ko_observation_schedule.records[-1].observation_time
    assert last == pytest.approx(549 * D, abs=1e-15)
    assert copy.maturity == last
    assert copy.exercise_date is None


def test_a_numeric_extension_settles_by_the_convention():
    product = _numeric(lag=2 * D)
    copy = _valid(product.with_absolute_schedule_2(184 * D, anchor=_anchor(product)))
    terminal = resolve_terminal_timing(copy, env(100.0, START))
    assert terminal.determination_time == copy.maturity
    assert terminal.payment_time == pytest.approx(copy.maturity + 2 * D, abs=1e-15)


def test_a_numeric_extension_runs_the_knocked_in_tenor_to_the_new_maturity():
    """D3: the knocked-in (O2) tenor runs to the copy's maturity, 549 days, not the booked 547."""
    product = _numeric(annualized=True)
    copy = product.with_absolute_schedule_2(184 * D, anchor=_anchor(product))
    assert copy._resolve_post_contract_tenor(env(100.0, START)) == pytest.approx(549 * D, abs=1e-15)
    assert product._resolve_post_contract_tenor(env(100.0, START)) == pytest.approx(547 * D, abs=1e-15)


def test_a_numeric_copy_ending_on_or_before_the_maturity_is_unchanged():
    product = _numeric(lag=2 * D)
    copy = product.with_absolute_schedule_2(182 * D, anchor=_anchor(product))
    assert copy.maturity == product.maturity
    assert copy._terminal_accrual["post"] == product._terminal_accrual["post"]


# --- AC3: the tracker ------------------------------------------------------------------------------------------------

def test_a_numeric_late_close_knock_in_lives_to_the_extended_maturity_and_books_the_put_there():
    """Knocked in on the day-184 close (the day-182 record, no close on 182-183): schedule 2 on days 275 and 549.
    Alive past the booked maturity (547), V1 on the day-549 close (60): min(60 - 100, 0) accrued to 549 days, paid 2
    days later (the settlement lag)."""
    t = tracker(_numeric(annualized=True, lag=2 * D))
    closes = {91: 90.0, 184: 70.0, 275: 90.0, 546: 85.0, 547: 85.0, 548: 80.0, 549: 60.0}
    assert run(t, closes) == [(184, E.KNOCK_IN, None), (549, E.MATURITY, "maturity")]
    cash = t.lifecycle.ledger.cashflows[-1]
    assert cash.amount == pytest.approx(-40.0 * 549 * D, rel=1e-12)
    assert cash.determination_time == pytest.approx(549 * D, abs=1e-12)
    assert cash.payment_time == pytest.approx(551 * D, abs=1e-12)


def test_a_dated_late_close_knock_in_lives_to_the_extended_exercise_and_books_the_put_there():
    convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS, calendar=CALENDAR)
    t = tracker(late_note(settlement_convention=convention, settlement_date=datetime(2026, 10, 9)))
    closes = {91: 90.0, _day(LATE_CLOSE): 70.0, 213: 90.0, _day(EXERCISE): 85.0, _day(EXTENDED): 60.0}
    assert run(t, closes) == [(_day(LATE_CLOSE), E.KNOCK_IN, None), (_day(EXTENDED), E.MATURITY, "maturity")]
    cash = t.lifecycle.ledger.cashflows[-1]
    assert cash.amount == pytest.approx(-40.0, abs=1e-12)
    assert pd.Timestamp(cash.determination_date).normalize() == pd.Timestamp(EXTENDED)
    assert pd.Timestamp(cash.payment_date).normalize() == pd.Timestamp(2026, 10, 12)
    assert t.terminal_date(env(60.0, EXTENDED)) == pd.Timestamp(EXTENDED)


def _carried(start):
    product = late_note()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", LATE_CLOSE)
    setattr(product, "_otc_lifecycle_knock_in_anchor", _anchor(product))
    return KOResetLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(start))


def test_a_carried_late_close_knock_in_retires_only_past_the_extended_exercise():
    import warnings

    on_exercise = _carried(EXERCISE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not on_exercise.matured_before_run(env(90.0, EXERCISE))
        on_extended = _carried(EXTENDED)
        assert not on_extended.matured_before_run(env(90.0, EXTENDED))
        assert run(on_extended, {_day(EXTENDED): 60.0}) == [(_day(EXTENDED), E.MATURITY, "maturity")]
    past = _carried(datetime(2026, 10, 7))
    assert past.matured_before_run(env(90.0, datetime(2026, 10, 7)))
    with pytest.warns(MaturedBeforeRunWarning):
        assert past.retire_if_matured_before_run(env(90.0, datetime(2026, 10, 7)))


# --- AC4: the simulated paths ----------------------------------------------------------------------------------------

def _sim_note(**kw):
    """Numeric REBASED, calendar days from Tue 2024-01-02: schedule 1 on days 2 and 9, knock-in records on day 1 (Wed)
    and day 5 (Sun, due on the Mon day-6 close), schedule 2 at offsets (4, 8); maturity day 13 (5 + 8)."""
    from .test_simulation_ko_reset import short_ko_reset

    return short_ko_reset(M.REBASED, pre=(2, 9), ki=(1, 5), post=(4, 8), **kw)


def _two_paths():
    """Trading-day index -> calendar day: 1 -> 1, 4 -> 6, 9 -> 13 (Mon), 10 -> 14 (Tue). Path 0 knocks in on index 1
    (its record's own close), path 1 on index 4 (the day-5 record's late close). Neither knocks out."""
    from simulation.conftest import SPOT, make_market_path

    path = make_market_path(n_paths=2, n_days=12)
    spot = np.full((2, 12), SPOT * 0.98)
    spot[0, 1] = spot[1, 4] = SPOT * 0.70
    path.spot[:, :] = spot
    return path


def test_the_calendar_places_a_late_close_knock_ins_maturity_at_the_extended_exercise():
    """D4: placements per knock-in day; the late close (index 4) ends at day 14 (index 10), the record's own close
    (index 1) at the booked maturity (day 13, index 9); the calendar's terminal is the latest. A one-day YEAR_FRACTION
    lag settles each a trading day later."""
    product = _sim_note()
    product.settlement_convention = SettlementConvention(lag=D, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    product.validate()
    dates = pd.DatetimeIndex(_two_paths().dates)
    t = KOResetLifecycleTracker(product=product, quantity=1.0, start_date=dates[0])
    calendar = t.resolve_calendar_schedule(dates, env(date=dates[0]))
    on_record, late = calendar.placement(1), calendar.placement(4)
    assert (on_record.final_due_day, on_record.final_settlement_day) == (9, 10)
    assert (late.final_due_day, late.final_settlement_day) == (10, 11)
    assert late.ko_due_day.tolist() == [8, 10]
    assert (calendar.terminal_due_day, calendar.terminal_settlement_day) == (10, 11)


def test_the_simulated_paths_settle_a_late_close_knock_in_at_the_extended_exercise():
    from quantark.backtest.simulation.conformance import run_oracle
    from quantark.backtest.simulation.engine import EnsembleBacktestEngine
    from simulation.conftest import ensemble_config

    from .test_simulation_ko_reset import _book, mc_engine_config

    cfg = ensemble_config(products=_book(_sim_note()), engine_config=mc_engine_config())
    paths = _two_paths()
    results = EnsembleBacktestEngine(cfg).run(paths)
    maturity = {e.path: e.day for e in results.events if e.event == "maturity"}
    knock_in = {e.path: e.day for e in results.events if e.event == "knock_in"}
    assert knock_in == {0: 1, 1: 4}
    assert maturity == {0: 9, 1: 10}
    for p in (0, 1):                                      # simulation == replay, cash included, to the bit
        report = run_oracle(cfg, paths, p)
        assert report.passed, report.summary()
