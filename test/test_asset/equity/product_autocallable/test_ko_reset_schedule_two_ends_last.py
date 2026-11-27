"""KO-reset validity: schedule 2 ends last (spec D35, AC38; owner ruling 2026-11-14).

A knock-in REPLACES schedule 1 by schedule 2, and the knocked-in note lives until schedule 2 ends. A contract whose
schedule 2 could end before schedule 1 has a knock-in with no schedule 2 left in force (the late knock-in T25e left
open); the owner ruled such a contract invalid. Schedule 2's last date must be on or after schedule 1's last date:

- ABSOLUTE: on the booked dates (or numeric times);
- REBASED: schedule 2 is rebased to the knock-in, so the earliest possible knock-in (the first knock-in observation; a
  REBASED contract monitors its knock-in discretely) plus schedule 2's last offset must not fall before schedule 1's
  last date -- any path could knock in then;
- equality is allowed; a violation raises ``ValidationError`` at construction, naming the dates, on the numeric and the
  dated clock, with discrete and continuous knock-in monitoring.

A valid REBASED note always converts to a valid ABSOLUTE copy (the lifecycle tracker's knocked-in contract): a knock-in
is never earlier than the first knock-in observation.
"""
from datetime import datetime

import pandas as pd
import pytest

from quantark.asset.equity.lifecycle.ko_reset import KOResetLifecycleTracker
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.calendar import add_months
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

QUARTERS = (0.25, 0.5, 0.75, 1.0)
START = datetime(2026, 1, 6)
DATED_S1 = (datetime(2026, 4, 6), datetime(2026, 7, 6), datetime(2026, 10, 6))


def _times(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=float(t), barrier=level) for t in times])


def _dates(dates, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level) for d in dates])


def _ki(ki, schedule):
    if ki == "continuous":
        return dict(ki_barrier=80.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    return dict(ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=schedule)


def numeric(mode, post, *, pre=QUARTERS, ki="discrete", ki_times=None, maturity=None):
    """Schedule 1 ``pre`` (KO 103), knock-in at 80 (discrete on ``ki_times``, default schedule 1's times), schedule 2
    ``post`` (absolute times, or offsets from the knock-in for REBASED). The default maturity is the latest possible
    end (owner ruling O1, 2026-11-27): ABSOLUTE, schedule 2's last time; REBASED, the last knock-in plus the last
    offset."""
    latest = max(post) if mode is M.ABSOLUTE else max(ki_times or pre) + max(post)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity or max(max(pre), latest),
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_times(pre, 103.0),
                                     **_ki(ki, _times(ki_times or pre, 80.0))),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_times(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=mode,
    )


def dated(mode, post, *, pre=DATED_S1, ki="discrete", ki_dates=None, exercise_date=None):
    """The dated twin: schedule 1 on ``pre`` dates; schedule 2 on dates (ABSOLUTE) or whole-month offsets ``k / 12``
    (REBASED)."""
    post_schedule = (_dates(post, 95.0) if mode is M.ABSOLUTE
                     else _times([k / 12 for k in post], 95.0))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, initial_date=START,
        exercise_date=exercise_date or (max(max(pre), max(post)) if mode is M.ABSOLUTE
                                        else max(max(pre), add_months(max(ki_dates or pre), max(post)))),
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dates(pre, 103.0),
                                     **_ki(ki, _dates(ki_dates or pre, 80.0))),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=post_schedule),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False),
        post_ko_mode=mode,
    )


# --- ABSOLUTE: the booked dates ------------------------------------------------------------------------------------


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_numeric_absolute_schedule_two_ending_first_is_refused(ki):
    with pytest.raises(ValidationError, match=r"schedule 2.*0\.5.*schedule 1.*1\.0"):
        numeric(M.ABSOLUTE, (0.25, 0.5), ki=ki)


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_numeric_absolute_schedule_two_ending_with_schedule_one_is_allowed(ki):
    note = numeric(M.ABSOLUTE, (0.5, 1.0), ki=ki)
    assert note.get_post_maturity_time(None) == note.get_pre_maturity_time(None) == 1.0
    numeric(M.ABSOLUTE, (0.5, 1.0, 1.5), ki=ki)


def test_numeric_absolute_legacy_date_lists_are_checked_too():
    """Schedules booked as float lists (``ko_observation_dates``) are built into records, then checked."""
    def build(post):
        return KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0, maturity=1.0,
            barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=list(QUARTERS),
                                         ki_barrier=80.0, ki_observation_dates=list(QUARTERS)),
            post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_dates=list(post)),
        )
    with pytest.raises(ValidationError, match="schedule 2"):
        build((0.25, 0.5))
    build((0.5, 1.0))


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_dated_absolute_schedule_two_ending_first_is_refused_naming_the_dates(ki):
    with pytest.raises(ValidationError, match=r"2026-08-06.*2026-10-06"):
        dated(M.ABSOLUTE, (datetime(2026, 5, 6), datetime(2026, 8, 6)), ki=ki,
              exercise_date=datetime(2026, 10, 6))


@pytest.mark.parametrize("ki", ("discrete", "continuous"))
def test_dated_absolute_schedule_two_ending_on_schedule_one_end_is_allowed(ki):
    dated(M.ABSOLUTE, (datetime(2026, 5, 6), datetime(2026, 10, 6)), ki=ki)


# --- REBASED: the earliest possible knock-in ------------------------------------------------------------------------


def test_numeric_rebased_refused_when_the_first_knock_in_plus_the_last_offset_ends_before_schedule_one():
    """First knock-in 0.25 + last offset 0.5 = 0.75 < 1.0: a path knocking in on 0.25 has no schedule 2 on 0.75..1.0.
    A knock-in on a later record would be fine; the earliest one decides."""
    with pytest.raises(ValidationError, match=r"0\.25.*0\.5.*0\.75.*1\.0"):
        numeric(M.REBASED, (0.25, 0.5))


def test_numeric_rebased_boundary_equality_is_allowed():
    numeric(M.REBASED, (0.25, 0.5, 0.75))                       # 0.25 + 0.75 == 1.0
    numeric(M.REBASED, (0.5,), ki_times=(0.5, 0.75, 1.0))     # 0.5 + 0.5 == 1.0


def test_numeric_rebased_reads_the_first_knock_in_observation_not_schedule_one():
    """The knock-in records, not schedule 1's, give the earliest knock-in."""
    with pytest.raises(ValidationError, match="schedule 2"):
        numeric(M.REBASED, (0.5,), ki_times=(0.25, 1.0))
    numeric(M.REBASED, (0.5,), ki_times=(0.5, 1.0))


def test_rebased_continuous_knock_in_is_refused_by_the_discrete_monitoring_rule():
    """A REBASED schedule hangs from a knock-in date, so it needs discrete knock-in monitoring (pre-existing rule): the
    earliest possible knock-in of a REBASED contract is therefore always its first knock-in observation."""
    with pytest.raises(ValidationError, match="discrete KI"):
        numeric(M.REBASED, (0.25, 0.5, 0.75, 1.0), ki="continuous")


def test_dated_rebased_refused_naming_the_knock_in_and_the_dates():
    """First knock-in 2026-04-06 + 2 months = 2026-06-06 < schedule 1's last 2026-10-06."""
    with pytest.raises(ValidationError, match=r"2026-04-06.*2026-06-06.*2026-10-06"):
        dated(M.REBASED, (1, 2))


def test_dated_rebased_boundary_equality_is_allowed():
    note = dated(M.REBASED, (3, 6))                             # 2026-04-06 + 6 months == 2026-10-06
    assert note.rebased_extended_schedule[0][-1] == DATED_S1[-1]


def test_dated_rebased_month_end_clamp_is_compared_as_generated():
    """``add_months`` clamps at the month end: 2026-01-31 + 1 month = 2026-02-28."""
    dated(M.REBASED, (1,), pre=(datetime(2026, 1, 31), datetime(2026, 2, 28)))
    with pytest.raises(ValidationError, match="2026-02-28.*2026-03-01"):
        dated(M.REBASED, (1,), pre=(datetime(2026, 1, 31), datetime(2026, 3, 1)))


def test_the_helper_refuses_a_rebased_schedule_two_too_short_for_the_first_knock_in():
    with pytest.raises(ValidationError, match="schedule 2"):
        create_ko_reset_snowball(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=0.5,
                                 ki_barrier=80.0, ki_continuous=False, post_ko_mode=M.REBASED)


# --- a valid REBASED note converts to a valid ABSOLUTE copy -------------------------------------------------------


@pytest.mark.parametrize("post", ((0.25, 0.5, 0.75), (0.5, 1.0), (1 / 12, 1.0)))
def test_every_knock_in_of_a_valid_numeric_rebased_note_converts_to_a_valid_copy(post):
    note = numeric(M.REBASED, post, ki_times=(0.25, 0.5, 0.75, 1.0))
    for index, record in enumerate(note.barrier_config.ki_observation_schedule.records):
        copy = note.with_absolute_schedule_2(float(record.observation_time), anchor=knock_in_anchor_for(record, index))
        copy.validate()
        assert copy.get_post_maturity_time(None) >= copy.get_pre_maturity_time(None)


def test_every_knock_in_of_a_valid_dated_rebased_note_converts_to_a_valid_copy():
    note = dated(M.REBASED, (3, 6), pre=(datetime(2026, 4, 6), datetime(2026, 7, 31), datetime(2026, 10, 6)),
                 ki_dates=(datetime(2026, 4, 6), datetime(2026, 7, 31), datetime(2026, 10, 6)))
    for index, record in enumerate(note.barrier_config.ki_observation_schedule.records):
        copy = note.with_absolute_schedule_2(record.observation_date, anchor=knock_in_anchor_for(record, index))
        copy.validate()
        assert max(r.observation_date for r in copy.post_barrier_config.ko_observation_schedule.records) >= \
            datetime(2026, 10, 6)


D = 1 / 365


@pytest.mark.parametrize("days", ((91.76, 181.479, 273.239), (91.4, 273.4, 364.8)), ids=("rounds_up", "rounds_short"))
def test_the_tracker_copy_of_a_boundary_rebased_note_is_valid_on_off_day_times(days):
    """The tracker snaps numeric times to whole days and knocks in on a whole day. A REBASED note booked on the equality
    boundary with off-day times (first knock-in + last offset == schedule 1's end) stays valid, and so does its copy on
    every knock-in close: where the two rounded terms fall a day short (91 + 273 < 365) the snapped last offset takes
    the day back (``snapping._keep_schedule_two_last``)."""
    first, offset, end = (d * D for d in days)
    pre = (first, 200 * D, end)
    note = numeric(M.REBASED, (offset,), pre=pre, ki_times=pre)
    for ki_index, ki_time in enumerate(pre):
        tracker = KOResetLifecycleTracker(product=note, quantity=1.0, start_date=pd.Timestamp(START))
        ki_day = int(round(ki_time * 365))
        tracker.lifecycle.knocked_in = True
        tracker.lifecycle.ki_date = (pd.Timestamp(START) + pd.Timedelta(days=ki_day)).to_pydatetime()
        tracker.lifecycle.knock_in_anchor = knock_in_anchor_for(
            tracker.product.barrier_config.ki_observation_schedule.records[ki_index], ki_index)
        tracker.product.validate()
        tracker.contract().validate()
