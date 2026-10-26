"""A REBASED KO-reset on the date clock (clock rule 2026-10-26, ship spec Decision 6).

Its schedule-2 offsets stay the product term (whole months, booked as ``k / 12`` years). At construction it generates
its extended schedule: for each knock-in record date, the schedule-2 dates ``add_months(knock-in date, k)``. Its
latest end is its ``exercise_date``. Monte Carlo places schedule 2 on the generated dates of the knock-in that fired,
and a knocked-in contract converts to ABSOLUTE on those dates.
"""
import math
from datetime import datetime

import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.lifecycle import LifecycleEventType as E
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env
from .test_tracker_rules import run, tracker

KI_DATES = [datetime(2026, 4, 6), datetime(2026, 7, 6)]                 # also schedule 1 (both follow ki_dates)
GENERATED = {0: (datetime(2026, 5, 6), datetime(2026, 6, 6)), 1: (datetime(2026, 8, 6), datetime(2026, 9, 6))}
LATEST_END = datetime(2026, 9, 6)


def _dated(dates, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level) for d in dates])


def dated_rebased(*, months=(1, 2), ki_dates=KI_DATES, exercise_date=LATEST_END, post=None, mode=M.REBASED,
                  **kw):
    post = post or ObservationSchedule(records=[ObservationRecord(observation_time=m / 12, barrier=95.0)
                                                for m in months])
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, initial_date=START.to_pydatetime(), exercise_date=exercise_date,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dated(ki_dates, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_dated(ki_dates, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=post),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode, **kw)


def _day(date):
    return (pd.Timestamp(date) - START).days


# --- construction ---------------------------------------------------------------------------------------------------

def test_a_dated_rebased_contract_generates_its_extended_schedule():
    assert dated_rebased().rebased_extended_schedule == GENERATED


def test_the_generated_dates_keep_the_day_and_clamp_at_the_month_end():
    note = dated_rebased(ki_dates=[datetime(2026, 1, 31), datetime(2026, 7, 6)])
    assert note.rebased_extended_schedule[0] == (datetime(2026, 2, 28), datetime(2026, 3, 31))


def test_an_exercise_date_before_the_latest_end_is_refused():
    with pytest.raises(ValidationError, match="latest"):
        dated_rebased(exercise_date=datetime(2026, 9, 5))
    assert dated_rebased(exercise_date=datetime(2026, 10, 1)).exercise_date == datetime(2026, 10, 1)


def test_the_exercise_date_is_derived_on_a_maturity_date_booking():
    note = dated_rebased(exercise_date=None, maturity_date=LATEST_END)
    assert note.exercise_date == LATEST_END


def test_an_offset_that_is_not_a_whole_month_is_refused():
    with pytest.raises(ValidationError, match="whole number of months"):
        dated_rebased(post=ObservationSchedule(records=[ObservationRecord(observation_time=30 / 365, barrier=95.0)]))


def test_a_schedule_two_settlement_offset_has_no_date_form_and_is_refused():
    post = ObservationSchedule(records=[ObservationRecord(observation_time=1 / 12, settlement_time=1 / 12 + 2 / 365,
                                                          barrier=95.0)])
    with pytest.raises(ValidationError, match="settlement"):
        dated_rebased(post=post)


def test_the_latest_end_is_the_max_maturity():
    e = env(100.0, START)
    assert dated_rebased().get_max_maturity_time(e) == pytest.approx((LATEST_END - START).days / 365.0, abs=0.0)


# --- conversion at the knock-in --------------------------------------------------------------------------------------

def test_a_knock_in_converts_to_absolute_on_the_generated_dates():
    note = dated_rebased()
    anchor = knock_in_anchor_for(note.barrier_config.ki_observation_schedule.records[0], 0)
    converted = note.with_absolute_schedule_2(KI_DATES[0], anchor=anchor)
    records = converted.post_barrier_config.ko_observation_schedule.records
    assert converted.post_ko_mode is M.ABSOLUTE
    assert [r.observation_date for r in records] == list(GENERATED[0])
    assert all(r.observation_time is None for r in records)
    assert converted.exercise_date == GENERATED[0][-1]


def test_a_numeric_knock_in_time_is_refused_on_a_dated_contract():
    with pytest.raises(ValidationError, match="knock-in date"):
        note = dated_rebased()
        note.with_absolute_schedule_2(91 / 365, anchor=knock_in_anchor_for(
            note.barrier_config.ki_observation_schedule.records[0], 0))


# --- Monte Carlo -----------------------------------------------------------------------------------------------------

def _mc():
    return SnowballMCEngine(params=MCParams(seed=3, num_paths=4096))


def test_mc_places_schedule_two_on_the_generated_dates():
    """One knock-in date: the REBASED contract is the ABSOLUTE one on that date's generated schedule."""
    rebased = dated_rebased(ki_dates=KI_DATES[:1], exercise_date=GENERATED[0][-1])
    absolute = dated_rebased(ki_dates=KI_DATES[:1], exercise_date=GENERATED[0][-1],
                             post=_dated(GENERATED[0], 95.0), mode=M.ABSOLUTE)
    e = env(100.0, START)
    assert _mc().price(rebased, e) == _mc().price(absolute, e)


# --- the lifecycle tracker -------------------------------------------------------------------------------------------

def test_the_tracker_knocks_in_and_out_on_the_generated_dates():
    closes = {_day(KI_DATES[0]): 70.0, _day(GENERATED[0][0]) - 1: 99.0, _day(GENERATED[0][0]): 96.0}
    assert run(tracker(dated_rebased()), closes) == [
        (_day(KI_DATES[0]), E.KNOCK_IN, None), (_day(GENERATED[0][0]), E.KNOCK_OUT, "knock-out:ko2:0")]


def test_the_tracker_matures_a_knocked_in_contract_at_its_generated_end():
    closes = {_day(KI_DATES[0]): 70.0, _day(GENERATED[0][0]): 90.0, _day(GENERATED[0][1]): 80.0}
    assert run(tracker(dated_rebased()), closes)[-1] == (_day(GENERATED[0][1]), E.MATURITY, "maturity")


@pytest.mark.parametrize("engine", [
    lambda: KOResetSnowballPDESolver(PDEParams()),
    lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
    _mc,
], ids=["pde", "quad", "mc"])
def test_a_knocked_in_dated_rebased_copy_prices_on_every_engine(engine):
    t = tracker(dated_rebased())
    run(t, {_day(KI_DATES[0]): 70.0})
    when = START + pd.Timedelta(days=_day(KI_DATES[0]) + 9)
    copy = t.product_for_pricing(when, env(90.0, when))
    assert [r.observation_date for r in copy.post_barrier_config.ko_observation_schedule.records] == list(GENERATED[0])
    assert math.isfinite(engine().price(copy, env(90.0, when)))
