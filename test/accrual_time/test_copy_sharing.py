"""Ageing a schedule keeps the accrual metadata its records share (0.11.0 deepcopy regression).

Every derived period of one product points at one ``AccrualBasis`` (one calendar) and one terms key. A per-record
``deepcopy`` with a fresh memo copied that calendar, and its 1,350-string identity, once per record; every later
``deepcopy(product)`` (engine prepare, greek bumps) then paid for all of them.
"""
from datetime import datetime, timedelta

from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.calendar.business_calendar import Calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC

from .conftest import T0, dates, snowball


def _product():
    cal = Calendar(holidays={datetime(1990, 1, 1) + timedelta(days=7 * i + 2) for i in range(50)}, name="DESK")
    ki = dates(*range(1, 60), level=75.0)
    return snowball(ko=dates(30, 60), ki=ki, initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60),
                    annualization_day_count=DC.BUSINESS_DAYS,
                    accrual=AccrualConfig(accrual_calendar=cal, accrual_bus_days_in_year=244.0)), cal


def _shared(schedule):
    periods = [r.accrual for r in schedule.records]
    return {id(a.basis) for a in periods}, {id(a.basis.calendar) for a in periods}, {id(a.terms) for a in periods}


def test_time_shift_keeps_records_sharing_one_basis_and_terms_key():
    product, _ = _product()
    original = product.barrier_config.ki_observation_schedule
    assert [len(s) for s in _shared(original)] == [1, 1, 1]
    for keep_landing in (False, True):
        shifted = original.time_shift(1 / 244, T0 + timedelta(days=1), keep_landing=keep_landing)
        assert len(shifted.records) > 1
        assert [len(s) for s in _shared(shifted)] == [1, 1, 1]


def test_time_shift_still_detaches_from_the_original():
    product, cal = _product()
    original = product.barrier_config.ki_observation_schedule
    shifted = original.time_shift(1 / 244, T0 + timedelta(days=1), keep_landing=True)
    (calendar_id,) = _shared(shifted)[1]
    assert calendar_id != id(cal)
    assert len(shifted.records) == len(original.records)   # the day-1 record lands on the bumped date
    for new, old in zip(shifted.records, original.records):
        assert new is not old
        assert (new.accrual.end, new.accrual.terms) == (old.accrual.end, old.accrual.terms)
        assert new.accrual.basis.same_terms(old.accrual.basis)   # a Calendar's == is object identity
