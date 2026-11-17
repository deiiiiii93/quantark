"""Ageing a schedule keeps the accrual metadata its records share (0.11.0 deepcopy regression).

Every derived period of one product points at one ``AccrualBasis`` (one calendar) and one terms key. A per-record
``deepcopy`` with a fresh memo copied that calendar, and its 1,350-string identity, once per record; every later
``deepcopy(product)`` (engine prepare, greek bumps) then paid for all of them.
"""
import copy as copy_module
from copy import deepcopy
from datetime import datetime, timedelta

from quantark.asset.equity.product.option.accrual import KnockInAnchor
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


def test_deepcopy_shares_immutable_accrual_members_and_copies_the_calendar_once():
    product, cal = _product()
    copy = deepcopy(product)
    old_records = product.barrier_config.ki_observation_schedule.records
    new_records = copy.barrier_config.ki_observation_schedule.records
    (calendar_id,) = _shared(copy.barrier_config.ki_observation_schedule)[1]
    assert calendar_id != id(cal) and calendar_id == id(copy.accrual_config.accrual_calendar)
    for old, new in zip(old_records, new_records):
        a, b = old.accrual, new.accrual
        assert b is not a and b.basis is not a.basis
        assert b.end is a.end and b.origin is a.origin       # datetimes are immutable: shared, never rebuilt
        assert (b.kind, b.start, b.end, b.terms) == (a.kind, a.start, a.end, a.terms)
        assert b.basis.same_terms(a.basis)
    cal.holidays.add(datetime(2026, 1, 7))                 # the copy does not see the original's calendar change
    assert len(copy.accrual_config.accrual_calendar.holidays) == len(cal.holidays) - 1
    assert len(new_records[0].accrual.basis.calendar.holidays) == len(cal.holidays) - 1


def _identities(terms):
    return [part for part in terms if isinstance(part, tuple) and part and part[0] == "DESK"]


def test_deepcopy_shares_the_calendar_identity_inside_every_terms_key(monkeypatch):
    # deepcopy never memoises a tuple it returns unchanged: a plain identity (one string per holiday) was walked
    # again for every distinct terms key on every product copy (34 KO keys -> 34 walks of the whole calendar).
    product, cal = _product()
    walked = []
    plain = copy_module._deepcopy_dispatch[tuple]

    def counting(x, memo, *args):
        if x and x[0] == "DESK":
            walked.append(id(x))
        return plain(x, memo, *args)

    monkeypatch.setitem(copy_module._deepcopy_dispatch, tuple, counting)
    copy = deepcopy(product)
    assert walked == []
    for schedule in ("ko_observation_schedule", "ki_observation_schedule"):
        for old, new in zip(getattr(product.barrier_config, schedule).records,
                            getattr(copy.barrier_config, schedule).records):
            (old_identity,), (new_identity,) = _identities(old.accrual.terms), _identities(new.accrual.terms)
            assert new_identity is old_identity and new_identity == cal.identity()
            assert new.accrual.terms == old.accrual.terms and hash(new.accrual.terms) == hash(old.accrual.terms)
    (identity,) = _identities(copy._accrual_terms_key)
    assert identity is _identities(product._accrual_terms_key)[0]


def test_deepcopy_of_a_knock_in_anchor_copies_its_calendar_through_the_memo():
    product, _ = _product()
    basis = product.barrier_config.ki_observation_schedule.records[0].accrual.basis
    anchor = KnockInAnchor(source_index=3, fixing=T0 + timedelta(days=9), origin=T0, basis=basis)
    pair = deepcopy((anchor, basis))
    assert pair[0] is not anchor and pair[0].basis is pair[1]
    assert pair[0].basis.calendar is not basis.calendar and pair[0].accrual() == anchor.accrual()


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
