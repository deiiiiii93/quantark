"""The intraday twin carries its source contract's periods (accrual time spec §6; review round 1 P1-4, round 2 P1-1,
round 3 P1-2)."""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.intraday.events import EventKind, EventPhase, _schedule_env, resolve_timeline
from quantark.intraday.fixings import Fixing
from quantark.intraday.provisional import reconstruct_lifecycle
from quantark.intraday.twin import build_numerical_contract
from quantark.util.calendar.day_counter import DayCountConvention as DC
from intraday.conftest import dated_phoenix, dated_snowball, flat_env, sse_calendar, sse_sessions  # noqa: F401

T0 = datetime(2026, 3, 16)


def _twin(sessions, prod, ts, fixings=()):
    env = flat_env(ts)
    tl = resolve_timeline(prod, sessions, env)
    rec = reconstruct_lifecycle(prod, tl, None, list(fixings), valuation_timestamp=ts, phase=EventPhase.AFTER,
                                spot=100.0, spot_timestamp=ts, schedule_env=_schedule_env(prod, env),
                                session_calendar=sessions)
    num = build_numerical_contract(prod, tl, rec, valuation_timestamp=ts, phase=EventPhase.AFTER,
                                   session_calendar=sessions)
    return tl, num


def _mid_life(sessions, prod, confirmed=5):
    kos = [e for e in resolve_timeline(prod, sessions, flat_env(datetime(2026, 9, 15))).events if e.kind is EventKind.KO]
    ts = kos[confirmed].timestamp - timedelta(hours=1)
    return _twin(sessions, prod, ts, [Fixing(k.timestamp, 100.0) for k in kos[:confirmed]])


def test_twin_records_carry_the_source_periods(sse_calendar, sse_sessions):
    prod = dated_phoenix(sse_calendar, T0)
    _, num = _mid_life(sse_sessions, prod)
    remaining = [e.index for e in num.remaining_events if e.kind is EventKind.KO]
    src = prod.barrier_config.ko_observation_schedule.records
    twin_periods = [r.accrual for r in num.product.barrier_config.ko_observation_schedule.records]
    assert [replace(p, terms=None) for p in twin_periods] == [replace(src[i].accrual, terms=None) for i in remaining]


def test_dated_twin_resolves_the_source_cash_exactly(sse_calendar, sse_sessions):
    prod = dated_phoenix(sse_calendar, T0)
    tl, num = _mid_life(sse_sessions, prod)
    twin = num.product
    assert twin.initial_date is None
    remaining = [e.index for e in num.remaining_events if e.kind is EventKind.KO]
    env = flat_env(T0)
    source_cash = {i: r.payoff for i, r in zip(range(len(prod.barrier_config.ko_observation_schedule.records)),
                                               prod.resolve_ko_observations(env))}
    twin_cash = [r.payoff for r in twin.resolve_ko_observations(flat_env(T0 + timedelta(days=200)))]
    assert twin_cash == [source_cash[i] for i in remaining]
    assert [twin.get_coupon_payoff(k) for k in range(len(remaining))] == [prod.get_coupon_payoff(i) for i in remaining]


def test_business_days_twin_needs_no_environment_calendar(sse_calendar, sse_sessions):
    base = dated_snowball(sse_calendar, T0)
    prod = dated_snowball(sse_calendar, T0)
    prod.annualization_day_count = DC.BUSINESS_DAYS
    prod.accrual_config = AccrualConfig(accrual_calendar=sse_calendar, accrual_bus_days_in_year=244.0)
    prod.validate()
    assert base is not prod
    _, num = _mid_life(sse_sessions, prod)
    remaining = [e for e in num.remaining_events if e.kind is EventKind.KO]
    twin_cash = [r.payoff for r in num.product.resolve_ko_observations(flat_env(T0 + timedelta(days=200)))]
    assert twin_cash == pytest.approx([e.cash for e in remaining], rel=0, abs=0)
