from datetime import datetime, time, timedelta

import pytest

from quantark.intraday.events import EventKind, EventPhase, resolve_timeline
from quantark.intraday.timestamp import SECONDS_PER_YEAR
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_phoenix, dated_snowball, digital, flat_env, snapshot

T0 = datetime(2026, 3, 16)


def test_dated_snowball_resolves_to_exchange_close_with_contractual_cash(sse_calendar, sse_sessions):
    prod = dated_snowball(sse_calendar, T0, months=12)
    before = snapshot(prod)
    tl = resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)))
    _assert_unchanged(before, prod)
    kos = [e for e in tl.events if e.kind is EventKind.KO]
    kis = [e for e in tl.events if e.kind is EventKind.KI]
    assert len(kos) == 12 and len(kis) == 12
    first = kos[0]
    assert first.timestamp.hour == 15 and first.timestamp.tzinfo == SHANGHAI and first.date_only
    d0 = prod.barrier_config.ko_observation_schedule.records[0].observation_date
    assert first.timestamp.date() == d0.date()
    # cash = principal(0, include_principal=False) + 100 * 1 * 0.12 * ACT365(T0 -> d0), the product's own day count
    assert first.cash == pytest.approx(100.0 * 0.12 * (d0 - T0).days / 365.0)
    assert first.payment_timestamp == first.timestamp            # no settlement lag on the fixture
    assert tl.terminal().kind is EventKind.TERMINAL and tl.terminal().timestamp == kos[-1].timestamp
    assert tl.contractual_tenor == pytest.approx((prod.exercise_date - T0).days / 365.0)
    assert tl.continuous_ki_barrier is None


def test_ordering_at_a_shared_timestamp_is_ki_ko_coupon_terminal(sse_calendar, sse_sessions):
    tl = resolve_timeline(dated_snowball(sse_calendar, T0), sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    last_ts = tl.terminal().timestamp
    kinds = [e.kind for e in tl.at(last_ts)]
    assert kinds == [EventKind.KI, EventKind.KO, EventKind.TERMINAL]


def test_due_and_remaining_respect_phase(sse_calendar, sse_sessions):
    tl = resolve_timeline(dated_snowball(sse_calendar, T0), sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    fix = [e for e in tl.events if e.kind is EventKind.KO][5].timestamp
    assert fix not in {e.timestamp for e in tl.due(fix, EventPhase.BEFORE)}
    assert fix in {e.timestamp for e in tl.due(fix, EventPhase.AFTER)}
    assert fix in {e.timestamp for e in tl.remaining(fix, EventPhase.BEFORE)}
    assert fix not in {e.timestamp for e in tl.remaining(fix, EventPhase.AFTER)}
    assert len(tl.due(fix - timedelta(seconds=1), EventPhase.AFTER)) == len(tl.due(fix, EventPhase.BEFORE))


def test_explicit_timestamp_and_fixing_time_override(sse_calendar, sse_sessions):
    prod = dated_snowball(sse_calendar, T0)
    rec = prod.barrier_config.ko_observation_schedule.records[0]
    rec.observation_timestamp = datetime(rec.observation_date.year, rec.observation_date.month, rec.observation_date.day,
                                         10, 0, tzinfo=SHANGHAI)
    tl = resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)), fixing_time_of_day=time(14, 0))
    kos = [e for e in tl.events if e.kind is EventKind.KO]
    assert kos[0].timestamp.hour == 10 and not kos[0].date_only          # explicit wins
    assert kos[1].timestamp.hour == 14 and kos[1].date_only               # per-contract override for date-only records


def test_float_schedule_needs_an_origin(sse_calendar, sse_sessions):
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    prod = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                                     ki_barrier=75.0, ki_observation_dates=[0.25, 0.5, 0.75, 1.0]),
        payoff_config=PayoffConfig(rebate_rate=0.12, include_principal=False))
    env = flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI))
    with pytest.raises(ValidationError, match="schedule_origin"):
        resolve_timeline(prod, sse_sessions, env)
    origin = datetime(2026, 3, 16, 15, 0, tzinfo=SHANGHAI)
    tl = resolve_timeline(prod, sse_sessions, env, schedule_origin=origin)
    kos = [e for e in tl.events if e.kind is EventKind.KO]
    assert kos[0].timestamp == origin + timedelta(seconds=0.25 * SECONDS_PER_YEAR)
    assert kos[0].cash == pytest.approx(100.0 * 0.12 * 0.25)


def test_digital_has_one_terminal_event(sse_calendar, sse_sessions):
    prod = digital(expiry=datetime(2026, 12, 15))
    tl = resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    assert [e.kind for e in tl.events] == [EventKind.TERMINAL]
    assert tl.events[0].timestamp == datetime(2026, 12, 15, 15, 0, tzinfo=SHANGHAI)


def test_observation_on_non_trading_day_is_rejected(sse_calendar, sse_sessions):
    prod = dated_snowball(sse_calendar, T0)
    prod.barrier_config.ko_observation_schedule.records[0].observation_date = datetime(2026, 4, 19)   # Sunday
    with pytest.raises(ValidationError, match="not a trading day"):
        resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))


def test_phoenix_coupon_cash_is_the_contracts_per_period_day_count(sse_calendar, sse_sessions):
    prod = dated_phoenix(sse_calendar, T0, months=12, coupon_rate=0.12)
    tl = resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    coupons = [e for e in tl.events if e.kind is EventKind.COUPON]
    kos = [e for e in tl.events if e.kind is EventKind.KO]
    dates = [T0] + [r.observation_date for r in prod.barrier_config.ko_observation_schedule.records]
    assert len(coupons) == 12 and [c.timestamp for c in coupons] == [k.timestamp for k in kos]
    for i, c in enumerate(coupons):
        assert c.cash == pytest.approx(100.0 * 0.12 * (dates[i + 1] - dates[i]).days / 365.0, rel=1e-14)
        assert c.barrier == 80.0
    assert kos[0].cash == pytest.approx(100.0)                     # principal only: ko_rate 0, coupons are separate
    assert [e.kind for e in tl.at(kos[3].timestamp)] == [EventKind.KI, EventKind.KO, EventKind.COUPON]


def test_continuous_ki_by_observation_type_is_a_continuous_barrier(sse_calendar, sse_sessions):
    from dataclasses import replace
    from quantark.util.enum.option_enums import ObservationType
    prod = dated_snowball(sse_calendar, T0)
    prod.barrier_config = replace(prod.barrier_config, ki_observation_type=ObservationType.CONTINUOUS,
                                  ki_observation_schedule=None)
    tl = resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    assert tl.continuous_ki_barrier == 75.0 and not any(e.kind is EventKind.KI for e in tl.events)
