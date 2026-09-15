import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind, EventPhase
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.param.vol import TradingClockVolSurface
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_snowball, flat_env, snapshot

T0 = datetime(2026, 3, 16)
TS = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _req(sse_calendar, sse_sessions, profile, ts=TS, **kw):
    prod = dated_snowball(sse_calendar, T0)
    return IntradayValuationRequest(product=prod, pricing_env=flat_env(ts), session_calendar=sse_sessions,
                                    variance_profile=profile, **kw)


def test_context_is_immutable_and_inputs_untouched(sse_calendar, sse_sessions, desk):
    req = _req(sse_calendar, sse_sessions, desk)
    before = (snapshot(req.product), snapshot(req.pricing_env))
    ctx = resolve_context(req)
    _assert_unchanged(before[0], req.product)
    _assert_unchanged(before[1], req.pricing_env)
    assert isinstance(ctx.pricing_env.vol_surface, TradingClockVolSurface) and ctx.pricing_env.vol_surface.time_map is ctx.time_map
    assert ctx.pricing_env.valuation_date == TS and ctx.pricing_env.bus_days_in_year == 244
    assert ctx.pricing_env.day_count_convention is DayCountConvention.CALENDAR_DAYS
    with pytest.raises((AttributeError, TypeError)):
        ctx.phase = EventPhase.AFTER


def test_naive_timestamp_business_days_and_prewrapped_surface_are_rejected(sse_calendar, sse_sessions, desk):
    with pytest.raises(ValidationError, match="timezone-aware"):
        resolve_context(_req(sse_calendar, sse_sessions, desk, ts=datetime(2026, 9, 15, 14, 59, 59)))
    req = _req(sse_calendar, sse_sessions, desk)
    env_bd = dataclasses.replace(req.pricing_env, day_count_convention=DayCountConvention.BUSINESS_DAYS,
                                 calendar=sse_calendar, bus_days_in_year=244)
    with pytest.raises(ValidationError, match="CALENDAR_DAYS"):
        resolve_context(dataclasses.replace(req, pricing_env=env_bd))
    ctx = resolve_context(req)
    env_wrapped = dataclasses.replace(req.pricing_env, vol_surface=ctx.pricing_env.vol_surface)
    with pytest.raises(ValidationError, match="inner"):
        resolve_context(dataclasses.replace(req, pricing_env=env_wrapped))


def test_identity_changes_with_time_phase_profile_market_and_assumptions(sse_calendar, sse_sessions, desk):
    base = resolve_context(_req(sse_calendar, sse_sessions, desk))
    later = resolve_context(_req(sse_calendar, sse_sessions, desk, ts=TS + timedelta(seconds=1)))
    other_profile = resolve_context(_req(sse_calendar, sse_sessions, VarianceProfile("desk", "2", 244, 0.25, (0.35, 0.35), (0.05,))))
    assert len({base.identity, later.identity, other_profile.identity}) == 3
    same = resolve_context(_req(sse_calendar, sse_sessions, desk))
    assert same.identity == base.identity and same.market_snapshot_id == base.market_snapshot_id
    kos = [e for e in base.timeline.events if e.kind is EventKind.KO]
    ts2 = kos[5].timestamp + timedelta(seconds=30)
    a = resolve_context(_req(sse_calendar, sse_sessions, desk, ts=ts2, event_phase=EventPhase.AFTER))          # assumption from spot 100
    b = resolve_context(_req(sse_calendar, sse_sessions, desk, ts=ts2, event_phase=EventPhase.AFTER,
                             fixings=(Fixing(kos[5].timestamp, 100.0),) + tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])))
    assert a.provisional and not b.provisional and a.identity != b.identity
    bumped = dataclasses.replace(_req(sse_calendar, sse_sessions, desk), pricing_env=flat_env(TS, vol=0.21))
    assert resolve_context(bumped).market_snapshot_id != base.market_snapshot_id


def test_identity_is_zone_independent(sse_calendar, sse_sessions, desk):
    a = resolve_context(_req(sse_calendar, sse_sessions, desk))
    b = resolve_context(_req(sse_calendar, sse_sessions, desk, ts=TS.astimezone(timezone.utc)))
    assert a.identity == b.identity and a.market_snapshot_id == b.market_snapshot_id


def test_horizon_covers_the_last_payment(sse_calendar, sse_sessions, desk):
    ctx = resolve_context(_req(sse_calendar, sse_sessions, desk))
    last_pay = max(e.payment_timestamp for e in ctx.timeline.events)
    assert ctx.time_map.horizon_date >= last_pay
    assert ctx.numerical.maturity_tau <= ctx.time_map.segments[-1].tau_end


def test_request_validation(sse_calendar, sse_sessions, desk):
    with pytest.raises(ValidationError, match="greek_convention"):
        _req(sse_calendar, sse_sessions, desk, greeks=("delta",))
    with pytest.raises(ValidationError, match="event_phase"):
        _req(sse_calendar, sse_sessions, desk, event_phase="during")
    r = _req(sse_calendar, sse_sessions, desk, event_phase="after", greeks=("delta",), greek_convention="point")
    assert r.event_phase is EventPhase.AFTER
