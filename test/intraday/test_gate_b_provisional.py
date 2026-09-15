"""Gate B: provisional-state integrity (purity, reversal on actual fixings, cache separation, revision)."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.intraday import Fixing, value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_snowball, flat_env, snapshot

T0 = datetime(2026, 3, 16)
ENGINE = SnowballQuadEngineV2()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _req(sse_calendar, sse_sessions, profile, ts, prod=None, **kw):
    spot = kw.pop("spot", 100.0)
    return IntradayValuationRequest(product=prod or dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts, spot=spot),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


def _kos(sse_calendar, sse_sessions, profile):
    ctx = resolve_context(_req(sse_calendar, sse_sessions, profile, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    return [e for e in ctx.timeline.events if e.kind is EventKind.KO]


def test_repeated_pricing_is_pure(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    prod = dated_snowball(sse_calendar, T0)
    cp = AutocallableLifecycleState()
    before = (snapshot(prod), snapshot(cp))
    req = _req(sse_calendar, sse_sessions, desk, kos[5].timestamp + timedelta(seconds=30), prod=prod, spot=104.0,
               event_phase="after", lifecycle_state=cp)
    r1, r2 = value_intraday(ENGINE, req), value_intraday(ENGINE, req)
    assert r1.to_dict() == r2.to_dict() and r1.provisional
    _assert_unchanged(before[0], prod)
    _assert_unchanged(before[1], cp)


def test_actual_fixing_reverses_the_assumed_ko_and_restores_the_live_claim(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    assumed = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=104.0, event_phase="after", fixings=fixings))
    actual = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=104.0, event_phase="after",
                                         fixings=fixings + (Fixing(kos[5].timestamp, 102.5),)))
    assert assumed.provisional and assumed.lifecycle["knocked_out"] and assumed.contingent_pv == 0.0
    assert not actual.provisional and actual.lifecycle["alive"] and actual.contingent_pv > 0.0 and actual.paid_cash == 0.0
    assert assumed.context_identity != actual.context_identity


def test_opposite_correction_assumed_alive_then_actual_ko(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    alive = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=101.0, event_phase="after", fixings=fixings))
    kod = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=101.0, event_phase="after",
                                      fixings=fixings + (Fixing(kos[5].timestamp, 103.2),)))
    assert alive.provisional and alive.lifecycle["alive"]
    assert not kod.provisional and kod.lifecycle["knocked_out"] and kod.paid_cash == pytest.approx(kos[5].cash, rel=1e-12)


def test_cache_separation_between_two_provisional_bases(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    a = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=74.0, event_phase="after"))     # assumed KI everywhere
    b = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=76.0, event_phase="after"))     # assumed alive, no KI
    assert a.lifecycle["knocked_in"] and not b.lifecycle["knocked_in"] and a.price != b.price
    assert a.context_identity != b.context_identity


def test_new_base_snapshot_revises_assumptions_and_says_so(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    a = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts, spot=102.9, event_phase="after", fixings=fixings))
    b = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts + timedelta(seconds=5), spot=103.1,
                                    event_phase="after", fixings=fixings))
    assert [x.assumed_value for x in a.assumptions] == [102.9] and [x.assumed_value for x in b.assumptions] == [103.1]
    assert a.assumptions[0].scheduled_at == b.assumptions[0].scheduled_at == kos[5].timestamp
    assert a.lifecycle["alive"] and b.lifecycle["knocked_out"]          # the revision is an assumption change, not market risk
