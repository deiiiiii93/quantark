from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.execution import PricingSession
from quantark.execution.errors import CapabilityError
from quantark.intraday import value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind, EventPhase
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.result import IntradayValuationResult
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_snowball, digital, flat_env, snapshot

T0 = datetime(2026, 3, 16)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _snow_req(sse_calendar, sse_sessions, profile, ts, **kw):
    return IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts, spot=kw.pop("spot", 100.0)),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


def _kos(sse_calendar, sse_sessions, profile):
    ctx = resolve_context(_snow_req(sse_calendar, sse_sessions, profile, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    return [e for e in ctx.timeline.events if e.kind is EventKind.KO]


def test_session_entry_point_returns_a_structured_result_and_leaves_inputs_alone(sse_calendar, sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    prod, env = dated_snowball(sse_calendar, T0), flat_env(ts)
    before = (snapshot(prod), snapshot(env))
    with PricingSession() as session:
        res = session.value_intraday(SnowballQuadEngineV2(), prod, env, session_calendar=sse_sessions, variance_profile=desk)
    _assert_unchanged(before[0], prod)
    _assert_unchanged(before[1], env)
    assert isinstance(res, IntradayValuationResult)
    assert res.valuation_timestamp == ts and res.phase is EventPhase.BEFORE
    assert res.provisional and len(res.assumptions) == 5          # five past monthly fixings, none supplied
    assert res.price == pytest.approx(res.contingent_pv + res.pending_receivable_pv)
    assert res.engine.endswith("SnowballQuadEngineV2") and res.method.startswith("quad_v2")
    assert res.profile_identity == desk.identity() and any(r.startswith("context:") for r in res.records)


def test_framework_parity_records_a_manifest_fingerprint(sse_calendar, sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    with PricingSession() as session:
        res = value_intraday(SnowballQuadEngineV2(), _snow_req(sse_calendar, sse_sessions, desk, ts), session=session)
    assert any(r.startswith("manifest:") for r in res.records)


def test_digital_through_the_service(sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    req = IntradayValuationRequest(product=digital(datetime(2026, 9, 15)), pricing_env=flat_env(ts, spot=100.3),
                                   session_calendar=sse_sessions, variance_profile=desk)
    res = value_intraday(DigitalOptionAnalyticalEngine(), req)
    assert 0.9 < res.price < 1.0 and not res.provisional and res.lifecycle == {}
    with PricingSession() as session:
        framed = value_intraday(DigitalOptionAnalyticalEngine(), req, session=session)
    assert framed.price == res.price and any(r.startswith("manifest:") for r in framed.records)


def test_assumed_ko_result_is_paid_cash_with_zero_contingent(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    with PricingSession() as session:
        res = value_intraday(SnowballQuadEngineV2(), _snow_req(sse_calendar, sse_sessions, desk, ts, spot=104.0,
                                                                event_phase="after", fixings=fixings), session=session)
    assert res.provisional and res.contingent_pv == 0.0 and res.lifecycle["knocked_out"]
    assert res.paid_cash == pytest.approx(kos[5].cash, rel=1e-12) and res.pending_receivable_pv == 0.0
    assert [c.kind for c in res.cashflows] == ["contingent", "paid"] and res.cashflows[1].provenance == "provisional"
    assert res.cashflows[1].event_id == kos[5].event_id


def test_confirmed_ko_cash_is_confirmed(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]) + (Fixing(kos[5].timestamp, 104.0),)
    res = value_intraday(SnowballQuadEngineV2(), _snow_req(sse_calendar, sse_sessions, desk, ts, spot=104.0,
                                                            event_phase="after", fixings=fixings))
    assert not res.provisional and res.cashflows[1].provenance == "confirmed"


def test_unsupported_engine_and_greeks_fail_closed(sse_calendar, sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    with pytest.raises(CapabilityError, match="intraday inventory"):
        value_intraday(SnowballQuadEngine(), _snow_req(sse_calendar, sse_sessions, desk, ts))
    with pytest.raises(CapabilityError, match="vanna"):
        value_intraday(SnowballQuadEngineV2(), _snow_req(sse_calendar, sse_sessions, desk, ts, greeks=("vanna",), greek_convention="point"))
