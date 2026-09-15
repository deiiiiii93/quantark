from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind, EventPhase
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _ctx(sse_calendar, sse_sessions, profile, ts, spot=100.0, phase=EventPhase.BEFORE, fixings=(), months=12):
    prod = dated_snowball(sse_calendar, T0, months=months)
    req = IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=spot), session_calendar=sse_sessions,
                                   variance_profile=profile, event_phase=phase, fixings=tuple(fixings))
    return resolve_context(req)


def _kos(ctx):
    return [e for e in ctx.timeline.events if e.kind is EventKind.KO]


def test_route_prices_the_twin_and_reconciles_components(sse_calendar, sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    kos = _kos(_ctx(sse_calendar, sse_sessions, desk, ts))
    ctx = _ctx(sse_calendar, sse_sessions, desk, ts, fixings=[Fixing(k.timestamp, 100.0) for k in kos[:5]])
    engine = SnowballQuadEngineV2()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.method.startswith("quad_v2") and abs(out.numerical["reconciliation_error"]) < 1e-10
    assert out.numerical["events"] > 0 and out.numerical["nodes"] > 0
    assert out.contingent_pv == pytest.approx(out.components["ko"] + out.components["coupon"] + out.components["terminal"], abs=1e-10)
    direct = SnowballQuadEngineV2().price(ctx.numerical.product, ctx.pricing_env,
                                          lifecycle_state=ctx.numerical.lifecycle_state, event_phase="before")
    assert out.contingent_pv == pytest.approx(direct, abs=1e-12)


def test_one_second_before_the_fixing_the_ko_is_almost_certain_above_barrier(sse_calendar, sse_sessions, desk):
    kos = _kos(_ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    fixings = [Fixing(k.timestamp, 100.0) for k in kos[:5]]
    engine = SnowballQuadEngineV2()
    ctx = _ctx(sse_calendar, sse_sessions, desk, kos[5].timestamp - timedelta(seconds=1), spot=103.5, fixings=fixings)
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.contingent_pv == pytest.approx(kos[5].cash, rel=1e-6)        # KO cash paid at determination, DF ~ 1
    assert out.components["ko"] / out.contingent_pv > 0.999


def test_before_and_after_at_the_fixing_instant_differ_only_by_the_event(sse_calendar, sse_sessions, desk):
    kos = _kos(_ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    fix = kos[5].timestamp
    fixings = [Fixing(k.timestamp, 100.0) for k in kos[:5]]
    engine = SnowballQuadEngineV2()
    before = _ctx(sse_calendar, sse_sessions, desk, fix, spot=104.0, phase=EventPhase.BEFORE, fixings=fixings)
    after = _ctx(sse_calendar, sse_sessions, desk, fix, spot=104.0, phase=EventPhase.AFTER, fixings=fixings + [Fixing(fix, 104.0)])
    ob = route_for(before, engine).price(before, engine)
    oa = route_for(after, engine).price(after, engine)
    assert ob.contingent_pv == pytest.approx(kos[5].cash, abs=1e-9)          # decided at spot 104 >= 103 -> KO cash now
    assert oa.method == "terminated" and oa.contingent_pv == 0.0             # after: KO is confirmed history
    assert after.numerical.paid_cash == pytest.approx(kos[5].cash, rel=1e-12)  # paid at determination


def test_zero_variance_profile_prices_lunch_deterministically(sse_calendar, sse_sessions):
    p = VarianceProfile.sessions_only(sse_sessions, 244)
    engine = SnowballQuadEngineV2()
    at_noon = _ctx(sse_calendar, sse_sessions, p, datetime(2026, 9, 15, 12, 0, tzinfo=SHANGHAI))
    at_break = _ctx(sse_calendar, sse_sessions, p, datetime(2026, 9, 15, 11, 30, tzinfo=SHANGHAI))
    out = route_for(at_noon, engine).price(at_noon, engine)
    out2 = route_for(at_break, engine).price(at_break, engine)
    # From 11:30 to 12:00 no variance accrues: the two prices differ only by 30 minutes of carry/discounting
    assert out.contingent_pv != out2.contingent_pv
    assert out.contingent_pv == pytest.approx(out2.contingent_pv, rel=2e-5)
