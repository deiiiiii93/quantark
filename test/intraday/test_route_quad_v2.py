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


# --- the maturity instant under BEFORE: every remaining event is decided on the known spot -------------------------
def _maturity_ctx(sse_calendar, sse_sessions, desk, spot):
    kos = _kos(_ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    fixings = [Fixing(k.timestamp, 100.0) for k in kos[:-1]]
    ctx = _ctx(sse_calendar, sse_sessions, desk, kos[-1].timestamp, spot=spot, fixings=fixings)
    return ctx, kos[-1], fixings


@pytest.mark.parametrize("spot, expected", [(104.0, 12.0), (90.0, 12.0), (70.0, -30.0)])
def test_at_the_maturity_instant_the_claim_is_decided_on_the_spot(sse_calendar, sse_sessions, desk, spot, expected):
    """KO, KI and the terminal payoff all sit at the valuation instant. QUAD V2's compiler has no zero-maturity
    twin, so the route values the decided claim: what the lifecycle books when those events are fixed at the spot."""
    from dataclasses import replace
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    ctx, last, fixings = _maturity_ctx(sse_calendar, sse_sessions, desk, spot)
    assert ctx.numerical.maturity_tau == 0.0 and not ctx.numerical.terminated
    engine = SnowballQuadEngineV2()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.method == "decided_at_valuation" and out.exact
    assert out.contingent_pv == pytest.approx(expected, abs=1e-12)
    # the BEFORE / AFTER identity every other fixing instant satisfies: decided now = paid at the instant
    after = resolve_context(replace(ctx.request, event_phase=EventPhase.AFTER,
                                    fixings=tuple(fixings) + (Fixing(last.timestamp, spot),)))
    assert after.numerical.terminated and out.contingent_pv == pytest.approx(after.numerical.paid_cash, abs=1e-12)
    pde = SnowballPDESolver(PDEParams())                      # an independent route that already prices the instant
    assert out.contingent_pv == pytest.approx(route_for(ctx, pde).price(ctx, pde).contingent_pv, abs=1e-9)


@pytest.mark.parametrize("spot, delta", [(70.0, 1.0), (90.0, 0.0)])
def test_greeks_at_the_maturity_instant_are_those_of_the_decided_payoff(sse_calendar, sse_sessions, desk, spot, delta):
    from dataclasses import replace
    from quantark.intraday import value_intraday
    ctx, _, _ = _maturity_ctx(sse_calendar, sse_sessions, desk, spot)
    engine = SnowballQuadEngineV2()
    for convention in ("point", "desk_bump"):
        res = value_intraday(engine, replace(ctx.request, greeks=("delta", "gamma"), greek_convention=convention))
        assert res.greek("delta").status == "ok" and res.greek("delta").value == pytest.approx(delta, abs=1e-9)
        assert res.greek("gamma").status == "ok" and res.greek("gamma").value == pytest.approx(0.0, abs=1e-6)


def test_a_desk_theta_rolling_onto_the_maturity_close_prices_the_decided_claim(sse_calendar, sse_sessions, desk):
    """The daily-KI study's maturity_day case: one hour before the last close, the roll lands on it."""
    from dataclasses import replace
    from quantark.intraday import value_intraday
    kos = _kos(_ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    fixings = [Fixing(k.timestamp, 100.0) for k in kos[:-1]]
    ctx = _ctx(sse_calendar, sse_sessions, desk, kos[-1].timestamp - timedelta(hours=1), fixings=fixings)
    theta = value_intraday(SnowballQuadEngineV2(), replace(ctx.request, greeks=("theta",),
                                                           greek_convention="desk_bump")).greek("theta")
    assert theta.status == "ok" and theta.bump == 3600.0
