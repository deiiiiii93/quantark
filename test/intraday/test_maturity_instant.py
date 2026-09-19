"""The maturity close under BEFORE: every remaining event sits at the valuation instant and is decided on the known spot.

Found 2026-09-19 while the daily-KI study ran. The day-level engines carry a zero-maturity shortcut that returns the
terminal payoff of the CARRIED knock-in state and never applies the observations still pending at the instant, although
each of them decides such observations correctly at every other instant. Through the intraday runtime's float-time twin
that shortcut became reachable: Monte Carlo missed the knock-in and the knock-out, the PDE solver missed the knock-out,
and QUAD V1 (no intraday route, so reachable only directly) missed both. The rebate here is 5% against a 12% knock-out
coupon ON PURPOSE: with equal rates a missed knock-out pays the same cash and no test can see it.
"""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.intraday import value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind, EventPhase
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)
REBATE, COUPON = 0.05, 0.12
#: spot -> the decided claim: knocked out at 104 >= 103 (the 12% coupon), survived at 90 (the 5% rebate),
#: knocked in at 70 <= 75 (the loss 70 - 100)
DECIDED = [(104.0, 12.0), (90.0, 5.0), (70.0, -30.0)]
ROUTED = {
    "mc": lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=2048)),
    "pde": lambda: SnowballPDESolver(PDEParams()),
    "quad_v2": lambda: SnowballQuadEngineV2(),
}
DAY_LEVEL = {**{k: v for k, v in ROUTED.items() if k != "quad_v2"},
             "quad_v1": lambda: SnowballQuadEngine(params=QuadParams(grid_points=1001))}


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _ctx(sse_calendar, sse_sessions, profile, ts, spot=100.0, phase=EventPhase.BEFORE, fixings=()):
    prod = dated_snowball(sse_calendar, T0, ko_rate=COUPON, rebate_rate=REBATE)
    return resolve_context(IntradayValuationRequest(
        product=prod, pricing_env=flat_env(ts, spot=spot), session_calendar=sse_sessions, variance_profile=profile,
        event_phase=phase, fixings=tuple(fixings)))


def _maturity(sse_calendar, sse_sessions, desk, spot, before=timedelta(0)):
    """(context at the maturity close less ``before``, the last KO event, the earlier fixings)."""
    probe = _ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    fixings = [Fixing(k.timestamp, 100.0) for k in kos[:-1]]
    return _ctx(sse_calendar, sse_sessions, desk, kos[-1].timestamp - before, spot=spot, fixings=fixings), kos[-1], fixings


@pytest.mark.parametrize("name", sorted(ROUTED))
@pytest.mark.parametrize("spot, expected", DECIDED)
def test_every_route_decides_the_maturity_close_on_the_known_spot(sse_calendar, sse_sessions, desk, name, spot, expected):
    ctx, last, fixings = _maturity(sse_calendar, sse_sessions, desk, spot)
    assert ctx.numerical.maturity_tau == 0.0 and not ctx.numerical.terminated
    engine = ROUTED[name]()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.contingent_pv == pytest.approx(expected, abs=1e-12)
    assert out.method == "decided_at_valuation" and out.exact                  # one resolution for every route
    # the identity every other fixing instant satisfies: decided now = what the lifecycle pays when fixed at the spot
    after = resolve_context(replace(ctx.request, event_phase=EventPhase.AFTER,
                                    fixings=tuple(fixings) + (Fixing(last.timestamp, spot),)))
    assert after.numerical.terminated and out.contingent_pv == pytest.approx(after.numerical.paid_cash, abs=1e-12)


@pytest.mark.parametrize("name", sorted(DAY_LEVEL))
@pytest.mark.parametrize("spot, expected", DECIDED)
def test_a_day_level_engine_handed_the_zero_maturity_twin_decides_the_pending_observations(
        sse_calendar, sse_sessions, desk, name, spot, expected):
    """No route in between: the engine's own zero-maturity shortcut. QUAD V1 has no intraday route at all, so this
    is the only way its shortcut can be reached, and the only place it can be fixed."""
    ctx, _, _ = _maturity(sse_calendar, sse_sessions, desk, spot)
    value = float(DAY_LEVEL[name]().price(ctx.numerical.product, ctx.pricing_env))
    assert value == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("name", sorted(DAY_LEVEL))
def test_a_carried_knock_in_still_decides_the_terminal_payoff_and_blocks_nothing_else(sse_calendar, sse_sessions, desk, name):
    """The shortcut's original job is kept: a claim knocked in on an earlier day pays the knocked-in payoff at 90."""
    probe = _ctx(sse_calendar, sse_sessions, desk, datetime(2026, 9, 15, tzinfo=SHANGHAI))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    fixings = [Fixing(k.timestamp, 70.0 if i == 3 else 100.0) for i, k in enumerate(kos[:-1])]     # knocked in in July
    ctx = _ctx(sse_calendar, sse_sessions, desk, kos[-1].timestamp, spot=90.0, fixings=fixings)
    assert ctx.numerical.knocked_in and ctx.numerical.maturity_tau == 0.0
    engine = DAY_LEVEL[name]()
    assert route_or_direct(ctx, engine) == pytest.approx(-10.0, abs=1e-12)


def route_or_direct(ctx, engine):
    try:
        return route_for(ctx, engine).price(ctx, engine).contingent_pv
    except Exception:                                   # QUAD V1: no intraday route
        return float(engine.price(ctx.numerical.product, ctx.pricing_env))


@pytest.mark.parametrize("name", ["mc", "pde"])
@pytest.mark.parametrize("spot, delta", [(70.0, 1.0), (90.0, 0.0)])
def test_greeks_at_the_maturity_close_are_those_of_the_decided_payoff(sse_calendar, sse_sessions, desk, name, spot, delta):
    ctx, _, _ = _maturity(sse_calendar, sse_sessions, desk, spot)
    for convention in ("point", "desk_bump"):
        res = value_intraday(ROUTED[name](), replace(ctx.request, greeks=("delta", "gamma"), greek_convention=convention))
        assert res.greek("delta").status == "ok" and res.greek("delta").value == pytest.approx(delta, abs=1e-9)
        assert res.greek("gamma").status == "ok" and res.greek("gamma").value == pytest.approx(0.0, abs=1e-6)


def test_a_desk_theta_rolling_onto_a_knock_out_at_the_maturity_close_agrees_across_routes(sse_calendar, sse_sessions, desk):
    """One hour before the last close at spot 104: the roll lands on the instant, where the claim knocks out. A route
    that pays the rebate there instead of the coupon reports a theta wrong by 7 per hour."""
    ctx, _, _ = _maturity(sse_calendar, sse_sessions, desk, 104.0, before=timedelta(hours=1))
    thetas = {}
    for name in ("pde", "quad_v2"):
        theta = value_intraday(ROUTED[name](), replace(ctx.request, greeks=("theta",), greek_convention="desk_bump")).greek("theta")
        assert theta.status == "ok"
        thetas[name] = theta.value
    assert thetas["pde"] == pytest.approx(thetas["quad_v2"], abs=5e-2), thetas
