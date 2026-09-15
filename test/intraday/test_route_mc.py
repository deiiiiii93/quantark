"""MC routes: sampling evidence, today's fixing, zero-variance steps. Agreement with exact routes is checked in
standard errors (a CROSS-CHECK); the independent reference decides qualification (Gate C)."""
from datetime import datetime, timedelta
from math import exp

import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine, DigitalOptionMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.execution import PricingSession
from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context, value_intraday
from quantark.intraday.engines import route_for
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.enum.option_enums import BarrierType, ObservationType, OptionType
from intraday.conftest import SHANGHAI, dated_snowball, digital, flat_env

T0 = datetime(2026, 3, 16)
SIGMAS = 4.0


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _mc(cls=SnowballMCEngine, method=MonteCarloMethod.RANDOMIZED_QUASI, num_paths=2 ** 14, **kw):
    return cls(params=MCParams(num_paths=num_paths, seed=7), method=method, **kw)


def _snow(sse_calendar, sse_sessions, profile, when, spot=100.0):
    prod = dated_snowball(sse_calendar, T0)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse_sessions, variance_profile=profile))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    ts, s = when(kos), spot(kos) if callable(spot) else spot
    return resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=s), session_calendar=sse_sessions,
                                                    variance_profile=profile, fixings=tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]))), kos


def test_snowball_rqmc_reports_its_sampling_evidence_and_cross_checks_quad_v2(sse_calendar, sse_sessions, desk):
    ctx, _ = _snow(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(days=1))
    out = route_for(ctx, _mc()).price(ctx, _mc())
    n = out.numerical
    assert n["estimator"] == "scramble_means" and n["std_error"] > 0.0 and n["n_paths"] == 2 ** 14 and n["seed"] == 7
    quad = route_for(ctx, SnowballQuadEngineV2()).price(ctx, SnowballQuadEngineV2()).contingent_pv
    assert abs(out.contingent_pv - quad) <= SIGMAS * n["std_error"], "cross-check with QUAD V2, not a qualification"
    with PricingSession() as session:
        res = value_intraday(_mc(), ctx.request, session=session)
    assert res.price == out.contingent_pv + res.pending_receivable_pv and any(r.startswith("manifest:") for r in res.records)


def test_todays_fixing_before_the_event_above_the_barrier_is_the_ko_cash(sse_calendar, sse_sessions, desk):
    ctx, kos = _snow(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp, spot=104.0)
    out = route_for(ctx, _mc(method=MonteCarloMethod.PSEUDO)).price(ctx, _mc(method=MonteCarloMethod.PSEUDO))
    assert out.contingent_pv == pytest.approx(kos[5].cash, rel=1e-12)


def test_zero_variance_lunch_steps_are_pure_drift_under_identical_draws(sse_calendar, sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    shift = timedelta(minutes=30)
    tau = shift.total_seconds() / (365 * 86400.0)
    early, _ = _snow(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp.replace(hour=11, minute=40))
    late, _ = _snow(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp.replace(hour=11, minute=40) + shift,
                    spot=100.0 * exp((0.03 - 0.01) * tau))
    engine = _mc(method=MonteCarloMethod.PSEUDO)   # sequential increments: the same draw per step for both anchors
    p_early = route_for(early, engine).price(early, engine).contingent_pv
    p_late = route_for(late, engine).price(late, engine).contingent_pv
    assert p_early / exp(-0.03 * tau) == pytest.approx(p_late, rel=1e-13)


def test_barrier_mc_with_bridge_cross_checks_the_exact_zero_carry_route(sse_sessions, desk):
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=115.0, barrier_type=BarrierType.UP_OUT,
                            exercise_date=datetime(2026, 12, 15), observation_type=ObservationType.CONTINUOUS)
    ctx = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI), r=0.0, q=0.0),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    engine = _mc(BarrierOptionMCEngine, num_paths=2 ** 15, use_brownian_bridge=True)
    out = route_for(ctx, engine).price(ctx, engine)
    exact = route_for(ctx, BarrierAnalyticalEngine()).price(ctx, BarrierAnalyticalEngine())
    assert out.numerical["bridge"] and abs(out.contingent_pv - exact.contingent_pv) <= SIGMAS * out.numerical["std_error"]


def test_digital_mc_cross_checks_the_closed_form(sse_sessions, desk):
    ctx = resolve_context(IntradayValuationRequest(product=digital(datetime(2026, 12, 15)),
                                                   pricing_env=flat_env(datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    out = route_for(ctx, _mc(DigitalOptionMCEngine)).price(ctx, _mc(DigitalOptionMCEngine))
    exact = route_for(ctx, DigitalOptionAnalyticalEngine()).price(ctx, DigitalOptionAnalyticalEngine())
    assert abs(out.contingent_pv - exact.contingent_pv) <= SIGMAS * out.numerical["std_error"]
