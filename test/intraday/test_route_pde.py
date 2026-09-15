"""PDE routes: resolution status, refinement, event handling. Agreement with other engines is a CROSS-CHECK only;
the independent reference decides qualification (Gate C)."""
from datetime import datetime, time, timedelta

import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.pde import BarrierPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.execution import PricingSession
from quantark.intraday import EventKind, Fixing, VarianceProfile, resolve_context, value_intraday
from quantark.intraday.engines import route_for
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.resolution import INTRADAY_PDE_MAX_POINTS
from quantark.util.enum.option_enums import BarrierType, ObservationType, OptionType
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)
CROSS_CHECK_REL = 5e-3        # standard-accuracy PDE vs QUAD V2 / closed form: a sanity bound, not an accuracy claim


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _snow_ctx(sse_calendar, sse_sessions, profile, when, spot=100.0, phase="before", extra_fixings=(), **kw):
    prod = dated_snowball(sse_calendar, T0)
    probe = resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse_sessions, variance_profile=profile, **kw))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    ts = when(kos)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]) + tuple(Fixing(kos[5].timestamp, v) for v in extra_fixings)
    return resolve_context(IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=spot), session_calendar=sse_sessions,
                                                    variance_profile=profile, fixings=fixings, event_phase=phase, **kw)), kos


def _pde():
    return SnowballPDESolver(PDEParams())


def test_one_day_out_prices_the_twin_unrefined_and_resolved(sse_calendar, sse_sessions, desk):
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(days=1))
    out = route_for(ctx, _pde()).price(ctx, _pde())
    assert out.numerical["resolution"] == "resolved" and out.records == ()
    assert out.contingent_pv == _pde().price(ctx.numerical.product, ctx.pricing_env)
    quad = route_for(ctx, SnowballQuadEngineV2()).price(ctx, SnowballQuadEngineV2()).contingent_pv
    assert out.contingent_pv == pytest.approx(quad, rel=CROSS_CHECK_REL), "cross-check with QUAD V2, not a qualification"


def test_fifteen_minutes_out_refines_the_grid_to_resolve_the_layer(sse_calendar, sse_sessions, desk):
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(minutes=15), spot=102.5)
    engine = _pde()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.numerical["points"] > 400 and out.numerical["resolution"] == "resolved" and out.records
    assert out.engine_used is not engine and out.numerical["fill_scaled"] is False
    with PricingSession() as session:                     # the kernel re-dispatches the refined clone
        res = value_intraday(engine, ctx.request, session=session)
    assert any(r.startswith("manifest:") for r in res.records) and res.contingent_pv == out.contingent_pv


def test_one_second_out_is_unqualified_at_the_point_cap(sse_calendar, sse_sessions, desk):
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(seconds=1), spot=102.99)
    out = route_for(ctx, _pde()).price(ctx, _pde())
    assert out.numerical["resolution"] == "unqualified" and out.numerical["points"] == INTRADAY_PDE_MAX_POINTS
    assert "points needed" in out.numerical["resolution_reason"]


def test_a_fixing_inside_a_zero_variance_window_reports_deterministic(sse_calendar, sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, only, lambda kos: kos[5].timestamp.replace(hour=12, minute=0),
                       fixing_time_of_day=time(13, 0))
    out = route_for(ctx, _pde()).price(ctx, _pde())
    assert out.numerical["resolution"] == "deterministic"


def test_before_at_the_fixing_above_the_barrier_is_the_ko_cash(sse_calendar, sse_sessions, desk):
    ctx, kos = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp, spot=104.0)
    out = route_for(ctx, _pde()).price(ctx, _pde())
    assert out.contingent_pv == pytest.approx(kos[5].cash, abs=1e-9)


def test_continuous_barrier_under_zero_carry_cross_checks_the_exact_closed_form(sse_sessions, desk):
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=115.0, barrier_type=BarrierType.UP_OUT,
                            exercise_date=datetime(2026, 12, 15), observation_type=ObservationType.CONTINUOUS)
    ctx = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI), r=0.0, q=0.0),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    pde = route_for(ctx, BarrierPDESolver(PDEParams())).price(ctx, BarrierPDESolver(PDEParams()))
    exact = route_for(ctx, BarrierAnalyticalEngine()).price(ctx, BarrierAnalyticalEngine())
    assert exact.method == "analytical_zero_carry_time_change"
    assert pde.contingent_pv == pytest.approx(exact.contingent_pv, rel=CROSS_CHECK_REL), "cross-check, not a qualification"
