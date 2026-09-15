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
from quantark.intraday.resolution import INTRADAY_PDE_MAX_POINTS, REQUIRED_STEPS_PER_LAYER
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


def test_far_from_the_fixing_prices_the_twin_unrefined_and_resolved(sse_calendar, sse_sessions, desk):
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[4].timestamp + timedelta(days=1))
    out = route_for(ctx, _pde()).price(ctx, _pde())
    assert out.numerical["resolution"] == "resolved" and out.records == ()
    assert out.contingent_pv == _pde().price(ctx.numerical.product, ctx.pricing_env)
    quad = route_for(ctx, SnowballQuadEngineV2()).price(ctx, SnowballQuadEngineV2()).contingent_pv
    assert out.contingent_pv == pytest.approx(quad, rel=CROSS_CHECK_REL), "cross-check with QUAD V2, not a qualification"


def test_one_day_out_refines_the_time_fill_so_no_step_jumps_the_layer(sse_calendar, sse_sessions, desk):
    # 4 steps per day on the desk profile put a whole session's variance into one Crank-Nicolson step
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(days=1))
    engine = _pde()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.numerical["resolution"] == "resolved" and out.numerical["points"] == 400
    assert out.numerical["steps_per_day"] > 4.0 and out.numerical["steps_per_layer"] >= REQUIRED_STEPS_PER_LAYER
    assert out.numerical["fill_scaled"] is False and "steps per day" in out.records[0]
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


def test_refinement_respects_the_grid_memory_budget_and_says_so(sse_calendar, sse_sessions, desk, monkeypatch):
    import quantark.intraday.engines.pde as pde_route
    budget = 2_000_000 * 25
    monkeypatch.setattr(pde_route, "INTRADAY_PDE_MAX_GRID_BYTES", budget)
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(minutes=15), spot=102.5)
    out = route_for(ctx, _pde()).price(ctx, _pde())
    n = out.numerical
    assert pde_route.bytes_per_grid_cell(out.engine_used, ctx.numerical.product) == 2 * 8 + 9      # two value surfaces
    assert n["points"] * (n["requested_steps"] + 1) * 25 <= budget
    assert n["resolution"] == "unqualified" and "grid memory budget" in n["resolution_reason"]


def test_a_memory_phoenix_budget_counts_every_coupon_state_surface(sse_calendar, sse_sessions, desk, monkeypatch):
    # a 12-coupon memory Phoenix keeps 2(12+1)+2 surfaces: a per-cell budget sized for a snowball let one price use GiBs
    from quantark.asset.equity.engine.pde import PhoenixPDESolver
    import quantark.intraday.engines.pde as pde_route
    from intraday.conftest import dated_phoenix
    monkeypatch.setattr(pde_route, "INTRADAY_PDE_MAX_GRID_BYTES", 200_000_000)       # keeps the test light
    product = dated_phoenix(sse_calendar, T0)
    ctx = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(datetime(2026, 4, 16, 14, 59, 50, tzinfo=SHANGHAI)),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    engine = PhoenixPDESolver(PDEParams())
    out = route_for(ctx, engine).price(ctx, engine)
    per_cell = pde_route.bytes_per_grid_cell(out.engine_used, ctx.numerical.product)
    assert per_cell == 8 * (2 * (12 + 1) + 2) + 9
    n = out.numerical
    assert n["points"] * (n["requested_steps"] + 1) * per_cell <= pde_route.INTRADAY_PDE_MAX_GRID_BYTES


def test_the_route_does_not_hand_back_full_value_surfaces(sse_calendar, sse_sessions, desk):
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(minutes=15), spot=102.5)
    out = route_for(ctx, _pde()).price(ctx, _pde())
    assert out.engine_used._grid_v0 is None and out.engine_used._grid_v1 is None
    again = out.engine_used.price(ctx.numerical.product, ctx.pricing_env)       # a later solve rebuilds them
    assert again == out.contingent_pv


def test_the_route_leaves_no_coefficient_memo_on_the_request_market(sse_calendar, sse_sessions, desk):
    # the legacy memo is keyed on the (request-owned) rate curve: a batch holding its requests would keep every
    # route's coefficient sets alive (a 100-item PDE batch passed 7 GiB)
    from quantark.asset.equity.engine.pde import base_pde_solver
    ctx, _ = _snow_ctx(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(minutes=15), spot=102.5)
    engine = _pde()
    out = route_for(ctx, engine).price(ctx, engine)
    assert id(ctx.pricing_env.rate_curve) not in base_pde_solver._ENV_STEP_COEFF_MEMO
    route_for(ctx, engine).point_greeks(ctx, engine)
    assert id(ctx.pricing_env.rate_curve) not in base_pde_solver._ENV_STEP_COEFF_MEMO
    assert route_for(ctx, engine).price(ctx, engine).contingent_pv == out.contingent_pv


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


def test_a_one_touch_barrier_snapped_to_the_next_node_is_unqualified(sse_sessions, desk):
    from quantark.asset.equity.engine.pde import OneTouchPDESolver
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
    from quantark.util.enum.option_enums import BarrierDirection
    product = OneTouchOption(barrier=103.0, barrier_direction=BarrierDirection.UP, rebate=1.0,
                             exercise_date=datetime(2026, 9, 16), observation_type=ObservationType.CONTINUOUS)
    ctx = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI),
                                                                                        spot=101.0, r=0.0, q=0.0),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    out = route_for(ctx, OneTouchPDESolver(PDEParams())).price(ctx, OneTouchPDESolver(PDEParams()))
    assert out.numerical["resolution"] == "unqualified" and "node overwrite" in out.numerical["resolution_reason"]
    ko = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=103.0, barrier_type=BarrierType.UP_OUT,
                       exercise_date=datetime(2026, 9, 16), observation_type=ObservationType.CONTINUOUS)
    ko_ctx = resolve_context(IntradayValuationRequest(product=ko, pricing_env=ctx.request.pricing_env,
                                                      session_calendar=sse_sessions, variance_profile=desk))
    edge = route_for(ko_ctx, BarrierPDESolver(PDEParams())).price(ko_ctx, BarrierPDESolver(PDEParams()))
    assert "node overwrite" not in edge.numerical["resolution_reason"], "a continuous knock-out barrier is a hard grid edge"


def test_continuous_barrier_under_zero_carry_cross_checks_the_exact_closed_form(sse_sessions, desk):
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=115.0, barrier_type=BarrierType.UP_OUT,
                            exercise_date=datetime(2026, 12, 15), observation_type=ObservationType.CONTINUOUS)
    ctx = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI), r=0.0, q=0.0),
                                                   session_calendar=sse_sessions, variance_profile=desk))
    pde = route_for(ctx, BarrierPDESolver(PDEParams())).price(ctx, BarrierPDESolver(PDEParams()))
    exact = route_for(ctx, BarrierAnalyticalEngine()).price(ctx, BarrierAnalyticalEngine())
    assert exact.method == "analytical_zero_carry_time_change"
    assert pde.contingent_pv == pytest.approx(exact.contingent_pv, rel=CROSS_CHECK_REL), "cross-check, not a qualification"
