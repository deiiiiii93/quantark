"""Point Greeks: derivatives of the resolved price function at the query spot, per route, with explicit statuses."""
from datetime import datetime, timedelta
from math import isfinite

import pytest

from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.execution.errors import CapabilityError
from quantark.intraday import Fixing, VarianceProfile, value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind
from quantark.intraday.greeks import POINT_PROXY_REASON
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.engine_enums import MonteCarloMethod
from intraday.conftest import SHANGHAI, dated_snowball, digital, flat_env
from intraday.reference import budgets
from intraday.reference.gaussian_reference import reference_digital, reference_snowball

T0 = datetime(2026, 3, 16)
QUAD = SnowballQuadEngineV2()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _kos(sse_calendar, sse_sessions, profile):
    probe = resolve_context(IntradayValuationRequest(product=dated_snowball(sse_calendar, T0),
                                                     pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse_sessions, variance_profile=profile))
    return [e for e in probe.timeline.events if e.kind is EventKind.KO]


def _snow_req(sse_calendar, sse_sessions, profile, when, spot=100.0, **kw):
    kos = _kos(sse_calendar, sse_sessions, profile)
    kw.setdefault("fixings", tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]))
    return IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(when(kos), spot=spot),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


def test_quad_v2_point_greeks_are_the_kernel_derivative_and_meet_the_reference(sse_calendar, sse_sessions, desk):
    req = _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=1),
                    greeks=("delta", "gamma"), greek_convention="point")
    res = value_intraday(QUAD, req)
    ctx = resolve_context(req)
    kernel = SnowballQuadEngineV2().calculate_point_greeks(ctx.numerical.product, ctx.pricing_env,
                                                          lifecycle_state=ctx.numerical.lifecycle_state, event_phase="before")
    delta, gamma = res.greek("delta"), res.greek("gamma")
    assert delta.convention == "point" and delta.status == "ok" and delta.unit == "per unit spot"
    assert delta.value == kernel["delta"] and gamma.value == kernel["gamma"]
    assert "point_evidence:kernel_derivative" in res.records
    ref = reference_snowball(ctx)
    k = budgets.REFERENCE_UNCERTAINTY_MULTIPLIER
    assert abs(delta.value - ref.delta) <= budgets.delta_budget(ref.delta, ctx.spot, 100.0) + k * ref.uncertainty_delta
    assert abs(gamma.value - ref.gamma) <= budgets.gamma_budget(ref.gamma, ctx.spot, 100.0) + k * ref.uncertainty_gamma


def test_before_the_fixing_with_the_spot_on_the_ko_barrier_is_undefined(sse_calendar, sse_sessions, desk):
    req = _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp, spot=103.0, event_phase="before",
                    greeks=("delta", "gamma"), greek_convention="point")
    for engine in (QUAD, SnowballPDESolver(PDEParams())):
        res = value_intraday(engine, req)
        for name in ("delta", "gamma"):
            g = res.greek(name)
            assert g.status == "undefined" and g.value is None and "discontinuity" in g.reason


def test_analytical_digital_point_greeks_are_the_closed_form(sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)
    req = IntradayValuationRequest(product=digital(datetime(2026, 9, 16)), pricing_env=flat_env(ts, spot=100.5),
                                   session_calendar=sse_sessions, variance_profile=desk, greeks=("gamma", "delta"),
                                   greek_convention="point")
    res = value_intraday(DigitalOptionAnalyticalEngine(), req)
    ref = reference_digital(resolve_context(req))
    assert [g.name for g in res.greeks] == ["gamma", "delta"]
    assert res.greek("delta").value == pytest.approx(ref.delta, rel=1e-12)
    assert res.greek("gamma").value == pytest.approx(ref.gamma, rel=1e-12)


def test_pde_point_greeks_follow_the_resolution_status(sse_calendar, sse_sessions, desk):
    engine = SnowballPDESolver(PDEParams())
    one_day = value_intraday(engine, _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(days=1),
                                               greeks=("delta", "gamma"), greek_convention="point"))
    assert one_day.greek("delta").status == "ok" and one_day.numerical["resolution"] == "resolved"
    assert "point_evidence:grid_stencil" in one_day.records
    one_second = value_intraday(engine, _snow_req(sse_calendar, sse_sessions, desk,
                                                  lambda kos: kos[5].timestamp - timedelta(seconds=1), spot=102.99,
                                                  greeks=("delta",), greek_convention="point"))
    d = one_second.greek("delta")
    assert d.status == "unqualified" and d.value is None and d.reason == one_second.numerical["resolution_reason"]


def test_mc_point_greeks_need_rqmc_and_report_their_uncertainty(sse_calendar, sse_sessions, desk):
    req = _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=1),
                    greeks=("delta",), greek_convention="point")
    ctx = resolve_context(req)
    rqmc = SnowballMCEngine(params=MCParams(num_paths=2 ** 12, seed=11), method=MonteCarloMethod.RANDOMIZED_QUASI)
    pg = route_for(ctx, rqmc).point_greeks(ctx, rqmc)
    assert pg.evidence == "paired_rqmc" and isfinite(pg.uncertainty["delta"]) and isfinite(pg.uncertainty["gamma"])
    assert pg.status in ("ok", "unqualified")
    pseudo = SnowballMCEngine(params=MCParams(num_paths=2 ** 12, seed=11), method=MonteCarloMethod.PSEUDO)
    with pytest.raises(CapabilityError, match="need RQMC"):
        value_intraday(pseudo, req)


def test_an_assumed_ko_has_zero_point_delta_from_the_terminated_claim(sse_calendar, sse_sessions, desk):
    req = _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp + timedelta(seconds=30), spot=104.0,
                    event_phase="after", greeks=("delta", "gamma"), greek_convention="point")
    res = value_intraday(QUAD, req)
    assert res.provisional and res.lifecycle["knocked_out"]
    assert res.greek("delta").value == 0.0 and res.greek("delta").status == "ok" and res.greek("gamma").value == 0.0
    assert "point_evidence:terminated" in res.records


def test_point_vega_and_rho_are_unqualified_proxies_until_demonstrated(sse_calendar, sse_sessions, desk):
    req = _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=1),
                    greeks=("vega", "rho", "dividend_rho"), greek_convention="point")
    res = value_intraday(QUAD, req)
    for g in res.greeks:
        assert g.status == "unqualified" and g.value is None and g.reason == POINT_PROXY_REASON and g.bump > 0.0


def test_a_demonstrated_proxy_is_a_central_difference_of_the_frozen_price_function(sse_calendar, sse_sessions, desk, monkeypatch):
    import quantark.intraday.greeks as G
    monkeypatch.setattr(G, "POINT_PROXY_DEMONSTRATED", frozenset({("QuadV2Route", "rho")}))
    req = _snow_req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=1),
                    greeks=("rho",), greek_convention="point")
    ctx = resolve_context(req)
    from quantark.asset.equity.riskmeasures.greeks import bump_envs
    up = G.cell_price(G.with_pricing_env(ctx, bump_envs.build_rate_bumped_env(ctx.pricing_env, 1e-6, direction=1.0), "u"), QUAD)
    down = G.cell_price(G.with_pricing_env(ctx, bump_envs.build_rate_bumped_env(ctx.pricing_env, 1e-6, direction=-1.0), "d"), QUAD)
    rho = value_intraday(QUAD, req).greek("rho")
    assert rho.status == "ok" and rho.value == pytest.approx((up - down) / 2e-6, rel=1e-12)
