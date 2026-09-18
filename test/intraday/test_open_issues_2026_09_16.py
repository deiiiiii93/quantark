"""Independent controls for the remaining re-review issues."""
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.pde import EuropeanPDESolver
from quantark.intraday import resolve_context, value_intraday
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.timestamp import SECONDS_PER_YEAR
from intraday.conftest import SHANGHAI, flat_env
from intraday.controls import fixtures as C
from intraday.controls import tolerances as budgets
from intraday.controls.fixtures import build_context


@pytest.mark.parametrize("offset", [-0.47, -0.05, 0.0, 0.05, 0.47])
def test_pde_gamma_is_evaluated_at_the_query_on_a_nonuniform_grid(offset):
    # Manufactured smooth value: cubic in log spot, with known derivatives.
    # Its curvature varies linearly, exposing nearest-node gamma immediately.
    x = np.log(100) + np.array([-0.06, -0.03, -0.009, 0.008, 0.029, 0.06])
    z = np.log(100) + offset * 0.017
    u = x - np.log(100)
    values = 7 + 2 * u + 3 * u**2 + 40 * u**3
    delta, gamma = EuropeanPDESolver()._calculate_delta_gamma(values, x, z, np.exp(z))
    q = z - np.log(100)
    d1, d2 = 2 + 6*q + 120*q*q, 6 + 240*q
    assert delta == pytest.approx(d1 / np.exp(z), abs=2e-14)
    assert gamma == pytest.approx((d2-d1) / np.exp(2*z), abs=2e-14)


def killed_density_uo_call(spot, strike, barrier, tau, variance, rate, dividend):
    """Independent absorbing Gaussian density integrated against a call payoff.

    Supports complex T/W for a cancellation-free derivative control. This uses
    reflection of the transition density, not the engine's RR case table or
    its finite-difference theta implementation. No rebate/settlement lag.
    """
    width = np.sqrt(variance)
    a, b, x = np.log(strike), np.log(barrier), np.log(spot)
    drift = (rate-dividend)*tau - variance/2

    def band(mean):
        cash = ndtr((b-mean)/width) - ndtr((a-mean)/width)
        asset = np.exp(mean + variance/2) * (ndtr((b-mean-variance)/width) - ndtr((a-mean-variance)/width))
        return asset - strike*cash

    weight = np.exp(2*drift/variance*(b-x))
    return np.exp(-rate*tau) * (band(x+drift) - weight*band(2*b-x+drift))


@pytest.mark.parametrize("profile,rate,dividend", [("desk", 0.0, 0.0), ("sessions_only", 0.0, 0.0),
                                                 ("uniform", 0.03, 0.01)])
@pytest.mark.parametrize("seconds", [1, 10, 60, 3600, 86400, 35*86400, 90*86400])
@pytest.mark.parametrize("offset", ["bp-1", "sd-1", "sd-2"])
def test_barrier_point_theta_matches_the_killed_density_control(profile, rate, dividend, seconds, offset):
    ctx, _ = build_context(C.Cell("barrier_uo_zero_carry", "analytical", profile,
                                  timedelta(seconds=seconds), offset, "ko"))
    env = replace(ctx.request.pricing_env, rate_curve=flat_env(ctx.valuation_timestamp, r=rate).rate_curve,
                  div_yield=flat_env(ctx.valuation_timestamp, q=dividend).div_yield)
    request = replace(ctx.request, pricing_env=env, greeks=("theta",), greek_convention="point")
    ctx = resolve_context(request)
    if profile == "uniform" and seconds > 86400:
        from quantark.execution.errors import CapabilityError
        # Weekends merge overnight segments, so this profile ceases to have
        # constant calendar variance; nonzero-carry first passage stays refused.
        with pytest.raises(CapabilityError, match="not exact"):
            value_intraday(BarrierAnalyticalEngine(), request)
        return
    result = value_intraday(BarrierAnalyticalEngine(), request)
    theta = result.greek("theta")
    assert theta.status == "ok", theta.reason
    tau = ctx.numerical.maturity_tau
    variance = ctx.pricing_env.vol_surface.total_variance(100.0, tau, ctx.spot)
    variance_rate = 0.2**2 * ctx.time_map.initial_slope()
    eps = 1e-20
    reference = killed_density_uo_call(ctx.spot, 100, 103, tau-1j*eps, variance-1j*eps*variance_rate,
                                       rate, dividend).imag / eps * 3600 / SECONDS_PER_YEAR
    tolerance = budgets.theta_budget(reference, 100.0)          # the frozen normalized theta tolerance, per hour
    assert abs(theta.value - reference) <= tolerance
    assert theta.error_estimate is not None and theta.error_estimate <= tolerance
    assert result.to_dict()["greeks"][0]["error_estimate"] == theta.error_estimate


def test_barrier_theta_units_and_lunch_zero_variance_are_preserved():
    ts = datetime(2026, 9, 16, 12, tzinfo=SHANGHAI)
    request = IntradayValuationRequest(product=C.product("barrier_uo_zero_carry"),
        pricing_env=flat_env(ts, spot=102, r=0, q=0), session_calendar=C.sse(),
        variance_profile=C.profile("sessions_only"), greeks=("theta",), greek_convention="point")
    hourly = value_intraday(BarrierAnalyticalEngine(), request).greek("theta")
    daily = value_intraday(BarrierAnalyticalEngine(), replace(request, theta_unit="day")).greek("theta")
    assert hourly.status == daily.status == "ok"
    assert hourly.error_estimate is not None and abs(hourly.value) <= budgets.theta_budget(0.0, 100.0)
    assert daily.value == pytest.approx(24*hourly.value)
    assert daily.error_estimate == pytest.approx(24*hourly.error_estimate)


def test_barrier_theta_reports_a_nonconvergent_ladder_as_a_diagnostic(monkeypatch):
    import quantark.intraday.greeks as G
    ctx, _ = build_context(C.Cell("barrier_uo_zero_carry", "analytical", "desk", timedelta(hours=1), "bp-1", "ko"))
    # Rolled prices at h/4, h/2, h, 2h. The kink sits in the FINEST step, so the finest stencil moves away from the
    # two coarser ones (which agree): the ladder does not converge, and the value is still reported with that note.
    values = iter((2.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(G, "_rolled_value", lambda *_: (next(values), True))
    greek = G.analytical_theta_limit(ctx, BarrierAnalyticalEngine(), price_base=1.0, unit="hour")
    assert greek.status == "ok" and greek.value is not None and greek.error_estimate is not None
    assert "did not converge" in greek.reason


@pytest.mark.parametrize("days", [29, 30, 35, 60, 90])
def test_long_gap_fixture_has_no_future_history_or_hidden_fixings(days):
    from quantark.intraday.greeks import seconds_to_first_event
    cell = C.Cell("snowball_long_gap", "quad_v2", "desk", timedelta(days=days), "bp+1", "ki")
    ctx, _ = build_context(cell)
    assert seconds_to_first_event(ctx) == days*86400
    assert not ctx.request.fixings and not ctx.provisional
    assert ctx.numerical.lifecycle_state.alive



def test_long_gap_and_monthly_history_have_the_same_conditional_economics():
    from quantark.intraday.capability import economic_identity
    a, _ = build_context(C.Cell("snowball_discrete_ki", "quad_v2", "desk", timedelta(days=29), "bp+1", "ki"))
    b, _ = build_context(C.Cell("snowball_long_gap", "quad_v2", "desk", timedelta(days=35), "sd-1", "ko"))
    assert economic_identity(a) == economic_identity(b)


def test_economic_identity_reads_the_delivered_market_of_a_bump_context():
    from quantark.intraday.capability import economic_identity
    from quantark.intraday.greeks import point_proxy_env, with_pricing_env
    ctx, _ = build_context(C.Cell("snowball_discrete_ki", "quad_v2", "desk", timedelta(hours=1), "bp+1", "ki"))
    bumped = with_pricing_env(ctx, point_proxy_env(ctx, "rho", 0.01, 1.0), "rate")
    assert bumped.request is ctx.request
    assert economic_identity(bumped) != economic_identity(ctx)



@pytest.mark.parametrize("convention", ["point", "desk_bump"])
def test_monthly_contract_immediately_after_a_fixing_is_inside_the_extended_window(convention):
    from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
    fixing, history = C.fixing_and_history("snowball_discrete_ki")
    now = history[-1].timestamp + timedelta(seconds=1)
    assert (fixing-now).total_seconds() > 29*86400
    request = IntradayValuationRequest(product=C.product("snowball_discrete_ki"),
        pricing_env=flat_env(now, spot=100.0), session_calendar=C.sse(), variance_profile=C.profile("desk"),
        fixings=history, greeks=("delta", "gamma", "vega", "rho", "dividend_rho", "theta"),
        greek_convention=convention)
    result = value_intraday(SnowballQuadEngineV2(), request)
    assert not result.provisional
    assert all(g.status == "ok" for g in result.greeks), [(g.name, g.reason) for g in result.greeks]
