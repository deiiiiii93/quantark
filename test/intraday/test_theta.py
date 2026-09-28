"""Intraday theta (a declared forward step on the frozen market) and the roll-through-events scenario."""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from golden_compare import GOLDEN_REL_TOL

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.intraday import Fixing, VarianceProfile, resolve_context, value_intraday
from quantark.intraday.events import EventKind
from quantark.intraday.greeks import resolve_theta_step
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.roll import roll_through_events
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

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


def _req(sse_calendar, sse_sessions, profile, ts, spot=100.0, **kw):
    kos = _kos(sse_calendar, sse_sessions, profile)
    kw.setdefault("fixings", tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]))
    return IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts, spot=spot),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


FIXING = datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI)          # the sixth KO/KI fixing of the fixture


def test_a_step_inside_the_segment_is_taken_as_requested(sse_calendar, sse_sessions, desk):
    req = _req(sse_calendar, sse_sessions, desk, FIXING - timedelta(minutes=1), greeks=("theta",), greek_convention="desk_bump",
               theta_step=timedelta(seconds=30), theta_unit="hour")
    step = resolve_theta_step(resolve_context(req), req.theta_step, req.theta_unit)
    assert step.actual == timedelta(seconds=30) and not step.adjusted and step.side == "forward"
    res = value_intraday(QUAD, req)
    theta = res.greek("theta")
    assert theta.unit == "PnL per hour" and theta.convention == "desk_bump" and theta.bump == 30.0
    assert res.numerical["theta_step_requested_s"] == 30.0 and res.numerical["theta_step_actual_s"] == 30.0
    assert res.numerical["theta_adjusted"] is False and res.numerical["theta_unit"] == "hour"
    # the roll is a finite move of the route's prices at the requested step; its accuracy is a study's business
    assert theta.status == "ok" and theta.value is not None and "finite roll" in theta.reason
    default = value_intraday(QUAD, replace(req, theta_step=None)).greek("theta")
    assert default.status == "ok" and default.bump == 60.0            # the default hour, clamped to the fixing


def test_a_step_crossing_the_fixing_is_clamped_to_land_on_it_before(sse_calendar, sse_sessions, desk):
    req = _req(sse_calendar, sse_sessions, desk, FIXING - timedelta(seconds=15), theta_step=timedelta(seconds=30))
    step = resolve_theta_step(resolve_context(req), req.theta_step, "hour")
    assert step.actual == timedelta(seconds=15) and step.adjusted and step.side == "forward_clamped_to_event"
    assert step.divisor == pytest.approx(15.0 / 3600.0, rel=1e-15)


def test_at_the_fixing_before_local_theta_is_undefined_never_zero(sse_calendar, sse_sessions, desk):
    res = value_intraday(QUAD, _req(sse_calendar, sse_sessions, desk, FIXING, greeks=("theta",), greek_convention="point"))
    theta = res.greek("theta")
    assert theta.status == "undefined" and theta.value is None and "event boundary" in theta.reason


def test_theta_across_the_lunch_plateau_is_the_pure_carry_of_direct_contexts(sse_calendar, sse_sessions):
    from quantark.intraday.greeks import _rolled_value
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ts = datetime(2026, 9, 16, 11, 45, tzinfo=SHANGHAI)
    req = _req(sse_calendar, sse_sessions, only, ts)
    res = value_intraday(QUAD, req)
    later = value_intraday(QUAD, _req(sse_calendar, sse_sessions, only, ts + timedelta(minutes=30)))
    # the frozen-market roll through the zero-weight lunch IS direct valuation at the later instant
    rolled, exact = _rolled_value(resolve_context(req), QUAD, timedelta(minutes=30))
    assert rolled == pytest.approx(later.price, abs=1e-10) and not exact
    theta = value_intraday(QUAD, replace(req, greeks=("theta",), greek_convention="desk_bump",
                                         theta_unit="minute")).greek("theta")
    assert theta.unit == "PnL per minute" and theta.convention == "desk_bump" and theta.bump == 3600.0
    assert theta.status == "ok" and theta.value * 60.0 == pytest.approx(
        _rolled_value(resolve_context(req), QUAD, timedelta(hours=1))[0] - res.price, abs=1e-12)


def test_roll_through_events_needs_an_outcome_for_every_crossed_instant(sse_calendar, sse_sessions, desk):
    req = _req(sse_calendar, sse_sessions, desk, FIXING - timedelta(minutes=1))
    to = FIXING + timedelta(seconds=30)
    with pytest.raises(ValidationError, match="15:00"):
        roll_through_events(QUAD, req, to, outcomes=())
    res = roll_through_events(QUAD, req, to, outcomes=(Fixing(FIXING, 104.0),))
    assert res.lifecycle["knocked_out"] and not res.provisional and res.valuation_timestamp == to
    assert "scenario:roll_through_events" in res.records and res.contingent_pv == 0.0


def test_roll_through_events_landing_on_an_event_is_after_that_event(sse_calendar, sse_sessions, desk):
    req = _req(sse_calendar, sse_sessions, desk, FIXING - timedelta(minutes=1))
    res = roll_through_events(QUAD, req, FIXING, outcomes=(Fixing(FIXING, 100.0),))
    assert res.phase.value == "after" and res.lifecycle["alive"] and not res.provisional


def test_legacy_numerical_theta_is_unchanged():
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
    from test_snowball_pde import create_pricing_env, create_standard_snowball
    theta = GreeksCalculator().calculate_numerical_theta(create_standard_snowball(), create_pricing_env(), SnowballQuadEngineV2())
    # frozen on the banking machine; a bumped QUAD solve drifts by ULPs across architectures
    assert theta == pytest.approx(float.fromhex(LEGACY_THETA_HEX), rel=GOLDEN_REL_TOL)


LEGACY_THETA_HEX = "0x1.3068ab6111800p+8"        # identical on the branch base 64b2832d
