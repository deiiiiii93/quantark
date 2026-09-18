"""Regressions for docs/superpowers/reviews/intraday-pricing-2026-09-16/REREVIEW.md (findings R1-R7).

One group per finding, named by its number. The reviewer's probes are in that
directory's ``reproduce_followup.py``; these tests pin the corrected behaviour.
"""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, BarrierLifecycleState, ValuationPoint
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.execution.errors import CapabilityError
from quantark.intraday import Fixing, VarianceProfile, resolve_context, spot_curve, value_intraday
from quantark.intraday.admissibility import analytical_barrier_admissibility
from quantark.intraday.events import EventKind, resolve_timeline
from quantark.intraday.result import CashflowComponent
from quantark.param.rrf.rate_curve import LinearRateCurve, LogLinearRateCurve
from quantark.util.enum.option_enums import BarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, dated_phoenix, dated_snowball, digital, flat_env

T0 = datetime(2026, 3, 16)
QUAD = SnowballQuadEngineV2()
DIGITAL = DigitalOptionAnalyticalEngine()
BARRIER = BarrierAnalyticalEngine()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _req(product, sessions, profile, ts, **kw):
    from quantark.intraday.request import IntradayValuationRequest
    env = kw.pop("env", flat_env(ts))
    return IntradayValuationRequest(product=product, pricing_env=env, session_calendar=sessions,
                                    variance_profile=profile, **kw)


def _events(product, sessions, kind):
    probe = flat_env(datetime(2026, 4, 1, tzinfo=SHANGHAI))
    return [e for e in resolve_timeline(product, sessions, probe).events if e.kind is kind]


def _uo_continuous(expiry=datetime(2026, 9, 16)):
    return BarrierOption(strike=100, option_type=OptionType.CALL, barrier=103, barrier_type=BarrierType.UP_OUT,
                         initial_date=T0, exercise_date=expiry, observation_type=ObservationType.CONTINUOUS)


# --- R1: a same-day close checkpoint reported before that close ----------------------------------------------------
def _ko_checkpoint(sse_calendar, sse_sessions, desk):
    """(product, sixth KO event, the confirmed state a reconstruction just after that KO's actual fixing wrote)."""
    product = dated_snowball(sse_calendar, T0)
    for record in product.barrier_config.ko_observation_schedule.records:
        record.settlement_date = record.observation_date + timedelta(days=5)
    kos = _events(product, sse_sessions, EventKind.KO)
    after = _req(product, sse_sessions, desk, kos[5].timestamp + timedelta(seconds=1),
                 fixings=tuple(Fixing(e.timestamp, 100.0) for e in kos[:5]) + (Fixing(kos[5].timestamp, 104.0),))
    state = resolve_context(after).reconstruction.state
    assert state.knocked_out and state.valuation_point.date == datetime(2026, 9, 16)      # the daily tracker's stamp
    return product, kos[5], state


def test_r1_a_date_only_checkpoint_is_its_close_and_never_moves_to_fit_the_request(sse_calendar, sse_sessions, desk):
    product, sixth, state = _ko_checkpoint(sse_calendar, sse_sessions, desk)
    # an hour before the close that checkpoint reports: its KO receivable is the future, not history
    with pytest.raises(ValidationError, match="cannot report the future"):
        value_intraday(QUAD, _req(product, sse_sessions, desk, sixth.timestamp - timedelta(hours=1),
                                  lifecycle_state=state))
    # after the close the same checkpoint is history, and the KO it carries stays confirmed
    res = value_intraday(QUAD, _req(product, sse_sessions, desk, sixth.timestamp + timedelta(seconds=1),
                                    lifecycle_state=state, event_phase="after"))
    assert res.lifecycle["knocked_out"] and not res.provisional
    assert [c.provenance for c in res.cashflows if c.kind == "pending_receivable"] == ["confirmed"]


def test_r1_an_explicit_checkpoint_carrying_a_later_outcome_is_rejected(sse_calendar, sse_sessions, desk):
    product, sixth, state = _ko_checkpoint(sse_calendar, sse_sessions, desk)
    # restamped 10:00 the timestamp no longer reports the future -- its contents still do
    morning = replace(state, valuation_point=ValuationPoint(date=sixth.timestamp.replace(hour=10, tzinfo=None)))
    with pytest.raises(ValidationError, match="carries outcomes determined after it"):
        value_intraday(QUAD, _req(product, sse_sessions, desk, sixth.timestamp - timedelta(hours=1),
                                  lifecycle_state=morning))


def test_r1_a_checkpoint_at_the_instant_cannot_have_decided_an_event_still_open(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    kos = _events(product, sse_sessions, EventKind.KO)
    at_close = AutocallableLifecycleState(valuation_point=ValuationPoint(date=kos[5].timestamp.replace(tzinfo=None)))
    with pytest.raises(ValidationError, match="valuation is BEFORE that event"):
        value_intraday(QUAD, _req(product, sse_sessions, desk, kos[5].timestamp, lifecycle_state=at_close))
    assert value_intraday(QUAD, _req(product, sse_sessions, desk, kos[5].timestamp, lifecycle_state=at_close,
                                     event_phase="after")).lifecycle["alive"]


def test_r1_a_same_day_checkpoint_no_longer_suppresses_the_continuous_history(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    product = _uo_continuous(expiry=datetime(2026, 9, 18))
    env = flat_env(ts, spot=104.0, r=0.0, q=0.0)
    dated = BarrierLifecycleState(valuation_point=ValuationPoint(date=datetime(2026, 9, 16)))
    with pytest.raises(ValidationError, match="cannot report the future"):
        value_intraday(BARRIER, _req(product, sse_sessions, desk, ts, env=env, lifecycle_state=dated))
    # an authoritative state AS OF the instant does supersede the scenario, and says so by its stamp
    now = BarrierLifecycleState(valuation_point=ValuationPoint(date=ts.replace(tzinfo=None)))
    res = value_intraday(BARRIER, _req(product, sse_sessions, desk, ts, env=env, lifecycle_state=now))
    assert res.continuous_assumption is None and not res.provisional


# --- R2: the first carry interval starts at the valuation instant --------------------------------------------------
def _barrier_request(sse_sessions, desk, rate_curve):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    req = _req(_uo_continuous(), sse_sessions, desk, ts)
    T = resolve_context(req).numerical.maturity_tau
    return replace(req, pricing_env=replace(req.pricing_env, rate_curve=rate_curve(T)))


def test_r2_a_zero_rate_sloping_from_an_origin_pillar_is_not_a_flat_forward(sse_sessions, desk):
    req = _barrier_request(sse_sessions, desk, lambda T: LinearRateCurve([(0.0, 0.01), (T, 0.10)]))
    adm = analytical_barrier_admissibility(resolve_context(req))
    assert not adm.admissible and "forward rate is not flat" in adm.reason
    with pytest.raises(CapabilityError, match="not exact here"):
        value_intraday(BARRIER, req)


def test_r2_the_same_pillars_log_linear_are_a_flat_forward(sse_sessions, desk):
    # ln DF is linear from DF(0) = 1 whatever the origin pillar says: one forward, 10%, on the whole window
    req = _barrier_request(sse_sessions, desk, lambda T: LogLinearRateCurve([(0.0, 0.01), (T, 0.10)]))
    adm = analytical_barrier_admissibility(resolve_context(req))
    assert adm.admissible and adm.mode == "uniform_calendar_rate"


# --- R3: exact prices difference exactly ----------------------------------------------------------------------------
def test_r3_exact_prices_difference_exactly_without_any_evidence(sse_sessions, desk):
    # a desk MOVE of exact prices is exact; a point theta on the same prices is a second-order stencil
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    res = value_intraday(DIGITAL, _req(digital(datetime(2026, 9, 16)), sse_sessions, desk, ts,
                                       greeks=("delta", "gamma", "vega", "rho", "dividend_rho", "theta"),
                                       greek_convention="desk_bump"))
    assert all(g.status == "ok" for g in res.greeks), [(g.name, g.reason) for g in res.greeks]
    point = value_intraday(DIGITAL, _req(digital(datetime(2026, 9, 16)), sse_sessions, desk, ts, greeks=("theta",),
                                         greek_convention="point")).greek("theta")
    assert point.status == "ok" and point.value is not None and "second-order" in point.reason


# --- R4: every quadrature configuration returns its own numbers ------------------------------------------------------
def _quad_cell(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    kos = _events(product, sse_sessions, EventKind.KO)
    ts = kos[5].timestamp - timedelta(hours=1)
    return _req(product, sse_sessions, desk, ts, env=flat_env(ts, spot=75.0 * 1.0001),
                fixings=tuple(Fixing(e.timestamp, 100.0) for e in kos[:5]), greeks=("delta", "gamma", "vega"),
                greek_convention="point")


def test_r4_a_coarser_quadrature_returns_its_own_numbers(sse_calendar, sse_sessions, desk):
    """Accuracy is a study's business: the runtime reports what each configuration computes."""
    req = _quad_cell(sse_calendar, sse_sessions, desk)
    coarse = value_intraday(SnowballQuadEngineV2(QuadV2Params(order=4, cells_per_sd=0.1)), req)
    fine = value_intraday(QUAD, req)
    assert all(g.status == "ok" for g in coarse.greeks) and all(g.status == "ok" for g in fine.greeks)
    assert coarse.greek("delta").value != fine.greek("delta").value


def test_r4_prepared_curve_points_report_the_kernel_derivative_at_every_spot(sse_calendar, sse_sessions, desk):
    req = replace(_quad_cell(sse_calendar, sse_sessions, desk), greeks=())
    curve = spot_curve(QUAD, req, [74.0, 76.0])
    assert [p.status for p in curve] == ["ok", "ok"] and all(p.delta is not None for p in curve)
    assert all(p.statuses == {"delta": "ok", "gamma": "ok"} for p in curve)


# --- R5: a finite roll is a desk theta; a point theta is a derivative -----------------------------------------------
def test_r5_point_theta_is_the_time_derivative_not_the_declared_roll(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    req = _req(digital(datetime(2026, 9, 16), strike=100.2), sse_sessions, desk, ts, env=flat_env(ts, r=0.0, q=0.0),
               greeks=("theta",), greek_convention="point", theta_step=timedelta(seconds=1800))
    exact_per_hour = -0.06884007638606393                  # the review's closed-form derivative in this variance segment
    point = value_intraday(DIGITAL, req).greek("theta")
    assert point.convention == "point" and point.status == "ok" and "finite roll" not in point.reason
    # a second-order stencil at h = 1e-3 of the segment: truncation ~1e-6 of the derivative, inside the 1e-4 budget
    assert point.value == pytest.approx(exact_per_hour, rel=1e-5) and point.bump == 3.6
    roll = value_intraday(DIGITAL, replace(req, greek_convention="desk_bump")).greek("theta")
    assert roll.convention == "desk_bump" and roll.bump == 1800.0 and "finite roll" in roll.reason
    assert roll.value == pytest.approx(-0.1106317613, rel=1e-9)            # the 1800 s roll, 60.7% off the derivative



# --- R6: the frozen-market roll wrappers are admitted coefficient families ------------------------------------------
@pytest.mark.parametrize("hour, profile_name, carry, mode", [
    (14, "desk", (0.03, 0.01), "analytical_uniform_calendar_rate"),
    (11, "sessions_only", (0.0, 0.0), "analytical_zero_carry_time_change"),
])
def test_r6_barrier_theta_reprices_the_rolled_context(sse_sessions, desk, hour, profile_name, carry, mode):
    profile = desk if profile_name == "desk" else VarianceProfile.sessions_only(sse_sessions, 244)
    ts = datetime(2026, 9, 16, hour, tzinfo=SHANGHAI)
    req = _req(_uo_continuous(), sse_sessions, profile, ts, env=flat_env(ts, r=carry[0], q=carry[1]),
               greeks=("theta",), greek_convention="desk_bump", theta_step=timedelta(seconds=60))
    res = value_intraday(BARRIER, req)
    theta = res.greek("theta")
    assert res.method == mode and theta.status == "ok" and theta.bump == 60.0
    later = value_intraday(BARRIER, replace(req, greeks=(), pricing_env=replace(req.pricing_env,
                                                                                 valuation_date=ts + timedelta(seconds=60))))
    assert later.method == mode
    # the point convention reprices the rolled context too, and no coefficient the roll introduced refuses it
    point = value_intraday(BARRIER, replace(req, greek_convention="point")).greek("theta")
    assert point.status == "ok" and "does not declare" not in (point.reason or "")


# --- R7: assumptions reach every flow they can change ---------------------------------------------------------------
def test_r7_a_memory_coupon_after_an_assumed_fixing_is_provisional(sse_calendar, sse_sessions, desk):
    product = dated_phoenix(sse_calendar, T0)
    product.coupon_config = replace(product.coupon_config, fixed_coupon_year_fraction=1 / 12)
    for record in product.barrier_config.ko_observation_schedule.records:
        record.settlement_date = record.observation_date + timedelta(days=90)
    kos = _events(product, sse_sessions, EventKind.KO)
    ts = kos[1].timestamp + timedelta(seconds=1)
    req = _req(product, sse_sessions, desk, ts, env=flat_env(ts, spot=79.0), fixings=(Fixing(kos[1].timestamp, 100.0),))
    assumed = value_intraday(PhoenixQuadEngineV2(), req)
    coupon = next(c for c in assumed.cashflows if c.cashflow_id == "coupon:1")
    # 79 missed the first coupon, so this one pays both periods -- and an actual first fixing at or above the KO
    # barrier would have ended the contract before it: the whole receivable rests on the assumption
    assert coupon.provenance == "provisional" and any(eid.startswith("ko[0]") for eid in coupon.depends_on)
    actual = value_intraday(PhoenixQuadEngineV2(), replace(req, fixings=(Fixing(kos[0].timestamp, 100.0),
                                                                         Fixing(kos[1].timestamp, 100.0))))
    assert {c.cashflow_id: (c.provenance, c.depends_on) for c in actual.cashflows if c.cashflow_id} == {
        "coupon:0": ("confirmed", ()), "coupon:1": ("confirmed", ())}


@pytest.mark.parametrize("disable_ko_after_ki", [False, True])
def test_r7_a_knock_in_assumption_reaches_only_the_flows_a_knock_in_can_change(sse_calendar, sse_sessions, desk,
                                                                              disable_ko_after_ki):
    product = dated_snowball(sse_calendar, T0)
    extra = datetime(2026, 5, 6)                           # a KI observation between the first two KO fixings
    records = list(product.barrier_config.ki_observation_schedule.records)
    schedule = ObservationSchedule(records=sorted(records + [ObservationRecord(observation_date=extra, barrier=75.0)],
                                                  key=lambda r: r.observation_date))
    product.barrier_config = replace(product.barrier_config, ki_observation_schedule=schedule,
                                     disable_ko_after_ki=disable_ko_after_ki)
    kos = _events(product, sse_sessions, EventKind.KO)
    ki_extra = next(e for e in _events(product, sse_sessions, EventKind.KI) if e.timestamp.date() == extra.date())
    ts = kos[1].timestamp + timedelta(seconds=1)
    res = value_intraday(QUAD, _req(product, sse_sessions, desk, ts, env=flat_env(ts, spot=104.0), event_phase="after",
                                    fixings=(Fixing(kos[0].timestamp, 100.0), Fixing(kos[1].timestamp, 104.0))))
    assert res.provisional and [a.event_ids for a in res.assumptions] == [(ki_extra.event_id,)]
    ko = next(c for c in res.cashflows if c.cashflow_id == "knock-out:1")
    if disable_ko_after_ki:          # an actual knock-in would have switched this KO off
        assert ko.provenance == "provisional" and ko.depends_on == (ki_extra.event_id,)
    else:                            # a knock-in changes neither whether nor what this KO pays
        assert ko.provenance == "confirmed" and ko.depends_on == ()


def test_r7_a_provisional_flow_names_what_it_depends_on():
    with pytest.raises(ValidationError, match="names what it depends on"):
        CashflowComponent("pending_receivable", 1.0, "provisional")
    with pytest.raises(ValidationError, match="names what it depends on"):
        CashflowComponent("paid", 1.0, "confirmed", depends_on=("ko[0]",))
