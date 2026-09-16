"""Regressions for the findings of docs/superpowers/reviews/intraday-pricing-2026-09-16/REVIEW.md.

One test per finding, named by its number, so the review can be audited against the
suite. Findings whose natural home was an existing file are asserted there instead
and named here in ``COVERED_ELSEWHERE``.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, time, timedelta
from math import exp

import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import PhoenixMCEngine
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, BarrierLifecycleState, ValuationPoint
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.execution import PricingSession
from quantark.execution.errors import CapabilityError
from quantark.intraday import Fixing, VarianceProfile, resolve_context, value_intraday
from quantark.intraday.admissibility import analytical_barrier_admissibility
from quantark.intraday.batch import spot_curve
from quantark.intraday.events import EventKind, resolve_timeline
from quantark.intraday.greeks import cell_price, resolve_theta_step
from quantark.intraday.request import IntradayValuationRequest
from quantark.param.vol import TermStructureVolSurface
from quantark.util.enum.option_enums import BarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, dated_phoenix, dated_snowball, digital, flat_env

#: Findings asserted in the file that owns the behaviour rather than here.
COVERED_ELSEWHERE = {
    4: "test_greeks_point.py::test_a_resolved_pde_mesh_is_a_diagnostic_and_not_a_greek_certificate "
       "and ::test_demonstrated_quad_proxies_... (the point half; the desk half is below)",
    11: "test_provisional.py::test_terminal_fixing_becomes_a_fixed_receivable",
}

T0 = datetime(2026, 3, 16)
QUAD = SnowballQuadEngineV2()
DIGITAL = DigitalOptionAnalyticalEngine()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _req(product, sessions, profile, ts, **kw):
    env = kw.pop("env", flat_env(ts))
    return IntradayValuationRequest(product=product, pricing_env=env, session_calendar=sessions,
                                    variance_profile=profile, **kw)


def _kos(product, sessions):
    probe = flat_env(datetime(2026, 4, 1, tzinfo=SHANGHAI))
    return [e for e in resolve_timeline(product, sessions, probe).events if e.kind is EventKind.KO]


def _uo_continuous(expiry=datetime(2026, 9, 16), **kw):
    return BarrierOption(strike=100, option_type=OptionType.CALL, barrier=103, barrier_type=BarrierType.UP_OUT,
                         initial_date=T0, exercise_date=expiry, observation_type=ObservationType.CONTINUOUS, **kw)


# --- 1: PDE and MC silently discard existing Phoenix coupon memory -------------------
def test_finding_1_initial_phoenix_memory_reaches_every_engine(sse_calendar, sse_sessions, desk):
    product = dated_phoenix(sse_calendar, T0, months=3, ko=1000.0)
    product.coupon_config = replace(product.coupon_config, fixed_coupon_year_fraction=1 / 12)
    first = _kos(product, sse_sessions)[0]
    ts = first.timestamp + timedelta(days=1)
    missed = AutocallableLifecycleState(coupon_memory_count=1, missed_coupon_indices={0}, observed_coupon_indices={0},
                                        observed_ki_indices={0}, observed_ko_indices={0},
                                        valuation_point=ValuationPoint(date=first.timestamp.replace(tzinfo=None)))
    none_missed = replace(missed, coupon_memory_count=0, missed_coupon_indices=set())
    req = _req(product, sse_sessions, desk, ts, env=flat_env(ts, r=0.0, q=0.0, vol=0.01), lifecycle_state=missed)
    engines = (PhoenixQuadEngineV2(), PhoenixPDESolver(), PhoenixMCEngine(params=MCParams(num_paths=4096, seed=42)))
    # one missed coupon of 1 on a remote KO: every route must carry the same outstanding arrears
    for engine in engines:
        with_memory = value_intraday(engine, req).price
        without = value_intraday(engine, replace(req, lifecycle_state=none_missed)).price
        assert with_memory - without == pytest.approx(1.0, abs=1e-9), type(engine).__name__


# --- 2: point rho of a terminated contract discards pending settlement ---------------
def test_finding_2_terminated_point_rho_differentiates_the_pending_ledger(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    for record in product.barrier_config.ko_observation_schedule.records:
        record.settlement_date = record.observation_date + timedelta(days=5)
    events = _kos(product, sse_sessions)
    ts = events[5].timestamp + timedelta(seconds=30)
    req = _req(product, sse_sessions, desk, ts, env=flat_env(ts, spot=104.0), event_phase="after",
               fixings=tuple(Fixing(e.timestamp, 100.0) for e in events[:5]),
               greeks=("rho",), greek_convention="point")
    res = value_intraday(QUAD, req)
    ctx = resolve_context(req)
    exact = -sum(amount * tau * exp(-0.03 * tau) for _, amount, tau in ctx.numerical.pending_cashflows)
    assert res.contingent_pv == 0.0 and res.pending_receivable_pv > 0.0
    rho = res.greek("rho")
    assert rho.status == "ok" and rho.value == pytest.approx(exact, rel=1e-8) and rho.value != 0.0
    # vol and dividends do not enter a fixed ledger, so those ARE exactly zero
    both = value_intraday(QUAD, replace(req, greeks=("vega", "dividend_rho")))
    assert both.greek("vega").value == 0.0 and both.greek("dividend_rho").value == 0.0


# --- 3: analytical barrier admissibility misses term-curve breakpoints ---------------
def test_finding_3_a_volatility_pillar_inside_the_session_breaks_admissibility(sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    req = _req(_uo_continuous(), sse_sessions, only, ts)
    u = resolve_context(req).time_map.to_trading(resolve_context(req).numerical.maturity_tau)
    term = TermStructureVolSurface(times=[u / 2, u], vols=[0.1, 0.3])
    ctx = resolve_context(replace(req, pricing_env=replace(req.pricing_env, vol_surface=term)))
    # the two halves of one session accrue variance 17x apart: no constant-coefficient first passage
    adm = analytical_barrier_admissibility(ctx)
    assert not adm.admissible and "variance rate differs across the coefficient intervals" in adm.reason
    with pytest.raises(CapabilityError, match="not exact here"):
        value_intraday(BarrierAnalyticalEngine(), replace(req, pricing_env=replace(req.pricing_env, vol_surface=term)))
    # a flat surface on the same clock is still admissible
    assert analytical_barrier_admissibility(resolve_context(req)).admissible


def test_finding_3_an_unqualified_curve_family_is_refused_rather_than_sampled(sse_sessions, desk):
    from quantark.param.rrf.rate_curve import CubicSplineRateCurve
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    spline = CubicSplineRateCurve([(0.01, 0.03), (0.5, 0.03), (1.0, 0.03), (2.0, 0.03)])
    req = _req(_uo_continuous(), sse_sessions, desk, ts,
               env=replace(flat_env(ts), rate_curve=spline))
    adm = analytical_barrier_admissibility(resolve_context(req))
    assert not adm.admissible and "CubicSplineRateCurve does not declare" in adm.reason


# --- 4: desk Greeks inherit the resolution of the prices they difference --------------
def test_finding_4_a_desk_greek_is_unqualified_on_an_unresolved_grid(sse_calendar, sse_sessions, desk):
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    product = dated_snowball(sse_calendar, T0)
    events = _kos(product, sse_sessions)
    fixings = tuple(Fixing(e.timestamp, 100.0) for e in events[:5])
    engine = SnowballPDESolver(PDEParams())

    def value(ts, spot):
        return value_intraday(engine, _req(product, sse_sessions, desk, ts, env=flat_env(ts, spot=spot),
                                           fixings=fixings, greeks=("delta", "gamma"), greek_convention="desk_bump"))

    # a day out the mesh resolves the layer -- which is necessary, not sufficient (re-review R3): no certificate
    far = value(events[5].timestamp - timedelta(days=1), 100.0)
    assert far.numerical["resolution"] == "resolved"
    assert all(g.status == "unqualified" and "discretisation or sampling error" in g.reason for g in far.greeks)
    # a second out, one bp above the barrier, it does not: the difference inherits that
    near = value(events[5].timestamp - timedelta(seconds=1), 102.99)
    assert near.numerical["resolution"] == "unqualified"
    for g in near.greeks:
        assert g.status == "unqualified" and g.value is None and "could not resolve" in g.reason


# --- 5: local theta crosses a variance-clock boundary --------------------------------
def test_finding_5_local_theta_stops_at_the_clock_boundary(sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ts = datetime(2026, 9, 16, 12, 59, 59, tzinfo=SHANGHAI)
    req = _req(digital(datetime(2026, 9, 16), strike=100.2), sse_sessions, only, ts,
               env=flat_env(ts, r=0.0, q=0.0), greeks=("theta",), greek_convention="point",
               theta_step=timedelta(seconds=30))
    step = resolve_theta_step(resolve_context(req), req.theta_step)
    assert step.actual == timedelta(seconds=1) and step.adjusted and step.side == "forward_clamped_to_clock"
    # a step longer than the whole clock horizon still lands on the boundary, never off the map
    far = resolve_theta_step(resolve_context(req), timedelta(days=3650))
    assert far.actual == timedelta(seconds=1) and far.side == "forward_clamped_to_clock"
    roll = value_intraday(DIGITAL, replace(req, greek_convention="desk_bump")).greek("theta")
    # frozen-market value is constant through the zero-weight break: the finite roll is exactly zero
    assert roll.status == "ok" and roll.value == 0.0 and "finite roll" in roll.reason and roll.bump == 1.0
    half = value_intraday(DIGITAL, replace(req, greek_convention="desk_bump",
                                           theta_step=timedelta(milliseconds=500))).greek("theta")
    assert half.value == 0.0
    # the derivative there is zero too; its stencil (h = 1 ms) sees only price round-off (re-review R5)
    point = value_intraday(DIGITAL, req).greek("theta")
    assert point.status == "ok" and point.value == pytest.approx(0.0, abs=1e-9) and point.bump == 0.001


# --- 6: different holiday calendars collide in valuation identity --------------------
def test_finding_6_a_holiday_change_changes_the_clock_identity(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    req = _req(digital(datetime(2026, 9, 18), strike=101.0), sse_sessions, desk, ts)
    changed = deepcopy(sse_sessions.calendar)
    changed.holidays.add(datetime(2026, 9, 17))
    other = replace(req, session_calendar=replace(sse_sessions, calendar=changed))
    a, b = value_intraday(DIGITAL, req), value_intraday(DIGITAL, other)
    assert a.price != b.price
    assert a.session_identity != b.session_identity and a.context_identity != b.context_identity


def test_finding_6_a_weekend_rule_change_changes_the_clock_identity(sse_sessions):
    changed = deepcopy(sse_sessions.calendar)
    changed.weekend_days = {6}
    assert replace(sse_sessions, calendar=changed).identity() != sse_sessions.identity()


# --- 7: checkpoint dates can suppress fixings they do not cover ----------------------
def test_finding_7_an_intraday_checkpoint_does_not_cover_a_later_fixing(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    sixth = _kos(product, sse_sessions)[5]
    ts = sixth.timestamp + timedelta(seconds=1)
    morning = AutocallableLifecycleState(valuation_point=ValuationPoint(date=sixth.timestamp.replace(hour=10, tzinfo=None)))
    res = value_intraday(QUAD, _req(product, sse_sessions, desk, ts, lifecycle_state=morning,
                                    env=flat_env(ts, spot=104.0), event_phase="after"))
    # 104 is above the 103 KO: the 15:00 fixing a 10:00 checkpoint never saw must still be assumed
    assert res.provisional and res.lifecycle["knocked_out"] and res.assumptions


def test_finding_7_a_date_only_checkpoint_still_covers_that_days_close(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    sixth = _kos(product, sse_sessions)[5]
    ts = sixth.timestamp + timedelta(seconds=1)
    dated = AutocallableLifecycleState(valuation_point=ValuationPoint(date=sixth.timestamp.replace(hour=0, tzinfo=None)))
    res = value_intraday(QUAD, _req(product, sse_sessions, desk, ts, lifecycle_state=dated,
                                    env=flat_env(ts, spot=104.0), event_phase="after"))
    assert not res.provisional and res.lifecycle["alive"] and not res.assumptions


def test_finding_7_a_checkpoint_from_the_future_is_rejected(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    sixth = _kos(product, sse_sessions)[5]
    ts = sixth.timestamp + timedelta(seconds=1)
    for point in (ValuationPoint(date=(sixth.timestamp + timedelta(days=1)).replace(tzinfo=None)),
                  ValuationPoint(date=(sixth.timestamp + timedelta(days=1)).replace(hour=0, tzinfo=None))):
        state = AutocallableLifecycleState(valuation_point=point)
        with pytest.raises(ValidationError, match="cannot report the future"):
            value_intraday(QUAD, _req(product, sse_sessions, desk, ts, lifecycle_state=state,
                                      env=flat_env(ts, spot=104.0), event_phase="after"))


# --- 8: the resolved context shares mutable caller-owned market objects --------------
def test_finding_8_a_resolved_context_owns_its_market(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    req = _req(digital(datetime(2026, 9, 16)), sse_sessions, desk, ts)
    ctx = resolve_context(req)
    before, identity = cell_price(ctx, DIGITAL), ctx.identity
    req.pricing_env.spot_quote.spot = 102.0
    req.pricing_env.rate_curve.rate = 0.25
    assert cell_price(ctx, DIGITAL) == before and ctx.identity == identity
    for name in ("spot_quote", "rate_curve", "div_yield"):
        assert getattr(ctx.pricing_env, name) is not getattr(req.pricing_env, name)
    assert ctx.pricing_env.vol_surface.inner is not req.pricing_env.vol_surface
    assert ctx.request.session_calendar.calendar is not req.session_calendar.calendar


# --- 9: a future digital fixing silently treated as already terminal -----------------
def test_finding_9_the_daily_expiry_tolerance_is_not_an_intraday_limit(sse_sessions, desk):
    expiry = datetime(2026, 9, 16, 15, tzinfo=SHANGHAI)
    ts = expiry - timedelta(milliseconds=1)
    req = _req(digital(expiry.replace(hour=0, tzinfo=None)), sse_sessions, desk, ts,
               greeks=("delta",), greek_convention="point")
    ctx = resolve_context(req)
    assert ctx.pricing_env.vol_surface.total_variance(100.0, ctx.numerical.maturity_tau, 100.0) > 0.0
    with pytest.raises(CapabilityError, match="MIN_MATURITY"):
        value_intraday(DIGITAL, req)
    # at the expiry instant itself the variance IS zero and the exact limit is published
    at_expiry = value_intraday(DIGITAL, _req(digital(expiry.replace(hour=0, tzinfo=None)), sse_sessions, desk, expiry,
                                             env=flat_env(expiry, spot=100.3)))
    assert at_expiry.method == "deterministic_zero_variance"


# --- 10: correctly paid Phoenix memory arrears fail the ledger cash check ------------
def test_finding_10_a_memory_coupon_reconciles_against_its_constituent_periods(sse_calendar, sse_sessions, desk):
    product = dated_phoenix(sse_calendar, T0)
    product.coupon_config = replace(product.coupon_config, fixed_coupon_year_fraction=1 / 12)
    events = _kos(product, sse_sessions)
    ts = events[1].timestamp + timedelta(seconds=1)
    res = value_intraday(PhoenixQuadEngineV2(), _req(product, sse_sessions, desk, ts,
                                                     fixings=(Fixing(events[0].timestamp, 79.0),
                                                              Fixing(events[1].timestamp, 100.0))))
    paid = [c for c in res.cashflows if c.kind == "paid"]
    # period 0 missed below its 80 barrier, period 1 triggered: one entry settles both
    assert len(paid) == 1 and paid[0].cashflow_id == "coupon:1"
    assert paid[0].amount_pv == pytest.approx(2.0) and res.lifecycle["coupon_memory_count"] == 0


# --- 12: point vega crashes on a zero-weight interval --------------------------------
def test_finding_12_a_zero_clock_has_exactly_zero_vega(sse_sessions):
    only = VarianceProfile.sessions_only(sse_sessions, 244)
    ts = datetime(2026, 9, 16, 12, tzinfo=SHANGHAI)
    req = _req(digital(datetime(2026, 9, 16)), sse_sessions, only, ts,
               fixing_time_of_day=time(12, 30), greeks=("vega",), greek_convention="point")
    vega = value_intraday(DIGITAL, req).greek("vega")
    assert vega.status == "ok" and vega.value == 0.0 and "zero-weight window" in vega.reason
    desk_vega = value_intraday(DIGITAL, replace(req, greek_convention="desk_bump")).greek("vega")
    assert desk_vega.status == "ok" and desk_vega.value == 0.0


# --- 13: the public session path re-dispatches a knocked-in vanilla -----------------
def test_finding_13_a_knocked_in_barrier_dispatches_the_engine_that_priced_it(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    product = _uo_continuous()
    product.barrier_type = BarrierType.UP_IN
    state = BarrierLifecycleState(knocked_in=True, valuation_point=ValuationPoint(date=datetime(2026, 9, 15)))
    req = _req(product, sse_sessions, desk, ts, lifecycle_state=state)
    direct = value_intraday(BarrierAnalyticalEngine(), req)
    with PricingSession() as session:
        through_session = value_intraday(BarrierAnalyticalEngine(), req, session=session)
    assert through_session.price == direct.price
    assert any(r.startswith("manifest:") for r in through_session.records)


# --- 14: published continuous-QUAD support is not wired to the intraday clock --------
def test_finding_14_continuous_quad_is_refused_with_a_typed_error(sse_calendar, sse_sessions, desk):
    product = dated_snowball(sse_calendar, T0)
    product.barrier_config = replace(product.barrier_config, ki_continuous=True,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_observation_schedule=None)
    ts = datetime(2026, 4, 1, 14, tzinfo=SHANGHAI)
    with pytest.raises(CapabilityError, match="interval survival/crossing operator"):
        value_intraday(QUAD, _req(product, sse_sessions, desk, ts))


# --- 15: spot curves drop continuous-history provenance and numerical status ---------
def test_finding_15_a_curve_carries_its_contexts_provenance(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    req = _req(_uo_continuous(), sse_sessions, desk, ts, env=flat_env(ts, r=0.0, q=0.0))
    curve = spot_curve(BarrierAnalyticalEngine(), req, [99.0, 100.0])
    single = value_intraday(BarrierAnalyticalEngine(), req)
    assert curve.provisional is True and curve.continuous_assumption is not None
    assert curve.context_identity == single.context_identity and curve.engine == single.engine
    assert curve.session_identity == single.session_identity and curve.profile_identity == single.profile_identity
    assert [p.spot for p in curve] == [99.0, 100.0] and len(curve) == 2
    assert all(p.numerical and p.method for p in curve)       # per-spot price evidence, not a bare float
    assert curve.to_dict()["provisional"] is True


# --- 16: an already assumed continuous KO is not nondifferentiable -------------------
def test_finding_16_an_assumed_hit_leaves_conditional_greeks_defined(sse_sessions, desk):
    ts = datetime(2026, 9, 16, 14, tzinfo=SHANGHAI)
    product = _uo_continuous(expiry=datetime(2026, 9, 18), rebate=1.0, pay_at_hit=False)
    res = value_intraday(BarrierAnalyticalEngine(),
                         _req(product, sse_sessions, desk, ts, env=flat_env(ts, spot=103.0),
                              greeks=("delta", "gamma"), greek_convention="point"))
    assert res.provisional and res.lifecycle["knocked_out"] and res.pending_receivable_pv > 0.0
    # the assumption is frozen across bumps, so the surviving claim is fixed cash: flat in spot
    assert res.greek("delta").value == 0.0 and res.greek("gamma").value == 0.0
