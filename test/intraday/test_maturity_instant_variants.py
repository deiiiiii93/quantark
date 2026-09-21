"""The maturity close under BEFORE for the Phoenix and the knock-out-reset snowball (2026-09-19, second pass).

``test_maturity_instant.py`` pins the plain snowball. The same defect sat in the other two autocallables, with two
more behind it:

* Phoenix: the zero-maturity shortcut of its Monte Carlo, PDE and QUAD V1 engines returned the redemption of the
  carried knock-in state, dropping the last coupon, the memory arrears it releases, the knock-out and the knock-in.
  The runtime could not resolve the instant either: the lifecycle tracker called ``get_payoff`` with the snowball
  signature and raised ``TypeError`` for every Phoenix alive at maturity. And the PDE solver dropped outstanding
  arrears from a knock-out decided at ANY valuation instant, not only the last.
* Knock-out-reset snowball: the shortcut ignored the pending knock-out of the schedule in force (pre-KI before a
  knock-in, post-KI after one), and QUAD V1 also the carried knock-in. The runtime has no replay for this product, so
  its routes value the decided claim from the twin's own decision.

Every fixture separates the outcomes on purpose (a rebate unlike the knock-out cash): with equal cash a missed
knock-out is invisible.
"""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, ValuationPoint
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.intraday import value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind, EventPhase, resolve_timeline
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode
from intraday.conftest import SHANGHAI, dated_ko_reset, dated_phoenix, flat_env

T0 = datetime(2026, 3, 16)
PROBE = datetime(2026, 9, 15, tzinfo=SHANGHAI)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


# =====================================================================================================================
# Phoenix: KO 103, coupon barrier 80, KI 75, principal 100, a 2% rebate so a survival pays more than a knock-out
# =====================================================================================================================
PHX_REBATE = 0.02
#: spot -> (the cash apart from the coupon, whether the period's coupon -- and the arrears with it -- is paid)
#:   104 knocks out: the principal            90 survives: principal + rebate, coupon paid (90 >= 80)
#:    78 survives below the coupon barrier    70 knocks in: 100 - (100 - 70), no rebate, no coupon
PHX_DECIDED = [(104.0, 100.0, True), (90.0, 102.0, True), (78.0, 102.0, False), (70.0, 70.0, False)]
PHX_ROUTED = {
    "mc": lambda: PhoenixMCEngine(params=MCParams(seed=7, num_paths=2048)),
    "pde": lambda: PhoenixPDESolver(PDEParams()),
    "quad_v2": lambda: PhoenixQuadEngineV2(),
}
PHX_DAY_LEVEL = {"mc": PHX_ROUTED["mc"], "pde": PHX_ROUTED["pde"],
                 "quad_v1": lambda: PhoenixQuadEngine(params=QuadParams(grid_points=1001))}
#: no arrears: every earlier coupon was paid, and the last period is 28 days at 12% ACT/365;
#: arrears: every earlier fixing sat at 78 (no coupon, no knock-in), eleven equal monthly coupons of 1.0 are owed
PHX_CASES = {"paid_up": dict(earlier=100.0, coupon=100.0 * 0.12 * 28.0 / 365.0, arrears=0.0, kw={}),
             "arrears": dict(earlier=78.0, coupon=1.0, arrears=11.0, kw=dict(fixed_fraction=1.0 / 12.0))}


def _phx_ctx(cal, sessions, profile, ts, spot=100.0, phase=EventPhase.BEFORE, fixings=(), **kw):
    return resolve_context(IntradayValuationRequest(
        product=dated_phoenix(cal, T0, rebate_rate=PHX_REBATE, **kw), pricing_env=flat_env(ts, spot=spot),
        session_calendar=sessions, variance_profile=profile, event_phase=phase, fixings=tuple(fixings)))


def _phx_at(cal, sessions, profile, spot, case, index=-1, before=timedelta(0)):
    """(context at KO[index] less ``before``, that KO event, the earlier fixings) for one of ``PHX_CASES``."""
    kos = [e for e in _phx_ctx(cal, sessions, profile, PROBE, **case["kw"]).timeline.events if e.kind is EventKind.KO]
    target = kos[index]
    fixings = [Fixing(k.timestamp, case["earlier"]) for k in kos if k.timestamp < target.timestamp]
    return _phx_ctx(cal, sessions, profile, target.timestamp - before, spot=spot, fixings=fixings, **case["kw"]), target, fixings


def _phx_expected(case, base, paid):
    return base + (case["coupon"] + case["arrears"] if paid else 0.0)


@pytest.mark.parametrize("name", sorted(PHX_ROUTED))
@pytest.mark.parametrize("case", sorted(PHX_CASES))
@pytest.mark.parametrize("spot, base, paid", PHX_DECIDED)
def test_every_phoenix_route_decides_the_maturity_close_on_the_known_spot(sse_calendar, sse_sessions, desk, name, case,
                                                                          spot, base, paid):
    case = PHX_CASES[case]
    ctx, last, fixings = _phx_at(sse_calendar, sse_sessions, desk, spot, case)
    assert ctx.numerical.maturity_tau == 0.0 and not ctx.numerical.terminated
    engine = PHX_ROUTED[name]()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.contingent_pv == pytest.approx(_phx_expected(case, base, paid), abs=1e-9)
    assert out.method == "decided_at_valuation" and out.exact
    # decided now = what the lifecycle pays when the instant is fixed at the spot (it raised TypeError for a Phoenix)
    after = resolve_context(replace(ctx.request, event_phase=EventPhase.AFTER,
                                    fixings=tuple(fixings) + (Fixing(last.timestamp, spot),)))
    assert after.numerical.terminated
    assert out.contingent_pv == pytest.approx(after.numerical.paid_cash - ctx.numerical.paid_cash, abs=1e-9)


@pytest.mark.parametrize("name", sorted(PHX_DAY_LEVEL))
@pytest.mark.parametrize("case", sorted(PHX_CASES))
@pytest.mark.parametrize("spot, base, paid", PHX_DECIDED)
def test_a_day_level_phoenix_engine_handed_the_zero_maturity_twin_decides_the_pending_observations(
        sse_calendar, sse_sessions, desk, name, case, spot, base, paid):
    """The engine's own zero-maturity shortcut, no route in between; the twin carries the arrears."""
    case = PHX_CASES[case]
    ctx, _, _ = _phx_at(sse_calendar, sse_sessions, desk, spot, case)
    assert ctx.numerical.product.coupon_config.initial_coupon_arrears == pytest.approx(case["arrears"])
    value = float(PHX_DAY_LEVEL[name]().price(ctx.numerical.product, ctx.pricing_env))
    assert value == pytest.approx(_phx_expected(case, base, paid), abs=1e-9)


@pytest.mark.parametrize("name", sorted(PHX_DAY_LEVEL))
def test_a_phoenix_knocked_in_earlier_redeems_at_a_loss_and_still_collects_its_last_coupon(sse_calendar, sse_sessions,
                                                                                           desk, name):
    probe = _phx_ctx(sse_calendar, sse_sessions, desk, PROBE)
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    fixings = [Fixing(k.timestamp, 70.0 if i == 3 else 100.0) for i, k in enumerate(kos[:-1])]      # knocked in in July
    ctx = _phx_ctx(sse_calendar, sse_sessions, desk, kos[-1].timestamp, spot=90.0, fixings=fixings)
    assert ctx.numerical.knocked_in and ctx.numerical.maturity_tau == 0.0
    value = float(PHX_DAY_LEVEL[name]().price(ctx.numerical.product, ctx.pricing_env))
    assert value == pytest.approx(90.0 + PHX_CASES["paid_up"]["coupon"], abs=1e-9)                 # 100 - 10, plus the coupon


@pytest.mark.parametrize("name", ["mc", "pde"])
@pytest.mark.parametrize("spot, delta", [(70.0, 1.0), (90.0, 0.0)])
def test_phoenix_greeks_at_the_maturity_close_are_those_of_the_decided_payoff(sse_calendar, sse_sessions, desk, name,
                                                                              spot, delta):
    ctx, _, _ = _phx_at(sse_calendar, sse_sessions, desk, spot, PHX_CASES["paid_up"])
    for convention in ("point", "desk_bump"):
        res = value_intraday(PHX_ROUTED[name](), replace(ctx.request, greeks=("delta", "gamma"), greek_convention=convention))
        assert res.greek("delta").status == "ok" and res.greek("delta").value == pytest.approx(delta, abs=1e-9)
        assert res.greek("gamma").status == "ok" and res.greek("gamma").value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", sorted(PHX_ROUTED))
def test_a_knock_out_decided_at_any_valuation_instant_releases_the_memory_arrears(sse_calendar, sse_sessions, desk, name):
    """Not the maturity close: the sixth fixing, five coupons owed, spot above the knock-out level. The claim pays the
    principal, this period's coupon and the five in arrears: 106. The PDE solver's immediate knock-out paid 101."""
    case = PHX_CASES["arrears"]
    ctx, target, fixings = _phx_at(sse_calendar, sse_sessions, desk, 104.0, case, index=5)
    assert ctx.numerical.maturity_tau > 0.0 and ctx.numerical.lifecycle_state.coupon_memory_count == 5
    engine = PHX_ROUTED[name]()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.contingent_pv == pytest.approx(106.0, abs=1e-9)
    after = resolve_context(replace(ctx.request, event_phase=EventPhase.AFTER,
                                    fixings=tuple(fixings) + (Fixing(target.timestamp, 104.0),)))
    assert out.contingent_pv == pytest.approx(after.numerical.paid_cash - ctx.numerical.paid_cash, abs=1e-9)


def test_a_phoenix_desk_theta_rolling_onto_the_maturity_close_agrees_across_routes(sse_calendar, sse_sessions, desk):
    """One hour before the last close at 90: the roll lands on the instant, where the last coupon is decided."""
    ctx, _, _ = _phx_at(sse_calendar, sse_sessions, desk, 90.0, PHX_CASES["paid_up"], before=timedelta(hours=1))
    thetas = {}
    for name in ("pde", "quad_v2"):
        theta = value_intraday(PHX_ROUTED[name](), replace(ctx.request, greeks=("theta",), greek_convention="desk_bump")).greek("theta")
        assert theta.status == "ok"
        thetas[name] = theta.value
    assert thetas["pde"] == pytest.approx(thetas["quad_v2"], abs=5e-2), thetas


# ---------------------------------------------------------------------------------------------------------------------
# the Phoenix decision itself, on a float-time contract
def _float_phoenix(arrears=0.0, disable_ko_after_ki=False):
    schedule = lambda level: ObservationSchedule(records=[ObservationRecord(observation_time=0.0, barrier=level)])   # noqa: E731
    product = PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=ObservationSchedule(
                                         records=[ObservationRecord(observation_time=1.0, barrier=103.0)]),
                                     ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=ObservationSchedule(
                                         records=[ObservationRecord(observation_time=1.0, barrier=75.0)]),
                                     disable_ko_after_ki=disable_ko_after_ki),
        coupon_config=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.01, memory_coupon=True,
                                          initial_coupon_arrears=arrears),
        accrual_config=AccrualConfig(is_annualized_coupon=False),
        payoff_config=PayoffConfig(include_principal=True, rebate_rate=PHX_REBATE))
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=schedule(103.0),
                                     ki_observation_schedule=schedule(75.0))
    product.maturity = 0.0
    return product


def test_the_phoenix_decision_orders_knock_out_knock_in_and_coupon_as_the_engines_do():
    env = flat_env(datetime(2027, 3, 16))
    product = _float_phoenix(arrears=3.0)
    out = product.decide_observations_at_valuation(104.0, env)
    assert out.knocked_out and out.ko_index == 0 and out.coupon == pytest.approx(1.0 + 3.0) and not out.knocked_in
    alive = product.decide_observations_at_valuation(90.0, env)
    assert not alive.knocked_out and not alive.knocked_in and alive.coupon == pytest.approx(4.0) and alive.coupon_index == 0
    missed = product.decide_observations_at_valuation(78.0, env)
    assert not missed.knocked_out and not missed.knocked_in and missed.coupon == 0.0
    hit = product.decide_observations_at_valuation(70.0, env)
    assert hit.knocked_in and not hit.knocked_out and hit.coupon == 0.0
    carried = product.decide_observations_at_valuation(104.0, env, knocked_in=True)
    assert carried.knocked_out and carried.knocked_in                     # a knocked-in Phoenix still knocks out ...
    blocked = _float_phoenix(disable_ko_after_ki=True).decide_observations_at_valuation(104.0, env, knocked_in=True)
    assert not blocked.knocked_out and blocked.coupon == pytest.approx(1.0)   # ... unless the contract disables it


# =====================================================================================================================
# Knock-out-reset snowball: pre-KI KO 103 paying 15, post-KI KO 95 paying 3, KI 75, a 5% rebate
# =====================================================================================================================
KR = dict(pre_months=12, post_months=12, rebate_rate=0.05)       # one final instant serves both schedules
#: never knocked in: 104 knocks out on the pre-KI schedule (15); 96 and 90 survive (the 5 rebate: the post-KI level
#: of 95 does not apply before a knock-in); 70 knocks in and redeems at the loss
KR_ALIVE = [(104.0, 15.0), (96.0, 5.0), (90.0, 5.0), (70.0, -30.0)]
#: knocked in on an earlier day: the post-KI schedule is in force, so 104 and 96 knock out at 3; 80 redeems at -20
KR_KNOCKED = [(104.0, 3.0), (96.0, 3.0), (80.0, -20.0)]
KR_STATES = [(False, spot, cash) for spot, cash in KR_ALIVE] + [(True, spot, cash) for spot, cash in KR_KNOCKED]
KR_ROUTED = {"pde": lambda: KOResetSnowballPDESolver(PDEParams()), "quad_v2": lambda: KOResetSnowballQuadEngineV2()}
KR_DAY_LEVEL = {"mc": lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=2048)), "pde": KR_ROUTED["pde"],
                "quad_v1": lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))}


def _kr_ctx(cal, sessions, profile, spot, knocked_in, before=timedelta(0), **kw):
    product = dated_ko_reset(cal, T0, **{**KR, **kw})
    last = resolve_timeline(product, sessions, flat_env(PROBE)).terminal().timestamp
    checkpoint = AutocallableLifecycleState(
        knocked_in=knocked_in, valuation_point=ValuationPoint(date=(last - timedelta(days=1)).replace(tzinfo=None)))
    return resolve_context(IntradayValuationRequest(
        product=product, pricing_env=flat_env(last - before, spot=spot), session_calendar=sessions,
        variance_profile=profile, event_phase=EventPhase.BEFORE, lifecycle_state=checkpoint))


@pytest.mark.parametrize("name", sorted(KR_ROUTED))
@pytest.mark.parametrize("knocked_in, spot, expected", KR_STATES)
def test_every_ko_reset_route_decides_the_maturity_close_on_the_known_spot(sse_calendar, sse_sessions, desk, name,
                                                                           knocked_in, spot, expected):
    ctx = _kr_ctx(sse_calendar, sse_sessions, desk, spot, knocked_in)
    assert ctx.numerical.maturity_tau == 0.0 and not ctx.numerical.terminated
    engine = KR_ROUTED[name]()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.contingent_pv == pytest.approx(expected, abs=1e-9)
    assert out.method == "decided_at_valuation" and out.exact


@pytest.mark.parametrize("name", sorted(KR_ROUTED))
@pytest.mark.parametrize("spot, expected", KR_KNOCKED)
def test_the_usual_contract_reaches_its_final_close_knocked_in(sse_calendar, sse_sessions, desk, name, spot, expected):
    """A six-month first schedule and a twelve-month second: only a knocked-in contract is alive at the final close
    (one not knocked in matured in September), and it is decided on the second schedule. The twin could not be built
    here before the contract's rules were applied to it."""
    ctx = _kr_ctx(sse_calendar, sse_sessions, desk, spot, True, pre_months=6)
    assert ctx.numerical.maturity_tau == 0.0 and not ctx.numerical.product.barrier_config.ko_observation_schedule.records
    engine = KR_ROUTED[name]()
    out = route_for(ctx, engine).price(ctx, engine)
    assert out.contingent_pv == pytest.approx(expected, abs=1e-9) and out.method == "decided_at_valuation"


@pytest.mark.parametrize("name", sorted(KR_DAY_LEVEL))
@pytest.mark.parametrize("knocked_in, spot, expected", KR_STATES)
def test_a_day_level_ko_reset_engine_handed_the_zero_maturity_twin_decides_the_pending_observations(
        sse_calendar, sse_sessions, desk, name, knocked_in, spot, expected):
    ctx = _kr_ctx(sse_calendar, sse_sessions, desk, spot, knocked_in)
    value = float(KR_DAY_LEVEL[name]().price(ctx.numerical.product, ctx.pricing_env))
    assert value == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", sorted(KR_ROUTED))
@pytest.mark.parametrize("spot, delta", [(70.0, 1.0), (90.0, 0.0)])
def test_ko_reset_greeks_at_the_maturity_close_are_those_of_the_decided_payoff(sse_calendar, sse_sessions, desk, name,
                                                                               spot, delta):
    ctx = _kr_ctx(sse_calendar, sse_sessions, desk, spot, False)
    for convention in ("point", "desk_bump"):
        res = value_intraday(KR_ROUTED[name](), replace(ctx.request, greeks=("delta", "gamma"), greek_convention=convention))
        assert res.greek("delta").status == "ok" and res.greek("delta").value == pytest.approx(delta, abs=1e-9)
        assert res.greek("gamma").status == "ok" and res.greek("gamma").value == pytest.approx(0.0, abs=1e-6)


def test_a_ko_reset_desk_theta_rolling_onto_a_knock_out_at_the_maturity_close_agrees_across_routes(sse_calendar,
                                                                                                  sse_sessions, desk):
    ctx = _kr_ctx(sse_calendar, sse_sessions, desk, 104.0, False, before=timedelta(hours=1))
    thetas = {}
    for name in ("pde", "quad_v2"):
        theta = value_intraday(KR_ROUTED[name](), replace(ctx.request, greeks=("theta",), greek_convention="desk_bump")).greek("theta")
        assert theta.status == "ok"
        thetas[name] = theta.value
    assert thetas["pde"] == pytest.approx(thetas["quad_v2"], abs=5e-2), thetas


# ---------------------------------------------------------------------------------------------------------------------
# the knock-out-reset decision itself, on a float-time contract
def _float_ko_reset(pre_times, post_times, ki_times, post_ko=95.0, mode=PostKOScheduleMode.ABSOLUTE):
    """Non-annualized cash: a pre-KI knock-out pays 15, a post-KI one 3, the rebate 5. A record whose time is
    negative is history (the aged contract drops it), so an empty ``pre_times`` is a contract past its pre-KI schedule."""
    schedule = lambda times, level: ObservationSchedule(                                                   # noqa: E731
        records=[ObservationRecord(observation_time=t, barrier=level) for t in times])
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule([1.0], 103.0),
                                     ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=schedule([1.0], 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=post_ko, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=schedule([1.0], post_ko)),
        accrual_config=AccrualConfig(is_annualized=False),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=schedule(pre_times, 103.0),
                                     ki_observation_schedule=schedule(ki_times, 75.0))
    product.post_barrier_config = replace(product.post_barrier_config, ko_observation_schedule=schedule(post_times, post_ko))
    product.maturity = 0.0
    return product


def test_the_ko_reset_decision_uses_the_schedule_the_knock_in_state_puts_in_force():
    env = flat_env(datetime(2027, 3, 16))
    both = _float_ko_reset([0.0], [0.0], [0.0])
    pre = both.decide_observations_at_valuation(104.0, env)
    assert pre.knocked_out and pre.ko_regime == "pre" and pre.ko_record.payoff == pytest.approx(15.0)
    assert not both.decide_observations_at_valuation(96.0, env).knocked_out          # 95 is not in force before a knock-in
    post = both.decide_observations_at_valuation(96.0, env, knocked_in=True)
    assert post.knocked_out and post.ko_regime == "post" and post.ko_record.payoff == pytest.approx(3.0)
    assert both.decide_observations_at_valuation(104.0, env, knocked_in=True).ko_regime == "post"   # never the pre-KI 15
    hit = both.decide_observations_at_valuation(70.0, env)
    assert hit.knocked_in and not hit.knocked_out
    # past its pre-KI schedule (that schedule ended earlier): only a knocked-in contract is still alive
    late = _float_ko_reset([-0.5], [0.0], [0.0])
    assert late.decide_observations_at_valuation(96.0, env, knocked_in=True).ko_regime == "post"
    assert not late.decide_observations_at_valuation(80.0, env, knocked_in=True).knocked_out
    # ... and a knock-in is never tested there: a contract not knocked in by the end of the first schedule matured
    assert late.first_schedule_ended(env) and not both.first_schedule_ended(env)
    assert not late.decide_observations_at_valuation(70.0, env).knocked_in


def test_a_post_ki_observation_at_the_knock_in_instant_is_in_force():
    """Only reachable when the post-KI level is at or below the knock-in level. The second schedule is in force from
    the observation that knocks the contract in, that observation included: every engine applies it so (Monte Carlo
    counted it strictly after the knock-in until 2026-09-21; ``test_ko_reset_contract_rules.py`` pins the engines)."""
    env = flat_env(datetime(2027, 3, 16))
    product = _float_ko_reset([0.0], [0.0], [0.0], post_ko=70.0)
    both = product.decide_observations_at_valuation(72.0, env)
    assert both.knocked_in and both.knocked_out and both.ko_regime == "post" and both.ko_record.payoff == pytest.approx(3.0)
    only_in = product.decide_observations_at_valuation(65.0, env)
    assert only_in.knocked_in and not only_in.knocked_out


def test_a_rebased_post_schedule_is_not_decided_here():
    """Its records are offsets from a knock-in time the carried state does not hold: nothing places them at this instant."""
    env = flat_env(datetime(2027, 3, 16))
    product = _float_ko_reset([0.0], [0.0], [0.0], mode=PostKOScheduleMode.REBASED)
    assert product.decide_observations_at_valuation(104.0, env).ko_regime == "pre"
    assert not product.decide_observations_at_valuation(96.0, env, knocked_in=True).knocked_out
