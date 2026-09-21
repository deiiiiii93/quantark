"""Regressions for the findings of docs/superpowers/reviews/intraday-pricing-2026-09-21/REVIEW.md.

One test group per finding, named by its number, so the review can be audited against the suite. The fixtures are
the review's: a monthly snowball, Phoenix and knock-out-reset snowball at their final close under BEFORE with every
earlier fixing confirmed, and a rebate unlike the knock-out cash so that a missed event shows. Findings whose natural
home is an existing file are asserted there and named in ``COVERED_ELSEWHERE``.
"""
import math
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2, PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, ValuationPoint
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.observation_schedule import (ObservationRecord, ObservationSchedule,
                                                                        at_valuation_instant)
from quantark.asset.equity.product.option.snowball_config import AirbagConfig
from quantark.execution import PricingSession
from quantark.intraday import value_intraday
from quantark.intraday.batch import spot_curve
from quantark.intraday.context import resolve_context
from quantark.intraday.engines.base import _redemption_piece
from quantark.intraday.events import EventKind, EventPhase, resolve_timeline
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.option_enums import ProtectionType
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, dated_ko_reset, dated_phoenix, dated_snowball, flat_env

#: Findings asserted in the file that owns the behaviour rather than here.
COVERED_ELSEWHERE = {
    8: "test/test_ko_reset_contract_rules.py::test_r8_*",
    9: "test/test_ko_reset_contract_rules.py::test_r9_*",
}

T0 = datetime(2026, 3, 16)
PROBE = datetime(2026, 9, 15, tzinfo=SHANGHAI)
LAG = timedelta(days=5)
ENGINES = {
    "snowball": {"mc": lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=2048)),
                 "pde": lambda: SnowballPDESolver(PDEParams()), "quad_v2": SnowballQuadEngineV2},
    "phoenix": {"mc": lambda: PhoenixMCEngine(params=MCParams(seed=7, num_paths=2048)),
                "pde": lambda: PhoenixPDESolver(PDEParams()), "quad_v2": PhoenixQuadEngineV2},
}
QUAD_V1 = {"snowball": SnowballQuadEngine, "phoenix": PhoenixQuadEngine}
ROUTES = [(kind, name) for kind in ENGINES for name in ENGINES[kind]]


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _lagged(product):
    """Every payment five calendar days after its observation: the terminal redemption and each knock-out."""
    product.settlement_date = product.exercise_date + LAG
    for config in (product.barrier_config, getattr(product, "post_barrier_config", None)):
        if config is not None:
            for record in config.ko_observation_schedule.records:
                record.settlement_date = record.observation_date + LAG
    return product


def _close(cal, sessions, profile, kind, spot=100.0, knocked_in_earlier=False, lag=False, last_ki=True):
    """The final close under BEFORE; every earlier fixing confirmed at 100 (70 at the fourth: an earlier knock-in).

    Snowball: KO 103 paying 12, KI 75, rebate 5, no principal. Phoenix: principal 100, KO 103, coupon barrier 80 with
    a 1.0 coupon a period, KI 75, rebate 2. ``last_ki=False`` drops the final knock-in record, so nothing is left to
    knock the claim in at maturity."""
    product = (dated_snowball(cal, T0, rebate_rate=0.05) if kind == "snowball"
               else dated_phoenix(cal, T0, rebate_rate=0.02, fixed_fraction=1.0 / 12.0))
    if lag:
        _lagged(product)
    if not last_ki:
        config = product.barrier_config
        product.barrier_config = replace(config, ki_observation_schedule=replace(
            config.ki_observation_schedule, records=config.ki_observation_schedule.records[:-1]))
    request = IntradayValuationRequest(product=product, pricing_env=flat_env(PROBE), session_calendar=sessions,
                                       variance_profile=profile, event_phase=EventPhase.BEFORE)
    kos = [e for e in resolve_context(request).timeline.events if e.kind is EventKind.KO]
    fixings = tuple(Fixing(e.timestamp, 70.0 if knocked_in_earlier and i == 3 else 100.0)
                    for i, e in enumerate(kos[:-1]))
    return replace(request, pricing_env=flat_env(kos[-1].timestamp, spot=spot), fixings=fixings)


def _ko_reset_close(cal, sessions, profile, spot=90.0, post_ko=95.0, lag=False):
    """A knocked-in knock-out-reset snowball at its second schedule's final close (the first ended in September):
    a knocked-in snowball on the second schedule, KO 95 paying 3, strike 100, no principal."""
    product = dated_ko_reset(cal, T0, pre_months=6, post_months=12, post_ko=post_ko, rebate_rate=0.05)
    if lag:
        _lagged(product)
    ts = resolve_timeline(product, sessions, flat_env(PROBE)).terminal().timestamp
    state = AutocallableLifecycleState(knocked_in=True,
                                       valuation_point=ValuationPoint(date=(ts - timedelta(days=1)).replace(tzinfo=None)))
    return IntradayValuationRequest(product=product, pricing_env=flat_env(ts, spot=spot), session_calendar=sessions,
                                    variance_profile=profile, lifecycle_state=state)


def _point(engine, request):
    result = value_intraday(engine, replace(request, greeks=("delta", "gamma"), greek_convention="point"))
    return {g.name: g for g in result.greeks}


def _framed(engine, request):
    with PricingSession() as session:
        return value_intraday(engine, request, session=session)


# =====================================================================================================================
# R1: a positive time, however small, is still ahead of the valuation instant
# =====================================================================================================================
def _millisecond_before_the_close(cal, sessions, profile):
    request = _close(cal, sessions, profile, "snowball", spot=103.0)
    return replace(request, pricing_env=flat_env(request.pricing_env.valuation_date - timedelta(milliseconds=1),
                                                 spot=103.0))


def _terminal_integral(ctx):
    """The review's independent value: S_T - 100 below 75, 5 from 75 to 103, 12 at and above 103, integrated over
    the lognormal law of the resolved positive variance (no production payoff, no reference solver)."""
    tau = ctx.numerical.maturity_tau
    variance = float(ctx.pricing_env.vol_surface.total_variance(100.0, tau, 103.0))
    mean, sd = 0.02 * tau - 0.5 * variance, math.sqrt(variance)
    cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))   # noqa: E731
    below_ki, below_ko = cdf((math.log(75.0 / 103.0) - mean) / sd), cdf(-mean / sd)
    lower_asset = 103.0 * math.exp(mean + 0.5 * variance) * cdf((math.log(75.0 / 103.0) - mean - variance) / sd)
    return math.exp(-0.03 * tau) * (lower_asset - 100.0 * below_ki + 5.0 * (below_ko - below_ki)
                                    + 12.0 * (1.0 - below_ko))


def test_r1_only_time_zero_is_the_valuation_instant():
    assert at_valuation_instant(0.0) and at_valuation_instant(-0.0)
    assert not at_valuation_instant(3.1709791983764586e-11)      # one millisecond
    assert not at_valuation_instant(1e-15)


def test_r1_the_decision_leaves_an_observation_a_millisecond_ahead_to_the_engines(sse_calendar, sse_sessions, desk):
    ctx = resolve_context(_millisecond_before_the_close(sse_calendar, sse_sessions, desk))
    assert 0.0 < ctx.numerical.maturity_tau < 1e-10
    decision = ctx.numerical.product.decide_observations_at_valuation(103.0, ctx.pricing_env, knocked_in=False)
    assert not decision.knocked_out and not decision.knocked_in


def test_r1_every_route_prices_the_last_millisecond_before_the_close(sse_calendar, sse_sessions, desk):
    request = _millisecond_before_the_close(sse_calendar, sse_sessions, desk)
    expected = _terminal_integral(resolve_context(request))
    assert expected == pytest.approx(8.5, abs=1e-4)          # half the paths close at or above the knock-out level

    assert value_intraday(SnowballQuadEngineV2(), request).price == pytest.approx(expected, abs=1e-9)

    mc = value_intraday(SnowballMCEngine(params=MCParams(seed=7, num_paths=32768)), request)
    assert mc.method.startswith("mc_")                        # simulated, not decided on the spot (which paid 12)
    assert abs(mc.price - expected) < 4.0 * mc.numerical["std_error"]

    # The PDE cannot resolve a 2.8e-6 diffusion layer: it prices the positive horizon and says so (it raised before).
    pde = value_intraday(SnowballPDESolver(PDEParams()), request)
    assert pde.method.startswith("pde_") and pde.numerical["resolution"] == "under_resolved"


@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_r1_quad_v1_refuses_a_positive_horizon_below_its_time_resolution(sse_calendar, sse_sessions, desk, kind):
    request = _close(sse_calendar, sse_sessions, desk, kind, spot=90.0)
    at_close = resolve_context(request)
    before = resolve_context(replace(request, pricing_env=flat_env(
        request.pricing_env.valuation_date - timedelta(milliseconds=1), spot=90.0)))
    engine = QUAD_V1[kind]()
    with pytest.raises(ValidationError, match="time resolution"):
        engine.price(before.numerical.product, before.pricing_env)
    engine.price(at_close.numerical.product, at_close.pricing_env)       # the instant itself is decided


# =====================================================================================================================
# R2: no point derivative at a kink of the decided redemption
# =====================================================================================================================
@pytest.mark.parametrize("kind, name", ROUTES)
def test_r2_point_greeks_at_the_strike_of_a_knocked_in_redemption_are_undefined(sse_calendar, sse_sessions, desk,
                                                                               kind, name):
    greeks = _point(ENGINES[kind][name](), _close(sse_calendar, sse_sessions, desk, kind, knocked_in_earlier=True))
    for greek in greeks.values():
        assert greek.status == "undefined" and greek.value is None, greek


@pytest.mark.parametrize("kind, name", ROUTES)
def test_r2_away_from_the_kink_the_decided_redemption_has_its_slope(sse_calendar, sse_sessions, desk, kind, name):
    greeks = _point(ENGINES[kind][name](), _close(sse_calendar, sse_sessions, desk, kind, spot=90.0,
                                                  knocked_in_earlier=True))
    assert greeks["delta"].status == "ok" and greeks["delta"].value == pytest.approx(1.0, abs=1e-6)
    assert greeks["gamma"].status == "ok" and greeks["gamma"].value == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("engine", [KOResetSnowballPDESolver, KOResetSnowballQuadEngineV2])
def test_r2_a_knocked_in_ko_reset_at_its_strike_has_no_point_greeks(sse_calendar, sse_sessions, desk, engine):
    greeks = _point(engine(), _ko_reset_close(sse_calendar, sse_sessions, desk, spot=100.0, post_ko=110.0))
    for greek in greeks.values():
        assert greek.status == "undefined" and greek.value is None, greek


def _snowball(cal, **payoff):
    product = dated_snowball(cal, T0, rebate_rate=0.05)
    airbag = payoff.pop("airbag", None)
    product.payoff_config = replace(product.payoff_config, **payoff)
    if airbag is not None:
        product.airbag_config = airbag
    return product


def _kinks(product, knocked_in, lo=50.0, hi=130.0, step=0.1):
    """The levels between which ``_redemption_piece`` changes, on a grid that avoids them."""
    env = flat_env(PROBE)
    spots = [lo + step * (i + 0.5) for i in range(int(round((hi - lo) / step)))]
    labels = [_redemption_piece(product, s, knocked_in, env) for s in spots]
    return [round(0.5 * (a + b), 6) for a, b, x, y in zip(spots, spots[1:], labels, labels[1:]) if x != y]


@pytest.mark.parametrize("payoff, knocked_in, kinks", [
    # knocked in: the loss S - 100 below the strike, nothing above it
    (dict(), True, [100.0]),
    # a 20% partial protection floors the loss at -20: flat again below 80
    (dict(protection_type=ProtectionType.PARTIAL, protection_rate=0.2), True, [80.0, 100.0]),
    # full protection: no loss at all, so the knocked-in redemption is flat
    (dict(protection_type=ProtectionType.FULL), True, []),
    # half participation moves the floor of a 20% protection to 60
    (dict(participation_rate=0.5, protection_type=ProtectionType.PARTIAL, protection_rate=0.2), True, [60.0, 100.0]),
    # an airbag below 70 halves the participation: the loss jumps from -30 to -15 there
    (dict(airbag=AirbagConfig(airbag_barrier=70.0, airbag_participation_rate=0.5)), True, [70.0, 100.0]),
    # not knocked in: a fixed rebate is flat, a call-style rebate kinks at its strike
    (dict(), False, []),
    (dict(call_rebate_enabled=True, call_strike=105.0), False, [105.0]),
])
def test_r2_the_redemption_piece_changes_exactly_at_its_breakpoints(sse_calendar, payoff, knocked_in, kinks):
    assert _kinks(_snowball(sse_calendar, **payoff), knocked_in) == kinks


def test_r2_the_phoenix_redemption_has_the_same_breakpoints(sse_calendar):
    product = dated_phoenix(sse_calendar, T0, rebate_rate=0.02)
    product.payoff_config = replace(product.payoff_config, protection_type=ProtectionType.PARTIAL, protection_rate=0.2)
    assert _kinks(product, True) == [80.0, 100.0]
    assert _kinks(product, False) == []


# =====================================================================================================================
# R3: the public session carries the maturity resolver
# =====================================================================================================================
@pytest.mark.parametrize("kind, name", ROUTES)
def test_r3_a_session_takes_the_decided_maturity_close_without_a_kernel_dispatch(sse_calendar, sse_sessions, desk,
                                                                                kind, name):
    request = _close(sse_calendar, sse_sessions, desk, kind, spot=90.0)
    scalar = value_intraday(ENGINES[kind][name](), request)
    framed = _framed(ENGINES[kind][name](), request)
    assert framed.method == scalar.method == "decided_at_valuation"
    assert framed.price == scalar.price == (5.0 if kind == "snowball" else 103.0)
    assert "manifest:not-dispatched (decided_at_valuation route calls no engine)" in framed.records


@pytest.mark.parametrize("engine", [KOResetSnowballPDESolver, KOResetSnowballQuadEngineV2])
def test_r3_a_session_takes_the_ko_reset_final_close(sse_calendar, sse_sessions, desk, engine):
    request = _ko_reset_close(sse_calendar, sse_sessions, desk)
    assert _framed(engine(), request).price == value_intraday(engine(), request).price == -10.0


# =====================================================================================================================
# R4: a QUAD V2 spot curve at the maturity close
# =====================================================================================================================
@pytest.mark.parametrize("kind, expected", [
    ("snowball", {70.0: (-30.0, 1.0), 90.0: (5.0, 0.0), 104.0: (12.0, 0.0)}),       # knocks in, survives, knocks out
    ("phoenix", {70.0: (70.0, 1.0), 90.0: (103.0, 0.0), 104.0: (101.0, 0.0)}),
])
def test_r4_a_quad_v2_curve_decides_each_spot_as_a_scalar_value_does(sse_calendar, sse_sessions, desk, kind, expected):
    request = _close(sse_calendar, sse_sessions, desk, kind)
    engine = ENGINES[kind]["quad_v2"]()
    curve = spot_curve(engine, request, list(expected))
    assert curve.method == "decided_at_valuation" and not curve.provisional
    for point in curve:
        price, delta = expected[point.spot]
        scalar = value_intraday(engine, replace(request, pricing_env=flat_env(request.pricing_env.valuation_date,
                                                                              spot=point.spot))).price
        assert point.price == pytest.approx(price, abs=1e-12) and point.price == scalar
        assert point.status == "ok" and point.delta == pytest.approx(delta, abs=1e-6)


def test_r4_a_curve_point_on_a_kink_has_no_point_greeks(sse_calendar, sse_sessions, desk):
    request = _close(sse_calendar, sse_sessions, desk, "snowball", knocked_in_earlier=True)
    point, = spot_curve(SnowballQuadEngineV2(), request, [100.0])
    assert point.price == 0.0 and point.status == "undefined" and point.delta is None and point.gamma is None


def test_r4_a_ko_reset_curve_at_its_final_close(sse_calendar, sse_sessions, desk):
    curve = spot_curve(KOResetSnowballQuadEngineV2(), _ko_reset_close(sse_calendar, sse_sessions, desk), [80.0, 90.0, 104.0])
    assert [p.price for p in curve] == [-20.0, -10.0, 3.0]


# =====================================================================================================================
# R5: a terminal redemption decided now is paid at its payment time
# =====================================================================================================================
@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_r5_every_engine_discounts_a_lagged_redemption_decided_at_the_close(sse_calendar, sse_sessions, desk, kind):
    request = _close(sse_calendar, sse_sessions, desk, kind, spot=90.0, lag=True)
    ctx = resolve_context(request)
    cash = 5.0 if kind == "snowball" else 103.0          # the rebate; the Phoenix principal, rebate and last coupon
    expected = cash * float(ctx.pricing_env.get_discount_factor(5.0 / 365.0))
    assert expected < cash - 1e-3
    twin, env = ctx.numerical.product, ctx.pricing_env
    for engine in (ENGINES[kind]["mc"](), ENGINES[kind]["pde"](), QUAD_V1[kind]()):
        assert float(engine.price(twin, env)) == pytest.approx(expected, abs=1e-12), type(engine).__name__
    for name in ENGINES[kind]:
        assert value_intraday(ENGINES[kind][name](), request).price == pytest.approx(expected, abs=1e-12)
        assert _framed(ENGINES[kind][name](), request).price == pytest.approx(expected, abs=1e-12)


def test_r5_the_ko_reset_redemption_is_discounted_too(sse_calendar, sse_sessions, desk):
    request = _ko_reset_close(sse_calendar, sse_sessions, desk, lag=True)
    ctx = resolve_context(request)
    expected = -10.0 * float(ctx.pricing_env.get_discount_factor(5.0 / 365.0))
    assert value_intraday(KOResetSnowballQuadEngineV2(), request).price == pytest.approx(expected, abs=1e-12)
    for engine in (KOResetSnowballPDESolver(PDEParams()), KOResetSnowballQuadEngine()):
        assert float(engine.price(ctx.numerical.product, ctx.pricing_env)) == pytest.approx(expected, abs=1e-12)


def test_r5_without_a_lag_the_redemption_is_the_cash(sse_calendar, sse_sessions, desk):
    ctx = resolve_context(_close(sse_calendar, sse_sessions, desk, "snowball", spot=90.0))
    for engine in (SnowballPDESolver(PDEParams()), SnowballQuadEngine()):
        assert float(engine.price(ctx.numerical.product, ctx.pricing_env)) == 5.0


# =====================================================================================================================
# R6 / R7: a discrete knock-in schedule that has ended knocks nothing in at maturity
# =====================================================================================================================
@pytest.mark.parametrize("kind, cash", [("snowball", 5.0), ("phoenix", 102.0)])
def test_r6_r7_a_finished_knock_in_schedule_leaves_the_survival_redemption(sse_calendar, sse_sessions, desk, kind,
                                                                           cash):
    request = _close(sse_calendar, sse_sessions, desk, kind, spot=70.0, last_ki=False)
    ctx = resolve_context(request)
    assert not ctx.numerical.knocked_in
    assert not [e for e in ctx.numerical.remaining_events if e.kind is EventKind.KI]
    twin, env = ctx.numerical.product, ctx.pricing_env
    assert twin.resolve_ki_observations(env) == []               # R7: an emptied schedule, not a missing one
    for engine in (ENGINES[kind]["mc"](), ENGINES[kind]["pde"](), QUAD_V1[kind]()):
        assert float(engine.price(twin, env)) == cash, type(engine).__name__
    for name in ENGINES[kind]:
        assert value_intraday(ENGINES[kind][name](), request).price == cash
        assert _framed(ENGINES[kind][name](), request).price == cash


def test_r7_a_missing_knock_in_schedule_is_still_an_error(sse_calendar):
    product, env = dated_phoenix(sse_calendar, T0), flat_env(datetime(2026, 9, 15))
    product.barrier_config = replace(product.barrier_config, ki_observation_schedule=None)
    with pytest.raises(ValidationError, match="KI observation schedule is required"):
        product.resolve_ki_observations(env)
    product.barrier_config = replace(product.barrier_config, ki_observation_schedule=ObservationSchedule(
        records=[ObservationRecord(observation_date=datetime(2027, 3, 16), barrier=75.0)]))
    assert len(product.resolve_ki_observations(env)) == 1
