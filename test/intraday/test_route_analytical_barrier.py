from datetime import date, datetime

import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, OneTouchAnalyticalEngine
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.execution.errors import CapabilityError
from quantark.intraday import value_intraday
from quantark.intraday.admissibility import analytical_barrier_admissibility
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.option_enums import BarrierDirection, BarrierType, ObservationType, OptionType, TouchType
from intraday.conftest import SHANGHAI, flat_env
from intraday.controls.gaussian_control import barrier_zero_carry

TS = datetime(2026, 9, 15, 10, 0, tzinfo=SHANGHAI)          # Tuesday
FRIDAY = datetime(2026, 9, 18)                              # same trading week: no weekend inside
DECEMBER = datetime(2026, 12, 15)
BARRIER_ENGINE, TOUCH_ENGINE = BarrierAnalyticalEngine(), OneTouchAnalyticalEngine()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


@pytest.fixture
def uniform(sse_sessions):
    return VarianceProfile.uniform(sse_sessions, 244, reference_date=date(2026, 9, 15))


def _uo(expiry, rebate=0.0, observation_type=ObservationType.CONTINUOUS, schedule=None):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=106.0, barrier_type=BarrierType.UP_OUT,
                         exercise_date=expiry, rebate=rebate, pay_at_hit=True, observation_type=observation_type,
                         observation_schedule=schedule)


def _ctx(sessions, profile, product, r=0.03, q=0.01, spot=100.0):
    return resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(TS, spot=spot, r=r, q=q),
                                                    session_calendar=sessions, variance_profile=profile))


def test_uniform_week_is_admissible_and_prices_through_the_engine(sse_sessions, uniform):
    ctx = _ctx(sse_sessions, uniform, _uo(FRIDAY))
    adm = analytical_barrier_admissibility(ctx)
    assert adm.admissible and adm.mode == "uniform_calendar_rate"
    out = route_for(ctx, BARRIER_ENGINE).price(ctx, BARRIER_ENGINE)
    assert out.method == "analytical_uniform_calendar_rate"
    assert out.contingent_pv == pytest.approx(BarrierAnalyticalEngine().price(ctx.numerical.product, ctx.pricing_env), rel=1e-14)


def test_uniform_week_under_zero_carry_matches_the_independent_formula(sse_sessions, uniform):
    ctx = _ctx(sse_sessions, uniform, _uo(FRIDAY), r=0.0, q=0.0)
    out = route_for(ctx, BARRIER_ENGINE).price(ctx, BARRIER_ENGINE)
    u = float(ctx.time_map.to_trading(ctx.numerical.maturity_tau))
    closed = barrier_zero_carry(100.0, 100.0, 106.0, u, 0.20, is_call=True, is_up=True, is_knock_out=True)
    assert out.contingent_pv == pytest.approx(closed, rel=1e-9)


def test_a_weekend_breaks_the_uniform_rate(sse_sessions, uniform):
    adm = analytical_barrier_admissibility(_ctx(sse_sessions, uniform, _uo(DECEMBER)))
    assert not adm.admissible and "weekends" in adm.reason


def test_zero_carry_time_change_is_exact_for_any_profile(sse_sessions, desk):
    for rebate in (0.0, 1.5):
        ctx = _ctx(sse_sessions, desk, _uo(DECEMBER, rebate=rebate), r=0.0, q=0.0)
        adm = analytical_barrier_admissibility(ctx)
        assert adm.mode == "zero_carry_time_change"
        out = route_for(ctx, BARRIER_ENGINE).price(ctx, BARRIER_ENGINE)
        u = float(ctx.time_map.to_trading(ctx.numerical.maturity_tau))
        closed = barrier_zero_carry(100.0, 100.0, 106.0, u, 0.20, is_call=True, is_up=True, is_knock_out=True, rebate=rebate)
        assert out.method == "analytical_zero_carry_time_change" and out.contingent_pv == pytest.approx(closed, rel=1e-9)


def test_carry_with_a_non_uniform_clock_is_refused_with_alternatives(sse_sessions, desk):
    ctx = _ctx(sse_sessions, desk, _uo(DECEMBER))
    with pytest.raises(CapabilityError) as ei:
        route_for(ctx, BARRIER_ENGINE).price(ctx, BARRIER_ENGINE)
    msg = str(ei.value)
    assert "drift per unit variance" in msg and "BarrierPDESolver" in msg and "BarrierOptionMCEngine" in msg


def test_discrete_monitoring_names_bgk(sse_sessions, uniform):
    schedule = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=106.0)
                                            for d in (datetime(2026, 9, 16), datetime(2026, 9, 17), FRIDAY)])
    product = _uo(FRIDAY, observation_type=ObservationType.DISCRETE, schedule=schedule)
    ctx = _ctx(sse_sessions, uniform, product)
    with pytest.raises(CapabilityError, match="BGK"):
        route_for(ctx, BARRIER_ENGINE).price(ctx, BARRIER_ENGINE)
    with pytest.raises(CapabilityError, match="BGK"):
        value_intraday(BARRIER_ENGINE, ctx.request)


def test_one_touch_pay_at_hit_in_a_uniform_week(sse_sessions, uniform):
    touch = OneTouchOption(barrier=104.0, barrier_direction=BarrierDirection.UP, exercise_date=FRIDAY, rebate=1.0,
                           payment_at_hit=True, touch_type=TouchType.ONE_TOUCH, observation_type=ObservationType.CONTINUOUS)
    ctx = _ctx(sse_sessions, uniform, touch)
    out = route_for(ctx, TOUCH_ENGINE).price(ctx, TOUCH_ENGINE)
    assert out.method == "analytical_uniform_calendar_rate"
    assert out.contingent_pv == pytest.approx(OneTouchAnalyticalEngine().price(ctx.numerical.product, ctx.pricing_env), rel=1e-14)
    res = value_intraday(TOUCH_ENGINE, ctx.request)
    assert res.provisional and res.continuous_assumption.assumed_hit_at is None and 0.0 < res.price < 1.0
