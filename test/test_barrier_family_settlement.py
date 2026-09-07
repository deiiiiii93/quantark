"""Leg-specific settlement timing across analytical and MC barrier families."""

import math
from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import (
    BarrierAnalyticalEngine,
    DoubleBarrierOptionAnalyticalEngine,
    DoubleSharkfinOptionAnalyticalEngine,
    OneTouchAnalyticalEngine,
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import (
    BarrierOptionMCEngine,
    DoubleSharkfinOptionMCEngine,
    LocalVolBarrierMCEngine,
    SingleSharkfinOptionMCEngine,
)
from quantark.asset.equity.engine.mc import barrier_vol_mc_engines
from quantark.asset.equity.engine.pde import BarrierPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    ObservationRecord,
    ObservationSchedule,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.execution.errors import CapabilityError
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationAggregation,
    ObservationType,
    OptionType,
    TouchType,
)


MATURITY = 1.0
LAG = 0.10


@pytest.fixture
def env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=LinearRateCurve(
            [(0.25, 0.01), (0.50, 0.025), (1.0, 0.06), (1.2, 0.07)]
        ),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


@pytest.fixture
def flat_env():
    """A delayed first-hit payment is only admitted on a FLAT curve (patch spec §5)."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _lagged():
    return SettlementConvention(
        lag=LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )


def test_no_touch_analytical_uses_terminal_payment_timing(env):
    engine = OneTouchAnalyticalEngine()

    def _product(convention):
        return OneTouchOption(
            barrier=120.0,
            barrier_direction=BarrierDirection.UP,
            maturity=MATURITY,
            rebate=10.0,
            payment_at_hit=False,
            touch_type=TouchType.NO_TOUCH,
            observation_type=ObservationType.CONTINUOUS,
            settlement_convention=convention,
        )

    immediate = engine.price(_product(None), env)
    delayed = engine.price(_product(_lagged()), env)

    assert delayed == pytest.approx(
        immediate
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY),
        rel=2.0e-12,
    )


EXPIRY_DATE = datetime(2027, 1, 1)          # one CALENDAR_DAYS year after the fixture's valuation date


def _expiry(dated):
    """Day-based settlement lags need an authoritative expiry date on the terminal leg."""
    return {"exercise_date": EXPIRY_DATE} if dated else {"maturity": MATURITY}


def _one_touch(convention, dated=False):
    return OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, rebate=10.0,
                          payment_at_hit=True, touch_type=TouchType.ONE_TOUCH,
                          observation_type=ObservationType.CONTINUOUS, settlement_convention=convention,
                          **_expiry(dated))


def _up_out_call(convention, rebate=10.0, dated=False):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                         rebate=rebate, pay_at_hit=True, observation_type=ObservationType.CONTINUOUS,
                         settlement_convention=convention, **_expiry(dated))


def _single_sharkfin(convention, knock_out_rebate=2.0, dated=False):
    return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                participation_rate=1.0, knock_out_rebate=knock_out_rebate, no_hit_rebate=0.0,
                                pay_at_hit=True, observation_type=ObservationType.CONTINUOUS,
                                settlement_convention=convention, **_expiry(dated))


def _double_sharkfin(convention, knock_out_rebate=2.0, dated=False, **kw):
    return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=70.0,
                                participation_rate=1.0, knock_out_rebate=knock_out_rebate,
                                no_hit_rebate=0.0, pay_at_hit=True, settlement_convention=convention,
                                **({"observation_type": ObservationType.CONTINUOUS} | _expiry(dated) | kw))


def test_continuous_first_hit_constant_lag_scales_by_exp_minus_r_lag(flat_env):
    """E[e^{-r(tau+L)} 1{tau<=T}] = e^{-rL} E[e^{-r tau} 1{tau<=T}] under the formula's flat r (patch spec §5.2)."""
    factor = math.exp(-flat_env.get_rate(MATURITY) * LAG)
    touch = OneTouchAnalyticalEngine()
    immediate = touch.price(_one_touch(None), flat_env)
    delayed = touch.price(_one_touch(_lagged()), flat_env)
    assert delayed != immediate
    assert delayed == pytest.approx(immediate * factor, rel=2e-12)
    # barrier: the option leg keeps its terminal delay, only the rebate leg carries exp(-r L)
    eng = BarrierAnalyticalEngine()
    rebate_leg = eng.price(_up_out_call(None), flat_env) - eng.price(_up_out_call(None, rebate=0.0), flat_env)
    assert rebate_leg > 0.0
    assert eng.price(_up_out_call(_lagged()), flat_env) == pytest.approx(
        eng.price(_up_out_call(_lagged(), rebate=0.0), flat_env) + rebate_leg * factor, rel=2e-12)


def test_delayed_first_hit_payment_is_rejected_on_a_sloped_curve(env, flat_env):
    """The formula defers at r(T) but the already-hit leg defers on the curve: gate, don't approximate.

    Admitting the lag under a term structure makes the price jump at the barrier by
    rebate * (DF(L) - exp(-r(T) L)), which a knock-out day would book as a spurious PnL.
    """
    touch = OneTouchAnalyticalEngine()
    with pytest.raises(CapabilityError, match="FLAT rate curve"):
        touch.price(_one_touch(_lagged()), env)
    # the same gate guards the barrier and both sharkfin cash legs
    with pytest.raises(CapabilityError, match="FLAT rate curve"):
        BarrierAnalyticalEngine().price(_up_out_call(_lagged()), env)
    with pytest.raises(CapabilityError, match="FLAT rate curve"):
        SingleSharkfinOptionAnalyticalEngine().price(_single_sharkfin(_lagged()), env)
    with pytest.raises(CapabilityError, match="FLAT rate curve"):
        DoubleSharkfinOptionAnalyticalEngine().price(_double_sharkfin(_lagged()), env)
    # a pay-at-hit contract with NO lag is unaffected by the curve's shape
    assert touch.price(_one_touch(None), env) > 0.0
    # and a flat curve quoted through a term-curve class still passes: the gate tests the
    # discount factor, not the curve's type
    sloped_but_flat = deepcopy(env)
    sloped_but_flat.rate_curve = LinearRateCurve([(0.25, 0.05), (0.5, 0.05), (1.0, 0.05), (1.2, 0.05)])
    assert touch.price(_one_touch(_lagged()), sloped_but_flat) == pytest.approx(
        touch.price(_one_touch(_lagged()), flat_env), rel=2e-12)


def test_calendar_day_lag_is_constant_only_under_act_style_day_counts(flat_env):
    """Day-based lags need a dated contract (terminal leg); the hit leg then scales by exp(-r n/basis)."""
    unadjusted = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS,
                                      business_day_convention=BusinessDayConvention.UNADJUSTED)
    engine = OneTouchAnalyticalEngine()

    def check_constant(pricing_env, basis):
        plain = _one_touch(None, dated=True)
        r = pricing_env.get_rate(plain.get_maturity(pricing_env))
        assert engine.price(_one_touch(unadjusted, dated=True), pricing_env) == pytest.approx(
            engine.price(plain, pricing_env) * math.exp(-r * 2 / basis), rel=2e-12)

    check_constant(flat_env, 365)                                        # CALENDAR_DAYS: days / 365
    env365 = deepcopy(flat_env)
    env365.day_count_convention = DayCountConvention.ACT_365
    check_constant(env365, 365)
    env360 = deepcopy(flat_env)
    env360.day_count_convention = DayCountConvention.ACT_360
    check_constant(env360, 360)
    env_isda = deepcopy(flat_env)
    env_isda.day_count_convention = DayCountConvention.ACT_ACT_ISDA
    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(_one_touch(unadjusted, dated=True), env_isda)
    # adjusted calendar-day and business-day lags are hit-date dependent (a calendar is needed
    # for the terminal leg to resolve at all)
    env_cal = deepcopy(flat_env)
    env_cal.calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    following = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS)     # FOLLOWING adjusts
    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(_one_touch(following, dated=True), env_cal)
    business = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS)
    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(_one_touch(business, dated=True), env_cal)
    # the same rules guard the barrier and sharkfin cash legs before pricing
    with pytest.raises(CapabilityError, match="first-hit"):
        BarrierAnalyticalEngine().price(_up_out_call(following, dated=True), env_cal)
    with pytest.raises(CapabilityError, match="first-hit"):
        SingleSharkfinOptionAnalyticalEngine().price(_single_sharkfin(business, dated=True), env_cal)
    with pytest.raises(CapabilityError, match="first-hit"):
        DoubleSharkfinOptionAnalyticalEngine().price(_double_sharkfin(following, dated=True), env_cal)


def test_per_record_settlement_timing_still_rejected(env):
    schedule = ObservationSchedule(records=[ObservationRecord(observation_time=0.5, settlement_time=0.6, barrier=120.0,
                                                              payoff=10.0)],
                                   aggregation_mode=ObservationAggregation.STOP_FIRST_HIT)
    product = OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=MATURITY, rebate=10.0,
                             payment_at_hit=True, touch_type=TouchType.ONE_TOUCH,
                             observation_type=ObservationType.DISCRETE, observation_schedule=schedule)
    with pytest.raises(CapabilityError, match="first-hit"):
        OneTouchAnalyticalEngine().price(product, env)


def test_pde_lag_effect_matches_analytical_scaling():
    """Flat rate: the lag EFFECT (immediate - delayed) agrees across engines; grid error cancels."""
    flat = PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.05), div_yield=ContinuousDividendYield(div_yield=0.0),
                              valuation_date=datetime(2026, 1, 1))
    lag = SettlementConvention(lag=0.5, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    an = BarrierAnalyticalEngine()
    pde = BarrierPDESolver(PDEParams(accuracy="high"))
    d_an = an.price(_up_out_call(None), flat) - an.price(_up_out_call(lag), flat)
    d_pde = pde.price(_up_out_call(None), flat) - pde.price(_up_out_call(lag), flat)
    assert d_an > 0.0
    # The PDE reproduces the analytical lag effect to well inside this gate; the tolerance is
    # set by the solver's own discretisation error, not by the size of the effect.
    assert d_pde == pytest.approx(d_an, rel=1e-3)


@pytest.mark.parametrize("engine,make", [(SingleSharkfinOptionAnalyticalEngine(), _single_sharkfin),
                                         (DoubleSharkfinOptionAnalyticalEngine(), _double_sharkfin)])
def test_sharkfin_first_hit_constant_lag_scales(flat_env, engine, make):
    """Only the hit-paid cash leg carries exp(-r L); the option leg keeps its own terminal delay."""
    factor = math.exp(-flat_env.get_rate(MATURITY) * LAG)
    base = engine.price(make(None), flat_env)
    no_cash = engine.price(make(None, knock_out_rebate=0.0), flat_env)
    assert base > no_cash                                             # the cash leg is material
    lagged = engine.price(make(_lagged()), flat_env)
    no_cash_lagged = engine.price(make(_lagged(), knock_out_rebate=0.0), flat_env)
    assert lagged == pytest.approx(no_cash_lagged + (base - no_cash) * factor, rel=2e-12)


def test_discrete_double_sharkfin_lag_discounts_each_node_at_its_settlement_time(env):
    """The discrete leg discounts every node at its own resolved settlement time (no constant factor).

    Under a flat rate every node ratio DF(t+L)/DF(t) is exp(-r L), so the lagged cash leg is exactly
    that multiple; under the term-structure fixture the lagged leg is a first-hit-probability-weighted
    average of the two node ratios and lies strictly between them.
    """
    times = (0.25, 0.5)                                       # BGK needs a regular interval: two nodes
    schedule = ObservationSchedule(records=[ObservationRecord(observation_time=t) for t in times])
    engine = DoubleSharkfinOptionAnalyticalEngine()

    def make(convention, knock_out_rebate=2.0):
        return _double_sharkfin(convention, knock_out_rebate, observation_type=ObservationType.DISCRETE,
                                observation_schedule=schedule)

    def cash_legs(pricing_env):
        plain = engine.price(make(None), pricing_env) - engine.price(make(None, knock_out_rebate=0.0), pricing_env)
        lagged = (engine.price(make(_lagged()), pricing_env)
                  - engine.price(make(_lagged(), knock_out_rebate=0.0), pricing_env))
        assert plain > 0.0
        return plain, lagged

    flat = deepcopy(env)
    flat.rate_curve = FlatRateCurve(rate=0.05)
    plain, lagged = cash_legs(flat)
    assert lagged == pytest.approx(plain * math.exp(-0.05 * LAG), rel=2e-12)
    plain, lagged = cash_legs(env)
    ratios = [env.get_discount_factor(t + LAG) / env.get_discount_factor(t) for t in times]
    assert ratios[0] != pytest.approx(ratios[1], rel=1e-6)                     # the curve makes them differ
    assert min(ratios) * plain < lagged < max(ratios) * plain


class _Paths:
    def generate_paths(self, **_kwargs):
        return np.array(
            [
                [100.0, 106.0, 104.0, 103.0],
                [100.0, 102.0, 106.0, 104.0],
            ]
        ), None


def test_discrete_barrier_mc_maps_first_hit_to_event_payment_df(
    env, monkeypatch
):
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.25,
                barrier=105.0,
                payoff=7.0,
            ),
            ObservationRecord(
                observation_time=0.50,
                barrier=105.0,
                payoff=7.0,
            ),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=105.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=MATURITY,
        rebate=7.0,
        pay_at_hit=True,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
        settlement_convention=_lagged(),
    )
    engine = BarrierOptionMCEngine(
        params=MCParams(num_paths=2, time_steps=3, seed=5)
    )
    monkeypatch.setattr(engine, "_create_path_generator", lambda *_args: _Paths())

    price = engine.price(product, env)

    assert price == pytest.approx(
        0.5
        * 7.0
        * (
            env.get_discount_factor(0.25 + LAG)
            + env.get_discount_factor(0.50 + LAG)
        )
    )


class _ContinuousPaths:
    def generate_paths(self, **_kwargs):
        return np.array(
            [
                [100.0, 106.0, 104.0],
                [100.0, 102.0, 99.0],
            ]
        ), None


def test_continuous_barrier_mc_maps_hit_index_to_event_payment_df(
    env, monkeypatch
):
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=105.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=MATURITY,
        rebate=7.0,
        pay_at_hit=True,
        observation_type=ObservationType.CONTINUOUS,
        settlement_convention=_lagged(),
    )
    engine = BarrierOptionMCEngine(
        params=MCParams(num_paths=2, time_steps=2, seed=5)
    )
    monkeypatch.setattr(
        engine, "_create_path_generator", lambda *_args: _ContinuousPaths()
    )

    price = engine.price(product, env)

    assert price == pytest.approx(
        0.5 * 7.0 * env.get_discount_factor(0.50 + LAG)
    )


def test_already_hit_barrier_mc_resolves_event_payment_from_valuation_date(env):
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=MATURITY,
        rebate=7.0,
        pay_at_hit=True,
        observation_type=ObservationType.CONTINUOUS,
        settlement_convention=_lagged(),
    )

    price = BarrierOptionMCEngine(
        params=MCParams(num_paths=2, time_steps=2, seed=5)
    ).price(product, env)

    assert price == pytest.approx(7.0 * env.get_discount_factor(LAG))


@pytest.mark.parametrize(
    ("engine", "make_product"),
    [
        (
            BarrierAnalyticalEngine(),
            lambda convention: BarrierOption(
                strike=100.0,
                option_type=OptionType.CALL,
                barrier=130.0,
                barrier_type=BarrierType.UP_OUT,
                maturity=MATURITY,
                rebate=2.0,
                pay_at_hit=False,
                observation_type=ObservationType.EXPIRY,
                settlement_convention=convention,
            ),
        ),
        (
            DoubleBarrierOptionAnalyticalEngine(),
            lambda convention: DoubleBarrierOption(
                strike=100.0,
                option_type=OptionType.CALL,
                upper_barrier=130.0,
                lower_barrier=70.0,
                barrier_type=DoubleBarrierType.KNOCK_OUT,
                maturity=MATURITY,
                rebate=2.0,
                observation_type=ObservationType.EXPIRY,
                settlement_convention=convention,
            ),
        ),
        (
            SingleSharkfinOptionAnalyticalEngine(),
            lambda convention: SingleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                barrier=130.0,
                maturity=MATURITY,
                participation_rate=1.0,
                knock_out_rebate=2.0,
                no_hit_rebate=1.0,
                pay_at_hit=False,
                observation_type=ObservationType.EXPIRY,
                settlement_convention=convention,
            ),
        ),
        (
            DoubleSharkfinOptionAnalyticalEngine(),
            lambda convention: DoubleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                upper_barrier=130.0,
                lower_barrier=70.0,
                maturity=MATURITY,
                participation_rate=1.0,
                knock_out_rebate=2.0,
                no_hit_rebate=1.0,
                pay_at_hit=False,
                observation_type=ObservationType.EXPIRY,
                settlement_convention=convention,
            ),
        ),
    ],
)
def test_expiry_barrier_family_cashflows_use_terminal_payment(
    env, engine, make_product
):
    immediate = engine.price(make_product(None), env)
    delayed = engine.price(make_product(_lagged()), env)

    assert delayed == pytest.approx(
        immediate
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY),
        rel=5.0e-11,
    )


@pytest.mark.parametrize(
    ("engine_factory", "make_product"),
    [
        (
            lambda: SingleSharkfinOptionMCEngine(
                params=MCParams(num_paths=256, time_steps=4, seed=11)
            ),
            lambda convention: SingleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                barrier=130.0,
                maturity=MATURITY,
                knock_out_rebate=2.0,
                no_hit_rebate=1.0,
                observation_type=ObservationType.EXPIRY,
                settlement_convention=convention,
            ),
        ),
        (
            lambda: DoubleSharkfinOptionMCEngine(
                params=MCParams(num_paths=256, time_steps=4, seed=11)
            ),
            lambda convention: DoubleSharkfinOption(
                strike=100.0,
                option_type=OptionType.CALL,
                upper_barrier=130.0,
                lower_barrier=70.0,
                maturity=MATURITY,
                knock_out_rebate=2.0,
                no_hit_rebate=1.0,
                observation_type=ObservationType.EXPIRY,
                settlement_convention=convention,
            ),
        ),
    ],
)
def test_sharkfin_mc_terminal_cashflows_use_payment_df(
    env, engine_factory, make_product
):
    immediate = engine_factory().price(make_product(None), env)
    delayed = engine_factory().price(make_product(_lagged()), env)

    assert delayed == pytest.approx(
        immediate
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY),
        rel=2.0e-12,
    )


def test_vol_model_barrier_mc_scales_terminal_kernel_value(
    env, monkeypatch
):
    monkeypatch.setattr(
        barrier_vol_mc_engines,
        "price_barrier_lv_mc",
        lambda *_args, **_kwargs: (8.0, 0.5),
    )
    engine = LocalVolBarrierMCEngine(
        params=MCParams(num_paths=2, time_steps=2, seed=5),
        local_vol_surface=object(),
    )

    def _product(convention):
        return BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=130.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=MATURITY,
            rebate=2.0,
            pay_at_hit=False,
            observation_type=ObservationType.EXPIRY,
            settlement_convention=convention,
        )

    immediate = engine.price(_product(None), env)
    delayed = engine.price(_product(_lagged()), env)

    assert delayed == pytest.approx(
        immediate
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY)
    )
    assert engine.get_last_std_error() == pytest.approx(
        0.5
        * env.get_discount_factor(MATURITY + LAG)
        / env.get_discount_factor(MATURITY)
    )


def test_vol_model_barrier_mc_rejects_delayed_mixed_hit_cashflow(env):
    product = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=130.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=MATURITY,
        rebate=2.0,
        pay_at_hit=True,
        observation_type=ObservationType.CONTINUOUS,
        settlement_convention=_lagged(),
    )
    engine = LocalVolBarrierMCEngine(
        params=MCParams(num_paths=2, time_steps=2, seed=5),
        local_vol_surface=object(),
    )

    with pytest.raises(CapabilityError, match="first-hit"):
        engine.price(product, env)
