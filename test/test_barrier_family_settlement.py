"""Leg-specific settlement timing across analytical and MC barrier families."""

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
from quantark.asset.equity.param import MCParams
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
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
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


def test_continuous_first_hit_analytical_rejects_unrepresentable_lag(env):
    product = OneTouchOption(
        barrier=120.0,
        barrier_direction=BarrierDirection.UP,
        maturity=MATURITY,
        rebate=10.0,
        payment_at_hit=True,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
        settlement_convention=_lagged(),
    )

    with pytest.raises(CapabilityError, match="first-hit"):
        OneTouchAnalyticalEngine().price(product, env)


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
