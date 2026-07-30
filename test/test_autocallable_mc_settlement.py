"""Settlement-aware cashflow decomposition for autocallable MC engines."""

from datetime import datetime

import numpy as np
import pytest

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import (
    ObservationRecord,
    ObservationSchedule,
    create_ko_reset_snowball,
)
from quantark.asset.equity.product.option.phoenix_config import (
    CouponBarrierConfig,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.settlement import (
    SettlementConvention,
    SettlementLagUnit,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    CouponPayType,
    ObservationType,
    PostKOScheduleMode,
)


LAG = 0.10
RATE = 0.05


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _lagged():
    return SettlementConvention(
        lag=LAG,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    )


def _barriers(schedule=None):
    return BarrierConfig(
        ko_barrier=105.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=None if schedule is not None else [0.5, 1.0],
        ko_observation_schedule=schedule,
        ki_barrier=70.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )


def _snowball(convention=None):
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=_barriers(),
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(
            coupon_pay_type=CouponPayType.INSTANT
        ),
        maturity=1.0,
        settlement_convention=convention,
    )


def _phoenix(convention=None, schedule=None):
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=_barriers(schedule),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=[80.0, 80.0],
            coupon_rate=0.04,
            coupon_pay_type=CouponPayType.INSTANT,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=True),
        maturity=1.0,
        settlement_convention=convention,
    )


@pytest.mark.parametrize(
    ("factory", "engine_type"),
    [
        (_snowball, SnowballMCEngine),
        (_phoenix, PhoenixMCEngine),
    ],
)
def test_uniform_lag_scales_pv_without_changing_path_events(
    factory, engine_type
):
    env = _env()
    params = MCParams(num_paths=4096, seed=17)
    immediate_engine = engine_type(params=params)
    delayed_engine = engine_type(params=params)

    immediate = immediate_engine.price(factory(None), env)
    delayed = delayed_engine.price(factory(_lagged()), env)
    immediate_result = immediate_engine.get_last_result()
    delayed_result = delayed_engine.get_last_result()

    assert delayed == pytest.approx(
        immediate * np.exp(-RATE * LAG),
        rel=2.0e-12,
    )
    assert delayed_result.ko_probability == immediate_result.ko_probability
    assert delayed_result.v0_probability == immediate_result.v0_probability
    assert delayed_result.v1_probability == immediate_result.v1_probability


def test_phoenix_bundle_preserves_record_and_terminal_payment_times():
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.5,
                settlement_time=0.56,
                barrier=105.0,
            ),
            ObservationRecord(
                observation_time=1.0,
                settlement_time=1.08,
                barrier=105.0,
            ),
        ]
    )
    product = _phoenix(_lagged(), schedule)
    engine = PhoenixMCEngine(params=MCParams(num_paths=1024, seed=9))
    engine.price(product, _env())
    timings = engine._payment_timings

    assert timings.observation_payment_times == pytest.approx([0.56, 1.08])
    assert timings.terminal.payment_time == pytest.approx(1.0 + LAG)


def test_ko_reset_pre_and_post_ko_payments_keep_event_lag():
    env = _env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=True,
    )
    product.settlement_convention = _lagged()
    engine = SnowballMCEngine(params=MCParams(num_paths=256, seed=5))

    for profile, maturity in (
        (product.get_pre_ko_observation_profile(env), 1.0),
        (product.get_post_ko_observation_profile(env), 2.0),
    ):
        _, payment_times = engine._compute_ko_schedule_payoffs(
            product,
            np.asarray(profile["observation_times"], dtype=float),
            np.asarray(profile["rates"], dtype=float),
            profile["records"],
            env,
            maturity,
        )
        assert payment_times == pytest.approx(
            np.asarray(profile["observation_times"]) + LAG
        )


def test_dcn_explicit_payment_dates_remain_authoritative():
    product = make_dcn(DCN_A)
    product.settlement_convention = _lagged()
    result = DCNMCEngine(
        num_paths=256,
        seed=3,
        use_sobol=False,
    ).price_detailed(product, flat_env(**FLAT))

    assert result.pv == (
        result.pv_fixed_coupons
        + result.pv_ko_coupons
        + result.pv_loss_leg
    )
