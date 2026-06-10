import math
from datetime import datetime

import pytest
from scipy import stats

from quantark.asset.equity.engine.analytical import (
    DoubleBarrierOptionAnalyticalEngine,
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.product.option import DoubleBarrierOption, DoubleSharkfinOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import ObservationFrequency, ObservationType, OptionType
from quantark.util.enum.option_enums import DoubleBarrierType
from quantark.util.exceptions import PricingError


def _pricing_env(
    spot: float = 100.0,
    rate: float = 0.03,
    div: float = 0.01,
    vol: float = 0.22,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    div: float,
    vol: float,
):
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * maturity) / (
        vol * sqrt_t
    )
    return d1, d1 - vol * sqrt_t


def test_expiry_call_matches_truncated_expectation():
    env = _pricing_env()
    product = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=75.0,
        upper_barrier=125.0,
        maturity=1.0,
        participation_rate=0.7,
        knock_out_rebate=2.0,
        no_hit_rebate=0.5,
        observation_type=ObservationType.EXPIRY,
        contract_multiplier=10.0,
    )
    engine = DoubleSharkfinOptionAnalyticalEngine()

    price = engine.price(product, env)

    spot = env.spot
    maturity = product.maturity
    rate = env.get_rate(maturity)
    div = env.get_div_yield(maturity)
    vol = env.get_vol(product.strike, maturity)
    discount = math.exp(-rate * maturity)

    d1_k, d2_k = _d1_d2(spot, product.strike, maturity, rate, div, vol)
    d1_u, d2_u = _d1_d2(spot, product.upper_barrier, maturity, rate, div, vol)
    _, d2_l = _d1_d2(spot, product.lower_barrier, maturity, rate, div, vol)

    option_leg = spot * math.exp(-div * maturity) * (
        stats.norm.cdf(d1_k) - stats.norm.cdf(d1_u)
    ) - product.strike * discount * (
        stats.norm.cdf(d2_k) - stats.norm.cdf(d2_u)
    )
    survival = stats.norm.cdf(d2_l) - stats.norm.cdf(d2_u)
    expected = (
        product.participation_rate * option_leg
        + product.knock_out_rebate * discount * (1.0 - survival)
        + product.no_hit_rebate * discount * survival
    ) * product.contract_multiplier

    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_expiry_put_matches_truncated_expectation():
    env = _pricing_env()
    product = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        lower_barrier=75.0,
        upper_barrier=125.0,
        maturity=1.0,
        participation_rate=0.6,
        knock_out_rebate=1.5,
        no_hit_rebate=0.25,
        observation_type=ObservationType.EXPIRY,
    )
    engine = DoubleSharkfinOptionAnalyticalEngine()

    price = engine.price(product, env)

    spot = env.spot
    maturity = product.maturity
    rate = env.get_rate(maturity)
    div = env.get_div_yield(maturity)
    vol = env.get_vol(product.strike, maturity)
    discount = math.exp(-rate * maturity)

    d1_l, d2_l = _d1_d2(spot, product.lower_barrier, maturity, rate, div, vol)
    d1_k, d2_k = _d1_d2(spot, product.strike, maturity, rate, div, vol)
    _, d2_u = _d1_d2(spot, product.upper_barrier, maturity, rate, div, vol)

    option_leg = product.strike * discount * (
        stats.norm.cdf(d2_l) - stats.norm.cdf(d2_k)
    ) - spot * math.exp(-div * maturity) * (
        stats.norm.cdf(d1_l) - stats.norm.cdf(d1_k)
    )
    survival = stats.norm.cdf(d2_l) - stats.norm.cdf(d2_u)
    expected = (
        product.participation_rate * option_leg
        + product.knock_out_rebate * discount * (1.0 - survival)
        + product.no_hit_rebate * discount * survival
    )

    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_continuous_no_rebate_matches_double_barrier_option_leg():
    env = _pricing_env(vol=0.2)
    product = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=70.0,
        upper_barrier=130.0,
        maturity=0.75,
        participation_rate=0.8,
        observation_type=ObservationType.CONTINUOUS,
    )
    sharkfin_engine = DoubleSharkfinOptionAnalyticalEngine()
    barrier_engine = DoubleBarrierOptionAnalyticalEngine()
    barrier_option = DoubleBarrierOption(
        strike=product.strike,
        option_type=product.option_type,
        lower_barrier=product.lower_barrier,
        upper_barrier=product.upper_barrier,
        barrier_type=DoubleBarrierType.KNOCK_OUT,
        maturity=product.maturity,
        observation_type=ObservationType.CONTINUOUS,
    )

    assert sharkfin_engine.price(product, env) == pytest.approx(
        product.participation_rate * barrier_engine.price(barrier_option, env),
        rel=1e-10,
        abs=1e-10,
    )


def test_continuous_pay_at_hit_rebate_is_worth_more_than_pay_at_expiry():
    env = _pricing_env(rate=0.05)
    at_expiry = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=80.0,
        upper_barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=False,
    )
    at_hit = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=80.0,
        upper_barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=True,
    )
    engine = DoubleSharkfinOptionAnalyticalEngine()

    assert engine.price(at_hit, env) > engine.price(at_expiry, env)


def test_discrete_daily_uses_bgk_shift_for_option_leg():
    env = _pricing_env(vol=0.2)
    discrete = DoubleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        lower_barrier=80.0,
        upper_barrier=120.0,
        maturity=1.0,
        observation_type=ObservationType.DISCRETE,
        observation_frequency=ObservationFrequency.DAILY,
    )
    vol = env.get_vol(discrete.strike, discrete.maturity)
    shifted = DoubleSharkfinOption(
        strike=discrete.strike,
        option_type=discrete.option_type,
        lower_barrier=apply_barrier_shift(
            discrete.lower_barrier,
            is_up_barrier=False,
            volatility=vol,
            observation_interval=1.0 / discrete.business_days_in_year,
        ),
        upper_barrier=apply_barrier_shift(
            discrete.upper_barrier,
            is_up_barrier=True,
            volatility=vol,
            observation_interval=1.0 / discrete.business_days_in_year,
        ),
        maturity=discrete.maturity,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = DoubleSharkfinOptionAnalyticalEngine()

    assert engine.price(discrete, env) == pytest.approx(
        engine.price(shifted, env), rel=1e-10, abs=1e-10
    )


def test_rejects_non_double_sharkfin_product():
    engine = DoubleSharkfinOptionAnalyticalEngine()
    with pytest.raises(PricingError, match="only supports DoubleSharkfinOption"):
        engine.price(
            DoubleBarrierOption(
                strike=100.0,
                option_type=OptionType.CALL,
                lower_barrier=80.0,
                upper_barrier=120.0,
                barrier_type=DoubleBarrierType.KNOCK_OUT,
                maturity=1.0,
            ),
            _pricing_env(),
        )
