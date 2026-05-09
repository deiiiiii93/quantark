import math
from datetime import datetime

import pytest
from scipy import stats

from asset.equity.engine.analytical import SingleSharkfinOptionAnalyticalEngine
from asset.equity.product.option import SingleSharkfinOption
from asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.barrier_shift import apply_barrier_shift
from util.enum import ObservationFrequency, ObservationType, OptionType
from util.exceptions import PricingError


def _pricing_env(
    spot: float = 100.0,
    rate: float = 0.03,
    div: float = 0.01,
    vol: float = 0.25,
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
    engine = SingleSharkfinOptionAnalyticalEngine()
    product = SingleSharkfinOption(
        strike=95.0,
        option_type=OptionType.CALL,
        barrier=115.0,
        maturity=1.0,
        participation_rate=0.6,
        knock_out_rebate=2.0,
        no_hit_rebate=1.5,
        observation_type=ObservationType.EXPIRY,
        contract_multiplier=10.0,
    )

    price = engine.price(product, env)

    spot = env.spot
    maturity = product.maturity
    rate = env.get_rate(maturity)
    div = env.get_div_yield(maturity)
    vol = env.get_vol(product.strike, maturity)
    discount = math.exp(-rate * maturity)

    d1_k, d2_k = _d1_d2(spot, product.strike, maturity, rate, div, vol)
    d1_b, d2_b = _d1_d2(spot, product.barrier, maturity, rate, div, vol)

    asset_between = spot * math.exp(-div * maturity) * (
        stats.norm.cdf(d1_k) - stats.norm.cdf(d1_b)
    )
    prob_between = stats.norm.cdf(d2_k) - stats.norm.cdf(d2_b)
    no_hit_option_leg = asset_between - product.strike * discount * prob_between
    hit_prob = stats.norm.cdf(d2_b)
    no_hit_prob = 1.0 - hit_prob

    expected = (
        product.participation_rate * no_hit_option_leg
        + product.knock_out_rebate * discount * hit_prob
        + product.no_hit_rebate * discount * no_hit_prob
    ) * product.contract_multiplier

    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_discrete_daily_call_uses_bgk_shift():
    env = _pricing_env(vol=0.2)
    engine = SingleSharkfinOptionAnalyticalEngine()
    discrete = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=1.0,
        knock_out_rebate=1.0,
        no_hit_rebate=0.5,
        observation_type=ObservationType.DISCRETE,
        observation_frequency=ObservationFrequency.DAILY,
    )

    shifted_barrier = apply_barrier_shift(
        barrier=discrete.barrier,
        is_up_barrier=True,
        volatility=env.get_vol(discrete.strike, discrete.maturity),
        observation_interval=1.0 / discrete.business_days_in_year,
    )
    shifted_continuous = SingleSharkfinOption(
        strike=discrete.strike,
        option_type=discrete.option_type,
        barrier=shifted_barrier,
        maturity=discrete.maturity,
        participation_rate=discrete.participation_rate,
        knock_out_rebate=discrete.knock_out_rebate,
        no_hit_rebate=discrete.no_hit_rebate,
        observation_type=ObservationType.CONTINUOUS,
    )

    assert engine.price(discrete, env) == pytest.approx(
        engine.price(shifted_continuous, env), rel=1e-10, abs=1e-10
    )


def test_continuous_put_scales_by_contract_multiplier():
    env = _pricing_env()
    engine = SingleSharkfinOptionAnalyticalEngine()
    base = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        barrier=80.0,
        maturity=1.0,
        participation_rate=0.8,
        knock_out_rebate=1.0,
        no_hit_rebate=0.25,
        observation_type=ObservationType.CONTINUOUS,
    )
    scaled = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.PUT,
        barrier=80.0,
        maturity=1.0,
        participation_rate=0.8,
        knock_out_rebate=1.0,
        no_hit_rebate=0.25,
        observation_type=ObservationType.CONTINUOUS,
        contract_multiplier=100.0,
    )

    assert engine.price(scaled, env) == pytest.approx(
        100.0 * engine.price(base, env), rel=1e-10, abs=1e-10
    )


def test_knock_out_rebate_can_pay_at_hit_or_expiry():
    env = _pricing_env(rate=0.05)
    engine = SingleSharkfinOptionAnalyticalEngine()
    at_expiry = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        no_hit_rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=False,
    )
    at_hit = SingleSharkfinOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        maturity=1.0,
        participation_rate=0.0,
        knock_out_rebate=10.0,
        no_hit_rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
        pay_at_hit=True,
    )

    assert engine.price(at_hit, env) > engine.price(at_expiry, env)


def test_rejects_non_sharkfin_product():
    engine = SingleSharkfinOptionAnalyticalEngine()
    with pytest.raises(PricingError, match="only supports SingleSharkfinOption"):
        engine.price(
            EuropeanVanillaOption(
                strike=100.0,
                option_type=OptionType.CALL,
                maturity=1.0,
            ),
            _pricing_env(),
        )
