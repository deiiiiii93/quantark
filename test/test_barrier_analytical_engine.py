import math
from datetime import datetime

import pytest
from scipy import stats

from asset.equity.engine.analytical import (
    BarrierAnalyticalEngine,
    BlackScholesEngine,
    OneTouchAnalyticalEngine,
)
from asset.equity.engine.pde_engine import PDEEngine
from asset.equity.product.option import (
    BarrierOption,
    EuropeanVanillaOption,
    OneTouchOption,
)
from param.quote.spot_quote import SpotQuote
from param.rrf.rate_curve import FlatRateCurve
from param.vol.vol_surface import FlatVolSurface
from priceenv import PricingEnvironment
from util.barrier_shift import apply_barrier_shift
from util.enum import (
    BarrierDirection,
    BarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from util.exceptions import PricingError


def _pricing_env(spot: float = 100.0) -> PricingEnvironment:
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.01),
        valuation_date=datetime(2020, 1, 1),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(0.2),
    )


def test_observation_type_includes_expiry():
    assert ObservationType.EXPIRY in ObservationType


def test_apply_barrier_shift():
    barrier = 100.0
    vol = 0.2
    dt = 1.0 / 252
    expected_up = barrier * math.exp(0.5825971579 * vol * math.sqrt(dt))
    expected_down = barrier * math.exp(-0.5825971579 * vol * math.sqrt(dt))
    assert math.isclose(
        apply_barrier_shift(barrier, True, vol, dt),
        expected_up,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    assert math.isclose(
        apply_barrier_shift(barrier, False, vol, dt),
        expected_down,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_barrier_analytical_engine_expiry_up_out_call_with_rebate():
    env = _pricing_env()
    T = 1.0
    option = BarrierOption(
        strike=90.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=T,
        rebate=2.0,
        observation_type=ObservationType.EXPIRY,
    )
    engine = BarrierAnalyticalEngine()
    price = engine.price(option, env)

    # Manual decomposition for expiry-only barrier
    bs_engine = BlackScholesEngine()
    vanilla = EuropeanVanillaOption(
        strike=option.strike,
        option_type=option.option_type,
        maturity=T,
    )
    vanilla_price = bs_engine.price(vanilla, env)

    S = env.spot
    r = env.get_rate(T)
    q = env.get_div_yield(T)
    sigma = env.get_vol(option.strike, T)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / option.barrier) + (r - q + 0.5 * sigma * sigma) * T) / (
        sigma * sqrt_t
    )
    d2 = d1 - sigma * sqrt_t
    prob_up = stats.norm.cdf(d2)
    asset_up = S * math.exp(-q * T) * stats.norm.cdf(d1)
    discount = math.exp(-r * T)
    portion_above = asset_up - option.strike * discount * prob_up
    ko_no_rebate = max(vanilla_price - portion_above, 0.0)
    expected = ko_no_rebate + option.rebate * discount * prob_up

    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_barrier_analytical_engine_rejects_rebate_continuous():
    env = _pricing_env()
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=1.5,
        pay_at_hit=False,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = BarrierAnalyticalEngine()
    price_with_rebate = engine.price(option, env)

    # Knock-out price without rebate
    option_no_rebate = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
    )
    ko_no_rebate = engine.price(option_no_rebate, env)

    ot_engine = OneTouchAnalyticalEngine()
    rebate_leg = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=1.5,
        payment_at_hit=False,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
    )
    rebate_price = ot_engine.price(rebate_leg, env)

    expected = ko_no_rebate + rebate_price
    assert price_with_rebate == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_barrier_analytical_engine_knock_in_rebate_not_hit():
    env = _pricing_env()
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_IN,
        maturity=1.0,
        rebate=2.0,
        pay_at_hit=False,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = BarrierAnalyticalEngine()
    price_with_rebate = engine.price(option, env)

    # Vanilla price
    bs = BlackScholesEngine()
    vanilla = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    vanilla_price = bs.price(vanilla, env)

    # Knock-out leg without rebate
    ko_option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
    )
    ko_no_rebate = engine.price(ko_option, env)

    # No-touch rebate leg (pays if barrier never hit)
    ot_engine = OneTouchAnalyticalEngine()
    no_touch = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=2.0,
        payment_at_hit=False,
        touch_type=TouchType.NO_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
    )
    rebate_price = ot_engine.price(no_touch, env)

    expected = max(vanilla_price - max(ko_no_rebate, 0.0), 0.0) + rebate_price
    assert price_with_rebate == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_participation_scales_payoff_not_rebate():
    env = _pricing_env()
    engine = BarrierAnalyticalEngine()

    base_option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.8,
        pay_at_hit=False,
        observation_type=ObservationType.CONTINUOUS,
    )
    price_base = engine.price(base_option, env)

    scaled_option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.8,
        pay_at_hit=False,
        observation_type=ObservationType.CONTINUOUS,
        participation_rate=0.4,
    )
    price_scaled = engine.price(scaled_option, env)

    # Extract rebate leg (not scaled)
    ot_engine = OneTouchAnalyticalEngine()
    rebate_leg = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=0.8,
        payment_at_hit=False,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
    )
    rebate_price = ot_engine.price(rebate_leg, env)

    payoff_base = price_base - rebate_price
    expected_scaled = rebate_price + 0.4 * payoff_base
    assert price_scaled == pytest.approx(expected_scaled, rel=1e-10, abs=1e-10)


def test_barrier_pde_engine_rejects_expiry_observation():
    env = _pricing_env()
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.EXPIRY,
    )
    engine = PDEEngine()
    with pytest.raises(PricingError):
        engine.price(option, env)
