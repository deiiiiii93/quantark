"""
Tests for the AccumulatorAnalyticalEngine.

The analytical engine statically replicates the accumulator as a strip of
up-and-out call/put legs across observation dates. The strongest exact check
(no Monte Carlo needed) is the degenerate no-barrier case: with the knock-out
barrier pushed far away, each daily leg collapses to a plain vanilla
``daily_shares * (call - gearing * put)``.
"""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import (
    AccumulatorAnalyticalEngine,
    BlackScholesEngine,
)
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    EuropeanVanillaOption,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AccumulatorKnockOutType, OptionType
from quantark.util.exceptions import PricingError


def _pricing_env(spot=100.0, rate=0.03, div=0.01, vol=0.25) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        rate_curve=FlatRateCurve(rate=rate),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _vanilla_strip_value(env, strike, obs_times, daily, gearing):
    """Reference: sum of daily_shares * (call - gearing * put) at each obs date."""
    bs = BlackScholesEngine()
    total = 0.0
    for t in obs_times:
        call = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.CALL, maturity=t
        )
        put = EuropeanVanillaOption(
            strike=strike, option_type=OptionType.PUT, maturity=t
        )
        total += daily * (bs.price(call, env) - gearing * bs.price(put, env))
    return total


def test_rejects_wrong_product_type():
    engine = AccumulatorAnalyticalEngine()
    bad = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    with pytest.raises(PricingError, match="AccumulatorOption"):
        engine.price(bad, _pricing_env())


def test_no_barrier_degenerate_matches_vanilla_strip():
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    acc = AccumulatorOption(
        strike=95.0,
        knock_out_barrier=1.0e6,  # effectively no knock-out
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.TERMINATION,
        observation_dates=obs,
    )
    price = AccumulatorAnalyticalEngine().price(acc, env)
    expected = _vanilla_strip_value(env, 95.0, obs, daily=1.0, gearing=2.0)
    assert price == pytest.approx(expected, rel=1e-4)


def test_contract_multiplier_scales_price():
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    common = dict(
        strike=95.0,
        knock_out_barrier=108.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        observation_dates=obs,
    )
    engine = AccumulatorAnalyticalEngine()
    base = engine.price(AccumulatorOption(**common, contract_multiplier=1.0), env)
    scaled = engine.price(AccumulatorOption(**common, contract_multiplier=10.0), env)
    assert scaled == pytest.approx(10.0 * base, rel=1e-10)


def test_realized_accrual_adds_to_price():
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    common = dict(
        strike=95.0,
        knock_out_barrier=108.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        observation_dates=obs,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
    )
    engine = AccumulatorAnalyticalEngine()
    without = engine.price(AccumulatorOption(**common), env)
    # one realized gain observation (S=100 >= K=95): +5 locked in, settled instantly
    with_past = engine.price(
        AccumulatorOption(**common, past_observations=[(-0.1, 100.0)]), env
    )
    assert with_past - without == pytest.approx(5.0, abs=1e-9)
