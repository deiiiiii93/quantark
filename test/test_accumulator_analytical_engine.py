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
    BarrierAnalyticalEngine,
    BlackScholesEngine,
)
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    BarrierOption,
    EuropeanVanillaOption,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    AccumulatorKnockOutType,
    BarrierType,
    ObservationType,
    OptionType,
)
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


# ---------------------------------------------------------------------------
# Review findings (Gate 2 zenmux, iter 1)
# ---------------------------------------------------------------------------

def _single_day_strip_value(env, strike, barrier, obs_times, daily, gearing):
    """Reference SINGLE_DAY price: each call leg is a single-day (expiry) up-out
    call; each put leg is a plain vanilla put. Legs are independent across days."""
    bs = BlackScholesEngine()
    barrier_engine = BarrierAnalyticalEngine()
    total = 0.0
    for t in obs_times:
        call = BarrierOption(
            strike=strike,
            option_type=OptionType.CALL,
            barrier=barrier,
            barrier_type=BarrierType.UP_OUT,
            maturity=t,
            observation_type=ObservationType.EXPIRY,
            contract_multiplier=daily,
        )
        put = EuropeanVanillaOption(
            strike=strike,
            option_type=OptionType.PUT,
            maturity=t,
            contract_multiplier=daily * gearing,
        )
        total += barrier_engine.price(call, env) - bs.price(put, env)
    return total


def test_single_day_legs_are_independent_per_day():
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    acc = AccumulatorOption(
        strike=95.0,
        knock_out_barrier=108.0,  # active barrier so DISCRETE vs EXPIRY differ
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_dates=obs,
    )
    price = AccumulatorAnalyticalEngine().price(acc, env)
    expected = _single_day_strip_value(
        env, 95.0, 108.0, obs, daily=1.0, gearing=2.0
    )
    assert price == pytest.approx(expected, rel=1e-9)


def test_single_day_extra_shares_not_cumulatively_knocked_out():
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    strike, barrier, extra = 95.0, 108.0, 0.5
    common = dict(
        strike=strike,
        knock_out_barrier=barrier,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        gearing=2.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_dates=obs,
    )
    engine = AccumulatorAnalyticalEngine()
    base = engine.price(AccumulatorOption(**common), env)
    with_extra = engine.price(
        AccumulatorOption(**common, extra_shares_at_expiry=extra), env
    )

    # For SINGLE_DAY the extra-shares leg is checked only at expiry (no cumulative
    # knockout): an expiry-monitored up-and-out put, scaled by extra shares.
    barrier_engine = BarrierAnalyticalEngine()
    extra_put = BarrierOption(
        strike=strike,
        option_type=OptionType.PUT,
        barrier=barrier,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        observation_type=ObservationType.EXPIRY,
        contract_multiplier=extra,
    )
    expected_delta = -barrier_engine.price(extra_put, env)
    assert (with_extra - base) == pytest.approx(expected_delta, rel=1e-9)


def test_expiry_monitoring_legs_checked_at_leg_maturity_only():
    # observation_type=EXPIRY: each leg's barrier is checked only at its own
    # maturity, even for TERMINATION (no cumulative knockout across dates).
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    strike, barrier, daily, gearing = 95.0, 108.0, 1.0, 2.0
    acc = AccumulatorOption(
        strike=strike,
        knock_out_barrier=barrier,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=daily,
        gearing=gearing,
        knock_out_type=AccumulatorKnockOutType.TERMINATION,
        observation_type=ObservationType.EXPIRY,
        observation_dates=obs,
    )
    price = AccumulatorAnalyticalEngine().price(acc, env)

    barrier_engine = BarrierAnalyticalEngine()
    expected = 0.0
    for t in obs:
        call = BarrierOption(
            strike=strike, option_type=OptionType.CALL, barrier=barrier,
            barrier_type=BarrierType.UP_OUT, maturity=t,
            observation_type=ObservationType.EXPIRY, contract_multiplier=daily,
        )
        put = BarrierOption(
            strike=strike, option_type=OptionType.PUT, barrier=barrier,
            barrier_type=BarrierType.UP_OUT, maturity=t,
            observation_type=ObservationType.EXPIRY,
            contract_multiplier=daily * gearing,
        )
        expected += barrier_engine.price(call, env) - barrier_engine.price(put, env)
    assert price == pytest.approx(expected, rel=1e-9)


def test_extra_shares_leg_uses_contract_maturity_not_last_observation():
    # Observation schedule ends before contract maturity; the extra-shares leg
    # must mature at the contract maturity, not the last observation date.
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75]  # last obs (0.75) < maturity (1.0)
    strike, barrier, extra = 95.0, 108.0, 0.5
    common = dict(
        strike=strike,
        knock_out_barrier=barrier,
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=1.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_dates=obs,
    )
    engine = AccumulatorAnalyticalEngine()
    base = engine.price(AccumulatorOption(**common), env)
    with_extra = engine.price(
        AccumulatorOption(**common, extra_shares_at_expiry=extra), env
    )

    barrier_engine = BarrierAnalyticalEngine()
    extra_put = BarrierOption(
        strike=strike, option_type=OptionType.PUT, barrier=barrier,
        barrier_type=BarrierType.UP_OUT, maturity=1.0,  # contract maturity
        observation_type=ObservationType.EXPIRY, contract_multiplier=extra,
    )
    assert (with_extra - base) == pytest.approx(
        -barrier_engine.price(extra_put, env), rel=1e-9
    )


def test_single_day_price_ignores_rebate_rate():
    env = _pricing_env()
    obs = [0.25, 0.5, 0.75, 1.0]
    common = dict(
        strike=95.0,
        knock_out_barrier=108.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        initial_price=100.0,
        notional=1_000_000.0,
        knock_out_type=AccumulatorKnockOutType.SINGLE_DAY,
        observation_dates=obs,
    )
    engine = AccumulatorAnalyticalEngine()
    no_rebate = engine.price(AccumulatorOption(**common, knock_out_rebate_rate=0.0), env)
    with_rebate = engine.price(
        AccumulatorOption(**common, knock_out_rebate_rate=0.05), env
    )
    assert with_rebate == pytest.approx(no_rebate, rel=1e-12)


def test_settlement_at_expiry_uses_curve_discount_factors():
    # Non-flat curve: deferral factor must be DF(0,T)/DF(0,t_i), not exp(-r_T*(T-t_i)).
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        rate_curve=LinearRateCurve([(0.25, 0.02), (1.0, 0.06)]),
        vol_surface=FlatVolSurface(volatility=0.25),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )
    obs = [0.25, 0.5, 0.75, 1.0]
    daily, gearing, strike = 1.0, 2.0, 95.0
    acc = AccumulatorOption(
        strike=strike,
        knock_out_barrier=1.0e6,  # no barrier -> exact vanilla strip
        option_type=OptionType.CALL,
        maturity=1.0,
        daily_share_accumulation=daily,
        gearing=gearing,
        knock_out_type=AccumulatorKnockOutType.TERMINATION,
        observation_dates=obs,
        settlement_at_expiry=True,
    )
    price = AccumulatorAnalyticalEngine().price(acc, env)

    bs = BlackScholesEngine()
    df_T = env.get_discount_factor(1.0)
    expected = 0.0
    for t in obs:
        call = EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=t)
        put = EuropeanVanillaOption(strike=strike, option_type=OptionType.PUT, maturity=t)
        instant = daily * (bs.price(call, env) - gearing * bs.price(put, env))
        expected += instant * df_T / env.get_discount_factor(t)
    assert price == pytest.approx(expected, rel=1e-4)
