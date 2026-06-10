"""
Unit tests for one-touch/no-touch analytical engine.
"""

import math
from datetime import datetime

import pytest
from scipy import stats

from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine, OneTouchAnalyticalEngine
from quantark.asset.equity.product.option import CashOrNothingDigitalOption, OneTouchOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import BarrierDirection, ObservationType, OptionType, TouchType


def _pricing_env(spot: float = 100.0, vol: float = 0.2, rate: float = 0.03, div: float = 0.01):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _instant_touch_price(spot: float, barrier: float, maturity: float, rate: float, div: float, vol: float, is_up: bool, rebate: float) -> float:
    b = rate - div
    mu = (b - 0.5 * vol * vol) / (vol * vol)
    lam = math.sqrt(mu * mu + 2.0 * rate / (vol * vol))
    sqrt_t = math.sqrt(maturity)
    z = math.log(barrier / spot) / (vol * sqrt_t) + lam * vol * sqrt_t
    eta = -1.0 if is_up else 1.0
    term1 = math.pow(barrier / spot, mu + lam) * stats.norm.cdf(eta * z)
    term2 = math.pow(barrier / spot, mu - lam) * stats.norm.cdf(
        eta * z - 2 * eta * lam * vol * sqrt_t
    )
    return rebate * (term1 + term2)


def _expiry_touch_price(spot: float, barrier: float, maturity: float, rate: float, div: float, vol: float, is_up: bool, rebate: float) -> float:
    b = rate - div
    mu = (b - 0.5 * vol * vol) / (vol * vol)
    sqrt_t = math.sqrt(maturity)
    log_s_b = math.log(spot / barrier)
    x2 = log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
    y2 = -log_s_b / (vol * sqrt_t) + (1 + mu) * vol * sqrt_t
    phi = 1.0 if is_up else -1.0
    eta = -1.0 if is_up else 1.0
    pow_term = math.pow(barrier / spot, 2 * mu)
    term = stats.norm.cdf(phi * x2 - phi * vol * sqrt_t) + pow_term * stats.norm.cdf(
        eta * y2 - eta * vol * sqrt_t
    )
    return rebate * math.exp(-rate * maturity) * term


def test_continuous_one_touch_pay_at_hit_up_barrier():
    env = _pricing_env()
    option = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=5.0,
        payment_at_hit=True,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = OneTouchAnalyticalEngine()
    price = engine.price(option, env)

    expected = _instant_touch_price(
        spot=env.spot,
        barrier=option.barrier,
        maturity=option.get_maturity(env),
        rate=env.get_rate(option.maturity),
        div=env.get_div_yield(option.maturity),
        vol=env.get_vol(option.barrier, option.maturity),
        is_up=True,
        rebate=option.rebate,
    )
    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_continuous_one_touch_pay_at_expiry_down_barrier():
    env = _pricing_env()
    option = OneTouchOption(
        barrier=90.0,
        barrier_direction=BarrierDirection.DOWN,
        maturity=1.0,
        rebate=7.5,
        payment_at_hit=False,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = OneTouchAnalyticalEngine()
    price = engine.price(option, env)

    expected = _expiry_touch_price(
        spot=env.spot,
        barrier=option.barrier,
        maturity=option.get_maturity(env),
        rate=env.get_rate(option.maturity),
        div=env.get_div_yield(option.maturity),
        vol=env.get_vol(option.barrier, option.maturity),
        is_up=False,
        rebate=option.rebate,
    )
    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_no_touch_ignores_payment_at_hit():
    env = _pricing_env()
    option = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=4.0,
        payment_at_hit=True,  # should be ignored for NO_TOUCH
        touch_type=TouchType.NO_TOUCH,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = OneTouchAnalyticalEngine()
    price = engine.price(option, env)

    T = option.get_maturity(env)
    r = env.get_rate(T)
    q = env.get_div_yield(T)
    sigma = env.get_vol(option.barrier, T)
    term = _expiry_touch_price(
        spot=env.spot,
        barrier=option.barrier,
        maturity=T,
        rate=r,
        div=q,
        vol=sigma,
        is_up=True,
        rebate=1.0,
    )
    prob_touch = term / math.exp(-r * T)
    expected = option.rebate * math.exp(-r * T) * (1.0 - prob_touch)
    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_discrete_monitoring_applies_barrier_shift():
    env = _pricing_env()
    obs_dates = [0.25, 0.5, 0.75, 1.0]
    option = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=6.0,
        payment_at_hit=False,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.DISCRETE,
        observation_dates=obs_dates,
    )
    engine = OneTouchAnalyticalEngine()
    price = engine.price(option, env)

    T = option.get_maturity(env)
    r = env.get_rate(T)
    q = env.get_div_yield(T)
    sigma = env.get_vol(option.barrier, T)
    freq = obs_dates[1] - obs_dates[0]
    shifted_barrier = apply_barrier_shift(
        barrier=option.barrier, is_up_barrier=True, volatility=sigma, observation_interval=freq
    )
    expected = _expiry_touch_price(
        spot=env.spot,
        barrier=shifted_barrier,
        maturity=T,
        rate=r,
        div=q,
        vol=sigma,
        is_up=True,
        rebate=option.rebate,
    )
    assert price == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_expiry_monitoring_matches_digital_pricing():
    env = _pricing_env()
    option = OneTouchOption(
        barrier=105.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=8.0,
        payment_at_hit=True,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.EXPIRY,
    )
    engine = OneTouchAnalyticalEngine()
    price = engine.price(option, env)

    # Digital call fallback
    digital = CashOrNothingDigitalOption(
        strike=option.barrier,
        payout=option.rebate,
        option_type=OptionType.CALL,
        maturity=option.maturity,
    )
    digital_engine = DigitalOptionAnalyticalEngine()
    expected = digital_engine.price(digital, env)

    assert price == pytest.approx(expected, rel=1e-12, abs=1e-12)

