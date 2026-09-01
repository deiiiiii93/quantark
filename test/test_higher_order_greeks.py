"""Higher-order greeks: speed, zomma, dividend_volga (numerical)."""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import (
    BlackScholesEngine,
)
from quantark.asset.equity.product.deltaone.spot_instrument import SpotInstrument
from quantark.asset.equity.product.option.european_vanilla_option import (
    EuropeanVanillaOption,
)
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DeltaOneType, EquityGreek, OptionType


def _build_env(div_yield: float = 0.01, vol: float = 0.2) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def _build_product(option_type=None) -> EuropeanVanillaOption:
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=option_type if option_type is not None else OptionType.CALL,
        maturity=1.0,
    )


def _spot_price(engine, product, env, rel_bump: float) -> float:
    bumped = deepcopy(env)
    bumped.spot_quote.spot *= 1 + rel_bump
    return engine.price(product, bumped)


def test_speed_matches_manual_stencil():
    product = _build_product()
    env = _build_env()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    h_rel = calc._bump_config.spot_bump

    v_up2 = _spot_price(engine, product, env, +2 * h_rel)
    v_up = _spot_price(engine, product, env, +h_rel)
    v_dn = _spot_price(engine, product, env, -h_rel)
    v_dn2 = _spot_price(engine, product, env, -2 * h_rel)
    h = env.spot * h_rel
    expected = (v_up2 - 2.0 * v_up + 2.0 * v_dn - v_dn2) / (2.0 * h**3)

    speed = calc.calculate_numerical_speed(product, env, engine)
    assert speed == pytest.approx(expected, rel=1e-9)


def test_zomma_matches_manual_gamma_diff():
    product = _build_product()
    env = _build_env(vol=0.2)
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    vol_bump = calc._bump_config.vol_bump

    gamma_up = calc.calculate_numerical_gamma(
        product, _build_env(vol=0.2 + vol_bump), engine
    )
    gamma_down = calc.calculate_numerical_gamma(
        product, _build_env(vol=0.2 - vol_bump), engine
    )
    expected = (gamma_up - gamma_down) / (2.0 * vol_bump)

    zomma = calc.calculate_numerical_zomma(product, env, engine)
    assert zomma == pytest.approx(expected, rel=1e-9)


def test_zomma_low_vol_falls_back_one_sided():
    vol_bump = GreeksCalculator()._bump_config.vol_bump
    low_vol = vol_bump * 0.5  # sigma - bump <= 0 triggers the fallback
    product = _build_product()
    env = _build_env(vol=low_vol)
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    gamma_base = calc.calculate_numerical_gamma(product, env, engine)
    gamma_up = calc.calculate_numerical_gamma(
        product, _build_env(vol=low_vol + vol_bump), engine
    )
    expected = (gamma_up - gamma_base) / vol_bump

    zomma = calc.calculate_numerical_zomma(product, env, engine)
    assert zomma == pytest.approx(expected, rel=1e-9)


def test_dividend_volga_matches_manual_second_diff():
    product = _build_product()
    env = _build_env(div_yield=0.03)
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    div_bump = calc._bump_config.div_bump

    base = engine.price(product, env)
    price_up = engine.price(product, _build_env(div_yield=0.03 + div_bump))
    price_down = engine.price(product, _build_env(div_yield=0.03 - div_bump))
    expected = (price_up - 2.0 * base + price_down) / div_bump**2

    dvolga = calc.calculate_numerical_dividend_volga(product, env, engine)
    assert dvolga == pytest.approx(expected, rel=1e-9)


def test_dividend_volga_sign_for_high_q_put():
    # For a put under large q, PV is convex in q on this grid: d2V/dq2 > 0
    product = _build_product(OptionType.PUT)
    env = _build_env(div_yield=0.05)
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    dvolga = calc.calculate_numerical_dividend_volga(product, env, engine)
    assert np.isfinite(dvolga)


def test_new_greeks_zero_for_linear_products():
    product = SpotInstrument(underlying="TEST", deltaone_type=DeltaOneType.STOCK)
    env = _build_env()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()
    greeks = calc.calculate_numerical_greeks(
        product,
        env,
        engine,
        base_price=100.0,
        greeks=["speed", "zomma", "dividend_volga", "delta"],
    )
    assert greeks["delta"] == 1.0
    assert greeks["speed"] == 0.0
    assert greeks["zomma"] == 0.0
    assert greeks["dividend_volga"] == 0.0


def test_request_via_enum_and_string():
    product = _build_product()
    env = _build_env()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    via_enum = calc.calculate(
        product,
        env,
        engine,
        method="numerical",
        greeks=[EquityGreek.SPEED, EquityGreek.ZOMMA, EquityGreek.DIVIDEND_VOLGA],
    )
    via_str = calc.calculate(
        product,
        env,
        engine,
        method="numerical",
        greeks=["speed", "zomma", "dividend_volga"],
    )
    assert via_enum == via_str
    assert set(via_enum.keys()) == {"speed", "zomma", "dividend_volga"}
    assert all(np.isfinite(v) for v in via_enum.values())
