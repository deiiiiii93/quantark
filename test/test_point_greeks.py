from datetime import datetime

import numpy as np
import pytest

from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from param import FlatRateCurve, FlatVolSurface, SpotQuote
from param.div import ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import EquityGreek, OptionType


def _build_env(div_yield: float = 0.01, vol: float = 0.2) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def _build_product() -> EuropeanVanillaOption:
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )


def test_point_greeks_accepts_enums():
    product = _build_product()
    env = _build_env()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    greeks = calc.calculate(
        product,
        env,
        engine,
        greeks=[EquityGreek.DELTA_Q, EquityGreek.VANNA, EquityGreek.VOLGA],
    )

    assert set(greeks.keys()) == {"delta_q", "vanna", "volga"}
    assert all(np.isfinite(value) for value in greeks.values())


def test_delta_q_matches_manual_diff():
    product = _build_product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    base_env = _build_env(div_yield=0.01)
    div_bump = calc._bump_config.div_bump
    spot_bump = calc._bump_config.spot_bump

    env_up = _build_env(div_yield=0.01 + div_bump)
    env_down = _build_env(div_yield=0.01 - div_bump)

    delta_up = calc.calculate_numerical_delta(
        product, env_up, engine, bump=spot_bump
    )
    delta_down = calc.calculate_numerical_delta(
        product, env_down, engine, bump=spot_bump
    )
    expected = (delta_up - delta_down) / (2.0 * div_bump)

    delta_q = calc.calculate_numerical_delta_q(
        product, base_env, engine, div_bump=div_bump
    )

    assert delta_q == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_vanna_and_volga_match_manual_diffs():
    product = _build_product()
    engine = BlackScholesEngine()
    calc = GreeksCalculator()

    base_env = _build_env(vol=0.2)
    vol_bump = calc._bump_config.vol_bump
    spot_bump = calc._bump_config.spot_bump

    env_up = _build_env(vol=0.2 + vol_bump)
    env_down = _build_env(vol=0.2 - vol_bump)

    delta_up = calc.calculate_numerical_delta(
        product, env_up, engine, bump=spot_bump
    )
    delta_down = calc.calculate_numerical_delta(
        product, env_down, engine, bump=spot_bump
    )
    expected_vanna = (delta_up - delta_down) / (2.0 * vol_bump)
    vanna = calc.calculate_numerical_vanna(
        product, base_env, engine, vol_bump=vol_bump
    )

    price_base = engine.price(product, base_env)
    price_up = engine.price(product, env_up)
    price_down = engine.price(product, env_down)
    expected_volga = (price_up - 2.0 * price_base + price_down) / (vol_bump**2)
    volga = calc.calculate_numerical_volga(
        product, base_env, engine, vol_bump=vol_bump
    )

    assert vanna == pytest.approx(expected_vanna, rel=1e-6, abs=1e-8)
    assert volga == pytest.approx(expected_volga, rel=1e-6, abs=1e-8)
