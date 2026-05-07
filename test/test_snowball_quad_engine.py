"""
Unit tests for SnowballQuadEngine.

Focus on direct regime-switching quadrature behavior for discrete KO with
discrete/continuous KI monitoring.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import QuadParams
from asset.equity.product.option.snowball_config import (
    AccrualConfig,
    AirbagConfig,
    BarrierConfig,
    PayoffConfig,
)
from asset.equity.product.option.snowball_option import SnowballOption
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import ObservationType


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div_yield: float = 0.02,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def create_barrier_config(
    ko_barrier: float,
    ki_barrier: float,
    ki_continuous: bool,
    ko_observation_dates: list[float] = None,
    ki_observation_dates: list[float] = None,
    disable_ko_after_ki: bool = False,
) -> BarrierConfig:
    if ko_observation_dates is None:
        ko_observation_dates = [0.25, 0.5, 0.75, 1.0]
    return BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_observation_dates,
        ki_barrier=ki_barrier,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_observation_dates=ki_observation_dates,
        ki_continuous=ki_continuous,
        disable_ko_after_ki=disable_ko_after_ki,
    )


def create_standard_snowball(barrier_config: BarrierConfig) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )


def create_reverse_snowball(barrier_config: BarrierConfig) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=True,
    )


def test_standard_snowball_quad_price_positive():
    env = create_pricing_env()
    barrier_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=75.0, ki_continuous=True
    )
    snowball = create_standard_snowball(barrier_config)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price = engine.price(snowball, env)
    assert np.isfinite(price)
    assert price > 0.0


def test_discrete_ki_vs_continuous_ki():
    env = create_pricing_env()
    barrier_cont = create_barrier_config(
        ko_barrier=103.0, ki_barrier=75.0, ki_continuous=True
    )
    barrier_disc = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=75.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
    )
    snowball_cont = create_standard_snowball(barrier_cont)
    snowball_disc = create_standard_snowball(barrier_disc)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_cont = engine.price(snowball_cont, env)
    price_disc = engine.price(snowball_disc, env)
    assert price_cont <= price_disc + 1e-6


def test_reverse_snowball_quad_price_positive():
    env = create_pricing_env()
    barrier_config = create_barrier_config(
        ko_barrier=97.0, ki_barrier=125.0, ki_continuous=True
    )
    snowball = create_reverse_snowball(barrier_config)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price = engine.price(snowball, env)
    assert np.isfinite(price)
    assert price > 0.0


def test_airbag_snowball_prices_higher_than_standard():
    env = create_pricing_env(spot=95.0, vol=0.30)
    barrier_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=85.0, ki_continuous=True
    )
    airbag_config = AirbagConfig(
        airbag_barrier=80.0,
        airbag_participation_rate=0.5,
        airbag_strike=90.0,
    )
    standard = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    airbag = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        airbag_config=airbag_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_standard = engine.price(standard, env)
    price_airbag = engine.price(airbag, env)
    assert price_airbag > price_standard or price_airbag == pytest.approx(
        price_standard, abs=1e-6
    )


def test_call_rebate_v0_supported():
    env = create_pricing_env()
    barrier_config = create_barrier_config(
        ko_barrier=103.0, ki_barrier=75.0, ki_continuous=True
    )
    no_rebate = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    call_rebate = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(
            rebate_rate=0.0,
            include_principal=True,
            call_rebate_enabled=True,
            call_strike=90.0,
            call_participation_rate=0.5,
        ),
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_no_rebate = engine.price(no_rebate, env)
    price_call_rebate = engine.price(call_rebate, env)
    assert price_call_rebate > price_no_rebate or price_call_rebate == pytest.approx(
        price_no_rebate, abs=1e-6
    )


def test_disable_ko_after_ki_reduces_value():
    env = create_pricing_env(vol=0.30)
    barrier_config = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=85.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
        disable_ko_after_ki=True,
    )
    barrier_config_enabled = create_barrier_config(
        ko_barrier=103.0,
        ki_barrier=85.0,
        ki_continuous=False,
        ki_observation_dates=[0.5, 1.0],
        disable_ko_after_ki=False,
    )
    snowball_disabled = create_standard_snowball(barrier_config)
    snowball_enabled = create_standard_snowball(barrier_config_enabled)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    price_disabled = engine.price(snowball_disabled, env)
    price_enabled = engine.price(snowball_enabled, env)
    assert price_disabled < price_enabled or price_disabled == pytest.approx(
        price_enabled, abs=1e-6
    )


def test_quad_applies_immediate_ko_at_valuation_observation():
    env = create_pricing_env(spot=150.0)
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.0, 0.5, 1.0],
        ki_barrier=None,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(is_annualized=False),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))

    price = engine.price(product, env)
    ko_record_0 = product.resolve_ko_observations(env)[0]

    assert price == pytest.approx(ko_record_0.payoff, abs=1e-10)

    stats = engine.calculate_event_stats(product, env)
    assert stats is not None
    assert stats.pv == pytest.approx(ko_record_0.payoff, abs=1e-10)
    assert stats.ko_times[0] == pytest.approx(0.0)
    assert stats.ko_probability[0] == pytest.approx(1.0)
    assert stats.expected_discounted_maturity_cashflow == pytest.approx(0.0)


def test_quad_applies_immediate_ki_at_valuation_observation():
    env = create_pricing_env(spot=70.0)
    barrier_config = BarrierConfig(
        ko_barrier=150.0,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[0.0, 0.5, 1.0],
        ki_continuous=False,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(is_annualized=False),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    lifecycle_product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(include_principal=True),
        accrual_config=AccrualConfig(is_annualized=False),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    setattr(lifecycle_product, "_otc_lifecycle_knocked_in", True)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))

    price = engine.price(product, env)
    lifecycle_price = engine.price(lifecycle_product, env)

    assert price == pytest.approx(lifecycle_price, abs=1e-10)

    stats = engine.calculate_event_stats(product, env)
    assert stats is not None
    assert stats.ki_probability == pytest.approx(1.0)
    assert stats.ki_times[0] == pytest.approx(0.0)
