"""
Unit tests for SnowballQuadEngine.

Focus on direct regime-switching quadrature behavior for discrete KO with
discrete/continuous KI monitoring.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import QuadParams
from asset.equity.product.option.snowball_config import BarrierConfig
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
