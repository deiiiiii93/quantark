"""
Unit tests for BarrierQuadEngine.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
from asset.equity.engine.quad import BarrierQuadEngine
from asset.equity.param import MCParams, QuadParams
from asset.equity.product.option import BarrierOption
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.03,
    div: float = 0.01,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def build_schedule(
    times=(0.25, 0.5, 0.75, 1.0), barrier=110.0, payoff=0.0
) -> ObservationSchedule:
    return ObservationSchedule(
        records=[
            ObservationRecord(observation_time=float(t), barrier=barrier, payoff=payoff)
            for t in times
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )


@pytest.mark.parametrize(
    "barrier_type, barrier",
    [
        (BarrierType.UP_OUT, 120.0),
        (BarrierType.DOWN_OUT, 80.0),
        (BarrierType.UP_IN, 120.0),
        (BarrierType.DOWN_IN, 80.0),
    ],
)
def test_barrier_quad_matches_mc_discrete(barrier_type, barrier):
    env = create_pricing_env()
    schedule = build_schedule(barrier=barrier, payoff=0.0)
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=barrier,
        barrier_type=barrier_type,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )

    quad_engine = BarrierQuadEngine(params=QuadParams(grid_points=801))
    mc_engine = BarrierOptionMCEngine(
        params=MCParams(num_paths=8000, time_steps=252, seed=11)
    )

    quad_price = quad_engine.price(option, env)
    mc_price = mc_engine.price(option, env)

    assert quad_price == pytest.approx(mc_price, rel=0.10, abs=0.3)


def test_discrete_schedule_matches_legacy_dates():
    env = create_pricing_env()
    dates = [0.25, 0.5, 0.75, 1.0]
    schedule = build_schedule(times=dates, barrier=110.0, payoff=0.0)

    option_schedule = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    option_legacy = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_dates=dates,
    )

    engine = BarrierQuadEngine(params=QuadParams(grid_points=801))
    price_schedule = engine.price(option_schedule, env)
    price_legacy = engine.price(option_legacy, env)

    assert price_schedule == pytest.approx(price_legacy, rel=1e-10, abs=1e-10)


def test_rebate_timing_immediate_knock_out():
    env = create_pricing_env(spot=120.0)
    schedule = build_schedule(barrier=110.0, payoff=5.0)
    engine = BarrierQuadEngine(params=QuadParams(grid_points=601))

    option_hit = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=5.0,
        pay_at_hit=True,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    price_hit = engine.price(option_hit, env)
    assert price_hit == pytest.approx(option_hit.rebate, rel=1e-10, abs=1e-10)

    option_expiry = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=5.0,
        pay_at_hit=False,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    price_expiry = engine.price(option_expiry, env)
    expected = option_expiry.rebate * math.exp(-env.get_rate(1.0) * 1.0)
    assert price_expiry == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_convergence_with_grid_size():
    env = create_pricing_env()
    schedule = build_schedule(barrier=110.0, payoff=0.0)
    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )

    engine_low = BarrierQuadEngine(params=QuadParams(grid_points=401))
    engine_mid = BarrierQuadEngine(params=QuadParams(grid_points=801))
    engine_high = BarrierQuadEngine(params=QuadParams(grid_points=1601))

    price_low = engine_low.price(option, env)
    price_mid = engine_mid.price(option, env)
    price_high = engine_high.price(option, env)

    error_low = abs(price_low - price_high)
    error_mid = abs(price_mid - price_high)

    assert error_mid <= error_low + 0.05


def test_barrier_at_spot_knock_out_rebate():
    env = create_pricing_env(spot=100.0)
    schedule = build_schedule(barrier=100.0, payoff=2.0)
    option = BarrierOption(
        strike=95.0,
        option_type=OptionType.CALL,
        barrier=100.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=2.0,
        pay_at_hit=True,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )
    engine = BarrierQuadEngine(params=QuadParams(grid_points=601))
    price = engine.price(option, env)

    assert price == pytest.approx(option.rebate, rel=1e-10, abs=1e-10)


def test_near_expiry_returns_intrinsic():
    env = create_pricing_env()
    option = BarrierOption(
        strike=90.0,
        option_type=OptionType.CALL,
        barrier=150.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1e-12,
        rebate=0.0,
        observation_type=ObservationType.DISCRETE,
        observation_dates=[1e-12],
    )
    engine = BarrierQuadEngine(params=QuadParams(grid_points=401))
    price = engine.price(option, env)

    assert price == pytest.approx(10.0, rel=1e-10, abs=1e-10)
