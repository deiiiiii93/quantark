"""
Unit tests for BarrierOptionMCEngine.
"""

import sys
from pathlib import Path
import math
from datetime import datetime

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationSchedule,
    ObservationRecord,
)
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import BarrierType, ObservationType, OptionType, ObservationAggregation
from quantark.util.enum.engine_enums import MonteCarloMethod


def _pricing_env(spot: float = 100.0) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(0.2),
        rate_curve=FlatRateCurve(0.01),
        div_yield=ContinuousDividendYield(0.0),
        valuation_date=datetime(2024, 1, 1),
    )


def test_barrier_mc_knock_in_out_decomposition():
    env = _pricing_env()
    params = MCParams(num_paths=8000, time_steps=252, seed=123)

    ko_option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
    )
    ki_option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=120.0,
        barrier_type=BarrierType.UP_IN,
        maturity=1.0,
        rebate=0.0,
        observation_type=ObservationType.CONTINUOUS,
    )

    engine = BarrierOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    ko_price = engine.price(ko_option, env)
    ki_price = engine.price(ki_option, env)

    vanilla = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    vanilla_engine = EuropeanMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    vanilla_price = vanilla_engine.price(vanilla, env)

    assert ko_price + ki_price == pytest.approx(vanilla_price, rel=1e-8, abs=1e-8)


def test_barrier_mc_immediate_knock_out_rebate_timing():
    env = _pricing_env()
    params = MCParams(num_paths=2000, time_steps=50, seed=7)

    option_hit = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=5.0,
        pay_at_hit=True,
        observation_type=ObservationType.CONTINUOUS,
    )
    engine = BarrierOptionMCEngine(params=params, method=MonteCarloMethod.PSEUDO)
    price_hit = engine.price(option_hit, env)
    assert price_hit == pytest.approx(option_hit.rebate, rel=1e-12, abs=1e-12)

    option_expiry = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=90.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=5.0,
        pay_at_hit=False,
        observation_type=ObservationType.CONTINUOUS,
    )
    price_expiry = engine.price(option_expiry, env)
    expected = option_expiry.rebate * math.exp(-env.get_rate(1.0) * 1.0)
    assert price_expiry == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_barrier_mc_discrete_schedule_with_bridge_option():
    env = _pricing_env()
    params = MCParams(num_paths=5000, time_steps=120, seed=21)

    schedule = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=0.25, barrier=110.0, payoff=2.0),
            ObservationRecord(observation_time=0.50, barrier=110.0, payoff=2.0),
            ObservationRecord(observation_time=0.75, barrier=110.0, payoff=2.0),
        ],
        aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
    )

    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        barrier=110.0,
        barrier_type=BarrierType.UP_OUT,
        maturity=1.0,
        rebate=2.0,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=schedule,
    )

    vanilla = EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )
    bs_engine = BlackScholesEngine()
    vanilla_price = bs_engine.price(vanilla, env)

    engine = BarrierOptionMCEngine(
        params=params,
        method=MonteCarloMethod.PSEUDO,
        use_brownian_bridge=True,
    )
    price = engine.price(option, env)

    assert price >= 0.0
    assert price <= vanilla_price + 0.5
    assert engine.get_last_std_error() is not None
