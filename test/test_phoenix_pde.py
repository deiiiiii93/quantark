"""
Unit tests for PhoenixPDESolver.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from asset.equity.engine.pde_engine import PDEEngine
from asset.equity.param import PDEParams
from asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from asset.equity.product.option.phoenix_config import CouponBarrierConfig
from asset.equity.product.option.phoenix_option import PhoenixOption
from asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.calendar.day_counter import DayCountConvention
from util.enum import CouponPayType, ObservationType, ProtectionType


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.2,
    rate: float = 0.03,
    div_yield: float = 0.0,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def create_phoenix(
    coupon_barrier,
    coupon_pay_type: CouponPayType,
    memory_coupon: bool,
) -> PhoenixOption:
    barrier_config = BarrierConfig(
        ko_barrier=1.0e9,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=None,
    )
    coupon_config = CouponBarrierConfig(
        coupon_barrier=coupon_barrier,
        coupon_rate=0.02,
        coupon_pay_type=coupon_pay_type,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=memory_coupon,
    )
    payoff_config = PayoffConfig(rebate_rate=0.0, include_principal=True)
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        contract_multiplier=1.0,
        maturity=1.0,
    )


def create_phoenix_schedule(
    ko_dates,
    ko_barrier: float,
    coupon_barrier,
    coupon_rate: float,
    coupon_pay_type: CouponPayType,
    memory_coupon: bool,
    include_principal: bool,
    maturity: float = 1.0,
) -> PhoenixOption:
    barrier_config = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_dates,
        ki_barrier=None,
    )
    coupon_config = CouponBarrierConfig(
        coupon_barrier=coupon_barrier,
        coupon_rate=coupon_rate,
        coupon_pay_type=coupon_pay_type,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=memory_coupon,
    )
    payoff_config = PayoffConfig(rebate_rate=0.0, include_principal=include_principal)
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        contract_multiplier=1.0,
        maturity=maturity,
    )


def test_phoenix_pde_price_positive():
    env = create_pricing_env()
    phoenix = create_phoenix(
        coupon_barrier=[80.0, 80.0],
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
    )
    engine = PhoenixPDESolver(params=PDEParams(grid_size=120, time_steps=80))
    price = engine.price(phoenix, env)
    assert np.isfinite(price)
    assert price > 0.0


def test_phoenix_pde_dispatcher():
    env = create_pricing_env()
    phoenix = create_phoenix(
        coupon_barrier=[80.0, 80.0],
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
    )
    direct = PhoenixPDESolver(params=PDEParams(grid_size=120, time_steps=80))
    dispatcher = PDEEngine(params=PDEParams(grid_size=120, time_steps=80))
    direct_price = direct.price(phoenix, env)
    dispatcher_price = dispatcher.price(phoenix, env)
    assert np.isfinite(dispatcher_price)
    assert abs(dispatcher_price - direct_price) <= 1e-6


def test_phoenix_pde_memory_coupon_effect():
    env = create_pricing_env()
    memory_phoenix = create_phoenix(
        coupon_barrier=[1.0e9, 1.0e-6],
        coupon_pay_type=CouponPayType.EXPIRY,
        memory_coupon=True,
    )
    non_memory_phoenix = create_phoenix(
        coupon_barrier=[1.0e9, 1.0e-6],
        coupon_pay_type=CouponPayType.EXPIRY,
        memory_coupon=False,
    )
    engine = PhoenixPDESolver(params=PDEParams(grid_size=120, time_steps=80))
    memory_price = engine.price(memory_phoenix, env)
    non_memory_price = engine.price(non_memory_phoenix, env)
    assert memory_price >= non_memory_price - 1e-6


def test_phoenix_pde_memory_variable_coupon_amounts():
    env = create_pricing_env()
    ko_dates = [0.25, 0.75]
    memory_phoenix = create_phoenix_schedule(
        ko_dates=ko_dates,
        ko_barrier=1.0e9,
        coupon_barrier=[1.0e9, 1.0e-6],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=True,
        include_principal=False,
        maturity=1.0,
    )
    non_memory_phoenix = create_phoenix_schedule(
        ko_dates=ko_dates,
        ko_barrier=1.0e9,
        coupon_barrier=[1.0e9, 1.0e-6],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
        include_principal=False,
        maturity=1.0,
    )
    engine = PhoenixPDESolver(params=PDEParams(grid_size=160, time_steps=120))
    memory_price = engine.price(memory_phoenix, env)
    non_memory_price = engine.price(non_memory_phoenix, env)
    df = math.exp(-0.03 * 0.75)
    expected_memory = df * 100.0 * 0.02 * (0.25 + 0.5)
    expected_non_memory = df * 100.0 * 0.02 * 0.5
    assert abs(memory_price - expected_memory) <= 1e-1
    assert abs(non_memory_price - expected_non_memory) <= 1e-1


def test_phoenix_pde_ko_without_coupon():
    env = create_pricing_env()
    phoenix = create_phoenix_schedule(
        ko_dates=[0.5],
        ko_barrier=1.0e-6,
        coupon_barrier=[1.0e9],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
        include_principal=False,
        maturity=1.0,
    )
    engine = PhoenixPDESolver(params=PDEParams(grid_size=160, time_steps=120))
    price = engine.price(phoenix, env)
    assert abs(price) <= 1e-2


def test_phoenix_pde_reverse_boundary_unbounded():
    env = create_pricing_env()
    barrier_config = BarrierConfig(
        ko_barrier=97.0,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=None,
    )
    coupon_config = CouponBarrierConfig(
        coupon_barrier=[1.0e9, 1.0e9],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=False,
    )
    payoff_config = PayoffConfig(
        rebate_rate=0.0,
        include_principal=True,
        participation_rate=1.0,
        protection_type=ProtectionType.NONE,
    )
    phoenix = PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=True,
    )
    engine = PhoenixPDESolver(params=PDEParams(grid_size=120, time_steps=80))
    spot = env.spot
    tau = phoenix.get_maturity(env)
    r = env.get_rate(tau)
    q = env.get_div_yield(tau)
    sigma = env.get_vol(phoenix.strike, tau)
    x_vec, s_vec, _, t_vec, _ = engine._build_grids(phoenix, env, spot, sigma, tau, r, q)
    grid = np.zeros((len(x_vec), len(t_vec)))
    engine._set_boundary_conditions_v1(grid, x_vec, s_vec, 0, tau, phoenix, env)
    df = math.exp(-r * tau)
    df_div = math.exp(-q * tau)
    principal = phoenix.initial_price * phoenix.contract_multiplier
    expected = (principal + phoenix.strike) * df - s_vec[-1] * df_div
    assert abs(grid[-1, 0] - expected) <= 1e-6


def test_phoenix_pde_variable_discrete_ki_barrier_uses_time_varying_levels():
    """
    Regression test: PDE KI jump must use the KI barrier at each observation time.

    If PDE incorrectly uses only the first KI barrier for all KI observations, both
    prices below become nearly identical (and often near zero) because KI almost
    never triggers. With correct time-varying KI handling, participation changes the
    V1 downside and therefore price.
    """
    env = create_pricing_env(vol=0.10, rate=0.02, div_yield=0.0)

    def build_product(participation_rate: float) -> PhoenixOption:
        ki_schedule = ObservationSchedule(
            records=[
                ObservationRecord(observation_time=0.5, barrier=1.0e-6),
                ObservationRecord(observation_time=1.0, barrier=120.0),
            ]
        )
        barrier_config = BarrierConfig(
            ko_barrier=1.0e9,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.5, 1.0],
            ki_barrier=[1.0e-6, 120.0],  # First KI never hits; second KI should hit often
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_schedule,
            ki_continuous=False,
        )
        coupon_config = CouponBarrierConfig(
            coupon_barrier=[1.0e9, 1.0e9],  # disable coupon effects
            coupon_rate=0.0,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        )
        payoff_config = PayoffConfig(
            rebate_rate=0.0,
            include_principal=False,
            participation_rate=participation_rate,
            protection_type=ProtectionType.NONE,
        )
        return PhoenixOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=barrier_config,
            coupon_config=coupon_config,
            payoff_config=payoff_config,
            contract_multiplier=1.0,
            maturity=1.0,
        )

    high_participation = build_product(participation_rate=1.0)
    low_participation = build_product(participation_rate=0.2)

    engine = PhoenixPDESolver(params=PDEParams(grid_size=180, time_steps=120))
    price_high = engine.price(high_participation, env)
    price_low = engine.price(low_participation, env)

    # Lower KI participation should reduce downside and increase price.
    assert price_low > price_high + 1e-3
