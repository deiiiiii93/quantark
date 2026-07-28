"""
Unit tests for PhoenixQuadEngine.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import EventProjectionMode
from quantark.util.exceptions import NumericalError


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


def test_phoenix_quad_price_positive():
    env = create_pricing_env()
    phoenix = create_phoenix(
        coupon_barrier=[80.0, 80.0],
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
    )
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=201))
    price = engine.price(phoenix, env)
    assert np.isfinite(price)
    assert price > 0.0


def test_phoenix_quad_memory_coupon_effect():
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
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=201))
    memory_price = engine.price(memory_phoenix, env)
    non_memory_price = engine.price(non_memory_phoenix, env)
    assert memory_price >= non_memory_price - 1e-6


def test_phoenix_quad_coupon_pay_type_discounting():
    env = create_pricing_env()
    instant_phoenix = create_phoenix(
        coupon_barrier=[1.0e-6, 1.0e-6],
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
    )
    expiry_phoenix = create_phoenix(
        coupon_barrier=[1.0e-6, 1.0e-6],
        coupon_pay_type=CouponPayType.EXPIRY,
        memory_coupon=False,
    )
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=201))
    price_instant = engine.price(instant_phoenix, env)
    price_expiry = engine.price(expiry_phoenix, env)
    assert price_instant >= price_expiry - 1e-6


def test_phoenix_quad_memory_variable_coupon_amounts():
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
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=201))
    memory_price = engine.price(memory_phoenix, env)
    non_memory_price = engine.price(non_memory_phoenix, env)
    df = math.exp(-0.03 * 0.75)
    expected_memory = df * 100.0 * 0.02 * (0.25 + 0.5)
    expected_non_memory = df * 100.0 * 0.02 * 0.5
    assert abs(memory_price - expected_memory) <= 1e-1
    assert abs(non_memory_price - expected_non_memory) <= 1e-1


def test_phoenix_quad_ko_without_coupon():
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
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=201))
    price = engine.price(phoenix, env)
    assert abs(price) <= 1e-2


def test_phoenix_quad_marker_ko_levels_do_not_change_spatial_grid():
    """Two 100x KO markers are economically disabled but remain coupon dates."""
    env = create_pricing_env(vol=0.27)
    common = dict(
        ko_dates=[0.1, 0.2, 3.0],
        coupon_barrier=[1.0e-6, 1.0e-6, 1.0e-6],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
        include_principal=True,
        maturity=3.0,
    )
    marked = create_phoenix_schedule(
        ko_barrier=[10_000.0, 10_000.0, 103.0],
        **common,
    )
    disabled = create_phoenix_schedule(
        ko_barrier=[1.0e300, 1.0e300, 103.0],
        **common,
    )
    params = QuadParams(grid_points=2001)
    marked_engine = PhoenixQuadEngine(params=params)
    disabled_engine = PhoenixQuadEngine(params=params)

    marked_price = marked_engine.price(marked, env)
    disabled_price = disabled_engine.price(disabled, env)
    marked_grid = marked_engine._last_spot_greeks_grid[0]
    disabled_grid = disabled_engine._last_spot_greeks_grid[0]

    assert np.array_equal(marked_grid, disabled_grid)
    assert marked_price == pytest.approx(disabled_price, abs=1e-8)
    assert marked_engine._last_ignored_ko_observation_indices == (0, 1)
    assert disabled_engine._last_ignored_ko_observation_indices == (0, 1)
    assert len(marked.resolve_ko_observations(env)) == 3


def test_quad_params_cell_average_default_and_nodal_opt_out():
    assert QuadParams().event_projection == EventProjectionMode.CELL_AVERAGE
    assert QuadParams().integration_rule == "trapezoid"
    assert (
        QuadParams(event_projection="nodal").event_projection
        == EventProjectionMode.NODAL
    )
    assert QuadParams(integration_rule="simpson").integration_rule == "simpson"


def test_joint_coupon_ko_projection_averages_coincident_threshold_once():
    grid = np.linspace(-0.2, 0.2, 9)
    zeros = np.zeros_like(grid)
    ones = np.ones_like(grid)

    projected_out, projected_in = PhoenixQuadEngine._project_joint_phoenix_event(
        grid=grid,
        spot=100.0,
        is_reverse=False,
        coupon_barrier=100.0,
        ko_barrier=100.0,
        ki_barrier=None,
        val_miss_out=zeros,
        val_pay_out=ones,
        val_miss_in=zeros,
        val_pay_in=ones,
        ko_value_without_coupon=2.0,
        ko_value_with_coupon=3.0,
        disable_ko_after_ki=False,
    )

    center = len(grid) // 2
    assert projected_out[center] == pytest.approx(1.5)
    assert projected_in[center] == pytest.approx(1.5)
    assert np.all(projected_out[:center] == 0.0)
    assert np.all(projected_out[center + 1 :] == 3.0)


def test_phoenix_quad_auto_convergence_returns_matching_finest_grid():
    env = create_pricing_env(vol=0.27)
    phoenix = create_phoenix_schedule(
        ko_dates=[0.25, 0.5, 1.0],
        ko_barrier=[110.0, 108.0, 105.0],
        coupon_barrier=[85.0, 85.0, 85.0],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
        include_principal=True,
    )
    engine = PhoenixQuadEngine(
        params=QuadParams(
            grid_points=201,
            auto_converge=True,
            convergence_rel_tol=5.0e-3,
            max_convergence_grid_points=1601,
        )
    )

    price = engine.price(phoenix, env)
    info = engine._last_convergence_info

    assert np.isfinite(price)
    assert info["converged"] is True
    assert len(info["estimates"]) >= 2
    assert len(engine._last_spot_greeks_grid[0]) == info["grid_points"]


def test_phoenix_quad_auto_convergence_fails_closed_at_cap():
    env = create_pricing_env(vol=0.27)
    phoenix = create_phoenix_schedule(
        ko_dates=[0.25, 0.5, 1.0],
        ko_barrier=[110.0, 108.0, 105.0],
        coupon_barrier=[85.0, 85.0, 85.0],
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        memory_coupon=False,
        include_principal=True,
    )
    engine = PhoenixQuadEngine(
        params=QuadParams(
            grid_points=201,
            auto_converge=True,
            convergence_rel_tol=1.0e-14,
            convergence_abs_tol=1.0e-14,
            max_convergence_grid_points=401,
        )
    )

    with pytest.raises(NumericalError, match="convergence was not reached"):
        engine.price(phoenix, env)
