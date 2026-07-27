"""Regression tests for fixed PDE bump grids in numerical Greeks."""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.param import BumpConfig, EngineParams, PDEParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType


def _pricing_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.035),
        div_yield=ContinuousDividendYield(div_yield=0.012),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball() -> SnowballOption:
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.12,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _phoenix() -> PhoenixOption:
    barrier_config = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=None,
    )
    coupon_config = CouponBarrierConfig(
        coupon_barrier=[80.0, 80.0, 80.0, 80.0],
        coupon_rate=0.08,
        coupon_pay_type=CouponPayType.INSTANT,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=True,
    )
    payoff_config = PayoffConfig(include_principal=True)
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        coupon_config=coupon_config,
        payoff_config=payoff_config,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _base_spatial_bounds(solver, product, pricing_env):
    # Layer-native: bounds come from binding the declarative request.
    prep = getattr(solver, "_prepare_for_request", None)
    tau = (
        prep(product, pricing_env)
        if prep is not None
        else product.get_maturity(pricing_env)
    )
    market = solver.market_snapshot(product, pricing_env)
    layout = solver.grid_binder.bind(
        solver.grid_request(product, market, tau), market
    )
    return layout.spatial.bounds


def _effective_spatial_bounds(solver, product, pricing_env):
    tau = product.get_maturity(pricing_env)
    spot = pricing_env.spot
    strike = getattr(product, "strike", spot)
    rate = pricing_env.get_rate(tau)
    div = pricing_env.get_div_yield(tau)
    vol = pricing_env.get_vol(strike, tau)
    _, s_vec, _, _, _ = solver._build_grids(
        product, pricing_env, spot, vol, tau, rate, div
    )
    return float(s_vec[0]), float(s_vec[-1])


def _rate_bumped_env(pricing_env, bump):
    bumped = deepcopy(pricing_env)
    maturity = 1.0
    bumped.rate_curve = FlatRateCurve(pricing_env.get_rate(maturity) + bump)
    return bumped


def _div_bumped_env(pricing_env, bump):
    bumped = deepcopy(pricing_env)
    maturity = 1.0
    bumped.div_yield = ContinuousDividendYield(
        pricing_env.get_div_yield(maturity) + bump
    )
    return bumped


@pytest.mark.parametrize(
    ("solver_cls", "product_factory"),
    [
        (SnowballPDESolver, _snowball),
        (PhoenixPDESolver, _phoenix),
    ],
)
def test_pde_bump_context_freezes_effective_spatial_bounds(
    solver_cls, product_factory
):
    product = product_factory()
    env = _pricing_env()
    params = PDEParams(
        grid_size=90,
        time_steps=45,
        cache_strategy="disable",
    )
    solver = solver_cls(params=params)
    bump = 0.0001

    base_bounds = _base_spatial_bounds(solver, product, env)
    rate_moved_bounds = _base_spatial_bounds(
        solver, product, _rate_bumped_env(env, bump)
    )
    div_moved_bounds = _base_spatial_bounds(
        solver, product, _div_bumped_env(env, bump)
    )

    assert np.max(np.abs(np.subtract(rate_moved_bounds, base_bounds))) > 1e-6
    assert np.max(np.abs(np.subtract(div_moved_bounds, base_bounds))) > 1e-6

    fixed_solver = solver.create_bump_context(product, env)

    assert fixed_solver is not solver
    if fixed_solver._uses_grid_layer():
        # Migrated solvers freeze the whole base Layout (grid redesign
        # spec §4.8) — a strictly stronger guarantee than frozen params:
        # every bumped re-solve reuses the SAME object.
        frozen = fixed_solver._frozen_base_layout
        assert frozen is not None
        for bumped_env in (
            _rate_bumped_env(env, bump),
            _div_bumped_env(env, bump),
        ):
            fixed_solver.price(product, bumped_env)
            assert fixed_solver._active_layout is frozen
    else:
        assert fixed_solver.params.s_min == pytest.approx(base_bounds[0])
        assert fixed_solver.params.s_max == pytest.approx(base_bounds[1])
        assert _effective_spatial_bounds(
            fixed_solver, product, _rate_bumped_env(env, bump)
        ) == pytest.approx(base_bounds)
        assert _effective_spatial_bounds(
            fixed_solver, product, _div_bumped_env(env, bump)
        ) == pytest.approx(base_bounds)


@pytest.mark.parametrize("product_factory", [_snowball, _phoenix])
def test_greeks_calculator_pde_rho_matches_manual_fixed_domain_repricing(
    product_factory,
):
    product = product_factory()
    env = _pricing_env()
    params = PDEParams(grid_size=90, time_steps=45, cache_strategy="disable")
    engine = PDEEngine(params=params)
    bump = 0.0001
    calc = GreeksCalculator(
        params=EngineParams(bump_config=BumpConfig(rate_bump=bump, div_bump=bump))
    )

    fixed_engine = engine.create_bump_context(product, env)
    base_price = fixed_engine.price(product, env)
    manual_rho = (
        fixed_engine.price(product, _rate_bumped_env(env, bump)) - base_price
    ) * (0.01 / bump)
    manual_dividend_rho = (
        fixed_engine.price(product, _div_bumped_env(env, bump)) - base_price
    ) * (0.01 / bump)

    assert calc.calculate_numerical_rho(product, env, engine, rate_bump=bump) == (
        pytest.approx(manual_rho)
    )
    assert calc.calculate_numerical_dividend_rho(
        product, env, engine, div_bump=bump
    ) == pytest.approx(manual_dividend_rho)
