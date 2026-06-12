from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine import BlackScholesEngine, EuropeanPDESolver, SnowballQuadEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType, OptionType, PostKOScheduleMode


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _product() -> EuropeanVanillaOption:
    return EuropeanVanillaOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
    )


def test_base_engine_spot_curve_falls_back_to_repricing(monkeypatch):
    engine = BlackScholesEngine()
    calls = 0
    original = engine.calculate_greeks

    def counted(product, env):
        nonlocal calls
        calls += 1
        return original(product, env)

    monkeypatch.setattr(engine, "calculate_greeks", counted)
    curve = engine.calculate_spot_greeks_curve(_product(), _env(), [90.0, 100.0, 110.0])

    assert calls == 3
    assert [row["spot"] for row in curve] == [90.0, 100.0, 110.0]
    assert {row["calculation_mode"] for row in curve} == {"reprice"}


def test_pde_spot_curve_uses_one_solve_and_matches_point_greeks(monkeypatch):
    engine = EuropeanPDESolver(params=PDEParams(grid_size=201, time_steps=100))
    calls = 0
    original = engine._solve

    def counted(product, env):
        nonlocal calls
        calls += 1
        return original(product, env)

    monkeypatch.setattr(engine, "_solve", counted)
    spots = [90.0, 100.0, 110.0]
    curve = engine.calculate_spot_greeks_curve(_product(), _env(), spots)

    assert calls == 1
    assert {row["calculation_mode"] for row in curve} == {"engine_grid"}
    for row, spot in zip(curve, spots, strict=True):
        env = deepcopy(_env())
        env.spot_quote.spot = spot
        expected = BlackScholesEngine().calculate_greeks(_product(), env)
        assert row["price"] == pytest.approx(expected["price"], rel=2e-2)
        assert row["delta"] == pytest.approx(expected["delta"], rel=5e-2, abs=2e-2)
        assert row["gamma"] == pytest.approx(expected["gamma"], rel=1e-1, abs=2e-3)


def test_snowball_quad_spot_curve_uses_one_recursion(monkeypatch):
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        contract_multiplier=10_000.0,
        maturity=1.0,
    )
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    calls = 0
    original = engine.price

    def counted(product, env):
        nonlocal calls
        calls += 1
        return original(product, env)

    monkeypatch.setattr(engine, "price", counted)
    curve = engine.calculate_spot_greeks_curve(product, _env(), [90.0, 100.0, 110.0])

    assert calls == 1
    assert {row["calculation_mode"] for row in curve} == {"engine_grid"}
    assert all(row["price"] > 0.0 for row in curve)


@pytest.mark.parametrize(
    ("engine", "product"),
    [
        (
            PhoenixQuadEngine(params=QuadParams(grid_points=201)),
            PhoenixOption(
                initial_price=100.0,
                strike=100.0,
                barrier_config=BarrierConfig(
                    ko_barrier=1.0e9,
                    ko_rate=0.0,
                    ko_observation_type=ObservationType.DISCRETE,
                    ko_observation_dates=[0.5, 1.0],
                    ki_barrier=None,
                ),
                coupon_config=CouponBarrierConfig(
                    coupon_barrier=[80.0, 80.0],
                    coupon_rate=0.02,
                    coupon_pay_type=CouponPayType.INSTANT,
                    day_count_convention=DayCountConvention.ACT_365,
                    memory_coupon=False,
                ),
                payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
                contract_multiplier=1.0,
                maturity=1.0,
            ),
        ),
        (
            KOResetSnowballQuadEngine(params=QuadParams(grid_points=201)),
            create_ko_reset_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity_pre=1.0,
                maturity_post=2.0,
                post_ko_mode=PostKOScheduleMode.ABSOLUTE,
                ki_continuous=True,
            ),
        ),
    ],
)
def test_other_quad_spot_curves_use_engine_grid(engine, product):
    curve = engine.calculate_spot_greeks_curve(product, _env(), [90.0, 100.0, 110.0])

    assert {row["calculation_mode"] for row in curve} == {"engine_grid"}
    assert all(np.isfinite(row["price"]) for row in curve)
