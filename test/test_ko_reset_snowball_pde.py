"""
Unit tests for KO-reset Snowball PDE solver.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import KOResetEventStats
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import PostKOScheduleMode
from quantark.util.exceptions import ValidationError


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


def test_ko_reset_pde_price_absolute():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=True,
    )
    solver = KOResetSnowballPDESolver(PDEParams())
    price = solver.price(product, pricing_env)
    assert np.isfinite(price)


def test_ko_reset_pde_engine_dispatch():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    engine = PDEEngine(PDEParams())
    price = engine.price(product, pricing_env)
    assert np.isfinite(price)


def test_ko_reset_pde_reject_rebased():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=1.0,
        post_ko_mode=PostKOScheduleMode.REBASED,
        ki_continuous=False,
    )
    solver = KOResetSnowballPDESolver(PDEParams())
    with pytest.raises(ValidationError):
        solver.price(product, pricing_env)


def test_ko_reset_quad_event_stats_absolute():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    engine = KOResetSnowballQuadEngine(QuadParams(grid_points=501))

    stats = engine.calculate_event_stats(product, pricing_env)

    assert isinstance(stats, KOResetEventStats)
    assert stats.pre_ko_times.shape == stats.pre_ko_probability.shape
    assert stats.post_ko_times.shape == stats.post_ko_probability.shape
    assert stats.ko_times.shape == stats.pre_ko_times.shape
    assert stats.ko_probability.shape == stats.pre_ko_probability.shape
    assert float(np.sum(stats.pre_ko_probability)) <= 1.0 + 1e-6
    assert float(np.sum(stats.post_ko_probability)) <= 1.0 + 1e-6

    pv_parts = (
        float(np.sum(stats.expected_discounted_ko_cashflow))
        + float(stats.expected_discounted_post_ko_cashflow)
        + float(stats.expected_discounted_maturity_cashflow)
    )
    assert stats.pv == pytest.approx(pv_parts, abs=1e-8)


def test_ko_reset_pde_event_stats_absolute():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    solver = KOResetSnowballPDESolver(PDEParams())

    stats = solver.calculate_event_stats(product, pricing_env)

    assert isinstance(stats, KOResetEventStats)
    assert stats.pre_ko_times.shape == stats.pre_ko_probability.shape
    assert stats.post_ko_times.shape == stats.post_ko_probability.shape

    pv_parts = (
        float(np.sum(stats.expected_discounted_ko_cashflow))
        + float(stats.expected_discounted_post_ko_cashflow)
        + float(stats.expected_discounted_maturity_cashflow)
    )
    assert stats.pv == pytest.approx(pv_parts, abs=1e-8)
