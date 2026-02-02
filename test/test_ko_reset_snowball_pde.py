"""
Unit tests for KO-reset Snowball PDE solver.
"""

from datetime import datetime

import numpy as np
import pytest

from asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from asset.equity.engine.pde_engine import PDEEngine
from asset.equity.param import PDEParams
from asset.equity.product.option import create_ko_reset_snowball
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import PostKOScheduleMode
from util.exceptions import ValidationError


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
    solver = KOResetSnowballPDESolver(PDEParams(grid_size=80, time_steps=40))
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
    engine = PDEEngine(PDEParams(grid_size=80, time_steps=40))
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
    solver = KOResetSnowballPDESolver(PDEParams(grid_size=60, time_steps=30))
    with pytest.raises(ValidationError):
        solver.price(product, pricing_env)
