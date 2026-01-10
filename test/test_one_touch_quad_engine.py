"""
Unit tests for OneTouchQuadEngine.
"""

from datetime import datetime

import pytest

from asset.equity.engine.analytical import OneTouchAnalyticalEngine
from asset.equity.engine.quad import OneTouchQuadEngine
from asset.equity.param import QuadParams
from asset.equity.product.option import OneTouchOption
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import BarrierDirection, ObservationType, TouchType


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


def test_expiry_only_one_touch_matches_analytical():
    env = create_pricing_env()
    option = OneTouchOption(
        barrier=110.0,
        barrier_direction=BarrierDirection.UP,
        maturity=1.0,
        rebate=5.0,
        payment_at_hit=False,
        touch_type=TouchType.ONE_TOUCH,
        observation_type=ObservationType.EXPIRY,
    )

    quad_engine = OneTouchQuadEngine(params=QuadParams(grid_points=801))
    ana_engine = OneTouchAnalyticalEngine()

    quad_price = quad_engine.price(option, env)
    ana_price = ana_engine.price(option, env)

    assert quad_price == pytest.approx(ana_price, rel=1e-6, abs=1e-6)


def test_expiry_only_no_touch_matches_analytical():
    env = create_pricing_env()
    option = OneTouchOption(
        barrier=90.0,
        barrier_direction=BarrierDirection.DOWN,
        maturity=1.0,
        rebate=5.0,
        payment_at_hit=False,
        touch_type=TouchType.NO_TOUCH,
        observation_type=ObservationType.EXPIRY,
    )

    quad_engine = OneTouchQuadEngine(params=QuadParams(grid_points=801))
    ana_engine = OneTouchAnalyticalEngine()

    quad_price = quad_engine.price(option, env)
    ana_price = ana_engine.price(option, env)

    assert quad_price == pytest.approx(ana_price, rel=1e-6, abs=1e-6)
