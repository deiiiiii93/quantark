"""
Unit tests for KO-reset snowball MC support.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.event_stats import KOResetEventStats
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import MCParams
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


def test_ko_reset_pricing_absolute():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=True,
    )
    engine = SnowballMCEngine(params=MCParams(num_paths=2000, time_steps=252))
    price = engine.price(product, pricing_env)
    assert np.isfinite(price)


def test_ko_reset_rebased_requires_discrete_ki():
    with pytest.raises(ValidationError):
        create_ko_reset_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity_pre=1.0,
            maturity_post=1.0,
            post_ko_mode=PostKOScheduleMode.REBASED,
            ki_continuous=True,
        )


def test_ko_reset_event_stats_shape():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.REBASED,
        ki_continuous=False,
    )
    engine = SnowballMCEngine(params=MCParams(num_paths=1000, time_steps=252))
    stats = engine.calculate_event_stats(product, pricing_env)
    assert isinstance(stats, KOResetEventStats)
    assert len(stats.pre_ko_times) == len(stats.pre_ko_probability)
    assert len(stats.post_ko_times) == len(stats.post_ko_probability)
