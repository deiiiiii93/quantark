"""Efficiency helpers: shallow bump envs, pre-grouped futures (Task 15)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay.product_replay import _env_with  # noqa: E402
from quantark.backtest.replay.market import SignedDividendYield  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402


def _env(spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="X"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=SignedDividendYield(0.01),
        valuation_date=datetime(2024, 1, 2),
    )


def test_env_with_shares_market_objects_and_replaces_spot():
    env = _env()
    bumped = _env_with(env, spot=101.0, underlying="X")
    assert bumped.vol_surface is env.vol_surface
    assert bumped.rate_curve is env.rate_curve
    assert bumped.div_yield is env.div_yield
    assert float(bumped.spot) == 101.0
    assert float(env.spot) == 100.0


def test_env_with_replaces_div_yield_only_when_given():
    env = _env()
    q = SignedDividendYield(0.03)
    bumped = _env_with(env, div_yield=q)
    assert bumped.div_yield is q
    assert bumped.spot_quote is env.spot_quote


def test_env_with_mutation_isolated():
    env = _env()
    bumped = _env_with(env, spot=105.0, underlying="X")
    bumped.spot_quote.spot = 999.0
    assert float(env.spot) == 100.0


def test_futures_slice_equals_filter_and_raises_on_missing():
    market = fixtures._market_data()
    date = pd.Timestamp("2024-01-03")
    grouped = market.get_futures_slice(date)
    filtered = market.futures_data[market.futures_data["date"] == date]
    pd.testing.assert_frame_equal(
        grouped.reset_index(drop=True), filtered.reset_index(drop=True)
    )
    with pytest.raises(ValidationError, match="No futures data"):
        market.get_futures_slice(pd.Timestamp("2030-01-01"))
