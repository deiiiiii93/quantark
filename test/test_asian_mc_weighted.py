"""Weighted Monte-Carlo averaging for Asian options (sub-project D, reference engine)."""
import math
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.product.option.asian_option import (
    AsianObservationRecord,
    AsianOption,
)
from quantark.asset.equity.engine.mc import AsianOptionMCEngine
from quantark.asset.equity.param import MCParams
from quantark.priceenv import PricingEnvironment
from quantark.param import (
    SpotQuote,
    FlatVolSurface,
    FlatRateCurve,
    ContinuousDividendYield,
)
from quantark.util.enum import AveragingType, OptionType, AsianStrikeType
from quantark.util.enum.engine_enums import MonteCarloMethod


def _env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


# --- deterministic kernel test (no MC noise) --------------------------------

def test_compute_averages_arithmetic_is_weighted():
    engine = AsianOptionMCEngine(params=MCParams(num_paths=10))
    paths = np.array([[100.0, 110.0, 120.0], [100.0, 90.0, 80.0]])
    obs_indices = np.array([0, 1])
    past_prices = np.array([])
    weights = np.array([0.25, 0.75])
    avg = engine._compute_averages(
        paths, obs_indices, past_prices, weights, 2, AveragingType.ARITHMETIC
    )
    # path0: .25*110 + .75*120 = 117.5 ; path1: .25*90 + .75*80 = 82.5
    assert avg == pytest.approx([117.5, 82.5])


def test_compute_averages_geometric_is_weighted():
    engine = AsianOptionMCEngine(params=MCParams(num_paths=10))
    paths = np.array([[100.0, 110.0, 120.0]])
    obs_indices = np.array([0, 1])
    past_prices = np.array([])
    weights = np.array([0.25, 0.75])
    avg = engine._compute_averages(
        paths, obs_indices, past_prices, weights, 2, AveragingType.GEOMETRIC
    )
    expected = math.exp(0.25 * math.log(110.0) + 0.75 * math.log(120.0))
    assert avg[0] == pytest.approx(expected)


# --- deterministic all-past weighted price ----------------------------------

def test_all_past_weighted_price_exact():
    engine = AsianOptionMCEngine(params=MCParams(num_paths=1000, seed=1))
    recs = [
        AsianObservationRecord(observation_time=-0.5, observed_price=100.0, weight=1.0),
        AsianObservationRecord(observation_time=-0.25, observed_price=200.0, weight=3.0),
    ]
    opt = AsianOption(
        strike=150.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
        observation_records=recs,
    )
    price = engine.price(opt, _env(rate=0.05))
    # weighted avg = (100*1 + 200*3)/4 = 175 ; call payoff = 25 ; discount exp(-0.05)
    assert price == pytest.approx(25.0 * math.exp(-0.05), rel=1e-9)


# --- weighting changes the simulated price ----------------------------------

def test_weighted_simulated_price_differs_from_uniform():
    engine = AsianOptionMCEngine(
        params=MCParams(num_paths=200000, seed=7), method=MonteCarloMethod.QUASI
    )
    env = _env(spot=100.0, rate=0.05, vol=0.30, div=0.0)
    times = [0.25, 0.5, 0.75, 1.0]
    uniform = AsianOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0,
        observation_records=[AsianObservationRecord(observation_time=t) for t in times],
    )
    # heavy weight on the last (highest-variance, highest-forward) fixing
    weighted = AsianOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0,
        observation_records=[
            AsianObservationRecord(observation_time=t, weight=w)
            for t, w in zip(times, [1.0, 1.0, 1.0, 7.0])
        ],
    )
    p_uniform = engine.price(uniform, env)
    p_weighted = engine.price(weighted, env)
    # Loading the back fixing raises both the average's mean and its variance,
    # so the call is worth strictly more than the equal-weight average.
    assert p_weighted > p_uniform + 0.5
