"""create_bump_context freezes the spatial grid across bumps [§11.4].

With include_spot_in_critical_points=True the base grid snaps a node onto spot;
without freezing, a spot bump moves that node and pollutes delta/gamma with grid
noise. The bump-context engine must hold x_vec/s_vec fixed under spot/vol/rate/
div bumps, and t_vec fixed too.
"""

from copy import deepcopy
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.param.rrf import FlatRateCurve as RRFFlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


def _env(spot=100.0, vol=0.2, rate=0.03, div=0.01):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def test_frozen_context_holds_grid_under_spot_bump():
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    bump_engine = solver.create_bump_context(product, env)

    base = bump_engine._solve(product, env)
    env_up = deepcopy(env)
    env_up.spot_quote.spot *= 1.01
    up = bump_engine._solve(product, env_up)
    env_dn = deepcopy(env)
    env_dn.spot_quote.spot *= 0.99
    dn = bump_engine._solve(product, env_dn)

    assert np.array_equal(base.x_vec, up.x_vec)
    assert np.array_equal(base.x_vec, dn.x_vec)
    assert np.array_equal(base.s_vec, up.s_vec)


def test_unfrozen_context_grid_moves_under_spot_bump():
    # Contrast: without the bump context, the spot critical point moves the grid.
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    base = solver._solve(product, env)
    env_up = deepcopy(env)
    env_up.spot_quote.spot *= 1.01
    up = solver._solve(product, env_up)
    assert not np.array_equal(base.x_vec, up.x_vec)


def test_frozen_context_holds_grid_under_rate_and_vol_bumps():
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    bump_engine = solver.create_bump_context(product, env)
    base = bump_engine._solve(product, env)

    env_vol = deepcopy(env)
    env_vol.vol_surface = FlatVolSurface(volatility=0.21)
    env_rate = deepcopy(env)
    env_rate.rate_curve = RRFFlatRateCurve(0.04)

    vol = bump_engine._solve(product, env_vol)
    rate = bump_engine._solve(product, env_rate)
    assert np.array_equal(base.x_vec, vol.x_vec)
    assert np.array_equal(base.x_vec, rate.x_vec)
