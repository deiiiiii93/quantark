"""Task 15a: frozen-layout bump identity tests (spec §4.8).

Per bump family, PROVE by object identity which layout parts are reused:
spot/vol/rate/div bumps reuse the whole Layout; a calendar (theta) roll
keeps the spatial object and rebuilds time; schedules rebuild per solve.
DCN (GridLayerMixin) is a first-class row.
"""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde import DCNPDEEngine, SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import SnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import NumericalError


def env(spot=100.0, vol=0.20, rate=0.05, div_yield=0.02, valuation=None):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=valuation or datetime(2024, 1, 1),
    )


def snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
        ki_barrier=75.0,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=1.0,
        maturity=1.0,
    )


def test_market_bumps_reuse_whole_layout_by_identity():
    base_env = env()
    product = snowball()
    solver = SnowballPDESolver(params=PDEParams())
    ctx = solver.create_bump_context(product, base_env)
    frozen = ctx._frozen_base_layout
    assert frozen is not None

    for bumped in (
        env(spot=101.0),          # spot
        env(vol=0.21),            # vol
        env(rate=0.06),           # rate
        env(div_yield=0.03),      # div
    ):
        ctx.price(product, bumped)
        assert ctx._active_layout is frozen  # SAME object — no rebuild


def test_calendar_roll_rebinds_time_keeps_spatial():
    # Float-maturity products keep tau constant under a valuation-date shift
    # (so the whole-layout reuse above is correct for them); a GENUINE roll —
    # tau and event times moved — must rebind time on the same spatial object.
    from quantark.asset.equity.engine.pde.grid import GridRequest

    base_env = env()
    product = snowball()
    solver = SnowballPDESolver(params=PDEParams())
    ctx = solver.create_bump_context(product, base_env)
    frozen = ctx._frozen_base_layout

    roll = 7.0 / 365.0
    fr = frozen.request
    rolled_request = GridRequest(
        tau=fr.tau - roll,
        bound_anchors=fr.bound_anchors,
        critical_prices=fr.critical_prices,
        hard_lower=None,
        hard_upper=None,
        event_times=tuple(t - roll for t in fr.event_times if t - roll > 0),
    )
    market = ctx.market_snapshot(product, base_env)
    rolled = ctx._bound_layout_for_solve(rolled_request, market)
    assert rolled is not frozen
    assert rolled.spatial is frozen.spatial  # spatial identity retained
    assert rolled.time is not frozen.time
    assert rolled.request.tau == pytest.approx(fr.tau - roll)


def test_schedule_rebuilt_each_solve():
    base_env = env()
    product = snowball()
    solver = SnowballPDESolver(params=PDEParams())
    solver.price(product, base_env)
    s1 = solver._active_schedule
    solver.grid_binder._cache.clear()  # force nothing — schedules are per-solve anyway
    solver.price(product, env(vol=0.25))
    assert solver._active_schedule is not s1


def test_out_of_domain_bump_raises():
    base_env = env()
    product = snowball()
    ctx = SnowballPDESolver(params=PDEParams()).create_bump_context(
        product, base_env
    )
    with pytest.raises(NumericalError):
        ctx.price(product, env(spot=500.0))


def test_dcn_bump_context_identity_matrix():
    import sys

    sys.path.insert(0, "test")
    from test_dcn_pde_solver import DCN_A, FLAT, flat_env, make_dcn

    product = make_dcn(DCN_A)
    base_env = flat_env(**FLAT, spot=DCN_A["initial_price"])
    eng = DCNPDEEngine()
    ctx = eng.create_bump_context(product, base_env)
    frozen = ctx._frozen_base_layout
    assert frozen is not None

    bumped = flat_env(**FLAT, spot=DCN_A["initial_price"] * 1.01)
    r = ctx.price_detailed(product, bumped)
    assert r is not None
    # identical geometry request -> frozen reuse (observable via binder cache
    # being bypassed: _bound_layout_for_solve returns the frozen object)
    market = ctx.market_snapshot(product, bumped)
    req = ctx.grid_request(product, market, frozen.request.tau)
    assert ctx._bound_layout_for_solve(req, market) is frozen
