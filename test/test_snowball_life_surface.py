"""One PDE solve with a node on every simulation day, read along the life."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.pde.base_pde_solver import LifeSurfaceSolution
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

SPOT = 100.0


def _product(maturity_days=60, ko_days=(20, 40, 60), ki_days=(10, 30, 50)):
    return create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=maturity_days / 365.0, contract_multiplier=1.0,
        ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.2, num_observations=len(ko_days),
        ko_observation_dates=[d / 365.0 for d in ko_days], ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False, ki_observation_dates=[d / 365.0 for d in ki_days], include_principal=True,
    )


def _env(spot=SPOT, day=0, vol=0.25):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot, asset_name="X"), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.02), valuation_date=datetime(2024, 1, 2 + day),
    )


def _daily(maturity_days=60):
    return tuple(d / 365.0 for d in range(1, maturity_days))


def test_the_surface_has_a_node_on_every_day_and_reads_the_engine_price_at_day_zero():
    engine = PDEEngine(params=PDEParams())
    product = _product()
    surface = engine.solve_life_surface(product, _env(), extra_times=_daily())
    assert isinstance(surface, LifeSurfaceSolution)
    assert surface.v0.shape == (surface.x.size, surface.t.size) == surface.v1.shape
    for t in _daily():
        assert surface.t[surface.step_of[t]] == pytest.approx(t, abs=1e-15)
    column = surface.t0_readout if surface.t0_readout is not None else surface.v0[:, 0]
    # Daily nodes change the time fill (240 steps here against 168 on the
    # plain grid), so the day-0 readout differs from price() by
    # discretisation only: measured 2.8e-6 relative.  Without extra nodes
    # the seam IS the ordinary solve, to the bit.
    assert float(np.interp(np.log(SPOT), surface.x, column)) == pytest.approx(engine.price(product, _env()), rel=1e-5)
    plain = engine.solve_life_surface(product, _env(), extra_times=())
    assert float(np.interp(np.log(SPOT), plain.x, plain.v0[:, 0])) == engine.price(product, _env())


def test_the_solver_is_left_as_it_was_found():
    engine = PDEEngine(params=PDEParams())
    product = _product()
    before = engine.price(product, _env(spot=97.0))
    engine.solve_life_surface(product, _env(), extra_times=_daily())
    solver = engine._get_solver(product)
    assert solver._extra_time_nodes == ()
    assert engine.price(product, _env(spot=97.0)) == before
    assert PDEEngine(params=PDEParams()).price(product, _env(spot=97.0)) == before


def test_an_interior_node_agrees_with_a_fresh_aged_solve_within_a_stated_bound():
    # The surface reads day 25 off the start-date grid; a fresh solve on day
    # 25 builds its own grid around the new spot.  They differ by
    # discretisation only; the bound below is the measured gap with margin.
    engine = PDEEngine(params=PDEParams())
    product = _product()
    surface = engine.solve_life_surface(product, _env(), extra_times=_daily())
    day = 25
    node = surface.step_of[day / 365.0]
    aged = _product()
    aged.maturity = (60 - day) / 365.0
    aged.barrier_config = aged.barrier_config.time_shift(day / 365.0, datetime(2024, 1, 2 + day), _env(day=day))[0]
    for spot in (90.0, 100.0, 102.0):
        read = float(np.interp(np.log(spot), surface.x, surface.v0[:, node]))
        fresh = engine.price(aged, _env(spot=spot, day=day))
        assert read == pytest.approx(fresh, rel=1e-2), (spot, read, fresh)


def test_the_knocked_in_surface_is_the_v1_slab():
    engine = PDEEngine(params=PDEParams())
    product = _product()
    surface = engine.solve_life_surface(product, _env(), extra_times=_daily())
    node = surface.step_of[25 / 365.0]
    assert float(np.interp(np.log(80.0), surface.x, surface.v1[:, node])) < float(
        np.interp(np.log(80.0), surface.x, surface.v0[:, node])
    )


def test_a_product_without_the_seam_is_refused():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    vanilla = EuropeanVanillaOption(strike=SPOT, option_type=OptionType.CALL, maturity=0.5)
    with pytest.raises(ValidationError):
        PDEEngine(params=PDEParams()).solve_life_surface(vanilla, _env(), extra_times=(0.1,))


def test_the_solution_reports_whether_its_time_fill_was_scaled():
    """A life surface asks for hundreds of intervals, so max_steps bites it
    first; the solution carries the requested and delivered step counts."""
    from quantark.asset.equity.engine.pde.grid import GridConfig

    product = _product()
    full = PDEEngine(params=PDEParams()).solve_life_surface(product, _env(), extra_times=_daily())
    assert not full.fill_scaled
    assert full.actual_steps == full.requested_steps == full.t.size - 1

    capped = PDEEngine(params=PDEParams(grid=GridConfig(max_steps=100))).solve_life_surface(
        product, _env(), extra_times=_daily())
    assert capped.fill_scaled
    assert capped.actual_steps <= 100 < capped.requested_steps == full.requested_steps
