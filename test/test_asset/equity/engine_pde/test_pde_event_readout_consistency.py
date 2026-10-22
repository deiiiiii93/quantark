"""Pointwise readout and greek extraction around a discrete barrier event.

A discrete KI observation writes a value JUMP onto the grid: the event
function is the knocked-in branch at or below the barrier and the alive
branch above it, and the two branches do not meet.  Differentiating the
post-event vector across that jump contaminates delta by O(J/h), so the
readout must take the branch, not the event trace.

The solver already captures the smooth 0+ branch before the valuation-date
event is written (``_t0_pre_event_cols``), and ``price()`` reads it.  These
tests pin that the greeks read the same object, and that the stencil is
evaluated at the query spot rather than at whichever node happens to be
nearest it.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.backtest.replay.market import SignedDividendYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

SPOT_0 = 100.0
KI_BARRIER = 90.0
OBS_DAYS = 30
VALUATION = datetime(2026, 1, 2)


def control_product():
    """A short snowball whose KI observations include the valuation date."""
    ki_times = [d / 365.0 for d in range(0, OBS_DAYS + 1)]
    product = create_standard_snowball(
        initial_price=SPOT_0, strike=SPOT_0, maturity=OBS_DAYS / 365.0,
        contract_multiplier=1.0, ko_barrier=103.0, ko_rate=0.12,
        ki_barrier=KI_BARRIER, num_observations=1, is_reverse=False,
        ko_observation_dates=[OBS_DAYS / 365.0], ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=ki_times,
        rebate_rate=0.12, include_principal=False,
    )
    product.initial_date = VALUATION
    return product


def engine(lower_bound: float = 40.0, points: int = 801):
    return PDEEngine(params=PDEParams(grid=GridConfig(
        points=points, bounds=(lower_bound, 160.0), steps_per_day=64)))


def env_at(spot: float) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot), asset_name="CTRL"),
        vol_surface=FlatVolSurface(volatility=0.26),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=SignedDividendYield(0.14),
        valuation_date=VALUATION,
    )


@pytest.fixture(scope="module")
def product():
    return control_product()


def test_the_greeks_price_is_the_price_on_a_valuation_date_observation(product):
    """``calculate_greeks``'s price must be ``price()``'s, not the event trace's.

    Just above the barrier the interpolation bracket reaches a node the
    valuation-date event has rewritten, so reading the post-event vector
    marks the position at a blend of the two branches.
    """
    eng = engine()
    for offset in (1e-5, 5e-5, 2e-4, 5e-4, 1e-3, 3e-3):
        env = env_at(KI_BARRIER * (1.0 + offset))
        assert eng.calculate_greeks(product, env)["price"] == pytest.approx(
            eng.price(product, env), rel=1e-12, abs=1e-12
        ), f"price and greeks disagree {offset:.2%} above the barrier"


def test_delta_above_the_barrier_follows_the_surviving_branch(product):
    """The post-observation alive branch is smooth above the barrier.

    A surviving observation leaves the alive value unchanged above the
    barrier, so its delta there is the continuation's delta.  Differentiating
    across the event jump instead inflates it several fold.
    """
    eng = engine()
    near = eng.calculate_greeks(product, env_at(KI_BARRIER * 1.0002))["delta"]
    away = eng.calculate_greeks(product, env_at(KI_BARRIER * 1.005))["delta"]
    assert near == pytest.approx(away, rel=0.25), (
        f"delta {near:.6f} just above the barrier against {away:.6f} a half "
        "percent above it: the stencil is crossing the event jump"
    )


def test_delta_does_not_step_as_the_nearest_node_changes(product):
    """Delta must be a function of the spot, not of the nearest grid node.

    Snapping the stencil to the nearest node makes delta piecewise constant
    in spot, with a step wherever the nearest node changes.
    """
    eng = engine()
    spots = KI_BARRIER * np.array([1.02, 1.0202, 1.0204, 1.0206, 1.0208, 1.021])
    deltas = np.array([eng.calculate_greeks(product, env_at(s))["delta"] for s in spots])
    assert len(np.unique(deltas)) == deltas.size, (
        f"delta is constant across distinct spots (snapped to a node): {deltas}"
    )
    steps = np.abs(np.diff(deltas))
    assert steps.max() <= 4.0 * steps.mean(), f"delta steps between nodes: {deltas}"


def test_delta_does_not_move_with_the_domain_bound(product):
    """The lower bound places the nodes; it must not place the answer.

    Far from the barrier the value is smooth, so two domains that differ
    only in where their nodes fall must agree on delta.
    """
    spot = 105.0
    a = engine(lower_bound=40.0).calculate_greeks(product, env_at(spot))["delta"]
    b = engine(lower_bound=39.8).calculate_greeks(product, env_at(spot))["delta"]
    assert a == pytest.approx(b, rel=2e-3), (
        f"delta {a:.6f} on [40, 160] against {b:.6f} on [39.8, 160]"
    )


def aged_at(day: int):
    """The control product rolled forward to ``day``, as the replay ages it."""
    aged = control_product()
    aged.maturity = (OBS_DAYS - day) / 365.0
    aged.barrier_config = aged.barrier_config.time_shift(
        day / 365.0, VALUATION + timedelta(days=day), env_at(SPOT_0)
    )[0]
    return aged


def test_a_life_surface_interior_column_reads_the_branch_not_the_event_trace(product):
    """Every column of a life surface is somebody's valuation date.

    The solver already shields the valuation-date readout from the event it
    writes onto the grid.  An interior column carries a discrete observation
    too, so reading the event-projected column marks a path at a blend of
    the surviving and knocked-in branches.
    """
    eng = engine()
    surface = eng.solve_life_surface(
        product, env_at(SPOT_0),
        extra_times=tuple(d / 365.0 for d in range(1, OBS_DAYS)),
    )
    day = 25
    node = surface.step_of[day / 365.0]
    aged = aged_at(day)
    for offset in (2e-4, 1e-3, 5e-3):
        spot = KI_BARRIER * (1.0 + offset)
        read = float(np.interp(np.log(spot), surface.x, surface.v0[:, node]))
        fresh = eng.price(aged, env_at(spot))
        assert read == pytest.approx(fresh, rel=1e-2), (
            f"{offset:.2%} above the barrier: surface column {read:.6f} "
            f"against a fresh aged solve {fresh:.6f}"
        )
