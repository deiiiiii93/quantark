"""Snowball PDE must price a product that is ALREADY knocked in.

Regression cover for a grid/alignment disagreement found by Gate G3 of the
vol-model backtest: ``_ki_monitor_times`` drops the interior daily-KI nodes
once ``_otc_lifecycle_knocked_in`` is set (monitoring is moot), so the time
grid is built without them - but ``_build_grids`` still demanded a grid node
for every KI observation and raised

    KI observation time 0.0027... does not align with PDE time grid

That made the first reprice after a knock-in fail for any discretely-monitored
snowball, which is exactly what an OTC replay does when the underlying breaches
the KI barrier mid-trade.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

ACT = 365.0


def _env(spot=100.0, vol=0.22, rate=0.025, q=0.01):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot), asset_name="TEST"),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=q),
        valuation_date=datetime(2026, 1, 5),
    )


def _daily_ki_snowball(maturity=1.0, n_ko=12):
    """Snowball with DISCRETE daily KI monitoring (the OTC replay pattern)."""
    n_days = int(round(maturity * ACT))
    ki_times = [d / ACT for d in range(1, n_days + 1)]
    ko_times = [(i + 1) / n_ko * maturity for i in range(n_ko)]
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=maturity,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=n_ko,
        ko_observation_dates=ko_times,
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=ki_times,
        rebate_rate=0.15,
        include_principal=False,
    )


def _price(product, env, **params):
    return float(SnowballPDESolver(params=PDEParams(**params)).price(product, env))


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------

def test_already_knocked_in_product_prices():
    """The exact Gate G3 failure: set the lifecycle flag and reprice."""
    product = _daily_ki_snowball()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    price = _price(product, _env())
    assert np.isfinite(price)


@pytest.mark.parametrize("maturity", [0.5, 1.0, 2.0])
def test_knocked_in_prices_across_maturities(maturity):
    product = _daily_ki_snowball(maturity=maturity)
    setattr(product, "_otc_lifecycle_knocked_in", True)
    assert np.isfinite(_price(product, _env()))


def test_not_knocked_in_still_prices():
    """The alive path must be untouched by the fix."""
    product = _daily_ki_snowball()
    assert np.isfinite(_price(product, _env()))


# ---------------------------------------------------------------------------
# The invariant the fix restores
# ---------------------------------------------------------------------------

def test_grid_and_alignment_agree_on_ki_nodes():
    """Whoever builds the grid and whoever checks it must use ONE predicate.

    If ``_time_grid_spec`` drops the KI monitor times but ``_build_grids``
    still demands aligned KI nodes, pricing raises.  Assert they agree in
    every regime rather than only that pricing happens to work.
    """
    solver = SnowballPDESolver(params=PDEParams())
    env = _env()
    for knocked_in in (False, True):
        product = _daily_ki_snowball()
        setattr(product, "_otc_lifecycle_knocked_in", knocked_in)
        tau = float(product.get_maturity(env))
        solver._configure_bgk(product, env, 0.22, tau)
        spec = solver._time_grid_spec(product, tau)
        nodes_in_grid = solver._ki_nodes_in_grid(product)
        assert bool(spec.monitor_times) == nodes_in_grid, (
            f"knocked_in={knocked_in}: grid has "
            f"{len(spec.monitor_times)} KI monitor times but the alignment "
            f"predicate says nodes_in_grid={nodes_in_grid}"
        )


def test_knocked_in_drops_ki_monitor_nodes():
    solver = SnowballPDESolver(params=PDEParams())
    env = _env()
    product = _daily_ki_snowball()
    tau = float(product.get_maturity(env))
    solver._configure_bgk(product, env, 0.22, tau)
    assert solver._ki_nodes_in_grid(product)
    assert solver._ki_monitor_times(product, tau)

    setattr(product, "_otc_lifecycle_knocked_in", True)
    assert not solver._ki_nodes_in_grid(product)
    assert solver._ki_monitor_times(product, tau) == []


# ---------------------------------------------------------------------------
# The economics must still be right, not merely finite
# ---------------------------------------------------------------------------

def test_knocked_in_is_worth_less_than_alive():
    """A knocked-in snowball has lost its downside protection.

    Same contract, same market; only the KI state differs.  For a standard
    (non-reverse) snowball the knocked-in value function carries the short
    put, so it must be strictly cheaper than the not-yet-knocked-in one.
    """
    env = _env(spot=90.0)  # below the strike, where the difference bites
    alive = _daily_ki_snowball()
    knocked = _daily_ki_snowball()
    setattr(knocked, "_otc_lifecycle_knocked_in", True)
    assert _price(knocked, env) < _price(alive, env)


def test_knocked_in_price_matches_a_ki_free_equivalent():
    """Pricing a knocked-in product must equal pricing its V1 surface directly.

    A product whose KI barrier is already breached is economically the
    same as one whose KI is certain.  Dropping the KI grid nodes must not
    change the answer, so compare against a continuous-KI product with the
    barrier set above spot (immediate, certain knock-in) - a construction
    that never used the discrete KI node path at all.
    """
    env = _env(spot=90.0)
    knocked = _daily_ki_snowball()
    setattr(knocked, "_otc_lifecycle_knocked_in", True)

    certain = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=95.0,  # spot 90 < 95 -> already in, continuously monitored
        num_observations=12,
        ki_continuous=True,
        rebate_rate=0.15,
        include_principal=False,
    )
    got, want = _price(knocked, env), _price(certain, env)
    assert abs(got - want) < 5e-3 * max(1.0, abs(want)), (got, want)


# ---------------------------------------------------------------------------
# Subclasses inherit the fix (they delegate to super()._build_grids)
# ---------------------------------------------------------------------------

def test_phoenix_already_knocked_in_prices():
    """PhoenixPDESolver shares SnowballPDESolver._build_grids, so it shared the bug."""
    from quantark.asset.equity.engine.pde import PhoenixPDESolver
    from quantark.asset.equity.product.option.observation_schedule import (
        ObservationRecord,
        ObservationSchedule,
    )
    from quantark.asset.equity.product.option.phoenix_helpers import (
        create_standard_phoenix,
    )

    n_days = int(round(1.0 * ACT))
    ki_times = [d / ACT for d in range(1, n_days + 1)]
    product = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=[75.0] * len(ki_times),
        coupon_barrier=85.0,
        coupon_rate=0.01,
        num_observations=12,
        memory_coupon=False,
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=ObservationSchedule(
            records=[
                ObservationRecord(observation_time=t, barrier=75.0) for t in ki_times
            ]
        ),
    )
    setattr(product, "_otc_lifecycle_knocked_in", True)
    price = float(PhoenixPDESolver(params=PDEParams()).price(product, _env()))
    assert np.isfinite(price)
