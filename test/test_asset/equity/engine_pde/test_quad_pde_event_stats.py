from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment


def _make_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime.now(),
    )


def test_snowball_quad_event_stats_shapes_and_mass():
    product = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        num_observations=6,
        include_principal=False,
        ko_rate=0.10,
    )
    env = _make_env()
    engine = SnowballQuadEngine(params=QuadParams(grid_points=301))
    stats = engine.calculate_event_stats(product, env)
    assert stats is not None

    obs = stats.observations
    assert obs.times.shape == (6,)
    assert obs.probability.shape == (6,)
    assert obs.survival.shape == (6,)
    assert obs.discounted_cash.shape == (6,)
    # ledger: six knock-out rows, the maturity and the knocked-in settlement
    assert stats.ledger.determination_times.shape == (8,)
    assert stats.ledger.payment_times.shape == (8,)

    assert float(np.sum(obs.probability)) <= 1.0 + 1e-6
    assert all(obs.survival[i] >= obs.survival[i + 1] - 1e-8 for i in range(5))

    # PV decomposition from directly computed outcomes (T17: the maturity is its own column, not a residual)
    assert abs(stats.reconciliation_error) < 1e-8
    assert abs(stats.pv - stats.ledger.discounted_cash.sum()) < 1e-8


def test_snowball_pde_event_stats_delegates_to_quad():
    product = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        num_observations=4,
        include_principal=False,
        ko_rate=0.10,
    )
    env = _make_env()
    pde = SnowballPDESolver(params=PDEParams())
    stats = pde.calculate_event_stats(product, env)
    assert stats is not None
    assert stats.observations.times.shape == (4,)
    # four knock-out rows, the maturity row and the knock-in settlement row (unified event statistics T12)
    assert stats.ledger.determination_times.shape == (6,)
    assert stats.ledger.payment_times.shape == (6,)
    assert abs(stats.pv - stats.ledger.discounted_cash.sum()) < 1e-8

    # Basic sanity: probabilities are between 0 and 1 (tolerances allow small numerical drift).
    assert float(stats.observations.probability.min()) >= -1e-6
    assert float(stats.observations.probability.max()) <= 1.0 + 1e-6
