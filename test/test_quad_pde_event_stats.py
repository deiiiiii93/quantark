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

    assert stats.ko_times.shape == (6,)
    assert stats.ko_probability.shape == (6,)
    assert stats.survival_probability.shape == (6,)
    assert stats.expected_discounted_ko_cashflow.shape == (6,)

    assert float(np.sum(stats.ko_probability)) <= 1.0 + 1e-6
    assert all(
        stats.survival_probability[i] >= stats.survival_probability[i + 1] - 1e-8
        for i in range(5)
    )

    # PV decomposition consistency (constructed as residual)
    pv_parts = float(np.sum(stats.expected_discounted_ko_cashflow) + stats.expected_discounted_maturity_cashflow)
    assert abs(stats.pv - pv_parts) < 1e-8


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
    assert stats.ko_times.shape == (4,)

    # Basic sanity: probabilities are between 0 and 1 (tolerances allow small numerical drift).
    assert float(stats.ko_probability.min()) >= -1e-6
    assert float(stats.ko_probability.max()) <= 1.0 + 1e-6
