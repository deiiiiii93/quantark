from datetime import datetime

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment


def test_snowball_mc_engine_event_stats_shapes():
    product = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=0.5,
        num_observations=4,
        ko_barrier=103.0,
        ko_rate=0.10,
        ki_barrier=75.0,
        include_principal=False,
    )
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime.now(),
    )

    engine = SnowballMCEngine(params=MCParams(num_paths=2000, time_steps=64, seed=7))
    stats = engine.calculate_event_stats(product, env)
    assert stats is not None

    assert stats.ko_times.shape == (4,)
    assert stats.ko_probability.shape == (4,)
    assert stats.survival_probability.shape == (4,)
    assert stats.expected_discounted_ko_cashflow.shape == (4,)
    assert stats.determination_times.shape == (5,)
    assert stats.payment_times.shape == (5,)
    assert stats.expected_discounted_cashflows.shape == (5,)
    assert stats.expected_undiscounted_cashflows.shape == (5,)

    assert 0.0 <= stats.ki_probability <= 1.0
    assert stats.pv == stats.pv  # not NaN

    # KO probabilities must sum to <= 1.0
    assert stats.ko_probability.sum() <= 1.0 + 1e-12
    # Survival probability is non-increasing
    assert all(
        stats.survival_probability[i] >= stats.survival_probability[i + 1] - 1e-12
        for i in range(3)
    )

    # PV reconciliation should be close (same simulation)
    assert abs(stats.reconciliation_error) < 1e-8
    assert abs(stats.pv - stats.expected_discounted_cashflows.sum()) < 1e-8
