from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType


KO_DATES = [0.25, 0.5, 0.75, 1.0]
KI_BARRIER = 75.0
PARACHUTE_KO_BARRIERS = [105.0, 104.0, 103.0, KI_BARRIER]


def _make_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime(2024, 1, 1),
    )


def _make_product(
    *,
    ko_barrier=None,
    ki_barrier=None,
    lifecycle_knocked_in: bool = False,
) -> SnowballOption:
    if ko_barrier is None:
        ko_barrier = [105.0, 104.0, 103.0, 102.0]

    barrier_config = BarrierConfig(
        ko_barrier=ko_barrier,
        ko_rate=0.10,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=KO_DATES,
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=KO_DATES if ki_barrier is not None else None,
        ki_continuous=False,
    )
    product = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        accrual_config=AccrualConfig(is_annualized=True),
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )
    if lifecycle_knocked_in:
        setattr(product, "_otc_lifecycle_knocked_in", True)
    return product


def _engine_factories():
    return {
        "quad": lambda: SnowballQuadEngine(
            QuadParams(grid_points=301, event_smoothing_cells=0)
        ),
        "pde": lambda: SnowballPDESolver(
            PDEParams()
        ),
        "mc": lambda: SnowballMCEngine(
            MCParams(num_paths=20_000, time_steps=64, seed=123)
        ),
    }


def _ki_view(stats):
    """``(P(KI before any KO), KI table times, KI table probability, KI table survival)`` off ``EventStatistics``
    (every producer since T17)."""
    assert isinstance(stats, EventStatistics)
    table = stats.ki_table
    return stats.knock_in.probability, table.times, table.probability, table.survival


def _assert_lifecycle_ki_event_stats(stats) -> None:
    """A note knocked in before valuation: KI is certain and happened at valuation (one KI row at t=0, nothing left
    alive in the KI table), every path still alive at maturity settles knocked in (``ki_settlement`` is the
    no-knock-out survival -- also on a contract whose KI barrier was dropped), and the ledger reconciles to ``pv``."""
    assert isinstance(stats, EventStatistics)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    np.testing.assert_array_equal(stats.ki_table.survival, [0.0])
    assert stats.ki_settlement.probability == pytest.approx(1.0 - float(np.sum(stats.observations.probability)),
                                                            abs=7e-9)
    assert stats.pv == pytest.approx(float(np.sum(stats.ledger.discounted_cash)), abs=1e-8)


def test_normal_ki_products_do_not_report_forced_lifecycle_state():
    env = _make_env()
    product = _make_product(ki_barrier=KI_BARRIER, lifecycle_knocked_in=False)

    for engine_factory in _engine_factories().values():
        engine = engine_factory()
        price = engine.price(product, env)
        stats = engine.calculate_event_stats(product, env)

        assert np.isfinite(price)
        assert stats is not None
        probability, times, _, _ = _ki_view(stats)
        assert 0.0 <= probability < 1.0
        assert times.shape != (1,) or times[0] != pytest.approx(0.0)


def test_no_ki_without_lifecycle_remains_v0_product():
    env = _make_env()
    product = _make_product(ki_barrier=None, lifecycle_knocked_in=False)

    for name, engine_factory in _engine_factories().items():
        engine = engine_factory()
        price = engine.price(product, env)
        stats = engine.calculate_event_stats(product, env)

        assert np.isfinite(price)
        assert stats is not None
        probability, times, _, _ = _ki_view(stats)
        assert probability == pytest.approx(0.0)
        assert times.size == 0
        if name == "mc":
            result = engine.get_last_result()
            assert result.v1_probability == pytest.approx(0.0)
            assert result.v0_probability > 0.0


def test_lifecycle_ki_without_future_ki_terms_matches_retained_ki_barrier():
    env = _make_env()
    collapsed = _make_product(ki_barrier=None, lifecycle_knocked_in=True)
    retained = _make_product(ki_barrier=KI_BARRIER, lifecycle_knocked_in=True)

    for engine_factory in _engine_factories().values():
        collapsed_price = engine_factory().price(collapsed, env)
        retained_price = engine_factory().price(retained, env)

        assert collapsed_price == pytest.approx(retained_price, abs=0.25)


def test_lifecycle_ki_event_stats_known_state_without_ki_barrier():
    env = _make_env()
    product = _make_product(ki_barrier=None, lifecycle_knocked_in=True)

    for name, engine_factory in _engine_factories().items():
        engine = engine_factory()
        engine.price(product, env)
        stats = engine.calculate_event_stats(product, env)

        assert stats is not None
        _assert_lifecycle_ki_event_stats(stats)
        if name == "mc":
            result = engine.get_last_result()
            assert result.v0_probability == pytest.approx(0.0)
            assert result.v1_probability == pytest.approx(1.0 - result.ko_probability)


def test_parachute_lifecycle_collapse_matches_retained_ki_barrier():
    env = _make_env()
    collapsed = _make_product(
        ko_barrier=PARACHUTE_KO_BARRIERS,
        ki_barrier=None,
        lifecycle_knocked_in=True,
    )
    retained = _make_product(
        ko_barrier=PARACHUTE_KO_BARRIERS,
        ki_barrier=KI_BARRIER,
        lifecycle_knocked_in=True,
    )

    for engine_factory in _engine_factories().values():
        collapsed_price = engine_factory().price(collapsed, env)
        retained_price = engine_factory().price(retained, env)

        assert collapsed_price == pytest.approx(retained_price, abs=0.25)
