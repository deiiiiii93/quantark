"""Payment-aware event statistics reconcile to engine PV."""

import numpy as np
import pytest

from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import (
    PhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_pde_solver import (
    SnowballPDESolver,
)
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.engine.quad.phoenix_quad_engine import (
    PhoenixQuadEngine,
)
from quantark.asset.equity.engine.quad.snowball_quad_engine import (
    SnowballQuadEngine,
)
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.util.enum import PostKOScheduleMode
from test_asset.equity.product_other.test_structured_numerical_settlement import (
    LAG,
    _convention,
    _env,
    _pde_params,
    _phoenix,
    _snowball,
)


def _assert_aligned_cashflow_ledger(stats, env, *, abs_tolerance=1.0e-8):
    """The ledger (DESIGN 4.6) is the payment-aware view: cash to pay discounted at its own payment time sums
    to ``pv``."""
    assert isinstance(stats, EventStatistics)
    determination_times = np.asarray(stats.ledger.determination_times, dtype=float)
    payment_times = np.asarray(stats.ledger.payment_times, dtype=float)
    undiscounted = np.asarray(stats.ledger.cash_to_pay, dtype=float)
    discounted = np.asarray(stats.ledger.discounted_cash, dtype=float)

    assert determination_times.ndim == 1
    assert len(determination_times) == len(payment_times)
    assert len(payment_times) == len(undiscounted)
    assert len(discounted) == len(payment_times)
    assert len(discounted) > 0
    assert np.all(payment_times >= determination_times)
    np.testing.assert_allclose(
        undiscounted
        * np.asarray(
            [env.get_discount_factor(float(t)) for t in payment_times],
            dtype=float,
        ),
        discounted,
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    assert stats.pv == pytest.approx(
        float(discounted.sum()),
        abs=abs_tolerance,
    )


def _expected_life_from_determination_grid(stats) -> float:
    """Each outcome's probability times its determination time. Unified statistics: the maturity outcomes are
    their own rows (``maturity``, ``ki_settlement``), no longer ``1 - sum(KO)`` at the last KO time."""
    obs = stats.observations
    return float(
        np.dot(obs.probability, obs.times)
        + stats.maturity.probability * stats.maturity.time
        + stats.ki_settlement.probability * stats.ki_settlement.time
    )


@pytest.mark.parametrize(
    ("product_factory", "engine_factory"),
    [
        (
            _snowball,
            lambda: SnowballMCEngine(
                params=MCParams(num_paths=4096, time_steps=48, seed=19)
            ),
        ),
        (
            _phoenix,
            lambda: PhoenixMCEngine(
                params=MCParams(num_paths=4096, time_steps=48, seed=19)
            ),
        ),
    ],
)
def test_autocallable_probabilities_stay_on_determination_grid(
    product_factory,
    engine_factory,
):
    env = _env()
    immediate = engine_factory().calculate_event_stats(
        product_factory(),
        env,
    )
    delayed = engine_factory().calculate_event_stats(
        product_factory(_convention()),
        env,
    )

    _assert_aligned_cashflow_ledger(delayed, env)
    np.testing.assert_array_equal(
        delayed.observations.times,
        immediate.observations.times,
    )
    np.testing.assert_array_equal(
        delayed.observations.probability,
        immediate.observations.probability,
    )
    assert _expected_life_from_determination_grid(delayed) == pytest.approx(
        _expected_life_from_determination_grid(immediate)
    )
    assert delayed.expected_lifetime == pytest.approx(immediate.expected_lifetime)
    assert np.any(
        np.asarray(delayed.ledger.payment_times)
        > np.asarray(delayed.ledger.determination_times)
    )


def test_event_distribution_keeps_determination_and_payment_times_separate():
    env = _env()
    stats = SnowballMCEngine(
        params=MCParams(num_paths=4096, time_steps=48, seed=23)
    ).calculate_event_stats(
        _snowball(_convention()),
        env,
    )

    distribution = EventDistribution.from_statistics(stats)
    ko_payment_times = np.asarray(
        distribution.payment_times[EventType.KO],
        dtype=float,
    )

    np.testing.assert_array_equal(distribution.stream_times[EventType.KO], stats.observations.times)
    np.testing.assert_allclose(
        ko_payment_times,
        np.asarray(stats.observations.times) + LAG,
    )

    leg = FixedPayoffLeg(
        direction=LegDirection.BUYER_RECEIVES,
        amount=1.0,
        trigger=PaymentTrigger.AT_KO,
    )
    expected = float(
        np.sum(
            np.asarray(stats.observations.probability)
            * np.asarray(
                [
                    env.get_discount_factor(float(t))
                    for t in ko_payment_times
                ]
            )
        )
    )
    assert leg.value(distribution, env, 0.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("product_factory", "engine_factory"),
    [
        (
            _snowball,
            lambda: SnowballPDESolver(params=_pde_params()),
        ),
        (
            _snowball,
            lambda: SnowballQuadEngine(
                QuadParams(grid_points=301)
            ),
        ),
        (
            _phoenix,
            lambda: PhoenixPDESolver(params=_pde_params()),
        ),
        (
            _phoenix,
            lambda: PhoenixQuadEngine(
                QuadParams(grid_points=301)
            ),
        ),
    ],
)
def test_native_structured_stats_reconcile_on_payment_grid(
    product_factory,
    engine_factory,
):
    env = _env()
    stats = engine_factory().calculate_event_stats(
        product_factory(_convention()),
        env,
    )

    _assert_aligned_cashflow_ledger(stats, env)
    assert np.any(
        np.asarray(stats.ledger.payment_times)
        > np.asarray(stats.ledger.determination_times)
    )


def test_ko_reset_stats_align_pre_post_and_terminal_cashflows():
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    product.settlement_convention = _convention()
    env = _env()
    stats = KOResetSnowballQuadEngine(
        QuadParams(grid_points=301)
    ).calculate_event_stats(product, env)

    _assert_aligned_cashflow_ledger(stats, env)
    # QUAD V1 since T17: both schedules' knock-out rows, the maturity (the first schedule's end) and the knocked-in
    # settlement (maturity), every one settling LAG after its determination
    assert len(stats.ledger.determination_times) == stats.observations.size + 2
    np.testing.assert_allclose(stats.ledger.payment_times, stats.ledger.determination_times + LAG)


def test_dcn_stats_use_observation_life_and_explicit_payment_dates():
    product = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    result = DCNMCEngine(
        num_paths=1024,
        seed=29,
        use_sobol=False,
    ).price_detailed(product, env)
    stats = result.event_stats

    # T11: an EventStatistics; its ledger is the payment-aware view (DESIGN 4.6), on the DCN axis (spec D12)
    _assert_aligned_cashflow_ledger(stats, env)
    assert stats.pv == result.pv
    assert stats.expected_lifetime == result.expected_life_years
    assert stats.ledger.determination_dates is not None
    assert stats.ledger.payment_dates is not None
    assert any(
        payment > determination
        for determination, payment in zip(
            stats.ledger.determination_dates,
            stats.ledger.payment_dates,
        )
    )
