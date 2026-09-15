"""Independent coupon event and payment controls for Phoenix QUAD V2."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from scipy.stats import norm, qmc, t as student_t

from test_quad_v2_engine import environment
from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2
from quantark.asset.equity.param import QuadV2Params
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.lifecycle.cashflows import RealizedCashflow, ValuationPoint
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.cashleg.event_distribution import EventType
from quantark.execution.context import default_context
from quantark.execution.contracts import PricingRequest, PricingOperation, OutputKind
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.util.enum import CouponPayType, ObservationType, ProtectionType


def phoenix(
    times=(0.25, 0.5, 0.75),
    ko=200.0,
    coupon=80.0,
    memory=False,
    pay_type=CouponPayType.INSTANT,
    reverse=False,
):
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        is_reverse=reverse,
        barrier_config=BarrierConfig(
            ko_barrier=ko,
            ko_rate=0.0,
            ko_observation_dates=list(times),
            ki_barrier=None,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=coupon,
            coupon_rate=0.12,
            fixed_coupon_year_fraction=0.25,
            memory_coupon=memory,
            coupon_pay_type=pay_type,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
    )


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
@pytest.mark.parametrize("memory", [False, True])
def test_coupon_memory_and_early_termination_payment_ledger(memory, pay_type):
    p = phoenix(
        times=(0.2, 0.4, 0.6, 0.8),
        ko=[200.0, 200.0, 90.0, 200.0],
        coupon=[110.0, 90.0, 90.0, 90.0],
        memory=memory,
        pay_type=pay_type,
    )
    env = environment(vol=0.0, r=0.1, q=0.1)
    engine = PhoenixQuadEngineV2()
    stats = engine.calculate_event_stats(p, env)
    assert isinstance(stats, PhoenixEventStats)
    np.testing.assert_allclose(stats.coupon_probability, [0, 1, 1, 0], atol=1e-13)
    np.testing.assert_allclose(stats.ko_probability, [0, 0, 1, 0], atol=1e-13)
    first = 6.0 if memory else 3.0
    first_payment = 0.4 if pay_type is CouponPayType.INSTANT else 0.6
    coupons = np.array([0, first * np.exp(-0.1 * first_payment), 3 * np.exp(-0.06), 0])
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow, coupons, atol=1e-12
    )
    expected = 100 * np.exp(-0.06) + coupons.sum()
    assert stats.pv == pytest.approx(expected, abs=1e-12)
    assert engine.price(p, env) == pytest.approx(expected, abs=1e-12)
    assert engine.price_components(p, env)["coupon"] == pytest.approx(
        coupons.sum(), abs=1e-12
    )
    assert stats.expected_discounted_cashflows.sum() == pytest.approx(
        expected, abs=1e-12
    )
    assert abs(stats.reconciliation_error) < 1e-12
    # In deferred mode both earned coupons settle with the actual early KO.
    expected_at_ko = 103.0 + (first if pay_type is CouponPayType.EXPIRY else 0.0)
    assert stats.expected_undiscounted_cashflows[
        np.isclose(stats.payment_times, 0.6)
    ].sum() == pytest.approx(expected_at_ko, abs=1e-12)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_single_coupon_matches_lognormal_probability_and_payment_time(
    reverse, pay_type
):
    p = phoenix(
        times=(0.5,),
        ko=85.0 if reverse else 110.0,
        coupon=105.0 if reverse else 95.0,
        reverse=reverse,
        pay_type=pay_type,
    )
    env = environment(vol=0.2, r=0.08, q=0.02)
    stats = PhoenixQuadEngineV2().calculate_event_stats(p, env)

    def probability(barrier):
        z = (np.log(barrier / 100) - (0.08 - 0.02 - 0.2**2 / 2) * 0.5) / (
            0.2 * np.sqrt(0.5)
        )
        return norm.cdf(z if reverse else -z)

    p_ko = probability(p.barrier_config.ko_barrier)
    p_coupon = probability(p.coupon_config.coupon_barrier)
    expected_coupon = 3 * (
        p_coupon * np.exp(-0.04)
        if pay_type is CouponPayType.INSTANT
        else p_ko * np.exp(-0.04) + (p_coupon - p_ko) * np.exp(-0.08)
    )
    np.testing.assert_allclose(stats.coupon_probability, [p_coupon], atol=1e-13)
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow, [expected_coupon], atol=1e-11
    )
    assert stats.pv == pytest.approx(
        100 * (p_ko * np.exp(-0.04) + (1 - p_ko) * np.exp(-0.08)) + expected_coupon,
        abs=1e-10,
    )
    assert stats.expected_discounted_cashflows.sum() == pytest.approx(
        stats.pv, abs=1e-10
    )


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_framework_distribution_and_lifecycle_preserve_coupon_ownership(pay_type):
    p = phoenix(
        times=(0.0, 0.25, 0.5),
        coupon=[110.0, 90.0, 110.0],
        memory=True,
        pay_type=pay_type,
    )
    env = environment(vol=0.0, r=0.1, q=0.1)
    state = AutocallableLifecycleState(
        coupon_memory_count=2,
        valuation_point=ValuationPoint(date=env.valuation_date),
    )
    state.ledger.register(
        RealizedCashflow(
            cashflow_id="known-coupon",
            event_type=LifecycleEventType.COUPON,
            amount=2.0,
            determination_date=env.valuation_date,
            payment_date=env.valuation_date + timedelta(days=10),
        )
    )
    engine = PhoenixQuadEngineV2()
    request = PricingRequest(
        p,
        env,
        operation=PricingOperation.EVENT_STATS,
        outputs=frozenset(
            {OutputKind.PV, OutputKind.EVENT_STATS, OutputKind.CASHFLOWS}
        ),
        operation_options=(("event_phase", "after"),),
        lifecycle_state=state,
    )
    stats = engine.execute(request, default_context(environ={})).value
    np.testing.assert_array_equal(stats.ko_times, [0.25, 0.5])
    np.testing.assert_allclose(stats.coupon_probability, [1.0, 0.0], atol=1e-13)
    pay_time = 0.25 if pay_type is CouponPayType.INSTANT else 1.0
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow,
        [9 * np.exp(-0.1 * pay_time), 0],
        atol=1e-12,
    )
    expected = (
        100 * np.exp(-0.1) + 9 * np.exp(-0.1 * pay_time) + 2 * np.exp(-0.1 * 10 / 365)
    )
    assert stats.pv == pytest.approx(expected, abs=1e-12)
    assert stats.expected_discounted_cashflows.sum() == pytest.approx(
        expected, abs=1e-12
    )
    result = engine.price_with_events(
        p, env, lifecycle_state=state, event_phase="after"
    )
    assert result.npv == pytest.approx(expected, abs=1e-12)
    dist = result.event_distribution
    np.testing.assert_allclose(dist.probabilities[EventType.COUPON], [1, 0], atol=1e-13)
    assert dist.payment_times_for(EventType.MATURITY_NO_KO) == 1.0
    if pay_type is CouponPayType.EXPIRY:
        assert stats.coupon_payment_is_path_dependent
        assert EventType.COUPON not in dist.payment_times
        with pytest.raises(NotImplementedError, match="path-dependent"):
            dist.payment_times_for(EventType.COUPON)
    else:
        np.testing.assert_allclose(
            dist.payment_times_for(EventType.COUPON), [0.25, 0.5]
        )


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_delayed_coupon_cashflows_use_nonflat_payment_discounts(pay_type):
    p = phoenix(times=(0.5,), ko=110.0, coupon=95.0, pay_type=pay_type)
    p.settlement_convention = SettlementConvention(
        lag=0.1, lag_unit=SettlementLagUnit.YEAR_FRACTION
    )
    env = environment(vol=0.2, q=0.02)
    env.rate_curve = LinearRateCurve(pillars=[(0.0, 0.02), (0.5, 0.04), (1.1, 0.12)])
    df = env.get_discount_factor
    mean = -np.log(df(0.5)) - 0.02 * 0.5 - 0.2**2 * 0.5 / 2
    p_ko, p_coupon = norm.cdf(
        (mean - np.log(np.array([110, 95]) / 100)) / (0.2 * np.sqrt(0.5))
    )
    expected_coupon = 3 * (
        p_coupon * df(0.6)
        if pay_type is CouponPayType.INSTANT
        else p_ko * df(0.6) + (p_coupon - p_ko) * df(1.1)
    )
    engine = PhoenixQuadEngineV2()
    stats = engine.calculate_event_stats(p, env)
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow, [expected_coupon], atol=1e-11
    )
    expected = 100 * (p_ko * df(0.6) + (1 - p_ko) * df(1.1)) + expected_coupon
    assert stats.pv == pytest.approx(expected, abs=1e-10)
    assert engine.price(p, env) == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(
        stats.expected_discounted_cashflows,
        stats.expected_undiscounted_cashflows
        * np.array([df(t) for t in stats.payment_times]),
        atol=1e-12,
    )
    np.testing.assert_allclose(stats.payment_times, [0.6, 0.6, 1.1, 1.1], atol=1e-13)


def test_zero_coupon_rate_still_reports_triggers_and_initial_ki_disables_ko():
    p = phoenix(times=(0.25, 0.5), ko=90.0)
    p.barrier_config = replace(
        p.barrier_config,
        ki_barrier=80.0,
        disable_ko_after_ki=True,
        ki_observation_dates=[0.25, 0.5],
    )
    p.coupon_config = replace(p.coupon_config, coupon_rate=0.0)
    state = AutocallableLifecycleState(knocked_in=True)
    stats = PhoenixQuadEngineV2().calculate_event_stats(
        p, environment(vol=0, r=0, q=0), lifecycle_state=state
    )
    np.testing.assert_allclose(stats.coupon_probability, [1, 1], atol=1e-13)
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow, [0, 0], atol=1e-13
    )
    np.testing.assert_allclose(stats.ko_probability, [0, 0], atol=1e-13)
    assert stats.ki_survive_knocked_in_probability == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_ko_releases_missed_memory_even_when_current_coupon_does_not_trigger(pay_type):
    p = phoenix(
        times=(0.25, 0.5),
        ko=[200.0, 90.0],
        coupon=110.0,
        memory=True,
        pay_type=pay_type,
    )
    engine = PhoenixQuadEngineV2()
    env = environment(vol=0, r=0, q=0)
    stats = engine.calculate_event_stats(p, env)
    np.testing.assert_allclose(stats.coupon_probability, [0, 0], atol=1e-13)
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow, [0, 3], atol=1e-13
    )
    assert stats.pv == pytest.approx(103, abs=1e-12)
    assert engine.price_components(p, env)["coupon"] == pytest.approx(3, abs=1e-12)


@pytest.mark.parametrize("pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY])
def test_coupon_threshold_equality_at_valuation_and_maturity(pay_type):
    p = phoenix(times=(0.0, 1.0), coupon=100.0, pay_type=pay_type)
    env = environment(vol=0, r=0.1, q=0.1)
    engine = PhoenixQuadEngineV2()
    stats = engine.calculate_event_stats(p, env)
    np.testing.assert_allclose(stats.coupon_probability, [1, 1], atol=1e-13)
    expected = [
        3 * (1 if pay_type is CouponPayType.INSTANT else np.exp(-0.1)),
        3 * np.exp(-0.1),
    ]
    np.testing.assert_allclose(
        stats.expected_discounted_coupon_cashflow, expected, atol=1e-12
    )
    assert stats.pv == pytest.approx(100 * np.exp(-0.1) + sum(expected), abs=1e-12)
    after = engine.calculate_event_stats(p, env, event_phase="after")
    np.testing.assert_allclose(
        after.expected_discounted_coupon_cashflow, expected[1:], atol=1e-12
    )


def test_deferred_coupon_channels_agree_across_direct_and_fft_backends():
    p = phoenix(
        times=(0.25, 0.5, 1.0),
        ko=105,
        coupon=[110, 90, 105],
        memory=True,
        pay_type=CouponPayType.EXPIRY,
    )
    env = environment()
    results = []
    for backend in ("direct", "fft"):
        engine = PhoenixQuadEngineV2(QuadV2Params(backend=backend))
        stats = engine.calculate_event_stats(p, env)
        assert engine._prepared.diagnostics["backends"] == (backend,)
        results.append(
            np.r_[
                stats.pv,
                stats.coupon_probability,
                stats.expected_discounted_coupon_cashflow,
                stats.expected_discounted_cashflows,
            ]
        )
    np.testing.assert_allclose(results[0], results[1], atol=1e-10, rtol=0)


def test_batched_coupon_stats_preserve_each_spots_probabilities_and_cashflows():
    p = phoenix(times=(0.5,), ko=110, coupon=95, pay_type=CouponPayType.EXPIRY)
    env = environment(vol=0.2, r=0.08, q=0.02)
    spots = np.array([99.0, 100.0, 101.0])
    stats = PhoenixQuadEngineV2().calculate_event_stats_at_spots(p, env, spots)
    mean = (0.08 - 0.02 - 0.2**2 / 2) * 0.5
    p_ko = norm.cdf((np.log(spots / 110) + mean) / np.sqrt(0.02))
    p_coupon = norm.cdf((np.log(spots / 95) + mean) / np.sqrt(0.02))
    expected_coupon = 3 * (p_ko * np.exp(-0.04) + (p_coupon - p_ko) * np.exp(-0.08))
    np.testing.assert_allclose(
        [s.coupon_probability[0] for s in stats], p_coupon, atol=1e-13
    )
    np.testing.assert_allclose(
        [s.expected_discounted_coupon_cashflow[0] for s in stats],
        expected_coupon,
        atol=1e-11,
    )
    for s in stats:
        assert s.expected_discounted_cashflows.sum() == pytest.approx(s.pv, abs=1e-10)


def _independent_qmc(memory, pay_type, reverse, continuous, seed):
    """Pathwise fixed-coupon note; no product/engine/compiler calls.

    8 independent scrambles of 65,536 paths form the Student-t interval used
    below. Continuous KI uses exact conditional GBM bridge crossing draws.
    """
    times = np.array([0.25, 0.5, 0.75, 1.0])
    samples = qmc.Sobol(8, scramble=True, seed=seed).random_base2(16)
    increments = (0.08 - 0.02 - 0.2**2 / 2) * 0.25 + 0.1 * norm.ppf(samples[:, :4])
    logs = np.log(100) + np.cumsum(increments, axis=1)
    alive = np.ones(len(samples), dtype=bool)
    hit = np.zeros(len(samples), dtype=bool)
    missed = np.zeros(len(samples))
    earned = np.zeros((len(samples), 3))
    termination = np.ones(len(samples))
    coupon_prob, ko_prob = np.zeros(3), np.zeros(3)
    barriers = np.array([100.0, 110.0, 100.0] if reverse else [100.0, 90.0, 100.0])
    sign = -1 if reverse else 1
    barrier = np.log(120 if reverse else 80)
    for i in range(4):
        ki = sign * (logs[:, i] - barrier) <= 0
        if continuous:
            previous = np.log(100) if i == 0 else logs[:, i - 1]
            d0, d1 = sign * (previous - barrier), sign * (logs[:, i] - barrier)
            bridge = np.exp(-2 * np.maximum(d0, 0) * np.maximum(d1, 0) / 0.01)
            ki |= samples[:, 4 + i] < bridge
        hit |= alive & ki
        if i == 3:
            break
        coupon_hit = alive & (sign * (logs[:, i] - np.log(barriers[i])) >= 0)
        ko = alive & ~hit & (sign * (logs[:, i] - np.log(92 if reverse else 108)) >= 0)
        coupon_prob[i] = coupon_hit.mean()
        ko_prob[i] = ko.mean()
        earned[coupon_hit, i] = 3 + missed[coupon_hit]
        # Contractual KO pays missed memory even if the current coupon misses.
        earned[ko & ~coupon_hit, i] = missed[ko & ~coupon_hit]
        if memory:
            missed[alive & ~coupon_hit & ~ko] += 3
            missed[coupon_hit | ko] = 0
        termination[ko] = times[i]
        alive &= ~ko
    payment = (
        times[:3][None, :]
        if pay_type is CouponPayType.INSTANT
        else termination[:, None]
    )
    discounted = earned * np.exp(-0.08 * payment)
    coupon_cash = discounted.mean(axis=0)
    ko_cash = 100 * np.exp(-0.08 * times[:3]) * ko_prob
    terminal_cash = 100 * np.exp(-0.08) * alive.mean()
    coupon_ledger = np.zeros(4)
    if pay_type is CouponPayType.INSTANT:
        coupon_ledger[:3] = coupon_cash
    else:
        for i, time in enumerate(times):
            coupon_ledger[i] = np.mean(discounted.sum(axis=1) * (termination == time))
    return np.r_[
        coupon_prob,
        ko_prob,
        coupon_cash,
        coupon_ledger,
        terminal_cash,
        (hit & alive).mean(),
        ko_cash.sum() + terminal_cash + coupon_cash.sum(),
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "memory,pay_type,reverse,continuous",
    [
        (False, CouponPayType.INSTANT, False, False),
        (True, CouponPayType.INSTANT, False, False),
        (True, CouponPayType.EXPIRY, False, False),
        (True, CouponPayType.EXPIRY, True, True),
    ],
)
def test_multi_event_coupon_stats_against_independent_qmc(
    memory, pay_type, reverse, continuous
):
    p = phoenix(
        ko=92 if reverse else 108,
        coupon=[100, 110, 100] if reverse else [100, 90, 100],
        memory=memory,
        pay_type=pay_type,
        reverse=reverse,
    )
    p.barrier_config = replace(
        p.barrier_config,
        ki_barrier=120 if reverse else 80,
        ki_continuous=continuous,
        disable_ko_after_ki=True,
        ki_observation_type=ObservationType.CONTINUOUS
        if continuous
        else ObservationType.DISCRETE,
        ki_observation_dates=[0.25, 0.5, 0.75, 1.0],
    )
    p.payoff_config = replace(p.payoff_config, protection_type=ProtectionType.FULL)
    env = environment(vol=0.2, r=0.08, q=0.02)
    estimates = np.array(
        [
            _independent_qmc(memory, pay_type, reverse, continuous, 970 + i)
            for i in range(8)
        ]
    )
    mean = estimates.mean(axis=0)
    ci = student_t.ppf(0.975, 7) * estimates.std(axis=0, ddof=1) / np.sqrt(8)
    # A CI must itself be precise enough to make the comparison informative.
    assert np.max(ci[:6]) < 0.002
    assert ci[-1] < 0.02
    levels = []
    for order, cells in [(6, 1.5), (8, 2.0), (10, 3.0)]:
        engine = PhoenixQuadEngineV2(QuadV2Params(order=order, cells_per_sd=cells))
        stats = engine.calculate_event_stats(p, env)
        values = engine._prepared.evaluate([100])["components"]
        terminal_coupon = float(
            values["terminal_alive_coupon"][0] + values["terminal_ki_coupon"][0]
        )
        levels.append(
            np.r_[
                stats.coupon_probability,
                stats.ko_probability,
                stats.expected_discounted_coupon_cashflow,
                stats.expected_discounted_cashflows[3:6],
                terminal_coupon,
                stats.expected_discounted_maturity_cashflow,
                stats.ki_survive_knocked_in_probability,
                stats.pv,
            ]
        )
        assert stats.expected_discounted_cashflows.sum() == pytest.approx(
            stats.pv, abs=1e-10
        )
        assert stats.expected_discounted_coupon_cashflow.sum() == pytest.approx(
            float(values["coupon"][0]), abs=1e-10
        )
    np.testing.assert_allclose(levels[-1], levels[-2], atol=2e-5, rtol=0)
    # Simultaneous checks use 99.9% intervals across the many correlated rows.
    simultaneous_ci = (
        student_t.ppf(0.9995, 7) * estimates.std(axis=0, ddof=1) / np.sqrt(8)
    )
    assert np.all(np.abs(levels[-1] - mean) <= simultaneous_ci + 2e-5), (
        levels[-1] - mean,
        simultaneous_ci,
    )
