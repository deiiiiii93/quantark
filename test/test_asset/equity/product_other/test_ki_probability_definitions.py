"""Cross-engine KI-probability definition tests.

Background (investigation 2026-07-01): engines used to disagree on "the" KI probability -- one reported
P(KI ever AND never KO), another P(KI ever). The gap was purely definitional (= P(KI AND KO)), not a
discretization or Brownian-bridge error.

Since the unified event statistics (T9 Snowball, T10 Phoenix MC; T12/T13 PDE; T16 QUAD V2; T17 QUAD V1) every
engine reports two unambiguous fields (DESIGN §4.5):

- ``knock_in.probability``       = P(KI before any KO) (a knock-in after a knock-out is no knock-in)
- ``knock_in.probability_no_ko`` = P(KI AND never KO) (== ``ki_settlement.probability``)

These tests pin the engines to the same two definitions and pin the ordering
P(KI before any KO) >= P(KI AND never KO).
"""

from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment


def _env(vol=0.22, rate=0.03, div=0.0, spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def _ref_phoenix():
    # KO=103 barely above spot -> many paths that knock in at 75 later recover
    # and autocall, so P(KI ever) and P(KI & never KO) differ materially.
    return create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=85.0,
        coupon_rate=0.02, num_observations=12, memory_coupon=False,
    )


def _quad_stats():
    return PhoenixQuadEngine(
        params=QuadParams(grid_points=401)
    ).calculate_event_stats(_ref_phoenix(), _env())


def _mc_stats():
    return PhoenixMCEngine(
        params=MCParams(num_paths=300_000, time_steps=252, use_qmc=True, seed=7)
    ).calculate_event_stats(_ref_phoenix(), _env())


# Reference truth values on the reference contract:
#   ki_ever    = P(underlying touches KI barrier in [0,T], KO-independent) ~ 0.185
#   ki_survive = P(KI ever AND never KO) = settles knocked-in            ~ 0.125
KI_EVER_REF = 0.185
KI_SURVIVE_REF = 0.125


def test_new_ki_fields_are_populated_by_both_engines():
    """QUAD V1 (unified statistics since T17) and Phoenix MC (since T10): P(KI before any KO) and P(KI and no KO)
    == the knocked-in settlement row."""
    q, m = _quad_stats(), _mc_stats()
    for stats in (q, m):
        assert stats.knock_in.probability is not None
        assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability


def test_quad_and_mc_agree_on_knock_in_before_ko():
    """Both count a knock-in only before any knock-out (DESIGN §4.5): they agree, and both lie well below the
    contract's KO-independent touch probability (KI_EVER_REF), which counted knock-ins after a knock-out."""
    q, m = _quad_stats(), _mc_stats()
    assert abs(q.knock_in.probability - m.knock_in.probability) < 0.015, (q.knock_in.probability,
                                                                          m.knock_in.probability)
    for stats in (q, m):
        assert stats.knock_in.probability_no_ko <= stats.knock_in.probability < KI_EVER_REF - 0.02


def test_ki_survive_agrees_across_engines():
    q, m = _quad_stats(), _mc_stats()
    assert abs(q.ki_settlement.probability - m.knock_in.probability_no_ko) < 0.015, (
        q.ki_settlement.probability, m.knock_in.probability_no_ko,
    )
    assert abs(m.knock_in.probability_no_ko - KI_SURVIVE_REF) < 0.02


def test_ki_survive_is_the_knocked_in_settlement():
    """P(KI and never KO) is the knocked-in settlement row itself, exactly."""
    q = _quad_stats()
    assert q.knock_in.probability_no_ko == q.ki_settlement.probability


def test_decomposition_identity_before_ge_survive():
    """P(KI before KO) - P(KI, no KO) == P(KI, then KO) >= 0 on both engines, and the two engines agree on it."""
    q, m = _quad_stats(), _mc_stats()
    gaps = [stats.knock_in.probability - stats.knock_in.probability_no_ko for stats in (q, m)]
    assert min(gaps) >= -1e-12, gaps
    assert abs(gaps[0] - gaps[1]) < 0.01, gaps


def _discrete_ki_phoenix(ki_barrier=90.0):
    """Discrete monthly-KI Phoenix (the PDE event-stats path supports discrete,
    not continuous, KI monitoring)."""
    from quantark.util.enum import ObservationType
    from quantark.asset.equity.product.option.observation_schedule import (
        ObservationRecord,
        ObservationSchedule,
    )

    sched = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=i / 12, barrier=ki_barrier)
            for i in range(1, 13)
        ]
    )
    return create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=103.0, ki_barrier=ki_barrier, coupon_barrier=85.0,
        coupon_rate=0.02, num_observations=12, memory_coupon=False,
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=sched,
    )


def test_pde_quad_mc_agree_on_both_ki_definitions_discrete():
    """All three autocallable engines (QUAD, PDE, MC) must agree on both KI
    definitions for a discrete-KI contract: P(KI before any KO) and P(KI and no KO)."""
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    from quantark.asset.equity.param import PDEParams

    env = _env()
    ph = _discrete_ki_phoenix(ki_barrier=75.0)
    q = PhoenixQuadEngine(params=QuadParams(grid_points=801)).calculate_event_stats(ph, env)
    p = PhoenixPDESolver(
        params=PDEParams()
    ).calculate_event_stats(ph, env)
    m = PhoenixMCEngine(
        params=MCParams(num_paths=300_000, time_steps=252, use_qmc=True, seed=7)
    ).calculate_event_stats(ph, env)

    # QUAD V1 (T17), PDE 1D (T12, T13) and MC: unified statistics -- P(KI before any KO) (the grid engines' from
    # their forward adjoint passes) and P(KI and no KO)
    for engine in (q, p):
        assert engine.knock_in.probability_no_ko == engine.ki_settlement.probability
    survs = [q.ki_settlement.probability, p.ki_settlement.probability, m.knock_in.probability_no_ko]
    befores = [q.knock_in.probability, p.knock_in.probability, m.knock_in.probability]
    assert max(survs) - min(survs) < 0.015, survs
    assert max(befores) - min(befores) < 0.015, befores
    # KI before any KO bounds settles-knocked-in from above (measured 0.104 vs 0.103: a knock-in after a knock-out
    # is no knock-in, DESIGN §4.5)
    for engine in (q, p, m):
        assert engine.knock_in.probability_no_ko <= engine.knock_in.probability


def test_snowball_mc_quad_agree_on_both_ki_definitions():
    """Snowball MC (T9) and QUAD V1 (T17) report the unified statistics: P(KI before any KO) and P(KI and no KO)
    (``knock_in.probability_no_ko == ki_settlement.probability``); both definitions agree across the engines."""
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
    from quantark.util.enum import ObservationType

    barrier = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[0.5, 1.0],
        ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    sb = SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier,
        contract_multiplier=1.0, maturity=1.0, is_reverse=False,
    )
    env = _env()
    q = SnowballQuadEngine(params=QuadParams(grid_points=801)).calculate_event_stats(sb, env)
    m = SnowballMCEngine(params=MCParams(num_paths=200_000, seed=7)).calculate_event_stats(sb, env)

    for engine in (q, m):
        assert engine.knock_in.probability is not None
        assert engine.knock_in.probability_no_ko == engine.ki_settlement.probability
    assert abs(q.ki_settlement.probability - m.knock_in.probability_no_ko) < 0.02
    assert abs(q.knock_in.probability - m.knock_in.probability) < 0.02
    assert q.knock_in.probability_no_ko <= q.knock_in.probability


def test_event_distribution_uses_survive_definition_for_mc():
    """The cash-leg EventDistribution's MATURITY_WITH_KI must use the 'settles
    knocked-in' definition regardless of which engine produced the stats, so an
    MC-sourced distribution no longer over-counts KI-then-autocalled paths."""
    from quantark.cashleg.event_distribution import EventDistribution, EventType

    m = _mc_stats()
    dist = EventDistribution.from_statistics(m)
    with_ki = float(dist.probabilities[EventType.MATURITY_WITH_KI])
    # Must reflect P(KI & never KO) (~0.125), NOT P(KI ever) (~0.185).
    assert abs(with_ki - m.ki_settlement.probability) < 1e-9
    assert with_ki < 0.16
