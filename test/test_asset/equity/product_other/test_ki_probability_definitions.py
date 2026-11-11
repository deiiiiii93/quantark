"""Cross-engine KI-probability definition tests.

Background (investigation 2026-07-01): QUAD/PDE report ``ki_probability`` as
P(KI ever AND never KO) while MC reports P(KI ever). The gap is purely
definitional (= P(KI AND KO)), not a discretization or Brownian-bridge error —
an independent fine-grid MC reproduces BOTH numbers by pure path counting.

To end the ambiguity, ``AutocallableEventStats`` now carries two unambiguous,
cross-engine-consistent fields:

- ``ki_ever_probability``               = P(KI ever)
- ``ki_survive_knocked_in_probability`` = P(KI ever AND never KO)

These tests pin both engines to the same definitions and pin the decomposition
identity ki_ever - ki_survive == P(KI AND KO) >= 0.

Since the unified event statistics (T9 Snowball, T10 Phoenix) the MC engines report
``knock_in.probability`` = P(KI before any KO) and ``knock_in.probability_no_ko`` =
P(KI AND never KO) (== the knocked-in settlement row); QUAD/PDE keep the legacy
fields until their producers migrate.
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
    """QUAD V1 (legacy object until T17): KI ever and KI-and-no-KO. Phoenix MC (unified statistics since T10):
    P(KI before any KO) and P(KI and no KO) == the knocked-in settlement row."""
    q, m = _quad_stats(), _mc_stats()
    assert q.ki_ever_probability is not None
    assert q.ki_survive_knocked_in_probability is not None
    assert m.knock_in.probability is not None
    assert m.knock_in.probability_no_ko == m.ki_settlement.probability


def test_mc_knock_in_before_ko_is_bounded_by_ki_ever():
    """MC counts a knock-in only before any knock-out (DESIGN §4.5): between settles-knocked-in and QUAD's KI ever."""
    q, m = _quad_stats(), _mc_stats()
    assert m.knock_in.probability_no_ko <= m.knock_in.probability
    assert m.knock_in.probability < float(q.ki_ever_probability) + 0.015, (
        q.ki_ever_probability, m.knock_in.probability
    )
    assert abs(float(q.ki_ever_probability) - KI_EVER_REF) < 0.02


def test_ki_survive_agrees_across_engines():
    q, m = _quad_stats(), _mc_stats()
    assert abs(float(q.ki_survive_knocked_in_probability) - m.knock_in.probability_no_ko) < 0.015, (
        q.ki_survive_knocked_in_probability, m.knock_in.probability_no_ko,
    )
    assert abs(m.knock_in.probability_no_ko - KI_SURVIVE_REF) < 0.02


def test_ki_survive_is_legacy_quad_value():
    """QUAD's legacy ki_probability already means 'KI & never KO', so the new
    precise field must equal it (the legacy semantics are preserved)."""
    q = _quad_stats()
    assert abs(
        float(q.ki_survive_knocked_in_probability) - float(q.ki_probability)
    ) < 1e-9


def test_decomposition_identity_ever_ge_survive():
    """QUAD: ki_ever - ki_survive == P(KI AND KO) >= 0 (~0.05 for the reference contract). MC: P(KI before KO) -
    P(KI, no KO) == P(KI, then KO) >= 0, measured ~0.003: almost all of the KI-and-KO mass is a knock-in AFTER the
    knock-out, which is no knock-in (DESIGN §4.5), so QUAD's KI ever exceeds MC's KI before KO by that mass."""
    q, m = _quad_stats(), _mc_stats()
    ever, survive = float(q.ki_ever_probability), float(q.ki_survive_knocked_in_probability)
    assert ever - survive > 0.02, (ever, survive)
    assert m.knock_in.probability >= m.knock_in.probability_no_ko
    assert ever - m.knock_in.probability > 0.02, (ever, m.knock_in.probability)


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
    definitions for a discrete-KI contract. This closes the definitional gap the
    barrier-smoothing spec had flagged as un-reconcilable (e.g. legacy QUAD 0.0995
    vs MC 0.1389): ki_ever ~0.14 and ki_survive ~0.10 now agree across engines."""
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

    # QUAD / PDE: legacy objects (KI ever, KI-and-no-KO) until T12/T17; MC: P(KI before any KO), P(KI and no KO)
    evers = [float(s.ki_ever_probability) for s in (q, p)]
    survs = [float(s.ki_survive_knocked_in_probability) for s in (q, p)] + [m.knock_in.probability_no_ko]
    assert max(evers) - min(evers) < 0.015, evers
    assert max(survs) - min(survs) < 0.015, survs
    # The two definitions genuinely differ (this is the whole point).
    for ever, surv in zip(evers, survs):
        assert ever > surv + 0.02, (ever, surv)
    # KI before any KO lies between them, well below KI ever: a knock-in after a knock-out is no knock-in
    # (DESIGN §4.5; measured 0.104 vs KI ever ~0.14, settles knocked in 0.103)
    assert m.knock_in.probability_no_ko <= m.knock_in.probability < min(evers) - 0.02
    assert abs(float(q.ki_probability) - float(q.ki_survive_knocked_in_probability)) < 1e-9
    assert abs(float(p.ki_probability) - float(p.ki_survive_knocked_in_probability)) < 1e-9


def test_snowball_mc_quad_agree_on_both_ki_definitions():
    """Snowball MC reports the unified statistics since T9: P(KI before any KO) and P(KI and no KO)
    (``knock_in.probability_no_ko == ki_settlement.probability``). The QUAD V1 legacy object still reports
    KI ever and KI-and-no-KO (until T17): the settles-knocked-in numbers agree; KI ever bounds KI before KO."""
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

    assert q.ki_ever_probability is not None and q.ki_survive_knocked_in_probability is not None
    assert m.knock_in.probability is not None
    assert m.knock_in.probability_no_ko == m.ki_settlement.probability
    assert abs(float(q.ki_survive_knocked_in_probability) - m.knock_in.probability_no_ko) < 0.02
    assert m.knock_in.probability_no_ko <= m.knock_in.probability < float(q.ki_ever_probability) + 0.02
    # Legacy QUAD meaning: ki_probability = ki_survive.
    assert abs(float(q.ki_probability) - float(q.ki_survive_knocked_in_probability)) < 1e-9


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
