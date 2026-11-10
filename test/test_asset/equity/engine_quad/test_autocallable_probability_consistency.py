"""Outcome probabilities must agree with cash added to the actual pricing route."""
import numpy as np
import pytest

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import (
    SnowballQuadEngineV2, PhoenixQuadEngineV2, KOResetSnowballQuadEngineV2,
)
from quantark.asset.equity.param import QuadParams
from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.modelvalidation.builders.equity_phoenix import make_phoenix
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from pde_3a.products import SNOWBALL, PHOENIX, KO_RESET, env


CASES = {
    'snowball': (make_snowball, dict(SNOWBALL, months=6, maturity=.5, ki_monitoring='discrete'), SnowballQuadEngine, SnowballQuadEngineV2),
    'phoenix': (make_phoenix, dict(PHOENIX, num_observations=6, maturity=.5, ki_monitoring='discrete'), PhoenixQuadEngine, PhoenixQuadEngineV2),
    'phoenix_memory': (make_phoenix, dict(PHOENIX, num_observations=6, maturity=.5, ki_monitoring='discrete', memory_coupon=True), PhoenixQuadEngine, PhoenixQuadEngineV2),
    'ko_reset': (make_ko_reset, dict(KO_RESET, maturity_pre=.5, maturity_post=1., ki_continuous=False, ki_monitoring='discrete'), KOResetSnowballQuadEngine, KOResetSnowballQuadEngineV2),
}


def unit_overlay(product, market, outcome, ko_index=None):
    structure = _event_structure(product, market)

    def cash(event, selected):
        return FoldedEventCash(event.schedule, event.determination_time,
                               event.payment_time, float(selected) / market.get_discount_factor(event.payment_time))

    return FoldOverlay(
        ko=tuple(cash(event, outcome == 'ko' and (ko_index is None or i == ko_index))
                 for i, event in enumerate(structure.ko)),
        maturity_no_ki=cash(structure.no_ki, outcome == 'no_ki'),
        maturity_ki=None if structure.ki is None else cash(structure.ki, outcome == 'ki'),
        constant_pv=0.,
    )


def assert_outcomes(engine, product, market, per_ko=False):
    stats = engine.calculate_event_stats(product, market)
    base = engine.price(product, market)
    increments = {outcome: engine.price_folded(product, market, unit_overlay(product, market, outcome)) - base
                  for outcome in ('ko', 'ki', 'no_ki')}
    ko = (stats.pre_ko_probability_total + stats.post_ko_probability_total
          if hasattr(stats, 'pre_ko_probability_total') else stats.ko_probability.sum())
    assert ko == pytest.approx(increments['ko'], abs=2e-11)
    assert stats.ki_survive_knocked_in_probability == pytest.approx(increments['ki'], abs=2e-11)
    assert sum(increments.values()) == pytest.approx(1., abs=2e-8)
    if per_ko:
        for i, probability in enumerate(stats.ko_probability):
            added = engine.price_folded(product, market, unit_overlay(product, market, 'ko', i)) - base
            assert probability == pytest.approx(added, abs=2e-11)


@pytest.mark.parametrize('family', CASES)
@pytest.mark.parametrize('carried', [False, True])
def test_v1_outcomes_equal_price_increments(family, carried):
    factory, spec, cls, _ = CASES[family]
    product = factory(spec)
    product._otc_lifecycle_knocked_in = carried
    assert_outcomes(cls(QuadParams(grid_points=301)), product, env(), per_ko=family != 'ko_reset')


@pytest.mark.parametrize('family', CASES)
def test_v1_continuous_outcomes_equal_price_increments(family):
    factory, spec, cls, _ = CASES[family]
    spec = dict(spec, ki_monitoring='continuous')
    if family == 'ko_reset':
        spec['ki_continuous'] = True
    assert_outcomes(cls(QuadParams(grid_points=301)), factory(spec), env())


@pytest.mark.parametrize('projection', ['cell_average', 'nodal'])
def test_v1_overlapping_ko_reset_events_equal_price(projection):
    product = make_ko_reset(dict(CASES['ko_reset'][1], pre_ko_barrier=80., ki_barrier=85.))
    assert_outcomes(KOResetSnowballQuadEngine(QuadParams(grid_points=301, event_projection=projection)), product, env())


@pytest.mark.parametrize('family', CASES)
def test_v2_ki_ever_is_independent_of_autocall(family):
    factory, spec, _, cls = CASES[family]
    market = env()
    stats = cls().calculate_event_stats(factory(spec), market)
    no_ko = dict(spec)
    if family == 'ko_reset':
        no_ko.update(pre_ko_barrier=1e9, post_ko_barrier=1e9)
    else:
        no_ko['ko_barrier'] = 1e9
    reference = cls().calculate_event_stats(factory(no_ko), market)
    assert stats.ki_ever_probability == pytest.approx(reference.ki_ever_probability, abs=2e-8)


@pytest.mark.parametrize('family', CASES)
@pytest.mark.parametrize('projection', ['cell_average', 'nodal'])
def test_v1_disabled_ko_after_ki_matches_price(family, projection):
    factory, spec, cls, _ = CASES[family]
    assert_outcomes(cls(QuadParams(grid_points=301, event_projection=projection)),
                    factory(dict(spec, disable_ko_after_ki=True)), env())


@pytest.mark.parametrize('family', ['snowball', 'phoenix', 'phoenix_memory'])
def test_v1_reverse_contract_matches_price(family):
    factory, spec, cls, _ = CASES[family]
    spec = dict(spec, is_reverse=True, ko_barrier=95., ki_barrier=115.)
    if family != 'snowball':
        spec['coupon_barrier'] = 105.
    assert_outcomes(cls(QuadParams(grid_points=301)), factory(spec), env())


@pytest.mark.parametrize('carried', [False, True])
def test_v1_transition_readout_matches_price(carried):
    factory, spec, cls, _ = CASES['snowball']
    product = factory(spec)
    product._otc_lifecycle_knocked_in = carried
    assert_outcomes(cls(QuadParams(grid_points=301, readout='transition')), product, env())


@pytest.mark.parametrize('memory', [False, True])
def test_v1_deferred_phoenix_outcomes_match_price(memory):
    factory, spec, cls, _ = CASES['phoenix']
    product = factory(dict(spec, memory_coupon=memory, coupon_pay_type='expiry', coupon_barrier=110.))
    assert_outcomes(cls(QuadParams(grid_points=301)), product, env(), per_ko=True)


@pytest.mark.parametrize('family', ['snowball', 'phoenix'])
def test_v1_carried_ki_and_valuation_ko_has_zero_ki_survival(family):
    from datetime import datetime
    from intraday.conftest import dated_phoenix, dated_snowball, flat_env
    from quantark.util.calendar import CalendarType, create_calendar

    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    factory = dated_snowball if family == 'snowball' else dated_phoenix
    product = factory(calendar, datetime(2026, 3, 16))
    product._otc_lifecycle_knocked_in = True
    date = product.barrier_config.ko_observation_schedule.records[2].observation_date
    market = flat_env(date, spot=110.)
    cls = SnowballQuadEngine if family == 'snowball' else PhoenixQuadEngine
    stats = cls().calculate_event_stats(product, market)
    assert stats.ki_ever_probability == 1.
    assert stats.ki_survive_knocked_in_probability == 0.
    assert stats.ki_probability == 0.


def test_forward_density_declares_different_statistics_model():
    factory, spec, cls, _ = CASES['snowball']
    product = factory(spec)
    product._otc_lifecycle_knocked_in = True
    stats = cls(QuadParams(grid_points=301, event_stats_mode='forward_density')).calculate_event_stats(product, env())
    assert stats.provenance.surrogate
    assert stats.ki_survive_knocked_in_probability == pytest.approx(1. - stats.ko_probability.sum(), abs=1e-8)


@pytest.mark.parametrize('family', CASES)
def test_v1_converged_statistics_use_the_accepted_price_grid(family):
    from dataclasses import replace

    factory, spec, cls, _ = CASES[family]
    product, market = factory(spec), env()
    engine = cls(QuadParams(grid_points=301, auto_converge=True,
                            convergence_abs_tol=1., max_convergence_grid_points=2401))
    stats = engine.calculate_event_stats(product, market)
    # KO-reset intentionally ignores auto_converge in both price and folded
    # price; statistics must preserve that same fixed-grid policy.
    accepted = (engine.params.grid_points if cls is KOResetSnowballQuadEngine
                else engine._last_convergence_info['grid_points'])
    fixed = cls(replace(engine.params, auto_converge=False, grid_points=accepted))
    reference = fixed.calculate_event_stats(product, market)
    assert stats.pv == reference.pv
    np.testing.assert_array_equal(stats.ko_probability, reference.ko_probability)
    assert stats.ki_survive_knocked_in_probability == reference.ki_survive_knocked_in_probability
    assert_outcomes(fixed, product, market)
