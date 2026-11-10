"""Native model statistics are prices of independently folded outcome cash."""
from dataclasses import replace

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import (
    HestonSnowballPDESolver, HestonPhoenixPDESolver, HestonKOResetSnowballPDESolver,
    HestonSLVSnowballPDESolver, HestonSLVPhoenixPDESolver, HestonSLVKOResetSnowballPDESolver,
    LocalVolSnowballPDESolver, LocalVolPhoenixPDESolver, LocalVolKOResetSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.volmodels.heston import HestonParams
from test_asset.equity.engine_quad.test_autocallable_probability_consistency import CASES, unit_overlay
from test_asset.equity.engine_pde.test_vol_pde_continuous_ki import _flat_env, _unit_leverage

HP = HestonParams(v0=.04, kappa=2., theta=.04, sigma=.3, rho=-.5)
CLASSES = {
    'lv': [LocalVolSnowballPDESolver, LocalVolPhoenixPDESolver, LocalVolKOResetSnowballPDESolver],
    'heston': [HestonSnowballPDESolver, HestonPhoenixPDESolver, HestonKOResetSnowballPDESolver],
    'slv': [HestonSLVSnowballPDESolver, HestonSLVPhoenixPDESolver, HestonSLVKOResetSnowballPDESolver],
}


def engine(model, family, **kwargs):
    cls = CLASSES[model][2 if family == 'ko_reset' else 1 if 'phoenix' in family else 0]
    if model == 'lv':
        return cls(PDEParams(accuracy='fast'))
    return cls(HP, **({'leverage_surface': _unit_leverage()} if model == 'slv' else {}),
               n_x=64, n_v=12, n_t=48, **kwargs)


def note(family, carried=False, reverse=False, discrete=False, disabled=False, no_ko=False):
    factory, spec, *_ = CASES[family]
    spec = dict(spec, ki_monitoring='discrete' if discrete else 'continuous',
                disable_ko_after_ki=disabled)
    if family == 'ko_reset':
        spec['ki_continuous'] = not discrete
    if reverse:
        spec.update(is_reverse=True, ki_barrier=120., ko_barrier=95., coupon_barrier=105.)
        if family == 'ko_reset':
            spec.update(pre_ko_barrier=95., post_ko_barrier=105.)
    if no_ko:
        spec["ko_barrier"] = 1e9
        if family == "ko_reset":
            spec.update(pre_ko_barrier=1e9, post_ko_barrier=1e9)
    p = factory(spec)
    p._otc_lifecycle_knocked_in = carried
    return p


@pytest.mark.parametrize('model', CLASSES)
@pytest.mark.parametrize('family', CASES)
@pytest.mark.parametrize('carried', [False, True])
def test_native_outcomes_match_folded_prices(model, family, carried):
    e, p, env = engine(model, family), note(family, carried), _flat_env()
    base = e.price(p, env)
    stats = e.calculate_event_stats(p, env)
    inc = {x: e.price_folded(p, env, unit_overlay(p, env, x)) - base for x in ('ko', 'ki', 'no_ki')}
    ko = (stats.pre_ko_probability_total + stats.post_ko_probability_total if family == 'ko_reset'
          else stats.ko_probability.sum())
    assert ko == pytest.approx(inc['ko'], abs=2e-10)
    assert stats.ki_survive_knocked_in_probability == pytest.approx(inc['ki'], abs=2e-10)
    # Raw claims are not normalized; native ADI integrates discounting exactly.
    assert sum(inc.values()) == pytest.approx(1., abs=2e-5)
    assert stats.pv == pytest.approx(base, abs=2e-10)
    assert abs(stats.reconciliation_error) < 2e-9
    assert not stats.provenance.surrogate
    if carried:
        assert stats.ki_ever_probability == 1.
        assert stats.ki_survive_knocked_in_probability < 1.


@pytest.mark.parametrize('family', CASES)
@pytest.mark.parametrize('reverse,discrete,disabled', [(True, False, False), (False, True, True), (False, False, True)])
def test_heston_event_order_and_reverse_match_price(family, reverse, discrete, disabled):
    e, p, env = engine('heston', family), note(family, reverse=reverse, discrete=discrete, disabled=disabled), _flat_env()
    base = e.price(p, env)
    stats = e.calculate_event_stats(p, env)
    inc = e.price_folded(p, env, unit_overlay(p, env, 'ki')) - base
    assert stats.ki_survive_knocked_in_probability == pytest.approx(inc, abs=2e-10)


@pytest.mark.parametrize('family', CASES)
def test_unit_leverage_is_heston_for_prices_and_probabilities(family):
    p, env = note(family), _flat_env()
    a, b = engine('heston', family).calculate_event_stats(p, env), engine('slv', family).calculate_event_stats(p, env)
    assert a.pv == pytest.approx(b.pv, abs=1e-12)
    assert a.ki_ever_probability == pytest.approx(b.ki_ever_probability, abs=1e-12)
    assert a.ki_survive_knocked_in_probability == pytest.approx(b.ki_survive_knocked_in_probability, abs=1e-12)


@pytest.mark.parametrize('memory', [False, True])
def test_deferred_memory_phoenix_native_cash_reconciles(memory):
    p = note('phoenix_memory' if memory else 'phoenix')
    from quantark.util.enum import CouponPayType
    p.coupon_config = replace(p.coupon_config, coupon_pay_type=CouponPayType.EXPIRY,
                              initial_coupon_arrears=.15 if memory else 0.)
    e, env = engine('heston', 'phoenix'), _flat_env()
    base = e.price(p, env)
    stats = e.calculate_event_stats(p, env)
    assert stats.ki_survive_knocked_in_probability == pytest.approx(
        e.price_folded(p, env, unit_overlay(p, env, 'ki')) - base, abs=2e-10)
    assert abs(stats.reconciliation_error) < 2e-9


@pytest.mark.parametrize('model', CLASSES)
@pytest.mark.parametrize('family', ['snowball', 'phoenix'])
def test_immediate_ko_keeps_underlying_ki_monitoring(model, family):
    from datetime import datetime
    from intraday.conftest import dated_phoenix, dated_snowball, flat_env
    from quantark.util.calendar import CalendarType, create_calendar
    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    p = (dated_snowball if family == 'snowball' else dated_phoenix)(calendar, datetime(2026, 3, 16))
    p.barrier_config = replace(p.barrier_config, ki_continuous=True)
    date = p.barrier_config.ko_observation_schedule.records[2].observation_date
    env = flat_env(date, spot=110.)
    env.vol_surface = _flat_env().vol_surface
    s = engine(model, family).calculate_event_stats(p, env)
    assert s.ko_probability.sum() == 1.
    assert s.ki_survive_knocked_in_probability == 0.
    assert s.ki_ever_probability > 0.


@pytest.mark.parametrize('family', CASES)
def test_ki_ever_keeps_monitoring_after_ko_on_identical_mesh(family):
    p, env, e = note(family), _flat_env(), engine('heston', family)
    # Freeze the S grid to isolate event semantics from changed critical levels.
    core = e._make_core(p, env, float(p.get_maturity(env)))
    e._layer_x_nodes = lambda *_: core.X_grid
    a = e.calculate_event_stats(p, env)
    b = e.calculate_event_stats(note(family, no_ko=True), env)
    assert a.ki_ever_probability == pytest.approx(b.ki_ever_probability, abs=2e-12)
    assert a.ki_survive_knocked_in_probability < b.ki_survive_knocked_in_probability


@pytest.mark.parametrize('model', ['heston', 'slv'])
@pytest.mark.parametrize('family', CASES)
@pytest.mark.parametrize('projection', ['cell_average', 'nodal'])
def test_batched_adi_is_scalar_adi_at_identical_settings(model, family, projection):
    p, env = note(family), _flat_env()
    a, b = engine(model, family, column_backend='scalar'), engine(model, family)
    a.params = b.params = replace(a.params, event_projection=projection)
    x, y = a.calculate_event_stats(p, env), b.calculate_event_stats(p, env)
    assert x.pv == pytest.approx(y.pv, abs=3e-12)
    np.testing.assert_allclose(x.ko_probability, y.ko_probability, rtol=0., atol=3e-14)
    assert x.ki_survive_knocked_in_probability == pytest.approx(y.ki_survive_knocked_in_probability, abs=3e-14)
    assert x.ki_ever_probability == y.ki_ever_probability


@pytest.mark.parametrize('model', ['heston', 'slv'])
@pytest.mark.parametrize('family', ['snowball', 'ko_reset'])
@pytest.mark.parametrize('spot', [90., 100.])
def test_characteristic_transport_preserves_outcome_cash_linearity(model, family, spot):
    p, env = note(family), _flat_env(s0=spot)
    e = engine(model, family, v_drift_scheme='semi_lagrangian')
    base = e.price(p, env)
    s = e.calculate_event_stats(p, env)
    inc = {x: e.price_folded(p, env, unit_overlay(p, env, x))-base for x in ('ko', 'ki', 'no_ki')}
    ko = (s.pre_ko_probability_total+s.post_ko_probability_total if family == 'ko_reset'
          else s.ko_probability.sum())
    assert s.ki_survive_knocked_in_probability == pytest.approx(inc['ki'], abs=2e-10)
    assert ko == pytest.approx(inc['ko'], abs=2e-10)
    assert sum(inc.values()) == pytest.approx(1., abs=2e-10)
    assert e._last_native_adi_diagnostics['variance_transport'] == 'linear_characteristic'


@pytest.mark.parametrize('model', ['heston', 'slv'])
def test_coarse_ko_reset_rich_outputs_use_the_native_probabilities(model):
    p, env, e = note('ko_reset'), _flat_env(), engine(model, 'ko_reset')
    s = e.calculate_event_stats(p, env)
    r = e.price_with_events(p, env)
    assert r.npv == pytest.approx(s.pv, abs=2e-11)
    assert r.event_distribution.provenance == s.provenance
