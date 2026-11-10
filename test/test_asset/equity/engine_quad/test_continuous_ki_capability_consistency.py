"""A common continuous monitoring contract with independent probability controls."""
import numpy as np
import pytest
from scipy.special import ndtr
from test_asset.equity.engine_quad.test_autocallable_probability_consistency import CASES, env
from quantark.asset.equity.param import QuadV2Params


def first_hit(spot, barrier, maturity, rate=.025, div=.03, vol=.22, reverse=False):
    d = np.log(barrier / spot) if reverse else np.log(spot / barrier)
    if d <= 0:
        return 1.0
    m = (rate - div - .5 * vol * vol) * maturity
    if reverse:
        m = -m
    variance = vol * vol * maturity
    sd = np.sqrt(variance)
    return float(ndtr((-d-m)/sd) + np.exp(-2*m*d/variance)*ndtr((m-d)/sd))


@pytest.mark.parametrize('family', CASES)
@pytest.mark.parametrize('carried', [False, True])
def test_v2_continuous_ki_ever_matches_independent_first_passage(family, carried):
    factory, spec, _, cls = CASES[family]
    spec = dict(spec, ki_monitoring='continuous')
    if family == 'ko_reset':
        spec['ki_continuous'] = True
    product = factory(spec)
    product._otc_lifecycle_knocked_in = carried
    engine, market = cls(), env()
    stats = engine.calculate_event_stats(product, market)
    horizon = spec['maturity_pre'] if family == 'ko_reset' else spec['maturity']
    expected = 1. if carried else first_hit(100., spec['ki_barrier'], horizon)
    assert stats.ki_ever_probability == pytest.approx(expected, abs=5e-13)
    assert engine._prepared.diagnostics['ki_monitoring_method'] == 'analytic_time_change'
    assert stats.ki_ever_probability >= stats.ki_survive_knocked_in_probability


@pytest.mark.parametrize('family', CASES)
def test_v2_continuous_monitoring_is_ko_independent_in_spot_batch(family):
    factory, spec, _, cls = CASES[family]
    spec = dict(spec, ki_monitoring='continuous')
    if family == 'ko_reset':
        spec.update(ki_continuous=True, pre_ko_barrier=80., post_ko_barrier=80.)
    else:
        spec['ko_barrier'] = 80.
    product, market = factory(spec), env()
    spots = [90., 100., 110.]
    stats = cls().calculate_event_stats_at_spots(product, market, spots)
    horizon = spec['maturity_pre'] if family == 'ko_reset' else spec['maturity']
    for spot, s in zip(spots, stats):
        assert s.ki_ever_probability == pytest.approx(first_hit(spot, spec['ki_barrier'], horizon), abs=5e-13)
        assert s.ki_survive_knocked_in_probability <= s.ki_ever_probability + 1e-12


def test_price_context_has_no_monitoring_companion():
    factory, spec, _, cls = CASES['phoenix_memory']
    p = factory(dict(spec, ki_monitoring='continuous'))
    engine = cls()
    engine.price(p, env())
    assert engine._prepared._ki_monitor is None
    price_states = engine._prepared.diagnostics['states']
    engine.calculate_event_stats(p, env())
    assert engine._prepared.diagnostics['states'] == price_states


def term_market():
    from quantark.param.div import ForwardCarryCurve
    from quantark.param.vol.vol_surface import TermStructureVolSurface
    market = env()
    market.div_yield = ForwardCarryCurve([(0.12, .012), (.29, -.004), (.5, .008), (1., .016)]).to_dividend_yield(market.rate_curve)
    market.vol_surface = TermStructureVolSurface(times=[.12, .29, .5, 1.], vols=[.18, .24, .22, .22])
    return market


@pytest.mark.parametrize('family', CASES)
def test_v2_piecewise_coefficients_support_continuous_monitoring(family):
    factory, spec, _, cls = CASES[family]
    spec = dict(spec, ki_monitoring='continuous')
    if family == 'ko_reset':
        spec['ki_continuous'] = True
    engine, market = cls(), term_market()
    stats = engine.calculate_event_stats(factory(spec), market)
    assert stats.ki_ever_probability is not None
    assert stats.ki_survive_knocked_in_probability <= stats.ki_ever_probability <= 1.
    assert engine._prepared.diagnostics['ki_monitoring_method'] == 'native_survival_recursion'
    assert engine._prepared.diagnostics['continuous_model'] == 'piecewise_constant_exact'
    for time in (.12, .29):
        assert time in [pair[1] for pair in engine._prepared.diagnostics['canonical_times']]
    fine = cls(QuadV2Params(order=10, cells_per_sd=3.))
    reference = fine.calculate_event_stats(factory(spec), market)
    assert stats.ki_ever_probability == pytest.approx(reference.ki_ever_probability, abs=2e-5)


@pytest.mark.parametrize('params_cls_name', ['MCParams', 'PDEParams', 'QuadParams', 'QuadV2Params'])
def test_shared_exact_model_guard_and_smooth_curve_opt_in(params_cls_name):
    import quantark.asset.equity.param as parameters
    from quantark.param.div import TermStructureDividendYield
    from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid

    cls = getattr(parameters, params_cls_name)
    market = env()
    market.div_yield = TermStructureDividendYield(times=[.1, .3, .5], yields=[.01, .05, .02])
    with pytest.raises(NotImplementedError, match='piecewise_constant'):
        continuous_time_grid(market, [.5], cls(continuous_term_structure='exact'))
    times, model = continuous_time_grid(market, [.5], cls(continuous_term_structure='piecewise_constant', continuous_steps_per_year=48))
    assert model == 'piecewise_constant_approximation'
    assert .1 in times and .3 in times and max(np.diff([0.] + times)) <= 1/48 + 1e-14


@pytest.mark.parametrize('family', ['snowball', 'phoenix'])
def test_mc_bridge_uses_actual_gbm_interval_variance(family):
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs

    cls = SnowballMCEngine if family == 'snowball' else PhoenixMCEngine
    engine, market = cls(MCParams(num_paths=4, seed=917)), term_market()
    engine._term_ctx = (market, 100.)
    dt = np.array([.12, .17, .21])
    engine._create_path_generator(100., .025, .03, .22, .5, dt)
    paths = np.full((4, 4), 100.)
    variance = engine._ki_bridge_step_log_variance(paths)
    term = build_mc_term_inputs(market, 100., .5, len(dt), dt_array=dt)
    np.testing.assert_allclose(variance[0], np.square(term.vol) * dt, rtol=0., atol=1e-15)
    assert not variance.flags.writeable
    assert np.shares_memory(variance, engine._native_ki_step_variance)


@pytest.mark.parametrize('family', ['snowball', 'phoenix', 'ko_reset'])
def test_continuous_ki_monitoring_continues_after_valuation_ko(family):
    from datetime import datetime
    from dataclasses import replace
    from intraday.conftest import dated_snowball, dated_phoenix, dated_ko_reset, flat_env
    from quantark.util.enum import ObservationType
    from quantark.util.calendar import CalendarType, create_calendar
    from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver

    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    factory = {'snowball': dated_snowball, 'phoenix': dated_phoenix, 'ko_reset': dated_ko_reset}[family]
    product = factory(calendar, datetime(2026, 3, 16))
    product.barrier_config = replace(product.barrier_config, ki_continuous=True,
                                    ki_observation_type=ObservationType.CONTINUOUS,
                                    ki_observation_schedule=None, ki_observation_dates=None)
    date = product.barrier_config.ko_observation_schedule.records[2].observation_date
    market = flat_env(date, spot=110.)
    _, _, v1, v2 = CASES[family]
    pde = {'snowball': SnowballPDESolver, 'phoenix': PhoenixPDESolver, 'ko_reset': KOResetSnowballPDESolver}[family]
    stats = [cls().calculate_event_stats(product, market) for cls in (v1, pde, v2)]
    expected = stats[-1].ki_ever_probability
    assert 0 < expected < 1
    for result in stats:
        assert result.ki_ever_probability == pytest.approx(expected, abs=1e-12)
        assert result.ki_survive_knocked_in_probability == 0.
        ko = result.pre_ko_probability_total + result.post_ko_probability_total if family == 'ko_reset' else result.ko_probability.sum()
        assert ko == 1.


def test_missing_dividend_input_is_exact_zero_carry():
    from quantark.asset.equity.engine.continuous_ki_support import continuous_time_grid
    market = env()
    market.div_yield = None
    times, model = continuous_time_grid(market, [.5], QuadV2Params())
    assert times == [.5] and model == 'piecewise_constant_exact'


@pytest.mark.parametrize('reverse', [False, True])
def test_reflected_grid_matches_pointwise_integral_on_same_field(reverse):
    from quantark.asset.equity.engine.quad.v2.basis import Mesh
    from quantark.asset.equity.engine.quad.v2.contract import grid_function
    from quantark.asset.equity.engine.quad.v2.operator import Operator
    from quantark.asset.equity.engine.quad.v2.continuous import continuous_grid_readout, continuous_readout

    barrier, sd = np.log(100.), .1
    mesh = Mesh(barrier-1., .05, 40, 6)
    x = mesh.nodes
    alive = grid_function(mesh, np.array([1.+.03*np.sin(2*x), .2+.02*np.cos(3*x)]))
    hit = grid_function(mesh, np.array([.5+.01*np.cos(x), .1+.01*np.sin(x)]))
    params = QuadV2Params(order=6)
    fast = continuous_grid_readout(alive, hit, .01, sd*sd, .99, barrier, reverse, Operator(mesh,params))
    # Compare the actual integral at interior queries, with no mesh-boundary extension.
    indices = np.flatnonzero(np.abs(x.ravel()-barrier)<.25)
    point = continuous_readout(alive, hit, x.ravel()[indices], .01, sd*sd, .99, barrier, reverse, params)[0]
    np.testing.assert_allclose(fast.reshape(2,-1)[:,indices], point, rtol=0., atol=2e-10)
