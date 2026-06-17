"""SA-CVA exposure-engine tests (spec §3.2). Real quant-ark objects only."""

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError


# --- Task 2: ExposureGrid -------------------------------------------------

def test_grid_unions_and_sorts_event_dates():
    from quantark.sacva.exposure.grid import ExposureGrid
    g = ExposureGrid.build(horizon=2.0, n_steps=4, event_times=[0.5, 1.7, 1.7])
    assert g.times[0] == 0.0 and g.times[-1] == pytest.approx(2.0)
    assert 0.5 in g.times and 1.7 in g.times
    assert list(g.times) == sorted(set(g.times.tolist()))
    assert all(t <= 2.0 for t in g.times)


def test_grid_rejects_nonpositive_horizon():
    from quantark.sacva.exposure.grid import ExposureGrid
    with pytest.raises(ValidationError):
        ExposureGrid.build(horizon=0.0, n_steps=4, event_times=[])


# --- Task 3: CorrelationModel ---------------------------------------------

def test_correlation_cholesky_and_pd_guard():
    from quantark.sacva.exposure.correlation import CorrelationModel
    cm = CorrelationModel(keys=["EQ:A", "FX:EUR"], matrix=[[1.0, 0.3], [0.3, 1.0]])
    L = cm.cholesky()
    assert np.allclose(L @ L.T, np.array([[1.0, 0.3], [0.3, 1.0]]))
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "b"], matrix=[[1.0, 1.5], [1.5, 1.0]]).cholesky()


def test_correlation_shape_and_symmetry_guards():
    from quantark.sacva.exposure.correlation import CorrelationModel
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "b"], matrix=[[1.0]])
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "b"], matrix=[[1.0, 0.2], [0.4, 1.0]])


# --- Task 4: ValueSurface backends ----------------------------------------

def test_grid_value_surface_interpolates():
    from quantark.sacva.exposure.value_surface import GridValueSurface
    spot_grid = np.array([80., 90., 100., 110., 120.])
    surf = np.array([0., 0., 5., 12., 20.])
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {"alive": (spot_grid, surf)}}, currency="USD")
    v = vs.value_at(np.array([95., 105.]), t=1.0, discrete_state="alive")
    assert np.all(v >= 0) and v[1] > v[0]


def test_grid_value_surface_rejects_extrapolation():
    from quantark.sacva.exposure.value_surface import GridValueSurface
    vs = GridValueSurface(times=np.array([1.0]),
                          grids={1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}},
                          currency="USD")
    with pytest.raises(ValidationError):
        vs.value_at(np.array([200.0]), t=1.0, discrete_state=None)


def test_analytic_value_surface_matches_engine():
    from quantark.sacva.exposure.value_surface import AnalyticValueSurface

    class Eng:
        def price(self, product, env):
            return max(env.spot - 90.0, 0.0)

    class _E:
        spot = None

    def as_of(env, spot, t):
        e = _E()
        e.spot = spot
        return e

    vs = AnalyticValueSurface(engine=Eng(), product=object(), base_env=object(),
                              as_of_env=as_of, currency="USD")
    v = vs.value_at(np.array([95., 80.]), t=0.5, discrete_state=None)
    assert v[0] == 5.0 and v[1] == 0.0


# --- Task 5: StatePathGenerator -------------------------------------------

def test_state_paths_shape_and_determinism():
    from quantark.sacva.exposure.paths import StatePathGenerator
    g = StatePathGenerator(keys=["EQ:A"], spots=[100.0], vols=[0.2], rates=[0.03],
                           divs=[0.0], corr=[[1.0]],
                           grid_times=np.linspace(0, 1, 13), num_paths=4000, seed=7)
    p1 = g.generate()
    p2 = g.generate()
    assert p1["EQ:A"].shape == (4000, 13)
    assert np.allclose(p1["EQ:A"], p2["EQ:A"])           # seeded determinism / CRN
    fwd = 100.0 * np.exp((0.03 - 0.0) * 1.0)
    assert abs(p1["EQ:A"][:, -1].mean() - fwd) / fwd < 0.05   # ~ risk-neutral forward


def test_state_paths_length_guards():
    from quantark.sacva.exposure.paths import StatePathGenerator
    with pytest.raises(ValidationError):
        StatePathGenerator(keys=["EQ:A"], spots=[100.0, 1.0], vols=[0.2], rates=[0.03],
                           divs=[0.0], corr=[[1.0]], grid_times=np.linspace(0, 1, 5))


# --- Task 6: BarrierStateMachine ------------------------------------------

def test_state_machine_knock_in_at_node():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    spots = np.array([[100., 85.], [100., 95.], [100., 80.]])
    sm = BarrierStateMachine(ki_barrier=90.0, ki_direction="down",
                             monitoring_idx=[1], times=np.array([0.0, 1.0]), seed=1)
    state = sm.run(spots)
    assert list(state["knocked_in"][:, 1]) == [True, False, True]


def test_state_machine_ko_terminates_and_records_index():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    spots = np.array([[100., 130., 100.], [100., 100., 100.]])
    sm = BarrierStateMachine(ko_barrier=120.0, ko_direction="up",
                             monitoring_idx=[1, 2], times=np.array([0., 1., 2.]), seed=1)
    st = sm.run(spots)
    assert st["ko_idx"][0] == 1 and st["ko_idx"][1] == -1
    assert st["alive"][0, 2] == False and st["alive"][1, 2] == True


def test_bridge_probability_matches_analytic_formula():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    sm = BarrierStateMachine(ki_barrier=95.0, ki_direction="down",
                             monitoring_idx=[1], times=np.array([0.0, 1.0]),
                             seed=3, continuous=True, vol=0.25)
    s0 = np.full(200000, 100.0)
    s1 = np.full(200000, 100.0)
    var = 0.25 ** 2 * 1.0
    x0 = np.log(100.0)
    b = np.log(95.0)
    p_expected = np.exp(-2.0 * (x0 - b) * (x0 - b) / var)
    crossed = sm._bridge_cross(s0, s1, 95.0, 1.0, "down", np.random.default_rng(3))
    assert abs(crossed.mean() - p_expected) < 0.01
    assert 0.0 < p_expected < 1.0          # NOT 1.0 (the inverted-formula bug)


def test_bridge_increases_ki_probability_vs_endpoint():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    rng = np.random.default_rng(0)
    spots = 100 * np.exp(np.cumsum(
        0.3 * np.sqrt(1 / 50) * rng.standard_normal((4000, 51)), axis=1))
    spots = np.column_stack([np.full(4000, 100.0), spots[:, 1:]])
    times = np.linspace(0, 1, 51)
    sm = BarrierStateMachine(ki_barrier=85.0, ki_direction="down",
                             monitoring_idx=list(range(51)), times=times, seed=2,
                             continuous=True, vol=0.3)
    ki_bridge = sm.run(spots)["knocked_in"][:, -1].mean()
    ki_endpoint = ((spots <= 85.0).any(axis=1)).mean()
    assert ki_bridge >= ki_endpoint


def test_continuous_ko_rejected():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    with pytest.raises(ValidationError):
        BarrierStateMachine(ko_barrier=120.0, continuous_ko=True,
                            times=np.array([0.0, 1.0]))


# --- Task 7: repricer ------------------------------------------------------

def _alive_state(n_paths, n_t):
    return {"alive": np.ones((n_paths, n_t), bool),
            "knocked_in": np.zeros((n_paths, n_t), bool),
            "ko_idx": np.full(n_paths, -1)}


def test_repricer_applies_quantity_once():
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    sg = np.array([90., 100., 110.])
    surf = np.array([0., 10., 20.])
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {None: (sg, surf)}}, currency="USD")
    spots = np.array([[100., 100.], [100., 110.]])
    vals = reprice_trade(vs, spots, _alive_state(2, 2), times=np.array([0.0, 1.0]),
                         quantity=2.0, exposure_idx=[1])
    assert vals[0, 0] == 20.0 and vals[1, 0] == 40.0


def test_repricer_negative_quantity_flips_sign_and_dead_zero():
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    sg = np.array([90., 110.])
    surf = np.array([0., 20.])
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {None: (sg, surf)}}, currency="USD")
    spots = np.array([[100., 110.], [100., 110.]])
    state = _alive_state(2, 2)
    state["alive"][1, 1] = False                 # path1 dead at the exposure date
    vals = reprice_trade(vs, spots, state, times=np.array([0.0, 1.0]),
                         quantity=-1.0, exposure_idx=[1])
    assert vals[0, 0] == -20.0 and vals[1, 0] == 0.0


def test_repricer_selects_knocked_in_surface():
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    sg = np.array([90., 110.])
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {"alive": (sg, np.array([1., 2.])),
                                       "knocked_in": (sg, np.array([10., 20.]))}},
                          currency="USD")
    spots = np.array([[110., 110.], [110., 110.]])
    state = _alive_state(2, 2)
    state["knocked_in"][1, 1] = True
    vals = reprice_trade(vs, spots, state, times=np.array([0.0, 1.0]), quantity=1.0,
                         exposure_idx=[1], state_labels=("alive", "knocked_in"))
    assert vals[0, 0] == 2.0 and vals[1, 0] == 20.0


# --- Task 8c: pending receivable (undiscounted) ---------------------------

def test_ko_pending_receivable_undiscounted_until_settlement():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    pe = pending_receivable_exposure(np.array([1, -1]), redemption=100.0,
                                     n_dates=4, settlement_idx=2)
    assert pe[0, 1] == 100.0
    assert pe[0, 2] == 0.0 and pe[0, 3] == 0.0
    assert np.all(pe[1] == 0.0)


def test_pending_receivable_per_path_settlement_offset():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    pe = pending_receivable_exposure(np.array([1, 2]), redemption=50.0, n_dates=5,
                                     settlement_offset_steps=1)
    assert pe[0, 1] == 50.0 and pe[0, 2] == 0.0     # KO@1 -> settle@2
    assert pe[1, 2] == 50.0 and pe[1, 3] == 0.0     # KO@2 -> settle@3


def test_pending_receivable_requires_one_settlement_spec():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    with pytest.raises(ValidationError):
        pending_receivable_exposure(np.array([1]), 100.0, 3)
