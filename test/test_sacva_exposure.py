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


def test_grid_rejects_bad_nsteps_and_nonfinite_event():
    from quantark.sacva.exposure.grid import ExposureGrid
    with pytest.raises(ValidationError):
        ExposureGrid.build(horizon=2.0, n_steps=True, event_times=[])   # bool n_steps
    with pytest.raises(ValidationError):
        ExposureGrid.build(horizon=2.0, n_steps=4, event_times=[float("nan")])


def test_grid_rejects_event_beyond_horizon():
    from quantark.sacva.exposure.grid import ExposureGrid
    with pytest.raises(ValidationError):     # event past horizon would be silently dropped
        ExposureGrid.build(horizon=2.0, n_steps=4, event_times=[2.5])
    with pytest.raises(ValidationError):     # negative event date is malformed
        ExposureGrid.build(horizon=2.0, n_steps=4, event_times=[-0.5])


def test_paths_empty_grid_times_raises_validation():
    from quantark.sacva.exposure.paths import StatePathGenerator
    with pytest.raises(ValidationError):
        StatePathGenerator(keys=["EQ:A"], spots=[100.0], vols=[0.2], rates=[0.03],
                           divs=[0.0], corr=[[1.0]], grid_times=[])


def test_paths_rejects_nonint_seed():
    from quantark.sacva.exposure.paths import StatePathGenerator
    with pytest.raises(ValidationError):
        StatePathGenerator(keys=["EQ:A"], spots=[100.0], vols=[0.2], rates=[0.03],
                           divs=[0.0], corr=[[1.0]], grid_times=np.linspace(0, 1, 5),
                           seed=None)


def test_exposure_grid_direct_construction_validates_times():
    from quantark.sacva.exposure.grid import ExposureGrid
    with pytest.raises(ValidationError):     # not starting at origin
        ExposureGrid(times=np.array([0.5, 1.0]))
    with pytest.raises(ValidationError):     # not strictly increasing
        ExposureGrid(times=np.array([0.0, 1.0, 1.0]))


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


def test_correlation_rejects_non_unit_diagonal_and_bounds():
    from quantark.sacva.exposure.correlation import CorrelationModel
    # symmetric & positive-definite but NOT a correlation matrix (diag != 1)
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "b"], matrix=[[2.0, 0.3], [0.3, 2.0]])
    # off-diagonal outside [-1, 1]
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "b"], matrix=[[1.0, 1.2], [1.2, 1.0]])
    # non-finite entry
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "b"], matrix=[[1.0, np.nan], [np.nan, 1.0]])


def test_correlation_rejects_duplicate_keys():
    from quantark.sacva.exposure.correlation import CorrelationModel
    with pytest.raises(ValidationError):
        CorrelationModel(keys=["a", "a"], matrix=[[1.0, 0.0], [0.0, 1.0]])


# --- Task 4: ValueSurface backends ----------------------------------------

def test_grid_value_surface_rejects_unsorted_times_and_unknown_grid_key():
    from quantark.sacva.exposure.value_surface import GridValueSurface
    g = {1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}}
    with pytest.raises(ValidationError):     # times not strictly increasing
        GridValueSurface(times=np.array([1.0, 1.0]), grids=g, currency="USD")
    with pytest.raises(ValidationError):     # grid time key absent from times
        GridValueSurface(times=np.array([0.0, 2.0]), grids=g, currency="USD")


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


def test_grid_value_surface_rejects_nonfinite_state():
    from quantark.sacva.exposure.value_surface import GridValueSurface
    vs = GridValueSurface(times=np.array([1.0]),
                          grids={1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}},
                          currency="USD")
    with pytest.raises(ValidationError):
        vs.value_at(np.array([np.nan]), t=1.0, discrete_state=None)


def test_grid_value_surface_rejects_unsorted_or_mismatched_grid():
    from quantark.sacva.exposure.value_surface import GridValueSurface
    # spot_grid not strictly increasing -> np.interp would silently mis-interpolate
    vs = GridValueSurface(times=np.array([1.0]),
                          grids={1.0: {None: (np.array([110., 90.]), np.array([0., 20.]))}},
                          currency="USD")
    with pytest.raises(ValidationError):
        vs.value_at(np.array([100.0]), t=1.0, discrete_state=None)
    # spot_grid / vals length mismatch
    vs2 = GridValueSurface(times=np.array([1.0]),
                           grids={1.0: {None: (np.array([90., 100., 110.]), np.array([0., 20.]))}},
                           currency="USD")
    with pytest.raises(ValidationError):
        vs2.value_at(np.array([100.0]), t=1.0, discrete_state=None)


def test_analytic_value_surface_rejects_stateful_pricing():
    # the analytic backend is single-state (vanilla); a discrete state would be
    # silently ignored -> state-insensitive exposure, so it must raise instead
    from quantark.sacva.exposure.value_surface import AnalyticValueSurface

    class Eng:
        def price(self, product, env):
            return 1.0

    vs = AnalyticValueSurface(engine=Eng(), product=object(), base_env=object(),
                              as_of_env=lambda e, s, t: e, currency="USD")
    with pytest.raises(ValidationError):
        vs.value_at(np.array([100.0]), t=0.5, discrete_state="knocked_in")


def test_analytic_value_surface_rejects_nonfinite_state():
    from quantark.sacva.exposure.value_surface import AnalyticValueSurface

    class Eng:
        def price(self, product, env):
            return 1.0

    vs = AnalyticValueSurface(engine=Eng(), product=object(), base_env=object(),
                              as_of_env=lambda e, s, t: e, currency="USD")
    with pytest.raises(ValidationError):
        vs.value_at(np.array([np.nan]), t=0.5, discrete_state=None)


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
    fwd = 100.0 * np.exp((0.03 - 0.0) * 1.0)             # risk-neutral drift r - q
    term = p1["EQ:A"][:, -1].mean()
    assert abs(term - fwd) / fwd < 0.02                  # close to the RN forward
    assert term > 101.5                                  # NOT the driftless mean (~100)


def test_state_paths_length_guards():
    from quantark.sacva.exposure.paths import StatePathGenerator
    with pytest.raises(ValidationError):
        StatePathGenerator(keys=["EQ:A"], spots=[100.0, 1.0], vols=[0.2], rates=[0.03],
                           divs=[0.0], corr=[[1.0]], grid_times=np.linspace(0, 1, 5))


def test_state_paths_reject_malformed_correlation():
    from quantark.sacva.exposure.paths import StatePathGenerator
    base = dict(spots=[100.0, 100.0], vols=[0.2, 0.2], rates=[0.03, 0.03],
                divs=[0.0, 0.0], grid_times=np.linspace(0, 1, 5))
    with pytest.raises(ValidationError):     # non-unit diagonal -> not a corr matrix
        StatePathGenerator(keys=["A", "B"], corr=[[2.0, 0.3], [0.3, 2.0]], **base)
    with pytest.raises(ValidationError):     # not positive definite
        StatePathGenerator(keys=["A", "B"], corr=[[1.0, 1.0], [1.0, 1.0]], **base)


def test_state_paths_reject_nonfinite_and_duplicate_keys():
    from quantark.sacva.exposure.paths import StatePathGenerator
    base = dict(rates=[0.03], divs=[0.0], corr=[[1.0]], grid_times=np.linspace(0, 1, 5))
    with pytest.raises(ValidationError):     # NaN vol slips past vols < 0
        StatePathGenerator(keys=["EQ:A"], spots=[100.0], vols=[float("nan")], **base)
    with pytest.raises(ValidationError):     # inf spot slips past spots <= 0
        StatePathGenerator(keys=["EQ:A"], spots=[float("inf")], vols=[0.2], **base)
    with pytest.raises(ValidationError):     # duplicate keys collide in output dict
        StatePathGenerator(keys=["EQ:A", "EQ:A"], spots=[100.0, 100.0], vols=[0.2, 0.2],
                           rates=[0.03, 0.03], divs=[0.0, 0.0],
                           corr=[[1.0, 0.0], [0.0, 1.0]], grid_times=np.linspace(0, 1, 5))


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


def test_state_machine_rejects_nonpositive_barrier_and_bad_vol():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    with pytest.raises(ValidationError):
        BarrierStateMachine(ki_barrier=-90.0, times=np.array([0.0, 1.0]))
    with pytest.raises(ValidationError):
        BarrierStateMachine(ki_barrier=90.0, continuous=True, vol=-0.2,
                            times=np.array([0.0, 1.0]))


def test_state_machine_rejects_noninteger_schedule_and_time_mismatch():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    # non-integer monitoring index silently never matches the int loop variable
    sm = BarrierStateMachine(ki_barrier=90.0, ki_direction="down",
                             monitoring_idx=[1.9], times=np.array([0.0, 1.0]), seed=1)
    with pytest.raises(ValidationError):
        sm.run(np.full((2, 2), 100.0))
    # times length must equal the spot time axis
    sm2 = BarrierStateMachine(ki_barrier=90.0, ki_direction="down",
                              monitoring_idx=[1], times=np.array([0.0, 1.0]), seed=1)
    with pytest.raises(ValidationError):
        sm2.run(np.full((2, 3), 100.0))


def test_state_machine_rejects_nonfinite_or_non2d_spots():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    sm = BarrierStateMachine(ki_barrier=90.0, ki_direction="down",
                             monitoring_idx=[1], times=np.array([0.0, 1.0]), seed=1)
    with pytest.raises(ValidationError):     # NaN spot defeats <= 0 check -> silent miss
        sm.run(np.array([[100.0, float("nan")]]))
    with pytest.raises(ValidationError):     # 1-D spots
        sm.run(np.array([100.0, 95.0]))


def test_bridge_up_barrier_matches_analytic_formula():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    sm = BarrierStateMachine(ki_barrier=105.0, ki_direction="up",
                             monitoring_idx=[1], times=np.array([0.0, 1.0]),
                             seed=4, continuous=True, vol=0.25)
    s0 = np.full(200000, 100.0)
    s1 = np.full(200000, 100.0)
    var = 0.25 ** 2 * 1.0
    b, x0 = np.log(105.0), np.log(100.0)
    p_expected = np.exp(-2.0 * (b - x0) * (b - x0) / var)
    crossed = sm._bridge_cross(s0, s1, 105.0, 1.0, "up", np.random.default_rng(4))
    assert abs(crossed.mean() - p_expected) < 0.01
    assert 0.0 < p_expected < 1.0


def test_continuous_ki_window_starting_after_origin():
    # KI activates at node 2; a dip during interval [0,1] (pre-window) must NOT knock in
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    times = np.linspace(0, 1, 5)
    spots = np.full((1, 5), 100.0)
    spots[0, 1] = 80.0                       # breach before the KI window opens
    sm = BarrierStateMachine(ki_barrier=85.0, ki_direction="down",
                             ki_monitoring_idx=[2, 3, 4], times=times,
                             continuous=True, vol=0.2, seed=1)
    assert sm.run(spots)["knocked_in"][0, -1] == False


def test_state_machine_rejects_barrier_with_empty_schedule():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    with pytest.raises(ValidationError):     # KI barrier set but never monitored
        BarrierStateMachine(ki_barrier=90.0, monitoring_idx=[], times=np.array([0.0, 1.0]))
    with pytest.raises(ValidationError):     # KO barrier set but never monitored
        BarrierStateMachine(ko_barrier=120.0, ko_monitoring_idx=[],
                            monitoring_idx=[1], times=np.array([0.0, 1.0]))


def test_state_machine_rejects_none_times_and_bad_seed():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    with pytest.raises(ValidationError):
        BarrierStateMachine(ki_barrier=90.0, monitoring_idx=[1], times=None)
    with pytest.raises(ValidationError):
        BarrierStateMachine(ki_barrier=90.0, monitoring_idx=[1],
                            times=np.array([0.0, 1.0]), seed=None)


def test_continuous_ki_rejects_noncontiguous_schedule():
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    # continuous KI with a gapped schedule would silently skip bridge intervals
    sm = BarrierStateMachine(ki_barrier=90.0, ki_direction="down",
                             monitoring_idx=[0, 1, 3], times=np.linspace(0, 1, 4),
                             continuous=True, vol=0.2, seed=1)
    with pytest.raises(ValidationError):
        sm.run(np.full((2, 4), 100.0))


def test_snowball_separate_ki_ko_schedules():
    # KO discrete (monthly nodes), KI continuous (every node) — distinct schedules.
    from quantark.sacva.exposure.statemachine import BarrierStateMachine
    times = np.linspace(0, 1, 13)                    # 12 monthly steps
    # path dips below KI between nodes 1-2 but never on a KO node; ends high
    spots = np.full((1, 13), 105.0)
    spots[0, 2] = 88.0                               # KI breach (node 2, not a KO node)
    sm = BarrierStateMachine(
        ki_barrier=90.0, ki_direction="down",
        ko_barrier=120.0, ko_direction="up",
        ki_monitoring_idx=list(range(13)),           # continuous KI
        ko_monitoring_idx=[6, 12],                   # discrete KO (semi-annual)
        times=times, continuous=True, vol=0.2, seed=1)
    st = sm.run(spots)
    assert st["knocked_in"][0, -1] == True           # KI caught on the daily schedule
    assert st["ko_idx"][0] == -1                     # never KO'd (105 < 120 on KO nodes)


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


def test_pending_receivable_rejects_bad_settlement_and_redemption():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    with pytest.raises(ValidationError):       # settlement before KO
        pending_receivable_exposure(np.array([2]), 100.0, 5, settlement_idx=1)
    with pytest.raises(ValidationError):       # negative offset
        pending_receivable_exposure(np.array([1]), 100.0, 5, settlement_offset_steps=-1)
    with pytest.raises(ValidationError):       # non-finite redemption
        pending_receivable_exposure(np.array([1]), float("nan"), 5, settlement_idx=2)


def test_pending_receivable_rejects_noninteger_ko_idx():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    # float KO indices would be silently truncated (1.9 -> 1) onto the wrong date
    with pytest.raises(ValidationError):
        pending_receivable_exposure(np.array([1.9]), 100.0, 5, settlement_idx=3)


def test_pending_receivable_rejects_offgrid_settlement():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    # settlement at n_dates is off-grid: redemption never zeroes on a represented node
    with pytest.raises(ValidationError):
        pending_receivable_exposure(np.array([1]), 100.0, 4, settlement_idx=4)
    # offset settlement overflowing the grid must raise (extend the grid), not clamp
    with pytest.raises(ValidationError):
        pending_receivable_exposure(np.array([3]), 100.0, 4, settlement_offset_steps=2)


def test_repricer_rejects_nonbool_state_and_bad_exposure_idx():
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}},
                          currency="USD")
    spots = np.array([[100.0, 105.0]])
    # integer (non-bool) state masks would be read as fancy indices, not masks
    bad_state = {"alive": np.ones((1, 2), dtype=int),
                 "knocked_in": np.zeros((1, 2), dtype=int), "ko_idx": np.full(1, -1)}
    with pytest.raises(ValidationError):
        reprice_trade(vs, spots, bad_state, times=np.array([0.0, 1.0]),
                      quantity=1.0, exposure_idx=[1])
    # non-integer exposure_idx
    good = _alive_state(1, 2)
    with pytest.raises(ValidationError):
        reprice_trade(vs, spots, good, times=np.array([0.0, 1.0]),
                      quantity=1.0, exposure_idx=[1.0])


def test_repricer_rejects_single_state_labels_with_knocked_in_paths():
    # a stateful (KI) trade priced with default single-state labels would silently
    # price knocked-in paths as alive; must raise so caller passes both states
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}},
                          currency="USD")
    state = _alive_state(2, 2)
    state["knocked_in"][1, 1] = True
    spots = np.array([[100.0, 105.0], [100.0, 105.0]])
    with pytest.raises(ValidationError):
        reprice_trade(vs, spots, state, times=np.array([0.0, 1.0]),
                      quantity=1.0, exposure_idx=[1])   # state_labels defaults to (None,)


def test_repricer_rejects_invalid_state_labels():
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}},
                          currency="USD")
    spots = np.array([[100.0, 105.0]])
    times = np.array([0.0, 1.0])
    for bad in (("alive", "bogus"), ("knocked_in",), (None, "alive"), ("alive",)):
        with pytest.raises(ValidationError):
            reprice_trade(vs, spots, _alive_state(1, 2), times=times,
                          quantity=1.0, exposure_idx=[1], state_labels=bad)


def test_repricer_rejects_nonfinite_spots():
    from quantark.sacva.exposure.repricer import reprice_trade
    from quantark.sacva.exposure.value_surface import GridValueSurface
    vs = GridValueSurface(times=np.array([0.0, 1.0]),
                          grids={1.0: {None: (np.array([90., 110.]), np.array([0., 20.]))}},
                          currency="USD")
    spots = np.array([[100.0, float("nan")]])
    with pytest.raises(ValidationError):
        reprice_trade(vs, spots, _alive_state(1, 2), times=np.array([0.0, 1.0]),
                      quantity=1.0, exposure_idx=[1])


def test_aggregate_epe_requires_bool_enforceable():
    from quantark.sacva.exposure.engine import aggregate_epe
    with pytest.raises(ValidationError):
        aggregate_epe([np.array([[1.0]])], enforceable="False", df=np.array([1.0]))


def test_aggregate_epe_rejects_zero_paths():
    from quantark.sacva.exposure.engine import aggregate_epe
    with pytest.raises(ValidationError):
        aggregate_epe([np.zeros((0, 1))], enforceable=True, df=np.array([1.0]))


def test_aggregate_epe_rejects_zero_date_columns():
    from quantark.sacva.exposure.engine import aggregate_epe
    with pytest.raises(ValidationError):
        aggregate_epe([np.zeros((2, 0))], enforceable=True, df=np.zeros(0))


def test_grid_value_surface_rejects_empty_grids():
    from quantark.sacva.exposure.value_surface import GridValueSurface
    with pytest.raises(ValidationError):
        GridValueSurface(times=np.array([0.0, 1.0]), grids={}, currency="USD")


# --- Task 8: ExposureProfile + aggregate_epe ------------------------------

def test_exposure_profile_carries_measure_tag():
    from quantark.sacva.exposure.engine import ExposureProfile, Measure
    p = ExposureProfile(times=np.array([0., 1.]), epe_discounted=np.array([5., 3.]),
                        measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)
    assert p.regulatory_eligible and p.measure is Measure.RISK_NEUTRAL
    assert p.times.flags.writeable is False        # validated arrays are immutable
    assert p.epe_discounted.flags.writeable is False


def test_eligible_must_be_risk_neutral():
    from quantark.sacva.exposure.engine import ExposureProfile, Measure
    with pytest.raises(ValidationError):
        ExposureProfile(times=np.array([0., 1.]), epe_discounted=np.array([0., 1.]),
                        measure=Measure.REAL_WORLD, regulatory_eligible=True)


def test_exposure_profile_requires_grid_origin_and_measure_type():
    from quantark.sacva.exposure.engine import ExposureProfile, Measure
    # times must start at valuation (t0 = 0); else CVA drops the [0, t0] interval
    with pytest.raises(ValidationError):
        ExposureProfile(times=np.array([0.5, 1.0]), epe_discounted=np.array([1., 1.]),
                        measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)
    # measure must be a Measure enum
    with pytest.raises(ValidationError):
        ExposureProfile(times=np.array([0., 1.]), epe_discounted=np.array([0., 1.]),
                        measure="risk_neutral", regulatory_eligible=True)
    # non-finite time rejected
    with pytest.raises(ValidationError):
        ExposureProfile(times=np.array([0., np.inf]), epe_discounted=np.array([0., 1.]),
                        measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)
    # regulatory_eligible must be a real bool (truthiness would mis-gate CVA)
    with pytest.raises(ValidationError):
        ExposureProfile(times=np.array([0., 1.]), epe_discounted=np.array([0., 1.]),
                        measure=Measure.RISK_NEUTRAL, regulatory_eligible=1)


def test_aggregate_epe_rejects_bad_df_and_nonfinite():
    from quantark.sacva.exposure.engine import aggregate_epe
    v = [np.array([[10.], [-10.]])]
    # df must be 1-D of length n_dates: a column df would broadcast to a matrix
    with pytest.raises(ValidationError):
        aggregate_epe(v, enforceable=True, df=np.array([[1.0]]))
    # df length mismatch
    with pytest.raises(ValidationError):
        aggregate_epe(v, enforceable=True, df=np.array([1.0, 0.9]))
    # non-finite trade value
    with pytest.raises(ValidationError):
        aggregate_epe([np.array([[np.inf], [1.0]])], enforceable=True, df=np.array([1.0]))
    # non-2D trade array
    with pytest.raises(ValidationError):
        aggregate_epe([np.array([1.0, 2.0])], enforceable=True, df=np.array([1.0, 1.0]))
    # ragged trade arrays
    with pytest.raises(ValidationError):
        aggregate_epe([np.array([[1.0]]), np.array([[1.0, 2.0]])],
                      enforceable=True, df=np.array([1.0]))


def test_netting_positive_part_is_pathwise_before_averaging():
    from quantark.sacva.exposure.engine import aggregate_epe
    v = [np.array([[10.], [-10.]]), np.array([[-4.], [4.]])]
    netted = aggregate_epe(v, enforceable=True, df=np.array([1.0]))
    assert netted[0] == pytest.approx(3.0)        # mean(max(+6,0),max(-6,0)); not 0
    gross = aggregate_epe(v, enforceable=False, df=np.array([1.0]))
    assert gross[0] == pytest.approx(7.0)         # mean(10, 4)


def test_pending_receivable_then_aggregate_discounts_once():
    from quantark.sacva.exposure.repricer import pending_receivable_exposure
    from quantark.sacva.exposure.engine import aggregate_epe
    pe = pending_receivable_exposure(np.array([1]), redemption=100.0,
                                     n_dates=3, settlement_idx=2)
    epe = aggregate_epe([pe], enforceable=True, df=np.array([1.0, 0.97, 0.95]))
    assert epe[1] == pytest.approx(0.97 * 100.0)   # discounted exactly once
