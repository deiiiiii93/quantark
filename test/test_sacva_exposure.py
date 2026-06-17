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
