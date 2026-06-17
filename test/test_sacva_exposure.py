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
