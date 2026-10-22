import numpy as np
from quantark.volmodels.slv.leverage import LeverageSurface


def test_diagnostics_defaults_none_and_is_optional():
    s = LeverageSurface(time_grid=np.array([0.0, 0.5]),
                        strike_grid=np.array([90.0, 100.0, 110.0]),
                        leverage_grid=np.ones((2, 3)))
    assert s.diagnostics is None
    assert np.isclose(s.leverage(100.0, 0.25), 1.0)        # interpolation unaffected


def test_diagnostics_roundtrips_when_provided():
    diag = {"mass_residual": [1e-9, 2e-9], "n_clipped": 0}
    s = LeverageSurface(time_grid=np.array([0.0]),
                        strike_grid=np.array([90.0, 110.0]),
                        leverage_grid=np.ones((1, 2)), diagnostics=diag)
    assert s.diagnostics["n_clipped"] == 0
