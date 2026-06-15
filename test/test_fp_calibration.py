import numpy as np
import pytest
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv.leverage import LeverageSurface
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.calibration import calibrate_leverage_surface_fp


def _flat_lv(level=0.20):
    x = np.linspace(40.0, 250.0, 60)
    t = np.linspace(0.0, 1.0, 12)
    return LocalVolSurface(strike_grid=x, time_grid=t, lv_grid=np.full((t.size, x.size), level))


def test_eta_zero_is_exact_deterministic_leverage():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    lv = _flat_lv(0.20)
    step_dt = np.full(10, 0.1)
    surf = calibrate_leverage_surface_fp(100.0, p, lv, step_dt, np.zeros(10), np.zeros(10),
                                         eta=0.0, config=FpCalibrationConfig())
    assert isinstance(surf, LeverageSurface)
    # deterministic variance nu_det(t)=theta+(v0-theta)e^{-k t}; with theta=v0 it is constant v0
    expected = 0.20 / np.sqrt(p.v0)
    assert np.allclose(surf.leverage(100.0, 0.3), expected, rtol=1e-6)


def test_returns_leverage_surface_with_diagnostics():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = calibrate_leverage_surface_fp(100.0, p, _flat_lv(0.20), np.full(20, 0.05),
                                         np.full(20, 0.0), np.full(20, 0.0), eta=1.0,
                                         config=FpCalibrationConfig(n_x=161, n_z=81))
    assert isinstance(surf, LeverageSurface)
    assert surf.diagnostics is not None and "mass_residual" in surf.diagnostics


@pytest.mark.parametrize("bad", [
    dict(v0=0.0, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5),
    dict(v0=0.04, kappa=2.0, theta=0.0, sigma=0.3, rho=-0.5),
])
def test_invalid_params_raise(bad):
    with pytest.raises(ValidationError):
        calibrate_leverage_surface_fp(100.0, HestonParams(**bad), _flat_lv(), np.full(5, 0.1),
                                      np.zeros(5), np.zeros(5), eta=1.0)
