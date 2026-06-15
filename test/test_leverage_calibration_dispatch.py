import numpy as np
import pytest
from quantark.util.enum.engine_enums import LeverageCalibrationMethod
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig


def _lv():
    x = np.linspace(40.0, 250.0, 60)
    t = np.linspace(0.0, 1.0, 12)
    return LocalVolSurface(strike_grid=x, time_grid=t, lv_grid=np.full((t.size, x.size), 0.20))


def test_default_method_is_ffp():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = calibrate_leverage_surface(100.0, p, _lv(), np.full(20, 0.05),
                                      np.zeros(20), np.zeros(20), eta=1.0,
                                      fp_config=FpCalibrationConfig(n_x=121, n_z=61))
    assert surf.diagnostics["method"] == "forward_fokker_planck"


def test_mc_binning_still_available_when_pinned():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = calibrate_leverage_surface(100.0, p, _lv(), np.full(20, 0.05),
                                      np.zeros(20), np.zeros(20), eta=1.0,
                                      method=LeverageCalibrationMethod.MC_BINNING,
                                      num_paths=20_000, seed=1)
    assert surf.diagnostics is None                       # MC path leaves diagnostics None


def test_mismatched_options_raise():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    with pytest.raises(ValidationError):                  # MC kwargs under FFP default
        calibrate_leverage_surface(100.0, p, _lv(), np.full(5, 0.1),
                                   np.zeros(5), np.zeros(5), eta=1.0, num_paths=10_000)
    with pytest.raises(ValidationError):                  # explicit seed=None is still an MC option under FFP
        calibrate_leverage_surface(100.0, p, _lv(), np.full(5, 0.1),
                                   np.zeros(5), np.zeros(5), eta=1.0, seed=None)
    with pytest.raises(ValidationError):                  # fp_config under MC
        calibrate_leverage_surface(100.0, p, _lv(), np.full(5, 0.1),
                                   np.zeros(5), np.zeros(5), eta=1.0,
                                   method=LeverageCalibrationMethod.MC_BINNING,
                                   fp_config=FpCalibrationConfig())


def test_positional_mc_options_bind_correctly():
    # old positional slots (num_paths, num_bins) bind to MC options, not to method/fp_config
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = calibrate_leverage_surface(100.0, p, _lv(), np.full(20, 0.05),
                                      np.zeros(20), np.zeros(20), 1.0, 15_000, 18,
                                      method=LeverageCalibrationMethod.MC_BINNING)
    assert surf.diagnostics is None                       # ran MC binning (positional num_paths/num_bins)


def test_mc_seed_none_is_nondeterministic_not_dropped():
    # explicit seed=None under MC means "no fixed seed" -> two runs differ (not silently seed=42)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    kw = dict(eta=1.0, method=LeverageCalibrationMethod.MC_BINNING, num_paths=8_000, seed=None)
    a = calibrate_leverage_surface(100.0, p, _lv(), np.full(15, 1.0 / 15),
                                   np.zeros(15), np.zeros(15), **kw)
    b = calibrate_leverage_surface(100.0, p, _lv(), np.full(15, 1.0 / 15),
                                   np.zeros(15), np.zeros(15), **kw)
    assert not np.array_equal(a.leverage_grid, b.leverage_grid)
