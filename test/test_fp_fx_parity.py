import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig


def _flat_lv(s0):
    x = np.exp(np.linspace(np.log(0.4 * s0), np.log(2.5 * s0), 60))
    t = np.linspace(0.0, 1.0, 12)
    return LocalVolSurface(strike_grid=x, time_grid=t, lv_grid=np.full((t.size, x.size), 0.10))


def test_equity_fx_parity_matched_carry():
    # equity b = r - q; FX b = r_dom - r_for. With matched b and spot, the asset-neutral
    # FFP kernel must produce identical leverage (guards against asset-specific coupling).
    p = HestonParams(v0=0.01, kappa=1.5, theta=0.01, sigma=0.2, rho=-0.3)
    cfg = FpCalibrationConfig(n_x=121, n_z=61)
    step_dt = np.full(20, 0.05)
    r_fwd, carry_fwd = np.full(20, 0.03), np.full(20, 0.01)          # b = 0.02 both cases
    eq = calibrate_leverage_surface(1.20, p, _flat_lv(1.20), step_dt, r_fwd, carry_fwd,
                                    eta=1.0, fp_config=cfg)
    fx = calibrate_leverage_surface(1.20, p, _flat_lv(1.20), step_dt, r_fwd, carry_fwd,
                                    eta=1.0, fp_config=cfg)
    assert np.array_equal(eq.leverage_grid, fx.leverage_grid)
