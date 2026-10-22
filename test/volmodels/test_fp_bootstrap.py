import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.fp_solver import ForwardFPADI
from quantark.volmodels.slv.fokkerplanck.bootstrap import conditional_variance, leverage_from_slice


def test_conditional_variance_of_seed_is_v0():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=0.0)
    cfg = FpCalibrationConfig(n_x=81, n_z=61)
    solver = ForwardFPADI.from_config(100.0, p, eta=1.0, b=0.0,
                                      step_dt=np.full(10, 0.1), config=cfg)
    f = solver.seed_dirac(100.0, p.v0)
    i = int(np.argmin(np.abs(solver.x - np.log(100.0))))
    ev = conditional_variance(solver, f, cfg)         # array over x
    # at the support node E[v|x] == variance of the seeded z-node ~= v0 (to grid resolution)
    assert np.isclose(ev[i], p.v0, rtol=1e-2)


def test_leverage_blends_to_unconditional_in_empty_tail():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=0.0)
    cfg = FpCalibrationConfig(n_x=81, n_z=61)
    solver = ForwardFPADI.from_config(100.0, p, eta=1.0, b=0.0,
                                      step_dt=np.full(10, 0.1), config=cfg)
    f = solver.seed_dirac(100.0, p.v0)
    sigma_lv = np.full(solver.x.size, 0.20)
    L, diag = leverage_from_slice(solver, f, sigma_lv, t=0.1, params=p, eta=1.0, cfg=cfg)
    assert np.all(np.isfinite(L)) and np.all(L > 0)
    # deep tail (no mass) -> uses unconditional mean E[v]=theta+(v0-theta)e^{-k t}
    ev_uncond = p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * 0.1)
    assert np.isclose(L[0], 0.20 / np.sqrt(ev_uncond), rtol=1e-3)
    assert diag["n_tail_blended"] > 0
