import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.analytical_kernel import heston_call_price
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.fp_solver import ForwardFPADI


def _run(scheme_implicit_all):
    s0, K, T, r = 100.0, 100.0, 1.0, 0.02
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=241, n_z=121, rannacher_steps=2)
    n_t = 80
    step_dt = np.full(n_t, T / n_t)
    solver = ForwardFPADI.from_config(s0, p, eta=1.0, b=r, step_dt=step_dt, config=cfg)
    f = solver.seed_dirac(s0, p.v0)
    L = np.ones(solver.x.size)
    for n in range(n_t):
        implicit = scheme_implicit_all or (n < cfg.rannacher_steps)
        f = solver.step(f, L, step_dt[n], implicit=implicit)
    marg = solver.spot_marginal(f)
    S = np.exp(solver.x)
    return np.exp(-r * T) * np.trapezoid(np.maximum(S - K, 0.0) * marg / S, S)


def test_craig_sneyd_matches_backward_euler_and_analytic():
    cs = _run(scheme_implicit_all=False)              # CS after Rannacher start
    be = _run(scheme_implicit_all=True)               # all backward-Euler (Task 5 solver)
    analytic = heston_call_price(
        100.0, 100.0, 1.0,
        HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5), 0.02, 0.0)
    assert abs(cs - be) < 0.05                         # CS agrees with the robust reference
    assert abs(cs - analytic) < 0.12                   # and with the analytic Heston price
