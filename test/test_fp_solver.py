import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.analytical_kernel import heston_call_price
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.fp_solver import ForwardFPADI


def test_seed_has_unit_mass():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=81, n_z=61)
    solver = ForwardFPADI.from_config(100.0, p, eta=1.0, b=0.0,
                                      step_dt=np.full(10, 0.1), config=cfg)
    f = solver.seed_dirac(100.0, p.v0)
    assert abs(solver.total_mass(f) - 1.0) < 1e-12


def test_heston_density_oracle_reprices_analytic_call():
    s0, K, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=241, n_z=121, rannacher_steps=2)
    n_t = 80
    step_dt = np.full(n_t, T / n_t)
    solver = ForwardFPADI.from_config(s0, p, eta=1.0, b=r - q, step_dt=step_dt, config=cfg)
    f = solver.seed_dirac(s0, p.v0)
    L = np.ones(solver.x.size)                        # Heston: leverage = 1 everywhere
    for n in range(n_t):
        f = solver.step(f, L, step_dt[n], implicit=(n < cfg.rannacher_steps))  # Rannacher start + CS
        assert abs(solver.total_mass(f) - 1.0) < cfg.mass_tol
    marg = solver.spot_marginal(f)                    # density in x = ln S
    S = np.exp(solver.x)
    payoff = np.maximum(S - K, 0.0)
    fp_price = np.exp(-r * T) * np.trapezoid(payoff * marg / S, S)
    analytic = heston_call_price(s0, K, T, p, r, q)
    assert abs(fp_price - analytic) < 0.15
