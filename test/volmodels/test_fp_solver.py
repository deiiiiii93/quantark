import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.analytical_kernel import heston_call_price
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.fp_solver import ForwardFPADI


_P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _solver(seed_split):
    cfg = FpCalibrationConfig(n_x=81, n_z=61, seed_split=seed_split)
    return ForwardFPADI.from_config(100.0, _P, eta=1.0, b=0.0,
                                    step_dt=np.full(10, 0.1), config=cfg)


def test_seed_has_unit_mass():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=81, n_z=61)
    solver = ForwardFPADI.from_config(100.0, p, eta=1.0, b=0.0,
                                      step_dt=np.full(10, 0.1), config=cfg)
    f = solver.seed_dirac(100.0, p.v0)
    assert abs(solver.total_mass(f) - 1.0) < 1e-12


def test_seed_unit_mass_both_modes():
    for split in (False, True):
        s = _solver(split)
        f = s.seed_dirac(100.0, _P.v0)
        assert abs(s.total_mass(f) - 1.0) < 1e-12


def test_default_seed_is_nearest_node():
    # seed_split=False (default) keeps the legacy single-node seed bit-identically
    s = _solver(False)
    f = s.seed_dirac(100.0, _P.v0)
    assert int(np.count_nonzero(f)) == 1


def test_bilinear_seed_preserves_mean_to_second_order():
    # seed placed strictly between nodes: bilinear split matches (ln s0, ln v0) to O(h^2), <=4 nodes
    s = _solver(True)
    xs = 0.5 * (s.x[40] + s.x[41]); zs = 0.5 * (s.z[30] + s.z[31])
    f = s.seed_dirac(np.exp(xs), np.exp(zs))
    assert abs(s.total_mass(f) - 1.0) < 1e-12
    F = f.reshape(s.nx, s.nz)
    mean_x = float((s.w.reshape(s.nx, s.nz) * F).sum(axis=1) @ s.x)
    mean_z = float((s.w.reshape(s.nx, s.nz) * F).sum(axis=0) @ s.z)
    hx = s.x[41] - s.x[40]; hz = s.z[31] - s.z[30]
    assert abs(mean_x - xs) < hx ** 2 + 1e-12
    assert abs(mean_z - zs) < hz ** 2 + 1e-12
    assert int(np.count_nonzero(f)) <= 4


def test_heston_density_oracle_reprices_analytic_call():
    s0, K, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=241, n_z=121)
    rannacher_steps = 2                               # implicit start-up steps (local test choice)
    n_t = 80
    step_dt = np.full(n_t, T / n_t)
    solver = ForwardFPADI.from_config(s0, p, eta=1.0, b=r - q, step_dt=step_dt, config=cfg)
    f = solver.seed_dirac(s0, p.v0)
    L = np.ones(solver.x.size)                        # Heston: leverage = 1 everywhere
    for n in range(n_t):
        f = solver.step(f, L, step_dt[n], implicit=(n < rannacher_steps))  # Rannacher start + CS
        assert abs(solver.total_mass(f) - 1.0) < cfg.mass_tol
    marg = solver.spot_marginal(f)                    # density in x = ln S
    S = np.exp(solver.x)
    payoff = np.maximum(S - K, 0.0)
    fp_price = np.exp(-r * T) * np.trapezoid(payoff * marg / S, S)
    analytic = heston_call_price(s0, K, T, p, r, q)
    assert abs(fp_price - analytic) < 0.15
