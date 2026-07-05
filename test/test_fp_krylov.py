import numpy as np
import pytest

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _surface():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    grid = np.full((6, 9), 0.20)
    return GridVolSurface(strikes, mats, grid)


def _calibrate(linear_solver, rho=-0.5, n=30, **cfg_over):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=rho)
    lv = build_dupire_local_vol(_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=101, linear_solver=linear_solver, **cfg_over)
    return calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                      np.full(n, 0.0), eta=1.0, fp_config=cfg)


@pytest.mark.parametrize("rho", [-0.5, -0.9])
def test_krylov_fresh_preconditioner_matches_direct(rho):
    # refactor_every=1 factorizes every step, so the preconditioner is exact and BiCGStab converges
    # in one iteration to machine precision -> Krylov reproduces direct to < 1e-10 (spec acceptance).
    a = _calibrate("direct", rho=rho)
    b = _calibrate("krylov_lagged", rho=rho, refactor_every=1)
    assert np.max(np.abs(a.leverage_grid - b.leverage_grid)) < 1e-10


@pytest.mark.parametrize("rho", [-0.5, -0.9])
def test_krylov_lagged_parity_is_calibration_negligible(rho):
    # With a LAGGED preconditioner (default refactor_every=5) the stale factorization + BiCGStab rtol
    # leaves a small, calibration-negligible gap vs direct (~1e-7 on leverage ~ O(1), sub-basis-point).
    # This is the genuine accuracy/speed knob of the opt-in Krylov mode; direct stays the default.
    a = _calibrate("direct", rho=rho)
    b = _calibrate("krylov_lagged", rho=rho, refactor_every=5)
    assert np.max(np.abs(a.leverage_grid - b.leverage_grid)) < 1e-6


def test_krylov_mode_actually_uses_bicgstab(monkeypatch):
    import quantark.volmodels.slv.fokkerplanck.fp_solver as mod
    calls = {"n": 0}
    real = mod.bicgstab

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "bicgstab", _spy)
    _calibrate("krylov_lagged", n=10)
    assert calls["n"] > 0                         # the march went through BiCGStab


def _count_splu(monkeypatch, **cfg_over):
    import quantark.volmodels.slv.fokkerplanck.fp_solver as mod
    calls = {"n": 0}
    real = mod.splu

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "splu", _spy)
    n = 12
    _calibrate("krylov_lagged", n=n, **cfg_over)
    return calls["n"], n


def test_krylov_refactor_cadence_advances_once_per_step(monkeypatch):
    # refactor_every=1 => one splu refresh per MARCH step (n steps). backward-Euler does one solve/step,
    # so the counter ticks exactly n times (no convergence-failure refactors on this benign fixture).
    c, n = _count_splu(monkeypatch, refactor_every=1)
    assert c == n
