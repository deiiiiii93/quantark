import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _surface():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    return GridVolSurface(strikes, mats, np.full((6, 9), 0.20))


def _calibrate(time_scheme, n, **cfg_over):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    lv = build_dupire_local_vol(_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=101, time_scheme=time_scheme, **cfg_over)
    return calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                      np.full(n, 0.0), eta=1.0, fp_config=cfg)


def _err_at(lev, ref):
    # ATM leverage path at t=0.5: a scalar functional of the surface, error vs the fine-dt reference
    return abs(lev.leverage(100.0, 0.5) - ref.leverage(100.0, 0.5))


def test_trbdf2_second_order_in_time():
    ref = _calibrate("tr_bdf2", 160)                  # fine-dt reference
    e = [_err_at(_calibrate("tr_bdf2", n), ref) for n in (20, 40, 80)]
    # successive halving of dt -> error ratio ~4 (second order). Require > 3.0 (spatial-floor slack).
    r1 = e[0] / max(e[1], 1e-300)
    r2 = e[1] / max(e[2], 1e-300)
    assert r1 > 3.0 and r2 > 3.0


def test_backward_euler_is_first_order_for_contrast():
    ref = _calibrate("backward_euler", 160)
    e = [_err_at(_calibrate("backward_euler", n), ref) for n in (20, 40, 80)]
    r1 = e[0] / max(e[1], 1e-300)
    # backward Euler halving gives ratio ~2 (first order): clearly below TR-BDF2's ~4
    assert 1.4 < r1 < 3.0


def test_trbdf2_mass_and_negativity_no_worse_than_be():
    be = _calibrate("backward_euler", 60)
    tr = _calibrate("tr_bdf2", 60)
    assert max(tr.diagnostics["mass_residual"]) < 1e-3
    assert tr.diagnostics["max_negative_mass"] <= 10.0 * be.diagnostics["max_negative_mass"] + 1e-9


def test_trbdf2_krylov_reuses_matrix_within_step(monkeypatch):
    # TR-BDF2 does two solves per step against the SAME M (gamma=2-sqrt2). With refactor_every=1 the
    # lagged-Krylov cadence must refresh only ONCE per step (advance=False on the 2nd substep), not twice.
    import quantark.volmodels.slv.fokkerplanck.fp_solver as mod
    calls = {"n": 0}
    real = mod.splu

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "splu", _spy)
    n = 12
    _calibrate("tr_bdf2", n, linear_solver="krylov_lagged", refactor_every=1)
    assert calls["n"] == n                            # once/step (step 0 backward-Euler start-up), NOT 2n-1
