"""WS-B3 benchmark: krylov_lagged vs direct FP march -- leverage parity + wall-clock.

The krylov_lagged mode reuses a stale splu factorization as a BiCGStab preconditioner, refreshed every
refactor_every steps. With refactor_every=1 the preconditioner is exact (parity ~ machine precision, no
speedup); with refactor_every>1 it trades a small, calibration-negligible parity gap for fewer
factorizations. direct stays the default. Run: python example/fp_krylov_benchmark.py
"""
import time
import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _run(linear_solver, n=60, refactor_every=5):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.9)
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), 0.20))
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=201, n_z=141, linear_solver=linear_solver, refactor_every=refactor_every)
    t0 = time.perf_counter()
    lev = calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                     np.full(n, 0.0), eta=1.0, fp_config=cfg)
    return lev, time.perf_counter() - t0


def main():
    direct, td = _run("direct")
    print(f"direct                        : {td:.2f}s")
    for refac in (1, 5, 10):
        k, tk = _run("krylov_lagged", refactor_every=refac)
        err = float(np.max(np.abs(direct.leverage_grid - k.leverage_grid)))
        print(f"krylov_lagged (refactor={refac:2d}) : {tk:.2f}s  (speedup {td / tk:.2f}x)  "
              f"max leverage diff = {err:.2e}")
    print("\nNote: refactor_every=1 => exact (machine-precision parity, no speedup); larger => faster")
    print("with a small calibration-negligible parity gap. Krylov is an opt-in escape hatch (default: direct).")


if __name__ == "__main__":
    main()
