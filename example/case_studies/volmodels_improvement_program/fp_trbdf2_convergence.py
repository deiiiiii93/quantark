"""WS-C6 benchmark: TR-BDF2 vs backward-Euler leverage-surface convergence in dt.

Prints the error-vs-fine-reference ladder and the observed temporal order (log2 of successive error
ratios). TR-BDF2 (gamma=2-sqrt2, L-stable, backward-Euler start-up) should show order ~2; backward
Euler ~1. Run: python example/fp_trbdf2_convergence.py
"""
import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _calibrate(time_scheme, n):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = GridVolSurface(list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9))),
                          list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), 0.20))
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=101, time_scheme=time_scheme)
    return calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                      np.full(n, 0.0), eta=1.0, fp_config=cfg)


def _order(scheme):
    ref = _calibrate(scheme, 160)
    errs = [abs(_calibrate(scheme, n).leverage(100.0, 0.5) - ref.leverage(100.0, 0.5))
            for n in (20, 40, 80)]
    orders = [round(np.log2(errs[i] / errs[i + 1]), 2) for i in range(len(errs) - 1)]
    print(f"{scheme:15s}: errs={['%.2e' % e for e in errs]}  orders={orders}")


def main():
    _order("backward_euler")
    _order("tr_bdf2")


if __name__ == "__main__":
    main()
