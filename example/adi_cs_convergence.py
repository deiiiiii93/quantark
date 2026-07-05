"""WS-C1 validation artifact: Craig-Sneyd time self-convergence for the Heston ADI.

After the corrector fix (base Y0, not Y2) + implicit -rU, the CS scheme is second-order in
time. Order is measured by self-convergence at FIXED FINE space (successive-halving price
differences), so the fixed spatial error does not contaminate the temporal order.
"""
import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde


def _orders(P, r, ladder, n_x=500, n_v=250):
    ps = [price_european_heston_pde(100.0, 100.0, True, 1.0, P, r, 0.0, n_x=n_x, n_v=n_v,
                                    n_t=n_t, scheme=ADIScheme.CRAIG_SNEYD, rannacher=True)
          for n_t in ladder]
    d = [abs(ps[i] - ps[i + 1]) for i in range(len(ps) - 1)]
    orders = [np.log(d[i] / d[i + 1]) / np.log(2.0) for i in range(len(d) - 1)]
    return ps, d, orders


def main():
    P = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
    print("Heston Craig-Sneyd time self-convergence (fixed fine space n_x=500, n_v=250)")
    for r, ladder in ((0.0, (10, 20, 40, 80)), (0.05, (15, 30, 60, 120))):
        ps, d, orders = _orders(P, r, ladder)
        print(f"\n  r = {r:.2f}   n_t ladder = {ladder}")
        print(f"    prices     = {[f'{p:.6f}' for p in ps]}")
        print(f"    |ΔP|       = {[f'{x:.2e}' for x in d]}")
        print(f"    time order = {[f'{o:.2f}' for o in orders]}   (target ~2)")


if __name__ == "__main__":
    main()
