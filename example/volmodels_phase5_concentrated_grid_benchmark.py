"""WS-C2 acceptance artifact: concentrated vs uniform ADI equal-node error reduction.

Concentration helps most where the near-strike payoff kink dominates the error
(short maturities): >=4x at T=0.1. At long maturity the error is dominated by
V-boundary truncation / diffused-kink smear, so the win is modest (~1.1x). Reported
honestly across maturities — the default stays uniform; the flip is deferred.
"""
import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def err(style, params, T, nx, nv, nt=100):
    ref = heston_call_price(100.0, 100.0, T, params, 0.03, 0.0)
    p = price_european_heston_pde(100.0, 100.0, True, T, params, 0.03, 0.0,
                                  n_x=nx, n_v=nv, n_t=nt, scheme=ADIScheme.CRAIG_SNEYD,
                                  grid_style=style)
    return abs(p - ref)


def main():
    base = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    hf = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)
    print("WS-C2  concentrated vs uniform ADI (equal nodes n_x=80 n_v=40 n_t=100)")
    print(f"{'case':>16} {'T':>5} | {'uniform err':>12} {'concentr err':>12} {'ratio':>7}")
    for name, p, T in [("base", base, 1.0), ("base", base, 0.5), ("base", base, 0.25),
                       ("base", base, 0.1), ("Hout-Foulon", hf, 1.0), ("Hout-Foulon", hf, 0.1)]:
        eu = err("uniform", p, T, 80, 40)
        ec = err("concentrated", p, T, 80, 40)
        print(f"{name:>16} {T:5.2f} | {eu:12.4e} {ec:12.4e} {eu / max(ec, 1e-12):7.2f}")
    print("\nTakeaway: >=4x where the near-strike kink dominates (short T); "
          "marginal/neutral where boundary truncation dominates (long T, low v0).")


if __name__ == "__main__":
    main()
