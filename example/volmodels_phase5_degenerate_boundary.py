"""WS-C3 acceptance artifact: degenerate v=0 boundary on Feller-violated Heston.

At v=0 the CIR diffusion vanishes; the Neumann row is inaccurate when Feller is
violated (2*kappa*theta << sigma^2). The degenerate PDE row (kappa*theta*U_v forward)
cuts the PDE error materially, while leaving Feller-satisfied cases essentially unchanged.
"""
import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def compare(params, label):
    ref = heston_call_price(100.0, 100.0, 1.0, params, 0.03, 0.0)
    common = dict(n_x=200, n_v=80, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    neu = price_european_heston_pde(100.0, 100.0, True, 1.0, params, 0.03, 0.0,
                                    v0_boundary="neumann", **common)
    deg = price_european_heston_pde(100.0, 100.0, True, 1.0, params, 0.03, 0.0,
                                    v0_boundary="degenerate_pde", **common)
    feller = 2.0 * params.kappa * params.theta / (params.sigma ** 2)
    print(f"{label:>18} (2kt/s2={feller:5.2f}): ref={ref:8.5f} "
          f"neumann_err={abs(neu-ref):.3e} degenerate_err={abs(deg-ref):.3e} "
          f"improvement={abs(neu-ref)/max(abs(deg-ref),1e-12):5.2f}x")


def main():
    print("WS-C3  degenerate v=0 boundary — PDE error vs analytical (n_x=200 n_v=80)")
    compare(HestonParams(kappa=0.5, theta=0.04, sigma=0.9, rho=-0.5, v0=0.04), "Feller-violated")
    compare(HestonParams(kappa=1.0, theta=0.04, sigma=0.7, rho=-0.5, v0=0.04), "Feller-violated 2")
    compare(HestonParams(kappa=3.0, theta=0.04, sigma=0.2, rho=-0.5, v0=0.04), "Feller-satisfied")


if __name__ == "__main__":
    main()
