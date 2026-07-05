"""WS-C4 investigation diagnostic: where does the forward-FP negativity come from?

This script reproduces the 2026-07 finding that motivated *not* shipping Chang-Cooper/SG fluxes:

  1. The x and z directional operators are ALREADY M-matrices (zero negative off-diagonals), so
     exponential-fitting (Scharfetter-Gummel / Chang-Cooper) on them cannot reduce negativity.
  2. ALL negativity is the mixed (correlation) cross-derivative term: at rho=0 the density stays
     non-negative to machine precision.
  3. The operator is extremely anisotropic in log-variance (b/a up to ~4900x), so |c| > min(a,b)
     almost everywhere and the rotated/seven-point positive mixed stencil is infeasible -- a genuine
     positivity fix needs a state-dependent sheared grid (a separate FP redesign), not WS-C4.
  4. The residual negativity is small (< 0.03 of unit mass on the worst realistic fixture) and bounded
     under refinement, which is why tol_neg is tightened to 0.05 as an empirical operating tripwire.

Run: python example/fp_positivity_diagnostic.py
"""
import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid
from quantark.volmodels.slv.fokkerplanck.fp_operators import build_directional_operators

_ZERO = (lambda t: 0.0)


def _skew_surface():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    ks = np.linspace(-0.6, 0.6, 9)
    grid = np.array([[0.22 - 0.04 * ks[j] for j in range(9)] for _ in mats])
    return GridVolSurface(strikes, mats, grid)


def _offdiag_min(A):
    Ad = A.toarray().copy()
    np.fill_diagonal(Ad, 0.0)
    return float(Ad.min())


def _max_neg(sigma, rho, n=60):
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=sigma, rho=rho)
    lv = build_dupire_local_vol(_skew_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO)
    t = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=121, tol_neg=0.99)
    lev = calibrate_leverage_surface(100.0, p, lv, np.diff(t), np.full(n, 0.02),
                                     np.full(n, 0.0), eta=1.0, fp_config=cfg)
    return lev.diagnostics["max_negative_mass"]


def main():
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.7, rho=-0.9)
    x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), 31, 0.1)
    z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(p.v0), 25, 0.1)
    L = np.linspace(0.8, 1.3, x.size)
    Ax, Az, Axz = build_directional_operators(x, z, L, p, 1.0, 0.05)
    print("=== off-diagonal minimum by operator (>=0 means M-matrix) ===")
    print(f"  Ax  (x-direction) : {_offdiag_min(Ax):+.3e}")
    print(f"  Az  (z-direction) : {_offdiag_min(Az):+.3e}")
    print(f"  Axz (mixed/corr)  : {_offdiag_min(Axz):+.3e}  <- sole M-matrix violator")

    nu = np.exp(z); se = 0.7
    a = 1.0 ** 2 * nu; b = se ** 2 / nu; c = abs(-0.9) * 1.0 * se
    print("\n=== anisotropy (a=L^2 nu, b=sigma^2/nu, |c|=|rho|L sigma) ===")
    print(f"  b/a up to {float((b / a).max()):.0f}x ;  a>=|c| at "
          f"{100 * float(np.mean(a >= c)):.0f}% of nodes (rotated positive scheme needs 100%)")

    print("\n=== max negative probability mass: mixed term is the whole budget ===")
    for sigma in (0.7, 0.5, 0.3):
        print(f"  sigma={sigma}: rho=0.0 -> {_max_neg(sigma, 0.0):.3e}   "
              f"rho=-0.9 -> {_max_neg(sigma, -0.9):.3e}")
    print("\nConclusion: negativity is intrinsic to the anisotropic mixed term; it is small (<0.03),")
    print("bounded, and clamped in the read-off. tol_neg=0.05 is an empirical operating tripwire.")


if __name__ == "__main__":
    main()
