import numpy as np

from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid, trapezoid_weights
from quantark.volmodels.slv.fokkerplanck.fp_operators import (
    build_forward_operator, build_directional_operators, _cc_delta,
)

_P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _grid(params=_P):
    x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), 61, 0.1)
    z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(params.v0), 41, 0.1)
    return x, z


def _mass_residual(A, x, z):
    w = np.outer(trapezoid_weights(x), trapezoid_weights(z)).ravel()
    Ad = A.toarray()
    return np.max(np.abs(w @ Ad)) / max(np.max(np.abs(Ad)), 1e-300)


def test_cc_delta_central_limit():
    # delta -> 1/2 as P -> 0 (guarded series), monotone-ish, finite for large |P|
    P = np.array([-50.0, -1e-3, 0.0, 1e-3, 50.0])
    d = _cc_delta(P)
    assert np.all(np.isfinite(d))
    assert abs(d[2] - 0.5) < 1e-12          # exactly 1/2 at P=0
    assert abs(d[1] - 0.5) < 1e-3 and abs(d[3] - 0.5) < 1e-3


def _smooth_density(x, z):
    X, Z = np.meshgrid(x, z, indexing="ij")
    f = np.exp(-((X - np.log(100.0)) ** 2) / 0.1 - ((Z - np.log(_P.v0)) ** 2) / 0.5).ravel()
    return f / f.sum()


def test_cc_converges_to_central_under_refinement():
    # CC differs from central by an artificial-diffusion flux with relative size P^2/12
    # (delta - 1/2 = -P/12, P = mu_f*h/D_f). The matrix ENTRIES differ by O(1) (flux-coeff O(h)
    # over quadrature weight O(h)), but the ACTION on a smooth density is an O(h^2)-relative
    # perturbation that shrinks under grid refinement. This is the defensible central-recovery
    # invariant (an entry-wise A_cc==A_c match is impossible: mu = b - D is never ~0 here).
    def gap(nx, nz):
        x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), nx, 0.1)
        z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(_P.v0), nz, 0.1)
        L = np.ones(x.size)
        Ac = build_forward_operator(x, z, L, params=_P, eta=1.0, b=0.05, flux_scheme="central")
        Acc = build_forward_operator(x, z, L, params=_P, eta=1.0, b=0.05, flux_scheme="chang_cooper")
        f = _smooth_density(x, z)
        return np.linalg.norm((Acc - Ac) @ f) / max(np.linalg.norm(Ac @ f), 1e-300)

    g1, g2 = gap(61, 41), gap(121, 81)
    assert g2 < g1                           # gap shrinks under refinement (CC -> central as h -> 0)
    assert g2 < 0.6 * g1                     # decreasing faster than first order (artificial diffusion ~ P^2)


def test_cc_mass_conserved_high_correlation():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.9)  # Feller-violated, |rho|=0.9
    x, z = _grid(p)
    L = np.linspace(0.8, 1.3, x.size)
    A = build_forward_operator(x, z, L, params=p, eta=1.0, b=0.05, flux_scheme="chang_cooper")
    assert _mass_residual(A, x, z) < 1e-9    # telescoping flux => still mass-conserving


def test_cc_directional_split_sums_to_full():
    x, z = _grid()
    L = np.linspace(0.9, 1.1, x.size)
    A = build_forward_operator(x, z, L, params=_P, eta=1.0, b=0.02, flux_scheme="chang_cooper")
    Ax, Az, Axz = build_directional_operators(x, z, L, params=_P, eta=1.0, b=0.02,
                                              flux_scheme="chang_cooper")
    assert np.max(np.abs((A - (Ax + Az + Axz)).toarray())) < 1e-12
