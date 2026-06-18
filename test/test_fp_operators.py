import numpy as np
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid, trapezoid_weights
from quantark.volmodels.slv.fokkerplanck.fp_operators import (
    build_forward_operator, build_directional_operators,
)


def _grid(params):
    x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), 61, 0.1)
    z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(params.v0), 41, 0.1)
    return x, z


def _mass_residual(A, x, z):
    w = np.outer(trapezoid_weights(x), trapezoid_weights(z)).ravel()
    Ad = A.toarray()
    return np.max(np.abs(w @ Ad)) / max(np.max(np.abs(Ad)), 1e-300)


def test_mass_conserved_heston_L1():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    x, z = _grid(p)
    A = build_forward_operator(x, z, np.ones(x.size), params=p, eta=1.0, b=0.01)
    assert _mass_residual(A, x, z) < 1e-9


def test_mass_conserved_with_varying_leverage_and_correlation():
    # the tricky case: leverage varies in x and rho != 0 (mixed flux must be conservative)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    x, z = _grid(p)
    L = np.linspace(0.8, 1.3, x.size)
    A = build_forward_operator(x, z, L, params=p, eta=1.0, b=0.02)
    assert _mass_residual(A, x, z) < 1e-9


def test_directional_split_sums_to_full_operator():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    x, z = _grid(p)
    L = np.linspace(0.9, 1.1, x.size)
    A = build_forward_operator(x, z, L, params=p, eta=1.0, b=0.01)
    Ax, Az, Axz = build_directional_operators(x, z, L, params=p, eta=1.0, b=0.01)
    diff = (A - (Ax + Az + Axz)).toarray()
    assert np.max(np.abs(diff)) < 1e-12
