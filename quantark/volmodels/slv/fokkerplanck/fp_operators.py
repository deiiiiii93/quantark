"""Forward Fokker-Planck operators for the SLV log-variance density f(x,z)=nu*p.

Conservative (flux-divergence) finite-volume discretization of the paper section 4.5.3
equation. Each direction is assembled from face fluxes so that the quadrature-weighted
constant vector is a left-null vector of the operator (mass conservation), and the
zero-flux boundary condition is enforced simply by carrying zero flux through the
domain-boundary faces. The three directional pieces (x, z, mixed) sum to the full
generator and are exposed separately for the Craig-Sneyd ADI split.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.fokkerplanck.coordinates import trapezoid_weights


def _check(x, z, L):
    x = np.asarray(x, float); z = np.asarray(z, float); L = np.asarray(L, float)
    if L.shape != (x.size,):
        raise ValidationError("L must have shape (nx,)")
    return x, z, L


def _x_operator(x, z, L, params: HestonParams, eta: float, b: float) -> sp.lil_matrix:
    """x-direction flux: J_x = (b - 0.5 nu L^2) f - d_x(0.5 nu L^2 f). Conservative central FV."""
    nx, nz = x.size, z.size
    nu = np.exp(z)
    wx = trapezoid_weights(x)
    A = sp.lil_matrix((nx * nz, nx * nz))
    idx = lambda i, j: i * nz + j
    for j in range(nz):
        D = 0.5 * nu[j] * L ** 2                      # D_x at each node (depends on x via L)
        mu = b - D                                    # advective drift at each node
        for i in range(nx - 1):                       # interior face i+1/2
            h = x[i + 1] - x[i]
            mu_f = 0.5 * (mu[i] + mu[i + 1])
            cL = 0.5 * mu_f + D[i] / h                 # coeff on f_i in J_{i+1/2}
            cR = 0.5 * mu_f - D[i + 1] / h             # coeff on f_{i+1}
            A[idx(i, j),     idx(i, j)]     += -cL / wx[i]
            A[idx(i, j),     idx(i + 1, j)] += -cR / wx[i]
            A[idx(i + 1, j), idx(i, j)]     += +cL / wx[i + 1]
            A[idx(i + 1, j), idx(i + 1, j)] += +cR / wx[i + 1]
    return A


def _z_operator(x, z, L, params: HestonParams, eta: float, b: float) -> sp.lil_matrix:
    """z-direction flux: J_z = ((k th - se^2/2)/nu - k) f - d_z(0.5 se^2 / nu f). Conservative FV."""
    nx, nz = x.size, z.size
    nu = np.exp(z)
    se2 = (eta * params.sigma) ** 2
    kappa, theta = params.kappa, params.theta
    Dz = 0.5 * se2 / nu                               # D_z at each z node
    mu_z = (kappa * theta - 0.5 * se2) / nu - kappa   # drift at each z node
    wz = trapezoid_weights(z)
    A = sp.lil_matrix((nx * nz, nx * nz))
    idx = lambda i, j: i * nz + j
    for i in range(nx):
        for j in range(nz - 1):                       # interior face j+1/2
            h = z[j + 1] - z[j]
            mu_f = 0.5 * (mu_z[j] + mu_z[j + 1])
            cL = 0.5 * mu_f + Dz[j] / h
            cR = 0.5 * mu_f - Dz[j + 1] / h
            A[idx(i, j),     idx(i, j)]     += -cL / wz[j]
            A[idx(i, j),     idx(i, j + 1)] += -cR / wz[j]
            A[idx(i, j + 1), idx(i, j)]     += +cL / wz[j + 1]
            A[idx(i, j + 1), idx(i, j + 1)] += +cR / wz[j + 1]
    return A


def _mixed_operator(x, z, L, params: HestonParams, eta: float, b: float) -> sp.lil_matrix:
    """Mixed flux assigned to z: J_z^mixed = -rho*se*d_x(L f). Conservative in z (telescoping)."""
    nx, nz = x.size, z.size
    rho_se = params.rho * eta * params.sigma
    A = sp.lil_matrix((nx * nz, nx * nz))
    if abs(rho_se) < 1e-15:
        return A
    wz = trapezoid_weights(z)
    idx = lambda i, j: i * nz + j
    for j in range(nz - 1):                           # z-face j+1/2
        for i in range(1, nx - 1):                    # interior x for central d_x
            hx = x[i + 1] - x[i - 1]
            cm = -rho_se / hx                          # J^m = cm * (L_{i+1} f_{i+1} - L_{i-1} f_{i-1})
            for ii, sgn in ((i + 1, +1.0), (i - 1, -1.0)):
                coef = cm * sgn * L[ii] * 0.5          # 0.5 from z-face average (f_{·,j}+f_{·,j+1})/2
                A[idx(i, j),     idx(ii, j)]     += -coef / wz[j]
                A[idx(i, j),     idx(ii, j + 1)] += -coef / wz[j]
                A[idx(i, j + 1), idx(ii, j)]     += +coef / wz[j + 1]
                A[idx(i, j + 1), idx(ii, j + 1)] += +coef / wz[j + 1]
    return A


def build_directional_operators(x, z, L, params: HestonParams, eta: float, b: float):
    """Return (Ax, Az, Axz) sparse CSR pieces of the forward generator for the ADI split."""
    x, z, L = _check(x, z, L)
    Ax = _x_operator(x, z, L, params, eta, b).tocsr()
    Az = _z_operator(x, z, L, params, eta, b).tocsr()
    Axz = _mixed_operator(x, z, L, params, eta, b).tocsr()
    return Ax, Az, Axz


def build_forward_operator(x, z, L, params: HestonParams, eta: float, b: float):
    """Sparse CSR forward Fokker-Planck generator A on the (x,z) grid (SLV log-variance density).

    Layout: row-major index k = i*nz + j for (x_i, z_j). Mass-conserving by construction
    (quadrature-weighted constant is a left-null vector). Densify via ``.toarray()`` only for
    small-grid tests; the production grid is far too large to densify.
    """
    Ax, Az, Axz = build_directional_operators(x, z, L, params, eta, b)
    return (Ax + Az + Axz).tocsr()
