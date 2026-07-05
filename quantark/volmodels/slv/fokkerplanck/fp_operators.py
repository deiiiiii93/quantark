"""Forward Fokker-Planck operators for the SLV log-variance density f(x,z)=nu*p.

Conservative (flux-divergence) finite-volume discretization of the paper section 4.5.3
equation. Each direction is assembled from face fluxes so that the quadrature-weighted
constant vector is a left-null vector of the operator (mass conservation), and the
zero-flux boundary condition is enforced simply by carrying zero flux through the
domain-boundary faces. The three directional pieces (x, z, mixed) sum to the full
generator and are exposed separately for the Craig-Sneyd ADI split.

Assembly is vectorized over grid faces (COO triplets) so the per-step operator rebuild
inside the leverage bootstrap stays cheap.
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


def _coo(rows, cols, vals, n):
    return sp.coo_matrix((np.asarray(vals).ravel(),
                          (np.asarray(rows).ravel(), np.asarray(cols).ravel())),
                         shape=(n, n)).tocsr()


def _x_operator(x, z, L, params: HestonParams, eta: float, b: float):
    """x-direction flux: J_x = (b - 0.5 nu L^2) f - d_x(0.5 nu L^2 f). Conservative central FV."""
    nx, nz = x.size, z.size
    n = nx * nz
    nu = np.exp(z)
    wx = trapezoid_weights(x)
    D = 0.5 * (L[:, None] ** 2) * nu[None, :]                  # (nx, nz)
    mu = b - D                                                  # (nx, nz)
    h = (x[1:] - x[:-1])[:, None]                               # (nx-1, 1)
    mu_f = 0.5 * (mu[:-1, :] + mu[1:, :])                       # (nx-1, nz)
    cL = 0.5 * mu_f + D[:-1, :] / h                             # coeff on f_i  at face i+1/2
    cR = 0.5 * mu_f - D[1:, :] / h                              # coeff on f_{i+1}
    iL = np.arange(nx - 1)[:, None]; iR = iL + 1
    jj = np.arange(nz)[None, :]
    kL = iL * nz + jj; kR = iR * nz + jj                        # node indices
    wxL = wx[:-1][:, None]; wxR = wx[1:][:, None]
    rows = [kL, kL, kR, kR]
    cols = [kL, kR, kL, kR]
    vals = [-cL / wxL, -cR / wxL, cL / wxR, cR / wxR]
    return _coo(np.concatenate([r.ravel() for r in rows]),
                np.concatenate([c.ravel() for c in cols]),
                np.concatenate([v.ravel() for v in vals]), n)


def _z_operator(x, z, L, params: HestonParams, eta: float, b: float):
    """z-direction flux: J_z = ((k th - se^2/2)/nu - k) f - d_z(0.5 se^2 / nu f). Conservative FV."""
    nx, nz = x.size, z.size
    n = nx * nz
    nu = np.exp(z)
    se2 = (eta * params.sigma) ** 2
    kappa, theta = params.kappa, params.theta
    Dz = 0.5 * se2 / nu                                         # (nz,)
    mu_z = (kappa * theta - 0.5 * se2) / nu - kappa             # (nz,)
    wz = trapezoid_weights(z)
    h = z[1:] - z[:-1]                                          # (nz-1,)
    mu_f = 0.5 * (mu_z[:-1] + mu_z[1:])                         # (nz-1,)
    cL = 0.5 * mu_f + Dz[:-1] / h                               # (nz-1,)  coeff on f_j at face j+1/2
    cR = 0.5 * mu_f - Dz[1:] / h
    ii = np.arange(nx)[:, None]
    jL = np.arange(nz - 1)[None, :]; jR = jL + 1
    kL = ii * nz + jL; kR = ii * nz + jR
    wzL = wz[:-1][None, :]; wzR = wz[1:][None, :]
    cLb = cL[None, :]; cRb = cR[None, :]
    rows = [kL, kL, kR, kR]
    cols = [kL, kR, kL, kR]
    vals = [-cLb / wzL, -cRb / wzL, cLb / wzR, cRb / wzR]
    return _coo(np.concatenate([r.ravel() for r in rows]),
                np.concatenate([c.ravel() for c in cols]),
                np.concatenate([np.broadcast_to(v, kL.shape).ravel() for v in vals]), n)


def _mixed_operator(x, z, L, params: HestonParams, eta: float, b: float):
    """Mixed flux assigned to z: J_z^mixed = -rho*se*d_x(L f). Conservative in z (telescoping)."""
    nx, nz = x.size, z.size
    n = nx * nz
    rho_se = params.rho * eta * params.sigma
    if abs(rho_se) < 1e-15:
        return sp.csr_matrix((n, n))
    wz = trapezoid_weights(z)
    # interior x nodes 1..nx-2; central d_x over hx = x[i+1]-x[i-1]
    i = np.arange(1, nx - 1)[:, None]                           # (nxm2, 1)
    j = np.arange(nz - 1)[None, :]                              # (1, nz-1)  z-face j+1/2
    hx = (x[2:] - x[:-2])[:, None]                              # (nxm2, 1)
    cm = -rho_se / hx                                           # (nxm2, 1)
    wzL = wz[:-1][None, :]; wzR = wz[1:][None, :]               # (1, nz-1)
    k_lo_j = i * nz + j; k_lo_jp = i * nz + (j + 1)
    rows_list, cols_list, vals_list = [], [], []
    for ii_off, sgn in ((1, +1.0), (-1, -1.0)):
        ineigh = i + ii_off
        Lneigh = L[ineigh.ravel()].reshape(i.shape)             # (nxm2,1)
        coef = cm * sgn * Lneigh * 0.5                          # (nxm2,1) broadcast over j
        k_nb_j = ineigh * nz + j
        k_nb_jp = ineigh * nz + (j + 1)
        # cell (i,j):   -J/wz_j  ; cell (i,j+1): +J/wz_{j+1}
        for k_target, w_t, s in ((k_lo_j, wzL, -1.0), (k_lo_jp, wzR, +1.0)):
            for k_src in (k_nb_j, k_nb_jp):
                rows_list.append(np.broadcast_to(k_target, (i.shape[0], nz - 1)).ravel())
                cols_list.append(np.broadcast_to(k_src, (i.shape[0], nz - 1)).ravel())
                vals_list.append(np.broadcast_to(s * coef / w_t, (i.shape[0], nz - 1)).ravel())
    return _coo(np.concatenate(rows_list), np.concatenate(cols_list),
                np.concatenate(vals_list), n)


def build_directional_operators(x, z, L, params: HestonParams, eta: float, b: float):
    """Return (Ax, Az, Axz) sparse CSR pieces of the forward generator for the ADI split."""
    x, z, L = _check(x, z, L)
    return (_x_operator(x, z, L, params, eta, b),
            _z_operator(x, z, L, params, eta, b),
            _mixed_operator(x, z, L, params, eta, b))


def build_forward_operator(x, z, L, params: HestonParams, eta: float, b: float):
    """Sparse CSR forward Fokker-Planck generator A on the (x,z) grid (SLV log-variance density).

    Layout: row-major index k = i*nz + j for (x_i, z_j). Mass-conserving by construction
    (quadrature-weighted constant is a left-null vector). Densify via ``.toarray()`` only for
    small-grid tests; the production grid is far too large to densify.
    """
    Ax, Az, Axz = build_directional_operators(x, z, L, params, eta, b)
    return (Ax + Az + Axz).tocsr()
