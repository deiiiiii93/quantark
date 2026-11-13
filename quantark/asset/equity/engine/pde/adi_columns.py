"""Batched claim sweeps with the HestonSLVADICore's native ADI coefficients.

Shape is (spot, variance, claim). Sparse and characteristic-transport modes
retain the core's scalar sweeps. The claim axis has no model coupling.
"""
import numpy as np
from quantark.volmodels.adi_core import solve_tridiag_batch

#: |rho| at or below which the batched step has no mixed-derivative term (no ``A0``, no Craig-Sneyd correction);
#: the forward adjoint's step transpose (``vol_model_adjoint.StepTranspose``) switches it off at the same value
RHO_OFF = 1e-12


def _bound_spot_caches(core):
    # A rolling claim sweep never revisits old forward intervals. Retaining
    # one matrix/factor pack per historical node can consume gigabytes on
    # Greek grids. Two keys cover the current step and terminal half steps.
    for name in ('_S_tri_cache', '_S_lu_cache', '_exact_discount_S_lu_cache'):
        cache = getattr(core, name, None)
        if cache is not None:
            while len(cache) > 2:
                cache.pop(next(iter(cache)))


def _characteristic_variance_transport(core, U, dt):
    """Characteristic interpolation with fixed weights, independent of the claim payoff.

    Clipping a cubic payoff interpolant is nonlinear: the price of added
    outcome cash then differs from its indicator claim. Native autocallables
    use the same fixed weights for prices, memory and statistics: the core's
    four-point Lagrange stencil (linear in its two edge brackets), unclipped.

    Fixed weights keep the transport linear in the claim, and they sum to one,
    so a sure unit is transported exactly. By Godunov's theorem a linear scheme
    cannot be both monotone and better than first order; this one keeps third
    order and gives up monotonicity. Fixed positive linear weights (79938a1c)
    were first order, and the error of a semi-Lagrangian step grows like
    dv**2/dt: in the convection-dominated sigma_collapse regime that moved the
    certified delta 11 reference standard errors.
    """
    if dt <= 0.:
        return U
    stencil = core._advection_stencil(dt)
    return np.einsum("ijk,jk->ij", U[:, stencil.indices], stencil.weights, optimize=True)


def step_columns(core, U, dt, tau, theta, t_mid, low, high, douglas, force_scalar=False):
    # Deterministic -r I commutes with the spatial generator. Integrate it
    # exponentially, rather than give a sure unit the ADI rational discount.
    r, q = core._rq_at(t_mid)
    discount = np.exp(-r*dt)
    original_low, original_high = low, high
    low, high = low/discount, high/discount
    if force_scalar or core.use_sparse or core._semi_lagrangian_v:
        previous = core._boundary_hook
        original_a1, original_tri = core._A1, core._tri_S
        original_transport = core._advect_v
        if core._semi_lagrangian_v:
            core._advect_v = lambda A, h: _characteristic_variance_transport(core, A, h)
        def a1_without_reaction(A, t):
            action = original_a1(A, t)
            action[1:-1, 1:-1] += r*A[1:-1, 1:-1]
            if core._degenerate_v0:
                action[1:-1, 0] += r*A[1:-1, 0]
            return action
        def tri_without_reaction(h, weight, t):
            a, b, c = original_tri(h, weight, t)
            b = b.copy()
            b[:, 1:-1] -= weight*h*r
            return a, b, c
        core._A1, core._tri_S = a1_without_reaction, tri_without_reaction
        # Sparse factors depend on the reaction convention too. Keep these
        # private caches scoped to the transformed step, never mix factors.
        old_lus = core._S_lu_cache
        if not hasattr(core, '_exact_discount_S_lu_cache'):
            core._exact_discount_S_lu_cache = {}
        core._S_lu_cache = core._exact_discount_S_lu_cache
        try:
            for c in range(U.shape[2]):
                def boundary(A, _tau, c=c):
                    A[0, :], A[-1, :] = low[c], high[c]
                    return A
                core._boundary_hook = boundary
                step = core._douglas_step if douglas else core._cs_step
                U[:, :, c] = step(U[:, :, c], dt, tau, theta, t_mid)
            U *= discount
            U[0], U[-1] = original_low, original_high
            return U
        finally:
            core._boundary_hook = previous
            core._A1, core._tri_S = original_a1, original_tri
            core._advect_v = original_transport
            core._S_lu_cache = old_lus
            _bound_spot_caches(core)
    n_s, n_v, n_c = U.shape
    lev = core._L(t_mid)
    diffusion = .5 * lev[:, None, None]**2 * core.V_grid[None, 1:-1, None]
    convection = (r-q)-diffusion
    a_s, b_s, c_s = core._tri_S(dt, theta, t_mid)
    b_s = b_s.copy()
    b_s[:, 1:-1] -= theta*dt*r
    a_v, b_v, c_v = core._tri_V(dt, theta)
    sub, diag, sup, *_ = core._build_v_generator_coefficients()

    def bc(A):
        if not core._degenerate_v0:
            A[:, 0, :] = A[:, 1, :]
        A[:, -1, :] = A[:, -2, :]
        A[0, :, :], A[-1, :, :] = low, high
        return A

    def A1(A):
        out = np.zeros_like(A)
        if core._uniform:
            xx = (A[2:, 1:-1]-2*A[1:-1, 1:-1]+A[:-2, 1:-1]) / core.dx**2
            x = (A[2:, 1:-1]-A[:-2, 1:-1]) / (2*core.dx)
            x0 = (A[2:, 0]-A[:-2, 0]) / (2*core.dx)
        else:
            wm2, w02, wp2 = core._xx
            wm1, w01, wp1 = core._x1
            xx = (A[:-2, 1:-1]*wm2[:, None, None] + A[1:-1, 1:-1]*w02[:, None, None]
                  + A[2:, 1:-1]*wp2[:, None, None])
            x = (A[:-2, 1:-1]*wm1[:, None, None] + A[1:-1, 1:-1]*w01[:, None, None]
                 + A[2:, 1:-1]*wp1[:, None, None])
            x0 = A[:-2, 0]*wm1[:, None] + A[1:-1, 0]*w01[:, None] + A[2:, 0]*wp1[:, None]
        out[1:-1, 1:-1] = diffusion*xx + convection*x
        if core._degenerate_v0:
            out[1:-1, 0] = (r-q)*x0
        return out

    def A2(A):
        out = np.zeros_like(A)
        out[1:-1, 1:-1] = (A[1:-1, :-2]*sub[None, :, None] + A[1:-1, 1:-1]*diag[None, :, None]
                            + A[1:-1, 2:]*sup[None, :, None])
        if core._degenerate_v0:
            out[1:-1, 0] = core.kappa*core.theta*(A[1:-1, 1]-A[1:-1, 0])/(core.V_grid[1]-core.V_grid[0])
        return out

    def A0(A):
        out = np.zeros_like(A)
        if abs(core.rho) <= RHO_OFF:
            return out
        if core._uniform:
            cross = (A[2:, 2:]-A[2:, :-2]-A[:-2, 2:]+A[:-2, :-2])/(4*core.dx*core.dV)
        else:
            xm, x0, xp = core._x1
            vm, v0, vp = core._v1
            x = A[:-2]*xm[:, None, None]+A[1:-1]*x0[:, None, None]+A[2:]*xp[:, None, None]
            cross = x[:, :-2]*vm[None, :, None]+x[:, 1:-1]*v0[None, :, None]+x[:, 2:]*vp[None, :, None]
        out[1:-1, 1:-1] = core.rho*core.sig_eff*lev[:, None, None]*core.V_grid[None, 1:-1, None]*cross
        return out

    # Batch every claim using the core's own tridiagonal solver and matrices.
    shape_s = (n_c, n_v, n_s)
    s_matrices = [np.broadcast_to(a, shape_s).reshape(-1, n_s) for a in (a_s, b_s, c_s)]
    shape_v = (n_c, n_s, n_v)
    v_matrices = [np.broadcast_to(a, shape_v).reshape(-1, n_v) for a in (a_v, b_v, c_v)]

    def solve_s(source, action):
        rhs = source-theta*dt*action
        rhs[0], rhs[-1] = low, high
        out = solve_tridiag_batch(*s_matrices, rhs.transpose(2, 1, 0).reshape(-1, n_s))
        return bc(out.reshape(shape_s).transpose(2, 1, 0))

    def solve_v(source, action):
        rhs = source-theta*dt*action
        if not core._degenerate_v0:
            rhs[:, 0] = 0.
        rhs[:, -1] = 0.
        out = solve_tridiag_batch(*v_matrices, rhs.transpose(2, 0, 1).reshape(-1, n_v))
        return bc(out.reshape(shape_v).transpose(1, 2, 0))

    a1, a2, a0 = A1(U), A2(U), A0(U)
    predictor = U+dt*(a1+a2+a0)
    first = solve_v(solve_s(predictor, a1), a2)
    if douglas:
        out = first
    else:
        corrected = bc(predictor+.5*dt*(A0(first)-a0)) if abs(core.rho)>RHO_OFF else predictor
        out = solve_v(solve_s(corrected, a1), a2)
    out *= discount
    out[0], out[-1] = original_low, original_high
    _bound_spot_caches(core)
    return out
