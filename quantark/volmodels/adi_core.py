"""Unified Heston / Heston-SLV ADI PDE core (Douglas / Craig-Sneyd).

2D finite-difference solver in (x = ln S, v). The Heston model is the special case of the
SLV model with leverage ``L ≡ 1`` and vol-of-vol scale ``η = 1``:

    A1 U = 0.5 L^2 v U_xx + ((r - carry) - 0.5 L^2 v) U_x - r U     (S-direction + reaction)
    A2 U = 0.5 (η σ)^2 v U_vv + κ(θ - v) U_v                        (v-direction)
    A0 U = ρ (η σ) L v U_xv                                         (mixed term)

``leverage=None`` selects the constant-``L`` (Heston) path, which enables coefficient
caching and the optional sparse-LU solve; a supplied ``LeverageSurface`` gives the SLV
path (S-coefficients rebuilt each step because ``L`` depends on t). Both paths support
``grid_spot`` pinning for clean bump Greeks. Dense batched-Thomas solves by default;
optional SuperLU for the constant-``L`` case. Deterministic — never invokes Monte Carlo.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.numerical import solve_tridiag_batch
from quantark.volmodels.heston.params import HestonParams


class HestonSLVADICore:
    """Shared ADI core for the Heston and Heston-SLV backward PDEs.

    Args:
        s0, strike, T, r, carry: option / market spec (carry = dividend or foreign rate).
        params: HestonParams (κ, θ, σ, ρ, v0).
        n_x, n_v, n_t: grid sizes (log-spot, variance, time).
        leverage: LeverageSurface, or None for constant L ≡ 1 (the Heston case).
        eta: vol-of-vol scale; ``sig_eff = eta * sigma``.
        use_sparse: use per-slice SuperLU in the S/V solves (only valid when L is constant).
        grid_spot: center the log-spot grid here (defaults to s0); pin it across spot
            bumps for clean delta/gamma.
    """

    def __init__(self, s0, strike, T, r, carry, params: HestonParams,
                 n_x, n_v, n_t, *, leverage=None, eta=1.0,
                 use_sparse=False, grid_spot=None):
        self.S0, self.K, self.T, self.r, self.q = s0, strike, T, r, carry
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = (
            params.kappa, params.theta, params.sigma, params.rho, params.v0,
        )
        self.lev = leverage
        self._constant_leverage = leverage is None
        self.sig_eff = float(eta) * self.sigma
        self.sig_eff2 = self.sig_eff ** 2
        # Sparse SuperLU is only valid when the S-operator is time-independent (L constant).
        self.use_sparse = bool(use_sparse) and self._constant_leverage
        self._opt_is_call = True
        self.N_S, self.N_V, self.N_T = n_x, n_v, n_t

        # ---- grid (identical extent logic to both original kernels) ----
        var_eff = max(self.theta, self.v0, 0.25 * self.sig_eff2, 0.04)
        x_width = 8.0 * np.sqrt(var_eff * max(T, 1e-12))
        grid_center = grid_spot if grid_spot is not None else s0
        x_center = float(np.log(max(grid_center, 1e-12)))
        self.x_min, self.x_max = x_center - x_width, x_center + x_width
        self.X_grid = np.linspace(self.x_min, self.x_max, n_x)
        self.S_grid = np.exp(self.X_grid)
        self.S_max = float(self.S_grid[-1])
        self.V_max = max(5.0 * self.theta, 0.5, 2.0 * self.v0)
        self.V_grid = np.linspace(0.0, self.V_max, n_v)
        self.dx = float(self.X_grid[1] - self.X_grid[0])
        self.dV = float(self.V_grid[1] - self.V_grid[0])
        self.dt = float(T / max(n_t, 1))
        self._S_int = self.S_grid[1:-1]
        self._ones_int = np.ones(self.N_S - 2)

        # caches: _tri_V always (time-independent); _tri_S only when L constant.
        self._S_tri_cache: dict = {}
        self._V_tri_cache: dict = {}
        self._S_lu_cache: dict = {}
        self._V_lu_cache: dict = {}

    def _L(self, t):
        if self._constant_leverage:
            return self._ones_int
        return np.asarray(self.lev.leverage(self._S_int, t), dtype=float)

    # ---- operators ----
    def _A1(self, U, t):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        v_int = self.V_grid[1:-1]
        L2v = (self._L(t) ** 2)[:, None] * v_int[None, :]
        U_xx = (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / (self.dx * self.dx)
        U_x = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2.0 * self.dx)
        # WS-C1: the -rU reaction is carried implicitly in the S-direction (folded into A1
        # here and into the _tri_S diagonal), so the predictor no longer applies it.
        out[1:-1, 1:-1] = (0.5 * L2v * U_xx
                           + ((self.r - self.q) - 0.5 * L2v) * U_x
                           - self.r * U[1:-1, 1:-1])
        return out

    def _A2(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        v_int = self.V_grid[1:-1]
        coef_d2 = 0.5 * self.sig_eff2 * v_int
        coef_d1 = self.kappa * (self.theta - v_int)
        U_VV = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (self.dV * self.dV)
        U_V = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * self.dV)
        out[1:-1, 1:-1] = coef_d2 * U_VV + coef_d1 * U_V
        return out

    def _A0(self, U, t):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3 or abs(self.rho) <= 1e-12:
            return out
        v_int = self.V_grid[1:-1]
        Lv = self._L(t)[:, None] * v_int[None, :]
        U_xv = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4.0 * self.dx * self.dV)
        out[1:-1, 1:-1] = self.rho * self.sig_eff * Lv * U_xv
        return out

    def _bc(self, U, tau):
        if self._opt_is_call:
            U[0, :] = 0.0
            U[-1, :] = max(0.0, self.S_max * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau))
        else:
            U[0, :] = self.K * np.exp(-self.r * tau)
            U[-1, :] = 0.0
        U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]

    def _terminal(self, is_call):
        S_mesh, _ = np.meshgrid(self.S_grid, self.V_grid, indexing="ij")
        self._opt_is_call = is_call
        U = np.maximum(S_mesh - self.K, 0.0) if is_call else np.maximum(self.K - S_mesh, 0.0)
        self._bc(U, 0.0)
        return U

    def _s_boundary_rhs(self, rhs, tau):
        if self._opt_is_call:
            rhs[0] = 0.0
            rhs[-1] = max(0.0, self.S_max * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau))
        else:
            rhs[0] = self.K * np.exp(-self.r * tau)
            rhs[-1] = 0.0
        return rhs

    # ---- tridiagonal builders (cached where the operator is time-independent) ----
    def _tri_S(self, dt_step, theta_loc, t_mid):
        if self._constant_leverage:
            key = (float(dt_step), float(theta_loc))
            cached = self._S_tri_cache.get(key)
            if cached is not None:
                return cached
        L2 = self._L(t_mid) ** 2                                # (N_S-2,) interior
        V = np.maximum(self.V_grid, 1e-10)[:, None]             # (N_V, 1)
        c2 = 0.5 * (L2[None, :] * V) / (self.dx * self.dx)      # (N_V, N_S-2)
        c1 = ((self.r - self.q) - 0.5 * (L2[None, :] * V)) / (2.0 * self.dx)
        a = np.zeros((self.N_V, self.N_S))
        b = np.ones((self.N_V, self.N_S))
        c = np.zeros((self.N_V, self.N_S))
        a[:, 1:-1] = -theta_loc * dt_step * (c2 - c1)
        # WS-C1: (I - theta*dt*A1) diagonal gains +theta*dt*r from the implicit -rU reaction.
        b[:, 1:-1] = 1.0 + theta_loc * dt_step * (2.0 * c2 + self.r)
        c[:, 1:-1] = -theta_loc * dt_step * (c2 + c1)
        if self._constant_leverage:
            self._S_tri_cache[key] = (a, b, c)
        return a, b, c

    def _tri_V(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        cached = self._V_tri_cache.get(key)
        if cached is not None:
            return cached
        N = self.N_V
        dV = self.dV
        v = np.maximum(self.V_grid, 1e-10)
        coef_d2 = 0.5 * self.sig_eff2 * v / (dV * dV)
        coef_d1 = self.kappa * (self.theta - v) / (2.0 * dV)
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        a[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] - coef_d1[1:-1])
        b[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * coef_d2[1:-1])
        c[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] + coef_d1[1:-1])
        b[0] = 1.0; c[0] = -1.0; a[-1] = -1.0; b[-1] = 1.0
        self._V_tri_cache[key] = (a, b, c)
        return a, b, c

    def _ensure_S_lus(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        if key in self._S_lu_cache:
            return self._S_lu_cache[key]
        alpha, beta, gamma = self._tri_S(dt_step, theta_loc, 0.0)   # constant L -> t_mid inert
        lus = []
        for j in range(self.N_V):
            A = sp.diags([alpha[j, 1:], beta[j], gamma[j, :-1]], offsets=[-1, 0, 1], format="csc")
            lus.append(spla.splu(A))
        self._S_lu_cache[key] = lus
        return lus

    def _ensure_V_lu(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        if key in self._V_lu_cache:
            return self._V_lu_cache[key]
        alpha, beta, gamma = self._tri_V(dt_step, theta_loc)
        A = sp.diags([alpha[1:], beta, gamma[:-1]], offsets=[-1, 0, 1], format="csc")
        lu = spla.splu(A)
        self._V_lu_cache[key] = lu
        return lu

    # ---- direction solves ----
    def _solve_S(self, source, A1U, dt_step, theta_loc, tau, t_mid):
        if self.use_sparse:
            Y = np.empty_like(source)
            lus = self._ensure_S_lus(dt_step, theta_loc)
            for j in range(self.N_V):
                rhs = self._s_boundary_rhs(source[:, j] - theta_loc * dt_step * A1U[:, j], tau)
                Y[:, j] = lus[j].solve(rhs)
            self._bc(Y, tau)
            return Y
        a, b, c = self._tri_S(dt_step, theta_loc, t_mid)
        rhs = source - theta_loc * dt_step * A1U                    # (N_S, N_V), fresh array
        if self._opt_is_call:
            rhs[0, :] = 0.0
            rhs[-1, :] = max(0.0, self.S_max * np.exp(-self.q * tau)
                             - self.K * np.exp(-self.r * tau))
        else:
            rhs[0, :] = self.K * np.exp(-self.r * tau)
            rhs[-1, :] = 0.0
        Y = solve_tridiag_batch(a, b, c, rhs.T).T                   # one system per V-slice
        self._bc(Y, tau)
        return Y

    def _solve_V(self, source, A2U, dt_step, theta_loc, tau):
        if self.use_sparse:
            U_out = np.empty_like(source)
            lu = self._ensure_V_lu(dt_step, theta_loc)
            for i in range(self.N_S):
                rhs = source[i, :] - theta_loc * dt_step * A2U[i, :]
                rhs[0] = 0.0; rhs[-1] = 0.0
                U_out[i, :] = lu.solve(rhs)
            self._bc(U_out, tau)
            return U_out
        a, b, c = self._tri_V(dt_step, theta_loc)
        rhs = source - theta_loc * dt_step * A2U                    # (N_S, N_V), fresh array
        rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0
        U_out = solve_tridiag_batch(np.broadcast_to(a, (self.N_S, self.N_V)),
                                    np.broadcast_to(b, (self.N_S, self.N_V)),
                                    np.broadcast_to(c, (self.N_S, self.N_V)),
                                    rhs)                             # one system per S-row
        self._bc(U_out, tau)
        return U_out

    # ---- ADI steps ----
    def _douglas_step(self, U, dt_step, tau, theta_loc, t_mid):
        A1U, A2U = self._A1(U, t_mid), self._A2(U)
        A0U = self._A0(U, t_mid)
        # WS-C1: -rU is inside A1U now, so the predictor drops the explicit - r*U term.
        Y0 = U + dt_step * (A1U + A2U + A0U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau, t_mid)
        return self._solve_V(Y1, A2U, dt_step, theta_loc, tau)

    def _cs_step(self, U, dt_step, tau, theta_loc, t_mid):
        A1U, A2U = self._A1(U, t_mid), self._A2(U)
        A0U = self._A0(U, t_mid)
        Y0 = U + dt_step * (A1U + A2U + A0U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau, t_mid)
        Y2 = self._solve_V(Y1, A2U, dt_step, theta_loc, tau)
        # Craig-Sneyd corrector: restart from the explicit predictor Y0 (NOT Y2 — restarting
        # from the already-swept Y2 re-applies the implicit operators and degrades CS to a
        # first-order double-Douglas). With Y0, CS is genuinely second-order.
        if abs(self.rho) > 1e-12:
            Ycorr = Y0 + 0.5 * dt_step * (self._A0(Y2, t_mid) - A0U)
            self._bc(Ycorr, tau)
        else:
            Ycorr = Y0
        Z1 = self._solve_S(Ycorr, A1U, dt_step, theta_loc, tau, t_mid)
        return self._solve_V(Z1, A2U, dt_step, theta_loc, tau)

    def solve(self, is_call, scheme, theta, rannacher):
        U = self._terminal(is_call)
        tau = 0.0
        if rannacher and self.N_T >= 1:
            dt_half = 0.5 * self.dt
            for _ in range(2):
                tau += dt_half
                U = self._douglas_step(U, dt_half, tau, 1.0, self.T - tau + 0.5 * dt_half)
            steps_remaining = self.N_T - 1
        else:
            steps_remaining = self.N_T
        for _ in range(steps_remaining):
            tau += self.dt
            t_mid = self.T - tau + 0.5 * self.dt
            if scheme == ADIScheme.DOUGLAS:
                U = self._douglas_step(U, self.dt, tau, theta, t_mid)
            else:  # CRAIG_SNEYD (MCS is rejected by the wrappers)
                U = self._cs_step(U, self.dt, tau, theta, t_mid)
        return U

    # ---- read-off ----
    def interpolate(self, U, x_val, v_val):
        X, V = self.X_grid, self.V_grid
        x_val = max(min(x_val, X[-1]), X[0])
        v_val = max(min(v_val, V[-1]), V[0])
        i0 = max(min(int(np.searchsorted(X, x_val, side="right") - 1), len(X) - 2), 0)
        j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
        i1, j1 = i0 + 1, j0 + 1
        X0, X1, V0, V1 = X[i0], X[i1], V[j0], V[j1]
        if X1 == X0 or V1 == V0:
            return float(U[i0, j0])
        t = (x_val - X0) / (X1 - X0)
        u = (v_val - V0) / (V1 - V0)
        return float(
            (1 - t) * (1 - u) * U[i0, j0] + t * (1 - u) * U[i1, j0]
            + (1 - t) * u * U[i0, j1] + t * u * U[i1, j1]
        )

    def price_delta_gamma(self, U, s0):
        """Price and SPOT delta/gamma from the solved surface.

        Gamma comes from spatial derivatives of the PDE solution on the S-grid (not from
        re-bumping bilinearly-interpolated prices, which has meaningless curvature).
        """
        V = self.V_grid
        v_val = min(max(self.v0, V[0]), V[-1])
        j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
        j1 = j0 + 1
        w = 0.0 if V[j1] == V[j0] else (v_val - V[j0]) / (V[j1] - V[j0])
        v_curve = (1.0 - w) * U[:, j0] + w * U[:, j1]   # price vs S at v0
        S = self.S_grid
        price = float(np.interp(s0, S, v_curve))
        dVdS = np.gradient(v_curve, S, edge_order=2)
        d2VdS2 = np.gradient(dVdS, S, edge_order=2)
        delta = float(np.interp(s0, S, dVdS))
        gamma = float(np.interp(s0, S, d2VdS2))
        return price, delta, gamma
