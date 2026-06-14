"""Backward SLV ADI PDE pricer for European vanillas (deterministic; no MC).

Heston ADI in (x=ln S, v) with a calibrated leverage L(S, t) entering the x-operators:
    A1 U = 0.5 L^2 v U_xx + ((r - carry) - 0.5 L^2 v) U_x
    A2 U = 0.5 (eta sigma)^2 v U_vv + kappa(theta - v) U_v
    A0 U = rho (eta sigma) L v U_xv
The leverage is supplied as a precomputed LeverageSurface (e.g. MC-binning calibrated);
this engine never invokes Monte Carlo. Douglas / Craig-Sneyd schemes with Rannacher
start-up; dense Thomas solves (S-tridiagonals rebuilt each step since L depends on t).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface


class _HestonSLVADI:
    def __init__(self, s0, strike, T, r, carry, params: HestonParams, lev: LeverageSurface,
                 eta, n_x, n_v, n_t):
        self.S0, self.K, self.T, self.r, self.q = s0, strike, T, r, carry
        self.kappa, self.theta, self.sigma, self.rho, self.v0 = (
            params.kappa, params.theta, params.sigma, params.rho, params.v0,
        )
        self.lev = lev
        self.sig_eff = float(eta) * self.sigma
        self.sig_eff2 = self.sig_eff ** 2
        self._opt_is_call = True
        self.N_S, self.N_V, self.N_T = n_x, n_v, n_t
        var_eff = max(self.theta, self.v0, 0.25 * self.sig_eff2, 0.04)
        x_width = 8.0 * np.sqrt(var_eff * max(T, 1e-12))
        x_center = float(np.log(max(s0, 1e-12)))
        self.X_grid = np.linspace(x_center - x_width, x_center + x_width, n_x)
        self.S_grid = np.exp(self.X_grid)
        self.S_max = float(self.S_grid[-1])
        self.V_max = max(5.0 * self.theta, 0.5, 2.0 * self.v0)
        self.V_grid = np.linspace(0.0, self.V_max, n_v)
        self.dx = float(self.X_grid[1] - self.X_grid[0])
        self.dV = float(self.V_grid[1] - self.V_grid[0])
        self.dt = T / max(n_t, 1)
        self._S_int = self.S_grid[1:-1]

    def _L(self, t):
        return np.asarray(self.lev.leverage(self._S_int, t), dtype=float)

    @staticmethod
    def _thomas(a, b, c, d):
        n = len(d)
        cp = np.zeros(n); dp = np.zeros(n); x = np.zeros(n)
        cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-14:
                denom = 1e-14
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        x[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def _A1(self, U, t):
        out = np.zeros_like(U)
        v_int = self.V_grid[1:-1]
        L2v = (self._L(t) ** 2)[:, None] * v_int[None, :]
        U_xx = (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / (self.dx * self.dx)
        U_x = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2.0 * self.dx)
        out[1:-1, 1:-1] = 0.5 * L2v * U_xx + ((self.r - self.q) - 0.5 * L2v) * U_x
        return out

    def _A2(self, U):
        out = np.zeros_like(U)
        v_int = self.V_grid[1:-1]
        coef_d2 = 0.5 * self.sig_eff2 * v_int
        coef_d1 = self.kappa * (self.theta - v_int)
        U_VV = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (self.dV * self.dV)
        U_V = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * self.dV)
        out[1:-1, 1:-1] = coef_d2 * U_VV + coef_d1 * U_V
        return out

    def _A0(self, U, t):
        out = np.zeros_like(U)
        if abs(self.rho) <= 1e-12:
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

    def _s_rhs_bc(self, rhs, tau):
        if self._opt_is_call:
            rhs[0] = 0.0
            rhs[-1] = max(0.0, self.S_max * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau))
        else:
            rhs[0] = self.K * np.exp(-self.r * tau)
            rhs[-1] = 0.0
        return rhs

    def _solve_S(self, source, A1U, dt_step, theta_loc, tau, t_mid):
        N = self.N_S
        L2 = self._L(t_mid) ** 2
        Y = np.empty_like(source)
        for j in range(self.N_V):
            vj = max(float(self.V_grid[j]), 1e-10)
            c2 = 0.5 * (L2 * vj) / (self.dx * self.dx)
            c1 = ((self.r - self.q) - 0.5 * (L2 * vj)) / (2.0 * self.dx)
            a = np.zeros(N); b = np.ones(N); c = np.zeros(N)
            a[1:-1] = -theta_loc * dt_step * (c2 - c1)
            b[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * c2)
            c[1:-1] = -theta_loc * dt_step * (c2 + c1)
            rhs = self._s_rhs_bc(source[:, j] - theta_loc * dt_step * A1U[:, j], tau)
            Y[:, j] = self._thomas(a, b, c, rhs)
        self._bc(Y, tau)
        return Y

    def _solve_V(self, source, A2U, dt_step, theta_loc, tau):
        N = self.N_V
        v = np.maximum(self.V_grid, 1e-10)
        coef_d2 = 0.5 * self.sig_eff2 * v / (self.dV * self.dV)
        coef_d1 = self.kappa * (self.theta - v) / (2.0 * self.dV)
        a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
        a[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] - coef_d1[1:-1])
        b[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * coef_d2[1:-1])
        c[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] + coef_d1[1:-1])
        b[0] = 1.0; c[0] = -1.0; a[-1] = -1.0; b[-1] = 1.0
        U_out = np.empty_like(source)
        for i in range(self.N_S):
            rhs = source[i, :] - theta_loc * dt_step * A2U[i, :]
            rhs[0] = 0.0; rhs[-1] = 0.0
            U_out[i, :] = self._thomas(a, b, c, rhs)
        self._bc(U_out, tau)
        return U_out

    def _douglas_step(self, U, dt_step, tau, theta_loc, t_mid):
        A1U, A2U = self._A1(U, t_mid), self._A2(U)
        A0U = self._A0(U, t_mid)
        Y0 = U + dt_step * (A1U + A2U + A0U - self.r * U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau, t_mid)
        return self._solve_V(Y1, A2U, dt_step, theta_loc, tau)

    def _cs_step(self, U, dt_step, tau, theta_loc, t_mid):
        A1U, A2U = self._A1(U, t_mid), self._A2(U)
        A0U = self._A0(U, t_mid)
        Y0 = U + dt_step * (A1U + A2U + A0U - self.r * U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau, t_mid)
        Y2 = self._solve_V(Y1, A2U, dt_step, theta_loc, tau)
        if abs(self.rho) > 1e-12:
            Ycorr = Y2 + 0.5 * dt_step * (self._A0(Y2, t_mid) - A0U)
            self._bc(Ycorr, tau)
        else:
            Ycorr = Y2
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
            else:
                U = self._cs_step(U, self.dt, tau, theta, t_mid)
        return U

    def price_delta_gamma(self, U, s0):
        V = self.V_grid
        v_val = min(max(self.v0, V[0]), V[-1])
        j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
        j1 = j0 + 1
        w = 0.0 if V[j1] == V[j0] else (v_val - V[j0]) / (V[j1] - V[j0])
        v_curve = (1.0 - w) * U[:, j0] + w * U[:, j1]
        S = self.S_grid
        price = float(np.interp(s0, S, v_curve))
        dVdS = np.gradient(v_curve, S, edge_order=2)
        gamma = float(np.interp(s0, S, np.gradient(dVdS, S, edge_order=2)))
        return price, float(np.interp(s0, S, dVdS)), gamma

    def interpolate(self, U, x_val, v_val):
        X, V = self.X_grid, self.V_grid
        x_val = max(min(x_val, X[-1]), X[0]); v_val = max(min(v_val, V[-1]), V[0])
        i0 = max(min(int(np.searchsorted(X, x_val, side="right") - 1), len(X) - 2), 0)
        j0 = max(min(int(np.searchsorted(V, v_val, side="right") - 1), len(V) - 2), 0)
        i1, j1 = i0 + 1, j0 + 1
        X0, X1, V0, V1 = X[i0], X[i1], V[j0], V[j1]
        if X1 == X0 or V1 == V0:
            return float(U[i0, j0])
        tx = (x_val - X0) / (X1 - X0); uv = (v_val - V0) / (V1 - V0)
        return float((1 - tx) * (1 - uv) * U[i0, j0] + tx * (1 - uv) * U[i1, j0]
                     + (1 - tx) * uv * U[i0, j1] + tx * uv * U[i1, j1])


def price_european_slv_pde(
    s0: float, strike: float, is_call: bool, T: float, params: HestonParams,
    lev_surface: LeverageSurface, r: float, carry: float, eta: float = 1.0,
    n_x: int = 200, n_v: int = 100, n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD, theta: float = 0.5, rannacher: bool = True,
) -> float:
    """Price a European vanilla under Heston SLV via backward ADI (given a LeverageSurface)."""
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if n_x < 3 or n_v < 3 or n_t < 1:
        raise ValidationError("require n_x>=3, n_v>=3, n_t>=1")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if not isinstance(scheme, ADIScheme):
        raise ValidationError("scheme must be an ADIScheme")
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the SLV PDE; use CRAIG_SNEYD or DOUGLAS")
    if eta < 0:
        raise ValidationError("eta must be non-negative")
    solver = _HestonSLVADI(s0, strike, T, r, carry, params, lev_surface, eta, n_x, n_v, n_t)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price = solver.interpolate(U, float(np.log(s0)), params.v0)
    if not np.isfinite(price):
        raise NumericalError("SLV PDE produced a non-finite price")
    return price


def price_delta_gamma_slv_pde(
    s0: float, strike: float, is_call: bool, T: float, params: HestonParams,
    lev_surface: LeverageSurface, r: float, carry: float, eta: float = 1.0,
    n_x: int = 200, n_v: int = 100, n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD, theta: float = 0.5, rannacher: bool = True,
) -> Tuple[float, float, float]:
    """(price, spot-delta, spot-gamma) from a single backward SLV PDE solve."""
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if n_x < 3 or n_v < 3 or n_t < 1:
        raise ValidationError("require n_x>=3, n_v>=3, n_t>=1")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if not isinstance(scheme, ADIScheme):
        raise ValidationError("scheme must be an ADIScheme")
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the SLV PDE; use CRAIG_SNEYD or DOUGLAS")
    if eta < 0:
        raise ValidationError("eta must be non-negative")
    solver = _HestonSLVADI(s0, strike, T, r, carry, params, lev_surface, eta, n_x, n_v, n_t)
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price, delta, gamma = solver.price_delta_gamma(U, s0)
    if not (np.isfinite(price) and np.isfinite(delta) and np.isfinite(gamma)):
        raise NumericalError("SLV PDE produced non-finite price/greeks")
    return price, delta, gamma
