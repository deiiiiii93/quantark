"""Heston ADI PDE pricer for European vanillas (Douglas / Craig-Sneyd).

2D finite-difference solver in (x=ln S, v) with implicit treatment in each direction
and an explicit cross-derivative term, ported from the SLV reference HestonPDE. Dense
Thomas solves by default; optional sparse SuperLU. Rannacher start-up (two backward-Euler
half-steps) damps the payoff kink. Asset-neutral: carry = dividend yield (equity) or
foreign rate (FX).

For a European option the Heston price depends only on the terminal forward and discount
factor, so constant curve-consistent (r, carry) is curve-exact (unlike state-dependent
local vol); the engine supplies the zero rate / yield to maturity.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.volmodels.heston.params import HestonParams

# Below this vol-of-vol the variance is effectively deterministic and the ADI grid is
# ill-conditioned; the exact deterministic-variance (BS) limit is used instead.
_SIGMA_MIN = 1e-4


class _HestonADI:
    def __init__(self, s0, strike, T, r, carry, params: HestonParams,
                 n_x: int, n_v: int, n_t: int, use_sparse: bool, grid_spot=None):
        self.S0, self.K, self.T, self.r, self.q = s0, strike, T, r, carry
        # The spatial (log-spot) grid is centered on grid_spot; the price is read at S0.
        # Keeping grid_spot fixed across spot bumps yields clean delta/gamma (the operators,
        # payoff and boundaries are independent of the interpolation spot).
        self._grid_spot = float(grid_spot) if grid_spot is not None else float(s0)
        self.v0, self.kappa, self.theta = params.v0, params.kappa, params.theta
        self.sigma, self.rho = params.sigma, params.rho
        self.use_sparse = use_sparse
        self._opt_is_call = True
        self._S_lu_cache: Dict[Tuple[float, float], List] = {}
        self._V_lu_cache: Dict[Tuple[float, float], object] = {}
        self._setup_grid(n_x, n_v, n_t)

    def _setup_grid(self, N_S, N_V, N_T):
        self.N_S, self.N_V, self.N_T = N_S, N_V, N_T
        # Floor the variance used for DOMAIN sizing so the log-spot grid stays wide
        # enough to resolve the payoff/boundaries even for low-variance parameters
        # (this affects only the domain extent, not the dynamics).
        var_eff = max(self.theta, self.v0, 0.25 * (self.sigma ** 2), 0.04)
        x_width = 8.0 * np.sqrt(var_eff * max(self.T, 1e-12))
        x_center = float(np.log(max(self._grid_spot, 1e-12)))
        self.x_min, self.x_max = x_center - x_width, x_center + x_width
        # Variance domain must cover v0 and the long-run level.
        self.V_max = max(5.0 * self.theta, 0.5, 2.0 * self.v0)
        self.X_grid = np.linspace(self.x_min, self.x_max, N_S)
        self.S_grid = np.exp(self.X_grid)
        self.S_max = float(np.exp(self.x_max))
        self.V_grid = np.linspace(0.0, self.V_max, N_V)
        self.dx = float(self.X_grid[1] - self.X_grid[0])
        self.dV = float(self.V_grid[1] - self.V_grid[0])
        self.dt = float(self.T / max(N_T, 1))

    # ---- dense tridiagonal builders ----
    def _tri_S(self, dt_step, theta_loc):
        # Returns per-V tridiagonals as (alpha, beta, gamma) arrays shape (N_V, N_S).
        N = self.N_S
        dx = self.dx
        out = []
        for j in range(self.N_V):
            V = max(float(self.V_grid[j]), 1e-10)
            c2 = 0.5 * V / (dx * dx)
            c1 = ((self.r - self.q) - 0.5 * V) / (2.0 * dx)
            alpha = np.zeros(N); beta = np.ones(N); gamma = np.zeros(N)
            alpha[1:-1] = -theta_loc * dt_step * (c2 - c1)
            beta[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * c2)
            gamma[1:-1] = -theta_loc * dt_step * (c2 + c1)
            out.append((alpha, beta, gamma))
        return out

    def _tri_V(self, dt_step, theta_loc):
        N = self.N_V
        dV = self.dV
        v = np.maximum(self.V_grid, 1e-10)
        coef_d2 = 0.5 * (self.sigma ** 2) * v / (dV * dV)
        coef_d1 = self.kappa * (self.theta - v) / (2.0 * dV)
        alpha = np.zeros(N); beta = np.zeros(N); gamma = np.zeros(N)
        alpha[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] - coef_d1[1:-1])
        beta[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * coef_d2[1:-1])
        gamma[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] + coef_d1[1:-1])
        beta[0] = 1.0; gamma[0] = -1.0; alpha[-1] = -1.0; beta[-1] = 1.0
        return alpha, beta, gamma

    @staticmethod
    def _thomas(a, b, c, d):
        n = len(d)
        cp = np.zeros(n); dp = np.zeros(n); x = np.zeros(n)
        cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-14:
                raise NumericalError("zero pivot in Heston ADI tridiagonal solve (refine grid)")
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        x[n - 1] = dp[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    def _ensure_S_lus(self, dt_step, theta_loc):
        key = (float(dt_step), float(theta_loc))
        if key in self._S_lu_cache:
            return self._S_lu_cache[key]
        lus = []
        for (alpha, beta, gamma) in self._tri_S(dt_step, theta_loc):
            A = sp.diags([alpha[1:], beta, gamma[:-1]], offsets=[-1, 0, 1], format="csc")
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

    # ---- operators ----
    def _A1(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        v_int = self.V_grid[1:-1]; dx = self.dx
        U_xx = (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / (dx * dx)
        U_x = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2.0 * dx)
        out[1:-1, 1:-1] = 0.5 * v_int * U_xx + ((self.r - self.q) - 0.5 * v_int) * U_x
        return out

    def _A2(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        dV = self.dV; v_int = self.V_grid[1:-1]
        coef_d2 = 0.5 * (self.sigma ** 2) * v_int
        coef_d1 = self.kappa * (self.theta - v_int)
        U_VV = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (dV * dV)
        U_V = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * dV)
        out[1:-1, 1:-1] = coef_d2 * U_VV + coef_d1 * U_V
        return out

    def _A0(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3 or abs(self.rho) <= 1e-12:
            return out
        dx = self.dx; dV = self.dV; v_int = self.V_grid[1:-1]
        U_xv = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4.0 * dx * dV)
        out[1:-1, 1:-1] = self.rho * self.sigma * v_int * U_xv
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

    def _solve_S(self, source, A1U, dt_step, theta_loc, tau):
        Y = np.empty_like(source)
        if self.use_sparse:
            lus = self._ensure_S_lus(dt_step, theta_loc)
            for j in range(self.N_V):
                rhs = self._s_boundary_rhs(source[:, j] - theta_loc * dt_step * A1U[:, j], tau)
                Y[:, j] = lus[j].solve(rhs)
        else:
            tris = self._tri_S(dt_step, theta_loc)
            for j in range(self.N_V):
                a, b, c = tris[j]
                rhs = self._s_boundary_rhs(source[:, j] - theta_loc * dt_step * A1U[:, j], tau)
                Y[:, j] = self._thomas(a, b, c, rhs)
        self._bc(Y, tau)
        return Y

    def _solve_V(self, source, A2U, dt_step, theta_loc, tau):
        U_out = np.empty_like(source)
        if self.use_sparse:
            lu = self._ensure_V_lu(dt_step, theta_loc)
            for i in range(self.N_S):
                rhs = source[i, :] - theta_loc * dt_step * A2U[i, :]
                rhs[0] = 0.0; rhs[-1] = 0.0
                U_out[i, :] = lu.solve(rhs)
        else:
            a, b, c = self._tri_V(dt_step, theta_loc)
            for i in range(self.N_S):
                rhs = source[i, :] - theta_loc * dt_step * A2U[i, :]
                rhs[0] = 0.0; rhs[-1] = 0.0
                U_out[i, :] = self._thomas(a, b, c, rhs)
        self._bc(U_out, tau)
        return U_out

    def _douglas_step(self, U, dt_step, tau, theta_loc):
        A1U, A2U = self._A1(U), self._A2(U)
        A0U = self._A0(U)
        Y0 = U + dt_step * (A1U + A2U + A0U - self.r * U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau)
        return self._solve_V(Y1, A2U, dt_step, theta_loc, tau)

    def _cs_step(self, U, dt_step, tau, theta_loc):
        A1U, A2U = self._A1(U), self._A2(U)
        A0U = self._A0(U)
        Y0 = U + dt_step * (A1U + A2U + A0U - self.r * U)
        Y1 = self._solve_S(Y0, A1U, dt_step, theta_loc, tau)
        Y2 = self._solve_V(Y1, A2U, dt_step, theta_loc, tau)
        if abs(self.rho) > 1e-12:
            Ycorr = Y2 + 0.5 * dt_step * (self._A0(Y2) - A0U)
            self._bc(Ycorr, tau)
        else:
            Ycorr = Y2
        Z1 = self._solve_S(Ycorr, A1U, dt_step, theta_loc, tau)
        return self._solve_V(Z1, A2U, dt_step, theta_loc, tau)

    def solve(self, is_call, scheme: ADIScheme, theta, rannacher):
        U = self._terminal(is_call)
        tau = 0.0
        if rannacher and self.N_T >= 1:
            dt_half = 0.5 * self.dt
            tau = dt_half
            U = self._douglas_step(U, dt_half, tau, 1.0)
            tau += dt_half
            U = self._douglas_step(U, dt_half, tau, 1.0)
            steps_remaining = self.N_T - 1
        else:
            steps_remaining = self.N_T
        for _ in range(steps_remaining):
            tau += self.dt
            if scheme == ADIScheme.DOUGLAS:
                U = self._douglas_step(U, self.dt, tau, theta)
            else:  # CRAIG_SNEYD or MCS (treated as CS, matching source)
                U = self._cs_step(U, self.dt, tau, theta)
        return U

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

        Gamma must come from the PDE solution itself (spatial derivatives on the S-grid),
        not from re-bumping bilinearly-interpolated prices (which is piecewise linear and
        yields meaningless curvature).
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


def price_european_heston_pde(
    s0: float,
    strike: float,
    is_call: bool,
    T: float,
    params: HestonParams,
    r: float,
    carry: float,
    n_x: int = 200,
    n_v: int = 100,
    n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD,
    theta: float = 0.5,
    rannacher: bool = True,
    use_sparse: bool = False,
    grid_spot: float = 0.0,
) -> float:
    """Price a European vanilla under Heston via ADI finite differences.

    grid_spot (> 0) pins the spatial grid centering for clean Greeks across spot bumps;
    when 0 the grid centers on s0.
    """
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if n_x < 3 or n_v < 3 or n_t < 1:
        raise ValidationError("require n_x>=3, n_v>=3, n_t>=1")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if not isinstance(scheme, ADIScheme):
        raise ValidationError("scheme must be an ADIScheme")
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the Heston PDE; use CRAIG_SNEYD or DOUGLAS")
    if params.sigma < _SIGMA_MIN:
        return _deterministic_pde_price(s0, strike, is_call, T, params, r, carry)
    solver = _HestonADI(s0, strike, T, r, carry, params, n_x, n_v, n_t, use_sparse,
                        grid_spot=(grid_spot if grid_spot > 0 else None))
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid (grid_spot too far from s0)")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price = solver.interpolate(U, float(np.log(s0)), params.v0)
    if not np.isfinite(price):
        raise NumericalError("Heston PDE produced a non-finite price")
    return price


def price_delta_gamma_heston_pde(
    s0: float, strike: float, is_call: bool, T: float, params: HestonParams,
    r: float, carry: float, n_x: int = 200, n_v: int = 100, n_t: int = 100,
    scheme: ADIScheme = ADIScheme.CRAIG_SNEYD, theta: float = 0.5, rannacher: bool = True,
    use_sparse: bool = False, grid_spot: float = 0.0,
) -> Tuple[float, float, float]:
    """Return (price, spot-delta, spot-gamma) from a single PDE solve.

    Delta/gamma are spatial derivatives of the solved surface (per unit of underlying);
    callers scale by notional/contract size.
    """
    if scheme == ADIScheme.MCS:
        raise ValidationError("MCS is not implemented for the Heston PDE; use CRAIG_SNEYD or DOUGLAS")
    if params.sigma < _SIGMA_MIN:
        from quantark.volmodels.black_scholes import bs_call_price, bs_put_price
        eps = 1e-4 * s0
        f = bs_call_price if is_call else bs_put_price
        v_eff = _deterministic_vol(T, params)
        p = f(s0, strike, T, v_eff, r, carry)
        pu = f(s0 + eps, strike, T, v_eff, r, carry)
        pd = f(s0 - eps, strike, T, v_eff, r, carry)
        return p, (pu - pd) / (2 * eps), (pu - 2 * p + pd) / (eps * eps)
    solver = _HestonADI(s0, strike, T, r, carry, params, n_x, n_v, n_t, use_sparse,
                        grid_spot=(grid_spot if grid_spot > 0 else None))
    if not (solver.S_grid[0] <= s0 <= solver.S_grid[-1]):
        raise ValidationError("s0 falls outside the PDE grid (grid_spot too far from s0)")
    U = solver.solve(is_call, scheme, theta, rannacher)
    price, delta, gamma = solver.price_delta_gamma(U, s0)
    if not (np.isfinite(price) and np.isfinite(delta) and np.isfinite(gamma)):
        raise NumericalError("Heston PDE produced non-finite price/greeks")
    return price, delta, gamma


def _deterministic_vol(T, params: HestonParams) -> float:
    if params.kappa > 1e-12:
        integrated = params.theta * T + (params.v0 - params.theta) * (
            -np.expm1(-params.kappa * T) / params.kappa
        )
    else:
        integrated = params.v0 * T
    return float(np.sqrt(max(integrated, 0.0) / T))


def _deterministic_pde_price(s0, strike, is_call, T, params, r, carry) -> float:
    from quantark.volmodels.black_scholes import bs_call_price, bs_put_price
    v_eff = _deterministic_vol(T, params)
    return (bs_call_price if is_call else bs_put_price)(s0, strike, T, v_eff, r, carry)
