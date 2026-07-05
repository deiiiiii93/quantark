"""Crank-Nicolson PDE pricer for European options under a LocalVolSurface.

Backward PDE: V_t + 0.5 sigma(S,t)^2 S^2 V_SS + (r(t) - carry(t)) S V_S - r(t) V = 0.
Uniform S-grid in [0, Smax], LAPACK banded tri-diagonal solve, local vol at the temporal
midpoint of each step. Rates/carry are PER-STEP forwards (no flat terminal rate);
boundary values use cumulative remaining discount factors so the term structure is
honored. Deterministic; never invokes Monte Carlo.
"""

from __future__ import annotations

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp, solve_tridiag
from quantark.volmodels.localvol.surface import LocalVolSurface


def _solve_lv_pde(
    s0: float,
    strike: float,
    is_call: bool,
    T: float,
    lv_surface: LocalVolSurface,
    step_dt: np.ndarray,
    r_fwd: np.ndarray,
    carry_fwd: np.ndarray,
    n_s: int = 300,
    s_max_mult: float = 4.0,
    s_max: float = 0.0,
    theta: float = 0.5,
    rannacher: bool = True,
):
    """Solve the local-vol Crank-Nicolson PDE; return ``(s_grid, v)`` (price curve vs S).

    Args:
        s0, strike, is_call, T: option/market spec (all positive; T > 0).
        lv_surface: positive LocalVolSurface.
        step_dt, r_fwd, carry_fwd: equal-length per-step arrays; sum(step_dt) should be T.
        n_s: number of spatial grid points (>= 4 -> >= 2 interior nodes).
        s_max_mult: upper-bound multiplier for the S grid (used when s_max <= 0).
        s_max: absolute upper bound for the S grid (overrides s_max_mult when > 0).
        theta: scheme parameter (0.5 = Crank-Nicolson).
        rannacher: replace the first (terminal) step with two fully-implicit half-steps to
            damp the payoff kink (default on, matches the Heston/SLV ADI convention). The
            S-grid spacing is also adjusted so the strike falls mid-cell (kink-averaging);
            both alter node placement / first-step semantics — LV goldens move deliberately.
    """
    dt = np.asarray(step_dt, dtype=float)
    rf = np.asarray(r_fwd, dtype=float)
    cf = np.asarray(carry_fwd, dtype=float)
    M = dt.size
    if M < 1 or rf.size != M or cf.size != M:
        raise ValidationError("step_dt, r_fwd, carry_fwd must be equal-length, length >= 1")
    if not (np.all(np.isfinite(dt)) and np.all(dt > 0)):
        raise ValidationError("step_dt must be finite and positive")
    if not (np.all(np.isfinite(rf)) and np.all(np.isfinite(cf))):
        raise ValidationError("r_fwd and carry_fwd must be finite")
    if s0 <= 0 or strike <= 0 or T <= 0:
        raise ValidationError("s0, strike, T must be positive")
    if not np.isclose(float(dt.sum()), float(T), rtol=1e-9, atol=1e-12):
        raise ValidationError(f"sum(step_dt)={dt.sum()} must equal T={T}")
    if n_s < 4:
        raise ValidationError("n_s must be >= 4 (>= 2 interior nodes)")
    if not 0.0 <= theta <= 1.0:
        raise ValidationError("theta must be in [0, 1]")
    if s_max_mult <= 0:
        raise ValidationError("s_max_mult must be positive")

    N = int(n_s)
    smax = float(s_max) if s_max > 0 else s_max_mult * max(s0, strike)
    if smax <= s0:
        raise ValidationError("s_max must exceed spot")
    # Strike mid-cell (kink-averaging): adjust the SPACING so K sits halfway between two
    # nodes while s_grid[0] stays exactly at S=0 and s_grid[-1] stays the upper boundary.
    # Deliberate golden move (WS-C7). (Never shift the whole grid off S=0.)
    ds_nom = smax / (N - 1)
    j_cell = max(int(round(strike / ds_nom - 0.5)), 0)   # cell index whose midpoint hosts K
    ds = strike / (j_cell + 0.5)                          # K == (j_cell + 0.5) * ds (mid-cell)
    smax = ds * (N - 1)                                   # upper bound moves by < ds
    if smax <= s0:
        raise ValidationError("s_max (after mid-cell adjustment) must exceed spot")
    s_grid = np.linspace(0.0, smax, N)                    # s_grid[0] == 0 preserved
    s_int = s_grid[1:-1]

    # node times t_0=0 .. t_M=T and cumulative remaining discount factors to T
    node_t = np.concatenate([[0.0], np.cumsum(dt)])
    step_df_r = np.asarray(safe_exp(-rf * dt), dtype=float)
    step_df_c = np.asarray(safe_exp(-cf * dt), dtype=float)
    df_r = np.ones(M + 1)       # df_r[k] = discount from t_k to T using r
    df_c = np.ones(M + 1)
    for k in range(M - 1, -1, -1):
        df_r[k] = df_r[k + 1] * step_df_r[k]
        df_c[k] = df_c[k + 1] * step_df_c[k]

    def bndry_from_df(dfr, dfc):
        if is_call:
            return 0.0, smax * float(dfc) - strike * float(dfr)
        return strike * float(dfr), 0.0

    def boundaries(node_index):
        return bndry_from_df(df_r[node_index], df_c[node_index])

    def theta_substep(v, dt_sub, theta_loc, t_mid, left_next, right_next,
                      left_curr, right_curr, r_m, carry_m):
        """One theta-step of size dt_sub (theta_loc=1 -> implicit Euler, 0.5 -> CN)."""
        sigma = np.asarray(lv_surface.local_vol(s_int, t_mid), dtype=float)
        alpha = 0.5 * sigma * sigma * s_int * s_int / (ds * ds)
        beta = (r_m - carry_m) * s_int / (2.0 * ds)
        A = alpha - beta            # coeff for V_{j-1}
        B = -2.0 * alpha - r_m      # coeff for V_j
        C = alpha + beta            # coeff for V_{j+1}
        v = v.copy()
        v[0], v[-1] = left_next, right_next
        sub_A = -theta_loc * dt_sub * A[1:]
        diag_A = 1.0 - theta_loc * dt_sub * B
        sup_A = -theta_loc * dt_sub * C[:-1]
        rhs = (1.0 + (1.0 - theta_loc) * dt_sub * B) * v[1:-1]
        rhs[:-1] += (1.0 - theta_loc) * dt_sub * C[:-1] * v[2:-1]
        rhs[1:] += (1.0 - theta_loc) * dt_sub * A[1:] * v[1:-2]
        rhs[0] += (1.0 - theta_loc) * dt_sub * A[0] * left_next + theta_loc * dt_sub * A[0] * left_curr
        rhs[-1] += (1.0 - theta_loc) * dt_sub * C[-1] * right_next + theta_loc * dt_sub * C[-1] * right_curr
        v[1:-1] = solve_tridiag(sub_A, diag_A, sup_A, rhs)
        v[0], v[-1] = left_curr, right_curr
        return v

    v = np.maximum(s_grid - strike, 0.0) if is_call else np.maximum(strike - s_grid, 0.0)

    for m in range(M - 1, -1, -1):
        dt_m = dt[m]
        r_m, carry_m = rf[m], cf[m]
        if rannacher and m == M - 1:
            # two fully-implicit half-steps for the first (terminal) step; exact
            # intermediate discount factors at the T - dt/2 half-node.
            h = 0.5 * dt_m
            dfr_h = df_r[m + 1] * float(np.sqrt(step_df_r[m]))   # discount to T at T - h
            dfc_h = df_c[m + 1] * float(np.sqrt(step_df_c[m]))
            l_hi, r_hi = boundaries(m + 1)
            l_mid, r_mid = bndry_from_df(dfr_h, dfc_h)
            l_lo, r_lo = boundaries(m)
            v = theta_substep(v, h, 1.0, node_t[m + 1] - 0.5 * h,
                              l_hi, r_hi, l_mid, r_mid, r_m, carry_m)
            v = theta_substep(v, h, 1.0, node_t[m + 1] - 1.5 * h,
                              l_mid, r_mid, l_lo, r_lo, r_m, carry_m)
        else:
            l_next, r_next = boundaries(m + 1)
            l_curr, r_curr = boundaries(m)
            v = theta_substep(v, dt_m, theta, node_t[m] + 0.5 * dt_m,
                              l_next, r_next, l_curr, r_curr, r_m, carry_m)

    return s_grid, v


def price_european_lv_pde(
    s0: float, strike: float, is_call: bool, T: float, lv_surface: LocalVolSurface,
    step_dt: np.ndarray, r_fwd: np.ndarray, carry_fwd: np.ndarray, n_s: int = 300,
    s_max_mult: float = 4.0, s_max: float = 0.0, theta: float = 0.5,
    rannacher: bool = True,
) -> float:
    """Price a European vanilla under local volatility via Crank-Nicolson.

    See ``_solve_lv_pde`` for argument semantics.
    """
    s_grid, v = _solve_lv_pde(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd,
                              n_s, s_max_mult, s_max, theta, rannacher)
    return float(np.interp(s0, s_grid, v))


def price_delta_gamma_european_lv_pde(
    s0: float, strike: float, is_call: bool, T: float, lv_surface: LocalVolSurface,
    step_dt: np.ndarray, r_fwd: np.ndarray, carry_fwd: np.ndarray, n_s: int = 300,
    s_max_mult: float = 4.0, s_max: float = 0.0, theta: float = 0.5,
    rannacher: bool = True,
) -> "tuple[float, float, float]":
    """(price, spot-delta, spot-gamma) from a single LV Crank-Nicolson solve.

    Delta/gamma are spatial derivatives of the solved price curve on the S-grid
    (``np.gradient``, edge_order=2), mirroring the ADI readers — not FD re-bumps of the
    interpolated price (which would give meaningless curvature). The price read-off shares
    the exact ``_solve_lv_pde`` curve used by ``price_european_lv_pde``.
    """
    s_grid, v = _solve_lv_pde(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd,
                              n_s, s_max_mult, s_max, theta, rannacher)
    price = float(np.interp(s0, s_grid, v))
    dVdS = np.gradient(v, s_grid, edge_order=2)
    d2VdS2 = np.gradient(dVdS, s_grid, edge_order=2)
    return price, float(np.interp(s0, s_grid, dVdS)), float(np.interp(s0, s_grid, d2VdS2))
