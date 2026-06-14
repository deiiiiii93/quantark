"""Crank-Nicolson PDE pricer for European options under a LocalVolSurface.

Backward PDE: V_t + 0.5 sigma(S,t)^2 S^2 V_SS + (r(t) - carry(t)) S V_S - r(t) V = 0.
Uniform S-grid in [0, Smax], Thomas tri-diagonal solve, local vol at the temporal
midpoint of each step. Rates/carry are PER-STEP forwards (no flat terminal rate);
boundary values use cumulative remaining discount factors so the term structure is
honored. Deterministic; never invokes Monte Carlo.
"""

from __future__ import annotations

import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import safe_exp
from quantark.volmodels.localvol.surface import LocalVolSurface


def _thomas_solve(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system (sub/sup length m-1, diag/rhs length m)."""
    diag = np.asarray(diag, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    m = diag.shape[0]
    if m == 1:
        if diag[0] == 0.0:
            raise NumericalError("singular tridiagonal system (m=1)")
        return np.array([rhs[0] / diag[0]])
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    c = np.empty(m - 1)
    d = np.empty(m)
    if diag[0] == 0.0:
        raise NumericalError("zero pivot in tridiagonal solve")
    c[0] = sup[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, m - 1):
        denom = diag[i] - sub[i - 1] * c[i - 1]
        if denom == 0.0:
            raise NumericalError("zero pivot in tridiagonal solve")
        c[i] = sup[i] / denom
        d[i] = (rhs[i] - sub[i - 1] * d[i - 1]) / denom
    denom_last = diag[m - 1] - sub[m - 2] * c[m - 2]
    if denom_last == 0.0:
        raise NumericalError("zero pivot in tridiagonal solve")
    d[m - 1] = (rhs[m - 1] - sub[m - 2] * d[m - 2]) / denom_last
    x = np.empty(m)
    x[m - 1] = d[m - 1]
    for i in range(m - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def price_european_lv_pde(
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
) -> float:
    """Price a European vanilla under local volatility via Crank-Nicolson.

    Args:
        s0, strike, is_call, T: option/market spec (all positive; T > 0).
        lv_surface: positive LocalVolSurface.
        step_dt, r_fwd, carry_fwd: equal-length per-step arrays; sum(step_dt) should be T.
        n_s: number of spatial grid points (>= 4 -> >= 2 interior nodes).
        s_max_mult: upper-bound multiplier for the S grid (used when s_max <= 0).
        s_max: absolute upper bound for the S grid (overrides s_max_mult when > 0).
        theta: scheme parameter (0.5 = Crank-Nicolson).
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
    ds = smax / (N - 1)
    s_grid = np.linspace(0.0, smax, N)
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

    def boundaries(node_index):
        if is_call:
            return 0.0, smax * float(df_c[node_index]) - strike * float(df_r[node_index])
        return strike * float(df_r[node_index]), 0.0

    v = np.maximum(s_grid - strike, 0.0) if is_call else np.maximum(strike - s_grid, 0.0)

    for m in range(M - 1, -1, -1):
        dt_m = dt[m]
        r_m, carry_m = rf[m], cf[m]
        t_mid = node_t[m] + 0.5 * dt_m
        left_next, right_next = boundaries(m + 1)
        left_curr, right_curr = boundaries(m)
        v[0], v[-1] = left_next, right_next

        sigma = np.asarray(lv_surface.local_vol(s_int, t_mid), dtype=float)
        alpha = 0.5 * sigma * sigma * s_int * s_int / (ds * ds)
        beta = (r_m - carry_m) * s_int / (2.0 * ds)
        A = alpha - beta            # coeff for V_{j-1}
        B = -2.0 * alpha - r_m      # coeff for V_j
        C = alpha + beta            # coeff for V_{j+1}

        sub_A = -theta * dt_m * A[1:]
        diag_A = 1.0 - theta * dt_m * B
        sup_A = -theta * dt_m * C[:-1]

        rhs = (1.0 + (1.0 - theta) * dt_m * B) * v[1:-1]
        rhs[:-1] += (1.0 - theta) * dt_m * C[:-1] * v[2:-1]
        rhs[1:] += (1.0 - theta) * dt_m * A[1:] * v[1:-2]
        # boundary contributions: known (next) + unknown-moved-to-RHS (curr)
        rhs[0] += (1.0 - theta) * dt_m * A[0] * left_next + theta * dt_m * A[0] * left_curr
        rhs[-1] += (1.0 - theta) * dt_m * C[-1] * right_next + theta * dt_m * C[-1] * right_curr

        v[1:-1] = _thomas_solve(sub_A, diag_A, sup_A, rhs)
        v[0], v[-1] = left_curr, right_curr

    return float(np.interp(s0, s_grid, v))
