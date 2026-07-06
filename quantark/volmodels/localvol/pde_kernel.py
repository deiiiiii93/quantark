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
    # GROW-ONLY: pick the largest cell index j whose midpoint hosts K with ds >= ds_nom, so
    # the recomputed smax can only grow past the original domain-covering bound, never shrink
    # below s0 (which would reject valid deep-ITM contracts — Codex code-review [high]).
    ds_nom = smax / (N - 1)
    j_cell = int(np.floor(strike / ds_nom - 0.5))         # largest j with strike/(j+0.5) >= ds_nom
    if j_cell >= 0:
        smax = (strike / (j_cell + 0.5)) * (N - 1)        # ds >= ds_nom => smax grows, never shrinks
    # else: strike sits in the first half-cell (already off-node); keep the original grid.
    s_grid = np.linspace(0.0, smax, N)                    # s_grid[0] == 0 preserved
    ds = smax / (N - 1)
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


def _barrier_boundary_value(rebate, pay_at_hit, df_to_T):
    """KO Dirichlet value at the barrier: rebate paid now (pay_at_hit) or discounted to T."""
    return float(rebate) if pay_at_hit else float(rebate) * float(df_to_T)


def _align_discrete_barrier_midcell(smax: float, barrier: float, n_s: int) -> float:
    """Grow a uniform [0, smax] grid so a discrete barrier sits halfway between nodes.

    Discrete barriers are applied as a hard mask at observation times. If the barrier drifts
    across node locations as ``n_s`` changes, the discontinuity can sit almost on a surviving
    node for one grid and almost one full cell away for the next, creating large odd-even
    oscillations. Placing the barrier at a cell midpoint gives the mask an unambiguous
    surviving side and knocked-out side. The domain only grows, so callers keep at least the
    requested far boundary.
    """
    base_smax = float(smax)
    if base_smax <= 0.0 or barrier <= 0.0 or n_s < 4:
        return base_smax
    ds_nom = base_smax / (int(n_s) - 1)
    j_cell = int(np.floor(float(barrier) / ds_nom - 0.5))
    if j_cell < 0:
        return base_smax
    aligned = (float(barrier) / (j_cell + 0.5)) * (int(n_s) - 1)
    return max(base_smax, float(aligned))


def _solve_ko_lv(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd, barrier, is_up,
                 rebate, pay_at_hit, terminal_one, observe_steps, n_s, s_max, theta):
    """Backward CN for a knock-OUT claim under local vol. Returns (s_grid, value_curve).

    terminal_one=True prices the survival (no-touch) leg (terminal payoff == 1, rebate 0);
    else the vanilla option payoff. observe_steps=None -> continuous (inject every step);
    else a set of backward step indices m at which to enforce the KO condition.
    """
    dt = np.asarray(step_dt, float); rf = np.asarray(r_fwd, float); cf = np.asarray(carry_fwd, float)
    M = dt.size
    N = int(n_s)
    continuous = observe_steps is None
    # Continuous monitoring: put the barrier exactly on the domain boundary so the CN
    # Dirichlet condition is enforced at EVERY implicit solve (no discrete-in-time bias).
    # Discrete monitoring: full [0, s_max] grid, KO injected only at observation steps.
    if continuous and is_up:
        smax = float(barrier); s_grid = np.linspace(0.0, smax, N)
    elif continuous and not is_up:
        smax = float(s_max) if s_max > 0 else max(4.0 * max(s0, strike), 3.0 * barrier)
        s_grid = np.linspace(float(barrier), smax, N)
    else:
        smax = float(s_max) if s_max > 0 else max(4.0 * max(s0, strike), 1.5 * barrier)
        if s_max <= 0:
            smax = _align_discrete_barrier_midcell(smax, barrier, N)
        s_grid = np.linspace(0.0, smax, N)
    ds = s_grid[1] - s_grid[0]; s_int = s_grid[1:-1]
    node_t = np.concatenate([[0.0], np.cumsum(dt)])
    step_df_r = np.exp(-rf * dt)
    df_r = np.ones(M + 1)
    for k in range(M - 1, -1, -1):
        df_r[k] = df_r[k + 1] * step_df_r[k]
    ko_mask = (s_grid >= barrier) if is_up else (s_grid <= barrier)
    # boundary node that coincides with the barrier under continuous truncation
    barrier_is_high = continuous and is_up
    barrier_is_low = continuous and (not is_up)

    def inject(v, node_index):
        v = v.copy()
        v[ko_mask] = 0.0 if terminal_one else _barrier_boundary_value(rebate, pay_at_hit, df_r[node_index])
        return v

    if terminal_one:
        v = np.ones(N)
    else:
        v = np.maximum(s_grid - strike, 0.0) if is_call else np.maximum(strike - s_grid, 0.0)
    # Knock out on a terminal breach ONLY when the barrier is monitored at maturity: always for
    # continuous monitoring, but for discrete monitoring only if maturity (backward step M) is an
    # observation. Otherwise a path finishing beyond the barrier in an UNMONITORED tail (last obs
    # < T) must still pay — silently monitoring maturity would over-knock and underprice.
    if continuous or M in observe_steps:
        v = inject(v, M)

    for m in range(M - 1, -1, -1):
        r_m, c_m, dt_m = rf[m], cf[m], dt[m]
        t_mid = node_t[m] + 0.5 * dt_m
        sigma = np.asarray(lv_surface.local_vol(s_int, t_mid), float)
        alpha = 0.5 * sigma * sigma * s_int * s_int / (ds * ds)
        beta = (r_m - c_m) * s_int / (2.0 * ds)
        A = alpha - beta; B = -2.0 * alpha - r_m; C = alpha + beta
        # Far-field Dirichlet ends for the vanilla claim.
        if terminal_one:
            lo_n = lo_c = float(df_r[m + 1]) if s_grid[0] > 0 else 0.0
            hi_n = float(df_r[m + 1]); hi_c = float(df_r[m])
            lo_c = float(df_r[m]) if s_grid[0] > 0 else 0.0
        elif is_call:
            lo_n = lo_c = 0.0
            hi_n = (smax - strike) * float(df_r[m + 1]); hi_c = (smax - strike) * float(df_r[m])
        else:
            lo_n = strike * float(df_r[m + 1]); lo_c = strike * float(df_r[m]); hi_n = hi_c = 0.0
        # Override the boundary that sits ON the barrier with the knock-out value.
        ko_hi = 0.0 if terminal_one else _barrier_boundary_value(rebate, pay_at_hit, df_r[m + 1])
        ko_hi_c = 0.0 if terminal_one else _barrier_boundary_value(rebate, pay_at_hit, df_r[m])
        if barrier_is_high:
            hi_n, hi_c = ko_hi, ko_hi_c
        if barrier_is_low:
            lo_n, lo_c = ko_hi, ko_hi_c
        v[0], v[-1] = lo_n, hi_n
        sub = -theta * dt_m * A[1:]; diag = 1.0 - theta * dt_m * B; sup = -theta * dt_m * C[:-1]
        rhs = (1.0 + (1.0 - theta) * dt_m * B) * v[1:-1]
        rhs[:-1] += (1.0 - theta) * dt_m * C[:-1] * v[2:-1]
        rhs[1:] += (1.0 - theta) * dt_m * A[1:] * v[1:-2]
        rhs[0] += (1.0 - theta) * dt_m * A[0] * lo_n + theta * dt_m * A[0] * lo_c
        rhs[-1] += (1.0 - theta) * dt_m * C[-1] * hi_n + theta * dt_m * C[-1] * hi_c
        v[1:-1] = solve_tridiag(sub, diag, sup, rhs)
        v[0], v[-1] = lo_c, hi_c
        if observe_steps is None or m in observe_steps:
            v = inject(v, m)
    return s_grid, v


def price_barrier_lv_pde(
    s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd, barrier, is_up, is_out,
    rebate=0.0, pay_at_hit=False, continuous=True, observe_steps=None, n_s=400, s_max=0.0, theta=0.5,
):
    """Single-barrier option under local vol via a 1D Crank-Nicolson KO PDE.

    Knock-in uses KI = Vanilla - KO(rebate=0) + rebate * NoTouch, where NoTouch is the
    survival value (terminal payoff 1, KO to 0 at the barrier). Continuous monitoring injects
    the KO condition every step; discrete injects at ``observe_steps`` (backward indices).
    """
    from quantark.volmodels.barrier import BarrierSpec, validate_barrier
    spec = BarrierSpec(bool(is_up), bool(is_out), bool(is_call), float(barrier), float(strike),
                       float(rebate), bool(pay_at_hit))
    validate_barrier(spec, s0)
    obs = None if continuous else set(int(m) for m in observe_steps)
    if is_out:
        sg, v = _solve_ko_lv(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd, barrier,
                             is_up, rebate, pay_at_hit, False, obs, n_s, s_max, theta)
        return float(np.interp(s0, sg, v))
    # knock-in decomposition
    van = price_european_lv_pde(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd,
                                n_s=n_s, s_max=s_max, theta=theta)
    sg, ko0 = _solve_ko_lv(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd, barrier,
                           is_up, 0.0, False, False, obs, n_s, s_max, theta)
    ki = van - float(np.interp(s0, sg, ko0))
    if rebate > 0.0:
        sg2, nt = _solve_ko_lv(s0, strike, is_call, T, lv_surface, step_dt, r_fwd, carry_fwd, barrier,
                               is_up, 0.0, False, True, obs, n_s, s_max, theta)
        ki += float(rebate) * float(np.interp(s0, sg2, nt))
    return ki
