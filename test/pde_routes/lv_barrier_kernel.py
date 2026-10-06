"""The independent skewed-surface oracle (spec 3C decision 14): the 1D local-vol barrier kernel, a second
discretization of a local-vol barrier on its own uniform grid. No library code calls it: the local-vol barrier
marches on the core (spec 3C Task 18) and so do the Heston / SLV barriers' deterministic sigma -> 0 limits.

Single barriers on a uniform S grid with a Crank-Nicolson knock-out march: continuous monitoring truncates the domain
at the barrier, discrete monitoring injects the knock-out at the observation steps; a knock-in is
Vanilla - KO(rebate=0) + rebate * NoTouch.
"""
from __future__ import annotations

import numpy as np

from quantark.util.numerical import solve_tridiag
from quantark.volmodels.localvol.pde_kernel import price_european_lv_pde


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
