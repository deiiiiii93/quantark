"""
Vanna-Volga barrier attenuation factors.

Two independent attenuation measures:
  - ``gamma_surv``: average single-barrier survival probability (closed form).
  - ``gamma_fet``: expected first-exit-time fraction, computed either by an
    implicit finite-difference PDE (default) OR by Monte-Carlo simulation.

The PDE and MC routines are deliberately separate sibling functions selected by
an explicit ``method`` argument; the PDE path never invokes the MC path (the
deterministic engine stays deterministic).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance

from quantark.param.vol.vannavolga import FXEnv

from .barrier_bs import survival_probability_single


def gamma_surv_single(env: FXEnv, barrier: float, is_up: bool, vol: float) -> float:
    return survival_probability_single(
        spot=env.spot, barrier=barrier, rd=env.rd, rf=env.rf,
        vol=vol, tau=env.tau, is_up=is_up,
    )


def gamma_surv(
    env: FXEnv,
    barrier_low: Optional[float],
    barrier_high: Optional[float],
    vol: float,
) -> float:
    """Average survival probability across measures.

    Single-barrier cases use the relevant barrier. For the two-barrier case this
    returns ``min(g_low, g_high)`` as a conservative single-barrier proxy
    (faithful to the legacy source). True double-barrier survival is the
    intersection of both survival events and is generally strictly lower than
    this proxy; the delivered one-touch VV pricer only uses single barriers.
    # TODO(double-no-touch): replace the min() proxy with a proper
    # double-no-touch survival probability for genuine two-barrier callers.
    """
    if barrier_low is not None and barrier_high is not None:
        g_low = survival_probability_single(
            env.spot, barrier_low, env.rd, env.rf, vol, env.tau, is_up=False
        )
        g_high = survival_probability_single(
            env.spot, barrier_high, env.rd, env.rf, vol, env.tau, is_up=True
        )
        return max(0.0, min(1.0, min(g_low, g_high)))
    if barrier_low is not None:
        return survival_probability_single(
            env.spot, barrier_low, env.rd, env.rf, vol, env.tau, is_up=False
        )
    if barrier_high is not None:
        return survival_probability_single(
            env.spot, barrier_high, env.rd, env.rf, vol, env.tau, is_up=True
        )
    return 1.0


def p_vanna_p_volga_from_gamma(
    gamma_value: float,
    a: float,
    b: float,
    c: float,
    gamma_star: float = 0.95,
) -> Tuple[float, float]:
    """Piecewise-linear vanna/volga attenuation weights."""
    g = max(0.0, min(1.0, gamma_value))
    if g <= gamma_star:
        p_vanna = a * g
        p_volga = b + c * g
    else:
        denom = 1.0 - gamma_star
        p_vanna = a * gamma_star * (1.0 - g) / denom + (g - gamma_star) / denom
        p_volga = (b + c * gamma_star) * (1.0 - g) / denom + (g - gamma_star) / denom
    return p_vanna, p_volga


def _fet_pde(
    spot: float,
    rd: float,
    rf: float,
    vol: float,
    tau: float,
    barrier_low: Optional[float],
    barrier_high: Optional[float],
    drift_extra: float,
    nS: int = 201,
    nT: int = 200,
) -> float:
    """Expected first-exit time via an implicit finite-difference PDE in x=ln S."""
    if tau <= 0.0:
        return 0.0
    mu = rd - rf + drift_extra
    a = 0.5 * vol * vol
    b = mu - 0.5 * vol * vol

    x0 = math.log(spot)
    k_std = 5.0
    # On a side with no real barrier, the truncation boundary is artificial but
    # still absorbing; place it far enough that diffusion AND drift rarely reach
    # it within tau (k_std*sigma*sqrt(tau) + |b|*tau), so it does not bias the
    # first-exit time. (vol<=0 is handled analytically upstream.)
    open_margin = k_std * vol * math.sqrt(tau) + abs(b) * tau
    xL = math.log(barrier_low) if barrier_low is not None else x0 - open_margin
    xR = math.log(barrier_high) if barrier_high is not None else x0 + open_margin
    if xR <= xL + Tolerance.LOG_MIN:
        xR = xL + 1e-3

    N = max(nS, 5)
    M = max(nT, 1)
    xs = np.linspace(xL, xR, N)
    dx = xs[1] - xs[0]
    dth = tau / M

    w_prev = np.full(N, tau, dtype=float)

    lower = np.full(N - 2, -dth * (a / (dx * dx) - b / (2.0 * dx)))
    diag = np.full(N - 2, 1.0 + dth * (2.0 * a / (dx * dx)))
    upper = np.full(N - 2, -dth * (a / (dx * dx) + b / (2.0 * dx)))

    for m in range(1, M + 1):
        th = m * dth
        wL = tau - th
        wR = tau - th
        rhs = w_prev[1:-1].copy()
        rhs[0] -= lower[0] * wL
        rhs[-1] -= upper[-1] * wR

        w_int = _solve_tridiagonal(lower, diag, upper, rhs)
        w_new = np.empty_like(w_prev)
        w_new[0] = wL
        w_new[-1] = wR
        w_new[1:-1] = w_int
        w_prev = w_new

    if x0 <= xL:
        return float(w_prev[0])
    if x0 >= xR:
        return float(w_prev[-1])
    i = int((x0 - xL) / dx)
    i = max(0, min(i, N - 2))
    t = (x0 - (xL + i * dx)) / dx
    return float((1.0 - t) * w_prev[i] + t * w_prev[i + 1])


def _solve_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system (lower/upper include the off-
    diagonal at the boundary rows, which are unused)."""
    n = diag.size
    cp = np.empty(n, dtype=float)
    dp = np.empty(n, dtype=float)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, n):
        denom = diag[k] - lower[k] * cp[k - 1]
        cp[k] = upper[k] / denom if k < n - 1 else 0.0
        dp[k] = (rhs[k] - lower[k] * dp[k - 1]) / denom
    x = np.empty(n, dtype=float)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


def _fet_deterministic(
    spot: float,
    rd: float,
    rf: float,
    tau: float,
    barrier_low: Optional[float],
    barrier_high: Optional[float],
    drift_extra: float,
) -> float:
    """Expected first-exit time for the zero-volatility deterministic path.

    With sigma=0 the log-price drifts linearly at rate ``b = rd - rf +
    drift_extra``; the exit time is the (capped) time for that monotonic path to
    reach a barrier, or ``tau`` if it never does.
    """
    if tau <= 0.0:
        return 0.0
    if (barrier_low is not None and spot <= barrier_low) or (
        barrier_high is not None and spot >= barrier_high
    ):
        return 0.0

    b = rd - rf + drift_extra
    x0 = math.log(spot)
    t_exit = tau
    if barrier_high is not None and b > 0.0:
        t = (math.log(barrier_high) - x0) / b
        if 0.0 < t < t_exit:
            t_exit = t
    if barrier_low is not None and b < 0.0:
        t = (math.log(barrier_low) - x0) / b
        if 0.0 < t < t_exit:
            t_exit = t
    return t_exit


def _simulate_fet(
    spot: float,
    rd: float,
    rf: float,
    vol: float,
    tau: float,
    barrier_low: Optional[float],
    barrier_high: Optional[float],
    drift_extra: float,
    num_paths: int = 20000,
    steps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte-Carlo expected first-exit time (min with tau) for GBM."""
    rng = rng if rng is not None else np.random.default_rng()
    dt = tau / steps
    sqrt_dt = math.sqrt(dt)
    mu = rd - rf + drift_extra
    S = np.full(num_paths, spot, dtype=float)
    hit_time = np.full(num_paths, tau, dtype=float)
    alive = np.ones(num_paths, dtype=bool)

    # Paths already at/beyond a barrier have exited at t=0 (matches the PDE,
    # whose interior excludes a spot sitting on the boundary).
    if (barrier_low is not None and spot <= barrier_low) or (
        barrier_high is not None and spot >= barrier_high
    ):
        return 0.0

    for k in range(1, steps + 1):
        if not alive.any():
            break
        Z = rng.standard_normal(size=num_paths)
        S[alive] = S[alive] * np.exp(
            (mu - 0.5 * vol * vol) * dt + vol * sqrt_dt * Z[alive]
        )
        t_now = k * dt
        if barrier_low is not None:
            crossed_low = alive & (S <= barrier_low)
            hit_time[crossed_low] = t_now
            alive[crossed_low] = False
        if barrier_high is not None:
            crossed_high = alive & (S >= barrier_high)
            hit_time[crossed_high] = t_now
            alive[crossed_high] = False

    return float(hit_time.mean())


def gamma_fet(
    env: FXEnv,
    barrier_low: Optional[float],
    barrier_high: Optional[float],
    vol: float,
    nS: int = 201,
    nT: int = 200,
    method: str = "pde",
    num_paths: int = 20000,
    steps: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """First-exit-time attenuation gamma_fet = 0.5*(lambda^d + lambda^f)/tau.

    ``method='pde'`` (default) uses the deterministic finite-difference solver;
    ``method='mc'`` uses Monte-Carlo simulation.
    """
    if env.tau <= 0.0:
        return 0.0
    if barrier_low is None and barrier_high is None:
        return 1.0

    if vol <= 0.0:
        # Deterministic limit (both PDE and MC degenerate); both measures share
        # the same zero-vol drift path.
        lam_d = _fet_deterministic(env.spot, env.rd, env.rf, env.tau, barrier_low, barrier_high, 0.0)
        lam_f = _fet_deterministic(env.spot, env.rd, env.rf, env.tau, barrier_low, barrier_high, vol * vol)
        return max(0.0, min(1.0, 0.5 * (lam_d + lam_f) / env.tau))

    if method == "pde":
        lam_d = _fet_pde(env.spot, env.rd, env.rf, vol, env.tau, barrier_low, barrier_high, 0.0, nS, nT)
        lam_f = _fet_pde(env.spot, env.rd, env.rf, vol, env.tau, barrier_low, barrier_high, vol * vol, nS, nT)
    elif method == "mc":
        lam_d = _simulate_fet(env.spot, env.rd, env.rf, vol, env.tau, barrier_low, barrier_high, 0.0, num_paths, steps, rng)
        lam_f = _simulate_fet(env.spot, env.rd, env.rf, vol, env.tau, barrier_low, barrier_high, vol * vol, num_paths, steps, rng)
    else:
        raise ValidationError(f"Unknown gamma_fet method '{method}'; expected 'pde' or 'mc'")

    g = 0.5 * (lam_d + lam_f) / env.tau
    return max(0.0, min(1.0, g))


__all__ = [
    "gamma_surv_single",
    "gamma_surv",
    "gamma_fet",
    "p_vanna_p_volga_from_gamma",
]
