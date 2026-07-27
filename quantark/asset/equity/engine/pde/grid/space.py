"""The ONE spatial builder (spec §4.4).

Multi-point sinh (Tavella-Randall ODE) concentration with a pinned auto-bounds
formula. Zero interior critical prices degenerates to uniform-in-log; hard
bounds override their side verbatim (they ARE the domain edge). Barrier
snapping is deliberately absent: the cell-average event projection keeps
second-order accuracy wherever a barrier falls relative to nodes, and NOT
snapping is what makes layouts shareable across products and stable across
bumps.

The ODE machinery (_ode_f / _ode_rk4_step / _ode_integrate / _ode_find_A) is
ported verbatim from the certified ``spatial_grid.py`` implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot

logger = logging.getLogger(__name__)

#: Domain half-width floor in log space (±10%) — near-expiry / near-zero vol.
_H_FLOOR = np.log(1.10)
#: Cells of margin a critical price keeps from a (non-hard) domain edge.
_CRIT_MARGIN_CELLS = 5.0
#: Max acceptable max/min dx ratio (certified check_grid_quality default).
_MAX_DX_RATIO = 100.0


@dataclass(frozen=True, eq=False)
class SpatialLayout:
    """Per-underlying spatial geometry; compared by identity (eq=False)."""

    s: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    bounds: Tuple[float, float]
    achieved_eps: float


# ---------------------------------------------------------------------------
# Tavella-Randall ODE machinery — ported verbatim from spatial_grid.py
# ---------------------------------------------------------------------------

def _ode_f(y: float, A: float, beta: float, crits: np.ndarray) -> float:
    j_sq = beta * beta + (y - crits) ** 2
    s = np.sum(1.0 / j_sq)
    return A / np.sqrt(s)


def _ode_rk4_step(y: float, h: float, A: float, beta: float, crits: np.ndarray) -> float:
    k1 = _ode_f(y, A, beta, crits)
    k2 = _ode_f(y + 0.5 * h * k1, A, beta, crits)
    k3 = _ode_f(y + 0.5 * h * k2, A, beta, crits)
    k4 = _ode_f(y + h * k3, A, beta, crits)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _ode_integrate(
    y_min: float, N: int, A: float, beta: float, crits: np.ndarray
) -> np.ndarray:
    h = 1.0 / N
    mesh = np.empty(N + 1, dtype=float)
    mesh[0] = y_min
    y = y_min
    for i in range(1, N + 1):
        y = _ode_rk4_step(y, h, A, beta, crits)
        mesh[i] = y
    return mesh


def _ode_find_A(
    y_min: float, y_max: float, N: int, beta: float, crits: np.ndarray
) -> float:
    a_lo = 0.0
    a_hi = max(4.0 * abs(y_max), abs(y_max - y_min))

    def residual(A: float) -> float:
        return _ode_integrate(y_min, N, A, beta, crits)[-1] - y_max

    f_lo = residual(a_lo)
    f_hi = residual(a_hi)
    for _ in range(20):
        if f_lo * f_hi <= 0:
            break
        a_hi *= 2.0
        f_hi = residual(a_hi)
    tol = 1e-10
    for _ in range(100):
        if abs(a_hi - a_lo) <= tol:
            break
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = residual(a_mid)
        if f_mid == 0.0:
            return a_mid
        if f_lo * f_mid < 0:
            a_hi, f_hi = a_mid, f_mid
        else:
            a_lo, f_lo = a_mid, f_mid
    return 0.5 * (a_lo + a_hi)


# ---------------------------------------------------------------------------
# Bounds + concentration
# ---------------------------------------------------------------------------

def _auto_bounds(
    request: GridRequest, market: MarketSnapshot, config: GridConfig, tau: float
) -> Tuple[float, float, "float | None", "float | None"]:
    """Pinned auto-bounds (spec §4.4): drift-adjusted anchor envelope, critical
    -price margin, ±10% floor; hard bounds override their side verbatim."""
    sigma, r, q = market.sigma_ref, market.r_ref, market.q_ref
    h = config.num_std * sigma * np.sqrt(tau) + abs(r - q - 0.5 * sigma * sigma) * tau
    h = max(h, _H_FLOOR)
    anchors_x = np.log(np.asarray(request.bound_anchors))
    x_lo = float(anchors_x.min() - h)
    x_hi = float(anchors_x.max() + h)

    margin = _CRIT_MARGIN_CELLS * np.log1p(config.eps_crit)
    for p in request.critical_prices:
        xp = np.log(p)
        x_lo = min(x_lo, xp - margin)
        x_hi = max(x_hi, xp + margin)

    # Expert per-side overrides (validated against hard bounds in the binder),
    # then product hard bounds (the domain edge — highest precedence). The
    # verbatim price is carried alongside so exp(log(.)) round-trip noise never
    # reaches the domain edge (a barrier boundary is exact by contract).
    cfg_lo, cfg_hi = config.bounds
    lo_exact = hi_exact = None
    if cfg_lo is not None:
        x_lo, lo_exact = np.log(cfg_lo), float(cfg_lo)
    if cfg_hi is not None:
        x_hi, hi_exact = np.log(cfg_hi), float(cfg_hi)
    if request.hard_lower is not None:
        x_lo, lo_exact = np.log(request.hard_lower), float(request.hard_lower)
    if request.hard_upper is not None:
        x_hi, hi_exact = np.log(request.hard_upper), float(request.hard_upper)
    return x_lo, x_hi, lo_exact, hi_exact


def _local_eps(x: np.ndarray, x_crits: np.ndarray) -> float:
    """Worst-case relative spacing at the critical prices (spec §4.4):
    per critical price the min of its two adjacent dx, maximized over
    criticals, converted to relative spacing via expm1."""
    dx = np.diff(x)
    worst = 0.0
    for xc in x_crits:
        idx = int(np.searchsorted(x, xc))
        cand = []
        if 0 < idx <= len(dx):
            cand.append(dx[idx - 1])
        if idx < len(dx):
            cand.append(dx[idx])
        if cand:
            worst = max(worst, float(min(cand)))
    return float(np.expm1(worst))


def build_space(
    request: GridRequest, market: MarketSnapshot, config: GridConfig
) -> SpatialLayout:
    """Build the spatial layout for one request (or a shared union request)."""
    tau = request.tau
    x_lo, x_hi, lo_exact, hi_exact = _auto_bounds(request, market, config, tau)

    # Interior critical prices only; outside-domain ones (possible only under
    # hard/expert bounds) are excluded from concentration and logged.
    crits = []
    for p in sorted(set(request.critical_prices)):
        xp = float(np.log(p))
        if x_lo < xp < x_hi:
            crits.append(xp)
        else:
            logger.info(
                "spatial grid: critical price %s outside domain [%s, %s]; "
                "excluded from concentration",
                p,
                np.exp(x_lo),
                np.exp(x_hi),
            )
    x_crits = np.array(crits, dtype=float)

    N = int(config.points) - 1
    dx_target = float(np.log1p(config.eps_crit))

    if len(x_crits) == 0:
        x = np.linspace(x_lo, x_hi, N + 1)
    elif (x_hi - x_lo) / N <= dx_target:
        # Uniform already satisfies the target — clamp, no bracket search.
        x = np.linspace(x_lo, x_hi, N + 1)
    else:
        x = _concentrated_mesh(x_lo, x_hi, N, x_crits, dx_target)

    achieved = _local_eps(x, x_crits) if len(x_crits) else float(
        np.expm1((x_hi - x_lo) / N)
    )
    if len(x_crits) and achieved > 2.0 * config.eps_crit:
        logger.warning(
            "spatial grid: achieved spacing %.5f exceeds 2x target eps_crit "
            "%.5f (accuracy degradation, not an error)",
            achieved,
            config.eps_crit,
        )

    s = np.exp(x)
    lo_price = lo_exact if lo_exact is not None else float(np.exp(x_lo))
    hi_price = hi_exact if hi_exact is not None else float(np.exp(x_hi))
    s[0], s[-1] = lo_price, hi_price  # domain edges are exact by contract
    dx = np.diff(x)
    for arr in (x, s, dx):
        arr.setflags(write=False)
    return SpatialLayout(
        s=s,
        x=x,
        dx=dx,
        bounds=(lo_price, hi_price),
        achieved_eps=achieved,
    )


def _concentrated_mesh(
    x_lo: float, x_hi: float, N: int, x_crits: np.ndarray, dx_target: float
) -> np.ndarray:
    """One beta search: log-scale bisection on the worst-critical-point
    spacing inequality (spec §4.4)."""
    beta_lo = 1e-12
    beta_hi = 1e3 * (x_hi - x_lo)

    # Beta is a shape parameter: search it on a coarse mesh with the target
    # scaled by N/N_coarse (certified optimization from spatial_grid.py —
    # full-N shooting inside a 30-iter bisection costs tens of seconds).
    N_coarse = min(N, 64)
    dx_target_adj = dx_target * (N / N_coarse)

    def worst_spacing(beta: float) -> float:
        A = _ode_find_A(x_lo, x_hi, N_coarse, beta, x_crits)
        mesh = _ode_integrate(x_lo, N_coarse, A, beta, x_crits)
        dx = np.diff(mesh)
        worst = 0.0
        for xc in x_crits:
            idx = int(np.searchsorted(mesh, xc))
            cand = []
            if 0 < idx <= len(dx):
                cand.append(dx[idx - 1])
            if idx < len(dx):
                cand.append(dx[idx])
            if cand:
                worst = max(worst, float(min(cand)))
        return worst

    if worst_spacing(beta_lo) > dx_target_adj:
        # Target unreachable even at max concentration: best achievable.
        beta = beta_lo
    else:
        lo, hi = beta_lo, beta_hi
        for _ in range(30):
            mid = float(np.sqrt(lo * hi))
            if worst_spacing(mid) > dx_target_adj:
                hi = mid  # too coarse at criticals -> concentrate more
            else:
                lo = mid  # meets target -> try smoother
        beta = lo

    def mesh_for(b: float) -> np.ndarray:
        A = _ode_find_A(x_lo, x_hi, N, b, x_crits)
        return _ode_integrate(x_lo, N, A, b, x_crits)

    def full_worst(mesh: np.ndarray) -> float:
        dx = np.diff(mesh)
        worst = 0.0
        for xc in x_crits:
            idx = int(np.searchsorted(mesh, xc))
            cand = []
            if 0 < idx <= len(dx):
                cand.append(dx[idx - 1])
            if idx < len(dx):
                cand.append(dx[idx])
            if cand:
                worst = max(worst, float(min(cand)))
        return worst

    # The coarse-mesh estimate lands within a few percent of the target;
    # refine on the full mesh until the inequality actually holds (bounded).
    mesh = mesh_for(beta)
    for _ in range(8):
        if full_worst(mesh) <= dx_target:
            break
        beta *= 0.85
        mesh = mesh_for(beta)

    # Tail-coarseness guard (ported from the certified check_grid_quality use):
    # a pointwise spacing target met by a degenerate grid (max/min dx ratio
    # beyond 100) is worse than missing the target, so relax concentration
    # until the grid is non-degenerate; achieved_eps then reports the miss.
    # This guard has final say over the refinement above.
    for _ in range(20):
        dx = np.diff(mesh)
        if float(dx.max() / dx.min()) <= _MAX_DX_RATIO:
            break
        beta *= 1.2
        mesh = mesh_for(beta)
    # Endpoint exactness (RK4 shooting lands within tol; pin the ends).
    mesh[0], mesh[-1] = x_lo, x_hi
    return mesh
