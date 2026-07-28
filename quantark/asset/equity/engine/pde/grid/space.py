"""The ONE spatial builder (spec §4.4).

Multi-point Tavella-Randall monitor concentration with a pinned auto-bounds
formula. Zero interior critical prices degenerates to uniform-in-log; hard
bounds override their side verbatim (they ARE the domain edge). Barrier
snapping is deliberately absent: the cell-average event projection keeps
second-order accuracy wherever a barrier falls relative to nodes, and NOT
snapping is what makes layouts shareable across products and stable across
bumps.

Dense or unstable critical sets solve the separable monitor ODE by cumulative
integration and inversion, avoiding nested shooting and overflow-prone RK4
trial paths while preserving certified small-set layouts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot
from quantark.util.exceptions import ValidationError

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
    active_critical_prices: Tuple[float, ...] = ()
    ignored_critical_prices: Tuple[float, ...] = ()


# ---------------------------------------------------------------------------
# Tavella-Randall monitor equidistribution
# ---------------------------------------------------------------------------

#: A critical price farther than twice the model envelope from every anchor
#: is outside the numerical reach of this grid.  Such levels are often used
#: downstream as "disabled observation" sentinels (for example 100x spot).
#: They must not widen or concentrate the live pricing domain.
_REACHABILITY_ENVELOPE_MULTIPLIER = 2.0
#: Fixed iteration caps make mesh construction cost independent of whether a
#: shooting bracket happens to converge.
_BETA_SEARCH_ITERS = 32
#: Preserve the certified legacy layout for small, well-behaved critical
#: sets.  Dense schedules take the stable cumulative path that fixes the
#: autocallable cliff; any numerical/quality failure also falls back to it.
_LEGACY_MAX_CRITICALS = 8
#: Chunking bounds temporary memory when a product has many unique barriers.
_MONITOR_CRIT_CHUNK = 64
#: Local support around each monitor singularity for cumulative integration.
_MONITOR_LOCAL_OFFSETS = np.array(
    (
        -32.0,
        -16.0,
        -8.0,
        -4.0,
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
    ),
    dtype=float,
)


def _integration_support(
    x_lo: float, x_hi: float, N: int, beta: float, crits: np.ndarray
) -> np.ndarray:
    """Integration support resolving both the domain and monitor peaks."""
    base_points = max(1025, min(8193, 8 * N + 1))
    pieces = [np.linspace(x_lo, x_hi, base_points)]
    for xc in crits:
        left = np.arcsinh((x_lo - xc) / beta)
        right = np.arcsinh((x_hi - xc) / beta)
        u = np.linspace(left, right, 65)
        pieces.append(xc + beta * np.sinh(u))
        pieces.append(xc + beta * _MONITOR_LOCAL_OFFSETS)
    support = np.unique(np.clip(np.concatenate(pieces), x_lo, x_hi))
    return support


def _monitor_values(
    support: np.ndarray, beta: float, crits: np.ndarray
) -> np.ndarray:
    """Stable values of sqrt(sum_k 1 / (beta^2 + (x-c_k)^2))."""
    monitor_sq = np.zeros_like(support)
    beta_sq = beta * beta
    for start in range(0, len(crits), _MONITOR_CRIT_CHUNK):
        chunk = crits[start : start + _MONITOR_CRIT_CHUNK]
        delta = support[:, None] - chunk[None, :]
        monitor_sq += np.sum(1.0 / (beta_sq + delta * delta), axis=1)
    return np.sqrt(monitor_sq)


def _equidistributed_mesh(
    x_lo: float, x_hi: float, N: int, beta: float, crits: np.ndarray
) -> np.ndarray:
    """Invert the cumulative Tavella-Randall monitor.

    The former implementation found an ODE shooting constant with nested
    RK4/bisection loops.  The ODE is separable: its solution is exactly the
    inverse cumulative integral of the monitor.  Constructing that cumulative
    integral directly removes the unstable shooting bracket and its overflow
    path while preserving the same equidistribution equation.
    """
    support = _integration_support(x_lo, x_hi, N, beta, crits)
    monitor = _monitor_values(support, beta, crits)
    widths = np.diff(support)
    cumulative = np.empty_like(support)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(
        0.5 * widths * (monitor[:-1] + monitor[1:])
    )
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise ValidationError(
            "spatial grid monitor integration failed; increase "
            "GridConfig(points=...) or select a higher accuracy profile"
        )
    targets = np.linspace(0.0, cumulative[-1], N + 1)
    mesh = np.interp(targets, cumulative, support)
    mesh[0], mesh[-1] = x_lo, x_hi
    return mesh


# ---------------------------------------------------------------------------
# Bounds + concentration
# ---------------------------------------------------------------------------

def _anchor_envelopes(
    request: GridRequest, market: MarketSnapshot, config: GridConfig, tau: float
) -> Tuple[float, float, float, float]:
    """Base and conservative reachability envelopes in log space."""
    sigma, r, q = market.sigma_ref, market.r_ref, market.q_ref
    h = config.num_std * sigma * np.sqrt(tau) + abs(r - q - 0.5 * sigma * sigma) * tau
    h = max(h, _H_FLOOR)
    anchors_x = np.log(np.asarray(request.bound_anchors))
    x_lo = float(anchors_x.min() - h)
    x_hi = float(anchors_x.max() + h)
    reach_h = _REACHABILITY_ENVELOPE_MULTIPLIER * h
    reach_lo = float(anchors_x.min() - reach_h)
    reach_hi = float(anchors_x.max() + reach_h)
    return x_lo, x_hi, reach_lo, reach_hi


def _reachable_critical_prices(
    request: GridRequest, market: MarketSnapshot, config: GridConfig, tau: float
) -> Tuple[float, ...]:
    """Critical prices inside the conservative model-reachability envelope."""
    _, _, reach_lo, reach_hi = _anchor_envelopes(request, market, config, tau)
    active = []
    for p in sorted(set(request.critical_prices)):
        xp = float(np.log(p))
        if reach_lo <= xp <= reach_hi:
            active.append(float(p))
        else:
            logger.info(
                "spatial grid: critical price %s outside conservative "
                "reachability envelope [%s, %s]; ignored for grid construction",
                p,
                np.exp(reach_lo),
                np.exp(reach_hi),
            )
    return tuple(active)


def _auto_bounds(
    request: GridRequest, market: MarketSnapshot, config: GridConfig, tau: float
) -> Tuple[float, float, "float | None", "float | None"]:
    """Pinned auto-bounds with reachable-critical margins and hard overrides."""
    x_lo, x_hi, _, _ = _anchor_envelopes(request, market, config, tau)

    margin = _CRIT_MARGIN_CELLS * np.log1p(config.eps_crit)
    for p in _reachable_critical_prices(request, market, config, tau):
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

    # Only reachable, interior critical prices participate in concentration.
    # Far operational sentinels therefore affect neither bounds nor spacing.
    crits = []
    active_prices = []
    for p in _reachable_critical_prices(request, market, config, tau):
        xp = float(np.log(p))
        if x_lo < xp < x_hi:
            crits.append(xp)
            active_prices.append(float(p))
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

    dx = np.diff(x)
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(dx))
        or np.any(dx <= 0.0)
    ):
        raise ValidationError(
            "spatial grid construction produced non-finite or non-monotone "
            "nodes; increase GridConfig(points=...) or select a higher "
            "accuracy profile"
        )
    dx_ratio = float(dx.max() / dx.min())
    if dx_ratio > _MAX_DX_RATIO * (1.0 + 1e-8):
        raise ValidationError(
            f"spatial grid max/min spacing ratio {dx_ratio:.3f} exceeds "
            f"{_MAX_DX_RATIO:.0f}; increase GridConfig(points=...) or select "
            "a higher accuracy profile"
        )

    achieved = _local_eps(x, x_crits) if len(x_crits) else float(
        np.expm1((x_hi - x_lo) / N)
    )
    if len(x_crits) and achieved > 2.0 * config.eps_crit:
        raise ValidationError(
            f"spatial grid achieved spacing {achieved:.5f} exceeds 2x target "
            f"eps_crit {config.eps_crit:.5f}; increase GridConfig(points=...) "
            "or select a higher accuracy profile"
        )

    s = np.exp(x)
    lo_price = lo_exact if lo_exact is not None else float(np.exp(x_lo))
    hi_price = hi_exact if hi_exact is not None else float(np.exp(x_hi))
    s[0], s[-1] = lo_price, hi_price  # domain edges are exact by contract
    for arr in (x, s, dx):
        arr.setflags(write=False)
    return SpatialLayout(
        s=s,
        x=x,
        dx=dx,
        bounds=(lo_price, hi_price),
        achieved_eps=achieved,
        active_critical_prices=tuple(active_prices),
        ignored_critical_prices=tuple(
            sorted(set(request.critical_prices) - set(active_prices))
        ),
    )


def _stable_concentrated_mesh(
    x_lo: float, x_hi: float, N: int, x_crits: np.ndarray, dx_target: float
) -> np.ndarray:
    """Bounded beta search over stable cumulative monitor inversion."""
    span = x_hi - x_lo
    beta_lo = max(span * 1e-10, 1e-12)
    beta_hi = 1e3 * span
    cache = {}

    def metrics(mesh: np.ndarray) -> Tuple[float, float]:
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
        if np.any(dx <= 0.0) or not np.all(np.isfinite(dx)):
            return worst, float("inf")
        return worst, float(dx.max() / dx.min())

    def evaluate(beta: float) -> Tuple[np.ndarray, float, float]:
        key = float(beta)
        hit = cache.get(key)
        if hit is None:
            mesh = _equidistributed_mesh(x_lo, x_hi, N, key, x_crits)
            worst, ratio = metrics(mesh)
            hit = (mesh, worst, ratio)
            cache[key] = hit
        return hit

    _, worst_lo, _ = evaluate(beta_lo)
    if worst_lo > dx_target:
        beta = beta_lo
    else:
        lo, hi = beta_lo, beta_hi
        for _ in range(_BETA_SEARCH_ITERS):
            mid = float(np.sqrt(lo * hi))
            _, worst, _ = evaluate(mid)
            if worst > dx_target:
                hi = mid
            else:
                lo = mid
        beta = lo

    mesh, _, ratio = evaluate(beta)
    if ratio > _MAX_DX_RATIO:
        # The quality guard has final say: find the least relaxation that
        # restores max/min dx <= 100, even if the pointwise target then becomes
        # infeasible (build_space applies the fail-closed 2x check).
        lo, hi = beta, beta_hi
        _, _, hi_ratio = evaluate(hi)
        if hi_ratio > _MAX_DX_RATIO:
            return np.linspace(x_lo, x_hi, N + 1)
        for _ in range(_BETA_SEARCH_ITERS):
            mid = float(np.sqrt(lo * hi))
            _, _, mid_ratio = evaluate(mid)
            if mid_ratio > _MAX_DX_RATIO:
                lo = mid
            else:
                hi = mid
        mesh = evaluate(hi)[0]

    mesh[0], mesh[-1] = x_lo, x_hi
    return mesh


def _legacy_ode_f(
    y: float, A: float, beta: float, crits: np.ndarray
) -> float:
    j_sq = beta * beta + (y - crits) ** 2
    return A / np.sqrt(np.sum(1.0 / j_sq))


def _legacy_ode_rk4_step(
    y: float, h: float, A: float, beta: float, crits: np.ndarray
) -> float:
    k1 = _legacy_ode_f(y, A, beta, crits)
    k2 = _legacy_ode_f(y + 0.5 * h * k1, A, beta, crits)
    k3 = _legacy_ode_f(y + 0.5 * h * k2, A, beta, crits)
    k4 = _legacy_ode_f(y + h * k3, A, beta, crits)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _legacy_ode_integrate(
    y_min: float, N: int, A: float, beta: float, crits: np.ndarray
) -> np.ndarray:
    h = 1.0 / N
    mesh = np.empty(N + 1, dtype=float)
    mesh[0] = y_min
    y = y_min
    for i in range(1, N + 1):
        y = _legacy_ode_rk4_step(y, h, A, beta, crits)
        mesh[i] = y
    return mesh


def _legacy_ode_find_A(
    y_min: float, y_max: float, N: int, beta: float, crits: np.ndarray
) -> float:
    a_lo = 0.0
    a_hi = max(4.0 * abs(y_max), abs(y_max - y_min))

    def residual(A: float) -> float:
        return (
            _legacy_ode_integrate(y_min, N, A, beta, crits)[-1] - y_max
        )

    f_lo = residual(a_lo)
    f_hi = residual(a_hi)
    for _ in range(20):
        if f_lo * f_hi <= 0:
            break
        a_hi *= 2.0
        f_hi = residual(a_hi)
    for _ in range(100):
        if abs(a_hi - a_lo) <= 1e-10:
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


def _legacy_concentrated_mesh(
    x_lo: float, x_hi: float, N: int, x_crits: np.ndarray, dx_target: float
) -> np.ndarray:
    """Certified pre-fix layout path for small critical sets."""
    beta_lo = 1e-12
    beta_hi = 1e3 * (x_hi - x_lo)
    n_coarse = min(N, 64)
    dx_target_adj = dx_target * (N / n_coarse)

    def worst_spacing(beta: float) -> float:
        A = _legacy_ode_find_A(x_lo, x_hi, n_coarse, beta, x_crits)
        mesh = _legacy_ode_integrate(x_lo, n_coarse, A, beta, x_crits)
        return _worst_spacing(mesh, x_crits)

    if worst_spacing(beta_lo) > dx_target_adj:
        beta = beta_lo
    else:
        lo, hi = beta_lo, beta_hi
        for _ in range(30):
            mid = float(np.sqrt(lo * hi))
            if worst_spacing(mid) > dx_target_adj:
                hi = mid
            else:
                lo = mid
        beta = lo

    def mesh_for(value: float) -> np.ndarray:
        A = _legacy_ode_find_A(x_lo, x_hi, N, value, x_crits)
        return _legacy_ode_integrate(x_lo, N, A, value, x_crits)

    mesh = mesh_for(beta)
    for _ in range(8):
        if _worst_spacing(mesh, x_crits) <= dx_target:
            break
        beta *= 0.85
        mesh = mesh_for(beta)
    for _ in range(20):
        dx = np.diff(mesh)
        if float(dx.max() / dx.min()) <= _MAX_DX_RATIO:
            break
        beta *= 1.2
        mesh = mesh_for(beta)
    mesh[0], mesh[-1] = x_lo, x_hi
    return mesh


def _worst_spacing(mesh: np.ndarray, x_crits: np.ndarray) -> float:
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


def _concentrated_mesh(
    x_lo: float, x_hi: float, N: int, x_crits: np.ndarray, dx_target: float
) -> np.ndarray:
    """Certified small-set layout, with bounded stable fallback for failures."""
    if len(x_crits) <= _LEGACY_MAX_CRITICALS:
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                mesh = _legacy_concentrated_mesh(
                    x_lo, x_hi, N, x_crits, dx_target
                )
            dx = np.diff(mesh)
            if (
                np.all(np.isfinite(mesh))
                and np.all(dx > 0.0)
                and float(dx.max() / dx.min())
                <= _MAX_DX_RATIO * (1.0 + 1e-8)
            ):
                return mesh
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            pass
        logger.info(
            "spatial grid: legacy concentration was unstable; using bounded "
            "cumulative monitor inversion"
        )
    return _stable_concentrated_mesh(x_lo, x_hi, N, x_crits, dx_target)
