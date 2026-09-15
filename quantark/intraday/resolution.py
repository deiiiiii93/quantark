"""Diffusion-layer resolution status for grid routes (PDE).

Near an event the value function is a jump smoothed over the remaining standard
deviation sqrt(W) to that event. A grid resolves it only with several cells
across that layer AND a time grid that carries the layer variance: the solver
marches in calendar time with per-step variance dW_k, so

* one step carrying the whole session's variance jumps the layer in a single
  Crank-Nicolson step however fine the space grid is (steps per layer), and
* the grid-scale sawtooth mode the jump excites must have decayed by the
  valuation: a theta-step multiplies it by |1 - 2(1-theta)mu| / (1 + 2 theta mu),
  mu = dW_k / dx_min^2, which tends to -1 for Crank-Nicolson when mu >> 1; the
  implicit (Rannacher) steps of the solver's own schedule count at theta = 1.

Below any floor the price can still be computed, but no accuracy claim can be
attached to it. W == 0 exactly is a deterministic
interval: there is no layer at all (the route's accuracy is then a matter of
barrier/spot placement on the grid, judged by Gate C, not of diffusion
resolution).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, sqrt
from typing import Optional, Sequence

import numpy as np

REQUIRED_CELLS_PER_LAYER = 4.0
#: The parabolic mesh ratio dW/dx^2 stays <= 1 at the layer scale once the layer spans the required cells.
REQUIRED_STEPS_PER_LAYER = REQUIRED_CELLS_PER_LAYER ** 2
#: nepers the grid-scale mode must lose between the first event and the valuation (e^-8 ~ 3e-4 of the jump)
REQUIRED_GRID_MODE_DAMPING = 8.0
INTRADAY_PDE_MAX_POINTS = 20_000
INTRADAY_PDE_MAX_STEPS = 100_000
#: Memory budget of one refined solve: points x time nodes. The two-surface autocallable solvers keep both value
#: surfaces over the whole time grid (16 bytes per cell) and build per-step coefficient sets alongside; the measured
#: route peak is ~25 bytes per cell, so 5e7 cells is ~1.2 GiB.
INTRADAY_PDE_MAX_GRID_CELLS = 50_000_000


@dataclass(frozen=True)
class ResolutionStatus:
    layer_log_width: float          # sqrt(W(0, tau_first)): tau_first = first remaining event (or maturity)
    cell_log_width: Optional[float]  # achieved dx at the spot (PDE)
    cells_per_layer: float
    status: str                     # "resolved" | "unqualified" | "deterministic"
    required_points: Optional[int]  # points for REQUIRED_CELLS_PER_LAYER over the current domain; None if deterministic
    reason: str
    #: layer^2 / max per-step variance over the steps up to the first event (inf when no time grid was given)
    steps_per_layer: float = inf
    #: nepers of grid-scale mode decay over those steps (inf when no time grid was given)
    grid_mode_damping: float = inf


def first_event_tau(ctx) -> float:
    taus = [t for t in ctx.numerical.event_taus.values() if t > 0.0] or [ctx.numerical.maturity_tau]
    return min(taus)


def diffusion_layer(ctx) -> float:
    """Remaining log-spot standard deviation to the first remaining positive-time event."""
    t_first = first_event_tau(ctx)
    if t_first <= 0.0:
        return 0.0
    strike = float(getattr(ctx.numerical.product, "strike", ctx.spot))
    return sqrt(max(float(ctx.pricing_env.vol_surface.total_variance(strike, t_first, ctx.spot)), 0.0))


def _step_variances(ctx, time_nodes: Sequence[float]) -> np.ndarray:
    """Variance of each step up to (and across) the first event; step k spans [t_k, t_k+1]."""
    t = np.asarray(time_nodes, dtype=float)
    k = min(int(np.searchsorted(t, first_event_tau(ctx))), len(t) - 1)
    strike = float(getattr(ctx.numerical.product, "strike", ctx.spot))
    surface, spot = ctx.pricing_env.vol_surface, ctx.spot
    w = np.array([float(surface.total_variance(strike, float(x), spot)) if x > 0.0 else 0.0 for x in t[: k + 1]])
    return np.diff(w)


def variance_steps_per_layer(ctx, time_nodes: Sequence[float]) -> float:
    """layer^2 / the largest variance one time step carries up to (and across) the first event."""
    layer = diffusion_layer(ctx)
    if layer == 0.0:
        return inf
    dw = _step_variances(ctx, time_nodes)
    largest = float(np.max(dw)) if len(dw) else 0.0
    return layer * layer / largest if largest > 0.0 else inf


def grid_mode_damping(ctx, time_nodes: Sequence[float], theta: Sequence[float], dx_min: float) -> float:
    """Nepers by which the theta-scheme damps the grid-scale mode over the steps up to the first event."""
    dw = _step_variances(ctx, time_nodes)
    th = np.asarray(theta, dtype=float)[: len(dw)]
    mu = dw / (dx_min * dx_min)
    factor = np.abs(1.0 - 2.0 * (1.0 - th) * mu) / (1.0 + 2.0 * th * mu)
    if np.any(factor == 0.0):
        return inf
    return float(-np.sum(np.log(factor)))


def pde_resolution(ctx, *, dx_at_spot: float, domain_log_width: float, time_nodes: Optional[Sequence[float]] = None,
                   theta: Optional[Sequence[float]] = None, dx_min: Optional[float] = None) -> ResolutionStatus:
    """``time_nodes``/``theta``/``dx_min`` (the solved layout, its per-step theta and smallest cell) add the time floors."""
    layer = diffusion_layer(ctx)
    if layer == 0.0:
        return ResolutionStatus(0.0, dx_at_spot, inf, "deterministic", None,
                                "the first interval carries no variance: exact shift, no diffusion layer")
    cells = layer / dx_at_spot
    required = int(ceil(domain_log_width / (layer / REQUIRED_CELLS_PER_LAYER))) + 1
    steps = variance_steps_per_layer(ctx, time_nodes) if time_nodes is not None else inf
    damping = grid_mode_damping(ctx, time_nodes, theta, dx_min) if theta is not None and time_nodes is not None else inf
    reasons = []
    if cells < REQUIRED_CELLS_PER_LAYER:
        reasons.append(f"diffusion layer {layer:.3e} spans {cells:.2f} cells (< {REQUIRED_CELLS_PER_LAYER:g}); "
                       f"{required} points needed")
    if steps < REQUIRED_STEPS_PER_LAYER:
        reasons.append(f"one time step carries 1/{steps:.2f} of the layer variance (< 1/{REQUIRED_STEPS_PER_LAYER:g}); "
                       "more steps per day needed")
    if damping < REQUIRED_GRID_MODE_DAMPING:
        reasons.append(f"the grid-scale mode of the event jump decays by {damping:.2f} nepers before the valuation "
                       f"(< {REQUIRED_GRID_MODE_DAMPING:g}); more steps per day needed")
    status = "unqualified" if reasons else "resolved"
    return ResolutionStatus(layer, dx_at_spot, cells, status, required, "; ".join(reasons), steps, damping)


def time_resolved(status: ResolutionStatus) -> bool:
    return status.steps_per_layer >= REQUIRED_STEPS_PER_LAYER and status.grid_mode_damping >= REQUIRED_GRID_MODE_DAMPING
