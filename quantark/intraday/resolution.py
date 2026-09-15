"""Diffusion-layer resolution status for grid routes (PDE).

Near an event the value function is a jump smoothed over the remaining standard
deviation sqrt(W) to that event. A grid resolves it only with several cells
across that layer; below that the price can still be computed, but no accuracy
claim can be attached to it. W == 0 exactly is a deterministic interval: there
is no layer at all (the route's accuracy is then a matter of barrier/spot
placement on the grid, judged by Gate C, not of diffusion resolution).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Optional

REQUIRED_CELLS_PER_LAYER = 4.0
INTRADAY_PDE_MAX_POINTS = 20_000


@dataclass(frozen=True)
class ResolutionStatus:
    layer_log_width: float          # sqrt(W(0, tau_first)): tau_first = first remaining event (or maturity)
    cell_log_width: Optional[float]  # achieved dx at the spot (PDE)
    cells_per_layer: float
    status: str                     # "resolved" | "unqualified" | "deterministic"
    required_points: Optional[int]  # points for REQUIRED_CELLS_PER_LAYER over the current domain; None if deterministic
    reason: str


def diffusion_layer(ctx) -> float:
    """Remaining log-spot standard deviation to the first remaining positive-time event."""
    taus = [t for t in ctx.numerical.event_taus.values() if t > 0.0] or [ctx.numerical.maturity_tau]
    t_first = min(taus)
    if t_first <= 0.0:
        return 0.0
    strike = float(getattr(ctx.numerical.product, "strike", ctx.spot))
    return sqrt(max(float(ctx.pricing_env.vol_surface.total_variance(strike, t_first, ctx.spot)), 0.0))


def pde_resolution(ctx, *, dx_at_spot: float, domain_log_width: float) -> ResolutionStatus:
    layer = diffusion_layer(ctx)
    if layer == 0.0:
        return ResolutionStatus(0.0, dx_at_spot, float("inf"), "deterministic", None,
                                "the first interval carries no variance: exact shift, no diffusion layer")
    cells = layer / dx_at_spot
    required = int(ceil(domain_log_width / (layer / REQUIRED_CELLS_PER_LAYER))) + 1
    if cells >= REQUIRED_CELLS_PER_LAYER:
        return ResolutionStatus(layer, dx_at_spot, cells, "resolved", required, "")
    return ResolutionStatus(layer, dx_at_spot, cells, "unqualified", required,
                            f"diffusion layer {layer:.3e} spans {cells:.2f} cells (< {REQUIRED_CELLS_PER_LAYER:g}); "
                            f"{required} points needed")
