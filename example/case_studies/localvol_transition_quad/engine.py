"""Local-vol transition QUAD: backward recursion over event dates on a barrier-aligned grid.

Values are carried per lifecycle channel (snowball: 0 = never knocked in,
1 = knocked in) as point values at the cell centres, per unit notional S0.
Between consecutive event dates a kernel applies one interval transition; at an
event date the KI switch and the KO payment act on whole cells (the barriers are
cell edges). Readout is the s0 cell; native delta/gamma use the uniform stencil
around it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .contract import Contract
from .grid import GridConfig, LogGrid, build_grid
from .market import Market


def event_times(c: Contract) -> list[float]:
    return sorted(set(c.ko_times) | set(c.ki_times) | {c.maturity})


def terminal_values(c: Contract, grid: LogGrid) -> np.ndarray:
    """Maturity payoff per channel at the cell centres, BEFORE maturity events."""
    s = np.exp(grid.centres)
    if c.kind == "vanilla":
        payoff = np.maximum(s - c.strike, 0.0) if c.option == "call" else np.maximum(c.strike - s, 0.0)
        return (payoff / c.s0)[None, :]
    if c.kind == "up_out":
        return (np.maximum(s - c.strike, 0.0) / c.s0)[None, :]
    never = np.full(grid.size, 1.0 + c.rebate_rate)
    knocked = 1.0 + np.minimum(s - c.strike, 0.0) / c.s0
    return np.stack((never, knocked))


def apply_events(values: np.ndarray, c: Contract, grid: LogGrid, t: float) -> np.ndarray:
    """KI switch then KO payment at observation time ``t`` (KO wins; regions are disjoint)."""
    out = values.copy()
    if c.kind == "snowball" and t in c.ki_times:
        below = grid.below(c.ki)
        out[0, below] = out[1, below]
    if t in c.ko_times:
        above = grid.above(c.ko)
        out[:, above] = c.ko_payment(t)
    return out


def readout(values: np.ndarray, grid: LogGrid) -> dict:
    """PV, delta = dPV/dS*S0 and gamma = d2PV/dS2*S0**2 from the uniform centred stencil."""
    i = grid.spot_index
    h = grid.widths[i]
    v = values[0]
    vx = (v[i + 1] - v[i - 1]) / (2 * h)
    vxx = (v[i + 1] - 2 * v[i] + v[i - 1]) / h ** 2
    return {"pv": float(v[i]), "delta": float(vx), "gamma": float(vxx - vx)}


def reference_vol(market: Market, c: Contract) -> float:
    """Largest local vol seen at s0 across the surface's time pillars (grid width scale)."""
    surface = market.surface
    return float(max(surface.local_vol(c.s0, float(t)) for t in surface.time_grid))


@dataclass
class LocalVolQuad:
    kernel: object
    grid_config: GridConfig = field(default_factory=GridConfig)

    def grid(self, c: Contract, market: Market) -> LogGrid:
        return build_grid(c, reference_vol(market, c), self.grid_config)

    def rollback(self, c: Contract, market: Market, grid: LogGrid | None = None) -> tuple[np.ndarray, LogGrid]:
        grid = self.grid(c, market) if grid is None else grid
        times = event_times(c)
        values = apply_events(terminal_values(c, grid), c, grid, times[-1])
        for right, left in zip(times[::-1], (times[-2::-1] + [0.0])):
            values = self.kernel.propagate(values, grid, market, left, right)
            if left > 0.0:
                values = apply_events(values, c, grid, left)
        return values, grid

    def price(self, c: Contract, market: Market) -> dict:
        if not math.isclose(market.spot, c.s0, rel_tol=0, abs_tol=1e-12 * c.s0):
            raise ValueError("the grid is anchored on the contract's s0; price at market.spot == s0")
        start = perf_counter()
        values, grid = self.rollback(c, market)
        out = readout(values, grid)
        return dict(values=out, seconds=perf_counter() - start, cells=grid.size)
