"""Barrier-aligned finite-volume grid in x = ln S (spec D7b).

States are cell centres; values are point values at the centres, and a transition
row maps a centre to the probability of landing in each cell. Every contract level
(KO, KI, strike) is a cell EDGE, so a discrete event acts on whole cells and its
indicator is integrated exactly; ln s0 is a cell CENTRE, so the readout needs no
interpolation. Between those anchors the spacing is piecewise uniform.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contract import Contract


class UnresolvedGrid(ValueError):
    """A measured transition identity, band tail or grid requirement is not met."""


@dataclass(frozen=True)
class GridConfig:
    cells: int = 400                 # target cell count across the domain
    half_width_std: float = 5.0      # domain half-width in terminal standard deviations
    readout_cells: int = 3           # uniform cells required on each side of the s0 cell
    subdivide: int = 1               # odd split of every cell: nested refinement keeping edges/centres

    def __post_init__(self):
        if not isinstance(self.cells, int) or self.cells < 16:
            raise ValueError("at least 16 cells required")
        if not math.isfinite(self.half_width_std) or self.half_width_std <= 0:
            raise ValueError("positive half-width required")
        if not isinstance(self.readout_cells, int) or self.readout_cells < 2:
            raise ValueError("at least two readout cells per side required")
        if not isinstance(self.subdivide, int) or self.subdivide < 1 or self.subdivide % 2 == 0:
            raise ValueError("subdivide must be a positive odd integer")


@dataclass(frozen=True)
class LogGrid:
    edges: np.ndarray                # ln S cell edges, strictly increasing, length N+1
    spot_index: int                  # cell whose centre is ln s0

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def size(self) -> int:
        return self.edges.size - 1

    def below(self, level: float) -> np.ndarray:
        """Cells lying entirely at or below ``level`` (an edge)."""
        return self.edges[1:] <= math.log(level) + 1e-12

    def above(self, level: float) -> np.ndarray:
        """Cells lying entirely at or above ``level`` (an edge)."""
        return self.edges[:-1] >= math.log(level) - 1e-12


def contract_levels(c: Contract) -> list[float]:
    levels = [c.strike]
    if math.isfinite(c.ko):
        levels.append(c.ko)
    if c.ki > 0:
        levels.append(c.ki)
    return levels


def build_grid(c: Contract, sigma_ref: float, config: GridConfig = GridConfig()) -> LogGrid:
    """Domain ln s0 +/- half_width_std * sigma_ref * sqrt(T); anchors on edges; s0 a centre.

    The s0 cell and ``readout_cells`` cells either side share one spacing, so the
    centred readout stencil is uniform. Raises ``UnresolvedGrid`` when a contract
    level falls inside that stencil (the grid would have to straddle it).
    """
    if not math.isfinite(sigma_ref) or sigma_ref <= 0:
        raise ValueError("positive reference volatility required")
    x0 = math.log(c.s0)
    half = config.half_width_std * sigma_ref * math.sqrt(c.maturity)
    lo, hi = x0 - half, x0 + half
    h = (hi - lo) / config.cells
    anchors = [math.log(level) for level in contract_levels(c)]
    if any(not lo < a < hi for a in anchors):
        raise UnresolvedGrid("a contract level lies outside the computational domain; widen half_width_std")
    k = config.readout_cells
    # A payoff KINK (the strike) costs only O(w**2) wherever it sits under the
    # midpoint rule, so a strike inside the readout block is not anchored (an ATM
    # strike sits on the s0 centre). Discontinuities (KO, KI) always are edges.
    stencil_lo, stencil_hi = x0 - (k + 0.5) * h, x0 + (k + 0.5) * h
    strike = math.log(c.strike)
    if stencil_lo - 1e-12 < strike < stencil_hi + 1e-12:
        anchors.remove(strike)
    if any(stencil_lo - 1e-12 < a < stencil_hi + 1e-12 for a in anchors):
        nearest = min(abs(a - x0) for a in anchors)
        # shrink the stencil spacing so the nearest level sits beyond it
        h_fit = nearest / (k + 0.5 + 1)
        if h_fit <= 0:
            raise UnresolvedGrid("a contract level coincides with s0")
        h = min(h, h_fit)
        stencil_lo, stencil_hi = x0 - (k + 0.5) * h, x0 + (k + 0.5) * h
    inner = x0 + h * (np.arange(-k, k + 2) - 0.5)            # 2k+2 edges of the readout block
    target = (hi - lo) / config.cells
    # Each barrier gets a symmetric pair of equal cells: a Markov chain's effective
    # threshold is the midpoint BETWEEN neighbouring centres, which is the barrier
    # only when the two cells either side of it have the same width.
    blocks = [inner]
    for a in anchors:
        # halve against other barriers and the domain ends; the readout block may abut the pair
        half_width = min([target] + [abs(a - b) / 2 for b in anchors + [lo, hi] if b != a])
        if a > stencil_hi:
            half_width = min(half_width, a - stencil_hi)
        if a < stencil_lo:
            half_width = min(half_width, stencil_lo - a)
        blocks.append(np.array([a - half_width, a, a + half_width]))
    blocks.sort(key=lambda b: b[0])

    def fill(a, b):
        n = max(1, math.ceil((b - a) / target - 1e-9))
        return list(a + (b - a) * np.arange(1, n) / n)

    edges_list, cursor = [lo], lo
    for block in blocks:
        if block[0] < cursor - 1e-12:
            raise UnresolvedGrid("overlapping grid blocks; contract levels too close for this grid")
        if block[0] > cursor + 1e-12:
            edges_list += fill(cursor, block[0]) + [block[0]]
        edges_list += list(block[1:])
        cursor = block[-1]
    if hi > cursor + 1e-12:
        edges_list += fill(cursor, hi) + [hi]
    edges = np.array(edges_list)
    if np.any(np.diff(edges) <= 0):
        raise UnresolvedGrid("non-increasing grid edges")
    spot_index = int(np.argmin(np.abs(edges - inner[0]))) + k
    m = config.subdivide
    if m > 1:
        frac = np.arange(m) / m
        edges = np.append((edges[:-1, None] + np.diff(edges)[:, None] * frac[None, :]).ravel(), edges[-1])
        spot_index = spot_index * m + m // 2
    grid = LogGrid(edges, spot_index)
    assert abs(grid.centres[spot_index] - x0) < 1e-12
    return grid
