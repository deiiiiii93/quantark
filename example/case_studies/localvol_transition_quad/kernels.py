"""Interval transition operators on the finite-volume log grid.

A kernel maps point values at the cell centres at ``t1`` to discounted point
values at ``t0``. Row i is the law of ln S_{t1} given ln S_{t0} = c_i, as a
probability per cell. Rounding each landing point to its cell centre adds the
cell's quantization variance w**2/12 per interval, so Gaussian rows are moment
matched (variance net of it). The end cells are semi-infinite in probability:
mass is conserved exactly and the out-of-domain region inherits the edge value.
"""
from __future__ import annotations

import math
from enum import Enum

import numpy as np
from scipy.special import ndtr

from .grid import LogGrid, UnresolvedGrid
from .market import Market


class KernelKind(Enum):
    BSM_EXACT = "bsm_exact"          # constant-vol control (tests only)


def _cell_probabilities(grid: LogGrid, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    cdf = ndtr((grid.edges[None, 1:-1] - mean[:, None]) / sd[:, None])
    out = np.empty((mean.size, grid.size))
    out[:, 0] = cdf[:, 0]
    out[:, 1:-1] = np.diff(cdf, axis=1)
    out[:, -1] = 1.0 - cdf[:, -1]
    return out


MOMENT_PASSES = 2


def gaussian_rows(grid: LogGrid, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Cell probabilities of a Gaussian per source row, moment matched on the CENTRES.

    Landing points are rounded to cell centres, which adds roughly w**2/12 of
    variance and, where widths change, a mean bias. The Gaussian's mean and
    variance are corrected so the realized law on the centres has the target
    first two moments (``MOMENT_PASSES`` fixed-point passes, from the source width).
    """
    c = grid.centres
    mu = np.array(mean, dtype=float)
    s2 = var - grid.widths[np.clip(np.searchsorted(grid.edges, mean) - 1, 0, grid.size - 1)] ** 2 / 12.0
    for _ in range(MOMENT_PASSES + 1):
        if np.any(s2 <= 0):
            raise UnresolvedGrid("interval variance below the cell quantization variance; refine the grid")
        p = _cell_probabilities(grid, mu, np.sqrt(s2))
        if _ == MOMENT_PASSES:
            return p
        m1 = p @ c
        mu = mu + (mean - m1)
        s2 = s2 + (var - (p @ (c * c) - m1 * m1))
    return p


class BSMExactKernel:
    """Exact Black-Scholes transition for a constant local-vol surface (validation control)."""

    kind = KernelKind.BSM_EXACT

    def __init__(self):
        self.diagnostics: list[dict] = []
        self._cache: dict = {}

    @staticmethod
    def _sigma(market: Market) -> float:
        grid = market.surface.lv_grid
        if np.ptp(grid) != 0.0:
            raise ValueError("BSM_EXACT requires a constant local-vol surface")
        return float(grid.flat[0])

    def matrix(self, grid: LogGrid, market: Market, t0: float, t1: float) -> np.ndarray:
        h = t1 - t0
        key = (id(grid), h, market.r, market.q, self._sigma(market))
        cached = self._cache.get(key)
        if cached is None:
            sigma = key[-1]
            mean = grid.centres + (market.r - market.q - 0.5 * sigma ** 2) * h
            cached = gaussian_rows(grid, mean, np.full(grid.size, sigma ** 2 * h))
            self._cache = {key: cached}
        return cached

    def propagate(self, values: np.ndarray, grid: LogGrid, market: Market, t0: float, t1: float) -> np.ndarray:
        p = self.matrix(grid, market, t0, t1)
        return math.exp(-market.r * (t1 - t0)) * (values @ p.T)
