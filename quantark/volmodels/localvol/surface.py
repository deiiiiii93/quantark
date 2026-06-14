"""Derived local-volatility surface (Dupire output): answers local_vol(S, t), not IV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from quantark.util.exceptions import ValidationError


@dataclass
class LocalVolSurface:
    """Local volatility sigma_LV(S, t) on a (time x strike) node grid.

    Bilinear interpolation in (t, S); flat extrapolation (clamp to edges) on both axes.

    Attributes:
        strike_grid: strictly increasing spot/strike grid (nK >= 2,).
        time_grid: strictly increasing time grid in years (nT >= 1,).
        lv_grid: local vols, shape (nT, nK), axis 0 = time, axis 1 = strike.
    """

    strike_grid: np.ndarray
    time_grid: np.ndarray
    lv_grid: np.ndarray

    def __post_init__(self) -> None:
        self.strike_grid = np.asarray(self.strike_grid, dtype=float)
        self.time_grid = np.asarray(self.time_grid, dtype=float)
        self.lv_grid = np.asarray(self.lv_grid, dtype=float)
        nT, nK = self.time_grid.size, self.strike_grid.size
        if nK < 2 or nT < 1:
            raise ValidationError("LocalVolSurface needs >= 2 strikes and >= 1 time")
        if self.lv_grid.shape != (nT, nK):
            raise ValidationError(
                f"lv_grid shape {self.lv_grid.shape} must equal (nT, nK)=({nT}, {nK})"
            )
        if np.any(np.diff(self.strike_grid) <= 0):
            raise ValidationError("strike_grid must be strictly increasing")
        if nT > 1 and np.any(np.diff(self.time_grid) <= 0):
            raise ValidationError("time_grid must be strictly increasing")
        if not np.all(np.isfinite(self.lv_grid)) or np.any(self.lv_grid <= 0):
            raise ValidationError("lv_grid must be finite and strictly positive")

    def local_vol(self, spot: ArrayLike, t: ArrayLike) -> "float | np.ndarray":
        """Bilinear (time, strike) interpolation with flat extrapolation."""
        s = np.asarray(spot, dtype=float)
        tt = np.asarray(t, dtype=float)
        s_b, t_b = np.broadcast_arrays(s, tt)
        s_c = np.clip(s_b, self.strike_grid[0], self.strike_grid[-1])
        if self.time_grid.size == 1:
            vals = np.interp(s_c.ravel(), self.strike_grid, self.lv_grid[0])
        else:
            t_c = np.clip(t_b, self.time_grid[0], self.time_grid[-1]).ravel()
            per_row = np.array(
                [np.interp(s_c.ravel(), self.strike_grid, self.lv_grid[i])
                 for i in range(self.time_grid.size)]
            )  # (nT, npts)
            vals = np.array([
                np.interp(t_c[j], self.time_grid, per_row[:, j]) for j in range(t_c.size)
            ])
        result = np.asarray(vals, dtype=float).reshape(s_b.shape)
        return result if result.shape else float(result)
