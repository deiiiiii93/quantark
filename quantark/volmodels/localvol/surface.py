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
        if not np.all(np.isfinite(self.strike_grid)) or np.any(self.strike_grid <= 0):
            raise ValidationError("strike_grid must be finite and positive")
        if not np.all(np.isfinite(self.time_grid)) or np.any(self.time_grid < 0):
            raise ValidationError("time_grid must be finite and non-negative")
        if np.any(np.diff(self.strike_grid) <= 0):
            raise ValidationError("strike_grid must be strictly increasing")
        if nT > 1 and np.any(np.diff(self.time_grid) <= 0):
            raise ValidationError("time_grid must be strictly increasing")
        if not np.all(np.isfinite(self.lv_grid)) or np.any(self.lv_grid <= 0):
            raise ValidationError("lv_grid must be finite and strictly positive")

    def local_vol(self, spot: ArrayLike, t: ArrayLike) -> "float | np.ndarray":
        """Vectorized bilinear (time, strike) interpolation with flat extrapolation.

        Gathers only the surrounding grid nodes (no per-point Python loop), so it is
        suitable for Monte Carlo path evaluation.
        """
        s = np.asarray(spot, dtype=float)
        tt = np.asarray(t, dtype=float)
        s_b, t_b = np.broadcast_arrays(s, tt)
        shape = s_b.shape

        K = self.strike_grid
        s_flat = np.clip(s_b.ravel(), K[0], K[-1])
        jK = np.clip(np.searchsorted(K, s_flat, side="right"), 1, K.size - 1)
        j0, j1 = jK - 1, jK
        wK = (s_flat - K[j0]) / (K[j1] - K[j0])

        g = self.lv_grid
        if self.time_grid.size == 1:
            row = g[0]
            vals = row[j0] * (1.0 - wK) + row[j1] * wK
        else:
            Tg = self.time_grid
            t_flat = np.clip(t_b.ravel(), Tg[0], Tg[-1])
            iT = np.clip(np.searchsorted(Tg, t_flat, side="right"), 1, Tg.size - 1)
            i0, i1 = iT - 1, iT
            wT = (t_flat - Tg[i0]) / (Tg[i1] - Tg[i0])
            bottom = g[i0, j0] * (1.0 - wK) + g[i0, j1] * wK
            top = g[i1, j0] * (1.0 - wK) + g[i1, j1] * wK
            vals = bottom * (1.0 - wT) + top * wT

        result = np.asarray(vals, dtype=float).reshape(shape)
        return result if result.shape else float(result)
