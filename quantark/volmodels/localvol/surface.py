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
        interp: strike-axis interpolation, "linear_s" (default, bilinear in S) or
            "linear_logs" (bilinear in log-strike, matching LeverageSurface). Time interp
            stays linear-in-vol in both modes.
    """

    strike_grid: np.ndarray
    time_grid: np.ndarray
    lv_grid: np.ndarray
    interp: str = "linear_s"

    def __post_init__(self) -> None:
        self.strike_grid = np.asarray(self.strike_grid, dtype=float)
        self.time_grid = np.asarray(self.time_grid, dtype=float)
        self.lv_grid = np.asarray(self.lv_grid, dtype=float)
        if self.interp not in ("linear_s", "linear_logs"):
            raise ValidationError("interp must be 'linear_s' or 'linear_logs'")
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

    def _strike_weights(self, s_flat: np.ndarray):
        """Bracketing strike indices and linear weights for clamped spots."""
        K = self.strike_grid
        jK = np.clip(np.searchsorted(K, s_flat, side="right"), 1, K.size - 1)
        j0, j1 = jK - 1, jK
        if self.interp == "linear_logs":
            lnK = np.log(K)
            wK = (np.log(s_flat) - lnK[j0]) / (lnK[j1] - lnK[j0])
        else:
            wK = (s_flat - K[j0]) / (K[j1] - K[j0])
        return j0, j1, wK

    def _local_vol_scalar_t(self, s: np.ndarray, tt: np.ndarray):
        """Scalar-t fast path: one time bracket, 1-D row gathers.

        Keeps the general path's arithmetic ORDER (strike interpolation per row
        first, then the time blend); blending the rows first is not bitwise.
        """
        shape = s.shape
        K = self.strike_grid
        s_flat = np.clip(s.ravel(), K[0], K[-1])
        j0, j1, wK = self._strike_weights(s_flat)

        g = self.lv_grid
        if self.time_grid.size == 1:
            row = g[0]
            vals = row[j0] * (1.0 - wK) + row[j1] * wK
        else:
            Tg = self.time_grid
            t_val = float(np.clip(tt, Tg[0], Tg[-1]))
            iT = int(np.clip(np.searchsorted(Tg, t_val, side="right"), 1, Tg.size - 1))
            i0, i1 = iT - 1, iT
            wT = (t_val - Tg[i0]) / (Tg[i1] - Tg[i0])
            g0, g1 = g[i0], g[i1]  # 1-D row views, not 2-D fancy gathers
            bottom = g0[j0] * (1.0 - wK) + g0[j1] * wK
            top = g1[j0] * (1.0 - wK) + g1[j1] * wK
            vals = bottom * (1.0 - wT) + top * wT

        result = np.asarray(vals, dtype=float).reshape(shape)
        return result if result.shape else float(result)

    def time_avg_var(self, spot: ArrayLike, t0: float, t1: float) -> "float | np.ndarray":
        """Exact time-averaged variance (1/(t1-t0)) * int_{t0}^{t1} sigma^2(spot, u) du.

        At fixed spot the bilinear surface is piecewise-linear in t with breakpoints
        exactly at ``time_grid`` (constant in the clamped extrapolation region), so
        sigma^2 is piecewise-quadratic and each segment [a, b] integrates in closed
        form: (b - a) * (sig_a^2 + sig_a*sig_b + sig_b^2) / 3. Used by MC kernels in
        ``time_sampling="integrated"`` mode; exact whenever sigma depends on t only.
        """
        t0 = float(t0)
        t1 = float(t1)
        if not (np.isfinite(t0) and np.isfinite(t1)) or not t1 > t0:
            raise ValidationError("time_avg_var requires finite t1 > t0")
        tg = self.time_grid
        inner = tg[(tg > t0) & (tg < t1)]
        pts = np.concatenate(([t0], inner, [t1]))
        sig_prev = np.asarray(self.local_vol(spot, float(pts[0])), dtype=float)
        acc = np.zeros_like(sig_prev)
        for k in range(1, pts.size):
            sig_next = np.asarray(self.local_vol(spot, float(pts[k])), dtype=float)
            w = float(pts[k] - pts[k - 1])
            acc = acc + w * (
                sig_prev * sig_prev + sig_prev * sig_next + sig_next * sig_next
            ) / 3.0
            sig_prev = sig_next
        result = acc / (t1 - t0)
        return result if result.shape else float(result)

    def local_vol(self, spot: ArrayLike, t: ArrayLike) -> "float | np.ndarray":
        """Vectorized bilinear (time, strike) interpolation with flat extrapolation.

        Gathers only the surrounding grid nodes (no per-point Python loop), so it is
        suitable for Monte Carlo path evaluation. A scalar ``t`` -- the shape Monte
        Carlo kernels call with, once per step -- takes a fast path that resolves
        the time bracket once instead of once per path.
        """
        s = np.asarray(spot, dtype=float)
        tt = np.asarray(t, dtype=float)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(tt))):
            raise ValidationError("spot and t must be finite")
        if tt.ndim == 0:
            return self._local_vol_scalar_t(s, tt)

        s_b, t_b = np.broadcast_arrays(s, tt)
        shape = s_b.shape

        K = self.strike_grid
        s_flat = np.clip(s_b.ravel(), K[0], K[-1])
        j0, j1, wK = self._strike_weights(s_flat)

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
