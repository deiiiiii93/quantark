"""SLV leverage surface and nonparametric conditional-expectation estimator.

The Heston Stochastic-Local-Volatility leverage is
    L(S, t) = sigma_LV(S, t) / sqrt(E[v_t | S_t = S]).
The conditional expectation E[v|S] is estimated nonparametrically by binning the
simulated (S, v) cloud (van der Stoep, Grzelak & Oosterlee 2014). A calibrated
LeverageSurface (node values over a (t, S) grid) is the artifact consumed by the
deterministic backward SLV PDE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError

# Shared clip band for CALIBRATION-GENERATED leverage (both the MC-binning and the
# forward-Fokker-Planck routes). Clip applies to L (not L^2). User-supplied precomputed
# LeverageSurface artifacts are consumed as-is and never re-clipped.
DEFAULT_LEVERAGE_CLIP: Tuple[float, float] = (0.05, 20.0)


class BinMethod(Enum):
    EQUIDISTANT = "equidistant"
    EQUAL_WEIGHTED = "equal_weighted"


def bin_conditional(stock_values, variance_values, num_bins, method):
    """Bin the (S, v) cloud and return (boundaries, bin_means) for E[v | S in bin]."""
    n = stock_values.size
    if n == 0:
        raise ValidationError("cannot estimate conditional expectation for empty arrays")
    order = np.argsort(stock_values)
    s_sorted = stock_values[order]
    v_sorted = variance_values[order]

    s_min, s_max = s_sorted[0], s_sorted[-1]
    boundaries = np.empty(num_bins + 1)
    boundaries[0] = s_min
    boundaries[-1] = s_max
    if method == BinMethod.EQUIDISTANT:
        for k in range(1, num_bins):
            boundaries[k] = s_min + (k / num_bins) * (s_max - s_min)
    elif method == BinMethod.EQUAL_WEIGHTED:
        for k in range(1, num_bins):
            idx = min(max(int(k * n / num_bins), 0), n - 1)
            boundaries[k] = s_sorted[idx]
    else:
        raise ValidationError(f"unknown binning method: {method}")

    # Contiguous segments on the sorted array. side="right" places a sample exactly on
    # an interior boundary into the LEFT bin, reproducing the historical mask convention
    # (bin 0 inclusive both edges [b_0, b_1]; bin k>0 half-open (b_k, b_{k+1}]). This
    # replaces the old O(num_bins * n) boolean-mask scan with a single searchsorted.
    splits = np.searchsorted(s_sorted, boundaries[1:-1], side="right")
    seg_starts = np.concatenate([[0], splits]).astype(int)
    seg_ends = np.concatenate([splits, [n]]).astype(int)
    bin_counts = seg_ends - seg_starts
    bin_means = np.zeros(num_bins)
    for k in range(num_bins):
        if bin_counts[k] > 0:
            bin_means[k] = float(np.mean(v_sorted[seg_starts[k]:seg_ends[k]]))
    global_mean = float(np.mean(v_sorted))
    for k in range(num_bins):
        if bin_counts[k] == 0:
            filled = False
            for off in range(1, num_bins):
                if k - off >= 0 and bin_counts[k - off] > 0:
                    bin_means[k] = bin_means[k - off]; filled = True; break
                if k + off < num_bins and bin_counts[k + off] > 0:
                    bin_means[k] = bin_means[k + off]; filled = True; break
            if not filled:
                bin_means[k] = global_mean
    return boundaries, bin_means


def eval_binned(query_S, boundaries, bin_means):
    """Evaluate the binned conditional expectation at arbitrary query stock values.

    Continuous piecewise-linear interpolation between bin MIDPOINTS (the Section 3.2
    scheme of van der Stoep et al.), with flat extrapolation beyond the outer midpoints.
    """
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
    return np.interp(np.asarray(query_S, dtype=float), midpoints, bin_means)


def estimate_conditional_expectation(
    stock_values: np.ndarray,
    variance_values: np.ndarray,
    num_bins: int,
    method: BinMethod = BinMethod.EQUAL_WEIGHTED,
) -> np.ndarray:
    """E[v | S] for each path via nonparametric binning of the (S, v) cloud."""
    boundaries, bin_means = bin_conditional(stock_values, variance_values, num_bins, method)
    return eval_binned(stock_values, boundaries, bin_means)


@dataclass
class LeverageSurface:
    """Calibrated SLV leverage L(S, t) on a (time x strike) node grid.

    Bilinear interpolation in (t, ln S); flat extrapolation. Construction validates
    finite, strictly positive node values and a monotone time grid.
    """

    time_grid: np.ndarray
    strike_grid: np.ndarray
    leverage_grid: np.ndarray  # shape (nT, nK)
    diagnostics: Optional[Mapping[str, Any]] = None   # FFP fills this; MC-binning leaves None

    def __post_init__(self) -> None:
        self.time_grid = np.asarray(self.time_grid, dtype=float)
        self.strike_grid = np.asarray(self.strike_grid, dtype=float)
        self.leverage_grid = np.asarray(self.leverage_grid, dtype=float)
        nT, nK = self.time_grid.size, self.strike_grid.size
        if nK < 2 or nT < 1:
            raise ValidationError("LeverageSurface needs >= 2 strikes and >= 1 time")
        if self.leverage_grid.shape != (nT, nK):
            raise ValidationError(
                f"leverage_grid shape {self.leverage_grid.shape} must equal ({nT}, {nK})"
            )
        if np.any(np.diff(self.strike_grid) <= 0):
            raise ValidationError("strike_grid must be strictly increasing")
        if nT > 1 and np.any(np.diff(self.time_grid) <= 0):
            raise ValidationError("time_grid must be strictly increasing")
        if not np.all(np.isfinite(self.leverage_grid)) or np.any(self.leverage_grid <= 0):
            raise ValidationError("leverage_grid must be finite and strictly positive")
        self._ln_k = np.log(self.strike_grid)

    def leverage(self, spot, t) -> "float | np.ndarray":
        s = np.asarray(spot, dtype=float)
        tt = np.asarray(t, dtype=float)
        s_b, t_b = np.broadcast_arrays(s, tt)
        shape = s_b.shape
        ln_s = np.log(np.clip(s_b.ravel(), self.strike_grid[0], self.strike_grid[-1]))
        K = self._ln_k
        jK = np.clip(np.searchsorted(K, ln_s, side="right"), 1, K.size - 1)
        j0, j1 = jK - 1, jK
        wK = (ln_s - K[j0]) / (K[j1] - K[j0])
        g = self.leverage_grid
        if self.time_grid.size == 1:
            vals = g[0, j0] * (1 - wK) + g[0, j1] * wK
        else:
            Tg = self.time_grid
            tc = np.clip(t_b.ravel(), Tg[0], Tg[-1])
            iT = np.clip(np.searchsorted(Tg, tc, side="right"), 1, Tg.size - 1)
            i0, i1 = iT - 1, iT
            wT = (tc - Tg[i0]) / (Tg[i1] - Tg[i0])
            bot = g[i0, j0] * (1 - wK) + g[i0, j1] * wK
            top = g[i1, j0] * (1 - wK) + g[i1, j1] * wK
            vals = bot * (1 - wT) + top * wT
        result = np.asarray(vals, dtype=float).reshape(shape)
        return result if result.shape else float(result)
