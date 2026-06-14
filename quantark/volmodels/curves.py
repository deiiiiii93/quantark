"""Convert term-structure curves into piecewise-constant per-step forward rates.

Kernels step in time and need the forward rate over each interval, not a single
terminal zero rate. For a deterministic curve the forward over [t0, t1] is exact:
f = -ln(DF(t1)/DF(t0)) / (t1 - t0)  (RateCurve.get_forward_rate).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from quantark.util.exceptions import ValidationError


def _validate_grid(t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValidationError("t_grid must be a 1D array with at least 2 points")
    if np.any(np.diff(t) <= 0):
        raise ValidationError("t_grid must be strictly increasing")
    if t[0] < 0:
        raise ValidationError("t_grid must start at a non-negative time")
    return t


def forward_rates_on_grid(rate_curve, t_grid: np.ndarray) -> np.ndarray:
    """Piecewise-constant forward rate over each [t_grid[i], t_grid[i+1]] interval.

    Uses RateCurve.get_forward_rate (DF-based, exact). t0=0 is valid:
    get_forward_rate(0, t1) = -ln(DF(t1))/t1 = the zero rate to t1.
    """
    t = _validate_grid(t_grid)
    return np.array(
        [float(rate_curve.get_forward_rate(t[i], t[i + 1])) for i in range(t.size - 1)]
    )


def forward_carry_on_grid(
    zero_yield: Callable[[float], float], t_grid: np.ndarray
) -> np.ndarray:
    """Piecewise-constant forward carry from a zero-yield term structure q(T).

    Forward carry over [t0, t1] = (q(t1) t1 - q(t0) t0) / (t1 - t0). ``zero_yield``
    maps maturity (years) to continuously-compounded zero yield (e.g. env.get_div_yield).
    """
    t = _validate_grid(t_grid)
    out = np.empty(t.size - 1, dtype=float)
    for i in range(t.size - 1):
        t0, t1 = t[i], t[i + 1]
        w0 = 0.0 if t0 <= 0.0 else float(zero_yield(t0)) * t0
        out[i] = (float(zero_yield(t1)) * t1 - w0) / (t1 - t0)
    return out
