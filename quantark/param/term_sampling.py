"""Convert term-structure curves into piecewise-constant per-step forward rates.

Kernels step in time and need the forward rate over each interval, not a single
terminal zero rate. For a deterministic curve the forward over [t0, t1] is exact:
f = -ln(DF(t1)/DF(t0)) / (t1 - t0)  (RateCurve.get_forward_rate).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError


def _validate_grid(t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValidationError("t_grid must be a 1D array with at least 2 points")
    if not np.all(np.isfinite(t)):
        raise ValidationError("t_grid must be finite")
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
    out = np.array(
        [float(rate_curve.get_forward_rate(t[i], t[i + 1])) for i in range(t.size - 1)]
    )
    if not np.all(np.isfinite(out)):
        raise NumericalError("rate_curve produced non-finite forward rates")
    return out


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
    if not np.all(np.isfinite(out)):
        raise NumericalError("zero_yield produced non-finite forward carry")
    return out


def discount_factors_on_grid(rate_curve, t_grid: np.ndarray) -> np.ndarray:
    """Node discount factors DF(0, t_i) for each grid node (DF at t=0 is 1)."""
    t = _validate_grid(t_grid)
    out = np.array(
        [1.0 if ti <= 0.0 else float(rate_curve.get_discount_factor(ti)) for ti in t]
    )
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise NumericalError("rate_curve produced invalid discount factors")
    return out


def step_vols_on_grid(
    get_vol: Callable[[float, float], float], ref_strike: float, t_grid: np.ndarray
) -> np.ndarray:
    """Per-interval vols from total-variance differencing at a reference strike.

    w(t) = get_vol(ref_strike, t)^2 * t;  step vol = sqrt((w1 - w0) / dt).
    Raises NumericalError if total variance decreases beyond tolerance
    (calendar arbitrage in the input surface).
    """
    t = _validate_grid(t_grid)
    w = np.empty(t.size, dtype=float)
    for i, ti in enumerate(t):
        if ti <= 0.0:
            w[i] = 0.0
        else:
            v = float(get_vol(float(ref_strike), float(ti)))
            if not np.isfinite(v) or v <= 0.0:
                raise NumericalError(
                    f"get_vol returned invalid vol {v} at t={ti} "
                    "(must be finite and strictly positive)"
                )
            w[i] = v * v * ti
    dw = np.diff(w)
    if np.any(dw < -1e-12):
        raise NumericalError(
            "total variance is decreasing on the grid (calendar arbitrage)"
        )
    out = np.sqrt(np.maximum(dw, 0.0) / np.diff(t))
    if not np.all(np.isfinite(out)):
        raise NumericalError("get_vol produced non-finite step vols")
    return out
