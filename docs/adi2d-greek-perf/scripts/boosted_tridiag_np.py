"""Pure-NumPy hoisted-guard variant — the zero-toolchain, zero-dependency boost.

Identical arithmetic to the stock Thomas sweep (bit-identical results); the
only change is that the per-iteration np.any pivot checks (2.35M calls per
march) are hoisted: denominators are stored and checked ONCE after the sweep.
A failing system still raises the same NumericalError with the same message —
it just raises after the arithmetic instead of before it.
"""
from __future__ import annotations

import numpy as np

from quantark.util.exceptions import NumericalError

_PIVOT_MIN = 1e-14


def solve_tridiag_batch_hoisted(sub, diag, sup, rhs) -> np.ndarray:
    diag = np.asarray(diag, dtype=float)
    n_sys, n = diag.shape
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    cp = np.empty((n_sys, n))
    dp = np.empty((n_sys, n))
    denoms = np.empty((n_sys, n))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denoms[:, 0] = diag[:, 0]
        cp[:, 0] = sup[:, 0] / diag[:, 0]
        dp[:, 0] = rhs[:, 0] / diag[:, 0]
        for i in range(1, n):
            denom = diag[:, i] - sub[:, i] * cp[:, i - 1]
            denoms[:, i] = denom
            cp[:, i] = sup[:, i] / denom
            dp[:, i] = (rhs[:, i] - sub[:, i] * dp[:, i - 1]) / denom
        x = np.empty((n_sys, n))
        x[:, n - 1] = dp[:, n - 1]
        for i in range(n - 2, -1, -1):
            x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    if np.any(np.abs(denoms) < _PIVOT_MIN):
        raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
    return x
