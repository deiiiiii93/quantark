"""Tridiagonal solvers: batched Thomas (bit-identical to the scalar sweep) and a
LAPACK-banded single-system wrapper. See the 2026-07-04 volmodels spec, WS-B2."""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, solve_banded

from quantark.util.exceptions import NumericalError

_PIVOT_MIN = 1e-14


def solve_tridiag_batch(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                        rhs: np.ndarray) -> np.ndarray:
    """Solve n_sys independent tridiagonal systems, all inputs shape (n_sys, N).

    Full-length convention: sub[:, 0] and sup[:, -1] are ignored. The sweep is the
    sequential Thomas recurrence over N vectorized across systems, so each system's
    result is bit-identical to a scalar Thomas solve with the same convention.
    Raises NumericalError if any pivot magnitude falls below 1e-14.
    """
    diag = np.asarray(diag, dtype=float)
    n_sys, n = diag.shape
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    cp = np.empty((n_sys, n))
    dp = np.empty((n_sys, n))
    # The pivot guard is checked ONCE after the sweep rather than once per row.
    # Every denominator is retained, so the first row whose magnitude collapses
    # is still detected before any result is returned -- and it is detected on
    # the value that caused the collapse, not on the nan it later propagates.
    # Profiling on 2026-08-10 attributed ~62% of a PDE march to this function,
    # with ~87k Python-level ufunc reductions coming from the in-sweep check.
    # Division by a collapsed pivot is allowed to produce inf/nan here, so the
    # sweep runs under suppressed floating-point warnings; the guard below
    # rejects the solve before a caller can observe it.
    denominators = np.empty((n_sys, n))
    denominators[:, 0] = diag[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cp[:, 0] = sup[:, 0] / diag[:, 0]
        dp[:, 0] = rhs[:, 0] / diag[:, 0]
        for i in range(1, n):
            denom = diag[:, i] - sub[:, i] * cp[:, i - 1]
            denominators[:, i] = denom
            cp[:, i] = sup[:, i] / denom
            dp[:, i] = (rhs[:, i] - sub[:, i] * dp[:, i - 1]) / denom
    if np.any(np.abs(denominators) < _PIVOT_MIN):
        raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
    x = np.empty((n_sys, n))
    x[:, n - 1] = dp[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def solve_tridiag(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                  rhs: np.ndarray) -> np.ndarray:
    """Solve one tridiagonal system via LAPACK (scipy solve_banded).

    Off-diagonals have length N-1. Arithmetic order differs from the Thomas sweep, so
    results are numerically equivalent (not bit-identical) to a scalar Thomas solve.
    Raises NumericalError on a singular system.
    """
    diag = np.asarray(diag, dtype=float)
    n = diag.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = np.asarray(sup, dtype=float)
    ab[1, :] = diag
    ab[2, :-1] = np.asarray(sub, dtype=float)
    try:
        return solve_banded((1, 1), ab, np.asarray(rhs, dtype=float), check_finite=False)
    except LinAlgError as exc:
        raise NumericalError(f"singular tridiagonal system: {exc}") from exc
