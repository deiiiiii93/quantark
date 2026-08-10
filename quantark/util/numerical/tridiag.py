"""Tridiagonal solvers: batched Thomas (bit-identical to the scalar sweep) and a
LAPACK-banded single-system wrapper. See the 2026-07-04 volmodels spec, WS-B2.

The batched solve is the single hottest routine in the 2D ADI march (~62% of it,
measured 2026-08-10), so it can optionally route through a small compiled kernel
(spec WS-A2). That accelerator is strictly opt-in-by-availability: if no shared
library has been built, everything falls back to the pure-NumPy sweep and the
package behaves identically. Build it with

    python -m quantark.util.numerical.build_thomas_kernel

The kernel performs the same recurrences in the same order with the same pivot
threshold, and is compiled with ``-ffp-contract=off`` so FMA contraction cannot
change the rounding sequence; results are therefore bit-identical, which
``test/test_tridiag_c_backend.py`` pins.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError, solve_banded

from quantark.util.exceptions import NumericalError

_PIVOT_MIN = 1e-14


def _library_candidates() -> "list[Path]":
    """Where a built kernel may live, most specific first."""
    here = Path(__file__).resolve().parent
    override = os.environ.get("QUANTARK_THOMAS_KERNEL")
    names = ("libthomas_kernel.dylib", "libthomas_kernel.so", "thomas_kernel.dll")
    candidates = [Path(override)] if override else []
    candidates.extend(here / name for name in names)
    return candidates


def _load_c_kernel():
    """Load the compiled kernel, or return None if it is not available.

    Failure is silent and total: a missing, stale, or unloadable library must
    leave the pure-NumPy path in charge rather than raise at import time.
    """
    for path in _library_candidates():
        try:
            if not path.exists():
                continue
            lib = ctypes.CDLL(str(path))
            lib.thomas_batch.restype = ctypes.c_int
            lib.thomas_batch.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                ctypes.c_ssize_t,
                ctypes.c_ssize_t,
                ctypes.c_double,
            ]
            return lib
        except (OSError, AttributeError):
            continue
    return None


_C_KERNEL = _load_c_kernel()


def tridiag_backend() -> str:
    """Which kernel is live: ``"c"`` when the accelerator loaded, else ``"numpy"``.

    Recorded by certification runs so evidence states what produced it.
    """
    return "c" if _C_KERNEL is not None else "numpy"


def _solve_tridiag_batch_c(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                           rhs: np.ndarray) -> np.ndarray:
    """Run the compiled Thomas kernel. Only called when it loaded."""
    sub = np.ascontiguousarray(sub, dtype=float)
    diag = np.ascontiguousarray(diag, dtype=float)
    sup = np.ascontiguousarray(sup, dtype=float)
    rhs = np.ascontiguousarray(rhs, dtype=float)
    n_sys, n = diag.shape
    x = np.empty((n_sys, n), dtype=float)
    cp = np.empty(n, dtype=float)
    dp = np.empty(n, dtype=float)
    status = _C_KERNEL.thomas_batch(
        sub, diag, sup, rhs, x, cp, dp, n_sys, n, _PIVOT_MIN
    )
    if status != 0:
        raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
    return x


def solve_tridiag_batch(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                        rhs: np.ndarray) -> np.ndarray:
    """Solve n_sys independent tridiagonal systems, all inputs shape (n_sys, N).

    Dispatches to the compiled kernel when one is available and to the NumPy
    sweep otherwise; both produce bit-identical results.
    """
    if _C_KERNEL is not None:
        return _solve_tridiag_batch_c(sub, diag, sup, rhs)
    return solve_tridiag_batch_numpy(sub, diag, sup, rhs)


def solve_tridiag_batch_numpy(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
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
