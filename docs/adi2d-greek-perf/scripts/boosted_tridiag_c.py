"""ctypes wrapper for the compiled Thomas kernel — the bitwise-exact boost.

Drop-in for solve_tridiag_batch: same signature, same convention, the same
1e-14 pivot guard raising the same NumericalError with the same message.

Uses the TRANSPOSED kernel variants: data is laid out (n, n_sys) so the inner
loops run across independent systems — clang vectorizes them (NEON fdiv.2d)
and the division latency pipelines instead of serializing per system. The
per-system operation order is unchanged, so results stay bit-identical to
the NumPy Thomas sweep.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from quantark.util.exceptions import NumericalError

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = ctypes.CDLL(os.path.join(_HERE, "thomas_kernel.dylib"))

_PTR = ctypes.POINTER(ctypes.c_double)
for _fn in (_LIB.thomas_batch, _LIB.thomas_multi_rhs,
            _LIB.thomas_batch_t, _LIB.thomas_multi_rhs_t):
    _fn.restype = ctypes.c_int
    _fn.argtypes = [_PTR] * 7 + [ctypes.c_ssize_t] * 2 + [ctypes.c_double]

_PIVOT_MIN = 1e-14
CALLS = {"multi_rhs": 0, "batch": 0}


def _p(arr: np.ndarray):
    return arr.ctypes.data_as(_PTR)


def solve_tridiag_batch_c(sub, diag, sup, rhs) -> np.ndarray:
    diag = np.asarray(diag, dtype=float)
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    n_sys, n = diag.shape
    rhsT = np.ascontiguousarray(np.asarray(rhs, dtype=float).T)   # (n, n_sys)
    xT = np.empty((n, n_sys))

    shared = (
        n_sys > 1
        and diag.strides[0] == 0
        and sub.strides[0] == 0
        and sup.strides[0] == 0
    )
    if shared:
        CALLS["multi_rhs"] += 1
        a1 = np.ascontiguousarray(sub[0])
        b1 = np.ascontiguousarray(diag[0])
        c1 = np.ascontiguousarray(sup[0])
        cp = np.empty(n)
        denoms = np.empty(n)
        ret = _LIB.thomas_multi_rhs_t(
            _p(a1), _p(b1), _p(c1), _p(rhsT), _p(xT), _p(cp), _p(denoms),
            n_sys, n, _PIVOT_MIN,
        )
    else:
        CALLS["batch"] += 1
        subT = np.ascontiguousarray(sub.T)
        diagT = np.ascontiguousarray(diag.T)
        supT = np.ascontiguousarray(sup.T)
        cpT = np.empty((n, n_sys))
        dpT = np.empty((n, n_sys))
        ret = _LIB.thomas_batch_t(
            _p(subT), _p(diagT), _p(supT), _p(rhsT), _p(xT), _p(cpT), _p(dpT),
            n_sys, n, _PIVOT_MIN,
        )
    if ret != 0:
        raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
    return xT.T
