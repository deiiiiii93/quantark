"""Boosted drop-in for quantark.util.numerical.tridiag.solve_tridiag_batch.

Same signature, same full-length convention (sub[:, 0] and sup[:, -1] ignored).
Pure SciPy — no new dependencies. Two LAPACK routes replace the Python-loop
Thomas recurrence:

  identical systems (stride-0 broadcast across axis 0, the ADI V-sweep)
      -> ONE gtsv call with n_sys right-hand sides

  distinct systems (the ADI S-sweep: coefficients vary per V-slice)
      -> block-diagonal concatenation into one tridiagonal system of size
         n_sys * N. The full-length convention's ignored entries become the
         zero seams that decouple the blocks, and partial pivoting cannot
         cross a seam because |0| never wins a pivot comparison.

Semantics vs the original:
  * numerically equivalent (~1e-16 per solve), NOT bit-identical — LAPACK
    gtsv applies partial pivoting, the unpivoted Thomas sweep does not;
  * the |pivot| < 1e-14 pre-check is replaced by LAPACK singularity handling
    plus a post-solve finiteness guard raising the same NumericalError.
    Pivoting survives some systems the unpivoted sweep would reject —
    strictly more robust, but a contract change to sign off before adoption.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, solve_banded

from quantark.util.exceptions import NumericalError

CALLS = {"identical": 0, "concat": 0}  # demo instrumentation


def solve_tridiag_batch_boosted(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                                rhs: np.ndarray) -> np.ndarray:
    """Solve n_sys independent tridiagonal systems, all inputs (n_sys, N)."""
    diag = np.asarray(diag, dtype=float)
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    rhs_arr = np.asarray(rhs, dtype=float)
    n_sys, n = diag.shape

    identical = (
        n_sys > 1
        and diag.strides[0] == 0
        and sub.strides[0] == 0
        and sup.strides[0] == 0
    )
    try:
        if identical:
            CALLS["identical"] += 1
            ab = np.zeros((3, n))
            ab[0, 1:] = sup[0, :-1]
            ab[1, :] = diag[0]
            ab[2, :-1] = sub[0, 1:]
            x = solve_banded((1, 1), ab, rhs_arr.T, check_finite=False).T
        else:
            CALLS["concat"] += 1
            # copies: the ignored seam entries must be zeroed WITHOUT mutating
            # the caller's (possibly cached) coefficient arrays
            sub_f = sub.copy()
            sub_f[:, 0] = 0.0
            sup_f = sup.copy()
            sup_f[:, -1] = 0.0
            m = n_sys * n
            ab = np.zeros((3, m))
            ab[0, 1:] = sup_f.ravel()[:-1]
            ab[1, :] = diag.ravel()
            ab[2, :-1] = sub_f.ravel()[1:]
            x = solve_banded(
                (1, 1), ab, rhs_arr.ravel(), check_finite=False
            ).reshape(n_sys, n)
    except LinAlgError as exc:
        raise NumericalError(
            f"singular system in boosted tridiagonal solve (refine grid): {exc}"
        ) from exc
    if not np.all(np.isfinite(x)):
        raise NumericalError(
            "non-finite result in boosted tridiagonal solve (refine grid)"
        )
    return np.ascontiguousarray(x)
