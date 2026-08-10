"""Shared GBM/BSM path-build tail, with an optional Numba backend.

Every BSM 1D engine builds paths through ``GBMPathGenerator``, whose tail
(exp of drift+diffusion, then cumprod) was ~70% of a European MC pricing and
allocates four ``(n_paths, n_steps)`` temporaries (measured 2026-08-10,
``docs/mc1d-perf/prof_baseline.py``). The fused per-path Numba loop performs
the SAME operations in the SAME order: ``c_k = c_{k-1} * exp(drift_dt[k] +
vol[k] * dW[p, k])`` is exactly ``np.cumprod``'s left-to-right fold, and the
``s0`` multiply is applied AFTER the fold -- never seeded into the accumulator,
because floating-point multiplication is not associative.

Numba is an OPTIONAL accelerator, exactly like the compiled Thomas kernel and
the QE variance kernel: with it absent the NumPy reference runs and behaviour
is unchanged. The Numba path is bit-identical -- which is asserted, not
assumed, because NumPy 2.x dispatches array transcendentals to SIMD
implementations that need not agree with the scalar libm calls a per-element
loop makes. Measured on this host they do agree exactly, and
``test_gbm_path_kernel.py`` pins that.

Measured speedup of the Numba path over NumPy (2026-08-10, arm64):
1.37x at 8192 paths, 1.61-1.64x at 100k-200k paths (252 steps), giving 1.33x
on a full European MC pricing (``docs/mc1d-perf/demo_gbm_numba_fusion.py``).
"""

from __future__ import annotations

import numpy as np


def gbm_path_tail_numpy(
    dW: np.ndarray,
    drift_dt: np.ndarray,
    vol_vec: np.ndarray,
    s0: float,
) -> np.ndarray:
    """Reference implementation: the historical generate_paths tail, verbatim.

    Args:
        dW: Brownian increments, shape ``(n_paths, n_steps)``.
        drift_dt: per-step ``(mu - sigma^2/2) * dt``, shape ``(n_steps,)``.
        vol_vec: per-step volatility, shape ``(n_steps,)``.
        s0: initial value.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == s0``.
    """
    paths = np.zeros((dW.shape[0], dW.shape[1] + 1), dtype=float)
    paths[:, 0] = s0
    exp_term = np.exp(drift_dt.reshape(1, -1) + vol_vec.reshape(1, -1) * dW)
    paths[:, 1:] = s0 * np.cumprod(exp_term, axis=1)
    return paths


def _build_numba_kernel():
    """Compile the fused kernel, or return None when Numba is unavailable."""
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    # fastmath stays OFF: it licenses reassociation, which would break
    # bit-identity with the NumPy reference.
    @njit(cache=True, fastmath=False)
    def _kernel(dW, drift_dt, vol_vec, s0, out):  # pragma: no cover - via dispatcher
        n_paths, n_steps = dW.shape
        for p in range(n_paths):
            c = 1.0
            for k in range(n_steps):
                c = c * np.exp(drift_dt[k] + vol_vec[k] * dW[p, k])
                out[p, k + 1] = s0 * c

    return _kernel


_NUMBA_KERNEL = _build_numba_kernel()


def gbm_backend() -> str:
    """``"numba"`` when the accelerator compiled, else ``"numpy"``."""
    return "numba" if _NUMBA_KERNEL is not None else "numpy"


def _gbm_path_tail_numba(dW, drift_dt, vol_vec, s0) -> np.ndarray:
    dW = np.ascontiguousarray(dW, dtype=float)
    drift_dt = np.ascontiguousarray(drift_dt, dtype=float)
    vol_vec = np.ascontiguousarray(vol_vec, dtype=float)
    paths = np.empty((dW.shape[0], dW.shape[1] + 1), dtype=float)
    paths[:, 0] = s0
    _NUMBA_KERNEL(dW, drift_dt, vol_vec, float(s0), paths)
    return paths


def gbm_path_tail(dW, drift_dt, vol_vec, s0) -> np.ndarray:
    """Build GBM/BSM paths from Brownian increments.

    Uses Numba when available and NumPy otherwise; the two are bit-identical.
    """
    if _NUMBA_KERNEL is not None:
        return _gbm_path_tail_numba(dW, drift_dt, vol_vec, s0)
    return gbm_path_tail_numpy(
        np.asarray(dW, dtype=float),
        np.asarray(drift_dt, dtype=float),
        np.asarray(vol_vec, dtype=float),
        float(s0),
    )
