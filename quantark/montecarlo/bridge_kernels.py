"""Shared Brownian-bridge transform, with an optional Numba backend.

Profiling the otc-price-adapter desk book on 2026-08-11 put this one method at
32% of a production autocallable Monte Carlo row -- roughly twice the share of
the path build. The reason is access pattern rather than arithmetic: the
reference walks time steps in Python and each iteration reads and writes
*columns* (``W[:, l]``, ``W[:, r]``, ``W[:, k]``) of a C-contiguous
``(n_paths, n_steps)`` array, so consecutive elements of every vector
operation sit ``n_steps * 8`` bytes apart. The fused kernel walks one path at
a time, keeping the bridge recursion in registers with row-major access; the
gain therefore grows with ``n_steps`` (measured 1.30x at 252 steps / 8k paths,
2.19x at 488 steps / 100k paths).

Numba is an OPTIONAL accelerator, exactly like the compiled Thomas kernel and
the QE and GBM kernels: with it absent the NumPy reference runs and behaviour
is unchanged. Unlike those, bit-identity here is not an empirical question --
every operation involved (``+``, ``-``, ``*``, ``/``, ``sqrt``) is IEEE-754
correctly rounded, so a scalar loop and a SIMD vector op must agree exactly.
It is still asserted in ``test_bridge_transform_kernel.py`` rather than
assumed.

Two order traps are load-bearing and must survive any future edit:

1. The conditional mean keeps the grouping ``(a*W_l + b*W_r) / denom``.
   Folding ``a/denom`` into a precomputed coefficient reassociates the
   expression and is NOT bit-identical.
2. When ``left == -1`` (an interval anchored at t=0) the ``a*0.0`` term is
   still evaluated and ADDED. Dropping it looks like a no-op but flips the
   sign of a zero -- ``0.0 + -0.0`` is ``+0.0`` while ``-0.0`` alone stays
   ``-0.0`` -- which allclose cannot see and a byte comparison can.
"""

from __future__ import annotations

import numpy as np

_INTERVAL_ERROR = "Invalid Brownian bridge interval length."


def bridge_transform_numpy(
    z: np.ndarray,
    times: np.ndarray,
    indices: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Reference implementation: the historical transform body, verbatim.

    Args:
        z: standard normals, shape ``(n_paths, n_steps)``.
        times, indices, left, right, variances: the ``BrownianBridge`` fields.

    Returns:
        Brownian increments ``dW`` of shape ``(n_paths, n_steps)``.
    """
    n_paths, n_steps = z.shape

    W = np.zeros((n_paths, n_steps), dtype=float)

    # First dimension: terminal time T
    idx0 = indices[0]
    std0 = np.sqrt(variances[0])
    W[:, idx0] = std0 * z[:, 0]

    # Remaining dimensions: midpoints
    for j in range(1, n_steps):
        k = indices[j]
        l = left[j]
        r = right[j]

        t_l = 0.0 if l == -1 else times[l]
        t_r = times[r]
        t_m = times[k]

        if l == -1:
            W_l = 0.0
        else:
            W_l = W[:, l]
        W_r = W[:, r]

        denom = t_r - t_l
        if denom <= 0.0:
            raise ValueError(_INTERVAL_ERROR)
        mean = ((t_r - t_m) * W_l + (t_m - t_l) * W_r) / denom
        std = np.sqrt(variances[j])

        W[:, k] = mean + std * z[:, j]

    dW = np.empty_like(W)
    dW[:, 0] = W[:, 0]
    if n_steps > 1:
        dW[:, 1:] = W[:, 1:] - W[:, :-1]

    return dW


def _build_numba_kernel():
    """Compile the fused kernel, or return None when Numba is unavailable."""
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    # fastmath stays OFF: it licenses reassociation, which would break
    # bit-identity with the NumPy reference.
    @njit(cache=True, fastmath=False)
    def _kernel(z, idx, left, right, a, b, denom, stds, W, dW):  # pragma: no cover
        n_paths, n_steps = z.shape
        for p in range(n_paths):
            W[p, idx[0]] = stds[0] * z[p, 0]
            for j in range(1, n_steps):
                k = idx[j]
                l = left[j]
                r = right[j]
                # trap 2: the a*W_l term is evaluated even when W_l is 0.0
                W_l = 0.0 if l == -1 else W[p, l]
                W_r = W[p, r]
                # trap 1: grouping preserved exactly
                mean = (a[j] * W_l + b[j] * W_r) / denom[j]
                W[p, k] = mean + stds[j] * z[p, j]
            prev = W[p, 0]
            dW[p, 0] = prev
            for k in range(1, n_steps):
                cur = W[p, k]
                dW[p, k] = cur - prev
                prev = cur

    return _kernel


_NUMBA_KERNEL = _build_numba_kernel()


def bridge_backend() -> str:
    """``"numba"`` when the accelerator compiled, else ``"numpy"``."""
    return "numba" if _NUMBA_KERNEL is not None else "numpy"


def _step_coefficients(times, indices, left, right):
    """Per-step ``(t_r - t_m, t_m - t_l, t_r - t_l)``, as the reference forms them.

    Entry 0 is unused by the kernel (the terminal draw takes no mean) and is
    filled with neutral values so no spurious division can occur.
    """
    n_steps = times.shape[0]
    a = np.zeros(n_steps, dtype=float)
    b = np.zeros(n_steps, dtype=float)
    denom = np.ones(n_steps, dtype=float)
    if n_steps > 1:
        j = np.arange(1, n_steps)
        l = left[j]
        anchored = l == -1
        t_l = np.where(anchored, 0.0, times[np.where(anchored, 0, l)])
        t_r = times[right[j]]
        t_m = times[indices[j]]
        d = t_r - t_l
        if np.any(d <= 0.0):
            raise ValueError(_INTERVAL_ERROR)
        a[j] = t_r - t_m
        b[j] = t_m - t_l
        denom[j] = d
    return a, b, denom


def _bridge_transform_numba(z, times, indices, left, right, variances) -> np.ndarray:
    z = np.ascontiguousarray(z, dtype=float)
    times = np.ascontiguousarray(times, dtype=float)
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    left = np.ascontiguousarray(left, dtype=np.int64)
    right = np.ascontiguousarray(right, dtype=np.int64)
    a, b, denom = _step_coefficients(times, indices, left, right)
    stds = np.sqrt(np.ascontiguousarray(variances, dtype=float))

    n_paths, n_steps = z.shape
    # Every W entry is written before it is read: the bridge recursion always
    # fills an interval's endpoints before its midpoint, so no zero-fill is
    # needed (the reference's np.zeros is a 400 MB memset at desk scale).
    W = np.empty((n_paths, n_steps), dtype=float)
    dW = np.empty((n_paths, n_steps), dtype=float)
    _NUMBA_KERNEL(z, indices, left, right, a, b, denom, stds, W, dW)
    return dW


def bridge_transform(z, times, indices, left, right, variances) -> np.ndarray:
    """Map standard normals to Brownian increments via the bridge.

    Uses Numba when available and NumPy otherwise; the two are bit-identical.
    """
    if _NUMBA_KERNEL is not None:
        return _bridge_transform_numba(z, times, indices, left, right, variances)
    return bridge_transform_numpy(
        np.asarray(z, dtype=float), times, indices, left, right, variances
    )
