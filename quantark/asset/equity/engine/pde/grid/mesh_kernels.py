"""Arithmetic hosts for the legacy concentrated mesh's RK4 integration (spec 3A §5.1).

``space._legacy_concentrated_mesh`` evaluates f(y) = A / sqrt(sum_i 1 / (beta^2 + (y - c_i)^2))
about 460,000 times per layout; through NumPy on a 1-8 element array nearly all of it is call
overhead (82% to 98% of a cold solve, spec §4.9). The kernels do the same IEEE operations in the
same order:

- the square is ``d * d`` (what NumPy evaluates for ``** 2``);
- the sum is left to right for fewer than eight terms, and for exactly eight (the builder's
  maximum, ``space._LEGACY_MAX_CRITICALS``) ``((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))``,
  NumPy's unrolled pairwise block. The order was measured, not assumed (evidence ``e10``).

A kernel never handles an unusual value: a non-finite or zero denominator, a non-finite sum or
node, or more than eight terms raises ``Deferred``, and ``space._kernel_integrate`` re-runs the
call on the NumPy reference under the caller's ``np.errstate``. Every exception and the fallback to
``_stable_concentrated_mesh`` therefore happen exactly as in 0.7.0.

Numba is the optional ``accel`` extra, as in ``quantark.montecarlo.qe_kernels``: absent, the Python
kernel runs. ``fastmath=False`` keeps the compiled loop free of FMA contraction and reassociation.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["Deferred", "KERNELS", "backend", "integrate", "integrate_python"]

_MAX_TERMS = 8


class Deferred(Exception):
    """The kernel met a value it does not handle: run the NumPy reference for this call."""


def integrate_python(y_min, n, a, beta, crits) -> np.ndarray:
    """RK4 for dy/dxi = a * g(y) on ``n`` steps of 1/n, on Python floats."""
    cs = [float(c) for c in crits]
    m = len(cs)
    if m == 0 or m > _MAX_TERMS:
        raise Deferred
    y_min, a, beta, n = float(y_min), float(a), float(beta), int(n)
    h = 1.0 / n
    half_h = 0.5 * h
    sixth_h = h / 6.0
    bb = beta * beta
    inf = math.inf
    sqrt = math.sqrt

    def f(y):
        if m == 8:
            r = [0.0] * 8
            for j in range(8):
                d = y - cs[j]
                q = bb + d * d
                if not 0.0 < q < inf:
                    raise Deferred
                r[j] = 1.0 / q
            s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        else:
            d = y - cs[0]
            q = bb + d * d
            if not 0.0 < q < inf:
                raise Deferred
            s = 1.0 / q
            for j in range(1, m):
                d = y - cs[j]
                q = bb + d * d
                if not 0.0 < q < inf:
                    raise Deferred
                s += 1.0 / q
        if not 0.0 < s < inf:
            raise Deferred
        return a / sqrt(s)

    out = [y_min]
    y = y_min
    for _ in range(n):
        k1 = f(y)
        k2 = f(y + half_h * k1)
        k3 = f(y + half_h * k2)
        k4 = f(y + h * k3)
        y = y + sixth_h * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(y)
    if not math.isfinite(y):
        raise Deferred
    return np.array(out, dtype=float)


def _build_numba():
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    @njit(cache=True, fastmath=False)
    def _kernel(y_min, n, a, beta, crits, out):          # 1 = done, 0 = deferred
        h = 1.0 / n
        half_h = 0.5 * h
        sixth_h = h / 6.0
        bb = beta * beta
        m = crits.shape[0]
        y = y_min
        out[0] = y_min
        ks = np.empty(4)
        r8 = np.empty(8)
        for i in range(n):
            yy = y
            for stage in range(4):
                if m == 8:
                    for j in range(8):
                        d = yy - crits[j]
                        q = bb + d * d
                        if not (q > 0.0 and q < np.inf):
                            return 0
                        r8[j] = 1.0 / q
                    s = ((r8[0] + r8[1]) + (r8[2] + r8[3])) + ((r8[4] + r8[5]) + (r8[6] + r8[7]))
                else:
                    d = yy - crits[0]
                    q = bb + d * d
                    if not (q > 0.0 and q < np.inf):
                        return 0
                    s = 1.0 / q
                    for j in range(1, m):
                        d = yy - crits[j]
                        q = bb + d * d
                        if not (q > 0.0 and q < np.inf):
                            return 0
                        s += 1.0 / q
                if not (s > 0.0 and s < np.inf):
                    return 0
                k = a / np.sqrt(s)
                ks[stage] = k
                if stage < 2:
                    yy = y + half_h * k
                elif stage == 2:
                    yy = y + h * k
            y = y + sixth_h * (ks[0] + 2.0 * ks[1] + 2.0 * ks[2] + ks[3])
            out[i + 1] = y
        if not np.isfinite(y):
            return 0
        return 1

    def integrate_numba(y_min, n, a, beta, crits) -> np.ndarray:
        c = np.ascontiguousarray(crits, dtype=np.float64)
        if c.shape[0] == 0 or c.shape[0] > _MAX_TERMS:
            raise Deferred
        out = np.empty(int(n) + 1)
        if not _kernel(float(y_min), int(n), float(a), float(beta), c, out):
            raise Deferred
        return out

    return integrate_numba


KERNELS = {"python": integrate_python}
_numba = _build_numba()
if _numba is not None:
    KERNELS["numba"] = _numba


def backend() -> str:
    """``"numba"`` when the accelerator compiled, else ``"python"``."""
    return "numba" if "numba" in KERNELS else "python"


#: The kernel ``space._kernel_integrate`` calls (a module attribute: tests select a kernel by patching it).
integrate = KERNELS[backend()]
