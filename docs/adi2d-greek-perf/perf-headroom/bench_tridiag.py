"""Micro-bench: hoisted-guard Thomas vs the shipped per-row np.any guard."""
import time
import numpy as np
from quantark.util.numerical.tridiag import solve_tridiag_batch, _PIVOT_MIN
from quantark.util.exceptions import NumericalError


def hoisted(sub, diag, sup, rhs):
    """Same arithmetic, same order; guard checked once after the sweep."""
    diag = np.asarray(diag, float); sub = np.asarray(sub, float)
    sup = np.asarray(sup, float); rhs = np.asarray(rhs, float)
    n_sys, n = diag.shape
    cp = np.empty((n_sys, n)); dp = np.empty((n_sys, n)); den = np.empty((n_sys, n))
    den[:, 0] = diag[:, 0]
    cp[:, 0] = sup[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0]
    for i in range(1, n):
        d = diag[:, i] - sub[:, i] * cp[:, i - 1]
        den[:, i] = d
        cp[:, i] = sup[:, i] / d
        dp[:, i] = (rhs[:, i] - sub[:, i] * dp[:, i - 1]) / d
    if np.any(np.abs(den) < _PIVOT_MIN):
        raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
    x = np.empty((n_sys, n))
    x[:, n - 1] = dp[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


rng = np.random.default_rng(3)
for n_sys, n in ((61, 120), (300, 135), (600, 270)):
    diag = 4.0 + rng.random((n_sys, n))
    sub = -1.0 - rng.random((n_sys, n))
    sup = -1.0 - rng.random((n_sys, n))
    rhs = rng.random((n_sys, n))
    a = solve_tridiag_batch(sub, diag, sup, rhs)
    b = hoisted(sub, diag, sup, rhs)
    bitwise = np.array_equal(a, b)
    def t(f, reps=20):
        best = 1e9
        for _ in range(reps):
            s = time.perf_counter(); f(sub, diag, sup, rhs); best = min(best, time.perf_counter()-s)
        return best
    t0, t1 = t(solve_tridiag_batch), t(hoisted)
    print(f"n_sys={n_sys:<4} n={n:<4} shipped {t0*1e3:7.3f} ms  hoisted {t1*1e3:7.3f} ms  "
          f"speedup {t0/t1:4.2f}x  bitwise={bitwise}")
