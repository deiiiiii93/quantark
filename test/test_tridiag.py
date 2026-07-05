import numpy as np
import pytest

from quantark.util.exceptions import NumericalError
from quantark.util.numerical import solve_tridiag, solve_tridiag_batch


def _thomas_reference(a, b, c, d):
    # scalar Thomas, full-length convention (a[0], c[-1] ignored) — mirrors the ADI solvers
    n = len(d)
    cp = np.zeros(n); dp = np.zeros(n); x = np.zeros(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def test_batch_bit_identical_to_scalar_thomas():
    rng = np.random.default_rng(0)
    n_sys, N = 7, 40
    sub = rng.normal(size=(n_sys, N))
    sup = rng.normal(size=(n_sys, N))
    diag = 4.0 + rng.random((n_sys, N))          # diagonally dominant
    rhs = rng.normal(size=(n_sys, N))
    out = solve_tridiag_batch(sub, diag, sup, rhs)
    for k in range(n_sys):
        ref = _thomas_reference(sub[k], diag[k], sup[k], rhs[k])
        assert np.array_equal(out[k], ref)        # bit-identical, not just close


def test_batch_zero_pivot_raises():
    sub = np.zeros((1, 3)); sup = np.zeros((1, 3))
    diag = np.array([[1.0, 0.0, 1.0]])
    with pytest.raises(NumericalError):
        solve_tridiag_batch(sub, diag, sup, np.ones((1, 3)))


def test_single_matches_dense_solve():
    rng = np.random.default_rng(1)
    N = 50
    sub = rng.normal(size=N - 1); sup = rng.normal(size=N - 1)
    diag = 4.0 + rng.random(N); rhs = rng.normal(size=N)
    A = np.diag(diag) + np.diag(sub, -1) + np.diag(sup, 1)
    np.testing.assert_allclose(solve_tridiag(sub, diag, sup, rhs),
                               np.linalg.solve(A, rhs), rtol=1e-10, atol=1e-12)


def test_single_singular_raises():
    with pytest.raises(NumericalError):
        solve_tridiag(np.zeros(1), np.zeros(2), np.zeros(1), np.ones(2))
