"""Optional compiled Thomas accelerator (spec WS-A2).

The batched tridiagonal solve is ~62% of a PDE march. A compiled kernel was
measured at 2-2.7x in the 2026-08-05 research round, bit-identical to the NumPy
sweep because it performs the same recurrences in the same order with the same
pivot guard.

It ships as an OPTIONAL accelerator: absent a built shared library the package
must behave exactly as the pure-NumPy implementation, so these tests all have
to pass on a machine that never compiles anything.
"""

import numpy as np
import pytest

from quantark.util.exceptions import NumericalError
from quantark.util.numerical import tridiag


def _systems(n_sys=5, n=24, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_sys, n)),
        4.0 + rng.random((n_sys, n)),
        rng.normal(size=(n_sys, n)),
        rng.normal(size=(n_sys, n)),
    )


def test_backend_reports_which_kernel_is_live():
    """Runs must be able to record what produced them (gate A2-G4)."""
    assert tridiag.tridiag_backend() in ("numpy", "c")


def test_numpy_backend_is_always_available():
    sub, diag, sup, rhs = _systems()
    out = tridiag.solve_tridiag_batch_numpy(sub, diag, sup, rhs)
    assert out.shape == diag.shape
    assert np.all(np.isfinite(out))


def test_dispatcher_matches_the_numpy_reference_bitwise():
    """Whichever backend is live, its output must be bit-identical (A2-G1)."""
    for seed in range(4):
        sub, diag, sup, rhs = _systems(seed=seed)
        assert np.array_equal(
            tridiag.solve_tridiag_batch(sub, diag, sup, rhs),
            tridiag.solve_tridiag_batch_numpy(sub, diag, sup, rhs),
        )


def test_dispatcher_matches_numpy_on_awkward_shapes():
    for n_sys, n in ((1, 2), (1, 3), (3, 2), (129, 7), (7, 129)):
        sub, diag, sup, rhs = _systems(n_sys=n_sys, n=n, seed=n_sys + n)
        assert np.array_equal(
            tridiag.solve_tridiag_batch(sub, diag, sup, rhs),
            tridiag.solve_tridiag_batch_numpy(sub, diag, sup, rhs),
        )


def test_dispatcher_raises_the_same_error_on_a_collapsed_pivot():
    """Guard parity: both backends reject, with the same exception type."""
    sub = np.array([[0.0, 1.0, 1.0]])
    sup = np.array([[1.0, 1.0, 0.0]])
    diag = np.array([[1.0, 1.0, 1.0]])
    rhs = np.ones((1, 3))
    with pytest.raises(NumericalError):
        tridiag.solve_tridiag_batch(sub, diag, sup, rhs)
    with pytest.raises(NumericalError):
        tridiag.solve_tridiag_batch_numpy(sub, diag, sup, rhs)


def test_dispatcher_raises_on_a_zero_first_pivot():
    sub = np.zeros((1, 3))
    sup = np.ones((1, 3))
    diag = np.array([[0.0, 1.0, 1.0]])
    rhs = np.ones((1, 3))
    with pytest.raises(NumericalError):
        tridiag.solve_tridiag_batch(sub, diag, sup, rhs)


def test_non_contiguous_input_is_handled():
    """A transposed view must not be reinterpreted by the compiled path."""
    sub, diag, sup, rhs = _systems(n_sys=6, n=10, seed=11)
    view = np.asfortranarray(diag)
    assert np.array_equal(
        tridiag.solve_tridiag_batch(sub, view, sup, rhs),
        tridiag.solve_tridiag_batch_numpy(sub, view, sup, rhs),
    )


def test_c_backend_when_present_is_bitwise_and_guard_compatible():
    """If a library was built, hold it to the same contract as NumPy."""
    if tridiag.tridiag_backend() != "c":
        pytest.skip("compiled kernel not built on this machine")
    sub, diag, sup, rhs = _systems(n_sys=8, n=32, seed=3)
    assert np.array_equal(
        tridiag._solve_tridiag_batch_c(sub, diag, sup, rhs),
        tridiag.solve_tridiag_batch_numpy(sub, diag, sup, rhs),
    )
    with pytest.raises(NumericalError):
        tridiag._solve_tridiag_batch_c(
            np.zeros((1, 3)), np.array([[0.0, 1.0, 1.0]]), np.ones((1, 3)), np.ones((1, 3))
        )
