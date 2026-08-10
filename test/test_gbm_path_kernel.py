"""Shared GBM path-build tail with an optional Numba backend.

Contract mirrors test_qe_variance_kernel.py: the NumPy reference reproduces
the legacy inline tail bit-for-bit, the Numba path (when installed) is
asserted bit-identical to the reference, and the live backend self-reports.
"""

import numpy as np
import pytest

from quantark.montecarlo import gbm_kernels


def _inputs(n_paths=257, n_steps=63, seed=11):
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(1.0 / n_steps)
    k = np.arange(n_steps)
    vol = 0.18 + 0.06 * np.sin(2 * np.pi * k / n_steps) ** 2
    drift_dt = (0.03 + 0.02 * k / n_steps - 0.5 * vol * vol) * (1.0 / n_steps)
    return dW, drift_dt, vol, 100.0


def _legacy_tail(dW, drift_dt, vol_vec, s0):
    """The pre-extraction generate_paths tail, verbatim."""
    paths = np.zeros((dW.shape[0], dW.shape[1] + 1), dtype=float)
    paths[:, 0] = s0
    exp_term = np.exp(drift_dt.reshape(1, -1) + vol_vec.reshape(1, -1) * dW)
    paths[:, 1:] = s0 * np.cumprod(exp_term, axis=1)
    return paths


def test_backend_reports_a_known_value():
    assert gbm_kernels.gbm_backend() in ("numba", "numpy")


def test_numpy_reference_matches_legacy_tail_bitwise():
    dW, drift_dt, vol, s0 = _inputs()
    a = gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, vol, s0)
    assert a.tobytes() == _legacy_tail(dW, drift_dt, vol, s0).tobytes()


def test_dispatcher_matches_reference_bitwise():
    dW, drift_dt, vol, s0 = _inputs()
    a = gbm_kernels.gbm_path_tail(dW, drift_dt, vol, s0)
    b = gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, vol, s0)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (7, 5), (1024, 252)])
def test_shapes_and_initial_column(n_paths, n_steps):
    dW, drift_dt, vol, s0 = _inputs(n_paths=n_paths, n_steps=n_steps)
    out = gbm_kernels.gbm_path_tail(dW, drift_dt, vol, s0)
    assert out.shape == (n_paths, n_steps + 1)
    assert np.all(out[:, 0] == s0)


def test_noncontiguous_input_matches_contiguous():
    dW, drift_dt, vol, s0 = _inputs(n_paths=64, n_steps=32)
    wide = np.empty((64, 64))
    wide[:, ::2] = dW
    view = wide[:, ::2]
    assert not view.flags.c_contiguous
    a = gbm_kernels.gbm_path_tail(view, drift_dt, vol, s0)
    assert a.tobytes() == gbm_kernels.gbm_path_tail(dW, drift_dt, vol, s0).tobytes()


@pytest.mark.skipif(
    gbm_kernels.gbm_backend() != "numba", reason="numba accelerator not installed"
)
def test_numba_path_is_bit_identical_across_regimes():
    for seed, n_paths, n_steps in ((1, 1024, 252), (2, 8192, 63), (3, 3, 1)):
        dW, drift_dt, vol, s0 = _inputs(n_paths=n_paths, n_steps=n_steps, seed=seed)
        a = gbm_kernels._gbm_path_tail_numba(dW, drift_dt, vol, s0)
        b = gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, vol, s0)
        assert a.tobytes() == b.tobytes()
