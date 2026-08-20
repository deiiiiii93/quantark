"""Shared Brownian-bridge transform with an optional Numba backend.

Contract mirrors test_gbm_path_kernel.py: the NumPy reference reproduces the
legacy inline loop bit-for-bit, the Numba path (when installed) is asserted
bit-identical to it, and the live backend self-reports.

Two order traps make the fused kernel's bit-identity conditional, and both are
pinned here:
  1. the conditional mean must keep the grouping ``(a*W_l + b*W_r) / denom``;
  2. when ``left == -1`` the ``a*0.0`` term must still be ADDED, because
     ``0.0 + -0.0 == +0.0`` while ``-0.0`` alone stays ``-0.0`` -- a
     sign-of-zero difference invisible to allclose but caught by a byte
     compare. ``test_all_zero_normals_preserve_signed_zeros`` is that trap.
"""

import numpy as np
import pytest

from quantark.montecarlo import bridge_kernels
from quantark.montecarlo.qmc_brownian_bridge import BrownianBridge


def _legacy_transform(bridge, z):
    """The pre-extraction BrownianBridge.transform body, verbatim."""
    z = np.asarray(z, dtype=float)
    n_paths, n_steps = z.shape
    W = np.zeros((n_paths, n_steps), dtype=float)
    idx0 = bridge.indices[0]
    std0 = np.sqrt(bridge.variances[0])
    W[:, idx0] = std0 * z[:, 0]
    for j in range(1, n_steps):
        k = bridge.indices[j]
        l = bridge.left[j]
        r = bridge.right[j]
        t_l = 0.0 if l == -1 else bridge.times[l]
        t_r = bridge.times[r]
        t_m = bridge.times[k]
        if l == -1:
            W_l = 0.0
        else:
            W_l = W[:, l]
        W_r = W[:, r]
        denom = t_r - t_l
        if denom <= 0.0:
            raise ValueError("Invalid Brownian bridge interval length.")
        mean = ((t_r - t_m) * W_l + (t_m - t_l) * W_r) / denom
        std = np.sqrt(bridge.variances[j])
        W[:, k] = mean + std * z[:, j]
    dW = np.empty_like(W)
    dW[:, 0] = W[:, 0]
    if n_steps > 1:
        dW[:, 1:] = W[:, 1:] - W[:, :-1]
    return dW


def _grids():
    grids = {f"uniform_{n}": np.linspace(1.0 / n, 1.0, n)
             for n in (1, 2, 3, 7, 63, 252)}
    rng = np.random.default_rng(5)
    grids["nonuniform_97"] = np.sort(rng.uniform(0.001, 3.0, size=97))
    grids["clustered_120"] = np.concatenate(
        [np.linspace(0.002, 0.25, 90), np.linspace(0.30, 2.0, 30)]
    )
    return grids


def _args(bridge):
    return (bridge.times, bridge.indices, bridge.left, bridge.right,
            bridge.variances)


def test_backend_reports_a_known_value():
    assert bridge_kernels.bridge_backend() in ("numba", "numpy")


@pytest.mark.parametrize("name", sorted(_grids()))
def test_numpy_reference_matches_legacy_bitwise(name):
    times = _grids()[name]
    bridge = BrownianBridge.from_time_grid(times)
    rng = np.random.default_rng(11)
    z = rng.standard_normal((257, times.size))
    got = bridge_kernels.bridge_transform_numpy(z, *_args(bridge))
    assert got.tobytes() == _legacy_transform(bridge, z).tobytes()


@pytest.mark.parametrize("name", sorted(_grids()))
def test_dispatcher_matches_legacy_bitwise(name):
    times = _grids()[name]
    bridge = BrownianBridge.from_time_grid(times)
    rng = np.random.default_rng(12)
    z = rng.standard_normal((513, times.size))
    got = bridge_kernels.bridge_transform(z, *_args(bridge))
    assert got.tobytes() == _legacy_transform(bridge, z).tobytes()


@pytest.mark.parametrize("name", sorted(_grids()))
def test_public_transform_matches_legacy_bitwise(name):
    """The wiring test: BrownianBridge.transform must be byte-stable."""
    times = _grids()[name]
    bridge = BrownianBridge.from_time_grid(times)
    rng = np.random.default_rng(13)
    z = rng.standard_normal((64, times.size))
    assert bridge.transform(z).tobytes() == _legacy_transform(bridge, z).tobytes()


@pytest.mark.parametrize("name", sorted(_grids()))
def test_all_zero_normals_preserve_signed_zeros(name):
    """Sign-of-zero trap: dropping the a*0.0 term would flip -0.0 to +0.0."""
    times = _grids()[name]
    bridge = BrownianBridge.from_time_grid(times)
    z = np.zeros((3, times.size))
    assert bridge.transform(z).tobytes() == _legacy_transform(bridge, z).tobytes()


def test_single_step_grid():
    bridge = BrownianBridge.from_time_grid(np.array([1.0]))
    rng = np.random.default_rng(3)
    z = rng.standard_normal((8, 1))
    got = bridge.transform(z)
    assert got.shape == (8, 1)
    assert got.tobytes() == _legacy_transform(bridge, z).tobytes()


def test_noncontiguous_input_matches_contiguous():
    times = np.linspace(1.0 / 32, 1.0, 32)
    bridge = BrownianBridge.from_time_grid(times)
    rng = np.random.default_rng(4)
    z = rng.standard_normal((48, 32))
    wide = np.empty((48, 64))
    wide[:, ::2] = z
    view = wide[:, ::2]
    assert not view.flags.c_contiguous
    assert bridge.transform(view).tobytes() == bridge.transform(z).tobytes()


def test_zero_paths():
    times = np.linspace(0.25, 1.0, 4)
    bridge = BrownianBridge.from_time_grid(times)
    got = bridge.transform(np.zeros((0, 4)))
    assert got.shape == (0, 4)


def test_rejects_bad_shapes():
    bridge = BrownianBridge.from_time_grid(np.linspace(0.25, 1.0, 4))
    with pytest.raises(ValueError, match="2D array"):
        bridge.transform(np.zeros(4))
    with pytest.raises(ValueError, match="time steps"):
        bridge.transform(np.zeros((5, 3)))


@pytest.mark.skipif(
    bridge_kernels.bridge_backend() != "numba",
    reason="numba accelerator not installed",
)
@pytest.mark.parametrize("name", sorted(_grids()))
def test_numba_path_is_bit_identical(name):
    times = _grids()[name]
    bridge = BrownianBridge.from_time_grid(times)
    for seed, n_paths in ((1, 1), (2, 5), (3, 1024)):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_paths, times.size))
        a = bridge_kernels._bridge_transform_numba(z, *_args(bridge))
        b = bridge_kernels.bridge_transform_numpy(z, *_args(bridge))
        assert a.tobytes() == b.tobytes()
