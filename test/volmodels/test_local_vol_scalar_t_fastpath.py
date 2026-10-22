"""Scalar-time fast path in LocalVolSurface.local_vol.

MC kernels call `local_vol(spot_vector, scalar_t)` once per step; profiling
2026-08-10 put 82% of an LV European MC run inside this method (broadcast
scalar t, n_paths-long searchsorted over identical values, four 2-D fancy
gathers). The fast path hoists the time bracket out of the per-path work.
These tests pin the invariant that makes the hoist legitimate: a scalar time
and a vector of that same time must agree EXACTLY.

Mirrors test_leverage_lookup_fastpath.py (2D program, b98a8d9).
"""

import numpy as np
import pytest

from quantark.volmodels.localvol.surface import LocalVolSurface

TIME_GRID = np.array([0.0, 0.5, 1.5, 3.0])
STRIKE_GRID = np.array([50.0, 80.0, 100.0, 125.0, 200.0])
LV_GRID = np.array(
    [
        [1.05, 1.00, 0.95, 0.92, 0.90],
        [1.08, 1.02, 0.97, 0.94, 0.91],
        [1.12, 1.05, 0.99, 0.96, 0.93],
        [1.20, 1.10, 1.02, 0.98, 0.95],
    ]
)
SPOTS = np.array([40.0, 50.0, 63.0, 80.0, 100.0, 117.0, 125.0, 200.0, 260.0])


def _surface(interp="linear_s"):
    return LocalVolSurface(STRIKE_GRID, TIME_GRID, LV_GRID, interp=interp)


def _single_time_surface():
    return LocalVolSurface(STRIKE_GRID, np.array([1.0]), LV_GRID[:1])


@pytest.mark.parametrize("interp", ["linear_s", "linear_logs"])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.1, 3.0, 4.0, -1.0])
def test_scalar_time_matches_vector_of_same_time_bitwise(interp, t):
    """The invariant the fast path rests on: broadcasting t must be redundant."""
    lv = _surface(interp)
    scalar = lv.local_vol(SPOTS, t)
    vector = lv.local_vol(SPOTS, np.full(SPOTS.shape, t))
    assert np.array_equal(scalar, vector)
    assert scalar.tobytes() == vector.tobytes()


def test_single_time_grid_scalar_matches_vector_bitwise():
    lv = _single_time_surface()
    assert np.array_equal(
        lv.local_vol(SPOTS, 0.7), lv.local_vol(SPOTS, np.full(SPOTS.shape, 0.7))
    )


def test_node_values_reproduced_exactly():
    lv = _surface()
    for i, t in enumerate(TIME_GRID):
        got = lv.local_vol(STRIKE_GRID, float(t))
        assert got == pytest.approx(LV_GRID[i], rel=0, abs=1e-15)


def test_time_midpoint_is_average_of_bracketing_rows():
    lv = _surface()
    got = lv.local_vol(STRIKE_GRID, 1.0)  # midway between 0.5 and 1.5
    assert got == pytest.approx(0.5 * (LV_GRID[1] + LV_GRID[2]), rel=1e-15)


def test_strike_clamps_flat_extrapolation():
    lv = _surface()
    lo = lv.local_vol(np.array([10.0, 50.0]), 0.25)
    hi = lv.local_vol(np.array([200.0, 500.0]), 0.25)
    assert lo[0] == lo[1] and hi[0] == hi[1]


def test_scalar_spot_scalar_t_returns_float():
    lv = _surface()
    out = lv.local_vol(100.0, 0.7)
    assert isinstance(out, float)
    vec = lv.local_vol(np.array([100.0]), 0.7)
    assert out == float(vec[0])


def test_large_vector_matches_bitwise():
    lv = _surface()
    rng = np.random.default_rng(7)
    spots = np.exp(rng.normal(np.log(100.0), 0.6, size=100_000))
    a = lv.local_vol(spots, 0.083333333)
    b = lv.local_vol(spots, np.full(spots.shape, 0.083333333))
    assert a.tobytes() == b.tobytes()
