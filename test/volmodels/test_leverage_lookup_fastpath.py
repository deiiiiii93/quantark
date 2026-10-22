"""Scalar-time fast path in LeverageSurface.leverage.

The SLV Monte Carlo path calls `leverage(spot_vector, scalar_t)` once per step,
and profiling on 2026-08-10 put ~47% of the reference-stack runtime inside this
one method: it broadcast the scalar time across every path and then ran an
n_paths-long searchsorted over identical values, plus four 2-D fancy-index
gathers.

The optimization hoists the time lookup out of the per-path work. These tests
pin the semantics it must preserve -- above all that a scalar time and a vector
of that same time agree EXACTLY, which is what makes the hoist legitimate.
"""

import numpy as np
import pytest

from quantark.volmodels.slv.leverage import LeverageSurface

TIME_GRID = np.array([0.0, 0.5, 1.5, 3.0])
STRIKE_GRID = np.array([50.0, 80.0, 100.0, 125.0, 200.0])
LEVERAGE_GRID = np.array(
    [
        [1.05, 1.00, 0.95, 0.92, 0.90],
        [1.08, 1.02, 0.97, 0.94, 0.91],
        [1.12, 1.05, 0.99, 0.96, 0.93],
        [1.20, 1.10, 1.02, 0.98, 0.95],
    ]
)


def _surface() -> LeverageSurface:
    return LeverageSurface(
        time_grid=TIME_GRID, strike_grid=STRIKE_GRID, leverage_grid=LEVERAGE_GRID
    )


def _flat_surface() -> LeverageSurface:
    return LeverageSurface(
        time_grid=np.array([0.0]),
        strike_grid=STRIKE_GRID,
        leverage_grid=LEVERAGE_GRID[:1],
    )


SPOTS = np.array([40.0, 50.0, 63.0, 80.0, 100.0, 117.0, 125.0, 200.0, 260.0])


@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.1, 3.0, 4.0])
def test_scalar_time_matches_a_vector_of_the_same_time_bitwise(t):
    """The invariant the fast path rests on: broadcasting t must be redundant."""
    surface = _surface()
    scalar = surface.leverage(SPOTS, t)
    vector = surface.leverage(SPOTS, np.full(SPOTS.shape, t))
    assert np.array_equal(scalar, vector)


def test_node_values_are_reproduced_exactly():
    """At a (time, strike) node the interpolant must return the node value."""
    surface = _surface()
    for i, t in enumerate(TIME_GRID):
        got = surface.leverage(STRIKE_GRID, t)
        assert got == pytest.approx(LEVERAGE_GRID[i], rel=0, abs=1e-15)


def test_time_midpoint_is_the_average_of_the_bracketing_rows():
    surface = _surface()
    got = surface.leverage(STRIKE_GRID, 1.0)  # midway between 0.5 and 1.5
    expected = 0.5 * (LEVERAGE_GRID[1] + LEVERAGE_GRID[2])
    assert got == pytest.approx(expected, rel=1e-15)


def test_extrapolation_is_flat_in_both_axes():
    surface = _surface()
    # Beyond the strike grid, the edge column; beyond the time grid, the edge row.
    assert surface.leverage(np.array([10.0]), 0.0)[0] == pytest.approx(
        LEVERAGE_GRID[0, 0]
    )
    assert surface.leverage(np.array([10_000.0]), 3.0)[0] == pytest.approx(
        LEVERAGE_GRID[-1, -1]
    )
    assert surface.leverage(np.array([100.0]), -5.0)[0] == pytest.approx(
        surface.leverage(np.array([100.0]), 0.0)[0]
    )
    assert surface.leverage(np.array([100.0]), 99.0)[0] == pytest.approx(
        surface.leverage(np.array([100.0]), 3.0)[0]
    )


def test_genuinely_varying_time_vector_still_interpolates_per_element():
    """The vector-t path must keep working; only the scalar case is hoisted."""
    surface = _surface()
    spots = np.array([100.0, 100.0, 100.0])
    times = np.array([0.0, 1.5, 3.0])
    got = surface.leverage(spots, times)
    expected = np.array(
        [LEVERAGE_GRID[0, 2], LEVERAGE_GRID[2, 2], LEVERAGE_GRID[3, 2]]
    )
    assert got == pytest.approx(expected, rel=1e-15)


def test_single_time_row_surface_ignores_time():
    surface = _flat_surface()
    for t in (0.0, 2.0, 50.0):
        assert surface.leverage(STRIKE_GRID, t) == pytest.approx(
            LEVERAGE_GRID[0], rel=1e-15
        )


def test_scalar_spot_returns_a_float():
    surface = _surface()
    got = surface.leverage(100.0, 1.0)
    assert isinstance(got, float)
    assert got == pytest.approx(0.5 * (LEVERAGE_GRID[1, 2] + LEVERAGE_GRID[2, 2]))


def test_output_shape_follows_broadcast():
    surface = _surface()
    spots = np.array([[100.0, 125.0], [80.0, 200.0]])
    assert surface.leverage(spots, 1.0).shape == (2, 2)
    assert surface.leverage(spots, np.zeros((2, 2))).shape == (2, 2)


def test_scalar_time_path_does_not_search_the_time_grid_per_path():
    """The whole point: the time lookup must not scale with path count.

    searchsorted on the time grid must receive a scalar-sized query, not one
    entry per path. Without the hoist this call sees SPOTS.size elements.
    """
    surface = _surface()
    seen = []
    real_searchsorted = np.searchsorted

    def recording(a, v, *args, **kwargs):
        if a is surface.time_grid or np.array_equal(a, surface.time_grid):
            seen.append(np.asarray(v).size)
        return real_searchsorted(a, v, *args, **kwargs)

    original = np.searchsorted
    np.searchsorted = recording
    try:
        surface.leverage(SPOTS, 1.1)
    finally:
        np.searchsorted = original

    assert seen, "expected a time-grid searchsorted call"
    assert max(seen) == 1, f"time lookup scaled with paths: query sizes {seen}"
