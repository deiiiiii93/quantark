"""
Unit tests for Brownian-bridge barrier crossing utilities.
"""
import math

import numpy as np

from quantark.asset.equity.process.bsm.qmc_brownian_bridge import (
    compute_step_crossing_probabilities,
)
from quantark.util.numerical import is_close


def test_crossing_and_touch_probabilities() -> None:
    """Crossing or touching the barrier should yield probability 1."""
    barrier = 100.0
    sigma = 0.2
    times = np.array([0.5, 1.0])
    paths = np.array([[90.0, 110.0, 100.0]])

    probs = compute_step_crossing_probabilities(paths, barrier, sigma, times)

    assert probs.shape == (1, 2)
    assert is_close(probs[0, 0], 1.0, abs_tol=1e-12)
    assert is_close(probs[0, 1], 1.0, abs_tol=1e-12)


def test_same_side_probability_formula() -> None:
    """Same-side endpoints follow Brownian-bridge formula."""
    barrier = 100.0
    sigma = 0.2
    times = np.array([0.5, 1.0])
    paths = np.array(
        [
            [110.0, 120.0, 130.0],  # Above barrier
            [90.0, 80.0, 70.0],  # Below barrier
        ]
    )

    probs = compute_step_crossing_probabilities(paths, barrier, sigma, times)

    assert probs.shape == (2, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    dt = np.array([0.5, 0.5])
    for row in range(paths.shape[0]):
        for col in range(2):
            s0 = paths[row, col]
            s1 = paths[row, col + 1]
            log_term = math.log(s0 / barrier) * math.log(s1 / barrier)
            expected = math.exp(-2.0 * log_term / (sigma * sigma * dt[col]))
            assert is_close(probs[row, col], expected, rel_tol=1e-12, abs_tol=1e-12)
