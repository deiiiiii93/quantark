"""Vectorized discount-factor helper (`make_df_fn`).

Profiling the SLV Monte Carlo reference on 2026-08-10 recorded 786,636 calls to
`pricing_env.get_discount_factor` from a single list comprehension inside this
helper. The curve is a function of time alone and the same handful of
contractual dates recur across every path, so the per-element Python call is
almost entirely redundant.
"""

import numpy as np
import pytest

from quantark.priceenv.term_sampling import make_df_fn


class _CountingEnv:
    """Minimal env that records how often the curve is queried."""

    def __init__(self, rate: float = 0.03):
        self.rate = rate
        self.calls = 0

    def get_discount_factor(self, t: float) -> float:
        self.calls += 1
        return float(np.exp(-self.rate * float(t)))


def test_scalar_time_returns_a_float():
    env = _CountingEnv()
    got = make_df_fn(env)(2.0)
    assert isinstance(got, float)
    assert got == pytest.approx(np.exp(-0.06))


def test_array_result_matches_elementwise_curve_evaluation_exactly():
    env = _CountingEnv()
    times = np.array([0.0, 0.25, 1.0, 1.0, 0.25, 3.5])
    got = make_df_fn(env)(times)
    expected = np.array([_CountingEnv().get_discount_factor(t) for t in times])
    assert np.array_equal(got, expected)


def test_shape_is_preserved_for_multidimensional_input():
    env = _CountingEnv()
    times = np.array([[0.5, 1.0], [1.5, 2.0]])
    assert make_df_fn(env)(times).shape == (2, 2)


def test_repeated_times_are_only_queried_once():
    """The whole point: cost must scale with distinct times, not element count."""
    env = _CountingEnv()
    times = np.repeat(np.array([0.25, 0.5, 0.75, 1.0]), 500)  # 2000 elements
    got = make_df_fn(env)(times)
    assert got.size == 2000
    assert env.calls == 4, f"queried the curve {env.calls} times for 4 distinct dates"


def test_empty_input_is_handled():
    env = _CountingEnv()
    got = make_df_fn(env)(np.array([]))
    assert got.shape == (0,)
    assert env.calls == 0
