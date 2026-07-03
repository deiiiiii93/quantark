"""Validate the shared term-structure benchmark harness itself."""
import numpy as np
import pytest
from scipy import stats

from term_structure_benchmarks import (
    make_term_env,
    reference_european_call_price,
)


def test_flat_shape_matches_hand_black_scholes():
    env = make_term_env("flat")
    S, K, T, r, q, vol = 100.0, 100.0, 1.0, 0.03, 0.01, 0.20
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    expected = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(
        -r * T
    ) * stats.norm.cdf(d2)
    assert reference_european_call_price(env, 100.0, 1.0) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_term_shapes_are_genuinely_non_flat(shape):
    env = make_term_env(shape)
    assert env.get_rate(0.25) != pytest.approx(env.get_rate(2.0), abs=1e-6)
    assert env.get_div_yield(0.25) != pytest.approx(
        env.get_div_yield(2.0), abs=1e-6
    )
    assert env.get_vol(100.0, 0.25) != pytest.approx(
        env.get_vol(100.0, 2.0), abs=1e-6
    )


def test_kinked_shape_has_negative_carry_segment():
    env = make_term_env("kinked")
    times = np.linspace(0.05, 2.0, 40)
    assert min(env.get_div_yield(float(t)) for t in times) < 0.0


def test_reference_price_uses_curve_df_not_flat_rate():
    env = make_term_env("up")
    px = reference_european_call_price(env, 100.0, 2.0)
    assert 0.0 < px < 100.0
    # recompute with the same formula, independently
    S = env.spot
    T = 2.0
    r, q = env.get_rate(T), env.get_div_yield(T)
    vol = env.get_vol(100.0, T)
    d1 = (np.log(S / 100.0) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    F = S * np.exp((r - q) * T)
    expected = env.get_discount_factor(T) * (
        F * stats.norm.cdf(d1) - 100.0 * stats.norm.cdf(d2)
    )
    assert px == pytest.approx(expected, abs=1e-12)
