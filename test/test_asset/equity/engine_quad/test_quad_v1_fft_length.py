"""QUAD V1 convolutions on fast FFT lengths (0.16.0 T5, owner-approved round-off moves).

The padded transform length was ``factor * (2N - 1)``; for most lattices it held a large prime factor (7346 =
2 x 3673) and pocketfft ran Bluestein. An even padded length at least ``2 * (2N - 1) - 1`` computes the full linear
convolution, so any longer even length computes the same one: the engine now runs the smallest even 5-smooth length
not below the legacy one, on real FFTs. These tests pin the length rule, the linear convolution against the direct
sum, and the transpose identities the forward adjoint pass relies on.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.quad_math import QuadratureMath, _even_5_smooth_at_least
from quantark.asset.equity.engine.quad.v1_adjoint import transition_readout_weights

# lattices whose legacy factor-2 length 4N - 2 held a prime factor >= 400 (Bluestein) in the synthetic book, plus
# two whose legacy length was already smooth and a small odd one
AWKWARD_N = (1837, 1681, 1525, 1365, 1209, 1001, 301, 37)


def _largest_prime(n: int) -> int:
    p, f = 2, 1
    while p * p <= n:
        while n % p == 0:
            f, n = p, n // p
        p += 1
    return max(f, n)


def _math(n: int, *, factor: int = 2, alpha: float = 0.0) -> QuadratureMath:
    return QuadratureMath(n, 100.0, 1.0, 0.25, fft_padding_factor=factor, fft_filter_alpha=alpha, fft_filter_power=8)


@pytest.mark.parametrize("n", AWKWARD_N)
@pytest.mark.parametrize("factor", [2, 4])
def test_an_even_padding_runs_on_the_smallest_even_5_smooth_length(n, factor):
    base = 2 * n - 1
    legacy = factor * base
    fft_len, real = _math(n, factor=factor).fft_length(base)
    assert real
    assert fft_len % 2 == 0 and fft_len >= legacy >= 2 * base - 1
    assert _largest_prime(fft_len) <= 5
    assert all(_largest_prime(m) > 5 for m in range(legacy, fft_len, 2))   # the smallest such even length


def test_the_book_lengths_that_ran_bluestein():
    assert _largest_prime(7346) == 3673 and _even_5_smooth_at_least(7346) == 7500
    assert _even_5_smooth_at_least(4834) == 4860
    assert _even_5_smooth_at_least(6722) == 6750
    assert _even_5_smooth_at_least(4800) == 4800


@pytest.mark.parametrize("factor", [1, 3])
def test_an_odd_padding_keeps_its_legacy_length(factor):
    """Factor 1 is circular and an odd length samples the filter on max|fftfreq| < 1/2: both stay as they were."""
    n = 1837
    base = 2 * n - 1
    assert _math(n, factor=factor).fft_length(base) == (factor * base, False)


@pytest.mark.parametrize("n", AWKWARD_N)
def test_the_convolution_is_the_direct_linear_convolution(n):
    rng = np.random.default_rng(n)
    m = _math(n)
    base = 2 * n - 1
    omega = rng.uniform(-1.0, 1.0, base)
    u = rng.uniform(-1.0, 1.0, base)            # full support: nothing wraps at any length >= 2 base - 1
    direct = np.convolve(omega, u)[n - 1 : 2 * n - 1] * (m.h / 3.0)
    got = m.convolution_fft(omega, u)
    rows = rng.uniform(-1.0, 1.0, (3, base))
    got_many = m.convolution_fft_many(omega, rows)
    direct_many = np.stack([np.convolve(omega, r)[n - 1 : 2 * n - 1] for r in rows]) * (m.h / 3.0)
    assert np.max(np.abs(got - direct)) <= 1e-13 * np.max(np.abs(direct))
    assert np.max(np.abs(got_many - direct_many)) <= 1e-13 * np.max(np.abs(direct_many))


@pytest.mark.parametrize("n", AWKWARD_N)
@pytest.mark.parametrize("alpha", [0.0, 12.0])
def test_the_reversed_kernel_convolution_is_the_transpose(n, alpha):
    """``<A x, y> = <x, A^T y>`` with ``A x = conv(omega, pad x)`` and ``A^T y = conv(omega reversed, pad y)`` -- the
    diffusion transpose of the forward adjoint pass (``v1_adjoint.diffuse_transpose``), filtered or not."""
    rng = np.random.default_rng(1000 + n)
    m = _math(n, alpha=alpha)
    base = 2 * n - 1
    omega = rng.uniform(0.0, 1.0, base)
    x = np.zeros(base)
    y = np.zeros(base)
    x[:n] = rng.uniform(-1.0, 1.0, n)
    y[:n] = rng.uniform(-1.0, 1.0, n)
    ax = m.convolution_fft(omega, x)
    aty = m.convolution_fft_many(np.ascontiguousarray(omega[::-1]), y[None, :])[0]
    lhs, rhs = float(ax @ y[:n]), float(x[:n] @ aty)
    scale = float(np.abs(ax) @ np.abs(y[:n]))
    assert abs(lhs - rhs) <= 1e-13 * scale


@pytest.mark.parametrize("n", (1837, 301))
def test_the_filtered_readout_transposes_on_the_same_length(n):
    """``_readout_at_spot`` filters the weighted values on ``fft_length``; ``transition_readout_weights`` filters
    the kernel on the same length -- the filter is real and even, so the two dot products agree."""
    rng = np.random.default_rng(7 + n)
    m = _math(n, alpha=12.0)
    base = 2 * n - 1
    tau, alpha = 0.02, 0.3
    values = rng.uniform(0.0, 1.0, n)
    weighted = m.filtered_weights(m.simpson_weights(values, 0, n - 1, (n - 1) % 2))
    assert weighted.shape == (m.fft_length(base)[0],)
    offsets = m.grid[0] + np.arange(weighted.size, dtype=float) * m.h
    omega = np.exp(-(offsets * offsets) / (4.0 * tau) + alpha * offsets)
    forward = (m.h / 3.0) * float(weighted @ omega)
    # the row's ends also carry the erfc tail terms, which the forward line above omits: prefactor 0 is the tails
    row = transition_readout_weights(m, 1.0, alpha, 0.0, tau) - transition_readout_weights(m, 0.0, alpha, 0.0, tau)
    transposed = float(row @ values)
    assert abs(forward - transposed) <= 1e-13 * abs(forward)
