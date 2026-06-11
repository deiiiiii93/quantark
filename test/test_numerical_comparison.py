"""Tests for shared floating-point comparison helpers."""

from quantark.util.numerical import (
    Tolerance,
    is_greater_than,
    is_greater_than_or_close,
    is_less_than,
    is_less_than_or_close,
)


def test_tolerant_ordering_treats_last_bit_noise_as_equality():
    boundary = 0.011904761904761862
    noisy_equal = 0.011904761904761904

    assert not is_greater_than(
        noisy_equal, boundary, abs_tol=Tolerance.PRECISION
    )
    assert is_greater_than_or_close(
        noisy_equal, boundary, abs_tol=Tolerance.PRECISION
    )


def test_tolerant_ordering_rejects_meaningful_boundary_violations():
    boundary = 1.0

    assert is_greater_than(1.01, boundary, abs_tol=Tolerance.PRECISION)
    assert is_less_than(0.99, boundary, abs_tol=Tolerance.PRECISION)
    assert not is_greater_than_or_close(
        0.99, boundary, abs_tol=Tolerance.PRECISION
    )
    assert not is_less_than_or_close(
        1.01, boundary, abs_tol=Tolerance.PRECISION
    )


def test_tolerant_inclusive_ordering_accepts_noise_on_either_side():
    boundary = 1.0
    noise = Tolerance.PRECISION / 2.0

    assert is_greater_than_or_close(
        boundary - noise, boundary, abs_tol=Tolerance.PRECISION
    )
    assert is_less_than_or_close(
        boundary + noise, boundary, abs_tol=Tolerance.PRECISION
    )
