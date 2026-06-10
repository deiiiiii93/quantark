"""
Float comparison utilities for stable numerical operations.

This module provides functions for comparing floating-point numbers
with configurable tolerances, addressing common issues with float
equality in financial calculations.
"""

import math
from typing import Union, Sequence
import numpy as np

from .constants import Tolerance


def is_close(
    a: float,
    b: float,
    rel_tol: float = Tolerance.RELATIVE,
    abs_tol: float = Tolerance.ABSOLUTE,
) -> bool:
    """
    Check if two numbers are close within specified tolerances.
    
    Uses the same algorithm as math.isclose() but with financial-appropriate
    default tolerances.
    
    Args:
        a: First value to compare.
        b: Second value to compare.
        rel_tol: Relative tolerance (default: 1e-9).
        abs_tol: Absolute tolerance (default: 0.0).
    
    Returns:
        True if |a - b| <= max(rel_tol * max(|a|, |b|), abs_tol).
    
    Examples:
        >>> is_close(1.0, 1.0 + 1e-10)
        True
        >>> is_close(100.0, 100.0001, rel_tol=1e-6)
        True
        >>> is_close(0.0, 1e-11, abs_tol=1e-10)
        True
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def is_zero(x: float, tol: float = Tolerance.ZERO) -> bool:
    """
    Check if a value is effectively zero.
    
    Commonly used for checking if time to expiry is negligible,
    or if a sensitivity is effectively zero.
    
    Args:
        x: Value to check.
        tol: Absolute tolerance for zero check (default: 1e-10).
    
    Returns:
        True if |x| < tol.
    
    Examples:
        >>> is_zero(1e-11)
        True
        >>> is_zero(0.001)
        False
        >>> is_zero(1e-5, tol=1e-4)
        True
    """
    return abs(x) < tol


def is_positive(x: float, tol: float = Tolerance.ZERO) -> bool:
    """
    Check if a value is strictly positive (greater than tolerance).
    
    Args:
        x: Value to check.
        tol: Tolerance threshold (default: 1e-10).
    
    Returns:
        True if x > tol.
    
    Examples:
        >>> is_positive(0.001)
        True
        >>> is_positive(1e-11)
        False
        >>> is_positive(-0.001)
        False
    """
    return x > tol


def is_negative(x: float, tol: float = Tolerance.ZERO) -> bool:
    """
    Check if a value is strictly negative (less than -tolerance).
    
    Args:
        x: Value to check.
        tol: Tolerance threshold (default: 1e-10).
    
    Returns:
        True if x < -tol.
    
    Examples:
        >>> is_negative(-0.001)
        True
        >>> is_negative(-1e-11)
        False
        >>> is_negative(0.001)
        False
    """
    return x < -tol


def is_non_negative(x: float, tol: float = Tolerance.ZERO) -> bool:
    """
    Check if a value is non-negative (zero or positive within tolerance).
    
    Args:
        x: Value to check.
        tol: Tolerance threshold (default: 1e-10).
    
    Returns:
        True if x >= -tol.
    
    Examples:
        >>> is_non_negative(0.0)
        True
        >>> is_non_negative(-1e-11)
        True
        >>> is_non_negative(-0.001)
        False
    """
    return x >= -tol


def is_non_positive(x: float, tol: float = Tolerance.ZERO) -> bool:
    """
    Check if a value is non-positive (zero or negative within tolerance).
    
    Args:
        x: Value to check.
        tol: Tolerance threshold (default: 1e-10).
    
    Returns:
        True if x <= tol.
    
    Examples:
        >>> is_non_positive(0.0)
        True
        >>> is_non_positive(1e-11)
        True
        >>> is_non_positive(0.001)
        False
    """
    return x <= tol


def almost_equal(
    a: float,
    b: float,
    tol: float = Tolerance.PRECISION,
) -> bool:
    """
    Check if two values are almost equal within absolute tolerance.
    
    Simpler than is_close() when you only need absolute tolerance.
    Useful for comparing prices, Greeks, and other financial quantities.
    
    Args:
        a: First value.
        b: Second value.
        tol: Absolute tolerance (default: 1e-6).
    
    Returns:
        True if |a - b| <= tol.
    
    Examples:
        >>> almost_equal(100.0, 100.000001)
        True
        >>> almost_equal(100.0, 100.01)
        False
        >>> almost_equal(0.0, 0.0001, tol=0.001)
        True
    """
    return abs(a - b) <= tol


def compare_arrays(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
    rel_tol: float = Tolerance.RELATIVE,
    abs_tol: float = Tolerance.PRECISION,
) -> bool:
    """
    Check if two arrays are element-wise close within tolerances.
    
    Uses numpy's allclose for efficient array comparison.
    
    Args:
        a: First array.
        b: Second array.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.
    
    Returns:
        True if all elements satisfy the tolerance condition.
    
    Examples:
        >>> compare_arrays([1.0, 2.0], [1.0, 2.0 + 1e-10])
        True
        >>> compare_arrays([1.0, 2.0], [1.0, 2.1])
        False
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    return np.allclose(arr_a, arr_b, rtol=rel_tol, atol=abs_tol)
