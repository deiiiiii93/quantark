"""
Safe mathematical operations with overflow/underflow protection.

This module provides wrappers around standard mathematical functions
that handle edge cases gracefully, preventing numerical errors in
financial calculations.
"""

import math
from typing import Union, Optional
import numpy as np

from .constants import Tolerance


def safe_log(
    x: Union[float, np.ndarray],
    min_value: float = Tolerance.LOG_MIN,
) -> Union[float, np.ndarray]:
    """
    Compute natural logarithm with underflow protection.
    
    Prevents log(0) or log(negative) errors by clamping input
    to a minimum positive value.
    
    Args:
        x: Input value(s). Can be scalar or numpy array.
        min_value: Minimum value to prevent log(0) (default: 1e-16).
    
    Returns:
        Natural logarithm of max(x, min_value).
    
    Examples:
        >>> safe_log(1.0)
        0.0
        >>> safe_log(0.0)  # Returns log(1e-16) instead of error
        -36.8413...
        >>> safe_log(-1.0)  # Returns log(1e-16) instead of error
        -36.8413...
    """
    if isinstance(x, np.ndarray):
        return np.log(np.maximum(x, min_value))
    return math.log(max(x, min_value))


def safe_exp(
    x: Union[float, np.ndarray],
    max_exp: float = Tolerance.EXP_MAX,
) -> Union[float, np.ndarray]:
    """
    Compute exponential with overflow/underflow protection.
    
    Prevents overflow for large positive exponents and underflow
    for large negative exponents by clamping the input.
    
    Args:
        x: Exponent value(s). Can be scalar or numpy array.
        max_exp: Maximum absolute exponent value (default: 700).
    
    Returns:
        exp(clip(x, -max_exp, max_exp)).
    
    Examples:
        >>> safe_exp(0.0)
        1.0
        >>> safe_exp(1000)  # Returns exp(700) instead of overflow
        1.0142...e+304
        >>> safe_exp(-1000)  # Returns exp(-700) instead of underflow
        9.8596...e-305
    """
    if isinstance(x, np.ndarray):
        clipped = np.clip(x, -max_exp, max_exp)
        return np.exp(clipped)
    clipped = max(-max_exp, min(x, max_exp))
    return math.exp(clipped)


def safe_sqrt(
    x: Union[float, np.ndarray],
    min_value: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Compute square root with negative value protection.
    
    Prevents sqrt(negative) errors by clamping input to non-negative.
    
    Args:
        x: Input value(s). Can be scalar or numpy array.
        min_value: Minimum value for clamping (default: 0.0).
    
    Returns:
        Square root of max(x, min_value).
    
    Examples:
        >>> safe_sqrt(4.0)
        2.0
        >>> safe_sqrt(-1.0)  # Returns 0.0 instead of error
        0.0
        >>> safe_sqrt(-0.0001)  # Small negative due to float precision
        0.0
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(x, min_value))
    return math.sqrt(max(x, min_value))


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    fallback: float = 0.0,
    min_denominator: float = Tolerance.DENOMINATOR_MIN,
) -> Union[float, np.ndarray]:
    """
    Perform division with zero denominator protection.
    
    Returns a fallback value when denominator is too small,
    preventing division by zero errors.
    
    Args:
        numerator: Numerator value(s).
        denominator: Denominator value(s).
        fallback: Value to return when denominator is too small (default: 0.0).
        min_denominator: Minimum absolute denominator (default: 1e-10).
    
    Returns:
        numerator / denominator if |denominator| >= min_denominator,
        otherwise returns fallback.
    
    Examples:
        >>> safe_divide(10.0, 2.0)
        5.0
        >>> safe_divide(10.0, 0.0)
        0.0
        >>> safe_divide(10.0, 1e-15, fallback=float('inf'))
        inf
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(
            np.abs(denominator) >= min_denominator,
            np.asarray(numerator) / denominator,
            fallback,
        )
        return result
    
    if abs(denominator) < min_denominator:
        return fallback
    return numerator / denominator


def safe_power(
    base: Union[float, np.ndarray],
    exponent: Union[float, np.ndarray],
    max_result: float = 1e308,
) -> Union[float, np.ndarray]:
    """
    Compute power with overflow protection.
    
    Handles edge cases like 0**0, negative bases with fractional exponents,
    and overflow from large results.
    
    Args:
        base: Base value(s).
        exponent: Exponent value(s).
        max_result: Maximum result value to prevent overflow (default: 1e308).
    
    Returns:
        base ** exponent, clipped to [-max_result, max_result].
    
    Examples:
        >>> safe_power(2.0, 10.0)
        1024.0
        >>> safe_power(0.0, 0.0)  # Mathematical convention: 0^0 = 1
        1.0
        >>> safe_power(10.0, 1000.0)  # Clipped to max_result
        1e+308
    """
    if isinstance(base, np.ndarray) or isinstance(exponent, np.ndarray):
        base_arr = np.asarray(base)
        exp_arr = np.asarray(exponent)
        # Handle 0^0 case
        result = np.where(
            (base_arr == 0) & (exp_arr == 0),
            1.0,
            np.power(np.abs(base_arr) + 1e-300, exp_arr),
        )
        return np.clip(result, -max_result, max_result)
    
    # Handle 0^0 case
    if base == 0 and exponent == 0:
        return 1.0
    
    # Handle negative base with fractional exponent
    if base < 0 and exponent != int(exponent):
        # Use absolute value to avoid complex result
        base = abs(base)
    
    try:
        result = base ** exponent
        return max(-max_result, min(result, max_result))
    except (OverflowError, ValueError):
        return max_result if exponent > 0 else 0.0


def clip_to_range(
    x: Union[float, np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Clip values to a specified range.
    
    Convenience wrapper around numpy clip that works with scalars.
    
    Args:
        x: Input value(s).
        min_val: Minimum value (None for no lower bound).
        max_val: Maximum value (None for no upper bound).
    
    Returns:
        Value(s) clipped to [min_val, max_val].
    
    Examples:
        >>> clip_to_range(1.5, 0.0, 1.0)
        1.0
        >>> clip_to_range(-0.5, 0.0, 1.0)
        0.0
        >>> clip_to_range(0.5, 0.0, 1.0)
        0.5
    """
    if isinstance(x, np.ndarray):
        return np.clip(x, min_val, max_val)
    
    result = x
    if min_val is not None and result < min_val:
        result = min_val
    if max_val is not None and result > max_val:
        result = max_val
    return result
