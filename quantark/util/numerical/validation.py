"""
Numerical validation utilities.

This module provides functions for validating numerical inputs,
checking for NaN/Inf values, and ensuring numerical stability
in financial calculations.
"""

import math
from typing import Union, Optional, Sequence, Tuple
import numpy as np

from .constants import Tolerance
from ..exceptions import ValidationError, NumericalError


def is_valid_number(x: float) -> bool:
    """
    Check if a value is a valid finite number.
    
    Args:
        x: Value to check.
    
    Returns:
        True if x is finite (not NaN and not Inf).
    
    Examples:
        >>> is_valid_number(1.0)
        True
        >>> is_valid_number(float('nan'))
        False
        >>> is_valid_number(float('inf'))
        False
    """
    return math.isfinite(x)


def is_finite(x: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
    """
    Check if value(s) are finite.
    
    Works with both scalars and numpy arrays.
    
    Args:
        x: Value(s) to check.
    
    Returns:
        Boolean or boolean array indicating finite values.
    
    Examples:
        >>> is_finite(1.0)
        True
        >>> is_finite(np.array([1.0, float('nan'), float('inf')]))
        array([ True, False, False])
    """
    if isinstance(x, np.ndarray):
        return np.isfinite(x)
    return math.isfinite(x)


def validate_positive(
    x: float,
    name: str = "value",
    allow_zero: bool = False,
    tol: float = Tolerance.ZERO,
) -> float:
    """
    Validate that a value is positive.
    
    Args:
        x: Value to validate.
        name: Name of the parameter for error messages.
        allow_zero: If True, zero is considered valid (default: False).
        tol: Tolerance for zero check (default: 1e-10).
    
    Returns:
        The validated value.
    
    Raises:
        ValidationError: If value is not positive (or non-negative if allow_zero).
    
    Examples:
        >>> validate_positive(5.0, "strike")
        5.0
        >>> validate_positive(0.0, "rate", allow_zero=True)
        0.0
        >>> validate_positive(-1.0, "volatility")
        Raises ValidationError
    """
    if not is_valid_number(x):
        raise ValidationError(f"{name} must be a valid number, got {x}")
    
    if allow_zero:
        if x < -tol:
            raise ValidationError(f"{name} must be non-negative, got {x}")
    else:
        if x <= tol:
            raise ValidationError(f"{name} must be positive, got {x}")
    
    return x


def validate_non_negative(
    x: float,
    name: str = "value",
    tol: float = Tolerance.ZERO,
) -> float:
    """
    Validate that a value is non-negative.
    
    Convenience wrapper around validate_positive with allow_zero=True.
    
    Args:
        x: Value to validate.
        name: Name of the parameter for error messages.
        tol: Tolerance for zero check (default: 1e-10).
    
    Returns:
        The validated value.
    
    Raises:
        ValidationError: If value is negative.
    
    Examples:
        >>> validate_non_negative(0.0, "price")
        0.0
        >>> validate_non_negative(5.0, "quantity")
        5.0
    """
    return validate_positive(x, name, allow_zero=True, tol=tol)


def validate_in_range(
    x: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
    inclusive: Tuple[bool, bool] = (True, True),
) -> float:
    """
    Validate that a value is within a specified range.
    
    Args:
        x: Value to validate.
        min_val: Minimum allowed value (None for no lower bound).
        max_val: Maximum allowed value (None for no upper bound).
        name: Name of the parameter for error messages.
        inclusive: Tuple of (lower_inclusive, upper_inclusive).
    
    Returns:
        The validated value.
    
    Raises:
        ValidationError: If value is outside the range.
    
    Examples:
        >>> validate_in_range(0.5, 0.0, 1.0, "correlation")
        0.5
        >>> validate_in_range(1.5, 0.0, 1.0, "probability")
        Raises ValidationError
    """
    if not is_valid_number(x):
        raise ValidationError(f"{name} must be a valid number, got {x}")
    
    lower_inclusive, upper_inclusive = inclusive
    
    if min_val is not None:
        if lower_inclusive:
            if x < min_val:
                raise ValidationError(
                    f"{name} must be >= {min_val}, got {x}"
                )
        else:
            if x <= min_val:
                raise ValidationError(
                    f"{name} must be > {min_val}, got {x}"
                )
    
    if max_val is not None:
        if upper_inclusive:
            if x > max_val:
                raise ValidationError(
                    f"{name} must be <= {max_val}, got {x}"
                )
        else:
            if x >= max_val:
                raise ValidationError(
                    f"{name} must be < {max_val}, got {x}"
                )
    
    return x


def validate_probability(
    x: float,
    name: str = "probability",
) -> float:
    """
    Validate that a value is a valid probability [0, 1].
    
    Args:
        x: Value to validate.
        name: Name of the parameter for error messages.
    
    Returns:
        The validated value.
    
    Raises:
        ValidationError: If value is not in [0, 1].
    
    Examples:
        >>> validate_probability(0.95, "confidence_level")
        0.95
        >>> validate_probability(1.5, "probability")
        Raises ValidationError
    """
    return validate_in_range(x, 0.0, 1.0, name)


def validate_array(
    arr: Union[Sequence[float], np.ndarray],
    name: str = "array",
    allow_nan: bool = False,
    allow_inf: bool = False,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Validate a numerical array.
    
    Args:
        arr: Array to validate.
        name: Name of the parameter for error messages.
        allow_nan: If True, allow NaN values (default: False).
        allow_inf: If True, allow infinite values (default: False).
        min_length: Minimum array length (None for no minimum).
        max_length: Maximum array length (None for no maximum).
    
    Returns:
        The validated array as numpy array.
    
    Raises:
        ValidationError: If array fails validation.
    
    Examples:
        >>> validate_array([1.0, 2.0, 3.0], "prices")
        array([1., 2., 3.])
        >>> validate_array([1.0, float('nan')], "values")
        Raises ValidationError
    """
    arr = np.asarray(arr)
    
    if min_length is not None and len(arr) < min_length:
        raise ValidationError(
            f"{name} must have at least {min_length} elements, got {len(arr)}"
        )
    
    if max_length is not None and len(arr) > max_length:
        raise ValidationError(
            f"{name} must have at most {max_length} elements, got {len(arr)}"
        )
    
    if not allow_nan and np.any(np.isnan(arr)):
        raise ValidationError(f"{name} contains NaN values")
    
    if not allow_inf and np.any(np.isinf(arr)):
        raise ValidationError(f"{name} contains infinite values")
    
    return arr


def check_numerical_stability(
    value: float,
    name: str = "value",
    max_abs_value: float = 1e100,
    check_moneyness: bool = False,
    moneyness_threshold: float = Tolerance.MONEYNESS_MAX,
) -> None:
    """
    Check for numerical stability issues.
    
    Raises NumericalError if the value indicates potential numerical
    instability (e.g., extreme values, near-overflow conditions).
    
    Args:
        value: Value to check.
        name: Name of the value for error messages.
        max_abs_value: Maximum absolute value allowed.
        check_moneyness: If True, check for extreme log-moneyness.
        moneyness_threshold: Threshold for extreme moneyness (default: 100).
    
    Raises:
        NumericalError: If numerical instability is detected.
    
    Examples:
        >>> check_numerical_stability(1e50, "intermediate_value")  # OK
        >>> check_numerical_stability(1e200, "price")
        Raises NumericalError
    """
    if not is_valid_number(value):
        raise NumericalError(
            f"Numerical instability: {name} is not finite ({value})"
        )
    
    if abs(value) > max_abs_value:
        raise NumericalError(
            f"Numerical instability: {name} = {value} exceeds maximum "
            f"allowed value of {max_abs_value}"
        )
    
    if check_moneyness and abs(value) > moneyness_threshold:
        raise NumericalError(
            f"Extreme moneyness detected: {name} = {value} "
            f"(threshold: {moneyness_threshold})"
        )
