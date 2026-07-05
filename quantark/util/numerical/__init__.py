"""
Numerical utilities for the QuantArk library.

This module provides centralized numerical operations commonly needed
in quantitative finance calculations, including:
- Tolerance constants for numerical comparisons
- Float comparison utilities with configurable precision
- Safe mathematical operations with overflow/underflow protection
- Number formatting for financial display
- Validation helpers for numerical inputs
"""

from .constants import Tolerance, FinancialConstants
from .comparison import (
    is_close,
    is_greater_than,
    is_less_than,
    is_greater_than_or_close,
    is_less_than_or_close,
    is_zero,
    is_positive,
    is_negative,
    is_non_negative,
    is_non_positive,
    almost_equal,
    compare_arrays,
)
from .safe_math import (
    safe_log,
    safe_exp,
    safe_sqrt,
    safe_divide,
    safe_power,
    clip_to_range,
)
from .formatting import (
    format_currency,
    format_percentage,
    format_basis_points,
    format_price,
    format_greeks,
    format_rate,
    format_volatility,
    format_with_sign,
    format_scientific,
    NumberFormatter,
)
from .finite_difference import (
    fd1_nonuniform,
    fd2_nonuniform,
)
from .tridiag import (
    solve_tridiag,
    solve_tridiag_batch,
)
from .validation import (
    is_valid_number,
    is_finite,
    validate_positive,
    validate_non_negative,
    validate_in_range,
    validate_probability,
    validate_array,
    check_numerical_stability,
)
from .pnl import pnl_pct_of_abs_baseline

__all__ = [
    # Constants
    'Tolerance',
    'FinancialConstants',
    # Comparison
    'is_close',
    'is_greater_than',
    'is_less_than',
    'is_greater_than_or_close',
    'is_less_than_or_close',
    'is_zero',
    'is_positive',
    'is_negative',
    'is_non_negative',
    'is_non_positive',
    'almost_equal',
    'compare_arrays',
    # Safe math
    'safe_log',
    'safe_exp',
    'safe_sqrt',
    'safe_divide',
    'safe_power',
    'clip_to_range',
    # Formatting
    'format_currency',
    'format_percentage',
    'format_basis_points',
    'format_price',
    'format_greeks',
    'format_rate',
    'format_volatility',
    'format_with_sign',
    'format_scientific',
    'NumberFormatter',
    # Finite differences
    'fd1_nonuniform',
    'fd2_nonuniform',
    # Tridiagonal solvers
    'solve_tridiag',
    'solve_tridiag_batch',
    # Validation
    'is_valid_number',
    'is_finite',
    'validate_positive',
    'validate_non_negative',
    'validate_in_range',
    'validate_probability',
    'validate_array',
    'check_numerical_stability',
    # P&L helpers
    'pnl_pct_of_abs_baseline',
]
