"""
Numerical constants and tolerances for financial calculations.

This module centralizes tolerance values and mathematical constants
used throughout the QuantArk library for numerical stability and
consistent precision handling.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Tolerance:
    """
    Standard tolerance values for numerical comparisons in financial calculations.

    These values are chosen based on typical precision requirements in
    derivatives pricing and risk management applications.

    Attributes:
        ZERO: Tolerance for zero checks (e.g., time to expiry < ZERO).
        PRECISION: General numerical precision for float comparisons.
        RELATIVE: Default relative tolerance for is_close comparisons.
        ABSOLUTE: Default absolute tolerance for is_close comparisons.
        BUMP_SPOT: Default bump size for spot price (1%).
        BUMP_RATE: Default bump size for rates (1 basis point).
        BUMP_VOL: Default bump size for volatility (1 vol point).
        LOG_MIN: Minimum value to prevent log(0).
        EXP_MAX: Maximum exponent to prevent overflow.
        DENOMINATOR_MIN: Minimum denominator to prevent division by zero.
        MONEYNESS_MAX: Maximum log moneyness before flagging extreme values.
        DISCOUNT_EXP_MAX: Maximum exponent for discount factors.
    """

    # Basic tolerances
    ZERO: Final[float] = 1e-10
    PRECISION: Final[float] = 1e-6
    RELATIVE: Final[float] = 1e-9
    ABSOLUTE: Final[float] = 0.0

    # Bump sizes for Greeks and sensitivities
    BUMP_SPOT: Final[float] = 0.01  # 1% spot bump
    BUMP_RATE: Final[float] = 0.0001  # 1bp rate bump
    BUMP_VOL: Final[float] = 0.01  # 1 vol point bump

    # Numerical stability thresholds
    LOG_MIN: Final[float] = 1e-16
    EXP_MAX: Final[float] = 700.0
    DENOMINATOR_MIN: Final[float] = 1e-10

    # Financial calculation thresholds
    MONEYNESS_MAX: Final[float] = 100.0
    DISCOUNT_EXP_MAX: Final[float] = 100.0

    # Matrix/covariance tolerances
    EIGENVALUE_MIN: Final[float] = 1e-10
    CONDITION_NUMBER_MAX: Final[float] = 1e12


@dataclass(frozen=True)
class FinancialConstants:
    """
    Common financial constants used in calculations.

    Attributes:
        BASIS_POINTS_PER_PERCENT: Conversion factor from percent to basis points.
        PERCENT_TO_DECIMAL: Conversion factor from percent to decimal.
    """

    BASIS_POINTS_PER_PERCENT: Final[float] = 100.0
    PERCENT_TO_DECIMAL: Final[float] = 0.01

    @staticmethod
    def to_basis_points(decimal_value: float) -> float:
        """Convert decimal rate to basis points (e.g., 0.0025 -> 25bp)."""
        return decimal_value * 10000.0

    @staticmethod
    def from_basis_points(bp_value: float) -> float:
        """Convert basis points to decimal rate (e.g., 25bp -> 0.0025)."""
        return bp_value / 10000.0

    @staticmethod
    def to_percentage(decimal_value: float) -> float:
        """Convert decimal to percentage (e.g., 0.05 -> 5%)."""
        return decimal_value * 100.0

    @staticmethod
    def from_percentage(pct_value: float) -> float:
        """Convert percentage to decimal (e.g., 5% -> 0.05)."""
        return pct_value / 100.0


# Singleton instances for convenient access
TOLERANCE = Tolerance()
FINANCIAL = FinancialConstants()
