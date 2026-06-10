"""
Basis yield representations for futures contracts.

The annualized basis yield is defined as:
    b = (F - S) / S / T

where:
    F = futures price
    S = spot price
    T = time to maturity (year fraction)

In cost-of-carry model: F = S * exp((r - q) * T)
Therefore: b = r - q (continuous compounding)

This module provides classes for representing basis yield which captures
the difference between futures price and spot price, normalized by time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Union  # noqa: F401
import numpy as np
from quantark.util.exceptions import ValidationError


class BasisRelationshipMode(Enum):
    """
    Defines how basis-dividend-rate relationship is handled during stress.

    In the cost-of-carry model: b = r - q
    When one parameter changes, this enum defines how others adjust.
    """

    INDEPENDENT = "independent"
    """Allow basis and dividend yield to move independently (may create arbitrage)."""

    AUTO_ADJUST_DIVIDEND = "auto_adjust_dividend"
    """When basis changes, automatically adjust dividend yield (assumes r fixed)."""

    AUTO_ADJUST_BASIS = "auto_adjust_basis"
    """When dividend yield changes, automatically adjust basis (assumes r fixed)."""


class BasisYield(ABC):
    """
    Abstract base class for annualized basis yield.

    Annualized basis yield represents the normalized difference between
    futures price and spot price:

        b = (F - S) / S / T

    In continuous compounding: F = S * exp(b * T)

    For index futures: b ≈ r - q (risk-free rate minus dividend yield)
    """

    @abstractmethod
    def get_basis_yield(self, time_to_maturity: float) -> float:
        """
        Get annualized basis yield for given maturity.

        Args:
            time_to_maturity: Time to maturity in years

        Returns:
            Annualized basis yield (continuously compounded)
        """
        pass

    def calculate_futures_price(
        self,
        spot: float,
        time_to_maturity: float
    ) -> float:
        """
        Calculate theoretical futures price from spot and basis yield.

        Args:
            spot: Current spot price
            time_to_maturity: Time to maturity in years

        Returns:
            Theoretical futures price: F = S * exp(b * T)
        """
        basis = self.get_basis_yield(time_to_maturity)
        return spot * np.exp(basis * time_to_maturity)

    def implied_from_prices(
        self,
        spot: float,
        futures_price: float,
        time_to_maturity: float
    ) -> float:
        """
        Calculate implied annualized basis yield from observed prices.

        Args:
            spot: Current spot price
            futures_price: Observed futures price
            time_to_maturity: Time to maturity in years

        Returns:
            Implied annualized basis yield

        Raises:
            ValidationError: If prices are invalid or time is zero
        """
        if spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {spot}")
        if futures_price <= 0:
            raise ValidationError(f"Futures price must be positive, got {futures_price}")
        if time_to_maturity <= 0:
            raise ValidationError(
                f"Time to maturity must be positive for basis calculation, got {time_to_maturity}"
            )

        # b = ln(F/S) / T
        return np.log(futures_price / spot) / time_to_maturity


@dataclass
class FlatBasisYield(BasisYield):
    """
    Constant annualized basis yield.

    Models basis as a constant yield across all maturities, suitable for
    simple futures pricing where the basis is assumed constant.

    Attributes:
        basis_yield: Constant annualized basis yield (continuously compounded)
    """

    basis_yield: float

    def __post_init__(self):
        """Validate basis yield."""
        if not np.isfinite(self.basis_yield):
            raise ValidationError(f"Basis yield must be finite, got {self.basis_yield}")
        # Basis can be negative (when q > r) or positive (when r > q)
        # Reasonable range check: -50% to +50%
        if abs(self.basis_yield) > 0.50:
            raise ValidationError(
                f"Basis yield seems unreasonably high: {self.basis_yield:.2%}"
            )

    def get_basis_yield(self, time_to_maturity: float) -> float:
        """
        Return constant basis yield.

        Args:
            time_to_maturity: Time to maturity (ignored)

        Returns:
            Constant basis yield
        """
        return self.basis_yield

    def __repr__(self):
        return f"FlatBasisYield(basis={self.basis_yield:.4f})"


@dataclass
class TermStructureBasisYield(BasisYield):
    """
    Term-structure basis yield (by maturity).

    Provides a time-dependent basis yield via linear interpolation.
    Useful for modeling futures curves with different maturities.

    Attributes:
        times: List of times (in years) for interpolation nodes
        yields: List of basis yields corresponding to each time
    """

    times: list[float]
    yields: list[float]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.yields):
            raise ValidationError("times and yields must have the same length.")
        if len(self.times) < 2:
            raise ValidationError("times must have at least 2 points.")
        if any(t <= 0 for t in self.times):
            raise ValidationError("times must be positive.")
        if any(self.times[i] >= self.times[i + 1] for i in range(len(self.times) - 1)):
            raise ValidationError("times must be strictly increasing.")
        if any(not np.isfinite(y) for y in self.yields):
            raise ValidationError("basis yields must be finite.")

    def get_basis_yield(self, time_to_maturity: float) -> float:
        """
        Get interpolated basis yield.

        Uses linear interpolation between nodes. Extrapolates flat
        outside the defined range.

        Args:
            time_to_maturity: Time to maturity in years

        Returns:
            Interpolated basis yield
        """
        t = float(time_to_maturity)
        if t <= self.times[0]:
            return float(self.yields[0])
        if t >= self.times[-1]:
            return float(self.yields[-1])
        return float(np.interp(t, self.times, self.yields))

    def __repr__(self):
        return f"TermStructureBasisYield(points={len(self.times)})"


@dataclass
class ZeroBasis(BasisYield):
    """
    Zero basis yield.

    Convenience class representing zero basis, meaning futures price
    equals spot price (F = S). This is rarely realistic but useful
    for testing or when basis is negligible.
    """

    def get_basis_yield(self, time_to_maturity: float) -> float:
        """
        Return zero basis yield.

        Args:
            time_to_maturity: Time to maturity (ignored)

        Returns:
            Zero
        """
        return 0.0

    def __repr__(self):
        return "ZeroBasis()"


def calculate_basis_from_rate_dividend(
    rate: float,
    dividend_yield: float
) -> float:
    """
    Calculate basis yield from rate and dividend yield.

    In the cost-of-carry model: b = r - q

    Args:
        rate: Risk-free interest rate (continuously compounded)
        dividend_yield: Dividend yield (continuously compounded)

    Returns:
        Annualized basis yield
    """
    return rate - dividend_yield


def calculate_dividend_from_rate_basis(
    rate: float,
    basis_yield: float
) -> float:
    """
    Calculate dividend yield from rate and basis yield.

    Rearranging the cost-of-carry model: q = r - b

    Args:
        rate: Risk-free interest rate (continuously compounded)
        basis_yield: Annualized basis yield

    Returns:
        Dividend yield (continuously compounded)
    """
    return rate - basis_yield


def validate_basis_dividend_consistency(
    rate: float,
    dividend_yield: float,
    basis_yield: float,
    tolerance: float = 1e-4
) -> bool:
    """
    Validate that basis, dividend yield, and rate are consistent.

    Checks if: b ≈ r - q (within tolerance)

    Args:
        rate: Risk-free interest rate
        dividend_yield: Dividend yield
        basis_yield: Annualized basis yield
        tolerance: Allowable deviation from the identity

    Returns:
        True if consistent, False otherwise
    """
    expected_basis = rate - dividend_yield
    return abs(basis_yield - expected_basis) <= tolerance
