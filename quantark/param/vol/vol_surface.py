"""
Volatility surface representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_divide, safe_sqrt, validate_positive
import numpy as np


class VolatilitySurface(ABC):
    """
    Abstract base class for volatility surfaces.

    Volatility surfaces provide implied volatility as a function of
    strike and time to maturity.
    """

    @abstractmethod
    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        Get implied volatility for given strike and maturity.

        Args:
            strike: Strike price
            time_to_maturity: Time to maturity in years
            spot: Current spot price

        Returns:
            Implied volatility (annualized)
        """
        pass


@dataclass
class FlatVolSurface(VolatilitySurface):
    """
    Flat (constant) volatility surface.

    Returns the same volatility regardless of strike and maturity.
    Suitable for Black-Scholes models with constant volatility.

    Attributes:
        volatility: Constant annualized volatility
    """

    volatility: float

    def __post_init__(self) -> None:
        """Validate volatility."""
        if not isinstance(self.volatility, (int, float)):
            raise ValidationError(f"Volatility must be numeric, got {self.volatility}")
        validate_positive(float(self.volatility), name="volatility")
        if self.volatility > 5.0:  # 500% vol - sanity check
            raise ValidationError(
                f"Volatility seems unreasonably high: {self.volatility}"
            )

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        Return constant volatility.

        Args:
            strike: Strike price (ignored)
            time_to_maturity: Time to maturity (ignored)
            spot: Spot price (ignored)

        Returns:
            Constant volatility
        """
        return self.volatility

    def __repr__(self):
        return f"FlatVolSurface(vol={self.volatility:.2%})"


@dataclass
class TermStructureVolSurface(VolatilitySurface):
    """
    Term-structure volatility surface (ATM by maturity).

    Provides a time-dependent volatility via total variance interpolation on maturity.
    Strike is ignored (ATM term structure).

    Attributes:
        times: Increasing maturities (year fractions).
        vols: Implied volatilities for each maturity.
    """

    times: list[float]
    vols: list[float]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.vols):
            raise ValidationError("times and vols must have the same length.")
        if len(self.times) < 2:
            raise ValidationError("times must have at least 2 points.")
        if any(t <= 0 for t in self.times):
            raise ValidationError("times must be positive.")
        if any(self.times[i] >= self.times[i + 1] for i in range(len(self.times) - 1)):
            raise ValidationError("times must be strictly increasing.")
        for v in self.vols:
            validate_positive(float(v), name="volatility")

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        t = float(time_to_maturity)
        if t <= self.times[0]:
            return float(self.vols[0])
        if t >= self.times[-1]:
            return float(self.vols[-1])
        total_variances = [float(v) ** 2 * float(tt) for v, tt in zip(self.vols, self.times)]
        interp_total_var = float(np.interp(t, self.times, total_variances))
        return float(safe_sqrt(safe_divide(interp_total_var, t)))

    def __repr__(self):
        return "TermStructureVolSurface(points=%d)" % len(self.times)
