"""
Volatility surface representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from util.exceptions import ValidationError
from util.numerical import validate_positive


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
