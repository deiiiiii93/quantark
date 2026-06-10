"""
Base class for equity derivative products.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from priceenv import PricingEnvironment


class BaseEquityProduct(ABC):
    """
    Abstract base class for all equity derivative products.

    This class defines the common interface that all equity derivatives must implement.
    """

    @abstractmethod
    def get_payoff(self, spot: float) -> float:
        """
        Calculate the payoff of the product given a spot price.

        Args:
            spot: Spot price at evaluation

        Returns:
            Payoff value
        """
        pass

    @abstractmethod
    def get_maturity(self) -> float:
        """
        Get time to maturity in years.

        Returns:
            Time to maturity
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """
        Validate product parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        pass

    @property
    def is_linear(self) -> bool:
        """Return True for linear (delta-one) products."""
        return False

    def time_shift(
        self,
        time_bump: float,
        bumped_date: datetime,
        pricing_env: "PricingEnvironment",
    ) -> bool:
        """
        Shift product state for theta bumping.

        Returns True if all observations were dropped.
        """
        return False

    def __repr__(self):
        return f"{self.__class__.__name__}()"
