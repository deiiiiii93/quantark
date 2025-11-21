"""
Base class for equity derivative products.
"""

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime


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

    def __repr__(self):
        return f"{self.__class__.__name__}()"
