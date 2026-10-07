"""
Base class for equity derivative products.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantark.priceenv import PricingEnvironment


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
        *,
        keep_landing: bool = False,
    ) -> None:
        """
        Shift product state for a theta bump (spec 2026-10-07 theta boundary).

        A product with no observation schedule has no state to move here;
        ``keep_landing`` (keep an observation landing on the bumped date) is
        accepted for the common signature.
        """
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}()"
