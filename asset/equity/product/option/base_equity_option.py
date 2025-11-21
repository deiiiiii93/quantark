"""
Base class for equity options.
"""

from abc import abstractmethod
from dataclasses import dataclass
from ..base_equity_product import BaseEquityProduct
from util.enum import OptionType, ExerciseType
from util.exceptions import ValidationError


@dataclass
class BaseEquityOption(BaseEquityProduct):
    """
    Abstract base class for equity options.

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
        exercise_type: EUROPEAN, AMERICAN, or BERMUDAN
    """

    strike: float
    maturity: float
    option_type: OptionType
    exercise_type: ExerciseType

    def __post_init__(self):
        """Validate option parameters."""
        self.validate()

    def validate(self) -> None:
        """
        Validate option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if self.maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option type: {self.option_type}")
        if not isinstance(self.exercise_type, ExerciseType):
            raise ValidationError(f"Invalid exercise type: {self.exercise_type}")

    def get_maturity(self) -> float:
        """
        Get time to maturity in years.

        Returns:
            Time to maturity
        """
        return self.maturity

    @abstractmethod
    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at maturity.

        Args:
            spot: Spot price at maturity

        Returns:
            Option payoff
        """
        pass

    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.option_type == OptionType.CALL

    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.option_type == OptionType.PUT

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.option_type}, K={self.strike:.2f}, T={self.maturity:.2f})"
        )
