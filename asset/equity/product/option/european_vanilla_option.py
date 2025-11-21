"""
European vanilla option implementation.
"""

from dataclasses import dataclass
from .base_equity_option import BaseEquityOption
from util.enum import OptionType, ExerciseType
from util.exceptions import ValidationError


@dataclass
class EuropeanVanillaOption(BaseEquityOption):
    """
    European vanilla call or put option.

    A European option can only be exercised at maturity.

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
    """

    def __init__(self, strike: float, maturity: float, option_type: OptionType):
        """
        Initialize European vanilla option.

        Args:
            strike: Strike price
            maturity: Time to maturity in years
            option_type: CALL or PUT
        """
        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
        )

    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at maturity.

        For a call: max(S - K, 0)
        For a put:  max(K - S, 0)

        Args:
            spot: Spot price at maturity

        Returns:
            Option payoff
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        if self.is_call():
            return max(spot - self.strike, 0.0)
        else:  # put
            return max(self.strike - spot, 0.0)

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate the intrinsic value of the option.

        Args:
            spot: Current spot price

        Returns:
            Intrinsic value
        """
        return self.get_payoff(spot)

    def __repr__(self):
        return (
            f"EuropeanVanillaOption("
            f"{self.option_type}, K={self.strike:.2f}, T={self.maturity:.4f})"
        )
