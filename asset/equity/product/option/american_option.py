"""
American option implementation.

American options can be exercised at any time before maturity.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from .base_equity_option import BaseEquityOption
from util.enum import OptionType, ExerciseType
from util.exceptions import ValidationError


@dataclass
class AmericanOption(BaseEquityOption):
    """
    American call or put option.

    An American option can be exercised at any time up to and including
    the expiration date. This early exercise feature makes American options
    more valuable than their European counterparts (or at least as valuable).

    For calls on non-dividend-paying stocks, early exercise is never optimal,
    so American calls equal European calls. However, for puts or when there
    are dividends, early exercise may be optimal.

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
    """

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ):
        """
        Initialize American option.

        Args:
            strike: Strike price
            option_type: CALL or PUT
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Date when option expires (optional if maturity provided)
            settlement_date: Date when settlement occurs (optional, defaults to exercise_date)

        Note:
            Either maturity OR exercise_date must be provided (not both).
        """
        # Default maturity to 0 if not provided (will be validated)
        if maturity is None and exercise_date is None:
            maturity = 0.0  # Will trigger validation error
        elif maturity is None:
            maturity = 0.0  # Placeholder when using dates

        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.AMERICAN,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
        )

    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at exercise.

        For a call: max(S - K, 0)
        For a put:  max(K - S, 0)

        Args:
            spot: Spot price at exercise

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

        For American options, this is the value that could be obtained
        by exercising immediately.

        Args:
            spot: Current spot price

        Returns:
            Intrinsic value
        """
        return self.get_payoff(spot)

    def __repr__(self):
        return (
            f"AmericanOption("
            f"{self.option_type}, K={self.strike:.2f}, T={self.maturity:.4f})"
        )

