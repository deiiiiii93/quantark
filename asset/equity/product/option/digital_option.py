"""
Cash-or-nothing European digital option implementation.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from .base_equity_option import BaseEquityOption
from util.enum import OptionType, ExerciseType
from util.exceptions import ValidationError


@dataclass
class CashOrNothingDigitalOption(BaseEquityOption):
    """
    European cash-or-nothing digital call or put.

    Pays a fixed cash amount if the terminal spot is on the paying side of
    the strike (S > K for calls, S < K for puts), otherwise pays zero.
    """

    payout: float = 0.0

    def __init__(
        self,
        strike: float,
        payout: float,
        option_type: OptionType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
    ):
        """
        Initialize cash-or-nothing digital option.

        Args:
            strike: Strike price
            payout: Fixed cash payout when in the money at maturity
            option_type: CALL or PUT
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Date when option can be exercised (optional if maturity provided)
            settlement_date: Date when settlement occurs (optional, defaults to exercise_date)

        Note:
            Either maturity OR exercise_date must be provided (not both).
        """
        if maturity is None and exercise_date is None:
            maturity = 0.0  # Will trigger validation error
        elif maturity is None:
            maturity = 0.0  # Placeholder when using dates

        self.payout = payout

        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
        )

    def validate(self) -> None:
        """Validate option parameters."""
        super().validate()

        if self.payout <= 0:
            raise ValidationError(f"Payout must be positive, got {self.payout}")

        if self.exercise_type != ExerciseType.EUROPEAN:
            raise ValidationError(
                f"CashOrNothingDigitalOption only supports European exercise, got {self.exercise_type}"
            )

    def get_payoff(self, spot: float) -> float:
        """
        Calculate the digital option payoff at maturity.

        For a call: payout if S > K else 0
        For a put:  payout if S < K else 0
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        if self.is_call():
            return self.payout if spot > self.strike else 0.0
        else:
            return self.payout if spot < self.strike else 0.0

    def intrinsic_value(self, spot: float) -> float:
        """Intrinsic value equals digital payoff."""
        return self.get_payoff(spot)

    def __repr__(self):
        return (
            f"CashOrNothingDigitalOption("
            f"{self.option_type}, K={self.strike:.4f}, P={self.payout:.4f}, T={self.maturity:.4f})"
        )

