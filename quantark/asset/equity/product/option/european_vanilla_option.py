"""
European vanilla option implementation.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
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

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        contract_multiplier: float = 1.0,
    ):
        """
        Initialize European vanilla option.

        Args:
            strike: Strike price
            option_type: CALL or PUT
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Date when option can be exercised (optional if maturity provided)
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
            exercise_type=ExerciseType.EUROPEAN,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            contract_multiplier=contract_multiplier,
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
            intrinsic = max(spot - self.strike, 0.0)
        else:  # put
            intrinsic = max(self.strike - spot, 0.0)

        return intrinsic * self.contract_multiplier

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
