"""
Base class for equity options.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from ..base_equity_product import BaseEquityProduct
from util.enum import OptionType, ExerciseType
from util.exceptions import ValidationError


@dataclass
class BaseEquityOption(BaseEquityProduct):
    """
    Abstract base class for equity options.

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years (used if dates not provided)
        option_type: CALL or PUT
        exercise_type: EUROPEAN, AMERICAN, or BERMUDAN
        exercise_date: Date when option can be exercised (optional)
        settlement_date: Date when settlement occurs (optional)
    """

    strike: float
    maturity: float
    option_type: OptionType
    exercise_type: ExerciseType
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None

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
        
        # Validate that either maturity or dates are provided (not both)
        has_dates = self.exercise_date is not None
        has_maturity = self.maturity is not None and self.maturity > 0
        
        if not has_dates and not has_maturity:
            raise ValidationError("Either maturity or exercise_date must be provided")
        
        if has_dates and has_maturity:
            raise ValidationError("Cannot provide both maturity and exercise_date")
        
        if has_maturity and self.maturity <= 0:
            raise ValidationError(f"Maturity must be positive, got {self.maturity}")
        
        # Validate dates if provided
        if self.exercise_date is not None:
            if self.settlement_date is not None and self.settlement_date < self.exercise_date:
                raise ValidationError(
                    f"Settlement date ({self.settlement_date}) must be >= exercise date ({self.exercise_date})"
                )
        
        if not isinstance(self.option_type, OptionType):
            raise ValidationError(f"Invalid option type: {self.option_type}")
        if not isinstance(self.exercise_type, ExerciseType):
            raise ValidationError(f"Invalid exercise type: {self.exercise_type}")

    def get_maturity(self, pricing_env=None) -> float:
        """
        Get time to maturity in years.
        
        If exercise_date is provided, calculates maturity from pricing_env.valuation_date
        to exercise_date using the day count convention. Otherwise returns the maturity float.

        Args:
            pricing_env: Pricing environment (required if using date-based calculation)

        Returns:
            Time to maturity in years
            
        Raises:
            ValidationError: If date-based but no pricing_env provided
        """
        if self.exercise_date is not None:
            if pricing_env is None:
                raise ValidationError(
                    "PricingEnvironment required for date-based maturity calculation"
                )
            from util.calendar import calculate_year_fraction
            
            # Validate valuation date is before exercise date
            if pricing_env.valuation_date >= self.exercise_date:
                raise ValidationError(
                    f"Valuation date ({pricing_env.valuation_date}) must be before "
                    f"exercise date ({self.exercise_date})"
                )
            
            return calculate_year_fraction(
                pricing_env.valuation_date,
                self.exercise_date,
                pricing_env.day_count_convention,
                pricing_env.bus_days_in_year
            )
        else:
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
