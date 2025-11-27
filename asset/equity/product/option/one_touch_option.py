"""
One-touch option implementation.

One-touch options are binary/digital options that pay a fixed rebate
if the underlying price touches a barrier at any point before expiry.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from ..base_equity_product import BaseEquityProduct
from util.enum import BarrierDirection, ObservationType, TouchType
from util.exceptions import ValidationError


@dataclass
class OneTouchOption(BaseEquityProduct):
    """
    One-touch (or no-touch) digital option.

    One-touch options pay a fixed rebate if the underlying price touches
    the barrier at any point during the option's life.

    No-touch options pay a fixed rebate if the barrier is NOT touched.

    Attributes:
        barrier: Barrier price level
        barrier_direction: UP (barrier above spot) or DOWN (barrier below spot)
        rebate: Amount paid when condition is satisfied (default: 1.0)
        maturity: Time to maturity in years
        payment_at_hit: If True, rebate paid immediately on touch;
                       If False, rebate paid at expiry
        touch_type: ONE_TOUCH or NO_TOUCH
        observation_type: CONTINUOUS or DISCRETE monitoring
        observation_dates: For discrete, list of observation times
    """

    barrier: float = 0.0
    barrier_direction: BarrierDirection = BarrierDirection.UP
    rebate: float = 1.0
    maturity: float = 0.0
    payment_at_hit: bool = True
    touch_type: TouchType = TouchType.ONE_TOUCH
    observation_type: ObservationType = ObservationType.CONTINUOUS
    observation_dates: Optional[List[float]] = None
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None

    def __init__(
        self,
        barrier: float,
        barrier_direction: BarrierDirection,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        rebate: float = 1.0,
        payment_at_hit: bool = True,
        touch_type: TouchType = TouchType.ONE_TOUCH,
        observation_type: ObservationType = ObservationType.CONTINUOUS,
        observation_dates: Optional[List[float]] = None,
    ):
        """
        Initialize one-touch option.

        Args:
            barrier: Barrier price level
            barrier_direction: UP or DOWN
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Expiration date (optional if maturity provided)
            settlement_date: Settlement date (optional)
            rebate: Payment amount on touch (default: 1.0)
            payment_at_hit: If True, pay on touch; if False, pay at expiry
            touch_type: ONE_TOUCH (pay if touched) or NO_TOUCH (pay if not touched)
            observation_type: CONTINUOUS or DISCRETE
            observation_dates: For discrete, list of observation times

        Raises:
            ValidationError: If parameters are invalid
        """
        # Handle maturity vs exercise_date
        if maturity is None and exercise_date is None:
            maturity = 0.0
        elif maturity is None:
            maturity = 0.0

        self.barrier = barrier
        self.barrier_direction = barrier_direction
        self.rebate = rebate
        self.maturity = maturity
        self.payment_at_hit = payment_at_hit
        self.touch_type = touch_type
        self.observation_type = observation_type
        self.observation_dates = observation_dates
        self.exercise_date = exercise_date
        self.settlement_date = settlement_date

        self.validate()

    def validate(self) -> None:
        """
        Validate one-touch option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")

        if self.rebate < 0:
            raise ValidationError(f"Rebate must be non-negative, got {self.rebate}")

        if not isinstance(self.barrier_direction, BarrierDirection):
            raise ValidationError(
                f"Invalid barrier direction: {self.barrier_direction}"
            )

        if not isinstance(self.touch_type, TouchType):
            raise ValidationError(f"Invalid touch type: {self.touch_type}")

        # Validate maturity
        has_dates = self.exercise_date is not None
        has_maturity = self.maturity is not None and self.maturity > 0

        if not has_dates and not has_maturity:
            raise ValidationError("Either maturity or exercise_date must be provided")

        # For discrete observation, must have observation dates
        if (
            self.observation_type == ObservationType.DISCRETE
            and (self.observation_dates is None or len(self.observation_dates) == 0)
        ):
            raise ValidationError(
                "Observation dates required for discrete barrier monitoring"
            )

    def get_maturity(self, pricing_env=None) -> float:
        """
        Get time to maturity in years.

        Args:
            pricing_env: Pricing environment (required if using dates)

        Returns:
            Time to maturity in years
        """
        if self.exercise_date is not None:
            if pricing_env is None:
                raise ValidationError(
                    "PricingEnvironment required for date-based maturity calculation"
                )
            from util.calendar import calculate_year_fraction

            return calculate_year_fraction(
                pricing_env.valuation_date,
                self.exercise_date,
                pricing_env.day_count_convention,
                pricing_env.bus_days_in_year
            )
        else:
            return self.maturity

    def get_payoff(self, spot: float, touched: bool = False) -> float:
        """
        Calculate the option payoff.

        For one-touch: pays rebate if touched
        For no-touch: pays rebate if NOT touched

        Args:
            spot: Spot price (not used for one-touch, kept for interface)
            touched: Whether the barrier was touched

        Returns:
            Option payoff
        """
        if self.touch_type == TouchType.ONE_TOUCH:
            return self.rebate if touched else 0.0
        else:  # NO_TOUCH
            return self.rebate if not touched else 0.0

    @property
    def is_one_touch(self) -> bool:
        """Check if this is a one-touch option."""
        return self.touch_type == TouchType.ONE_TOUCH

    @property
    def is_no_touch(self) -> bool:
        """Check if this is a no-touch option."""
        return self.touch_type == TouchType.NO_TOUCH

    @property
    def is_up_barrier(self) -> bool:
        """Check if barrier is above spot."""
        return self.barrier_direction == BarrierDirection.UP

    @property
    def is_down_barrier(self) -> bool:
        """Check if barrier is below spot."""
        return self.barrier_direction == BarrierDirection.DOWN

    def is_barrier_hit(self, spot: float) -> bool:
        """
        Check if the barrier would be hit at a given spot price.

        Args:
            spot: Spot price to check

        Returns:
            True if barrier is hit
        """
        if self.is_up_barrier:
            return spot >= self.barrier
        else:
            return spot <= self.barrier

    def __repr__(self):
        touch_str = "OneTouch" if self.is_one_touch else "NoTouch"
        dir_str = "Up" if self.is_up_barrier else "Down"
        return (
            f"{touch_str}Option("
            f"B={self.barrier:.2f}, {dir_str}, "
            f"rebate={self.rebate:.2f}, T={self.maturity:.4f})"
        )

