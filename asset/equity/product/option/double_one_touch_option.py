"""
Double one-touch option implementation.

Double one-touch options have two barriers (upper and lower) and pay
a fixed rebate if either barrier is touched (or neither, for double no-touch).
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from ..base_equity_product import BaseEquityProduct
from .observation_schedule import ObservationRecord, ObservationSchedule
from util.enum import (
    ObservationType,
    ObservationAggregation,
    TouchType,
)
from util.exceptions import ValidationError


@dataclass
class DoubleOneTouchOption(BaseEquityProduct):
    """
    Double one-touch or double no-touch option.

    Double one-touch options pay a fixed rebate if EITHER barrier is touched.
    Double no-touch options pay a fixed rebate if NEITHER barrier is touched.

    These are also known as:
        - Double one-touch: "range" or "boundary" options
        - Double no-touch: "range binary" or "range accrual" options

    Attributes:
        upper_barrier: Upper barrier price
        lower_barrier: Lower barrier price
        rebate: Amount paid when condition is satisfied (default: 1.0)
        maturity: Time to maturity in years
        payment_at_hit: For one-touch, True = pay on hit, False = pay at expiry
        touch_type: DOUBLE_ONE_TOUCH or DOUBLE_NO_TOUCH
        observation_type: CONTINUOUS or DISCRETE monitoring
        observation_dates: For discrete, list of observation times
    """

    upper_barrier: float = 0.0
    lower_barrier: float = 0.0
    rebate: float = 1.0
    maturity: float = 0.0
    payment_at_hit: bool = True
    touch_type: TouchType = TouchType.DOUBLE_ONE_TOUCH
    observation_type: ObservationType = ObservationType.CONTINUOUS
    observation_dates: Optional[List[float]] = None
    observation_schedule: Optional[ObservationSchedule] = None
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None

    def __init__(
        self,
        upper_barrier: float,
        lower_barrier: float,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        rebate: float = 1.0,
        payment_at_hit: bool = True,
        touch_type: TouchType = TouchType.DOUBLE_ONE_TOUCH,
        observation_type: ObservationType = ObservationType.CONTINUOUS,
        observation_dates: Optional[List[float]] = None,
        observation_schedule: Optional[ObservationSchedule] = None,
    ):
        """
        Initialize double one-touch option.

        Args:
            upper_barrier: Upper barrier price
            lower_barrier: Lower barrier price
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Expiration date (optional if maturity provided)
            settlement_date: Settlement date (optional)
            rebate: Payment amount (default: 1.0)
            payment_at_hit: For one-touch, True = pay on hit (default: True)
            touch_type: DOUBLE_ONE_TOUCH or DOUBLE_NO_TOUCH
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

        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.rebate = rebate
        self.maturity = maturity
        self.payment_at_hit = payment_at_hit
        self.touch_type = touch_type
        self.observation_type = observation_type
        self.observation_dates = observation_dates
        self.observation_schedule = observation_schedule
        self.exercise_date = exercise_date
        self.settlement_date = settlement_date

        self.validate()

    def validate(self) -> None:
        """
        Validate double one-touch option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        if self.upper_barrier <= 0:
            raise ValidationError(
                f"Upper barrier must be positive, got {self.upper_barrier}"
            )

        if self.lower_barrier <= 0:
            raise ValidationError(
                f"Lower barrier must be positive, got {self.lower_barrier}"
            )

        if self.lower_barrier >= self.upper_barrier:
            raise ValidationError(
                f"Lower barrier ({self.lower_barrier}) must be less than "
                f"upper barrier ({self.upper_barrier})"
            )

        if self.rebate < 0:
            raise ValidationError(f"Rebate must be non-negative, got {self.rebate}")

        if self.touch_type not in (TouchType.DOUBLE_ONE_TOUCH, TouchType.DOUBLE_NO_TOUCH):
            raise ValidationError(
                f"Touch type must be DOUBLE_ONE_TOUCH or DOUBLE_NO_TOUCH, "
                f"got {self.touch_type}"
            )

        # Validate maturity
        has_dates = self.exercise_date is not None
        has_maturity = self.maturity is not None and self.maturity > 0

        if not has_dates and not has_maturity:
            raise ValidationError("Either maturity or exercise_date must be provided")

        # For discrete observation, must have observation dates
        if (
            self.observation_type == ObservationType.DISCRETE
            and self.observation_schedule is None
            and (self.observation_dates is None or len(self.observation_dates) == 0)
        ):
            raise ValidationError(
                "Observation dates required for discrete barrier monitoring"
            )

        # Normalize observation schedule (preferred) or legacy dates for discrete monitoring
        if self.observation_schedule is not None:
            if self.observation_type == ObservationType.CONTINUOUS:
                raise ValidationError("ObservationSchedule requires DISCRETE observation_type.")
            normalized_schedule = ObservationSchedule(
                records=[
                    ObservationRecord(
                        observation_time=rec.observation_time,
                        observation_date=rec.observation_date,
                        upper_barrier=rec.upper_barrier if rec.upper_barrier is not None else self.upper_barrier,
                        lower_barrier=rec.lower_barrier if rec.lower_barrier is not None else self.lower_barrier,
                        payoff=rec.payoff if rec.payoff is not None else self.rebate,
                        return_rate=rec.return_rate,
                    )
                    for rec in self.observation_schedule.records
                ],
                aggregation_mode=self.observation_schedule.aggregation_mode,
                frequency=self.observation_schedule.frequency,
            )
            normalized_schedule.validate(require_double=True)
            self.observation_schedule = normalized_schedule
            if self.observation_schedule.times:
                self.observation_dates = self.observation_schedule.times
            self.observation_type = ObservationType.DISCRETE
        elif self.observation_type == ObservationType.DISCRETE:
            self.observation_schedule = ObservationSchedule.from_legacy(
                observation_dates=self.observation_dates or [],
                default_barrier=None,
                default_payoff=self.rebate,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
                upper_barrier=self.upper_barrier,
                lower_barrier=self.lower_barrier,
            )
            self.observation_dates = self.observation_schedule.times

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

    def time_shift(self, time_bump: float, bumped_date: datetime, pricing_env) -> bool:
        """Shift observation schedule and maturity for theta bumping."""
        schedule = getattr(self, "observation_schedule", None)
        if schedule is not None:
            if schedule.uses_dates():
                pricing_env.valuation_date = bumped_date
            bumped_schedule = schedule.time_shift(time_bump, bumped_date)
            if bumped_schedule is None:
                return True
            self.observation_schedule = bumped_schedule
            if hasattr(self, "observation_dates") and bumped_schedule.uses_times():
                self.observation_dates = bumped_schedule.times

        if getattr(self, "exercise_date", None) is None:
            if getattr(self, "maturity", None) is not None:
                self.maturity -= time_bump
        else:
            pricing_env.valuation_date = bumped_date

        return False

    def get_payoff(self, spot: float, touched: bool = False) -> float:
        """
        Calculate the option payoff.

        For double one-touch: pays rebate if either barrier touched
        For double no-touch: pays rebate if neither barrier touched

        Args:
            spot: Spot price (not directly used)
            touched: Whether any barrier was touched

        Returns:
            Option payoff
        """
        if self.is_double_one_touch:
            return self.rebate if touched else 0.0
        else:  # Double no-touch
            return self.rebate if not touched else 0.0

    @property
    def is_double_one_touch(self) -> bool:
        """Check if this is a double one-touch option."""
        return self.touch_type == TouchType.DOUBLE_ONE_TOUCH

    @property
    def is_double_no_touch(self) -> bool:
        """Check if this is a double no-touch option."""
        return self.touch_type == TouchType.DOUBLE_NO_TOUCH

    def is_barrier_hit(self, spot: float) -> bool:
        """
        Check if either barrier would be hit at a given spot price.

        Args:
            spot: Spot price to check

        Returns:
            True if either barrier is hit
        """
        return spot >= self.upper_barrier or spot <= self.lower_barrier

    def is_in_corridor(self, spot: float) -> bool:
        """
        Check if spot is within the corridor.

        Args:
            spot: Spot price to check

        Returns:
            True if spot is between lower and upper barriers
        """
        return self.lower_barrier < spot < self.upper_barrier

    @property
    def corridor_width(self) -> float:
        """Get the width of the corridor in price terms."""
        return self.upper_barrier - self.lower_barrier

    @property
    def corridor_width_log(self) -> float:
        """Get the width of the corridor in log-price terms."""
        import math
        return math.log(self.upper_barrier / self.lower_barrier)

    def __repr__(self):
        touch_str = "DoubleOneTouch" if self.is_double_one_touch else "DoubleNoTouch"
        return (
            f"{touch_str}Option("
            f"L={self.lower_barrier:.2f}, U={self.upper_barrier:.2f}, "
            f"rebate={self.rebate:.2f}, T={self.maturity:.4f})"
        )
