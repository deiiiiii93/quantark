"""
One-touch option implementation.

One-touch options are binary/digital options that pay a fixed rebate
if the underlying price touches a barrier at any point before expiry.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from ..base_equity_product import BaseEquityProduct
from .observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.settlement import SettlementConvention
from quantark.util.enum import (
    BarrierDirection,
    ObservationType,
    ObservationAggregation,
    TouchType,
)
from quantark.util.exceptions import ValidationError


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
    observation_schedule: Optional[ObservationSchedule] = None
    exercise_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    settlement_convention: Optional[SettlementConvention] = None

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
        observation_schedule: Optional[ObservationSchedule] = None,
        settlement_convention: Optional[SettlementConvention] = None,
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
        self.observation_schedule = observation_schedule
        self.exercise_date = exercise_date
        self.settlement_date = settlement_date
        self.settlement_convention = settlement_convention

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
        if (
            self.settlement_convention is not None
            and not isinstance(self.settlement_convention, SettlementConvention)
        ):
            raise ValidationError(
                "settlement_convention must be SettlementConvention or None"
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
                        barrier=rec.barrier if rec.barrier is not None else self.barrier,
                        payoff=rec.payoff if rec.payoff is not None else self.rebate,
                        return_rate=rec.return_rate,
                        is_rate_annualized=rec.is_rate_annualized,
                        initial_date=rec.initial_date,
                        settlement_date=rec.settlement_date,
                        settlement_time=rec.settlement_time,
                        maturity_date=rec.maturity_date,
                        day_count_convention=rec.day_count_convention,
                        tenor_end=rec.tenor_end,
                        day_count_fraction=rec.day_count_fraction,
                    )
                    for rec in self.observation_schedule.records
                ],
                aggregation_mode=self.observation_schedule.aggregation_mode,
                frequency=self.observation_schedule.frequency,
            )
            normalized_schedule.validate(require_single=True)
            self.observation_schedule = normalized_schedule
            if self.observation_schedule.times:
                self.observation_dates = self.observation_schedule.times
            self.observation_type = ObservationType.DISCRETE
        elif self.observation_type == ObservationType.DISCRETE:
            self.observation_schedule = ObservationSchedule.from_legacy(
                observation_dates=self.observation_dates or [],
                default_barrier=self.barrier,
                default_payoff=self.rebate,
                aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
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
            from quantark.util.calendar import calculate_year_fraction

            return calculate_year_fraction(
                pricing_env.valuation_date,
                self.exercise_date,
                pricing_env.day_count_convention,
                pricing_env.bus_days_in_year,
                calendar=getattr(pricing_env, "calendar", None),
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
