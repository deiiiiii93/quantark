"""
Double barrier option implementation.

Double barrier options have both an upper and lower barrier.
The option knocks in or out if either barrier is breached.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from .base_equity_option import BaseEquityOption
from .observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import (
    OptionType,
    ExerciseType,
    DoubleBarrierType,
    ObservationType,
    ObservationAggregation,
)
from quantark.util.exceptions import ValidationError


@dataclass
class DoubleBarrierOption(BaseEquityOption):
    """
    Double barrier option (knock-in or knock-out with two barriers).

    Double barrier options have both an upper barrier (above spot) and
    a lower barrier (below spot). The option:
        - KNOCK_OUT: Deactivates if either barrier is hit
        - KNOCK_IN: Activates if either barrier is hit

    These are sometimes called "corridor options" or "range options".

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
        upper_barrier: Upper barrier price (above current spot)
        lower_barrier: Lower barrier price (below current spot)
        barrier_type: KNOCK_IN or KNOCK_OUT
        rebate: Amount paid if knocked out (or not knocked in)
        observation_type: CONTINUOUS or DISCRETE monitoring
        observation_dates: For discrete, list of observation times (year fractions)
    """

    upper_barrier: float = 0.0
    lower_barrier: float = 0.0
    barrier_type: DoubleBarrierType = DoubleBarrierType.KNOCK_OUT
    rebate: float = 0.0
    observation_type: ObservationType = ObservationType.CONTINUOUS
    observation_dates: Optional[List[float]] = None
    observation_schedule: Optional[ObservationSchedule] = None

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        upper_barrier: float,
        lower_barrier: float,
        barrier_type: DoubleBarrierType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        rebate: float = 0.0,
        observation_type: ObservationType = ObservationType.CONTINUOUS,
        observation_dates: Optional[List[float]] = None,
        observation_schedule: Optional[ObservationSchedule] = None,
        contract_multiplier: float = 1.0,
    ):
        """
        Initialize double barrier option.

        Args:
            strike: Strike price
            option_type: CALL or PUT
            upper_barrier: Upper barrier price
            lower_barrier: Lower barrier price
            barrier_type: KNOCK_IN or KNOCK_OUT
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Expiration date (optional if maturity provided)
            settlement_date: Settlement date (optional)
            rebate: Rebate amount (default: 0)
            observation_type: CONTINUOUS or DISCRETE (default: CONTINUOUS)
            observation_dates: For discrete, list of observation times

        Raises:
            ValidationError: If parameters are invalid
        """
        # Default maturity handling
        if maturity is None and exercise_date is None:
            maturity = 0.0
        elif maturity is None:
            maturity = 0.0

        # Store barrier parameters before calling super().__init__
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.barrier_type = barrier_type
        self.rebate = rebate
        self.observation_type = observation_type
        self.observation_dates = observation_dates
        self.observation_schedule = observation_schedule

        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            contract_multiplier=contract_multiplier,
        )

    def validate(self) -> None:
        """
        Validate double barrier option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        super().validate()

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

        if not isinstance(self.barrier_type, DoubleBarrierType):
            raise ValidationError(f"Invalid barrier type: {self.barrier_type}")

        if not isinstance(self.observation_type, ObservationType):
            raise ValidationError(f"Invalid observation type: {self.observation_type}")

        # Strike should be between barriers for knock-out to make sense
        if self.barrier_type == DoubleBarrierType.KNOCK_OUT:
            if not (self.lower_barrier < self.strike < self.upper_barrier):
                # This is a warning, not an error - could be intentional
                pass

        # For discrete observation, must have observation dates unless schedule supplied
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

    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at maturity.

        Note: This returns the vanilla payoff. The actual payoff depends on
        whether either barrier was hit during the option's life.

        Args:
            spot: Spot price at maturity

        Returns:
            Option payoff (call or put style)
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        if self.is_call():
            intrinsic = max(spot - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot, 0.0)

        return intrinsic * self.contract_multiplier

    def intrinsic_value(self, spot: float) -> float:
        """
        Calculate intrinsic value.

        Args:
            spot: Current spot price

        Returns:
            Intrinsic value
        """
        return self.get_payoff(spot)

    @property
    def is_knock_in(self) -> bool:
        """Check if this is a knock-in barrier."""
        return self.barrier_type == DoubleBarrierType.KNOCK_IN

    @property
    def is_knock_out(self) -> bool:
        """Check if this is a knock-out barrier."""
        return self.barrier_type == DoubleBarrierType.KNOCK_OUT

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
        Check if spot is within the corridor (between barriers).

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
        return (
            f"DoubleBarrierOption("
            f"{self.option_type}, K={self.strike:.2f}, "
            f"L={self.lower_barrier:.2f}, U={self.upper_barrier:.2f}, "
            f"{self.barrier_type}, T={self.maturity:.4f})"
        )
