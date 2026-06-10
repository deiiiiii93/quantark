"""
Barrier option implementation.

Barrier options have a payoff that depends on whether the underlying
price crosses a barrier level during the option's life.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from .base_equity_option import BaseEquityOption
from .observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import (
    OptionType,
    ExerciseType,
    BarrierType,
    ObservationType,
    ObservationAggregation,
)
from quantark.util.exceptions import ValidationError


@dataclass
class BarrierOption(BaseEquityOption):
    """
    Single barrier option (knock-in or knock-out).

    Barrier options are path-dependent options whose existence or payoff
    depends on whether the underlying asset price reaches a specified barrier.

    Types:
        - UP_IN: Activates (knocks in) when price goes UP above barrier
        - UP_OUT: Deactivates (knocks out) when price goes UP above barrier
        - DOWN_IN: Activates when price goes DOWN below barrier
        - DOWN_OUT: Deactivates when price goes DOWN below barrier

    Attributes:
        strike: Strike price
        maturity: Time to maturity in years
        option_type: CALL or PUT
        barrier: Barrier price level
        barrier_type: UP_IN, UP_OUT, DOWN_IN, or DOWN_OUT
        rebate: Amount paid if knocked out (or not knocked in at expiry)
        observation_type: CONTINUOUS or DISCRETE monitoring
        observation_dates: For discrete monitoring, list of observation times (year fractions)
    """

    barrier: float = 0.0
    barrier_type: BarrierType = BarrierType.DOWN_OUT
    rebate: float = 0.0
    participation_rate: float = 1.0
    pay_at_hit: bool = False
    observation_type: ObservationType = ObservationType.CONTINUOUS
    observation_dates: Optional[List[float]] = None
    observation_schedule: Optional[ObservationSchedule] = None

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        barrier: float,
        barrier_type: BarrierType,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        rebate: float = 0.0,
        participation_rate: float = 1.0,
        pay_at_hit: bool = False,
        observation_type: ObservationType = ObservationType.CONTINUOUS,
        observation_dates: Optional[List[float]] = None,
        observation_schedule: Optional[ObservationSchedule] = None,
        contract_multiplier: float = 1.0,
    ):
        """
        Initialize barrier option.

        Args:
            strike: Strike price
            option_type: CALL or PUT
            barrier: Barrier price level
            barrier_type: UP_IN, UP_OUT, DOWN_IN, or DOWN_OUT
            maturity: Time to maturity in years (optional if exercise_date provided)
            exercise_date: Expiration date (optional if maturity provided)
            settlement_date: Settlement date (optional)
            rebate: Rebate amount (default: 0)
            observation_type: CONTINUOUS or DISCRETE (default: CONTINUOUS)
            observation_dates: For discrete, list of observation times in year fractions

        Raises:
            ValidationError: If parameters are invalid
        """
        # Default maturity handling
        if maturity is None and exercise_date is None:
            maturity = 0.0
        elif maturity is None:
            maturity = 0.0

        # Store barrier parameters before calling super().__init__
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.rebate = rebate
        self.participation_rate = participation_rate
        self.pay_at_hit = pay_at_hit
        self.observation_type = observation_type
        self.observation_dates = observation_dates
        self.observation_schedule = observation_schedule

        super().__init__(
            strike=strike,
            maturity=maturity,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,  # Barrier options are typically European
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            contract_multiplier=contract_multiplier,
        )

    def validate(self) -> None:
        """
        Validate barrier option parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        super().validate()

        if self.barrier <= 0:
            raise ValidationError(f"Barrier must be positive, got {self.barrier}")

        if self.rebate < 0:
            raise ValidationError(f"Rebate must be non-negative, got {self.rebate}")
        if not isinstance(self.pay_at_hit, bool):
            raise ValidationError(f"pay_at_hit must be boolean, got {self.pay_at_hit}")
        if self.participation_rate <= 0:
            raise ValidationError(
                f"Participation rate must be positive, got {self.participation_rate}"
            )

        if not isinstance(self.barrier_type, BarrierType):
            raise ValidationError(f"Invalid barrier type: {self.barrier_type}")

        if not isinstance(self.observation_type, ObservationType):
            raise ValidationError(f"Invalid observation type: {self.observation_type}")

        # For discrete observation, must have observation dates
        if (
            self.observation_type == ObservationType.DISCRETE
            and self.observation_schedule is None
            and (self.observation_dates is None or len(self.observation_dates) == 0)
        ):
            raise ValidationError(
                "Observation dates required for discrete barrier monitoring"
            )

        # Validate observation dates if provided
        if self.observation_dates is not None:
            for t in self.observation_dates:
                if t < 0:
                    raise ValidationError(
                        f"Observation dates must be non-negative, got {t}"
                    )

        # Normalize observation schedule (preferred) or legacy dates for discrete monitoring
        if self.observation_schedule is not None:
            if self.observation_type in (
                ObservationType.CONTINUOUS,
                ObservationType.EXPIRY,
            ):
                raise ValidationError(
                    "ObservationSchedule requires DISCRETE observation_type."
                )
            normalized_schedule = ObservationSchedule(
                records=[
                    ObservationRecord(
                        observation_time=rec.observation_time,
                        observation_date=rec.observation_date,
                        barrier=(
                            rec.barrier if rec.barrier is not None else self.barrier
                        ),
                        payoff=rec.payoff if rec.payoff is not None else self.rebate,
                        return_rate=rec.return_rate,
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
        else:
            # EXPIRY monitoring does not use schedules or legacy dates
            self.observation_schedule = None
            self.observation_dates = self.observation_dates or []

    def get_payoff(self, spot: float) -> float:
        """
        Calculate the option payoff at maturity (assuming barrier not hit for knock-out
        or barrier hit for knock-in).

        Note: This returns the vanilla payoff. The actual payoff depends on
        whether the barrier was hit during the option's life.

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
        return self.barrier_type.is_knock_in

    @property
    def is_knock_out(self) -> bool:
        """Check if this is a knock-out barrier."""
        return self.barrier_type.is_knock_out

    @property
    def is_up_barrier(self) -> bool:
        """Check if barrier is above spot."""
        return self.barrier_type.is_up

    @property
    def is_down_barrier(self) -> bool:
        """Check if barrier is below spot."""
        return self.barrier_type.is_down

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
        else:  # down barrier
            return spot <= self.barrier

    @property
    def pay_at_expiry(self) -> bool:
        """Rebate paid at expiry if True (i.e., not at hit)."""
        return not self.pay_at_hit

    def __repr__(self):
        return (
            f"BarrierOption("
            f"{self.option_type}, K={self.strike:.2f}, "
            f"B={self.barrier:.2f}, {self.barrier_type}, T={self.maturity:.4f}, "
            f"rebate={self.rebate:.2f}, pay_at_hit={self.pay_at_hit}, "
            f"participation={self.participation_rate:.4f})"
        )
