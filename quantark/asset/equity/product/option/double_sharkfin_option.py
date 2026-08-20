"""
Double sharkfin option product definition.

Double sharkfin options are capped double-barrier structures. The product
knocks out if either the upper or lower barrier is hit. If no barrier is hit,
it pays a vanilla-style call or put payoff capped by the relevant barrier.
"""

from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import List, Optional, Sequence

from quantark.util.calendar import DayCountConvention
from quantark.util.enum import (
    ExerciseType,
    ObservationAggregation,
    ObservationFrequency,
    ObservationType,
    OptionType,
    TenorEnd,
)
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from .base_equity_option import BaseEquityOption
from .observation_schedule import ObservationRecord, ObservationSchedule


@dataclass
class DoubleSharkfinOption(BaseEquityOption):
    """
    Double sharkfin option with upper and lower knock-out barriers.

    Payoff states:
        1. Either barrier hit: knock_out_rebate
        2. No barrier hit:
           - CALL: no_hit_rebate + participation_rate * max(min(S, U) - K, 0)
           - PUT:  no_hit_rebate + participation_rate * max(K - max(S, L), 0)

    Monitoring styles:
        - EXPIRY: barriers checked only at terminal spot
        - DISCRETE: barriers checked on observation dates, including daily schedules
        - CONTINUOUS: barriers checked continuously by the pricing engine/path logic

    Attributes:
        strike: Strike price.
        option_type: CALL or PUT payoff direction.
        upper_barrier: Upper knock-out barrier level.
        lower_barrier: Lower knock-out barrier level.
        maturity: Time to maturity in years.
        participation_rate: Participation in the capped no-hit payoff.
        knock_out_rebate: Fixed payoff if either barrier is hit.
        no_hit_rebate: Fixed payoff added when no barrier is hit.
        pay_at_hit: If True, pay knock-out rebate immediately on hit;
            otherwise pay it at expiry.
        observation_type: EXPIRY, DISCRETE, or CONTINUOUS monitoring.
        observation_dates: Legacy discrete observation times in years.
        observation_schedule: Preferred discrete observation schedule.
        observation_frequency: Frequency used when generating a regular schedule.
        contract_multiplier: Underlying units represented by one contract.
    """

    upper_barrier: float = 0.0
    lower_barrier: float = 0.0
    participation_rate: float = 1.0
    knock_out_rebate: float = 0.0
    no_hit_rebate: float = 0.0
    pay_at_hit: bool = False
    observation_type: ObservationType = ObservationType.CONTINUOUS
    observation_dates: Optional[List[float]] = None
    observation_schedule: Optional[ObservationSchedule] = None
    observation_frequency: ObservationFrequency = ObservationFrequency.CUSTOM
    use_business_days_for_frequency: bool = True
    business_days_in_year: float = 252.0

    def __init__(
        self,
        strike: float,
        option_type: OptionType,
        upper_barrier: float,
        lower_barrier: float,
        maturity: Optional[float] = None,
        exercise_date: Optional[datetime] = None,
        settlement_date: Optional[datetime] = None,
        participation_rate: float = 1.0,
        knock_out_rebate: float = 0.0,
        no_hit_rebate: float = 0.0,
        pay_at_hit: bool = False,
        observation_type: ObservationType = ObservationType.CONTINUOUS,
        observation_dates: Optional[List[float]] = None,
        observation_schedule: Optional[ObservationSchedule] = None,
        observation_frequency: ObservationFrequency = ObservationFrequency.CUSTOM,
        use_business_days_for_frequency: bool = True,
        business_days_in_year: float = 252.0,
        contract_multiplier: float = 1.0,
        tenor: Optional[float] = None,
        initial_date: Optional[datetime] = None,
        maturity_date: Optional[datetime] = None,
        tenor_end: TenorEnd = TenorEnd.EXERCISE,
        annualization_day_count: DayCountConvention = DayCountConvention.ACT_365,
    ):
        """
        Initialize a double sharkfin option.

        Args:
            strike: Strike price.
            option_type: CALL or PUT payoff direction.
            upper_barrier: Upper knock-out barrier level.
            lower_barrier: Lower knock-out barrier level.
            maturity: Time to maturity in years (optional if exercise_date provided).
            exercise_date: Expiration date (optional if maturity provided).
            settlement_date: Settlement date.
            participation_rate: Participation in the capped no-hit payoff.
            knock_out_rebate: Fixed payoff if either barrier is hit.
            no_hit_rebate: Fixed payoff added if no barrier is hit.
            pay_at_hit: If True, knock-out rebate is paid immediately on hit.
                If False, it is paid at expiry.
            observation_type: EXPIRY, DISCRETE, or CONTINUOUS monitoring.
            observation_dates: Discrete observation times in year fractions.
            observation_schedule: Preferred discrete observation schedule.
            observation_frequency: Frequency for generated discrete schedules.
            use_business_days_for_frequency: Use business-day spacing for frequency.
            business_days_in_year: Business-day count for generated schedules.
            contract_multiplier: Underlying units represented by one contract.

        Raises:
            ValidationError: If parameters are invalid.
        """
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.participation_rate = participation_rate
        self.knock_out_rebate = knock_out_rebate
        self.no_hit_rebate = no_hit_rebate
        self.pay_at_hit = pay_at_hit
        self.observation_type = observation_type
        self.observation_dates = observation_dates
        self.observation_schedule = observation_schedule
        self.observation_frequency = observation_frequency
        self.use_business_days_for_frequency = use_business_days_for_frequency
        self.business_days_in_year = business_days_in_year

        super().__init__(
            strike=strike,
            option_type=option_type,
            exercise_type=ExerciseType.EUROPEAN,
            maturity=maturity,
            tenor=tenor,
            initial_date=initial_date,
            exercise_date=exercise_date,
            settlement_date=settlement_date,
            maturity_date=maturity_date,
            tenor_end=tenor_end,
            annualization_day_count=annualization_day_count,
            contract_multiplier=contract_multiplier,
        )

    def validate(self) -> None:
        """
        Validate double sharkfin option parameters.

        Raises:
            ValidationError: If any parameter is invalid.
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
        if self.participation_rate < 0:
            raise ValidationError(
                f"Participation rate must be non-negative, got {self.participation_rate}"
            )
        if self.knock_out_rebate < 0:
            raise ValidationError(
                f"Knock-out rebate must be non-negative, got {self.knock_out_rebate}"
            )
        if self.no_hit_rebate < 0:
            raise ValidationError(
                f"No-hit rebate must be non-negative, got {self.no_hit_rebate}"
            )
        if not isinstance(self.pay_at_hit, bool):
            raise ValidationError(f"pay_at_hit must be boolean, got {self.pay_at_hit}")
        if not isinstance(self.observation_type, ObservationType):
            raise ValidationError(f"Invalid observation type: {self.observation_type}")
        if not isinstance(self.observation_frequency, ObservationFrequency):
            raise ValidationError(
                f"Invalid observation frequency: {self.observation_frequency}"
            )
        if self.business_days_in_year <= 0:
            raise ValidationError(
                f"business_days_in_year must be positive, got {self.business_days_in_year}"
            )

        self._validate_strike_in_corridor()
        self._normalize_observation_schedule()

    def _validate_strike_in_corridor(self) -> None:
        """Validate that the strike is inside the double sharkfin corridor."""
        if not (self.lower_barrier < self.strike < self.upper_barrier):
            raise ValidationError(
                "Double sharkfin requires lower_barrier < strike < upper_barrier, "
                f"got lower_barrier={self.lower_barrier}, strike={self.strike}, "
                f"upper_barrier={self.upper_barrier}"
            )

    def _normalize_observation_schedule(self) -> None:
        """Normalize discrete monitoring inputs into ObservationSchedule."""
        if self.observation_type in (ObservationType.CONTINUOUS, ObservationType.EXPIRY):
            if self.observation_schedule is not None:
                raise ValidationError(
                    "ObservationSchedule requires DISCRETE observation_type."
                )
            self.observation_dates = self.observation_dates or []
            return

        if self.observation_schedule is not None:
            normalized_schedule = ObservationSchedule(
                records=[
                    ObservationRecord(
                        observation_time=rec.observation_time,
                        observation_date=rec.observation_date,
                        upper_barrier=(
                            rec.upper_barrier
                            if rec.upper_barrier is not None
                            else self.upper_barrier
                        ),
                        lower_barrier=(
                            rec.lower_barrier
                            if rec.lower_barrier is not None
                            else self.lower_barrier
                        ),
                        payoff=(
                            rec.payoff
                            if rec.payoff is not None
                            else self.knock_out_rebate
                        ),
                        return_rate=rec.return_rate,
                        is_rate_annualized=rec.is_rate_annualized,
                        initial_date=rec.initial_date,
                        settlement_date=rec.settlement_date,
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
            self._validate_observation_times(normalized_schedule.times)
            normalized_schedule.validate(require_double=True)
            self.observation_schedule = normalized_schedule
            if self.observation_schedule.times:
                self.observation_dates = self.observation_schedule.times
            return

        if self.observation_dates is None or len(self.observation_dates) == 0:
            self.observation_dates = self._generate_observation_dates()
        self._validate_observation_times(self.observation_dates)

        self.observation_schedule = ObservationSchedule.from_legacy(
            observation_dates=self.observation_dates,
            default_barrier=None,
            default_payoff=self.knock_out_rebate,
            aggregation_mode=ObservationAggregation.STOP_FIRST_HIT,
            frequency=self.observation_frequency,
            upper_barrier=self.upper_barrier,
            lower_barrier=self.lower_barrier,
        )
        self.observation_dates = self.observation_schedule.times

    def _validate_observation_times(self, times: List[float]) -> None:
        """Validate numeric observation times when present."""
        if any(t < 0 for t in times):
            raise ValidationError("Observation dates must be non-negative.")
        if times != sorted(times):
            raise ValidationError("Observation dates must be sorted in ascending order.")

    def _generate_observation_dates(self) -> List[float]:
        """
        Generate regular discrete observation times from observation_frequency.

        Returns:
            Observation times in years.

        Raises:
            ValidationError: If a schedule cannot be generated.
        """
        if self.observation_frequency == ObservationFrequency.CUSTOM:
            raise ValidationError(
                "Observation dates required for discrete double sharkfin monitoring "
                "when observation_frequency is CUSTOM."
            )
        if self.maturity is None or self.maturity <= 0:
            raise ValidationError(
                "Positive maturity is required to generate discrete observations."
            )

        days_in_year = (
            self.business_days_in_year
            if self.use_business_days_for_frequency
            else 365.0
        )
        dt = self.observation_frequency.to_year_fraction(
            use_business_days=self.use_business_days_for_frequency,
            days_in_year=days_in_year,
        )
        if dt <= 0:
            raise ValidationError(f"Observation frequency produced invalid dt={dt}")

        count = max(1, int(ceil(self.maturity / dt)))
        times = [min((idx + 1) * dt, self.maturity) for idx in range(count)]
        if not is_close(times[-1], self.maturity):
            times.append(self.maturity)
        return times

    def is_barrier_hit(self, spot: float) -> bool:
        """
        Check whether a single observed spot hits either sharkfin barrier.

        Args:
            spot: Observed underlying price.

        Returns:
            True if either barrier is hit.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")
        return spot >= self.upper_barrier or spot <= self.lower_barrier

    def has_barrier_hit(self, path: Sequence[float]) -> bool:
        """
        Check whether any observed spot in a path hits either barrier.

        Args:
            path: Sequence of observed underlying prices.

        Returns:
            True if any observation hits either barrier.
        """
        if len(path) == 0:
            raise ValidationError("Path must contain at least one observed spot.")
        return any(self.is_barrier_hit(spot) for spot in path)

    def is_in_corridor(self, spot: float) -> bool:
        """
        Check whether a spot is strictly inside the sharkfin corridor.

        Args:
            spot: Observed underlying price.

        Returns:
            True if spot is between lower and upper barriers.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")
        return self.lower_barrier < spot < self.upper_barrier

    def get_no_hit_payoff(self, spot: float) -> float:
        """
        Calculate the capped no-hit payoff.

        Args:
            spot: Terminal underlying price.

        Returns:
            Payoff assuming neither barrier was hit.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        if self.is_call():
            capped_spot = min(spot, self.upper_barrier)
            intrinsic = max(capped_spot - self.strike, 0.0)
        else:
            capped_spot = max(spot, self.lower_barrier)
            intrinsic = max(self.strike - capped_spot, 0.0)

        payoff = self.no_hit_rebate + self.participation_rate * intrinsic
        return payoff * self.contract_multiplier

    def get_barrier_payoff(self) -> float:
        """
        Calculate payoff when either sharkfin barrier has been hit.

        Returns:
            Knock-out rebate scaled by contract multiplier.
        """
        return self.knock_out_rebate * self.contract_multiplier

    def get_payoff(self, spot: float, barrier_hit: Optional[bool] = None) -> float:
        """
        Calculate the terminal double sharkfin payoff.

        Args:
            spot: Terminal underlying price.
            barrier_hit: Optional observed barrier state. When omitted, terminal
                spot is used to infer a hit. Pricing engines for DISCRETE or
                CONTINUOUS monitoring should pass the path-derived barrier state.

        Returns:
            Sharkfin payoff scaled by contract multiplier.
        """
        if spot < 0:
            raise ValidationError(f"Spot price must be non-negative, got {spot}")

        if barrier_hit is None:
            barrier_hit = self.is_barrier_hit(spot)

        if barrier_hit:
            return self.get_barrier_payoff()
        return self.get_no_hit_payoff(spot)

    def get_observation_times(self) -> List[float]:
        """
        Get discrete observation times.

        Returns:
            Observation times in years. Empty for EXPIRY or CONTINUOUS monitoring.
        """
        if self.observation_schedule is not None:
            return self.observation_schedule.times
        return self.observation_dates or []

    @property
    def corridor_width(self) -> float:
        """Get the width of the corridor in price terms."""
        return self.upper_barrier - self.lower_barrier

    @property
    def corridor_width_log(self) -> float:
        """Get the width of the corridor in log-price terms."""
        from quantark.util.numerical import safe_log

        return safe_log(self.upper_barrier / self.lower_barrier)

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
            if bumped_schedule.uses_times():
                self.observation_dates = bumped_schedule.times

        if getattr(self, "exercise_date", None) is None and self.maturity is not None:
            self.maturity -= time_bump

        return False

    def __repr__(self) -> str:
        return (
            "DoubleSharkfinOption("
            f"{self.option_type}, K={self.strike:.2f}, "
            f"L={self.lower_barrier:.2f}, U={self.upper_barrier:.2f}, "
            f"{self._format_maturity()})"
        )
