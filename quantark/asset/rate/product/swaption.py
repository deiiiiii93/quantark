"""
Swaption (option on interest rate swap) product.

A swaption gives the holder the right (but not the obligation) to enter
into an interest rate swap at a future date (the exercise date).

- Payer swaption: right to enter a payer swap (pay fixed, receive floating)
- Receiver swaption: right to enter a receiver swap (receive fixed, pay floating)

The underlying swap parameters are stored directly rather than as a
pre-built InterestRateSwap, since the swap starts in the future and
constructing it eagerly may generate invalid schedules. The method
create_underlying_swap() builds the swap on demand.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from quantark.asset.rate.product.irs import (
    InterestRateSwap,
    NotionalSchedule,
    SwapDirection,
    create_vanilla_irs,
)
from quantark.param.index import RateIndex
from quantark.util.calendar import (
    DayCountConvention,
    Calendar,
    create_calendar,
)
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError


class SwaptionType(Enum):
    """Type of swaption."""

    PAYER = "payer"  # Right to enter a payer swap (pay fixed)
    RECEIVER = "receiver"  # Right to enter a receiver swap (receive fixed)


class SwaptionExerciseStyle(Enum):
    """Exercise style of the swaption."""

    EUROPEAN = "european"  # Exercise only at expiry
    BERMUDAN = "bermudan"  # Exercise at specific dates (future extension)


@dataclass
class Swaption:
    """
    Swaption — an option to enter into an interest rate swap.

    The holder has the right to enter into the underlying swap on the
    exercise date at the pre-agreed fixed rate (strike).

    Attributes:
        exercise_date: Option expiry / exercise date
        swaption_type: PAYER or RECEIVER
        exercise_style: EUROPEAN or BERMUDAN

        swap_start_date: Start date of the underlying swap (= exercise_date for spot-starting)
        swap_end_date: Maturity date of the underlying swap
        notional: Notional principal
        fixed_rate: Strike rate (fixed rate of the underlying swap)
        index: Reference rate index for the floating leg
        payment_frequency: Payment frequency for both legs
        fixed_day_count: Day count convention for the fixed leg
        float_day_count: Day count convention for the floating leg
        calendar: Business day calendar
        notional_schedule: Optional amortizing notional schedule
        trade_date: Date the swaption was traded
    """

    # Option parameters
    exercise_date: datetime
    swaption_type: SwaptionType
    exercise_style: SwaptionExerciseStyle = SwaptionExerciseStyle.EUROPEAN

    # Underlying swap parameters
    swap_start_date: datetime = None  # type: ignore[assignment]
    swap_end_date: datetime = None  # type: ignore[assignment]
    notional: float = 0.0
    fixed_rate: float = 0.0
    index: RateIndex = None  # type: ignore[assignment]
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360_US
    float_day_count: DayCountConvention = DayCountConvention.ACT_360
    calendar: Optional[Calendar] = None
    notional_schedule: Optional[NotionalSchedule] = None
    trade_date: Optional[datetime] = None

    def __post_init__(self):
        """Initialize and validate the swaption."""
        # Default swap_start_date to exercise_date (spot-starting)
        if self.swap_start_date is None:
            self.swap_start_date = self.exercise_date

        if self.calendar is None and self.index is not None:
            self.calendar = create_calendar(self.index.calendar_type)

        self.validate()

    def validate(self) -> None:
        """Validate swaption parameters."""
        if self.index is None:
            raise ValidationError("Index must be specified")
        if self.notional <= 0:
            raise ValidationError(
                f"Notional must be positive, got {self.notional}"
            )
        if self.fixed_rate < -0.10 or self.fixed_rate > 0.50:
            raise ValidationError(
                f"Fixed rate {self.fixed_rate} seems unreasonable"
            )
        if self.swap_end_date is None:
            raise ValidationError("Swap end date must be specified")
        if self.swap_end_date <= self.swap_start_date:
            raise ValidationError(
                f"Swap end date {self.swap_end_date} must be after "
                f"swap start date {self.swap_start_date}"
            )
        if self.swap_start_date < self.exercise_date:
            raise ValidationError(
                f"Swap start date {self.swap_start_date} must be on or after "
                f"exercise date {self.exercise_date}"
            )
        if (
            self.trade_date is not None
            and self.trade_date > self.exercise_date
        ):
            raise ValidationError(
                f"Trade date {self.trade_date} must be on or before "
                f"exercise date {self.exercise_date}"
            )

    def get_notional(self, as_of_date: Optional[datetime] = None) -> float:
        """
        Get the notional amount.

        Args:
            as_of_date: Date to look up notional (for amortizing)

        Returns:
            Notional amount
        """
        if self.notional_schedule is not None and as_of_date is not None:
            return self.notional_schedule.get_notional(as_of_date)
        return self.notional

    def get_exercise_date(self) -> datetime:
        """Get the option exercise date."""
        return self.exercise_date

    def get_swap_tenor(self) -> float:
        """
        Get the tenor of the underlying swap in years.

        Returns:
            Swap tenor in years
        """
        return (self.swap_end_date - self.swap_start_date).days / 365.0

    def get_option_tenor(self, valuation_date: datetime) -> float:
        """
        Get the option tenor (time to exercise) in years.

        Args:
            valuation_date: Date from which to measure

        Returns:
            Time to exercise in years
        """
        return (self.exercise_date - valuation_date).days / 365.0

    def get_maturity(self) -> datetime:
        """Get the maturity date (end of underlying swap)."""
        return self.swap_end_date

    def time_to_expiry(self, valuation_date: datetime) -> float:
        """
        Calculate time to option expiry in years.

        Args:
            valuation_date: Date from which to measure

        Returns:
            Time to expiry in years
        """
        return (self.exercise_date - valuation_date).days / 365.0

    def is_expired(self, valuation_date: datetime) -> bool:
        """
        Check if the swaption has expired.

        Args:
            valuation_date: Date to check

        Returns:
            True if past the exercise date
        """
        return valuation_date >= self.exercise_date

    def create_underlying_swap(self) -> InterestRateSwap:
        """
        Build the underlying InterestRateSwap on demand.

        Maps swaption type to swap direction:
        - PAYER swaption -> PAYER swap (pay fixed)
        - RECEIVER swaption -> RECEIVER swap (receive fixed)

        Returns:
            InterestRateSwap representing the underlying
        """
        direction = (
            SwapDirection.PAYER
            if self.swaption_type == SwaptionType.PAYER
            else SwapDirection.RECEIVER
        )

        return create_vanilla_irs(
            effective_date=self.swap_start_date,
            maturity_date=self.swap_end_date,
            denominator=self.notional,
            fixed_rate=self.fixed_rate,
            index=self.index,
            direction=direction,
            payment_frequency=self.payment_frequency,
            fixed_day_count=self.fixed_day_count,
            float_day_count=self.float_day_count,
        )

    def __repr__(self):
        type_str = self.swaption_type.value.capitalize()
        expiry_str = self.exercise_date.date()
        swap_end_str = self.swap_end_date.date()
        return (
            f"Swaption({type_str}, K={self.fixed_rate:.4%}, "
            f"expiry={expiry_str}, "
            f"swap_end={swap_end_str}, "
            f"notional={self.notional:,.0f}, "
            f"index={self.index.name})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_payer_swaption(
    exercise_date: datetime,
    swap_tenor_years: int,
    notional: float,
    fixed_rate: float,
    index: RateIndex,
    exercise_style: SwaptionExerciseStyle = SwaptionExerciseStyle.EUROPEAN,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360_US,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Optional[Calendar] = None,
    trade_date: Optional[datetime] = None,
) -> Swaption:
    """
    Create a Payer Swaption (right to pay fixed).

    The underlying swap starts on the exercise date and has the
    specified tenor.

    Args:
        exercise_date: Option exercise/expiry date
        swap_tenor_years: Tenor of the underlying swap in years
        notional: Notional principal
        fixed_rate: Strike rate
        index: Reference rate index
        exercise_style: EUROPEAN or BERMUDAN
        payment_frequency: Payment frequency for both legs
        fixed_day_count: Day count convention for fixed leg
        float_day_count: Day count convention for floating leg
        calendar: Business day calendar
        trade_date: Trade date

    Returns:
        Swaption object with PAYER type
    """
    if swap_tenor_years <= 0:
        raise ValidationError(
            f"Swap tenor must be positive, got {swap_tenor_years}"
        )

    from dateutil.relativedelta import relativedelta

    swap_end_date = exercise_date + relativedelta(years=swap_tenor_years)

    return Swaption(
        exercise_date=exercise_date,
        swaption_type=SwaptionType.PAYER,
        exercise_style=exercise_style,
        swap_start_date=exercise_date,
        swap_end_date=swap_end_date,
        notional=notional,
        fixed_rate=fixed_rate,
        index=index,
        payment_frequency=payment_frequency,
        fixed_day_count=fixed_day_count,
        float_day_count=float_day_count,
        calendar=calendar,
        trade_date=trade_date,
    )


def create_receiver_swaption(
    exercise_date: datetime,
    swap_tenor_years: int,
    notional: float,
    fixed_rate: float,
    index: RateIndex,
    exercise_style: SwaptionExerciseStyle = SwaptionExerciseStyle.EUROPEAN,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360_US,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Optional[Calendar] = None,
    trade_date: Optional[datetime] = None,
) -> Swaption:
    """
    Create a Receiver Swaption (right to receive fixed).

    The underlying swap starts on the exercise date and has the
    specified tenor.

    Args:
        exercise_date: Option exercise/expiry date
        swap_tenor_years: Tenor of the underlying swap in years
        notional: Notional principal
        fixed_rate: Strike rate
        index: Reference rate index
        exercise_style: EUROPEAN or BERMUDAN
        payment_frequency: Payment frequency for both legs
        fixed_day_count: Day count convention for fixed leg
        float_day_count: Day count convention for floating leg
        calendar: Business day calendar
        trade_date: Trade date

    Returns:
        Swaption object with RECEIVER type
    """
    if swap_tenor_years <= 0:
        raise ValidationError(
            f"Swap tenor must be positive, got {swap_tenor_years}"
        )

    from dateutil.relativedelta import relativedelta

    swap_end_date = exercise_date + relativedelta(years=swap_tenor_years)

    return Swaption(
        exercise_date=exercise_date,
        swaption_type=SwaptionType.RECEIVER,
        exercise_style=exercise_style,
        swap_start_date=exercise_date,
        swap_end_date=swap_end_date,
        notional=notional,
        fixed_rate=fixed_rate,
        index=index,
        payment_frequency=payment_frequency,
        fixed_day_count=fixed_day_count,
        float_day_count=float_day_count,
        calendar=calendar,
        trade_date=trade_date,
    )
