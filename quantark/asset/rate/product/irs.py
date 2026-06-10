"""
Interest Rate Swap (IRS) products.

This module provides comprehensive IRS implementation including:
- Vanilla IRS (fixed vs floating)
- Basis Swaps (floating vs floating)
- Amortizing notional schedules
- SOFR overnight rate compounding
- Rate caps/floors on floating legs

The implementation follows market conventions and supports:
- In-advance and in-arrears rate resets
- Lookback and lockout for overnight rates
- Multiple day count conventions
- Business day adjustments
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Tuple, Union
from dateutil.relativedelta import relativedelta

from quantark.asset.bond.schedule.cashflow import (
    CashFlow,
    FixedCashFlow,
    FloatingCashFlow,
    CompoundingMethod,
)
from quantark.param.index import RateIndex, IndexFixingStore
from quantark.param.rrf import RateCurve
from quantark.util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    Calendar,
    CalendarType,
    create_calendar,
    calculate_day_count_fraction,
)
from quantark.util.enum import PaymentFrequency, StubType, ResetConvention
from quantark.util.exceptions import ValidationError


class SwapDirection(Enum):
    """Direction of swap from the perspective of the holder."""

    PAYER = "payer"  # Pay fixed, receive floating (for vanilla IRS)
    RECEIVER = "receiver"  # Receive fixed, pay floating (for vanilla IRS)


@dataclass
class NotionalSchedule:
    """
    Denominator schedule for amortizing/accreting swaps.

    Supports arbitrary denominator changes over the life of the swap.

    Attributes:
        notional_dates: List of dates when denominator changes
        notional_amounts: List of denominator amounts corresponding to each date
        initial_notional: Initial denominator amount (before any amortization)
    """

    notional_dates: List[datetime]
    notional_amounts: List[float]
    initial_notional: float

    def __post_init__(self):
        """Validate notional schedule."""
        if len(self.notional_dates) != len(self.notional_amounts):
            raise ValidationError(
                "notional_dates and notional_amounts must have same length"
            )
        if self.initial_notional <= 0:
            raise ValidationError(
                f"Initial notional must be positive, got {self.initial_notional}"
            )
        for amount in self.notional_amounts:
            if amount < 0:
                raise ValidationError(f"Notional amounts must be non-negative")

        # Ensure dates are sorted
        sorted_pairs = sorted(zip(self.notional_dates, self.notional_amounts))
        self.notional_dates = [d for d, _ in sorted_pairs]
        self.notional_amounts = [a for _, a in sorted_pairs]

    def get_notional(self, as_of_date: datetime) -> float:
        """
        Get the denominator amount for a given date.

        Args:
            as_of_date: Date to look up denominator

        Returns:
            Denominator amount effective on that date
        """
        # Before first change, use initial notional
        if not self.notional_dates or as_of_date < self.notional_dates[0]:
            return self.initial_notional

        # Find the most recent notional change
        for i in range(len(self.notional_dates) - 1, -1, -1):
            if as_of_date >= self.notional_dates[i]:
                return self.notional_amounts[i]

        return self.initial_notional

    def get_denominator(self, as_of_date: datetime) -> float:
        """
        Get the denominator amount for a given date.

        Args:
            as_of_date: Date to look up denominator

        Returns:
            Denominator amount effective on that date
        """
        return self.get_notional(as_of_date)

    @classmethod
    def constant(cls, notional: float) -> "NotionalSchedule":
        """
        Create a constant denominator schedule (bullet swap).

        Args:
            notional: Constant denominator amount

        Returns:
            NotionalSchedule with no changes
        """
        return cls(notional_dates=[], notional_amounts=[], initial_notional=notional)

    @classmethod
    def linear_amortizing(
        cls,
        initial_notional: float,
        start_date: datetime,
        end_date: datetime,
        num_periods: int,
        final_notional: float = 0.0,
    ) -> "NotionalSchedule":
        """
        Create a linearly amortizing denominator schedule.

        Args:
            initial_notional: Starting denominator
            start_date: When amortization starts
            end_date: When amortization ends
            num_periods: Number of amortization periods
            final_notional: Final denominator amount (default: 0)

        Returns:
            NotionalSchedule with linear amortization
        """
        if num_periods < 1:
            raise ValidationError("num_periods must be at least 1")

        total_days = (end_date - start_date).days
        period_days = total_days / num_periods
        notional_step = (initial_notional - final_notional) / num_periods

        dates = []
        amounts = []

        for i in range(1, num_periods + 1):
            date = start_date + timedelta(days=int(i * period_days))
            amount = initial_notional - i * notional_step
            dates.append(date)
            amounts.append(max(0, amount))

        return cls(
            notional_dates=dates,
            notional_amounts=amounts,
            initial_notional=initial_notional,
        )

    def __repr__(self):
        if not self.notional_dates:
            return f"NotionalSchedule(constant={self.initial_notional:.2f})"
        return f"NotionalSchedule(initial={self.initial_notional:.2f}, changes={len(self.notional_dates)})"


class SwapLeg(ABC):
    """
    Abstract base class for swap legs.

    A swap leg represents one side of an interest rate swap,
    generating a series of cashflows based on either a fixed
    or floating rate.
    """

    @abstractmethod
    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """
        Get all future cashflows from valuation date.

        Args:
            valuation_date: Date to value the leg

        Returns:
            List of future cashflows
        """
        pass

    @abstractmethod
    def get_all_cashflows(self) -> List[CashFlow]:
        """
        Get all cashflows (past and future).

        Returns:
            List of all cashflows
        """
        pass

    @abstractmethod
    def get_notional(self, as_of_date: datetime) -> float:
        """
        Get the denominator amount for a given date.

        Args:
            as_of_date: Date to look up denominator

        Returns:
            Denominator amount
        """
        pass

    def get_denominator(self, as_of_date: datetime) -> float:
        """
        Get the denominator amount for a given date.

        Args:
            as_of_date: Date to look up denominator

        Returns:
            Denominator amount
        """
        return self.get_notional(as_of_date)

    @abstractmethod
    def get_start_date(self) -> datetime:
        """Get the start date of the leg."""
        pass

    @abstractmethod
    def get_end_date(self) -> datetime:
        """Get the end/maturity date of the leg."""
        pass

    @abstractmethod
    def calculate_accrued_interest(self, settlement_date: datetime) -> float:
        """
        Calculate accrued interest as of settlement date.

        Args:
            settlement_date: Settlement date for calculation

        Returns:
            Accrued interest amount
        """
        pass

    @abstractmethod
    def is_fixed(self) -> bool:
        """Return True if this is a fixed rate leg."""
        pass


@dataclass
class FixedLeg(SwapLeg):
    """
    Fixed rate leg of an interest rate swap.

    Pays a fixed rate on the denominator amount at specified
    payment intervals.

    Attributes:
        start_date: Effective date of the leg
        end_date: Maturity date of the leg
        notional_schedule: Denominator schedule (supports amortization)
        fixed_rate: Fixed interest rate (annual)
        payment_frequency: Payment frequency
        day_count_convention: Day count convention
        calendar: Business day calendar
        business_day_convention: Convention for adjusting payment dates
        settlement_days: Days between accrual end and payment
    """

    start_date: datetime
    end_date: datetime
    notional_schedule: NotionalSchedule
    fixed_rate: float
    payment_frequency: PaymentFrequency
    day_count_convention: DayCountConvention = DayCountConvention.THIRTY_360_US
    calendar: Optional[Calendar] = None
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.MODIFIED_FOLLOWING
    )
    settlement_days: int = 0

    # Cached schedule
    _cached_schedule: Optional[List[FixedCashFlow]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize and validate the fixed leg."""
        if self.calendar is None:
            self.calendar = create_calendar(CalendarType.US)

        self.validate()
        self._cached_schedule = self._generate_schedule()

    def validate(self) -> None:
        """Validate leg parameters."""
        if self.end_date <= self.start_date:
            raise ValidationError(
                f"End date {self.end_date} must be after start date {self.start_date}"
            )
        if self.fixed_rate < -0.10 or self.fixed_rate > 0.50:
            raise ValidationError(f"Fixed rate {self.fixed_rate} seems unreasonable")

    def _generate_unadjusted_dates(self) -> List[datetime]:
        """Generate unadjusted payment dates."""
        dates = []
        months_per_period = 12 // self.payment_frequency.periods_per_year

        current_date = self.end_date
        while True:
            dates.append(current_date)
            next_date = current_date - relativedelta(months=months_per_period)
            if next_date <= self.start_date:
                break
            current_date = next_date

        dates.reverse()
        return dates

    def _generate_schedule(self) -> List[FixedCashFlow]:
        """Generate the complete payment schedule."""
        unadjusted_dates = self._generate_unadjusted_dates()

        if not unadjusted_dates:
            raise ValidationError("No payment dates generated")

        adjusted_dates = [
            self.calendar.adjust_date(date, self.business_day_convention)
            for date in unadjusted_dates
        ]

        cashflows = []
        accrual_start = self.start_date

        for i, payment_date in enumerate(adjusted_dates):
            accrual_end = unadjusted_dates[i]

            # Get notional for this period
            notional = self.notional_schedule.get_notional(accrual_start)

            # Calculate day count fraction
            dcf = calculate_day_count_fraction(
                accrual_start, accrual_end, self.day_count_convention
            )

            # Apply settlement delay
            settlement_date = payment_date
            if self.settlement_days > 0:
                settlement_date = self.calendar.add_business_days(
                    payment_date, self.settlement_days
                )

            cashflow = FixedCashFlow(
                payment_date=settlement_date,
                accrual_start_date=accrual_start,
                accrual_end_date=accrual_end,
                notional=notional,
                fixed_rate=self.fixed_rate,
                day_count_fraction=dcf,
            )

            cashflows.append(cashflow)
            accrual_start = accrual_end

        return cashflows

    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """Get all future cashflows from valuation date."""
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return [
            cf.to_cashflow()
            for cf in self._cached_schedule
            if cf.payment_date > valuation_date
        ]

    def get_all_cashflows(self) -> List[CashFlow]:
        """Get all cashflows."""
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return [cf.to_cashflow() for cf in self._cached_schedule]

    def get_fixed_cashflows(self) -> List[FixedCashFlow]:
        """Get all fixed cashflows (with full detail)."""
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()
        return self._cached_schedule

    def get_notional(self, as_of_date: datetime) -> float:
        """Get the denominator amount for a given date."""
        return self.notional_schedule.get_notional(as_of_date)

    def get_start_date(self) -> datetime:
        return self.start_date

    def get_end_date(self) -> datetime:
        return self.end_date

    def calculate_accrued_interest(self, settlement_date: datetime) -> float:
        """Calculate accrued interest as of settlement date."""
        if settlement_date <= self.start_date:
            return 0.0
        if settlement_date >= self.end_date:
            return 0.0

        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        for cf in self._cached_schedule:
            if cf.accrual_start_date <= settlement_date < cf.accrual_end_date:
                accrued_dcf = calculate_day_count_fraction(
                    cf.accrual_start_date, settlement_date, self.day_count_convention
                )
                return cf.notional * self.fixed_rate * accrued_dcf

        return 0.0

    def is_fixed(self) -> bool:
        return True

    def __repr__(self):
        return (
            f"FixedLeg(rate={self.fixed_rate:.4%}, "
            f"start={self.start_date.date()}, "
            f"end={self.end_date.date()})"
        )


@dataclass
class FloatingLeg(SwapLeg):
    """
    Floating rate leg of an interest rate swap.

    Pays a floating rate (index + spread) on the denominator amount.
    Supports overnight rate compounding with lookback/lockout.

    Attributes:
        start_date: Effective date of the leg
        end_date: Maturity date of the leg
        notional_schedule: Denominator schedule (supports amortization)
        index: Reference rate index
        spread: Spread over the index rate
        payment_frequency: Payment frequency
        reset_convention: When rate is fixed (in-advance or in-arrears)
        day_count_convention: Day count convention
        calendar: Business day calendar
        business_day_convention: Convention for adjusting payment dates
        settlement_days: Days between accrual end and payment
        lookback_days: Days to look back for fixing (in-arrears only)
        lockout_days: Days before period end to lock rate
        rate_cap: Optional maximum rate
        rate_floor: Optional minimum rate
        compounding_method: Method for compounding overnight rates
        fixing_store: Store for historical index fixings
    """

    start_date: datetime
    end_date: datetime
    notional_schedule: NotionalSchedule
    index: RateIndex
    spread: float
    payment_frequency: PaymentFrequency
    reset_convention: ResetConvention = ResetConvention.IN_ADVANCE
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360
    calendar: Optional[Calendar] = None
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.MODIFIED_FOLLOWING
    )
    settlement_days: int = 0
    lookback_days: int = 0
    lockout_days: int = 0
    rate_cap: Optional[float] = None
    rate_floor: Optional[float] = None
    compounding_method: CompoundingMethod = CompoundingMethod.NONE
    fixing_store: Optional[IndexFixingStore] = None

    # Cached schedule
    _cached_schedule: Optional[List[FloatingCashFlow]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize and validate the floating leg."""
        if self.calendar is None:
            self.calendar = create_calendar(self.index.calendar_type)

        if self.fixing_store is None:
            self.fixing_store = IndexFixingStore()

        self.index.set_fixing_store(self.fixing_store)

        self.validate()
        self._cached_schedule = self._generate_schedule()

    def validate(self) -> None:
        """Validate leg parameters."""
        if self.end_date <= self.start_date:
            raise ValidationError(
                f"End date {self.end_date} must be after start date {self.start_date}"
            )
        if self.lookback_days < 0:
            raise ValidationError(
                f"Lookback days must be non-negative, got {self.lookback_days}"
            )
        if self.lockout_days < 0:
            raise ValidationError(
                f"Lockout days must be non-negative, got {self.lockout_days}"
            )
        if self.rate_cap is not None and self.rate_floor is not None:
            if self.rate_cap < self.rate_floor:
                raise ValidationError(
                    f"Rate cap {self.rate_cap} must be >= floor {self.rate_floor}"
                )

    def _generate_unadjusted_dates(self) -> List[datetime]:
        """Generate unadjusted payment dates."""
        dates = []
        months_per_period = 12 // self.payment_frequency.periods_per_year

        current_date = self.end_date
        while True:
            dates.append(current_date)
            next_date = current_date - relativedelta(months=months_per_period)
            if next_date <= self.start_date:
                break
            current_date = next_date

        dates.reverse()
        return dates

    def _calculate_fixing_date(
        self, accrual_start: datetime, accrual_end: datetime
    ) -> datetime:
        """Calculate the fixing date for an accrual period."""
        if self.reset_convention == ResetConvention.IN_ADVANCE:
            fixing_date = accrual_start
            if self.index.fixing_lag_days > 0:
                fixing_date = self.calendar.add_business_days(
                    fixing_date, -self.index.fixing_lag_days
                )
            return fixing_date
        else:
            # In-arrears
            if self.lookback_days > 0:
                fixing_date = self.calendar.add_business_days(
                    accrual_end, -self.lookback_days
                )
            elif self.lockout_days > 0:
                fixing_date = self.calendar.add_business_days(
                    accrual_end, -self.lockout_days
                )
            else:
                fixing_date = accrual_end
                if self.index.fixing_lag_days > 0:
                    fixing_date = self.calendar.add_business_days(
                        fixing_date, -self.index.fixing_lag_days
                    )
            return fixing_date

    def _generate_schedule(self) -> List[FloatingCashFlow]:
        """Generate the complete payment schedule."""
        unadjusted_dates = self._generate_unadjusted_dates()

        if not unadjusted_dates:
            raise ValidationError("No payment dates generated")

        adjusted_dates = [
            self.calendar.adjust_date(date, self.business_day_convention)
            for date in unadjusted_dates
        ]

        cashflows = []
        accrual_start = self.start_date

        for i, payment_date in enumerate(adjusted_dates):
            accrual_end = unadjusted_dates[i]

            # Get notional for this period
            notional = self.notional_schedule.get_notional(accrual_start)

            # Calculate fixing date
            fixing_date = self._calculate_fixing_date(accrual_start, accrual_end)

            # Calculate day count fraction
            dcf = calculate_day_count_fraction(
                accrual_start, accrual_end, self.day_count_convention
            )

            # Check if we have a historical fixing
            fixing = self.fixing_store.get_fixing(self.index.name, fixing_date)

            # Apply settlement delay
            settlement_date = payment_date
            if self.settlement_days > 0:
                settlement_date = self.calendar.add_business_days(
                    payment_date, self.settlement_days
                )

            cashflow = FloatingCashFlow(
                payment_date=settlement_date,
                accrual_start_date=accrual_start,
                accrual_end_date=accrual_end,
                fixing_date=fixing_date,
                notional=notional,
                spread=self.spread,
                day_count_fraction=dcf,
                index_fixing=fixing,
                forward_rate=None,  # Set by engine
                is_projected=fixing is None,
                rate_cap=self.rate_cap,
                rate_floor=self.rate_floor,
                compounding_method=self.compounding_method,
            )

            cashflows.append(cashflow)
            accrual_start = accrual_end

        return cashflows

    def update_forward_rates(
        self, forward_curve: RateCurve, valuation_date: datetime
    ) -> None:
        """
        Update projected cashflows with forward rates from a curve.

        Args:
            forward_curve: Rate curve for projecting forward rates
            valuation_date: Valuation date
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        for cf in self._cached_schedule:
            if cf.is_projected:
                time_to_fixing = (cf.fixing_date - valuation_date).days / 365.0

                if time_to_fixing < 0:
                    time_to_fixing = 0.0

                if self.index.is_overnight:
                    cf.forward_rate = forward_curve.get_rate(time_to_fixing)
                else:
                    t1 = time_to_fixing
                    t2 = t1 + self.index.tenor_years
                    cf.forward_rate = forward_curve.get_forward_rate(t1, t2)

    def add_fixing(self, fixing_date: datetime, rate: float) -> None:
        """Add a historical fixing for the index."""
        from quantark.param.index import IndexFixing

        self.fixing_store.add_fixing(
            IndexFixing(fixing_date=fixing_date, rate=rate, index_name=self.index.name)
        )
        self._cached_schedule = self._generate_schedule()

    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """Get all future cashflows from valuation date."""
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return [
            cf.to_cashflow()
            for cf in self._cached_schedule
            if cf.payment_date > valuation_date
        ]

    def get_all_cashflows(self) -> List[CashFlow]:
        """Get all cashflows."""
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return [cf.to_cashflow() for cf in self._cached_schedule]

    def get_floating_cashflows(
        self, valuation_date: Optional[datetime] = None
    ) -> List[FloatingCashFlow]:
        """
        Get floating cashflows with full detail.

        Args:
            valuation_date: If provided, only return future cashflows

        Returns:
            List of FloatingCashFlow objects
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        if valuation_date is None:
            return self._cached_schedule

        return [cf for cf in self._cached_schedule if cf.payment_date > valuation_date]

    def get_notional(self, as_of_date: datetime) -> float:
        """Get the denominator amount for a given date."""
        return self.notional_schedule.get_notional(as_of_date)

    def get_start_date(self) -> datetime:
        return self.start_date

    def get_end_date(self) -> datetime:
        return self.end_date

    def calculate_accrued_interest(self, settlement_date: datetime) -> float:
        """Calculate accrued interest as of settlement date."""
        if settlement_date <= self.start_date:
            return 0.0
        if settlement_date >= self.end_date:
            return 0.0

        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        for cf in self._cached_schedule:
            if cf.accrual_start_date <= settlement_date < cf.accrual_end_date:
                accrued_dcf = calculate_day_count_fraction(
                    cf.accrual_start_date, settlement_date, self.day_count_convention
                )
                return cf.notional * cf.effective_rate * accrued_dcf

        return 0.0

    def get_current_rate(self, as_of_date: datetime) -> Optional[float]:
        """Get the current all-in rate for the active period."""
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        for cf in self._cached_schedule:
            if cf.accrual_start_date <= as_of_date < cf.accrual_end_date:
                return cf.effective_rate

        return None

    def is_fixed(self) -> bool:
        return False

    def __repr__(self):
        return (
            f"FloatingLeg(index={self.index.name}, "
            f"spread={self.spread:.2%}, "
            f"start={self.start_date.date()}, "
            f"end={self.end_date.date()})"
        )


@dataclass
class InterestRateSwap:
    """
    Interest Rate Swap (IRS).

    A swap that exchanges fixed rate payments for floating rate payments.
    The holder is either a "payer" (pays fixed, receives floating) or
    a "receiver" (receives fixed, pays floating).

    Attributes:
        fixed_leg: The fixed rate leg
        floating_leg: The floating rate leg
        direction: Whether holder pays or receives fixed
        trade_date: Date the swap was traded
        effective_date: Date the swap becomes effective
    """

    fixed_leg: FixedLeg
    floating_leg: FloatingLeg
    direction: SwapDirection = SwapDirection.PAYER
    trade_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None

    def __post_init__(self):
        """Validate swap parameters."""
        self.validate()

        if self.effective_date is None:
            self.effective_date = self.fixed_leg.start_date
        if self.trade_date is None:
            self.trade_date = self.effective_date

    def validate(self) -> None:
        """Validate swap parameters."""
        # Legs should have matching dates (allowing small differences)
        if abs((self.fixed_leg.start_date - self.floating_leg.start_date).days) > 5:
            raise ValidationError(
                "Fixed and floating legs should have matching start dates"
            )
        if abs((self.fixed_leg.end_date - self.floating_leg.end_date).days) > 5:
            raise ValidationError(
                "Fixed and floating legs should have matching end dates"
            )

    @property
    def pay_leg(self) -> SwapLeg:
        """Get the leg that the holder pays."""
        if self.direction == SwapDirection.PAYER:
            return self.fixed_leg
        else:
            return self.floating_leg

    @property
    def receive_leg(self) -> SwapLeg:
        """Get the leg that the holder receives."""
        if self.direction == SwapDirection.PAYER:
            return self.floating_leg
        else:
            return self.fixed_leg

    def get_start_date(self) -> datetime:
        """Get the effective start date of the swap."""
        return min(self.fixed_leg.start_date, self.floating_leg.start_date)

    def get_end_date(self) -> datetime:
        """Get the maturity date of the swap."""
        return max(self.fixed_leg.end_date, self.floating_leg.end_date)

    def get_notional(self, as_of_date: datetime) -> float:
        """Get the denominator amount (from fixed leg) for a given date."""
        return self.fixed_leg.get_notional(as_of_date)

    def get_denominator(self, as_of_date: datetime) -> float:
        """Get the denominator amount (from fixed leg) for a given date."""
        return self.fixed_leg.get_denominator(as_of_date)

    def get_fixed_rate(self) -> float:
        """Get the fixed rate of the swap."""
        return self.fixed_leg.fixed_rate

    def get_spread(self) -> float:
        """Get the floating leg spread."""
        return self.floating_leg.spread

    def time_to_maturity(self, valuation_date: datetime) -> float:
        """Calculate time to maturity in years."""
        end_date = self.get_end_date()
        return (end_date - valuation_date).days / 365.0

    def is_expired(self, valuation_date: datetime) -> bool:
        """Check if swap has matured."""
        return valuation_date >= self.get_end_date()

    def update_forward_rates(
        self, forward_curve: RateCurve, valuation_date: datetime
    ) -> None:
        """Update floating leg with forward rates."""
        self.floating_leg.update_forward_rates(forward_curve, valuation_date)

    def __repr__(self):
        direction_str = "Pay" if self.direction == SwapDirection.PAYER else "Receive"
        return (
            f"InterestRateSwap({direction_str} {self.fixed_leg.fixed_rate:.4%} vs "
            f"{self.floating_leg.index.name}+{self.floating_leg.spread:.2%}, "
            f"maturity={self.get_end_date().date()})"
        )


@dataclass
class BasisSwap:
    """
    Basis Swap (floating vs floating).

    A swap that exchanges payments based on two different floating rate
    indices. One leg typically has a spread to compensate for the
    difference in index levels.

    Attributes:
        leg1: First floating rate leg
        leg2: Second floating rate leg
        trade_date: Date the swap was traded
        effective_date: Date the swap becomes effective
    """

    leg1: FloatingLeg
    leg2: FloatingLeg
    trade_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None

    def __post_init__(self):
        """Validate swap parameters."""
        self.validate()

        if self.effective_date is None:
            self.effective_date = self.leg1.start_date
        if self.trade_date is None:
            self.trade_date = self.effective_date

    def validate(self) -> None:
        """Validate swap parameters."""
        if abs((self.leg1.start_date - self.leg2.start_date).days) > 5:
            raise ValidationError("Legs should have matching start dates")
        if abs((self.leg1.end_date - self.leg2.end_date).days) > 5:
            raise ValidationError("Legs should have matching end dates")

    @property
    def pay_leg(self) -> FloatingLeg:
        """Get the leg that the holder pays (leg1)."""
        return self.leg1

    @property
    def receive_leg(self) -> FloatingLeg:
        """Get the leg that the holder receives (leg2)."""
        return self.leg2

    def get_start_date(self) -> datetime:
        """Get the effective start date of the swap."""
        return min(self.leg1.start_date, self.leg2.start_date)

    def get_end_date(self) -> datetime:
        """Get the maturity date of the swap."""
        return max(self.leg1.end_date, self.leg2.end_date)

    def get_notional(self, as_of_date: datetime) -> float:
        """Get the denominator amount (from leg1) for a given date."""
        return self.leg1.get_notional(as_of_date)

    def get_denominator(self, as_of_date: datetime) -> float:
        """Get the denominator amount (from leg1) for a given date."""
        return self.leg1.get_denominator(as_of_date)

    def get_basis_spread(self) -> float:
        """Get the net basis spread (leg2.spread - leg1.spread)."""
        return self.leg2.spread - self.leg1.spread

    def time_to_maturity(self, valuation_date: datetime) -> float:
        """Calculate time to maturity in years."""
        end_date = self.get_end_date()
        return (end_date - valuation_date).days / 365.0

    def is_expired(self, valuation_date: datetime) -> bool:
        """Check if swap has matured."""
        return valuation_date >= self.get_end_date()

    def update_forward_rates(
        self, forward_curve: RateCurve, valuation_date: datetime
    ) -> None:
        """Update both legs with forward rates."""
        self.leg1.update_forward_rates(forward_curve, valuation_date)
        self.leg2.update_forward_rates(forward_curve, valuation_date)

    def __repr__(self):
        return (
            f"BasisSwap({self.leg1.index.name}+{self.leg1.spread:.2%} vs "
            f"{self.leg2.index.name}+{self.leg2.spread:.2%}, "
            f"maturity={self.get_end_date().date()})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_vanilla_irs(
    effective_date: datetime,
    maturity_date: datetime,
    denominator: float,
    fixed_rate: float,
    index: RateIndex,
    spread: float = 0.0,
    direction: SwapDirection = SwapDirection.PAYER,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360_US,
    float_day_count: DayCountConvention = DayCountConvention.ACT_360,
) -> InterestRateSwap:
    """
    Create a vanilla interest rate swap.

    Args:
        effective_date: Start date of the swap
        maturity_date: Maturity date of the swap
        denominator: Denominator amount
        fixed_rate: Fixed rate (annual)
        index: Floating rate index
        spread: Spread over floating index
        direction: PAYER (pay fixed) or RECEIVER (receive fixed)
        payment_frequency: Payment frequency for both legs
        fixed_day_count: Day count convention for fixed leg
        float_day_count: Day count convention for floating leg

    Returns:
        InterestRateSwap object
    """
    notional_schedule = NotionalSchedule.constant(denominator)
    calendar = create_calendar(index.calendar_type)

    fixed_leg = FixedLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        fixed_rate=fixed_rate,
        payment_frequency=payment_frequency,
        day_count_convention=fixed_day_count,
        calendar=calendar,
    )

    floating_leg = FloatingLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        index=index,
        spread=spread,
        payment_frequency=payment_frequency,
        day_count_convention=float_day_count,
        calendar=calendar,
    )

    return InterestRateSwap(
        fixed_leg=fixed_leg,
        floating_leg=floating_leg,
        direction=direction,
        effective_date=effective_date,
    )


def create_basis_swap(
    effective_date: datetime,
    maturity_date: datetime,
    denominator: float,
    index1: RateIndex,
    index2: RateIndex,
    spread1: float = 0.0,
    spread2: float = 0.0,
    payment_frequency1: PaymentFrequency = PaymentFrequency.QUARTERLY,
    payment_frequency2: PaymentFrequency = PaymentFrequency.QUARTERLY,
    day_count1: DayCountConvention = DayCountConvention.ACT_360,
    day_count2: DayCountConvention = DayCountConvention.ACT_360,
) -> BasisSwap:
    """
    Create a basis swap (floating vs floating).

    Args:
        effective_date: Start date of the swap
        maturity_date: Maturity date of the swap
        denominator: Denominator amount
        index1: First floating rate index (pay leg)
        index2: Second floating rate index (receive leg)
        spread1: Spread over first index
        spread2: Spread over second index
        payment_frequency1: Payment frequency for leg 1
        payment_frequency2: Payment frequency for leg 2
        day_count1: Day count convention for leg 1
        day_count2: Day count convention for leg 2

    Returns:
        BasisSwap object
    """
    notional_schedule = NotionalSchedule.constant(denominator)
    calendar = create_calendar(index1.calendar_type)

    leg1 = FloatingLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        index=index1,
        spread=spread1,
        payment_frequency=payment_frequency1,
        day_count_convention=day_count1,
        calendar=calendar,
    )

    leg2 = FloatingLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        index=index2,
        spread=spread2,
        payment_frequency=payment_frequency2,
        day_count_convention=day_count2,
        calendar=calendar,
    )

    return BasisSwap(
        leg1=leg1,
        leg2=leg2,
        effective_date=effective_date,
    )


def create_amortizing_irs(
    effective_date: datetime,
    maturity_date: datetime,
    initial_notional: float,
    fixed_rate: float,
    index: RateIndex,
    amortization_schedule: List[Tuple[datetime, float]],
    spread: float = 0.0,
    direction: SwapDirection = SwapDirection.PAYER,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
) -> InterestRateSwap:
    """
    Create an amortizing interest rate swap.

    Args:
        effective_date: Start date of the swap
        maturity_date: Maturity date of the swap
        initial_notional: Initial denominator amount
        fixed_rate: Fixed rate (annual)
        index: Floating rate index
        amortization_schedule: List of (date, new_denominator) tuples
        spread: Spread over floating index
        direction: PAYER or RECEIVER
        payment_frequency: Payment frequency

    Returns:
        InterestRateSwap object
    """
    dates = [d for d, _ in amortization_schedule]
    amounts = [a for _, a in amortization_schedule]

    notional_schedule = NotionalSchedule(
        notional_dates=dates,
        notional_amounts=amounts,
        initial_notional=initial_notional,
    )

    calendar = create_calendar(index.calendar_type)

    fixed_leg = FixedLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        fixed_rate=fixed_rate,
        payment_frequency=payment_frequency,
        calendar=calendar,
    )

    floating_leg = FloatingLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        index=index,
        spread=spread,
        payment_frequency=payment_frequency,
        calendar=calendar,
    )

    return InterestRateSwap(
        fixed_leg=fixed_leg,
        floating_leg=floating_leg,
        direction=direction,
        effective_date=effective_date,
    )


def create_compounding_irs(
    effective_date: datetime,
    maturity_date: datetime,
    denominator: float,
    fixed_rate: float,
    index: RateIndex,
    spread: float = 0.0,
    direction: SwapDirection = SwapDirection.PAYER,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    compounding_method: CompoundingMethod = CompoundingMethod.SPREAD_EXCLUSIVE,
    lookback_days: int = 2,
) -> InterestRateSwap:
    """
    Create an IRS with overnight rate compounding on floating leg.

    This is typically used for SOFR or other overnight rate swaps
    where the floating rate is compounded over the accrual period.

    Args:
        effective_date: Start date of the swap
        maturity_date: Maturity date of the swap
        denominator: Denominator amount
        fixed_rate: Fixed rate (annual)
        index: Overnight rate index (e.g., SOFR)
        spread: Spread over compounded rate
        direction: PAYER or RECEIVER
        payment_frequency: Payment frequency
        compounding_method: How to compound (SPREAD_EXCLUSIVE or SPREAD_INCLUSIVE)
        lookback_days: Lookback for rate observation

    Returns:
        InterestRateSwap object
    """
    notional_schedule = NotionalSchedule.constant(denominator)
    calendar = create_calendar(index.calendar_type)

    fixed_leg = FixedLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        fixed_rate=fixed_rate,
        payment_frequency=payment_frequency,
        calendar=calendar,
    )

    floating_leg = FloatingLeg(
        start_date=effective_date,
        end_date=maturity_date,
        notional_schedule=notional_schedule,
        index=index,
        spread=spread,
        payment_frequency=payment_frequency,
        calendar=calendar,
        reset_convention=ResetConvention.IN_ARREARS,
        lookback_days=lookback_days,
        compounding_method=compounding_method,
    )

    return InterestRateSwap(
        fixed_leg=fixed_leg,
        floating_leg=floating_leg,
        direction=direction,
        effective_date=effective_date,
    )
