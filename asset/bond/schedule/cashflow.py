"""
Bond schedule generation and cashflow management.

This module provides:
- CashFlow: Basic cashflow representation
- FixedCashFlow: Fixed rate cashflow for fixed legs
- FloatingCashFlow: Floating rate cashflow with index reset and compounding
- ScheduleGenerator: Payment schedule generation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dateutil.relativedelta import relativedelta
from enum import Enum

from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    Calendar,
    calculate_day_count_fraction,
)
from util.enum import PaymentFrequency, StubType
from util.exceptions import ValidationError


class CompoundingMethod(Enum):
    """Compounding method for overnight rates."""

    NONE = "none"  # No compounding (simple rate)
    FLAT = "flat"  # Flat compounding (sum of daily rates)
    SPREAD_EXCLUSIVE = "spread_exclusive"  # Compound index, add spread at end
    SPREAD_INCLUSIVE = "spread_inclusive"  # Compound index + spread together


@dataclass
class CashFlow:
    """
    Represents a single cash flow in a bond schedule.

    Attributes:
        payment_date: Date when payment is made
        accrual_start_date: Start of accrual period
        accrual_end_date: End of accrual period
        notional: Notional amount for this period
        rate: Interest rate for this period (annual rate)
        day_count_fraction: Year fraction for this period
        amount: Cash flow amount (calculated)
    """

    payment_date: datetime
    accrual_start_date: datetime
    accrual_end_date: datetime
    notional: float
    rate: float
    day_count_fraction: float
    amount: float

    def __post_init__(self):
        """Validate cashflow."""
        if self.payment_date < self.accrual_end_date:
            raise ValidationError(
                f"Payment date {self.payment_date} must be >= accrual end date {self.accrual_end_date}"
            )
        if self.accrual_end_date <= self.accrual_start_date:
            raise ValidationError(
                f"Accrual end {self.accrual_end_date} must be > accrual start {self.accrual_start_date}"
            )

    def __repr__(self):
        return (
            f"CashFlow(payment={self.payment_date.date()}, "
            f"amount={self.amount:.2f})"
        )


@dataclass
class FixedCashFlow:
    """
    Represents a single cash flow for a fixed rate leg.

    Used for fixed legs of interest rate swaps and fixed rate bonds.

    Attributes:
        payment_date: Date when payment is made
        accrual_start_date: Start of accrual period
        accrual_end_date: End of accrual period
        notional: Notional amount for this period
        fixed_rate: Fixed interest rate (annual rate)
        day_count_fraction: Year fraction for this period
    """

    payment_date: datetime
    accrual_start_date: datetime
    accrual_end_date: datetime
    notional: float
    fixed_rate: float
    day_count_fraction: float

    @property
    def amount(self) -> float:
        """
        Calculate the cash flow amount.

        Returns:
            Fixed coupon payment amount
        """
        return self.notional * self.fixed_rate * self.day_count_fraction

    def to_cashflow(self) -> CashFlow:
        """
        Convert to standard CashFlow object.

        Returns:
            CashFlow instance
        """
        return CashFlow(
            payment_date=self.payment_date,
            accrual_start_date=self.accrual_start_date,
            accrual_end_date=self.accrual_end_date,
            notional=self.notional,
            rate=self.fixed_rate,
            day_count_fraction=self.day_count_fraction,
            amount=self.amount,
        )

    def __repr__(self):
        return (
            f"FixedCashFlow(payment={self.payment_date.date()}, "
            f"rate={self.fixed_rate:.4%}, amount={self.amount:.2f})"
        )


@dataclass
class FloatingCashFlow:
    """
    Represents a single cash flow for a floating rate instrument.

    Extends the basic CashFlow concept with floating rate specific fields
    including index fixing, forward rate projection, and compounding support.

    Attributes:
        payment_date: Date when payment is made
        accrual_start_date: Start of accrual period
        accrual_end_date: End of accrual period
        fixing_date: Date when the rate is fixed (for in-advance)
        notional: Notional amount for this period
        spread: Spread over the index rate (annual, e.g., 0.01 for 100bp)
        day_count_fraction: Year fraction for this period
        index_fixing: Actual fixing if known (None if projected)
        forward_rate: Projected rate if fixing is unknown
        is_projected: Whether the rate is projected (vs known fixing)
        rate_cap: Optional rate cap
        rate_floor: Optional rate floor
        compounding_method: Method for compounding overnight rates
        daily_fixings: Dictionary of daily fixings for compounding (date -> rate)
    """

    payment_date: datetime
    accrual_start_date: datetime
    accrual_end_date: datetime
    fixing_date: datetime
    notional: float
    spread: float
    day_count_fraction: float
    index_fixing: Optional[float] = None
    forward_rate: Optional[float] = None
    is_projected: bool = True
    rate_cap: Optional[float] = None
    rate_floor: Optional[float] = None
    compounding_method: CompoundingMethod = CompoundingMethod.NONE
    daily_fixings: Optional[Dict[datetime, float]] = field(default=None, repr=False)

    @property
    def base_rate(self) -> float:
        """
        Get the base index rate (before spread).

        Returns:
            Base rate from fixing or forward projection
        """
        if self.index_fixing is not None:
            return self.index_fixing
        elif self.forward_rate is not None:
            return self.forward_rate
        else:
            return 0.0

    @property
    def effective_rate(self) -> float:
        """
        Get the effective coupon rate (index + spread), applying cap/floor.

        For compounding methods, this returns the all-in rate after compounding.

        Returns:
            Effective rate for this period
        """
        if self.compounding_method != CompoundingMethod.NONE and self.daily_fixings:
            return self._calculate_compounded_rate()

        total_rate = self.base_rate + self.spread

        # Apply cap and floor
        if self.rate_cap is not None:
            total_rate = min(total_rate, self.rate_cap)
        if self.rate_floor is not None:
            total_rate = max(total_rate, self.rate_floor)

        return total_rate

    def _calculate_compounded_rate(self) -> float:
        """
        Calculate the compounded rate for overnight indices.

        Uses the daily fixings to compute compounded rate based on
        the compounding method.

        Returns:
            Compounded rate for the period
        """
        if not self.daily_fixings:
            return self.base_rate + self.spread

        sorted_dates = sorted(self.daily_fixings.keys())

        if self.compounding_method == CompoundingMethod.FLAT:
            # Simple averaging/summing of daily rates
            total_rate = sum(self.daily_fixings.values()) / len(self.daily_fixings)
            total_rate += self.spread
        elif self.compounding_method == CompoundingMethod.SPREAD_EXCLUSIVE:
            # Compound the index rate, then add spread
            compounded = 1.0
            for i, date in enumerate(sorted_dates):
                rate = self.daily_fixings[date]
                # Assume daily rate, 1/360 or 1/365 fraction
                days = 1
                if i < len(sorted_dates) - 1:
                    days = (sorted_dates[i + 1] - date).days
                compounded *= 1 + rate * days / 360.0
            # Annualize the compounded return
            total_days = (sorted_dates[-1] - sorted_dates[0]).days + 1
            compounded_rate = (compounded - 1) * 360.0 / total_days
            total_rate = compounded_rate + self.spread
        elif self.compounding_method == CompoundingMethod.SPREAD_INCLUSIVE:
            # Compound index + spread together
            compounded = 1.0
            for i, date in enumerate(sorted_dates):
                rate = self.daily_fixings[date] + self.spread
                days = 1
                if i < len(sorted_dates) - 1:
                    days = (sorted_dates[i + 1] - date).days
                compounded *= 1 + rate * days / 360.0
            total_days = (sorted_dates[-1] - sorted_dates[0]).days + 1
            total_rate = (compounded - 1) * 360.0 / total_days
        else:
            total_rate = self.base_rate + self.spread

        # Apply cap and floor
        if self.rate_cap is not None:
            total_rate = min(total_rate, self.rate_cap)
        if self.rate_floor is not None:
            total_rate = max(total_rate, self.rate_floor)

        return total_rate

    @property
    def amount(self) -> float:
        """
        Calculate the cash flow amount.

        Returns:
            Coupon payment amount
        """
        return self.notional * self.effective_rate * self.day_count_fraction

    def to_cashflow(self) -> CashFlow:
        """
        Convert to standard CashFlow object.

        Returns:
            CashFlow instance
        """
        return CashFlow(
            payment_date=self.payment_date,
            accrual_start_date=self.accrual_start_date,
            accrual_end_date=self.accrual_end_date,
            notional=self.notional,
            rate=self.effective_rate,
            day_count_fraction=self.day_count_fraction,
            amount=self.amount,
        )

    def __repr__(self):
        rate_type = "Fixed" if not self.is_projected else "Projected"
        return (
            f"FloatingCashFlow(payment={self.payment_date.date()}, "
            f"{rate_type}, rate={self.effective_rate:.4%}, "
            f"amount={self.amount:.2f})"
        )


class ScheduleGenerator:
    """
    Generates payment schedules for bonds with comprehensive features.

    Supports:
    - Regular and irregular first/last periods
    - Business day adjustments
    - Multiple payment frequencies
    - Various day count conventions
    - Settlement delays
    """

    def __init__(
        self,
        issue_date: datetime,
        maturity_date: datetime,
        payment_frequency: PaymentFrequency,
        day_count_convention: DayCountConvention,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        settlement_days: int = 0,
        stub_type: StubType = StubType.NONE,
        first_coupon_date: Optional[datetime] = None,
        penultimate_coupon_date: Optional[datetime] = None,
    ):
        """
        Initialize schedule generator.

        Args:
            issue_date: Bond issue date
            maturity_date: Bond maturity date
            payment_frequency: Payment frequency
            day_count_convention: Day count convention
            calendar: Business day calendar
            business_day_convention: Convention for adjusting payment dates
            settlement_days: Days between accrual end and payment
            stub_type: Type of stub period (if any)
            first_coupon_date: Date of first coupon (for irregular first period)
            penultimate_coupon_date: Date of penultimate coupon (for irregular last period)
        """
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.payment_frequency = payment_frequency
        self.day_count_convention = day_count_convention
        self.calendar = calendar
        self.business_day_convention = business_day_convention
        self.settlement_days = settlement_days
        self.stub_type = stub_type
        self.first_coupon_date = first_coupon_date
        self.penultimate_coupon_date = penultimate_coupon_date

        self._validate()

    def _validate(self):
        """Validate schedule parameters."""
        if self.maturity_date <= self.issue_date:
            raise ValidationError(
                f"Maturity date {self.maturity_date} must be after issue date {self.issue_date}"
            )

        if self.settlement_days < 0:
            raise ValidationError(
                f"Settlement days must be non-negative, got {self.settlement_days}"
            )

        if self.first_coupon_date and self.first_coupon_date <= self.issue_date:
            raise ValidationError(
                f"First coupon date {self.first_coupon_date} must be after issue date"
            )

        if (
            self.penultimate_coupon_date
            and self.penultimate_coupon_date >= self.maturity_date
        ):
            raise ValidationError(
                f"Penultimate coupon date {self.penultimate_coupon_date} must be before maturity"
            )

    def generate_unadjusted_dates(self) -> List[datetime]:
        """
        Generate unadjusted payment dates (before business day adjustments).

        Returns:
            List of unadjusted payment dates
        """
        dates = []

        # Calculate period in months
        months_per_period = 12 // self.payment_frequency.periods_per_year

        # Handle irregular first period
        if self.first_coupon_date:
            dates.append(self.first_coupon_date)
            current_date = self.first_coupon_date
        else:
            # Start from maturity and work backwards for regular schedule
            current_date = self.maturity_date

        # Generate dates working backwards from maturity
        if not self.first_coupon_date:
            dates = []
            current_date = self.maturity_date

            while True:
                dates.append(current_date)
                # Move back by one period
                next_date = current_date - relativedelta(months=months_per_period)

                if next_date <= self.issue_date:
                    break

                current_date = next_date

            # Reverse to get chronological order
            dates.reverse()
        else:
            # Forward generation from first coupon date
            while current_date < self.maturity_date:
                next_date = current_date + relativedelta(months=months_per_period)
                if next_date >= self.maturity_date:
                    break
                dates.append(next_date)
                current_date = next_date

            # Always include maturity
            if dates[-1] != self.maturity_date:
                dates.append(self.maturity_date)

        return dates

    def generate_schedule(self, notional: float, coupon_rate: float) -> List[CashFlow]:
        """
        Generate complete payment schedule with cashflows.

        Args:
            notional: Bond notional/face value
            coupon_rate: Annual coupon rate

        Returns:
            List of CashFlow objects
        """
        if notional <= 0:
            raise ValidationError(f"Notional must be positive, got {notional}")

        if coupon_rate < 0:
            raise ValidationError(
                f"Coupon rate must be non-negative, got {coupon_rate}"
            )

        # Generate unadjusted dates
        unadjusted_dates = self.generate_unadjusted_dates()

        if not unadjusted_dates:
            raise ValidationError("No payment dates generated")

        # Adjust dates for business days
        adjusted_dates = [
            self.calendar.adjust_date(date, self.business_day_convention)
            for date in unadjusted_dates
        ]

        # Generate cashflows
        cashflows = []
        accrual_start = self.issue_date

        for i, payment_date in enumerate(adjusted_dates):
            accrual_end = unadjusted_dates[i]  # Use unadjusted for accrual

            # Calculate day count fraction
            day_count_fraction = calculate_day_count_fraction(
                accrual_start, accrual_end, self.day_count_convention
            )

            # Calculate coupon amount
            coupon_amount = notional * coupon_rate * day_count_fraction

            # Add principal repayment at maturity
            if i == len(adjusted_dates) - 1:
                coupon_amount += notional

            # Apply settlement delay
            settlement_date = payment_date
            if self.settlement_days > 0:
                settlement_date = self.calendar.add_business_days(
                    payment_date, self.settlement_days
                )

            cashflow = CashFlow(
                payment_date=settlement_date,
                accrual_start_date=accrual_start,
                accrual_end_date=accrual_end,
                notional=notional,
                rate=coupon_rate,
                day_count_fraction=day_count_fraction,
                amount=coupon_amount,
            )

            cashflows.append(cashflow)
            accrual_start = accrual_end

        return cashflows

    def __repr__(self):
        return (
            f"ScheduleGenerator(issue={self.issue_date.date()}, "
            f"maturity={self.maturity_date.date()}, "
            f"frequency={self.payment_frequency.name})"
        )


def calculate_accrued_interest(
    last_coupon_date: datetime,
    settlement_date: datetime,
    next_coupon_date: datetime,
    coupon_rate: float,
    notional: float,
    day_count_convention: DayCountConvention,
) -> float:
    """
    Calculate accrued interest for a bond.

    Args:
        last_coupon_date: Date of last coupon payment
        settlement_date: Settlement date for trade
        next_coupon_date: Date of next coupon payment
        coupon_rate: Annual coupon rate
        notional: Bond notional/face value
        day_count_convention: Day count convention

    Returns:
        Accrued interest amount

    Raises:
        ValidationError: If dates are invalid
    """
    if settlement_date <= last_coupon_date:
        return 0.0

    if settlement_date >= next_coupon_date:
        raise ValidationError(
            f"Settlement date {settlement_date} must be before next coupon date {next_coupon_date}"
        )

    # Calculate fraction of period that has accrued
    accrued_fraction = calculate_day_count_fraction(
        last_coupon_date, settlement_date, day_count_convention
    )

    # Calculate full period fraction
    period_fraction = calculate_day_count_fraction(
        last_coupon_date, next_coupon_date, day_count_convention
    )

    # Accrued interest is proportional to time elapsed
    if period_fraction > 0:
        accrued = notional * coupon_rate * accrued_fraction
    else:
        accrued = 0.0

    return accrued


def find_coupon_dates_for_settlement(
    cashflows: List[CashFlow], settlement_date: datetime
) -> tuple:
    """
    Find the last and next coupon dates relative to a settlement date.

    Args:
        cashflows: List of cashflows
        settlement_date: Settlement date

    Returns:
        Tuple of (last_coupon_date, next_coupon_date, next_cashflow_index)

    Raises:
        ValidationError: If settlement date is outside cashflow range
    """
    if not cashflows:
        raise ValidationError("No cashflows provided")

    # Find position in schedule
    for i, cf in enumerate(cashflows):
        if settlement_date <= cf.accrual_end_date:
            if i == 0:
                # Before first coupon
                return (cf.accrual_start_date, cf.accrual_end_date, i)
            else:
                # Between coupons
                return (cashflows[i - 1].accrual_end_date, cf.accrual_end_date, i)

    # After all cashflows
    raise ValidationError(f"Settlement date {settlement_date} is after all cashflows")
