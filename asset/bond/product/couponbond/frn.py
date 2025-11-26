"""
Floating Rate Note (FRN) product.

Implements floating rate bonds that pay coupons based on a reference rate
index plus a spread, with support for various reset conventions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from dateutil.relativedelta import relativedelta

from asset.bond.product.base_bond_product import BaseBondProduct
from asset.bond.schedule.cashflow import CashFlow
from param.index import RateIndex, IndexFixingStore
from param.rrf import RateCurve
from util.enum import ResetConvention
from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    Calendar,
    CalendarType,
    create_calendar,
    calculate_day_count_fraction,
)
from util.enum import PaymentFrequency, StubType
from util.exceptions import ValidationError


@dataclass
class FloatingCashFlow:
    """
    Represents a single cash flow for a floating rate bond.

    Extends the basic CashFlow concept with floating rate specific fields.

    Attributes:
        payment_date: Date when payment is made
        accrual_start_date: Start of accrual period
        accrual_end_date: End of accrual period
        fixing_date: Date when the rate is fixed
        notional: Notional amount for this period
        spread: Spread over the index rate (annual, e.g., 0.01 for 100bp)
        day_count_fraction: Year fraction for this period
        index_fixing: Actual fixing if known (None if projected)
        forward_rate: Projected rate if fixing is unknown
        is_projected: Whether the rate is projected (vs known fixing)
        rate_cap: Optional rate cap
        rate_floor: Optional rate floor
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

    @property
    def effective_rate(self) -> float:
        """
        Get the effective coupon rate (index + spread), applying cap/floor.

        Returns:
            Effective rate for this period
        """
        if self.index_fixing is not None:
            base_rate = self.index_fixing
        elif self.forward_rate is not None:
            base_rate = self.forward_rate
        else:
            base_rate = 0.0

        total_rate = base_rate + self.spread

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


@dataclass
class FloatingRateBond(BaseBondProduct):
    """
    Floating Rate Note (FRN).

    A bond that pays coupons based on a floating reference rate plus a spread.
    The reference rate is reset periodically according to the reset convention.

    Attributes:
        issue_date: Bond issue date
        maturity_date: Bond maturity date
        notional: Face value/principal amount
        index: Reference rate index (e.g., SOFR, EURIBOR_3M)
        spread: Spread over the index (e.g., 0.0050 for 50bp)
        payment_frequency: Coupon payment frequency
        reset_convention: When rate is fixed (in-advance or in-arrears)
        day_count_convention: Day count convention for coupon calculation
        calendar: Business day calendar
        business_day_convention: Convention for adjusting payment dates
        settlement_days: Days between accrual end and payment
        lookback_days: Days to look back for fixing (in-arrears only)
        lockout_days: Days before period end to lock rate (in-arrears only)
        rate_cap: Optional maximum coupon rate
        rate_floor: Optional minimum coupon rate
        fixing_store: Store for historical index fixings
        stub_type: Type of stub period if any
        first_coupon_date: Date of first coupon for irregular first period
    """

    issue_date: datetime
    maturity_date: datetime
    notional: float
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
    fixing_store: Optional[IndexFixingStore] = None
    stub_type: StubType = StubType.NONE
    first_coupon_date: Optional[datetime] = None

    # Cached schedule
    _cached_schedule: Optional[List[FloatingCashFlow]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize and validate the FRN."""
        # Set default calendar if not provided
        if self.calendar is None:
            self.calendar = create_calendar(self.index.calendar_type)

        # Initialize fixing store if not provided
        if self.fixing_store is None:
            self.fixing_store = IndexFixingStore()

        # Link fixing store to index
        self.index.set_fixing_store(self.fixing_store)

        # Validate parameters
        self.validate()

        # Generate initial schedule (without forward rates - those come from engine)
        self._cached_schedule = self._generate_schedule()

    def validate(self) -> None:
        """
        Validate FRN parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        if self.maturity_date <= self.issue_date:
            raise ValidationError(
                f"Maturity date {self.maturity_date} must be after issue date {self.issue_date}"
            )

        if self.notional <= 0:
            raise ValidationError(f"Notional must be positive, got {self.notional}")

        if self.settlement_days < 0:
            raise ValidationError(
                f"Settlement days must be non-negative, got {self.settlement_days}"
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
                    f"Rate cap {self.rate_cap} must be >= rate floor {self.rate_floor}"
                )

        if self.first_coupon_date and self.first_coupon_date <= self.issue_date:
            raise ValidationError(
                f"First coupon date {self.first_coupon_date} must be after issue date"
            )

    def _generate_unadjusted_dates(self) -> List[datetime]:
        """
        Generate unadjusted payment dates.

        Returns:
            List of unadjusted payment dates
        """
        dates = []
        months_per_period = 12 // self.payment_frequency.periods_per_year

        if self.first_coupon_date:
            dates.append(self.first_coupon_date)
            current_date = self.first_coupon_date

            while current_date < self.maturity_date:
                next_date = current_date + relativedelta(months=months_per_period)
                if next_date >= self.maturity_date:
                    break
                dates.append(next_date)
                current_date = next_date

            if dates[-1] != self.maturity_date:
                dates.append(self.maturity_date)
        else:
            # Generate backwards from maturity
            current_date = self.maturity_date

            while True:
                dates.append(current_date)
                next_date = current_date - relativedelta(months=months_per_period)

                if next_date <= self.issue_date:
                    break

                current_date = next_date

            dates.reverse()

        return dates

    def _calculate_fixing_date(
        self, accrual_start: datetime, accrual_end: datetime
    ) -> datetime:
        """
        Calculate the fixing date for an accrual period.

        Args:
            accrual_start: Start of accrual period
            accrual_end: End of accrual period

        Returns:
            The fixing date
        """
        if self.reset_convention == ResetConvention.IN_ADVANCE:
            # In-advance: fix at start of period minus fixing lag
            fixing_date = accrual_start
            if self.index.fixing_lag_days > 0:
                fixing_date = self.calendar.add_business_days(
                    fixing_date, -self.index.fixing_lag_days
                )
            return fixing_date
        else:
            # In-arrears: fix near end of period with lookback/lockout
            if self.lookback_days > 0:
                fixing_date = self.calendar.add_business_days(
                    accrual_end, -self.lookback_days
                )
            elif self.lockout_days > 0:
                fixing_date = self.calendar.add_business_days(
                    accrual_end, -self.lockout_days
                )
            else:
                # Default: fix at period end minus fixing lag
                fixing_date = accrual_end
                if self.index.fixing_lag_days > 0:
                    fixing_date = self.calendar.add_business_days(
                        fixing_date, -self.index.fixing_lag_days
                    )
            return fixing_date

    def _generate_schedule(self) -> List[FloatingCashFlow]:
        """
        Generate the complete payment schedule with floating cashflows.

        Returns:
            List of FloatingCashFlow objects
        """
        unadjusted_dates = self._generate_unadjusted_dates()

        if not unadjusted_dates:
            raise ValidationError("No payment dates generated")

        # Adjust dates for business days
        adjusted_dates = [
            self.calendar.adjust_date(date, self.business_day_convention)
            for date in unadjusted_dates
        ]

        cashflows = []
        accrual_start = self.issue_date

        for i, payment_date in enumerate(adjusted_dates):
            accrual_end = unadjusted_dates[i]

            # Calculate fixing date
            fixing_date = self._calculate_fixing_date(accrual_start, accrual_end)

            # Calculate day count fraction
            day_count_fraction = calculate_day_count_fraction(
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
                notional=self.notional,
                spread=self.spread,
                day_count_fraction=day_count_fraction,
                index_fixing=fixing,
                forward_rate=None,  # Set by engine
                is_projected=fixing is None,
                rate_cap=self.rate_cap,
                rate_floor=self.rate_floor,
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
                # Calculate time to fixing date
                time_to_fixing = (cf.fixing_date - valuation_date).days / 365.0

                if time_to_fixing < 0:
                    # Fixing date is in the past but no fixing - use current rate
                    time_to_fixing = 0.0

                # Get forward rate for the index tenor
                if self.index.is_overnight:
                    # For overnight rates, use instantaneous forward
                    cf.forward_rate = forward_curve.get_rate(time_to_fixing)
                else:
                    # For term rates, calculate forward rate for the tenor
                    t1 = time_to_fixing
                    t2 = t1 + self.index.tenor_years
                    cf.forward_rate = forward_curve.get_forward_rate(t1, t2)

    def add_fixing(self, fixing_date: datetime, rate: float) -> None:
        """
        Add a historical fixing for the index.

        Args:
            fixing_date: Date of the fixing
            rate: Fixed rate value
        """
        from param.index import IndexFixing

        self.fixing_store.add_fixing(
            IndexFixing(fixing_date=fixing_date, rate=rate, index_name=self.index.name)
        )

        # Refresh schedule to incorporate new fixing
        self._cached_schedule = self._generate_schedule()

    def get_floating_cashflows(
        self, valuation_date: datetime
    ) -> List[FloatingCashFlow]:
        """
        Get all future floating cashflows from valuation date.

        Args:
            valuation_date: Date to value the bond

        Returns:
            List of future FloatingCashFlow objects
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return [cf for cf in self._cached_schedule if cf.payment_date > valuation_date]

    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """
        Get all future cashflows as standard CashFlow objects.

        Note: For FRNs, projected cashflows use estimated forward rates.
        Use get_floating_cashflows for more detail.

        Args:
            valuation_date: Date to value the bond

        Returns:
            List of future cashflows
        """
        floating_cfs = self.get_floating_cashflows(valuation_date)

        # Convert to standard cashflows
        cashflows = [cf.to_cashflow() for cf in floating_cfs]

        # Add principal repayment to last cashflow
        if cashflows:
            last_cf = cashflows[-1]
            cashflows[-1] = CashFlow(
                payment_date=last_cf.payment_date,
                accrual_start_date=last_cf.accrual_start_date,
                accrual_end_date=last_cf.accrual_end_date,
                notional=last_cf.notional,
                rate=last_cf.rate,
                day_count_fraction=last_cf.day_count_fraction,
                amount=last_cf.amount + self.notional,
            )

        return cashflows

    def get_all_floating_cashflows(self) -> List[FloatingCashFlow]:
        """
        Get all cashflows (past and future).

        Returns:
            List of all FloatingCashFlow objects
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return self._cached_schedule

    def get_maturity_date(self) -> datetime:
        """Get the maturity date of the bond."""
        return self.maturity_date

    def get_issue_date(self) -> datetime:
        """Get the issue date of the bond."""
        return self.issue_date

    def get_notional(self) -> float:
        """Get the notional/face value of the bond."""
        return self.notional

    def calculate_accrued_interest(self, settlement_date: datetime) -> float:
        """
        Calculate accrued interest as of settlement date.

        For FRNs, accrued interest is based on the current period's
        rate (either fixed or projected).

        Args:
            settlement_date: Settlement date for the calculation

        Returns:
            Accrued interest amount
        """
        if settlement_date <= self.issue_date:
            return 0.0

        if settlement_date >= self.maturity_date:
            return 0.0

        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        # Find the current accrual period
        for cf in self._cached_schedule:
            if cf.accrual_start_date <= settlement_date < cf.accrual_end_date:
                # Calculate fraction of period that has accrued
                accrued_fraction = calculate_day_count_fraction(
                    cf.accrual_start_date, settlement_date, self.day_count_convention
                )

                # Accrued interest based on current rate
                return self.notional * cf.effective_rate * accrued_fraction

        return 0.0

    def get_current_coupon_rate(self, as_of_date: datetime) -> Optional[float]:
        """
        Get the current coupon rate (index + spread) for a given date.

        Args:
            as_of_date: Date to check

        Returns:
            Current coupon rate if in an active period, None otherwise
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        for cf in self._cached_schedule:
            if cf.accrual_start_date <= as_of_date < cf.accrual_end_date:
                return cf.effective_rate

        return None

    def get_next_reset_date(self, as_of_date: datetime) -> Optional[datetime]:
        """
        Get the next rate reset date.

        Args:
            as_of_date: Date to check from

        Returns:
            Next fixing date, or None if bond has matured
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        for cf in self._cached_schedule:
            if cf.fixing_date > as_of_date:
                return cf.fixing_date

        return None

    def __repr__(self):
        return (
            f"FloatingRateBond("
            f"index={self.index.name}, "
            f"spread={self.spread:.2%}, "
            f"maturity={self.maturity_date.date()}, "
            f"notional={self.notional:.2f})"
        )


def create_simple_frn(
    issue_date: datetime,
    maturity_date: datetime,
    notional: float,
    index: RateIndex,
    spread: float,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    reset_convention: ResetConvention = ResetConvention.IN_ADVANCE,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360,
) -> FloatingRateBond:
    """
    Create a simple FRN with standard conventions.

    This is a convenience function for creating an FRN with:
    - Calendar based on index
    - Modified following business day convention
    - No settlement delay
    - No caps or floors

    Args:
        issue_date: Bond issue date
        maturity_date: Bond maturity date
        notional: Face value/principal amount
        index: Reference rate index
        spread: Spread over index (e.g., 0.005 for 50bp)
        payment_frequency: Payment frequency (default: quarterly)
        reset_convention: Rate reset convention (default: in-advance)
        day_count_convention: Day count convention (default: ACT/360)

    Returns:
        FloatingRateBond object
    """
    return FloatingRateBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        notional=notional,
        index=index,
        spread=spread,
        payment_frequency=payment_frequency,
        reset_convention=reset_convention,
        day_count_convention=day_count_convention,
        calendar=create_calendar(index.calendar_type),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        settlement_days=0,
        stub_type=StubType.NONE,
    )
