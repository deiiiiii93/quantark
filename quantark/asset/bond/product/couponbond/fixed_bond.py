"""
Fixed rate coupon bond product.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from asset.bond.product.base_bond_product import BaseBondProduct
from asset.bond.schedule.cashflow import (
    CashFlow,
    ScheduleGenerator,
    calculate_accrued_interest,
    find_coupon_dates_for_settlement
)
from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    Calendar,
    CalendarType,
    create_calendar
)
from util.enum import PaymentFrequency, StubType
from util.exceptions import ValidationError


@dataclass
class FixedBond(BaseBondProduct):
    """
    Fixed rate coupon bond.
    
    A bond that pays fixed coupon payments at regular intervals and
    returns the principal at maturity.
    
    Attributes:
        issue_date: Bond issue date
        maturity_date: Bond maturity date
        denominator: Minimum tradable notional amount
        coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
        payment_frequency: Payment frequency (annual, semi-annual, etc.)
        day_count_convention: Day count convention for calculating accruals
        calendar: Business day calendar
        business_day_convention: Convention for adjusting payment dates
        settlement_days: Days between accrual end and payment (default: 0)
        stub_type: Type of stub period if any (default: NONE)
        first_coupon_date: Date of first coupon for irregular first period
        penultimate_coupon_date: Date of penultimate coupon for irregular last period
    """
    issue_date: datetime
    maturity_date: datetime
    denominator: float
    coupon_rate: float
    payment_frequency: PaymentFrequency
    day_count_convention: DayCountConvention
    calendar: Optional[Calendar] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    settlement_days: int = 0
    stub_type: StubType = StubType.NONE
    first_coupon_date: Optional[datetime] = None
    penultimate_coupon_date: Optional[datetime] = None
    
    # Cached schedule
    _cached_schedule: Optional[List[CashFlow]] = None
    
    def __post_init__(self):
        """Initialize and validate the bond."""
        # Set default calendar if not provided
        if self.calendar is None:
            self.calendar = create_calendar(CalendarType.NONE)
        
        # Validate parameters
        self.validate()
        
        # Generate schedule
        self._cached_schedule = self._generate_schedule()
    
    def validate(self) -> None:
        """
        Validate bond parameters.
        
        Raises:
            ValidationError: If parameters are invalid
        """
        if self.maturity_date <= self.issue_date:
            raise ValidationError(
                f"Maturity date {self.maturity_date} must be after issue date {self.issue_date}"
            )
        
        if self.denominator <= 0:
            raise ValidationError(
                f"Denominator must be positive, got {self.denominator}"
            )
        
        if self.coupon_rate < 0:
            raise ValidationError(f"Coupon rate must be non-negative, got {self.coupon_rate}")
        
        if self.settlement_days < 0:
            raise ValidationError(
                f"Settlement days must be non-negative, got {self.settlement_days}"
            )
        
        if self.first_coupon_date and self.first_coupon_date <= self.issue_date:
            raise ValidationError(
                f"First coupon date {self.first_coupon_date} must be after issue date"
            )
        
        if self.first_coupon_date and self.first_coupon_date >= self.maturity_date:
            raise ValidationError(
                f"First coupon date {self.first_coupon_date} must be before maturity"
            )
    
    def _generate_schedule(self) -> List[CashFlow]:
        """
        Generate the complete payment schedule.
        
        Returns:
            List of CashFlow objects
        """
        generator = ScheduleGenerator(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            payment_frequency=self.payment_frequency,
            day_count_convention=self.day_count_convention,
            calendar=self.calendar,
            business_day_convention=self.business_day_convention,
            settlement_days=self.settlement_days,
            stub_type=self.stub_type,
            first_coupon_date=self.first_coupon_date,
            penultimate_coupon_date=self.penultimate_coupon_date
        )
        
        return generator.generate_schedule(
            notional=self.denominator,
            coupon_rate=self.coupon_rate
        )
    
    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """
        Get all future cashflows from valuation date.
        
        Args:
            valuation_date: Date to value the bond
            
        Returns:
            List of future cashflows
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()
        
        # Filter for future cashflows
        future_cashflows = [
            cf for cf in self._cached_schedule
            if cf.payment_date > valuation_date
        ]
        
        return future_cashflows
    
    def get_all_cashflows(self) -> List[CashFlow]:
        """
        Get all cashflows (past and future).
        
        Returns:
            List of all cashflows
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
    
    def get_denominator(self) -> float:
        """Get the minimum tradable notional (denominator) of the bond."""
        return self.denominator
    
    def calculate_accrued_interest(self, settlement_date: datetime) -> float:
        """
        Calculate accrued interest as of settlement date.
        
        Args:
            settlement_date: Settlement date for the calculation
            
        Returns:
            Accrued interest amount
        """
        if settlement_date <= self.issue_date:
            return 0.0
        
        if settlement_date >= self.maturity_date:
            return 0.0
        
        # Get all cashflows
        all_cashflows = self.get_all_cashflows()
        
        if not all_cashflows:
            return 0.0
        
        try:
            # Find the relevant coupon period
            last_coupon_date, next_coupon_date, _ = find_coupon_dates_for_settlement(
                all_cashflows,
                settlement_date
            )
            
            # Calculate accrued interest
            accrued = calculate_accrued_interest(
                last_coupon_date=last_coupon_date,
                settlement_date=settlement_date,
                next_coupon_date=next_coupon_date,
                coupon_rate=self.coupon_rate,
                notional=self.denominator,
                day_count_convention=self.day_count_convention
            )
            
            return accrued
            
        except ValidationError:
            # Settlement date is outside the schedule
            return 0.0
    
    def get_coupon_payment(self) -> float:
        """
        Get the regular coupon payment amount (for regular periods).
        
        Returns:
            Coupon payment amount
        """
        return (
            self.denominator * self.coupon_rate / self.payment_frequency.periods_per_year
        )
    
    def __repr__(self):
        return (f"FixedBond("
                f"issue={self.issue_date.date()}, "
                f"maturity={self.maturity_date.date()}, "
                f"denominator={self.denominator:.2f}, "
                f"coupon={self.coupon_rate:.2%}, "
                f"freq={self.payment_frequency.name})")


def create_simple_fixed_bond(
    issue_date: datetime,
    maturity_date: datetime,
    denominator: float,
    coupon_rate: float,
    payment_frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_ACT_ISDA,
) -> FixedBond:
    """
    Create a simple fixed bond with standard conventions.
    
    This is a convenience function for creating a bond with:
    - No holidays calendar
    - Modified following business day convention
    - No settlement delay
    - No stub periods
    
    Args:
        issue_date: Bond issue date
        maturity_date: Bond maturity date
        denominator: Minimum tradable notional amount
        coupon_rate: Annual coupon rate
        payment_frequency: Payment frequency (default: semi-annual)
        day_count_convention: Day count convention (default: ACT/ACT ISDA)
        
    Returns:
        FixedBond object
    """
    return FixedBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=denominator,
        coupon_rate=coupon_rate,
        payment_frequency=payment_frequency,
        day_count_convention=day_count_convention,
        calendar=create_calendar(CalendarType.NONE),
        business_day_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        settlement_days=0,
        stub_type=StubType.NONE
    )
