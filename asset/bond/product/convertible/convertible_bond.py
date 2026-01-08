"""
Convertible bond product definition.

This module provides the ConvertibleBond dataclass representing a convertible
bond instrument with comprehensive contract terms including:
- Bond characteristics (face value, maturity, coupon)
- Conversion features (conversion ratio, conversion schedule)
- Call/put schedules
- Credit attributes (credit spread, hazard rate, recovery rate)
- Dividend handling (continuous and discrete)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from asset.bond.product.base_bond_product import BaseBondProduct
from asset.bond.schedule.cashflow import (
    CashFlow,
    ScheduleGenerator,
    calculate_accrued_interest,
    find_coupon_dates_for_settlement,
)
from util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    Calendar,
    CalendarType,
    create_calendar,
)
from util.enum import PaymentFrequency, StubType
from util.exceptions import ValidationError
from util.numerical import validate_positive, validate_probability


@dataclass
class CallScheduleEntry:
    """
    Represents a call schedule entry for callable bonds.

    The issuer can call (redeem early) the bond at the call price
    starting from the call date.

    Attributes:
        call_date: Date from which the bond is callable
        call_price: Price at which issuer can redeem (typically par=100 or premium)
        soft_call: If True, requires stock price above trigger level to call
        trigger_level: Stock price trigger for soft call (as % of conversion price)
    """

    call_date: datetime
    call_price: float
    soft_call: bool = False
    trigger_level: Optional[float] = None

    def __post_init__(self):
        """Validate call schedule entry."""
        if self.call_price <= 0:
            raise ValidationError(f"Call price must be positive, got {self.call_price}")
        if self.soft_call and self.trigger_level is None:
            raise ValidationError("Soft call requires trigger_level to be specified")
        if self.trigger_level is not None and self.trigger_level <= 0:
            raise ValidationError(
                f"Trigger level must be positive, got {self.trigger_level}"
            )


@dataclass
class PutScheduleEntry:
    """
    Represents a put schedule entry for puttable bonds.

    The holder can put (sell back to issuer) the bond at the put price
    on the put date.

    Attributes:
        put_date: Date on which the bond is puttable
        put_price: Price at which holder can sell back (typically par=100)
    """

    put_date: datetime
    put_price: float

    def __post_init__(self):
        """Validate put schedule entry."""
        if self.put_price <= 0:
            raise ValidationError(f"Put price must be positive, got {self.put_price}")


@dataclass
class DiscreteDividend:
    """
    Represents a discrete dividend payment.

    Attributes:
        ex_date: Ex-dividend date (stock drops by dividend amount on this date)
        amount: Dividend amount per share (absolute, not yield)
    """

    ex_date: datetime
    amount: float

    def __post_init__(self):
        """Validate discrete dividend."""
        if self.amount < 0:
            raise ValidationError(f"Dividend amount must be non-negative, got {self.amount}")


@dataclass
class ConvertibleBond(BaseBondProduct):
    """
    Convertible bond product.

    A convertible bond is a hybrid security combining debt and equity features.
    The holder has the right to convert the bond into a predetermined number
    of shares of the issuer's common stock.

    Attributes:
        issue_date: Bond issue date
        maturity_date: Bond maturity date
        face_value: Par value of the bond (typically 100 or 1000)
        coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
        conversion_ratio: Number of shares received upon conversion per bond
        payment_frequency: Coupon payment frequency
        day_count_convention: Day count convention for coupon calculations

        # Conversion features
        conversion_price: Price per share at which conversion occurs
            (calculated as face_value / conversion_ratio if not provided)
        conversion_start_date: Date from which conversion is allowed
            (defaults to issue_date)
        conversion_end_date: Date until which conversion is allowed
            (defaults to maturity_date)

        # Call/Put schedules
        call_schedule: List of call schedule entries (issuer's option)
        put_schedule: List of put schedule entries (holder's option)

        # Credit attributes
        credit_spread: Credit spread over risk-free rate (e.g., 0.02 for 200bp)
        hazard_rate: Default intensity for jump-diffusion model (annualized)
        recovery_rate: Recovery rate upon default (typically 0.4)
        stock_jump_on_default: Stock price jump multiplier on default (eta, typically 0.4)

        # Dividend handling
        continuous_dividend_yield: Continuous dividend yield (annualized)
        discrete_dividends: List of discrete dividend payments

        # Calendar and conventions
        calendar: Business day calendar
        business_day_convention: Convention for adjusting payment dates
        settlement_days: Days between trade and settlement
    """

    # Core bond attributes
    issue_date: datetime
    maturity_date: datetime
    face_value: float
    coupon_rate: float
    conversion_ratio: float
    payment_frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL
    day_count_convention: DayCountConvention = DayCountConvention.ACT_ACT_ISDA

    # Conversion features
    conversion_price: Optional[float] = None
    conversion_start_date: Optional[datetime] = None
    conversion_end_date: Optional[datetime] = None

    # Call/Put schedules
    call_schedule: List[CallScheduleEntry] = field(default_factory=list)
    put_schedule: List[PutScheduleEntry] = field(default_factory=list)

    # Credit attributes
    credit_spread: float = 0.0
    hazard_rate: float = 0.0
    recovery_rate: float = 0.4
    stock_jump_on_default: float = 0.4

    # Dividend handling
    continuous_dividend_yield: float = 0.0
    discrete_dividends: List[DiscreteDividend] = field(default_factory=list)

    # Calendar and conventions
    calendar: Optional[Calendar] = None
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING
    settlement_days: int = 0

    # Cached schedule
    _cached_schedule: Optional[List[CashFlow]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        """Initialize and validate the convertible bond."""
        # Set default calendar if not provided
        if self.calendar is None:
            self.calendar = create_calendar(CalendarType.NONE)

        # Set default conversion dates
        if self.conversion_start_date is None:
            self.conversion_start_date = self.issue_date
        if self.conversion_end_date is None:
            self.conversion_end_date = self.maturity_date

        # Calculate conversion price if not provided
        if self.conversion_price is None:
            if self.conversion_ratio <= 0:
                raise ValidationError(
                    f"Conversion ratio must be positive, got {self.conversion_ratio}"
                )
            self.conversion_price = self.face_value / self.conversion_ratio

        # Validate parameters
        self.validate()

        # Generate coupon schedule
        self._cached_schedule = self._generate_schedule()

    def validate(self) -> None:
        """
        Validate convertible bond parameters.

        Raises:
            ValidationError: If parameters are invalid
        """
        # Date validations
        if self.maturity_date <= self.issue_date:
            raise ValidationError(
                f"Maturity date {self.maturity_date} must be after issue date {self.issue_date}"
            )

        if self.conversion_start_date > self.conversion_end_date:
            raise ValidationError(
                f"Conversion start date {self.conversion_start_date} must be <= "
                f"conversion end date {self.conversion_end_date}"
            )

        if self.conversion_end_date > self.maturity_date:
            raise ValidationError(
                f"Conversion end date {self.conversion_end_date} cannot be after "
                f"maturity date {self.maturity_date}"
            )

        # Value validations
        self.face_value = validate_positive(self.face_value, "face_value")
        self.conversion_ratio = validate_positive(self.conversion_ratio, "conversion_ratio")

        if self.coupon_rate < 0:
            raise ValidationError(
                f"Coupon rate must be non-negative, got {self.coupon_rate}"
            )

        if self.conversion_price is not None and self.conversion_price <= 0:
            raise ValidationError(
                f"Conversion price must be positive, got {self.conversion_price}"
            )

        # Credit validations
        if self.credit_spread < 0:
            raise ValidationError(
                f"Credit spread must be non-negative, got {self.credit_spread}"
            )

        if self.hazard_rate < 0:
            raise ValidationError(
                f"Hazard rate must be non-negative, got {self.hazard_rate}"
            )

        self.recovery_rate = validate_probability(self.recovery_rate, "recovery_rate")

        if self.stock_jump_on_default < 0 or self.stock_jump_on_default > 1:
            raise ValidationError(
                f"Stock jump on default must be in [0, 1], got {self.stock_jump_on_default}"
            )

        # Dividend validation
        if self.continuous_dividend_yield < 0:
            raise ValidationError(
                f"Continuous dividend yield must be non-negative, "
                f"got {self.continuous_dividend_yield}"
            )

        if self.settlement_days < 0:
            raise ValidationError(
                f"Settlement days must be non-negative, got {self.settlement_days}"
            )

        # Validate call schedule dates are in order and within bond lifetime
        for i, call_entry in enumerate(self.call_schedule):
            if call_entry.call_date < self.issue_date:
                raise ValidationError(
                    f"Call date {call_entry.call_date} cannot be before issue date"
                )
            if call_entry.call_date > self.maturity_date:
                raise ValidationError(
                    f"Call date {call_entry.call_date} cannot be after maturity date"
                )
            if i > 0 and call_entry.call_date < self.call_schedule[i - 1].call_date:
                raise ValidationError("Call schedule must be in chronological order")

        # Validate put schedule dates
        for i, put_entry in enumerate(self.put_schedule):
            if put_entry.put_date < self.issue_date:
                raise ValidationError(
                    f"Put date {put_entry.put_date} cannot be before issue date"
                )
            if put_entry.put_date > self.maturity_date:
                raise ValidationError(
                    f"Put date {put_entry.put_date} cannot be after maturity date"
                )
            if i > 0 and put_entry.put_date < self.put_schedule[i - 1].put_date:
                raise ValidationError("Put schedule must be in chronological order")

        # Validate discrete dividends
        for div in self.discrete_dividends:
            if div.ex_date > self.maturity_date:
                raise ValidationError(
                    f"Dividend ex-date {div.ex_date} cannot be after maturity date"
                )

    def _generate_schedule(self) -> List[CashFlow]:
        """
        Generate the coupon payment schedule.

        Returns:
            List of CashFlow objects representing coupon payments and principal
        """
        generator = ScheduleGenerator(
            issue_date=self.issue_date,
            maturity_date=self.maturity_date,
            payment_frequency=self.payment_frequency,
            day_count_convention=self.day_count_convention,
            calendar=self.calendar,
            business_day_convention=self.business_day_convention,
            settlement_days=self.settlement_days,
        )

        return generator.generate_schedule(
            notional=self.face_value, coupon_rate=self.coupon_rate
        )

    def get_cashflows(self, valuation_date: datetime) -> List[CashFlow]:
        """
        Get all future cashflows from valuation date.

        Note: This returns the scheduled cashflows assuming no conversion.
        Conversion value is handled separately by the pricing engine.

        Args:
            valuation_date: Date to value the bond

        Returns:
            List of future cashflows
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._generate_schedule()

        return [cf for cf in self._cached_schedule if cf.payment_date > valuation_date]

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
        return self.face_value

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

        all_cashflows = self.get_all_cashflows()
        if not all_cashflows:
            return 0.0

        try:
            last_coupon_date, next_coupon_date, _ = find_coupon_dates_for_settlement(
                all_cashflows, settlement_date
            )

            return calculate_accrued_interest(
                last_coupon_date=last_coupon_date,
                settlement_date=settlement_date,
                next_coupon_date=next_coupon_date,
                coupon_rate=self.coupon_rate,
                notional=self.face_value,
                day_count_convention=self.day_count_convention,
            )
        except ValidationError:
            return 0.0

    def parity(self, stock_price: float) -> float:
        """
        Calculate conversion parity (value if converted immediately).

        Conversion parity is the value of the shares the bondholder
        would receive upon conversion, expressed per bond.

        Args:
            stock_price: Current stock price

        Returns:
            Conversion parity value
        """
        return stock_price * self.conversion_ratio

    def conversion_premium(self, stock_price: float, bond_price: float) -> float:
        """
        Calculate conversion premium.

        The conversion premium is the percentage by which the bond price
        exceeds its conversion parity.

        Args:
            stock_price: Current stock price
            bond_price: Current bond price (clean price per face_value)

        Returns:
            Conversion premium as a decimal (e.g., 0.20 for 20% premium)
        """
        parity = self.parity(stock_price)
        if parity <= 0:
            return float("inf")
        return (bond_price - parity) / parity

    def is_callable_at(self, date: datetime, stock_price: Optional[float] = None) -> bool:
        """
        Check if the bond is callable at a given date.

        Args:
            date: Date to check callability
            stock_price: Current stock price (needed for soft call checks)

        Returns:
            True if the bond can be called at this date
        """
        if not self.call_schedule:
            return False

        # Find the latest applicable schedule entry
        active_entry = None
        for entry in self.call_schedule:
            if date >= entry.call_date:
                active_entry = entry

        if active_entry is None:
            return False

        if not active_entry.soft_call:
            return True

        # Soft call: check trigger condition
        if stock_price is not None and active_entry.trigger_level is not None:
            trigger_price = self.conversion_price * active_entry.trigger_level / 100.0
            if stock_price >= trigger_price:
                return True
        return False

    def get_call_price_at(self, date: datetime) -> Optional[float]:
        """
        Get the call price at a given date.

        Args:
            date: Date to get call price for

        Returns:
            Call price if callable, None otherwise
        """
        if not self.call_schedule:
            return None

        # Find the applicable call price (latest entry that's <= date)
        applicable_entry = None
        for entry in self.call_schedule:
            if date >= entry.call_date:
                applicable_entry = entry
        return applicable_entry.call_price if applicable_entry else None

    def is_puttable_at(self, date: datetime) -> bool:
        """
        Check if the bond is puttable at a given date.

        Args:
            date: Date to check puttability

        Returns:
            True if the bond can be put at this date
        """
        check_date = date.date()
        return any(entry.put_date.date() == check_date for entry in self.put_schedule)

    def get_put_price_at(self, date: datetime) -> Optional[float]:
        """
        Get the put price at a given date.

        Args:
            date: Date to get put price for

        Returns:
            Put price if puttable, None otherwise
        """
        check_date = date.date()
        for entry in self.put_schedule:
            if entry.put_date.date() == check_date:
                return entry.put_price
        return None

    def is_convertible_at(self, date: datetime) -> bool:
        """
        Check if conversion is allowed at a given date.

        Args:
            date: Date to check convertibility

        Returns:
            True if conversion is allowed
        """
        return self.conversion_start_date <= date <= self.conversion_end_date

    def get_discrete_dividends_between(
        self, start_date: datetime, end_date: datetime
    ) -> List[DiscreteDividend]:
        """
        Get discrete dividends with ex-dates between start and end dates.

        Args:
            start_date: Start of the period
            end_date: End of the period

        Returns:
            List of discrete dividends in the period
        """
        return [
            div
            for div in self.discrete_dividends
            if start_date <= div.ex_date <= end_date
        ]

    def get_coupon_payment(self) -> float:
        """
        Get the regular coupon payment amount (for regular periods).

        Returns:
            Coupon payment amount
        """
        return self.face_value * self.coupon_rate / self.payment_frequency.periods_per_year

    def time_to_maturity(self, valuation_date: datetime) -> float:
        """
        Calculate time to maturity in years.

        Args:
            valuation_date: Valuation date

        Returns:
            Time to maturity in years
        """
        delta = self.maturity_date - valuation_date
        return delta.days / 365.0

    def __repr__(self):
        return (
            f"ConvertibleBond("
            f"issue={self.issue_date.date()}, "
            f"maturity={self.maturity_date.date()}, "
            f"face_value={self.face_value:.2f}, "
            f"coupon={self.coupon_rate:.2%}, "
            f"conv_ratio={self.conversion_ratio:.4f})"
        )
