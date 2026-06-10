"""
Forward Rate Agreement (FRA) product.

A FRA is a single-period forward contract on an interest rate.
The buyer pays a fixed rate and receives the floating reference rate
on a notional amount for a future accrual period.

Settlement is typically at the start of the accrual period (FRA settlement date),
discounted back from the accrual end date.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dateutil.relativedelta import relativedelta

from quantark.param.index import RateIndex, IndexFixingStore
from quantark.util.calendar import (
    DayCountConvention,
    Calendar,
    calculate_day_count_fraction,
    create_calendar,
)
from quantark.util.exceptions import ValidationError


@dataclass
class ForwardRateAgreement:
    """
    Forward Rate Agreement (FRA).

    A single-period forward contract where the buyer agrees to pay a fixed rate
    and receive the floating reference rate on a notional for a future period.

    The FRA settles at the accrual start date (settlement date), with the
    settlement amount discounted from the accrual end date.

    Attributes:
        notional: Notional principal amount
        fixed_rate: Agreed FRA rate (annualized)
        accrual_start: Start of the FRA accrual period (settlement date)
        accrual_end: End of the FRA accrual period
        index: Reference rate index (e.g., SOFR_3M)
        day_count_convention: Day count convention for accrual calculation
        calendar: Business day calendar
        trade_date: Date the FRA was traded
        fixing_store: Store for historical index fixings
    """

    notional: float
    fixed_rate: float
    accrual_start: datetime
    accrual_end: datetime
    index: RateIndex
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360
    calendar: Optional[Calendar] = None
    trade_date: Optional[datetime] = None
    fixing_store: Optional[IndexFixingStore] = None

    def __post_init__(self):
        """Initialize and validate the FRA."""
        if self.calendar is None:
            self.calendar = create_calendar(self.index.calendar_type)

        if self.fixing_store is None:
            self.fixing_store = IndexFixingStore()

        self.index.set_fixing_store(self.fixing_store)
        self.validate()

    def validate(self) -> None:
        """Validate FRA parameters."""
        if self.notional <= 0:
            raise ValidationError(
                f"Notional must be positive, got {self.notional}"
            )
        if self.accrual_end <= self.accrual_start:
            raise ValidationError(
                f"Accrual end {self.accrual_end} must be after "
                f"accrual start {self.accrual_start}"
            )
        if self.fixed_rate < -0.10 or self.fixed_rate > 0.50:
            raise ValidationError(
                f"Fixed rate {self.fixed_rate} seems unreasonable"
            )
        if self.trade_date is not None and self.trade_date > self.accrual_start:
            raise ValidationError(
                f"Trade date {self.trade_date} must be on or before "
                f"accrual start {self.accrual_start}"
            )

    def get_notional(self) -> float:
        """Get the notional amount."""
        return self.notional

    def get_accrual_start(self) -> datetime:
        """Get the accrual start (settlement) date."""
        return self.accrual_start

    def get_maturity(self) -> datetime:
        """Get the maturity date (accrual end)."""
        return self.accrual_end

    def time_to_settlement(self, valuation_date: datetime) -> float:
        """
        Calculate time to settlement in years.

        Args:
            valuation_date: Date from which to measure

        Returns:
            Time to settlement in years (ACT/365 basis)
        """
        return (self.accrual_start - valuation_date).days / 365.0

    def time_to_maturity(self, valuation_date: datetime) -> float:
        """
        Calculate time to maturity in years.

        Args:
            valuation_date: Date from which to measure

        Returns:
            Time to maturity in years (ACT/365 basis)
        """
        return (self.accrual_end - valuation_date).days / 365.0

    def day_count_fraction(self) -> float:
        """
        Calculate the day count fraction for the accrual period.

        Returns:
            Day count fraction using the configured convention
        """
        return calculate_day_count_fraction(
            self.accrual_start, self.accrual_end, self.day_count_convention
        )

    def is_expired(self, valuation_date: datetime) -> bool:
        """
        Check if the FRA has expired (past settlement date).

        Args:
            valuation_date: Date to check against

        Returns:
            True if the FRA has expired
        """
        return valuation_date >= self.accrual_start

    def __repr__(self):
        return (
            f"ForwardRateAgreement(notional={self.notional:,.0f}, "
            f"rate={self.fixed_rate:.4%}, "
            f"settlement={self.accrual_start.date()}, "
            f"maturity={self.accrual_end.date()}, "
            f"index={self.index.name})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_fra(
    trade_date: datetime,
    settlement_date: datetime,
    tenor_months: int,
    notional: float,
    fixed_rate: float,
    index: RateIndex,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Optional[Calendar] = None,
) -> ForwardRateAgreement:
    """
    Create a Forward Rate Agreement from trade date and tenor.

    Args:
        trade_date: Date the FRA is traded
        settlement_date: FRA settlement date (accrual start)
        tenor_months: Length of the accrual period in months
        notional: Notional principal amount
        fixed_rate: Agreed FRA rate (annualized)
        index: Reference rate index
        day_count_convention: Day count convention
        calendar: Business day calendar

    Returns:
        ForwardRateAgreement object

    Example:
        >>> fra = create_fra(
        ...     trade_date=datetime(2024, 1, 15),
        ...     settlement_date=datetime(2024, 4, 15),
        ...     tenor_months=3,
        ...     notional=10_000_000,
        ...     fixed_rate=0.05,
        ...     index=SOFR_3M,
        ... )
    """
    if tenor_months <= 0:
        raise ValidationError(
            f"Tenor months must be positive, got {tenor_months}"
        )

    accrual_end = settlement_date + relativedelta(months=tenor_months)

    return ForwardRateAgreement(
        notional=notional,
        fixed_rate=fixed_rate,
        accrual_start=settlement_date,
        accrual_end=accrual_end,
        index=index,
        day_count_convention=day_count_convention,
        calendar=calendar,
        trade_date=trade_date,
    )
