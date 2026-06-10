"""
Interest Rate Cap, Floor, and Collar products.

A Cap is a portfolio of caplets, each being a call option on a forward rate.
A Floor is a portfolio of floorlets, each being a put option on a forward rate.
A Collar is a combination of a long Cap and a short Floor (or vice versa).

Cap/Floor are symmetric products sharing the same structure, distinguished
by CapFloorType. Each caplet/floorlet covers one accrual period and pays
max(0, L - K) * dcf * N (for caplets) or max(0, K - L) * dcf * N (for floorlets),
where L is the reference rate, K is the strike, dcf is the day count fraction,
and N is the notional.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from dateutil.relativedelta import relativedelta

from quantark.asset.rate.product.irs import NotionalSchedule
from quantark.param.index import RateIndex, IndexFixingStore
from quantark.util.calendar import (
    DayCountConvention,
    BusinessDayConvention,
    Calendar,
    calculate_day_count_fraction,
    create_calendar,
)
from quantark.util.enum import PaymentFrequency
from quantark.util.exceptions import ValidationError


class CapFloorType(Enum):
    """Type of cap/floor instrument."""

    CAP = "cap"
    FLOOR = "floor"


@dataclass
class Caplet:
    """
    A single caplet or floorlet within a Cap/Floor.

    Represents one period's option on the reference rate.

    Attributes:
        accrual_start: Start of the accrual period
        accrual_end: End of the accrual period
        payment_date: Date the cashflow is paid
        fixing_date: Date the reference rate is observed
        notional: Notional for this period
        strike: Cap/floor strike rate
        day_count_fraction: DCF for the accrual period
        is_projected: True if fixing is in the future (not yet observed)
        index_fixing: Observed rate (if already fixed)
    """

    accrual_start: datetime
    accrual_end: datetime
    payment_date: datetime
    fixing_date: datetime
    notional: float
    strike: float
    day_count_fraction: float
    is_projected: bool
    index_fixing: Optional[float] = None

    def __repr__(self):
        status = "projected" if self.is_projected else f"fixed={self.index_fixing:.4%}"
        return (
            f"Caplet({self.accrual_start.date()}->{self.accrual_end.date()}, "
            f"K={self.strike:.4%}, N={self.notional:,.0f}, {status})"
        )


@dataclass
class CapFloor:
    """
    Interest Rate Cap or Floor.

    A Cap is a series of caplets (call options on the reference rate).
    A Floor is a series of floorlets (put options on the reference rate).

    The cap/floor pays at each period end:
    - Cap:   max(0, reference_rate - strike) * dcf * notional
    - Floor: max(0, strike - reference_rate) * dcf * notional

    Attributes:
        notional: Notional principal amount
        strike: Cap/floor strike rate (annualized)
        cap_floor_type: CAP or FLOOR
        start_date: Effective date (start of first accrual period)
        end_date: Maturity date (end of last accrual period)
        index: Reference rate index
        payment_frequency: Frequency of caplet/floorlet periods
        day_count_convention: Day count convention for accrual
        calendar: Business day calendar
        business_day_convention: Convention for adjusting dates
        notional_schedule: Optional amortizing notional schedule
        fixing_store: Store for historical index fixings
    """

    notional: float
    strike: float
    cap_floor_type: CapFloorType
    start_date: datetime
    end_date: datetime
    index: RateIndex
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360
    calendar: Optional[Calendar] = None
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.MODIFIED_FOLLOWING
    )
    notional_schedule: Optional[NotionalSchedule] = None
    fixing_store: Optional[IndexFixingStore] = None

    # Cached caplets
    _cached_caplets: Optional[List[Caplet]] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize and validate the cap/floor."""
        if self.calendar is None:
            self.calendar = create_calendar(self.index.calendar_type)

        if self.fixing_store is None:
            self.fixing_store = IndexFixingStore()

        self.index.set_fixing_store(self.fixing_store)
        self.validate()
        self._cached_caplets = self._generate_caplets()

    def validate(self) -> None:
        """Validate cap/floor parameters."""
        if self.notional <= 0:
            raise ValidationError(
                f"Notional must be positive, got {self.notional}"
            )
        if self.strike < -0.10 or self.strike > 0.50:
            raise ValidationError(
                f"Strike rate {self.strike} seems unreasonable"
            )
        if self.end_date <= self.start_date:
            raise ValidationError(
                f"End date {self.end_date} must be after start date {self.start_date}"
            )

    def _generate_unadjusted_dates(self) -> List[datetime]:
        """Generate unadjusted period end dates (backward from end_date)."""
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

    def _generate_caplets(self) -> List[Caplet]:
        """Generate the caplet/floorlet schedule."""
        unadjusted_dates = self._generate_unadjusted_dates()

        if not unadjusted_dates:
            raise ValidationError("No caplet periods generated")

        adjusted_dates = [
            self.calendar.adjust_date(date, self.business_day_convention)
            for date in unadjusted_dates
        ]

        caplets = []
        accrual_start = self.start_date

        for i, payment_date in enumerate(adjusted_dates):
            accrual_end = unadjusted_dates[i]

            # Get notional for this period
            if self.notional_schedule is not None:
                period_notional = self.notional_schedule.get_notional(accrual_start)
            else:
                period_notional = self.notional

            # Fixing date: typically at period start with fixing lag
            fixing_date = accrual_start
            if self.index.fixing_lag_days > 0:
                fixing_date = self.calendar.add_business_days(
                    accrual_start, -self.index.fixing_lag_days
                )

            # Day count fraction
            dcf = calculate_day_count_fraction(
                accrual_start, accrual_end, self.day_count_convention
            )

            # Check for historical fixing
            fixing = self.fixing_store.get_fixing(self.index.name, fixing_date)

            caplet = Caplet(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                payment_date=payment_date,
                fixing_date=fixing_date,
                notional=period_notional,
                strike=self.strike,
                day_count_fraction=dcf,
                is_projected=fixing is None,
                index_fixing=fixing,
            )

            caplets.append(caplet)
            accrual_start = accrual_end

        return caplets

    def get_caplets(self) -> List[Caplet]:
        """
        Get all caplets/floorlets.

        Returns:
            List of Caplet objects (usable for both caps and floors)
        """
        if self._cached_caplets is None:
            self._cached_caplets = self._generate_caplets()
        return list(self._cached_caplets)

    def get_floorlets(self) -> List[Caplet]:
        """
        Alias for get_caplets() — floorlets use the same structure.

        Returns:
            List of Caplet objects
        """
        return self.get_caplets()

    def get_future_caplets(self, valuation_date: datetime) -> List[Caplet]:
        """
        Get caplets whose fixing date is in the future.

        Args:
            valuation_date: Reference date

        Returns:
            List of future Caplet objects
        """
        if self._cached_caplets is None:
            self._cached_caplets = self._generate_caplets()
        return [c for c in self._cached_caplets if c.payment_date > valuation_date]

    def get_notional(self, as_of_date: datetime) -> float:
        """
        Get the notional amount for a given date.

        Args:
            as_of_date: Date to look up notional

        Returns:
            Notional amount
        """
        if self.notional_schedule is not None:
            return self.notional_schedule.get_notional(as_of_date)
        return self.notional

    def get_maturity(self) -> datetime:
        """Get the maturity date."""
        return self.end_date

    def time_to_maturity(self, valuation_date: datetime) -> float:
        """
        Calculate time to maturity in years.

        Args:
            valuation_date: Date from which to measure

        Returns:
            Time to maturity in years
        """
        return (self.end_date - valuation_date).days / 365.0

    def is_expired(self, valuation_date: datetime) -> bool:
        """
        Check if the cap/floor has fully expired.

        Args:
            valuation_date: Date to check

        Returns:
            True if all periods have expired
        """
        return valuation_date >= self.end_date

    def num_periods(self) -> int:
        """
        Get the number of caplet/floorlet periods.

        Returns:
            Number of periods
        """
        if self._cached_caplets is None:
            self._cached_caplets = self._generate_caplets()
        return len(self._cached_caplets)

    def __repr__(self):
        type_str = self.cap_floor_type.value.upper()
        return (
            f"{type_str}(notional={self.notional:,.0f}, "
            f"strike={self.strike:.4%}, "
            f"start={self.start_date.date()}, "
            f"end={self.end_date.date()}, "
            f"index={self.index.name}, "
            f"periods={self.num_periods()})"
        )


# Type aliases for convenience
Cap = CapFloor
Floor = CapFloor


@dataclass
class Collar:
    """
    Interest Rate Collar.

    A collar combines a long cap with a short floor (for a borrower hedging),
    or a long floor with a short cap (for a lender hedging). The cap strike
    is above the floor strike.

    Attributes:
        cap: The cap leg
        floor: The floor leg
    """

    cap: CapFloor
    floor: CapFloor

    def __post_init__(self):
        """Validate collar parameters."""
        self.validate()

    def validate(self) -> None:
        """Validate collar parameters."""
        if self.cap.cap_floor_type != CapFloorType.CAP:
            raise ValidationError("Collar cap leg must have type CAP")
        if self.floor.cap_floor_type != CapFloorType.FLOOR:
            raise ValidationError("Collar floor leg must have type FLOOR")
        if self.cap.strike <= self.floor.strike:
            raise ValidationError(
                f"Cap strike {self.cap.strike} must be above "
                f"floor strike {self.floor.strike}"
            )
        # Dates should match
        if self.cap.start_date != self.floor.start_date:
            raise ValidationError(
                "Cap and floor start dates must match"
            )
        if self.cap.end_date != self.floor.end_date:
            raise ValidationError(
                "Cap and floor end dates must match"
            )

    def get_maturity(self) -> datetime:
        """Get the maturity date."""
        return self.cap.end_date

    def time_to_maturity(self, valuation_date: datetime) -> float:
        """Calculate time to maturity in years."""
        return self.cap.time_to_maturity(valuation_date)

    def is_expired(self, valuation_date: datetime) -> bool:
        """Check if the collar has expired."""
        return self.cap.is_expired(valuation_date)

    def __repr__(self):
        return (
            f"Collar(cap_strike={self.cap.strike:.4%}, "
            f"floor_strike={self.floor.strike:.4%}, "
            f"notional={self.cap.notional:,.0f}, "
            f"start={self.cap.start_date.date()}, "
            f"end={self.cap.end_date.date()})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_cap(
    start_date: datetime,
    end_date: datetime,
    notional: float,
    strike: float,
    index: RateIndex,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Optional[Calendar] = None,
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.MODIFIED_FOLLOWING
    ),
    notional_schedule: Optional[NotionalSchedule] = None,
) -> CapFloor:
    """
    Create an interest rate Cap.

    Args:
        start_date: Effective date
        end_date: Maturity date
        notional: Notional principal
        strike: Cap strike rate
        index: Reference rate index
        payment_frequency: Caplet frequency
        day_count_convention: Day count convention
        calendar: Business day calendar
        business_day_convention: Date adjustment convention
        notional_schedule: Optional amortizing schedule

    Returns:
        CapFloor object with type CAP
    """
    return CapFloor(
        notional=notional,
        strike=strike,
        cap_floor_type=CapFloorType.CAP,
        start_date=start_date,
        end_date=end_date,
        index=index,
        payment_frequency=payment_frequency,
        day_count_convention=day_count_convention,
        calendar=calendar,
        business_day_convention=business_day_convention,
        notional_schedule=notional_schedule,
    )


def create_floor(
    start_date: datetime,
    end_date: datetime,
    notional: float,
    strike: float,
    index: RateIndex,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Optional[Calendar] = None,
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.MODIFIED_FOLLOWING
    ),
    notional_schedule: Optional[NotionalSchedule] = None,
) -> CapFloor:
    """
    Create an interest rate Floor.

    Args:
        start_date: Effective date
        end_date: Maturity date
        notional: Notional principal
        strike: Floor strike rate
        index: Reference rate index
        payment_frequency: Floorlet frequency
        day_count_convention: Day count convention
        calendar: Business day calendar
        business_day_convention: Date adjustment convention
        notional_schedule: Optional amortizing schedule

    Returns:
        CapFloor object with type FLOOR
    """
    return CapFloor(
        notional=notional,
        strike=strike,
        cap_floor_type=CapFloorType.FLOOR,
        start_date=start_date,
        end_date=end_date,
        index=index,
        payment_frequency=payment_frequency,
        day_count_convention=day_count_convention,
        calendar=calendar,
        business_day_convention=business_day_convention,
        notional_schedule=notional_schedule,
    )


def create_collar(
    start_date: datetime,
    end_date: datetime,
    notional: float,
    cap_strike: float,
    floor_strike: float,
    index: RateIndex,
    payment_frequency: PaymentFrequency = PaymentFrequency.QUARTERLY,
    day_count_convention: DayCountConvention = DayCountConvention.ACT_360,
    calendar: Optional[Calendar] = None,
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.MODIFIED_FOLLOWING
    ),
    notional_schedule: Optional[NotionalSchedule] = None,
) -> Collar:
    """
    Create an interest rate Collar (long cap + short floor).

    Args:
        start_date: Effective date
        end_date: Maturity date
        notional: Notional principal
        cap_strike: Cap strike rate (upper bound)
        floor_strike: Floor strike rate (lower bound)
        index: Reference rate index
        payment_frequency: Period frequency
        day_count_convention: Day count convention
        calendar: Business day calendar
        business_day_convention: Date adjustment convention
        notional_schedule: Optional amortizing schedule

    Returns:
        Collar object
    """
    cap = create_cap(
        start_date=start_date,
        end_date=end_date,
        notional=notional,
        strike=cap_strike,
        index=index,
        payment_frequency=payment_frequency,
        day_count_convention=day_count_convention,
        calendar=calendar,
        business_day_convention=business_day_convention,
        notional_schedule=notional_schedule,
    )

    floor = create_floor(
        start_date=start_date,
        end_date=end_date,
        notional=notional,
        strike=floor_strike,
        index=index,
        payment_frequency=payment_frequency,
        day_count_convention=day_count_convention,
        calendar=calendar,
        business_day_convention=business_day_convention,
        notional_schedule=notional_schedule,
    )

    return Collar(cap=cap, floor=floor)
