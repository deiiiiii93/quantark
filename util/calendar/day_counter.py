"""
Day count conventions and year fraction calculations.
"""

from enum import Enum
from datetime import datetime
from util.exceptions import ValidationError


class DayCountConvention(Enum):
    """
    Day count conventions for calculating year fractions.

    Standard bond market conventions:
    - ACT_360: Actual days / 360 (money market)
    - ACT_365: Actual days / 365 (simple)
    - ACT_ACT_ISDA: Actual/Actual ISDA (treasury bonds)
    - ACT_ACT_ISMA: Actual/Actual ISMA/ICMA (international bonds)
    - ACT_ACT_BOND: Actual/Actual Bond (same as ISMA)
    - ACT_ACT_CHINABOND: Actual/Actual China Bond (Chinese interbank market)
    - THIRTY_360_US: 30/360 US Bond Basis
    - THIRTY_360_EUROPEAN: 30/360 European
    - ACT_365L: Actual/365 with leap year adjustment

    Legacy conventions:
    - CALENDAR_DAYS: Uses actual calendar days (365 or 366 for leap years)
    - BUSINESS_DAYS: Uses business days (typically 252 per year)
    """

    CALENDAR_DAYS = "calendar_days"
    BUSINESS_DAYS = "business_days"
    ACT_360 = "act_360"
    ACT_365 = "act_365"
    ACT_ACT_ISDA = "act_act_isda"
    ACT_ACT_ISMA = "act_act_isma"  # Also known as ICMA
    ACT_ACT_ICMA = "act_act_isma"  # Alias for ISMA
    ACT_ACT_BOND = "act_act_bond"
    ACT_ACT_CHINABOND = "act_act_chinabond"
    THIRTY_360_US = "30_360_us"
    THIRTY_360_EUROPEAN = "30_360_european"
    ACT_365L = "act_365l"


def calculate_year_fraction(
    start_date: datetime,
    end_date: datetime,
    convention: DayCountConvention,
    bus_days_in_year: int = 252,
    period_start: datetime = None,
    period_end: datetime = None,
    frequency: int = None,
) -> float:
    """
    Calculate year fraction between two dates using specified convention.

    Args:
        start_date: Start date (valuation date)
        end_date: End date (exercise or settlement date)
        convention: Day count convention to use
        bus_days_in_year: Number of business days per year (default: 252)
        period_start: Optional coupon period start (for ISMA/ICMA/BOND/CHINABOND)
        period_end: Optional coupon period end (for ISMA/ICMA/BOND/CHINABOND)
        frequency: Optional coupon frequency per year (for ISMA/ICMA/BOND/CHINABOND)

    Returns:
        Year fraction as a float

    Raises:
        ValidationError: If dates are invalid or convention is unsupported

    Examples:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2025, 1, 1)
        >>> calculate_year_fraction(start, end, DayCountConvention.CALENDAR_DAYS)
        1.0027397260273974  # 366 days / 365 (2024 is leap year)
    """
    # Validate inputs
    if not isinstance(start_date, datetime):
        raise ValidationError(f"start_date must be datetime, got {type(start_date)}")
    if not isinstance(end_date, datetime):
        raise ValidationError(f"end_date must be datetime, got {type(end_date)}")
    if end_date <= start_date:
        raise ValidationError(
            f"end_date ({end_date}) must be after start_date ({start_date})"
        )
    if not isinstance(convention, DayCountConvention):
        raise ValidationError(f"Invalid convention: {convention}")

    # Calculate days between dates
    delta = end_date - start_date
    num_days = delta.days

    if convention == DayCountConvention.CALENDAR_DAYS:
        # Use actual/365 convention (or actual/366 for leap years)
        # More accurate: use average over the period
        # Simple approach: use actual days / 365.0
        return num_days / 365.0

    elif convention == DayCountConvention.BUSINESS_DAYS:
        # Business days convention
        if bus_days_in_year <= 0:
            raise ValidationError(
                f"bus_days_in_year must be positive, got {bus_days_in_year}"
            )
        # Approximate: assume 5/7 of calendar days are business days
        # More accurate would require a business day calendar
        business_days = num_days * (bus_days_in_year / 365.0)
        return business_days / bus_days_in_year

    # For all other conventions, delegate to calculate_day_count_fraction
    else:
        return calculate_day_count_fraction(
            start_date, end_date, convention, period_start, period_end, frequency
        )


def is_leap_year(year: int) -> bool:
    """
    Check if a year is a leap year.

    Args:
        year: Year to check

    Returns:
        True if leap year, False otherwise
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_year(year: int) -> int:
    """
    Get number of days in a year.

    Args:
        year: Year to check

    Returns:
        366 if leap year, 365 otherwise
    """
    return 366 if is_leap_year(year) else 365


def calculate_day_count_fraction(
    start_date: datetime,
    end_date: datetime,
    convention: DayCountConvention,
    period_start: datetime = None,
    period_end: datetime = None,
    frequency: int = None,
) -> float:
    """
    Calculate day count fraction between two dates using specified convention.

    This is the comprehensive bond market implementation supporting all major
    day count conventions used in fixed income markets.

    Args:
        start_date: Accrual start date
        end_date: Accrual end date
        convention: Day count convention to use
        period_start: Optional coupon period start (required for ISMA/ICMA/BOND)
        period_end: Optional coupon period end (required for ISMA/ICMA/BOND)
        frequency: Optional coupon frequency per year (required for ISMA/ICMA/BOND)

    Returns:
        Day count fraction as a float

    Raises:
        ValidationError: If dates are invalid or convention is unsupported

    Examples:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 7, 1)
        >>> calculate_day_count_fraction(start, end, DayCountConvention.ACT_360)
        0.5055555555555556  # 182 days / 360

        # For ACT/ACT ISMA with coupon period info:
        >>> period_start = datetime(2024, 1, 1)
        >>> period_end = datetime(2024, 7, 1)
        >>> calculate_day_count_fraction(start, end, DayCountConvention.ACT_ACT_ISMA,
        ...                              period_start, period_end, frequency=2)
        0.5  # Fraction of the coupon period
    """
    # Validate inputs
    if not isinstance(start_date, datetime):
        raise ValidationError(f"start_date must be datetime, got {type(start_date)}")
    if not isinstance(end_date, datetime):
        raise ValidationError(f"end_date must be datetime, got {type(end_date)}")
    if end_date <= start_date:
        raise ValidationError(
            f"end_date ({end_date}) must be after start_date ({start_date})"
        )
    if not isinstance(convention, DayCountConvention):
        raise ValidationError(f"Invalid convention: {convention}")

    # Calculate based on convention
    if convention == DayCountConvention.ACT_360:
        return _act_360(start_date, end_date)
    elif convention == DayCountConvention.ACT_365:
        return _act_365(start_date, end_date)
    elif convention == DayCountConvention.ACT_ACT_ISDA:
        return _act_act_isda(start_date, end_date)
    elif convention in (
        DayCountConvention.ACT_ACT_ISMA,
        DayCountConvention.ACT_ACT_ICMA,
        DayCountConvention.ACT_ACT_BOND,
    ):
        return _act_act_isma(start_date, end_date, period_start, period_end, frequency)
    elif convention == DayCountConvention.ACT_ACT_CHINABOND:
        return _act_act_chinabond(
            start_date, end_date, period_start, period_end, frequency
        )
    elif convention == DayCountConvention.THIRTY_360_US:
        return _thirty_360_us(start_date, end_date)
    elif convention == DayCountConvention.THIRTY_360_EUROPEAN:
        return _thirty_360_european(start_date, end_date)
    elif convention == DayCountConvention.ACT_365L:
        return _act_365l(start_date, end_date)
    elif convention == DayCountConvention.CALENDAR_DAYS:
        # Legacy support
        return (end_date - start_date).days / 365.0
    elif convention == DayCountConvention.BUSINESS_DAYS:
        # Legacy support - approximate
        num_days = (end_date - start_date).days
        business_days = num_days * (252.0 / 365.0)
        return business_days / 252.0
    else:
        raise ValidationError(f"Unsupported convention: {convention}")


def _act_360(start_date: datetime, end_date: datetime) -> float:
    """
    ACT/360 day count fraction.
    Used in money markets and some corporate bonds.

    Formula: (Actual days between dates) / 360
    """
    actual_days = (end_date - start_date).days
    return actual_days / 360.0


def _act_365(start_date: datetime, end_date: datetime) -> float:
    """
    ACT/365 day count fraction.
    Simple actual days over 365.

    Formula: (Actual days between dates) / 365
    """
    actual_days = (end_date - start_date).days
    return actual_days / 365.0


def _act_act_isda(start_date: datetime, end_date: datetime) -> float:
    """
    ACT/ACT ISDA day count fraction.
    Used for US Treasury bonds and many government bonds.

    Formula: Sum over each year of (days in year / total days in that year)
    """
    if start_date.year == end_date.year:
        # Same year - simple case
        actual_days = (end_date - start_date).days
        days_in_year_val = days_in_year(start_date.year)
        return actual_days / days_in_year_val

    # Multiple years - split calculation
    total_fraction = 0.0

    # First partial year
    year_end = datetime(start_date.year, 12, 31, 23, 59, 59)
    if year_end < end_date:
        days_first_year = (year_end - start_date).days + 1
        total_fraction += days_first_year / days_in_year(start_date.year)
        current_year = start_date.year + 1
    else:
        # End date is in first year
        actual_days = (end_date - start_date).days
        return actual_days / days_in_year(start_date.year)

    # Full years in between
    while current_year < end_date.year:
        total_fraction += 1.0  # Full year
        current_year += 1

    # Last partial year
    if current_year == end_date.year:
        year_start = datetime(end_date.year, 1, 1)
        days_last_year = (end_date - year_start).days
        total_fraction += days_last_year / days_in_year(end_date.year)

    return total_fraction


def _thirty_360_us(start_date: datetime, end_date: datetime) -> float:
    """
    30/360 US (Bond Basis) day count fraction.
    Used for US corporate and municipal bonds.

    Formula: (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360
    With adjustments for month-end dates.
    """
    d1 = start_date.day
    m1 = start_date.month
    y1 = start_date.year

    d2 = end_date.day
    m2 = end_date.month
    y2 = end_date.year

    # Adjust d1 if it's the last day of February or day 31
    if _is_last_day_of_february(start_date):
        d1 = 30
    elif d1 == 31:
        d1 = 30

    # Adjust d2 if d1 is 30 or 31 and d2 is 31
    if d2 == 31 and d1 >= 30:
        d2 = 30

    days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return days / 360.0


def _thirty_360_european(start_date: datetime, end_date: datetime) -> float:
    """
    30/360 European day count fraction.
    Used for Eurobonds.

    Formula: (360*(Y2-Y1) + 30*(M2-M1) + (D2-D1)) / 360
    Day 31 is always adjusted to 30.
    """
    d1 = min(start_date.day, 30)
    m1 = start_date.month
    y1 = start_date.year

    d2 = min(end_date.day, 30)
    m2 = end_date.month
    y2 = end_date.year

    days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return days / 360.0


def _act_365l(start_date: datetime, end_date: datetime) -> float:
    """
    ACT/365L day count fraction.
    Actual days divided by 366 if Feb 29 is in the period, 365 otherwise.

    Formula: (Actual days) / (366 if leap year in period, else 365)
    """
    actual_days = (end_date - start_date).days

    # Check if Feb 29 falls within the period
    contains_leap_day = False
    for year in range(start_date.year, end_date.year + 1):
        if is_leap_year(year):
            feb_29 = datetime(year, 2, 29)
            if start_date <= feb_29 < end_date:
                contains_leap_day = True
                break

    denominator = 366 if contains_leap_day else 365
    return actual_days / denominator


def _act_act_isma(
    start_date: datetime,
    end_date: datetime,
    period_start: datetime = None,
    period_end: datetime = None,
    frequency: int = None,
) -> float:
    """
    ACT/ACT ISMA (ICMA) day count fraction.
    Used for international bonds and European government bonds.
    Also known as ACT/ACT BOND.

    Formula: (Actual days in accrual period) / (Frequency × Actual days in coupon period)

    When period information is not provided, falls back to a simplified calculation
    using assumed semi-annual frequency and estimated period length.

    Args:
        start_date: Accrual start date
        end_date: Accrual end date
        period_start: Coupon period start date
        period_end: Coupon period end date
        frequency: Number of coupon payments per year (e.g., 1, 2, 4, 12)

    Returns:
        Day count fraction
    """
    actual_days = (end_date - start_date).days

    # If full period information is provided, use proper ISMA calculation
    if period_start is not None and period_end is not None and frequency is not None:
        period_days = (period_end - period_start).days
        if period_days <= 0:
            raise ValidationError("period_end must be after period_start")
        if frequency <= 0:
            raise ValidationError("frequency must be positive")
        return actual_days / (frequency * period_days)

    # Fallback: estimate based on common assumptions
    # Use semi-annual frequency (2) if not specified
    freq = frequency if frequency is not None else 2

    # If period info not available, use the accrual period itself as the coupon period
    # This is an approximation for mid-period calculations
    if period_start is None or period_end is None:
        # Assume the period is ~6 months for semi-annual, ~3 months for quarterly, etc.
        estimated_period_days = 365.0 / freq
        return actual_days / (freq * estimated_period_days)

    period_days = (period_end - period_start).days
    return actual_days / (freq * period_days)


def _act_act_chinabond(
    start_date: datetime,
    end_date: datetime,
    period_start: datetime = None,
    period_end: datetime = None,
    frequency: int = None,
) -> float:
    """
    ACT/ACT China Bond day count fraction.
    Used for Chinese interbank bond market.

    The China Bond convention follows similar principles to ACT/ACT ISMA but
    with specific rules for the Chinese market:

    1. For bonds with regular coupon periods:
       Fraction = Actual days / (Frequency × Days in coupon period)

    2. For irregular periods or when period info is not available:
       Uses ACT/365 as fallback (common in Chinese market)

    3. Chinese bonds typically use:
       - Annual payment (frequency=1) for most government and corporate bonds
       - Semi-annual (frequency=2) for some policy bank bonds

    Args:
        start_date: Accrual start date
        end_date: Accrual end date
        period_start: Coupon period start date
        period_end: Coupon period end date
        frequency: Number of coupon payments per year

    Returns:
        Day count fraction
    """
    actual_days = (end_date - start_date).days

    # If full period information is provided, use proper calculation
    if period_start is not None and period_end is not None and frequency is not None:
        period_days = (period_end - period_start).days
        if period_days <= 0:
            raise ValidationError("period_end must be after period_start")
        if frequency <= 0:
            raise ValidationError("frequency must be positive")
        return actual_days / (frequency * period_days)

    # Chinese bond market conventions:
    # When period info is not available, use ACT/365 as the standard fallback
    # This is common practice for Chinese interbank bonds

    # If frequency is specified but not period info, estimate period
    if frequency is not None and frequency > 0:
        # Estimate using standard year length
        estimated_period_days = 365.0 / frequency
        return actual_days / (frequency * estimated_period_days)

    # Default fallback to ACT/365 (standard for Chinese bonds without period info)
    return actual_days / 365.0


def _is_last_day_of_february(date: datetime) -> bool:
    """
    Check if a date is the last day of February.

    Args:
        date: Date to check

    Returns:
        True if last day of February, False otherwise
    """
    if date.month != 2:
        return False

    # Check if it's Feb 29 in a leap year or Feb 28 in a non-leap year
    if is_leap_year(date.year):
        return date.day == 29
    else:
        return date.day == 28
