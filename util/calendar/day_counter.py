"""
Day count conventions and year fraction calculations.
"""
from enum import Enum
from datetime import datetime
from util.exceptions import ValidationError


class DayCountConvention(Enum):
    """
    Day count conventions for calculating year fractions.
    
    CALENDAR_DAYS: Uses actual calendar days (365 or 366 for leap years)
    BUSINESS_DAYS: Uses business days (typically 252 per year)
    """
    CALENDAR_DAYS = "calendar_days"
    BUSINESS_DAYS = "business_days"


def calculate_year_fraction(
    start_date: datetime,
    end_date: datetime,
    convention: DayCountConvention,
    bus_days_in_year: int = 252
) -> float:
    """
    Calculate year fraction between two dates using specified convention.
    
    Args:
        start_date: Start date (valuation date)
        end_date: End date (exercise or settlement date)
        convention: Day count convention to use
        bus_days_in_year: Number of business days per year (default: 252)
        
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
    
    else:
        raise ValidationError(f"Unsupported convention: {convention}")


def is_leap_year(year: int) -> bool:
    """
    Check if a year is a leap year.
    
    Args:
        year: Year to check
        
    Returns:
        True if leap year, False otherwise
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


