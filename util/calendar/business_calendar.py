"""
Business day calendar for settlement date adjustments.
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Set, Optional
from util.exceptions import ValidationError


class BusinessDayConvention(Enum):
    """
    Business day conventions for date adjustments.

    - FOLLOWING: Roll forward to next business day
    - MODIFIED_FOLLOWING: Roll forward unless crosses month, then roll backward
    - PRECEDING: Roll backward to previous business day
    - MODIFIED_PRECEDING: Roll backward unless crosses month, then roll forward
    - UNADJUSTED: No adjustment
    """

    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"
    UNADJUSTED = "unadjusted"


class CalendarType(Enum):
    """
    Standard calendar types with predefined holiday schedules.
    """

    US = "us"  # US holidays (Federal Reserve)
    UK = "uk"  # UK holidays
    TARGET = "target"  # Trans-European Automated Real-time Gross settlement (ECB)
    CHINA = "china"  # China holidays (PBOC/SSE)
    NONE = "none"  # No holidays, only weekends


class Calendar:
    """
    Business day calendar supporting holidays and business day adjustments.

    A calendar defines which days are business days (non-holidays, non-weekends)
    and provides methods to adjust dates according to business day conventions.

    Attributes:
        holidays: Set of holiday dates
        weekend_days: Set of weekend day numbers (0=Monday, 6=Sunday)
        name: Calendar name for identification
    """

    def __init__(
        self,
        holidays: Optional[Set[datetime]] = None,
        weekend_days: Optional[Set[int]] = None,
        name: str = "Custom",
    ):
        """
        Initialize calendar.

        Args:
            holidays: Set of holiday dates (default: empty set)
            weekend_days: Set of weekend day numbers, 0=Mon, 6=Sun (default: {5, 6})
            name: Calendar name
        """
        self.holidays = holidays if holidays is not None else set()
        self.weekend_days = (
            weekend_days if weekend_days is not None else {5, 6}
        )  # Sat, Sun
        self.name = name

        # Normalize holiday dates to midnight
        self.holidays = {self._normalize_date(d) for d in self.holidays}

    @staticmethod
    def _normalize_date(date: datetime) -> datetime:
        """Normalize date to midnight."""
        return datetime(date.year, date.month, date.day)

    def is_business_day(self, date: datetime) -> bool:
        """
        Check if a date is a business day.

        Args:
            date: Date to check

        Returns:
            True if business day, False if weekend or holiday
        """
        normalized = self._normalize_date(date)

        # Check if weekend
        if normalized.weekday() in self.weekend_days:
            return False

        # Check if holiday
        if normalized in self.holidays:
            return False

        return True

    def is_holiday(self, date: datetime) -> bool:
        """
        Check if a date is a holiday.

        Args:
            date: Date to check

        Returns:
            True if holiday, False otherwise
        """
        normalized = self._normalize_date(date)
        return normalized in self.holidays

    def adjust_date(
        self, date: datetime, convention: BusinessDayConvention
    ) -> datetime:
        """
        Adjust a date according to business day convention.

        Args:
            date: Date to adjust
            convention: Business day convention to apply

        Returns:
            Adjusted date

        Raises:
            ValidationError: If convention is invalid
        """
        if convention == BusinessDayConvention.UNADJUSTED:
            return date

        if convention == BusinessDayConvention.FOLLOWING:
            return self._adjust_following(date)

        if convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            adjusted = self._adjust_following(date)
            # If adjustment crosses month boundary, use preceding instead
            if adjusted.month != date.month:
                return self._adjust_preceding(date)
            return adjusted

        if convention == BusinessDayConvention.PRECEDING:
            return self._adjust_preceding(date)

        if convention == BusinessDayConvention.MODIFIED_PRECEDING:
            adjusted = self._adjust_preceding(date)
            # If adjustment crosses month boundary, use following instead
            if adjusted.month != date.month:
                return self._adjust_following(date)
            return adjusted

        raise ValidationError(f"Unsupported business day convention: {convention}")

    def _adjust_following(self, date: datetime) -> datetime:
        """Roll forward to next business day."""
        current = date
        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        while not self.is_business_day(current) and iterations < max_iterations:
            current = current + timedelta(days=1)
            iterations += 1

        if iterations >= max_iterations:
            raise ValidationError(
                f"Could not find business day within {max_iterations} days"
            )

        return current

    def _adjust_preceding(self, date: datetime) -> datetime:
        """Roll backward to previous business day."""
        current = date
        max_iterations = 10  # Prevent infinite loops
        iterations = 0

        while not self.is_business_day(current) and iterations < max_iterations:
            current = current - timedelta(days=1)
            iterations += 1

        if iterations >= max_iterations:
            raise ValidationError(
                f"Could not find business day within {max_iterations} days"
            )

        return current

    def add_business_days(self, date: datetime, num_days: int) -> datetime:
        """
        Add a number of business days to a date.

        Args:
            date: Starting date
            num_days: Number of business days to add (can be negative)

        Returns:
            Date after adding business days
        """
        if num_days == 0:
            return date

        direction = 1 if num_days > 0 else -1
        remaining = abs(num_days)
        current = date

        while remaining > 0:
            current = current + timedelta(days=direction)
            if self.is_business_day(current):
                remaining -= 1

        return current

    def __repr__(self):
        return f"Calendar(name={self.name}, holidays={len(self.holidays)})"


def create_calendar(
    calendar_type: CalendarType, year_range: tuple = (2020, 2030)
) -> Calendar:
    """
    Create a calendar with predefined holidays.

    Args:
        calendar_type: Type of calendar to create
        year_range: (start_year, end_year) for generating holidays

    Returns:
        Calendar with appropriate holidays
    """
    if calendar_type == CalendarType.NONE:
        return Calendar(holidays=set(), name="No Holidays")

    if calendar_type == CalendarType.US:
        return _create_us_calendar(year_range)

    if calendar_type == CalendarType.UK:
        return _create_uk_calendar(year_range)

    if calendar_type == CalendarType.TARGET:
        return _create_target_calendar(year_range)

    if calendar_type == CalendarType.CHINA:
        return _create_china_calendar(year_range)

    raise ValidationError(f"Unsupported calendar type: {calendar_type}")


def _create_us_calendar(year_range: tuple) -> Calendar:
    """
    Create US Federal Reserve holiday calendar.

    Includes: New Year's Day, MLK Day, Presidents Day, Memorial Day,
    Independence Day, Labor Day, Columbus Day, Veterans Day,
    Thanksgiving, Christmas
    """
    holidays = set()
    start_year, end_year = year_range

    for year in range(start_year, end_year + 1):
        # New Year's Day (Jan 1)
        holidays.add(datetime(year, 1, 1))

        # Martin Luther King Jr. Day (3rd Monday in January)
        holidays.add(_nth_weekday(year, 1, 0, 3))

        # Presidents Day (3rd Monday in February)
        holidays.add(_nth_weekday(year, 2, 0, 3))

        # Memorial Day (last Monday in May)
        holidays.add(_last_weekday(year, 5, 0))

        # Independence Day (July 4)
        holidays.add(datetime(year, 7, 4))

        # Labor Day (1st Monday in September)
        holidays.add(_nth_weekday(year, 9, 0, 1))

        # Columbus Day (2nd Monday in October)
        holidays.add(_nth_weekday(year, 10, 0, 2))

        # Veterans Day (Nov 11)
        holidays.add(datetime(year, 11, 11))

        # Thanksgiving (4th Thursday in November)
        holidays.add(_nth_weekday(year, 11, 3, 4))

        # Christmas (Dec 25)
        holidays.add(datetime(year, 12, 25))

    return Calendar(holidays=holidays, name="US Federal Reserve")


def _create_uk_calendar(year_range: tuple) -> Calendar:
    """
    Create UK holiday calendar.

    Includes: New Year's Day, Good Friday, Easter Monday, Early May,
    Spring Bank Holiday, Summer Bank Holiday, Christmas, Boxing Day
    """
    holidays = set()
    start_year, end_year = year_range

    for year in range(start_year, end_year + 1):
        # New Year's Day
        holidays.add(datetime(year, 1, 1))

        # Good Friday and Easter Monday (based on Easter calculation)
        easter = _calculate_easter(year)
        holidays.add(easter - timedelta(days=2))  # Good Friday
        holidays.add(easter + timedelta(days=1))  # Easter Monday

        # Early May Bank Holiday (1st Monday in May)
        holidays.add(_nth_weekday(year, 5, 0, 1))

        # Spring Bank Holiday (last Monday in May)
        holidays.add(_last_weekday(year, 5, 0))

        # Summer Bank Holiday (last Monday in August)
        holidays.add(_last_weekday(year, 8, 0))

        # Christmas Day
        holidays.add(datetime(year, 12, 25))

        # Boxing Day
        holidays.add(datetime(year, 12, 26))

    return Calendar(holidays=holidays, name="UK")


def _create_target_calendar(year_range: tuple) -> Calendar:
    """
    Create TARGET (Trans-European Automated Real-time Gross settlement) calendar.
    Used by European Central Bank.

    Includes: New Year's Day, Good Friday, Easter Monday, Labour Day,
    Christmas, Boxing Day
    """
    holidays = set()
    start_year, end_year = year_range

    for year in range(start_year, end_year + 1):
        # New Year's Day
        holidays.add(datetime(year, 1, 1))

        # Good Friday and Easter Monday
        easter = _calculate_easter(year)
        holidays.add(easter - timedelta(days=2))  # Good Friday
        holidays.add(easter + timedelta(days=1))  # Easter Monday

        # Labour Day (May 1)
        holidays.add(datetime(year, 5, 1))

        # Christmas Day
        holidays.add(datetime(year, 12, 25))

        # Boxing Day (Dec 26)
        holidays.add(datetime(year, 12, 26))

    return Calendar(holidays=holidays, name="TARGET")


def _create_china_calendar(year_range: tuple) -> Calendar:
    """
    Create China holiday calendar (PBOC/SSE).

    Note: China holidays are complex with lunar calendar festivals.
    This provides a simplified version with fixed dates.
    For production use, consider loading exact dates from a data source.

    Includes: New Year's Day, Spring Festival (approximate), Qingming,
    Labour Day, Dragon Boat Festival (approximate), Mid-Autumn Festival (approximate),
    National Day
    """
    holidays = set()
    start_year, end_year = year_range

    for year in range(start_year, end_year + 1):
        # New Year's Day (Jan 1)
        holidays.add(datetime(year, 1, 1))

        # Spring Festival / Chinese New Year (approximate - late Jan/early Feb)
        # Using approximate dates; actual dates vary by lunar calendar
        # Typically a week-long holiday
        for day in range(1, 8):
            holidays.add(datetime(year, 2, day))

        # Qingming Festival (April 4 or 5)
        holidays.add(datetime(year, 4, 4))
        holidays.add(datetime(year, 4, 5))

        # Labour Day (May 1-3)
        holidays.add(datetime(year, 5, 1))
        holidays.add(datetime(year, 5, 2))
        holidays.add(datetime(year, 5, 3))

        # Dragon Boat Festival (approximate - June)
        holidays.add(datetime(year, 6, 12))
        holidays.add(datetime(year, 6, 13))
        holidays.add(datetime(year, 6, 14))

        # Mid-Autumn Festival (approximate - September)
        holidays.add(datetime(year, 9, 15))
        holidays.add(datetime(year, 9, 16))
        holidays.add(datetime(year, 9, 17))

        # National Day (Oct 1-7)
        for day in range(1, 8):
            holidays.add(datetime(year, 10, day))

    return Calendar(holidays=holidays, name="China")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
    """
    Find the nth occurrence of a weekday in a month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)
        n: Which occurrence (1=first, 2=second, etc.)

    Returns:
        Date of nth weekday
    """
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()

    # Calculate days until first occurrence of target weekday
    days_until = (weekday - first_weekday) % 7
    first_occurrence = first_day + timedelta(days=days_until)

    # Add weeks to get to nth occurrence
    return first_occurrence + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> datetime:
    """
    Find the last occurrence of a weekday in a month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)

    Returns:
        Date of last weekday in month
    """
    # Start from last day of month
    if month == 12:
        last_day = datetime(year, 12, 31)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)

    last_weekday_val = last_day.weekday()

    # Calculate days back to target weekday
    days_back = (last_weekday_val - weekday) % 7
    return last_day - timedelta(days=days_back)


def _calculate_easter(year: int) -> datetime:
    """
    Calculate Easter Sunday using Meeus/Jones/Butcher algorithm.

    Args:
        year: Year

    Returns:
        Date of Easter Sunday
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1

    return datetime(year, month, day)
