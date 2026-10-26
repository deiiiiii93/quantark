"""``add_months``: the calendar's "+nM" (user ruling 2026-10-26: REBASED gaps are calendar months, through the
calendar utility). Same day of month, clamped to the month end; an optional business-day adjustment."""
from datetime import datetime

import pytest

from quantark.util.calendar import BusinessDayConvention, CalendarType, add_months, create_calendar
from quantark.util.exceptions import ValidationError


@pytest.mark.parametrize("start, months, expected", [
    (datetime(2026, 1, 15), 1, datetime(2026, 2, 15)),
    (datetime(2026, 1, 15), 0, datetime(2026, 1, 15)),
    (datetime(2026, 1, 31), 1, datetime(2026, 2, 28)),          # clamped to the month end
    (datetime(2028, 1, 31), 1, datetime(2028, 2, 29)),          # leap year
    (datetime(2026, 1, 31), 3, datetime(2026, 4, 30)),
    (datetime(2026, 11, 30), 3, datetime(2027, 2, 28)),         # across the year end
    (datetime(2026, 3, 31), -1, datetime(2026, 2, 28)),
    (datetime(2026, 1, 15, 15, 0), 12, datetime(2027, 1, 15, 15, 0)),   # the time of day is kept
])
def test_add_months_keeps_the_day_and_clamps_at_the_month_end(start, months, expected):
    assert add_months(start, months) == expected


def test_add_months_adjusts_on_a_calendar_when_asked():
    calendar = create_calendar(CalendarType.NONE)
    tuesday = add_months(datetime(2026, 1, 3), 2)                    # 2026-03-03 is a Tuesday: no roll
    assert add_months(datetime(2026, 1, 3), 2, calendar, BusinessDayConvention.FOLLOWING) == tuesday
    # 2026-02-28 is a Saturday
    assert add_months(datetime(2026, 1, 28), 1, calendar, BusinessDayConvention.FOLLOWING) == datetime(2026, 3, 2)
    assert add_months(datetime(2026, 1, 28), 1, calendar, BusinessDayConvention.MODIFIED_FOLLOWING) \
        == datetime(2026, 2, 27)


def test_add_months_refuses_a_fractional_count_and_a_convention_without_a_calendar():
    with pytest.raises(ValidationError, match="whole number"):
        add_months(datetime(2026, 1, 15), 1.5)
    with pytest.raises(ValidationError, match="calendar"):
        add_months(datetime(2026, 1, 15), 1, None, BusinessDayConvention.FOLLOWING)
