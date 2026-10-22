"""Tests for Calendar.trading_days_after (DCN/desk settlement convention)."""
from datetime import datetime

import pytest

from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.exceptions import ValidationError


def _sse():
    return create_calendar(CalendarType.CHINA_SSE)


def test_dcn_offset_fingerprint():
    # Problem-verified: 2025-01-03 + 2 trading days -> 2025-01-07
    assert _sse().trading_days_after(datetime(2025, 1, 3), 2) == datetime(2025, 1, 7)


def test_zero_offset_is_identity():
    assert _sse().trading_days_after(datetime(2025, 1, 3), 0) == datetime(2025, 1, 3)


def test_matches_add_business_days():
    cal = _sse()
    d = datetime(2023, 4, 28)
    for n in range(0, 6):
        assert cal.trading_days_after(d, n) == (
            d if n == 0 else cal.add_business_days(d, n)
        )


def test_returns_datetime_type():
    out = _sse().trading_days_after(datetime(2025, 1, 3), 1)
    assert isinstance(out, datetime)


def test_rejects_non_trading_base_date():
    with pytest.raises(ValidationError):
        _sse().trading_days_after(datetime(2025, 1, 4), 1)  # Saturday


def test_rejects_negative_n():
    with pytest.raises(ValidationError):
        _sse().trading_days_after(datetime(2025, 1, 3), -1)
