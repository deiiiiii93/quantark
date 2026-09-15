from datetime import datetime, timedelta, timezone

import pytest

from quantark.intraday.timestamp import (
    SECONDS_PER_YEAR, add_year_fraction, calendar_year_fraction, require_aware, same_instant,
)
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI


def test_require_aware_rejects_naive_and_non_datetime():
    with pytest.raises(ValidationError, match="valuation_timestamp"):
        require_aware(datetime(2026, 9, 15, 14, 59, 59), "valuation_timestamp")
    with pytest.raises(ValidationError):
        require_aware("2026-09-15T14:59:59+08:00", "x")
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    assert require_aware(ts, "x") is ts


def test_calendar_year_fraction_is_seconds_exact_and_signed():
    a = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    b = datetime(2026, 9, 15, 15, 0, 0, tzinfo=SHANGHAI)
    assert calendar_year_fraction(a, b) == 1.0 / SECONDS_PER_YEAR
    assert calendar_year_fraction(b, a) == -1.0 / SECONDS_PER_YEAR
    assert calendar_year_fraction(a, a) == 0.0


def test_calendar_year_fraction_is_instant_based_across_zones():
    a = datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)
    b = datetime(2026, 9, 15, 7, 0, tzinfo=timezone.utc)   # same instant
    assert calendar_year_fraction(a, b) == 0.0 and same_instant(a, b)


def test_add_year_fraction_round_trips():
    a = datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)
    tau = 37.0 / SECONDS_PER_YEAR
    assert add_year_fraction(a, tau) == a + timedelta(seconds=37)
    assert calendar_year_fraction(a, add_year_fraction(a, 0.25)) == pytest.approx(0.25, abs=1e-15)


def test_arithmetic_is_instant_based_across_a_dst_change():
    try:
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
    except Exception:  # ZoneInfoNotFoundError when no tz database
        pytest.skip("no tz database")
    fri = datetime(2026, 3, 6, 16, 0, tzinfo=ny)      # EST
    mon = datetime(2026, 3, 9, 16, 0, tzinfo=ny)      # EDT: the weekend lost an hour
    assert calendar_year_fraction(fri, mon) == (71 * 3600.0) / SECONDS_PER_YEAR
    moved = add_year_fraction(fri, 72 * 3600.0 / SECONDS_PER_YEAR)
    assert moved == datetime(2026, 3, 9, 17, 0, tzinfo=ny) and moved.tzinfo is ny
