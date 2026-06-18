"""Tests for trading-day helper methods added to Calendar for TRS support."""

from datetime import datetime

import pytest

from quantark.util.calendar.business_calendar import Calendar


@pytest.fixture
def cal():
    # Weekends Sat/Sun by default; add one holiday: Mon 2024-01-08.
    return Calendar(holidays={datetime(2024, 1, 8)}, name="Test")


# ---------------------------------------------------------------------------
# get_calendar_days
# ---------------------------------------------------------------------------
def test_calendar_days_both_inclusive(cal):
    days = cal.get_calendar_days("2024-01-01", "2024-01-05", side="both")
    assert [d.strftime("%Y-%m-%d") for d in days] == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_calendar_days_side_trimming(cal):
    assert [d.strftime("%Y-%m-%d") for d in cal.get_calendar_days(
        "2024-01-01", "2024-01-04", side="left")] == [
        "2024-01-01", "2024-01-02", "2024-01-03"]
    assert [d.strftime("%Y-%m-%d") for d in cal.get_calendar_days(
        "2024-01-01", "2024-01-04", side="right")] == [
        "2024-01-02", "2024-01-03", "2024-01-04"]
    assert [d.strftime("%Y-%m-%d") for d in cal.get_calendar_days(
        "2024-01-01", "2024-01-04", side="neither")] == [
        "2024-01-02", "2024-01-03"]


# ---------------------------------------------------------------------------
# get_working_days
# ---------------------------------------------------------------------------
def test_working_days_skip_weekend_and_holiday(cal):
    # Jan 5 Fri, 6 Sat, 7 Sun, 8 Mon(holiday), 9 Tue, 10 Wed
    days = cal.get_working_days("2024-01-05", "2024-01-10", side="both")
    assert [d.strftime("%Y-%m-%d") for d in days] == [
        "2024-01-05",
        "2024-01-09",
        "2024-01-10",
    ]


def test_working_days_accepts_datetime_inputs(cal):
    days = cal.get_working_days(datetime(2024, 1, 5), datetime(2024, 1, 9))
    assert [d.strftime("%Y-%m-%d") for d in days] == ["2024-01-05", "2024-01-09"]


def test_working_days_left_keeps_interior_trading_day_when_endpoint_nonbusiness(cal):
    # Fri 2024-01-05 -> Sat 2024-01-06; side="left" drops the Saturday calendar
    # endpoint (a non-business day), so the Friday trading day is retained.
    days = cal.get_working_days("2024-01-05", "2024-01-06", side="left")
    assert [d.strftime("%Y-%m-%d") for d in days] == ["2024-01-05"]


# ---------------------------------------------------------------------------
# get_next_trading_date
# ---------------------------------------------------------------------------
def test_next_trading_date_rolls_weekend_and_holiday(cal):
    # Saturday -> next business day is Tue Jan 9 (Mon Jan 8 is a holiday).
    assert cal.get_next_trading_date("2024-01-06", n=1, only_holidays=True) == "2024-01-09"


def test_next_trading_date_advances_full_step_from_business_day(cal):
    # From a business day with only_holidays=False, n=1 advances one trading day.
    assert cal.get_next_trading_date("2024-01-09", n=1, only_holidays=False) == "2024-01-10"


def test_next_trading_date_n_zero_returns_same(cal):
    assert cal.get_next_trading_date("2024-01-09", n=0) == "2024-01-09"


# ---------------------------------------------------------------------------
# get_num_of_calendar_days
# ---------------------------------------------------------------------------
def test_num_calendar_days_sides(cal):
    assert cal.get_num_of_calendar_days("2024-01-01", "2024-01-05", side="left") == 4
    assert cal.get_num_of_calendar_days("2024-01-01", "2024-01-05", side="both") == 5
    assert cal.get_num_of_calendar_days("2024-01-01", "2024-01-05", side="neither") == 3


def test_num_calendar_days_empty_interval_not_negative(cal):
    # Degenerate (start == end) open interval must not return a negative count.
    assert cal.get_num_of_calendar_days("2024-01-01", "2024-01-01", side="neither") == 0
    assert cal.get_num_of_calendar_days("2024-01-01", "2024-01-01", side="both") == 1


def test_num_calendar_days_reversed_interval_is_zero(cal):
    # A reversed range spans no days for any side (matches get_calendar_days).
    for side in ("left", "right", "both", "neither"):
        assert cal.get_num_of_calendar_days(
            "2024-01-05", "2024-01-01", side=side) == 0
