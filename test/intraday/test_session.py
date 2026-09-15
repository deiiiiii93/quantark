from datetime import date, datetime, time, timedelta

import pytest

from quantark.intraday.session import TradingSession, TradingSessionCalendar
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI


def _sse(cal, **kw):
    return TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=cal,
                                  sessions=(TradingSession(time(9, 30), time(11, 30)),
                                            TradingSession(time(13, 0), time(15, 0))), **kw)


def test_close_and_open_are_aware_and_in_local_zone(sse_calendar):
    s = _sse(sse_calendar)
    d = date(2026, 9, 15)   # Tuesday, trading day
    assert s.close_at(d) == datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)
    assert s.open_at(d) == datetime(2026, 9, 15, 9, 30, tzinfo=SHANGHAI)
    assert s.close_at(d).utcoffset() == timedelta(hours=8)


def test_non_trading_day_is_rejected_for_close_but_allowed_for_payment(sse_calendar):
    s = _sse(sse_calendar)
    sat = date(2026, 9, 19)
    with pytest.raises(ValidationError, match="not a trading day"):
        s.close_at(sat)
    assert s.payment_at(sat) == datetime(2026, 9, 19, 15, 0, tzinfo=SHANGHAI)


def test_session_bounds_and_early_close(sse_calendar):
    s = _sse(sse_calendar, early_closes={date(2026, 9, 15): time(10, 30)})
    bounds = s.session_bounds(date(2026, 9, 15))
    assert bounds == ((datetime(2026, 9, 15, 9, 30, tzinfo=SHANGHAI), datetime(2026, 9, 15, 10, 30, tzinfo=SHANGHAI)),)
    assert s.close_at(date(2026, 9, 15)) == datetime(2026, 9, 15, 10, 30, tzinfo=SHANGHAI)
    full = s.session_bounds(date(2026, 9, 16))
    assert len(full) == 2 and full[1][1].hour == 15


def test_next_previous_trading_day_skip_weekend(sse_calendar):
    s = _sse(sse_calendar)
    assert s.next_trading_day(date(2026, 9, 18)) == date(2026, 9, 21)      # Fri -> Mon
    assert s.previous_trading_day(date(2026, 9, 21)) == date(2026, 9, 18)


def test_sessions_must_be_ordered_and_non_overlapping(sse_calendar):
    with pytest.raises(ValidationError):
        TradingSessionCalendar(name="bad", tz=SHANGHAI, calendar=sse_calendar,
                               sessions=(TradingSession(time(9, 30), time(11, 30)),
                                         TradingSession(time(11, 0), time(15, 0))))
    with pytest.raises(ValidationError):
        TradingSession(time(11, 30), time(9, 30))


def _ny_or_skip():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:  # ZoneInfoNotFoundError when no tz database
        pytest.skip("no tz database")


def test_dst_gap_and_fold_are_rejected_explicitly(sse_calendar):
    ny = _ny_or_skip()
    s = TradingSessionCalendar(name="NYSE", tz=ny, calendar=sse_calendar,
                               sessions=(TradingSession(time(9, 30), time(16, 0)),))
    with pytest.raises(ValidationError, match="ambiguous"):
        s.localize(date(2026, 11, 1), time(1, 30))      # fold: 01:30 occurs twice
    with pytest.raises(ValidationError, match="nonexistent"):
        s.localize(date(2026, 3, 8), time(2, 30))       # gap: 02:30 does not exist
    assert s.localize(date(2026, 11, 1), time(16, 0)).utcoffset() == timedelta(hours=-5)


def test_identity_is_hashable_and_stable(sse_calendar):
    a, b = _sse(sse_calendar), _sse(sse_calendar)
    assert a.identity() == b.identity() and hash(a.identity())
