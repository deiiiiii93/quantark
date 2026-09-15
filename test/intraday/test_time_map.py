from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from quantark.intraday.profile import IntradayTimeMap, SegmentKind, VarianceProfile
from quantark.intraday.session import TradingSessionCalendar
from quantark.intraday.timestamp import SECONDS_PER_YEAR, calendar_year_fraction
from quantark.param import FlatVolSurface
from quantark.param.vol import TradingClockVolSurface
from quantark.util.calendar import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI

D = 244
TS = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)      # Tue, 1s before close
HORIZON = datetime(2027, 9, 30, tzinfo=SHANGHAI)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", D, overnight_weight=0.25, session_weights=(0.35, 0.35), break_weights=(0.05,))


def _tau(a, b):
    return calendar_year_fraction(a, b)


def test_one_trading_day_between_closes_is_exactly_one_over_d(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    close_today = datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)
    close_next = datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI)
    close_fri = datetime(2026, 9, 18, 15, 0, tzinfo=SHANGHAI)
    close_mon = datetime(2026, 9, 21, 15, 0, tzinfo=SHANGHAI)        # Fri -> Mon: one trading day, 3 calendar days
    u0, u1 = m.to_trading(_tau(TS, close_today)), m.to_trading(_tau(TS, close_next))
    assert u1 - u0 == pytest.approx(1.0 / D, abs=1e-15)
    assert m.to_trading(_tau(TS, close_mon)) - m.to_trading(_tau(TS, close_fri)) == pytest.approx(1.0 / D, abs=1e-15)


def test_partial_first_segment_accrues_the_last_second_of_the_afternoon(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    # afternoon session = 2h = 7200s carries 0.35/D; the last second carries 0.35/D/7200
    assert m.to_trading(1.0 / SECONDS_PER_YEAR) == pytest.approx(0.35 / D / 7200.0, rel=1e-12)
    assert m.segments[0].kind is SegmentKind.SESSION and m.segments[0].start == TS
    assert m.initial_slope() == pytest.approx((0.35 / D) / (7200.0 / SECONDS_PER_YEAR), rel=1e-12)


def test_zero_weight_segment_is_a_bitwise_plateau(sse_sessions):
    p = VarianceProfile.sessions_only(sse_sessions, D)
    anchor = datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)        # exactly at close -> overnight segment first
    m = IntradayTimeMap(sse_sessions, p, anchor, HORIZON)
    assert m.segments[0].kind is SegmentKind.OVERNIGHT and m.segments[0].weight == 0.0
    taus = np.array([0.0, 3600.0, 12 * 3600.0, 18.5 * 3600.0 - 1.0]) / SECONDS_PER_YEAR
    u = m.to_trading(taus)
    assert u.tolist() == [0.0, 0.0, 0.0, 0.0]                        # exact
    assert m.initial_slope() == 0.0
    assert m.variance_time_between(taus[1], taus[3]) == 0.0
    lunch = _tau(anchor, datetime(2026, 9, 16, 11, 30, tzinfo=SHANGHAI))
    after_lunch = _tau(anchor, datetime(2026, 9, 16, 13, 0, tzinfo=SHANGHAI))
    assert m.to_trading(lunch) == m.to_trading(after_lunch) == pytest.approx(0.5 / D)


def test_additivity_and_monotonicity_on_a_dense_grid(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    taus = np.linspace(0.0, 0.9, 20001)
    u = m.to_trading(taus)
    assert np.all(np.diff(u) >= 0.0)
    a, b, c = 0.11, 0.37, 0.62
    whole = m.to_trading(c) - m.to_trading(a)
    parts = (m.to_trading(b) - m.to_trading(a)) + (m.to_trading(c) - m.to_trading(b))
    assert whole == pytest.approx(parts, abs=1e-15)


def test_early_close_keeps_the_days_budget(sse_sessions, desk):
    cal = TradingSessionCalendar(name="SSE", tz=sse_sessions.tz, calendar=sse_sessions.calendar,
                                 sessions=sse_sessions.sessions, early_closes={date(2026, 9, 16): time(10, 30)})
    m = IntradayTimeMap(cal, desk, TS, HORIZON)
    close_today = datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI)
    early = datetime(2026, 9, 16, 10, 30, tzinfo=SHANGHAI)
    assert m.to_trading(_tau(TS, early)) - m.to_trading(_tau(TS, close_today)) == pytest.approx(1.0 / D, abs=1e-15)


def test_beyond_horizon_and_negative_raise(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    with pytest.raises(ValidationError):
        m.to_trading(-1e-9)
    with pytest.raises(ValidationError):
        m.to_trading(_tau(TS, HORIZON) + 1.0)


def test_to_calendar_agrees_with_to_trading_at_knots_and_is_strictly_increasing(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    for seg in m.segments[:50]:
        if seg.weight > 0.0:
            assert m.to_calendar(seg.u_end) == pytest.approx(seg.tau_end, abs=1e-15)
    us = np.linspace(0.0, m.segments[200].u_end, 500)
    assert np.all(np.diff(m.to_calendar(us)) > 0.0)


def test_re_anchored_preserves_absolute_schedule(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    later = TS + timedelta(hours=20)
    m2 = m.re_anchored(later)
    target = datetime(2026, 10, 15, 15, 0, tzinfo=SHANGHAI)
    assert m.to_trading(_tau(TS, target)) - m.to_trading(_tau(TS, later)) == pytest.approx(
        m2.to_trading(_tau(later, target)), abs=1e-15)


def test_daily_business_time_map_is_the_one_day_special_case(sse_calendar, sse_sessions, desk):
    # Between consecutive trading-day closes both maps advance exactly 1/D; the legacy map places it
    # midnight-to-midnight, the intraday map close-to-close (the profile's declared ownership).
    anchor_midnight = datetime(2026, 9, 15, tzinfo=SHANGHAI)
    legacy = BusinessTimeMap(TradingClock(sse_calendar, D), datetime(2026, 9, 15), datetime(2027, 9, 30))
    m = IntradayTimeMap(sse_sessions, desk, anchor_midnight, HORIZON)
    for k in (1, 5, 22, 60):
        assert legacy.to_trading(k / 365.0) - legacy.to_trading((k - 1) / 365.0) in (0.0, pytest.approx(1.0 / D, abs=1e-15))
    c0 = _tau(anchor_midnight, datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI))
    c1 = _tau(anchor_midnight, datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI))
    assert m.to_trading(c1) - m.to_trading(c0) == pytest.approx(1.0 / D, abs=1e-15)


def test_trading_clock_vol_surface_accepts_the_map_and_differences_exactly(sse_sessions, desk):
    m = IntradayTimeMap(sse_sessions, desk, TS, HORIZON)
    surf = TradingClockVolSurface(FlatVolSurface(0.20), m)
    t1 = _tau(TS, datetime(2026, 9, 16, 11, 30, tzinfo=SHANGHAI))
    t2 = _tau(TS, datetime(2026, 9, 16, 13, 0, tzinfo=SHANGHAI))
    w = surf.total_variance(100.0, np.array([t1, t2]), 100.0)
    assert (w[1] - w[0]) == pytest.approx(0.04 * 0.05 / D, rel=1e-12)      # lunch weight 0.05
    p0 = VarianceProfile.sessions_only(sse_sessions, D)
    surf0 = TradingClockVolSurface(FlatVolSurface(0.20), IntradayTimeMap(sse_sessions, p0, TS, HORIZON))
    w0 = surf0.total_variance(100.0, np.array([t1, t2]), 100.0)
    assert w0[1] - w0[0] == 0.0                                            # bitwise zero across lunch
    assert surf.get_vol(100.0, 0.0, 100.0) > 0.0                          # initial_slope path


def test_segments_are_instant_based_across_a_dst_change(sse_calendar):
    try:
        from zoneinfo import ZoneInfo
        ny = ZoneInfo("America/New_York")
    except Exception:  # ZoneInfoNotFoundError when no tz database
        pytest.skip("no tz database")
    from quantark.intraday.session import TradingSession
    from quantark.util.calendar import CalendarType, create_calendar
    us = create_calendar(CalendarType.US, year_range=(2026, 2027))
    nyse = TradingSessionCalendar(name="NYSE", tz=ny, calendar=us, sessions=(TradingSession(time(9, 30), time(16, 0)),))
    p = VarianceProfile.uniform(nyse, 252, reference_date=date(2026, 3, 4))
    fri_close = nyse.close_at(date(2026, 3, 6))
    mon_close = nyse.close_at(date(2026, 3, 9))
    anchor = datetime(2026, 3, 5, 12, 0, tzinfo=ny)
    m = IntradayTimeMap(nyse, p, anchor, datetime(2026, 4, 30, tzinfo=ny))
    overnight = next(s for s in m.segments if s.kind is SegmentKind.OVERNIGHT and s.trading_day == date(2026, 3, 9))
    # Friday 16:00 EST -> Monday 09:30 EDT is 65.5 wall-clock hours but 64.5 elapsed hours
    assert (overnight.tau_end - overnight.tau_start) * SECONDS_PER_YEAR == pytest.approx(64.5 * 3600.0, abs=1e-6)
    assert m.to_trading(_tau(anchor, mon_close)) - m.to_trading(_tau(anchor, fri_close)) == pytest.approx(1.0 / 252, abs=1e-15)
