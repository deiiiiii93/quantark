"""BusinessTimeMap property tests (spec 2026-09-01-trading-clock-vol §4.1, §7.1)."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError

ANCHOR = datetime(2026, 2, 9)   # Monday; CNY 2026 holidays fall mid-February
HORIZON = datetime(2027, 2, 9)


def _cn_calendar():
    return create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def _map(days_per_year=244):
    clock = TradingClock(calendar=_cn_calendar(), days_per_year=days_per_year)
    return BusinessTimeMap(clock, ANCHOR, HORIZON)


def test_to_trading_monotone_and_bounded():
    m = _map()
    taus = np.linspace(0.0, 0.9, 4001)
    out = m.to_trading(taus)
    assert np.all(np.diff(out) >= 0.0)
    assert out[0] == 0.0


def test_holiday_plateau_is_bitwise_flat():
    """Spec §4.2(a): inside a plateau to_trading returns the stored knot VALUE."""
    m = _map()
    cal = _cn_calendar()
    d = ANCHOR
    while cal.is_business_day(d):
        d += timedelta(days=1)
    t0 = (d - ANCHOR).days / 365.0
    a = m.to_trading(t0 + 0.10 / 365.0)
    b = m.to_trading(t0 + 0.90 / 365.0)
    assert a == b                      # bitwise, not approx


def test_trading_day_slope():
    m = _map()
    cal = _cn_calendar()
    d = ANCHOR
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    t0 = (d - ANCHOR).days / 365.0
    lo, hi = m.to_trading(t0), m.to_trading(t0 + 1.0 / 365.0)
    assert (hi - lo) == pytest.approx(1.0 / 244.0, abs=1e-15)


def test_knot_round_trip_exact():
    """to_trading and to_calendar agree exactly at every trading-date knot."""
    m = _map()
    cal = _cn_calendar()
    k, d = 0, ANCHOR
    while d < ANCHOR + timedelta(days=120):
        if cal.is_business_day(d):
            k += 1
            u = k / 244.0
            c = m.to_calendar(u)
            assert m.to_trading(c) == pytest.approx(u, abs=1e-15)
        d += timedelta(days=1)


def test_to_calendar_is_continuous_and_strictly_increasing():
    """Spec §4.1 (review iter-2 P2a): no plateau-start jumps in the cash clock."""
    m = _map()
    us = np.linspace(0.0, 200.0 / 244.0, 20001)
    cs = m.to_calendar(us)
    diffs = np.diff(cs)
    assert np.all(diffs > 0.0)                       # strictly increasing
    assert float(np.max(diffs)) < 15.0 / 365.0 / 40  # no jump anywhere near a holiday span


def test_horizon_fail_closed():
    m = _map()
    with pytest.raises(ValidationError):
        m.to_trading(1.5)          # past horizon
    with pytest.raises(ValidationError):
        m.to_trading(-0.1)


def test_cfets_denominator():
    m = _map(days_per_year=252)
    cal = _cn_calendar()
    d = ANCHOR
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    t0 = (d - ANCHOR).days / 365.0
    assert (m.to_trading(t0 + 1.0 / 365.0) - m.to_trading(t0)) == pytest.approx(
        1.0 / 252.0, abs=1e-15
    )


def test_ndarray_and_scalar_agree():
    m = _map()
    taus = np.array([0.01, 0.1, 0.3])
    arr = m.to_trading(taus)
    for i, t in enumerate(taus):
        assert arr[i] == m.to_trading(float(t))
