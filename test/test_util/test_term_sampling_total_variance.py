"""Plateau exactness through the full step-vol path (spec §4.2, §7.7)."""
from datetime import datetime

import numpy as np
import pytest

from quantark.param.term_sampling import step_vols_on_grid
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import NumericalError

ANCHOR = datetime(2026, 2, 9)


def _surface():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 2, 9))
    return TradingClockVolSurface(FlatVolSurface(0.20), m), m


def test_holiday_interval_step_vol_is_exactly_zero():
    surface, m = _surface()
    # dense grid: 8 steps per calendar day across 40 days (covers CNY week)
    t = np.linspace(0.0, 40.0 / 365.0, 40 * 8 + 1)
    tv = lambda k, ts: surface.total_variance(k, ts, 100.0)
    out = step_vols_on_grid(
        lambda k, ts: surface.get_vol(k, ts, 100.0), 100.0, t, total_variance=tv
    )
    tau_td = m.to_trading(t)
    holiday_steps = np.diff(tau_td) == 0.0
    assert holiday_steps.any()                       # CNY is in range
    assert np.all(out[holiday_steps] == 0.0)         # BITWISE zero (spec trigger)
    assert np.all(out[~holiday_steps] > 0.0)


def test_without_total_variance_existing_path_is_bitwise_unchanged():
    t = np.linspace(0.01, 1.0, 250)
    get_vol = lambda k, ts: 0.20
    a = step_vols_on_grid(get_vol, 100.0, t)
    b = step_vols_on_grid(get_vol, 100.0, t, total_variance=None)
    assert np.array_equal(a, b)


def test_fast_path_rejects_decreasing_w():
    t = np.array([0.0, 0.1, 0.2])
    bad_tv = lambda k, ts: np.where(np.asarray(ts) > 0.15, 0.001, 0.004)
    with pytest.raises(NumericalError):
        step_vols_on_grid(lambda k, ts: 0.2, 100.0, t, total_variance=bad_tv)
