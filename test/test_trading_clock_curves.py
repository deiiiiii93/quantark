"""Trading-axis cash wrappers (spec §4.3, review iter-1 P1 + iter-2 P2a)."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

ANCHOR = datetime(2026, 2, 9)
R, Q = 0.02, 0.01


def _map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 6, 9)), cal


def test_df_invariance_at_every_trading_knot():
    """Spec §7.3 / D5: DF_td(u_k) equals DF_cal(c_k) at every trading date."""
    m, cal = _map()
    curve = TradingClockRateCurve(inner=FlatRateCurve(R), time_map=m)
    k, d = 0, ANCHOR
    while d < ANCHOR + timedelta(days=200):
        if cal.is_business_day(d):
            k += 1
            u = k / 244.0
            c = (d + timedelta(days=1) - ANCHOR).days / 365.0  # end-of-day knot
            assert curve.get_discount_factor(u) == pytest.approx(
                math.exp(-R * c), rel=1e-15
            )
        d += timedelta(days=1)


def test_df_is_continuous_in_trading_time():
    """Review iter-2 P2a: no holiday-carry jump between adjacent substeps."""
    m, _ = _map()
    curve = TradingClockRateCurve(inner=FlatRateCurve(R), time_map=m)
    us = np.linspace(1e-6, 200.0 / 244.0, 40001)
    dfs = np.array([curve.get_discount_factor(float(u)) for u in us])
    rel_jumps = np.abs(np.diff(dfs)) / dfs[:-1]
    assert float(np.max(rel_jumps)) < R * (15.0 / 365.0) / 40  # spread, not spiked


def test_dividend_wrapper_preserves_cumulative_yield():
    """Review iter-1 P1: q_td(u)*u == q_cal(c)*c, NOT q_cal(c)*u."""
    m, _ = _map()
    y = TradingClockDividendYield(inner=ContinuousDividendYield(Q), time_map=m)
    for u in (5 / 244.0, 30 / 244.0, 130 / 244.0):
        c = m.to_calendar(u)
        assert y.get_yield(u) * u == pytest.approx(Q * c, rel=1e-14)


def test_forward_rates_telescope_exactly():
    m, _ = _map()
    curve = TradingClockRateCurve(inner=FlatRateCurve(R), time_map=m)
    u = np.linspace(1 / 244.0, 100 / 244.0, 100)
    dfs = np.array([curve.get_discount_factor(float(x)) for x in u])
    total = -math.log(dfs[-1] / dfs[0])
    parts = -np.log(dfs[1:] / dfs[:-1])
    assert float(np.sum(parts)) == pytest.approx(total, rel=1e-13)
