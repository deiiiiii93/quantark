"""TradingClockVolSurface: variance preservation (spec §4.2, §7)."""
from datetime import datetime

import numpy as np
import pytest

from quantark.param.vol.vol_surface import FlatVolSurface, TermStructureVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

ANCHOR = datetime(2026, 2, 9)
HORIZON = datetime(2027, 2, 9)


def _map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return BusinessTimeMap(TradingClock(cal, 244), ANCHOR, HORIZON)


def test_total_variance_is_preserved_flat():
    """sigma_td flat: w_cal(tau) == sigma_td^2 * tau_td(tau) for any tau."""
    m = _map()
    s = TradingClockVolSurface(inner=FlatVolSurface(0.20), time_map=m)
    for tau in (0.05, 0.11, 0.37, 0.80):
        tau_td = m.to_trading(tau)
        sigma_cal = s.get_vol(100.0, tau, 100.0)
        assert sigma_cal**2 * tau == pytest.approx(0.04 * tau_td, rel=1e-14)


def test_total_variance_method_matches_get_vol_reconstruction():
    m = _map()
    inner = TermStructureVolSurface(
        times=[10 / 244, 60 / 244, 200 / 244], vols=[0.25, 0.22, 0.20]
    )
    s = TradingClockVolSurface(inner=inner, time_map=m)
    tau = 0.33
    w = s.total_variance(100.0, tau, 100.0)
    sigma = s.get_vol(100.0, tau, 100.0)
    assert w == pytest.approx(sigma * sigma * tau, rel=1e-13)


def test_is_smile_passthrough():
    m = _map()
    assert TradingClockVolSurface(FlatVolSurface(0.2), m).is_smile is False


def test_zero_time_limit_uses_initial_slope():
    m = _map()
    s = TradingClockVolSurface(FlatVolSurface(0.20), m)
    v = s.get_vol(100.0, 0.0, 100.0)
    assert v == pytest.approx(0.20 * np.sqrt(m.initial_slope()), rel=1e-14)


def test_total_variance_vectorized_over_time():
    m = _map()
    s = TradingClockVolSurface(FlatVolSurface(0.20), m)
    taus = np.array([0.05, 0.11, 0.37])
    w = s.total_variance(100.0, taus, 100.0)
    assert w.shape == taus.shape
    for i, t in enumerate(taus):
        assert w[i] == s.total_variance(100.0, float(t), 100.0)
