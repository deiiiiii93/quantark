"""TradingClockBasisYield: cumulative-preserving basis on the trading axis (0.15.0 spec D11, AC9)."""
from datetime import datetime

import pytest

from quantark.param.basis import FlatBasisYield, TermStructureBasisYield, TradingClockBasisYield, ZeroBasis
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError

ANCHOR = datetime(2026, 2, 9)


def _map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 6, 9))


@pytest.mark.parametrize("inner", [
    FlatBasisYield(0.013),
    TermStructureBasisYield(times=[0.25, 1.0], yields=[0.01, 0.02]),
])
def test_definition_is_bitwise(inner):
    m = _map()
    b = TradingClockBasisYield(inner, m)
    for u in (1 / 244.0, 5 / 244.0, 30 / 244.0, 130 / 244.0, 0.9):
        c = float(m.to_calendar(u))
        assert b.get_basis_yield(u) == inner.get_basis_yield(c) * c / u


def test_cumulative_identity_within_ulps():
    m = _map()
    b = TradingClockBasisYield(FlatBasisYield(0.013), m)
    for u in (5 / 244.0, 30 / 244.0, 130 / 244.0):
        c = float(m.to_calendar(u))
        assert b.get_basis_yield(u) * u == pytest.approx(0.013 * c, rel=1e-14)


def test_zero_time_is_zero():
    assert TradingClockBasisYield(FlatBasisYield(0.013), _map()).get_basis_yield(0.0) == 0.0


def test_exposes_clock_and_rewraps():
    m = _map()
    inner = FlatBasisYield(0.013)
    b = TradingClockBasisYield(inner, m)
    assert b.inner is inner and b.time_map is m and b.clock is m.clock
    m2 = m.re_anchored(datetime(2026, 3, 2))
    b2 = b.with_time_map(m2)
    assert isinstance(b2, TradingClockBasisYield) and b2.inner is inner and b2.time_map is m2
    assert b.to_inner_time(30 / 244.0) == float(m.to_calendar(30 / 244.0))


def test_rejects_non_basis_inner():
    with pytest.raises(ValidationError):
        TradingClockBasisYield(0.01, _map())


def test_is_identically_zero_flags():
    assert ZeroBasis().is_identically_zero()
    assert FlatBasisYield(0.0).is_identically_zero()
    assert not FlatBasisYield(0.001).is_identically_zero()
    assert not TradingClockBasisYield(ZeroBasis(), _map()).is_identically_zero()
