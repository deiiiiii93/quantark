"""AC10c: a clock-wrapped basis yield (new in 0.15.0) is a clock field of the PnL-explain factor model (spec D17).

The TIME state re-anchors every wrapped field at its own valuation date, and two wrappers that differ only by the
map's anchor are the same market (sticky trading tenor), exactly as the rate and dividend wrappers already are.
"""
from copy import deepcopy
from datetime import datetime

from quantark.param import SpotQuote
from quantark.param.basis import FlatBasisYield
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.pnlexplain.equity.clock import CLOCK_FIELDS, re_anchor
from quantark.pnlexplain.equity.factor_diff import basis_market_equal
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
CQ = CurveQuotingBasis.CALENDAR_YEAR


def _env(d):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.02),
                              basis_yield=FlatBasisYield(0.01), valuation_date=d, spot_quote=SpotQuote(100.0),
                              vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS,
                              bus_days_in_year=244, calendar=CAL, rate_curve_quoting=CQ, div_yield_quoting=CQ,
                              basis_yield_quoting=CQ, trading_clock_horizon=datetime(2027, 6, 1))


def test_basis_is_a_clock_field_and_re_anchors():
    assert "basis_yield" in CLOCK_FIELDS
    env = deepcopy(_env(datetime(2026, 2, 13)))
    object.__setattr__(env, "valuation_date", datetime(2026, 2, 24))  # bypass the env's own hook: re_anchor's job
    re_anchor(env)
    assert env.basis_yield.time_map.anchor_date == datetime(2026, 2, 24)


def test_anchor_only_difference_is_no_basis_move():
    e0, e1 = _env(datetime(2026, 2, 13)), _env(datetime(2026, 2, 24))
    assert basis_market_equal(e0.basis_yield, e1.basis_yield)
    moved = _env(datetime(2026, 2, 24))
    object.__setattr__(moved, "basis_yield", moved.basis_yield.with_time_map(moved.basis_yield.time_map))
    assert basis_market_equal(e0.basis_yield, moved.basis_yield)
    assert not basis_market_equal(e0.basis_yield, FlatBasisYield(0.01))  # wrapped vs native is a clock change
