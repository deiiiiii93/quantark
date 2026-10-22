"""The factor coordinate on a clock-wrapped pair (patch spec 2026-09-03 §14).

A clock wrapper converts the coordinate tenor with its OWN anchor, and the two
snapshots are anchored at their own valuation dates by construction. Sampling a
wrapped object through the environment therefore reads two different points of
one unchanged market: a phantom move that the change detector (which compares
class, inner and clock, ignoring the anchor) correctly says did not happen, and
that the waterfall books under TIME by re-anchoring on the time step.

Both sides are now sampled on the INNER's own axis at the t1 coordinate, so a
frozen market differences to exactly zero and the move carries the same unit as
the bump (sigma_td for vol, the calendar-quoted r and q for the rate and
dividend wrappers).
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.param.vol.vol_surface import TermStructureVolSurface
from quantark.pnlexplain import (
    ExplainMethod, Factor, PnLExplainConfig, ValuationSnapshot, explain,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import OptionType

D = 244
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
EXPIRY = datetime(2026, 12, 15)
HORIZON = datetime(2027, 3, 1)
ENG = BlackScholesEngine()
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))
CALL = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=EXPIRY)

# National Day week: 9 calendar days, 2 trading days. The anchor slide is largest
# across a closed block, so this step maximises the phantom move.
T0, T1 = datetime(2026, 9, 30), datetime(2026, 10, 9)


def _map(anchor):
    return BusinessTimeMap(TradingClock(CAL, D), anchor, HORIZON)


def _cal_env(date, spot=100.0, inner=None):
    """Calendar axis: the vol surface is wrapped, r and q are native."""
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.02), valuation_date=date, spot_quote=SpotQuote(spot),
        vol_surface=TradingClockVolSurface(inner or FlatVolSurface(0.20), _map(date)),
        div_yield=ContinuousDividendYield(0.01),
    )


def _td_env(date, spot=100.0, r=0.02, q=0.01):
    """Trading axis: r and q are wrapped, the vol surface is native."""
    m = _map(date)
    return PricingEnvironment(
        rate_curve=TradingClockRateCurve(FlatRateCurve(r), m), valuation_date=date,
        spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(0.20),
        div_yield=TradingClockDividendYield(ContinuousDividendYield(q), m),
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D, calendar=CAL,
    )


def _moves(e0, e1):
    return explain(ValuationSnapshot(CALL, ENG, e0, date=T0),
                   ValuationSnapshot(CALL, ENG, e1, date=T1), config=WF).moves


def test_frozen_clock_vol_surface_is_exactly_no_vol_move():
    """The change detector already says the market did not move; the sampled
    move must agree instead of reading the anchor slide as a vol move."""
    mv = _moves(_cal_env(T0), _cal_env(T1, spot=101.0))
    assert Factor.VOL not in mv.changed
    assert mv.d_vol == 0.0
    assert mv.vol_t0 == mv.vol_t1 == 0.20


def test_clock_vol_move_is_the_inner_move_in_trading_points():
    """A real surface move is reported per sigma_td point, the unit the vega
    bump moves, with no sqrt(tau_td/tau_cal) conversion applied to it."""
    mv = _moves(_cal_env(T0), _cal_env(T1, inner=FlatVolSurface(0.21)))
    assert Factor.VOL in mv.changed
    assert mv.d_vol == pytest.approx(0.01, abs=1e-15)


def test_clock_vol_coordinate_is_the_t1_trading_tenor():
    """Both sides read the SAME point of their inner: the t1 coordinate mapped
    onto the trading axis by the t1 wrapper."""
    inner_t1 = TermStructureVolSurface(times=[0.05, 0.2, 0.5, 1.0], vols=[0.24, 0.22, 0.20, 0.18])
    e0, e1 = _cal_env(T0), _cal_env(T1, inner=inner_t1)
    mv = _moves(e0, e1)
    point = e1.vol_surface.to_inner_time(mv.coordinate.tenor_t1)
    assert mv.vol_t1 == pytest.approx(inner_t1.get_vol(100.0, point, 100.0), abs=1e-15)
    assert mv.vol_t0 == 0.20


def test_frozen_clock_rate_and_dividend_are_exactly_no_move():
    mv = _moves(_td_env(T0), _td_env(T1, spot=101.0))
    assert Factor.RATE not in mv.changed and Factor.DIVIDEND not in mv.changed
    assert mv.d_rate == 0.0 and mv.d_div == 0.0
    assert mv.rate_t0 == mv.rate_t1 == 0.02


def test_clock_rate_and_dividend_moves_are_the_calendar_quoted_inner_moves():
    """The wrappers' inners are calendar-quoted, so the move is per calendar
    year, matching what build_rate_bumped_env shifts."""
    mv = _moves(_td_env(T0), _td_env(T1, r=0.03, q=0.015))
    assert mv.d_rate == pytest.approx(0.01, abs=1e-15)
    assert mv.d_div == pytest.approx(0.005, abs=1e-15)


def test_native_environment_sampling_is_untouched():
    """Regression guard: off the clock the move is still the environment's own
    (K, T1) reading on both sides."""
    s0 = TermStructureVolSurface(times=[0.1, 0.5, 1.0], vols=[0.22, 0.20, 0.19])
    s1 = TermStructureVolSurface(times=[0.1, 0.5, 1.0], vols=[0.23, 0.21, 0.20])

    def env(date, surface):
        return PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=date,
                                  spot_quote=SpotQuote(100.0), vol_surface=surface,
                                  div_yield=ContinuousDividendYield(0.01))

    e0, e1 = env(T0, s0), env(T1, s1)
    mv = _moves(e0, e1)
    tenor = mv.coordinate.tenor_t1
    assert mv.vol_t0 == e0.get_vol(100.0, tenor)
    assert mv.vol_t1 == e1.get_vol(100.0, tenor)
