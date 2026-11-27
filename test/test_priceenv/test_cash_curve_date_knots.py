"""Business-day cash curves read the actual date (run 8, .ship/2026-11-27-cash-wrapper-date-knots, D1-D4 + R1-R6).

On a BUSINESS_DAYS environment the CALENDAR_YEAR-declared cash curves (rate, dividend, basis) sit on the env's
``DateKnotTimeMap``: one knot per business date d, (env_time(d), (d - valuation).days / 365). Every business date --
Mondays and the first trading day after a holiday included -- reads the calendar curve at its actual ACT/365 time.
The 0.15.0 ``BusinessTimeMap`` mapped a date after a non-trading day back to the day after the previous trading day
(a Monday was discounted for 1 day from Friday instead of 3).
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import SpotQuote
from quantark.param.basis.basis_yield import BasisRelationshipMode, FlatBasisYield, TermStructureBasisYield
from quantark.param.basis.trading_clock_basis import TradingClockBasisYield
from quantark.param.div.dividend_yield import ContinuousDividendYield, TermStructureDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol import CalendarQuotedVolSurface, FlatVolSurface, TermStructureVolSurface
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.date_axis import env_time, env_times
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, DateKnotTimeMap, TradingClock
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
D = 244
VAL = datetime(2026, 1, 16)              # a Friday
HORIZON = datetime(2027, 6, 30)
WEDNESDAY = datetime(2026, 1, 21)
MONDAY = datetime(2026, 1, 19)
NEXT_MONDAY = datetime(2026, 1, 26)
POST_FESTIVAL = datetime(2026, 2, 24)    # first SSE trading day after the 2026 Spring Festival block
POST_QINGMING = datetime(2026, 4, 7)     # first SSE trading day after Qingming
MONDAY_YEAR_OUT = datetime(2027, 1, 18)
CAL_Q = CurveQuotingBasis.CALENDAR_YEAR
SPOT = 100.0


def _business_dates(start, end, calendar=CAL):
    out, d = [], start + timedelta(days=1)
    while d <= end:
        if calendar.is_business_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _c(d, anchor=VAL):
    return (d - anchor).days / 365.0


def _inners(kind):
    if kind == "flat":
        return FlatRateCurve(0.03), ContinuousDividendYield(0.01), FlatBasisYield(0.004)
    return (
        LinearRateCurve(pillars=[(0.05, 0.021), (0.25, 0.026), (0.5, 0.029), (1.0, 0.031), (2.0, 0.034)]),
        TermStructureDividendYield(times=[0.05, 0.3, 1.0, 2.0], yields=[0.004, 0.012, 0.018, 0.02]),
        TermStructureBasisYield(times=[0.05, 0.5, 2.0], yields=[0.001, 0.005, 0.007]),
    )


_DEFAULT = object()


def _env(rate=None, div=_DEFAULT, basis=_DEFAULT, valuation_date=VAL, horizon=HORIZON, calendar=CAL,
         holiday=BusinessDayConvention.PRECEDING, vol=None, **kw):
    r0, q0, b0 = _inners("flat")
    quoting = dict(rate_curve_quoting=CAL_Q, div_yield_quoting=CAL_Q, basis_yield_quoting=CAL_Q)
    quoting.update(kw)
    return PricingEnvironment(
        rate_curve=r0 if rate is None else rate, div_yield=q0 if div is _DEFAULT else div,
        basis_yield=b0 if basis is _DEFAULT else basis, valuation_date=valuation_date, spot_quote=SpotQuote(SPOT),
        vol_surface=FlatVolSurface(0.2) if vol is None else vol,
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D, calendar=calendar,
        holiday_time_adjustment=holiday, trading_clock_horizon=horizon, **quoting)


def test_window_has_the_non_trading_blocks_it_claims():
    assert VAL.weekday() == 4 and MONDAY.weekday() == 0 and NEXT_MONDAY.weekday() == 0
    assert MONDAY_YEAR_OUT.weekday() == 0 and WEDNESDAY.weekday() == 2
    for d, gap in ((POST_FESTIVAL, 7), (POST_QINGMING, 3)):
        assert CAL.is_business_day(d)
        assert not any(CAL.is_business_day(d - timedelta(days=k)) for k in range(1, gap + 1))


# ---------------------------------------------------------------------------------------------------- D1 home + R6


def test_date_knot_map_lives_next_to_business_time_map_and_vol_re_exports_it():
    from quantark.param import vol

    assert DateKnotTimeMap.__module__ == "quantark.util.calendar.trading_clock"
    assert vol.DateKnotTimeMap is DateKnotTimeMap
    assert issubclass(DateKnotTimeMap, BusinessTimeMap)


def test_business_time_map_exposes_its_knots_read_only():
    m = BusinessTimeMap(TradingClock(CAL, D), VAL, HORIZON)
    assert m.u_knots is m._u_knots
    with pytest.raises(AttributeError):
        m.u_knots = np.zeros(2)
    with pytest.raises(ValueError):
        m.u_knots[0] = 1.0


def test_env_map_knots_are_the_env_times_of_every_business_date():
    """R6: the business-day count the map's knots use is ``date_axis.env_times`` itself (function-local import)."""
    env = _env()
    m = env.rate_curve.time_map
    dates = _business_dates(VAL, HORIZON)
    assert type(m) is DateKnotTimeMap
    assert list(m.knot_dates) == [VAL] + dates
    assert np.array_equal(m.u_knots, np.r_[0.0, env_times(env, dates)])


# ---------------------------------------------------------------------------------------------------- AC1 / D2 / R4


@pytest.mark.parametrize("kind", ["flat", "term"])
def test_cash_curves_read_the_calendar_curve_at_every_business_date(kind):
    rate, div, basis = _inners(kind)
    env = _env(rate, div, basis)
    assert type(env.rate_curve) is TradingClockRateCurve
    assert type(env.div_yield) is TradingClockDividendYield
    assert type(env.basis_yield) is TradingClockBasisYield
    dates = _business_dates(VAL, datetime(2027, 3, 31))
    for d in (MONDAY, NEXT_MONDAY, POST_FESTIVAL, POST_QINGMING, MONDAY_YEAR_OUT):
        assert d in dates
    for d, u in zip(dates, env_times(env, dates)):
        c = _c(d)
        assert env.get_discount_factor(u) == pytest.approx(rate.get_discount_factor(c), rel=1e-15, abs=0.0)
        assert env.div_yield.get_yield(u) * u == pytest.approx(div.get_yield(c) * c, rel=1e-15, abs=0.0)
        assert env.basis_yield.get_basis_yield(u) * u == pytest.approx(
            basis.get_basis_yield(c) * c, rel=1e-15, abs=0.0)


def test_measured_lag_is_closed_at_three_percent():
    """The run 8 Goal's measurements: DF error +1.64e-4 on a Monday, +8.20e-4 on 02-24 -- now zero."""
    env = _env(FlatRateCurve(0.03))
    for d, days in ((MONDAY, 3), (NEXT_MONDAY, 10), (POST_FESTIVAL, 39), (MONDAY_YEAR_OUT, 367)):
        assert (d - VAL).days == days
        assert env.get_discount_factor(env_time(env, d)) == pytest.approx(math.exp(-0.03 * days / 365.0),
                                                                          rel=1e-15, abs=0.0)


def test_cash_curves_are_linear_in_trading_time_between_business_dates():
    """Between two business-date knots the calendar time is linear in u (a fractional env time)."""
    env = _env(FlatRateCurve(0.03))
    m = env.rate_curve.time_map
    u0, u1 = env_time(env, datetime(2026, 1, 23)), env_time(env, NEXT_MONDAY)
    u = 0.5 * (u0 + u1)
    c = 0.5 * (_c(datetime(2026, 1, 23)) + _c(NEXT_MONDAY))
    assert m.to_calendar(u) == pytest.approx(c, rel=1e-15)
    assert env.get_discount_factor(u) == pytest.approx(math.exp(-0.03 * c), rel=1e-15)


# ---------------------------------------------------------------------------------------------------- AC2


@pytest.mark.parametrize("vol_kind", ["flat", "term"])
@pytest.mark.parametrize("expiry", [WEDNESDAY, NEXT_MONDAY, POST_FESTIVAL])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_european_on_the_business_day_clock_equals_the_calendar_clock(vol_kind, expiry, option_type):
    """r = 3%, q = 1%, run 7's calendar-quoted vol: run 7 recorded -0.83% (Mon 01-26) / -1.73% (02-24) gaps."""
    inner = FlatVolSurface(0.21) if vol_kind == "flat" else TermStructureVolSurface(
        times=[0.02, 0.1, 0.4, 1.0, 1.5], vols=[0.32, 0.27, 0.23, 0.21, 0.205])
    vol = CalendarQuotedVolSurface(inner, DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON))
    bd_env = _env(FlatRateCurve(0.03), ContinuousDividendYield(0.01), vol=vol, basis_yield_quoting=None,
                  basis=None)
    cal_env = PricingEnvironment(rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                                 valuation_date=VAL, spot_quote=SpotQuote(SPOT), vol_surface=inner)
    opt = EuropeanVanillaOption(strike=103.0, option_type=option_type, exercise_date=expiry)
    engine = BlackScholesEngine()
    assert opt.get_maturity(bd_env) == env_time(bd_env, expiry)
    assert engine.price(opt, bd_env) == pytest.approx(engine.price(opt, cal_env), rel=1e-12, abs=0.0)


# ---------------------------------------------------------------------------------------------------- AC3


def test_undeclared_hand_wrapped_business_time_map_curve_is_untouched():
    hand_map = BusinessTimeMap(TradingClock(CAL, D), VAL, HORIZON)
    hand = TradingClockRateCurve(FlatRateCurve(0.03), hand_map)
    env = PricingEnvironment(rate_curve=hand, valuation_date=VAL, spot_quote=SpotQuote(SPOT),
                             vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS,
                             bus_days_in_year=D, calendar=CAL)
    assert env.rate_curve is hand and type(env.rate_curve.time_map) is BusinessTimeMap
    for d in (MONDAY, POST_FESTIVAL):
        u = env_time(env, d)
        assert env.get_discount_factor(u) == FlatRateCurve(0.03).get_discount_factor(float(hand_map.to_calendar(u)))


def test_trading_year_and_calendar_days_curves_are_read_as_is():
    rate = FlatRateCurve(0.03)
    trd = _env(rate, rate_curve_quoting=CurveQuotingBasis.TRADING_YEAR)
    assert trd.rate_curve is rate
    cal = PricingEnvironment(rate_curve=rate, valuation_date=VAL, rate_curve_quoting=CAL_Q)
    assert cal.rate_curve is rate


# ---------------------------------------------------------------------------------------------------- AC4


@pytest.mark.parametrize("div", [ContinuousDividendYield(0.02), None], ids=["declared", "declared-none"])
def test_carry_rhoq_bucket_bump_on_the_date_knot_map(div):
    from quantark.asset.equity.report.term_structure import BucketedDividendYield
    from quantark.asset.equity.riskmeasures.bucketed_coordinates.carry_rhoq import _bucket_bump

    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), div_yield=div, valuation_date=VAL, spot_quote=SpotQuote(SPOT),
        vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D,
        calendar=CAL, rate_curve_quoting=CAL_Q, div_yield_quoting=CAL_Q, trading_clock_horizon=HORIZON)
    m = env.rate_curve.time_map
    assert type(m) is DateKnotTimeMap
    base = div if div is not None else ContinuousDividendYield(0.0)
    u_max = float(m.u_knots[-1])
    start, end, past = env_time(env, MONDAY), env_time(env, POST_FESTIVAL), u_max + 0.5
    for a, b, hand_a, hand_b in (
        (start, end, _c(MONDAY), _c(POST_FESTIVAL)),
        (end, past, _c(POST_FESTIVAL), float(m.c_knots[-1]) + 0.5),
    ):
        bumped = _bucket_bump(env, a, b, 0.001)
        assert type(bumped) is TradingClockDividendYield and type(bumped.time_map) is DateKnotTimeMap
        assert np.array_equal(bumped.time_map.u_knots, m.u_knots)
        hand = TradingClockDividendYield(BucketedDividendYield(base=base, bucket_start=hand_a, bucket_end=hand_b,
                                                               bump=0.001), m)
        for d in (MONDAY, WEDNESDAY, POST_FESTIVAL, POST_QINGMING, MONDAY_YEAR_OUT):
            u = env_time(env, d)
            assert bumped.get_yield(u) == hand.get_yield(u)


# ---------------------------------------------------------------------------------------------------- AC5 / D3


def test_hand_wrapped_declared_curve_on_a_business_time_map_moves_onto_the_env_map():
    hand = TradingClockRateCurve(FlatRateCurve(0.03), BusinessTimeMap(TradingClock(CAL, D), VAL, HORIZON))
    env = _env(hand)
    assert env.rate_curve is not hand and env.rate_curve.inner is hand.inner
    assert type(env.rate_curve.time_map) is DateKnotTimeMap
    assert env.rate_curve.time_map is env.div_yield.time_map
    u = env_time(env, MONDAY)
    assert env.get_discount_factor(u) == FlatRateCurve(0.03).get_discount_factor(3.0 / 365.0)


def test_hand_wrapped_declared_curve_on_the_env_date_knot_map_is_kept():
    own = DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON)
    hand = TradingClockRateCurve(FlatRateCurve(0.03), own)
    env = _env(hand)
    assert env.rate_curve is hand


def test_hand_wrapped_declared_curve_on_another_holiday_convention_moves():
    other = DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON, BusinessDayConvention.FOLLOWING)
    hand = TradingClockRateCurve(FlatRateCurve(0.03), other)
    env = _env(hand)
    assert env.rate_curve.inner is hand.inner and env.rate_curve.time_map is not other
    assert env.rate_curve.time_map.holiday_time_adjustment is BusinessDayConvention.PRECEDING


def test_after_a_valuation_move_the_three_declared_wrappers_share_one_map():
    own = DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON)
    env = _env(TradingClockRateCurve(FlatRateCurve(0.03), own))
    env.valuation_date = datetime(2026, 2, 11)
    maps = [env.rate_curve.time_map, env.div_yield.time_map, env.basis_yield.time_map]
    assert all(m is maps[0] for m in maps) and type(maps[0]) is DateKnotTimeMap
    assert maps[0].anchor_date == datetime(2026, 2, 11) and maps[0].horizon_date == HORIZON
    u = env_time(env, POST_FESTIVAL)
    assert env.get_discount_factor(u) == pytest.approx(math.exp(-0.03 * 13 / 365.0), rel=1e-15)


# ---------------------------------------------------------------------------------------------------- R1 reach


def test_date_knot_map_reaches_the_first_business_date_on_or_after_the_horizon():
    saturday = datetime(2026, 1, 24)
    m = DateKnotTimeMap(TradingClock(CAL, D), VAL, saturday)
    assert m.knot_dates[-1] == NEXT_MONDAY
    festival = DateKnotTimeMap(TradingClock(CAL, D), datetime(2026, 2, 13), datetime(2026, 2, 20))
    assert festival.knot_dates[1:] == (POST_FESTIVAL,)
    exact = DateKnotTimeMap(TradingClock(CAL, D), VAL, WEDNESDAY)
    assert exact.knot_dates[-1] == WEDNESDAY


def test_last_date_plus_one_day_horizon_reads_friday_and_next_monday():
    env = _env(FlatRateCurve(0.03), horizon=datetime(2026, 1, 24))        # last date Fri 01-23 + 1 day
    for d in (datetime(2026, 1, 23), NEXT_MONDAY):
        assert env.get_discount_factor(env_time(env, d)) == pytest.approx(math.exp(-0.03 * _c(d)), rel=1e-15)


def test_following_env_reads_a_weekend_last_date_at_its_placed_business_date():
    env = _env(FlatRateCurve(0.03), horizon=datetime(2026, 1, 25), holiday=BusinessDayConvention.FOLLOWING)
    u = env_time(env, datetime(2026, 1, 24))                                 # a Saturday, placed on Mon 01-26
    assert u == env_time(env, NEXT_MONDAY)
    assert env.get_discount_factor(u) == pytest.approx(math.exp(-0.03 * _c(NEXT_MONDAY)), rel=1e-15)


# ---------------------------------------------------------------------------------------------------- R3


def test_following_us_holiday_flow_discounts_at_the_placed_business_date():
    """A holiday-dated flow (DCN-style ``env_time``) on a FOLLOWING env discounts to DF_cal of the date the env
    places it on (the next business date), not the day after the previous trading day."""
    us = create_calendar(CalendarType.US, year_range=(2025, 2028))
    val, flow = datetime(2026, 6, 1), datetime(2026, 7, 4)
    assert not us.is_business_day(flow)
    placed = flow
    while not us.is_business_day(placed):
        placed += timedelta(days=1)
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=val, spot_quote=SpotQuote(SPOT),
        vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252,
        calendar=us, holiday_time_adjustment=BusinessDayConvention.FOLLOWING, rate_curve_quoting=CAL_Q,
        trading_clock_horizon=datetime(2026, 12, 31))
    u = env_time(env, flow)
    assert u == env_time(env, placed)
    assert env.get_discount_factor(u) == pytest.approx(math.exp(-0.03 * (placed - val).days / 365.0), rel=1e-15)


# ---------------------------------------------------------------------------------------------------- AC6 (D4) / R5


def test_following_env_valued_on_a_holiday_refuses_at_construction():
    saturday = datetime(2026, 1, 17)
    with pytest.raises(ValidationError, match="FOLLOWING"):
        _env(FlatRateCurve(0.03), valuation_date=saturday, holiday=BusinessDayConvention.FOLLOWING)
    env = _env(FlatRateCurve(0.03), valuation_date=saturday)                # PRECEDING default unaffected
    assert env.get_discount_factor(env_time(env, MONDAY)) == pytest.approx(math.exp(-0.03 * 2 / 365.0), rel=1e-15)


def test_following_env_moved_onto_a_holiday_refuses_and_stays_as_it_was():
    env = _env(FlatRateCurve(0.03), holiday=BusinessDayConvention.FOLLOWING)
    before = (env.rate_curve, env.div_yield, env.basis_yield)
    with pytest.raises(ValidationError, match="FOLLOWING"):
        env.valuation_date = datetime(2026, 1, 17)
    assert env.valuation_date == VAL
    assert (env.rate_curve, env.div_yield, env.basis_yield) == before


# ---------------------------------------------------------------------------------------------------- R10


def test_mixed_maps_refuse_a_relationship_at_a_monday_tenor():
    """A declared wrapper (date knots) next to an undeclared hand-wrapped one (``BusinessTimeMap``) map a Monday
    tenor to different calendar times: a relationship mode refuses them as not on one clock (documented, D3/R10)."""
    from quantark.priceenv.clock_validation import relationship_tenor

    hand_div = TradingClockDividendYield(ContinuousDividendYield(0.01), BusinessTimeMap(TradingClock(CAL, D), VAL,
                                                                                           HORIZON))
    env = _env(FlatRateCurve(0.03), hand_div, div_yield_quoting=None, basis_yield_quoting=None, basis=None)
    assert type(env.div_yield.time_map) is BusinessTimeMap
    assert relationship_tenor(env, BasisRelationshipMode.AUTO_ADJUST_BASIS, env_time(env, WEDNESDAY)) == \
        pytest.approx(_c(WEDNESDAY), rel=1e-12)
    with pytest.raises(ValidationError, match="not on one clock"):
        relationship_tenor(env, BasisRelationshipMode.AUTO_ADJUST_BASIS, env_time(env, MONDAY))


# ---------------------------------------------------------------------------------------------------- AC6 (D5) / R8


def test_trading_clock_vol_surface_refuses_a_date_knot_map():
    """``TradingClockVolSurface`` reads ``to_trading`` / ``initial_slope`` as the variance clock; on a
    ``DateKnotTimeMap`` they mean the opposite (inverse of the carry map, calendar per trading time)."""
    from quantark.param.vol import TradingClockVolSurface

    with pytest.raises(ValidationError, match="DateKnotTimeMap"):
        TradingClockVolSurface(FlatVolSurface(0.2), DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON))
    TradingClockVolSurface(FlatVolSurface(0.2), BusinessTimeMap(TradingClock(CAL, D), VAL, HORIZON))


def test_env_refuses_a_vol_date_knot_map_on_another_holiday_convention():
    vol = CalendarQuotedVolSurface(FlatVolSurface(0.2), DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON,
                                                                        BusinessDayConvention.FOLLOWING))
    with pytest.raises(ValidationError, match="holiday_time_adjustment"):
        _env(FlatRateCurve(0.03), vol=vol)
    env = _env(FlatRateCurve(0.03), vol=vol, holiday=BusinessDayConvention.FOLLOWING)
    assert env.vol_surface is vol


def test_env_refuses_an_undeclared_cash_date_knot_map_on_another_holiday_convention():
    hand = TradingClockDividendYield(ContinuousDividendYield(0.01), DateKnotTimeMap(
        TradingClock(CAL, D), VAL, HORIZON, BusinessDayConvention.FOLLOWING))
    with pytest.raises(ValidationError, match="holiday_time_adjustment"):
        _env(FlatRateCurve(0.03), hand, div_yield_quoting=None)


def test_calendar_quoted_vol_on_a_calendar_day_env_refuses_at_construction_and_re_anchor():
    vol = CalendarQuotedVolSurface(FlatVolSurface(0.2), DateKnotTimeMap(TradingClock(CAL, D), VAL, HORIZON))
    for convention in (DayCountConvention.CALENDAR_DAYS, DayCountConvention.ACT_365):
        with pytest.raises(ValidationError, match="BUSINESS_DAYS"):
            PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=VAL, spot_quote=SpotQuote(SPOT),
                               vol_surface=vol, day_count_convention=convention, calendar=CAL)
    env = PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=VAL, spot_quote=SpotQuote(SPOT),
                             vol_surface=FlatVolSurface(0.2), calendar=CAL)
    env.vol_surface = vol            # a post-construction assignment is not checked (documented) ...
    with pytest.raises(ValidationError, match="BUSINESS_DAYS"):
        env.valuation_date = WEDNESDAY  # ... but the next re-anchor refuses it, leaving the env as it was
    assert env.valuation_date == VAL and env.vol_surface is vol
