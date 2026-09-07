"""Clock-wrapped environments: waterfall exact on both axes, Taylor fails closed (patch spec §8)."""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.pnlexplain import ExplainMethod, Factor, PnLExplainConfig, ValuationSnapshot, explain, value
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

D = 244
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
EXPIRY = datetime(2026, 12, 15)
HORIZON = datetime(2027, 3, 1)
ENG = BlackScholesEngine()
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _map(anchor):
    return BusinessTimeMap(TradingClock(CAL, D), anchor, HORIZON)


def _cal_env(date, spot=100.0, sigma_td=0.20, r=0.02, q=0.01, anchor=None):
    m = _map(anchor or date)
    return PricingEnvironment(rate_curve=FlatRateCurve(r), valuation_date=date, spot_quote=SpotQuote(spot),
                              vol_surface=TradingClockVolSurface(FlatVolSurface(sigma_td), m),
                              div_yield=ContinuousDividendYield(q))


def _td_env(date, spot=100.0, sigma_td=0.20, r=0.02, q=0.01):
    m = _map(date)
    return PricingEnvironment(rate_curve=TradingClockRateCurve(FlatRateCurve(r), m), valuation_date=date,
                              spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(sigma_td),
                              div_yield=TradingClockDividendYield(ContinuousDividendYield(q), m),
                              day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D, calendar=CAL)


CALL = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=EXPIRY)
STEPS = [(datetime(2026, 3, 10), datetime(2026, 3, 11)),          # plain step
         (datetime(2026, 9, 30), datetime(2026, 10, 9))]          # National Day week: 9 calendar days, 2 trading days


def test_fixture_calendar_facts():
    """Pin the calendar the steps rely on (CHINA_SSE: Oct 1-7 closed, Oct 8-9 open)."""
    assert CAL.is_business_day(datetime(2026, 3, 10)) and CAL.is_business_day(datetime(2026, 3, 11))
    assert CAL.count_business_days(datetime(2026, 9, 30), datetime(2026, 10, 9),
                                   include_start=False, include_end=True) == 2
    assert not any(CAL.is_business_day(datetime(2026, 10, d)) for d in range(1, 8))


@pytest.mark.parametrize("t0,t1", STEPS)
def test_calendar_axis_waterfall_reconciles_and_same_inner_is_not_a_vol_change(t0, t1):
    e0, e1 = _cal_env(t0), _cal_env(t1, spot=101.0)               # fresh wrapper + fresh map, same inner sigma
    r = explain(ValuationSnapshot(CALL, ENG, e0, date=t0), ValuationSnapshot(CALL, ENG, e1, date=t1), config=WF)
    assert Factor.VOL not in r.moves.changed and Factor.RATE not in r.moves.changed
    assert Factor.SPOT in r.moves.changed
    assert r.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-12)
    # TIME = same inner, same clock, map re-anchored at t1
    re_anchored = _cal_env(t1, anchor=t1)
    direct = (value(ValuationSnapshot(CALL, ENG, re_anchored, date=t1)).total
              - value(ValuationSnapshot(CALL, ENG, e0, date=t0)).total)
    assert r.metadata["time_pure"] == pytest.approx(direct, abs=1e-12)
    shap = explain(ValuationSnapshot(CALL, ENG, e0, date=t0), ValuationSnapshot(CALL, ENG, e1, date=t1),
                   config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,), interaction="shapley"))
    assert shap.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-12)
    assert shap.total_pnl == pytest.approx(r.total_pnl, abs=1e-12)


def test_axis_equivalence_of_rows():
    t0, t1 = datetime(2026, 3, 10), datetime(2026, 3, 12)
    kw1 = dict(spot=102.0, sigma_td=0.22, r=0.025)
    a = explain(ValuationSnapshot(CALL, ENG, _cal_env(t0), date=t0),
                ValuationSnapshot(CALL, ENG, _cal_env(t1, **kw1), date=t1), config=WF)
    b = explain(ValuationSnapshot(CALL, ENG, _td_env(t0), date=t0),
                ValuationSnapshot(CALL, ENG, _td_env(t1, **kw1), date=t1), config=WF)
    for f in (Factor.TIME, Factor.SPOT, Factor.VOL, Factor.RATE):
        ra = [x for x in a.rows if x.factor is f and x.method is ExplainMethod.WATERFALL][0].pnl
        rb = [x for x in b.rows if x.factor is f and x.method is ExplainMethod.WATERFALL][0].pnl
        assert ra == pytest.approx(rb, abs=1e-9), f
    assert a.total_pnl == pytest.approx(b.total_pnl, abs=1e-9)
    assert Factor.VOL in a.moves.changed and Factor.RATE in b.moves.changed


def test_fail_closed_rules():
    t0, t1 = datetime(2026, 3, 10), datetime(2026, 3, 11)
    ok0, ok1 = _cal_env(t0), _cal_env(t1)
    with pytest.raises(ValidationError, match="anchor"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0),
                ValuationSnapshot(CALL, ENG, _cal_env(t1, anchor=t0), date=t1), config=WF)
    native = PricingEnvironment(rate_curve=FlatRateCurve(0.02), valuation_date=t1, spot_quote=SpotQuote(100.0),
                                vol_surface=FlatVolSurface(0.2), div_yield=ContinuousDividendYield(0.01))
    with pytest.raises(ValidationError, match="clock"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, native, date=t1), config=WF)
    other_clock = PricingEnvironment(
        rate_curve=FlatRateCurve(0.02), valuation_date=t1, spot_quote=SpotQuote(100.0),
        vol_surface=TradingClockVolSurface(FlatVolSurface(0.2), BusinessTimeMap(TradingClock(CAL, 252), t1, HORIZON)),
        div_yield=ContinuousDividendYield(0.01))
    with pytest.raises(ValidationError, match="clock"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, other_clock, date=t1),
                config=WF)
    # A float maturity on a BUSINESS_DAYS environment used to be turned away outright, because
    # the roll rule was fixed at days/365. It now ages on the environment's clock, so a
    # calendar-rolled contract fails against the trading-time number instead of being rejected
    # for existing, and a correctly rolled one goes through.
    floating = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=0.5)
    with pytest.raises(ValidationError, match="environment's own clock"):
        explain(ValuationSnapshot(floating, ENG, _td_env(t0), date=t0),
                ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                                        maturity=0.5 - 1 / 365), ENG, _td_env(t1), date=t1),
                config=WF)
    rolled = explain(
        ValuationSnapshot(floating, ENG, _td_env(t0), date=t0),
        ValuationSnapshot(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                                maturity=0.5 - 1 / D), ENG, _td_env(t1), date=t1),
        config=WF)
    assert rolled.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValidationError, match="Taylor"):
        explain(ValuationSnapshot(CALL, ENG, ok0, date=t0), ValuationSnapshot(CALL, ENG, ok1, date=t1))
