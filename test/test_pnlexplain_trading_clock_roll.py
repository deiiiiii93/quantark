"""A float maturity ages on the environment's clock, not always on calendar days.

The roll rule was fixed at calendar_days/365, so a contract whose float maturity is quoted
in trading time was rejected outright on a BUSINESS_DAYS environment. It now takes the
decrement from the environment: trading days elapsed over bus_days_in_year. The calendar
default is untouched, and the check is no looser: a contract rolled on the wrong clock
still fails, against the right number.

A Friday-to-Monday step separates the two rules cleanly: three calendar days, one trading day.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig, ValuationSnapshot, explain
from quantark.pnlexplain.equity.lifecycle import float_maturity_decrement
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

FRI, MON = datetime(2026, 9, 4), datetime(2026, 9, 7)     # 3 calendar days, 1 trading day
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
BUS_YEAR = 244
M0 = 1.0
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _env(date, spot=100.0, business=True, calendar=CAL):
    kw = dict(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.22),
              rate_curve=FlatRateCurve(rate=0.02),
              div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=date)
    if business:
        kw.update(day_count_convention=DayCountConvention.BUSINESS_DAYS,
                  bus_days_in_year=BUS_YEAR, calendar=calendar)
    elif calendar is not None:
        kw.update(calendar=calendar, bus_days_in_year=BUS_YEAR)
    return PricingEnvironment(**kw)


def _call(maturity):
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=maturity)


def _explain(m1, business=True, calendar=CAL):
    eng = BlackScholesEngine()
    return explain(
        ValuationSnapshot(_call(M0), eng, _env(FRI, business=business, calendar=calendar),
                          date=FRI),
        ValuationSnapshot(_call(m1), eng, _env(MON, spot=101.0, business=business,
                                               calendar=calendar), date=MON),
        config=WF)


def test_the_decrement_comes_from_the_environments_clock():
    """Three calendar days, but only one trading day of ageing."""
    assert float_maturity_decrement(_env(FRI), FRI, 3) == pytest.approx(1 / BUS_YEAR)
    assert float_maturity_decrement(_env(FRI, business=False), FRI, 3) is None


def test_a_trading_time_contract_is_accepted_on_a_business_days_environment():
    """This raised outright before: the days/365 rule could not describe the contract."""
    res = _explain(M0 - 1 / BUS_YEAR)
    assert res.total_pnl == pytest.approx(res.total_pnl)      # finite, and it got this far
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-9)


def test_the_calendar_roll_is_rejected_on_a_trading_clock():
    """No looser than before: the wrong clock still fails, against the right number."""
    with pytest.raises(ValidationError, match="environment's own clock"):
        _explain(M0 - 3 / 365)


def test_the_calendar_default_is_untouched():
    """A calendar environment keeps days/365 exactly as before."""
    res = _explain(M0 - 3 / 365, business=False)
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValidationError, match="calendar_days/365"):
        _explain(M0 - 1 / BUS_YEAR, business=False)


def test_a_trading_clock_without_a_calendar_says_so():
    """Trading time cannot be measured without a calendar; name that, do not guess 365."""
    env_no_cal = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=FRI,
        day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=BUS_YEAR)
    eng = BlackScholesEngine()
    with pytest.raises(ValidationError, match="calendar"):
        explain(ValuationSnapshot(_call(M0), eng, env_no_cal, date=FRI),
                ValuationSnapshot(_call(M0 - 1 / BUS_YEAR), eng,
                                  PricingEnvironment(
                                      spot_quote=SpotQuote(spot=101.0),
                                      vol_surface=FlatVolSurface(volatility=0.22),
                                      rate_curve=FlatRateCurve(rate=0.02),
                                      div_yield=ContinuousDividendYield(div_yield=0.01),
                                      valuation_date=MON,
                                      day_count_convention=DayCountConvention.BUSINESS_DAYS,
                                      bus_days_in_year=BUS_YEAR),
                                  date=MON),
                config=WF)
