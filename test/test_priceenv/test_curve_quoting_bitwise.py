"""AC8: an environment that wraps its own calendar-quoted curves prices bitwise like the same curves wrapped by hand.

``CALENDAR_YEAR`` + ``trading_clock_horizon`` on a ``BUSINESS_DAYS`` environment must be exactly the hand recipe of
``docs/trading-clock.md`` (``TradingClockRateCurve`` / ``TradingClockDividendYield`` over a ``DateKnotTimeMap``
anchored at the valuation date with the environment's holiday convention -- run 8 D2/D6 moved the env's map from the
0.14.0 ``BusinessTimeMap``, which lagged after non-trading days): one declaration replaces the hand wrapping, never a
different model. The gate is ``==`` on every engine family, not a tolerance.
"""
from datetime import datetime, timedelta

import pytest
from dateutil.relativedelta import relativedelta

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import SpotQuote
from quantark.param.basis import FlatBasisYield, TradingClockBasisYield
from quantark.param.div import TradingClockDividendYield
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf import TradingClockRateCurve
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import DateKnotTimeMap, TradingClock
from quantark.util.enum import ObservationType, OptionType

from dcn_fixtures import SAMPLE_DCN_A, SSE, make_dcn

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
ANCHOR = datetime(2026, 1, 15)
HORIZON = ANCHOR + timedelta(days=400)
R, Q, B, SIGMA, SPOT, D = 0.02, 0.01, 0.013, 0.20, 100.0, 244
CAL_Q = CurveQuotingBasis.CALENDAR_YEAR


def _pair(calendar=CAL, anchor=ANCHOR, horizon=HORIZON, spot=SPOT):
    """(auto, hand): the same market, wrapped by the environment and by hand."""
    common = dict(valuation_date=anchor, spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(SIGMA),
                  day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=D, calendar=calendar)
    auto = PricingEnvironment(rate_curve=FlatRateCurve(R), div_yield=ContinuousDividendYield(Q),
                              basis_yield=FlatBasisYield(B), rate_curve_quoting=CAL_Q, div_yield_quoting=CAL_Q,
                              basis_yield_quoting=CAL_Q, trading_clock_horizon=horizon, **common)
    m = DateKnotTimeMap(TradingClock(calendar, D), anchor, horizon)   # run 8 D6: the env's recipe
    hand = PricingEnvironment(rate_curve=TradingClockRateCurve(FlatRateCurve(R), m),
                              div_yield=TradingClockDividendYield(ContinuousDividendYield(Q), m),
                              basis_yield=TradingClockBasisYield(FlatBasisYield(B), m), **common)
    return auto, hand


def _trading_times(months=6):
    ko, d = [], ANCHOR
    for k in range(1, months + 1):
        d = ANCHOR + relativedelta(months=k)
        while not CAL.is_business_day(d):
            d += timedelta(days=1)
        ko.append(CAL.count_business_days(ANCHOR, d, include_start=False, include_end=True) / float(D))
    ki = [n / float(D) for n in range(1, int(round(ko[-1] * D)) + 1)]
    return ko, ki


def _snowball():
    ko, ki = _trading_times()
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, maturity=ko[-1], is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=ko,
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=ki,
            ki_continuous=False,
        ),
    )


def test_curve_reads_are_bitwise():
    auto, hand = _pair()
    for u in (1 / 244.0, 17 / 244.0, 0.5, 1.0):
        assert auto.get_discount_factor(u) == hand.get_discount_factor(u)
        assert auto.get_div_yield(u) == hand.get_div_yield(u)
        assert auto.get_basis_yield(u) == hand.get_basis_yield(u)
        assert auto.get_rate(u) == hand.get_rate(u)


def test_analytical_european_bitwise():
    auto, hand = _pair()
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=0.5)
    engine = BlackScholesEngine()
    assert engine.price(option, auto) == engine.price(option, hand)


@pytest.mark.parametrize("engine", [
    pytest.param(lambda: SnowballPDESolver(params=PDEParams(accuracy="fast")), id="pde"),
    pytest.param(lambda: SnowballQuadEngine(params=QuadParams(grid_points=301)), id="quad"),
    pytest.param(lambda: SnowballMCEngine(params=MCParams(num_paths=20_000, time_steps=252, seed=42)), id="mc"),
])
def test_snowball_bitwise(engine):
    auto, hand = _pair()
    product = _snowball()
    assert float(engine().price(product, auto)) == float(engine().price(product, hand))


@pytest.mark.parametrize("engine", [
    pytest.param(lambda: DCNPDEEngine(params=PDEParams(accuracy="fast")), id="pde"),
    pytest.param(lambda: DCNMCEngine(num_paths=4096, seed=7), id="mc"),
])
def test_dcn_on_env_axis_bitwise(engine):
    product = make_dcn(SAMPLE_DCN_A)
    val = product.schedule.valuation_date
    auto, hand = _pair(calendar=SSE, anchor=val, horizon=datetime(2026, 3, 1), spot=product.initial_price)
    assert float(engine().price(product, auto)) == float(engine().price(product, hand))
