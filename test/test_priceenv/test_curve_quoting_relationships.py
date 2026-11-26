"""Basis <-> dividend relationship modes across trading-clock wrappers (0.16.0 trading-clock run D2).

A relationship derives one cash curve from the others at a tenor ``T`` on the environment's axis. On a wrapped
environment the curves are calendar quotes behind TradingClock wrappers: the relationship is evaluated on the
calendar-quoted inners at ``T_cal = to_inner_time(T)`` (every read at the tenor, the source's own included) and the
derived curve is re-exposed on the same clock. A partially wrapped environment has no single clock and is refused.
"""
from datetime import datetime

import pytest

from quantark.dynamicscenario.config import DynamicScenarioConfig
from quantark.dynamicscenario.engine import DynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import ParameterChange
from quantark.param import SpotQuote
from quantark.param.basis import FlatBasisYield, TradingClockBasisYield, ZeroBasis
from quantark.param.basis.basis_yield import (
    TermStructureBasisYield,
    calculate_basis_from_rate_dividend,
    calculate_dividend_from_rate_basis,
)
from quantark.param.div import TradingClockDividendYield
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf import TradingClockRateCurve
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.portfolio import Portfolio
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.clock_validation import relationship_tenor
from quantark.stresstest.scenario.scenario import Stress
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.stresstest.stress.stress_types import BasisDividendRelationshipMode as Mode
from quantark.util.exceptions import ValidationError

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
D0, H = datetime(2026, 2, 13), datetime(2027, 6, 1)
CQ, TQ = CurveQuotingBasis.CALENDAR_YEAR, CurveQuotingBasis.TRADING_YEAR
T = 0.5
#: a sloped rate inner: r(T) differs from r(T_cal), so a read at the wrong tenor shows
RATE = LinearRateCurve(pillars=[(0.25, 0.02), (1.5, 0.04)])
BASIS_TS = TermStructureBasisYield(times=[0.25, 1.5], yields=[0.005, 0.015])
STEP = 0.001


def declared(rate=RATE, div=ContinuousDividendYield(0.02), basis=FlatBasisYield(0.01), **kw):
    base = dict(rate_curve=rate, div_yield=div, basis_yield=basis, valuation_date=D0, spot_quote=SpotQuote(100.0),
                vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS,
                bus_days_in_year=244, calendar=CAL, rate_curve_quoting=CQ, div_yield_quoting=CQ,
                basis_yield_quoting=CQ, trading_clock_horizon=H)
    base.update(kw)
    return PricingEnvironment(**base)


def hand_wrapped(basis):
    m = BusinessTimeMap(TradingClock(CAL, 244), D0, H)
    return PricingEnvironment(rate_curve=TradingClockRateCurve(RATE, m), valuation_date=D0,
                              div_yield=TradingClockDividendYield(ContinuousDividendYield(0.02), m), basis_yield=basis,
                              spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
                              day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=244,
                              calendar=CAL)


def stress(parameter, mode):
    return Stress(parameter, StressType.ABSOLUTE, STEP, StressLevel.PORTFOLIO,
                  metadata={"relationship_mode": mode.value, "time_to_maturity": T})


def t_cal(env):
    return env.rate_curve.to_inner_time(T)


def apply_dynamic(env, parameter, mode):
    engine = DynamicScenarioEngine(config=DynamicScenarioConfig(basis_dividend_relationship_mode=mode))
    pf = Portfolio(portfolio_name="p", pricing_environments={"X": env})
    change = ParameterChange(parameter, StressType.ABSOLUTE, STEP, metadata={"time_to_maturity": T})
    engine._apply_parameter_change(env, change, pf, "X")


def test_tenor_maps_onto_the_calendar_axis():
    env = declared()
    assert relationship_tenor(env, Mode.AUTO_ADJUST_BASIS, T) == t_cal(env) != T
    native = PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=D0)
    assert relationship_tenor(native, Mode.AUTO_ADJUST_BASIS, T) == T


@pytest.mark.parametrize("apply", ["stress", "dynamic"])
def test_dividend_drives_basis_on_the_inners(apply):
    env = declared()
    c = t_cal(env)
    if apply == "stress":
        StressApplicator._apply_stress_to_env(env, stress("dividend_yield", Mode.AUTO_ADJUST_BASIS))
    else:
        apply_dynamic(env, "dividend_yield", Mode.AUTO_ADJUST_BASIS)
    assert type(env.basis_yield) is TradingClockBasisYield and type(env.div_yield) is TradingClockDividendYield
    expected = calculate_basis_from_rate_dividend(RATE.get_rate(c), 0.02 + STEP)
    assert env.basis_yield.inner.get_basis_yield(1.0) == pytest.approx(expected, abs=1e-15)
    assert env.basis_yield.time_map.anchor_date == env.valuation_date


@pytest.mark.parametrize("apply", ["stress", "dynamic"])
def test_basis_drives_dividend_on_the_inners(apply):
    env = declared(basis=BASIS_TS)
    c = t_cal(env)
    if apply == "stress":
        StressApplicator._apply_stress_to_env(env, stress("basis", Mode.AUTO_ADJUST_DIVIDEND))
        new_basis = BASIS_TS.get_basis_yield(c) + STEP      # the stress reads the source at T_cal
    else:
        apply_dynamic(env, "basis", Mode.AUTO_ADJUST_DIVIDEND)
        new_basis = BASIS_TS.get_basis_yield(c) + STEP
    assert type(env.div_yield) is TradingClockDividendYield
    expected = calculate_dividend_from_rate_basis(RATE.get_rate(c), new_basis)
    assert env.div_yield.inner.get_yield(1.0) == pytest.approx(expected, abs=1e-15)


def test_synchronized_reads_the_target_at_t_cal():
    env = declared(basis=BASIS_TS)
    c = t_cal(env)
    StressApplicator._apply_stress_to_env(env, stress("dividend_yield", Mode.SYNCHRONIZED))
    assert type(env.basis_yield) is TradingClockBasisYield
    shift = (BASIS_TS.get_basis_yield(c) + STEP) - BASIS_TS.get_basis_yield(c)
    assert list(env.basis_yield.inner.yields) == pytest.approx([y + shift for y in BASIS_TS.yields], abs=1e-15)


@pytest.mark.parametrize("basis", [None, ZeroBasis()], ids=["none", "zero"])
def test_hand_wrapped_env_wraps_the_derived_curve_on_the_sources_map(basis):
    env = hand_wrapped(basis)
    c = t_cal(env)
    StressApplicator._apply_stress_to_env(env, stress("dividend_yield", Mode.AUTO_ADJUST_BASIS))
    assert type(env.basis_yield) is TradingClockBasisYield
    assert env.basis_yield.time_map.anchor_date == env.rate_curve.time_map.anchor_date
    expected = calculate_basis_from_rate_dividend(RATE.get_rate(c), 0.02 + STEP)
    assert env.basis_yield.inner.get_basis_yield(1.0) == pytest.approx(expected, abs=1e-15)


def test_partially_wrapped_env_still_refuses():
    env = declared(div=ContinuousDividendYield(0.02), div_yield_quoting=TQ)
    assert type(env.div_yield) is ContinuousDividendYield and type(env.rate_curve) is TradingClockRateCurve
    before = {f: getattr(env, f) for f in ("rate_curve", "div_yield", "basis_yield")}
    with pytest.raises(ValidationError, match="relationship"):
        StressApplicator._apply_stress_to_env(env, stress("dividend_yield", Mode.AUTO_ADJUST_BASIS))
    for f, obj in before.items():
        assert getattr(env, f) is obj


def test_trading_year_target_slot_refuses():
    env = declared(basis=None, basis_yield_quoting=TQ)
    with pytest.raises(ValidationError, match="relationship"):
        StressApplicator._apply_stress_to_env(env, stress("dividend_yield", Mode.AUTO_ADJUST_BASIS))


def test_independent_basis_stress_reads_the_source_at_t_cal():
    env = declared(basis=BASIS_TS)
    c = t_cal(env)
    s = Stress("basis", StressType.VALUE, 0.02, StressLevel.PORTFOLIO, metadata={"time_to_maturity": T})
    StressApplicator._apply_stress_to_env(env, s)
    # a VALUE stress on a term structure moves every node by the same amount; T_cal only picks the reported node
    assert type(env.basis_yield) is TradingClockBasisYield
    assert BASIS_TS.get_basis_yield(c) != BASIS_TS.get_basis_yield(T)
