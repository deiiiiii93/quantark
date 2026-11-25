"""AC10b: scenario and stress shifts on clock-wrapped cash curves (0.15.0 spec D16).

A shift is per calendar year (like ``parallel_shifted``): it moves the calendar-quoted inner curve, and the result
is re-exposed on the trading clock it came from. Never skipped, never replaced by a native curve built from a
trading-axis reading.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.report.autocallable_risk_report import _clone_env, _shift_dividend_yield
from quantark.dynamicscenario.engine import DynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import ParameterChange
from quantark.param import SpotQuote
from quantark.param.basis import FlatBasisYield, TradingClockBasisYield
from quantark.param.div import TradingClockDividendYield
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf import TradingClockRateCurve
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.portfolio import Portfolio
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.clock_validation import calendar_quoted_view
from quantark.stresstest.scenario.scenario import Stress
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.exceptions import ValidationError

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2028))
D0, H = datetime(2026, 2, 13), datetime(2027, 6, 1)
CQ = CurveQuotingBasis.CALENDAR_YEAR
US = (1 / 244.0, 0.3, 1.0)


def _env(rate=None, div=None, basis=None, **kw):
    base = dict(rate_curve=rate or FlatRateCurve(0.03), div_yield=div or ContinuousDividendYield(0.02),
                basis_yield=basis or FlatBasisYield(0.01), valuation_date=D0, spot_quote=SpotQuote(100.0),
                vol_surface=FlatVolSurface(0.2), day_count_convention=DayCountConvention.BUSINESS_DAYS,
                bus_days_in_year=244, calendar=CAL, rate_curve_quoting=CQ, div_yield_quoting=CQ,
                basis_yield_quoting=CQ, trading_clock_horizon=H)
    base.update(kw)
    return PricingEnvironment(**base)


def _same_map(after, before, cls):
    assert type(after) is cls and after.time_map is before.time_map


# --- calendar_quoted_view ----------------------------------------------------------------------------------------

def test_view_exposes_inners_and_restores_untouched_wrappers():
    env = _env()
    before = {f: getattr(env, f) for f in ("rate_curve", "div_yield", "basis_yield")}
    with calendar_quoted_view(env):
        assert env.rate_curve is before["rate_curve"].inner
        assert env.div_yield is before["div_yield"].inner
    for f, obj in before.items():
        assert getattr(env, f) is obj


def test_view_restores_everything_on_error():
    env = _env()
    before = env.rate_curve
    with pytest.raises(RuntimeError):
        with calendar_quoted_view(env):
            env.rate_curve = FlatRateCurve(0.05)
            raise RuntimeError("boom")
    assert env.rate_curve is before


def test_view_wraps_a_curve_created_where_there_was_none():
    env = _env(div=None)
    object.__setattr__(env, "div_yield", None)
    with calendar_quoted_view(env):
        env.div_yield = ContinuousDividendYield(0.04)
    assert type(env.div_yield) is TradingClockDividendYield and env.div_yield.inner.div_yield == 0.04


def test_view_leaves_natives_alone_on_undeclared_envs():
    rate = FlatRateCurve(0.0)
    env = PricingEnvironment(rate_curve=rate, valuation_date=D0, day_count_convention=DayCountConvention.BUSINESS_DAYS,
                             calendar=CAL, bus_days_in_year=244)
    with calendar_quoted_view(env):
        env.rate_curve = FlatRateCurve(0.01)
    assert type(env.rate_curve) is FlatRateCurve


# --- stress applicator -----------------------------------------------------------------------------------------------

def test_rate_stress_shifts_the_calendar_inner():
    env = _env()
    before = env.rate_curve
    StressApplicator._apply_stress_to_env(env, Stress("rate", StressType.ABSOLUTE, 0.01, StressLevel.PORTFOLIO))
    _same_map(env.rate_curve, before, TradingClockRateCurve)
    assert env.rate_curve.inner == FlatRateCurve(0.04)


def test_pillar_rate_stress_shifts_the_calendar_inner():
    env = _env(rate=LinearRateCurve(pillars=[(0.5, 0.02), (2.0, 0.03)]))
    before = env.rate_curve
    StressApplicator._apply_stress_to_env(env, Stress("rate", StressType.ABSOLUTE, 0.01, StressLevel.PORTFOLIO))
    _same_map(env.rate_curve, before, TradingClockRateCurve)
    assert [r for _, r in env.rate_curve.inner.pillars] == pytest.approx([0.03, 0.04], abs=1e-15)


def test_dividend_and_basis_stress_shift_the_calendar_inners():
    env = _env()
    div0, basis0 = env.div_yield, env.basis_yield
    StressApplicator._apply_stress_to_env(env, Stress("dividend_yield", StressType.ABSOLUTE, 0.01,
                                                      StressLevel.PORTFOLIO))
    _same_map(env.div_yield, div0, TradingClockDividendYield)
    assert env.div_yield.inner.get_yield(1.0) == pytest.approx(0.03, abs=1e-15)
    StressApplicator._apply_stress_to_env(env, Stress("basis", StressType.ABSOLUTE, 0.005, StressLevel.PORTFOLIO))
    _same_map(env.basis_yield, basis0, TradingClockBasisYield)
    assert env.basis_yield.inner.get_basis_yield(1.0) == pytest.approx(0.015, abs=1e-15)


def test_native_stress_is_unchanged():
    native = PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=D0)
    StressApplicator._apply_stress_to_env(native, Stress("rate", StressType.ABSOLUTE, 0.01, StressLevel.PORTFOLIO))
    assert native.rate_curve == FlatRateCurve(0.04)


# --- dynamic scenario --------------------------------------------------------------------------------------------

def _apply(env, change):
    pf = Portfolio(portfolio_name="p", pricing_environments={"X": env})
    DynamicScenarioEngine()._apply_parameter_change(env, change, pf, "X")


def test_scenario_rate_and_div_changes_shift_the_inners():
    env = _env()
    rate0, div0 = env.rate_curve, env.div_yield
    _apply(env, ParameterChange("rate", StressType.ABSOLUTE, 0.01))
    _same_map(env.rate_curve, rate0, TradingClockRateCurve)
    assert env.rate_curve.inner == FlatRateCurve(0.04)
    _apply(env, ParameterChange("dividend_yield", StressType.ABSOLUTE, 0.01))
    _same_map(env.div_yield, div0, TradingClockDividendYield)
    assert env.div_yield.inner == ContinuousDividendYield(0.03)


def test_scenario_change_on_unshiftable_wrapped_inner_raises():
    env = _env(rate=LinearRateCurve(pillars=[(0.5, 0.02), (2.0, 0.03)]))
    before = env.rate_curve
    with pytest.raises(ValidationError, match="LinearRateCurve"):
        _apply(env, ParameterChange("rate", StressType.ABSOLUTE, 0.01))
    assert env.rate_curve is before


# --- risk report overrides -------------------------------------------------------------------------------------

def test_report_dividend_shift_moves_the_inner():
    env = _env()
    shifted = _shift_dividend_yield(env.div_yield, 0.01)
    _same_map(shifted, env.div_yield, TradingClockDividendYield)
    for u in US:
        assert shifted.inner.get_yield(u) == pytest.approx(0.03, abs=1e-15)


def test_report_q_override_is_wrapped_on_a_declared_env():
    clone = _clone_env(_env(), q=0.05)
    assert type(clone.div_yield) is TradingClockDividendYield and clone.div_yield.inner.div_yield == 0.05
    assert clone.div_yield.clock.calendar is clone.calendar
