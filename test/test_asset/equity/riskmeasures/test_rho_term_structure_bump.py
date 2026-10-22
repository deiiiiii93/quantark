"""Scalar rho bumps must be PARALLEL shifts of the rate curve, not a flat
replacement at r(T)+bump (patch spec 2026-09-03 §14 finding, confirmed
2026-09-05).

`FlatRateCurve(r(T) + bump)` on a term curve differs from the base by
r(T) - r(t) + bump at every t < T: the finite difference then measures a
curve RESHAPING of hundreds of bp divided by one bp. On a flat curve the
replacement is exactly a parallel shift, so the flat path must stay bitwise.
"""
import math
from copy import deepcopy
from datetime import datetime

import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import FlatRateCurve, SpotQuote
from quantark.param.rrf.rate_curve import LinearRateCurve, ParallelShiftRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.portfolio.equity.position import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

from dcn_fixtures import DCN_A, FLAT, make_dcn, term_env

BUMP = 1e-4
STEEP = [(0.5, 0.010), (1.0, 0.030), (2.0, 0.050)]
SAMPLE_T = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _steep_env():
    env = term_env(**FLAT)
    env.rate_curve = LinearRateCurve(STEEP)
    return env


def _engine():
    return DCNMCEngine(num_paths=2 ** 11, seed=42)


def _parallel(env, shift):
    up = deepcopy(env)
    up.rate_curve = ParallelShiftRateCurve(env.rate_curve, shift)
    return up


# --- the helper -------------------------------------------------------------

def test_rate_bumped_env_flat_curve_stays_flat_and_bitwise():
    from quantark.asset.equity.riskmeasures.greeks import bump_envs

    env = term_env(**FLAT)
    env.rate_curve = FlatRateCurve(0.0356)
    bumped = bump_envs.build_rate_bumped_env(env, BUMP, direction=1.0)
    assert type(bumped.rate_curve) is FlatRateCurve
    assert bumped.rate_curve.rate == 0.0356 + BUMP          # bitwise, not approx
    down = bump_envs.build_rate_bumped_env(env, BUMP, direction=-1.0)
    assert down.rate_curve.rate == 0.0356 - BUMP
    assert env.rate_curve.rate == 0.0356                    # input not mutated


def test_rate_bumped_env_term_curve_is_a_parallel_shift():
    from quantark.asset.equity.riskmeasures.greeks import bump_envs

    env = _steep_env()
    bumped = bump_envs.build_rate_bumped_env(env, BUMP, direction=1.0)
    for t in SAMPLE_T:
        assert bumped.rate_curve.get_rate(t) == pytest.approx(
            env.rate_curve.get_rate(t) + BUMP, abs=1e-12
        )
        assert bumped.rate_curve.get_discount_factor(t) == pytest.approx(
            env.rate_curve.get_discount_factor(t) * math.exp(-BUMP * t), rel=1e-14
        )


def test_rate_bumped_env_trading_clock_shifts_the_calendar_quoted_inner():
    """r/q live on the calendar clock: one bp per CALENDAR year, re-exposed
    through the same time map (not one bp per trading year)."""
    from quantark.asset.equity.riskmeasures.greeks import bump_envs

    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2028, 2, 9))
    wrapped = TradingClockRateCurve(LinearRateCurve(STEEP), m)
    env = PricingEnvironment(
        rate_curve=wrapped, valuation_date=datetime(2026, 2, 9),
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=244, calendar=cal,
    )
    bumped = bump_envs.build_rate_bumped_env(env, BUMP, direction=1.0)
    assert type(bumped.rate_curve) is TradingClockRateCurve
    assert bumped.rate_curve.time_map.anchor_date == m.anchor_date
    assert bumped.rate_curve.time_map.clock.days_per_year == 244
    for t_td in (0.1, 0.5, 1.0, 1.7):
        t_cal = float(m.to_calendar(t_td))
        assert bumped.rate_curve.get_discount_factor(t_td) == pytest.approx(
            wrapped.get_discount_factor(t_td) * math.exp(-BUMP * t_cal), rel=1e-14
        )


# --- the four call sites ----------------------------------------------------

def test_numerical_rho_on_term_curve_equals_parallel_shift_rho():
    env = _steep_env()
    product = make_dcn(DCN_A)
    engine = _engine()
    ctx = engine.create_bump_context(product, env)
    base = ctx.price(product, env)
    expected = (ctx.price(product, _parallel(env, BUMP)) - base) * (0.01 / BUMP)
    rho = GreeksCalculator().calculate_numerical_rho(product, env, engine, rate_bump=BUMP)
    assert rho == pytest.approx(expected, rel=1e-9)


def test_numerical_rho_on_flat_curve_is_bitwise_unchanged():
    """The flat path is the legacy formula: FlatRateCurve(r + bump), same floats."""
    env = term_env(**FLAT)
    env.rate_curve = FlatRateCurve(FLAT["r"])
    product = make_dcn(DCN_A)
    engine = _engine()
    ctx = engine.create_bump_context(product, env)
    base = ctx.price(product, env)
    legacy = deepcopy(env)
    legacy.rate_curve = FlatRateCurve(env.get_rate(product.get_maturity(env)) + BUMP)
    expected = (ctx.price(product, legacy) - base) * (0.01 / BUMP)
    rho = GreeksCalculator().calculate_numerical_rho(product, env, engine, rate_bump=BUMP)
    assert rho == expected


def _position(env):
    return EquityPosition(
        product=make_dcn(DCN_A), quantity=1.0, entry_price=0.0, underlying="TEST",
        engine=_engine(), entry_timestamp=env.valuation_date, cash_legs=[],
    )


def test_position_trade_greeks_central_rho_on_term_curve_is_parallel():
    env = _steep_env()
    pos = _position(env)
    b = pos.engine.params.get_effective_bump_config().rate_bump
    expected = (
        pos.get_trade_value(_parallel(env, b)) - pos.get_trade_value(_parallel(env, -b))
    ) / (2.0 * b)
    assert pos.get_trade_greeks(env, GreeksCalculator())["rho"] == pytest.approx(expected, rel=1e-9)


def test_position_trade_risk_rho_on_term_curve_is_parallel():
    env = _steep_env()
    pos = _position(env)
    gc = GreeksCalculator()
    b = gc._bump_config.rate_bump
    ctx = pos.engine.create_bump_context(pos.product, env)
    base = ctx.price(pos.product, env)
    expected = (ctx.price(pos.product, _parallel(env, b)) - base) * (0.01 / b)
    risk = pos.get_trade_risk(env, gc, ["rho"])
    assert risk.product["rho"] == pytest.approx(expected, rel=1e-9)


def test_execution_rate_up_cell_on_term_curve_is_parallel():
    from quantark.execution.greeks import TradeState, apply_greek_bump

    env = _steep_env()
    gc = GreeksCalculator()
    state = TradeState(
        product=make_dcn(DCN_A), pricing_env=env, cash_legs=(), engine=_engine(),
        streams=None, quantity=1.0, greeks_params=gc.params,
    )
    bumped = apply_greek_bump("rate_up", state, gc).pricing_env
    for t in SAMPLE_T:
        assert bumped.get_rate(t) == pytest.approx(
            env.get_rate(t) + gc._bump_config.rate_bump, abs=1e-12
        )
    assert state.pricing_env is env                          # never in place
