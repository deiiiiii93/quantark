"""Spec §7.4: the same CNY-straddling product priced on both axes agrees.

Calendar axis: ACT/365 times of real trading dates + TradingClockVolSurface.
Trading axis:  n_td/244 times of the SAME dates + native trading surface +
               TradingClockRateCurve / TradingClockDividendYield.
Both integrate the same (Dtau_cal, Dn_td/244) per date-step, so prices agree
to discretization tolerance. The European leg is exact (analytical); the
snowball legs are the engine gates.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
from dateutil.relativedelta import relativedelta

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import ObservationType, OptionType

ANCHOR = datetime(2026, 1, 15)      # ~1 month before CNY 2026
R, Q, SIGMA_TD, SPOT = 0.02, 0.01, 0.20, 100.0
D = 244


def _calendar():
    return create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def _next_trading_day(cal, d):
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


def _dates_and_times(cal, months=6):
    """Monthly KO dates snapped to next trading day; both time coordinates."""
    ko_dates = []
    for k in range(1, months + 1):
        ko_dates.append(_next_trading_day(cal, ANCHOR + relativedelta(months=k)))
    ko_dates = sorted(set(ko_dates))
    maturity_date = ko_dates[-1]
    ki_dates = []
    d = ANCHOR + timedelta(days=1)
    while d <= maturity_date:
        if cal.is_business_day(d):
            ki_dates.append(d)
        d += timedelta(days=1)

    def t_cal(d_):
        return (d_ - ANCHOR).days / 365.0

    def t_td(d_):
        n = cal.count_business_days(
            ANCHOR, d_, include_start=False, include_end=True
        )
        return n / float(D)

    return ko_dates, ki_dates, t_cal, t_td


def _env_calendar_axis(cal, horizon_days=400):
    m = BusinessTimeMap(
        TradingClock(cal, D), ANCHOR, ANCHOR + timedelta(days=horizon_days)
    )
    return PricingEnvironment(
        rate_curve=FlatRateCurve(R),
        valuation_date=ANCHOR,
        spot_quote=SpotQuote(SPOT),
        vol_surface=TradingClockVolSurface(FlatVolSurface(SIGMA_TD), m),
        div_yield=ContinuousDividendYield(Q),
    ), m


def _env_trading_axis(cal, horizon_days=400):
    m = BusinessTimeMap(
        TradingClock(cal, D), ANCHOR, ANCHOR + timedelta(days=horizon_days)
    )
    return PricingEnvironment(
        rate_curve=TradingClockRateCurve(FlatRateCurve(R), m),
        valuation_date=ANCHOR,
        spot_quote=SpotQuote(SPOT),
        vol_surface=FlatVolSurface(SIGMA_TD),
        div_yield=TradingClockDividendYield(ContinuousDividendYield(Q), m),
    ), m


def _snowball(times_of, ko_dates, ki_dates):
    """Zero-coupon snowball: every cashflow is date-anchored.

    ko_rate accrues as rate * elapsed ENGINE time, so a nonzero coupon is a
    different contract on each axis (0.15*t_td != 0.15*t_cal at the same
    date) — the coupon's day-count basis is CONTRACT data, not an engine
    property. With ko_rate=0 the payoffs (principal at KO, put at maturity
    after KI) depend only on the dates, which is exactly what axis
    equivalence is entitled to test.
    """
    ko_times = [times_of(d) for d in ko_dates]
    ki_times = [times_of(d) for d in ki_dates]
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=ko_times,
            ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=ki_times,
            ki_continuous=False,
        ),
        contract_multiplier=1.0,
        maturity=ko_times[-1],
        is_reverse=False,
    )


def test_european_axis_equivalence_exact():
    """Analytical BS both ways — agreement to 1e-12 relative (no grids)."""
    cal = _calendar()
    ko_dates, _ki, t_cal, t_td = _dates_and_times(cal)
    expiry = ko_dates[-1]
    env_a, _ = _env_calendar_axis(cal)
    env_b, _ = _env_trading_axis(cal)
    opt_a = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=t_cal(expiry)
    )
    opt_b = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=t_td(expiry)
    )
    engine = BlackScholesEngine()
    price_a = engine.price(opt_a, env_a)
    price_b = engine.price(opt_b, env_b)
    assert price_a == pytest.approx(price_b, rel=1e-12)


def test_snowball_axis_equivalence_pde_quad_mc():
    """Each engine family prices the same term sheet both ways.

    Tolerances: PDE and QUAD 1e-3 relative (the two axes discretize the same
    dates differently); MC 1e-2 relative at 50k paths."""
    cal = _calendar()
    ko_dates, ki_dates, t_cal, t_td = _dates_and_times(cal)
    env_a, _ = _env_calendar_axis(cal)
    env_b, _ = _env_trading_axis(cal)
    product_a = _snowball(t_cal, ko_dates, ki_dates)
    product_b = _snowball(t_td, ko_dates, ki_dates)

    pde_a = float(SnowballPDESolver(params=PDEParams(accuracy="fast")).price(product_a, env_a))
    pde_b = float(SnowballPDESolver(params=PDEParams(accuracy="fast")).price(product_b, env_b))
    assert pde_a == pytest.approx(pde_b, rel=1e-3)

    quad_a = float(SnowballQuadEngine(params=QuadParams(grid_points=301)).price(product_a, env_a))
    quad_b = float(SnowballQuadEngine(params=QuadParams(grid_points=301)).price(product_b, env_b))
    assert quad_a == pytest.approx(quad_b, rel=1e-3)

    mc_a = float(
        SnowballMCEngine(params=MCParams(num_paths=50_000, time_steps=252, seed=42)).price(product_a, env_a)
    )
    mc_b = float(
        SnowballMCEngine(params=MCParams(num_paths=50_000, time_steps=252, seed=42)).price(product_b, env_b)
    )
    assert mc_a == pytest.approx(mc_b, rel=1e-2)

    # cross-family sanity on the calendar axis
    assert pde_a == pytest.approx(quad_a, rel=5e-3)
    assert np.isfinite(mc_a)
