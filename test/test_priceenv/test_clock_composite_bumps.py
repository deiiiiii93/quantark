"""Bump curves composed over a trading-clock wrapper shift the calendar-quoted inner (0.16.0 trading-clock run D5).

A rate bump written ``ParallelShiftRateCurve(env.rate_curve, b)`` on a wrapped curve adds ``b`` per TRADING year
(the composite sits on the trading axis, outside the wrapper), while every other cash shift since 0.15.0 moves the
calendar quote. The sites now shift the inner and re-wrap on the same map.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.localvol_greeks import local_vol_rho_theta
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import SpotQuote
from quantark.param.rrf import ParallelShiftRateCurve, TradingClockRateCurve
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.clock_validation import clock_parallel_shift_rate
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
VAL, HORIZON = datetime(2026, 2, 9), datetime(2027, 6, 9)
CQ = CurveQuotingBasis.CALENDAR_YEAR
CALL = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=datetime(2026, 12, 15))
B = 1e-4


def wrapped_env(rate=FlatRateCurve(0.02)):
    return PricingEnvironment(valuation_date=VAL, spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
                              rate_curve=rate, day_count_convention=DayCountConvention.BUSINESS_DAYS,
                              bus_days_in_year=244, calendar=CAL, rate_curve_quoting=CQ, trading_clock_horizon=HORIZON)


def assert_shifted_inner(curve, base, shift):
    assert type(curve) is TradingClockRateCurve
    assert type(curve.inner) is ParallelShiftRateCurve and curve.inner.shift == shift
    assert curve.inner.base_curve == base.inner
    assert curve.time_map.anchor_date == base.time_map.anchor_date


def test_helper_shifts_the_inner_of_a_wrapper():
    env = wrapped_env()
    assert_shifted_inner(clock_parallel_shift_rate(env.rate_curve, B), env.rate_curve, B)


def test_helper_is_the_old_composite_on_a_native_curve():
    native = FlatRateCurve(0.02)
    out = clock_parallel_shift_rate(native, B)
    assert type(out) is ParallelShiftRateCurve and out.base_curve is native and out.shift == B


def test_local_vol_rho_bumps_the_calendar_quote():
    env = wrapped_env()
    seen = []

    def price(product, pricing_env, surface):
        seen.append(pricing_env.rate_curve)
        return BlackScholesEngine().price(product, pricing_env)

    rho = local_vol_rho_theta(price, CALL, env, None, rate_bump=B, base=0.0)["rho"]
    assert_shifted_inner(seen[0], env.rate_curve, B)
    assert_shifted_inner(seen[1], env.rate_curve, -B)
    up = BlackScholesEngine().price(CALL, wrapped_env(ParallelShiftRateCurve(FlatRateCurve(0.02), B)))
    dn = BlackScholesEngine().price(CALL, wrapped_env(ParallelShiftRateCurve(FlatRateCurve(0.02), -B)))
    assert rho == (up - dn) / (2.0 * B) / 100.0


@pytest.mark.parametrize("solver", ["heston", "slv"])
def test_heston_pde_rho_bumps_the_calendar_quote(solver, monkeypatch):
    import numpy as np

    from quantark.asset.equity.engine.pde import heston_pde_solver, heston_slv_pde_solver
    from quantark.asset.equity.engine.pde import HestonPDESolver, HestonSLVPDESolver
    from quantark.volmodels.heston import HestonParams
    from quantark.volmodels.slv import LeverageSurface

    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.5)
    if solver == "heston":
        engine = HestonPDESolver(params, n_x=16, n_v=8, n_t=8)
        monkeypatch.setattr(heston_pde_solver, "price_delta_gamma_heston_pde", lambda **kw: (1.0, 0.0, 0.0))
    else:
        lev = LeverageSurface(time_grid=np.linspace(0.0, 1.0, 6), strike_grid=np.array([40.0, 100.0, 250.0]),
                              leverage_grid=np.ones((6, 3)))
        engine = HestonSLVPDESolver(params, lev, n_x=16, n_v=8, n_t=8)
        monkeypatch.setattr(heston_slv_pde_solver, "price_delta_gamma_slv_pde",
                            lambda **kw: (1.0, 0.0, 0.0), raising=False)
    seen = []

    def spy(product, pricing_env, **kw):
        seen.append(pricing_env.rate_curve)
        return 0.0

    monkeypatch.setattr(engine, "price", spy)
    env = wrapped_env()
    try:
        engine.calculate_greeks(CALL, env)
    except Exception:  # theta/other legs after rho may need a real solve; rho's two reprices come first
        pass
    rate_bumps = [c for c in seen if type(getattr(c, "inner", None)) is ParallelShiftRateCurve]
    assert len(rate_bumps) >= 2 and all(type(c) is TradingClockRateCurve for c in seen)
    assert_shifted_inner(rate_bumps[0], env.rate_curve, engine.params.get_effective_bump_config().rate_bump)


def test_convertible_floor_and_full_dv01_bump_the_calendar_quote(monkeypatch):
    from quantark.asset.bond.engine.convertible.convertible_bond_engine import ConvertibleBondEngine
    from quantark.asset.bond.product.convertible import ConvertibleBond

    env = wrapped_env()
    bond = ConvertibleBond(issue_date=datetime(2025, 1, 1), maturity_date=datetime(2027, 1, 1), face_value=100.0,
                           coupon_rate=0.025, conversion_ratio=10.0)
    engine = ConvertibleBondEngine.__new__(ConvertibleBondEngine)
    engine.pricing_env = env
    seen = []
    monkeypatch.setattr(engine, "floor_bond_price", lambda b: 1.0)
    monkeypatch.setattr(engine, "_floor_bond_price_with_env", lambda b, e: seen.append(e.rate_curve) or 1.0)
    engine.floor_bond_dv01(bond)
    engine.floor_bond_convexity(bond)
    assert len(seen) == 4
    for curve, sign in zip(seen, (1, -1, 1, -1)):
        assert_shifted_inner(curve, env.rate_curve, sign * 0.0001)


def test_convertible_full_dv01_and_convexity_bump_the_calendar_quote(monkeypatch):
    from quantark.asset.bond.engine.convertible.convertible_bond_engine import ConvertibleBondEngine
    from quantark.asset.bond.product.convertible import ConvertibleBond

    env = wrapped_env()
    bond = ConvertibleBond(issue_date=datetime(2025, 1, 1), maturity_date=datetime(2027, 1, 1), face_value=100.0,
                           coupon_rate=0.025, conversion_ratio=10.0)
    engine = ConvertibleBondEngine.__new__(ConvertibleBondEngine)
    engine.pricing_env = env
    seen = []

    class _Priced:
        def price(self, b):
            return 1.0

    monkeypatch.setattr(engine, "_create_bumped_engine", lambda e: seen.append(e.rate_curve) or _Priced())
    monkeypatch.setattr(engine, "price", lambda b: 1.0)
    engine.dv01(bond)
    engine.convexity(bond)
    assert len(seen) == 4
    for curve, sign in zip(seen, (1, -1, 1, -1)):
        assert_shifted_inner(curve, env.rate_curve, sign * 0.0001)
