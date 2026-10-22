"""A DCN valued on a day that is not a trading day of its schedule calendar (amendment A7'(a), T12).

The builder used to refuse such a valuation date. Now the monthly rows and daily-KI dates are those on or after it (so
none falls on it), times are ACT/365F from it, and the valuation instant is not a monitoring date: the first knock-in
test is the next trading day's close. Raise -> price; no moved values (``lifecycle_state=None`` on a trading day is
pinned bitwise by the T12 capture).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from dcn_fixtures import DCN_A, FLAT, SSE, make_dcn, schedule_kwargs

from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row
from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_schedule import build_dcn_schedule
from quantark.asset.equity.riskmeasures.greek_conventions_report import build_cash_greeks_report
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.volmodels.heston import HestonParams

FRI, SAT, SUN = datetime(2024, 6, 7), datetime(2024, 6, 8), datetime(2024, 6, 9)
HOLIDAY = datetime(2024, 6, 10)          # Dragon Boat Festival: SSE closed on a Monday
TUE = datetime(2024, 6, 11)              # the first trading day after
DAYS = (FRI, SAT, SUN, HOLIDAY, TUE)
MC = dict(num_paths=1 << 14, seed=42, use_sobol=True, num_batches=8, num_workers=1)
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)


def env_on(d, spot=5500.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=FLAT["sigma"]),
        rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(div_yield=FLAT["q"]),
        valuation_date=d, day_count_convention=DayCountConvention.ACT_365)


@pytest.mark.parametrize("d", [SAT, SUN, HOLIDAY])
def test_schedule_builds_on_a_non_trading_day(d):
    assert not SSE.is_business_day(d)
    s = build_dcn_schedule(**schedule_kwargs(dict(DCN_A, valuation_date=d)))
    trading = build_dcn_schedule(**schedule_kwargs(dict(DCN_A, valuation_date=FRI)))
    assert s.valuation_date == d
    assert s.daily_ki_dates[0] == TUE and s.daily_ki_dates == tuple(x for x in trading.daily_ki_dates if x > d)
    assert s.monthly == tuple(m for m in trading.monthly if m.observation_date > d)


@pytest.mark.parametrize("d", [SAT, HOLIDAY])
def test_grid_starts_at_an_unmonitored_instant(d):
    p = make_dcn(DCN_A, valuation_date=d)
    ctx = build_dcn_grid_context(p)
    assert not ctx.valuation_monitored
    assert ctx.times[0] == 0.0 and ctx.times[1] == (TUE - d).days / 365.0
    assert int(ctx.obs_cols[0]) >= 1
    assert observe_valuation_row(p, env_on(d)) is None
    assert p.get_maturity() == (p.schedule.monthly[-1].observation_date - d).days / 365.0


def test_trading_day_grid_is_monitored():
    assert build_dcn_grid_context(make_dcn(DCN_A, valuation_date=FRI)).valuation_monitored


@pytest.mark.parametrize("d", [SAT, HOLIDAY])
def test_weekend_and_holiday_valuations_price_and_agree(d):
    p, env = make_dcn(DCN_A, valuation_date=d), env_on(d)
    mc = DCNMCEngine(**MC).price_detailed(p, env)
    pde = DCNPDEEngine(PDEParams(accuracy="standard")).price(p, env)
    assert abs(pde - mc.pv) < max(4.0 * mc.std_error, 5e-4 * p.notional)
    heston = HestonDCNPDESolver(HESTON, n_x=101, n_v=31).price(p, env)
    assert heston == heston                                    # finite, no crash


@pytest.mark.parametrize("engine", ["mc", "pde"])
def test_values_are_continuous_across_the_closed_days(engine):
    """Same spot, Friday to Tuesday: each closed day moves the value by one calendar day of carry. The closed-day
    values lie between Friday's and Tuesday's (monotone in the date), never jump."""
    make = (lambda: DCNMCEngine(**MC)) if engine == "mc" else (lambda: DCNPDEEngine(PDEParams(accuracy="standard")))
    values = [make().price(make_dcn(DCN_A, valuation_date=d), env_on(d)) for d in DAYS]
    step = values[-1] - values[0]
    assert step != 0.0
    span = sorted((values[0], values[-1]))
    slack = 0.05 * abs(step)
    for v in values[1:-1]:
        assert span[0] - slack <= v <= span[1] + slack
    diffs = [b - a for a, b in zip(values, values[1:])]
    assert all(x * step >= -slack for x in diffs)


def test_closed_day_is_not_a_knock_in_test():
    """Spot below the KI barrier on a Saturday: no knock-in until Tuesday's close (some paths recover)."""
    spot = 0.98 * make_dcn(DCN_A).ki_barrier
    fri = DCNMCEngine(**MC).price_detailed(make_dcn(DCN_A, valuation_date=FRI), env_on(FRI, spot))
    sat = DCNMCEngine(**MC).price_detailed(make_dcn(DCN_A, valuation_date=SAT), env_on(SAT, spot))
    assert fri.ki_probability == 1.0
    assert 0.0 < sat.ki_probability < 1.0
    pde_sat = DCNPDEEngine(PDEParams(accuracy="standard")).price(make_dcn(DCN_A, valuation_date=SAT),
                                                                 env_on(SAT, spot))
    assert abs(pde_sat - sat.pv) < max(4.0 * sat.std_error, 5e-4 * 1e6)
    pde_fri = DCNPDEEngine(PDEParams(accuracy="standard")).price(make_dcn(DCN_A, valuation_date=FRI),
                                                                 env_on(FRI, spot))
    assert pde_sat > pde_fri                                   # the buyer keeps a chance of the coupons


def test_cash_greeks_theta_rolls_from_a_closed_day():
    p, env = make_dcn(DCN_A, valuation_date=SAT), env_on(SAT)
    report = build_cash_greeks_report(p, env, DCNPDEEngine(PDEParams(accuracy="fast")), calendar=SSE)
    rolled = DCNPDEEngine(PDEParams(accuracy="fast")).price(make_dcn(DCN_A, valuation_date=TUE), env)
    assert report.theta_1d == rolled - report.pv
