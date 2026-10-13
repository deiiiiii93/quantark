"""W15: a DCN's ``settlement_convention`` times ALL of its cash (spec I2, contract C4, amendment A2).

``settlement_convention=None`` keeps the schedule's offset-derived coupon / KO payment dates and ``settlement_date``
(bitwise, pinned by the existing DCN suites). With a convention, every flow pays at
``payment_date_by_convention(convention, determination_date, calendar)``: coupon and KO from their observation date,
the loss leg from the final observation date; calendar = the convention's own, else the schedule's. ``YEAR_FRACTION``
lags are refused: a dated-only DCN needs payment DATES. ``DCNOption.payment_dates()`` (C4) is the single source; every
route reads it through ``build_dcn_grid_context`` and the MC event statistics read it directly.
"""
from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from dcn_fixtures import DCN_A, DCN_B, FLAT, SSE, flat_env, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNOption, DCNPaymentDates
from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementConvention,
    SettlementLagUnit,
    SettlementRequest,
    SettlementResolver,
    payment_date_by_convention,
)
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.contracts import OutputKind
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention, create_calendar
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

R = FLAT["r"]
BD3 = SettlementConvention(lag=3, lag_unit=SettlementLagUnit.BUSINESS_DAYS)
CD5 = SettlementConvention(lag=5, lag_unit=SettlementLagUnit.CALENDAR_DAYS)
YF = SettlementConvention(lag=0.01, lag_unit=SettlementLagUnit.YEAR_FRACTION)
WEEKENDS_ONLY = create_calendar(CalendarType.NONE)
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
CONVENTIONS = {"BD3": BD3, "CD5": CD5}
_CTOR_FIELDS = (
    "notional", "initial_price", "direction", "coupon_barrier_ratio", "ko_barrier_ratio", "ki_barrier_ratio",
    "ki_put_strike_ratio", "coupon_rate", "ko_coupon_rate", "participation", "coupon_counted_days",
    "coupon_days_denom", "schedule", "settlement_date", "knocked_in_at_valuation", "settlement_convention",
)


def rebuild(product: DCNOption, **changes) -> DCNOption:
    kw = {k: getattr(product, k) for k in _CTOR_FIELDS}
    kw.update(changes)
    return DCNOption(**kw)


def with_convention(contract, convention, **overrides) -> DCNOption:
    return rebuild(make_dcn(contract, **overrides), settlement_convention=convention)


def schedule_twin(product: DCNOption) -> DCNOption:
    """A convention-free DCN whose SCHEDULE carries the C4 dates: what every route must price identically."""
    dates = product.payment_dates()
    rows = tuple(
        dataclasses.replace(m, coupon_payment_date=c, ko_payment_date=k)
        for m, c, k in zip(product.schedule.monthly, dates.coupon_payment_dates, dates.ko_payment_dates)
    )
    schedule = dataclasses.replace(product.schedule, monthly=rows)
    return rebuild(product, schedule=schedule, settlement_date=dates.loss_payment_date, settlement_convention=None)


def env_on(date, spot=6000.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=FLAT["sigma"]),
        rate_curve=FlatRateCurve(rate=R), div_yield=ContinuousDividendYield(div_yield=FLAT["q"]),
        valuation_date=date, day_count_convention=DayCountConvention.ACT_365)


def grid_env():
    env = flat_env(**FLAT)
    strikes = np.array([3000.0, 4500.0, 6000.0, 7500.0, 9000.0])
    mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    base = FLAT["sigma"] + 0.05 * (6000.0 - strikes) / 6000.0
    env.vol_surface = GridVolSurface(strikes=strikes.tolist(), maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    return env


def yf(d0, d1):
    return (d1 - d0).days / 365.0


# --- the settlement.py helper (A2) ------------------------------------------------------------------------------

def test_helper_business_days_rolls_on_the_given_calendar():
    obs = datetime(2024, 4, 3)                      # SSE closed 4-5 April 2024 (Qingming)
    assert payment_date_by_convention(BD3, obs, SSE) == datetime(2024, 4, 10)
    assert payment_date_by_convention(BD3, obs, WEEKENDS_ONLY) == datetime(2024, 4, 8)


def test_helper_calendar_days_then_adjusts():
    obs = datetime(2024, 4, 3)                      # +5 = Monday 8 April, an SSE trading day
    assert payment_date_by_convention(CD5, obs, SSE) == datetime(2024, 4, 8)
    sat = SettlementConvention(lag=3, lag_unit=SettlementLagUnit.CALENDAR_DAYS)
    assert payment_date_by_convention(sat, datetime(2024, 4, 3), SSE) == datetime(2024, 4, 8)   # Sat -> Following
    unadjusted = dataclasses.replace(sat, business_day_convention=BusinessDayConvention.UNADJUSTED)
    assert payment_date_by_convention(unadjusted, datetime(2024, 4, 3), None) == datetime(2024, 4, 6)


def test_helper_zero_lag_pays_at_determination():
    zero = SettlementConvention(lag=0, lag_unit=SettlementLagUnit.BUSINESS_DAYS)
    assert payment_date_by_convention(zero, datetime(2024, 4, 6), SSE) == datetime(2024, 4, 6)


def test_helper_refuses_year_fraction_and_a_missing_calendar():
    with pytest.raises(ValidationError, match="YEAR_FRACTION"):
        payment_date_by_convention(YF, datetime(2024, 4, 3), SSE)
    with pytest.raises(ValidationError, match="calendar"):
        payment_date_by_convention(BD3, datetime(2024, 4, 3), None)
    with pytest.raises(ValidationError, match="calendar"):
        payment_date_by_convention(CD5, datetime(2024, 4, 3), None)


@pytest.mark.parametrize("convention", [BD3, CD5], ids=["BD3", "CD5"])
def test_resolver_derives_the_same_date_as_the_helper(convention):
    """``_resolve_payment`` is refactored onto the helper: one rule for contingent flows and the DCN."""
    env = env_on(datetime(2024, 4, 1))
    env.calendar = SSE
    product = SimpleNamespace(settlement_convention=convention, settlement_date=None)
    resolved = SettlementResolver.resolve_contingent(
        product, SettlementRequest(kind=CashflowKind.COUPON, determination_date=datetime(2024, 4, 3)), env)
    assert resolved.payment_date == payment_date_by_convention(convention, datetime(2024, 4, 3), SSE)


# --- C4: DCNOption.payment_dates() ------------------------------------------------------------------------------

def test_no_convention_returns_the_schedule_dates_and_settlement_date():
    p = make_dcn(DCN_A)
    dates = p.payment_dates()
    assert isinstance(dates, DCNPaymentDates)
    assert dates.coupon_payment_dates == tuple(m.coupon_payment_date for m in p.schedule.monthly)
    assert dates.ko_payment_dates == tuple(m.ko_payment_date for m in p.schedule.monthly)
    assert dates.loss_payment_date == p.settlement_date


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_convention_dates_every_flow_from_its_determination_date(name):
    convention = CONVENTIONS[name]
    p = with_convention(DCN_A, convention)
    dates = p.payment_dates()
    want = tuple(payment_date_by_convention(convention, m.observation_date, SSE) for m in p.schedule.monthly)
    assert dates.coupon_payment_dates == want
    assert dates.ko_payment_dates == want
    final = p.schedule.monthly[-1].observation_date
    assert dates.loss_payment_date == payment_date_by_convention(convention, final, SSE)
    assert dates.loss_payment_date != p.settlement_date          # 2025-01-07 under the offsets
    assert want != tuple(m.coupon_payment_date for m in p.schedule.monthly)


def test_convention_calendar_wins_over_the_schedule_calendar():
    """A7': not SSE-only. A convention naming its own calendar rolls on it."""
    own = dataclasses.replace(BD3, calendar=WEEKENDS_ONLY)
    p = with_convention(DCN_A, own)
    want = tuple(payment_date_by_convention(BD3, m.observation_date, WEEKENDS_ONLY) for m in p.schedule.monthly)
    sse = tuple(payment_date_by_convention(BD3, m.observation_date, SSE) for m in p.schedule.monthly)
    assert p.payment_dates().coupon_payment_dates == want
    assert want != sse                               # e.g. the 2024-04-03 row (Qingming)


def test_zero_lag_convention_pays_at_observation():
    zero = SettlementConvention(lag=0, lag_unit=SettlementLagUnit.BUSINESS_DAYS)
    p = with_convention(DCN_A, zero)
    obs = tuple(m.observation_date for m in p.schedule.monthly)
    dates = p.payment_dates()
    assert dates.coupon_payment_dates == obs == dates.ko_payment_dates
    assert dates.loss_payment_date == obs[-1]


def test_year_fraction_convention_is_refused():
    with pytest.raises(ValidationError, match="YEAR_FRACTION"):
        with_convention(DCN_A, YF)
    zero_yf = SettlementConvention(lag=0.0, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    with pytest.raises(ValidationError, match="YEAR_FRACTION"):
        with_convention(DCN_A, zero_yf)
    mutated = make_dcn(DCN_A)
    mutated.settlement_convention = YF               # set after validation: refused where the dates are read
    with pytest.raises(ValidationError, match="YEAR_FRACTION"):
        mutated.payment_dates()
    with pytest.raises(ValidationError, match="YEAR_FRACTION"):
        DCNMCEngine(num_paths=256, seed=3, num_workers=1).price(mutated, flat_env(**FLAT))


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_grid_context_pay_times_are_the_c4_dates(name):
    p = with_convention(DCN_A, CONVENTIONS[name])
    ctx, dates, v = build_dcn_grid_context(p), p.payment_dates(), p.schedule.valuation_date
    np.testing.assert_array_equal(ctx.coupon_pay_times, [yf(v, d) for d in dates.coupon_payment_dates])
    np.testing.assert_array_equal(ctx.ko_pay_times, [yf(v, d) for d in dates.ko_payment_dates])
    assert ctx.loss_pay_time == yf(v, dates.loss_payment_date)


# --- AC5: every route discounts every leg at the C4 dates ------------------------------------------------------

MC = dict(num_paths=4096, seed=7, use_sobol=True, num_batches=2, num_workers=1)
ROUTES = {
    "bsm_mc": (lambda: DCNMCEngine(**MC), lambda: flat_env(**FLAT)),
    "lv_mc": (lambda: LocalVolDCNMCEngine(**MC), grid_env),
    "heston_mc": (lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=2048, seed=7, num_workers=1),
                  lambda: flat_env(**FLAT)),
    "bsm_pde": (lambda: DCNPDEEngine(PDEParams(accuracy="fast")), lambda: flat_env(**FLAT)),
    "lv_pde": (lambda: LocalVolDCNPDEEngine(params=PDEParams(grid=GridConfig(points=301))), grid_env),
    "heston_pde": (lambda: HestonDCNPDESolver(HESTON, n_x=101, n_v=31), lambda: flat_env(**FLAT)),
}


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_route_prices_the_convention_as_its_schedule_twin(route, name):
    """Bitwise equal to a convention-free DCN whose schedule carries the C4 dates; and the convention moves PV."""
    make_engine, make_env = ROUTES[route]
    p = with_convention(DCN_A, CONVENTIONS[name])
    priced = make_engine().price(p, make_env())
    assert priced == make_engine().price(schedule_twin(p), make_env())
    assert priced != make_engine().price(make_dcn(DCN_A), make_env())


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_mc_legs_rediscount_by_hand_dfs(name):
    """Same paths, same decisions: each period's coupon / KO PV and the loss leg move by exactly the hand DF ratio
    from the schedule date (DCN_B: offset 0, the observation date) to the C4 date."""
    base_p = make_dcn(DCN_B)
    conv_p = with_convention(DCN_B, CONVENTIONS[name])
    env = flat_env(**FLAT)
    base = DCNMCEngine(**MC).price_detailed(base_p, env)
    conv = DCNMCEngine(**MC).price_detailed(conv_p, env)
    v = base_p.schedule.valuation_date
    dates = conv_p.payment_dates()
    ratio = lambda d_old, d_new: math.exp(-R * (yf(v, d_new) - yf(v, d_old)))  # noqa: E731
    for j, m in enumerate(base_p.schedule.monthly):
        assert conv.pv_fixed_coupons_by_period[j] == pytest.approx(
            base.pv_fixed_coupons_by_period[j] * ratio(m.coupon_payment_date, dates.coupon_payment_dates[j]),
            rel=1e-12, abs=1e-9)
        assert conv.pv_ko_coupons_by_period[j] == pytest.approx(
            base.pv_ko_coupons_by_period[j] * ratio(m.ko_payment_date, dates.ko_payment_dates[j]),
            rel=1e-12, abs=1e-9)
    assert conv.pv_loss_leg == pytest.approx(
        base.pv_loss_leg * ratio(base_p.settlement_date, dates.loss_payment_date), rel=1e-12)
    assert conv.pv_loss_leg != base.pv_loss_leg


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_mc_event_stat_payment_dates_are_c4(name):
    p = with_convention(DCN_A, CONVENTIONS[name])
    result = DCNMCEngine(**MC).price_detailed(p, flat_env(**FLAT))
    dates = p.payment_dates()
    assert result.event_stats.payment_dates == (
        *dates.coupon_payment_dates, *dates.ko_payment_dates, dates.loss_payment_date)
    v = p.schedule.valuation_date
    np.testing.assert_array_equal(
        result.event_stats.payment_times, [yf(v, d) for d in result.event_stats.payment_dates])


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_mc_and_pde_agree_under_the_convention(name):
    p = with_convention(DCN_A, CONVENTIONS[name])
    env = flat_env(**FLAT)
    mc = DCNMCEngine(num_paths=1 << 14, seed=42, use_sobol=True, num_batches=8, num_workers=1).price_detailed(p, env)
    pde = DCNPDEEngine(PDEParams(accuracy="standard")).price(p, env)
    assert abs(pde - mc.pv) < max(4.0 * mc.std_error, 5e-4 * p.notional)


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_session_routes_read_the_convention(name):
    p = with_convention(DCN_A, CONVENTIONS[name])
    twin = schedule_twin(p)
    with PricingSession() as s:
        for make_engine in (lambda: DCNMCEngine(**MC), lambda: DCNPDEEngine(PDEParams(accuracy="fast"))):
            got, want = (
                s.execute(make_engine(), PricingRequest(product=q, pricing_env=flat_env(**FLAT),
                                                        outputs=frozenset([OutputKind.PV]))).value
                for q in (p, twin)
            )
            assert got == want


# --- the valuation-date row (W10 helper) pays on the C4 dates --------------------------------------------------

def _valuation_routes():
    return (DCNMCEngine(**MC), DCNPDEEngine(PDEParams(accuracy="fast")), HestonDCNPDESolver(HESTON, n_x=101, n_v=31))


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_knock_out_today_pays_the_ko_coupon_at_the_c4_date(name):
    today = datetime(2023, 4, 3)                     # DCN_A's first KO row
    p = with_convention(DCN_A, CONVENTIONS[name], valuation_date=today)
    pay = p.payment_dates().ko_payment_dates[0]
    assert pay > today + timedelta(days=2)
    want = p.participation * p.ko_coupon_rate * p.accrual_per_period * p.notional * math.exp(-R * yf(today, pay))
    for engine in _valuation_routes():
        assert engine.price(p, env_on(today, 6100.0)) == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("name", sorted(CONVENTIONS))
def test_final_observation_loss_leg_pays_at_the_c4_date(name):
    final = datetime(2025, 1, 3)
    p = with_convention(DCN_A, CONVENTIONS[name], valuation_date=final)
    pay = p.payment_dates().loss_payment_date
    assert pay != p.settlement_date
    spot = 4300.0
    want = -(p.notional / p.initial_price) * p.participation * (p.k_loss - spot) * math.exp(-R * yf(final, pay))
    for engine in _valuation_routes():
        assert engine.price(p, env_on(final, spot)) == pytest.approx(want, rel=1e-12)
