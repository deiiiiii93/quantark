"""DCN prices on the environment's axis: cross-engine gate and zero-vol hand check (DCN time-axis spec AC6, AC7, AC8).

* AC6: on an ACT/360 environment and on a BUSINESS_DAYS environment (SSE calendar, 244 days a year), DCN MC vs DCN
  PDE (GBM) and local-vol DCN MC vs local-vol DCN PDE agree within the DCN gate ``max(3 s.e., 5 bp N)`` -- the gate of
  ``engine_pde/test_dcn_pde_solver.py::test_mc_pde_cross_validation_gate`` -- and the PDE price really moved off the
  calendar-day axis (by more than the gate, so a gate passing on the old axis would be noticed).
* AC7: at zero volatility the MC price equals a hand computation on the environment's axis: the deterministic path
  ``S0 exp((r - q) t)`` at the env times of the KI dates, the contract's rules walked row by row, every flow discounted
  at the env time of its C4 payment date. No DCN engine or grid code is used by the hand price.
* AC8 (gate part): a DCN whose SSE schedule holds Fri 2025-07-04 (a US holiday) on a US-calendar BUSINESS_DAYS
  environment: 07-04 shares Thu 07-03's instant (PRECEDING) or Mon 07-07's (FOLLOWING), and MC vs PDE (GBM) stays
  within the gate.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import LocalVolDCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import LocalVolDCNPDEEngine
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, GridVolSurface, SpotQuote
from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.priceenv import CurveQuotingBasis
from quantark.priceenv.date_axis import env_time, env_times
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.business_calendar import BusinessDayConvention as BDC

from dcn_fixtures import DCN_A, FLAT, SAMPLE_DCN_A, SSE, make_dcn

US = create_calendar(CalendarType.US, year_range=(2020, 2030))

#: the clocks under test: ACT/360, and SSE business days at 244 a year
CLOCKS = {
    "act_360": dict(day_count=DayCountConvention.ACT_360),
    "business_days_sse": dict(day_count=DayCountConvention.BUSINESS_DAYS, calendar=SSE, days=244),
}
#: Sobol RQMC settings: stderr must stay below 2 bp N (raise paths, never widen the gate)
GBM_MC = dict(num_paths=2 ** 20, num_batches=16, seed=42)
LV_MC = dict(num_paths=2 ** 19, num_batches=16, seed=42)


def _pde_params() -> PDEParams:
    # an explicit 2401 points, as the calendar-axis gate (the shared presets cap at 2000)
    return PDEParams(grid=GridConfig(points=2401, max_points=2401))


def _env(product, *, day_count=DayCountConvention.CALENDAR_DAYS, calendar=None, days=252, vol=None,
         r=FLAT["r"], q=FLAT["q"], adjust=BDC.PRECEDING):
    """An environment dated at the DCN's schedule valuation date, spot at the initial price."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=product.initial_price),
        vol_surface=vol if vol is not None else FlatVolSurface(volatility=FLAT["sigma"]),
        rate_curve=FlatRateCurve(rate=r), div_yield=ContinuousDividendYield(div_yield=q),
        valuation_date=product.schedule.valuation_date, day_count_convention=day_count, calendar=calendar,
        bus_days_in_year=days, holiday_time_adjustment=adjust,
        **({"rate_curve_quoting": CurveQuotingBasis.TRADING_YEAR, "div_yield_quoting": CurveQuotingBasis.TRADING_YEAR}
           if day_count == DayCountConvention.BUSINESS_DAYS else {}),
    )


def _skewed_grid(s0: float) -> GridVolSurface:
    """A put-skewed implied grid (5 vol points across 0.5 S0 .. 1.5 S0) through 2.5 years of the env's clock."""
    moneyness = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
    maturities = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
    base = FLAT["sigma"] + 0.05 * (1.0 - moneyness)
    return GridVolSurface(strikes=(s0 * moneyness).tolist(), maturities=maturities,
                          iv_grid=np.tile(base, (len(maturities), 1)))


def _assert_gate(mc, pde: float, notional: float) -> None:
    assert mc.std_error < 2e-4 * notional, f"stderr gate: raise paths, not tolerance (s.e. {mc.std_error:.2f})"
    gate = max(3.0 * mc.std_error, 5e-4 * notional)
    assert abs(mc.pv - pde) < gate, (
        f"|MC-PDE| = {abs(mc.pv - pde):.2f} exceeds gate {gate:.2f} "
        f"(MC={mc.pv:.2f} +/- {mc.std_error:.2f}, PDE={pde:.2f})"
    )


# --- AC6: cross-engine gate on ACT/360 and SSE BUSINESS_DAYS ----------------------------------------------------------


@pytest.mark.parametrize("clock", sorted(CLOCKS))
def test_gbm_mc_matches_pde_on_the_env_axis(clock):
    product = make_dcn(SAMPLE_DCN_A)
    env = _env(product, **CLOCKS[clock])
    pde = DCNPDEEngine(params=_pde_params()).price(product, env)
    # the env axis really moved the price (by more than the gate): the gate is not passing on the calendar-day axis
    on_calendar_days = DCNPDEEngine(params=_pde_params()).price(product, _env(product))
    assert abs(pde - on_calendar_days) > 5e-4 * product.notional
    mc = DCNMCEngine(**GBM_MC).price_detailed(product, env)
    _assert_gate(mc, pde, product.notional)


@pytest.mark.parametrize("clock", sorted(CLOCKS))
def test_local_vol_mc_matches_local_vol_pde_on_the_env_axis(clock):
    product = make_dcn(SAMPLE_DCN_A)
    grid = _skewed_grid(product.initial_price)
    env = _env(product, vol=grid, **CLOCKS[clock])
    pde = LocalVolDCNPDEEngine(params=_pde_params()).price(product, env)
    on_calendar_days = LocalVolDCNPDEEngine(params=_pde_params()).price(product, _env(product, vol=grid))
    assert abs(pde - on_calendar_days) > 5e-4 * product.notional
    mc = LocalVolDCNMCEngine(**LV_MC).price_detailed(product, env)
    _assert_gate(mc, pde, product.notional)


# --- AC7: zero volatility against a hand computation on the env axis --------------------------------------------------


class _ZeroVariance(BlackImpliedVolSurface):
    """Exactly zero volatility, through the exact total-variance protocol of the term sampler.

    ``FlatVolSurface`` refuses 0.0 and the sampler's ``get_vol`` route requires a strictly positive vol; the exact
    total-variance route (``exposes_exact_total_variance``, the trading-clock surface's protocol) accepts a zero
    variance increment by design (a holiday plateau), so every step vol is exactly 0.0 and every path is the
    deterministic forward.
    """

    exposes_exact_total_variance = True

    def total_variance(self, strike, time_to_maturity, spot):
        t = np.asarray(time_to_maturity, dtype=float)
        return 0.0 if t.ndim == 0 else np.zeros_like(t)

    def get_vol(self, strike, time_to_maturity, spot):
        return 0.0


def _hand_price(product, env, r: float, q: float):
    """Zero-vol DCN PV by hand on ``env``'s axis, and the branch taken.

    Path ``S(d) = S0 exp((r - q) t(d))`` with ``t(d) = env_time(env, d, anchor=schedule.valuation_date)``. In date
    order over the daily KI dates: KI is sticky once ``S <= KI barrier``; on a monthly row, a KO-eligible row with
    ``S >= KO barrier`` pays ``participation * ko_coupon_rate * ko_accrual * N`` and ends the note, else a
    coupon-eligible row pays ``participation * coupon_rate * coupon_accrual * N`` when not knocked in and
    ``S >= coupon barrier``; after the final row a knocked-in survivor pays ``-(N/S0) participation max(K - S_T, 0)``.
    Accruals are 30/360 contract data: ``counted_days / denom`` per eligible month index of the row. Every flow
    discounts at ``DF(env_time of its C4 payment date)``; the PV carries ``direction_sign``.
    """
    s = product.schedule
    anchor = s.valuation_date
    s0 = product.initial_price
    path = dict(zip(s.daily_ki_dates, s0 * np.exp((r - q) * env_times(env, s.daily_ki_dates, anchor=anchor))))
    # a branch decided by round-off would not be a hand check: every node is far from every barrier
    levels = np.array(list(path.values()))
    for barrier in (product.ki_barrier, product.coupon_barrier, product.ko_barrier):
        assert np.min(np.abs(levels / barrier - 1.0)) > 1e-6

    def df(d):
        return env.get_discount_factor(env_time(env, d, anchor=anchor))

    pay = product.payment_dates()
    a = product.coupon_counted_days / float(product.coupon_days_denom)
    rows = {m.observation_date: (j, m) for j, m in enumerate(s.monthly)}
    part, n = product.participation, product.notional
    knocked_in, ki_date, ko_row, coupon_rows, pv = product.knocked_in_at_valuation, None, None, [], 0.0
    for d in s.daily_ki_dates:
        spot = path[d]
        if spot <= product.ki_barrier and not knocked_in:
            knocked_in, ki_date = True, d
        if d not in rows:
            continue
        j, m = rows[d]
        if m.is_ko_obs and spot >= product.ko_barrier:
            periods = sum(k >= s.spec.ko_lock_months for k in m.month_indices)
            pv += part * product.ko_coupon_rate * a * periods * n * df(pay.ko_payment_dates[j])
            ko_row = j
            break
        if m.is_coupon_obs and not knocked_in and spot >= product.coupon_barrier:
            periods = sum(k >= s.spec.lock_months for k in m.month_indices)
            pv += part * product.coupon_rate * a * periods * n * df(pay.coupon_payment_dates[j])
            coupon_rows.append(j)
    else:
        if knocked_in:
            s_t = path[s.monthly[-1].observation_date]
            pv += -(n / s0) * part * max(product.k_loss - s_t, 0.0) * df(pay.loss_payment_date)
    return product.direction_sign * pv, dict(ko_row=ko_row, ki_date=ki_date, coupon_rows=coupon_rows)


#: (r, q, direction, expected branch). Carry +25%: rows 0 and 1 pay the coupon, row 2 (2023-06-05, S ~ 1.11 S0 on
#: both clocks) knocks out. Carry -25%: rows 0..4 pay (S >= 0.85 S0), then S falls below 0.80 S0 between rows 7 and 8
#: (knock-in) and the note runs to maturity with the loss leg (S_T ~ 0.60 S0), seen from the seller.
ZERO_VOL_CASES = {
    "knock_out": (0.30, 0.05, DCNDirection.BUYER, dict(ko_row=2, ki=False, coupon_rows=[0, 1])),
    "knock_in_loss": (0.02, 0.27, DCNDirection.SELLER, dict(ko_row=None, ki=True, coupon_rows=[0, 1, 2, 3, 4])),
}


@pytest.mark.parametrize("case", sorted(ZERO_VOL_CASES))
@pytest.mark.parametrize("clock", sorted(CLOCKS))
def test_zero_vol_mc_equals_the_hand_price_on_the_env_axis(clock, case):
    r, q, direction, want = ZERO_VOL_CASES[case]
    product = make_dcn(SAMPLE_DCN_A, direction=direction)
    env = _env(product, vol=_ZeroVariance(), r=r, q=q, **CLOCKS[clock])
    hand, branch = _hand_price(product, env, r, q)
    assert branch["ko_row"] == want["ko_row"]
    assert (branch["ki_date"] is not None) == want["ki"]
    assert branch["coupon_rows"] == want["coupon_rows"]
    if want["ki"]:   # the knock-in falls strictly between rows 7 and 8
        monthly = product.schedule.monthly
        assert monthly[7].observation_date < branch["ki_date"] < monthly[8].observation_date

    mc = DCNMCEngine(num_paths=64, seed=1).price_detailed(product, env)
    # every path is the same deterministic path (the s.e. is the variance formula's round-off): one branch for all
    assert mc.std_error < 1e-9 * product.notional
    assert mc.ko_probability == (1.0 if want["ko_row"] is not None else 0.0)
    assert mc.ki_probability == (1.0 if want["ki"] else 0.0)
    # Round-off bound 1e-9 N: the MC builds log S by summing ~485 steps of (fwd_rate - fwd_carry) dt, each forward
    # rate being -log(DF ratio) / dt, so log S_T carries <~ 485 eps ~ 1e-13 absolute error against (r - q) t; the
    # loss leg's sensitivity is (N/S0) S_T < N, so the price differs by <~ 1e-13 N (coupon/KO cash is exact --
    # identical payment times and DFs). 1e-9 N leaves 1e4 x room and stays far below what a clock change moves (the
    # same hand price on ACT/365F times misses by orders of magnitude more).
    assert abs(mc.pv - hand) <= 1e-9 * product.notional, f"MC {mc.pv!r} vs hand {hand!r}"


# --- AC8: collapsed instants on a US-calendar BUSINESS_DAYS environment -----------------------------------------------

#: an SSE-scheduled DCN whose daily KI dates include Fri 2025-07-04 (an SSE trading day, a US holiday)
DCN_US = dict(DCN_A, initial_date=datetime(2025, 1, 3), valuation_date=datetime(2025, 6, 4),
              maturity_date=datetime(2026, 1, 5), tenor_months=12, settlement_date=datetime(2026, 1, 7))
JUL3, JUL4, JUL7 = datetime(2025, 7, 3), datetime(2025, 7, 4), datetime(2025, 7, 7)
MERGED = {BDC.PRECEDING: (JUL3, JUL4), BDC.FOLLOWING: (JUL4, JUL7)}


@pytest.mark.parametrize("adjust", [BDC.PRECEDING, BDC.FOLLOWING], ids=["preceding", "following"])
def test_gbm_mc_matches_pde_with_collapsed_instants(adjust):
    product = make_dcn(DCN_US)
    env = _env(product, day_count=DayCountConvention.BUSINESS_DAYS, calendar=US, days=252, adjust=adjust)
    ctx = build_dcn_grid_context(product, env)
    assert MERGED[adjust] in ctx.column_dates   # 07-04 really shares one instant with its neighbour
    pde = DCNPDEEngine(params=_pde_params()).price(product, env)
    mc = DCNMCEngine(**GBM_MC).price_detailed(product, env)
    assert np.isfinite(pde) and np.isfinite(mc.pv)
    _assert_gate(mc, pde, product.notional)
