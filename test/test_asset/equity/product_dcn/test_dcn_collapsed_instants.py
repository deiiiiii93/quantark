"""DCN schedule dates on one env instant (DCN time-axis spec D6/D7/D10/D15-D17; AC8, AC9, AC9b, AC9c).

An SSE DCN priced on a US-calendar BUSINESS_DAYS environment: SSE trading day Fri 2025-07-04 is a US holiday, so it
shares an instant with Thu 07-03 (PRECEDING) or Mon 07-07 (FOLLOWING). Every engine prices it, the statistics keep one
KI row per instant (dated by its earliest date), a future date at time 0 is decided at the valuation instant, and a
pending receivable paid on the holiday is timed on the same axis as the price.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.dcn_valuation import settle_final_observation
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine, LocalVolDCNMCEngine, QEDCNMCEngine, coupled_heston_ladder_pair,
)
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.lifecycle import DCNLifecycleState, ValuationPoint
from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, RealizedCashflow
from quantark.asset.equity.lifecycle.dcn import DCNLifecycleTracker
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.priceenv import CurveQuotingBasis
from quantark.priceenv.date_axis import env_time
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.business_calendar import BusinessDayConvention as BDC
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, FLAT, make_dcn

US = create_calendar(CalendarType.US, year_range=(2020, 2030))
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
DCN_25 = dict(DCN_A, initial_date=datetime(2025, 1, 3), valuation_date=datetime(2025, 1, 3),
              maturity_date=datetime(2026, 1, 5), tenor_months=12, settlement_date=datetime(2026, 1, 7))
#: a DCN whose final observation is Fri 2025-07-04 (an SSE trading day, a US holiday)
DCN_JUL4 = dict(DCN_A, initial_date=datetime(2024, 7, 4), valuation_date=datetime(2024, 7, 4),
                maturity_date=datetime(2025, 7, 4), tenor_months=12, settlement_date=datetime(2025, 7, 8))
JUN3, JUN4, JUL3, JUL4, JUL7 = datetime(2025, 6, 3), datetime(2025, 6, 4), datetime(2025, 7, 3), datetime(2025, 7, 4), datetime(2025, 7, 7)


def us_env(d, spot=6000.0, adjust=BDC.PRECEDING, grid=False):
    vol = FlatVolSurface(volatility=FLAT["sigma"])
    if grid:   # the local-vol engines build Dupire from a grid surface
        from quantark.param.vol.vol_surface import GridVolSurface

        strikes, mats = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0], [0.25, 0.5, 1.0, 1.5, 2.0]
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.full((len(mats), len(strikes)), FLAT["sigma"]))
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol,
        rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(div_yield=FLAT["q"]),
        valuation_date=d, day_count_convention=DayCountConvention.BUSINESS_DAYS, calendar=US,
        holiday_time_adjustment=adjust, rate_curve_quoting=CurveQuotingBasis.TRADING_YEAR, div_yield_quoting=CurveQuotingBasis.TRADING_YEAR)


ENGINES = {
    "mc": lambda: DCNMCEngine(num_paths=4096, seed=7),
    "mc_lv": lambda: LocalVolDCNMCEngine(num_paths=4096, seed=7),
    "mc_heston": lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=4096, seed=7, num_batches=2),
    "mc_qe": lambda: QEDCNMCEngine(model_params=HESTON, num_paths=4096, seed=7, num_batches=2),
    "mc_coupled": lambda: coupled_heston_ladder_pair(HESTON, 1, HestonMCScheme.QUADEXP_M, num_paths=4096, seed=7)[0],
    "pde": lambda: DCNPDEEngine(PDEParams(accuracy="fast")),
    "pde_lv": lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="fast")),
    "adi_heston": lambda: HestonDCNPDESolver(HESTON, n_x=51, n_v=21),
}


@pytest.mark.parametrize("adjust", [BDC.PRECEDING, BDC.FOLLOWING])
@pytest.mark.parametrize("name", sorted(ENGINES))
def test_every_engine_prices_merged_instants(name, adjust):
    """AC8 / D10: finite price, pv == price() bitwise, one KI row per instant dated by its earliest date."""
    product, env = make_dcn(DCN_25, valuation_date=JUN4), us_env(JUN4, adjust=adjust, grid="lv" in name)
    price = ENGINES[name]().price(product, env)
    assert np.isfinite(price)
    stats = ENGINES[name]().calculate_event_stats(product, env)
    assert stats.pv == price
    table = stats.ki_table
    assert np.all(np.diff(np.asarray(table.times)) > 0.0)
    dates = list(table.dates)
    assert dates == sorted(set(dates))
    assert JUL3 in dates and JUL4 not in dates if adjust is BDC.PRECEDING else JUL4 in dates and JUL7 not in dates
    assert np.all(np.isfinite(np.asarray(table.probability)))


def test_ki_rows_follow_the_grid_columns():
    product, env = make_dcn(DCN_25, valuation_date=JUN4), us_env(JUN4)
    ctx = build_dcn_grid_context(product, env)
    table = DCNPDEEngine(PDEParams(accuracy="fast")).calculate_event_stats(product, env).ki_table
    assert tuple(table.dates) == tuple(c[0] for c in ctx.column_dates)
    np.testing.assert_array_equal(np.asarray(table.times), ctx.times)


@pytest.mark.parametrize("name", ["mc", "pde", "adi_heston"])
def test_future_date_at_zero_is_decided_at_valuation(name):
    """AC9 / D7: valued Thu 07-03, the Fri 07-04 KI date is at time 0 -- today's spot below KI knocks in now."""
    product, env = make_dcn(DCN_25, valuation_date=JUL3), us_env(JUL3, spot=4400.0)
    stats = ENGINES[name]().calculate_event_stats(product, env)
    assert stats.knock_in.probability == 1.0
    assert tuple(stats.ki_table.dates) == (JUL3,)


@pytest.mark.parametrize("name", ["mc", "pde"])
def test_unmonitored_valuation_with_a_date_at_zero_is_monitored(name):
    """AC9 / D17: valued Sat 2025-01-18; Mon 01-20 (SSE open, US holiday) is at time 0 and monitors today's spot."""
    sat = datetime(2025, 1, 18)
    product, env = make_dcn(DCN_25, valuation_date=sat), us_env(sat, spot=4400.0)
    stats = ENGINES[name]().calculate_event_stats(product, env)
    assert stats.knock_in.probability == 1.0
    assert tuple(stats.ki_table.dates) == (sat,)


@pytest.mark.parametrize("name", ["mc", "pde"])
@pytest.mark.parametrize("spot", [6300.0, 5500.0, 4000.0])
def test_final_observation_at_zero_settles(name, spot):
    """AC9: the final observation Fri 07-04 is at time 0 from Thu 07-03 -- the settled final row."""
    product, env = make_dcn(DCN_JUL4, valuation_date=JUL3), us_env(JUL3, spot=spot)
    assert build_dcn_grid_context(product, env).times.size == 1
    assert ENGINES[name]().price(product, env) == settle_final_observation(product, env).pv


def _state(d, decided, paid, amount=10_000.0):
    cf = RealizedCashflow(cashflow_id=f"dcn:coupon:{decided:%Y-%m-%d}", event_type=LifecycleEventType.COUPON,
                          amount=amount, determination_date=decided, payment_date=paid)
    return DCNLifecycleState(valuation_point=ValuationPoint(date=d), ledger=LifecycleCashflowLedger([cf]))


@pytest.mark.parametrize("adjust", [BDC.PRECEDING, BDC.FOLLOWING])
def test_pending_receivable_on_an_env_holiday(adjust):
    """AC9b / D15: a coupon decided 06-03 and paid Fri 07-04 (US holiday), valued Thu 07-03: priced, and timed as the
    grid times the same date (PRECEDING: time 0, DF 1)."""
    product, env = make_dcn(DCN_25, valuation_date=JUL3), us_env(JUL3, adjust=adjust)
    stats = DCNMCEngine(num_paths=4096, seed=7).calculate_event_stats(product, env, lifecycle_state=_state(JUL3, JUN3,
                                                                                                          JUL4))
    receivables = stats.receivables
    t_pay = env_time(env, JUL4, anchor=JUL3)
    assert float(receivables.payment_times[0]) == t_pay
    assert float(receivables.decision_times[0]) == env_time(env, JUN3, anchor=JUL3) < 0.0
    if adjust is BDC.PRECEDING:
        assert t_pay == 0.0 and float(receivables.discounted_cash[0]) == 10_000.0


def test_tracker_needs_no_env_and_maturity_takes_one():
    """AC9c / D16."""
    product = make_dcn(DCN_25, valuation_date=JUN3)
    tracker = DCNLifecycleTracker(product=product, quantity=1.0)
    env = us_env(JUN3)
    final = product.schedule.monthly[-1].observation_date
    assert tracker.remaining_maturity(JUN3, env) == env_time(env, final, anchor=JUN3)
    assert tracker.remaining_maturity(JUN3) == (final - JUN3).days / 365.0


@pytest.mark.parametrize("adjust", [BDC.PRECEDING, BDC.FOLLOWING])
def test_portfolio_ledger_times_dcn_receivables_as_the_price(adjust):
    """D15 in the run's ledger: a DCN flow paid on an env holiday right after the run date is valued as DCN price()
    values it (PRECEDING: time 0, DF 1 -- not refused; FOLLOWING: the next business day's time)."""
    from types import SimpleNamespace

    import pandas as pd

    from quantark.asset.equity.lifecycle.events import LifecycleEvent
    from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager

    def coupon(cashflow_id):
        return RealizedCashflow(cashflow_id=cashflow_id, event_type=LifecycleEventType.COUPON, amount=10_000.0,
                                determination_date=JUN3, payment_date=JUL4)

    manager = PortfolioLifecycleManager(JUN3)
    # booked as a run books them: the tracker's id, namespaced by position (``pos:dcn:coupon:...``)
    for position_id, cashflow_id in (("dcn_pos", f"dcn:coupon:{JUN3:%Y-%m-%d}"), ("other_pos", "coupon:2025-06-03")):
        manager._book_events(position_id, "SPX", [LifecycleEvent(event_type=LifecycleEventType.COUPON, date=JUN3,
                                                                 spot=6000.0, realized_cashflow=coupon(cashflow_id))])
    env = us_env(JUL3, adjust=adjust)
    portfolio = SimpleNamespace(pricing_environments={"SPX": env})
    # the non-DCN flow keeps the settlement resolver's business-day counting, which puts 07-04 at the valuation
    # instant: 0.14.0 refused it (the whole book raised); since 0.16.0 (DCN/dated run D7) it is timed at 0, DF(0) = 1
    manager._revalue_ledger(portfolio, pd.Timestamp(JUL3))
    dcn_pv = 10_000.0 * env.get_discount_factor(env_time(env, JUL4))
    assert manager.pending_receivable_pv == dcn_pv + 10_000.0
    manager.ledger = LifecycleCashflowLedger([c for c in manager.ledger.cashflows
                                              if c.cashflow_id.startswith("dcn_pos:")])
    manager._revalue_ledger(portfolio, pd.Timestamp(JUL3))
    assert manager.pending_receivable_pv == dcn_pv
    if adjust is BDC.PRECEDING:
        assert dcn_pv == 10_000.0
    else:
        assert dcn_pv == 10_000.0 * env.get_discount_factor(1 / 252)   # FOLLOWING: Mon 07-07, one business day
