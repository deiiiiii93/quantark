"""Spec test 5: KO day, coupon day, terminal carry, receivable payment, barrier KI substitution."""
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd
import pytest

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.barrier import BarrierLifecycleTracker
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType, OptionType
from quantark.util.enum.option_enums import BarrierType
from quantark.util.exceptions import ValidationError

START = datetime(2024, 1, 1)
Q = -3.0            # short the note, as a desk would be
KO_DATES = [0.25, 0.5, 0.75, 1.0]
WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _env(date, spot):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02), div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=date)


def _snowball():
    return SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=[105.0, 104.0, 103.0, 102.0], ko_rate=0.10,
            ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=KO_DATES,
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=KO_DATES, ki_continuous=False),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        accrual_config=AccrualConfig(is_annualized=True),
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def _find_event_day(tracker, event_type, spot_quiet, spot_trigger, max_days=400):
    """Advance the tracker with a quiet spot until the day BEFORE `event_type` would fire
    at `spot_trigger`; return (t0, t1)."""
    d = START
    for _ in range(max_days):
        d = d + timedelta(days=1)
        probe = deepcopy(tracker)
        ev = probe.observe(pd.Timestamp(d), probe.product_for_lifecycle(), _env(d, spot_trigger), spot_trigger)
        if any(e.event_type is event_type for e in ev):
            return d - timedelta(days=1), d
        tracker.observe(pd.Timestamp(d), tracker.product_for_lifecycle(), _env(d, spot_quiet), spot_quiet)
    raise AssertionError(f"no {event_type} within {max_days} days")


def _transition_pair(tracker, engine, t0, t1, spot0, spot1):
    e0, e1 = _env(t0, spot0), _env(t1, spot1)
    state_before = deepcopy(tracker.lifecycle if hasattr(tracker, "lifecycle") else tracker.state)
    product_t0 = tracker.product_for_pricing(pd.Timestamp(t0), e0)
    product_alive = tracker.product_for_pricing(pd.Timestamp(t1), e1)
    if hasattr(tracker, "lifecycle"):
        events = tracker.observe(pd.Timestamp(t1), tracker.product_for_lifecycle(), e1, spot1)
    else:
        events = tracker.observe(pd.Timestamp(t1), e1, spot1)
    state_after = deepcopy(tracker.lifecycle if hasattr(tracker, "lifecycle") else tracker.state)
    product_t1 = tracker.product_for_pricing(pd.Timestamp(t1), e1)
    engine_t1 = engine
    override = getattr(tracker, "engine_for_pricing", None)
    if override is not None and override() is not None:
        engine_t1 = override()
    # Float-schedule products keep their ledger in contract time: the tracker stamps a
    # numeric ValuationPoint on the state at every observation, and the snapshot carries it.
    s0 = ValuationSnapshot(product_t0, engine, e0, date=t0, quantity=Q, lifecycle_state=state_before,
                           valuation_point=getattr(state_before, "valuation_point", None))
    s1 = ValuationSnapshot(product_t1, engine_t1, e1, date=t1, quantity=Q, lifecycle_state=state_after,
                           valuation_point=getattr(state_after, "valuation_point", None))
    tr = LifecycleTransition(product_alive_t1=product_alive, engine_alive_t1=engine,
                             state_before=state_before, state_after=state_after, events=tuple(events))
    return s0, s1, tr


def _tol(x):
    return 1e-10 * max(1.0, abs(x))


def test_snowball_ko_day_event_row_and_exact_waterfall():
    tracker = AutocallableLifecycleTracker(product=_snowball(), quantity=Q, start_date=pd.Timestamp(START))
    t0, t1 = _find_event_day(tracker, LifecycleEventType.KNOCK_OUT, spot_quiet=100.0, spot_trigger=110.0)
    engine = SnowballPDESolver(PDEParams(accuracy="fast"))
    s0, s1, tr = _transition_pair(tracker, engine, t0, t1, 100.0, 110.0)
    assert tr.changed and tr.events[0].event_type is LifecycleEventType.KNOCK_OUT
    res = explain(s0, s1, transition=tr)
    assert res.pv_t1.contingent_mtm == 0.0
    assert res.pv_t1.paid_cash + res.pv_t1.pending_receivable_pv == pytest.approx(
        s1.lifecycle_state.ledger.cashflows[0].amount, rel=1e-12)
    event = [r for r in res.rows if r.factor is Factor.LIFECYCLE_EVENT][0]
    assert event.pnl == pytest.approx(res.pv_t1.total - res.pv_alive_t1.total, abs=_tol(res.total_pnl))
    assert event.metadata["events"][0]["event_type"] == "KO"
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    assert res.reconcile(ExplainMethod.TAYLOR) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    with pytest.raises(ValidationError):           # state changed, no transition
        explain(s0, s1)


def test_phoenix_coupon_day_books_cash_and_reconciles():
    phoenix = create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=1.0, ko_barrier=130.0,
                                      ki_barrier=70.0, coupon_barrier=90.0, coupon_rate=0.01,
                                      num_observations=12)
    tracker = AutocallableLifecycleTracker(product=phoenix, quantity=Q, start_date=pd.Timestamp(START))
    t0, t1 = _find_event_day(tracker, LifecycleEventType.COUPON, spot_quiet=80.0, spot_trigger=100.0)
    engine = PhoenixQuadEngine(QuadParams())
    s0, s1, tr = _transition_pair(tracker, engine, t0, t1, 80.0, 100.0)
    assert any(e.event_type is LifecycleEventType.COUPON for e in tr.events)
    res = explain(s0, s1, transition=tr, config=WF)
    assert res.pv_t1.paid_cash + res.pv_t1.pending_receivable_pv != 0.0
    assert res.pv_t0.paid_cash == 0.0
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))


def test_terminal_position_carries_receivable_then_pays_inside_time():
    st = AutocallableLifecycleState()
    st.mark_ko(START + timedelta(days=90), cashflow=-300.0, settlement_date=START + timedelta(days=95))
    prod = _snowball()
    eng = SnowballPDESolver(PDEParams(accuracy="fast"))
    d0, d1, d2 = START + timedelta(days=91), START + timedelta(days=92), START + timedelta(days=96)
    # the same (unrolled) product on every date: a terminal position is never priced, so the
    # contract-roll check is skipped for it (spec §8)
    s0 = ValuationSnapshot(prod, eng, _env(d0, 100.0), date=d0, quantity=Q, lifecycle_state=st)
    s1 = ValuationSnapshot(prod, eng, _env(d1, 101.0), date=d1, quantity=Q, lifecycle_state=st)
    res = explain(s0, s1, config=WF)
    by = res.by_factor(ExplainMethod.WATERFALL)
    assert res.pv_t0.contingent_mtm == 0.0 and res.pv_t0.pending_receivable_pv < 0.0
    assert by["spot"] == 0.0 and by["lifecycle_event"] == 0.0 and by["time"] != 0.0
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    s2 = ValuationSnapshot(prod, eng, _env(d2, 101.0), date=d2, quantity=Q, lifecycle_state=st)
    res2 = explain(s1, s2, config=WF)
    assert res2.pv_t1.pending_receivable_pv == 0.0 and res2.pv_t1.paid_cash == pytest.approx(-300.0)
    assert res2.by_factor(ExplainMethod.WATERFALL)["lifecycle_event"] == 0.0
    assert res2.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res2.total_pnl))


def test_barrier_ki_substitution_lands_in_event_row_not_model():
    dip = BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=85.0,
                        barrier_type=BarrierType.DOWN_IN, maturity=1.0)
    tracker = BarrierLifecycleTracker(product=dip, quantity=Q, start_date=pd.Timestamp(START))
    t0 = START + timedelta(days=10)
    t1 = START + timedelta(days=11)
    tracker.observe(pd.Timestamp(t0), _env(t0, 95.0), 95.0)
    engine = BarrierAnalyticalEngine()
    s0, s1, tr = _transition_pair(tracker, engine, t0, t1, 95.0, 80.0)
    assert tr.changed and s1.lifecycle_state.knocked_in
    assert isinstance(s1.product, EuropeanVanillaOption) and isinstance(s1.engine, BlackScholesEngine)
    res = explain(s0, s1, transition=tr, config=WF)
    by = res.by_factor(ExplainMethod.WATERFALL)
    assert by["model"] == 0.0
    assert res.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(res.total_pnl))
    # the wrong transition (engine swap without an event) is rejected
    bad = LifecycleTransition(product_alive_t1=tr.product_alive_t1, engine_alive_t1=engine,
                              state_before=tr.state_before, state_after=tr.state_before, events=())
    with pytest.raises(ValidationError):
        explain(s0, ValuationSnapshot(s1.product, s1.engine, s1.pricing_env, date=t1, quantity=Q,
                                      lifecycle_state=tr.state_before), transition=bad, config=WF)


def test_manager_pricing_products_is_pure():
    env = _env(START, 100.0)
    portfolio = Portfolio(portfolio_name="book", pricing_environments={"IDX": env})
    note = portfolio.add_position(product=_snowball(), quantity=Q, entry_price=0.0, underlying="IDX",
                                  engine=SnowballPDESolver(PDEParams(accuracy="fast")), entry_timestamp=START)
    vanilla = portfolio.add_position(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=1.0, entry_price=5.0, underlying="IDX", engine=BlackScholesEngine(), entry_timestamp=START)
    manager = PortfolioLifecycleManager(base_date=START)
    manager.register_positions(portfolio)
    day = START + timedelta(days=30)
    before = {pid: p.product for pid, p in portfolio.positions.items()}
    products = manager.pricing_products(portfolio, pd.Timestamp(day))
    assert set(products) == {note.position_id, vanilla.position_id}
    # the untracked schedule-free vanilla is rolled too (patch spec 2026-09-03 §4): a fresh copy
    assert products[vanilla.position_id] is not vanilla.product
    assert products[vanilla.position_id].maturity == pytest.approx(1.0 - 30 / 365)
    assert vanilla.product.maturity == 1.0
    assert products[note.position_id] is not note.product
    assert products[note.position_id].maturity == pytest.approx(1.0 - 30 / 365)
    assert {pid: p.product for pid, p in portfolio.positions.items()} == before   # no mutation
