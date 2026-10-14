"""W14b: a DCN through multi-day runs (spec I1, C3, AC2 tracker-aged, AC2b, AC4, AC4b, AC4c; A1, A4, A5, A6, A7').

``DCNLifecycleTracker`` decides each day's close on the BOOKED schedule's rows and daily-KI dates: knock-in on a KI
date, knock-out (then nothing more) on a KO row, the fixed coupon on a coupon row, the loss leg on the final row. Cash is
a dated ``RealizedCashflow`` on the state's ledger, keyed by observation date, paid on ``DCNOption.payment_dates()``.
Observations the run's clock never visits are decided at the previous run instant's spot (``gap=True``, the W6 rule);
observations before the run start are history (signed rule 3: no barrier state assumed, a past coupon not owed).

Pricing views (A1): pre-close ``product_for_pricing(d, env, initial=True)`` = ``spec.build(valuation_date=d)``;
post-close = the same ``.without_observed_through(d)`` with the day's cash in the ledger. Run consumers value the
contingent claim with the engine and the ledger separately (live MTM + pending PV + paid cash).
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dcn_fixtures import DCN_A, FLAT, SSE, make_dcn
from dcn_run_fixtures import (
    BOOKED, COUPON_PAY, COUPON_ROW, KI_DAY, KO_PAY, KO_ROW, backtest_run, booked_dcn, calendar_days, dynamic_run,
    flat_env_on, nyse_like_days, replay_run, spot_on, sse_days,
)

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import HestonDCNMCEngine, LocalVolDCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.lifecycle import DCNLifecycleState, LifecycleEventType, ValuationPoint, tracker_for
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.equity.lifecycle.dcn import DCNLifecycleTracker
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.dcn_option import DCNOption
from quantark.backtest.replay import AutocallableEngineConfig
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

R = FLAT["r"]
CASH = 10000.0
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
MC = dict(num_paths=2048, seed=7, use_sobol=True, num_batches=2, num_workers=1)
_CTOR = (
    "notional", "initial_price", "direction", "coupon_barrier_ratio", "ko_barrier_ratio", "ki_barrier_ratio",
    "ki_put_strike_ratio", "coupon_rate", "ko_coupon_rate", "participation", "coupon_counted_days",
    "coupon_days_denom", "schedule", "settlement_date", "knocked_in_at_valuation", "settlement_convention",
)


def rebuild(product, **changes):
    kw = {k: getattr(product, k) for k in _CTOR}
    kw.update(changes)
    return DCNOption(**kw)


def tracker(product=None, start=BOOKED, quantity=1.0):
    return tracker_for(product or booked_dcn(), quantity=quantity, start_date=pd.Timestamp(start))


def run_tracker(t, days, product=None):
    """Observe each day's close; returns the events in order."""
    events = []
    for d in days:
        events.extend(t.observe(pd.Timestamp(d), t.product_for_lifecycle(), flat_env_on(d), spot_on(d)))
    return events


def through(end, start=BOOKED):
    return [d for d in sse_days(start, end)]


# --- C3 / A5: the tracker and its state ------------------------------------------------------------------------

def test_tracker_for_a_dcn_and_its_clock():
    t = tracker()
    assert isinstance(t, DCNLifecycleTracker)
    assert isinstance(t.lifecycle, DCNLifecycleState)
    assert t.clock_origin == pd.Timestamp(DCN_A["initial_date"])
    assert not t.lifecycle.knocked_in
    assert tracker(booked_dcn(knocked_in_at_valuation=True)).lifecycle.knocked_in


def test_decisions_on_the_booked_rows():
    t = tracker(quantity=3.0)
    events = run_tracker(t, sse_days())
    kinds = [(e.event_type, pd.Timestamp(e.date).to_pydatetime()) for e in events]
    assert kinds == [(LifecycleEventType.COUPON, COUPON_ROW), (LifecycleEventType.KNOCK_IN, KI_DAY),
                     (LifecycleEventType.KNOCK_OUT, KO_ROW)]
    coupon, _ki, ko = events
    assert coupon.cashflow == pytest.approx(3.0 * CASH, rel=1e-14) and ko.terminates_position
    assert ko.cashflow == pytest.approx(3.0 * CASH, rel=1e-14)
    assert coupon.realized_cashflow.payment_date == COUPON_PAY and ko.realized_cashflow.payment_date == KO_PAY
    assert coupon.realized_cashflow.cashflow_id == "dcn:coupon:2023-04-03"
    assert ko.realized_cashflow.cashflow_id == "dcn:knock-out:2023-05-04"
    s = t.lifecycle
    assert s.knocked_out and s.ko_date == KO_ROW and s.knocked_in and s.ki_date == KI_DAY and not s.alive
    assert s.observed_row_indices == {0, 1}
    assert s.settled                                   # 2023-05-12: the KO coupon was paid on 2023-05-08
    assert s.realized_cashflows == pytest.approx(6.0 * CASH, rel=1e-14)


def test_no_coupon_once_knocked_in_and_nothing_after_the_knock_out():
    t = tracker()
    events = run_tracker(t, sse_days(BOOKED, datetime(2023, 5, 31)))
    assert [e.event_type for e in events][-1] is LifecycleEventType.KNOCK_OUT
    assert all(pd.Timestamp(e.date) <= pd.Timestamp(KO_ROW) for e in events)


def test_pending_cash_until_its_payment_date():
    t = tracker()
    run_tracker(t, through(datetime(2023, 4, 4)))
    s = t.lifecycle
    assert s.pending_settlement_cashflow == pytest.approx(CASH, rel=1e-14)
    assert s.settlement_date == COUPON_PAY and s.realized_cashflows == 0.0
    assert not s.settled                               # alive
    run_tracker(t, [datetime(2023, 4, 6)])
    assert s.pending_settlement_cashflow == 0.0 and s.realized_cashflows == pytest.approx(CASH, rel=1e-14)


def test_loss_leg_settles_at_the_final_observation():
    start = datetime(2024, 12, 27)
    p = make_dcn(DCN_A, valuation_date=start, knocked_in_at_valuation=True)
    t = tracker(p, start=start)
    final, pay = datetime(2025, 1, 3), datetime(2025, 1, 7)
    events = []
    for d in sse_days(start, datetime(2025, 1, 8)):
        events.extend(t.observe(pd.Timestamp(d), t.product_for_lifecycle(), flat_env_on(d, 4300.0), 4300.0))
    (maturity,) = events
    assert maturity.event_type is LifecycleEventType.MATURITY and maturity.terminates_position
    loss = -(p.notional / p.initial_price) * p.participation * (p.k_loss - 4300.0)
    assert maturity.cashflow == pytest.approx(loss, rel=1e-15)
    assert maturity.realized_cashflow.payment_date == pay and pd.Timestamp(maturity.date) == pd.Timestamp(final)
    assert t.lifecycle.matured and t.lifecycle.settled
    assert t.settle_maturity_if_due(pd.Timestamp(final), p, flat_env_on(final), 4300.0) is None


def test_matured_before_the_run_is_retired_with_a_warning():
    t = tracker(start=datetime(2025, 1, 6))
    with pytest.warns(MaturedBeforeRunWarning):
        assert t.retire_if_matured_before_run(flat_env_on(datetime(2025, 1, 6)))
    assert t.lifecycle.matured and not t.lifecycle.alive and t.lifecycle.settled
    assert t.lifecycle.ledger.cashflows == ()


def test_observations_before_the_run_start_are_history():
    """Rule 3 (signed 2026-10-09): no carried barrier state, a past coupon not owed."""
    start = datetime(2023, 4, 13)
    t = tracker(start=start)
    events = run_tracker(t, sse_days(start, datetime(2023, 4, 20)))
    assert events == []                                  # the 04-11/12 KI and the 04-03 coupon precede the run
    assert t.lifecycle.observed_row_indices == {0} and not t.lifecycle.knocked_in


def test_a_first_run_day_after_an_unvisited_observation_is_refused():
    t = tracker(start=COUPON_ROW)
    with pytest.raises(ValidationError, match="dcn-run-start"):
        t.observe(pd.Timestamp("2023-04-04"), t.product_for_lifecycle(), flat_env_on(datetime(2023, 4, 4)), 5500.0)


def test_a_skipped_observation_is_decided_at_the_previous_instant():
    t = tracker()
    days = [d for d in sse_days(BOOKED, datetime(2023, 4, 6)) if d != COUPON_ROW]
    events = []
    for d in days:
        spot = 5600.0 if d == datetime(2023, 3, 31) else spot_on(d)
        events.extend(t.observe(pd.Timestamp(d), t.product_for_lifecycle(), flat_env_on(d, spot), spot))
    (coupon,) = events
    assert pd.Timestamp(coupon.date) == pd.Timestamp(COUPON_ROW) and coupon.spot == 5600.0
    assert coupon.metadata["gap"] is True and coupon.realized_cashflow.metadata["gap"] is True


def test_product_views():
    t = tracker()
    assert t.product_for_pricing(pd.Timestamp("2023-03-20"), flat_env_on(datetime(2023, 3, 20))) is t.product
    run_tracker(t, through(datetime(2023, 3, 31)))
    pre = t.product_for_pricing(pd.Timestamp(COUPON_ROW), flat_env_on(COUPON_ROW), initial=True)
    assert pre.schedule.monthly[0].observation_date == COUPON_ROW
    run_tracker(t, [COUPON_ROW])
    post = t.product_for_pricing(pd.Timestamp(COUPON_ROW), flat_env_on(COUPON_ROW))
    assert post.schedule.monthly[0].observation_date == KO_ROW
    assert post.schedule.daily_ki_dates[0] > COUPON_ROW and post.schedule.valuation_date == COUPON_ROW


# --- AC2: tracker-aged == freshly booked with the same state, both views -----------------------------------------

def _fresh(d, knocked_in, post):
    p = make_dcn(DCN_A, valuation_date=d, knocked_in_at_valuation=knocked_in)
    return rebuild(p, schedule=p.schedule.without_observed_through(d)) if post else p


AC2_ROUTES = {
    "bsm_mc": lambda: DCNMCEngine(**MC),
    "bsm_pde": lambda: DCNPDEEngine(PDEParams(accuracy="fast")),
}


@pytest.mark.parametrize("route", sorted(AC2_ROUTES))
@pytest.mark.parametrize("day", [COUPON_ROW, datetime(2023, 4, 4), KI_DAY, datetime(2023, 4, 20)])
@pytest.mark.parametrize("post", [False, True])
def test_tracker_aged_equals_booked(route, day, post):
    t = tracker()
    run_tracker(t, [d for d in sse_days(BOOKED, day) if d < day])
    env = flat_env_on(day)
    if post:
        run_tracker(t, [day])
        aged = t.product_for_pricing(pd.Timestamp(day), env)
    else:
        aged = t.product_for_pricing(pd.Timestamp(day), env, initial=True)
    s = t.lifecycle
    if not post:
        s = dataclasses.replace(s, valuation_point=ValuationPoint(date=day))      # priced as of the day's open
    knocked_in = s.knocked_in if post else bool(s.knocked_in and s.ki_date is not None and s.ki_date < day)
    fresh = _fresh(day, knocked_in, post)
    engine = AC2_ROUTES[route]
    assert engine().price(aged, env, lifecycle_state=s) == engine().price(fresh, env, lifecycle_state=s)


@pytest.mark.parametrize("make", [lambda: LocalVolDCNMCEngine(**MC), lambda: LocalVolDCNPDEEngine(),
                                  lambda: HestonDCNPDESolver(HESTON, n_x=101, n_v=31)])
def test_tracker_aged_equals_booked_on_the_vol_routes(make):
    from test_dcn_lifecycle_state import env_on as grid_env_on

    day = datetime(2023, 4, 4)
    t = tracker()
    run_tracker(t, sse_days(BOOKED, day))
    env = grid_env_on(day, spot_on(day), grid=True)
    aged = t.product_for_pricing(pd.Timestamp(day), env)
    fresh = _fresh(day, False, True)
    s = t.lifecycle
    assert make().price(aged, env, lifecycle_state=s) == make().price(fresh, env, lifecycle_state=s)


# --- AC2b at the tracker: pre-close == post-close (engine + ledger) -----------------------------------------------

@pytest.mark.parametrize("route", sorted(AC2_ROUTES))
@pytest.mark.parametrize("day", [COUPON_ROW, KI_DAY, KO_ROW])
def test_pre_close_equals_post_close_through_the_tracker(route, day):
    t = tracker()
    run_tracker(t, [d for d in sse_days(BOOKED, day) if d < day])
    env = flat_env_on(day)
    engine = AC2_ROUTES[route]
    pre = engine().price(t.product_for_pricing(pd.Timestamp(day), env, initial=True), env, lifecycle_state=None) \
        + t.lifecycle.ledger.pending_pv(ValuationPoint(date=day), env)
    run_tracker(t, [day])
    post_product = t.product_for_pricing(pd.Timestamp(day), env) if t.lifecycle.alive else None
    post = (engine().price(post_product, env) if post_product is not None else 0.0) \
        + t.lifecycle.ledger.pending_pv(t.lifecycle.valuation_point, env)
    assert post == pytest.approx(pre, rel=1e-10)


# --- AC4: runs -------------------------------------------------------------------------------------------------

def _pde():
    return DCNPDEEngine(PDEParams(accuracy="fast"))


def _dynamic_frame(results):
    return pd.DataFrame([{"date": pd.Timestamp(x.date).normalize(), "value": x.portfolio_value,
                          "pending": x.pending_receivable_pv, "paid": x.paid_cash, "n": len(x.positions),
                          "events": [e.event_type for e in x.lifecycle_events]}
                         for x in results.day_results]).set_index("date")


def _check_cash(frame, quantity=1.0):
    """Coupon receivable until 04-06, then cash; KO coupon receivable until 05-08, then cash."""
    df_coupon = math.exp(-R * (COUPON_PAY - datetime(2023, 4, 4)).days / 365.0)
    assert frame.loc["2023-04-04", "pending"] == pytest.approx(quantity * CASH * df_coupon, rel=1e-12)
    assert frame.loc["2023-04-04", "paid"] == 0.0
    assert frame.loc["2023-04-06", "pending"] == 0.0
    assert frame.loc["2023-04-06", "paid"] == pytest.approx(quantity * CASH, rel=1e-14)
    df_ko = math.exp(-R * (KO_PAY - KO_ROW).days / 365.0)
    assert frame.loc["2023-05-04", "pending"] == pytest.approx(quantity * CASH * df_ko, rel=1e-12)
    assert frame.loc["2023-05-04", "n"] == 0
    assert frame.loc["2023-05-12", "paid"] == pytest.approx(2.0 * quantity * CASH, rel=1e-14)
    assert frame.loc["2023-05-12", "value"] == pytest.approx(2.0 * quantity * CASH, rel=1e-12)


def _value_on_observation_day_is_the_pre_close_value(frame, quantity=1.0):
    """Concern 2 / AC2b in a run: the day's (post-close) value == the pre-close engine value at the same spot."""
    pre = quantity * _pde().price(make_dcn(DCN_A, valuation_date=COUPON_ROW), flat_env_on(COUPON_ROW))
    assert frame.loc[pd.Timestamp(COUPON_ROW), "value"] == pytest.approx(pre, rel=1e-10)


def test_dynamic_scenario_run():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = dynamic_run(_pde())
    frame = _dynamic_frame(results)
    assert frame.loc[pd.Timestamp(COUPON_ROW), "events"] == ["COUPON"]
    assert frame.loc[pd.Timestamp(KI_DAY), "events"] == ["KI"]
    assert frame.loc[pd.Timestamp(KO_ROW), "events"] == ["KO"]
    _check_cash(frame)
    _value_on_observation_day_is_the_pre_close_value(frame)
    # weekends price (calendar days): Saturday 2023-04-08 holds a live, priced position
    assert frame.loc["2023-04-08", "n"] == 1 and np.isfinite(frame.loc["2023-04-08", "value"])


def _backtest_frame(results):
    states = results.states_df.copy()
    states.index = pd.to_datetime(states.index).normalize()
    return states.rename(columns={"portfolio_value": "value", "pending_receivable_pv": "pending",
                                  "paid_cash": "paid", "num_positions": "n"})


def test_equity_backtest_run():
    results = backtest_run(_pde())
    frame = _backtest_frame(results)
    events = results.get_lifecycle_events()
    assert list(events["event_type"]) == ["COUPON", "KI", "KO"]
    _check_cash(frame)
    _value_on_observation_day_is_the_pre_close_value(frame)


def _states(results):
    states = results.states_df
    return (states() if callable(states) else states).copy()


def _replay_frame(results):
    states = _states(results)
    if "date" in states.columns:
        states = states.set_index("date")
    states.index = pd.to_datetime(states.index).normalize()
    states["value"] = states["product_mtm"] + states["pending_receivable_pv"] + states["paid_cash"]
    states["n"] = [0 if not alive else 1 for alive in states["alive"]]
    return states.rename(columns={"pending_receivable_pv": "pending", "paid_cash": "paid"})


def test_replay_run():
    results = replay_run(AutocallableEngineConfig(pricing_engine_type=EngineType.PDE,
                                                  pde_params=PDEParams(accuracy="fast")))
    frame = _replay_frame(results)
    actions = results.actions_df() if callable(results.actions_df) else results.actions_df
    assert list(actions["action_type"]) == ["COUPON", "KI", "KO"]
    assert list(actions["gap"]) == [False, False, False]
    _check_cash(frame)
    _value_on_observation_day_is_the_pre_close_value(frame)


@pytest.mark.parametrize("consumer", ["dynamic", "backtest", "replay"])
def test_loss_leg_settles_in_every_run(consumer):
    start, end = datetime(2024, 12, 27), datetime(2025, 1, 10)
    p = make_dcn(DCN_A, valuation_date=start, knocked_in_at_valuation=True)
    loss = -(p.notional / p.initial_price) * p.participation * (p.k_loss - 4300.0)
    import dcn_run_fixtures as fx

    original = fx.spot_on
    fx.spot_on = lambda d: 4300.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if consumer == "dynamic":
                frame = _dynamic_frame(dynamic_run(_pde(), product=p, days=calendar_days(start, end)))
            elif consumer == "backtest":
                frame = _backtest_frame(backtest_run(_pde(), product=p, days=sse_days(start, end)))
            else:
                frame = _replay_frame(replay_run(
                    AutocallableEngineConfig(pricing_engine_type=EngineType.PDE,
                                             pde_params=PDEParams(accuracy="fast")),
                    product=p, days=sse_days(start, end)))
    finally:
        fx.spot_on = original
    df = math.exp(-R * 4 / 365.0)
    assert frame.loc["2025-01-03", "pending"] == pytest.approx(loss * df, rel=1e-12)
    assert frame.loc["2025-01-07", "paid"] == pytest.approx(loss, rel=1e-15)
    assert frame.loc["2025-01-10", "value"] == pytest.approx(loss, rel=1e-12)


# --- AC4b: replay builds DCN engines -----------------------------------------------------------------------------

def test_replay_builds_mc_and_pde_dcn_engines_and_runs():
    from quantark.backtest.replay.engine_factory import create_pricing_engine

    # an MC run's surface engine defaults to QUAD, which a DCN does not have: name it (no silent swap)
    config = AutocallableEngineConfig(pricing_engine_type=EngineType.MONTE_CARLO,
                                      surface_engine_type=EngineType.MONTE_CARLO,
                                      mc_params=MCParams(num_paths=1024, seed=5, use_qmc=True))
    with pytest.raises(ValidationError, match="Quad"):
        replay_run(AutocallableEngineConfig(pricing_engine_type=EngineType.MONTE_CARLO))
    engine = create_pricing_engine(booked_dcn(), config)
    assert type(engine) is DCNMCEngine and engine.num_paths == 1024 and engine.seed == 5 and engine.use_sobol
    pde = create_pricing_engine(booked_dcn(), AutocallableEngineConfig(pricing_engine_type=EngineType.PDE))
    assert type(pde) is DCNPDEEngine
    results = replay_run(config, days=sse_days(BOOKED, datetime(2023, 4, 7)))
    actions = results.actions_df() if callable(results.actions_df) else results.actions_df
    assert list(actions["action_type"]) == ["COUPON"]


@pytest.mark.parametrize("vol_model, solver, cls", [
    ("localvol", "mc", LocalVolDCNMCEngine), ("localvol", "pde", LocalVolDCNPDEEngine),
    ("heston", "mc", HestonDCNMCEngine), ("heston", "pde", HestonDCNPDESolver),
])
def test_vol_model_factory_builds_dcn_engines(vol_model, solver, cls):
    from pde_routes.cases import dcn_flat_surface
    from quantark.backtest.replay.engine_factory import create_vol_model_engine
    from quantark.volcalibration.calibrate import CalibratedVolModel

    calibrated = CalibratedVolModel(variant=vol_model, surface_date=BOOKED.date(), surface_sha="x",
                                    local_vol_surface=dcn_flat_surface() if vol_model == "localvol" else None,
                                    heston_params=HESTON if vol_model == "heston" else None)
    engine = create_vol_model_engine(vol_model=vol_model, solver=solver, calibrated=calibrated,
                                     mc_params=MCParams(num_paths=512, seed=3), product=booked_dcn())
    assert type(engine) is cls


def test_replay_config_admits_a_dcn_under_a_vol_model():
    from quantark.backtest.replay import BookAutocallableBacktestConfig, BookProduct, HedgeSpec
    from quantark.volcalibration.config import VolModelCalibrationConfig

    BookAutocallableBacktestConfig(
        products=[BookProduct(product=booked_dcn(), quantity=1.0, position_id=1, has_lifecycle=True)],
        market_data=__import__("dcn_run_fixtures").replay_market(), hedge=HedgeSpec(kind="spot"),
        engine_config=AutocallableEngineConfig(vol_source="surface", surface_vol_mode="full_grid",
                                               vol_model="localvol", vol_model_solver="mc",
                                               vol_model_calibration=VolModelCalibrationConfig()),
        calculate_event_probabilities=False,
    )


# --- AC4c (A7'): any run calendar ------------------------------------------------------------------------------

def _decisions(events):
    return [(str(e), pd.Timestamp(d).normalize(), round(float(c), 6)) for e, d, c in events]


def test_any_run_calendar_makes_the_same_decisions():
    sse = backtest_run(_pde())
    nyse = backtest_run(_pde(), days=nyse_like_days())          # 2023-04-03 is a holiday of the run calendar
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal = dynamic_run(_pde(), days=calendar_days())          # weekends and SSE holidays included

    def bt(results):
        ev = results.get_lifecycle_events()
        return _decisions(zip(ev["event_type"], ev.index, ev["cashflow"]))

    cal_events = [(e.event_type.value if hasattr(e.event_type, "value") else e.event_type, x.date, e.cashflow)
                  for x in cal.day_results for e in x.lifecycle_events]
    assert bt(sse) == _decisions(cal_events)
    # the run never visits 2023-04-03: the coupon is still decided on it, at 2023-03-31's spot (equal here)
    assert bt(nyse) == bt(sse)
    assert pd.Timestamp(COUPON_ROW) not in pd.DatetimeIndex([pd.Timestamp(d) for d in nyse_like_days()])
    replay = replay_run(AutocallableEngineConfig(pricing_engine_type=EngineType.PDE,
                                                 pde_params=PDEParams(accuracy="fast")), days=nyse_like_days())
    actions = replay.actions_df() if callable(replay.actions_df) else replay.actions_df
    assert list(actions["action_type"]) == ["COUPON", "KI", "KO"]
    assert list(actions["gap"]) == [True, False, False]                  # flagged as a gap decision
    assert pd.Timestamp(actions["date"].iloc[0]) == pd.Timestamp(COUPON_ROW)


# --- CP5 fixes -------------------------------------------------------------------------------------------------

def _term_dividend_config():
    return AutocallableEngineConfig(pricing_engine_type=EngineType.PDE, pde_params=PDEParams(accuracy="fast"),
                                    dividend_source="futures_curve")


def test_term_dividend_replay_runs_past_a_knock_out():
    """The replay samples replays[0]'s remaining maturity every day, alive or not (``_remaining_maturity_years``)."""
    results = replay_run(_term_dividend_config(), fixed_dividend_yield=None)
    frame = _replay_frame(results)
    assert frame.loc["2023-05-12", "paid"] == pytest.approx(2.0 * CASH, rel=1e-14)


def test_term_dividend_replay_runs_through_the_loss_leg_lag():
    start, end = datetime(2024, 12, 27), datetime(2025, 1, 10)
    p = make_dcn(DCN_A, valuation_date=start, knocked_in_at_valuation=True)
    import dcn_run_fixtures as fx

    original = fx.spot_on
    fx.spot_on = lambda d: 4300.0
    try:
        frame = _replay_frame(replay_run(_term_dividend_config(), product=p, days=sse_days(start, end),
                                         fixed_dividend_yield=None))
    finally:
        fx.spot_on = original
    loss = -(p.notional / p.initial_price) * p.participation * (p.k_loss - 4300.0)
    assert frame.loc["2025-01-07", "paid"] == pytest.approx(loss, rel=1e-15)


def test_dead_or_expired_pricing_copy_is_terminal():
    start = datetime(2024, 12, 27)
    p = make_dcn(DCN_A, valuation_date=start, knocked_in_at_valuation=True)
    t = tracker(p, start=start)
    for d in sse_days(start, datetime(2025, 1, 3)):
        t.observe(pd.Timestamp(d), p, flat_env_on(d, 4300.0), 4300.0)
    assert t.lifecycle.matured
    for d in (datetime(2025, 1, 3), datetime(2025, 1, 6)):
        copy = t.product_for_pricing(pd.Timestamp(d), flat_env_on(d))
        assert copy.get_maturity() == 0.0


def test_pricing_accessor_is_pure():
    t = tracker()
    run_tracker(t, through(datetime(2023, 3, 31)))
    point = t.lifecycle.valuation_point
    t.product_for_pricing(pd.Timestamp(COUPON_ROW), flat_env_on(COUPON_ROW), initial=True)
    assert t.lifecycle.valuation_point == point


def test_pre_close_view_refuses_undecided_dates():
    t = tracker()
    run_tracker(t, through(datetime(2023, 3, 30)))
    with pytest.raises(ValidationError, match="undecided"):
        t.product_for_pricing(pd.Timestamp("2023-04-04"), flat_env_on(datetime(2023, 4, 4)), initial=True)


def test_time_of_day_run_instants_are_refused():
    t = tracker()
    with pytest.raises(ValidationError, match="time of day"):
        t.observe(pd.Timestamp("2023-03-27 15:00"), t.product, flat_env_on(BOOKED), 5500.0)
    t.observe(pd.Timestamp("2023-03-27 00:00"), t.product, flat_env_on(BOOKED), 5500.0)   # midnight: a date


@pytest.mark.parametrize("consumer", ["backtest", "replay"])
def test_pnl_explain_of_a_dcn_run_is_refused(consumer):
    from quantark.pnlexplain.config import PnLExplainConfig

    with pytest.raises(ValidationError, match="dcn-explain"):
        if consumer == "backtest":
            backtest_run(_pde(), pnl_explain=PnLExplainConfig())
        else:
            replay_run(AutocallableEngineConfig(pricing_engine_type=EngineType.PDE,
                                                pde_params=PDEParams(accuracy="fast")),
                       pnl_explain=PnLExplainConfig())


# --- CP5 round 2 -----------------------------------------------------------------------------------------------

def test_gap_calendar_replay_with_a_term_dividend_source():
    """A7': the replay samples the remaining maturity BEFORE the day's observation; on the day after an unvisited
    schedule date that sample must not refuse."""
    sse = replay_run(_term_dividend_config(), fixed_dividend_yield=None)
    gap = replay_run(_term_dividend_config(), days=nyse_like_days(), fixed_dividend_yield=None)

    def actions(results):
        frame = results.actions_df() if callable(results.actions_df) else results.actions_df
        return frame

    a_sse, a_gap = actions(sse), actions(gap)
    assert list(a_gap["action_type"]) == list(a_sse["action_type"]) == ["COUPON", "KI", "KO"]
    assert list(a_gap["cashflow"]) == pytest.approx(list(a_sse["cashflow"]), rel=1e-15)
    assert list(a_gap["gap"]) == [True, False, False]
    assert _replay_frame(gap).loc["2023-05-12", "paid"] == pytest.approx(2.0 * CASH, rel=1e-14)


def test_remaining_maturity_never_refuses():
    t = tracker()
    run_tracker(t, through(datetime(2023, 3, 31)))
    final = t.final_observation_date
    # 2023-04-03 skipped: undecided, yet the probe answers
    d = pd.Timestamp("2023-04-04")
    assert t.remaining_maturity(d) == (final - d.to_pydatetime()).days / 365.0
    with pytest.raises(ValidationError, match="undecided"):
        t.product_for_pricing(d, flat_env_on(d.to_pydatetime()))          # a copy that would be priced: refused
    assert t.remaining_maturity(pd.Timestamp(final) + pd.Timedelta(days=3)) == 0.0
    assert t.remaining_maturity(pd.Timestamp("2023-03-20")) == t.product.get_maturity()
    run_tracker(t, [datetime(2023, 4, 4)])
    t.lifecycle.knocked_out = True
    assert t.remaining_maturity(pd.Timestamp("2023-04-05")) == 0.0


def test_alive_note_past_its_final_observation_is_not_priced_before_observe():
    start = datetime(2024, 12, 27)
    p = make_dcn(DCN_A, valuation_date=start)
    t = tracker(p, start=start)
    for d in sse_days(start, datetime(2025, 1, 2)):
        t.observe(pd.Timestamp(d), p, flat_env_on(d, 5500.0), 5500.0)
    after = pd.Timestamp("2025-01-06")
    with pytest.raises(ValidationError, match="observe"):
        t.product_for_pricing(after, flat_env_on(after.to_pydatetime()))
    t.observe(after, p, flat_env_on(after.to_pydatetime(), 5500.0), 5500.0)   # decides 01-03 at 01-02's spot
    assert t.lifecycle.matured
    assert t.product_for_pricing(after, flat_env_on(after.to_pydatetime())).get_maturity() == 0.0
