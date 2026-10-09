"""The lifecycle tracker's one clock and its records dated before the run (tracker clock decisions, 2026-10-09).

Decision A: the clock origin is ``initial_date`` when the contract has one, else the run's ``start_date``; the
pricing copy, the valuation point, the terminal, the ledger and a REBASED schedule 2 all run on it.

Decision 2: a record dated before ``start_date`` with no carried state was observed before the run. A knock-out or
knock-in record is assumed not to have fired and is never tested against the run's first close; a Phoenix coupon
record is history (a memory coupon's was paid: no cash in the run, nothing memorized). A record dated ON
``start_date`` is today's and is tested on the first close.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quantark.asset.equity.lifecycle.ko_reset import tracker_for
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState

from test_tracker_clock_parity import (
    BUILDERS, START, env, phoenix, walk,
)

FORMS = ("numeric", "dated")


def _initial(gap):
    return START - pd.Timedelta(days=gap)


def _tracker(kind, form, gap, **kw):
    return tracker_for(BUILDERS[kind](form, _initial(gap)), quantity=1.0, start_date=START, **kw)


def _close(t, r, spot):
    day = START + pd.Timedelta(days=r)
    e = env(spot, day)
    product = t.product_for_lifecycle()
    events = t.observe(day, product, e, spot)
    maturity = t.settle_maturity_if_due(day, product, e, spot)
    return events + ([maturity] if maturity is not None else [])


# -- decision A: one clock --------------------------------------------------------------------------------------------

def test_the_clock_origin_is_the_initial_date_when_set():
    from quantark.asset.equity.lifecycle.autocallable import lifecycle_clock_origin
    product = BUILDERS["snowball"]("numeric", _initial(30))
    assert lifecycle_clock_origin(product, START) == _initial(30)
    product.initial_date = None
    assert lifecycle_clock_origin(product, START) == START
    assert lifecycle_clock_origin(product, None) is None


@pytest.mark.parametrize("gap", [30, 100])
def test_the_valuation_point_runs_from_the_initial_date(gap):
    t = _tracker("snowball", "numeric", gap)
    assert t.clock_origin == _initial(gap)
    assert t._valuation_point(START, t.product).time == gap / 365.0


@pytest.mark.parametrize("gap", [30, 100])
def test_the_pricing_copy_drops_a_record_on_the_close_that_books_it(gap):
    """Record 3 (contract day 122) is booked on run day 122 - N; the copy made after that close no longer holds it
    (findings 2.1: it held it N more days)."""
    t = _tracker("snowball", "numeric", gap)
    for r in range(122 - gap + 1):
        _close(t, r, 100.0)
    assert 3 in t.lifecycle.observed_ko_indices
    day = START + pd.Timedelta(days=122 - gap)
    copy = t.product_for_pricing(day, env(100.0, day))
    assert copy.barrier_config.ko_observation_count == 8                       # records 4..11
    assert copy.maturity == pytest.approx((365 - 122) / 365.0)


def test_an_initial_date_with_a_time_of_day_ages_the_copy_in_whole_days():
    """A booking stamped 09:30 dates its records on whole days; the copy made after the close that books record 3
    must not hold it for one more day."""
    product = BUILDERS["snowball"]("numeric", _initial(30))
    product.initial_date = (_initial(30) + pd.Timedelta(hours=9, minutes=30)).to_pydatetime()
    t = tracker_for(product, quantity=1.0, start_date=START)
    for r in range(122 - 30 + 1):
        _close(t, r, 100.0)
    day = START + pd.Timedelta(days=122 - 30)
    assert 3 in t.lifecycle.observed_ko_indices
    assert t.product_for_pricing(day, env(100.0, day)).barrier_config.ko_observation_count == 8


@pytest.mark.parametrize("gap", [30, 100])
def test_the_numeric_terminal_falls_due_with_the_last_record(gap):
    events = walk("snowball", "numeric", gap, "quiet")["events"]
    assert events[-1][:2] == (365 - gap, "MATURITY")


def test_a_rebased_knock_in_time_runs_from_the_initial_date():
    t = _tracker("ko_reset_rebased", "numeric", 100)
    for r in range(83):
        _close(t, r, 70.0 if r == 82 else 100.0)                                # record 5, contract day 182
    assert t.lifecycle.knocked_in
    assert t.knock_in_time() == 182 / 365.0
    assert t._schedule_two_base() == _initial(100)


# -- decision 2: records dated before the run --------------------------------------------------------------------------

@pytest.mark.parametrize("form", FORMS)
def test_knock_out_records_before_the_run_are_never_tested(form):
    """N = 100: records 0-2 (contract days 30, 61, 91) lie before the run; a first close above the barrier tests none
    of them. They are observed, so they are never due again."""
    t = _tracker("snowball", form, 100)
    assert _close(t, 0, 104.0) == []
    assert t.lifecycle.alive and t.lifecycle.observed_ko_indices == {0, 1, 2}
    assert [(e.event_type.value, e.observation_index) for e in _close(t, 22, 104.0)] == [("KO", 3)]


@pytest.mark.parametrize("form", FORMS)
def test_a_record_dated_on_the_run_start_is_tested_on_the_first_close(form):
    t = _tracker("snowball", form, 30)                                          # record 0 on contract day 30
    assert [(e.event_type.value, e.observation_index) for e in _close(t, 0, 104.0)] == [("KO", 0)]


@pytest.mark.parametrize("form", FORMS)
def test_knock_in_records_before_the_run_are_never_tested(form):
    """A KO-reset's discrete knock-in records at contract days 30, 61, 91 lie before a run issued 100 days earlier;
    a first close below the knock-in level does not knock it in."""
    t = _tracker("ko_reset_absolute", form, 100)
    assert _close(t, 0, 70.0) == []
    assert not t.lifecycle.knocked_in and t.lifecycle.observed_ki_indices == {0, 1, 2}
    # the next knock-in record (contract day 122) is tested
    assert [e.event_type.value for e in _close(t, 22, 70.0)] == ["KI"]


@pytest.mark.parametrize("form", FORMS)
def test_memory_coupons_before_the_run_were_paid(form):
    """Coupons 0-2 lie before the run: no cash, nothing memorized, even on a first close below the coupon barrier;
    coupon 3 then pays its own period only."""
    t = _tracker("phoenix", form, 100)
    assert _close(t, 0, 75.0) == []
    assert t.lifecycle.observed_coupon_indices == {0, 1, 2}
    assert not t.lifecycle.missed_coupon_indices and t.lifecycle.coupon_memory_count == 0
    coupon = _close(t, 22, 100.0)
    assert [(e.event_type.value, e.observation_index) for e in coupon] == [("COUPON", 3)]
    assert coupon[0].metadata == {}                                             # no arrears released
    assert [cf.cashflow_id for cf in t.lifecycle.ledger.cashflows] == ["coupon:3"]


@pytest.mark.parametrize("form", FORMS)
def test_non_memory_coupons_before_the_run_are_history(form):
    """A first close above the coupon barrier pays none of the coupons dated before the run."""
    product = phoenix(annualized=False, memory=False)(form, _initial(100))
    t = tracker_for(product, quantity=1.0, start_date=START)
    assert _close(t, 0, 100.0) == []
    assert t.lifecycle.observed_coupon_indices == {0, 1, 2} and not t.lifecycle.ledger.cashflows


def test_carried_observations_are_still_honoured():
    state = AutocallableLifecycleState(observed_ko_indices={0, 1, 2, 3})
    t = _tracker("snowball", "dated", 100, lifecycle=state)
    assert _close(t, 22, 104.0) == []                                           # record 3 was carried as observed
    assert [(e.event_type.value, e.observation_index) for e in _close(t, 52, 104.0)] == [("KO", 4)]


# -- the run calendar: a record before the run is due on no day of it ------------------------------------------------

@pytest.mark.parametrize("form", FORMS)
def test_the_calendar_schedule_never_fires_a_record_before_the_run(form):
    t = _tracker("snowball", form, 100)
    dates = pd.date_range(START, periods=300, freq="D")
    schedule = t.resolve_calendar_schedule(dates, env(100.0, START))
    assert list(schedule.ko_due_day[:4]) == [-1, -1, -1, 22]
    assert schedule.terminal_due_day == 265
    assert schedule.clock_origin == _initial(100)


@pytest.mark.parametrize("form", FORMS)
def test_the_ko_reset_calendar_never_fires_a_knock_in_before_the_run(form):
    t = _tracker("ko_reset_rebased", form, 100)
    dates = pd.date_range(START, periods=300, freq="D")
    schedule = t.resolve_calendar_schedule(dates, env(100.0, START))
    assert list(schedule.ki_due_day) == [-1, -1, -1, 22, 52, 82]
    assert list(schedule.placement(82).ko_due_day) == [112, 143, 173]           # 182 + 30/61/91 - 100


# -- the portfolio manager's run ledger values each flow on its own tracker's clock --------------------------------

class _Book:
    """The duck-typed portfolio ``PortfolioLifecycleManager`` drives: positions, environments, removal."""

    def __init__(self, product, spot):
        from types import SimpleNamespace
        self.positions = {"p1": SimpleNamespace(product=product, quantity=1.0, underlying="IDX", engine=None)}
        self.pricing_environments = {"IDX": env(spot, START)}

    def remove_position(self, position_id):
        del self.positions[position_id]


def test_the_manager_pays_a_numeric_flow_on_the_trackers_clock():
    """N = 30 with a 2-day YEAR_FRACTION lag: record 0 knocks out on the first close and pays two days later.
    The run ledger discounts it over 2 days and pays it on run day 2, not 30 days late."""
    from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    from quantark.param import FlatRateCurve

    product = BUILDERS["snowball"]("numeric", _initial(30))
    product.settlement_convention = SettlementConvention(lag=2 / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    product._validate_settlement_convention()
    book = _Book(product, 104.0)
    manager = PortfolioLifecycleManager(base_date=START.to_pydatetime())
    manager.register_positions(book)
    (event,) = [p.event for p in manager.process_day(book, 0, START.to_pydatetime())]
    amount = event.realized_cashflow.amount
    assert event.realized_cashflow.determination_time == 30 / 365.0
    assert manager.pending_receivable_pv == pytest.approx(amount * FlatRateCurve(0.03).get_discount_factor(2 / 365.0))
    for r in (1, 2):
        day = START + pd.Timedelta(days=r)
        book.pricing_environments["IDX"] = env(104.0, day)
        manager.process_day(book, r, day.to_pydatetime())
    assert manager.pending_receivable_pv == 0.0 and manager.paid_cash == amount


def test_the_pnl_explain_recorder_reads_a_numeric_ledger_on_the_trackers_clock():
    """The recorder rebuilds a contract-time state's valuation point for each snapshot day: an autocallable's from
    its contract's clock origin, any other tracker's (a barrier's) from the manager's base date."""
    from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
    from quantark.asset.equity.lifecycle.state import BarrierLifecycleState
    from quantark.pnlexplain.equity.recorder import _state_point

    product = BUILDERS["snowball"]("numeric", _initial(30))
    day = (START + pd.Timedelta(days=5)).to_pydatetime()
    state = AutocallableLifecycleState(valuation_point=ValuationPoint(time=35 / 365.0))
    assert _state_point(state, day, START.to_pydatetime(), product).time == 35 / 365.0
    barrier = BarrierLifecycleState(valuation_point=ValuationPoint(time=5 / 365.0))
    assert _state_point(barrier, day, START.to_pydatetime(), product).time == 5 / 365.0


# -- the replay dashboard ages its copy on the same clock --------------------------------------------------------------

@pytest.mark.parametrize("gap", [30, 100])
def test_the_dashboard_copy_is_the_trackers_copy(gap):
    """The dashboard re-ages the booked contract itself; it must hold what the tracker's copy holds."""
    from types import SimpleNamespace

    from quantark.backtest.replay.dashboard import AutocallableBacktestDashboard

    product = BUILDERS["snowball"]("numeric", _initial(gap))
    day = START + pd.Timedelta(days=10)
    dash = object.__new__(AutocallableBacktestDashboard)
    dash.results = SimpleNamespace(config=SimpleNamespace(product=product, start_date=START.to_pydatetime()))
    aged = dash._product_for_date(day, env(100.0, day), pd.DataFrame({"knocked_in": [False]}, index=[day]))
    reference = tracker_for(product, quantity=1.0, start_date=START).product_for_pricing(day, env(100.0, day))
    assert aged.maturity == reference.maturity == pytest.approx((365 - gap - 10) / 365.0)
    # the dashboard ages the booked (unsnapped) k/12 times, the tracker its snapped days: same records, < 1 day apart
    held = [r.observation_time for r in aged.barrier_config.ko_observation_schedule.records]
    assert held == pytest.approx(
        [r.observation_time for r in reference.barrier_config.ko_observation_schedule.records], abs=0.5 / 365.0)
