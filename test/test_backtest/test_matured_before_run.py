"""A contract whose maturity falls before the run's start (user ruling 2026-10-09, on the W1 hand-back).

The tracker does not settle it on the run's first close: it warns once (``MaturedBeforeRunWarning``, naming the
contract, its maturity and the run start) and the lifecycle returns zeroes -- no terminal cash in the ledger, zero PV
on every day of the run. "Before" is the decision-2 rule: the terminal's date strictly before ``start_date``; a
maturity ON ``start_date`` is today's and is settled on the first close. A numeric contract behaves exactly like its
dated twin.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.lifecycle.ko_reset import tracker_for

from test_tracker_clock_parity import BUILDERS, START, env

FORMS = ("numeric", "dated")
#: the day each contract's terminal falls due, in days after its initial_date (a KO-reset not knocked in matures
#: on schedule 1's last record, contract day 182)
MATURITY_DAY = {"snowball": 365, "phoenix": 365, "ko_reset_absolute": 182, "ko_reset_rebased": 182}


def _warning_class():
    from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
    return MaturedBeforeRunWarning


def _tracker(kind, form, gap, **kw):
    product = BUILDERS[kind](form, START - pd.Timedelta(days=gap))
    for name, value in kw.items():
        setattr(product, name, value)
    return tracker_for(product, quantity=1.0, start_date=START)


def _close(t, r, spot):
    day = START + pd.Timedelta(days=r)
    e = env(spot, day)
    product = t.product_for_lifecycle()
    events = t.observe(day, product, e, spot)
    maturity = t.settle_maturity_if_due(day, product, e, spot)
    return events + ([maturity] if maturity is not None else [])


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("kind", list(MATURITY_DAY))
def test_a_contract_matured_before_the_run_books_nothing_and_warns(kind, form):
    t = _tracker(kind, form, MATURITY_DAY[kind] + 1)                # matured the day before the run
    with pytest.warns(_warning_class()) as caught:
        assert _close(t, 0, 90.0) == []
    assert len(caught) == 1
    message = str(caught[0].message)
    assert type(t.product).__name__ in message and str(START.date()) in message
    assert str((START - pd.Timedelta(days=1)).date()) in message
    assert not t.lifecycle.alive and t.lifecycle.matured and t.lifecycle.settled
    assert not t.lifecycle.ledger.cashflows and t.lifecycle.realized_cashflows == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # one warning per run
        assert _close(t, 1, 90.0) == []


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("kind", list(MATURITY_DAY))
def test_a_maturity_on_the_run_start_is_settled_on_the_first_close(kind, form):
    t = _tracker(kind, form, MATURITY_DAY[kind])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        events = _close(t, 0, 90.0)
    assert [e.event_type.value for e in events][-1] == "MATURITY"
    assert [cf.cashflow_id for cf in t.lifecycle.ledger.cashflows][-1] == "maturity"


@pytest.mark.parametrize("form", FORMS)
def test_a_ko_reset_knocked_in_before_the_run_matures_at_its_knocked_in_end(form):
    """A carried knock-in makes the knocked-in end the maturity: ABSOLUTE schedule 2 ends on contract day 365."""
    carried = {"_otc_lifecycle_knocked_in": True,
               "_otc_lifecycle_knock_in_date": (START - pd.Timedelta(days=300)).to_pydatetime()}
    alive = _tracker("ko_reset_absolute", form, 300, **carried)     # schedule 1 ended, knocked-in end ahead
    assert _close(alive, 0, 90.0) == [] and alive.lifecycle.alive
    dead = _tracker("ko_reset_absolute", form, 366, **carried)
    with pytest.warns(_warning_class()):
        assert _close(dead, 0, 90.0) == []
    assert not dead.lifecycle.alive and not dead.lifecycle.ledger.cashflows


def test_an_unknocked_ko_reset_matured_with_schedule_one_although_schedule_two_runs_on():
    """R2: not knocked in when schedule 1 ended (contract day 182), it matured there, before a run 300 days in."""
    t = _tracker("ko_reset_absolute", "numeric", 300)
    with pytest.warns(_warning_class()):
        assert _close(t, 0, 90.0) == []
    assert not t.lifecycle.alive


# -- the portfolio manager: the position leaves the book before the run prices it -----------------------------------

class _Book:
    def __init__(self, product):
        from types import SimpleNamespace
        self.positions = {"p1": SimpleNamespace(product=product, quantity=1.0, underlying="IDX", engine=None)}
        self.pricing_environments = {"IDX": env(90.0, START)}

    def remove_position(self, position_id):
        del self.positions[position_id]


@pytest.mark.parametrize("form", FORMS)
def test_the_manager_drops_a_matured_position_with_no_cash(form):
    from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager

    book = _Book(BUILDERS["snowball"](form, START - pd.Timedelta(days=400)))
    manager = PortfolioLifecycleManager(base_date=START.to_pydatetime())
    with pytest.warns(_warning_class()):
        manager.register_positions(book)
    assert book.positions == {}
    assert manager.process_day(book, 0, START.to_pydatetime()) == []
    assert not manager.ledger.cashflows
    assert manager.pending_receivable_pv == 0.0 and manager.paid_cash == 0.0


# -- the replay: zero initial value, zero marks, no maturity action ---------------------------------------------------

def test_the_replay_values_a_matured_contract_at_zero_on_every_day():
    from test_book_backtest import _single_config, _synthetic_market
    from quantark.backtest.replay import AutocallableBacktestEngine

    config = _single_config(_synthetic_market(end="2024-01-31"))
    config.end_date = datetime(2024, 1, 31)
    config.product.initial_date = datetime(2022, 12, 1)               # matured 2023-12-01
    with pytest.warns(_warning_class()):
        results = AutocallableBacktestEngine(config).run()
    states = results.states_df
    assert len(states) == 22
    for column in ("product_mtm", "cashflows", "pending_receivable_pv", "paid_cash", "portfolio_value"):
        assert (states[column] == 0.0).all(), column
    assert not states["alive"].any() and states["matured"].all()
    summary = results.get_summary()
    assert summary["initial_portfolio_value"] == 0.0 and summary["total_pnl"] == 0.0
    assert summary["num_actions"] == 0 and results.actions_df.empty


# -- the simulated paths: the product is dead from day 0 --------------------------------------------------------------

def test_the_calendar_schedule_flags_a_contract_matured_before_the_run():
    t = _tracker("snowball", "dated", 400)
    dates = pd.date_range(START, periods=30, freq="D")
    assert t.resolve_calendar_schedule(dates, env(100.0, START)).matured_before_run
    fresh = _tracker("snowball", "dated", 365)
    assert not fresh.resolve_calendar_schedule(dates, env(100.0, START)).matured_before_run


def test_the_vector_lifecycle_books_nothing_for_a_contract_matured_before_the_run():
    from quantark.backtest.simulation.lifecycle import VectorLifecycle

    dates = pd.date_range(START, periods=30, freq="D")
    schedule = _tracker("snowball", "numeric", 400).resolve_calendar_schedule(dates, env(100.0, START))
    vec = VectorLifecycle([schedule], np.array([1.0]), n_paths=2)
    assert not vec.alive.any()
    for d in range(30):
        assert vec.step(d, np.array([90.0, 110.0]), lambda p, idx, ki: np.full(idx.size, 7.0)) == []
    assert (vec.realized == 0.0).all() and (vec.pending == 0.0).all()


def test_the_ensemble_values_a_contract_matured_before_the_run_at_zero():
    from quantark.backtest.replay import ReplayProduct
    from quantark.backtest.simulation.engine import EnsembleBacktestEngine
    from simulation.conftest import ensemble_config, short_snowball
    from simulation.test_engine import _paths

    product = short_snowball()
    product.initial_date = datetime(2023, 11, 1)                      # a 6-day contract, long matured
    cfg = ensemble_config(products=[ReplayProduct(product=product, quantity=-1000.0, position_id=1,
                                                  has_lifecycle=True)])
    with pytest.warns(_warning_class()):
        results = EnsembleBacktestEngine(cfg).run(_paths())
    assert results.initial_book_value == pytest.approx(0.0)
    assert (results.cube.product_mtm == 0.0).all()
    assert results.events == []
