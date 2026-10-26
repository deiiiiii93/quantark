"""A numeric note under the clock rule (2026-10-26; ship spec AC6).

A contract is all dates or all floats. A numeric Snowball / Phoenix / KO-reset carries no ``initial_date``: its times
count from the day it is booked on, so a note struck before today is booked aged to today (``time_shift``), the form
the lifecycle trackers' copies already have. Point-in-time valuation therefore never ages the product:
``EquityPosition.on_valuation_clock`` only rolls a dated trade's ``CONTRACT``-clock cash legs, and the numeric-note
ageing of 2026-10-24 (``age_note_to_valuation_date``) is gone. Every tool that reads the note off a position (the
static PnL explain's book snapshot, the trade theta split, SIMM's vega tenor) reads it as booked.
"""
import copy
from datetime import timedelta

import pytest

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.backtest import BacktestEngine
from quantark.cashleg import DeterministicLeg, EventType, LegDirection
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.pnlexplain.equity.portfolio import BookSnapshot
from quantark.portfolio import EquityPosition, Portfolio
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close
from test_backtest import test_trade_value_runs as runs
from test_portfolio import test_leg_clock_initial_date as dated

D = 1 / 365.0
START = runs.START
AGE = 14
OBS = (28, 56, 84)                               # days from the strike, AGE days before START


def _note():
    """Struck AGE days before START, booked on the numeric clock alone: aged to START."""
    note = runs._snowball()                      # numeric: KO at OBS days, maturity 84 days
    note.time_shift(AGE * D, START, dated._env(valuation_date=START), keep_landing=True)
    return note


def _position(product=None, legs=(), **kwargs):
    return EquityPosition(product=product if product is not None else _note(), quantity=1.0, entry_price=0.0,
                          underlying=runs.UNDERLYING, engine=SnowballQuadEngine(params=QuadParams()),
                          entry_timestamp=START, cash_legs=list(legs), entry_trade_value=-500.0, **kwargs)


def _legs():
    """The trade's legs as seen from START (a numeric trade has no initial_date to count them from)."""
    obs = tuple((d - AGE) * D for d in OBS)
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=7 * D, leg_id="fee"),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO,
                       leg_id="ko"),
        AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                            notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                            settlement_schedule=obs, terminal_accrual_factor=obs[-1],
                            terminal_settlement_time=obs[-1], accrual_basis=AccrualBasis.KO_MATURITY,
                            terminal_events=frozenset({EventType.MATURITY_NO_KO}), leg_id="rebate"),
    ]


def test_the_numeric_note_ageing_is_gone():
    from quantark.asset.equity.lifecycle import ko_reset
    assert not hasattr(ko_reset, "age_note_to_valuation_date")


def test_a_numeric_note_dated_from_an_initial_date_is_refused():
    note = runs._snowball()
    note.initial_date = START - timedelta(days=AGE)
    with pytest.raises(ValidationError, match="numeric clock .*initial_date"):
        note.validate()


def test_on_valuation_clock_leaves_a_numeric_position_as_booked():
    """No initial_date: nothing to roll, the product untouched, the legs already counted from the valuation date."""
    env = dated._env(valuation_date=START + timedelta(days=3))
    for position in (_position(), _position(legs=_legs())):
        assert position.on_valuation_clock(env) is position


def test_on_valuation_clock_rolls_a_dated_trades_legs_but_never_its_product():
    position = dated._position()
    aged = position.on_valuation_clock(dated._env())
    assert aged is not position and aged.product is position.product


def test_the_book_snapshot_holds_the_note_as_booked():
    env = dated._env(valuation_date=START)
    pf = Portfolio(portfolio_name="numeric", pricing_environments={runs.UNDERLYING: env}, creation_date=START)
    pf.positions["note"] = _position()
    snap = BookSnapshot.from_portfolio(pf, START).positions["note"].snapshot
    assert snap.product is pf.positions["note"].product


@pytest.mark.parametrize("trade_value", [False, True])
def test_an_aged_numeric_notes_day_zero_matches_the_backtest(trade_value):
    pf = Portfolio(portfolio_name="numeric", pricing_environments={runs.UNDERLYING: dated._env()},
                   creation_date=START, trade_value=trade_value)
    pf.positions["note"] = _position(legs=_legs() if trade_value else ())
    value = copy.deepcopy(pf).get_portfolio_value()
    position = _position(legs=_legs() if trade_value else ())
    backtest = BacktestEngine(runs._config(trade_value=trade_value, positions=[position])).run()
    assert is_close(backtest.states_df["portfolio_value"].iloc[0], value, rel_tol=1e-12)
