"""A numeric autocallable's times count from its initial date in static valuation too (user decision 2026-10-24).

A numeric Snowball / Phoenix / KO-reset carries one number per observation that is both its time and its accrual;
``accrued_offset`` banks how far the contract has been aged from its origin. Point-in-time valuation through an
``EquityPosition`` ages the note by ``run_time(initial_date, valuation_date) - accrued_offset`` with the lifecycle
tracker's own steps (``product_for_pricing``, pre-close), so a fresh booking prices like the tracker's day copy and a
copy the tracker already aged (offset == elapsed) is not aged again. Dated contracts, products without an
``initial_date`` and valuations on or before it are unchanged.
"""
import copy
from datetime import timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle.ko_reset import tracker_for
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.backtest import BacktestEngine
from quantark.cashleg import DeterministicLeg, EventType, LegDirection
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.portfolio import EquityPosition, Portfolio
from quantark.util.numerical import is_close
from test_backtest import test_trade_value_runs as runs
from test_portfolio import test_leg_clock_initial_date as dated

D = 1 / 365.0
START = runs.START
AGE = 14
INITIAL = START - timedelta(days=AGE)
OBS = (28, 56, 84)                               # days from INITIAL


def _note(initial_date=INITIAL):
    note = runs._snowball()                      # numeric: KO at OBS days, maturity 84 days, from its own origin
    note.initial_date = initial_date
    note.validate()
    return note


def _position(product=None, legs=(), **kwargs):
    return EquityPosition(product=product if product is not None else _note(), quantity=1.0, entry_price=0.0,
                          underlying=runs.UNDERLYING, engine=SnowballQuadEngine(params=QuadParams()),
                          entry_timestamp=START, cash_legs=list(legs), entry_trade_value=-500.0, **kwargs)


def _tracker_copy(product, date=START):
    env = dated._env(valuation_date=date)
    return tracker_for(product, quantity=1.0, start_date=date).product_for_pricing(date, env, initial=True)


def _legs():
    obs = tuple(d * D for d in OBS)
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=21 * D, leg_id="fee"),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO,
                       leg_id="ko"),
        AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                            notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                            settlement_schedule=obs, terminal_accrual_factor=OBS[-1] * D,
                            terminal_settlement_time=OBS[-1] * D, accrual_basis=AccrualBasis.KO_MATURITY,
                            terminal_events=frozenset({EventType.MATURITY_NO_KO}), leg_id="rebate"),
    ]


# --- the note's clock --------------------------------------------------------------------------------------------

def test_a_fresh_numeric_booking_prices_like_the_trackers_day_copy():
    env = dated._env()
    engine = SnowballQuadEngine(params=QuadParams())
    want = engine.price(_tracker_copy(_note()), env)
    assert _position().get_market_value(env) == want
    assert _position().get_trade_value(env) == want                     # flag on and off agree
    aged = _position().on_valuation_clock(env).product
    assert aged.accrual_config.accrued_offset == AGE / 365.0


def test_the_trackers_copy_is_not_aged_again():
    copy_ = _tracker_copy(_note())
    position = _position(product=copy_)
    assert position.on_valuation_clock(dated._env()) is position


def test_a_partly_aged_booking_ages_by_the_remainder():
    env = dated._env()
    half = _tracker_copy(_note(), date=START - timedelta(days=7))      # aged 7 of the 14 days
    assert half.accrual_config.accrued_offset == 7 / 365.0
    aged = _position(product=half).on_valuation_clock(env).product
    assert is_close(aged.accrual_config.accrued_offset, AGE / 365.0, abs_tol=1e-15)
    engine = SnowballQuadEngine(params=QuadParams())
    assert is_close(engine.price(aged, env), engine.price(_tracker_copy(_note()), env), rel_tol=1e-12)


def test_no_initial_date_a_dated_contract_or_no_age_leave_the_note_as_is():
    env = dated._env()
    for position in (_position(product=_note(initial_date=None)),
                     _position(product=dated._dated_snowball())):
        assert position.on_valuation_clock(env) is position
    fresh = _position()
    assert fresh.on_valuation_clock(dated._env(valuation_date=INITIAL)) is fresh
    assert fresh.on_valuation_clock(dated._env(valuation_date=INITIAL - timedelta(days=3))) is fresh


def test_a_carried_knock_in_survives_the_ageing():
    note = _note()
    note._otc_lifecycle_knocked_in = True
    aged = _position(product=note).on_valuation_clock(dated._env()).product
    assert aged._otc_lifecycle_knocked_in is True
    assert note.accrual_config.accrued_offset == 0.0                   # the booked product is untouched


def test_product_greeks_and_trade_risk_use_the_aged_note():
    env = dated._env()
    gc = GreeksCalculator()
    by_hand = _position(product=_tracker_copy(_note()))
    assert _position().get_greeks(env, gc, use_analytical=False) == by_hand.get_greeks(env, gc, use_analytical=False)
    assert _position().get_trade_risk(env, gc).total == by_hand.get_trade_risk(env, gc).total


# --- one day-0 number across the families --------------------------------------------------------------------------

def _portfolio(trade_value):
    pf = Portfolio(portfolio_name="numeric", pricing_environments={runs.UNDERLYING: dated._env()},
                   creation_date=START, trade_value=trade_value)
    pf.positions["note"] = _position(legs=_legs() if trade_value else ())
    return pf


@pytest.mark.parametrize("trade_value", [False, True])
def test_an_aged_numeric_notes_day_zero_matches_the_backtest(trade_value):
    value = copy.deepcopy(_portfolio(trade_value)).get_portfolio_value()
    position = _position(legs=_legs() if trade_value else ())
    backtest = BacktestEngine(runs._config(trade_value=trade_value, positions=[position])).run()
    assert is_close(backtest.states_df["portfolio_value"].iloc[0], value, rel_tol=1e-12)
