"""Tools that start from an ``EquityPosition`` see its numeric note on the valuation clock (user decision 2026-10-26).

A numeric note's times count from its ``initial_date``; ``EquityPosition.on_valuation_clock`` ages it to the
valuation date (``test_numeric_note_clock``). Every tool that reads the note off a position takes it through that
chokepoint: the static PnL explain's book snapshot, the trade Greeks' theta split and SIMM's vega tenor. A raw
``engine.price(product, env)`` still prices the product as given.
"""
import copy
from datetime import timedelta

import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.pnlexplain.equity.portfolio import BookSnapshot, explain_portfolio
from quantark.portfolio import Portfolio
from quantark.simm.engines.risk_class.equity_engine import EquitySensitivityEngine
from test_portfolio import test_leg_clock_initial_date as dated
from test_portfolio import test_numeric_note_clock as nn

UNDERLYING = nn.runs.UNDERLYING
AGED_MATURITY = (nn.OBS[-1] - nn.AGE) / 365.0          # the note matures 84 days after INITIAL, 14 days ago


def _book(day=0, trade_value=False, initial_date=nn.INITIAL):
    env = dated._env(valuation_date=nn.START + timedelta(days=day))
    pf = Portfolio(portfolio_name="numeric", pricing_environments={UNDERLYING: env}, creation_date=nn.START,
                   trade_value=trade_value)
    pf.positions["note"] = nn._position(product=nn._note(initial_date=initial_date))
    return pf


def _snapshot(pf):
    return BookSnapshot.from_portfolio(pf, pf.pricing_environments[UNDERLYING].valuation_date)


def test_the_book_snapshot_holds_the_note_on_the_valuation_clock():
    pf = _book()
    snap = _snapshot(pf).positions["note"].snapshot
    assert snap.product.get_maturity(snap.pricing_env) == AGED_MATURITY
    assert pf.positions["note"].product.accrual_config.accrued_offset == 0.0       # the booked note is untouched


def test_a_static_explain_of_an_aged_numeric_note_explains_the_portfolio_pnl():
    pf0, pf1 = _book(0), _book(1)
    result = explain_portfolio(_snapshot(pf0), _snapshot(pf1))
    want = copy.deepcopy(pf1).get_portfolio_value() - copy.deepcopy(pf0).get_portfolio_value()
    assert result.total_pnl == pytest.approx(want, rel=1e-12)


def test_a_note_without_an_initial_date_is_snapshotted_as_booked():
    pf = _book(initial_date=None)
    assert _snapshot(pf).positions["note"].snapshot.product is pf.positions["note"].product


def test_the_trade_theta_split_reads_the_aged_maturity():
    gc = GreeksCalculator()
    pf = _book(trade_value=True)
    by_hand = _book(trade_value=True)
    by_hand.positions["note"] = nn._position(product=nn._tracker_copy(nn._note()))
    got = pf.position_greeks(pf.positions["note"], gc)
    want = by_hand.position_greeks(by_hand.positions["note"], gc)
    for name in ("convexity_theta", "r_theta", "q_theta"):
        assert got[name] == want[name], name


def test_simm_reads_the_vega_tenor_off_the_aged_note():
    pf = _book()
    engine = EquitySensitivityEngine.__new__(EquitySensitivityEngine)
    env = pf.pricing_environments[UNDERLYING]
    assert engine._get_option_expiry_tenor(pf.positions["note"], env) == AGED_MATURITY
