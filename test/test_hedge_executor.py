"""Simple HedgeExecutor: average-cost accounting (patch spec 2026-09-03 §3)."""
from datetime import datetime

import pytest

from quantark.backtest import ZeroCostModel
from quantark.backtest.equity.hedge_executor import HedgeExecutor
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment

U = "TEST"
T = datetime(2026, 1, 5)


def _env(spot):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name=U), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(rate=0.0), valuation_date=T)


@pytest.fixture
def book():
    env = _env(100.0)
    portfolio = Portfolio(portfolio_name="t", pricing_environments={U: env}, creation_date=T)
    return portfolio, env, HedgeExecutor(portfolio, ZeroCostModel(), "spot")


def _hedge(ex, env, size, spot):
    env.spot_quote = SpotQuote(spot=spot, asset_name=U)
    return ex.execute_hedge(U, size, env, T)


def _identity(portfolio, ex, fills, spot):
    """portfolio_pnl + realized_pnl must equal the cash PnL of the fill sequence marked at `spot`."""
    return portfolio.get_portfolio_pnl() + ex.realized_pnl == pytest.approx(sum(q * (spot - p) for q, p in fills))


def test_open_then_increase_blends_entry_price(book):
    portfolio, env, ex = book
    assert _hedge(ex, env, 10.0, 100.0).trade_type == "open"
    rec = _hedge(ex, env, 10.0, 110.0)
    pos = ex.get_hedge_position(U)
    assert rec.trade_type == "adjust" and rec.metadata["action"] == "increase_hedge"
    assert pos.quantity == 20.0 and pos.entry_price == pytest.approx(105.0)
    assert ex.realized_pnl == 0.0
    assert _identity(portfolio, ex, [(10, 100.0), (10, 110.0)], 110.0)


def test_reduce_realizes_the_closed_lot(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    _hedge(ex, env, 10.0, 110.0)
    rec = _hedge(ex, env, -4.0, 110.0)
    pos = ex.get_hedge_position(U)
    assert rec.metadata["action"] == "reduce_hedge"
    assert pos.quantity == 16.0 and pos.entry_price == pytest.approx(105.0)      # entry unchanged
    assert ex.realized_pnl == pytest.approx(20.0)                                # (110 - 105) * 4
    assert _identity(portfolio, ex, [(10, 100.0), (10, 110.0), (-4, 110.0)], 110.0)


def test_flip_realizes_old_lot_and_reenters(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    _hedge(ex, env, 10.0, 110.0)
    _hedge(ex, env, -4.0, 110.0)
    rec = _hedge(ex, env, -26.0, 120.0)
    pos = ex.get_hedge_position(U)
    assert rec.metadata["action"] == "flip_hedge"
    assert pos.quantity == -10.0 and pos.entry_price == pytest.approx(120.0)
    assert ex.realized_pnl == pytest.approx(20.0 + (120.0 - 105.0) * 16.0)
    assert _identity(portfolio, ex, [(10, 100.0), (10, 110.0), (-4, 110.0), (-26, 120.0)], 120.0)


def test_net_to_zero_closes_and_the_next_hedge_opens_a_new_id(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    old_id = ex.get_hedge_position(U).position_id
    rec = _hedge(ex, env, -10.0, 130.0)                     # exactly flat: was ValidationError before the patch
    assert rec.trade_type == "close" and rec.position_id == old_id
    assert rec.metadata["entry_price_after"] is None
    assert ex.get_hedge_position(U) is None and ex.get_hedge_quantity(U) == 0.0
    assert old_id not in portfolio.positions
    assert ex.realized_pnl == pytest.approx(300.0)
    assert _identity(portfolio, ex, [(10, 100.0), (-10, 130.0)], 130.0)
    rec2 = _hedge(ex, env, 5.0, 130.0)
    assert rec2.trade_type == "open" and rec2.position_id != old_id
    assert ex.get_statistics()["realized_pnl"] == pytest.approx(300.0)


def test_close_hedge_position_realizes(book):
    portfolio, env, ex = book
    _hedge(ex, env, 10.0, 100.0)
    env.spot_quote = SpotQuote(spot=90.0, asset_name=U)
    rec = ex.close_hedge_position(U, env, T)
    assert rec.trade_type == "close" and rec.quantity == -10.0
    assert ex.realized_pnl == pytest.approx(-100.0)
    assert not portfolio.positions
