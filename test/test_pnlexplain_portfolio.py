"""Spec test 5b: the §9 case table."""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain import explain
from quantark.pnlexplain.base import ExplainMethod, Factor, RowKind
from quantark.pnlexplain.equity.portfolio import (
    BookSnapshot, PositionSnapshot, QuotedLegSnapshot, explain_portfolio, explain_position, explain_quoted_leg,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, value
from quantark.pnlexplain.equity.trades import ExplainTrade
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 6, 26)
T1 = datetime(2026, 6, 29)
ENG = BlackScholesEngine()


def _env(spot, date, vol=0.2):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=date)


E0, E1 = _env(100.0, T0), _env(103.0, T1, vol=0.22)


def _call(strike=100.0, days=0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=1.0 - days / 365)


def _pos(pid, product_t0, q, env=E0, date=T0, state=None, tombstone=False, product_t1=None):
    prod = product_t0 if date == T0 else (product_t1 or _call(product_t0.strike, days=3))
    snap = ValuationSnapshot(prod, ENG, env, date=date, quantity=q, lifecycle_state=state)
    return PositionSnapshot(position_id=pid, underlying="IDX", snapshot=snap, tombstone=tombstone)


def _unit_t1(product_t1):
    return ENG.price(product_t1, E1)


def _tol(x):
    return 1e-10 * max(1.0, abs(x))


def test_unchanged_position_promotes_instrument_rows():
    p0 = _pos("a", _call(), 2.0)
    p1 = _pos("a", _call(), 2.0, env=E1, date=T1)
    res = explain_position(p0, p1)
    inst = explain(p0.snapshot, p1.snapshot)
    assert res.instrument.total_pnl == pytest.approx(inst.total_pnl)
    assert all(r.level == "position" for r in res.rows)
    assert sum(1 for r in res.rows if r.kind is RowKind.SUMMARY) == 1
    assert res.total_pnl == pytest.approx(inst.total_pnl)
    assert not res.trade_rows


def test_quantity_change_with_trade_reconciles():
    p0 = _pos("h", _call(), 2.0)
    p1 = _pos("h", _call(), 3.0, env=E1, date=T1)
    u1 = _unit_t1(p1.snapshot.product)
    trade = ExplainTrade(position_id="h", quantity=1.0, price=u1 - 0.1, kind="adjust", timestamp=T1)
    res = explain_position(p0, p1, trades=[trade])
    assert len(res.trade_rows) == 1 and res.trade_rows[0].pnl == pytest.approx(0.1)
    expected = value(p1.snapshot).total - value(p0.snapshot).total + trade.cash
    assert res.total_pnl == pytest.approx(expected)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(expected))
    with pytest.raises(ValidationError):            # trades do not reconcile
        explain_position(p0, p1, trades=[ExplainTrade("h", 2.0, u1)])


def test_opened_today_and_closed_by_trading_and_terminal_tombstone():
    p1 = _pos("n", _call(), 4.0, env=E1, date=T1)
    u1 = _unit_t1(p1.snapshot.product)
    new = explain_position(None, p1, trades=[ExplainTrade("n", 4.0, u1 + 0.05, kind="open")])
    assert new.instrument is None
    assert new.total_pnl == pytest.approx(4.0 * u1 - 4.0 * (u1 + 0.05))
    with pytest.raises(ValidationError):
        explain_position(None, p1)

    p0 = _pos("c", _call(), 4.0)
    tomb = _pos("c", _call(), 4.0, env=E1, date=T1, tombstone=True)
    closed = explain_position(p0, tomb, trades=[ExplainTrade("c", -4.0, u1 - 0.2, kind="close")])
    v0 = value(p0.snapshot).total
    assert closed.total_pnl == pytest.approx(-v0 + 4.0 * (u1 - 0.2))
    assert closed.reconcile(ExplainMethod.WATERFALL) == pytest.approx(0.0, abs=_tol(v0))
    assert closed.trade_rows[0].pnl == pytest.approx(-4.0 * 0.2)
    with pytest.raises(ValidationError):            # non-terminal tombstone without trades
        explain_position(p0, tomb)

    st = AutocallableLifecycleState()
    st.mark_ko(T1, cashflow=12.0)
    from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
    before = AutocallableLifecycleState()
    t0 = _pos("k", _call(), 4.0, state=before)
    t1 = _pos("k", _call(), 4.0, env=E1, date=T1, state=st, tombstone=True)
    from quantark.asset.equity.lifecycle.events import LifecycleEvent, LifecycleEventType
    ev = LifecycleEvent(event_type=LifecycleEventType.KNOCK_OUT, date=T1, spot=103.0, cashflow=12.0,
                        terminates_position=True)
    tr = LifecycleTransition(product_alive_t1=t1.snapshot.product, engine_alive_t1=ENG,
                             state_before=before, state_after=st, events=(ev,))
    term = explain_position(t0, t1, transition=tr)
    assert term.total_pnl == pytest.approx(12.0 - value(t0.snapshot).total)
    with pytest.raises(ValidationError):            # trades on a terminal tombstone
        explain_position(t0, t1, transition=tr, trades=[ExplainTrade("k", -4.0, 1.0)])


def test_quoted_leg_rolls_with_contract_specific_ids():
    a0 = QuotedLegSnapshot("hedge:IF2401", "IDX", units=300.0, price=100.4, spot=100.0, date=T0)
    a1 = QuotedLegSnapshot("hedge:IF2401", "IDX", units=300.0, price=103.3, spot=103.0, date=T1, tombstone=True)
    b1 = QuotedLegSnapshot("hedge:IF2402", "IDX", units=300.0, price=104.0, spot=103.0, date=T1)
    close = ExplainTrade.from_contracts("hedge:IF2401", contracts=-1.0, price=103.3, multiplier=300.0, kind="roll_close")
    open_ = ExplainTrade.from_contracts("hedge:IF2402", contracts=1.0, price=104.0, multiplier=300.0, kind="roll_open")
    old = explain_quoted_leg(a0, a1, trades=[close])
    assert old.total_pnl == pytest.approx(300.0 * (103.3 - 100.4))
    rows = {r.term: r for r in old.rows if r.kind is RowKind.COMPONENT}
    assert rows["spot"].pnl == pytest.approx(300.0 * 3.0)
    assert rows["basis"].pnl == pytest.approx(300.0 * (2.9 - 3.0))
    assert rows["trade:roll_close"].pnl == pytest.approx(0.0)
    new = explain_quoted_leg(None, b1, trades=[open_])
    assert new.total_pnl == pytest.approx(0.0)


def test_portfolio_aggregation_costs_and_reconciliation():
    pos0 = {"a": _pos("a", _call(100.0), 2.0), "b": _pos("b", _call(110.0), -1.0), "h": _pos("h", _call(), 2.0)}
    pos1 = {"a": _pos("a", _call(100.0), 2.0, env=E1, date=T1),
            "b": _pos("b", _call(110.0), -1.0, env=E1, date=T1),
            "h": _pos("h", _call(), 3.0, env=E1, date=T1)}
    b0 = BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0})
    b1 = BookSnapshot(date=T1, positions=pos1, environments={"IDX": E1})
    u1 = _unit_t1(pos1["h"].snapshot.product)
    trades = [ExplainTrade("h", 1.0, u1, transaction_cost=0.5, timestamp=T1)]
    res = explain_portfolio(b0, b1, trades=trades, transaction_costs=0.25)
    cost = [r for r in res.rows if r.factor is Factor.TRANSACTION_COST][0]
    assert cost.pnl == pytest.approx(-0.75) and cost.method is ExplainMethod.SHARED
    expected = sum(res.positions[k].total_pnl for k in res.positions) - 0.75
    assert res.total_pnl == pytest.approx(expected)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(expected))
        assert res.metadata["reconciliation"][m.value]["ok"] is True
    # two strikes under one underlying: aggregated display columns are null
    book_rows = {(r.factor, r.term): r for r in res.rows if r.method is ExplainMethod.TAYLOR}
    assert book_rows[(Factor.VOL, "vega")].greek is None and book_rows[(Factor.VOL, "vega")].moves == {}
    per_u = res.by_underlying()["IDX"]
    assert all(r.level == "portfolio" for r in per_u)
    frame = res.to_frame()
    assert set(frame["level"]) == {"position", "portfolio"}
    assert {"a", "b", "h", "portfolio"} <= set(frame["position_id"])
    # single-position book keeps the display columns
    single = explain_portfolio(BookSnapshot(T0, {"a": pos0["a"]}, {"IDX": E0}),
                               BookSnapshot(T1, {"a": pos1["a"]}, {"IDX": E1}))
    srow = [r for r in single.rows if r.method is ExplainMethod.TAYLOR and r.term == "vega"][0]
    assert srow.greek is not None and "vol_pts" in srow.moves


def test_from_portfolio_snapshots_rolled_products_without_mutation():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.lifecycle.manager import PortfolioLifecycleManager
    from quantark.asset.equity.param import PDEParams
    from quantark.portfolio import Portfolio
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_pnlexplain_lifecycle_days import _snowball, _env as _lc_env

    start = datetime(2024, 1, 1)
    day = datetime(2024, 1, 31)
    portfolio = Portfolio(portfolio_name="book", pricing_environments={"IDX": _lc_env(start, 100.0)})
    note = portfolio.add_position(product=_snowball(), quantity=-3.0, entry_price=0.0, underlying="IDX",
                                  engine=SnowballPDESolver(PDEParams(accuracy="fast")), entry_timestamp=start)
    manager = PortfolioLifecycleManager(base_date=start)
    manager.register_positions(portfolio)
    portfolio.pricing_environments["IDX"] = _lc_env(day, 101.0)
    manager.process_day(portfolio, day_index=30, day_date=day)          # substitutes the rolled product
    before = {pid: (p.product, p.engine, p.quantity) for pid, p in portfolio.positions.items()}
    book = BookSnapshot.from_portfolio(portfolio, day)
    snap = book.positions[note.position_id].snapshot
    assert snap.product is portfolio.positions[note.position_id].product     # the rolled product
    assert snap.product.maturity == pytest.approx(1.0 - 30 / 365)
    assert snap.pricing_env is not portfolio.pricing_environments["IDX"]     # a copy, same date
    assert snap.pricing_env.valuation_date == day and snap.lifecycle_state is not note.lifecycle_state
    assert {pid: (p.product, p.engine, p.quantity) for pid, p in portfolio.positions.items()} == before
    with pytest.raises(ValidationError):                                      # env dated differently
        BookSnapshot.from_portfolio(portfolio, day + timedelta(days=1))


def test_portfolio_errors():
    pos0 = {"a": _pos("a", _call(), 2.0)}
    b0 = BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0})
    with pytest.raises(ValidationError):            # absent at t1, no tombstone
        explain_portfolio(b0, BookSnapshot(date=T1, positions={}, environments={"IDX": E1}))
    with pytest.raises(ValidationError):            # new at t1 without trades
        explain_portfolio(BookSnapshot(date=T0, positions={}, environments={"IDX": E0}),
                          BookSnapshot(date=T1, positions={"a": _pos("a", _call(), 2.0, env=E1, date=T1)},
                                       environments={"IDX": E1}))
    with pytest.raises(ValidationError):            # currency mismatch
        explain_portfolio(BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0}, currency="CNY"),
                          BookSnapshot(date=T1, positions={"a": _pos("a", _call(), 2.0, env=E1, date=T1)},
                                       environments={"IDX": E1}, currency="USD"))
    with pytest.raises(ValidationError):            # trades for an unknown position id
        explain_portfolio(b0, BookSnapshot(date=T1, positions={"a": _pos("a", _call(), 2.0, env=E1, date=T1)},
                                           environments={"IDX": E1}),
                          trades=[ExplainTrade("zzz", 1.0, 1.0)])


def test_book_infers_and_rejects_currencies():
    usd = ValuationSnapshot(_call(), ENG, E0, date=T0, currency="USD")
    eur = ValuationSnapshot(_call(), ENG, E0, date=T0, currency="EUR")
    book = BookSnapshot(date=T0, positions={"a": PositionSnapshot("a", "IDX", usd)}, environments={"IDX": E0})
    assert book.currency == "USD"                                          # inferred from the positions
    with pytest.raises(ValidationError):                                   # mixed labels, no book label
        BookSnapshot(date=T0, positions={"a": PositionSnapshot("a", "IDX", usd),
                                         "b": PositionSnapshot("b", "IDX", eur)}, environments={"IDX": E0})
    with pytest.raises(ValidationError):                                   # the book label disagrees
        BookSnapshot(date=T0, positions={"a": PositionSnapshot("a", "IDX", usd)}, environments={"IDX": E0},
                     currency="CNY")


def test_trading_tombstone_consumes_the_whole_old_position():
    p0 = _pos("c", _call(), 10.0)
    half = _pos("c", _call(), 5.0, env=E1, date=T1, tombstone=True)        # written with a stale quantity
    u1 = _unit_t1(half.snapshot.product)
    with pytest.raises(ValidationError):                                   # 5 sold, but the position held 10
        explain_position(p0, half, trades=[ExplainTrade("c", -5.0, u1)])
    res = explain_position(p0, half, trades=[ExplainTrade("c", -10.0, u1 - 0.2)])
    v0 = value(p0.snapshot).total
    assert res.total_pnl == pytest.approx(-v0 + 10.0 * (u1 - 0.2))
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=_tol(v0))


def test_quoted_leg_validation():
    with pytest.raises(ValidationError):                                   # zero spot
        QuotedLegSnapshot("hedge:X", "IDX", units=300.0, price=100.0, spot=0.0, date=T0)
    with pytest.raises(ValidationError):                                   # non-finite price
        QuotedLegSnapshot("hedge:X", "IDX", units=300.0, price=float("nan"), spot=100.0, date=T0)
    with pytest.raises(ValidationError):                                   # a live leg with zero units
        QuotedLegSnapshot("hedge:X", "IDX", units=0.0, price=100.0, spot=100.0, date=T0)
    leg = QuotedLegSnapshot("hedge:X", "IDX", units=300.0, price=100.0, spot=100.0, date=T0)
    with pytest.raises(ValidationError):                                   # key != position_id
        BookSnapshot(date=T0, positions={}, environments={"IDX": E0}, quoted_legs={"hedge:Y": leg})
    with pytest.raises(ValidationError):                                   # off-date leg
        BookSnapshot(date=T1, positions={}, environments={"IDX": E1}, quoted_legs={"hedge:X": leg})
    new = QuotedLegSnapshot("hedge:X", "IDX", units=300.0, price=101.0, spot=101.0, date=T1)
    with pytest.raises(ValidationError):                                   # a foreign trade on a new leg
        explain_quoted_leg(None, new, trades=[ExplainTrade("hedge:Z", 300.0, 101.0)])
    with pytest.raises(ValidationError):                                   # a trade dated after the leg
        explain_quoted_leg(None, new, trades=[ExplainTrade("hedge:X", 300.0, 101.0,
                                                           timestamp=T1 + timedelta(days=1))])


def test_portfolio_rejects_trades_outside_the_step():
    pos0 = {"a": _pos("a", _call(), 2.0)}
    b0 = BookSnapshot(date=T0, positions=pos0, environments={"IDX": E0})
    p1 = _pos("n", _call(), 4.0, env=E1, date=T1)
    b1 = BookSnapshot(date=T1, positions={"a": _pos("a", _call(), 2.0, env=E1, date=T1), "n": p1},
                      environments={"IDX": E1})
    u1 = _unit_t1(p1.snapshot.product)
    stale = ExplainTrade("n", 4.0, u1, kind="open", timestamp=T0 - timedelta(days=30))
    with pytest.raises(ValidationError):
        explain_portfolio(b0, b1, trades=[stale])
    ok = ExplainTrade("n", 4.0, u1, kind="open", timestamp=T1)
    assert explain_portfolio(b0, b1, trades=[ok]).positions["n"].instrument is None
