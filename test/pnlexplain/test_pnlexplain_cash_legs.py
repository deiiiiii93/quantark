"""PnL explain with cash legs in trade mode (ship spec 2026-10-23 portfolio-trade-value D18; AC12).

The equity backtest and replay recorders add one ``cash_legs`` row per day: the change in the live legs' PV, the
change in leg receivable PV and the leg cash paid that day. The product explain is unchanged and the explain
total reconciles to the engine's state PnL every day.
"""
import pandas as pd
import pytest

from quantark.backtest import BacktestEngine
from quantark.backtest.replay import ReplayBacktestEngine
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig
from test_backtest import test_replay_trade_value as replay_fx
from test_backtest import test_trade_value_runs as backtest_fx

WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _assert_reconciles(recon):
    port = recon[recon["level"] == "portfolio"]
    assert not port.empty
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()


def _cash_leg_rows(explain):
    return explain[(explain["factor"] == "cash_legs") & (explain["level"] == "portfolio")]


def test_backtest_explain_has_a_cash_legs_line_and_reconciles_to_state_pnl():
    config = backtest_fx._config()
    config.pnl_explain = WF
    results = BacktestEngine(config).run()
    _assert_reconciles(results.explain_reconciliation_df)
    rows = _cash_leg_rows(results.explain_df)
    assert not rows.empty
    assert (rows["pnl"].abs() > 0.0).any()


def test_backtest_explain_without_legs_has_no_cash_legs_line():
    config = backtest_fx._config(trade_value=False)
    config.pnl_explain = WF
    results = BacktestEngine(config).run()
    _assert_reconciles(results.explain_reconciliation_df)
    assert _cash_leg_rows(results.explain_df).empty


@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_replay_explain_has_a_cash_legs_line_and_reconciles_to_state_pnl(kind):
    config = replay_fx._config(kind, days=replay_fx.PATH + replay_fx.TAIL)
    config.pnl_explain = WF
    result = ReplayBacktestEngine(config).run()
    _assert_reconciles(result.explain_reconciliation_df())
    rows = _cash_leg_rows(result.explain_df())
    assert not rows.empty and (rows["pnl"].abs() > 0.0).any()


def test_replay_explain_flag_off_is_unchanged_by_legs():
    with_legs = replay_fx._config(trade_value=False)
    with_legs.pnl_explain = WF
    legless = replay_fx._config(trade_value=False, legs=[])
    legless.pnl_explain = WF
    a = ReplayBacktestEngine(with_legs).run()
    b = ReplayBacktestEngine(legless).run()
    pd.testing.assert_frame_equal(a.explain_df(), b.explain_df(), check_exact=True)
    assert _cash_leg_rows(a.explain_df()).empty
