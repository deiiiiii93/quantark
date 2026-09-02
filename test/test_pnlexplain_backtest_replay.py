"""Spec tests 6 (replay: futures hedge with a roll), 8 (LV model row)."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay.engine import ReplayBacktestEngine  # noqa: E402
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig  # noqa: E402
from quantark.pnlexplain.equity.recorder import _replay_trade, trade_kind  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))
D = datetime(2024, 1, 2)


def _row(**kw):
    base = dict(date=D, trade_type="hedge_rebalance", instrument_type="futures", contract="IF2401",
                quantity=2.0, price=101.5, multiplier=300.0, notional=60900.0, transaction_cost=0.0, reason="x")
    base.update(kw)
    return base


def test_replay_book_explain_off_identical_and_on_reconciles():
    off = ReplayBacktestEngine(fixtures.make_book_config()).run()
    cfg = fixtures.make_book_config()
    cfg.pnl_explain = WF
    on = ReplayBacktestEngine(cfg).run()
    pd.testing.assert_frame_equal(on.states_df(), off.states_df())
    pd.testing.assert_frame_equal(on.greeks_df(), off.greeks_df())
    pd.testing.assert_frame_equal(on.trades_df(), off.trades_df())
    pd.testing.assert_frame_equal(on.actions_df(), off.actions_df())
    assert off.explain_df().empty and off.explain_reconciliation_df().empty
    recon = on.explain_reconciliation_df()
    port = recon[recon["level"] == "portfolio"]
    assert len(port) == len(on.states_df()) - 1
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    ex = on.explain_df()
    hedge_rows = ex[ex["position_id"].str.startswith("hedge:")]
    assert set(hedge_rows["term"]) >= {"spot", "basis"}
    pos = ex[ex["level"] == "position"]
    assert (pos["position_id"].str.startswith("hedge:") | pos["position_id"].isin(["1", "2"])).all()


def test_replay_roll_produces_close_and_open_legs():
    from quantark.backtest.futures_ledger import FuturesRollPolicy
    # The default policy (roll 5 days before expiry) picks IF2402 from day one, so nothing ever
    # rolls in the 5-day golden window. Rolling 3 days before the 2024-01-07 expiry holds IF2401
    # for two days and rolls into IF2402 on 2024-01-04, the day the book knocks out: the new leg
    # is opened by the roll and closed by the hedge the same afternoon (an intraday round trip).
    cfg = fixtures.make_book_config()
    cfg.hedge.roll_policy = FuturesRollPolicy(roll_days_before_expiry=3)
    cfg.pnl_explain = WF
    results = ReplayBacktestEngine(cfg).run()
    trades = results.trades_df()
    assert set(trades["trade_type"]) >= {"roll_close", "roll_open"}
    ex = results.explain_df()
    assert (ex["term"] == "trade:roll_close").any() and (ex["term"] == "trade:roll_open").any()
    assert (ex["position_id"] == "hedge:IF2401").any() and (ex["position_id"] == "hedge:IF2402").any()
    recon = results.explain_reconciliation_df()
    port = recon[recon["level"] == "portfolio"]
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    # the explain is a pure observer: the same run without it books the same states and trades
    off_cfg = fixtures.make_book_config()
    off_cfg.hedge.roll_policy = FuturesRollPolicy(roll_days_before_expiry=3)
    off = ReplayBacktestEngine(off_cfg).run()
    pd.testing.assert_frame_equal(results.states_df(), off.states_df())
    pd.testing.assert_frame_equal(results.trades_df(), off.trades_df())


def test_single_engine_passthrough():
    cfg = fixtures.make_scalar_bsm_config()
    cfg.pnl_explain = WF
    results = AutocallableBacktestEngine(cfg).run()
    assert not results.explain_df.empty
    assert results.explain_reconciliation_df.query("level == 'portfolio'")["ok"].all()
    off = AutocallableBacktestEngine(fixtures.make_scalar_bsm_config()).run()
    assert off.explain_df.empty and off.explain_reconciliation_df.empty
    pd.testing.assert_frame_equal(results.states_df, off.states_df)


def test_unknown_replay_trade_type_is_rejected():
    with pytest.raises(ValidationError, match="unknown backtest trade type"):
        trade_kind("hedge_swap")
    with pytest.raises(ValidationError, match="unknown backtest trade type"):
        _replay_trade(_row(trade_type="hedge_swap"), D)


def test_nonzero_trade_without_position_id_is_rejected():
    assert _replay_trade(_row(quantity=0.0), D) is None                 # explicit no-trade row
    with pytest.raises(ValidationError, match="contract id"):
        _replay_trade(_row(contract=""), D)
    with pytest.raises(ValidationError, match="contract id"):
        _replay_trade(_row(contract=None), D)
    with pytest.raises(ValidationError, match="lacks 'contract'"):
        _replay_trade({k: v for k, v in _row().items() if k != "contract"}, D)


def test_missing_or_non_finite_trade_price_is_rejected():
    with pytest.raises(ValidationError, match="price"):
        _replay_trade(_row(price=float("nan")), D)
    with pytest.raises(ValidationError, match="lacks 'price'"):
        _replay_trade({k: v for k, v in _row().items() if k != "price"}, D)
    with pytest.raises(ValidationError, match="multiplier"):
        _replay_trade(_row(multiplier=None), D)
    t = _replay_trade(_row(), D)
    assert t.position_id == "hedge:IF2401" and t.quantity == 600.0 and t.kind == "adjust"


def test_localvol_recalibration_lands_in_model_row(tmp_path):
    root = fixtures.write_localvol_history(tmp_path / "lv")
    cfg = fixtures.make_localvol_config(root)
    cfg.pnl_explain = WF
    results = AutocallableBacktestEngine(cfg).run()
    ex = results.explain_df
    pos = ex[(ex["level"] == "position") & (ex["position_id"] == "0")]
    model = pos[pos["term"] == "model"]["pnl"].abs().sum()
    vol = pos[pos["term"] == "vol"]["pnl"].abs().sum()
    total = pos[pos["term"] == "total"]["pnl"].abs().sum()
    assert vol <= 1e-8 * max(1.0, total)
    assert model > 0.0
    assert results.explain_reconciliation_df.query("level == 'portfolio'")["ok"].all()


def test_pnl_explain_field_is_appended_after_metadata_in_both_replay_configs():
    """Positional construction of the configs must keep its existing slots (review finding)."""
    from dataclasses import fields
    from quantark.backtest.replay.config import AutocallableBacktestConfig, ReplayBacktestConfig
    for cfg in (AutocallableBacktestConfig, ReplayBacktestConfig):
        names = [f.name for f in fields(cfg)]
        assert names[-2:] == ["metadata", "pnl_explain"], cfg.__name__
