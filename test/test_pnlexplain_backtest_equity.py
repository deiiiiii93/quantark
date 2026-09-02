"""Spec test 6 (equity engine): reconciliation, tombstone across a settlement lag, explain-on isolation."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_backtest_lifecycle import (  # noqa: E402
    ConstantEngine, RampAdapter, _down_out_put_position, UNDERLYING, START,
)
from test_multi_greek_backtest import make_config as make_multi_config  # noqa: E402

from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit  # noqa: E402
from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel  # noqa: E402
from quantark.backtest.strategy import DeltaNeutralStrategy  # noqa: E402
from quantark.backtest.strategy.multi_greek_strategy import DeltaGammaNeutralStrategy  # noqa: E402
from quantark.pnlexplain import ExplainMethod, PnLExplainConfig, FRAME_COLUMNS  # noqa: E402
from quantark.pnlexplain.equity.recorder import RECON_COLUMNS, _trade_from_record  # noqa: E402
from quantark.util.calendar import BusinessDayConvention  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

WF = PnLExplainConfig(methods=(ExplainMethod.WATERFALL,))


def _lifecycle_config(pnl_explain=None, delta_threshold=1e12, *, lagged=False):
    """Spot ramps 100 -> 70 and knocks the 85 barrier out.

    ``lagged`` books the KO cash two calendar days after the hit. The analytical
    barrier engine cannot price a delayed first-hit payment, so that variant
    prices on the constant engine with greeks off (as the lifecycle suite does);
    the immediate-settlement variant keeps the analytical engine and greeks.
    """
    adapter = RampAdapter(start_spot=100.0, end_spot=70.0, num_days=40)
    if lagged:
        convention = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.CALENDAR_DAYS,
                                          business_day_convention=BusinessDayConvention.UNADJUSTED)
        position = _down_out_put_position(2.0, 10.0, settlement_convention=convention, engine=ConstantEngine())
    else:
        position = _down_out_put_position(2.0, 10.0)
    return BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=delta_threshold),
        start_date=START, end_date=adapter.dates[-1].to_pydatetime(), underlying=UNDERLYING,
        initial_positions=[position],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(),
        handle_lifecycle_events=True, calculate_greeks=not lagged, pnl_explain=pnl_explain,
    )


def _frames(results):
    return results.states_df, results.trades_df, results.explain_df, results.explain_reconciliation_df


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _normalized(frame):
    """Object columns (lifecycle event records) compared by their string form; numbers exactly.

    Position ids are fresh UUIDs per constructed Position (they also prefix
    cashflow ids), so UUIDs are masked; everything else must match exactly.
    """
    out = frame.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        out[col] = s.astype(str).str.replace(_UUID, "<id>", regex=True)
    return out


def _record(**kw):
    from quantark.backtest.equity.state import TradeRecord
    base = dict(timestamp=START, trade_type="adjust", instrument_type="spot", underlying=UNDERLYING,
                quantity=1.0, price=100.0, notional=100.0, transaction_cost=0.0, reason="x", position_id="p")
    base.update(kw)
    return TradeRecord(**base)


@pytest.mark.parametrize("lagged, config", [(False, PnLExplainConfig()), (True, WF)])
def test_explain_off_is_byte_identical_and_frames_are_empty(lagged, config):
    off = BacktestEngine(_lifecycle_config(lagged=lagged)).run()
    states, trades, explain_df, recon = _frames(off)
    assert list(explain_df.columns) == ["date", *FRAME_COLUMNS] and explain_df.empty
    assert list(recon.columns) == RECON_COLUMNS and recon.empty
    on = BacktestEngine(_lifecycle_config(config, lagged=lagged)).run()
    # exact comparisons: the explain is an observer and must not move a single bit
    pd.testing.assert_frame_equal(_normalized(on.states_df), _normalized(states), check_exact=True)
    pd.testing.assert_frame_equal(on.trades_df, trades, check_exact=True)
    # position ids are fresh UUIDs per constructed Position: compare everything else
    pd.testing.assert_frame_equal(_normalized(on.get_lifecycle_events()), _normalized(off.get_lifecycle_events()),
                                  check_exact=True)
    assert not on.explain_df.empty and not on.explain_reconciliation_df.empty


def test_unknown_trade_type_is_rejected():
    with pytest.raises(ValidationError, match="unknown backtest trade type"):
        _trade_from_record(_record(trade_type="bogus"))


def test_zero_quantity_record_is_skipped_but_nonzero_without_position_id_is_rejected():
    assert _trade_from_record(_record(trade_type="no_trade", quantity=0.0, price=0.0, position_id=None)) is None
    with pytest.raises(ValidationError, match="position_id"):
        _trade_from_record(_record(position_id=None))
    with pytest.raises(ValidationError, match="position_id"):
        _trade_from_record(_record(position_id=""))


def test_missing_or_non_finite_trade_price_is_rejected():
    with pytest.raises(ValidationError, match="price"):
        _trade_from_record(_record(price=float("nan")))
    with pytest.raises(ValidationError, match="price"):
        _trade_from_record(_record(price=None))
    with pytest.raises(ValidationError, match="quantity"):
        _trade_from_record(_record(quantity=float("inf")))


def test_lifecycle_ko_with_settlement_lag_reconciles_every_day():
    results = BacktestEngine(_lifecycle_config(WF, lagged=True)).run()
    recon = results.explain_reconciliation_df
    port = recon[recon["level"] == "portfolio"]
    assert len(port) == len(results.states_df) - 1               # one method, from day two
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    ex = results.explain_df
    events = results.get_lifecycle_events()             # indexed by event date
    ko_day = pd.Timestamp(events.index[0])
    ko_rows = ex[(ex["date"] == ko_day) & (ex["factor"] == "lifecycle_event") & (ex["level"] == "position")]
    assert len(ko_rows) == 1 and ko_rows.iloc[0]["pnl"] != 0.0
    # the tombstone lives on for the settlement lag: time rows exist after the KO day, event rows are zero
    later = ex[(ex["date"] > ko_day) & (ex["level"] == "position")]
    later_events = later[later["factor"] == "lifecycle_event"]
    assert len(later_events) >= 1 and set(later_events["date"]) == set(later["date"])
    assert (later_events["pnl"] == 0.0).all()
    assert set(later["term"]) >= {"time", "rate"}
    assert later["date"].nunique() >= 1
    # the receivable is paid inside TIME on the settlement date and the position then leaves the book
    paid_day = ko_day + pd.Timedelta(days=2)
    assert (later["date"] <= paid_day).all()
    states = results.states_df
    assert states.loc[paid_day, "paid_cash"] != 0.0


def test_spot_hedge_adjusts_identity_holds_states_gap_documented():
    # Vanilla short-call book: the delta never vanishes, so the simple executor adjusts daily.
    # (On the barrier book the executor would try to set its hedge to exactly zero after the
    # KO and Portfolio.update_position rejects that with the explain OFF as well — a
    # pre-existing engine limitation, not a fixture for this test.)
    cfg = make_multi_config(DeltaNeutralStrategy(delta_threshold=0.0))
    cfg.pnl_explain = PnLExplainConfig()
    results = BacktestEngine(cfg).run()
    recon = results.explain_reconciliation_df
    port = recon[recon["level"] == "portfolio"]
    assert len(port) == 2 * (len(results.states_df) - 1)          # two methods, from day two
    assert port["ok"].all()
    trades = results.trades_df
    assert len(trades) > 1
    # the simple executor keeps the original entry price on adjusts: gap_states is reported, and it
    # is genuinely non-zero on adjust days (the documented pre-existing accounting quirk)
    assert port["gap_states"].notna().all()
    assert (port["gap_states"].abs() > 1e-8 * port["expected"].abs().clip(lower=1.0)).any()
    ex = results.explain_df
    assert (ex["method"] == "taylor").any() and (ex["factor"] == "trade").any()


def test_multi_instrument_hedge_states_gap_is_zero():
    cfg = make_multi_config(DeltaGammaNeutralStrategy(delta_threshold=1.0, gamma_threshold=0.5,
                                                      rebalance_frequency="continuous"))
    cfg.pnl_explain = WF
    results = BacktestEngine(cfg).run()
    port = results.explain_reconciliation_df.query("level == 'portfolio'")
    assert len(port) == len(results.states_df) - 1               # one method, from day two
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    assert (results.explain_df["level"] == "position").any()


def test_pnl_explain_field_is_appended_after_metadata():
    """Positional construction of the config must keep its existing slots (review finding)."""
    from dataclasses import fields
    names = [f.name for f in fields(BacktestConfig)]
    assert names[-2:] == ["metadata", "pnl_explain"]
