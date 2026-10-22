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
    pd.testing.assert_frame_equal(on.states_df(), off.states_df(), check_exact=True)
    pd.testing.assert_frame_equal(on.greeks_df(), off.greeks_df(), check_exact=True)
    pd.testing.assert_frame_equal(on.trades_df(), off.trades_df(), check_exact=True)
    pd.testing.assert_frame_equal(on.actions_df(), off.actions_df(), check_exact=True)
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
    assert len(port) == len(results.states_df()) - 1
    assert port["ok"].all()
    assert (port["gap_states"].abs() <= 1e-8 * port["expected"].abs().clip(lower=1.0)).all()
    # the explain is a pure observer: the same run without it books the same states and trades
    off_cfg = fixtures.make_book_config()
    off_cfg.hedge.roll_policy = FuturesRollPolicy(roll_days_before_expiry=3)
    off = ReplayBacktestEngine(off_cfg).run()
    pd.testing.assert_frame_equal(results.states_df(), off.states_df(), check_exact=True)
    pd.testing.assert_frame_equal(results.trades_df(), off.trades_df(), check_exact=True)


def test_single_engine_passthrough():
    cfg = fixtures.make_scalar_bsm_config()
    cfg.pnl_explain = WF
    results = AutocallableBacktestEngine(cfg).run()
    assert not results.explain_df.empty
    port = results.explain_reconciliation_df.query("level == 'portfolio'")
    assert len(port) == len(results.states_df) - 1
    assert port["ok"].all()
    off = AutocallableBacktestEngine(fixtures.make_scalar_bsm_config()).run()
    assert off.explain_df.empty and off.explain_reconciliation_df.empty
    pd.testing.assert_frame_equal(results.states_df, off.states_df, check_exact=True)


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
    port = results.explain_reconciliation_df.query("level == 'portfolio'")
    assert len(port) == len(results.states_df) - 1
    assert port["ok"].all()


def test_pnl_explain_field_is_appended_after_metadata_in_both_replay_configs():
    """Positional construction of the configs must keep its existing slots (review finding)."""
    from dataclasses import fields
    from quantark.backtest.replay.config import AutocallableBacktestConfig, ReplayBacktestConfig
    for cfg in (AutocallableBacktestConfig, ReplayBacktestConfig):
        names = [f.name for f in fields(cfg)]
        # Later fields are APPENDED, never inserted: metadata and pnl_explain
        # keep the slots they had, whatever is added after them.  Assert the
        # SLOTS, not the tail of the list, so a genuinely appended field does
        # not read as a broken one.
        appended = ["metadata", "pnl_explain", "dividend_roll_policy"]
        first = names.index("metadata")
        assert names[first : first + 3] == appended, cfg.__name__
        for offset, name in enumerate(appended):
            assert names.index(name) == first + offset, (cfg.__name__, name)
        # Anything after them is later work, and must stay after them.
        assert set(names[first + 3 :]) <= {
            "record_carry_exposure",
            "carry_audit_mode",
            "carry_audit_dates",
            "carry_risk_settings",
        }, cfg.__name__


def test_replay_first_day_intraday_leg_is_not_carried():
    """A leg opened and closed on day one is a tombstone in the baseline, not a held leg."""
    from datetime import timedelta
    from types import SimpleNamespace
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.pnlexplain.equity.recorder import ReplayPnLExplainRecorder
    from quantark.priceenv import PricingEnvironment

    def env(d):
        return PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.2),
                                  rate_curve=FlatRateCurve(rate=0.03),
                                  div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=d)

    engine = SimpleNamespace(_replays=[], _pricing_engines=[], _quantities=[], _start_date=pd.Timestamp(D),
                             config=SimpleNamespace(underlying="IDX"),
                             hedge_position=SimpleNamespace(contract=None, quantity=0.0), _transaction_costs=0.0)
    rec = ReplayPnLExplainRecorder(WF)
    market = {"spot": 100.0}
    selected = {"futures_price": 101.0, "multiplier": 300.0, "contract": "IF2401"}
    d1, d2 = D, D + timedelta(days=1)
    rec.begin_day(engine, d1, env(d1))
    round_trip = [_row(trade_type="hedge_rebalance", quantity=1.0, price=101.0, date=d1),
                  _row(trade_type="hedge_close", quantity=-1.0, price=101.5, date=d1)]
    rec.end_day(engine, d1, env(d1), market, selected, round_trip, {"total_pnl": 0.0})
    assert dict(rec._prev_book.quoted_legs) == {}                 # not carried into tomorrow's t0
    rec.begin_day(engine, d2, env(d2))
    rec.end_day(engine, d2, env(d2), market, selected, [], {"total_pnl": 0.0})   # must not raise
    _, recon = rec.frames()
    assert len(recon) == 1 and bool(recon.iloc[0]["ok"])


def test_replay_event_buffer_is_reset_once_per_date_not_per_observation():
    """A settlement-only day must not inherit yesterday's events, and a settlement recorded
    before the observation step must not be erased by it (review finding)."""
    from types import SimpleNamespace
    from quantark.pnlexplain.equity.recorder import ReplayPnLExplainRecorder
    stale = SimpleNamespace(events_today=["yesterday"], record_events=True)
    engine = SimpleNamespace(_replays=[stale], _pricing_engines=[None], _quantities=[1.0])
    ReplayPnLExplainRecorder(WF).begin_day(engine, D, None)          # first day: no previous book
    assert stale.events_today == []
    # the observation step appends without clearing
    replay = ReplayBacktestEngine(fixtures.make_book_config())._replays[0]
    replay.record_events = True
    replay.events_today = ["settlement-before-observation"]
    replay._tracker = SimpleNamespace(observe=lambda *a: [])
    replay.apply_lifecycle_events(D, None, None, 100.0)
    assert replay.events_today == ["settlement-before-observation"]
