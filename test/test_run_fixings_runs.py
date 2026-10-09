"""Run fixings through every run path (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md §7 X1-X3, X6)."""
import copy
import warnings
from datetime import timedelta

import pytest

import run_fixings_fixtures as F
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning
from quantark.util.numerical import is_close


def _rows(recorder, pid=None):
    return [(f.index, f.outcome, f.spot, f.gap) for f in recorder.recorded if pid is None or f.position_id == pid]


def _hand_rows(kind, spot_of=F.spot_on):
    """A DAYS-day run's recordings: every fixing dated on/before day DAYS - 2 (the last day's is still live), at its
    own day's spot, never a gap."""
    outs = F.hand_outcomes(kind, F.DAYS - 2, spot_of)
    return [(i, outs[i], spot_of(F.day_of(F.fixing_dates()[i])), False) for i in sorted(outs)]


# ================================================================ equity backtest
def _backtest(product, kind, adapter, tmp_path, frequency="D"):
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position

    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=F.START,
        end_date=adapter.dates[-1].to_pydatetime(), underlying=F.UNDERLYING,
        initial_positions=[Position(product=product, quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING,
                                    engine=F.ENGINES[kind](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
        frequency=frequency, results_path=str(tmp_path),
    )
    engine = BacktestEngine(cfg)
    return engine, engine.run()


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_backtest_x1_x2_x6(kind, tmp_path):
    booked = F.MAKERS[(kind, "dated")]()
    before = copy.deepcopy(booked)
    (eng_d, res_d), warns = F.count_missing(_backtest, booked, kind, F.DailyAdapter(F.DAYS), tmp_path)
    eng_n, res_n = _backtest(F.MAKERS[(kind, "numeric")](), kind, F.DailyAdapter(F.DAYS), tmp_path)
    values_d, values_n = res_d.get_value_series().tolist(), res_n.get_value_series().tolist()
    engine = F.ENGINES[kind]()
    for k in range(F.DAYS):                                                      # X1a: bitwise vs by hand
        ref = F.quiet_price(engine, F.by_hand(kind, k - 1), F.eq_env(k))
        assert values_d[k] == ref, (k, values_d[k], ref)
        assert is_close(values_n[k], values_d[k], rel_tol=1e-12)                 # X2 values
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _hand_rows(kind)  # X2 sequences, X1b spot
    pid = list(eng_d.portfolio.positions)[0]                                     # X6
    assert eng_d.portfolio.positions[pid].product.observation_records == F.by_hand(kind, F.DAYS - 2).observation_records
    assert booked.observation_records == before.observation_records             # X3g


def test_equity_backtest_x1b_the_fixing_does_not_move_later(tmp_path):
    other = lambda k: F.spot_on(k) if k <= 5 else 80.0                          # noqa: E731  differs from day 6 on
    a, _ = _backtest(F.ra_dated(), "ra", F.DailyAdapter(F.DAYS), tmp_path)
    b, _ = _backtest(F.ra_dated(), "ra", F.DailyAdapter(F.DAYS, spot_of=other), tmp_path)
    assert _rows(a.run_fixings)[0] == _rows(b.run_fixings)[0] == (0, True, 105.0, False)


def test_equity_backtest_x3b_weekly_gap_takes_the_previous_instant(tmp_path):
    adapter = F.DailyAdapter(5, freq="W-MON")                                  # days 0, 7, 14, 21, 28
    (eng, res), warns = F.count_missing(_backtest, F.ra_dated(), "ra", adapter, tmp_path, "W")
    assert warns == 0
    assert _rows(eng.run_fixings) == [(0, True, 105.0, True), (1, False, 115.0, True)]
    assert res.get_value_series().tolist()[3] == F.quiet_price(F.ENGINES["ra"](), F.ra_dated({0: True, 1: False}),
                                                               F.eq_env(21))


def test_equity_backtest_x3e_hourly_refuses_a_participant(tmp_path):
    with pytest.raises(ValidationError, match="time of day"):
        _backtest(F.ra_dated(), "ra", F.DailyAdapter(3, freq="h"), tmp_path, "H")


def test_equity_backtest_numeric_participant_no_longer_warns_constant_maturity(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _backtest(F.ra_numeric(), "ra", F.DailyAdapter(6), tmp_path)
    assert not [w for w in caught if "no roll rule" in str(w.message)]
