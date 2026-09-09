from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import quantark
from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.backtest.simulation.measures import MEASURE_COLUMNS, path_measures
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.results import SUMMARY_COLUMNS, EnsembleResults
from quantark.backtest.simulation.runner import run_ensemble
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry

START = date(2024, 1, 2)


def _paths(n_paths=8, n_days=8, sigma=1.2, seed=3):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=seed)


@pytest.fixture(scope="module")
def results() -> EnsembleResults:
    return EnsembleBacktestEngine(ensemble_config()).run(_paths())


def test_the_manifest_records_notional_and_library_version(results):
    assert results.manifest["book_notional"] == 1000.0 * SPOT * 1.0
    assert results.notional == results.manifest["book_notional"]
    assert results.manifest["library_version"] == quantark.__version__


def test_summary_has_one_row_per_path_with_the_library_measures(results):
    summary = results.summary
    assert list(summary.columns) == list(SUMMARY_COLUMNS)
    assert len(summary) == results.n_paths and list(summary["path"]) == list(range(results.n_paths))
    for i in range(results.n_paths):
        expected = path_measures(results.path_states(i), results.path_trades(i), notional=results.notional)
        row = summary.iloc[i]
        for name in MEASURE_COLUMNS:
            a, b = row[name], expected[name]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (i, name, a, b)
        assert row["days"] == int(results.last_day[i]) + 1
        assert row["terminal_pnl"] == results.cube.total_pnl[i, int(results.last_day[i])]


def test_termination_reasons_and_flags_follow_the_events(results):
    summary = results.summary
    reasons = set(summary["termination_reason"])
    assert reasons <= {"knock_out", "maturity", "data_end"} and {"knock_out", "maturity"} <= reasons
    for i, row in summary.iterrows():
        last = int(results.last_day[i])
        assert bool(row["knocked_in"]) == bool(results.cube.knocked_in[i, last])
        mine = [e for e in results.events if e.path == i and e.event in ("knock_out", "maturity")]
        if row["termination_reason"] == "knock_out":
            assert bool(results.cube.knocked_out[i, last]) and row["ko_observation_index"] == mine[-1].index >= 0
        elif row["termination_reason"] == "maturity":
            assert bool(results.cube.matured[i, last]) and row["ko_observation_index"] == -1


def test_a_batched_run_has_the_same_summary(results):
    batched = run_ensemble(ensemble_config(batch_paths=3), _paths())
    pd.testing.assert_frame_equal(batched.summary, results.summary)


def test_summary_is_memoised(results):
    assert results.summary is results.summary


def test_distribution_reports_moments_quantiles_and_the_loss_tail(results):
    d = results.distribution("terminal_pnl_bp", es_level=0.25)
    values = results.summary["terminal_pnl_bp"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    assert d["n"] == values.size and d["n_paths"] == results.n_paths
    assert d["mean"] == float(values.mean()) and d["std"] == float(values.std(ddof=1))
    assert d["quantiles"]["q50"] == pytest.approx(float(np.quantile(values, 0.5)), rel=1e-12)
    assert set(d["quantiles"]) == {"q01", "q05", "q25", "q50", "q75", "q95", "q99"}
    cut = np.quantile(values, 0.25)
    assert d["expected_shortfall"] == pytest.approx(float(values[values <= cut].mean()), rel=1e-12)
    assert d["share_positive"] == float((values > 0).mean())
    upper = results.distribution("cost_bp", es_level=0.25, tail="upper")
    costs = results.summary["cost_bp"].to_numpy(dtype=float)
    assert upper["expected_shortfall"] == pytest.approx(float(costs[costs >= np.quantile(costs, 0.75)].mean()), rel=1e-12)


def test_distribution_frequencies_cover_every_path(results):
    d = results.distribution("terminal_pnl", es_level=0.05)
    reasons = results.summary["termination_reason"]
    assert d["ko_frequency"] == float((reasons == "knock_out").mean())
    assert d["maturity_frequency"] == float((reasons == "maturity").mean())
    assert d["ki_frequency"] == float(results.summary["knocked_in"].mean())
    assert d["ko_frequency"] + d["maturity_frequency"] + d["data_end_frequency"] == pytest.approx(1.0)


def test_distribution_fails_closed_on_bad_arguments(results):
    with pytest.raises(ValidationError):
        results.distribution("sharpe", es_level=0.05)
    with pytest.raises(ValidationError):
        results.distribution("terminal_pnl_bp", es_level=0.0)
    with pytest.raises(ValidationError):
        results.distribution("terminal_pnl_bp", es_level=0.05, tail="middle")


from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy  # noqa: E402
from quantark.backtest.simulation.results import PairedComparison  # noqa: E402


def test_paired_with_itself_is_identically_zero(results):
    comparison = results.paired(results)
    assert isinstance(comparison, PairedComparison)
    assert comparison.n_paths == results.n_paths
    for name in MEASURE_COLUMNS + ("terminal_pnl",):
        diffs = comparison.differences[name].to_numpy(dtype=float)
        assert np.all((diffs == 0.0) | np.isnan(diffs)), name
    assert comparison.differences["same_termination"].all()
    d = comparison.describe("terminal_pnl_bp")
    assert d["n"] == results.n_paths and d["mean"] == 0.0 and d["share_positive"] == 0.0 and d["t_stat"] is None


def test_paired_measures_a_different_hedge_on_the_same_paths(results):
    half = EnsembleBacktestEngine(ensemble_config(
        strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=0.5, target_delta=0.0),
    )).run(_paths())
    comparison = half.paired(results)
    diffs = comparison.differences["terminal_pnl_bp"].to_numpy(dtype=float)
    assert np.any(diffs != 0.0)
    expected = half.summary["terminal_pnl_bp"].to_numpy() - results.summary["terminal_pnl_bp"].to_numpy()
    assert np.array_equal(diffs, expected)
    d = comparison.describe("terminal_pnl_bp")
    assert d["n"] == results.n_paths and d["t_stat"] is not None and 0.0 <= d["share_positive"] <= 1.0
    assert comparison.path_fingerprint == results.manifest["path_fingerprint"]


def test_paired_refuses_unmatched_paths(results):
    other = EnsembleBacktestEngine(ensemble_config()).run(_paths(seed=4))
    with pytest.raises(ValidationError):
        results.paired(other)
    with pytest.raises(ValidationError):
        results.paired(results.take([0, 1]))


def test_take_keeps_the_chosen_paths_and_renumbers_the_logs(results):
    sub = results.take([5, 2])
    assert sub.n_paths == 2 and sub.manifest["path_indices"] == [5, 2]
    assert np.array_equal(sub.cube.total_pnl[0], results.cube.total_pnl[5])
    assert np.array_equal(sub.last_day, results.last_day[[5, 2]])
    pd.testing.assert_frame_equal(sub.path_states(1), results.path_states(2))
    pd.testing.assert_frame_equal(sub.path_trades(0), results.path_trades(5).assign(path=0))
    assert {e.path for e in sub.events} <= {0, 1}
    assert sub.summary.drop(columns="path").iloc[1].equals(results.summary.drop(columns="path").iloc[2])
    with pytest.raises(ValidationError):
        results.take([results.n_paths])
