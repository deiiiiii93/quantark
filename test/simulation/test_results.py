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
