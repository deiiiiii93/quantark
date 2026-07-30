"""Schema is the single source of truth for replay record columns (Task 13)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay import schema  # noqa: E402


@pytest.fixture(scope="module")
def single_frames():
    from quantark.backtest.replay.single import AutocallableBacktestEngine

    results = AutocallableBacktestEngine(fixtures.make_scalar_bsm_config()).run()
    return fixtures.result_frames(results)


@pytest.fixture(scope="module")
def localvol_frames(tmp_path_factory):
    from quantark.backtest.replay.single import AutocallableBacktestEngine

    results = AutocallableBacktestEngine(
        fixtures.make_localvol_config(tmp_path_factory.mktemp("hist"))
    ).run()
    return fixtures.result_frames(results), results


@pytest.fixture(scope="module")
def book_frames():
    from quantark.backtest.replay import ReplayBacktestEngine

    results = ReplayBacktestEngine(fixtures.make_book_config()).run()
    return fixtures.result_frames(results)


def test_single_frames_match_schema(single_frames):
    assert tuple(single_frames["states"].columns) == schema.STATE_COLUMNS
    assert tuple(single_frames["greeks"].columns) == schema.GREEK_COLUMNS
    assert tuple(single_frames["rebalances"].columns) == schema.REBALANCE_COLUMNS
    assert tuple(single_frames["daily_event_summary"].columns) == schema.DAILY_EVENT_COLUMNS
    assert tuple(single_frames["event_probabilities"].columns) == schema.EVENT_PROB_COLUMNS
    assert tuple(single_frames["surfaces"].columns) == schema.SURFACE_COLUMNS
    actions = tuple(single_frames["actions"].columns)
    assert actions[: len(schema.ACTION_COLUMNS)] == schema.ACTION_COLUMNS
    trades = single_frames["trades"]
    if not trades.empty:
        assert tuple(trades.columns) == schema.TRADE_COLUMNS


def test_surface_mode_states_append_provenance(localvol_frames):
    frames, results = localvol_frames
    assert tuple(frames["states"].columns) == (
        schema.STATE_COLUMNS + schema.SURFACE_PROVENANCE_COLUMNS
    )
    for record in results.calibration_records:
        for key in schema.CALIBRATION_RECORD_KEYS:
            assert key in record, key


def test_book_frames_match_schema(book_frames):
    assert tuple(book_frames["states"].columns) == schema.STATE_COLUMNS
    assert tuple(book_frames["greeks"].columns) == schema.BOOK_GREEK_COLUMNS
    trades = book_frames["trades"]
    if not trades.empty:
        assert tuple(trades.columns) == schema.TRADE_COLUMNS
