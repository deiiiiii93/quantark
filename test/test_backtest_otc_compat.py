"""Behavioral compatibility of the deprecated quantark.backtest.otc paths
(spec §9.4): old imports construct, run, and expose EVERY public accessor
with the legacy shapes — properties on single results, callable methods and
the products_meta kwarg on book results — until 0.5.0 removes the shims.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

SINGLE_PROPERTIES = (
    "states_df",
    "greeks_df",
    "rebalance_df",
    "trades_df",
    "actions_df",
    "surfaces_df",
    "daily_event_summary_df",
    "event_probability_df",
)
BOOK_METHODS = (
    "states_df",
    "greeks_df",
    "rebalances_df",
    "trades_df",
    "actions_df",
    "daily_event_summary_df",
    "event_probability_df",
    "surfaces_df",
)


def test_single_engine_end_to_end_via_old_paths():
    from quantark.backtest.otc import (
        AutocallableBacktestEngine,
        AutocallableBacktestResults,
    )

    results = AutocallableBacktestEngine(fixtures.make_scalar_bsm_config()).run()
    assert isinstance(results, AutocallableBacktestResults)
    for name in SINGLE_PROPERTIES:
        frame = getattr(results, name)
        assert isinstance(frame, pd.DataFrame), name
    assert isinstance(results.calibration_records, list)
    assert isinstance(results.get_summary(), dict)
    assert isinstance(results.get_total_pnl(), float)
    assert isinstance(results.get_total_return(), float)
    assert isinstance(results.get_pnl_series(), pd.Series)
    assert isinstance(results.get_value_series(), pd.Series)
    assert isinstance(results.get_hedge_trades(), pd.DataFrame)
    assert isinstance(results.get_lifecycle_events(), pd.DataFrame)


def test_book_engine_end_to_end_via_old_paths():
    from quantark.backtest.otc.book_engine import (
        BookAutocallableBacktestEngine,
        BookBacktestResults,
    )

    results = BookAutocallableBacktestEngine(fixtures.make_book_config()).run()
    assert isinstance(results, BookBacktestResults)
    for name in BOOK_METHODS:
        accessor = getattr(results, name)
        assert callable(accessor), f"{name} must remain a callable method"
        assert isinstance(accessor(), pd.DataFrame), name
    assert isinstance(results.get_summary(), dict)


def test_book_results_products_meta_constructor_kwarg():
    from quantark.backtest.otc.book_engine import BookBacktestResults

    results = BookBacktestResults(
        config=None, states=[], greeks=[], rebalances=[], trades=[], actions=[],
        daily_event_summary=[], event_probabilities=[], surfaces=[],
        products_meta=[{"position_id": 1}],
    )
    assert results.get_summary()["num_products"] == 1


def test_root_backtest_exports_and_factory_dispatch():
    import quantark.backtest as b

    assert b.AutocallableBacktestEngine is not None
    assert b.ReplayBacktestEngine is not None
    engine = b.get_backtest_engine(fixtures.make_scalar_bsm_config())
    assert type(engine).__name__ == "AutocallableBacktestEngine"
    engine = b.get_backtest_engine(fixtures.make_book_config())
    assert type(engine).__name__ == "ReplayBacktestEngine"


@pytest.mark.parametrize(
    "module",
    [
        "quantark.backtest.otc",
        "quantark.backtest.otc.config",
        "quantark.backtest.otc.engine",
        "quantark.backtest.otc.book_engine",
        "quantark.backtest.otc.market",
        "quantark.backtest.otc.engine_factory",
        "quantark.backtest.otc.state",
        "quantark.backtest.otc.results",
        "quantark.backtest.otc._replay",
        "quantark.backtest.otc.vol_calibrators",
        "quantark.backtest.otc.vol_history",
    ],
)
def test_every_shim_warns_deprecation(module):
    importlib.import_module(module)  # ensure importable
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(sys.modules[module])
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), module
