"""
Cross-engine interface unification for the backtest module.

Both the portfolio equity engine and the single-product OTC autocallable
engine expose ``run()`` (``BaseBacktestEngine``) and produce results offering
the same accessor surface (``BaseBacktestResults``), so downstream code can
treat them uniformly. ``get_backtest_engine`` dispatches a config to its
engine.
"""

from datetime import datetime

import pandas as pd

from quantark.backtest import (
    BacktestConfig,
    BacktestEngine,
    BaseBacktestEngine,
    BaseBacktestResults,
    ZeroCostModel,
    get_backtest_engine,
)
from quantark.backtest.otc.results import AutocallableBacktestResults
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.util.marketdata.adapter.mock_adapter import MockMarketDataAdapter

UNDERLYING = "EQUITY_INDEX"


def _equity_config():
    return BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12),
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 2, 1),
        underlying=UNDERLYING,
        initial_positions=[],
        market_data_adapter=MockMarketDataAdapter(seed=7),
        transaction_cost_model=ZeroCostModel(),
    )


# The accessor surface every backtest result must expose.
RESULTS_SURFACE = (
    "get_summary",
    "get_total_pnl",
    "get_total_return",
    "get_pnl_series",
    "get_value_series",
    "get_hedge_trades",
    "get_lifecycle_events",
)


class TestEngineInterface:
    def test_equity_engine_satisfies_protocol(self):
        engine = BacktestEngine(_equity_config())
        assert isinstance(engine, BaseBacktestEngine)

    def test_factory_dispatches_equity_config(self):
        engine = get_backtest_engine(_equity_config())
        assert isinstance(engine, BacktestEngine)
        assert isinstance(engine, BaseBacktestEngine)

    def test_factory_rejects_unknown_config(self):
        from quantark.util.exceptions import ValidationError
        import pytest

        with pytest.raises(ValidationError):
            get_backtest_engine(object())


class TestResultsInterface:
    def test_equity_results_satisfy_protocol(self):
        results = BacktestEngine(_equity_config()).run()
        assert isinstance(results, BaseBacktestResults)
        for name in RESULTS_SURFACE:
            assert hasattr(results, name)
        # Lifecycle accessor returns a DataFrame even with no events.
        assert isinstance(results.get_lifecycle_events(), pd.DataFrame)

    def test_otc_results_satisfy_protocol(self):
        # An empty OTC result container still honours the full surface.
        results = AutocallableBacktestResults(
            config=None,
            states=[],
            greeks=[],
            rebalances=[],
            trades=[],
            actions=[],
            surfaces=[],
            daily_event_summary=[],
            event_probabilities=[],
        )
        assert isinstance(results, BaseBacktestResults)
        for name in RESULTS_SURFACE:
            assert hasattr(results, name)
        assert results.get_total_pnl() == 0.0
        assert results.get_total_return() == 0.0
        assert isinstance(results.get_pnl_series(), pd.Series)
        assert isinstance(results.get_value_series(), pd.Series)
        assert isinstance(results.get_lifecycle_events(), pd.DataFrame)


class TestReplayFactoryDispatch:
    def test_factory_dispatches_replay_config(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from replay_golden import fixtures

        from quantark.backtest import get_backtest_engine
        from quantark.backtest.replay import ReplayBacktestEngine

        engine = get_backtest_engine(fixtures.make_book_config())
        assert isinstance(engine, ReplayBacktestEngine)

    def test_factory_dispatches_single_autocallable_config(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from replay_golden import fixtures

        from quantark.backtest import get_backtest_engine
        from quantark.backtest.replay.single import AutocallableBacktestEngine

        engine = get_backtest_engine(fixtures.make_scalar_bsm_config())
        assert isinstance(engine, AutocallableBacktestEngine)
