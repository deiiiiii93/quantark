"""Contract tests for the metrics split (plan Task 12)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.backtest.metrics import CorePerformanceMetrics  # noqa: E402
from quantark.backtest.equity.metrics import PerformanceMetrics  # noqa: E402

CORE_METRICS = (
    "total_pnl", "total_return", "sharpe_ratio", "max_drawdown",
    "max_drawdown_duration", "win_rate", "profit_factor", "value_at_risk",
    "conditional_var", "volatility", "skewness", "kurtosis",
)
EQUITY_ONLY = (
    "hedge_frequency", "average_hedge_cost", "total_hedge_cost_ratio",
    "delta_tracking_error", "average_absolute_delta",
    "delta_rebalance_efficiency",
)


@pytest.fixture(scope="module")
def replay_results():
    from quantark.backtest.replay.single import AutocallableBacktestEngine

    return AutocallableBacktestEngine(fixtures.make_scalar_bsm_config()).run()


@pytest.fixture(scope="module")
def book_results():
    from quantark.backtest.replay import ReplayBacktestEngine

    return ReplayBacktestEngine(fixtures.make_book_config()).run()


def _assert_core_metrics(metrics: CorePerformanceMetrics) -> None:
    for name in CORE_METRICS:
        value = getattr(metrics, name)()
        if name == "max_drawdown_duration":
            assert value is None or isinstance(value, pd.Timedelta)
        else:
            assert isinstance(value, float) or hasattr(value, "__float__"), name
    all_metrics = metrics.calculate_all_metrics()
    assert "sharpe_ratio" in all_metrics and "var_95" in all_metrics
    assert isinstance(metrics.to_dataframe(), pd.DataFrame)


def test_single_replay_results_have_metrics(replay_results):
    _assert_core_metrics(replay_results.metrics)


def test_book_results_have_metrics(book_results):
    _assert_core_metrics(book_results.metrics)
    assert isinstance(book_results.get_pnl_series(), pd.Series)
    assert isinstance(book_results.get_value_series(), pd.Series)
    assert isinstance(book_results.get_hedge_trades(), pd.DataFrame)
    assert isinstance(book_results.get_lifecycle_events(), pd.DataFrame)


class _EquityResultsStub:
    """Minimal stand-in exposing the equity-only attributes."""

    def __init__(self):
        idx = pd.date_range("2024-01-02", periods=4, freq="D")
        self._values = pd.Series([100.0, 101.0, 99.0, 102.0], index=idx)
        self.num_hedges = 2
        self.total_transaction_costs = 10.0
        self.initial_value = 100.0
        self.final_value = 102.0
        self.state_tracker = [1, 2, 3, 4]
        self.states_df = pd.DataFrame(index=idx)

        class _Cfg:
            class strategy:
                target_delta = 0.0

        self.config = _Cfg()

    def get_value_series(self):
        return self._values

    def get_pnl_series(self):
        return self._values - self._values.iloc[0]

    def get_total_pnl(self):
        return float(self._values.iloc[-1] - self._values.iloc[0])

    def get_total_return(self):
        return self.get_total_pnl() / float(self._values.iloc[0])

    def get_hedge_trades(self):
        return pd.DataFrame()

    def get_delta_series(self):
        return pd.Series([1.0, -1.0, 0.5, -0.5], index=self._values.index)


def test_equity_metrics_full_surface():
    metrics = PerformanceMetrics(_EquityResultsStub())
    _assert_core_metrics(metrics)
    for name in EQUITY_ONLY:
        assert isinstance(getattr(metrics, name)(), float), name
    all_metrics = metrics.calculate_all_metrics()
    assert "delta_tracking_error" in all_metrics
    assert "num_hedges" in all_metrics
    assert isinstance(metrics, CorePerformanceMetrics)
