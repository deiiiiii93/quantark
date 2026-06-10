"""
Equity backtest module.

Provides equity-specific backtest components for simulating delta-neutral
hedging strategies on equity derivative portfolios.
"""
from .config import BacktestConfig, EquityBacktestConfig
from .state import BacktestState, StateTracker, TradeRecord
from .results import BacktestResults, EquityBacktestResults
from .metrics import PerformanceMetrics, EquityPerformanceMetrics
from .engine import BacktestEngine, EquityBacktestEngine
from .hedge_executor import HedgeExecutor, EquityHedgeExecutor

__all__ = [
    # Config
    'BacktestConfig',
    'EquityBacktestConfig',
    # State
    'BacktestState',
    'StateTracker',
    'TradeRecord',
    # Results
    'BacktestResults',
    'EquityBacktestResults',
    # Metrics
    'PerformanceMetrics',
    'EquityPerformanceMetrics',
    # Engine
    'BacktestEngine',
    'EquityBacktestEngine',
    # Hedge Executor
    'HedgeExecutor',
    'EquityHedgeExecutor',
]

