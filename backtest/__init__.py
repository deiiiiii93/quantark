"""
QuantArk Backtest Module

A comprehensive backtesting framework for hedging strategies with:
- Delta-neutral and custom strategy support
- Transaction cost modeling
- Comprehensive logging and metrics
- Static and interactive visualizations
"""

from backtest.engine import BacktestEngine
from backtest.config import BacktestConfig
from backtest.state import BacktestState
from backtest.results import BacktestResults
from backtest.metrics import PerformanceMetrics
from backtest.transaction_costs import (
    TransactionCostModel,
    ZeroCostModel,
    FixedCostModel,
    ProportionalCostModel,
    CompleteCostModel
)
from backtest.hedge_executor import HedgeExecutor
from backtest.logger import BacktestLogger
from backtest.visualizer import StaticVisualizer
from backtest.dashboard import InteractiveDashboard
from backtest.report_generator import ReportGenerator

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestState',
    'BacktestResults',
    'PerformanceMetrics',
    'TransactionCostModel',
    'ZeroCostModel',
    'FixedCostModel',
    'ProportionalCostModel',
    'CompleteCostModel',
    'HedgeExecutor',
    'BacktestLogger',
    'StaticVisualizer',
    'InteractiveDashboard',
    'ReportGenerator',
]

