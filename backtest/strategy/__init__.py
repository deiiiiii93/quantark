"""
Strategy framework for backtesting.

Includes abstract base strategy and concrete implementations.
"""

from backtest.strategy.base_strategy import BaseStrategy
from backtest.strategy.delta_neutral_strategy import DeltaNeutralStrategy

__all__ = [
    'BaseStrategy',
    'DeltaNeutralStrategy',
]

