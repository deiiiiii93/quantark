"""
Strategy framework for backtesting.

Includes abstract base strategy and concrete implementations for:
- Equity: Delta-neutral hedging
- Fixed Income: DV01-neutral and convexity-neutral hedging
"""

from backtest.strategy.base_strategy import (
    BaseStrategy,
    AssetClass,
    HedgingTarget,
)
from backtest.strategy.delta_neutral_strategy import DeltaNeutralStrategy
from backtest.strategy.dv01_neutral_strategy import DV01NeutralStrategy
from backtest.strategy.convexity_neutral_strategy import ConvexityNeutralStrategy

__all__ = [
    # Base
    'BaseStrategy',
    'AssetClass',
    'HedgingTarget',
    # Equity strategies
    'DeltaNeutralStrategy',
    # Fixed Income strategies
    'DV01NeutralStrategy',
    'ConvexityNeutralStrategy',
]
