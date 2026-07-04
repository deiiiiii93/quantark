"""
Equity portfolio module.

Provides equity-specific position and portfolio classes for managing
equity derivative positions.
"""
from .position import EquityPosition, Position
from .swap_position import EquitySwapPosition
from .portfolio import EquityPortfolio, Portfolio
from .futures_buckets import (
    aggregate_futures_delta_buckets,
    aggregate_futures_rhoq_buckets,
)

__all__ = [
    'EquityPosition',
    'EquitySwapPosition',
    'EquityPortfolio',
    'aggregate_futures_delta_buckets',
    'aggregate_futures_rhoq_buckets',
    # Backward compatibility
    'Position',
    'Portfolio',
]

