"""
Strategy framework for backtesting.

Includes abstract base strategy and concrete implementations for:
- Equity: delta-neutral, Whalley-Wilmott band, minimum-variance delta,
  and multi-Greek (delta+gamma, delta+vega, delta+gamma+vega) hedging
- Fixed Income: DV01-neutral and convexity-neutral hedging
"""

from quantark.backtest.strategy.base_strategy import (
    BaseStrategy,
    AssetClass,
    HedgingTarget,
    passes_frequency_gate,
)
from quantark.backtest.strategy.delta_neutral_strategy import DeltaNeutralStrategy
from quantark.backtest.strategy.dv01_neutral_strategy import DV01NeutralStrategy
from quantark.backtest.strategy.convexity_neutral_strategy import ConvexityNeutralStrategy
from quantark.backtest.strategy.hedge_optimizer import HedgeOptimizer, HedgeTarget
from quantark.backtest.strategy.hedge_instruments import (
    BaseHedgeInstrument,
    SpotHedgeInstrument,
    FuturesHedgeInstrument,
    OptionHedgeInstrument,
)
from quantark.backtest.strategy.multi_greek_strategy import (
    MultiGreekHedgeStrategy,
    DeltaGammaNeutralStrategy,
    DeltaVegaNeutralStrategy,
    DeltaGammaVegaNeutralStrategy,
)
from quantark.backtest.strategy.whalley_wilmott_strategy import WhalleyWilmottStrategy
from quantark.backtest.strategy.min_variance_delta_strategy import (
    MinimumVarianceDeltaStrategy,
)

__all__ = [
    # Base
    'BaseStrategy',
    'AssetClass',
    'HedgingTarget',
    'passes_frequency_gate',
    # Equity strategies
    'DeltaNeutralStrategy',
    'WhalleyWilmottStrategy',
    'MinimumVarianceDeltaStrategy',
    # Multi-Greek hedging
    'HedgeOptimizer',
    'HedgeTarget',
    'BaseHedgeInstrument',
    'SpotHedgeInstrument',
    'FuturesHedgeInstrument',
    'OptionHedgeInstrument',
    'MultiGreekHedgeStrategy',
    'DeltaGammaNeutralStrategy',
    'DeltaVegaNeutralStrategy',
    'DeltaGammaVegaNeutralStrategy',
    # Fixed Income strategies
    'DV01NeutralStrategy',
    'ConvexityNeutralStrategy',
]
