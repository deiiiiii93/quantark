"""Configuration for FX backtests."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from quantark.backtest.strategy.base_strategy import BaseStrategy
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from quantark.portfolio.fx import FXPortfolio
from quantark.util.exceptions import ValidationError


@dataclass
class FXBacktestConfig:
    """
    Configuration for an FX hedging backtest.

    Attributes:
        portfolio: The FX book to manage.
        market_path: DataFrame of per-pair *levels* indexed by date, with columns
            ``{pair}_spot`` (required) and optionally ``{pair}_vol``,
            ``{pair}_dom_rate``, ``{pair}_for_rate``.
        strategy: Hedging strategy (e.g. FXDeltaNeutralStrategy).
        transaction_cost_model: Cost model for hedge trades (default zero).
        calculate_greeks: Whether to compute greeks each step (required for hedging).
        metadata: Free-form metadata.
    """

    portfolio: FXPortfolio
    market_path: pd.DataFrame
    strategy: BaseStrategy
    transaction_cost_model: Optional[TransactionCostModel] = None
    calculate_greeks: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio, FXPortfolio):
            raise ValidationError("FXBacktestConfig requires an FXPortfolio")
        if self.market_path is None or len(self.market_path) == 0:
            raise ValidationError("market_path must be a non-empty DataFrame")
        if self.strategy is None:
            raise ValidationError("A hedging strategy is required")
        if self.transaction_cost_model is None:
            self.transaction_cost_model = ZeroCostModel()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "asset_class": "fx",
            "strategy": self.strategy.name,
            "steps": len(self.market_path),
            "calculate_greeks": self.calculate_greeks,
        }
