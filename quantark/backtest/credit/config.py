"""Configuration for credit backtests."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from quantark.backtest.strategy.base_strategy import BaseStrategy
from quantark.backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from quantark.portfolio.credit import CreditPortfolio
from quantark.util.exceptions import ValidationError


@dataclass
class CreditBacktestConfig:
    """
    Configuration for a credit hedging backtest.

    Attributes:
        portfolio: The credit book to manage.
        market_path: DataFrame of per-entity *levels* indexed by date, with
            columns ``{entity}_hazard`` (required) and optionally ``{entity}_rate``.
        strategy: Hedging strategy (e.g. CreditSpreadNeutralStrategy).
        transaction_cost_model: Cost model for hedge trades (default zero).
        calculate_greeks: Whether to compute greeks each step (required for hedging).
        metadata: Free-form metadata.
    """

    portfolio: CreditPortfolio
    market_path: pd.DataFrame
    strategy: BaseStrategy
    transaction_cost_model: Optional[TransactionCostModel] = None
    calculate_greeks: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio, CreditPortfolio):
            raise ValidationError("CreditBacktestConfig requires a CreditPortfolio")
        if self.market_path is None or len(self.market_path) == 0:
            raise ValidationError("market_path must be a non-empty DataFrame")
        if self.strategy is None:
            raise ValidationError("A hedging strategy is required")
        if self.transaction_cost_model is None:
            self.transaction_cost_model = ZeroCostModel()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "asset_class": "credit",
            "strategy": self.strategy.name,
            "steps": len(self.market_path),
            "calculate_greeks": self.calculate_greeks,
        }
