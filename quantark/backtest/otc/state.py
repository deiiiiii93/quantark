"""
State and strategy objects for OTC autocallable backtests.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.futures_ledger import FuturesHedgePosition  # noqa: F401  (canonical home)
from quantark.util.exceptions import ValidationError


@dataclass
class AutocallableDeltaHedgeStrategy:
    """
    Delta-neutral futures hedge sizing.

    Product delta is assumed to be currency value per one index point. A futures
    contract contributes approximately ``multiplier`` per one index point.
    """

    delta_threshold: float = 0.0
    hedge_ratio: float = 1.0
    target_delta: float = 0.0
    round_contracts: bool = True

    def __post_init__(self) -> None:
        if self.delta_threshold < 0:
            raise ValidationError("delta_threshold must be non-negative")
        if not 0 <= self.hedge_ratio <= 1:
            raise ValidationError("hedge_ratio must be between 0 and 1")

    def target_contracts(
        self,
        *,
        product_delta: float,
        product_quantity: float,
        futures_multiplier: float,
    ) -> float:
        if futures_multiplier <= 0:
            raise ValidationError("futures_multiplier must be positive")
        net_delta = float(product_delta) * float(product_quantity)
        target = -((net_delta - self.target_delta) / float(futures_multiplier))
        target *= self.hedge_ratio
        if self.round_contracts:
            return float(round(target))
        return float(target)

    def should_rebalance(self, current_contracts: float, target_contracts: float) -> bool:
        return abs(float(target_contracts) - float(current_contracts)) > self.delta_threshold

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "delta_threshold": self.delta_threshold,
            "hedge_ratio": self.hedge_ratio,
            "target_delta": self.target_delta,
            "round_contracts": self.round_contracts,
        }
