"""
State and strategy objects for OTC autocallable backtests.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
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


@dataclass
class FuturesHedgePosition:
    """Single active futures hedge position with realized PnL tracking."""

    contract: Optional[str] = None
    quantity: float = 0.0
    avg_price: float = 0.0
    multiplier: float = 1.0
    realized_pnl: float = 0.0

    def mark_to_market(self, price: float) -> float:
        if self.contract is None or abs(self.quantity) == 0:
            return float(self.realized_pnl)
        return float(
            self.realized_pnl
            + self.quantity * (float(price) - self.avg_price) * self.multiplier
        )

    def trade(self, quantity_delta: float, price: float, contract: str, multiplier: float) -> None:
        quantity_delta = float(quantity_delta)
        price = float(price)
        multiplier = float(multiplier)
        if abs(quantity_delta) < 1e-12:
            return

        if self.contract is None or abs(self.quantity) < 1e-12:
            self.contract = contract
            self.quantity = quantity_delta
            self.avg_price = price
            self.multiplier = multiplier
            return

        if self.contract != contract:
            raise ValidationError("Cannot trade a different contract without rolling")

        same_direction = self.quantity * quantity_delta > 0
        if same_direction:
            new_qty = self.quantity + quantity_delta
            self.avg_price = (
                self.avg_price * abs(self.quantity) + price * abs(quantity_delta)
            ) / abs(new_qty)
            self.quantity = new_qty
            self.multiplier = multiplier
            return

        close_qty = min(abs(self.quantity), abs(quantity_delta))
        direction = 1.0 if self.quantity > 0 else -1.0
        self.realized_pnl += direction * close_qty * (price - self.avg_price) * multiplier
        new_qty = self.quantity + quantity_delta
        if abs(new_qty) < 1e-12:
            self.quantity = 0.0
            self.avg_price = 0.0
            self.contract = None
            self.multiplier = multiplier
            return

        if self.quantity * new_qty < 0:
            self.avg_price = price
        self.quantity = new_qty
        self.contract = contract
        self.multiplier = multiplier
