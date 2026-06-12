"""
Lifecycle state objects for equity structured products.

``AutocallableLifecycleState`` was extracted from
``quantark/backtest/otc/state.py`` (which re-exports it for backward
compatibility). ``BarrierLifecycleState`` is the simpler analogue for the
vanilla barrier product family.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AutocallableLifecycleState:
    """Realized product lifecycle state during historical replay."""

    alive: bool = True
    knocked_in: bool = False
    knocked_out: bool = False
    matured: bool = False
    ki_date: Optional[datetime] = None
    ko_date: Optional[datetime] = None
    maturity_date: Optional[datetime] = None
    coupon_memory_count: int = 0
    realized_cashflows: float = 0.0
    observed_ko_indices: set[int] = field(default_factory=set)
    observed_ki_indices: set[int] = field(default_factory=set)
    observed_coupon_indices: set[int] = field(default_factory=set)

    def mark_ki(self, timestamp: datetime) -> bool:
        if self.knocked_in or self.knocked_out:
            return False
        self.knocked_in = True
        self.ki_date = timestamp
        return True

    def mark_ko(self, timestamp: datetime, cashflow: float = 0.0) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.knocked_out = True
        self.alive = False
        self.ko_date = timestamp
        self.realized_cashflows += float(cashflow)
        return True

    def mark_maturity(self, timestamp: datetime, cashflow: float = 0.0) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.matured = True
        self.alive = False
        self.maturity_date = timestamp
        self.realized_cashflows += float(cashflow)
        return True

    def add_cashflow(self, amount: float) -> None:
        self.realized_cashflows += float(amount)


@dataclass
class BarrierLifecycleState:
    """Realized lifecycle state for vanilla barrier-family products."""

    alive: bool = True
    knocked_in: bool = False
    knocked_out: bool = False
    expired: bool = False
    hit_date: Optional[datetime] = None
    realized_cashflows: float = 0.0
