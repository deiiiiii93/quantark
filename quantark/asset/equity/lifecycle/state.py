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
    # Pending-settlement state: a terminal cashflow whose settlement date lies
    # after its observation date is parked here (economic termination and cash
    # posting are separate moments); ``settled`` marks cash fully posted.
    pending_settlement_cashflow: float = 0.0
    settlement_date: Optional[datetime] = None
    settled: bool = False

    def mark_ki(self, timestamp: datetime) -> bool:
        if self.knocked_in or self.knocked_out:
            return False
        self.knocked_in = True
        self.ki_date = timestamp
        return True

    def mark_ko(
        self,
        timestamp: datetime,
        cashflow: float = 0.0,
        settlement_date: Optional[datetime] = None,
    ) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.knocked_out = True
        self.alive = False
        self.ko_date = timestamp
        if settlement_date is not None and settlement_date > timestamp:
            # Delayed settlement: park the cash as a receivable.
            self.pending_settlement_cashflow += float(cashflow)
            self.settlement_date = settlement_date
        else:
            # T+0 (or unspecified): exactly the historical behavior.
            self.realized_cashflows += float(cashflow)
            self.settlement_date = settlement_date or timestamp
            self.settled = True
        return True

    def mark_maturity(self, timestamp: datetime, cashflow: float = 0.0) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.matured = True
        self.alive = False
        self.maturity_date = timestamp
        self.realized_cashflows += float(cashflow)
        # Maturity settlement is immediate — without this, clean/KI-maturity
        # runs would never satisfy the all-settled termination predicate.
        self.settlement_date = timestamp
        self.settled = True
        return True

    def settle(self) -> float:
        """Post the pending receivable to realized cash; returns the amount."""
        amount = float(self.pending_settlement_cashflow)
        if amount != 0.0 or (not self.settled and not self.alive):
            self.realized_cashflows += amount
            self.pending_settlement_cashflow = 0.0
            self.settled = True
        return amount

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
