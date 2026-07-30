"""
Lifecycle state objects for equity structured products.

``AutocallableLifecycleState`` was extracted from
``quantark/backtest/otc/state.py`` (which re-exports it for backward
compatibility). ``BarrierLifecycleState`` is the simpler analogue for the
vanilla barrier product family.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

from quantark.util.exceptions import ValidationError

from .cashflows import (
    LifecycleCashflowLedger,
    RealizedCashflow,
    ValuationPoint,
)
from .events import LifecycleEventType


class EquityOptionLifecycleState(Protocol):
    """Structural lifecycle state consumed by settlement-aware engines."""

    valuation_point: Optional[ValuationPoint]
    ledger: LifecycleCashflowLedger


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
    valuation_point: Optional[ValuationPoint] = None
    ledger: LifecycleCashflowLedger = field(
        default_factory=LifecycleCashflowLedger
    )
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
        self._advance_valuation_point(timestamp)
        self.ledger.register(
            RealizedCashflow(
                cashflow_id=f"knock-out:{timestamp.isoformat()}",
                event_type=LifecycleEventType.KNOCK_OUT,
                amount=cashflow,
                determination_date=timestamp,
                payment_date=settlement_date or timestamp,
            )
        )
        if settlement_date is not None and settlement_date > timestamp:
            # Delayed settlement: the cash is determined now but pays later.
            # The ledger carries the dated flow; this scalar mirror is what
            # backtest.replay reads to price the receivable day by day.
            self.pending_settlement_cashflow += float(cashflow)
            self.settlement_date = settlement_date
        else:
            # T+0 (or unspecified): exactly the historical behavior.
            self.settlement_date = settlement_date or timestamp
            self.settled = True
        return True

    def mark_maturity(self, timestamp: datetime, cashflow: float = 0.0) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.matured = True
        self.alive = False
        self.maturity_date = timestamp
        self._advance_valuation_point(timestamp)
        self.ledger.register(
            RealizedCashflow(
                cashflow_id=f"maturity:{timestamp.isoformat()}",
                event_type=LifecycleEventType.MATURITY,
                amount=cashflow,
                determination_date=timestamp,
                payment_date=timestamp,
            )
        )
        # Maturity settlement is immediate — without this, clean/KI-maturity
        # runs would never satisfy the all-settled termination predicate.
        self.settlement_date = timestamp
        self.settled = True
        return True

    def settle(self) -> float:
        """Post the pending receivable to paid cash; returns the amount.

        The ledger already holds the dated flow (registered by ``mark_ko``);
        posting means advancing the valuation point past its payment date so
        ``realized_cashflows`` counts it as paid.
        """
        amount = float(self.pending_settlement_cashflow)
        if amount != 0.0 or (not self.settled and not self.alive):
            if self.settlement_date is not None:
                self._advance_valuation_point(self.settlement_date)
            self.pending_settlement_cashflow = 0.0
            self.settled = True
        return amount

    def add_cashflow(
        self,
        amount: float,
        *,
        cashflow_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        timestamp = timestamp or (
            self.valuation_point.date
            if self.valuation_point is not None
            else None
        )
        if timestamp is None:
            raise ValidationError(
                "timestamp or date-based valuation_point is required "
                "to register a cashflow"
            )
        self._advance_valuation_point(timestamp)
        self.ledger.register(
            RealizedCashflow(
                cashflow_id=(
                    cashflow_id
                    or f"coupon:{timestamp.isoformat()}:{len(self.ledger.cashflows)}"
                ),
                event_type=LifecycleEventType.COUPON,
                amount=amount,
                determination_date=timestamp,
                payment_date=timestamp,
            )
        )

    def _advance_valuation_point(self, date: datetime) -> None:
        """Move the valuation point forward, never backward: replay feeds
        events chronologically, and a paid flow must stay paid."""
        current = (
            self.valuation_point.date if self.valuation_point is not None else None
        )
        if current is None or date > current:
            self.valuation_point = ValuationPoint(date=date)

    @property
    def realized_cashflows(self) -> float:
        if self.valuation_point is None:
            return 0.0
        return self.ledger.paid_total(self.valuation_point)


@dataclass
class BarrierLifecycleState:
    """Realized lifecycle state for vanilla barrier-family products."""

    alive: bool = True
    knocked_in: bool = False
    knocked_out: bool = False
    expired: bool = False
    hit_date: Optional[datetime] = None
    valuation_point: Optional[ValuationPoint] = None
    ledger: LifecycleCashflowLedger = field(
        default_factory=LifecycleCashflowLedger
    )

    @property
    def realized_cashflows(self) -> float:
        if self.valuation_point is None:
            return 0.0
        return self.ledger.paid_total(self.valuation_point)
