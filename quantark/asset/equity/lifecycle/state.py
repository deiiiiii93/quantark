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
        *,
        realized_cashflow: Optional[RealizedCashflow] = None,
        valuation_point: Optional[ValuationPoint] = None,
    ) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.knocked_out = True
        self.alive = False
        self.ko_date = timestamp
        entry = realized_cashflow or RealizedCashflow(
            cashflow_id=f"knock-out:{timestamp.isoformat()}",
            event_type=LifecycleEventType.KNOCK_OUT,
            amount=cashflow,
            determination_date=timestamp,
            payment_date=settlement_date or timestamp,
        )
        self._advance_to(valuation_point or self._point_for(entry))
        self.ledger.register(entry)
        self._mirror_terminal_settlement(entry, timestamp)
        return True

    def mark_maturity(
        self,
        timestamp: datetime,
        cashflow: float = 0.0,
        *,
        realized_cashflow: Optional[RealizedCashflow] = None,
        valuation_point: Optional[ValuationPoint] = None,
    ) -> bool:
        if self.knocked_out or self.matured:
            return False
        self.matured = True
        self.alive = False
        self.maturity_date = timestamp
        entry = realized_cashflow or RealizedCashflow(
            cashflow_id=f"maturity:{timestamp.isoformat()}",
            event_type=LifecycleEventType.MATURITY,
            amount=cashflow,
            determination_date=timestamp,
            payment_date=timestamp,
        )
        self._advance_to(valuation_point or self._point_for(entry))
        self.ledger.register(entry)
        # A default maturity entry pays at determination, so this marks the
        # position settled immediately — without that, clean/KI-maturity runs
        # would never satisfy the all-settled termination predicate. An entry
        # carrying a later payment date parks the cash as pending instead.
        self._mirror_terminal_settlement(entry, timestamp)
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
        realized_cashflow: Optional[RealizedCashflow] = None,
        valuation_point: Optional[ValuationPoint] = None,
    ) -> bool:
        if realized_cashflow is not None:
            self._advance_to(valuation_point or self._point_for(realized_cashflow))
            return self.ledger.register(realized_cashflow)

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
        return self.ledger.register(
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

    def _advance_to(self, point: ValuationPoint) -> None:
        """Advance to a resolved point; time-based points replace outright
        (numeric-only products never mix representations mid-run)."""
        if point.date is not None:
            self._advance_valuation_point(point.date)
        else:
            self.valuation_point = point

    def _mirror_terminal_settlement(
        self, entry: RealizedCashflow, timestamp: datetime
    ) -> None:
        """Mirror a terminal flow into the scalar receivable fields.

        The ledger carries the dated flow; ``pending_settlement_cashflow`` /
        ``settlement_date`` / ``settled`` are what backtest.replay reads to
        price the receivable day by day and to terminate the run.
        """
        delayed = (
            entry.payment_date is not None
            and entry.determination_date is not None
            and entry.payment_date > entry.determination_date
        ) or (
            entry.payment_date is None
            and entry.payment_time is not None
            and entry.determination_time is not None
            and entry.payment_time > entry.determination_time
        )
        if delayed:
            self.pending_settlement_cashflow += float(entry.amount)
            if entry.payment_date is not None:
                self.settlement_date = entry.payment_date
        else:
            # Paid at determination (or unspecified): historical behavior.
            self.settlement_date = entry.payment_date or timestamp
            self.settled = True

    @staticmethod
    def _point_for(cashflow: RealizedCashflow) -> ValuationPoint:
        if cashflow.determination_date is not None:
            return ValuationPoint(date=cashflow.determination_date)
        return ValuationPoint(time=cashflow.determination_time)

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
