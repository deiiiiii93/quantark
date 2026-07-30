"""Append-only realized-cashflow ledger for equity option lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, TYPE_CHECKING

from quantark.util.exceptions import ValidationError

from .events import LifecycleEventType

if TYPE_CHECKING:
    from quantark.priceenv import PricingEnvironment


@dataclass(frozen=True)
class ValuationPoint:
    """One authoritative date or numeric-time lifecycle valuation point."""

    date: Optional[datetime] = None
    time: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.date is None) == (self.time is None):
            raise ValidationError(
                "valuation point requires exactly one representation"
            )
        if self.date is not None and not isinstance(self.date, datetime):
            raise ValidationError(
                "valuation point date must be datetime, "
                f"got {type(self.date).__name__}"
            )
        if self.time is not None:
            try:
                value = float(self.time)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"valuation point time must be numeric, got {self.time!r}"
                ) from exc
            if not isfinite(value):
                raise ValidationError(
                    f"valuation point time must be finite, got {self.time!r}"
                )
            object.__setattr__(self, "time", value)


@dataclass(frozen=True)
class RealizedCashflow:
    """Immutable cashflow fixed by a realized lifecycle event."""

    cashflow_id: str
    event_type: LifecycleEventType
    amount: float
    determination_date: Optional[datetime] = None
    determination_time: Optional[float] = None
    payment_date: Optional[datetime] = None
    payment_time: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.cashflow_id, str) or not self.cashflow_id.strip():
            raise ValidationError("cashflow_id must be a non-empty string")
        if not isinstance(self.event_type, LifecycleEventType):
            raise ValidationError(
                f"invalid lifecycle event type: {self.event_type!r}"
            )

        try:
            amount = float(self.amount)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"cashflow amount must be numeric, got {self.amount!r}"
            ) from exc
        if not isfinite(amount):
            raise ValidationError(
                f"cashflow amount must be finite, got {self.amount!r}"
            )
        object.__setattr__(self, "amount", amount)

        self._validate_representation_pair(
            self.determination_date,
            self.payment_date,
            "date",
        )
        self._validate_representation_pair(
            self.determination_time,
            self.payment_time,
            "time",
        )
        if self.determination_date is None and self.determination_time is None:
            raise ValidationError(
                "cashflow determination requires a date or numeric time"
            )
        if self.payment_date is None and self.payment_time is None:
            raise ValidationError(
                "cashflow payment requires a date or numeric time"
            )

        for name in ("determination_date", "payment_date"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, datetime):
                raise ValidationError(
                    f"{name} must be datetime, got {type(value).__name__}"
                )

        for name in ("determination_time", "payment_time"):
            value = getattr(self, name)
            if value is None:
                continue
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{name} must be numeric, got {value!r}"
                ) from exc
            if not isfinite(normalized):
                raise ValidationError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, normalized)

        if (
            self.determination_date is not None
            and self.payment_date < self.determination_date
        ):
            raise ValidationError(
                "cashflow payment date cannot be before determination date"
            )
        if (
            self.determination_time is not None
            and self.payment_time < self.determination_time
        ):
            raise ValidationError(
                "cashflow payment time cannot be before determination time"
            )

        try:
            metadata = dict(self.metadata)
        except (TypeError, ValueError) as exc:
            raise ValidationError("cashflow metadata must be a mapping") from exc
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    @staticmethod
    def _validate_representation_pair(
        determination_value,
        payment_value,
        label: str,
    ) -> None:
        if (determination_value is None) != (payment_value is None):
            raise ValidationError(
                f"cashflow {label} representation must include both "
                "determination and payment"
            )

    def __deepcopy__(self, memo):
        """Frozen payloads are safe to share between ledger snapshots."""
        return self


class LifecycleCashflowLedger:
    """Append-only ledger keyed by stable, externally meaningful IDs."""

    def __init__(
        self, cashflows: Iterable[RealizedCashflow] = ()
    ) -> None:
        self._cashflows: dict[str, RealizedCashflow] = {}
        for cashflow in cashflows:
            self.register(cashflow)

    @property
    def cashflows(self) -> tuple[RealizedCashflow, ...]:
        """Return all cashflows in deterministic ID order."""
        return tuple(
            self._cashflows[cashflow_id]
            for cashflow_id in sorted(self._cashflows)
        )

    def register(self, cashflow: RealizedCashflow) -> bool:
        """Append a cashflow, returning False for an identical retry."""
        if not isinstance(cashflow, RealizedCashflow):
            raise ValidationError(
                "ledger entries must be RealizedCashflow, "
                f"got {type(cashflow).__name__}"
            )
        existing = self._cashflows.get(cashflow.cashflow_id)
        if existing is None:
            self._cashflows[cashflow.cashflow_id] = cashflow
            return True
        if existing == cashflow:
            return False
        raise ValidationError(
            f"conflicting cashflow payload for id={cashflow.cashflow_id!r}"
        )

    def pending(
        self, valuation_point: ValuationPoint
    ) -> tuple[RealizedCashflow, ...]:
        """Return cashflows payable strictly after the valuation point."""
        return tuple(
            cashflow
            for cashflow in self.cashflows
            if self._is_pending(cashflow, valuation_point)
        )

    def paid(
        self, valuation_point: ValuationPoint
    ) -> tuple[RealizedCashflow, ...]:
        """Return cashflows payable at or before the valuation point."""
        return tuple(
            cashflow
            for cashflow in self.cashflows
            if not self._is_pending(cashflow, valuation_point)
        )

    def paid_total(self, valuation_point: ValuationPoint) -> float:
        return float(
            sum(cashflow.amount for cashflow in self.paid(valuation_point))
        )

    def pending_pv(
        self,
        valuation_point: ValuationPoint,
        pricing_env: "PricingEnvironment",
    ) -> float:
        """Discount pending fixed receivables from valuation to payment."""
        from quantark.asset.equity.settlement import SettlementResolver

        if valuation_point.date is not None:
            if pricing_env is None:
                raise ValidationError(
                    "pricing environment is required for pending cashflow PV"
                )
            if pricing_env.valuation_date != valuation_point.date:
                raise ValidationError(
                    "pricing environment valuation_date must match "
                    "the ledger valuation point"
                )

        total = 0.0
        for cashflow in self.pending(valuation_point):
            timing = SettlementResolver.resolve_pending(
                cashflow,
                pricing_env,
                valuation_point=valuation_point,
            )
            total += cashflow.amount * timing.payment_df
        return float(total)

    @staticmethod
    def _is_pending(
        cashflow: RealizedCashflow,
        valuation_point: ValuationPoint,
    ) -> bool:
        if valuation_point.date is not None:
            if cashflow.payment_date is None:
                raise ValidationError(
                    f"cashflow {cashflow.cashflow_id!r} has no payment_date "
                    "for date-based valuation"
                )
            return cashflow.payment_date > valuation_point.date

        if cashflow.payment_time is None:
            raise ValidationError(
                f"cashflow {cashflow.cashflow_id!r} has no payment_time "
                "for time-based valuation"
            )
        return cashflow.payment_time > valuation_point.time

    def __deepcopy__(self, memo):
        copied = type(self)()
        memo[id(self)] = copied
        copied._cashflows = dict(self._cashflows)
        return copied


__all__ = [
    "LifecycleCashflowLedger",
    "RealizedCashflow",
    "ValuationPoint",
]
