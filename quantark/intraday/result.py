"""Structured intraday valuation result: price, PV components, provenance, identities, per-greek status."""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Optional, Tuple

from quantark.intraday.events import EventPhase
from quantark.intraday.fixings import AssumedFixing, ContinuousHistoryAssumption
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

GREEK_STATUSES = ("ok", "undefined", "failed", "not_requested")
GREEK_CONVENTIONS = ("point", "desk_bump", "engine")
_RECONCILE_TOL = 1e-12


@dataclass(frozen=True)
class GreekValue:
    """One sensitivity: a finite value iff ``status == "ok"``, otherwise a reason.

    ``reason`` on an ``ok`` value is a disclosure (the bump, the stencil, a resolution
    verdict); ``error_estimate`` is the estimator's own diagnostic, never a gate.
    """

    name: str
    value: Optional[float]
    unit: str
    convention: str
    bump: Optional[float] = None
    status: str = "ok"
    reason: Optional[str] = None
    error_estimate: Optional[float] = None

    def __post_init__(self):
        if self.error_estimate is not None and (not isfinite(float(self.error_estimate)) or self.error_estimate < 0.0):
            raise ValidationError(f"greek {self.name}: error_estimate must be finite and nonnegative")
        if self.status not in GREEK_STATUSES:
            raise ValidationError(f"unknown greek status {self.status!r}")
        if self.convention not in GREEK_CONVENTIONS:
            raise ValidationError(f"unknown greek convention {self.convention!r}")
        if self.status == "ok":
            if self.value is None or not isfinite(float(self.value)):
                raise ValidationError(f"greek {self.name}: status ok requires a finite value")
            object.__setattr__(self, "value", float(self.value))
        else:
            if self.value is not None:
                raise ValidationError(f"greek {self.name}: status {self.status} must not carry a value")
            if not self.reason:
                raise ValidationError(f"greek {self.name}: status {self.status} needs a reason")


@dataclass(frozen=True)
class CashflowComponent:
    kind: str                     # "contingent" | "pending_receivable" | "paid"
    amount_pv: float              # PV at valuation for contingent/pending; nominal for paid
    provenance: str               # "model" | "confirmed" | "provisional"
    event_id: Optional[str] = None
    cashflow_id: Optional[str] = None
    payment_tau: Optional[float] = None
    #: Why a ``provisional`` flow is conditional: the assumed events (or ``"continuous_history"``) whose actual
    #: outcome can change whether it exists or what it pays. Empty for a confirmed flow.
    depends_on: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if (self.provenance == "provisional") != bool(self.depends_on):
            raise ValidationError(f"cashflow {self.cashflow_id!r}: a provisional flow names what it depends on, "
                                  "and only a provisional flow does")
        if self.kind not in ("contingent", "pending_receivable", "paid"):
            raise ValidationError(f"unknown cashflow component kind {self.kind!r}")
        if self.provenance not in ("model", "confirmed", "provisional"):
            raise ValidationError(f"unknown provenance {self.provenance!r}")
        if not isfinite(float(self.amount_pv)):
            raise ValidationError("cashflow component amount must be finite")


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class IntradayValuationResult:
    """``price`` is the remaining claim's PV per contract (contingent + pending receivables); paid cash is reported apart."""

    price: float
    contingent_pv: float
    pending_receivable_pv: float
    paid_cash: float
    units: str
    valuation_timestamp: datetime
    phase: EventPhase
    provisional: bool
    assumptions: Tuple[AssumedFixing, ...]
    continuous_assumption: Optional[ContinuousHistoryAssumption]
    lifecycle: Mapping[str, object]
    cashflows: Tuple[CashflowComponent, ...]
    greeks: Tuple[GreekValue, ...]
    profile_identity: tuple
    session_identity: tuple
    market_snapshot_id: str
    context_identity: str
    engine: str
    method: str
    numerical: Mapping[str, object] = field(default_factory=dict)
    records: Tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("price", "contingent_pv", "pending_receivable_pv", "paid_cash"):
            if not isfinite(float(getattr(self, name))):
                raise ValidationError(f"{name} must be finite")
        if not is_close(self.price, self.contingent_pv + self.pending_receivable_pv,
                        rel_tol=_RECONCILE_TOL, abs_tol=_RECONCILE_TOL):
            raise ValidationError("price must reconcile to contingent_pv + pending_receivable_pv")
        object.__setattr__(self, "lifecycle", MappingProxyType(dict(self.lifecycle)))
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))
        object.__setattr__(self, "greeks", tuple(self.greeks))
        object.__setattr__(self, "cashflows", tuple(self.cashflows))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "records", tuple(self.records))

    def greek(self, name: str) -> GreekValue:
        for g in self.greeks:
            if g.name == name:
                return g
        raise ValidationError(f"greek {name!r} was not requested")

    def total_value(self, *, include_paid_cash: bool) -> float:
        """Price with or without the paid cash; the caller states which convention it reports."""
        return self.price + (self.paid_cash if include_paid_cash else 0.0)

    def to_dict(self) -> dict:
        """JSON-serialisable view (isoformat timestamps, enum values)."""
        return _jsonable(self)
