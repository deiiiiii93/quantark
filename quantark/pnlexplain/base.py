"""Row schema and result container shared by every explain method.

Additivity contract (spec §5.4): for a method M, the COMPONENT rows with
method in {M, SHARED} sum to total_pnl. INFORMATIONAL rows (sub-decompositions)
and SUMMARY rows (total) are never summed.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

from quantark.util.exceptions import NumericalError, ValidationError


class Factor(Enum):
    TIME = "time"
    SPOT = "spot"
    VOL = "vol"
    RATE = "rate"
    DIVIDEND = "dividend"
    BASIS = "basis"
    MODEL = "model"
    LIFECYCLE_EVENT = "lifecycle_event"
    TRADE = "trade"
    TRANSACTION_COST = "transaction_cost"
    UNEXPLAINED = "unexplained"
    TOTAL = "total"


MARKET_FACTORS: Tuple[Factor, ...] = (
    Factor.TIME, Factor.SPOT, Factor.VOL, Factor.RATE, Factor.DIVIDEND,
    Factor.BASIS, Factor.MODEL,
)


class ExplainMethod(Enum):
    WATERFALL = "waterfall"
    TAYLOR = "taylor"
    SHARED = "shared"


class RowKind(Enum):
    COMPONENT = "component"
    INFORMATIONAL = "informational"
    SUMMARY = "summary"


LEVELS: Tuple[str, ...] = ("instrument", "position", "portfolio")
MOVE_KEYS: Tuple[str, ...] = (
    "spot_return", "vol_pts", "rate_pct", "div_pct", "basis_pct", "days",
    "trading_days", "tenor",
)
FRAME_COLUMNS = [
    "level", "position_id", "underlying", "method", "kind", "factor", "term",
    "step", "pnl", "greek", "cash_greek", *MOVE_KEYS,
]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


def _frozen(mapping: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if mapping is None:
        return _EMPTY
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))


def _finite(value: Any, what: str) -> float:
    """Coerce a numeric row field to a finite float.

    A value ``float()`` cannot read is bad input (ValidationError); a value
    that reads as NaN/inf is a non-finite result (NumericalError).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(f):
        raise NumericalError(f"non-finite {what}: {value!r}")
    return f


@dataclass(frozen=True)
class ExplainRow:
    """One attribution row (money amounts for the whole position)."""

    factor: Factor
    term: str
    method: ExplainMethod
    kind: RowKind
    level: str
    pnl: float
    moves: Mapping[str, float] = field(default_factory=lambda: _EMPTY)
    greek: Optional[float] = None
    cash_greek: Optional[float] = None
    step: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValidationError(f"unknown row level {self.level!r}")
        if not isinstance(self.factor, Factor) or not isinstance(self.method, ExplainMethod) \
                or not isinstance(self.kind, RowKind):
            raise ValidationError("factor/method/kind must be the pnlexplain enums")
        where = f"row {self.method.value}/{self.term}"
        object.__setattr__(self, "pnl", _finite(self.pnl, f"PnL in {where}"))
        for name in ("greek", "cash_greek"):
            val = getattr(self, name)
            if val is not None:
                object.__setattr__(self, name, _finite(val, f"{name} in {where}"))
        moves = dict(self.moves or {})
        unknown = sorted(set(moves) - set(MOVE_KEYS))
        if unknown:
            raise ValidationError(f"unknown move keys {unknown}; allowed {MOVE_KEYS}")
        object.__setattr__(self, "moves", MappingProxyType(
            {k: _finite(v, f"move {k} in {where}") for k, v in moves.items()}
        ))
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def relabel(self, level: str) -> "ExplainRow":
        return dataclasses.replace(self, level=level)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "level": self.level, "position_id": "", "underlying": "",
            "method": self.method.value, "kind": self.kind.value,
            "factor": self.factor.value, "term": self.term, "step": self.step,
            "pnl": self.pnl, "greek": self.greek, "cash_greek": self.cash_greek,
        }
        for key in MOVE_KEYS:
            out[key] = self.moves.get(key, math.nan)
        return out


def component_sum(rows: Iterable[ExplainRow], method: ExplainMethod) -> float:
    """Exactly rounded sum of the COMPONENT rows of `method` (and SHARED rows)."""
    return math.fsum(
        row.pnl for row in rows
        if row.kind is RowKind.COMPONENT and row.method in (method, ExplainMethod.SHARED)
    )


def make_total_row(level: str, pnl: float, **metadata: Any) -> ExplainRow:
    return ExplainRow(
        factor=Factor.TOTAL, term="total", method=ExplainMethod.SHARED,
        kind=RowKind.SUMMARY, level=level, pnl=pnl, metadata=metadata,
    )


def rows_to_frame(
    rows: Iterable[ExplainRow],
    *,
    date: Optional[datetime] = None,
    position_id: str = "",
    underlying: str = "",
) -> pd.DataFrame:
    records = []
    for row in rows:
        rec = row.to_dict()
        rec["position_id"] = position_id
        rec["underlying"] = underlying
        if date is not None:
            rec = {"date": pd.Timestamp(date), **rec}
        records.append(rec)
    columns = (["date"] if date is not None else []) + FRAME_COLUMNS
    frame = pd.DataFrame.from_records(records, columns=columns)
    if date is not None:
        # Explicit nanosecond resolution: pandas>=3 infers datetime64[us] from
        # Python datetimes, which would make the dtype depend on the pandas version.
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    return frame


@dataclass(frozen=True)
class ValueBreakdown:
    """Position value = contingent MTM + pending receivable PV + paid cash."""

    contingent_mtm: float
    pending_receivable_pv: float
    paid_cash: float

    def __post_init__(self) -> None:
        for name in ("contingent_mtm", "pending_receivable_pv", "paid_cash"):
            val = float(getattr(self, name))
            if not math.isfinite(val):
                raise NumericalError(f"non-finite {name}: {getattr(self, name)!r}")
            object.__setattr__(self, name, val)

    @property
    def total(self) -> float:
        return self.contingent_mtm + self.pending_receivable_pv + self.paid_cash

    def to_dict(self) -> Dict[str, float]:
        return {
            "contingent_mtm": self.contingent_mtm,
            "pending_receivable_pv": self.pending_receivable_pv,
            "paid_cash": self.paid_cash, "total": self.total,
        }


@dataclass(frozen=True)
class PnLExplainResult:
    date_t0: datetime
    date_t1: datetime
    pv_t0: ValueBreakdown
    pv_alive_t1: ValueBreakdown
    pv_t1: ValueBreakdown
    total_pnl: float
    moves: Any                      # FactorMoves (equity); typed loosely to avoid a cycle
    rows: Tuple[ExplainRow, ...]
    unexplained: Optional[float]
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_pnl", _finite(self.total_pnl, "total_pnl"))
        if self.unexplained is not None:
            object.__setattr__(self, "unexplained", _finite(self.unexplained, "unexplained"))
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", _frozen(self.metadata))

    def rows_for(self, method: ExplainMethod, *, kind: Optional[RowKind] = None
                 ) -> Tuple[ExplainRow, ...]:
        return tuple(
            r for r in self.rows
            if r.method in (method, ExplainMethod.SHARED) and (kind is None or r.kind is kind)
        )

    def by_factor(self, method: ExplainMethod) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in self.rows_for(method, kind=RowKind.COMPONENT):
            out[r.factor.value] = out.get(r.factor.value, 0.0) + r.pnl
        return out

    def reconcile(self, method: ExplainMethod) -> float:
        return self.total_pnl - component_sum(self.rows, method)

    def to_frame(self) -> pd.DataFrame:
        return rows_to_frame(self.rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date_t0": self.date_t0.isoformat(), "date_t1": self.date_t1.isoformat(),
            "pv_t0": self.pv_t0.to_dict(), "pv_alive_t1": self.pv_alive_t1.to_dict(),
            "pv_t1": self.pv_t1.to_dict(), "total_pnl": self.total_pnl,
            "unexplained": self.unexplained, "rows": [r.to_dict() for r in self.rows],
            "metadata": dict(self.metadata),
        }
