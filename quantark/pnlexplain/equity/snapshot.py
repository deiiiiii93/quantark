"""ValuationSnapshot and the value identity (spec §5.1-5.2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
from quantark.pnlexplain.base import ValueBreakdown
from quantark.util.exceptions import NumericalError, ValidationError

_MISSING = object()


def is_terminal(state: Any) -> bool:
    """Same predicate as settlement_support.terminal_lifecycle_pv."""
    if state is None:
        return False
    return (
        getattr(state, "alive", None) is False
        or bool(getattr(state, "matured", False))
        or bool(getattr(state, "expired", False))
        or bool(getattr(state, "knocked_out", False))
    )


@dataclass(frozen=True)
class ValuationSnapshot:
    product: Any
    engine: Any
    pricing_env: Any
    date: datetime
    quantity: float = 1.0
    lifecycle_state: Any = None
    valuation_point: Optional[ValuationPoint] = None
    currency: Optional[str] = None
    label: str = ""

    def __post_init__(self) -> None:
        d = self.date
        if not isinstance(d, datetime) or d.tzinfo is not None:
            raise ValidationError("snapshot date must be a naive datetime")
        if (d.hour, d.minute, d.second, d.microsecond) != (0, 0, 0, 0):
            raise ValidationError("snapshot date must be midnight (date-only snapshots)")
        env_date = self.pricing_env.valuation_date
        if env_date != d:
            raise ValidationError(
                f"snapshot date {d} must equal pricing_env.valuation_date {env_date}"
            )
        spot = float(self.pricing_env.spot)
        if not math.isfinite(spot) or spot <= 0.0:
            raise ValidationError(f"spot must be positive and finite, got {spot}")
        q = float(self.quantity)
        if not math.isfinite(q) or q == 0.0:
            raise ValidationError(f"quantity must be non-zero and finite, got {self.quantity}")
        object.__setattr__(self, "quantity", q)
        vp = self.valuation_point
        if vp is not None and vp.date is not None and vp.date != d:
            raise ValidationError("a date-based valuation_point must equal the snapshot date")

    @property
    def point(self) -> ValuationPoint:
        return self.valuation_point if self.valuation_point is not None else ValuationPoint(date=self.date)


def value(
    snapshot: ValuationSnapshot,
    *,
    engine: Any = None,
    product: Any = None,
    pricing_env: Any = None,
    valuation_point: Optional[ValuationPoint] = None,
    lifecycle_state: Any = _MISSING,
) -> ValueBreakdown:
    """contingent MTM (quantity x engine price; 0 once terminal) + ledger PV + paid cash.

    The engine is NOT handed the lifecycle state: receivables are valued here
    from the ledger exactly as the backtests do, so nothing is counted twice.
    """
    engine = snapshot.engine if engine is None else engine
    product = snapshot.product if product is None else product
    env = snapshot.pricing_env if pricing_env is None else pricing_env
    point = snapshot.point if valuation_point is None else valuation_point
    state = snapshot.lifecycle_state if lifecycle_state is _MISSING else lifecycle_state

    if is_terminal(state):
        contingent = 0.0
    else:
        price = float(engine.price(product, env))
        if not math.isfinite(price):
            raise NumericalError(
                f"engine {type(engine).__name__} returned a non-finite price {price!r}"
            )
        contingent = snapshot.quantity * price
    pending = paid = 0.0
    if state is not None:
        ledger = getattr(state, "ledger", None)
        if ledger is None:
            raise ValidationError("lifecycle_state requires a cashflow ledger")
        if ledger.cashflows:
            pending = float(ledger.pending_pv(point, env))
            paid = float(ledger.paid_total(point))
    return ValueBreakdown(contingent_mtm=contingent, pending_receivable_pv=pending, paid_cash=paid)
