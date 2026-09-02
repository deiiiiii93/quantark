"""Normalised trade schema (spec §9)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Optional

from quantark.util.exceptions import ValidationError

TRADE_KINDS = ("open", "adjust", "close", "roll_close", "roll_open")


@dataclass(frozen=True)
class ExplainTrade:
    position_id: str
    quantity: float                      # signed position UNITS: buy > 0, sell < 0
    price: float                         # per unit, same convention as engine.price
    transaction_cost: float = 0.0
    timestamp: Optional[datetime] = None
    kind: str = "adjust"
    instrument_type: str = ""
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.position_id:
            raise ValidationError("trade requires a position_id")
        q, p, c = float(self.quantity), float(self.price), float(self.transaction_cost)
        if not math.isfinite(q) or q == 0.0:
            raise ValidationError(f"trade quantity must be non-zero and finite, got {self.quantity}")
        if not math.isfinite(p):
            raise ValidationError(f"trade price must be finite, got {self.price}")
        if not math.isfinite(c) or c < 0.0:
            raise ValidationError(f"transaction_cost must be >= 0, got {self.transaction_cost}")
        if self.kind not in TRADE_KINDS:
            raise ValidationError(f"trade kind must be one of {TRADE_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "quantity", q)
        object.__setattr__(self, "price", p)
        object.__setattr__(self, "transaction_cost", c)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def cash(self) -> float:
        """Cash flow of the trade: buying costs cash."""
        return -self.quantity * self.price

    @classmethod
    def from_contracts(cls, position_id: str, contracts: float, price: float, multiplier: float,
                       **kw: Any) -> "ExplainTrade":
        if not math.isfinite(float(multiplier)) or float(multiplier) <= 0.0:
            raise ValidationError(f"multiplier must be positive, got {multiplier}")
        return cls(position_id=position_id, quantity=float(contracts) * float(multiplier),
                   price=float(price), **kw)
