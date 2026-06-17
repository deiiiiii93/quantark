"""Position-level inputs for SA-CVA exposure (spec §3.1).

A ``CVATrade`` is data plus a handle to its pricing engine. ``engine.price(
product, env)`` returns the UNIT product value; the signed ``quantity`` is
applied once by the repricer (bank-perspective MtM, positive when the bank is
exposed to the counterparty).
"""

import math
from dataclasses import dataclass
from typing import Any

from quantark.util.exceptions import ValidationError


@dataclass
class CVATrade:
    """One derivative in a netting set."""

    trade_id: str
    product: Any
    engine: Any
    env: Any
    quantity: float = 1.0
    trade_currency: str = "USD"
    hedge: bool = False

    def __post_init__(self) -> None:
        if not self.trade_id:
            raise ValidationError("CVATrade requires a trade_id")
        if self.product is None or self.engine is None or self.env is None:
            raise ValidationError(f"{self.trade_id}: product/engine/env required")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, (int, float)):
            raise ValidationError(f"{self.trade_id}: quantity must be numeric")
        if not math.isfinite(self.quantity):
            raise ValidationError(f"{self.trade_id}: quantity must be finite")
        if self.quantity == 0.0:
            raise ValidationError(f"{self.trade_id}: quantity must be non-zero")
        if not self.trade_currency:
            raise ValidationError(f"{self.trade_id}: trade_currency required")
        if not isinstance(self.hedge, bool):
            raise ValidationError(f"{self.trade_id}: hedge must be bool")
        self.trade_currency = self.trade_currency.upper()


@dataclass
class CVAHedge(CVATrade):
    """Eligible CVA hedge; its market-value sensitivity is S_k^Hdg (spec §3.3)."""

    def __post_init__(self) -> None:
        self.hedge = True
        super().__post_init__()
