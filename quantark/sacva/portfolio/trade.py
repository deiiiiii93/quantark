"""Position-level inputs for SA-CVA exposure (spec §3.1).

A ``CVATrade`` is data plus a handle to its pricing engine. ``engine.price(
product, env)`` returns the UNIT product value; the signed ``quantity`` is
applied once by the repricer (bank-perspective MtM, positive when the bank is
exposed to the counterparty).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

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
    equity_bucket: Optional[int] = None   # SA-CVA equity bucket (MAR50.70), for market sens
    fx_currency: Optional[str] = None     # SA-CVA FX factor (MAR50.59): the foreign ccy

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
        if self.equity_bucket is not None and (
                isinstance(self.equity_bucket, bool)
                or not isinstance(self.equity_bucket, int) or self.equity_bucket < 1):
            raise ValidationError(f"{self.trade_id}: equity_bucket must be a positive int")
        if self.fx_currency is not None:
            if not isinstance(self.fx_currency, str) or not self.fx_currency:
                raise ValidationError(f"{self.trade_id}: fx_currency must be a non-empty str")
            self.fx_currency = self.fx_currency.upper()
        if self.equity_bucket is not None and self.fx_currency is not None:
            raise ValidationError(
                f"{self.trade_id}: a trade declares one market factor — equity_bucket "
                "XOR fx_currency, not both")
        self.trade_currency = self.trade_currency.upper()


@dataclass
class CVAHedge(CVATrade):
    """Eligible CVA hedge; its market-value sensitivity is S_k^Hdg (spec §3.3)."""

    def __post_init__(self) -> None:
        self.hedge = True
        super().__post_init__()
