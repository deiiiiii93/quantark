"""
Shared futures hedge ledger: position accounting and contract roll policy.

Extracted verbatim from the OTC replay module so any backtest engine can
carry a rolled futures hedge with average-cost realized-PnL accounting.
(The equity multi-instrument executor migrates onto this in a later change.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from quantark.util.exceptions import ValidationError


@dataclass
class FuturesRollPolicy:
    """
    Rule for selecting and rolling Chinese equity index futures contracts.
    """

    roll_days_before_expiry: int = 5

    def __post_init__(self) -> None:
        if self.roll_days_before_expiry < 0:
            raise ValidationError("roll_days_before_expiry must be non-negative")

    def select_contract(
        self, futures_slice, valuation_date, current_contract: Optional[str] = None
    ):
        """
        Select the active contract from a daily futures-chain slice.
        """
        valuation_date = valuation_date.normalize()
        rows = futures_slice.copy()
        rows = rows[rows["expiry_date"] > valuation_date]
        if rows.empty:
            raise ValidationError(
                f"No non-expired futures contract on {valuation_date.date()}"
            )

        if current_contract is not None:
            current = rows[rows["contract"] == current_contract]
            if not current.empty:
                current_row = current.sort_values("expiry_date").iloc[0]
                days_to_expiry = (
                    current_row["expiry_date"] - valuation_date
                ).days
                if days_to_expiry > self.roll_days_before_expiry:
                    return current_row

        min_expiry = valuation_date + timedelta(days=self.roll_days_before_expiry)
        candidates = rows[rows["expiry_date"] > min_expiry]
        if candidates.empty:
            candidates = rows
        return candidates.sort_values(["expiry_date", "contract"]).iloc[0]


@dataclass
class FuturesHedgePosition:
    """Single active futures hedge position with realized PnL tracking."""

    contract: Optional[str] = None
    quantity: float = 0.0
    avg_price: float = 0.0
    multiplier: float = 1.0
    realized_pnl: float = 0.0

    def mark_to_market(self, price: float) -> float:
        if self.contract is None or abs(self.quantity) == 0:
            return float(self.realized_pnl)
        return float(
            self.realized_pnl
            + self.quantity * (float(price) - self.avg_price) * self.multiplier
        )

    def trade(self, quantity_delta: float, price: float, contract: str, multiplier: float) -> None:
        quantity_delta = float(quantity_delta)
        price = float(price)
        multiplier = float(multiplier)
        if abs(quantity_delta) < 1e-12:
            return

        if self.contract is None or abs(self.quantity) < 1e-12:
            self.contract = contract
            self.quantity = quantity_delta
            self.avg_price = price
            self.multiplier = multiplier
            return

        if self.contract != contract:
            raise ValidationError("Cannot trade a different contract without rolling")

        same_direction = self.quantity * quantity_delta > 0
        if same_direction:
            new_qty = self.quantity + quantity_delta
            self.avg_price = (
                self.avg_price * abs(self.quantity) + price * abs(quantity_delta)
            ) / abs(new_qty)
            self.quantity = new_qty
            self.multiplier = multiplier
            return

        close_qty = min(abs(self.quantity), abs(quantity_delta))
        direction = 1.0 if self.quantity > 0 else -1.0
        self.realized_pnl += direction * close_qty * (price - self.avg_price) * multiplier
        new_qty = self.quantity + quantity_delta
        if abs(new_qty) < 1e-12:
            self.quantity = 0.0
            self.avg_price = 0.0
            self.contract = None
            self.multiplier = multiplier
            return

        if self.quantity * new_qty < 0:
            self.avg_price = price
        self.quantity = new_qty
        self.contract = contract
        self.multiplier = multiplier
