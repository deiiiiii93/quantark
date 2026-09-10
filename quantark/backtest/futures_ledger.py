"""
Shared futures hedge ledger: position accounting and contract roll policy.

Extracted verbatim from the OTC replay module so any backtest engine can
carry a rolled futures hedge with average-cost realized-PnL accounting.
(The equity multi-instrument executor migrates onto this in a later change.)
"""

from __future__ import annotations

import math
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


#: The historical flat threshold.  A trade smaller than this is a no-op and a
#: residual quantity smaller than this closes the leg.
FLAT_QUANTITY = 1e-12


@dataclass(frozen=True)
class LegState:
    """One contract's open position: quantity, average cost and contract size."""

    contract: Optional[str] = None
    quantity: float = 0.0
    avg_price: float = 0.0
    multiplier: float = 1.0

    @property
    def is_flat(self) -> bool:
        return self.contract is None or abs(self.quantity) < FLAT_QUANTITY

    def unrealized(self, mark: float) -> float:
        """``quantity * (mark - avg_price) * multiplier``, in that order."""
        return self.quantity * (float(mark) - self.avg_price) * self.multiplier


def apply_trade(
    leg: LegState,
    realized_pnl: float,
    *,
    quantity_delta: float,
    price: float,
    contract: str,
    multiplier: float,
) -> tuple:
    """The shared accounting transition: ``(leg, book realized) -> (leg, realized)``.

    ``realized_pnl`` is the BOOK's single accumulator, not a per-leg one:
    accumulating per leg and summing afterwards would reorder the additions
    and drift from the historical total.

    Every expression below is copied unchanged from the original
    ``FuturesHedgePosition.trade``, including the order in which the average
    price is computed from the OLD quantity before the quantity is replaced.
    """
    quantity_delta = float(quantity_delta)
    price = float(price)
    multiplier = float(multiplier)
    if abs(quantity_delta) < FLAT_QUANTITY:
        return leg, realized_pnl

    if leg.contract is None or abs(leg.quantity) < FLAT_QUANTITY:
        return (
            LegState(
                contract=contract,
                quantity=quantity_delta,
                avg_price=price,
                multiplier=multiplier,
            ),
            realized_pnl,
        )

    if leg.contract != contract:
        raise ValidationError("Cannot trade a different contract without rolling")

    same_direction = leg.quantity * quantity_delta > 0
    if same_direction:
        new_qty = leg.quantity + quantity_delta
        avg_price = (
            leg.avg_price * abs(leg.quantity) + price * abs(quantity_delta)
        ) / abs(new_qty)
        return (
            LegState(
                contract=leg.contract,
                quantity=new_qty,
                avg_price=avg_price,
                multiplier=multiplier,
            ),
            realized_pnl,
        )

    close_qty = min(abs(leg.quantity), abs(quantity_delta))
    direction = 1.0 if leg.quantity > 0 else -1.0
    realized_pnl += direction * close_qty * (price - leg.avg_price) * multiplier
    new_qty = leg.quantity + quantity_delta
    if abs(new_qty) < FLAT_QUANTITY:
        return (
            LegState(
                contract=None, quantity=0.0, avg_price=0.0, multiplier=multiplier
            ),
            realized_pnl,
        )

    avg_price = price if leg.quantity * new_qty < 0 else leg.avg_price
    return (
        LegState(
            contract=contract,
            quantity=new_qty,
            avg_price=avg_price,
            multiplier=multiplier,
        ),
        realized_pnl,
    )


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
        leg, realized = apply_trade(
            LegState(
                contract=self.contract,
                quantity=self.quantity,
                avg_price=self.avg_price,
                multiplier=self.multiplier,
            ),
            self.realized_pnl,
            quantity_delta=quantity_delta,
            price=price,
            contract=contract,
            multiplier=multiplier,
        )
        self.contract = leg.contract
        self.quantity = leg.quantity
        self.avg_price = leg.avg_price
        self.multiplier = leg.multiplier
        self.realized_pnl = realized


class FuturesHedgeBook:
    """Several futures legs with ONE book-level realized-P&L accumulator.

    The single-leg path is the historical one, expression for expression, so
    a book that only ever holds one contract reproduces the old goldens bit
    for bit.  Realized P&L is added once at the book level, in trade order;
    unrealized P&L is summed over open legs in contract order so a mark is
    reproducible.

    A fully closed leg is dropped from the open set but its multiplier is
    remembered, because the legacy single-leg view reports the last contract
    size after a full close and a fresh default would report ``1.0``.
    """

    def __init__(self) -> None:
        self._legs: dict = {}
        self.realized_pnl: float = 0.0
        self._last_multiplier: float = 1.0

    # -- inspection -----------------------------------------------------

    def contracts(self) -> tuple:
        """Open contracts, in a deterministic (sorted) order."""
        return tuple(sorted(self._legs))

    def quantity(self, contract: str) -> float:
        leg = self._legs.get(contract)
        return 0.0 if leg is None else leg.quantity

    def multiplier(self, contract: str) -> float:
        leg = self._legs.get(contract)
        return self._last_multiplier if leg is None else leg.multiplier

    def holdings(self) -> dict:
        return {c: self._legs[c].quantity for c in self.contracts()}

    def gross(self) -> float:
        """Gross open contracts, ``sum |h_i|``."""
        return sum(abs(self._legs[c].quantity) for c in self.contracts())

    def gross_notional(self, prices) -> float:
        """``sum |h_i| m_i F_i`` over open legs."""
        self._require_prices(prices)
        return sum(
            abs(self._legs[c].quantity) * self._legs[c].multiplier * float(prices[c])
            for c in self.contracts()
        )

    def spot_delta(self, prices, spot: float) -> float:
        """HEDGE-ONLY currency spot sensitivity ``sum h_i m_i F_i / S``.

        This is not the net book delta: the product's own delta is not in it.
        """
        spot = float(spot)
        if not math.isfinite(spot) or spot <= 0.0:
            raise ValidationError("spot must be finite and positive")
        self._require_prices(prices)
        return sum(
            self._legs[c].quantity * self._legs[c].multiplier * float(prices[c]) / spot
            for c in self.contracts()
        )

    # -- accounting -----------------------------------------------------

    def trade(
        self,
        contract: str,
        quantity_delta: float,
        price: float,
        multiplier: float,
    ) -> None:
        """Apply one trade after validating the WHOLE input."""
        if not isinstance(contract, str) or not contract.strip():
            raise ValidationError("contract must be a non-empty identifier")
        quantity_delta = _finite(quantity_delta, "quantity_delta")
        price = _positive(price, "price")
        multiplier = _positive(multiplier, "multiplier")
        existing = self._legs.get(contract)
        if (
            existing is not None
            and abs(quantity_delta) >= FLAT_QUANTITY
            and existing.multiplier != multiplier
        ):
            raise ValidationError(
                f"multiplier conflict on open leg {contract!r}: "
                f"{existing.multiplier} vs {multiplier}"
            )
        leg, realized = apply_trade(
            existing if existing is not None else LegState(),
            self.realized_pnl,
            quantity_delta=quantity_delta,
            price=price,
            contract=contract,
            multiplier=multiplier,
        )
        self.realized_pnl = realized
        if leg.is_flat:
            self._legs.pop(contract, None)
            self._last_multiplier = leg.multiplier
        else:
            self._legs[contract] = leg
            self._last_multiplier = leg.multiplier

    def mark_to_market(self, prices) -> float:
        """``realized + sum_open quantity * (mark - avg) * multiplier``.

        Realized P&L enters once, not once per leg.  With a single open leg
        the arithmetic is the legacy expression in its original order.
        """
        self._require_prices(prices)
        open_contracts = self.contracts()
        if not open_contracts:
            return float(self.realized_pnl)
        if len(open_contracts) == 1:
            leg = self._legs[open_contracts[0]]
            return float(
                self.realized_pnl
                + leg.quantity
                * (float(prices[open_contracts[0]]) - leg.avg_price)
                * leg.multiplier
            )
        unrealized = 0.0
        for contract in open_contracts:
            unrealized += self._legs[contract].unrealized(prices[contract])
        return float(self.realized_pnl + unrealized)

    # -- compatibility --------------------------------------------------

    @property
    def single_leg(self) -> FuturesHedgePosition:
        """A legacy position SNAPSHOT when at most one leg is open.

        Mutating the returned object does not trade: the book owns the state,
        and an accidental write here would otherwise be a silent, unrecorded
        position change.
        """
        open_contracts = self.contracts()
        if len(open_contracts) > 1:
            raise ValidationError(
                "single_leg is unavailable for a multi-leg book: "
                f"{open_contracts}. Use the book API."
            )
        if not open_contracts:
            return FuturesHedgePosition(
                contract=None,
                quantity=0.0,
                avg_price=0.0,
                multiplier=self._last_multiplier,
                realized_pnl=self.realized_pnl,
            )
        leg = self._legs[open_contracts[0]]
        return FuturesHedgePosition(
            contract=leg.contract,
            quantity=leg.quantity,
            avg_price=leg.avg_price,
            multiplier=leg.multiplier,
            realized_pnl=self.realized_pnl,
        )

    def copy(self) -> "FuturesHedgeBook":
        """An independent book: trading either one leaves the other alone."""
        clone = FuturesHedgeBook()
        clone._legs = dict(self._legs)
        clone.realized_pnl = self.realized_pnl
        clone._last_multiplier = self._last_multiplier
        return clone

    # -- helpers --------------------------------------------------------

    def _require_prices(self, prices) -> None:
        missing = [c for c in self.contracts() if c not in (prices or {})]
        if missing:
            raise ValidationError(
                f"missing futures mark for open leg(s): {missing}"
            )
        for contract in self.contracts():
            _positive(prices[contract], f"mark[{contract}]")

    def __repr__(self) -> str:
        return (
            f"FuturesHedgeBook(legs={self.holdings()}, "
            f"realized={self.realized_pnl:.2f})"
        )


def _finite(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return value


def _positive(value, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return value
