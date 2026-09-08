"""The futures hedge over a batch of paths (spec 4.2, 8.1, 8.6).

``FuturesHedgePosition``'s average-cost rules, applied to arrays: every
path holds the same contract (the roll policy reads only expiry dates,
which are common) in its own size, at its own price.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError

#: The replay's own dead-zone for "no position" and "no trade".  Copied
#: verbatim from ``FuturesHedgePosition`` and ``ReplayBacktestEngine._rebalance``
#: rather than routed through ``util.numerical``: the goal is a ledger that
#: agrees with the replay bit for bit, so it must share the exact threshold.
POSITION_EPS = 1e-12


class VectorHedgeLedger:
    """Average-cost futures ledger, one lane per path."""

    def __init__(self, n_paths: int) -> None:
        if n_paths < 1:
            raise ValidationError("n_paths must be positive")
        self.n_paths = int(n_paths)
        self.quantity = np.zeros(self.n_paths)
        self.avg_price = np.zeros(self.n_paths)
        self.realized_pnl = np.zeros(self.n_paths)
        self.contract: Optional[str] = None
        self.multiplier = 1.0

    def trade(
        self, quantity_delta: np.ndarray, price: np.ndarray, contract: str, multiplier: float
    ) -> None:
        """Book a per-path trade in a common contract."""
        delta = np.asarray(quantity_delta, dtype=float)
        px = np.asarray(price, dtype=float)
        if delta.shape != (self.n_paths,) or px.shape != (self.n_paths,):
            raise ValidationError(f"trade arrays must have shape {(self.n_paths,)}")
        active = np.abs(delta) >= POSITION_EPS
        if not active.any():
            return
        flat = np.abs(self.quantity) < POSITION_EPS
        if self.contract is not None and contract != self.contract and (~flat & active).any():
            raise ValidationError("Cannot trade a different contract without rolling")

        opening = active & flat
        self.quantity[opening] = delta[opening]
        self.avg_price[opening] = px[opening]

        held = active & ~flat
        same = held & (self.quantity * delta > 0.0)
        new_qty = self.quantity[same] + delta[same]
        self.avg_price[same] = (
            self.avg_price[same] * np.abs(self.quantity[same]) + px[same] * np.abs(delta[same])
        ) / np.abs(new_qty)
        self.quantity[same] = new_qty

        against = held & ~same
        if against.any():
            close_qty = np.minimum(np.abs(self.quantity[against]), np.abs(delta[against]))
            direction = np.where(self.quantity[against] > 0.0, 1.0, -1.0)
            self.realized_pnl[against] += (
                direction * close_qty * (px[against] - self.avg_price[against]) * multiplier
            )
            after = self.quantity[against] + delta[against]
            flipped = self.quantity[against] * after < 0.0
            closed = np.abs(after) < POSITION_EPS
            avg = self.avg_price[against]
            avg = np.where(flipped, px[against], avg)
            avg = np.where(closed, 0.0, avg)
            self.avg_price[against] = avg
            self.quantity[against] = np.where(closed, 0.0, after)

        self.multiplier = float(multiplier)
        # The scalar ledger drops the contract when a lane goes flat; with
        # one contract shared by every lane, the batch keeps it until every
        # lane is flat, which is the same statement for the book.
        self.contract = None if np.all(np.abs(self.quantity) < POSITION_EPS) else contract

    def mark_to_market(self, price: np.ndarray) -> np.ndarray:
        """Realised P&L plus the open position's mark, per path."""
        px = np.asarray(price, dtype=float)
        open_ = np.abs(self.quantity) >= POSITION_EPS
        return self.realized_pnl + np.where(
            open_, self.quantity * (px - self.avg_price) * self.multiplier, 0.0
        )

    def close_all(self, price: np.ndarray, contract: str, multiplier: float) -> np.ndarray:
        """Flatten every lane; returns the sizes traded."""
        traded = -self.quantity.copy()
        self.trade(traded, price, contract, multiplier)
        return traded
