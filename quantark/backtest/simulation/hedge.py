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


def day_active_contract(chain, roll_policy, current_contract: Optional[str]):
    """The day's hedge contract and its column in ``chain``.

    ``FuturesRollPolicy.select_contract`` reads only ``expiry_date`` and the
    contract code, never a price, and every path shares the listing
    calendar -- so one selection serves the whole batch and only the
    execution price differs per path.  Path 0's frame is therefore
    representative, and the roll test pins that.
    """
    selected = roll_policy.select_contract(chain.frame(0), chain.date, current_contract)
    code = str(selected["contract"])
    return code, chain.contracts.index(code)


def target_contracts_vector(strategy, net_delta: np.ndarray, multiplier: float) -> np.ndarray:
    """``AutocallableDeltaHedgeStrategy.target_contracts`` over an array."""
    if multiplier <= 0:
        raise ValidationError("futures_multiplier must be positive")
    target = -((np.asarray(net_delta, dtype=float) - float(strategy.target_delta)) / float(multiplier))
    target = target * float(strategy.hedge_ratio)
    if getattr(strategy, "round_contracts", True):
        # np.round is half-to-even, like Python's round(), which the scalar
        # strategy uses; a half-up rule would differ on exact halves.
        return np.round(target)
    return target


def should_rebalance_vector(strategy, current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``strategy.should_rebalance`` over arrays."""
    return np.abs(np.asarray(target, dtype=float) - np.asarray(current, dtype=float)) > float(
        strategy.delta_threshold
    )
