"""
Equity hedge execution logic for backtesting.

Handles creation and management of hedge positions using spot or futures.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from quantark.portfolio import Portfolio
from quantark.asset.equity.product.deltaone import SpotInstrument, Futures
from quantark.asset.equity.engine.analytical import DeltaOneEngine
from quantark.priceenv import PricingEnvironment
from .state import TradeRecord
from quantark.backtest.transaction_costs import TransactionCostModel
from quantark.util.enum.deltaone_enums import DeltaOneType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


class HedgeExecutor:
    """
    Executes hedge trades and manages hedge positions.

    Responsible for:
    - Creating hedge instruments (spot or futures)
    - Executing trades through portfolio
    - Calculating transaction costs
    - Recording trade details

    Average-cost accounting (patch spec 2026-09-03 §3): increases blend the
    entry price, reductions / flips / closes realise P&L into
    ``realized_pnl``, and a hedge that nets to exactly zero is closed and
    removed from the portfolio (a zero quantity is not a position).

    Attributes:
        portfolio: Portfolio to manage
        transaction_cost_model: Cost model for trades
        hedge_instrument_type: 'spot' or 'futures'
        futures_maturity: Maturity for futures contracts (if used)
        futures_multiplier: Multiplier for futures contracts
        realized_pnl: Cumulative realised P&L of reduced, flipped and closed
            hedge lots (added to the backtest's net P&L)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        transaction_cost_model: TransactionCostModel,
        hedge_instrument_type: str = "spot",
        futures_maturity: Optional[float] = None,
        futures_multiplier: float = 1.0,
    ):
        """
        Initialize hedge executor.

        Args:
            portfolio: Portfolio instance
            transaction_cost_model: Transaction cost model
            hedge_instrument_type: 'spot' or 'futures'
            futures_maturity: Maturity for futures (required if type='futures')
            futures_multiplier: Contract multiplier for futures

        Raises:
            ValidationError: If parameters are invalid
        """
        if hedge_instrument_type not in ["spot", "futures"]:
            raise ValidationError(
                f"Hedge instrument type must be 'spot' or 'futures', got '{hedge_instrument_type}'"
            )

        if hedge_instrument_type == "futures" and futures_maturity is None:
            raise ValidationError(
                "futures_maturity is required when hedge_instrument_type='futures'"
            )

        self.portfolio = portfolio
        self.transaction_cost_model = transaction_cost_model
        self.hedge_instrument_type = hedge_instrument_type
        self.futures_maturity = futures_maturity
        self.futures_multiplier = futures_multiplier

        # Track hedge positions
        self._hedge_position_ids: Dict[str, str] = {}  # underlying -> position_id
        self._engine = DeltaOneEngine()
        # Average-cost accounting (patch spec 2026-09-03 §3): realised P&L of reduced,
        # flipped and closed hedge lots. BacktestEngine adds it to net P&L.
        self.realized_pnl: float = 0.0

    def execute_hedge(
        self,
        underlying: str,
        hedge_size: float,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str = "hedge",
    ) -> TradeRecord:
        """
        Execute a hedge trade.

        If a hedge position already exists for the underlying, updates it.
        Otherwise, creates a new hedge position.

        Args:
            underlying: Underlying asset identifier
            hedge_size: Size to hedge (positive=buy, negative=sell)
            pricing_env: Current pricing environment
            current_time: Execution timestamp
            reason: Reason for the hedge

        Returns:
            TradeRecord with execution details

        Raises:
            ValidationError: If parameters are invalid
        """
        if abs(hedge_size) < 1e-10:
            # No hedge needed
            return self._create_zero_trade_record(underlying, current_time, reason)

        # Get hedge price
        hedge_price = pricing_env.spot
        notional = abs(hedge_size * hedge_price)

        # Calculate transaction cost
        transaction_cost = self.transaction_cost_model.calculate_cost(
            quantity=hedge_size,
            price=hedge_price,
            notional=notional,
            instrument_type=self.hedge_instrument_type,
            trade_type="hedge",
        )

        # Check if hedge position already exists
        existing_position_id = self._hedge_position_ids.get(underlying)

        if existing_position_id and existing_position_id in self.portfolio.positions:
            # Update existing hedge position
            trade_record = self._update_hedge_position(
                underlying=underlying,
                position_id=existing_position_id,
                hedge_size=hedge_size,
                hedge_price=hedge_price,
                notional=notional,
                transaction_cost=transaction_cost,
                current_time=current_time,
                reason=reason,
            )
        else:
            # Create new hedge position
            trade_record = self._create_hedge_position(
                underlying=underlying,
                hedge_size=hedge_size,
                hedge_price=hedge_price,
                notional=notional,
                transaction_cost=transaction_cost,
                pricing_env=pricing_env,
                current_time=current_time,
                reason=reason,
            )

        return trade_record

    def _create_hedge_position(
        self,
        underlying: str,
        hedge_size: float,
        hedge_price: float,
        notional: float,
        transaction_cost: float,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str,
    ) -> TradeRecord:
        """Create a new hedge position."""
        # Create hedge instrument
        if self.hedge_instrument_type == "spot":
            hedge_product = SpotInstrument(
                underlying=underlying, deltaone_type=DeltaOneType.STOCK
            )
        else:  # futures
            hedge_product = Futures(
                underlying=underlying,
                multiplier=self.futures_multiplier,
                maturity=self.futures_maturity,
            )

        # Add position to portfolio
        position = self.portfolio.add_position(
            product=hedge_product,
            quantity=hedge_size,
            entry_price=hedge_price,
            underlying=underlying,
            engine=self._engine,
            entry_timestamp=current_time,
        )

        # Track hedge position
        self._hedge_position_ids[underlying] = position.position_id

        # Create trade record
        trade_record = TradeRecord(
            timestamp=current_time,
            trade_type="open",
            instrument_type=self.hedge_instrument_type,
            underlying=underlying,
            quantity=hedge_size,
            price=hedge_price,
            notional=notional,
            transaction_cost=transaction_cost,
            reason=reason,
            position_id=position.position_id,
            metadata={"action": "create_hedge"},
        )

        return trade_record

    def _update_hedge_position(
        self,
        underlying: str,
        position_id: str,
        hedge_size: float,
        hedge_price: float,
        notional: float,
        transaction_cost: float,
        current_time: datetime,
        reason: str,
    ) -> TradeRecord:
        """Update an existing hedge position at average cost.

        Mirrors ``MultiInstrumentHedgeExecutor._adjust_contract``: an increase
        blends the entry price, a reduction realises P&L on the closed part, a
        sign flip realises the old lot and re-enters at today's price, and a
        quantity that nets to zero closes and removes the position.
        """
        position = self.portfolio.positions[position_id]
        old_quantity = position.quantity
        entry = position.entry_price
        new_quantity = old_quantity + hedge_size
        entry_after: Optional[float]

        if is_zero(new_quantity):
            # net to zero: realise the whole lot and drop the position (a zero quantity
            # is not a position; this used to raise inside Portfolio.update_position)
            self.realized_pnl += (hedge_price - entry) * old_quantity
            self.portfolio.remove_position(position_id)
            del self._hedge_position_ids[underlying]
            trade_type, action, entry_after = "close", "close_hedge", None
        elif old_quantity * new_quantity < 0:
            # sign flip: realise the old lot, re-enter the remainder at today's price
            self.realized_pnl += (hedge_price - entry) * old_quantity
            self.portfolio.update_position(position_id, quantity=new_quantity, entry_price=hedge_price)
            trade_type, action, entry_after = "adjust", "flip_hedge", hedge_price
        elif abs(new_quantity) > abs(old_quantity):
            # increase: blend the entry price, realise nothing
            blended = (entry * old_quantity + hedge_price * hedge_size) / new_quantity
            self.portfolio.update_position(position_id, quantity=new_quantity, entry_price=blended)
            trade_type, action, entry_after = "adjust", "increase_hedge", blended
        else:
            # reduce: realise P&L on the closed part, keep the entry price
            self.realized_pnl += (hedge_price - entry) * (old_quantity - new_quantity)
            self.portfolio.update_position(position_id, quantity=new_quantity)
            trade_type, action, entry_after = "adjust", "reduce_hedge", entry

        return TradeRecord(
            timestamp=current_time,
            trade_type=trade_type,
            instrument_type=self.hedge_instrument_type,
            underlying=underlying,
            quantity=hedge_size,
            price=hedge_price,
            notional=notional,
            transaction_cost=transaction_cost,
            reason=reason,
            position_id=position_id,
            metadata={
                "action": action,
                "old_quantity": old_quantity,
                "new_quantity": new_quantity,
                "entry_price_after": entry_after,
            },
        )

    def _create_zero_trade_record(
        self, underlying: str, current_time: datetime, reason: str
    ) -> TradeRecord:
        """Create a trade record for a zero-size trade (no hedge)."""
        return TradeRecord(
            timestamp=current_time,
            trade_type="no_trade",
            instrument_type=self.hedge_instrument_type,
            underlying=underlying,
            quantity=0.0,
            price=0.0,
            notional=0.0,
            transaction_cost=0.0,
            reason=reason,
            metadata={"action": "no_hedge_needed"},
        )

    def get_hedge_position(self, underlying: str) -> Optional[Any]:
        """
        Get the hedge position for an underlying.

        Args:
            underlying: Underlying asset identifier

        Returns:
            Position object or None if no hedge exists
        """
        position_id = self._hedge_position_ids.get(underlying)
        if position_id:
            return self.portfolio.positions.get(position_id)
        return None

    def get_hedge_quantity(self, underlying: str) -> float:
        """
        Get the current hedge quantity for an underlying.

        Args:
            underlying: Underlying asset identifier

        Returns:
            Current hedge quantity (0 if no hedge)
        """
        position = self.get_hedge_position(underlying)
        if position:
            return position.quantity
        return 0.0

    def close_hedge_position(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str = "close_hedge",
    ) -> Optional[TradeRecord]:
        """
        Close a hedge position completely.

        Args:
            underlying: Underlying asset identifier
            pricing_env: Current pricing environment
            current_time: Execution timestamp
            reason: Reason for closing

        Returns:
            TradeRecord or None if no position exists
        """
        position_id = self._hedge_position_ids.get(underlying)
        if not position_id or position_id not in self.portfolio.positions:
            return None

        position = self.portfolio.positions[position_id]
        close_quantity = -position.quantity  # Opposite of current position
        close_price = pricing_env.spot
        notional = abs(close_quantity * close_price)

        # Calculate transaction cost
        transaction_cost = self.transaction_cost_model.calculate_cost(
            quantity=close_quantity,
            price=close_price,
            notional=notional,
            instrument_type=self.hedge_instrument_type,
            trade_type="close",
        )

        # Realise the lot, then remove the position
        self.realized_pnl += (close_price - position.entry_price) * position.quantity
        self.portfolio.remove_position(position_id)
        del self._hedge_position_ids[underlying]

        # Create trade record
        trade_record = TradeRecord(
            timestamp=current_time,
            trade_type="close",
            instrument_type=self.hedge_instrument_type,
            underlying=underlying,
            quantity=close_quantity,
            price=close_price,
            notional=notional,
            transaction_cost=transaction_cost,
            reason=reason,
            position_id=position_id,
            metadata={"action": "close_hedge"},
        )

        return trade_record

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get hedge executor statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "hedge_instrument_type": self.hedge_instrument_type,
            "num_hedge_positions": len(self._hedge_position_ids),
            "underlyings_hedged": list(self._hedge_position_ids.keys()),
            "realized_pnl": self.realized_pnl,
        }

    def __repr__(self) -> str:
        return (
            f"HedgeExecutor("
            f"type={self.hedge_instrument_type}, "
            f"hedges={len(self._hedge_position_ids)})"
        )


# Alias for explicit naming
EquityHedgeExecutor = HedgeExecutor
