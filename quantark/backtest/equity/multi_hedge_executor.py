"""
Multi-instrument hedge execution for multi-Greek strategies.

Manages one live contract per hedge instrument spec, supplies the per-unit
Greeks the optimizer sizes against, executes the resulting trades, rolls
expiring option contracts, and tracks realized P&L with average-cost
accounting (Portfolio.get_portfolio_pnl only sums live positions, so the
P&L of closed/rolled hedge contracts would otherwise be lost).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.backtest.strategy.hedge_instruments import BaseHedgeInstrument
from quantark.backtest.transaction_costs import TransactionCostModel
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

from .state import TradeRecord


@dataclass
class _ActiveContract:
    """A live hedge position held against one instrument spec."""

    position_id: str
    product: BaseEquityProduct
    engine: BaseEngine


class MultiInstrumentHedgeExecutor:
    """
    Executes multi-instrument hedges and manages their lifecycle.

    Per-step call order (enforced by the backtest engine):
    1. process_rolls()          close expiring/drifted contracts first, so
                                portfolio Greeks and the hedge solve never
                                reference a contract about to vanish
    2. get_instrument_greeks()  per-unit Greeks of the contract that would
                                actually be traded (live contract if held,
                                else a fresh candidate cached for step 3)
    3. execute_hedges()         trade the solved quantities

    Attributes:
        portfolio: Portfolio receiving hedge positions
        transaction_cost_model: Cost model for trades
        instruments: Hedge instrument specs keyed by name
        greeks_calculator: Calculator shared with the backtest engine
        realized_pnl: Cumulative realized P&L from closed/reduced hedge
            contracts (average-cost accounting)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        transaction_cost_model: TransactionCostModel,
        instruments: Sequence[BaseHedgeInstrument],
        greeks_calculator: Optional[GreeksCalculator] = None,
    ):
        """
        Initialize multi-instrument hedge executor.

        Args:
            portfolio: Portfolio instance
            transaction_cost_model: Transaction cost model
            instruments: Hedge instrument specifications (unique names)
            greeks_calculator: Greeks calculator (defaults to a new one)

        Raises:
            ValidationError: If no instruments given or their names collide
        """
        if not instruments:
            raise ValidationError("At least one hedge instrument is required")
        names = [inst.name for inst in instruments]
        if len(set(names)) != len(names):
            raise ValidationError(f"Duplicate hedge instrument names: {names}")

        self.portfolio = portfolio
        self.transaction_cost_model = transaction_cost_model
        self.instruments: Dict[str, BaseHedgeInstrument] = {
            inst.name: inst for inst in instruments
        }
        self.greeks_calculator = (
            greeks_calculator if greeks_calculator is not None else GreeksCalculator()
        )
        self.realized_pnl: float = 0.0

        self._active: Dict[str, _ActiveContract] = {}
        self._candidates: Dict[str, Tuple[BaseEquityProduct, BaseEngine]] = {}

    def process_rolls(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
    ) -> List[TradeRecord]:
        """
        Close held contracts whose specs demand a roll.

        Must be called before portfolio Greeks are computed each step.

        Args:
            underlying: Underlying asset identifier
            pricing_env: Current pricing environment
            current_time: Current timestamp

        Returns:
            Trade records for the closing trades (possibly empty)
        """
        records = []
        for name, contract in list(self._active.items()):
            if contract.position_id not in self.portfolio.positions:
                # Position removed externally: drop the stale handle
                del self._active[name]
                continue
            instrument = self.instruments[name]
            if instrument.requires_roll(contract.product, pricing_env):
                records.append(
                    self._close_contract(
                        name, underlying, pricing_env, current_time, reason="roll"
                    )
                )
        return records

    def get_instrument_greeks(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
    ) -> Dict[str, Dict[str, float]]:
        """
        Per-unit Greeks of each instrument's tradeable contract.

        For instruments without a live contract, a fresh candidate is
        created and cached so that the subsequent execute_hedges() trades
        exactly the contract these Greeks describe.

        Args:
            underlying: Underlying asset identifier
            pricing_env: Current pricing environment
            current_time: Current timestamp

        Returns:
            Dictionary mapping instrument name to its per-unit Greeks
        """
        greeks: Dict[str, Dict[str, float]] = {}
        self._candidates = {}
        for name, instrument in self.instruments.items():
            contract = self._active.get(name)
            if contract is not None:
                product, engine = contract.product, contract.engine
            else:
                product = instrument.create_product(
                    underlying, pricing_env, current_time
                )
                engine = instrument.create_engine()
                self._candidates[name] = (product, engine)
            greeks[name] = instrument.unit_greeks(
                product, engine, pricing_env, self.greeks_calculator
            )
        return greeks

    def execute_hedges(
        self,
        underlying: str,
        quantities: Dict[str, float],
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str = "multi_greek_hedge",
    ) -> List[TradeRecord]:
        """
        Execute the solved hedge quantities.

        Args:
            underlying: Underlying asset identifier
            quantities: Quantity to trade per instrument name
            pricing_env: Current pricing environment
            current_time: Execution timestamp
            reason: Reason recorded on the trades

        Returns:
            Trade records for executed trades (zero quantities are skipped)

        Raises:
            ValidationError: If a quantity references an unknown instrument
        """
        records = []
        for name, quantity in quantities.items():
            if name not in self.instruments:
                raise ValidationError(
                    f"Unknown hedge instrument '{name}'; expected one of "
                    f"{sorted(self.instruments)}"
                )
            if is_zero(quantity):
                continue
            if name in self._active:
                records.append(
                    self._adjust_contract(
                        name, underlying, quantity, pricing_env, current_time, reason
                    )
                )
            else:
                records.append(
                    self._open_contract(
                        name, underlying, quantity, pricing_env, current_time, reason
                    )
                )
        return records

    def get_position_id(self, name: str) -> Optional[str]:
        """Position id of the live contract for an instrument, if any."""
        contract = self._active.get(name)
        return contract.position_id if contract else None

    def get_statistics(self) -> Dict[str, object]:
        """Executor statistics."""
        return {
            "instruments": sorted(self.instruments),
            "active_contracts": sorted(self._active),
            "realized_pnl": self.realized_pnl,
        }

    def _open_contract(
        self,
        name: str,
        underlying: str,
        quantity: float,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str,
    ) -> TradeRecord:
        """Open a new contract, preferring the cached candidate."""
        instrument = self.instruments[name]
        candidate = self._candidates.pop(name, None)
        if candidate is not None:
            product, engine = candidate
        else:
            product = instrument.create_product(underlying, pricing_env, current_time)
            engine = instrument.create_engine()

        price = instrument.unit_price(product, engine, pricing_env)
        position = self.portfolio.add_position(
            product=product,
            quantity=quantity,
            entry_price=price,
            underlying=underlying,
            engine=engine,
            entry_timestamp=current_time,
        )
        self._active[name] = _ActiveContract(position.position_id, product, engine)

        return self._make_record(
            "open", instrument, underlying, quantity, price, current_time,
            reason, position.position_id, trade_type_for_cost="hedge",
        )

    def _adjust_contract(
        self,
        name: str,
        underlying: str,
        quantity: float,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str,
    ) -> TradeRecord:
        """
        Adjust a live contract with average-cost accounting.

        - increase (same sign, larger):  blend entry price, realize nothing
        - reduce  (same sign, smaller):  realize P&L on the closed lot
        - sign flip:                     realize the full old lot, re-enter
        - net to zero:                   close and remove the position
        """
        instrument = self.instruments[name]
        contract = self._active[name]
        position = self.portfolio.positions[contract.position_id]
        price = instrument.unit_price(contract.product, contract.engine, pricing_env)

        old_quantity = position.quantity
        entry = position.entry_price
        new_quantity = old_quantity + quantity

        if is_zero(new_quantity):
            self.realized_pnl += (price - entry) * old_quantity
            self.portfolio.remove_position(contract.position_id)
            del self._active[name]
            trade_type = "close"
        elif old_quantity * new_quantity < 0:
            self.realized_pnl += (price - entry) * old_quantity
            self.portfolio.update_position(
                contract.position_id, quantity=new_quantity, entry_price=price
            )
            trade_type = "adjust"
        elif abs(new_quantity) > abs(old_quantity):
            blended = (entry * old_quantity + price * quantity) / new_quantity
            self.portfolio.update_position(
                contract.position_id, quantity=new_quantity, entry_price=blended
            )
            trade_type = "adjust"
        else:
            self.realized_pnl += (price - entry) * (old_quantity - new_quantity)
            self.portfolio.update_position(
                contract.position_id, quantity=new_quantity
            )
            trade_type = "adjust"

        return self._make_record(
            trade_type, instrument, underlying, quantity, price, current_time,
            reason, contract.position_id, trade_type_for_cost="hedge",
        )

    def _close_contract(
        self,
        name: str,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str,
    ) -> TradeRecord:
        """Close a live contract entirely, realizing its P&L."""
        instrument = self.instruments[name]
        contract = self._active[name]
        position = self.portfolio.positions[contract.position_id]
        price = instrument.unit_price(contract.product, contract.engine, pricing_env)

        quantity = position.quantity
        self.realized_pnl += (price - position.entry_price) * quantity
        self.portfolio.remove_position(contract.position_id)
        del self._active[name]

        return self._make_record(
            "close", instrument, underlying, -quantity, price, current_time,
            reason, contract.position_id, trade_type_for_cost="close",
        )

    def _make_record(
        self,
        trade_type: str,
        instrument: BaseHedgeInstrument,
        underlying: str,
        quantity: float,
        price: float,
        current_time: datetime,
        reason: str,
        position_id: str,
        trade_type_for_cost: str,
    ) -> TradeRecord:
        """Build a trade record with transaction costs."""
        notional = abs(quantity * price)
        transaction_cost = self.transaction_cost_model.calculate_cost(
            quantity=quantity,
            price=price,
            notional=notional,
            instrument_type=instrument.instrument_type,
            trade_type=trade_type_for_cost,
        )
        return TradeRecord(
            timestamp=current_time,
            trade_type=trade_type,
            instrument_type=instrument.instrument_type,
            underlying=underlying,
            quantity=quantity,
            price=price,
            notional=notional,
            transaction_cost=transaction_cost,
            reason=reason,
            position_id=position_id,
            metadata={"hedge_instrument": instrument.name},
        )

    def __repr__(self) -> str:
        return (
            f"MultiInstrumentHedgeExecutor("
            f"instruments={sorted(self.instruments)}, "
            f"active={sorted(self._active)})"
        )
