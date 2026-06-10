"""
Fixed Income hedge execution logic for backtesting.

Handles creation and management of hedge positions using bond futures.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
from portfolio.fi import FIPortfolio
from asset.bond.product.futures.bond_futures import BondFutures
from priceenv import PricingEnvironment
from .state import FITradeRecord
from backtest.transaction_costs import TransactionCostModel
from util.exceptions import ValidationError


class FIHedgeExecutor:
    """
    Executes hedge trades using bond futures for FI portfolios.
    
    Responsible for:
    - Creating and managing bond futures hedge positions
    - Executing trades through portfolio
    - Calculating transaction costs
    - Recording trade details with DV01 impact
    
    Attributes:
        portfolio: FI portfolio to manage
        transaction_cost_model: Cost model for trades
        futures_spec: Bond futures specification for hedging
        futures_dv01: DV01 per futures contract
        contract_size: Notional per contract
    """
    
    def __init__(
        self,
        portfolio: FIPortfolio,
        transaction_cost_model: TransactionCostModel,
        futures_spec: Optional[BondFutures] = None,
        futures_dv01: float = 1000.0,
        contract_size: float = 100000.0
    ):
        """
        Initialize FI hedge executor.
        
        Args:
            portfolio: FI portfolio instance
            transaction_cost_model: Transaction cost model
            futures_spec: Bond futures specification (optional)
            futures_dv01: DV01 per futures contract
            contract_size: Notional per contract
        """
        self.portfolio = portfolio
        self.transaction_cost_model = transaction_cost_model
        self.futures_spec = futures_spec
        self.futures_dv01 = futures_dv01
        self.contract_size = contract_size
        
        # Track hedge positions
        self._hedge_position_quantity: float = 0.0
        self._hedge_position_id: Optional[str] = None
        self._cumulative_hedge_pnl: float = 0.0
        self._last_hedge_price: float = 0.0
    
    def execute_hedge(
        self,
        underlying: str,
        hedge_size: float,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str = "hedge"
    ) -> FITradeRecord:
        """
        Execute a hedge trade using bond futures.
        
        Args:
            underlying: Underlying identifier
            hedge_size: Number of futures contracts (positive=buy, negative=sell)
            pricing_env: Current pricing environment
            current_time: Execution timestamp
            reason: Reason for the hedge
            
        Returns:
            FITradeRecord with execution details
        """
        if abs(hedge_size) < 1e-10:
            return self._create_zero_trade_record(
                underlying, current_time, reason
            )
        
        # Round to whole contracts
        hedge_size = round(hedge_size)
        
        # Get futures price (simplified: use rate as proxy)
        futures_price = 100.0 - pricing_env.get_rate(1.0) * 100
        notional = abs(hedge_size * self.contract_size)
        
        # Calculate transaction cost
        transaction_cost = self.transaction_cost_model.calculate_cost(
            quantity=hedge_size,
            price=futures_price,
            notional=notional,
            instrument_type='bond_futures',
            trade_type='hedge'
        )
        
        # Calculate DV01 impact
        dv01_impact = hedge_size * self.futures_dv01
        
        # Update hedge position tracking
        old_quantity = self._hedge_position_quantity
        new_quantity = old_quantity + hedge_size
        
        # Track P&L on existing position
        if abs(old_quantity) > 0:
            price_change = futures_price - self._last_hedge_price
            self._cumulative_hedge_pnl += old_quantity * price_change * self.contract_size / 100
        
        self._hedge_position_quantity = new_quantity
        self._last_hedge_price = futures_price
        
        # Determine trade type
        if abs(old_quantity) < 1e-10:
            trade_type = 'open'
        elif abs(new_quantity) < 1e-10:
            trade_type = 'close'
        else:
            trade_type = 'adjust'
        
        # Create trade record
        trade_record = FITradeRecord(
            timestamp=current_time,
            trade_type=trade_type,
            instrument_type='bond_futures',
            underlying=underlying,
            quantity=hedge_size,
            price=futures_price,
            notional=notional,
            transaction_cost=transaction_cost,
            dv01_impact=dv01_impact,
            reason=reason,
            position_id=self._hedge_position_id or str(uuid.uuid4()),
            metadata={
                'old_quantity': old_quantity,
                'new_quantity': new_quantity,
                'futures_dv01': self.futures_dv01
            }
        )
        
        if self._hedge_position_id is None:
            self._hedge_position_id = trade_record.position_id
        
        return trade_record
    
    def _create_zero_trade_record(
        self,
        underlying: str,
        current_time: datetime,
        reason: str
    ) -> FITradeRecord:
        """Create a trade record for a zero-size trade."""
        return FITradeRecord(
            timestamp=current_time,
            trade_type='no_trade',
            instrument_type='bond_futures',
            underlying=underlying,
            quantity=0.0,
            price=0.0,
            notional=0.0,
            transaction_cost=0.0,
            dv01_impact=0.0,
            reason=reason,
            metadata={'action': 'no_hedge_needed'}
        )
    
    def get_hedge_position(self, underlying: str) -> Optional[Dict[str, Any]]:
        """
        Get current hedge position details.
        
        Args:
            underlying: Underlying identifier
            
        Returns:
            Dictionary with position details or None
        """
        if abs(self._hedge_position_quantity) < 1e-10:
            return None
        
        return {
            'quantity': self._hedge_position_quantity,
            'dv01': self._hedge_position_quantity * self.futures_dv01,
            'position_id': self._hedge_position_id,
            'last_price': self._last_hedge_price
        }
    
    def get_hedge_quantity(self, underlying: str) -> float:
        """Get current hedge quantity (number of contracts)."""
        return self._hedge_position_quantity
    
    def get_hedge_dv01(self) -> float:
        """Get current hedge DV01."""
        return self._hedge_position_quantity * self.futures_dv01
    
    def close_hedge_position(
        self,
        underlying: str,
        pricing_env: PricingEnvironment,
        current_time: datetime,
        reason: str = "close_hedge"
    ) -> Optional[FITradeRecord]:
        """
        Close the hedge position completely.
        
        Args:
            underlying: Underlying identifier
            pricing_env: Current pricing environment
            current_time: Execution timestamp
            reason: Reason for closing
            
        Returns:
            FITradeRecord or None if no position exists
        """
        if abs(self._hedge_position_quantity) < 1e-10:
            return None
        
        close_quantity = -self._hedge_position_quantity
        trade_record = self.execute_hedge(
            underlying=underlying,
            hedge_size=close_quantity,
            pricing_env=pricing_env,
            current_time=current_time,
            reason=reason
        )
        
        return trade_record
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hedge executor statistics."""
        return {
            'hedge_instrument_type': 'bond_futures',
            'current_position': self._hedge_position_quantity,
            'current_dv01': self.get_hedge_dv01(),
            'futures_dv01_per_contract': self.futures_dv01,
            'contract_size': self.contract_size,
            'cumulative_hedge_pnl': self._cumulative_hedge_pnl
        }
    
    def __repr__(self) -> str:
        return (
            f"FIHedgeExecutor("
            f"position={self._hedge_position_quantity} contracts, "
            f"dv01=${self.get_hedge_dv01():,.0f})"
        )

