"""
Fixed Income backtest state management.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from copy import deepcopy


@dataclass
class FITradeRecord:
    """
    Record of a single FI trade execution.
    
    Attributes:
        timestamp: When the trade was executed
        trade_type: Type of trade ('hedge', 'open', 'close', 'adjust')
        instrument_type: Type of instrument ('bond_futures')
        underlying: Underlying identifier
        quantity: Quantity traded (contracts)
        price: Execution price
        notional: Total notional value
        transaction_cost: Transaction cost incurred
        dv01_impact: DV01 change from this trade
        reason: Reason for trade
        position_id: Associated position ID
        metadata: Additional metadata
    """
    timestamp: datetime
    trade_type: str
    instrument_type: str
    underlying: str
    quantity: float
    price: float
    notional: float
    transaction_cost: float
    dv01_impact: float
    reason: str
    position_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'trade_type': self.trade_type,
            'instrument_type': self.instrument_type,
            'underlying': self.underlying,
            'quantity': self.quantity,
            'price': self.price,
            'notional': self.notional,
            'transaction_cost': self.transaction_cost,
            'dv01_impact': self.dv01_impact,
            'reason': self.reason,
            'position_id': self.position_id,
            **self.metadata
        }


@dataclass
class FIBacktestState:
    """
    State snapshot at a single point in time during FI backtest.
    
    Tracks all relevant information about the portfolio, market,
    and performance at a specific timestamp.
    
    Attributes:
        timestamp: Current timestamp
        portfolio_value: Total portfolio market value
        pnl: Cumulative P&L
        bond_pnl: P&L from bond positions
        hedge_pnl: P&L from hedge positions
        transaction_costs: Cumulative transaction costs
        num_positions: Number of positions
        num_hedges: Number of active hedge positions
        risk_measures: Portfolio risk measures (dv01, convexity, duration)
        market_data: Current market data (rates, spreads)
        trades: List of trades at this timestamp
        metadata: Additional state information
    """
    timestamp: datetime
    portfolio_value: float
    pnl: float = 0.0
    bond_pnl: float = 0.0
    hedge_pnl: float = 0.0
    transaction_costs: float = 0.0
    num_positions: int = 0
    num_hedges: int = 0
    risk_measures: Dict[str, float] = field(default_factory=dict)
    market_data: Dict[str, float] = field(default_factory=dict)
    trades: List[FITradeRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        base_dict = {
            'timestamp': self.timestamp,
            'portfolio_value': self.portfolio_value,
            'pnl': self.pnl,
            'bond_pnl': self.bond_pnl,
            'hedge_pnl': self.hedge_pnl,
            'transaction_costs': self.transaction_costs,
            'num_positions': self.num_positions,
            'num_hedges': self.num_hedges,
        }
        
        # Add risk measures with prefix
        for key, value in self.risk_measures.items():
            base_dict[f'risk_{key}'] = value
        
        # Add market data with prefix
        for key, value in self.market_data.items():
            base_dict[f'market_{key}'] = value
        
        base_dict.update(self.metadata)
        
        return base_dict
    
    def copy(self) -> 'FIBacktestState':
        """Create a deep copy of this state."""
        return deepcopy(self)


class FIStateTracker:
    """
    Tracks FI backtest state over time.
    """
    
    def __init__(self):
        """Initialize empty state tracker."""
        self.states: List[FIBacktestState] = []
        self.trades: List[FITradeRecord] = []
        self._current_state: Optional[FIBacktestState] = None
    
    def add_state(self, state: FIBacktestState):
        """Add a new state snapshot."""
        self.states.append(state)
        self._current_state = state
        
        for trade in state.trades:
            self.trades.append(trade)
    
    def get_current_state(self) -> Optional[FIBacktestState]:
        """Get the most recent state."""
        return self._current_state
    
    def get_state_at_time(self, timestamp: datetime) -> Optional[FIBacktestState]:
        """Get state at or before a specific timestamp."""
        for state in reversed(self.states):
            if state.timestamp <= timestamp:
                return state
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert state history to DataFrame."""
        if not self.states:
            return pd.DataFrame()
        
        data = [state.to_dict() for state in self.states]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Convert trade history to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.states:
            return {}
        
        df = self.to_dataframe()
        
        return {
            'num_snapshots': len(self.states),
            'num_trades': len(self.trades),
            'start_date': self.states[0].timestamp,
            'end_date': self.states[-1].timestamp,
            'initial_value': self.states[0].portfolio_value,
            'final_value': self.states[-1].portfolio_value,
            'total_pnl': self.states[-1].pnl,
            'total_transaction_costs': self.states[-1].transaction_costs,
            'final_dv01': self.states[-1].risk_measures.get('dv01', 0),
        }
    
    def clear(self):
        """Clear all tracked states and trades."""
        self.states.clear()
        self.trades.clear()
        self._current_state = None
    
    def __len__(self) -> int:
        return len(self.states)
    
    def __repr__(self) -> str:
        return f"FIStateTracker(snapshots={len(self.states)}, trades={len(self.trades)})"

