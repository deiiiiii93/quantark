"""
Backtest state management for tracking portfolio state over time.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from copy import deepcopy


@dataclass
class TradeRecord:
    """
    Record of a single trade execution.
    
    Attributes:
        timestamp: When the trade was executed
        trade_type: Type of trade ('hedge', 'open', 'close', 'adjust')
        instrument_type: Type of instrument ('spot', 'futures', 'option')
        underlying: Underlying asset
        quantity: Quantity traded (positive=buy, negative=sell)
        price: Execution price per unit
        notional: Total notional value of trade
        transaction_cost: Total transaction cost incurred
        reason: Reason for the trade (e.g., 'delta_rebalance')
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
            'reason': self.reason,
            'position_id': self.position_id,
            **self.metadata
        }


@dataclass
class BacktestState:
    """
    State snapshot at a single point in time during backtest.
    
    Tracks all relevant information about the portfolio, market,
    and performance at a specific timestamp.
    
    Attributes:
        timestamp: Current timestamp
        portfolio_value: Total portfolio market value
        cash: Cash balance
        pnl: Cumulative P&L
        options_pnl: P&L from options positions
        hedge_pnl: P&L from hedge positions
        transaction_costs: Cumulative transaction costs
        num_positions: Number of positions in portfolio
        num_hedges: Number of active hedge positions
        greeks: Portfolio Greeks (delta, gamma, vega, theta, rho)
        market_data: Current market data (spot, vol, rate, div_yield)
        trades: List of trades executed at this timestamp
        pending_receivable_pv: PV of determined lifecycle cashflows not yet paid
        paid_cash: Cumulative lifecycle cash that has reached its payment date
        realized_cash: Backward-compatible alias of ``paid_cash``
        lifecycle_events: Lifecycle events that fired at this timestamp
            (``ProcessedLifecycleEvent`` records; empty when none)
        metadata: Additional state information
    """
    timestamp: datetime
    portfolio_value: float
    cash: float = 0.0
    pnl: float = 0.0
    options_pnl: float = 0.0
    hedge_pnl: float = 0.0
    transaction_costs: float = 0.0
    num_positions: int = 0
    num_hedges: int = 0
    greeks: Dict[str, float] = field(default_factory=dict)
    market_data: Dict[str, float] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)
    pending_receivable_pv: float = 0.0
    paid_cash: float = 0.0
    realized_cash: float = 0.0
    lifecycle_events: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            Dictionary representation of state
        """
        base_dict = {
            'timestamp': self.timestamp,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'pnl': self.pnl,
            'options_pnl': self.options_pnl,
            'hedge_pnl': self.hedge_pnl,
            'transaction_costs': self.transaction_costs,
            'num_positions': self.num_positions,
            'num_hedges': self.num_hedges,
            'pending_receivable_pv': self.pending_receivable_pv,
            'paid_cash': self.paid_cash,
            'realized_cash': self.realized_cash,
        }
        
        # Add Greeks with prefix
        for key, value in self.greeks.items():
            base_dict[f'greek_{key}'] = value
        
        # Add market data with prefix
        for key, value in self.market_data.items():
            base_dict[f'market_{key}'] = value
        
        # Add metadata
        base_dict.update(self.metadata)
        
        return base_dict
    
    def copy(self) -> 'BacktestState':
        """Create a deep copy of this state."""
        return deepcopy(self)


class StateTracker:
    """
    Tracks backtest state over time.
    
    Maintains a time series of BacktestState snapshots and provides
    methods for querying and analyzing state history.
    """
    
    def __init__(self):
        """Initialize empty state tracker."""
        self.states: List[BacktestState] = []
        self.trades: List[TradeRecord] = []
        self.lifecycle_events: List[Any] = []
        self._current_state: Optional[BacktestState] = None

    def add_state(self, state: BacktestState):
        """
        Add a new state snapshot.

        Args:
            state: BacktestState to add
        """
        self.states.append(state)
        self._current_state = state

        # Also track trades
        for trade in state.trades:
            self.trades.append(trade)

        # Track lifecycle events flat for cross-day reporting
        for event in state.lifecycle_events:
            self.lifecycle_events.append(event)
    
    def get_current_state(self) -> Optional[BacktestState]:
        """Get the most recent state."""
        return self._current_state
    
    def get_state_at_time(self, timestamp: datetime) -> Optional[BacktestState]:
        """
        Get state at or before a specific timestamp.
        
        Args:
            timestamp: Target timestamp
            
        Returns:
            BacktestState at or before timestamp, or None if not found
        """
        for state in reversed(self.states):
            if state.timestamp <= timestamp:
                return state
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert state history to DataFrame.
        
        Returns:
            DataFrame with all state snapshots
        """
        if not self.states:
            return pd.DataFrame()
        
        data = [state.to_dict() for state in self.states]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Convert trade history to DataFrame.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(data)

        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        return df

    def get_lifecycle_events_dataframe(self) -> pd.DataFrame:
        """
        Convert realized lifecycle events to a DataFrame.

        One row per ``ProcessedLifecycleEvent`` (knock-out, knock-in, coupon,
        maturity, expiry), indexed by event date.

        Returns:
            DataFrame with all lifecycle events (empty if none fired)
        """
        if not self.lifecycle_events:
            return pd.DataFrame()

        rows = []
        for item in self.lifecycle_events:
            event = item.event
            rows.append({
                'date': event.date,
                'position_id': item.position_id,
                'underlying': item.underlying,
                'product_type': item.product_type,
                'event_type': event.event_type.value,
                'observation_index': event.observation_index,
                'spot': event.spot,
                'barrier': event.barrier,
                'payoff': event.payoff,
                'cashflow': event.cashflow,
                'cashflow_id': (
                    event.realized_cashflow.cashflow_id
                    if event.realized_cashflow is not None
                    else None
                ),
                'determination_date': (
                    event.realized_cashflow.determination_date
                    if event.realized_cashflow is not None
                    else None
                ),
                'determination_time': (
                    event.realized_cashflow.determination_time
                    if event.realized_cashflow is not None
                    else None
                ),
                'payment_date': (
                    event.realized_cashflow.payment_date
                    if event.realized_cashflow is not None
                    else None
                ),
                'payment_time': (
                    event.realized_cashflow.payment_time
                    if event.realized_cashflow is not None
                    else None
                ),
                'terminates_position': event.terminates_position,
            })

        df = pd.DataFrame(rows)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        return df

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of state history.
        
        Returns:
            Dictionary with summary statistics
        """
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
            'max_portfolio_value': df['portfolio_value'].max() if 'portfolio_value' in df else 0,
            'min_portfolio_value': df['portfolio_value'].min() if 'portfolio_value' in df else 0,
        }
    
    def clear(self):
        """Clear all tracked states, trades, and lifecycle events."""
        self.states.clear()
        self.trades.clear()
        self.lifecycle_events.clear()
        self._current_state = None
    
    def __len__(self) -> int:
        """Return number of state snapshots."""
        return len(self.states)
    
    def __repr__(self) -> str:
        return f"StateTracker(snapshots={len(self.states)}, trades={len(self.trades)})"
