"""
Base protocols for backtest engine and hedge executor.

These protocols define the interface for backtest components that can be
implemented by different asset classes (equity, fixed income, etc.).
"""
from typing import Protocol, Dict, Any, Optional, runtime_checkable
from datetime import datetime
from dataclasses import dataclass


@dataclass
class BaseTradeRecord:
    """
    Base trade record for all asset classes.
    
    Attributes:
        timestamp: Trade execution time
        trade_type: Type of trade (open, adjust, close, no_trade)
        instrument_type: Type of hedge instrument
        underlying: Underlying identifier
        quantity: Trade quantity
        price: Execution price
        notional: Trade notional value
        transaction_cost: Transaction cost
        reason: Reason for trade
        position_id: Position identifier (if applicable)
        metadata: Additional trade details
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
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class BaseHedgeExecutor(Protocol):
    """
    Protocol for hedge execution across asset classes.
    
    Defines the interface for creating, managing, and closing hedge positions.
    Asset-specific implementations handle the details of hedge instruments
    (spot/futures for equity, bond futures for fixed income).
    """
    
    def execute_hedge(
        self,
        underlying: str,
        hedge_size: float,
        pricing_context: Any,
        current_time: datetime,
        reason: str = "hedge"
    ) -> BaseTradeRecord:
        """
        Execute a hedge trade.
        
        Args:
            underlying: Underlying asset identifier
            hedge_size: Size to hedge (positive=buy, negative=sell)
            pricing_context: Asset-specific pricing context
            current_time: Execution timestamp
            reason: Reason for the hedge
            
        Returns:
            Trade record with execution details
        """
        ...
    
    def get_hedge_position(self, underlying: str) -> Optional[Any]:
        """
        Get the current hedge position for an underlying.
        
        Args:
            underlying: Underlying identifier
            
        Returns:
            Position or None if no hedge exists
        """
        ...
    
    def get_hedge_quantity(self, underlying: str) -> float:
        """
        Get the current hedge quantity.
        
        Args:
            underlying: Underlying identifier
            
        Returns:
            Current hedge quantity (0 if no hedge)
        """
        ...
    
    def close_hedge_position(
        self,
        underlying: str,
        pricing_context: Any,
        current_time: datetime,
        reason: str = "close_hedge"
    ) -> Optional[BaseTradeRecord]:
        """
        Close a hedge position completely.
        
        Args:
            underlying: Underlying identifier
            pricing_context: Asset-specific pricing context
            current_time: Execution timestamp
            reason: Reason for closing
            
        Returns:
            Trade record or None if no position exists
        """
        ...
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get hedge executor statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...


@runtime_checkable
class BaseBacktestEngine(Protocol):
    """
    Protocol for backtest engines across asset classes.
    
    Defines the interface for running backtests, including initialization,
    stepping through time, and generating results.
    """
    
    def run(self) -> Any:
        """
        Execute the backtest.
        
        Returns:
            Backtest results object
        """
        ...
    
    def _initialize(self) -> None:
        """Initialize portfolio, pricing environment, and hedge executor."""
        ...
    
    def _step(self, timestamp: datetime) -> None:
        """
        Execute a single backtest step.
        
        Args:
            timestamp: Current timestamp
        """
        ...
    
    def _finalize(self) -> Any:
        """
        Finalize backtest and create results.
        
        Returns:
            Backtest results object
        """
        ...


@runtime_checkable
class BaseBacktestResults(Protocol):
    """
    Protocol for backtest results across asset classes.
    
    Defines the interface for accessing backtest results including
    time series data, trade history, and summary statistics.
    """
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get backtest summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        ...
    
    def get_total_pnl(self) -> float:
        """
        Get total P&L.
        
        Returns:
            Total profit/loss
        """
        ...
    
    def get_total_return(self) -> float:
        """
        Get total return as decimal.
        
        Returns:
            Total return (e.g., 0.10 for 10%)
        """
        ...
    
    def get_pnl_series(self) -> Any:
        """
        Get P&L time series.
        
        Returns:
            Time series of P&L values
        """
        ...
    
    def get_value_series(self) -> Any:
        """
        Get portfolio value time series.
        
        Returns:
            Time series of portfolio values
        """
        ...
    
    def get_hedge_trades(self) -> Any:
        """
        Get hedge trade history.
        
        Returns:
            DataFrame or list of trades
        """
        ...


@runtime_checkable
class BaseBacktestConfig(Protocol):
    """
    Protocol for backtest configuration across asset classes.
    """
    
    start_date: datetime
    end_date: datetime
    underlying: str
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.
        
        Returns:
            Dictionary with configuration details
        """
        ...
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        ...

