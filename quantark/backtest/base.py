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

    The cross-asset contract is intentionally just ``run() -> results``: how an
    engine steps through time is an implementation detail (the equity and FI
    engines use ``_initialize``/``_step``/``_finalize``; the OTC autocallable
    engine drives a single-product futures-basis replay loop). All engines are
    constructed with their asset-specific config and produce a result object
    satisfying :class:`BaseBacktestResults`.
    """

    def run(self) -> Any:
        """
        Execute the backtest.

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

    def get_lifecycle_events(self) -> Any:
        """
        Get realized product lifecycle events.

        One row per realized event (knock-out, knock-in, coupon, maturity,
        expiry) detected during the backtest. Empty when the run had no
        lifecycle-bearing products or lifecycle handling was disabled.

        Returns:
            DataFrame of lifecycle events
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


def get_backtest_engine(config: Any) -> BaseBacktestEngine:
    """
    Build the backtest engine matching a config.

    Dispatches on the config type so callers can stay engine-agnostic:

    - ``AutocallableBacktestConfig`` -> ``AutocallableBacktestEngine``
      (single-product OTC futures-basis replay)
    - ``FIBacktestConfig`` -> ``FIBacktestEngine`` (fixed income portfolio)
    - ``BacktestConfig`` / ``EquityBacktestConfig`` -> ``BacktestEngine``
      (equity portfolio)

    All returned engines expose ``run()`` and produce results satisfying
    :class:`BaseBacktestResults`. Imports are deferred to avoid importing every
    asset-class engine at module load.

    Args:
        config: An asset-class backtest configuration.

    Returns:
        The engine instance for that config.

    Raises:
        ValidationError: If no engine matches the config type.
    """
    from quantark.util.exceptions import ValidationError

    from quantark.backtest.replay.config import BookAutocallableBacktestConfig
    from quantark.backtest.replay.config import AutocallableBacktestConfig

    if isinstance(config, BookAutocallableBacktestConfig):
        from quantark.backtest.replay.engine import BookAutocallableBacktestEngine

        return BookAutocallableBacktestEngine(config)

    if isinstance(config, AutocallableBacktestConfig):
        from quantark.backtest.replay.single import AutocallableBacktestEngine

        return AutocallableBacktestEngine(config)

    try:
        from quantark.backtest.fi.config import FIBacktestConfig
    except Exception:  # pragma: no cover - FI optional at import time
        FIBacktestConfig = None

    if FIBacktestConfig is not None and isinstance(config, FIBacktestConfig):
        from quantark.backtest.fi.engine import FIBacktestEngine

        return FIBacktestEngine(config)

    from quantark.backtest.equity.config import BacktestConfig

    if isinstance(config, BacktestConfig):
        from quantark.backtest.equity.engine import BacktestEngine

        return BacktestEngine(config)

    raise ValidationError(
        f"No backtest engine registered for config type "
        f"{type(config).__name__!r}"
    )

