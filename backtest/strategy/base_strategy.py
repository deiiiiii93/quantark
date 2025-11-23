"""
Abstract base strategy for backtesting.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


class BaseStrategy(ABC):
    """
    Abstract base class for hedging strategies.
    
    All strategies must implement methods for:
    - Determining when to hedge
    - Calculating hedge sizes
    - Reacting to market events
    
    Subclasses should define their own parameters and logic.
    """
    
    def __init__(self, name: str):
        """
        Initialize base strategy.
        
        Args:
            name: Name identifier for the strategy
        """
        self.name = name
        self._last_hedge_time: Optional[datetime] = None
    
    @abstractmethod
    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> bool:
        """
        Determine if hedging should be performed.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Current portfolio Greeks
            market_data: Current market data (spot, vol, rate, etc.)
            **kwargs: Additional context
            
        Returns:
            True if hedging should be executed
        """
        pass
    
    @abstractmethod
    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> float:
        """
        Calculate the size of the hedge to execute.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Current portfolio Greeks
            market_data: Current market data
            **kwargs: Additional context
            
        Returns:
            Hedge size (positive=buy, negative=sell)
        """
        pass
    
    def on_step(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ):
        """
        Called at each backtest timestep before hedging decision.
        
        Can be used for strategy state updates, logging, etc.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Current portfolio Greeks
            market_data: Current market data
            **kwargs: Additional context
        """
        pass
    
    def on_hedge_executed(
        self,
        current_time: datetime,
        hedge_size: float,
        hedge_price: float,
        **kwargs
    ):
        """
        Called after a hedge is executed.
        
        Can be used for updating internal state.
        
        Args:
            current_time: Timestamp of execution
            hedge_size: Size of hedge executed
            hedge_price: Execution price
            **kwargs: Additional context
        """
        self._last_hedge_time = current_time
    
    def get_last_hedge_time(self) -> Optional[datetime]:
        """Get timestamp of last hedge execution."""
        return self._last_hedge_time
    
    def time_since_last_hedge(self, current_time: datetime) -> Optional[timedelta]:
        """
        Calculate time elapsed since last hedge.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Time delta since last hedge, or None if no hedge yet
        """
        if self._last_hedge_time is None:
            return None
        return current_time - self._last_hedge_time
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        pass
    
    def reset(self):
        """Reset strategy state."""
        self._last_hedge_time = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

