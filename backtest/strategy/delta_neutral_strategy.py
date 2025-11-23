"""
Delta-neutral hedging strategy implementation.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from backtest.strategy.base_strategy import BaseStrategy
from util.exceptions import ValidationError


class DeltaNeutralStrategy(BaseStrategy):
    """
    Delta-neutral hedging strategy.
    
    This strategy monitors portfolio delta and triggers hedges when:
    1. Delta exceeds a threshold, OR
    2. A rebalance frequency is reached
    
    The hedge aims to bring portfolio delta to a target level (typically 0).
    
    Attributes:
        delta_threshold: Absolute delta level to trigger hedge (e.g., 100)
        rebalance_frequency: Frequency for periodic rebalancing
        hedge_instrument: Type of hedge ('spot' or 'futures')
        hedge_ratio: Proportion of delta to hedge (0-1, default 1.0)
        target_delta: Target delta after hedging (default 0.0)
        min_time_between_hedges: Minimum time between hedges to avoid over-trading
    
    Example:
        >>> strategy = DeltaNeutralStrategy(
        ...     name="DN_100",
        ...     delta_threshold=100.0,
        ...     rebalance_frequency='daily',
        ...     hedge_instrument='spot'
        ... )
    """
    
    VALID_FREQUENCIES = ['daily', 'hourly', 'on_threshold', 'continuous']
    VALID_INSTRUMENTS = ['spot', 'futures']
    
    def __init__(
        self,
        name: str = "DeltaNeutral",
        delta_threshold: float = 100.0,
        rebalance_frequency: str = 'daily',
        hedge_instrument: str = 'spot',
        hedge_ratio: float = 1.0,
        target_delta: float = 0.0,
        min_time_between_hedges: Optional[timedelta] = None
    ):
        """
        Initialize delta-neutral strategy.
        
        Args:
            name: Strategy name
            delta_threshold: Absolute delta to trigger hedge
            rebalance_frequency: When to rebalance ('daily', 'hourly', 'on_threshold', 'continuous')
            hedge_instrument: Hedge instrument type ('spot' or 'futures')
            hedge_ratio: Proportion of delta to hedge (0-1)
            target_delta: Target delta after hedging
            min_time_between_hedges: Minimum time between hedges
            
        Raises:
            ValidationError: If parameters are invalid
        """
        super().__init__(name)
        
        # Validate parameters
        if delta_threshold < 0:
            raise ValidationError(
                f"Delta threshold must be non-negative, got {delta_threshold}"
            )
        
        if rebalance_frequency not in self.VALID_FREQUENCIES:
            raise ValidationError(
                f"Invalid rebalance_frequency '{rebalance_frequency}'. "
                f"Must be one of {self.VALID_FREQUENCIES}"
            )
        
        if hedge_instrument not in self.VALID_INSTRUMENTS:
            raise ValidationError(
                f"Invalid hedge_instrument '{hedge_instrument}'. "
                f"Must be one of {self.VALID_INSTRUMENTS}"
            )
        
        if not 0 <= hedge_ratio <= 1:
            raise ValidationError(
                f"Hedge ratio must be between 0 and 1, got {hedge_ratio}"
            )
        
        self.delta_threshold = delta_threshold
        self.rebalance_frequency = rebalance_frequency
        self.hedge_instrument = hedge_instrument
        self.hedge_ratio = hedge_ratio
        self.target_delta = target_delta
        self.min_time_between_hedges = min_time_between_hedges
        
        # Internal state
        self._hedge_count = 0
        self._total_delta_hedged = 0.0
        self._last_rebalance_date: Optional[datetime] = None
    
    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> bool:
        """
        Determine if hedging should be performed.
        
        Hedging triggers when:
        1. Absolute delta exceeds threshold, AND
        2. Frequency condition is met (if applicable), AND
        3. Minimum time between hedges has elapsed (if set)
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio Greeks
            market_data: Market data
            **kwargs: Additional context
            
        Returns:
            True if hedging should be executed
        """
        current_delta = portfolio_greeks.get('delta', 0.0)
        
        # Check 1: Does delta exceed threshold?
        delta_exceeds_threshold = abs(current_delta) > self.delta_threshold
        
        # Check 2: Has minimum time between hedges elapsed?
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if time_since_hedge is not None and time_since_hedge < self.min_time_between_hedges:
                return False
        
        # Check 3: Frequency-based conditions
        if self.rebalance_frequency == 'on_threshold':
            # Only hedge when threshold is breached
            return delta_exceeds_threshold
        
        elif self.rebalance_frequency == 'continuous':
            # Hedge whenever delta exceeds threshold
            return delta_exceeds_threshold
        
        elif self.rebalance_frequency == 'daily':
            # Hedge once per day if threshold breached
            if not delta_exceeds_threshold:
                return False
            
            # Check if we've already hedged today
            if self._last_rebalance_date is not None:
                if current_time.date() == self._last_rebalance_date.date():
                    return False
            
            return True
        
        elif self.rebalance_frequency == 'hourly':
            # Hedge once per hour if threshold breached
            if not delta_exceeds_threshold:
                return False
            
            # Check if we've hedged this hour
            if self._last_rebalance_date is not None:
                current_hour = current_time.replace(minute=0, second=0, microsecond=0)
                last_hour = self._last_rebalance_date.replace(minute=0, second=0, microsecond=0)
                if current_hour == last_hour:
                    return False
            
            return True
        
        return False
    
    def calculate_hedge_size(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> float:
        """
        Calculate hedge size to bring delta to target.
        
        Formula:
            hedge_size = -(current_delta - target_delta) * hedge_ratio
        
        Positive hedge_size means buy, negative means sell.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio Greeks
            market_data: Market data
            **kwargs: Additional context
            
        Returns:
            Hedge size (shares/contracts to trade)
        """
        current_delta = portfolio_greeks.get('delta', 0.0)
        
        # Calculate required hedge to reach target delta
        delta_to_hedge = current_delta - self.target_delta
        
        # Apply hedge ratio (allows partial hedging)
        hedge_size = -delta_to_hedge * self.hedge_ratio
        
        return hedge_size
    
    def on_step(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ):
        """Update strategy state at each step."""
        # Can be used for custom logic, logging, etc.
        pass
    
    def on_hedge_executed(
        self,
        current_time: datetime,
        hedge_size: float,
        hedge_price: float,
        **kwargs
    ):
        """Update strategy state after hedge execution."""
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        
        self._hedge_count += 1
        self._total_delta_hedged += abs(hedge_size)
        self._last_rebalance_date = current_time
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'delta_threshold': self.delta_threshold,
            'rebalance_frequency': self.rebalance_frequency,
            'hedge_instrument': self.hedge_instrument,
            'hedge_ratio': self.hedge_ratio,
            'target_delta': self.target_delta,
            'min_time_between_hedges': str(self.min_time_between_hedges) if self.min_time_between_hedges else None
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get strategy statistics.
        
        Returns:
            Dictionary with hedge statistics
        """
        return {
            'hedge_count': self._hedge_count,
            'total_delta_hedged': self._total_delta_hedged,
            'last_hedge_time': self._last_hedge_time,
            'last_rebalance_date': self._last_rebalance_date
        }
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._hedge_count = 0
        self._total_delta_hedged = 0.0
        self._last_rebalance_date = None
    
    def __repr__(self) -> str:
        return (
            f"DeltaNeutralStrategy("
            f"threshold={self.delta_threshold}, "
            f"freq={self.rebalance_frequency}, "
            f"instrument={self.hedge_instrument})"
        )

