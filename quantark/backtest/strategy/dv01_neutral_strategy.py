"""
DV01-neutral hedging strategy for Fixed Income portfolios.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from quantark.backtest.strategy.base_strategy import BaseStrategy, AssetClass, HedgingTarget
from quantark.util.exceptions import ValidationError


class DV01NeutralStrategy(BaseStrategy):
    """
    DV01-neutral hedging strategy for Fixed Income portfolios.
    
    This strategy monitors portfolio DV01 and triggers hedges when:
    1. DV01 exceeds a threshold, OR
    2. A rebalance frequency is reached
    
    The hedge aims to bring portfolio DV01 to a target level (typically 0)
    using bond futures as the hedging instrument.
    
    Attributes:
        dv01_threshold: Absolute DV01 level to trigger hedge (e.g., $50,000)
        rebalance_frequency: Frequency for periodic rebalancing
        hedge_instrument: Type of hedge ('bond_futures')
        hedge_ratio: Proportion of DV01 to hedge (0-1, default 1.0)
        target_dv01: Target DV01 after hedging (default 0.0)
        futures_dv01: DV01 per futures contract (for sizing)
        min_time_between_hedges: Minimum time between hedges
    
    Example:
        >>> strategy = DV01NeutralStrategy(
        ...     name="DV01_Neutral",
        ...     dv01_threshold=50000.0,
        ...     futures_dv01=1000.0,
        ...     rebalance_frequency='daily'
        ... )
    """
    
    VALID_FREQUENCIES = ['daily', 'hourly', 'on_threshold', 'continuous']
    VALID_INSTRUMENTS = ['bond_futures']
    
    def __init__(
        self,
        name: str = "DV01Neutral",
        dv01_threshold: float = 50000.0,
        rebalance_frequency: str = 'daily',
        hedge_instrument: str = 'bond_futures',
        hedge_ratio: float = 1.0,
        target_dv01: float = 0.0,
        futures_dv01: float = 1000.0,
        min_time_between_hedges: Optional[timedelta] = None
    ):
        """
        Initialize DV01-neutral strategy.
        
        Args:
            name: Strategy name
            dv01_threshold: Absolute DV01 to trigger hedge (in dollars)
            rebalance_frequency: When to rebalance ('daily', 'hourly', 'on_threshold', 'continuous')
            hedge_instrument: Hedge instrument type ('bond_futures')
            hedge_ratio: Proportion of DV01 to hedge (0-1)
            target_dv01: Target DV01 after hedging (default 0)
            futures_dv01: DV01 per futures contract (for hedge sizing)
            min_time_between_hedges: Minimum time between hedges
            
        Raises:
            ValidationError: If parameters are invalid
        """
        super().__init__(
            name=name,
            asset_class=AssetClass.FIXED_INCOME,
            hedging_target=HedgingTarget.DV01,
            hedge_instrument=hedge_instrument
        )
        
        # Validate parameters
        if dv01_threshold < 0:
            raise ValidationError(
                f"DV01 threshold must be non-negative, got {dv01_threshold}"
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
        
        if futures_dv01 <= 0:
            raise ValidationError(
                f"Futures DV01 must be positive, got {futures_dv01}"
            )
        
        self.dv01_threshold = dv01_threshold
        self.rebalance_frequency = rebalance_frequency
        self.hedge_ratio = hedge_ratio
        self.target_dv01 = target_dv01
        self.futures_dv01 = futures_dv01
        self.min_time_between_hedges = min_time_between_hedges
        
        # Internal state
        self._hedge_count = 0
        self._total_dv01_hedged = 0.0
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
        1. Absolute DV01 exceeds threshold, AND
        2. Frequency condition is met (if applicable), AND
        3. Minimum time between hedges has elapsed (if set)
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio risk measures (should contain 'dv01')
            market_data: Market data
            **kwargs: Additional context
            
        Returns:
            True if hedging should be executed
        """
        current_dv01 = portfolio_greeks.get('dv01', 0.0)
        
        # Check 1: Does DV01 exceed threshold?
        dv01_exceeds_threshold = abs(current_dv01) > self.dv01_threshold
        
        # Check 2: Has minimum time between hedges elapsed?
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if time_since_hedge is not None and time_since_hedge < self.min_time_between_hedges:
                return False
        
        # Check 3: Frequency-based conditions
        if self.rebalance_frequency == 'on_threshold':
            return dv01_exceeds_threshold
        
        elif self.rebalance_frequency == 'continuous':
            return dv01_exceeds_threshold
        
        elif self.rebalance_frequency == 'daily':
            if not dv01_exceeds_threshold:
                return False
            
            if self._last_rebalance_date is not None:
                if current_time.date() == self._last_rebalance_date.date():
                    return False
            
            return True
        
        elif self.rebalance_frequency == 'hourly':
            if not dv01_exceeds_threshold:
                return False
            
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
        Calculate hedge size in number of futures contracts.
        
        Formula:
            num_contracts = -(current_dv01 - target_dv01) / futures_dv01 * hedge_ratio
        
        Positive = buy futures, Negative = sell futures.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio risk measures
            market_data: Market data
            **kwargs: Additional context
            
        Returns:
            Number of futures contracts to trade (can be fractional, round in executor)
        """
        current_dv01 = portfolio_greeks.get('dv01', 0.0)
        
        # Calculate DV01 to hedge
        dv01_to_hedge = current_dv01 - self.target_dv01
        
        # Calculate number of contracts (short futures to reduce long DV01)
        num_contracts = -dv01_to_hedge / self.futures_dv01 * self.hedge_ratio
        
        return num_contracts
    
    def on_step(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ):
        """Update strategy state at each step."""
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
        self._total_dv01_hedged += abs(hedge_size * self.futures_dv01)
        self._last_rebalance_date = current_time
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'asset_class': self.asset_class.value,
            'hedging_target': self.hedging_target.value,
            'dv01_threshold': self.dv01_threshold,
            'rebalance_frequency': self.rebalance_frequency,
            'hedge_instrument': self.hedge_instrument,
            'hedge_ratio': self.hedge_ratio,
            'target_dv01': self.target_dv01,
            'futures_dv01': self.futures_dv01,
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
            'total_dv01_hedged': self._total_dv01_hedged,
            'last_hedge_time': self._last_hedge_time,
            'last_rebalance_date': self._last_rebalance_date
        }
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._hedge_count = 0
        self._total_dv01_hedged = 0.0
        self._last_rebalance_date = None
    
    def __repr__(self) -> str:
        return (
            f"DV01NeutralStrategy("
            f"threshold=${self.dv01_threshold:,.0f}, "
            f"freq={self.rebalance_frequency}, "
            f"futures_dv01=${self.futures_dv01:,.0f})"
        )

