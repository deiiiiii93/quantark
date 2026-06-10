"""
Convexity-neutral hedging strategy for Fixed Income portfolios.

Extends DV01-neutral to also manage convexity exposure.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from quantark.backtest.strategy.dv01_neutral_strategy import DV01NeutralStrategy
from quantark.backtest.strategy.base_strategy import HedgingTarget
from quantark.util.exceptions import ValidationError


class ConvexityNeutralStrategy(DV01NeutralStrategy):
    """
    Convexity-neutral hedging strategy for Fixed Income portfolios.
    
    This strategy extends DV01-neutral hedging to also monitor and manage
    portfolio convexity. It triggers hedges when either:
    1. DV01 exceeds its threshold, OR
    2. Convexity exceeds its threshold
    
    The strategy uses two hedging instruments:
    - Bond futures for DV01 hedging
    - (Optional) Options on bond futures for convexity hedging
    
    For this implementation, we focus on DV01 hedging while monitoring convexity.
    
    Attributes:
        dv01_threshold: Absolute DV01 level to trigger hedge
        convexity_threshold: Absolute convexity level to trigger hedge
        rebalance_frequency: Frequency for periodic rebalancing
        hedge_ratio: Proportion of risk to hedge (0-1)
        target_dv01: Target DV01 after hedging
        target_convexity: Target convexity after hedging
        futures_dv01: DV01 per futures contract
        futures_convexity: Convexity per futures contract
    
    Example:
        >>> strategy = ConvexityNeutralStrategy(
        ...     name="Conv_Neutral",
        ...     dv01_threshold=50000.0,
        ...     convexity_threshold=1000000.0,
        ...     futures_dv01=1000.0,
        ...     rebalance_frequency='daily'
        ... )
    """
    
    def __init__(
        self,
        name: str = "ConvexityNeutral",
        dv01_threshold: float = 50000.0,
        convexity_threshold: float = 1000000.0,
        rebalance_frequency: str = 'daily',
        hedge_instrument: str = 'bond_futures',
        hedge_ratio: float = 1.0,
        target_dv01: float = 0.0,
        target_convexity: float = 0.0,
        futures_dv01: float = 1000.0,
        futures_convexity: float = 10000.0,
        min_time_between_hedges: Optional[timedelta] = None
    ):
        """
        Initialize convexity-neutral strategy.
        
        Args:
            name: Strategy name
            dv01_threshold: Absolute DV01 to trigger hedge
            convexity_threshold: Absolute convexity to trigger hedge
            rebalance_frequency: When to rebalance
            hedge_instrument: Hedge instrument type
            hedge_ratio: Proportion of risk to hedge (0-1)
            target_dv01: Target DV01 after hedging
            target_convexity: Target convexity after hedging
            futures_dv01: DV01 per futures contract
            futures_convexity: Convexity per futures contract
            min_time_between_hedges: Minimum time between hedges
            
        Raises:
            ValidationError: If parameters are invalid
        """
        super().__init__(
            name=name,
            dv01_threshold=dv01_threshold,
            rebalance_frequency=rebalance_frequency,
            hedge_instrument=hedge_instrument,
            hedge_ratio=hedge_ratio,
            target_dv01=target_dv01,
            futures_dv01=futures_dv01,
            min_time_between_hedges=min_time_between_hedges
        )
        
        # Override hedging target
        self.hedging_target = HedgingTarget.CONVEXITY
        
        # Additional convexity-specific parameters
        if convexity_threshold < 0:
            raise ValidationError(
                f"Convexity threshold must be non-negative, got {convexity_threshold}"
            )
        
        self.convexity_threshold = convexity_threshold
        self.target_convexity = target_convexity
        self.futures_convexity = futures_convexity
        
        # Track convexity hedging
        self._total_convexity_hedged = 0.0
    
    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs
    ) -> bool:
        """
        Determine if hedging should be performed.
        
        Triggers when either:
        1. DV01 exceeds its threshold, OR
        2. Convexity exceeds its threshold
        
        AND frequency/timing conditions are met.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio risk measures (dv01, convexity)
            market_data: Market data
            **kwargs: Additional context
            
        Returns:
            True if hedging should be executed
        """
        current_dv01 = portfolio_greeks.get('dv01', 0.0)
        current_convexity = portfolio_greeks.get('convexity', 0.0)
        
        # Check if either threshold is exceeded
        dv01_exceeds = abs(current_dv01) > self.dv01_threshold
        convexity_exceeds = abs(current_convexity) > self.convexity_threshold
        
        threshold_exceeded = dv01_exceeds or convexity_exceeds
        
        # Check timing conditions
        if self.min_time_between_hedges is not None:
            time_since_hedge = self.time_since_last_hedge(current_time)
            if time_since_hedge is not None and time_since_hedge < self.min_time_between_hedges:
                return False
        
        # Check frequency
        if self.rebalance_frequency == 'on_threshold':
            return threshold_exceeded
        
        elif self.rebalance_frequency == 'continuous':
            return threshold_exceeded
        
        elif self.rebalance_frequency == 'daily':
            if not threshold_exceeded:
                return False
            
            if self._last_rebalance_date is not None:
                if current_time.date() == self._last_rebalance_date.date():
                    return False
            
            return True
        
        elif self.rebalance_frequency == 'hourly':
            if not threshold_exceeded:
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
        Calculate hedge size prioritizing DV01 neutralization.
        
        For now, we primarily hedge DV01 using futures. Full convexity
        hedging would require options, which is left for future enhancement.
        
        Args:
            current_time: Current timestamp
            portfolio_greeks: Portfolio risk measures
            market_data: Market data
            **kwargs: Additional context
            
        Returns:
            Number of futures contracts to trade
        """
        # Use base class DV01 hedge calculation
        return super().calculate_hedge_size(
            current_time, portfolio_greeks, market_data, **kwargs
        )
    
    def on_hedge_executed(
        self,
        current_time: datetime,
        hedge_size: float,
        hedge_price: float,
        **kwargs
    ):
        """Update strategy state after hedge execution."""
        super().on_hedge_executed(current_time, hedge_size, hedge_price, **kwargs)
        
        # Track convexity impact
        self._total_convexity_hedged += abs(hedge_size * self.futures_convexity)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        params = super().get_parameters()
        params.update({
            'convexity_threshold': self.convexity_threshold,
            'target_convexity': self.target_convexity,
            'futures_convexity': self.futures_convexity,
        })
        return params
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = super().get_statistics()
        stats['total_convexity_hedged'] = self._total_convexity_hedged
        return stats
    
    def reset(self):
        """Reset strategy state."""
        super().reset()
        self._total_convexity_hedged = 0.0
    
    def __repr__(self) -> str:
        return (
            f"ConvexityNeutralStrategy("
            f"dv01_thresh=${self.dv01_threshold:,.0f}, "
            f"conv_thresh=${self.convexity_threshold:,.0f}, "
            f"freq={self.rebalance_frequency})"
        )

