"""
Abstract base strategy for backtesting.

Supports multiple asset classes with configurable hedging targets:
- Equity: delta, gamma, vega hedging
- Fixed Income: DV01, convexity hedging
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from enum import Enum


class AssetClass(Enum):
    """Supported asset classes for hedging strategies."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    GENERIC = "generic"


class HedgingTarget(Enum):
    """
    Risk measures that can be used as hedging targets.

    Equity-oriented:
        DELTA: First-order price sensitivity to underlying
        GAMMA: Second-order price sensitivity
        VEGA: Sensitivity to volatility

    Fixed Income-oriented:
        DV01: Dollar value of 1 basis point rate change
        CONVEXITY: Second-order rate sensitivity
        DURATION: Modified duration
    """

    # Equity
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"

    # Fixed Income
    DV01 = "dv01"
    CONVEXITY = "convexity"
    DURATION = "duration"


VALID_REBALANCE_FREQUENCIES = ["daily", "hourly", "on_threshold", "continuous"]


def passes_frequency_gate(
    frequency: str,
    breach: bool,
    current_time: datetime,
    last_rebalance_time: Optional[datetime],
) -> bool:
    """
    Shared rebalance-frequency gating for threshold-driven strategies.

    Args:
        frequency: One of VALID_REBALANCE_FREQUENCIES
        breach: Whether the hedging threshold is currently breached
        current_time: Current timestamp
        last_rebalance_time: Timestamp of the last executed rebalance

    Returns:
        True if a hedge is allowed at this timestep

    Semantics match DeltaNeutralStrategy: 'on_threshold' and 'continuous'
    hedge whenever the threshold is breached; 'daily'/'hourly' additionally
    allow at most one hedge per day/hour.
    """
    if not breach:
        return False

    if frequency in ("on_threshold", "continuous"):
        return True

    if frequency == "daily":
        if last_rebalance_time is not None:
            if current_time.date() == last_rebalance_time.date():
                return False
        return True

    if frequency == "hourly":
        if last_rebalance_time is not None:
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            last_hour = last_rebalance_time.replace(
                minute=0, second=0, microsecond=0
            )
            if current_hour == last_hour:
                return False
        return True

    return False


class BaseStrategy(ABC):
    """
    Abstract base class for hedging strategies.

    All strategies must implement methods for:
    - Determining when to hedge
    - Calculating hedge sizes
    - Reacting to market events

    Subclasses should define their own parameters and logic.

    Attributes:
        name: Strategy identifier
        asset_class: Target asset class (equity, fixed_income, generic)
        hedging_target: Primary risk measure to hedge
        hedge_instrument: Type of instrument used for hedging
    """

    def __init__(
        self,
        name: str,
        asset_class: AssetClass = AssetClass.GENERIC,
        hedging_target: HedgingTarget = HedgingTarget.DELTA,
        hedge_instrument: str = "spot",
    ):
        """
        Initialize base strategy.

        Args:
            name: Name identifier for the strategy
            asset_class: Target asset class
            hedging_target: Primary risk measure to hedge
            hedge_instrument: Type of hedge instrument
        """
        self.name = name
        self.asset_class = asset_class
        self.hedging_target = hedging_target
        self.hedge_instrument = hedge_instrument
        self._last_hedge_time: Optional[datetime] = None

    @abstractmethod
    def should_hedge(
        self,
        current_time: datetime,
        portfolio_greeks: Dict[str, float],
        market_data: Dict[str, float],
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        self, current_time: datetime, hedge_size: float, hedge_price: float, **kwargs
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

    def get_primary_risk_measure(self, risk_measures: Dict[str, float]) -> float:
        """
        Get the primary risk measure value from risk measures dictionary.

        Uses self.hedging_target to determine which measure to extract.

        Args:
            risk_measures: Dictionary of risk measures (greeks or FI measures)

        Returns:
            Value of the primary hedging target
        """
        target_key = self.hedging_target.value
        return risk_measures.get(target_key, 0.0)

    def get_supported_risk_measures(self) -> List[str]:
        """
        Get list of risk measures this strategy monitors.

        Returns:
            List of risk measure names
        """
        if self.asset_class == AssetClass.EQUITY:
            return ["delta", "gamma", "vega", "theta", "rho"]
        elif self.asset_class == AssetClass.FIXED_INCOME:
            return ["dv01", "convexity", "modified_duration", "yield"]
        else:
            return [self.hedging_target.value]

    def reset(self):
        """Reset strategy state."""
        self._last_hedge_time = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"target={self.hedging_target.value})"
        )
