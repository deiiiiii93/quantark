"""
Base protocols for dynamic scenario analysis.

This module defines the core protocols (interfaces) that all dynamic scenario
engines must implement, enabling asset-class-agnostic dynamic scenario analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
from datetime import datetime

from quantark.util.numerical import pnl_pct_of_abs_baseline


@runtime_checkable
class RiskMetricsAdapter(Protocol):
    """
    Protocol for computing risk metrics for a portfolio.
    
    Different asset classes have different risk measures:
    - Equity: delta, gamma, vega, theta
    - Fixed Income: DV01, convexity, modified duration
    
    Engines use adapters to compute appropriate metrics for their asset class.
    """
    
    def compute_metrics(
        self,
        portfolio: Any,
        pricing_environments: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute risk metrics for a portfolio.
        
        Args:
            portfolio: Portfolio to compute metrics for
            pricing_environments: Pricing environments by underlying
            
        Returns:
            Dictionary of computed risk metrics
        """
        ...
    
    def get_metric_names(self) -> List[str]:
        """
        Get list of metric names this adapter computes.
        
        Returns:
            List of metric names (e.g., ['delta', 'gamma'] or ['dv01', 'convexity'])
        """
        ...


@dataclass
class BaseDayResult:
    """
    Base class for day result data.
    
    Contains common fields across all asset classes. Subclasses add
    asset-class-specific fields.
    """
    day_index: int
    date: Optional[datetime] = None
    label: Optional[str] = None
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    transaction_costs_today: float = 0.0
    cumulative_transaction_costs: float = 0.0
    net_pnl: float = 0.0
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_trades(self) -> int:
        """Get number of trades executed today (override in subclass)."""
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'day_index': self.day_index,
            'date': self.date.isoformat() if self.date else None,
            'label': self.label,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'transaction_costs_today': self.transaction_costs_today,
            'cumulative_transaction_costs': self.cumulative_transaction_costs,
            'net_pnl': self.net_pnl,
            'risk_metrics': self.risk_metrics,
            'metadata': self.metadata,
        }


@dataclass
class BaseScenarioResults:
    """
    Base class for dynamic scenario results.
    
    Contains common fields across all asset classes. Subclasses add
    asset-class-specific analysis methods.
    """
    path_name: str
    baseline_value: float
    final_value: float
    day_results: List[BaseDayResult]
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    total_transaction_costs: float = 0.0
    net_pnl: float = 0.0
    total_hedges: int = 0
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    asset_class: str = "unknown"
    
    def __post_init__(self):
        """Calculate derived fields if not set."""
        if self.total_pnl == 0.0 and self.day_results:
            self.total_pnl = self.final_value - self.baseline_value
        
        if self.total_pnl_pct == 0.0 and self.baseline_value != 0:
            self.total_pnl_pct = pnl_pct_of_abs_baseline(
                self.total_pnl,
                self.baseline_value,
            )
        
        if self.net_pnl == 0.0:
            self.net_pnl = self.total_pnl - self.total_transaction_costs
    
    @property
    def num_days(self) -> int:
        """Get number of days in scenario."""
        return len(self.day_results)
    
    def get_day_result(self, day_index: int) -> Optional[BaseDayResult]:
        """Get result for specific day."""
        if 0 <= day_index < len(self.day_results):
            return self.day_results[day_index]
        return None
    
    def get_worst_day(self) -> Optional[BaseDayResult]:
        """Get day with worst P&L."""
        if not self.day_results:
            return None
        return min(self.day_results, key=lambda d: d.daily_pnl)
    
    def get_best_day(self) -> Optional[BaseDayResult]:
        """Get day with best P&L."""
        if not self.day_results:
            return None
        return max(self.day_results, key=lambda d: d.daily_pnl)
    
    def get_max_drawdown(self) -> tuple:
        """
        Calculate maximum drawdown.
        
        Returns:
            Tuple of (max_drawdown_amount, max_drawdown_pct, peak_day, trough_day)
        """
        if not self.day_results:
            return 0.0, 0.0, None, None
        
        peak_value = self.baseline_value
        peak_day = 0
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        trough_day = 0
        
        for result in self.day_results:
            if result.portfolio_value > peak_value:
                peak_value = result.portfolio_value
                peak_day = result.day_index
            
            drawdown = peak_value - result.portfolio_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = (drawdown / peak_value) * 100 if peak_value > 0 else 0
                trough_day = result.day_index
        
        return max_drawdown, max_drawdown_pct, peak_day, trough_day
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'path_name': self.path_name,
            'baseline_value': self.baseline_value,
            'final_value': self.final_value,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'total_transaction_costs': self.total_transaction_costs,
            'net_pnl': self.net_pnl,
            'total_hedges': self.total_hedges,
            'num_days': self.num_days,
            'execution_timestamp': self.execution_timestamp.isoformat(),
            'total_execution_time': self.total_execution_time,
            'config_summary': self.config_summary,
            'day_results': [d.to_dict() for d in self.day_results],
            'metadata': self.metadata,
            'asset_class': self.asset_class,
        }


class BaseDynamicScenarioEngine(ABC):
    """
    Abstract base class for dynamic scenario engines.
    
    Defines the interface that all dynamic scenario engines must implement,
    regardless of asset class.
    """
    
    @abstractmethod
    def supports_portfolio(self, portfolio: Any) -> bool:
        """
        Check if this engine supports the given portfolio type.
        
        Args:
            portfolio: Portfolio to check
            
        Returns:
            True if this engine can process the portfolio
        """
        pass
    
    @abstractmethod
    def run(
        self,
        portfolio: Any,
        day_path: Any,
        hedge_strategy: Optional[Any] = None,
        transaction_cost_model: Optional[Any] = None
    ) -> BaseScenarioResults:
        """
        Run dynamic scenario simulation.
        
        Args:
            portfolio: Portfolio to simulate
            day_path: Day path defining market evolution
            hedge_strategy: Optional hedging strategy
            transaction_cost_model: Optional transaction cost model
            
        Returns:
            BaseScenarioResults (or subclass) with day-by-day evolution
        """
        pass
    
    @abstractmethod
    def get_asset_class(self) -> str:
        """
        Get the asset class this engine handles.
        
        Returns:
            Asset class identifier (e.g., 'equity', 'fixed_income')
        """
        pass


def get_engine_for_portfolio(
    portfolio: Any,
    config: Optional[Any] = None,
    engines: Optional[List[BaseDynamicScenarioEngine]] = None
) -> BaseDynamicScenarioEngine:
    """
    Factory function to get appropriate engine for a portfolio.
    
    Args:
        portfolio: Portfolio to get engine for
        config: Optional configuration
        engines: Optional list of available engines
        
    Returns:
        Appropriate engine for the portfolio type
        
    Raises:
        ValueError: If no engine supports the portfolio type
    """
    # Import here to avoid circular imports
    from quantark.portfolio import Portfolio
    from quantark.portfolio.fi import FIPortfolio
    
    # Determine portfolio type and return appropriate engine
    if isinstance(portfolio, FIPortfolio):
        from quantark.dynamicscenario.fi.engine import FIDynamicScenarioEngine
        from quantark.dynamicscenario.fi.config import FIDynamicScenarioConfig
        fi_config = config if isinstance(config, FIDynamicScenarioConfig) else FIDynamicScenarioConfig()
        return FIDynamicScenarioEngine(fi_config)
    
    elif isinstance(portfolio, Portfolio):
        from quantark.dynamicscenario.engine import DynamicScenarioEngine
        from quantark.dynamicscenario.config import DynamicScenarioConfig
        eq_config = config if isinstance(config, DynamicScenarioConfig) else DynamicScenarioConfig()
        return DynamicScenarioEngine(eq_config)
    
    raise ValueError(f"No engine available for portfolio type: {type(portfolio).__name__}")
