"""
Base classes and protocols for VaR engines.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd

from var.config import VaRMethod


@runtime_checkable
class VaREngine(Protocol):
    """Protocol defining the interface for VaR calculation engines."""
    
    def calculate_var(
        self,
        portfolio: Any,
        historical_data: Union[Any, pd.DataFrame],
    ) -> "VaRResult":
        """
        Calculate VaR for the portfolio.
        
        Args:
            portfolio: Portfolio object (EquityPortfolio or FIPortfolio)
            historical_data: Historical market data (MarketDataSet or DataFrame)
            
        Returns:
            VaRResult containing VaR metrics and attribution
        """
        ...
    
    def supports_portfolio(self, portfolio: Any) -> bool:
        """
        Check if this engine supports the portfolio type.
        
        Args:
            portfolio: Portfolio object to check
            
        Returns:
            True if supported, False otherwise
        """
        ...


@dataclass
class VaRResult:
    """Results from a VaR calculation."""
    
    var: float
    cvar: float
    confidence_level: float
    holding_period: int
    method: VaRMethod
    
    portfolio_value: float
    var_as_pct: float
    
    component_var: Optional[Dict[str, float]] = None
    marginal_var: Optional[Dict[str, float]] = None
    factor_var: Optional[Dict[str, float]] = None
    incremental_var: Optional[Dict[str, float]] = None
    
    scenarios: Optional[pd.DataFrame] = None
    worst_scenarios: Optional[List[Dict]] = None
    
    stressed_var: Optional[float] = None
    stressed_cvar: Optional[float] = None
    stressed_period: Optional[Dict[str, datetime]] = None
    
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    execution_time_seconds: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
