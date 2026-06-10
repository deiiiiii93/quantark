"""
Base protocol for risk factors.
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class RiskFactor(Protocol):
    """Protocol defining the interface for risk factors."""
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract risk factor values from a DataFrame.
        
        Args:
            df: DataFrame with historical market data
            
        Returns:
            Series of risk factor values (e.g., returns, changes)
        """
        ...
    
    def extract_from_market_data(self, market_data: any) -> pd.Series:
        """
        Extract risk factor values from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            Series of risk factor values
        """
        ...
    
    @property
    def name(self) -> str:
        """Name of the risk factor."""
        ...
