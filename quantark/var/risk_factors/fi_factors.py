"""
Fixed income-specific risk factors.
"""

import pandas as pd
import numpy as np
from typing import List


class ParallelShiftFactor:
    """Parallel yield curve shift risk factor."""
    
    @property
    def name(self) -> str:
        return "parallel_shift"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract parallel shifts from DataFrame.
        
        Args:
            df: DataFrame with 'parallel_shift' or 'rate' column
            
        Returns:
            Series of parallel shifts (basis points)
        """
        if 'parallel_shift' in df.columns:
            return df['parallel_shift']
        elif 'rate' in df.columns:
            return df['rate'].diff().dropna()
        else:
            raise ValueError("DataFrame must contain 'parallel_shift' or 'rate' column")
    
    def extract_from_market_data(self, market_data: any) -> pd.Series:
        """
        Extract parallel shifts from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            Series of parallel shifts
        """
        rate_history = market_data.get_rate_history()
        return rate_history.diff().dropna()


class KeyRateShiftFactor:
    """Key-rate shift risk factor for specific tenor points."""
    
    def __init__(self, tenors: List[float]):
        """
        Initialize with tenor points.
        
        Args:
            tenors: List of tenor points in years (e.g., [2.0, 5.0, 10.0, 30.0])
        """
        self.tenors = tenors
    
    @property
    def name(self) -> str:
        return f"key_rate_shift_{len(self.tenors)}pt"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract key-rate shifts from DataFrame.
        
        Args:
            df: DataFrame with columns like 'rate_2y', 'rate_5y', etc.
            
        Returns:
            DataFrame with key-rate shift columns
        """
        shift_cols = []
        for tenor in self.tenors:
            col_name = f"rate_{int(tenor)}y" if tenor == int(tenor) else f"rate_{tenor}y"
            if col_name in df.columns:
                shift_cols.append(df[col_name].diff().dropna())
            else:
                raise ValueError(f"DataFrame must contain '{col_name}' column for tenor {tenor}")
        
        return pd.concat(shift_cols, axis=1, keys=[f"shift_{int(t)}y" for t in self.tenors])
    
    def extract_from_market_data(self, market_data: any) -> pd.DataFrame:
        """
        Extract key-rate shifts from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            DataFrame with key-rate shift columns
        """
        shifts = {}
        for tenor in self.tenors:
            rate_history = market_data.get_rate_history(tenor=tenor)
            shifts[f"shift_{int(tenor)}y"] = rate_history.diff().dropna()
        
        return pd.DataFrame(shifts)
