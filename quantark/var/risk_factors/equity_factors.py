"""
Equity-specific risk factors.
"""

import pandas as pd
import numpy as np


class SpotReturnFactor:
    """Spot price return risk factor."""
    
    @property
    def name(self) -> str:
        return "spot_return"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract spot returns from DataFrame.
        
        Args:
            df: DataFrame with 'spot' or 'spot_return' column
            
        Returns:
            Series of spot returns
        """
        if 'spot_return' in df.columns:
            return df['spot_return']
        elif 'spot' in df.columns:
            return df['spot'].pct_change().dropna()
        else:
            raise ValueError("DataFrame must contain 'spot' or 'spot_return' column")
    
    def extract_from_market_data(self, market_data: any) -> pd.Series:
        """
        Extract spot returns from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            Series of spot returns
        """
        spot_history = market_data.get_spot_history()
        return spot_history.pct_change().dropna()


class VolChangeFactor:
    """Implied volatility change risk factor."""
    
    @property
    def name(self) -> str:
        return "vol_change"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract volatility changes from DataFrame.
        
        Args:
            df: DataFrame with 'vol' or 'vol_change' column
            
        Returns:
            Series of vol changes (absolute changes, not percentage)
        """
        if 'vol_change' in df.columns:
            return df['vol_change']
        elif 'vol' in df.columns:
            return df['vol'].diff().dropna()
        else:
            raise ValueError("DataFrame must contain 'vol' or 'vol_change' column")
    
    def extract_from_market_data(self, market_data: any) -> pd.Series:
        """
        Extract volatility changes from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            Series of vol changes
        """
        vol_history = market_data.get_vol_history()
        return vol_history.diff().dropna()


class RateShiftFactor:
    """Interest rate shift risk factor."""
    
    @property
    def name(self) -> str:
        return "rate_shift"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract rate shifts from DataFrame.
        
        Args:
            df: DataFrame with 'rate' or 'rate_shift' column
            
        Returns:
            Series of rate shifts (absolute changes in basis points)
        """
        if 'rate_shift' in df.columns:
            return df['rate_shift']
        elif 'rate' in df.columns:
            return df['rate'].diff().dropna()
        else:
            raise ValueError("DataFrame must contain 'rate' or 'rate_shift' column")
    
    def extract_from_market_data(self, market_data: any) -> pd.Series:
        """
        Extract rate shifts from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            Series of rate shifts
        """
        rate_history = market_data.get_rate_history()
        return rate_history.diff().dropna()


class DivYieldShiftFactor:
    """Dividend yield shift risk factor."""
    
    @property
    def name(self) -> str:
        return "div_yield_shift"
    
    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract dividend yield shifts from DataFrame.
        
        Args:
            df: DataFrame with 'div_yield' or 'div_yield_shift' column
            
        Returns:
            Series of div yield shifts (absolute changes)
        """
        if 'div_yield_shift' in df.columns:
            return df['div_yield_shift']
        elif 'div_yield' in df.columns:
            return df['div_yield'].diff().dropna()
        else:
            raise ValueError("DataFrame must contain 'div_yield' or 'div_yield_shift' column")
    
    def extract_from_market_data(self, market_data: any) -> pd.Series:
        """
        Extract dividend yield shifts from MarketDataSet.
        
        Args:
            market_data: MarketDataSet object
            
        Returns:
            Series of div yield shifts
        """
        div_yield_history = market_data.get_div_yield_history()
        return div_yield_history.diff().dropna()
