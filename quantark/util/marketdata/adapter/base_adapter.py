"""
Base adapter for fetching market data from different sources.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from util.marketdata.models import TimeSeriesData, MarketDataPoint, MarketDataSet


class BaseMarketDataAdapter(ABC):
    """
    Abstract base class for market data adapters.
    
    Each data source (Bloomberg, Wind, Yahoo Finance, etc.) should
    implement this interface to provide a consistent API for fetching
    market data.
    """
    
    def __init__(self, source_name: str):
        """
        Initialize adapter.
        
        Args:
            source_name: Name of the data source (e.g., 'Bloomberg', 'Mock')
        """
        self.source_name = source_name
        self._cache = {}  # Simple cache for repeated queries
    
    @abstractmethod
    def get_spot_history(self, asset_name: str, 
                        start_date: datetime, 
                        end_date: datetime,
                        frequency: str = 'D') -> TimeSeriesData:
        """
        Get historical spot prices.
        
        Args:
            asset_name: Asset identifier (e.g., 'AAPL', 'SPX')
            start_date: Start date of the history
            end_date: End date of the history
            frequency: Data frequency ('D' for daily, 'H' for hourly, etc.)
            
        Returns:
            TimeSeriesData containing spot prices
        """
        pass
    
    @abstractmethod
    def get_vol_history(self, asset_name: str,
                       start_date: datetime,
                       end_date: datetime,
                       frequency: str = 'D') -> TimeSeriesData:
        """
        Get historical implied volatility.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date of the history
            end_date: End date of the history
            frequency: Data frequency
            
        Returns:
            TimeSeriesData containing volatility data
        """
        pass
    
    @abstractmethod
    def get_rate_history(self, currency: str,
                        start_date: datetime,
                        end_date: datetime,
                        tenor: str = '1Y',
                        frequency: str = 'D') -> TimeSeriesData:
        """
        Get historical interest rates.
        
        Args:
            currency: Currency code (e.g., 'USD', 'EUR')
            start_date: Start date of the history
            end_date: End date of the history
            tenor: Rate tenor (e.g., '1Y', '3M', '10Y')
            frequency: Data frequency
            
        Returns:
            TimeSeriesData containing rate data
        """
        pass
    
    @abstractmethod
    def get_div_yield_history(self, asset_name: str,
                             start_date: datetime,
                             end_date: datetime,
                             frequency: str = 'D') -> TimeSeriesData:
        """
        Get historical dividend yields.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date of the history
            end_date: End date of the history
            frequency: Data frequency
            
        Returns:
            TimeSeriesData containing dividend yield data
        """
        pass
    
    def get_market_data_set(self, asset_name: str,
                           start_date: datetime,
                           end_date: datetime,
                           currency: str = 'USD',
                           frequency: str = 'D') -> MarketDataSet:
        """
        Get complete market data set for an asset.
        
        This is a convenience method that fetches all required data
        (spot, vol, rates, div yields) in one call.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date of the history
            end_date: End date of the history
            currency: Currency for rate curve
            frequency: Data frequency
            
        Returns:
            MarketDataSet containing all market data
        """
        spot_data = self.get_spot_history(asset_name, start_date, end_date, frequency)
        vol_data = self.get_vol_history(asset_name, start_date, end_date, frequency)
        rate_data = self.get_rate_history(currency, start_date, end_date, frequency=frequency)
        div_yield_data = self.get_div_yield_history(asset_name, start_date, end_date, frequency)
        
        return MarketDataSet(
            spot_data=spot_data,
            vol_data=vol_data,
            rate_data=rate_data,
            div_yield_data=div_yield_data,
            asset_name=asset_name,
            metadata={
                'source': self.source_name,
                'start_date': start_date,
                'end_date': end_date,
                'frequency': frequency
            }
        )
    
    def get_current_data(self, asset_name: str) -> MarketDataPoint:
        """
        Get current/latest market data point.
        
        Args:
            asset_name: Asset identifier
            
        Returns:
            MarketDataPoint with current data
        """
        # Default implementation: get latest from history
        end_date = datetime.now()
        start_date = datetime(end_date.year, end_date.month, end_date.day)
        
        spot_ts = self.get_spot_history(asset_name, start_date, end_date, 'D')
        vol_ts = self.get_vol_history(asset_name, start_date, end_date, 'D')
        rate_ts = self.get_rate_history('USD', start_date, end_date, frequency='D')
        div_ts = self.get_div_yield_history(asset_name, start_date, end_date, 'D')
        
        if len(spot_ts) == 0:
            raise ValueError(f"No current data available for {asset_name}")
        
        return MarketDataPoint(
            timestamp=spot_ts.data.index[-1],
            spot=spot_ts.data['spot'].iloc[-1],
            volatility=vol_ts.data['volatility'].iloc[-1] if len(vol_ts) > 0 else 0.2,
            rate=rate_ts.data['rate'].iloc[-1] if len(rate_ts) > 0 else 0.05,
            div_yield=div_ts.data['div_yield'].iloc[-1] if len(div_ts) > 0 else 0.0,
            asset_name=asset_name
        )
    
    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
    
    def validate_date_range(self, start_date: datetime, end_date: datetime):
        """
        Validate date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Raises:
            ValueError if dates are invalid
        """
        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")
        if end_date > datetime.now():
            raise ValueError(f"End date {end_date} is in the future")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source_name})"

