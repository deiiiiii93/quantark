"""
Market data models for time series storage and manipulation.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
from quantark.util.exceptions import ValidationError


@dataclass
class MarketDataPoint:
    """
    Single timestamp snapshot of market data.
    
    Attributes:
        timestamp: Timestamp of the data point
        spot: Spot price
        volatility: Implied volatility (or realized vol)
        rate: Risk-free interest rate
        div_yield: Dividend yield
        asset_name: Name/identifier of the asset
        metadata: Additional metadata (optional)
    """
    timestamp: datetime
    spot: float
    volatility: float
    rate: float
    div_yield: float = 0.0
    asset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate market data point."""
        if self.spot <= 0:
            raise ValidationError(f"Spot price must be positive, got {self.spot}")
        if self.volatility <= 0:
            raise ValidationError(f"Volatility must be positive, got {self.volatility}")
        if abs(self.div_yield) > 0.20:
            raise ValidationError(f"Dividend yield magnitude seems unreasonably high: {self.div_yield}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'spot': self.spot,
            'volatility': self.volatility,
            'rate': self.rate,
            'div_yield': self.div_yield,
            'asset_name': self.asset_name,
            **self.metadata
        }


@dataclass
class OptionMarketData:
    """
    Option-specific market data including prices, Greeks, and bid/ask.
    
    Attributes:
        timestamp: Timestamp of the data
        strike: Strike price
        maturity: Time to maturity in years
        option_type: 'call' or 'put'
        price: Option price
        implied_vol: Implied volatility
        bid: Bid price (optional)
        ask: Ask price (optional)
        greeks: Dictionary of Greeks (delta, gamma, vega, theta, rho)
        asset_name: Name/identifier of the underlying asset
        metadata: Additional metadata
    """
    timestamp: datetime
    strike: float
    maturity: float
    option_type: str
    price: float
    implied_vol: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    greeks: Dict[str, float] = field(default_factory=dict)
    asset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate option market data."""
        if self.strike <= 0:
            raise ValidationError(f"Strike must be positive, got {self.strike}")
        if self.maturity < 0:
            raise ValidationError(f"Maturity must be non-negative, got {self.maturity}")
        if self.option_type.lower() not in ['call', 'put']:
            raise ValidationError(f"Option type must be 'call' or 'put', got {self.option_type}")
        if self.price < 0:
            raise ValidationError(f"Price must be non-negative, got {self.price}")
        if self.implied_vol <= 0:
            raise ValidationError(f"Implied vol must be positive, got {self.implied_vol}")
        if self.bid is not None and self.bid < 0:
            raise ValidationError(f"Bid must be non-negative, got {self.bid}")
        if self.ask is not None and self.ask < 0:
            raise ValidationError(f"Ask must be non-negative, got {self.ask}")
        if self.bid is not None and self.ask is not None and self.bid > self.ask:
            raise ValidationError(f"Bid {self.bid} cannot be greater than ask {self.ask}")
    
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask if available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.price
    
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread if available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


class TimeSeriesData:
    """
    Container for historical time series data with pandas-like operations.
    
    This class wraps a pandas DataFrame and provides convenient methods
    for working with market data time series.
    """
    
    def __init__(self, data: pd.DataFrame, asset_name: Optional[str] = None,
                 data_type: str = "market_data", metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize time series data.
        
        Args:
            data: DataFrame with DatetimeIndex and market data columns
            asset_name: Name/identifier of the asset
            data_type: Type of data (e.g., 'spot', 'vol', 'rates', 'market_data')
            metadata: Additional metadata
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValidationError("Data must have DatetimeIndex")
        
        self.data = data.sort_index()
        self.asset_name = asset_name
        self.data_type = data_type
        self.metadata = metadata or {}
    
    @classmethod
    def from_market_data_points(cls, points: List[MarketDataPoint], 
                                asset_name: Optional[str] = None) -> 'TimeSeriesData':
        """
        Create TimeSeriesData from list of MarketDataPoint objects.
        
        Args:
            points: List of MarketDataPoint objects
            asset_name: Override asset name
            
        Returns:
            TimeSeriesData object
        """
        if not points:
            raise ValidationError("Cannot create TimeSeriesData from empty list")
        
        data_dicts = [p.to_dict() for p in points]
        df = pd.DataFrame(data_dicts)
        df.set_index('timestamp', inplace=True)
        
        # Use asset name from points if not provided
        if asset_name is None and 'asset_name' in df.columns:
            asset_name = df['asset_name'].iloc[0] if not df['asset_name'].isna().all() else None
            df.drop(columns=['asset_name'], inplace=True, errors='ignore')
        
        return cls(df, asset_name=asset_name, data_type='market_data')
    
    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.data)
    
    def __getitem__(self, key):
        """Support indexing and slicing."""
        return self.data[key]
    
    def filter_by_date(self, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> 'TimeSeriesData':
        """
        Filter data by date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            New TimeSeriesData with filtered data
        """
        filtered_data = self.data.copy()
        
        if start_date is not None:
            filtered_data = filtered_data[filtered_data.index >= start_date]
        if end_date is not None:
            filtered_data = filtered_data[filtered_data.index <= end_date]
        
        return TimeSeriesData(filtered_data, self.asset_name, self.data_type, self.metadata.copy())
    
    def resample(self, freq: str, agg_method: str = 'last') -> 'TimeSeriesData':
        """
        Resample time series to different frequency.
        
        Args:
            freq: Frequency string (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly)
            agg_method: Aggregation method ('last', 'first', 'mean', 'median')
            
        Returns:
            New TimeSeriesData with resampled data
        """
        if agg_method == 'last':
            resampled = self.data.resample(freq).last()
        elif agg_method == 'first':
            resampled = self.data.resample(freq).first()
        elif agg_method == 'mean':
            resampled = self.data.resample(freq).mean()
        elif agg_method == 'median':
            resampled = self.data.resample(freq).median()
        else:
            raise ValidationError(f"Unknown aggregation method: {agg_method}")
        
        # Drop NaN rows
        resampled = resampled.dropna()
        
        return TimeSeriesData(resampled, self.asset_name, self.data_type, self.metadata.copy())
    
    def get_at_date(self, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Get data point closest to specified date.
        
        Args:
            date: Target date
            
        Returns:
            Dictionary of values at that date, or None if no data
        """
        if len(self.data) == 0:
            return None
        
        # Find nearest index
        idx = self.data.index.get_indexer([date], method='nearest')[0]
        if idx == -1:
            return None
        
        row = self.data.iloc[idx]
        result = row.to_dict()
        result['timestamp'] = self.data.index[idx]
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return underlying DataFrame."""
        return self.data.copy()
    
    def describe(self) -> pd.DataFrame:
        """Return statistical summary of the data."""
        return self.data.describe()
    
    def __repr__(self) -> str:
        date_range = f"{self.data.index.min()} to {self.data.index.max()}" if len(self.data) > 0 else "empty"
        return (f"TimeSeriesData(asset={self.asset_name}, type={self.data_type}, "
                f"points={len(self.data)}, range={date_range})")


@dataclass
class MarketDataSet:
    """
    Complete dataset bundle for backtesting.
    
    Combines multiple time series into a single container for
    convenience in backtesting scenarios.
    
    Attributes:
        spot_data: Spot price time series
        vol_data: Volatility time series
        rate_data: Interest rate time series
        div_yield_data: Dividend yield time series
        option_data: Option market data (optional)
        asset_name: Name/identifier of the asset
        metadata: Additional metadata
    """
    spot_data: TimeSeriesData
    vol_data: TimeSeriesData
    rate_data: TimeSeriesData
    div_yield_data: Optional[TimeSeriesData] = None
    option_data: Optional[TimeSeriesData] = None
    asset_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate market data set."""
        # Ensure all required data is present
        if self.spot_data is None:
            raise ValidationError("Spot data is required")
        if self.vol_data is None:
            raise ValidationError("Volatility data is required")
        if self.rate_data is None:
            raise ValidationError("Rate data is required")
        
        # Set asset name from data if not provided
        if self.asset_name is None:
            self.asset_name = (self.spot_data.asset_name or 
                             self.vol_data.asset_name or 
                             self.rate_data.asset_name)
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """Get the common date range across all data series."""
        start_dates = [self.spot_data.data.index.min(), 
                      self.vol_data.data.index.min(),
                      self.rate_data.data.index.min()]
        end_dates = [self.spot_data.data.index.max(),
                    self.vol_data.data.index.max(),
                    self.rate_data.data.index.max()]
        
        if self.div_yield_data is not None:
            start_dates.append(self.div_yield_data.data.index.min())
            end_dates.append(self.div_yield_data.data.index.max())
        
        return max(start_dates), min(end_dates)
    
    def align_dates(self) -> 'MarketDataSet':
        """
        Align all time series to common dates.
        
        Returns:
            New MarketDataSet with aligned data
        """
        # Get common date range
        start_date, end_date = self.get_date_range()
        
        # Filter all series
        aligned_spot = self.spot_data.filter_by_date(start_date, end_date)
        aligned_vol = self.vol_data.filter_by_date(start_date, end_date)
        aligned_rate = self.rate_data.filter_by_date(start_date, end_date)
        
        aligned_div = None
        if self.div_yield_data is not None:
            aligned_div = self.div_yield_data.filter_by_date(start_date, end_date)
        
        aligned_option = None
        if self.option_data is not None:
            aligned_option = self.option_data.filter_by_date(start_date, end_date)
        
        return MarketDataSet(
            spot_data=aligned_spot,
            vol_data=aligned_vol,
            rate_data=aligned_rate,
            div_yield_data=aligned_div,
            option_data=aligned_option,
            asset_name=self.asset_name,
            metadata=self.metadata.copy()
        )
    
    def __repr__(self) -> str:
        start_date, end_date = self.get_date_range()
        return (f"MarketDataSet(asset={self.asset_name}, "
                f"range={start_date.date()} to {end_date.date()})")

