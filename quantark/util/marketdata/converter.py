"""
Converter utilities between market data and PricingEnvironment.

Provides bidirectional conversion between raw market data formats
and the PricingEnvironment objects used by the pricing engines.
"""
from datetime import datetime
from typing import Optional, Iterator, Tuple
import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from util.marketdata.models import MarketDataPoint, TimeSeriesData, MarketDataSet
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield, NoDividend
from priceenv import PricingEnvironment
from util.calendar import DayCountConvention
from util.exceptions import ValidationError


class MarketDataConverter:
    """
    Converter between market data formats and PricingEnvironment.
    
    Provides static methods for converting between:
    - MarketDataPoint <-> PricingEnvironment
    - TimeSeriesData -> Iterator[PricingEnvironment]
    - MarketDataSet -> Iterator[PricingEnvironment]
    """
    
    @staticmethod
    def to_pricing_environment(data_point: MarketDataPoint,
                               day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS,
                               bus_days_in_year: int = 252) -> PricingEnvironment:
        """
        Convert MarketDataPoint to PricingEnvironment.
        
        Args:
            data_point: Market data point
            day_count_convention: Day count convention
            bus_days_in_year: Business days per year
            
        Returns:
            PricingEnvironment object
        """
        # Create spot quote
        spot_quote = SpotQuote(
            spot=data_point.spot,
            timestamp=data_point.timestamp,
            asset_name=data_point.asset_name
        )
        
        # Create flat vol surface
        vol_surface = FlatVolSurface(volatility=data_point.volatility)
        
        # Create flat rate curve
        rate_curve = FlatRateCurve(rate=data_point.rate)
        
        # Create dividend yield
        if data_point.div_yield > 0:
            div_yield = ContinuousDividendYield(div_yield=data_point.div_yield)
        else:
            div_yield = NoDividend()
        
        # Create pricing environment
        return PricingEnvironment(
            spot_quote=spot_quote,
            vol_surface=vol_surface,
            rate_curve=rate_curve,
            div_yield=div_yield,
            valuation_date=data_point.timestamp,
            day_count_convention=day_count_convention,
            bus_days_in_year=bus_days_in_year
        )
    
    @staticmethod
    def from_pricing_environment(pricing_env: PricingEnvironment,
                                asset_name: Optional[str] = None) -> MarketDataPoint:
        """
        Extract market data from PricingEnvironment.
        
        Args:
            pricing_env: PricingEnvironment object
            asset_name: Override asset name
            
        Returns:
            MarketDataPoint object
        """
        # Extract data
        spot = pricing_env.spot
        volatility = pricing_env.get_vol(spot, 1.0)  # Get ATM 1Y vol
        rate = pricing_env.get_rate(1.0)  # Get 1Y rate
        div_yield = pricing_env.get_div_yield(1.0)  # Get 1Y div yield
        
        # Use asset name from spot quote if not provided
        if asset_name is None:
            asset_name = pricing_env.spot_quote.asset_name
        
        return MarketDataPoint(
            timestamp=pricing_env.valuation_date,
            spot=spot,
            volatility=volatility,
            rate=rate,
            div_yield=div_yield,
            asset_name=asset_name
        )
    
    @staticmethod
    def time_series_to_pricing_envs(spot_data: TimeSeriesData,
                                    vol_data: TimeSeriesData,
                                    rate_data: TimeSeriesData,
                                    div_yield_data: Optional[TimeSeriesData] = None,
                                    asset_name: Optional[str] = None,
                                    day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS,
                                    bus_days_in_year: int = 252) -> Iterator[Tuple[datetime, PricingEnvironment]]:
        """
        Convert aligned time series to iterator of PricingEnvironments.
        
        Yields one PricingEnvironment per timestamp.
        
        Args:
            spot_data: Spot price time series
            vol_data: Volatility time series
            rate_data: Interest rate time series
            div_yield_data: Dividend yield time series (optional)
            asset_name: Asset identifier
            day_count_convention: Day count convention
            bus_days_in_year: Business days per year
            
        Yields:
            Tuple of (timestamp, PricingEnvironment)
        """
        # Ensure data is aligned (same dates)
        common_dates = spot_data.data.index.intersection(vol_data.data.index)
        common_dates = common_dates.intersection(rate_data.data.index)
        
        if div_yield_data is not None:
            common_dates = common_dates.intersection(div_yield_data.data.index)
        
        if len(common_dates) == 0:
            raise ValidationError("No common dates found across time series")
        
        # Use first asset name found if not provided
        if asset_name is None:
            asset_name = spot_data.asset_name or vol_data.asset_name or rate_data.asset_name
        
        # Iterate over common dates
        for date in common_dates:
            # Extract values at this date
            spot = spot_data.data.loc[date, 'spot']
            vol = vol_data.data.loc[date, 'volatility']
            rate = rate_data.data.loc[date, 'rate']
            
            if div_yield_data is not None:
                div_yield = div_yield_data.data.loc[date, 'div_yield']
            else:
                div_yield = 0.0
            
            # Create market data point
            data_point = MarketDataPoint(
                timestamp=date,
                spot=spot,
                volatility=vol,
                rate=rate,
                div_yield=div_yield,
                asset_name=asset_name
            )
            
            # Convert to pricing environment
            pricing_env = MarketDataConverter.to_pricing_environment(
                data_point, day_count_convention, bus_days_in_year
            )
            
            yield date, pricing_env
    
    @staticmethod
    def market_data_set_to_pricing_envs(dataset: MarketDataSet,
                                       day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS,
                                       bus_days_in_year: int = 252) -> Iterator[Tuple[datetime, PricingEnvironment]]:
        """
        Convert MarketDataSet to iterator of PricingEnvironments.
        
        Args:
            dataset: MarketDataSet object
            day_count_convention: Day count convention
            bus_days_in_year: Business days per year
            
        Yields:
            Tuple of (timestamp, PricingEnvironment)
        """
        return MarketDataConverter.time_series_to_pricing_envs(
            spot_data=dataset.spot_data,
            vol_data=dataset.vol_data,
            rate_data=dataset.rate_data,
            div_yield_data=dataset.div_yield_data,
            asset_name=dataset.asset_name,
            day_count_convention=day_count_convention,
            bus_days_in_year=bus_days_in_year
        )
    
    @staticmethod
    def create_pricing_env_at_date(dataset: MarketDataSet,
                                   target_date: datetime,
                                   day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS,
                                   bus_days_in_year: int = 252) -> Optional[PricingEnvironment]:
        """
        Create PricingEnvironment for a specific date from MarketDataSet.
        
        Args:
            dataset: MarketDataSet object
            target_date: Target date
            day_count_convention: Day count convention
            bus_days_in_year: Business days per year
            
        Returns:
            PricingEnvironment at target date, or None if date not found
        """
        # Get data at target date
        spot_dict = dataset.spot_data.get_at_date(target_date)
        vol_dict = dataset.vol_data.get_at_date(target_date)
        rate_dict = dataset.rate_data.get_at_date(target_date)
        
        if spot_dict is None or vol_dict is None or rate_dict is None:
            return None
        
        # Get div yield if available
        div_yield = 0.0
        if dataset.div_yield_data is not None:
            div_dict = dataset.div_yield_data.get_at_date(target_date)
            if div_dict is not None:
                div_yield = div_dict['div_yield']
        
        # Create market data point
        data_point = MarketDataPoint(
            timestamp=spot_dict['timestamp'],
            spot=spot_dict['spot'],
            volatility=vol_dict['volatility'],
            rate=rate_dict['rate'],
            div_yield=div_yield,
            asset_name=dataset.asset_name
        )
        
        return MarketDataConverter.to_pricing_environment(
            data_point, day_count_convention, bus_days_in_year
        )


def create_backtest_pricing_envs(dataset: MarketDataSet,
                                 align_dates: bool = True,
                                 day_count_convention: DayCountConvention = DayCountConvention.CALENDAR_DAYS,
                                 bus_days_in_year: int = 252) -> pd.DataFrame:
    """
    Create a DataFrame of PricingEnvironments for backtesting.
    
    This is a convenience function that creates a pandas DataFrame
    where each row represents a pricing environment at a specific date.
    
    Args:
        dataset: MarketDataSet object
        align_dates: Whether to align dates across all time series
        day_count_convention: Day count convention
        bus_days_in_year: Business days per year
        
    Returns:
        DataFrame with columns [spot, volatility, rate, div_yield] and PricingEnvironment objects
    """
    if align_dates:
        dataset = dataset.align_dates()
    
    # Create list of pricing environments
    pricing_envs = []
    dates = []
    
    for date, pricing_env in MarketDataConverter.market_data_set_to_pricing_envs(
        dataset, day_count_convention, bus_days_in_year
    ):
        pricing_envs.append(pricing_env)
        dates.append(date)
    
    # Create DataFrame with market data
    df = pd.DataFrame({
        'spot': [pe.spot for pe in pricing_envs],
        'volatility': [pe.get_vol(pe.spot, 1.0) for pe in pricing_envs],
        'rate': [pe.get_rate(1.0) for pe in pricing_envs],
        'div_yield': [pe.get_div_yield(1.0) for pe in pricing_envs],
        'pricing_env': pricing_envs
    }, index=dates)
    
    return df

