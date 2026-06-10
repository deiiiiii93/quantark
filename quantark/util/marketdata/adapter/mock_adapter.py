"""
Mock adapter for generating synthetic market data.

This adapter uses the MockDataGenerator to create realistic
synthetic data on-demand. Useful for testing and backtesting
when real data is not available.
"""
from datetime import datetime
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quantark.util.marketdata.adapter.base_adapter import BaseMarketDataAdapter
from quantark.util.marketdata.generator.mock_generator import MockDataGenerator
from quantark.util.marketdata.models import TimeSeriesData, MarketDataPoint


class MockMarketDataAdapter(BaseMarketDataAdapter):
    """
    Mock adapter that generates synthetic market data.
    
    Supports multiple assets with different characteristics.
    All data is generated on-the-fly using stochastic models.
    """
    
    # Default configurations for different asset types
    DEFAULT_CONFIGS = {
        'equity': {
            'initial_spot': 100.0,
            'initial_vol': 0.25,
            'initial_rate': 0.05,
            'initial_div_yield': 0.02,
            'drift': 0.08,
            'vol_of_vol': 0.4,
            'jump_intensity': 2.0,  # 2 jumps per year on average
        },
        'index': {
            'initial_spot': 4000.0,
            'initial_vol': 0.18,
            'initial_rate': 0.05,
            'initial_div_yield': 0.018,
            'drift': 0.07,
            'vol_of_vol': 0.3,
            'jump_intensity': 1.0,
        },
        'commodity': {
            'initial_spot': 50.0,
            'initial_vol': 0.35,
            'initial_rate': 0.05,
            'initial_div_yield': 0.0,
            'drift': 0.0,
            'vol_of_vol': 0.5,
            'jump_intensity': 5.0,
        },
        'fx': {
            'initial_spot': 1.0,
            'initial_vol': 0.10,
            'initial_rate': 0.03,
            'initial_div_yield': 0.0,
            'drift': 0.0,
            'vol_of_vol': 0.2,
            'jump_intensity': 0.5,
        },
        'fixed_income': {
            'initial_spot': 100.0,  # Bond price (par)
            'initial_vol': 0.01,    # Low vol for bonds
            'initial_rate': 0.04,   # 4% initial rate
            'initial_div_yield': 0.0,
            'drift': 0.0,           # Rates mean-revert
            'vol_of_vol': 0.15,     # Rate volatility
            'jump_intensity': 0.5,  # Rare rate jumps
        }
    }
    
    def __init__(self, seed: Optional[int] = None, 
                 asset_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize mock adapter.
        
        Args:
            seed: Random seed for reproducibility
            asset_configs: Custom configurations for specific assets
                          Format: {asset_name: {param1: value1, ...}}
        """
        super().__init__("Mock")
        self.generator = MockDataGenerator(seed=seed)
        self.asset_configs = asset_configs or {}
    
    def _get_asset_config(self, asset_name: str) -> Dict[str, Any]:
        """
        Get configuration for an asset.
        
        First checks custom configs, then tries to match asset type,
        finally defaults to 'equity' configuration.
        
        Args:
            asset_name: Asset identifier
            
        Returns:
            Configuration dictionary
        """
        # Check custom configs
        if asset_name in self.asset_configs:
            return self.asset_configs[asset_name]
        
        # Try to infer type from asset name
        asset_lower = asset_name.lower()
        if any(x in asset_lower for x in ['spx', 'dji', 'ndx', 'index']):
            return self.DEFAULT_CONFIGS['index']
        elif any(x in asset_lower for x in ['eur', 'gbp', 'jpy', 'fx']):
            return self.DEFAULT_CONFIGS['fx']
        elif any(x in asset_lower for x in ['oil', 'gold', 'silver', 'commodity']):
            return self.DEFAULT_CONFIGS['commodity']
        elif any(x in asset_lower for x in ['bond', 'ust', 'treasury', 'fi', 'fixed_income', 'bund', 'gilt']):
            return self.DEFAULT_CONFIGS['fixed_income']
        else:
            # Default to equity
            return self.DEFAULT_CONFIGS['equity']
    
    def get_spot_history(self, asset_name: str,
                        start_date: datetime,
                        end_date: datetime,
                        frequency: str = 'D') -> TimeSeriesData:
        """
        Generate synthetic spot price history.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            TimeSeriesData with spot prices
        """
        self.validate_date_range(start_date, end_date)
        
        config = self._get_asset_config(asset_name)
        
        spot_ts, _, _, _ = self.generator.generate_market_data_series(
            start_date=start_date,
            end_date=end_date,
            asset_name=asset_name,
            frequency=frequency,
            **config
        )
        
        return spot_ts
    
    def get_vol_history(self, asset_name: str,
                       start_date: datetime,
                       end_date: datetime,
                       frequency: str = 'D') -> TimeSeriesData:
        """
        Generate synthetic volatility history.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            TimeSeriesData with volatility
        """
        self.validate_date_range(start_date, end_date)
        
        config = self._get_asset_config(asset_name)
        
        _, vol_ts, _, _ = self.generator.generate_market_data_series(
            start_date=start_date,
            end_date=end_date,
            asset_name=asset_name,
            frequency=frequency,
            **config
        )
        
        return vol_ts
    
    def get_rate_history(self, currency: str,
                        start_date: datetime,
                        end_date: datetime,
                        tenor: str = '1Y',
                        frequency: str = 'D') -> TimeSeriesData:
        """
        Generate synthetic interest rate history.
        
        Args:
            currency: Currency code
            start_date: Start date
            end_date: End date
            tenor: Rate tenor
            frequency: Data frequency
            
        Returns:
            TimeSeriesData with interest rates
        """
        self.validate_date_range(start_date, end_date)
        
        # Use default equity config for rates (can be customized)
        config = self.DEFAULT_CONFIGS['equity']
        
        _, _, rate_ts, _ = self.generator.generate_market_data_series(
            start_date=start_date,
            end_date=end_date,
            asset_name=f"{currency}_{tenor}",
            frequency=frequency,
            **config
        )
        
        return rate_ts
    
    def get_div_yield_history(self, asset_name: str,
                             start_date: datetime,
                             end_date: datetime,
                             frequency: str = 'D') -> TimeSeriesData:
        """
        Generate synthetic dividend yield history.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            
        Returns:
            TimeSeriesData with dividend yields
        """
        self.validate_date_range(start_date, end_date)
        
        config = self._get_asset_config(asset_name)
        
        _, _, _, div_ts = self.generator.generate_market_data_series(
            start_date=start_date,
            end_date=end_date,
            asset_name=asset_name,
            frequency=frequency,
            **config
        )
        
        return div_ts
    
    def set_asset_config(self, asset_name: str, config: Dict[str, Any]):
        """
        Set custom configuration for a specific asset.
        
        Args:
            asset_name: Asset identifier
            config: Configuration dictionary
        """
        self.asset_configs[asset_name] = config
    
    def reset_seed(self, seed: int):
        """
        Reset the random seed for reproducibility.
        
        Args:
            seed: New random seed
        """
        self.generator = MockDataGenerator(seed=seed)
        self.clear_cache()
    
    def __repr__(self) -> str:
        return f"MockMarketDataAdapter(assets={len(self.asset_configs)}, seed={self.generator.seed})"

