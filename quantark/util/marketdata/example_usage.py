"""
Example usage of market data utilities.

Demonstrates:
1. Generating mock market data
2. Saving/loading data with Parquet storage
3. Converting to PricingEnvironment
4. Using data for option pricing
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantark.util.marketdata import (
    MockMarketDataAdapter,
    ParquetStorage,
    MarketDataConverter,
    create_backtest_pricing_envs
)
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.util.enum import OptionType


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def example_1_generate_mock_data():
    """Example 1: Generate synthetic market data."""
    print_section("EXAMPLE 1: Generate Mock Market Data")
    
    # Create mock adapter with seed for reproducibility
    adapter = MockMarketDataAdapter(seed=42)
    
    # Define date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    print(f"\nGenerating data for AAPL from {start_date.date()} to {end_date.date()}")
    
    # Fetch complete market data set
    dataset = adapter.get_market_data_set(
        asset_name='AAPL',
        start_date=start_date,
        end_date=end_date,
        frequency='D'  # Daily data
    )
    
    print(f"\nDataset: {dataset}")
    print(f"\nSpot data points: {len(dataset.spot_data)}")
    print(f"Vol data points: {len(dataset.vol_data)}")
    print(f"Rate data points: {len(dataset.rate_data)}")
    print(f"Div yield data points: {len(dataset.div_yield_data)}")
    
    # Show first few data points
    print("\nFirst 5 spot prices:")
    print(dataset.spot_data.data.head())
    
    print("\nFirst 5 volatilities:")
    print(dataset.vol_data.data.head())
    
    # Show statistics
    print("\nSpot price statistics:")
    print(dataset.spot_data.describe())
    
    return dataset


def example_2_save_load_data(dataset):
    """Example 2: Save and load data using Parquet storage."""
    print_section("EXAMPLE 2: Save and Load Data with Parquet")
    
    # Create storage instance
    storage = ParquetStorage()
    
    print(f"\nStorage: {storage}")
    
    # Save the dataset
    print("\nSaving dataset to Parquet files...")
    saved_paths = storage.save_market_data_set(dataset, overwrite=True)
    
    print(f"Saved {len(saved_paths)} files:")
    for path in saved_paths:
        print(f"  - {path}")
    
    # Get storage info
    print("\nStorage info:")
    info = storage.get_storage_info()
    print(f"  Base path: {info['base_path']}")
    print(f"  Number of assets: {info['num_assets']}")
    print(f"  Total files: {info['total_files']}")
    print(f"  Total size: {info['total_size_mb']:.2f} MB")
    
    # List files for AAPL
    print("\nFiles for AAPL:")
    files = storage.list_files('AAPL')
    for file_info in files:
        print(f"  {file_info['data_type']}: {file_info['size_bytes']} bytes")
    
    # Load data back
    print("\nLoading data from Parquet files...")
    start_date = dataset.spot_data.data.index.min()
    end_date = dataset.spot_data.data.index.max()
    
    loaded_dataset = storage.load_market_data_set(
        asset_name='AAPL',
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"Loaded dataset: {loaded_dataset}")
    print(f"Loaded {len(loaded_dataset.spot_data)} spot data points")
    
    # Verify data matches
    original_spot = dataset.spot_data.data['spot'].iloc[0]
    loaded_spot = loaded_dataset.spot_data.data['spot'].iloc[0]
    print(f"\nData verification:")
    print(f"  Original first spot: ${original_spot:.2f}")
    print(f"  Loaded first spot: ${loaded_spot:.2f}")
    print(f"  Match: {abs(original_spot - loaded_spot) < 1e-6}")
    
    return loaded_dataset


def example_3_convert_to_pricing_env(dataset):
    """Example 3: Convert market data to PricingEnvironment."""
    print_section("EXAMPLE 3: Convert to PricingEnvironment")
    
    # Get a specific date
    target_date = dataset.spot_data.data.index[100]  # 100th data point
    
    print(f"\nCreating PricingEnvironment for {target_date.date()}")
    
    # Create pricing environment at target date
    pricing_env = MarketDataConverter.create_pricing_env_at_date(
        dataset=dataset,
        target_date=target_date
    )
    
    print(f"\nPricingEnvironment:")
    print(f"  Spot: ${pricing_env.spot:.2f}")
    print(f"  Volatility: {pricing_env.get_vol(pricing_env.spot, 1.0):.2%}")
    print(f"  Rate: {pricing_env.get_rate(1.0):.2%}")
    print(f"  Div Yield: {pricing_env.get_div_yield(1.0):.2%}")
    
    # Create multiple pricing environments
    print("\nCreating pricing environments for all dates...")
    pe_df = create_backtest_pricing_envs(dataset, align_dates=True)
    
    print(f"\nCreated {len(pe_df)} pricing environments")
    print("\nFirst 5 pricing environments:")
    print(pe_df[['spot', 'volatility', 'rate', 'div_yield']].head())
    
    return pricing_env


def example_4_price_options(dataset):
    """Example 4: Price options using market data."""
    print_section("EXAMPLE 4: Price Options with Market Data")
    
    # Create pricing engine
    engine = BlackScholesEngine()
    
    # Create an ATM call option with 3 months maturity
    target_date = dataset.spot_data.data.index[50]
    pricing_env = MarketDataConverter.create_pricing_env_at_date(dataset, target_date)
    
    strike = pricing_env.spot  # ATM
    maturity = 0.25  # 3 months
    
    call_option = EuropeanVanillaOption(
        strike=strike,
        option_type=OptionType.CALL,
        maturity=maturity
    )
    
    print(f"\nOption details:")
    print(f"  Type: Call")
    print(f"  Strike: ${strike:.2f}")
    print(f"  Maturity: {maturity} years")
    print(f"\nMarket data at {target_date.date()}:")
    print(f"  Spot: ${pricing_env.spot:.2f}")
    print(f"  Vol: {pricing_env.get_vol(strike, maturity):.2%}")
    print(f"  Rate: {pricing_env.get_rate(maturity):.2%}")
    
    # Price the option
    price = engine.price(call_option, pricing_env)
    
    print(f"\nOption price: ${price:.4f}")
    
    # Price options at multiple dates
    print("\nPricing option at multiple dates...")
    pe_df = create_backtest_pricing_envs(dataset, align_dates=True)
    
    prices = []
    for pricing_env in pe_df['pricing_env'].iloc[:10]:  # First 10 dates
        # Adjust strike to be ATM at each date
        atm_call = EuropeanVanillaOption(
            strike=pricing_env.spot,
            option_type=OptionType.CALL,
            maturity=maturity
        )
        price = engine.price(atm_call, pricing_env)
        prices.append(price)
    
    print(f"\nPriced {len(prices)} ATM calls")
    print(f"Price range: ${min(prices):.4f} - ${max(prices):.4f}")
    print(f"Average price: ${sum(prices) / len(prices):.4f}")


def example_5_multiple_assets():
    """Example 5: Generate data for multiple assets."""
    print_section("EXAMPLE 5: Multiple Assets")
    
    # Create adapter
    adapter = MockMarketDataAdapter(seed=123)
    
    # Configure different assets with different characteristics
    adapter.set_asset_config('AAPL', {
        'initial_spot': 150.0,
        'initial_vol': 0.30,
        'drift': 0.10,
        'vol_of_vol': 0.4,
        'jump_intensity': 2.0
    })
    
    adapter.set_asset_config('SPX', {
        'initial_spot': 4500.0,
        'initial_vol': 0.18,
        'drift': 0.08,
        'vol_of_vol': 0.3,
        'jump_intensity': 1.0
    })
    
    # Generate data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 1)
    
    storage = ParquetStorage()
    
    for asset in ['AAPL', 'SPX']:
        print(f"\nGenerating data for {asset}...")
        dataset = adapter.get_market_data_set(
            asset_name=asset,
            start_date=start_date,
            end_date=end_date,
            frequency='D'
        )
        
        print(f"  Generated {len(dataset.spot_data)} points")
        print(f"  Initial spot: ${dataset.spot_data.data['spot'].iloc[0]:.2f}")
        print(f"  Final spot: ${dataset.spot_data.data['spot'].iloc[-1]:.2f}")
        print(f"  Return: {(dataset.spot_data.data['spot'].iloc[-1] / dataset.spot_data.data['spot'].iloc[0] - 1) * 100:.2f}%")
        
        # Save to storage
        storage.save_market_data_set(dataset, overwrite=True)
    
    # Show storage info
    print(f"\n{storage}")
    print(f"\nAssets in storage: {storage.list_assets()}")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  QUANTARK - Market Data Utilities Examples".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    try:
        # Example 1: Generate mock data
        dataset = example_1_generate_mock_data()
        
        # Example 2: Save and load data
        loaded_dataset = example_2_save_load_data(dataset)
        
        # Example 3: Convert to PricingEnvironment
        pricing_env = example_3_convert_to_pricing_env(loaded_dataset)
        
        # Example 4: Price options
        example_4_price_options(loaded_dataset)
        
        # Example 5: Multiple assets
        example_5_multiple_assets()
        
        print_section("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("\nKey takeaways:")
        print("  1. MockMarketDataAdapter generates realistic synthetic data")
        print("  2. ParquetStorage provides efficient data persistence")
        print("  3. MarketDataConverter integrates with existing pricing framework")
        print("  4. Full workflow: generate → store → load → price")
        print("\nData is stored in: util/marketdata/data/")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

