# Portfolio Management Module

A comprehensive portfolio management module for tracking positions, calculating valuations, and aggregating risk metrics across multiple assets and products.

## Overview

The portfolio module provides a complete solution for managing multi-asset portfolios with the following features:

- **Position Tracking**: Track individual positions with quantities, entry prices, timestamps, and position-specific pricing engines
- **Multi-Underlying Support**: Manage positions across multiple underlyings, each with its own pricing environment
- **Valuation & P&L**: Calculate portfolio-level market values and profit/loss
- **Risk Aggregation**: Compute aggregated Greeks (delta, gamma, vega, theta, rho) across all positions
- **Snapshots**: Capture point-in-time portfolio states for historical tracking
- **Export Functionality**: Export to Excel (multi-sheet) and Parquet formats

## Architecture

### Core Components

1. **Position** (`position.py`)
   - Individual position with product, quantity, entry details, and engine
   - Each position has its own pricing engine (supports different asset types)
   - Methods: price, market value, P&L, Greeks calculation

2. **Portfolio** (`portfolio.py`)
   - Container managing multiple positions
   - Separate pricing environment per underlying
   - Position management: add, remove, update
   - Aggregation: valuation, P&L, Greeks

3. **PortfolioSnapshot** (`portfolio_snapshot.py`)
   - Point-in-time snapshot of portfolio state
   - Captures positions, values, and Greeks at specific timestamp
   - Useful for backtesting and performance analysis

4. **PortfolioExporter** (`portfolio_storage.py`)
   - Export to Parquet (efficient columnar storage)
   - Export to Excel (multi-sheet with positions, summary, Greeks)
   - Load portfolio data from storage

## Usage

### Basic Example

```python
from portfolio import Portfolio, PortfolioExporter
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType
from datetime import datetime

# Setup pricing environment
pricing_env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0, asset_name="AAPL"),
    vol_surface=FlatVolSurface(volatility=0.20),
    rate_curve=FlatRateCurve(rate=0.05),
    div_yield=ContinuousDividendYield(div_yield=0.02),
    valuation_date=datetime(2024, 1, 1)
)

# Create portfolio
portfolio = Portfolio(
    portfolio_name="My Portfolio",
    pricing_environments={'AAPL': pricing_env},
    creation_date=datetime(2024, 1, 1)
)

# Add positions
engine = BlackScholesEngine()
call_option = EuropeanVanillaOption(
    strike=100.0,
    option_type=OptionType.CALL,
    maturity=1.0
)

position = portfolio.add_position(
    product=call_option,
    quantity=10,
    entry_price=5.0,
    underlying='AAPL',
    engine=engine
)

# Calculate portfolio value and P&L
total_value = portfolio.get_portfolio_value()
total_pnl = portfolio.get_portfolio_pnl()

# Calculate Greeks
greeks_calc = GreeksCalculator()
portfolio_greeks = portfolio.get_portfolio_greeks(greeks_calc)

# Export
exporter = PortfolioExporter()
exporter.export_to_excel(portfolio, greeks_calculator=greeks_calc)
```

## Key Design Decisions

### Engine per Position

Each position has its own pricing engine. This design allows:
- Different products with different pricing requirements in the same portfolio
- Mixing analytical and numerical pricing methods
- Future extensibility to other asset classes (fixed income, commodities, etc.)

### Pricing Environment per Underlying

Each underlying has its own pricing environment, enabling:
- Different market data (spot, vol, rates, dividends) per underlying
- Cross-asset portfolios
- Independent valuation date management

### Aggregated Greeks

Portfolio-level Greeks are computed by:
1. Calculating position-level Greeks using each position's engine
2. Scaling Greeks by position quantity
3. Summing across all positions

This provides accurate portfolio-level risk metrics while respecting position-specific pricing logic.

## Export Formats

### Excel Export Structure

The Excel export creates a workbook with multiple sheets:

1. **Positions**: All position details with current prices and P&L
2. **Summary**: Portfolio-level statistics and metrics
3. **Greeks_by_Position**: Position-level Greeks for each position
4. **Greeks_by_Underlying**: Aggregated Greeks per underlying

### Parquet Export

Parquet exports use efficient columnar storage with:
- Snappy compression
- Embedded metadata (portfolio name, timestamps, statistics)
- Fast read/write operations
- Suitable for large-scale backtesting

## Testing

Run the test suite:
```bash
pytest test/test_portfolio.py -v
```

Run the demo:
```bash
python example/portfolio_demo.py
```

## Integration with Backtest Module

This portfolio module serves as the foundation for the backtest implementation:
- **Position History**: Snapshots enable tracking portfolio evolution over time
- **Performance Attribution**: Greeks by underlying help analyze risk contributions
- **Export Infrastructure**: Parquet format suitable for historical data storage
- **Multi-Asset Support**: Ready for cross-asset backtesting scenarios

## Dependencies

- `pandas>=2.0.0`: DataFrame operations
- `pyarrow>=12.0.0`: Parquet file format
- `openpyxl>=3.0.0`: Excel export
- `pytest>=7.0.0`: Testing framework

All dependencies are listed in `requirements.txt`.

