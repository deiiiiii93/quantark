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

### Trade value (opt-in)

`EquityPortfolio(..., trade_value=True)` values every `EquityPosition` at its **full trade value**: the product ×
the signed quantity plus the position's cash legs (absolute, in the holder's direction). With the flag on:

- `get_portfolio_value` sums `get_trade_value`; `get_portfolio_greeks` / `get_greeks_by_underlying` sum the
  `get_trade_risk(...).total` bump Greeks for every position (no analytical Greeks; `use_analytical` is ignored)
  over `market_value, delta, gamma, vega, theta, rho, dividend_rho`, with `market_value` equal to the value;
- `get_portfolio_pnl` sums `get_trade_pnl` = trade value − `EquityPosition.entry_trade_value`. A position with cash
  legs must carry `entry_trade_value` (the signed trade total at entry, legs included) or trade PnL raises
  `ValidationError`; a position without legs uses `entry_price × quantity`;
- `position_value`, `position_greeks` and `position_pnl` value one position in the portfolio's mode — every
  equity consumer (VaR, SIMM via the portfolio adapter, static stress, dynamic scenarios, backtests) goes through
  them, so all of them follow the one flag.

`EquitySwapPosition` (TRS) keeps its own path in both modes. The flag defaults to `False`, the product path,
bitwise as before.

**Cash-leg times count from the trade's initial date.** A leg's payment and accrual times are years from the
contract's `initial_date` (from the valuation date when the product has none). Every trade valuation
(`get_trade_value`, `get_trade_value_breakdown`, `get_trade_greeks`, `get_trade_risk`, so the portfolio, static
stress, VaR and SIMM) first rolls the legs to the valuation date by whole days / 365, exactly as the multi-day runs'
`LegCashBook` does: a flow paid before the valuation date is history, a flow due that day counts at t = 0, and each
accrual period keeps its full amount. Example: struck 2026-10-01, valued 2026-10-10, a 5 % leg paying at
2027-10-01 pays 1.0 × 5 % of its base, discounted over the 356 days to payment. `EquityPosition.on_valuation_clock(env)`
returns the rolled position; `cash_legs_clock=LegClock.VALUATION` marks legs already on the valuation date (the
lifecycle manager and the replay engine set it on the legs they roll), so they are not rolled twice.

**A numeric note counts from its initial date too.** A Snowball / Phoenix / KO-reset with numeric observation times
and an `initial_date` is aged to the valuation date before every position valuation (product and trade mode) by
`elapsed − accrual_config.accrued_offset`, with the lifecycle tracker's own steps. A fresh booking prices like the
tracker's copy for that day; a copy already aged that far (`accrued_offset` equal to the elapsed time) is left alone.

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
pytest test/test_asset/equity/product_other/test_portfolio.py -v
```

Run the demo:
```bash
python example/portfolio/portfolio_demo.py
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

