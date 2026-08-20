# Delta One Products

This module provides complete support for delta one products (stocks, indices, ETFs, and futures) with full term structure pricing capabilities.

## Products Implemented

### 1. SpotInstrument
- Supports: STOCK, INDEX, ETF
- Perpetual instruments (no maturity)
- Forward pricing using cost-of-carry: `F(t,T) = S(t) * exp((r - q) * T)`

### 2. Futures
- Full futures contract with basis handling
- Contract multiplier support
- Optional market price for mark-to-market valuation
- Theoretical pricing: `F(t,T) = S(t) * exp((r - q) * T) + basis(t) * exp(-λ * T)`
- Basis converges to zero at maturity

## Key Features

### Pricing Capabilities
- **Spot instruments**: Current spot value and forward pricing
- **Futures**: Theoretical pricing with basis OR mark-to-market from observed prices
- Full term structure support with rate and dividend yield curves

### Greeks Calculation
All delta one products provide:
- **Delta**: ~1.0 (tracks underlying directly)
- **Gamma**: 0 (linear payoff)
- **Vega**: 0 (no volatility exposure)
- **Theta**: Carry costs
- **Rho**: Rate sensitivity

### Portfolio Integration
Delta one products integrate seamlessly with the existing Portfolio system:
- Can be added as positions with quantity (long/short)
- Greeks aggregate across portfolio
- Support hedging analysis

## Engine: DeltaOneEngine

The `DeltaOneEngine` supports:
1. Theoretical pricing for all delta one products
2. Mark-to-market pricing for futures (when enabled)
3. Analytical Greeks calculation
4. Forward price calculation at any time horizon

## Usage Example

```python
from asset.equity.product.deltaone import SpotInstrument, Futures
from asset.equity.engine.analytical import DeltaOneEngine
from util.enum import DeltaOneType

# Create a stock
stock = SpotInstrument(
    underlying="AAPL",
    deltaone_type=DeltaOneType.STOCK
)

# Create a futures contract
futures = Futures(
    underlying="ES",
    multiplier=50.0,
    maturity=0.25,  # 3 months
    basis=2.5,
    market_price=4515.25  # Optional MTM price
)

# Create engines
engine_theoretical = DeltaOneEngine(use_market_price=False)
engine_mtm = DeltaOneEngine(use_market_price=True)

# Price products
stock_price = engine_theoretical.price(stock, pricing_env)
futures_theoretical = engine_theoretical.price(futures, pricing_env)
futures_mtm = engine_mtm.price(futures, pricing_env)

# Calculate Greeks
greeks = engine_theoretical.calculate_greeks(futures, pricing_env)
```

## Files Structure

```
deltaone/
├── __init__.py                    # Module exports
├── base_deltaone_product.py       # Abstract base class
├── spot_instrument.py             # Stock/Index/ETF implementation
├── futures.py                     # Futures contract implementation
└── README.md                      # This file
```

## For Backtesting

Delta one products are essential for backtesting because they enable:
1. **Delta hedging**: Short futures to hedge long option positions
2. **Portfolio rebalancing**: Adjust underlying exposure
3. **Cash management**: Futures provide leveraged exposure
4. **Mark-to-market tracking**: Monitor P&L from market-observed prices

The multiplier attribute on futures allows proper contract sizing, and the market_price attribute enables realistic backtesting with observed prices rather than just theoretical models.

