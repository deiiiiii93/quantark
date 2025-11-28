# Change: Add Fixed Income Backtest Support

## Why

The current backtest module is designed specifically for equity derivatives with delta-neutral hedging strategies. We need to extend it to support Fixed Income assets (bonds, bond futures) with DV01/convexity-based hedging, enabling comprehensive portfolio risk management across asset classes.

## What Changes

### Phase 1: Abstract Base Protocols
- **BREAKING** Create abstract base classes/protocols for backtest components that are asset-agnostic
- Create `BasePosition` and `BasePortfolio` protocols in portfolio module
- Create `BaseHedgeExecutor` and `BaseBacktestEngine` protocols in backtest module
- Refactor existing equity implementations to implement these protocols

### Phase 2: Fixed Income Backtest Implementation
- Implement `FIPosition` and `FIPortfolio` for Fixed Income assets
- Implement `FIHedgeExecutor` for bond futures hedging
- Implement `DV01NeutralStrategy` for DV01-based hedging
- Implement `ConvexityNeutralStrategy` for convexity-based hedging
- Implement `FIBacktestEngine` following the equity backtest architecture
- Create FI-specific visualizations and metrics

### Phase 3: Example and Documentation
- Create example for Fixed Income portfolio backtest with bond positions and bond futures hedging

## Impact

- Affected specs: backtest-protocols (new), fi-backtest (new), fi-portfolio (new)
- Affected code:
  - `portfolio/` → restructure into `equity/` and `fi/` subdirectories
  - `backtest/` → restructure into `equity/` and `fi/` subdirectories
  - `backtest/strategy/` → add FI-specific strategies
  - New: `portfolio/base.py` with base protocols
  - New: `portfolio/equity/` directory (moved from root)
  - New: `portfolio/fi/` directory for FI implementations
  - New: `backtest/base.py` with base protocols
  - New: `backtest/equity/` directory (moved from root)
  - New: `backtest/fi/` directory for FI implementations
  - New: `backtest/examples/fi_dv01_hedge.py`

