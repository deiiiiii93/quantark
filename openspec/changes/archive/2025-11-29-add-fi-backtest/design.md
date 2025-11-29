## Context

QuantArk has a mature backtest module for equity derivatives with delta-neutral hedging. The module includes:
- Portfolio and Position management tied to `BaseEquityProduct`
- `BacktestEngine` with `HedgeExecutor` for spot/futures hedging
- `DeltaNeutralStrategy` for delta-based hedging decisions
- Comprehensive metrics, visualization, and reporting

We need to extend this to Fixed Income (FI) assets while:
1. Maintaining backward compatibility for existing equity backtests
2. Reusing common infrastructure (state tracking, logging, visualization patterns)
3. Supporting FI-specific risk measures (DV01, convexity, modified duration)

## Goals / Non-Goals

### Goals
- Create asset-agnostic base protocols for portfolio and backtest components
- Implement FI-specific portfolio and position classes
- Implement FI-specific hedging strategies (DV01-neutral, convexity-neutral)
- Support bond futures as hedging instruments
- Enable comparison between hedged and unhedged FI portfolios
- Follow existing backtest architecture patterns

### Non-Goals
- Supporting other asset classes (credit, FX) in this change
- Real-time trading or order management
- External market data integration
- Interest rate swap hedging (future enhancement)

## Decisions

### Decision 1: Protocol-Based Architecture
**What**: Use Python Protocols (typing.Protocol) for base classes  
**Why**: Protocols enable structural subtyping without inheritance, allowing cleaner separation between equity and FI implementations  
**Alternatives**: 
- Abstract base classes (ABC) - More restrictive, requires explicit inheritance
- No protocols - Would duplicate code between asset classes

### Decision 2: Separate Directory Structure
**What**: Create `portfolio/fi/` and `backtest/fi/` directories for FI implementations  
**Why**: Keeps asset-specific code isolated, easier to maintain and extend  
**Alternatives**:
- Single file with all implementations - Would become unwieldy
- Per-asset module structure - Overkill for current scope

### Decision 3: Risk Measure Mapping
**What**: Map FI risk measures to hedging targets
- DV01 (dollar value of 01) → Primary hedging target (like delta for equity)
- Convexity → Secondary hedging target (like gamma for equity)
- Modified Duration → Informational metric

**Why**: DV01 is the most common FI hedge measure; convexity adds precision for larger rate moves

### Decision 4: Hedge Instrument
**What**: Use Bond Futures as primary hedging instrument  
**Why**: 
- Bond futures have defined DV01 per contract
- Liquid and commonly used for duration hedging
- Already implemented in `asset/bond/product/futures/bond_futures.py`

### Decision 5: Pricing Environment Adaptation
**What**: Create FI-specific pricing environment handling  
**Why**: FI products use rate curves differently than equity (discount curves vs. spot quotes)

## Component Architecture

```
portfolio/
├── __init__.py          # MODIFIED: Re-export from submodules with aliases
├── base.py              # NEW: BasePosition, BasePortfolio protocols
├── equity/              # NEW: Equity implementations (refactored from root)
│   ├── __init__.py
│   ├── position.py      # MOVED: EquityPosition (was portfolio/position.py)
│   └── portfolio.py     # MOVED: EquityPortfolio (was portfolio/portfolio.py)
└── fi/                  # NEW: Fixed Income implementations
    ├── __init__.py
    ├── position.py      # NEW: FIPosition
    └── portfolio.py     # NEW: FIPortfolio

backtest/
├── __init__.py          # MODIFIED: Re-export from submodules with aliases
├── base.py              # NEW: BaseBacktestEngine, BaseHedgeExecutor protocols
├── strategy/
│   ├── __init__.py
│   ├── base_strategy.py # MODIFIED: Add hedging_target abstraction
│   ├── delta_neutral_strategy.py
│   ├── dv01_neutral_strategy.py     # NEW
│   └── convexity_neutral_strategy.py # NEW
├── equity/              # NEW: Equity implementations (refactored from root)
│   ├── __init__.py
│   ├── engine.py        # MOVED: EquityBacktestEngine (was backtest/engine.py)
│   ├── hedge_executor.py # MOVED: EquityHedgeExecutor (was backtest/hedge_executor.py)
│   ├── config.py        # MOVED: EquityBacktestConfig (was backtest/config.py)
│   ├── state.py         # MOVED: (was backtest/state.py)
│   ├── results.py       # MOVED: (was backtest/results.py)
│   └── metrics.py       # MOVED: (was backtest/metrics.py)
└── fi/                  # NEW: Fixed Income implementations
    ├── __init__.py
    ├── engine.py        # NEW: FIBacktestEngine
    ├── hedge_executor.py # NEW: FIHedgeExecutor
    ├── config.py        # NEW: FIBacktestConfig
    ├── state.py         # NEW: FIBacktestState
    ├── results.py       # NEW: FIBacktestResults
    └── metrics.py       # NEW: FI-specific metrics
```

## Risk Measures Comparison

| Equity | Fixed Income | Description |
|--------|--------------|-------------|
| Delta | DV01 | First-order price sensitivity |
| Gamma | Convexity | Second-order price sensitivity |
| Vega | Rate Vega | Volatility sensitivity |
| Theta | Carry/Roll | Time decay |

## Migration Plan

1. **Create base protocols** (no breaking changes)
   - Add `portfolio/base.py` with `BasePosition`, `BasePortfolio`
   - Add `backtest/base.py` with `BaseHedgeExecutor`, `BaseBacktestEngine`

2. **Refactor existing code to use protocols** (backward compatible)
   - Update `Position` → `EquityPosition` (with alias for compatibility)
   - Update `Portfolio` → `EquityPortfolio` (with alias for compatibility)
   - Update backtest components similarly

3. **Implement FI components** (additive)
   - Implement FI portfolio classes
   - Implement FI backtest engine and strategies
   - Create example

## Risks / Trade-offs

- **Risk**: Refactoring existing code may introduce bugs
  → **Mitigation**: Keep existing APIs, use aliases, comprehensive tests

- **Risk**: FI market data may not be available in MockMarketDataAdapter
  → **Mitigation**: Extend mock adapter to support rate curves and bond prices

- **Trade-off**: Protocols add complexity but enable cleaner multi-asset support

## Open Questions

1. Should we support cross-asset hedging (e.g., equity volatility hedged with rates)?
   - **Proposed answer**: Not in this change, design for future extensibility

2. Should FI metrics use the same MetricsCalculator or a separate one?
   - **Proposed answer**: Create FI-specific metrics class that extends base functionality

3. How to handle CTD (Cheapest-to-Deliver) selection in backtest?
   - **Proposed answer**: Use pre-specified CTD or simple static selection initially

