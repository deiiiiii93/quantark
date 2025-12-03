# QuantArk - Project Overview

## Project Purpose
QuantArk is a **professional-grade Python library** for pricing and risk management of financial derivatives. It provides a modular, clean architecture for quantitative finance applications including:
- European and American option pricing
- Bond and interest rate swap pricing
- Portfolio Value-at-Risk (VaR) calculations
- Greeks calculation (analytical and numerical)
- Monte Carlo simulation
- PDE-based pricing methods
- Portfolio management and backtesting

## Target Users
- Quantitative analysts and researchers
- Risk management professionals
- Financial engineers
- Academic researchers in quantitative finance
- Fintech developers

## Key Features

### Options Pricing
- **European Vanilla Options**: Black-Scholes analytical pricing
- **American Options**: Barone-Adesi-Whaley and Longstaff-Schwartz methods
- **Path-dependent options**: Monte Carlo engine for Asian, barrier, lookback options
- **PDE Engine**: Finite difference methods for American options

### Fixed Income
- Fixed rate bonds
- Floating rate notes (FRNs)
- Bond options
- Interest rate swaps (IRS)
- DV01 and duration calculations

### Portfolio VaR
- **Parametric VaR**: Variance-covariance method using Greeks/DV01
- **Historical VaR**: Full portfolio revaluation under historical scenarios
- **Monte Carlo VaR**: Simulation-based with stress testing
- **Stressed VaR**: Basel III/IV compliant crisis period analysis
- **Risk Attribution**: Component, Marginal, Incremental, and Factor VaR

### Risk Measures
- **Greeks**: Delta, Gamma, Vega, Theta, Rho
- **Analytical**: Closed-form formulas
- **Numerical**: Finite difference methods
- **Bond Greeks**: DV01, duration, convexity

## Core Design Principles
1. **Modularity**: Each component is independent and composable
2. **Extensibility**: Easy to add new products, processes, and engines
3. **Type Safety**: Extensive use of dataclasses and type hints
4. **Validation**: Input validation at every level
5. **Numerical Stability**: Handle edge cases and boundary conditions
6. **Professional Error Handling**: Custom exception hierarchy
7. **Engine-Agnostic Products**: Products don't know their pricing method

## Exception Hierarchy
- `QuantArkException`: Base exception for all QuantArk errors
  - `ValidationError`: Invalid input parameters or data
  - `NumericalError`: Numerical instability or convergence failures
  - `MarketDataError`: Missing or invalid market data
  - `PricingError`: General pricing calculation failures

## Current Implementation Status
The project is **production-ready** for:
- European/American option pricing
- Bond and IRS pricing
- Portfolio VaR calculations (all three methods)
- Risk attribution and backtesting
- Fixed income instruments

## Development Approach
- Specification-driven development using OpenSpec
- Comprehensive test coverage (pytest framework)
- Professional documentation with examples
- Modular, layered architecture
- Separation of concerns across components

## Performance Considerations
- Uses vectorized NumPy operations for speed
- Optimized for large portfolios (1000+ positions)
- Efficient covariance matrix calculations
- Monte Carlo with variance reduction techniques

## Documentation
- README.md: Project overview and quick start
- var/README.md: Comprehensive VaR module documentation
- Inline docstrings: All public APIs documented
- Example scripts: Located in `example/` directory
- Technical docs: `docs/` directory (when available)
