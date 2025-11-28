# Project Context

## Purpose
QuantArk is a professional-grade Python library for pricing and risk management of financial derivatives. It provides modular, extensible components for pricing various derivative instruments (options, swaps, bonds) using multiple numerical methods (analytical, Monte Carlo, PDE, quadrature), calculating risk measures (Greeks), and running strategy backtests.

## Tech Stack
- **Language**: Python 3.10+
- **Numerical Computing**: NumPy, SciPy, Pandas
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Storage**: Parquet (via PyArrow), Excel (via openpyxl)
- **Testing**: pytest

## Project Conventions

### Code Style
- **Formatting**: Follow PEP 8 guidelines
- **Naming**:
  - Classes: PascalCase (e.g., `EuropeanVanillaOption`, `BlackScholesEngine`)
  - Functions/methods: snake_case (e.g., `calculate_greeks`, `get_spot_price`)
  - Constants: UPPER_SNAKE_CASE
  - Private methods: single underscore prefix (e.g., `_validate_inputs`)
- **Type hints**: Use dataclasses and type hints extensively
- **Docstrings**: Triple-quoted docstrings for all public classes and methods

### Architecture Patterns
- **Separation of Concerns**:
  - `product/`: Instrument specifications (what to price)
  - `process/`: Stochastic models (how prices evolve)
  - `engine/`: Pricing algorithms (how to compute prices)
  - `param/`: Market data (inputs for pricing)
  - `priceenv/`: Unified pricing environment bundling market data
  - `riskmeasures/`: Greeks and sensitivity calculations
- **Base Class Pattern**: Abstract base classes define interfaces (e.g., `BaseEngine`, `BaseEquityOption`)
- **Engine-Agnostic Products**: Products don't know their pricing method; engines handle that
- **Immutable Market Data**: Parameter objects should be treated as immutable

### Module Structure
```
asset/
├── equity/           # Equity derivatives
├── bond/             # Fixed income instruments
├── rate/             # Interest rate derivatives
├── credit/           # Credit derivatives
└── currency/         # FX derivatives
```

Each asset class follows the same internal structure:
- `product/`: Instrument definitions
- `engine/`: Pricing engines (analytical/, mc/, pde/, quad/)
- `process/`: Stochastic processes
- `riskmeasures/`: Risk calculations

### Testing Strategy
- **Framework**: pytest
- **Location**: `test/` directory
- **Naming**: `test_<module>.py`
- **Coverage**: Tests for all public APIs and edge cases
- **Validation**: Input validation tests and boundary condition checks

### Git Workflow
- **Main branch**: `main` - stable, production-ready code
- **Feature branches**: Create branches for new features
- **Commit messages**: Clear, descriptive messages

## Domain Context
This is a **quantitative finance** library. Key concepts:
- **Derivatives**: Financial instruments whose value derives from underlying assets
- **Greeks**: Sensitivities of option prices (Delta, Gamma, Vega, Theta, Rho)
- **Black-Scholes-Merton (BSM)**: Foundational option pricing model
- **Monte Carlo**: Simulation-based pricing for path-dependent options
- **PDE**: Partial differential equation methods for option pricing
- **Hedging**: Managing portfolio risk through offsetting positions
- **Mark-to-Market**: Valuing positions at current market prices
- **DV01**: Dollar value of a basis point (bond risk measure)

## Important Constraints
- **Numerical Stability**: All engines must handle edge cases (near-expiry, deep ITM/OTM)
- **Input Validation**: Validate at every level (negative prices, invalid dates, etc.)
- **Performance**: Pricing calculations should be efficient (vectorized NumPy operations)
- **Backward Compatibility**: New features should not break existing APIs
- **No External Market Data**: Library provides pricing logic, not data feeds

## External Dependencies
- **NumPy/SciPy**: Core numerical computations
- **Pandas**: Time series data and DataFrames for results
- **Matplotlib/Plotly**: Visualization of results and reports
- **PyArrow**: Parquet file I/O for results storage

## Exception Hierarchy
```
QuantArkException (base)
├── ValidationError    # Invalid input parameters
├── NumericalError     # Numerical instability/convergence
├── MarketDataError    # Missing/invalid market data
└── PricingError       # General pricing failures
```
