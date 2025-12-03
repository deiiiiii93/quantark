# QuantArk - Professional Financial Derivatives Pricing Library

A modular, professional-grade Python library for pricing and risk management of financial derivatives.

## Overview

QuantArk is designed with a clean, modular architecture that separates concerns across different components:

- **Products**: Define instrument specifications (options, swaps, etc.)
- **Processes**: Stochastic models (Black-Scholes-Merton, Heston, Local Vol, etc.)
- **Engines**: Pricing algorithms (Analytical, Monte Carlo, PDE, Quadrature)
- **Parameters**: Market data (spot prices, volatility surfaces, rate curves, dividends)
- **PriceEnv**: Unified pricing environment bundling all market data
- **RiskMeasures**: Greeks calculation (both analytical and numerical)

## Features

### Current Implementation

- **European Vanilla Options**: Full support for calls and puts
- **American Options**: Analytical and numerical methods (Barone-Adesi-Whaley, Longstaff-Schwartz)
- **Black-Scholes-Merton Model**: With continuous dividend yield
- **Analytical Pricing**: Closed-form Black-Scholes formula
- **Monte Carlo Engine**: Path-dependent pricing with variance reduction techniques
- **PDE Engine**: Finite difference methods for American options
- **Portfolio Value-at-Risk (VaR)**: Three calculation methods
  - Historical VaR (full revaluation under historical scenarios)
  - Parametric VaR (variance-covariance with Greeks/DV01)
  - Monte Carlo VaR (simulation-based with stress testing)
- **Bond Pricing**: Fixed rate bonds, FRNs, and bond options
- **Interest Rate Swaps**: Pricing and risk metrics (DV01)
- **Greeks Calculation**:
  - Analytical Greeks using closed-form formulas
  - Numerical Greeks using finite difference method (FDM)
  - Delta, Gamma, Vega, Theta, Rho, DV01
- **Robust Error Handling**: Professional exception hierarchy
- **Numerical Stability**: Careful boundary checking and validation

### Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Extensibility**: Easy to add new products, processes, and engines
3. **Type Safety**: Extensive use of dataclasses and type hints
4. **Validation**: Input validation at every level
5. **Professional Exception Handling**: Custom exception hierarchy for different error types

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantArk.git
cd QuantArk

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.riskmeasures import GreeksCalculator
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType

# Set up market data
spot = SpotQuote(spot=100.0)
vol = FlatVolSurface(volatility=0.20)  # 20% vol
rate = FlatRateCurve(rate=0.05)  # 5% risk-free rate
div = ContinuousDividendYield(div_yield=0.02)  # 2% dividend yield

pricing_env = PricingEnvironment(
    spot_quote=spot,
    vol_surface=vol,
    rate_curve=rate,
    div_yield=div
)

# Create a European call option
call_option = EuropeanVanillaOption(
    strike=100.0,
    maturity=1.0,  # 1 year
    option_type=OptionType.CALL
)

# Price the option
engine = BlackScholesEngine()
price = engine.price(call_option, pricing_env)
print(f"Call Price: ${price:.6f}")

# Calculate Greeks
greeks_calc = GreeksCalculator()
analytical_greeks = greeks_calc.calculate_analytical_greeks(
    call_option, pricing_env, price
)

print(f"Delta: {analytical_greeks['delta']:.6f}")
print(f"Gamma: {analytical_greeks['gamma']:.6f}")
print(f"Vega:  {analytical_greeks['vega']:.6f}")
print(f"Theta: {analytical_greeks['theta']:.6f} (per day)")
print(f"Rho:   {analytical_greeks['rho']:.6f}")

# Calculate portfolio VaR
from var import (
    ParametricVaREngine,
    HistoricalVaREngine,
    VaRConfig,
    EquityRiskFactorConfig,
)
from portfolio.equity.portfolio import EquityPortfolio

# Create a portfolio
portfolio = EquityPortfolio(
    positions={
        "AAPL": {"quantity": 100, "cost_basis": 150.0},
        "MSFT": {"quantity": 50, "cost_basis": 300.0}
    }
)

# Configure VaR calculation
var_config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    equity_factors=EquityRiskFactorConfig(
        include_spot=True,
        include_vol=True,
        include_rate=True
    )
)

# Calculate parametric VaR
parametric_engine = ParametricVaREngine(config=var_config)
parametric_result = parametric_engine.calculate_var(portfolio, historical_data)

print(f"\nParametric VaR (99%): ${parametric_result.var:.2f}")
print(f"CVaR: ${parametric_result.cvar:.2f}")

# Calculate historical VaR
historical_engine = HistoricalVaREngine(config=var_config)
historical_result = historical_engine.calculate_var(portfolio, historical_data)

print(f"Historical VaR (99%): ${historical_result.var:.2f}")
print(f"CVaR: ${historical_result.cvar:.2f}")
```

## Running the Demo

A comprehensive demonstration is provided:

```bash
python example/european_option_demo.py
```

The demo showcases:
1. European Call option pricing and Greeks
2. European Put option pricing and Greeks
3. Put-Call Parity verification
4. Comparison between analytical and numerical Greeks

## Project Structure

```
QuantArk/
├── asset/              # Asset classes
│   ├── equity/
│   │   ├── engine/     # Pricing engines
│   │   │   ├── analytical/
│   │   │   ├── mc/
│   │   │   ├── pde/
│   │   │   └── quad/
│   │   ├── param/      # Engine parameters
│   │   ├── process/    # Stochastic processes
│   │   │   ├── bsm/
│   │   │   ├── heston/
│   │   │   ├── localvol/
│   │   │   └── slv/
│   │   ├── product/    # Derivative products
│   │   │   └── option/
│   │   └── riskmeasures/  # Greeks calculation
│   ├── bond/          # Fixed income instruments
│   │   ├── engine/    # Bond pricing engines
│   │   ├── product/   # Bond products
│   │   └── riskmeasures/  # Bond risk measures
│   └── rate/          # Interest rate derivatives
│       ├── engine/    # IR pricing engines
│       ├── product/   # IR products
│       └── riskmeasures/  # IR risk measures
├── param/              # Market data parameters
│   ├── div/           # Dividend yields
│   ├── quote/         # Spot quotes
│   ├── rrf/           # Risk-free rates
│   └── vol/           # Volatility surfaces
├── priceenv/          # Pricing environment
├── var/               # Value-at-Risk (VaR) calculations
│   ├── engines/       # VaR calculation engines
│   │   ├── historical.py  # Historical VaR
│   │   ├── parametric.py  # Parametric VaR
│   │   └── monte_carlo.py # Monte Carlo VaR
│   ├── risk_factors/  # Risk factor models
│   │   ├── base.py
│   │   ├── equity_factors.py
│   │   └── fi_factors.py
│   ├── backtest/      # VaR backtesting framework
│   ├── base.py        # VaR base classes
│   └── config.py      # VaR configuration
├── portfolio/         # Portfolio management
│   ├── equity/        # Equity portfolios
│   └── fi/            # Fixed income portfolios
├── backtest/          # Hedging strategy backtesting
├── dynamicscenario/   # Multi-day scenario simulation
├── stresstest/        # Stress testing framework
├── util/              # Utilities
│   ├── enum/          # Enumerations
│   └── exceptions.py  # Exception hierarchy
├── example/           # Example scripts
└── test/              # Unit tests
```

## Exception Hierarchy

QuantArk uses a professional exception hierarchy:

- `QuantArkException`: Base exception
  - `ValidationError`: Invalid input parameters
  - `NumericalError`: Numerical instability/convergence issues
  - `MarketDataError`: Missing or invalid market data
  - `PricingError`: General pricing failures

## Numerical Stability

The Black-Scholes engine includes extensive checks for numerical stability:

- Input parameter validation (spot, strike, volatility, rates)
- Overflow protection in exponential calculations
- Boundary condition handling (near expiry, deep ITM/OTM)
- Sanity checks on computed prices vs intrinsic values
- Extreme parameter detection and rejection

## Roadmap

### Short-term
- [x] American options (analytical and numerical methods)
- [ ] Asian options
- [ ] Barrier options
- [x] Monte Carlo engine implementation
- [x] PDE engine implementation
- [x] Portfolio VaR calculations
- [x] Fixed income instruments (bonds, swaps)

### Medium-term
- [ ] Heston stochastic volatility model
- [ ] Local volatility model
- [ ] Credit derivatives
- [ ] Calibration framework
- [ ] XVA calculations
- [ ] Multi-asset derivatives
- [ ] Hybrid models

### Long-term
- [ ] Performance optimization (Cython, GPU)
- [ ] Real-time risk metrics
- [ ] Portfolio optimization
- [ ] Market data integration
- [ ] Cloud deployment support

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow the existing code structure and style
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Validate inputs and handle edge cases
5. Add professional error handling

## License

MIT License - see LICENSE file for details

## Authors

QuantArk Development Team

## Acknowledgments

- Black-Scholes-Merton model from Fischer Black, Myron Scholes, and Robert Merton
- Greeks formulas from standard derivatives textbooks
- Design patterns inspired by QuantLib and similar professional libraries

