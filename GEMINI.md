# QuantArk - Project Context

## Project Overview

QuantArk is a professional-grade, modular Python library for pricing and risk management of financial derivatives. It is designed with a clean architecture separating concerns between instrument specifications, stochastic models, pricing engines, and market data.

**Key Features:**
*   **Derivatives Pricing:** European/American options, Bonds, Swaps, FRNs.
*   **Pricing Engines:** Analytical (BSM), Monte Carlo, PDE, Quadrature.
*   **Risk Management:** Greeks (Delta, Gamma, Vega, etc.), Value-at-Risk (VaR), Standard Initial Margin Model (SIMM).
*   **Simulation:** Backtesting framework for hedging strategies, Dynamic Scenario generation, Stress Testing.

## Directory Structure & Key Modules

The codebase follows a modular structure where core financial logic is separated into distinct packages at the root level.

*   **`asset/`**: Core definitions for financial instruments and pricing.
    *   `equity/`, `bond/`, `rate/`: Asset class specific implementations.
    *   `product/`: Instrument specifications (e.g., `EuropeanVanillaOption`, `FixedBond`).
    *   `engine/`: Pricing algorithms (e.g., `BlackScholesEngine`, `MonteCarloEngine`).
    *   `process/`: Stochastic processes (e.g., `GeometricBrownianMotion`).
    *   `riskmeasures/`: Calculators for Greeks and sensitivities.
*   **`param/`**: Market data parameters.
    *   `quote/` (Spot), `vol/` (Volatility Surfaces), `rrf/` (Rates), `div/` (Dividends).
*   **`priceenv/`**: `PricingEnvironment` class bundling all market data for pricing.
*   **`portfolio/`**: Portfolio management for Equity and Fixed Income.
*   **`backtest/`**: Framework for backtesting hedging strategies (Delta-neutral, DV01-neutral).
*   **`var/`**: Value-at-Risk engines (Historical, Parametric, Monte Carlo).
*   **`simm/`**: Standard Initial Margin Model (ISDA SIMM) implementation.
*   **`stresstest/`**: Framework for stress testing portfolios under extreme scenarios.
*   **`dynamicscenario/`**: Multi-day market path simulation.
*   **`util/`**: Shared utilities, enumerations (`OptionType`, `EngineType`), exception hierarchy, and **numerical utilities**.
*   **`example/`**: comprehensive demo scripts for all major features.
*   **`test/`**: Unit and integration tests using `pytest`.

## Architecture Patterns

*   **Separation of Concerns:**
    *   **Product:** "What" is being priced (e.g., Strike, Maturity).
    *   **Market Data:** "State" of the market (e.g., Spot=100, Vol=20%).
    *   **Engine:** "How" to price it (e.g., Analytical vs. Monte Carlo).
    *   **Process:** "How" the underlying moves (e.g., Geometric Brownian Motion).
*   **Type Safety:** Extensive use of Python `dataclasses` and type hints.
*   **Error Handling:** Custom exception hierarchy rooted in `QuantArkException`.
*   **Numerical Operations:** Always use `util/numerical/` for float comparisons, safe math, and formatting.

## Numerical Utilities (IMPORTANT)

**Always use `util/numerical/` for numerical operations.** Do NOT use raw float comparisons or hardcoded tolerances.

| Need | Module | Functions |
|------|--------|-----------|
| Float comparison | `comparison` | `is_zero()`, `is_close()`, `almost_equal()` |
| Safe math | `safe_math` | `safe_log()`, `safe_exp()`, `safe_sqrt()`, `safe_divide()` |
| Formatting | `formatting` | `format_currency()`, `format_percentage()`, `format_basis_points()` |
| Validation | `validation` | `validate_positive()`, `validate_probability()`, `is_valid_number()` |
| Constants | `constants` | `Tolerance.ZERO`, `Tolerance.PRECISION`, `FinancialConstants` |

**Example:**
```python
from util.numerical import is_zero, safe_log, format_currency, validate_positive

if is_zero(time_to_expiry):          # NOT: if T < 1e-10
    return intrinsic_value
log_m = safe_log(spot / strike)       # NOT: math.log(spot / strike)
print(format_currency(price))         # NOT: f"${price:.2f}"
vol = validate_positive(vol, "vol")   # Raises ValidationError if invalid
```

## Setup & Usage

**Installation:**
```bash
pip install -r requirements.txt
# Optional: Install in editable mode
# pip install -e .
```

**Running Demos:**
The `example/` directory contains ready-to-run scripts showcasing the library's capabilities.
```bash
python example/european_option_demo.py    # Basic Option Pricing
python example/parametric_var_demo.py     # VaR Calculation
python example/fixed_bond_demo.py         # Bond Pricing
python example/backtest_demo.py           # Hedging Strategy Backtest
```

## Development & Testing

**Testing:**
The project uses `pytest` for testing.
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test/test_european_option.py

# Run with coverage
python -m pytest --cov=.
```

**Conventions:**
*   **Code Style:** Follow PEP 8.
*   **Documentation:** Add docstrings to all public methods and classes.
*   **New Features:** Follow the OpenSpec workflow (see `openspec/AGENTS.md`) for proposing and implementing significant changes.

## Key Files for Context
*   `README.md`: Main entry point and quick start.
*   `PROJECT_INDEX.md`: Detailed map of the project structure (highly recommended for navigation).
*   `requirements.txt`: Python dependencies.
*   `backtest/README.md`: Specific documentation for the backtesting module.
*   `test/`: Reference implementations for how to use the various engines and products.
