# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Quick Commands

## Slash Commands

- `/pro-impl-report [path]`: Generate a professional implementation report in a
  Methods/Results/Discussion layout. If `path` is omitted, write to
  `IMPLEMENTATION_REPORT.md`. Template: `commands/pro-impl-report.md`.

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest test/test_european_option.py

# Run with verbose output
python -m pytest -v

# Run tests matching a keyword
python -m pytest -k "test_name_pattern"

# Run coverage
python -m pytest --cov=.
```

### Running Examples
```bash
# European option pricing demo
python example/european_option_demo.py

# American option pricing demo
python example/american_option_demo.py

# Monte Carlo demo
python example/european_mc_demo.py

# PDE pricing demo
python example/pde_pricing_demo.py

# VaR demo
python example/parametric_var_demo.py

# Stress test demo
python example/stress_test_demo.py

# Other examples are located in the example/ directory
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# The project uses a virtual environment named 'quantark'
```

<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

## Project Index

`PROJECT_INDEX.md` is a generated snapshot of the repo structure and entry points.

## Code Style Guidelines

### Imports and Dependencies
- Order: Standard library, third-party, local imports
- Local imports: Use relative imports for modules within the same package
- No wildcard imports: Explicitly import required classes and functions
- Dependencies: Core libraries are scipy>=1.10.0, numpy>=1.24.0, pandas>=2.0.0

### Naming Conventions
- Classes: PascalCase (`EuropeanVanillaOption`, `BaseEquityProduct`)
- Methods and Functions: snake_case (`get_payoff`, `get_maturity`)
- Variables: snake_case (`spot_price`, `strike_price`)
- Constants: UPPER_SNAKE_CASE (`MAX_ITERATIONS`)
- Private methods: Leading underscore (`_validate_inputs`)

### Type Hints and Documentation
- Type hints: Always use for function parameters and return values
- Optional types: Use `Optional[Type]` and `Union[Type1, Type2]`
- Docstrings: Google-style with Args, Returns, Raises sections
- Complex types: Use `from typing import Optional, Union, List, Dict`

### Error Handling
- Exception hierarchy: `QuantArkException` -> `ValidationError`, `NumericalError`, `MarketDataError`, `PricingError`
- Input validation: Validate at construction time with descriptive messages
- Numerical stability: Check for overflow or underflow, division by zero
- Market data: Validate missing or inconsistent data with `MarketDataError`

### Code Organization
- Data structures: Use `@dataclass` for simple data holders
- Abstract classes: Use `ABC` and `@abstractmethod` for interfaces
- Engine pattern: Two-level enum pattern for engine methods (`EngineType.ANALYTICAL(method)`)
- Module structure: Clear separation with `__init__.py` and `__all__` exports

### Testing
- Test files: `test/test_<module>.py` naming convention
- Test methods: `test_<functionality>` naming
- Coverage: Test both positive and negative cases, edge conditions

## Architecture Overview

QuantArk is a professional-grade financial derivatives pricing library with a modular, layered architecture.

### Core Design Pattern: Modular Component Architecture

The library separates concerns across independent, composable components:

1. Products (`asset/*/product/`) - Define instrument specifications (strike, maturity, type)
2. Processes (`asset/*/process/`) - Stochastic models (BSM, Heston, Local Vol)
3. Engines (`asset/*/engine/`) - Pricing algorithms (Analytical, PDE, Monte Carlo, Quadrature, Tree)
4. Parameters (`param/`) - Market data (spot, vol surface, rate curve, dividends)
5. PricingEnvironment (`priceenv/`) - Unified pricing environment bundling all market data
6. RiskMeasures (`asset/*/riskmeasures/`) - Greeks calculation (analytical and numerical)

### Engine Method Selection Pattern

For engines supporting multiple methods, use the two-level enum pattern:

**Enum Definition** (`util/enum/engine_enums.py`):
```python
class AmericanAnalyticalMethod(Enum):
    BS93 = "BS93"
    BS02 = "BS02"
    BAW = "BAW"

class EngineType(Enum):
    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    PDE = auto()
    QUADRATURE = auto()
    TREE = auto()

    def __call__(self, method=None):
        # Enables EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
        if method is not None:
            return (self, method)
        return self
```

**Engine Implementation**:
```python
from util.enum.engine_enums import AmericanAnalyticalMethod, EngineType

class AmericanOptionAnalyticalEngine(BaseEngine):
    DEFAULT_METHOD = AmericanAnalyticalMethod.BS93

    def __init__(self, params=None, method: Union[str, AmericanAnalyticalMethod, tuple] = None):
        if method is None:
            self.method = self.DEFAULT_METHOD
        elif isinstance(method, tuple):
            # EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
            engine_type, analytical_method = method
            self.method = analytical_method
        elif isinstance(method, AmericanAnalyticalMethod):
            self.method = method
        elif isinstance(method, str):
            self.method = AmericanAnalyticalMethod[method.upper()]
```

**Usage** (in order of preference):
```python
# Preferred: Two-level enum pattern
engine = AmericanOptionAnalyticalEngine(
    method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
)

# Alternative: Direct method enum
engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)

# Backward compatibility: String
engine = AmericanOptionAnalyticalEngine(method="BS93")
```

IMPORTANT: Always follow this two-level enum pattern (EngineType.ANALYTICAL(method)) for new engine implementations with multiple methods.

### Key Asset Classes

**Equity** (`asset/equity/`):
- Products: European/American options, barriers, one-touch options, delta-one products
- Engines: Analytical, PDE, Monte Carlo, Quadrature
- Risk: Delta, Gamma, Vega, Theta, Rho (both analytical and finite-difference methods)

**Fixed Income** (`asset/bond/`):
- Products: Fixed and floating rate bonds, swaps, bond options, forwards, futures, convertibles
- Engines: Analytical discount-based pricing, tree and PDE where applicable
- Risk: DV01, convexity, duration

**Rates** (`asset/rate/`):
- Interest rate derivatives

### Supporting Modules

**Portfolio** (`portfolio/`):
- Base portfolio classes with position tracking
- Equity and FI portfolio implementations
- Portfolio snapshot and storage functionality
- Greek aggregation at portfolio level

**Backtest** (`backtest/`):
- Framework for testing hedging strategies (delta-neutral, DV01-neutral)
- Transaction cost modeling (fixed, proportional, slippage, bid-ask)
- Logging and visualization (matplotlib, plotly)
- Performance metrics (Sharpe, drawdown, VaR, CVaR)
- Separate implementations for equity and fixed income

**Dynamic Scenario** (`dynamicscenario/`):
- Multi-day scenario simulation with day-by-day parameter evolution
- Path modeling (spot, vol, rate curves)
- Hedging strategy simulation with rebalancing
- Greeks and risk measure evolution tracking
- Both equity and FI support

**VaR** (`var/`):
- Parametric, historical, and Monte Carlo VaR engines
- Risk factor configuration and attribution
- Backtesting and reporting

**Stress Test** (`stresstest/`):
- Scenario construction and stress application
- Equity and FI stress engines
- Results aggregation and reporting

**SIMM** (`simm/`):
- Standard Initial Margin Model calibration and CRIF parsing
- Risk class engines and aggregation
- Reporting and result utilities

**Utilities** (`util/`):
- `exceptions.py`: Exception hierarchy (QuantArkException -> ValidationError, NumericalError, MarketDataError, PricingError)
- `enum/`: OptionType, ExerciseStyle, BarrierType, engine enums
- `calendar/`: Day count conventions
- `marketdata/`: Market data utilities
- `numerical/`: Numerical utilities (see below)

### Numerical Utilities (IMPORTANT - Always Use These)

The `util/numerical/` module provides standardized utilities for all numerical operations. Always use these instead of raw float comparisons or hardcoded tolerances.

**Module Structure:**
- `constants.py`: `Tolerance` (ZERO=1e-10, PRECISION=1e-6, etc.), `FinancialConstants`
- `comparison.py`: `is_zero()`, `is_close()`, `almost_equal()`, `is_positive()`, etc.
- `safe_math.py`: `safe_log()`, `safe_exp()`, `safe_sqrt()`, `safe_divide()`
- `formatting.py`: `format_currency()`, `format_percentage()`, `format_basis_points()`, `format_greeks()`
- `validation.py`: `validate_positive()`, `validate_probability()`, `is_valid_number()`

**Usage Examples:**
```python
from util.numerical import (
    is_zero, is_close, Tolerance,           # Comparison
    safe_log, safe_exp, safe_sqrt,          # Safe math
    format_currency, format_percentage,      # Formatting
    validate_positive, is_valid_number       # Validation
)

# Float comparison (NOT: if T < 1e-10)
if is_zero(time_to_expiry):
    return intrinsic_value

# Safe math (NOT: math.log(x) which can fail)
log_moneyness = safe_log(spot / strike)

# Formatting (NOT: f"${value:.2f}")
print(format_currency(option_price))  # $12.50

# Validation
vol = validate_positive(volatility, "volatility")
```

### PricingEnvironment Structure

`PricingEnvironment` is the central data container that bundles all market parameters:
- Required: `rate_curve`, `valuation_date`
- Optional: `spot_quote`, `vol_surface`, `div_yield`
- Defaults: `day_count_convention=CALENDAR_DAYS`, `bus_days_in_year=252`
- Convenience methods: `.spot`, `.get_vol(K, T)`, `.get_rate(T)`, `.get_div_yield(T)`

### Error Handling Philosophy

All components use professional exception handling with a clear hierarchy:
- Input validation at construction time
- Numerical stability checks during computation
- Market data validation at pricing time
- Descriptive error messages indicating what failed and why

## Repo Workflow Notes

- Search: use `rg` for content search and `rg --files` for file discovery
- Edits: prefer `apply_patch` for small, single-file edits
- ExecPlan: for complex features or significant refactors, use `.agent/PLAN.md`

## Skills
- Load all skills in $CODEX_HOME/skills (including $CODEX_HOME/skills/.system)
- Always use the `engine-creator` skill when creating new pricing engines in this repository
- Always use the `engine-validator` skill when creating validation reports for pricing engines in this repository

### Available Skills
- autocallable-risk-report: Generate and validate risk profile analysis reports for autocallable products
- draft-commit-message: Draft a Conventional Commit message when the user asks for help writing a commit message
- engine-creator: Create new pricing engine scripts in the asset/ directory following QuantArk patterns
- engine-validator: Validate pricing engine scripts and generate validation reports
- gh-address-comments: Address review or issue comments on the open GitHub PR using gh CLI
- product-creator: Create new financial product classes in the asset/ directory following QuantArk patterns
- risk-metric-analyzer: Analyze and report comprehensive risk metrics for financial products
- skill-creator: Create or update Codex skills
- skill-installer: Install Codex skills into $CODEX_HOME/skills
