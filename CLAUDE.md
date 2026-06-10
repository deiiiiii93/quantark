
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Note**: The project virtual environment lives at `.venv/`. Either activate it first (`source .venv/bin/activate`) or use `.venv/bin/python` and `.venv/bin/pip` for all commands below. (`quantark/` is the library package — the old venv of that name is gone.)

### Dependencies
```bash
# Create the venv and install the library editable with dev extras
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

### Package layout & imports
All library code lives under the single top-level package `quantark` (e.g. `quantark.asset`, `quantark.util`, `quantark.param`). Always write new code with canonical `quantark.*` imports. The 12 historical flat top-level names (`asset`, `util`, `param`, …) still import via a compatibility shim (`quantark/_compat.py`, registered by `quantark_compat.pth`) that aliases them to the same module objects with a `DeprecationWarning` — existing consumers keep working, but do not write new flat imports. `example/` scripts intentionally keep flat imports as a live exerciser of the shim.

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

# Run tests with coverage (if coverage installed)
python -m pytest --cov=.
```

### Running Examples
```bash
# Vanilla options
python example/european_option_demo.py
python example/american_option_demo.py
python example/asian_option_demo.py
python example/barrier_analytical_demo.py

# Autocallable products
python example/snowball_mc_demo.py
python example/phoenix_option_demo.py
python example/ko_reset_snowball_demo.py

# Engine comparisons
python example/european_mc_demo.py           # Monte Carlo
python example/pde_engine_demo.py            # PDE
python example/european_quad_demo.py         # Quadrature
python example/phoenix_engine_compare_demo.py

# Fixed income
python example/fixed_bond_demo.py
python example/bond_option_demo.py
python example/bond_forward_futures_demo.py
python example/convertible_bond_demo.py
python example/frn_demo.py
python example/irs_demo.py

# Risk & portfolio
python example/parametric_var_demo.py
python example/portfolio_var_demo.py
python example/var_backtest_demo.py
python example/portfolio_demo.py
python example/dynamic_scenario_demo.py
python example/stress_test_demo.py

# List all 50+ available examples: ls example/*.py
```

### OpenSpec Workflow (Specification-Driven Development)
The project uses OpenSpec (v1.2.0) for managing changes. Key commands:

```bash
# List active changes
openspec list

# List all specifications
openspec list --specs

# Show details of a change or spec
openspec show <change-id>
openspec show <spec-id> --type spec

# Validate a change (always use --strict)
openspec validate <change-id> --strict

# Archive a completed change
openspec archive <change-id> --yes

# Update instruction files (run after upgrading OpenSpec)
openspec update

# New AI assistant commands
/opsx:new       Start a new change
/opsx:continue  Create the next artifact
/opsx:apply     Implement tasks
/opsx:explore   Explore ideas
/opsx:propose  Propose a new change
/opsx:archive  Archive a completed change
```

## High-Level Architecture

QuantArk is a professional-grade financial derivatives pricing library with a modular, layered architecture:

### Core Design Pattern: Modular Component Architecture

The library separates concerns across independent, composable components:

1. **Products** (`quantark/asset/*/product/`) - Define instrument specifications (strike, maturity, type)
   - Vanilla: `EuropeanVanillaOption`, `AmericanOption`, `AsianOption`
   - Exotic: `BarrierOption`, `OneTouchOption`, `CashOrNothingDigitalOption`
   - Autocallable: `SnowballOption`, `PhoenixOption`, `KOResetSnowballOption`, `RangeAccrualOption`
   - Fixed Income: `FixedBond`, `ConvertibleBond`, `BondForward`, `BondFutures`, `InterestRateSwap`
2. **Processes** (`quantark/asset/*/process/`) - Stochastic models (BSM, Heston, Local Vol)
   - Examples: `GeometricBrownianMotion`, `HestonProcess`
3. **Engines** (`quantark/asset/*/engine/`) - Pricing algorithms (Analytical, PDE, Monte Carlo, Quadrature)
   - Examples: `BlackScholesEngine`, `MonteCarloEngine`, `SnowballPDESolver`, `PhoenixQuadEngine`
4. **Parameters** (`quantark/param/`) - Market data (spot, vol surface, rate curve, dividends)
   - Examples: `SpotQuote`, `FlatVolSurface`, `FlatRateCurve`, `ContinuousDividendYield`
5. **PriceEnv** (`quantark/priceenv/`) - Unified pricing environment bundling all market data
   - `PricingEnvironment` is the central data container
6. **RiskMeasures** (`quantark/asset/*/riskmeasures/`) - Greeks calculation (analytical and numerical)
   - Examples: `GreeksCalculator`, `BondGreeksCalculator`
7. **VaR** (`quantark/var/`) - Portfolio Value-at-Risk calculations with multiple methodologies
   - Engines: `ParametricVaREngine`, `HistoricalVaREngine`, `MonteCarloVaREngine`
   - Risk Factors: `SpotReturnFactor`, `RateShiftFactor`, `VolChangeFactor`
   - Results: `VaRResult`, `IncrementalVaRResult`, `VaRReportGenerator`

### Engine Method Selection Pattern

For engines supporting multiple methods, use the two-level enum pattern defined in `quantark/util/enum/engine_enums.py`:

```python
from quantark.util.enum.engine_enums import AmericanAnalyticalMethod, EngineType

# Preferred: Two-level enum pattern
engine = AmericanOptionAnalyticalEngine(
    method=EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
)

# Alternative: Direct method enum
engine = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)

# Backward compatibility: String
engine = AmericanOptionAnalyticalEngine(method="BS93")
```

Always follow this two-level enum pattern (EngineType.ANALYTICAL(method)) for new engine implementations with multiple methods.

### Asset Class Structure

Each asset class (`quantark/asset/equity/`, `quantark/asset/bond/`, `quantark/asset/rate/`) follows the same internal structure:
- `product/` - Instrument definitions
- `engine/` - Pricing engines (analytical/, mc/, pde/, quad/, tree/)
- `process/` - Stochastic processes
- `riskmeasures/` - Risk calculations
- `analysis/` - Path analysis tools (equity autocallables)
- `report/` - Risk reporting and visualization

### Product vs Position Sizing

Products represent one contract/unit, while positions carry quantity:
- Equity options use `contract_multiplier` to scale per-contract payoffs.
- Bonds use `denominator` as the minimum tradable notional; position quantity scales.

### Supporting Modules

- **VaR** (`quantark/var/`) - Portfolio Value-at-Risk calculations (parametric, historical, Monte Carlo) with attribution
- **SIMM** (`quantark/simm/`) - ISDA SIMM v2.6 initial margin calculations
- **Portfolio** (`quantark/portfolio/`) - Portfolio management with position tracking (equity and fixed income)
- **Backtest** (`quantark/backtest/`) - Framework for testing hedging strategies (delta-neutral, DV01-neutral, convexity-neutral)
- **Dynamic Scenario** (`quantark/dynamicscenario/`) - Multi-day scenario simulation (equity and FI)
- **Stress Test** (`quantark/stresstest/`) - Stress testing framework with scenario definitions (equity and FI)
- **Utilities** (`quantark/util/`) - Exceptions, enums, calendar (China holidays/business-day), market data adapter framework, numerical utilities

### Exception Hierarchy

Professional exception handling with clear hierarchy:
- `QuantArkException` (base)
  - `ValidationError` - Invalid input parameters
  - `NumericalError` - Numerical instability/convergence issues
  - `MarketDataError` - Missing or invalid market data
  - `PricingError` - General pricing failures

## Development Workflow

### OpenSpec Three-Stage Workflow

1. **Stage 1: Creating Changes** - Create proposal for new features, breaking changes, architecture shifts
   - Scaffold: `proposal.md`, `tasks.md`, `design.md` (if needed), and delta specs
   - Validate: `openspec validate <change-id> --strict`
   - Do not start implementation until proposal is approved

2. **Stage 2: Implementing Changes** - Follow tasks.md sequentially
   - Complete all items before marking as done
   - Update checklist after implementation

3. **Stage 3: Archiving Changes** - After deployment
   - Move changes to archive directory
   - Update specs if capabilities changed

### When to Create a Proposal

Create proposal when:
- Adding new features or functionality
- Making breaking changes (API, schema)
- Changing architecture or patterns
- Optimizing performance (changes behavior)
- Updating security patterns

Skip proposal for:
- Bug fixes (restore intended behavior)
- Typos, formatting, comments
- Dependency updates (non-breaking)
- Configuration changes
- Tests for existing behavior

## Key Conventions

### Code Style
- **Formatting**: PEP 8
- **Naming**: Classes PascalCase, functions snake_case, constants UPPER_SNAKE_CASE
- **Type hints**: Extensive use of dataclasses and type hints
- **Docstrings**: Triple-quoted docstrings for all public classes and methods

### Design Principles
- **Separation of Concerns**: Products, processes, engines, parameters are independent
- **Engine-Agnostic Products**: Products don't know their pricing method
- **Immutable Market Data**: Parameter objects should be treated as immutable
- **Input Validation**: Validate at every level
- **Numerical Stability**: Handle edge cases (near-expiry, deep ITM/OTM)

### Numerical Operations (IMPORTANT)
Always use `quantark/util/numerical/` utilities for numerical operations. **Do NOT** use raw float comparisons or hardcoded tolerances.

**Float Comparison** - Use `util.numerical.comparison`:
```python
from quantark.util.numerical import is_zero, is_close, almost_equal, Tolerance

# CORRECT: Use is_zero for expiry checks
if is_zero(time_to_expiry):  # Uses Tolerance.ZERO (1e-10)
    return intrinsic_value

# WRONG: Hardcoded tolerance
if time_to_expiry < 1e-10:  # Don't do this
    return intrinsic_value
```

**Safe Math** - Use `util.numerical.safe_math`:
```python
from quantark.util.numerical import safe_log, safe_exp, safe_sqrt, safe_divide

# CORRECT: Protected math operations
log_moneyness = safe_log(spot / strike)  # Prevents log(0)
discount = safe_exp(-r * T)               # Prevents overflow
sigma_sqrt_t = safe_sqrt(variance)        # Prevents sqrt(negative)

# WRONG: Unprotected operations
log_moneyness = math.log(spot / strike)  # Can fail with log(0)
```

**Number Formatting** - Use `util.numerical.formatting`:
```python
from quantark.util.numerical import format_currency, format_percentage, format_basis_points

# CORRECT: Standardized formatting
print(format_currency(price))        # $1,234.56
print(format_percentage(0.05))       # 5.00%
print(format_basis_points(0.0025))   # 25.0bp

# WRONG: Inconsistent formatting
print(f"${price:.2f}")  # Don't hardcode format strings
```

**Validation** - Use `util.numerical.validation`:
```python
from quantark.util.numerical import validate_positive, validate_probability, is_valid_number

# CORRECT: Standardized validation
strike = validate_positive(strike, "strike")
confidence = validate_probability(confidence_level, "confidence_level")
```

### Testing
- **Framework**: pytest
- **Location**: `test/` directory
- **Naming**: `test_<module>.py`
- **Coverage**: Tests for all public APIs and edge cases

## Reference Documentation

- `README.md` - Project overview, features, installation, quick start
- `AGENTS.md` - Detailed guidance for AI assistants with architecture overview
- `openspec/AGENTS.md` - OpenSpec instructions for spec-driven development
- `openspec/project.md` - Project conventions, tech stack, domain context
- `docs/` - Technical implementation details (backtest theory, stress test theory, quad engines, convertible bonds, engine param guide)
- Module-level `CLAUDE.md` - Detailed guides in `quantark/asset/equity/`, `quantark/var/`, `quantark/backtest/`, `quantark/stresstest/`, `quantark/dynamicscenario/`, `quantark/simm/`

## Important Notes

- The library is **quantitative finance** focused - understand derivatives pricing concepts
- Performance matters: Use vectorized NumPy operations
- Backward compatibility: New features should not break existing APIs
- No external market data: Library provides pricing logic, not data feeds
