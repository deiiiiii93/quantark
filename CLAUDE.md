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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Note**: The project includes a pre-configured virtual environment `quantark/`. Either activate it first (`source quantark/bin/activate`) or use `quantark/bin/python` and `quantark/bin/pip` for all commands below.

### Dependencies
```bash
# Install Python dependencies (using the virtual environment's pip)
pip install -r requirements.txt
```

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
# European option pricing demo
python example/european_option_demo.py

# American option pricing demo
python example/american_option_demo.py

# Monte Carlo pricing demo
python example/european_mc_demo.py

# PDE pricing demo
python example/pde_pricing_demo.py

# Bond pricing demos
python example/fixed_bond_demo.py
python example/bond_option_demo.py
python example/frn_demo.py
python example/irs_demo.py

# VaR calculation demos
python example/parametric_var_demo.py
python example/historical_var_demo.py
python example/monte_carlo_var_demo.py
python example/var_backtest_demo.py

# Portfolio and backtesting demos
python example/portfolio_demo.py
python example/dynamic_scenario_demo.py
python example/stress_test_demo.py

# List all available examples: ls example/*.py
```

### OpenSpec Workflow (Specification-Driven Development)
The project uses OpenSpec for managing changes. Key commands:

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

# Update instruction files
openspec update
```

For detailed OpenSpec instructions, see `openspec/AGENTS.md`.

## High-Level Architecture

QuantArk is a professional-grade financial derivatives pricing library with a modular, layered architecture:

### Core Design Pattern: Modular Component Architecture

The library separates concerns across independent, composable components:

1. **Products** (`asset/*/product/`) - Define instrument specifications (strike, maturity, type)
   - Examples: `EuropeanVanillaOption`, `FixedBond`, `InterestRateSwap`
2. **Processes** (`asset/*/process/`) - Stochastic models (BSM, Heston, Local Vol)
   - Examples: `GeometricBrownianMotion`, `HestonProcess`
3. **Engines** (`asset/*/engine/`) - Pricing algorithms (Analytical, PDE, Monte Carlo, Quadrature)
   - Examples: `BlackScholesEngine`, `AmericanOptionAnalyticalEngine`, `MonteCarloEngine`
4. **Parameters** (`param/`) - Market data (spot, vol surface, rate curve, dividends)
   - Examples: `SpotQuote`, `FlatVolSurface`, `FlatRateCurve`, `ContinuousDividendYield`
5. **PriceEnv** (`priceenv/`) - Unified pricing environment bundling all market data
   - `PricingEnvironment` is the central data container
6. **RiskMeasures** (`asset/*/riskmeasures/`) - Greeks calculation (analytical and numerical)
   - Examples: `GreeksCalculator`, `BondGreeksCalculator`
7. **VaR** (`var/`) - Portfolio Value-at-Risk calculations with multiple methodologies
   - Engines: `ParametricVaREngine`, `HistoricalVaREngine`, `MonteCarloVaREngine`
   - Risk Factors: `SpotReturnFactor`, `RateShiftFactor`, `VolChangeFactor`
   - Results: `VaRResult`, `IncrementalVaRResult`, `VaRReportGenerator`

### Engine Method Selection Pattern

For engines supporting multiple methods, use the two-level enum pattern defined in `util/enum/engine_enums.py`:

```python
from util.enum.engine_enums import AmericanAnalyticalMethod, EngineType

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

Each asset class (`asset/equity/`, `asset/bond/`, `asset/rate/`) follows the same internal structure:
- `product/` - Instrument definitions
- `engine/` - Pricing engines (analytical/, mc/, pde/, quad/)
- `process/` - Stochastic processes
- `riskmeasures/` - Risk calculations

### Supporting Modules

- **VaR** (`var/`) - Portfolio Value-at-Risk calculations (parametric, historical, Monte Carlo)
- **Portfolio** (`portfolio/`) - Portfolio management with position tracking (equity and fixed income)
- **Backtest** (`backtest/`) - Framework for testing hedging strategies (delta-neutral, DV01-neutral)
- **Dynamic Scenario** (`dynamicscenario/`) - Multi-day scenario simulation
- **Stress Test** (`stresstest/`) - Stress testing framework with scenario definitions
- **Utilities** (`util/`) - Exceptions, enums, calendar, market data utilities

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
- `docs/` - Technical implementation details

## Important Notes

- The library is **quantitative finance** focused - understand derivatives pricing concepts
- Performance matters: Use vectorized NumPy operations
- Backward compatibility: New features should not break existing APIs
- No external market data: Library provides pricing logic, not data feeds