# QuantArk Project Structure

## Root Directory Layout

```
QuantArk/
├── asset/                      # Asset class implementations
│   ├── equity/                 # Equity derivatives
│   ├── bond/                   # Fixed income instruments
│   └── rate/                   # Interest rate derivatives
├── param/                      # Market data parameters
├── priceenv/                   # Pricing environment
├── var/                        # Value-at-Risk module
├── portfolio/                  # Portfolio management
├── backtest/                   # Hedging strategy backtesting
├── dynamicscenario/            # Multi-day scenario simulation
├── stresstest/                 # Stress testing framework
├── util/                       # Utilities and helpers
├── example/                    # Example scripts
├── test/                       # Unit tests
├── docs/                       # Documentation (when available)
├── openspec/                   # OpenSpec specifications
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── CLAUDE.md                   # AI assistant guidance
├── LICENSE                     # License file
└── quantark/                   # Virtual environment (pre-configured)
```

## Detailed Module Structure

### 1. Asset Module (`asset/`)

#### Equity Derivatives (`asset/equity/`)
```
asset/equity/
├── engine/                     # Pricing engines
│   ├── analytical/             # Analytical pricing methods
│   │   ├── black_scholes_engine.py
│   │   └── american_analytical_engine.py
│   ├── mc/                     # Monte Carlo engines
│   ├── pde/                    # PDE-based engines
│   └── quad/                   # Quadrature engines
├── process/                    # Stochastic processes
│   ├── bsm/                    # Black-Scholes-Merton
│   ├── heston/                 # Heston stochastic volatility
│   ├── localvol/               # Local volatility
│   └── slv/                    # Stochastic local volatility
├── product/                    # Derivative products
│   └── option/
│       ├── __init__.py
│       ├── european_vanilla.py
│       └── american_vanilla.py
├── param/                      # Engine parameters
├── riskmeasures/               # Greeks calculation
│   ├── greeks_calculator.py
│   └── numerical_greeks.py
└── __init__.py
```

#### Fixed Income (`asset/bond/`)
```
asset/bond/
├── engine/                     # Bond pricing engines
├── product/                    # Bond products
│   ├── fixed_bond.py
│   ├── bond_option.py
│   ├── frn.py                  # Floating rate notes
│   └── irs.py                  # Interest rate swaps
├── riskmeasures/               # Bond risk measures
│   ├── bond_greeks.py          # DV01, duration, convexity
│   └── irs_greeks.py
└── __init__.py
```

#### Interest Rate Derivatives (`asset/rate/`)
```
asset/rate/
├── engine/
├── product/
├── riskmeasures/
└── __init__.py
```

### 2. Market Data Parameters (`param/`)

```
param/
├── div/                        # Dividend yield models
│   └── continuous_dividend_yield.py
├── quote/                      # Spot price quotes
│   └── spot_quote.py
├── rrf/                        # Risk-free rate curves
│   └── flat_rate_curve.py
├── vol/                        # Volatility surfaces
│   ├── flat_vol_surface.py
│   └── implied_vol_surface.py
└── __init__.py
```

### 3. Pricing Environment (`priceenv/`)

```
priceenv/
├── pricing_environment.py      # Central data container
└── __init__.py
```

### 4. VaR Module (`var/`)

```
var/
├── engines/                    # VaR calculation engines
│   ├── historical.py           # Historical VaR engine
│   ├── parametric.py           # Parametric VaR engine
│   └── monte_carlo.py          # Monte Carlo VaR engine
├── risk_factors/               # Risk factor models
│   ├── base.py                 # RiskFactor protocol
│   ├── equity_factors.py       # Spot, vol, rate, dividend factors
│   └── fi_factors.py           # Parallel shift, key rate factors
├── backtest/                   # VaR backtesting
│   ├── var_backtester.py
│   └── var_backtest_result.py
├── results/                    # Result objects
│   ├── var_result.py           # VaRResult dataclass
│   ├── incremental_var_result.py
│   └── var_report_generator.py
├── attribution.py              # Attribution calculators
│   ├── component_var.py
│   ├── marginal_var.py
│   └── factor_var.py
├── base.py                     # Base classes and protocols
├── config.py                   # Configuration dataclasses
├── __init__.py
└── README.md                   # VaR module documentation
```

### 5. Portfolio Module (`portfolio/`)

```
portfolio/
├── equity/                     # Equity portfolios
│   ├── equity_portfolio.py
│   └── position.py
├── fi/                         # Fixed income portfolios
│   ├── fi_portfolio.py
│   └── fi_position.py
└── __init__.py
```

### 6. Backtest Module (`backtest/`)

```
backtest/
├── hedging_strategies.py       # Delta-neutral, DV01-neutral
├── backtest_framework.py       # Core backtesting logic
└── __init__.py
```

### 7. Dynamic Scenario Module (`dynamicscenario/`)

```
dynamicscenario/
├── multi_day_scenario.py       # Multi-day scenario simulation
├── scenario_generator.py
└── __init__.py
```

### 8. Stress Test Module (`stresstest/`)

```
stresstest/
├── stress_scenarios.py         # Stress test definitions
├── stress_test_engine.py
└── __init__.py
```

### 9. Utilities (`util/`)

```
util/
├── enum/                       # Enumerations
│   ├── engine_enums.py         # Engine method enums
│   └── option_enums.py         # Option-related enums
├── calendar.py                 # Date utilities
├── exceptions.py               # Exception hierarchy
└── __init__.py
```

### 10. Examples (`example/`)

```
example/
├── european_option_demo.py
├── american_option_demo.py
├── european_mc_demo.py
├── pde_pricing_demo.py
├── fixed_bond_demo.py
├── bond_option_demo.py
├── frn_demo.py
├── irs_demo.py
├── parametric_var_demo.py
├── historical_var_demo.py
├── monte_carlo_var_demo.py
├── var_backtest_demo.py
├── portfolio_demo.py
├── dynamic_scenario_demo.py
└── stress_test_demo.py
```

### 11. Tests (`test/`)

```
test/
├── test_european_option.py
├── test_american_option_*.py
├── test_euro_mc_engine.py
├── test_pde_engine.py
├── test_fixed_bond.py
├── test_bond_option.py
├── test_frn.py
├── test_irs.py
├── test_backtest.py
├── test_var_*.py               # VaR-related tests
│   ├── test_var_config.py
│   ├── test_var_attribution.py
│   ├── test_stressed_var.py
│   ├── test_incremental_var.py
│   ├── test_var_integration.py
│   ├── test_benchmark_var.py
│   └── test_var_backtest.py
└── conftest.py                 # pytest configuration
```

### 12. OpenSpec (`openspec/`)

```
openspec/
├── AGENTS.md                   # OpenSpec guidance for AI
├── project.md                  # Project conventions
├── changes/                    # Change specifications
│   └── add-portfolio-var/
│       ├── proposal.md
│       ├── tasks.md
│       ├── design.md
│       └── archive/           # Completed changes
└── specs/                      # Technical specifications
```

## Key Architectural Patterns

### Modular Component Architecture
Each component is **independent** and **composable**:
1. **Products** define instrument specifications
2. **Processes** define stochastic models
3. **Engines** implement pricing algorithms
4. **Parameters** contain market data
5. **PriceEnv** bundles all market data
6. **RiskMeasures** calculate Greeks

### Engine-Agnostic Design
Products don't know their pricing method:
```python
# Products are engine-agnostic
option = EuropeanVanillaOption(strike=100, maturity=1.0)

# Any engine can price the same product
bs_engine = BlackScholesEngine()  # Analytical
mc_engine = MonteCarloEngine()    # Simulation

# Both produce the same result type
price = engine.price(option, pricing_env)
```

### VaR Module Architecture
```
Portfolio
    ↓
VaREngine (Protocol)
    ↓
┌──────────────┬──────────────┬──────────────┐
│  Parametric  │ Historical   │ Monte Carlo  │
│  VaR Engine  │ VaR Engine   │ VaR Engine   │
└──────────────┴──────────────┴──────────────┘
    ↓              ↓              ↓
VaRResult ← VaRResult ← VaRResult
    ↓              ↓              ↓
┌──────────────┬──────────────┬──────────────┐
│ Attribution  │   Backtest   │   Reports    │
│  Calculators │   Framework  │  Generator   │
└──────────────┴──────────────┴──────────────┘
```

## File Naming Conventions

### Python Files
- **snake_case**: `black_scholes_engine.py`, `var_config.py`
- **Private**: `_internal_helper.py` (leading underscore)
- **Test files**: `test_<module>.py` (in `test/` directory)
- **Init files**: `__init__.py` (in every package)

### Configuration Files
- **requirements.txt**: Python dependencies
- **README.md**: Project documentation
- **CLAUDE.md**: AI assistant guidance
- **LICENSE**: License file
- **.gitignore**: Git ignore patterns (if exists)

### Documentation
- **AGENTS.md**: OpenSpec documentation
- **project.md**: Project-specific conventions
- **proposal.md**: Feature proposals
- **tasks.md**: Implementation tasks
- **README.md**: Module-specific docs (in `var/`, etc.)

## Import Patterns

### Relative Imports (within module)
```python
# Inside var/engines/parametric.py
from ..base import VaREngine
from ..config import VaRConfig
from .historical import HistoricalVaREngine
```

### Absolute Imports (public API)
```python
# In user code or cross-module
from var import VaRConfig, VaRMethod
from var.engines import ParametricVaREngine
from asset.equity.product.option import EuropeanVanillaOption
```

### Public API Exports
```python
# var/__init__.py
"""VaR module public API."""

from .config import VaRConfig, VaRMethod, EquityRiskFactorConfig, FIRiskFactorConfig
from .engines import ParametricVaREngine, HistoricalVaREngine, MonteCarloVaREngine
from .base import VaREngine
from .results import VaRResult, IncrementalVaRResult

__all__ = [
    'VaRConfig',
    'VaRMethod',
    'VaREngine',
    'VaRResult',
    'ParametricVaREngine',
    'HistoricalVaREngine',
    'MonteCarloVaREngine',
    # ... more exports
]
```

## Data Flow Examples

### Option Pricing Flow
```
EuropeanVanillaOption
    ↓
PricingEnvironment (spot, vol, rate, dividend)
    ↓
BlackScholesEngine
    ↓
Price + Greeks
```

### VaR Calculation Flow
```
Portfolio (positions)
    ↓
Market Data (historical)
    ↓
VaRConfig (confidence, method, parameters)
    ↓
┌─────────────────────────────┐
│ ParametricVaREngine         │
│ - Extract sensitivities     │
│ - Build covariance matrix   │
│ - Calculate VaR             │
└─────────────────────────────┘
    ↓
VaRResult (var, cvar, attribution)
    ↓
ReportGenerator (formatted output)
```

### Portfolio Backtesting Flow
```
Portfolio
    ↓
Historical Data
    ↓
DynamicScenario (multi-day simulation)
    ↓
BacktestFramework (strategy testing)
    ↓
BacktestResult (performance metrics)
```

## Module Dependencies

### Dependency Graph (High Level)
```
asset/*/product/*       →  No dependencies (leaf nodes)
asset/*/engine/*        →  Depends on asset/*/product/*
asset/*/process/*       →  No dependencies
param/*                 →  No dependencies (leaf nodes)
priceenv/*              →  Depends on param/*
var/engines/*           →  Depends on var/config, var/base
portfolio/*             →  Depends on asset/*
backtest/*              →  Depends on portfolio/*
dynamicscenario/*       →  Depends on priceenv
stresstest/*            →  Depends on var/*
util/*                  →  No dependencies (utilities)
```

### No Circular Dependencies
The architecture is **acyclic**:
- Products are leaf nodes (no dependencies)
- Engines depend on products
- PriceEnv depends on param
- VaR engines depend on config and base
- Higher-level modules (backtest) depend on lower-level (portfolio, var)

This ensures **maintainability** and **testability**.
