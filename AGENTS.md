# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

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

## Commands

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
```

### Running Examples
```bash
# European option pricing demo
python example/european_option_demo.py

# American option pricing demo
python example/american_option_demo.py

# Other examples are located in the example/ directory
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# The project uses a virtual environment named 'quantark'
```

## Architecture Overview

QuantArk is a professional-grade financial derivatives pricing library with a modular, layered architecture:

### Core Design Pattern: Modular Component Architecture

The library separates concerns across independent, composable components:

1. **Products** (`asset/*/product/`) - Define instrument specifications (strike, maturity, type)
2. **Processes** (`asset/*/process/`) - Stochastic models (BSM, Heston, Local Vol)
3. **Engines** (`asset/*/engine/`) - Pricing algorithms (Analytical, PDE, Monte Carlo)
4. **Parameters** (`param/`) - Market data (spot, vol surface, rate curve, dividends)
5. **PriceEnv** (`priceenv/`) - Unified pricing environment bundling all market data
6. **RiskMeasures** (`asset/*/riskmeasures/`) - Greeks calculation (analytical and numerical)

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
    
    def __call__(self, method=None):
        # Enables EngineType.ANALYTICAL(AmericanAnalyticalMethod.BS93)
        if self == EngineType.ANALYTICAL and method is not None:
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
- Engines: 
  - BlackScholesEngine (analytical)
  - AmericanOptionAnalyticalEngine (BS93, BS02, BAW methods)
  - PDE solvers (Crank-Nicolson, explicit/implicit Euler)
- Risk: Delta, Gamma, Vega, Theta, Rho (both analytical and finite-difference methods)

**Fixed Income** (`asset/bond/`):
- Products: Fixed/floating rate bonds, swaps, bond options, forwards, futures, convertibles
- Engines: Analytical discount-based pricing
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
- Comprehensive logging and visualization (matplotlib, plotly)
- Performance metrics (Sharpe, drawdown, VaR, CVaR)
- Separate implementations for equity and fixed income

**Dynamic Scenario** (`dynamicscenario/`):
- Multi-day scenario simulation with day-by-day parameter evolution
- Path modeling (spot, vol, rate curves)
- Hedging strategy simulation with rebalancing
- Greeks/risk measure evolution tracking
- Both equity and FI support

**Utilities** (`util/`):
- `exceptions.py`: Exception hierarchy (QuantArkException → ValidationError, NumericalError, MarketDataError, PricingError)
- `enum/`: OptionType, ExerciseStyle, BarrierType, engine enums (AmericanAnalyticalMethod, etc.)
- `calendar/`: Day count conventions
- `marketdata/`: Market data utilities

### PricingEnvironment Structure

`PricingEnvironment` is the central data container that bundles all market parameters:
- Required: `rate_curve`, `valuation_date`
- Optional: `spot_quote`, `vol_surface`, `div_yield`
- Provides convenience methods: `.spot`, `.get_vol(K, T)`, `.get_rate(T)`, `.get_div_yield(T)`

### Error Handling Philosophy

All components use professional exception handling with a clear hierarchy:
- Input validation at construction time
- Numerical stability checks during computation
- Market data validation at pricing time
- Descriptive error messages indicating what failed and why


═══════════════════════════════════════════════════════
FAST APPLY - PRIMARY FILE EDIT TOOL - USE THIS FOR EDITS
═══════════════════════════════════════════════════════

IMPORTANT: Use `edit_file` over `str_replace` or full file writes.

This tool handles:
• Automatic indentation correction
• Fuzzy matching for code blocks
• Faster execution than alternatives

→ Prefer this over manual file editing tools.
→ Works with partial code snippets—no need for full file content.

═══════════════════════════════════════════════════════
FAST CONTEXT - PRIMARY CODE SEARCH TOOL - USE THIS FIRST
═══════════════════════════════════════════════════════

IMPORTANT: If you need to explore the codebase, use `warpgrep_codebase_search` FIRST instead of manually running search commands. 

This tool runs parallel grep and readfile calls to locate relevant files and line ranges. Ideal for:
• "Find where authentication is handled"
• "Locate the payment processing logic"
• "Find the bug where users get redirected incorrectly"

Pass a targeted natural language query describing what you're trying to accomplish. Add inferred context when helpful.

→ Always start your search here.
→ Use classical search tools afterward if needed to fill gaps.

CANNOT BE CALLED IN PARALLEL - one invocation at a time."