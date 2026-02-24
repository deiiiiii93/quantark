# Dynamic Scenario Analysis Module - Developer Guide

## Overview

The Dynamic Scenario Analysis module simulates portfolio evolution through multi-day market scenarios with optional hedging. Unlike static stress testing that applies instantaneous shocks, dynamic scenarios model **day-by-day parameter evolution**, enabling:

- Time-dependent P&L attribution
- Hedge execution and rebalancing simulation
- Greeks/DV01/duration evolution tracking
- Transaction cost impact analysis

## Architecture

### Core Design Pattern: Protocol-Based Multi-Asset

```
┌─────────────────────────────────────────────────────┐
│                  Base Protocols                      │
│  base.py: BaseDynamicScenarioEngine, RiskMetrics    │
│           get_engine_for_portfolio() (Factory)       │
└───────────────────────┬─────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
   ┌──────▼──────┐             ┌──────▼───────┐
   │   Equity    │             │ Fixed Income │
   │   Engine    │             │   Engine     │
   │ (engine.py) │             │ (fi/engine)  │
   └──────┬──────┘             └──────┬───────┘
          │                           │
   ┌──────▼──────┐             ┌──────▼───────┐
   │   Equity    │             │     FI       │
   │   Results   │             │   Results    │
   │ (results/)  │             │ (fi/results) │
   └─────────────┘             └──────────────┘
```

## Module Structure

```
dynamicscenario/
├── __init__.py                      # Main exports
├── base.py                          # Base protocols, factory function
├── config.py                        # DynamicScenarioConfig (equity)
├── engine.py                        # DynamicScenarioEngine (equity)
├── equity/                          # Equity subpackage (module wrapper)
├── fi/                              # Fixed Income subpackage
│   ├── config.py                    # FIDynamicScenarioConfig
│   ├── engine.py                    # FIDynamicScenarioEngine
│   └── results.py                   # FIDayResult, FIDynamicScenarioResults
├── path/                            # Path components
│   ├── day_path.py                  # DayPath, DayStep, ParameterChange
│   ├── path_builder.py              # PathBuilder (fluent API)
│   ├── path_library.py              # PathLibrary (equity patterns)
│   └── fi_path_library.py           # FIPathLibrary (FI patterns)
├── results/                         # Equity results and export
│   ├── dynamic_results.py           # DayResult, DynamicScenarioResults
│   └── result_exporter.py           # Export utilities (CSV/JSON/Parquet)
└── report/                          # Visualization and reporting
    ├── dynamic_report.py            # HTML report generator
    └── visualizer.py                # Visualization utilities
```

## Core Components

### Path Layer

**DayPath** - Container for multi-day scenario:
```python
@dataclass
class DayPath:
    name: str
    steps: List[DayStep]
    description: Optional[str] = None
```

**DayStep** - Single day's market changes:
```python
@dataclass
class DayStep:
    day_index: int
    changes: List[ParameterChange]
    label: Optional[str] = None
```

**ParameterChange** - Individual parameter modification:
```python
@dataclass
class ParameterChange:
    parameter: str      # "spot", "volatility", "rate"
    stress_type: StressType  # PERCENTAGE, ABSOLUTE, VALUE
    stress_value: float
    target: Optional[str] = None  # Optional underlying or position
```

### PathBuilder (Fluent API)

```python
path = (PathBuilder(num_days=5, name="Custom Rally")
    .description("5 days of gradual price increase")
    .spot_trend(daily_change=0.02)   # +2% per day
    .vol_trend(daily_change=-0.05)   # -5% vol per day
    .rate_trend(daily_change=0.001)  # +10bps rate per day
    .set_day_label(0, "Rally Start")
    .build())
```

### PathLibrary (Equity Patterns)

```python
from dynamicscenario import PathLibrary

# Standard scenarios
PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
PathLibrary.consecutive_decline(days=5, daily_pct=0.03)
PathLibrary.v_shaped_recovery(down_days=3, up_days=5, magnitude=0.20)
PathLibrary.volatility_spike_decay(spike_pct=0.50, decay_days=5)
PathLibrary.gradual_crash(days=10, total_decline=0.25)
PathLibrary.rate_hike_cycle(days=10, total_hike_bps=100)

# Historical scenarios
PathLibrary.historical_black_monday()   # 1987-style crash
PathLibrary.historical_covid_crash()    # March 2020-style
```

### FIPathLibrary (Fixed Income Patterns)

```python
from dynamicscenario import FIPathLibrary

FIPathLibrary.parallel_shift(days=5, total_bps=50)
FIPathLibrary.steepener(days=5, short_bps=-25, long_bps=50)
FIPathLibrary.flattener(days=5, short_bps=50, long_bps=-25)
FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)
FIPathLibrary.rate_cut_cycle(days=10, total_bps=-100)
FIPathLibrary.bear_steepener(days=5, short_bps=25, long_bps=75)
FIPathLibrary.historical_fed_tightening_2022()
```

### Configuration

**DynamicScenarioConfig** (equity):
```python
@dataclass
class DynamicScenarioConfig:
    calculate_greeks: bool = True
    greeks_method: str = 'analytical'  # or 'numerical'
    export_formats: List[str] = field(default_factory=lambda: ['parquet'])
    output_dir: str = './dynamic_results'
    save_detailed_results: bool = True
    generate_report: bool = True
    include_charts: bool = True
```

**FIDynamicScenarioConfig** (fixed income):
```python
@dataclass
class FIDynamicScenarioConfig:
    calculate_dv01: bool = True
    calculate_duration: bool = True
    calculate_convexity: bool = True
    calculate_key_rate_dv01: bool = False
    key_rate_tenors: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 30])
    hedge_enabled: bool = False
    hedge_dv01_threshold: float = 50000.0
    futures_dv01_per_contract: float = 80.0
    export_formats: List[str] = field(default_factory=lambda: ['parquet'])
```

### Engines

**Factory pattern** - Automatic engine selection:
```python
from dynamicscenario import get_engine_for_portfolio

engine = get_engine_for_portfolio(portfolio, config)  # Returns appropriate engine
```

**DynamicScenarioEngine** (equity):
```python
results = engine.run(portfolio, path)
results = engine.run(portfolio, path, hedge_strategy=strategy, transaction_cost_model=cost_model)
```

**FIDynamicScenarioEngine** (FI):
```python
fi_engine = FIDynamicScenarioEngine(fi_config)
fi_results = fi_engine.run(fi_portfolio, path)
```

### Results

**DynamicScenarioResults** (equity):
```python
results.get_summary()               # Dict with all stats
results.get_pnl_evolution()         # pd.DataFrame
results.get_greeks_evolution()      # pd.DataFrame
results.get_max_drawdown()          # float
results.get_worst_day()             # DayResult
results.get_trades_dataframe()      # pd.DataFrame
```

**FIDynamicScenarioResults** (FI):
```python
fi_results.get_dv01_evolution()     # pd.DataFrame
fi_results.get_duration_evolution() # pd.DataFrame
fi_results.get_hedge_effectiveness()# float
fi_results.get_hedge_trades()       # pd.DataFrame
```

## Usage Examples

### Basic Equity Scenario

```python
from dynamicscenario import (
    DynamicScenarioConfig,
    DynamicScenarioEngine,
    PathLibrary,
)

config = DynamicScenarioConfig(calculate_greeks=True)
engine = DynamicScenarioEngine(config)

path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
results = engine.run(portfolio, path)

print(results.get_summary())
pnl_df = results.get_pnl_evolution()
```

### Equity with Hedging

```python
from backtest.strategy import DeltaNeutralStrategy
from backtest.transaction_costs import ProportionalCostModel

strategy = DeltaNeutralStrategy(delta_threshold=100.0, rebalance_frequency='daily')
cost_model = ProportionalCostModel(commission_rate=0.001)

results = engine.run(
    portfolio=portfolio,
    day_path=path,
    hedge_strategy=strategy,
    transaction_cost_model=cost_model,
)
```

### Fixed Income Scenario

```python
from dynamicscenario import (
    FIDynamicScenarioConfig,
    FIDynamicScenarioEngine,
    FIPathLibrary,
)

config = FIDynamicScenarioConfig(
    calculate_dv01=True,
    hedge_enabled=True,
    hedge_dv01_threshold=50000,
    futures_dv01_per_contract=80.0,
)

engine = FIDynamicScenarioEngine(config)
path = FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)
results = engine.run(fi_portfolio, path)

dv01_df = results.get_dv01_evolution()
```

### Custom Path with PathBuilder

```python
from dynamicscenario import PathBuilder

path = (PathBuilder(num_days=5, name="Custom")
    .spot_trend(daily_change=0.02)
    .vol_trend(daily_change=-0.05)
    .rate_trend(daily_change=0.001)
    .build())
```

### Manual Path Construction

```python
from dynamicscenario import DayPath, DayStep, ParameterChange
from stresstest.stress.stress_types import StressType

steps = [
    DayStep(0, [
        ParameterChange("spot", StressType.PERCENTAGE, 0.02),
        ParameterChange("volatility", StressType.PERCENTAGE, 0.05),
    ]),
    DayStep(1, [
        ParameterChange("spot", StressType.PERCENTAGE, -0.01),
    ]),
]

path = DayPath(name="Custom", steps=steps)
```

### Export and Reporting

```python
from dynamicscenario import DynamicResultExporter, DynamicReportGenerator, DynamicScenarioVisualizer

# Export data
DynamicResultExporter.export(results, "./output", formats=['parquet', 'csv'])

# Generate HTML report
DynamicReportGenerator().generate(results, "report.html")

# Create visualizations
viz = DynamicScenarioVisualizer()
viz.plot_pnl_evolution(results, "pnl.png")
viz.plot_greeks_evolution(results, "greeks.png")
```

## Testing

```bash
# All dynamic scenario tests
python -m pytest test/test_dynamic_scenario.py -v

# With coverage
python -m pytest test/test_dynamic_scenario.py --cov=dynamicscenario
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Portfolio must contain at least one position" | Empty portfolio | Add positions before running |
| "Day path must have at least one day" | Empty path | Ensure path has DaySteps |
| "Spot cannot be <= 0" | Large negative shock | Use smaller percentage changes |
| "No engine available for portfolio type" | Unknown portfolio | Use Portfolio or FIPortfolio |

## Integration Points

- **Portfolio**: `portfolio.Portfolio` (equity) or `portfolio.fi.FIPortfolio`
- **Pricing**: `priceenv.PricingEnvironment` (updated day-by-day)
- **Asset Engines**: `asset.equity.engine.analytical.BlackScholesEngine`, `asset.bond.engine.BondDiscountEngine`
- **Greeks**: `asset.equity.riskmeasures.GreeksCalculator`
- **Hedging**: `backtest.strategy.DeltaNeutralStrategy`, `backtest.transaction_costs.*`
- **Stress Types**: `stresstest.stress.stress_types.StressType` (PERCENTAGE, ABSOLUTE)

## Summary

- **Asset Classes**: Equity + Fixed Income
- **Path Patterns**: 8+ equity, 7+ FI predefined patterns
- **Equity Features**: Greeks calculation, delta hedging, transaction costs
- **FI Features**: DV01/duration/convexity, DV01 hedging, rate scenarios
- **Outputs**: Results + export (CSV/JSON/Parquet) + visualization + HTML reports
