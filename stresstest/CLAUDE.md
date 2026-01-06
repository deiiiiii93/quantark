# Stress Test Module - Developer Guide

## Overview

The stress test module provides comprehensive scenario analysis for portfolios. It supports flexible parameter stressing at portfolio, underlying, or position level with predefined and custom scenarios.

## Architecture

### Core Design Pattern: Protocol-Based Multi-Asset Architecture

```
stresstest/
├── base.py                    # Protocol interfaces
├── config.py                  # StressTestConfig (wrapper)
├── engine.py                  # StressTestEngine (wrapper)
├── scenario/                  # Scenario management
│   ├── scenario.py            # Scenario, Stress dataclasses
│   ├── scenario_builder.py    # Fluent builder API
│   ├── scenario_library.py    # Predefined scenarios
│   └── scenario_storage.py    # YAML/JSON persistence
├── stress/                    # Stress application
│   ├── stress_types.py        # StressType, StressLevel enums
│   └── stress_applicator.py   # Apply stresses to environments
├── results/                   # Results management
│   ├── stress_results.py      # StressTestResults container
│   ├── result_aggregator.py   # Analysis utilities
│   └── result_exporter.py     # Export to parquet/csv/json
├── report/                    # Reporting
│   ├── report_generator.py    # HTML reports
│   └── visualizer.py          # matplotlib/plotly plots
├── equity/                    # Equity implementation
│   ├── engine.py             # EquityStressEngine
│   ├── config.py             # EquityStressTestConfig
│   └── results.py            # EquityStressResults
└── fi/                        # Fixed Income implementation
    ├── engine.py             # FIStressEngine
    ├── config.py             # FIStressConfig
    └── results.py            # FIStressResults
```

### Exports

```python
from stresstest import (
    StressTestConfig,
    StressTestEngine,
    FIStressConfig,
    FIStressEngine,
    Scenario,
    Stress,
    ScenarioBuilder,
    StressType,
    StressLevel,
)

from stresstest.scenario import ScenarioLibrary, ScenarioStorage
from stresstest.results import StressTestResults, ResultExporter, ResultAggregator
from stresstest.report import ReportGenerator, StressTestVisualizer
```

## Core Components

### 1. Stress Types and Levels

**StressType** - How stress is applied:
```python
class StressType(Enum):
    ABSOLUTE = "absolute"      # Add/subtract absolute value: rate + 200bps
    PERCENTAGE = "percentage"  # Relative change: spot * (1 - 20%)
    VALUE = "value"            # Set to specific value: vol = 0.80
```

**StressLevel** - Where stress is applied:
```python
class StressLevel(Enum):
    PORTFOLIO = "portfolio"    # All positions
    UNDERLYING = "underlying"  # Specific underlying (e.g., "AAPL")
    POSITION = "position"      # Specific position ID
```

### 2. Stress and Scenario Classes

**Stress** - Single parameter stress:
```python
@dataclass
class Stress:
    parameter: str                    # "spot", "volatility", "rate", "key_rate"
    stress_type: StressType
    stress_value: float               # Magnitude of stress
    level: StressLevel = StressLevel.PORTFOLIO
    target: Optional[str] = None      # Required for UNDERLYING/POSITION
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # e.g., tenor_bucket
```

**Scenario** - Container for multiple stresses:
```python
@dataclass
class Scenario:
    name: str
    stresses: List[Stress] = field(default_factory=list)
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stress(self, stress: Stress) -> 'Scenario': ...
    def get_stresses_for_level(self, level: StressLevel) -> List[Stress]: ...
```

### 3. ScenarioBuilder (Fluent API)

```python
scenario = (ScenarioBuilder()
    .name("Market Crash")
    .description("20% drop, 50% vol spike")
    .spot_stress(-0.20)                                    # Percentage default
    .vol_stress(0.50)
    .rate_stress(0.02, stress_type=StressType.ABSOLUTE)   # 200bps hike
    .key_rate_stress(0.01, tenor_bucket="5Y")             # 5Y key rate
    .spread_stress(0.005, spread_curve="CDX HY")          # Credit spread
    .build())
```

Builder methods: `name()`, `description()`, `spot_stress()`, `vol_stress()`, `rate_stress()`, `key_rate_stress()`, `spread_stress()`, `custom_stress()`, `metadata()`, `build()`

### 4. ScenarioLibrary (Predefined Scenarios)

```python
from stresstest.scenario import ScenarioLibrary

# Standard scenarios
ScenarioLibrary.market_crash()      # -20% spot, +50% vol
ScenarioLibrary.market_rally()      # +15% spot, -30% vol
ScenarioLibrary.vol_spike()         # +80% vol
ScenarioLibrary.rate_hike()         # +200bps rate
ScenarioLibrary.severe_downturn()   # -35% spot, +100% vol, -100bps rate

# Historical scenarios
ScenarioLibrary.black_monday_1987()      # -22.6% spot
ScenarioLibrary.financial_crisis_2008()  # -40% spot, +120% vol
ScenarioLibrary.covid_crash_2020()       # -34% spot, +200% vol
```

### 5. Configuration

**StressTestConfig**:
```python
@dataclass
class StressTestConfig:
    calculate_greeks: bool = True
    greeks_method: str = "analytical"    # or "numerical"
    export_formats: List[str] = field(default_factory=lambda: ['parquet'])
    output_dir: str = "./stress_results"
    save_detailed_results: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 6. Engine

**StressTestEngine** (equity, wrapper for backward compat):
```python
engine = StressTestEngine(config)
results = engine.run_static_scenarios(portfolio, scenarios)

# Single scenario
envelope = engine.evaluate_scenario(portfolio, scenario, baseline_value)
```

**FIStressEngine** (fixed income):
```python
fi_engine = FIStressEngine(fi_config)
fi_results = fi_engine.run_static_scenarios(fi_portfolio, scenarios)
```

### 7. Results

**StressResultEnvelope** - Top-level container:
```python
@dataclass
class StressResultEnvelope:
    baseline_value: float
    baseline_greeks: Optional[Dict[str, float]]
    scenario_results: Sequence[ScenarioEnvelope]
    execution_timestamp: datetime
    total_execution_time: float
    config_summary: Dict[str, Any]
```

**ScenarioEnvelope** - Per-scenario results:
```python
@dataclass
class ScenarioEnvelope:
    scenario: Scenario
    portfolio_value: float
    portfolio_pnl: float
    portfolio_pnl_pct: float
    greeks: Optional[Dict[str, float]]
    position_results: List[Dict[str, Any]]
    underlying_results: Dict[str, Dict[str, Any]]
    extra_metrics: Dict[str, Dict[str, Any]]  # Asset-specific
```

**StressTestResults** - Query wrapper:
```python
results.get_summary()               # Dict with all stats
results.get_worst_scenario()        # Scenario with max loss
results.get_best_scenario()         # Scenario with max gain
results.to_summary_dataframe()      # pd.DataFrame
results.to_position_dataframe(scenario_name)
```

### 8. Export and Reporting

**ResultExporter**:
```python
ResultExporter.export_to_parquet(results, "./output")
ResultExporter.export_to_csv(results, "./output")
ResultExporter.export_to_json(results, "./output/results.json")
ResultExporter.export(results, "./output", formats=['parquet', 'csv', 'json'])
```

**ScenarioStorage** (YAML/JSON persistence):
```python
ScenarioStorage.save_scenarios(scenarios, "scenarios.yaml")
scenarios = ScenarioStorage.load_scenarios("scenarios.yaml")
ScenarioStorage.generate_template("template.yaml")
```

**ReportGenerator**:
```python
ReportGenerator().generate_report(results, "report.html", title="Q4 Stress Test")
```

**StressTestVisualizer**:
```python
viz = StressTestVisualizer()
viz.plot_pnl_waterfall(results, "waterfall.png")
viz.plot_pnl_distribution(results, "distribution.png")
viz.plot_scenario_comparison(results, "comparison.png")
viz.create_interactive_dashboard(results, "dashboard.html")
```

## Usage Examples

### Basic Stress Test

```python
from stresstest import StressTestEngine, StressTestConfig
from stresstest.scenario import ScenarioLibrary

config = StressTestConfig(calculate_greeks=True)
engine = StressTestEngine(config)

scenarios = [
    ScenarioLibrary.market_crash(),
    ScenarioLibrary.market_rally(),
    ScenarioLibrary.vol_spike(),
]

results = engine.run_static_scenarios(portfolio, scenarios)
print(results.get_summary())
```

### Custom Scenario with ScenarioBuilder

```python
from stresstest import ScenarioBuilder, StressType

scenario = (ScenarioBuilder()
    .name("Custom Stress")
    .spot_stress(-0.15)
    .vol_stress(0.30)
    .rate_stress(0.005, stress_type=StressType.ABSOLUTE)
    .build())

results = engine.run_static_scenarios(portfolio, [scenario])
```

### Targeted Stress (Single Underlying)

```python
scenario = (ScenarioBuilder()
    .name("AAPL Shock")
    .spot_stress(-0.25, target="AAPL")
    .vol_stress(0.60, target="AAPL")
    .build())
```

### Scenario Persistence

```python
from stresstest.scenario import ScenarioStorage

# Save
ScenarioStorage.save_scenarios(scenarios, "my_scenarios.yaml")

# Load
loaded = ScenarioStorage.load_scenarios("my_scenarios.yaml")
```

### Full Export and Reporting

```python
from stresstest.results import ResultExporter
from stresstest.report import ReportGenerator, StressTestVisualizer

# Export data
ResultExporter.export(results, "./output", formats=['parquet', 'csv'])

# Generate report
ReportGenerator().generate_report(results, "./reports/stress_report.html")

# Create visualizations
viz = StressTestVisualizer()
viz.plot_pnl_waterfall(results, "./plots/waterfall.png")
```

## Testing

```bash
# All stress test tests
python -m pytest test/test_stress_test.py -v

# Specific component
python -m pytest test/test_stress_test.py::TestScenarioBuilder -v

# With coverage
python -m pytest test/test_stress_test.py --cov=stresstest
```

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Scenario name is required" | Missing `.name()` in builder | Call `.name()` before `.build()` |
| "At least one stress is required" | Empty scenario | Add stress via `spot_stress()`, etc. |
| "Target required for UNDERLYING level" | UNDERLYING stress without target | Add `target="SYMBOL"` parameter |
| "tenor_bucket metadata required" | key_rate_stress missing bucket | Use `.key_rate_stress(val, tenor_bucket="5Y")` |

## Integration Points

- **Portfolio**: `portfolio.Portfolio` or `portfolio.fi.FIPortfolio`
- **Pricing**: `priceenv.PricingEnvironment` (cloned and stressed)
- **Greeks**: `asset.equity.riskmeasures.GreeksCalculator`
- **Default adapter**: `EquityStressMetricsAdapter` (computes equity Greeks)

## Summary

- **Stress Types**: 3 (absolute, percentage, value)
- **Stress Levels**: 3 (portfolio, underlying, position)
- **Predefined Scenarios**: 8+ (crashes, rallies, historical)
- **Export Formats**: parquet, csv, json
- **Asset Support**: Equity (complete), FI (engine exists)
