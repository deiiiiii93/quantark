# Dynamic Scenario Analysis Module

Multi-day scenario simulation engine for equity and fixed income portfolios with optional hedging strategies.

## Overview

The Dynamic Scenario module enables simulation of portfolio evolution through multi-day market scenarios. Unlike static stress testing that applies instantaneous shocks, dynamic scenarios model day-by-day parameter evolution, allowing for:

- Time-dependent P&L attribution
- Hedge execution and rebalancing simulation
- Greeks/risk measure evolution tracking
- Transaction cost impact analysis

## Supported Asset Classes

### Equity Portfolios
- Delta, gamma, vega, theta evolution tracking
- Delta-neutral hedging strategies
- Spot, volatility, and rate path modeling

### Fixed Income Portfolios (NEW)
- DV01, convexity, duration evolution tracking
- DV01-neutral hedging with bond futures
- Rate curve path modeling (parallel shifts, steepeners, flatteners)
- Key-rate DV01 tracking (optional)

## Quick Start

### Equity Dynamic Scenario

```python
from dynamicscenario import (
    DynamicScenarioConfig,
    DynamicScenarioEngine,
    PathLibrary,
)

# Create config
config = DynamicScenarioConfig(calculate_greeks=True)
engine = DynamicScenarioEngine(config)

# Use a predefined path
path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)

# Run simulation
results = engine.run(portfolio, path)
print(results.get_summary())
```

### FI Dynamic Scenario

```python
from dynamicscenario import (
    FIDynamicScenarioConfig,
    FIDynamicScenarioEngine,
    FIPathLibrary,
)

# Create FI config
config = FIDynamicScenarioConfig(
    calculate_dv01=True,
    calculate_duration=True,
    hedge_enabled=True,
    hedge_dv01_threshold=50000,
)
engine = FIDynamicScenarioEngine(config)

# Use FI-specific path
path = FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)

# Run simulation
results = engine.run(fi_portfolio, path)
print(results.get_summary())
```

## Path Components

### DayPath Structure

A `DayPath` consists of:
- **DayStep**: Market changes for a single day
- **ParameterChange**: Individual parameter modification

```python
from dynamicscenario import DayPath, DayStep, ParameterChange
from stresstest.stress.stress_types import StressType

# Manual path construction
steps = [
    DayStep(0, [ParameterChange("spot", StressType.PERCENTAGE, 0.02)]),
    DayStep(1, [ParameterChange("spot", StressType.PERCENTAGE, -0.01)]),
]
path = DayPath(name="Custom", steps=steps)
```

### PathBuilder (Fluent API)

```python
from dynamicscenario import PathBuilder

path = (PathBuilder(num_days=5, name="Rally with Vol Decay")
    .spot_trend(daily_change=0.02)
    .vol_decay(daily_change=-0.01)
    .rate_trend(daily_change=0.001)
    .build())
```

### PathLibrary (Equity Patterns)

Predefined equity scenarios:

| Pattern | Description |
|---------|-------------|
| `consecutive_rally(days, daily_pct)` | N days of consistent price increase |
| `consecutive_decline(days, daily_pct)` | N days of consistent price decline |
| `v_shaped_recovery(down_days, up_days, magnitude)` | Sharp decline followed by recovery |
| `volatility_spike_decay(spike_pct, decay_days)` | Vol spike with gradual decay |
| `gradual_crash(days, total_decline)` | Steady decline with vol spike |
| `rate_hike_cycle(days, total_hike_bps)` | Rate increases with equity pressure |
| `historical_black_monday()` | 1987-style crash simulation |
| `historical_covid_crash()` | March 2020-style crash |

### FIPathLibrary (Fixed Income Patterns)

Predefined FI scenarios:

| Pattern | Description |
|---------|-------------|
| `parallel_shift(days, total_bps)` | Uniform rate change across tenors |
| `steepener(days, short_bps, long_bps)` | Short down, long up |
| `flattener(days, short_bps, long_bps)` | Short up, long down |
| `rate_hike_cycle(days, total_bps)` | Gradual rate increases |
| `rate_cut_cycle(days, total_bps)` | Gradual rate decreases |
| `bear_steepener(days, short_bps, long_bps)` | Both up, long more |
| `bull_flattener(days, short_bps, long_bps)` | Both down, long more |
| `historical_fed_tightening_2022()` | 2022 Fed cycle simulation |

## Configuration

### Equity Config

```python
DynamicScenarioConfig(
    calculate_greeks=True,
    greeks_method='analytical',  # or 'numerical'
    export_formats=['csv', 'parquet', 'json'],
    output_dir='./results',
    save_detailed_results=True,
    generate_report=True,
)
```

### FI Config

```python
FIDynamicScenarioConfig(
    calculate_dv01=True,
    calculate_convexity=True,
    calculate_duration=True,
    calculate_key_rate_dv01=False,
    key_rate_tenors=[1, 2, 5, 10, 30],
    
    # Hedging
    hedge_enabled=True,
    hedge_dv01_threshold=50000,
    futures_dv01_per_contract=80.0,
    
    # Export
    export_formats=['csv', 'json'],
    output_dir='./results',
)
```

## Results

### Equity Results

```python
results.get_pnl_evolution()      # DataFrame with P&L over days
results.get_greeks_evolution()   # DataFrame with Greeks over days
results.get_market_evolution()   # DataFrame with market params
results.get_trades_dataframe()   # All hedge trades
results.get_max_drawdown()       # (amount, pct, peak_day, trough_day)
```

### FI Results

```python
results.get_dv01_evolution()     # Pre-hedge and post-hedge DV01
results.get_duration_evolution() # Modified duration over days
results.get_rate_evolution()     # Rate levels over days
results.get_hedge_trades()       # Futures hedge trades
results.get_hedge_effectiveness()  # DV01 tracking error, hedge frequency
```

## Visualization

```python
from dynamicscenario import DynamicScenarioVisualizer

viz = DynamicScenarioVisualizer()

# Equity plots
viz.create_all_plots(equity_results, output_dir)

# FI-specific plots
viz.create_fi_all_plots(fi_results, output_dir)
viz.plot_dv01_evolution(fi_results, "dv01.png")
viz.plot_duration_evolution(fi_results, "duration.png")
viz.plot_rate_evolution(fi_results, "rates.png")
viz.plot_fi_risk_dashboard(fi_results, "fi_dashboard.png")
```

## Report Generation

```python
from dynamicscenario import DynamicReportGenerator

reporter = DynamicReportGenerator()

# Equity report
reporter.generate_report(equity_results, "report.html")

# FI report
reporter.generate_fi_report(fi_results, "fi_report.html")
```

## Export

```python
from dynamicscenario import DynamicResultExporter
from dynamicscenario.results.result_exporter import FIResultExporter

# Equity export
exporter = DynamicResultExporter(results)
exporter.export_all(output_dir, formats=['csv', 'json'])

# FI export
fi_exporter = FIResultExporter(fi_results)
fi_exporter.export_all(output_dir, formats=['csv', 'json'])
```

## Example

See `example/dynamic_scenario_fi_demo.py` for a complete FI example:

```bash
python example/dynamic_scenario_fi_demo.py
```

## Module Structure

```
dynamicscenario/
├── __init__.py              # Main exports
├── base.py                  # Base protocols
├── config.py                # Equity config
├── engine.py                # Equity engine
├── equity/                  # Equity subpackage
│   └── __init__.py
├── fi/                      # Fixed Income subpackage
│   ├── __init__.py
│   ├── config.py            # FIDynamicScenarioConfig
│   ├── engine.py            # FIDynamicScenarioEngine
│   └── results.py           # FIDayResult, FIDynamicScenarioResults
├── path/
│   ├── day_path.py          # DayPath, DayStep, ParameterChange
│   ├── path_builder.py      # PathBuilder
│   ├── path_library.py      # Equity PathLibrary
│   └── fi_path_library.py   # FIPathLibrary
├── results/
│   ├── dynamic_results.py   # Equity results
│   └── result_exporter.py   # Export utilities
└── report/
    ├── dynamic_report.py    # HTML report generator
    └── visualizer.py        # Visualization utilities
```

