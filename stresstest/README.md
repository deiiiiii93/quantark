# Stress Test Module

Comprehensive stress testing and scenario analysis for portfolio risk management.

## Overview

The Stress Test module provides powerful tools for analyzing how different market conditions affect portfolio P&L and risk exposures. It supports both static scenario analysis (current implementation) and is designed with APIs ready for future dynamic scenario analysis with time dimension and hedging strategies.

## Key Features

- **Flexible Parameter Stressing**: Stress any parameter in PricingEnvironment at three levels:
  - Portfolio level: Apply to all positions
  - Underlying level: Target specific underlyings
  - Position level: Target individual positions

- **Multi-Parameter Scenarios**: Apply multiple simultaneous stresses (e.g., spot -20% AND vol +50%)

- **Predefined Scenario Library**: Common scenarios including:
  - Market crash, rally, vol spike/crush
  - Rate hikes and cuts
  - Historical scenarios (Black Monday 1987, 2008 Crisis, COVID 2020)

- **Scenario Management**: Save and load scenarios from YAML/JSON files for reusability

- **Comprehensive Results**: Calculate P&L, Greeks, and risk metrics across scenarios

- **Rich Reporting**:
  - Export to Parquet, CSV, JSON
  - HTML reports with executive summary
  - Static plots (matplotlib) and interactive dashboards (plotly)

- **Future-Ready**: API designed to support dynamic scenario analysis with time evolution and hedging strategies

## Installation

The module is part of the QuantArk library. Ensure you have the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn plotly pyyaml pyarrow
```

## Quick Start

```python
from datetime import datetime
from portfolio import Portfolio
from stresstest import StressTestEngine, StressTestConfig
from stresstest.scenario.scenario_library import ScenarioLibrary
from stresstest.scenario.scenario_builder import ScenarioBuilder

# 1. Create or load your portfolio
portfolio = Portfolio(...)  # Your existing portfolio

# 2. Define scenarios
scenarios = [
    ScenarioLibrary.market_crash(),
    ScenarioLibrary.vol_spike(),
    ScenarioBuilder()
        .name("Custom Scenario")
        .spot_stress(-0.15)
        .vol_stress(0.30)
        .build()
]

# 3. Configure and run stress test
config = StressTestConfig(
    calculate_greeks=True,
    export_formats=['parquet', 'csv']
)
engine = StressTestEngine(config)
results = engine.run_static_scenarios(portfolio, scenarios)

# 4. View and export results
print(results.get_summary())
results.to_summary_dataframe().to_csv("results.csv")
```

## Architecture

### Core Components

```
stresstest/
├── scenario/           # Scenario definition and management
│   ├── scenario.py           # Core Scenario and Stress classes
│   ├── scenario_builder.py   # Fluent API for building scenarios
│   ├── scenario_library.py   # Predefined scenarios
│   └── scenario_storage.py   # YAML/JSON I/O
├── stress/            # Stress application logic
│   ├── stress_types.py       # Stress type enums (ABSOLUTE, PERCENTAGE, VALUE)
│   └── stress_applicator.py  # Apply stresses to pricing environments
├── results/           # Results management
│   ├── stress_results.py     # Results container classes
│   ├── result_aggregator.py  # Analysis and comparison
│   └── result_exporter.py    # Export to various formats
├── report/            # Reporting and visualization
│   ├── report_generator.py   # HTML report generation
│   └── visualizer.py          # Static and interactive plots
├── engine.py          # Main stress test execution engine
└── config.py          # Configuration classes
```

## Usage Guide

### Creating Scenarios

#### Using the Builder API

The ScenarioBuilder provides a fluent interface for creating scenarios:

```python
from stresstest.scenario.scenario_builder import ScenarioBuilder
from stresstest.stress.stress_types import StressType

# Simple scenario: spot down 20%, vol up 50%
scenario = (ScenarioBuilder()
    .name("Market Stress")
    .description("Significant market downturn")
    .spot_stress(-0.20)      # Percentage stress (default)
    .vol_stress(0.50)
    .build()
)

# Multi-parameter with different stress types
scenario = (ScenarioBuilder()
    .name("Complex Scenario")
    .spot_stress(-0.15)                                    # Percentage
    .vol_stress(0.05, stress_type=StressType.ABSOLUTE)   # Absolute
    .rate_stress(0.02, stress_type=StressType.ABSOLUTE)   # +200bps
    .build()
)

# Target specific underlying
scenario = (ScenarioBuilder()
    .name("AAPL Specific")
    .spot_stress(-0.25, underlying="AAPL")
    .vol_stress(0.60, underlying="AAPL")
    .build()
)
```

#### Using Predefined Scenarios

```python
from stresstest.scenario.scenario_library import ScenarioLibrary

# Standard scenarios
scenarios = [
    ScenarioLibrary.market_crash(),          # -20% spot, +50% vol
    ScenarioLibrary.market_rally(),          # +15% spot, -30% vol
    ScenarioLibrary.vol_spike(),             # +80% vol
    ScenarioLibrary.rate_hike(),             # +200bps
    ScenarioLibrary.severe_downturn(),       # -35% spot, +100% vol, -100bps
]

# Historical scenarios
historical = [
    ScenarioLibrary.black_monday_1987(),     # -22.6% drop
    ScenarioLibrary.financial_crisis_2008(), # -40% equity, +120% vol
    ScenarioLibrary.covid_crash_2020(),      # -34% equity, +200% vol
]

# Get all predefined
all_predefined = ScenarioLibrary.get_all_predefined()
```

#### Manual Scenario Creation

```python
from stresstest.scenario.scenario import Scenario, Stress
from stresstest.stress.stress_types import StressType, StressLevel

scenario = Scenario(
    name="Custom Scenario",
    description="Detailed custom scenario",
    stresses=[
        Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO),
        Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.PORTFOLIO),
        Stress("rate", StressType.ABSOLUTE, 0.01, StressLevel.PORTFOLIO),
    ],
    metadata={"category": "custom", "severity": "high"}
)
```

### Saving and Loading Scenarios

```python
from stresstest.scenario.scenario_storage import ScenarioStorage

# Save to YAML (recommended)
ScenarioStorage.save_scenarios(scenarios, "my_scenarios.yaml")

# Save to JSON
ScenarioStorage.save_scenarios(scenarios, "my_scenarios.json")

# Load from file
scenarios = ScenarioStorage.load_scenarios("my_scenarios.yaml")

# Load single scenario
scenario = ScenarioStorage.load_scenario("my_scenarios.yaml", scenario_index=0)
```

### Running Stress Tests

```python
from stresstest import StressTestEngine, StressTestConfig

# Configure
config = StressTestConfig(
    calculate_greeks=True,
    greeks_method='analytical',      # or 'numerical'
    export_formats=['parquet', 'csv', 'json'],
    output_dir='./stress_results',
    save_detailed_results=True
)

# Create engine
engine = StressTestEngine(config)

# Run stress test
results = engine.run_static_scenarios(
    portfolio=my_portfolio,
    scenarios=my_scenarios,
    baseline_label="Current Market"
)

# Print summary
print(results.get_summary())
```

### Analyzing Results

```python
from stresstest.results.result_aggregator import ResultAggregator

# Get worst and best scenarios
worst = results.get_worst_scenario()
best = results.get_best_scenario()

print(f"Worst: {worst.scenario.name}, P&L: ${worst.portfolio_pnl:,.2f}")
print(f"Best: {best.scenario.name}, P&L: ${best.portfolio_pnl:,.2f}")

# Get risk summary
risk_summary = ResultAggregator.get_risk_summary(results)
print(f"Average P&L: ${risk_summary['avg_pnl']:,.2f}")
print(f"Max Drawdown: {risk_summary['max_drawdown_pct']:.2f}%")

# Calculate VaR and CVaR
var_cvar = ResultAggregator.calculate_var_cvar(results, confidence_level=0.95)
print(f"95% VaR: ${var_cvar['var']:,.2f}")
print(f"95% CVaR: ${var_cvar['cvar']:,.2f}")

# Compare scenarios
comparison_df = ResultAggregator.compare_scenarios(results, metric='portfolio_pnl')
print(comparison_df)

# Get position-level details for a scenario
position_df = results.to_position_dataframe("Market Crash")
print(position_df)
```

### Exporting Results

```python
from stresstest.results.result_exporter import ResultExporter

# Export to Parquet (recommended for large datasets)
ResultExporter.export_to_parquet(
    results, 
    "./output/stress_results",
    include_positions=True
)

# Export to CSV
ResultExporter.export_to_csv(
    results,
    "./output/stress_results"
)

# Export to JSON
ResultExporter.export_to_json(
    results,
    "./output/stress_results.json"
)

# Export to multiple formats at once
ResultExporter.export(
    results,
    output_dir="./output",
    formats=['parquet', 'csv', 'json'],
    base_name="my_stress_test"
)

# Export risk metrics summary
ResultExporter.export_risk_metrics(
    results,
    "./output/risk_metrics.csv"
)
```

### Generating Reports and Visualizations

```python
from stresstest.report import ReportGenerator, StressTestVisualizer

# Generate HTML report
report_gen = ReportGenerator()
report_gen.generate_report(
    results,
    "./reports/stress_test_report.html",
    title="Q4 2024 Stress Test"
)

# Create visualizations
visualizer = StressTestVisualizer()

# Individual plots
visualizer.plot_pnl_waterfall(results, "./plots/waterfall.png")
visualizer.plot_pnl_distribution(results, "./plots/distribution.png")
visualizer.plot_scenario_comparison(results, "./plots/comparison.png")
visualizer.plot_greeks_comparison(results, "./plots/greeks.png")

# Interactive dashboard
visualizer.create_interactive_dashboard(
    results,
    "./plots/interactive_dashboard.html"
)

# Generate all plots at once
visualizer.create_all_plots(
    results,
    output_dir="./plots",
    prefix="stress_test"
)
```

## Stress Types and Levels

### Stress Types

- **ABSOLUTE**: Add/subtract absolute value
  ```python
  Stress("rate", StressType.ABSOLUTE, 0.02, ...)  # +200bps
  ```

- **PERCENTAGE**: Apply percentage change
  ```python
  Stress("spot", StressType.PERCENTAGE, -0.20, ...)  # -20%
  ```

- **VALUE**: Set to specific value
  ```python
  Stress("rate", StressType.VALUE, 0.05, ...)  # Set to 5%
  ```

### Stress Levels

- **PORTFOLIO**: Apply to all pricing environments
  ```python
  Stress(..., level=StressLevel.PORTFOLIO, target=None)
  ```

- **UNDERLYING**: Apply to specific underlying
  ```python
  Stress(..., level=StressLevel.UNDERLYING, target="AAPL")
  ```

- **POSITION**: Apply to specific position
  ```python
  Stress(..., level=StressLevel.POSITION, target=position_id)
  ```

## Advanced Usage

### Custom Stress Applicators

For complex stressing logic, you can extend the stress applicator:

```python
from stresstest.stress.stress_applicator import StressApplicator

# Get stress summary showing changes
stressed_env = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
summary = StressApplicator.get_stress_summary(original_env, stressed_env)
```

### Parallel Execution (Future)

The API is designed to support parallel execution in the future:

```python
config = StressTestConfig(
    parallel_execution=True,
    max_workers=4
)
```

### Dynamic Scenario Analysis (Future)

Placeholder for time-series scenario analysis:

```python
# This will be available in future versions
results = engine.run_dynamic_scenarios(
    portfolio=portfolio,
    scenarios=scenarios,
    time_steps=[date1, date2, date3, ...],
    hedge_strategy=my_hedge_strategy
)
```

## Examples

See `example/stress_test_demo.py` for a comprehensive demonstration including:
- Portfolio creation
- Custom scenario definition
- Scenario storage and loading
- Stress test execution
- Results export and visualization
- HTML report generation

Run the demo:
```bash
python example/stress_test_demo.py
```

## API Reference

### Main Classes

- **StressTestEngine**: Main execution engine
- **StressTestConfig**: Configuration for stress tests
- **Scenario**: Container for multiple stresses
- **Stress**: Individual parameter stress
- **ScenarioBuilder**: Fluent API for building scenarios
- **StressTestResults**: Container for results
- **ResultAggregator**: Analysis utilities
- **ResultExporter**: Export to various formats
- **ReportGenerator**: HTML report generation
- **StressTestVisualizer**: Plotting and visualization

### Enums

- **StressType**: ABSOLUTE, PERCENTAGE, VALUE
- **StressLevel**: PORTFOLIO, UNDERLYING, POSITION

## Best Practices

1. **Start with Predefined Scenarios**: Use the scenario library as a baseline
2. **Save Scenarios**: Store scenario definitions in version control
3. **Use Parquet for Large Results**: More efficient than CSV for big datasets
4. **Calculate Greeks**: Essential for understanding risk exposure changes
5. **Review Position-Level Details**: Identify which positions drive P&L
6. **Test Extreme Scenarios**: Include tail risk scenarios in your analysis
7. **Document Assumptions**: Use scenario metadata to document assumptions

## Troubleshooting

### Common Issues

**Issue**: Stressed volatility becomes negative
- **Solution**: Check stress values; volatility must remain positive

**Issue**: Position-level stress not working
- **Solution**: Ensure position_id is correct; use `portfolio.positions.keys()`

**Issue**: Greeks calculation fails
- **Solution**: Check if products support analytical Greeks; use 'numerical' method

**Issue**: Large memory usage
- **Solution**: Disable `save_detailed_results` in config for large portfolios

## Future Enhancements

The module is designed with placeholders for:
- **Dynamic Scenario Analysis**: Time-series scenarios with hedging strategies
- **Parallel Execution**: Run scenarios in parallel for better performance
- **Monte Carlo Scenarios**: Generate scenarios from distributions
- **Historical Replay**: Replay actual historical market moves
- **Custom Stress Functions**: User-defined stress application logic

## Contributing

When adding new features:
1. Maintain API consistency with existing patterns
2. Add comprehensive docstrings
3. Include examples in the README
4. Add unit tests for new functionality

## License

Part of the QuantArk quantitative finance library.

