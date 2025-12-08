# Dynamic Scenario Analysis Module - Developer Guide

## Overview

The Dynamic Scenario Analysis module (`dynamicscenario/`) is a sophisticated framework for simulating portfolio evolution through multi-day market scenarios with optional hedging strategies. Unlike static stress testing that applies instantaneous shocks, dynamic scenarios model **day-by-day parameter evolution**, enabling:

- Time-dependent P&L attribution
- Hedge execution and rebalancing simulation
- Greeks/risk measure evolution tracking
- Transaction cost impact analysis
- Multi-asset class support (Equity and Fixed Income)

## Architecture

### Core Design Pattern: Asset-Class-Agnostic with Specialized Engines

The module uses a protocol-based architecture that separates core concerns from asset-class-specific implementations:

```
┌─────────────────────────────────────────────────────────────┐
│                    Base Protocols                           │
│  (base.py)                                                 │
│  - BaseDynamicScenarioEngine (ABC)                         │
│  - BaseScenarioResults (dataclass)                         │
│  - RiskMetricsAdapter (Protocol)                           │
│  - get_engine_for_portfolio() (Factory)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼───────┐           ┌───────▼────────┐
        │   Equity      │           │ Fixed Income   │
        │   Engine      │           │   Engine       │
        │ (engine.py)   │           │   (fi/engine.py)│
        └───────┬───────┘           └───────┬────────┘
                │                           │
        ┌───────▼───────┐           ┌───────▼────────┐
        │   Equity      │           │ Fixed Income   │
        │   Results     │           │    Results     │
        │(results/dynamic│          │  (fi/results.py)│
        │  _results.py)  │           └────────────────┘
        └────────────────┘
```

### Key Components

#### 1. **Base Layer** (`base.py`)
Core protocols that all dynamic scenario engines must implement:
- `BaseDynamicScenarioEngine`: Abstract base for all engines
- `BaseScenarioResults`: Base results class with common metrics
- `RiskMetricsAdapter`: Protocol for computing asset-class-specific risk metrics
- `get_engine_for_portfolio()`: Factory function to select appropriate engine

#### 2. **Path Layer** (`path/`)
Defines how market parameters evolve over time:

##### Path Components
- **`DayPath`**: Container for multi-day scenario (sequence of `DayStep`)
- **`DayStep`**: Single day's market changes (collection of `ParameterChange`)
- **`ParameterChange`**: Individual parameter modification (spot, vol, rate)

##### Path Construction
- **`PathBuilder`**: Fluent API for custom path creation
- **`PathLibrary`**: Predefined equity scenarios (rally, crash, etc.)
- **`FIPathLibrary`**: Predefined FI scenarios (rate hikes, curve twists)

##### Available Path Patterns

**Equity (PathLibrary)**:
- `consecutive_rally(days, daily_pct)`: N days of consistent price increase
- `consecutive_decline(days, daily_pct)`: N days of consistent price decline
- `v_shaped_recovery(down_days, up_days, magnitude)`: Sharp decline followed by recovery
- `volatility_spike_decay(spike_pct, decay_days)`: Vol spike with gradual decay
- `gradual_crash(days, total_decline)`: Steady decline with vol spike
- `rate_hike_cycle(days, total_hike_bps)`: Rate increases with equity pressure
- `historical_black_monday()`: 1987-style crash simulation
- `historical_covid_crash()`: March 2020-style crash

**Fixed Income (FIPathLibrary)**:
- `parallel_shift(days, total_bps)`: Uniform rate change across tenors
- `steepener(days, short_bps, long_bps)`: Short down, long up
- `flattener(days, short_bps, long_bps)`: Short up, long down
- `rate_hike_cycle(days, total_bps)`: Gradual rate increases
- `rate_cut_cycle(days, total_bps)`: Gradual rate decreases
- `bear_steepener(days, short_bps, long_bps)`: Both up, long more
- `bull_flattener(days, short_bps, long_bps)`: Both down, long more
- `historical_fed_tightening_2022()`: 2022 Fed cycle simulation

#### 3. **Engine Layer**
Asset-class-specific simulation engines:

**Equity Engine** (`engine.py`):
- `DynamicScenarioEngine`: Main equity engine
- Handles: Portfolio value calculation, Greeks calculation, Delta hedging
- Integrates with: `backtest.strategy` for hedging strategies

**Fixed Income Engine** (`fi/engine.py`):
- `FIDynamicScenarioEngine`: FI-specific engine
- Handles: DV01/duration/convexity calculation, DV01 hedging with futures
- Integrates with: Bond pricing engines

#### 4. **Results Layer** (`results/`)
Rich result objects with day-by-day evolution:

**Equity Results** (`results/dynamic_results.py`):
- `DynamicScenarioResults`: Main results container
- `DayResult`: Single day result with positions, trades, market state
- Methods: `get_pnl_evolution()`, `get_greeks_evolution()`, `get_max_drawdown()`

**FI Results** (`fi/results.py`):
- `FIDynamicScenarioResults`: FI-specific results
- `FIDayResult`: FI day result with DV01/duration tracking
- Methods: `get_dv01_evolution()`, `get_duration_evolution()`, `get_hedge_effectiveness()`

**Export** (`results/result_exporter.py`):
- `DynamicResultExporter`: Equity results export
- `FIResultExporter`: FI results export
- Formats: CSV, JSON, Parquet

#### 5. **Visualization & Reporting** (`report/`)
- **`DynamicScenarioVisualizer`**: Plot generation (equity and FI)
- **`DynamicReportGenerator`**: HTML report generation

## Usage Patterns

### Basic Equity Scenario

```python
from dynamicscenario import (
    DynamicScenarioConfig,
    DynamicScenarioEngine,
    PathLibrary,
)

# Create config
config = DynamicScenarioConfig(
    calculate_greeks=True,
    greeks_method='analytical',
    export_formats=['csv', 'json'],
)

# Create engine
engine = DynamicScenarioEngine(config)

# Use predefined path
path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)

# Run simulation
results = engine.run(portfolio, path)

# Analyze results
print(results.get_summary())
pnl_df = results.get_pnl_evolution()
greeks_df = results.get_greeks_evolution()
```

### Equity Scenario with Hedging

```python
from backtest.strategy.delta_neutral_strategy import DeltaNeutralStrategy
from backtest.transaction_costs import ProportionalCostModel

# Create hedging strategy
hedge_strategy = DeltaNeutralStrategy(
    delta_threshold=100.0,
    rebalance_frequency='daily',
)

# Create transaction cost model
cost_model = ProportionalCostModel(commission_rate=0.001)

# Run with hedging
results = engine.run(
    portfolio=portfolio,
    day_path=path,
    hedge_strategy=hedge_strategy,
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

# Create FI config
config = FIDynamicScenarioConfig(
    calculate_dv01=True,
    calculate_duration=True,
    calculate_convexity=True,
    hedge_enabled=True,
    hedge_dv01_threshold=50000,
    futures_dv01_per_contract=80.0,
)

# Create engine
engine = FIDynamicScenarioEngine(config)

# Use FI-specific path
path = FIPathLibrary.rate_hike_cycle(days=10, total_bps=100)

# Run simulation
results = engine.run(fi_portfolio, path)

# Analyze FI-specific results
print(results.get_summary())
dv01_df = results.get_dv01_evolution()
duration_df = results.get_duration_evolution()
hedge_trades = results.get_hedge_trades()
```

### Custom Path with PathBuilder

```python
from dynamicscenario import PathBuilder

path = (PathBuilder(num_days=5, name="Custom Rally")
    .description("5 days of gradual price increase with vol decay")
    .spot_trend(daily_change=0.02)  # +2% per day
    .vol_trend(daily_change=-0.05)  # -5% vol per day
    .rate_trend(daily_change=0.001) # +10bps rate per day
    .set_day_label(0, "Rally Start")
    .set_day_label(4, "Rally Peak")
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
        ParameterChange("volatility", StressType.PERCENTAGE, -0.02),
    ]),
]

path = DayPath(name="Custom", steps=steps)
```

## Module Structure

```
dynamicscenario/
├── __init__.py                      # Main exports
├── base.py                          # Base protocols
├── config.py                        # Equity config (DynamicScenarioConfig)
├── engine.py                        # Equity engine (DynamicScenarioEngine)
├── equity/                          # Equity subpackage
│   └── __init__.py
├── fi/                              # Fixed Income subpackage
│   ├── __init__.py
│   ├── config.py                    # FIDynamicScenarioConfig
│   ├── engine.py                    # FIDynamicScenarioEngine
│   └── results.py                   # FIDayResult, FIDynamicScenarioResults
├── path/                            # Path components
│   ├── __init__.py
│   ├── day_path.py                  # DayPath, DayStep, ParameterChange
│   ├── path_builder.py              # PathBuilder (fluent API)
│   ├── path_library.py              # PathLibrary (equity patterns)
│   └── fi_path_library.py           # FIPathLibrary (FI patterns)
├── results/                         # Results and export
│   ├── __init__.py
│   ├── dynamic_results.py           # Equity results
│   └── result_exporter.py           # Export utilities
└── report/                          # Visualization and reporting
    ├── __init__.py
    ├── dynamic_report.py            # HTML report generator
    └── visualizer.py                # Visualization utilities
```

## Integration Points

### With Portfolio Module
- `portfolio.Portfolio`: Equity portfolios
- `portfolio.fi.FIPortfolio`: Fixed Income portfolios
- Engine automatically detects portfolio type and uses appropriate engine

### With Pricing Module (`priceenv/`)
- `PricingEnvironment`: Market data container
- Parameters: spot, volatility surface, rate curve, dividend yield
- Engines update these parameters day-by-day during simulation

### With Asset Engines
- `asset.equity.engine.analytical.BlackScholesEngine`: Option pricing
- `asset.equity.engine.analytical.DeltaOneEngine`: Spot instruments
- `asset.bond.engine.BondDiscountEngine`: Bond pricing
- Used to re-price positions at each day

### With Risk Measures
- `asset.equity.riskmeasures.GreeksCalculator`: Equity Greeks
- FI risk measures integrated in `FIPortfolio`

### With Backtest Module
- `backtest.strategy`: Hedging strategies (DeltaNeutralStrategy, etc.)
- `backtest.transaction_costs`: Transaction cost models
- Dynamic scenario engines integrate these for optional hedging

### With Stress Test Module
- `stresstest.stress.stress_types`: StressType enum (PERCENTAGE, ABSOLUTE)
- ParameterChange uses these stress types

## Configuration Reference

### DynamicScenarioConfig (Equity)

```python
DynamicScenarioConfig(
    # Greeks calculation
    calculate_greeks: bool = True,
    greeks_method: str = 'analytical',  # or 'numerical'

    # Export settings
    export_formats: List[str] = ['parquet'],
    output_dir: str = './dynamic_results',
    save_detailed_results: bool = True,
    save_intermediate_states: bool = True,

    # Report settings
    generate_report: bool = True,
    include_charts: bool = True,

    # Metadata
    metadata: Dict[str, Any] = {},
)
```

### FIDynamicScenarioConfig (Fixed Income)

```python
FIDynamicScenarioConfig(
    # Risk measures
    calculate_dv01: bool = True,
    calculate_convexity: bool = True,
    calculate_duration: bool = True,
    calculate_key_rate_dv01: bool = False,
    key_rate_tenors: List[int] = [1, 2, 5, 10, 30],

    # Hedging
    hedge_enabled: bool = False,
    hedge_dv01_threshold: float = 50000.0,
    futures_dv01_per_contract: float = 80.0,

    # Export
    export_formats: List[str] = ['parquet'],
    output_dir: str = './dynamic_results',
)
```

## Examples

See these example files for complete demonstrations:

1. **`/example/dynamic_scenario_demo.py`**: Comprehensive equity demonstration
   - Path creation (PathBuilder, PathLibrary)
   - Basic scenarios without hedging
   - Scenarios with delta hedging
   - Market crash scenarios
   - Hedged vs unhedged comparison
   - Export, visualization, and reporting

2. **`/example/dynamic_scenario_fi_demo.py`**: Fixed Income demonstration
   - Creating FI portfolios with Treasury bonds
   - Rate scenario patterns (parallel shift, steepener, rate hike)
   - DV01 hedging with bond futures
   - FI-specific visualization and reporting

Run examples:
```bash
python example/dynamic_scenario_demo.py
python example/dynamic_scenario_fi_demo.py
```

## Testing

### Test Location
- `test/test_dynamic_scenario.py`: Main test suite (if exists)

### Test Categories
1. **Path Creation Tests**
   - PathBuilder fluent API
   - PathLibrary patterns
   - FIPathLibrary patterns
   - Manual path construction

2. **Engine Tests**
   - Basic scenario execution
   - Hedging integration
   - Transaction cost application
   - Error handling

3. **Results Tests**
   - Result object creation
   - DataFrame accessors
   - Export functionality
   - Serialization

4. **Integration Tests**
   - Portfolio integration
   - Pricing engine integration
   - Risk measure calculation

### Running Tests
```bash
# Run all dynamic scenario tests
python -m pytest test/test_dynamic_scenario.py -v

# Run specific test
python -m pytest test/test_dynamic_scenario.py::TestPathLibrary -v

# Run with coverage
python -m pytest test/test_dynamic_scenario.py --cov=dynamicscenario
```

## Current State & Capabilities

### ✅ Completed Features

1. **Equity Support**
   - Multi-day scenario simulation
   - Greeks calculation (analytical and numerical)
   - Delta-neutral hedging integration
   - Transaction cost modeling
   - Predefined path patterns (rally, crash, V-shaped, etc.)
   - Historical scenarios (Black Monday, COVID crash)
   - Rich results with day-by-day evolution
   - Visualization and HTML reporting
   - Export to CSV/JSON/Parquet

2. **Fixed Income Support**
   - DV01/duration/convexity calculation
   - DV01 hedging with bond futures
   - Rate scenario patterns (parallel shift, steepener, flattener)
   - Rate cycle scenarios (hike/cut cycles)
   - Historical Fed tightening scenarios
   - FI-specific visualization

3. **Architecture**
   - Protocol-based design for extensibility
   - Asset-class-agnostic base classes
   - Factory pattern for engine selection
   - Modular path construction (Builder + Library patterns)
   - Comprehensive results objects

### 📋 Current TODOs & Future Enhancements

Based on code analysis, potential TODOs include:

#### High Priority

1. **Multi-Asset Portfolio Support**
   - Currently engines support single-asset portfolios
   - **TODO**: Extend to multi-asset portfolios (equity + FI in same portfolio)
   - **Challenge**: Different risk metrics (Greeks vs DV01) in same simulation
   - **Approach**: Use RiskMetricsAdapter protocol more extensively

2. **Key Rate DV01 for FI**
   - Currently FI config has `calculate_key_rate_dv01` flag
   - **TODO**: Implement key rate DV01 calculation and tracking
   - **Impact**: Better visualization of curve risk
   - **Files**: `fi/results.py`, `fi/engine.py`

3. **Position-Level Stress Testing**
   - Current stress testing applies to entire portfolio or underlying
   - **TODO**: Add Position-level stress (different shock per position)
   - **Approach**: Extend `ParameterChange` to support position_id target

4. **Enhanced Hedging Strategies**
   - Currently uses backtest strategies (DeltaNeutralStrategy)
   - **TODO**: Add FI-specific hedging strategies (DV01-neutral, duration-neutral)
   - **Files**: New module `fi/hedging.py` or extend `backtest.strategy`

5. **Correlation and Multi-Factor Models**
   - Current paths are independent parameter changes
   - **TODO**: Add correlated factor movements (e.g., equity down + vol up)
   - **Approach**: PathBuilder enhancements for correlated changes

#### Medium Priority

6. **Monte Carlo Scenario Generation**
   - Currently only predefined or manual paths
   - **TODO**: Add stochastic path generation (geometric Brownian motion, etc.)
   - **Use case**: Generate many scenarios for Monte Carlo VaR
   - **Files**: New module `path/stochastic_paths.py`

7. **Scenario Optimization**
   - No optimization of hedging frequency/costs
   - **TODO**: Optimize hedge timing to minimize P&L variance vs transaction costs
   - **Approach**: Dynamic programming or optimization routine

8. **Real-Time Scenario Execution**
   - Currently batch simulation
   - **TODO**: Support streaming scenarios (update portfolio daily)
   - **Use case**: Production risk management
   - **Files**: New module `realtime/` or extend `engine.py`

9. **Advanced Visualization**
   - Current plots are basic (line charts, bar charts)
   - **TODO**: Add heatmaps (correlation, P&L attribution)
   - **TODO**: Add interactive dashboards (currently HTML with basic charts)
   - **Files**: `report/visualizer.py`

10. **Scenario Library Persistence**
    - PathLibrary and FIPathLibrary are code-based
    - **TODO**: Load scenarios from JSON/YAML files
    - **Use case**: Risk team configures scenarios without code changes
    - **Files**: New module `persistence/` or extend `path/`

#### Low Priority

11. **Parallel Execution**
    - Currently single-threaded
    - **TODO**: Parallel scenario execution for Monte Carlo
    - **Use case**: Run 1000 scenarios faster
    - **Approach**: Use `concurrent.futures` or `multiprocessing`

12. **Scenario Comparison**
    - No built-in comparison utilities
    - **TODO**: Compare multiple scenarios side-by-side
    - **Files**: New module `comparison.py`

13. **Backtesting Integration**
    - Limited integration with `backtest/` module
    - **TODO**: Run backtest strategies through dynamic scenarios
    - **Use case**: Test hedging strategy performance under stress

14. **Extreme Value Theory**
    - No tail risk modeling
    - **TODO**: Fit EVT distributions to scenarios
    - **Use case**: Estimate tail risk beyond historical scenarios

15. **Credit Spread Scenarios**
    - Currently supports equity and rates
    - **TODO**: Add credit spread scenarios (for corporate bonds)
    - **Asset class**: Extend to credit portfolios

### 🔧 Known Limitations & Workarounds

1. **Single Underlying per Portfolio**
   - **Limitation**: Each pricing environment supports one underlying
   - **Workaround**: Create multiple environments (one per underlying)
   - **Note**: This works for equity but needs testing for FI

2. **Flat Volatility Surface Only**
   - **Limitation**: Equity paths use `FlatVolSurface`
   - **Workaround**: For term structure, extend to `VolatilitySurface`
   - **Files**: `param/` module

3. **Limited Historical Scenarios**
   - Only Black Monday and COVID crash
   - **Workaround**: Use PathBuilder to create custom historical replications
   - **Enhancement**: Add 2008 Financial Crisis, Flash Crash, etc.

4. **Transaction Cost Simplification**
   - Uses proportional cost model
   - **Workaround**: Extend `backtest.transaction_costs` for more models
   - **Enhancement**: Add bid-ask spread, market impact models

## Design Decisions & Rationale

### Why Protocol-Based Architecture?

1. **Separation of Concerns**: Base protocols define interface, implementations handle details
2. **Extensibility**: Easy to add new asset classes without modifying core
3. **Type Safety**: Protocols provide runtime type checking
4. **Testing**: Can mock protocols for unit tests

### Why Day-by-Day Evolution?

1. **Realism**: Markets evolve gradually, not instantaneously
2. **Hedging**: Hedge rebalancing happens over time
3. **P&L Attribution**: Can see when losses/gains occur
4. **Transaction Costs**: Costs accrue over time

### Why Separate Equity and FI Engines?

1. **Different Risk Metrics**: Greeks vs DV01/duration
2. **Different Hedging**: Delta hedging vs DV01 hedging
3. **Different Parameter Spaces**: Spot/vol vs rates/curve
4. **Optimization**: Can optimize each engine independently

### Why PathBuilder + PathLibrary?

1. **Flexibility**: PathBuilder for custom scenarios
2. **Reusability**: PathLibrary for common patterns
3. **Readability**: Fluent API is self-documenting
4. **Validation**: Centralized validation in one place

## Common Patterns & Anti-Patterns

### ✅ Good Patterns

1. **Use PathLibrary for Common Scenarios**
   ```python
   path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
   ```

2. **Enable Hedging for Realistic Simulation**
   ```python
   results = engine.run(portfolio, path, hedge_strategy=strategy)
   ```

3. **Use Appropriate Greeks Method**
   ```python
   # For speed: analytical
   config.greeks_method = 'analytical'

   # For accuracy: numerical (slower)
   config.greeks_method = 'numerical'
   ```

4. **Save Detailed Results for Analysis**
   ```python
   config.save_detailed_results = True
   ```

### ❌ Anti-Patterns

1. **Creating New Engine Instead of Using Factory**
   ```python
   # Bad: Direct instantiation
   engine = DynamicScenarioEngine(config)

   # Good: Use factory
   engine = get_engine_for_portfolio(portfolio, config)
   ```

2. **Manually Creating Paths When Library Exists**
   ```python
   # Bad: Manual construction
   steps = [DayStep(i, [...]) for i in range(5)]
   path = DayPath(steps=steps)

   # Good: Use library
   path = PathLibrary.consecutive_rally(days=5, daily_pct=0.02)
   ```

3. **Not Cloning Portfolio Before Simulation**
   ```python
   # Bad: Modifies original portfolio
   engine.run(portfolio, path)

   # Good: Engine creates working copy internally
   results = engine.run(portfolio, path)
   ```

## Performance Considerations

1. **Greeks Calculation**
   - Analytical: Fast, suitable for most cases
   - Numerical: Slower (5-10x), use only when necessary
   - **Tip**: Use analytical for path simulation, numerical for final validation

2. **Hedging Frequency**
   - Daily hedging: More accurate, higher transaction costs
   - Threshold-based: Balance between accuracy and cost
   - **Tip**: Test different thresholds to find optimal balance

3. **Result Export**
   - Parquet: Fastest for large datasets
   - CSV: Human-readable, slower
   - JSON: Good for web APIs, slower

4. **Visualization**
   - PNG: Fast, static
   - HTML: Interactive, slower
   - **Tip**: Generate PNG for reports, HTML for exploration

## Debugging Tips

1. **Enable Detailed Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check Day-by-Day Results**
   ```python
   for day in results.day_results:
       print(f"Day {day.day_index}: P&L ${day.daily_pnl:,.2f}")
   ```

3. **Validate Market Evolution**
   ```python
   market_df = results.get_market_evolution()
   print(market_df[['day_index', 'spot', 'volatility', 'rate']])
   ```

4. **Inspect Hedge Trades**
   ```python
   trades_df = results.get_trades_dataframe()
   print(trades_df[['day_index', 'quantity', 'price', 'transaction_cost']])
   ```

5. **Check Worst/Best Days**
   ```python
   worst = results.get_worst_day()
   best = results.get_best_day()
   ```

## Common Errors & Solutions

### Error: "Portfolio must contain at least one position"
**Cause**: Empty portfolio passed to engine
**Solution**: Add positions to portfolio before running

### Error: "Day path must have at least one day"
**Cause**: Empty or invalid DayPath
**Solution**: Ensure path has at least one DayStep

### Error: "Spot cannot be <= 0"
**Cause**: ParameterChange caused spot to go negative
**Solution**: Use percentage changes instead of absolute, or add bounds checking

### Error: "No engine available for portfolio type"
**Cause**: Portfolio is not Portfolio or FIPortfolio
**Solution**: Use correct portfolio type or extend engine factory

### Error: "Invalid export format"
**Cause**: Unsupported format in export_formats
**Solution**: Use only 'csv', 'json', 'parquet', 'html'

## Extending the Module

### Adding a New Asset Class

1. **Create Asset-Specific Results Class**
   ```python
   # results/myasset_results.py
   from dynamicscenario.base import BaseDayResult, BaseScenarioResults

   class MyAssetDayResult(BaseDayResult):
       my_risk_metric: float = 0.0

   class MyAssetScenarioResults(BaseScenarioResults):
       asset_class: str = "my_asset"
   ```

2. **Create Risk Metrics Adapter**
   ```python
   # myasset/risk_adapter.py
   from dynamicscenario.base import RiskMetricsAdapter

   class MyAssetRiskAdapter(RiskMetricsAdapter):
       def compute_metrics(self, portfolio, pricing_environments):
           # Compute my_asset-specific risk metrics
           return {'my_metric': 123.4}

       def get_metric_names(self):
           return ['my_metric']
   ```

3. **Create Engine**
   ```python
   # myasset/engine.py
   from dynamicscenario.base import BaseDynamicScenarioEngine

   class MyAssetDynamicScenarioEngine(BaseDynamicScenarioEngine):
       def supports_portfolio(self, portfolio):
           return isinstance(portfolio, MyAssetPortfolio)

       def run(self, portfolio, day_path, hedge_strategy=None):
           # Implement simulation
           pass

       def get_asset_class(self):
           return "my_asset"
   ```

4. **Update Factory Function**
   ```python
   # base.py - get_engine_for_portfolio()
   elif isinstance(portfolio, MyAssetPortfolio):
       from myasset.engine import MyAssetDynamicScenarioEngine
       return MyAssetDynamicScenarioEngine(config)
   ```

5. **Create Path Library**
   ```python
   # path/myasset_path_library.py
   class MyAssetPathLibrary:
       @staticmethod
       def my_scenario(days, param1):
           # Create scenario-specific path
           pass
   ```

6. **Update Exports**
   ```python
   # __init__.py
   from myasset import (
       MyAssetDynamicScenarioEngine,
       MyAssetScenarioResults,
       MyAssetPathLibrary,
   )
   ```

### Adding a New Path Pattern

1. **To PathLibrary (Equity)**
   ```python
   # path/path_library.py
   class PathLibrary:
       @staticmethod
       def my_new_pattern(days, param1, param2):
           """Description of the pattern."""
           steps = []
           for day in range(days):
               changes = [
                   ParameterChange("spot", StressType.PERCENTAGE, param1),
                   ParameterChange("volatility", StressType.PERCENTAGE, param2),
               ]
               steps.append(DayStep(day, changes))

           return DayPath(
               name="My New Pattern",
               steps=steps,
               description=f"Custom pattern with {param1} and {param2}"
           )
   ```

2. **To FIPathLibrary (Fixed Income)**
   ```python
   # path/fi_path_library.py
   class FIPathLibrary:
       @staticmethod
       def my_new_rate_pattern(days, param1):
           """Description of the pattern."""
           steps = []
           for day in range(days):
               changes = [
                   ParameterChange("rate", StressType.BASIS_POINTS, param1),
               ]
               steps.append(DayStep(day, changes))

           return DayPath(
               name="My New Rate Pattern",
               steps=steps,
               description=f"Rate pattern with {param1}"
           )
   ```

## References

### Internal Dependencies
- `portfolio/`: Portfolio management
- `priceenv/`: Pricing environments
- `asset/`: Pricing engines and products
- `backtest/`: Hedging strategies
- `stresstest/`: Stress types and levels
- `util/`: Utilities and enums

### External Dependencies
- `pandas`: DataFrame operations
- `numpy`: Numerical computations
- `matplotlib`: Plotting (PNG)
- `plotly`: Interactive plots (HTML)

## Support & Contribution

### Getting Help
1. Check this guide first
2. Review `/dynamicscenario/README.md` for user documentation
3. Look at example files in `/example/`
4. Check tests in `/test/test_dynamic_scenario.py`

### Reporting Issues
Include:
1. Scenario configuration
2. Portfolio structure
3. Expected vs actual behavior
4. Full error traceback
5. Minimal reproducible example

### Contributing
1. Follow existing code patterns
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility
5. Run full test suite before submitting

---

**Module Version**: 1.0.0 (as of 2024)
**Last Updated**: 2024-12-08
**Maintainer**: QuantArk Development Team
