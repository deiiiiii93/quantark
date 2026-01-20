# Stress Test Module - AI Agent Guide

## Purpose

This guide is specifically for AI agents working with the QuantArk stress test module. It provides targeted guidance on common tasks, patterns, and pitfalls to help you work effectively with the stress testing framework.

## Quick Start for AI Agents

### Understanding the Stress Test Module Structure

```
stresstest/
├── base.py                    # Protocol interfaces (shared)
├── config.py                  # Configuration (shared)
├── engine.py                  # Main engine (backward compatibility)
├── README.md                  # User-facing documentation
├── IMPLEMENTATION_SUMMARY.md  # Implementation notes
├── scenario/                  # Scenario management
│   ├── scenario.py            # Core Scenario/Stress classes
│   ├── scenario_builder.py    # Fluent builder API
│   ├── scenario_library.py    # Predefined scenarios
│   └── scenario_storage.py    # YAML/JSON persistence
├── stress/                    # Stress application
│   ├── stress_types.py        # StressType, StressLevel enums
│   └── stress_applicator.py   # Apply stresses to environments
├── results/                   # Results management
│   ├── stress_results.py      # Results containers
│   ├── result_aggregator.py   # Analysis utilities
│   └── result_exporter.py     # Export to formats
├── report/                    # Reporting
│   ├── report_generator.py    # HTML reports
│   └── visualizer.py          # Static/interactive plots
├── equity/                    # Equity-specific implementation
│   ├── engine.py             # EquityStressEngine
│   ├── config.py             # Equity config
│   ├── results.py            # Equity results
│   └── report/               # Equity reporting
└── fi/                        # Fixed Income implementation
    ├── engine.py             # FI stress engine
    ├── config.py             # FI config
    ├── results.py            # FI results
    └── metrics.py            # FI-specific metrics
```

### Common Imports

```python
# Core engine and config
from stresstest.equity.engine import EquityStressEngine, StressTestEngine
from stresstest.config import StressTestConfig
from stresstest.equity.config import EquityStressTestConfig

# Scenarios
from stresstest.scenario.scenario import Scenario, Stress
from stresstest.scenario.scenario_builder import ScenarioBuilder
from stresstest.scenario.scenario_library import ScenarioLibrary
from stresstest.scenario.scenario_storage import ScenarioStorage

# Stress types
from stresstest.stress.stress_types import StressType, StressLevel

# Results
from stresstest.results.stress_results import StressTestResults
from stresstest.results.result_aggregator import ResultAggregator
from stresstest.results.result_exporter import ResultExporter

# Reporting
from stresstest.report import ReportGenerator, StressTestVisualizer
```

## Task-Oriented Guidance

### Task 1: Adding a New Predefined Scenario

**When**: When you need to add a new common scenario to the ScenarioLibrary.

**Steps**:

1. **Add scenario method** in `stresstest/scenario/scenario_library.py`:

```python
@staticmethod
def your_custom_scenario() -> Scenario:
    """
    Your custom predefined scenario.

    Describe what this scenario tests and when to use it.

    Returns:
        Scenario with your stresses
    """
    return ScenarioBuilder() \
        .name("Your Custom Scenario") \
        .description("Description of the scenario") \
        .spot_stress(-0.20)  # Example stress
        .vol_stress(0.50)
        # Add more stresses as needed
        .build()
```

2. **Add to exports** in `stresstest/scenario/__init__.py`:

```python
from stresstest.scenario.scenario_library import ScenarioLibrary

__all__ = [
    # ... existing exports
    "ScenarioLibrary",
]
```

3. **Add tests** in `test/test_stress_test.py`:

```python
def test_scenario_library_your_custom_scenario():
    """Test your custom scenario."""
    scenario = ScenarioLibrary.your_custom_scenario()

    # Validate scenario
    assert scenario.name == "Your Custom Scenario"
    assert len(scenario.stresses) > 0

    # Validate specific stresses
    spot_stress = next((s for s in scenario.stresses if s.parameter == "spot"), None)
    assert spot_stress is not None
    assert spot_stress.stress_value == -0.20
    assert spot_stress.stress_type == StressType.PERCENTAGE
```

**Requirements**:
- ✅ Use descriptive name and description
- ✅ Follow existing pattern (ScenarioBuilder().name().description().stress().build())
- ✅ Validate in tests
- ✅ Document when to use this scenario

### Task 2: Adding a New Asset Class

**When**: When you need to support stress testing for a new asset class (e.g., Crypto, Commodities, FX).

**Steps**:

1. **Create asset-specific directory**:
   ```bash
   mkdir -p stresstest/crypto
   ```

2. **Create config** in `stresstest/crypto/config.py`:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from stresstest.config import StressTestConfig

@dataclass
class CryptoStressTestConfig(StressTestConfig):
    """Configuration for crypto stress tests."""

    # Add crypto-specific configuration
    include_funding_rates: bool = True
    include_borrow_rates: bool = True

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        summary = super().get_summary()
        summary.update({
            "include_funding_rates": self.include_funding_rates,
            "include_borrow_rates": self.include_borrow_rates,
        })
        return summary
```

3. **Create results** in `stresstest/crypto/results.py`:

```python
from dataclasses import dataclass
from typing import Dict, Any, List
from stresstest.base import ScenarioEnvelope

@dataclass
class CryptoScenarioEnvelope(ScenarioEnvelope):
    """Crypto-specific scenario result."""
    # Add crypto-specific fields
    funding_rate_impact: Optional[float] = None
    borrow_rate_impact: Optional[float] = None
    extra_metrics: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize crypto-specific metrics."""
        super().__post_init__()
        if self.extra_metrics is None:
            self.extra_metrics = {}
```

4. **Create metrics adapter** in `stresstest/crypto/metrics.py`:

```python
from stresstest.base import StressMetricsAdapter
from portfolio.base import BasePortfolio

class CryptoStressMetricsAdapter(StressMetricsAdapter):
    """Calculate crypto-specific stress metrics."""

    def supports(self, portfolio: BasePortfolio) -> bool:
        """Check if supports crypto portfolio."""
        from portfolio.crypto.portfolio import CryptoPortfolio
        return isinstance(portfolio, CryptoPortfolio)

    def compute_metrics(
        self,
        original_portfolio: BasePortfolio,
        stressed_portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
        stressed_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate crypto-specific metrics."""
        # Calculate funding rate impact
        funding_impact = self._calculate_funding_impact(
            original_portfolio, stressed_portfolio
        )

        # Calculate borrow rate impact
        borrow_impact = self._calculate_borrow_impact(
            original_portfolio, stressed_portfolio
        )

        return {
            "crypto": {
                "funding_rate_impact": funding_impact,
                "borrow_rate_impact": borrow_impact,
            }
        }
```

5. **Create scenario runner** in `stresstest/crypto/runner.py`:

```python
from stresstest.base import ScenarioRunner
from stresstest.scenario.scenario import Scenario

class CryptoScenarioRunner(ScenarioRunner):
    """Evaluate scenarios on crypto portfolios."""

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> CryptoScenarioEnvelope:
        """Evaluate single scenario on crypto portfolio."""
        # 1. Apply stresses
        stressed_portfolio = StressApplicator.apply_scenario_to_portfolio(
            portfolio, scenario
        )

        # 2. Calculate stressed value
        stressed_value = stressed_portfolio.value()

        # 3. Calculate crypto-specific metrics
        # ... implementation

        # 4. Create envelope
        return CryptoScenarioEnvelope(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=stressed_value - baseline_value,
            portfolio_pnl_pct=(stressed_value - baseline_value) / baseline_value,
            # ... other fields
        )
```

6. **Create engine** in `stresstest/crypto/engine.py`:

```python
from stresstest.base import BaseStressEngine
from stresstest.config import StressTestConfig

class CryptoStressEngine(BaseStressEngine):
    """Crypto stress testing engine."""

    def __init__(
        self,
        config: CryptoStressTestConfig,
        metrics_adapter: Optional[StressMetricsAdapter] = None,
    ):
        self.config = config
        self.metrics_adapter = metrics_adapter or CryptoStressMetricsAdapter()
        self.scenario_runner = CryptoScenarioRunner()

    def supports_portfolio(self, portfolio: BasePortfolio) -> bool:
        """Check if supports portfolio type."""
        from portfolio.crypto.portfolio import CryptoPortfolio
        return isinstance(portfolio, CryptoPortfolio)

    def run_static_scenarios(
        self,
        portfolio: BasePortfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> StressResultEnvelope:
        """Run static scenario analysis."""
        # Implementation similar to EquityStressEngine
        pass

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> CryptoScenarioEnvelope:
        """Evaluate single scenario."""
        return self.scenario_runner.evaluate_scenario(
            portfolio, scenario, baseline_value
        )
```

7. **Create __init__.py** in `stresstest/crypto/__init__.py`:

```python
from stresstest.crypto.config import CryptoStressTestConfig
from stresstest.crypto.engine import CryptoStressEngine
from stresstest.crypto.results import CryptoScenarioEnvelope

__all__ = [
    "CryptoStressTestConfig",
    "CryptoStressEngine",
    "CryptoScenarioEnvelope",
]
```

8. **Add to main exports** in `stresstest/__init__.py`:

```python
from stresstest.crypto import (
    CryptoStressTestConfig,
    CryptoStressEngine,
    CryptoScenarioEnvelope,
)

__all__ = [
    # ... existing exports
    "CryptoStressTestConfig",
    "CryptoStressEngine",
    "CryptoScenarioEnvelope",
]
```

### Task 3: Adding New Stress Types

**When**: When you need to support new types of stress application or new stress parameters.

**Steps**:

1. **Add to StressType enum** in `stresstest/stress/stress_types.py`:

```python
class StressType(Enum):
    """Type of stress to apply."""
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    VALUE = "value"
    YOUR_NEW_TYPE = "your_new_type"  # Add new type

    def apply(self, current_value: float, stress_value: float) -> float:
        """Apply stress to current value."""
        if self == StressType.ABSOLUTE:
            return current_value + stress_value
        elif self == StressType.PERCENTAGE:
            return current_value * (1.0 + stress_value)
        elif self == StressType.VALUE:
            return stress_value
        elif self == StressType.YOUR_NEW_TYPE:  # Add handling
            # Your logic here
            return your_implementation(current_value, stress_value)
        else:
            raise ValueError(f"Unknown stress type: {self}")
```

2. **Register adapter** in `stresstest/stress/stress_applicator.py`:

```python
# Add new stress parameter handler
def apply_your_new_stress(env, stress):
    """Apply your new stress type."""
    # Your implementation
    pass

# Register in __init__ or register_adapter method
StressApplicator.register_adapter("your_parameter", apply_your_new_stress)
```

3. **Add builder method** in `stresstest/scenario/scenario_builder.py`:

```python
def your_param_stress(
    self,
    stress_value: float,
    stress_type: StressType = StressType.PERCENTAGE,
    level: StressLevel = StressLevel.PORTFOLIO,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> 'ScenarioBuilder':
    """Add your parameter stress."""
    stress = Stress(
        parameter="your_parameter",
        stress_type=stress_type,
        stress_value=stress_value,
        level=level,
        target=target,
        metadata=metadata or {},
    )
    self._scenario.stresses.append(stress)
    return self
```

4. **Add tests** in `test/test_stress_test.py`:

```python
def test_stress_type_your_new_type():
    """Test your new stress type."""
    # Test apply method
    result = StressType.YOUR_NEW_TYPE.apply(100, 10)
    assert result == expected_value

    # Test in scenario
    scenario = (ScenarioBuilder()
        .name("Test")
        .your_param_stress(10)
        .build())

    assert len(scenario.stresses) == 1
    assert scenario.stresses[0].parameter == "your_parameter"
    assert scenario.stresses[0].stress_type == StressType.YOUR_NEW_TYPE
```

### Task 4: Adding Custom Stress Parameters

**When**: When you need to stress parameters beyond spot, volatility, rate, etc.

**Steps**:

1. **Register adapter** for custom parameter:

```python
from stresstest.stress.stress_applicator import StressApplicator

def apply_custom_parameter(env, stress):
    """Apply custom parameter stress."""
    # Get current value
    current_value = env.get_custom_parameter(stress.parameter)

    # Apply stress
    stressed_value = stress.stress_type.apply(current_value, stress.stress_value)

    # Set stressed value
    env.set_custom_parameter(stress.parameter, stressed_value)

# Register adapter
StressApplicator.register_adapter("custom_param", apply_custom_parameter)
```

2. **Use in scenario builder**:

```python
scenario = (ScenarioBuilder()
    .name("Custom Stress")
    .spot_stress(-0.20)  # Built-in
    .custom_param_stress(0.15)  # Your custom parameter
    .build())
```

3. **Add builder helper** (optional):

```python
# In ScenarioBuilder class
def custom_param_stress(
    self,
    stress_value: float,
    stress_type: StressType = StressType.PERCENTAGE,
    level: StressLevel = StressLevel.PORTFOLIO,
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> 'ScenarioBuilder':
    """Add custom parameter stress."""
    stress = Stress(
        parameter="custom_param",
        stress_type=stress_type,
        stress_value=stress_value,
        level=level,
        target=target,
        metadata=metadata or {},
    )
    self._scenario.stresses.append(stress)
    return self
```

### Task 5: Adding New Export Formats

**When**: When you need to export results to a new format (e.g., Excel, HDF5, database).

**Steps**:

1. **Add export method** in `stresstest/results/result_exporter.py`:

```python
@staticmethod
def export_to_your_format(
    results: StressTestResults,
    output_path: str,
    **kwargs,
) -> None:
    """Export to your custom format."""
    # Convert results to your format
    # Write to output_path

    # Example: Excel export
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = results.to_summary_dataframe()
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Position details
        for scenario_result in results.envelope.scenario_results:
            position_df = results.to_position_dataframe(scenario_result.scenario.name)
            sheet_name = scenario_result.scenario.name[:31]  # Excel limit
            position_df.to_excel(writer, sheet_name=sheet_name, index=False)
```

2. **Add batch export support**:

```python
@(
    results: StressTestResults,
    output_dir:staticmethod
def export str,
    formats: List[str],
    base_name: str = "stress_test",
) -> None:
    """Export to multiple formats."""
    # Existing formats
    if 'parquet' in formats:
        ResultExporter.export_to_parquet(results, output_dir, include_positions=True)

    if 'csv' in formats:
        ResultExporter.export_to_csv(results, output_dir)

    if 'json' in formats:
        ResultExporter.export_to_json(results, f"{output_dir}/{base_name}.json")

    # Add your format
    if 'your_format' in formats:
        ResultExporter.export_to_your_format(
            results,
            f"{output_dir}/{base_name}.your_ext"
        )
```

3. **Add tests**:

```python
def test_result_exporter_your_format(tmp_path):
    """Test export to your format."""
    results = create_test_results()
    output_path = tmp_path / "test.your_ext"

    ResultExporter.export_to_your_format(results, str(output_path))

    assert output_path.exists()
    # Validate your format
```

### Task 6: Adding New Visualization Types

**When**: When you need to create custom plots or charts for stress test results.

**Steps**:

1. **Add plot method** in `stresstest/report/visualizer.py`:

```python
def plot_your_custom_visualization(
    self,
    results: StressTestResults,
    output_path: str,
    **kwargs,
) -> plt.Figure:
    """
    Create your custom visualization.

    Args:
        results: Stress test results
        output_path: Path to save plot
        **kwargs: Additional parameters

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get data for your visualization
    # Example: Correlation heatmap
    correlation_data = self._calculate_correlation_matrix(results)

    # Create plot
    sns.heatmap(correlation_data, annot=True, cmap='RdYlBu_r', ax=ax)

    # Customize
    ax.set_title("Scenario Correlation Matrix")
    ax.set_xlabel("Scenarios")
    ax.set_ylabel("Scenarios")

    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig
```

2. **Add batch generation support**:

```python
def create_all_plots(
    self,
    results: StressTestResults,
    output_dir: str,
    prefix: str = "stress_test",
) -> None:
    """Generate all plots."""
    # Existing plots
    self.plot_pnl_waterfall(results, f"{output_dir}/{prefix}_waterfall.png")
    self.plot_pnl_distribution(results, f"{output_dir}/{prefix}_distribution.png")
    self.plot_scenario_comparison(results, f"{output_dir}/{prefix}_comparison.png")

    # Add your plot
    self.plot_your_custom_visualization(
        results,
        f"{output_dir}/{prefix}_custom.png"
    )
```

### Task 7: Fixing Bugs in Stress Application

**When**: When you encounter issues with stress application, scenario evaluation, or results calculation.

**Debugging Steps**:

1. **Check stress definition**:

```python
# Validate stress parameters
for stress in scenario.stresses:
    print(f"Parameter: {stress.parameter}")
    print(f"Type: {stress.stress_type}")
    print(f"Value: {stress.stress_value}")
    print(f"Level: {stress.level}")
    print(f"Target: {stress.target}")
```

2. **Check stress application**:

```python
# Apply stress and check result
stressed_portfolio = StressApplicator.apply_scenario_to_portfolio(
    portfolio, scenario
)

# Get stress summary
summary = StressApplicator.get_stress_summary(
    original_env,
    stressed_env
)
print(summary)
```

3. **Check scenario evaluation**:

```python
# Evaluate single scenario
result = engine.evaluate_scenario(portfolio, scenario, baseline_value)

print(f"Scenario: {result.scenario.name}")
print(f"Baseline value: {baseline_value:,.2f}")
print(f"Stressed value: {result.portfolio_value:,.2f}")
print(f"P&L: {result.portfolio_pnl:,.2f}")
print(f"P&L %: {result.portfolio_pnl_pct:.2%}")
```

4. **Validate results**:

```python
# Check result consistency
assert abs(result.portfolio_pnl - (result.portfolio_value - baseline_value)) < 0.01
assert abs(result.portfolio_pnl_pct - (result.portfolio_pnl / baseline_value)) < 0.0001

# Check position results
for pos_result in result.position_results:
    pos_pnl = pos_result['pnl']
    pos_value = pos_result['value']
    # Validate position calculations
```

**Common Bugs and Solutions**:

1. **Bug**: Stress not applied correctly
   - **Cause**: Wrong stress type or level
   - **Fix**: Check stress definition and adapter registration

2. **Bug**: Negative volatility after stress
   - **Cause**: Percentage stress on low volatility
   - **Fix**: Use VALUE type to set minimum vol

3. **Bug**: Position-level stress not working
   - **Cause**: Wrong position_id or level
   - **Fix**: Verify position exists and target is correct

4. **Bug**: Greeks calculation fails
   - **Cause**: Unsupported product for analytical Greeks
   - **Fix**: Use numerical Greeks or check product support

### Task 8: Performance Optimization

**When**: When stress tests are too slow for large portfolios or many scenarios.

**Optimization Strategies**:

1. **Disable Greeks for Large Portfolios**:

```python
# Fast: Skip Greeks calculation
config = StressTestConfig(
    calculate_greeks=False,
    save_detailed_results=False
)
```

2. **Reduce Position Details**:

```python
# Skip position-level details for speed
config = StressTestConfig(
    save_detailed_results=False
)
```

3. **Use Parquet for Large Datasets**:

```python
# Efficient export format
config = StressTestConfig(
    export_formats=['parquet']
)
```

4. **Parallel Execution** (Future):

```python
# Enable parallel execution (when implemented)
config = StressTestConfig(
    parallel_execution=True,
    max_workers=4
)
```

5. **Filter Scenarios**:

```python
# Only run relevant scenarios
relevant_scenarios = [
    s for s in all_scenarios
    if s.metadata.get('category') == 'equity'
]
```

### Task 9: Adding Scenario Analysis Utilities

**When**: When you need new analysis methods for comparing and analyzing scenarios.

**Steps**:

1. **Add method** in `stresstest/results/result_aggregator.py`:

```python
@staticmethod
def calculate_scenario_correlation(
    results: StressTestResults,
) -> pd.DataFrame:
    """Calculate correlation between scenarios."""
    # Get P&L matrix
    pnl_matrix = results.to_summary_dataframe()[['scenario_name', 'portfolio_pnl']]
    pnl_matrix = pnl_matrix.pivot_table(
        index=results.envelope.scenario_results[0].scenario.name,
        columns='scenario_name',
        values='portfolio_pnl'
    )

    # Calculate correlation
    correlation = pnl_matrix.corr()

    return correlation

@staticmethod
def identify_outlier_scenarios(
    results: StressTestResults,
    threshold: float = 2.0,
) -> List[Scenario]:
    """Identify scenarios beyond threshold standard deviations."""
    pnls = [
        result.portfolio_pnl
        for result in results.envelope.scenario_results
    ]

    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)

    outliers = []
    for result in results.envelope.scenario_results:
        z_score = abs((result.portfolio_pnl - mean_pnl) / std_pnl)
        if z_score > threshold:
            outliers.append(result.scenario)

    return outliers
```

2. **Add tests**:

```python
def test_result_aggregator_scenario_correlation():
    """Test scenario correlation calculation."""
    results = create_test_results()
    correlation = ResultAggregator.calculate_scenario_correlation(results)

    assert isinstance(correlation, pd.DataFrame)
    assert correlation.shape[0] == len(results.envelope.scenario_results)
    assert correlation.shape[1] == len(results.envelope.scenario_results)

def test_result_aggregator_identify_outliers():
    """Test outlier identification."""
    results = create_test_results()
    outliers = ResultAggregator.identify_outlier_scenarios(results, threshold=2.0)

    assert isinstance(outliers, list)
    assert all(isinstance(s, Scenario) for s in outliers)
```

### Task 10: Creating Custom Scenario Templates

**When**: When you need to generate scenario templates for users to fill in.

**Steps**:

1. **Add template generation** in `stresstest/scenario/scenario_storage.py`:

```python
@staticmethod
def generate_comprehensive_template(filepath: str) -> None:
    """Generate comprehensive scenario template with examples."""
    template_scenarios = [
        # Equity scenarios
        Scenario(
            name="Equity Market Crash",
            description="Significant equity market decline",
            stresses=[
                Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO),
                Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.PORTFOLIO),
            ],
            metadata={"category": "equity", "severity": "high"}
        ),
        Scenario(
            name="Rate Hike",
            description="Interest rate increase",
            stresses=[
                Stress("rate", StressType.ABSOLUTE, 0.02, StressLevel.PORTFOLIO),
            ],
            metadata={"category": "rates", "severity": "medium"}
        ),
        # Template for user to fill
        Scenario(
            name="Your Custom Scenario",
            description="Describe your scenario",
            stresses=[
                # User fills these in:
                # Stress("spot", StressType.PERCENTAGE, 0.0, StressLevel.PORTFOLIO),
            ],
            metadata={"category": "custom", "severity": "TBD"}
        ),
    ]

    ScenarioStorage.save_scenarios(template_scenarios, filepath)
```

## Common Patterns and Anti-Patterns

### ✅ DO: Follow These Patterns

1. **Use ScenarioBuilder for Custom Scenarios**:

```python
# Good: Use builder API
scenario = (ScenarioBuilder()
    .name("My Scenario")
    .spot_stress(-0.20)
    .vol_stress(0.50)
    .build())
```

2. **Validate Scenarios Before Running**:

```python
# Good: Validate before execution
try:
    scenario = ScenarioBuilder().name("Test").spot_stress(-0.20).build()
    scenario.validate()
except ValidationError as e:
    print(f"Invalid scenario: {e}")
```

3. **Use Appropriate Stress Types**:

```python
# Good: Use percentage for spot
Stress("spot", StressType.PERCENTAGE, -0.20)

# Good: Use absolute for rates
Stress("rate", StressType.ABSOLUTE, 0.02)

# Good: Use value to set levels
Stress("volatility", StressType.VALUE, 0.80)
```

4. **Document Scenarios with Metadata**:

```python
# Good: Add metadata
scenario.metadata = {
    "category": "equity",
    "severity": "high",
    "source": "Internal research",
    "approved_by": "Risk Committee"
}
```

5. **Export Results in Multiple Formats**:

```python
# Good: Export for different use cases
ResultExporter.export(
    results,
    output_dir="./output",
    formats=['parquet', 'csv', 'json']
)
```

### ❌ DON'T: Avoid These Anti-Patterns

1. **Don't Create Scenarios Without Builder**:

```python
# Bad: Manual construction
scenario = Scenario(
    name="Test",
    stresses=[
        Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO)
    ]
)

# Good: Use builder
scenario = ScenarioBuilder().name("Test").spot_stress(-0.20).build()
```

2. **Don't Ignore Validation**:

```python
# Bad: Skip validation
scenario = ScenarioBuilder().spot_stress(-0.20).build()

# Good: Validate
try:
    scenario = ScenarioBuilder().spot_stress(-0.20).build()
    scenario.validate()
except ValidationError as e:
    print(f"Fix scenario: {e}")
```

3. **Don't Use Wrong Stress Levels**:

```python
# Bad: Position level without target
Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.POSITION)

# Good: Specify target
Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.POSITION, "pos_123")
```

4. **Don't Mix Stress Types for Same Parameter**:

```python
# Bad: Multiple stresses on spot at same level
scenario = ScenarioBuilder()
scenario.spot_stress(-0.20)
scenario.spot_stress(-0.15)  # Last one wins

# Good: Single combined stress
scenario = ScenarioBuilder()
scenario.spot_stress(-0.20)  # One stress
```

5. **Don't Forget to Save/Load Scenarios**:

```python
# Bad: Recreate scenarios each time
scenarios = [ScenarioBuilder().name("X").spot_stress(-0.20).build()]

# Good: Save and load
ScenarioStorage.save_scenarios(scenarios, "scenarios.yaml")
loaded = ScenarioStorage.load_scenarios("scenarios.yaml")
```

## Quick Reference

### Stress Type Selection

| Parameter | Stress Type | Example | When to Use |
|-----------|-------------|---------|-------------|
| Spot | PERCENTAGE | -20% | Relative changes |
| Volatility | PERCENTAGE or VALUE | +50% or 80% vol | Relative or absolute |
| Rate | ABSOLUTE | +200bps | Basis points |
| Dividend Yield | ABSOLUTE | +100bps | Basis points |
| Key Rate | ABSOLUTE | +25bps | Specific tenor |

### Stress Level Selection

| Level | Use Case | Example |
|-------|----------|---------|
| PORTFOLIO | All positions | Market-wide shock |
| UNDERLYING | Specific asset | AAPL-specific news |
| POSITION | Specific position | Single position stress |

### Export Format Selection

| Format | Use Case | Advantages |
|--------|----------|------------|
| Parquet | Large datasets | Fast, efficient, typed |
| CSV | Human-readable | Excel compatible, simple |
| JSON | Structured data | Complete, web-friendly |
| Excel | Business users | Multiple sheets, formatting |

### Configuration Best Practices

| Parameter | Recommended Value | Reason |
|-----------|-------------------|--------|
| calculate_greeks | True (small portfolios) | Risk analysis |
| calculate_greeks | False (large portfolios) | Performance |
| export_formats | ['parquet', 'csv'] | Balance of efficiency and usability |
| save_detailed_results | True (analysis) | Position-level details |
| save_detailed_results | False (summary) | Performance |

## Testing Strategy

### Test Categories

1. **Unit Tests** (test individual components)
   - Stress validation
   - Scenario builder
   - Scenario library
   - Storage I/O

2. **Integration Tests** (test workflows)
   - Engine execution
   - Results generation
   - Export functionality

3. **Asset-Specific Tests**:
   - Equity engine and metrics
   - FI engine and metrics (future)

### Example Test Structure

```python
class TestScenarioBuilder:
    """Test suite for ScenarioBuilder."""

    def test_fluent_api_chain(self):
        """Test builder API chaining."""
        scenario = (ScenarioBuilder()
            .name("Test Scenario")
            .description("Test description")
            .spot_stress(-0.20)
            .vol_stress(0.50)
            .build())

        assert scenario.name == "Test Scenario"
        assert scenario.description == "Test description"
        assert len(scenario.stresses) == 2
        assert scenario.stresses[0].parameter == "spot"
        assert scenario.stresses[0].stress_value == -0.20

    def test_validation_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError, match="Scenario name is required"):
            ScenarioBuilder().spot_stress(-0.20).build()

    def test_validation_at_least_one_stress(self):
        """Test that at least one stress is required."""
        with pytest.raises(ValidationError, match="At least one stress is required"):
            ScenarioBuilder().name("Test").build()
```

## Integration with Other Modules

### Portfolio Module

```python
from portfolio import Portfolio
from portfolio.equity.portfolio import EquityPortfolio

# Engine checks portfolio type
engine = EquityStressEngine(config)
if engine.supports_portfolio(portfolio):
    results = engine.run_static_scenarios(portfolio, scenarios)
```

### PriceEnv Module

```python
from priceenv import PricingEnvironment

# Stress applicator modifies PricingEnvironment
stressed_env = StressApplicator.apply_stress_to_environment(
    original_env,
    stress
)
```

### Results Module

```python
from stresstest.results import StressTestResults

# Results provide query and export methods
results = StressTestResults(envelope)
summary = results.get_summary()
df = results.to_summary_dataframe()
```

## Working with Results

### Basic Results Access

```python
results = engine.run_static_scenarios(portfolio, scenarios)

# Summary
summary = results.get_summary()
print(f"Worst scenario: {summary['worst_scenario']}")
print(f"Average P&L: ${summary['avg_pnl']:,.2f}")

# Best/worst scenarios
worst = results.get_worst_scenario()
best = results.get_best_scenario()
```

### Advanced Analysis

```python
# Scenario comparison
comparison_df = ResultAggregator.compare_scenarios(results, 'portfolio_pnl')
print(comparison_df)

# Risk metrics
risk_summary = ResultAggregator.get_risk_summary(results)
print(f"Max loss: ${risk_summary['max_loss']:,.2f}")
print(f"Volatility: {risk_summary['volatility']:.2%}")

# VaR/CVaR
var_cvar = ResultAggregator.calculate_var_cvar(results, confidence_level=0.95)
print(f"95% VaR: ${var_cvar['var']:,.2f}")
print(f"95% CVaR: ${var_cvar['cvar']:,.2f}")
```

### Position-Level Analysis

```python
# Get position details for worst scenario
worst = results.get_worst_scenario()
position_df = results.to_position_dataframe(worst.scenario.name)

# Sort by P&L impact
position_df = position_df.sort_values('pnl', ascending=True)

print("Top 5 losing positions:")
for _, pos in position_df.head().iterrows():
    print(f"  {pos['position_id']}: ${pos['pnl']:,.2f}")
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will log:
# - Scenario evaluation
# - Stress application
# - Results calculation
```

### Check Scenario Definition

```python
# Inspect scenario
for scenario in scenarios:
    print(f"\nScenario: {scenario.name}")
    print(f"  Description: {scenario.description}")
    print(f"  Stresses:")
    for stress in scenario.stresses:
        print(f"    - {stress.parameter}: {stress.stress_value} "
              f"({stress.stress_type.value}) at {stress.level.value}")
```

### Validate Stress Application

```python
# Apply single stress and check
stressed_portfolio = StressApplicator.apply_scenario_to_portfolio(
    portfolio, scenario[0]  # First scenario
)

# Compare values
baseline_value = portfolio.value()
stressed_value = stressed_portfolio.value()
print(f"Baseline: {baseline_value:,.2f}")
print(f"Stressed: {stressed_value:,.2f}")
print(f"Impact: {(stressed_value - baseline_value) / baseline_value:.2%}")
```

### Check Position Results

```python
# Inspect position results
for result in results.envelope.scenario_results:
    print(f"\nScenario: {result.scenario.name}")
    print(f"  Portfolio P&L: ${result.portfolio_pnl:,.2f}")
    print(f"  Position details: {len(result.position_results)}")
    for pos in result.position_results[:5]:  # First 5
        print(f"    {pos['position_id']}: ${pos['pnl']:,.2f}")
```

## Common Issues and Solutions

### Issue: "ValidationError: Scenario name is required"

**Error**: Scenario built without name.

**Solution**:
```python
# Add name before build
scenario = (ScenarioBuilder()
    .name("My Scenario")  # Add this
    .spot_stress(-0.20)
    .build())
```

### Issue: "ValidationError: Target underlying is required"

**Error**: UNDERLYING level stress without target.

**Solution**:
```python
# Specify target for UNDERLYING level
stress = Stress(
    "spot",
    StressType.PERCENTAGE,
    -0.20,
    StressLevel.UNDERLYING,
    target="AAPL"  # Add this
)
```

### Issue: "ValidationError: tenor_bucket metadata required"

**Error**: key_rate_stress without tenor bucket.

**Solution**:
```python
# Specify tenor bucket
scenario = (ScenarioBuilder()
    .key_rate_stress(0.01, tenor_bucket="5Y")  # Add tenor_bucket
    .build())
```

### Issue: "Greeks calculation failed"

**Error**: Product doesn't support analytical Greeks.

**Solution**:
```python
# Use numerical Greeks
config = StressTestConfig(
    calculate_greeks=True,
    greeks_method='numerical'  # Instead of 'analytical'
)
```

### Issue: "Stress not applied to specific position"

**Error**: Position-level stress not working.

**Solution**:
```python
# Verify position exists
print(f"Available positions: {list(portfolio.positions.keys())}")

# Use correct position_id
stress = Stress(
    "spot",
    StressType.PERCENTAGE,
    -0.20,
    StressLevel.POSITION,
    target="correct_position_id"  # Must match exactly
)
```

## Performance Monitoring

### Benchmark Function

```python
import time

def benchmark_stress_test(portfolio, scenarios):
    """Benchmark stress test execution."""
    config = StressTestConfig(calculate_greeks=True)
    engine = EquityStressEngine(config)

    start = time.time()
    results = engine.run_static_scenarios(portfolio, scenarios)
    elapsed = time.time() - start

    print(f"Stress test completed in {elapsed:.2f}s")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Positions: {len(portfolio.positions)}")
    print(f"  Time per scenario: {elapsed / len(scenarios):.3f}s")
    print(f"  Time per position: {elapsed / len(scenarios) / len(portfolio.positions):.6f}s")

    return results
```

### Performance Regression Testing

```python
@pytest.mark.performance
def test_large_portfolio_performance():
    """Ensure stress test completes within time limit."""
    portfolio = create_large_portfolio(1000)
    scenarios = create_test_scenarios(10)

    config = StressTestConfig(
        calculate_greeks=False,  # Disable for speed
        save_detailed_results=False
    )
    engine = EquityStressEngine(config)

    start = time.time()
    results = engine.run_static_scenarios(portfolio, scenarios)
    elapsed = time.time() - start

    # Must complete within 30 seconds
    assert elapsed < 30.0, f"Stress test took {elapsed:.2f}s (>30s limit)"
```

## Documentation Guidelines

### For New Features

When adding new functionality, document:

1. **Purpose**: What problem does this solve?
2. **Usage**: How to use it (with examples)?
3. **Configuration**: What parameters control it?
4. **Performance**: What's the performance impact?
5. **Testing**: How is correctness verified?

### For Scenario Additions

Document:
1. **Market Event**: What historical/current event does this represent?
2. **Parameters**: What stresses are applied and why?
3. **Use Case**: When should this scenario be used?
4. **Interpretation**: How to interpret results?

### Code Comments

Add comments for:

1. **Complex stress logic** (why, not just what)
2. **Asset-specific handling** (equity vs FI vs crypto)
3. **Adapter registration** (what parameters are supported)
4. **Performance considerations** (why certain choices were made)

```python
# Good: Explains why
# Use ABSOLUTE stress for rates to apply basis point changes
# This matches market convention for rate shocks
Stress("rate", StressType.ABSOLUTE, 0.02)

# Good: Explains asset-specific logic
# For equity: calculate Greeks
# For FI: calculate DV01/convexity
if self.metrics_adapter.supports(portfolio):
    extra_metrics = self.metrics_adapter.compute_metrics(...)

# Good: Explains adapter registration
# Register handler for key rate stresses
# Looks for tenor_bucket in stress metadata
StressApplicator.register_adapter("key_rate", apply_key_rate)
```

## Checklist for AI Agents

Before submitting code changes:

- [ ] New scenarios use ScenarioBuilder pattern
- [ ] All new methods have docstrings with examples
- [ ] Type hints added to all functions
- [ ] Validation added in __post_init__ or validate()
- [ ] Exports updated in __init__.py files
- [ ] Unit tests added for new functionality
- [ ] Integration tests verify end-to-end workflows
- [ ] Performance impact assessed and documented
- [ ] Documentation updated (CLAUDE.md, AGENTS.md)
- [ ] Examples added to README.md if user-facing
- [ ] Backwards compatibility maintained
- [ ] Error handling comprehensive
- [ ] Code follows project style guidelines

## Summary

This guide provides AI agents with targeted guidance for working with the stress test module. Key takeaways:

1. **Always use ScenarioBuilder** - Don't manually construct scenarios
2. **Validate early** - Catch errors before running stress tests
3. **Use appropriate stress types** - PERCENTAGE for spot, ABSOLUTE for rates
4. **Document scenarios** - Add metadata for context and interpretation
5. **Choose right export format** - Parquet for efficiency, CSV for readability
6. **Test thoroughly** - Stress testing is critical for risk management
7. **Document complex logic** - Future maintainers will thank you

For detailed implementation guidance, see `stresstest/CLAUDE.md`.
For user-facing documentation, see `stresstest/README.md`.
