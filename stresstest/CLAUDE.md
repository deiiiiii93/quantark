# Stress Test Module Developer Guide

## Overview

The QuantArk stress test module is a comprehensive, production-grade framework for portfolio scenario analysis. It provides sophisticated tools for stress testing portfolios under various market conditions, supporting both current static scenario analysis and future-ready APIs for dynamic scenario evolution with time dimensions and hedging strategies.

## Architecture

### Core Design Pattern: Protocol-Based Multi-Asset Architecture

The module follows a **dual-layer architecture** with base protocols and asset-specific implementations:

```
stresstest/
├── base.py                    # Protocol interfaces (shared)
├── config.py                  # Configuration (shared)
├── engine.py                  # Main engine (backward compatibility)
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

### 1. Base Protocols (`base.py`)

Defines **Protocol interfaces** ensuring consistency across asset classes and future extensibility.

#### Key Protocols:

##### **ScenarioEnvelope**
```python
@dataclass
class ScenarioEnvelope:
    """Container for scenario-level stress results."""
    scenario: Scenario
    portfolio_value: float
    portfolio_pnl: float
    portfolio_pnl_pct: float
    greeks: Optional[Dict[str, float]] = None
    position_results: List[Dict[str, Any]] = field(default_factory=list)
    underlying_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    extra_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Purpose**: Standard container for individual scenario results
**Usage**: Returned by `evaluate_scenario()` for each scenario
**Contents**:
- Portfolio-level metrics (value, P&L, P&L %)
- Greeks under stress
- Position-level details
- Underlying-level aggregation
- Asset-specific metrics (via `extra_metrics`)

##### **StressResultEnvelope**
```python
@dataclass
class StressResultEnvelope:
    """Top-level container for all stress test results."""
    baseline_value: float
    baseline_greeks: Optional[Dict[str, float]]
    scenario_results: Sequence[ScenarioEnvelope]
    execution_timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

**Purpose**: Complete stress test results container
**Usage**: Returned by `run_static_scenarios()` for entire test
**Contents**:
- Baseline (non-stressed) values
- All scenario results
- Execution metadata
- Configuration summary

##### **StressMetricsAdapter**
```python
@runtime_checkable
class StressMetricsAdapter(Protocol):
    """Adapter interface for asset-specific stress metrics."""

    def supports(self, portfolio: BasePortfolio) -> bool:
        """Check if adapter supports portfolio type."""
        ...

    def compute_metrics(
        self,
        original_portfolio: BasePortfolio,
        stressed_portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
        stressed_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Return asset-specific metrics keyed by namespace."""
        ...
```

**Purpose**: Calculate asset-specific metrics (equity, FI, etc.)
**Implementation Pattern**:
- Equity: Greeks (delta, gamma, vega, theta, rho)
- Fixed Income: DV01, convexity, duration
- Extensible to new asset classes

**Usage**:
```python
# Engine uses adapter to calculate metrics
if self.metrics_adapter.supports(portfolio):
    extra_metrics = self.metrics_adapter.compute_metrics(
        original_portfolio, stressed_portfolio, scenario,
        baseline_value, stressed_value
    )
```

##### **ScenarioRunner**
```python
@runtime_checkable
class ScenarioRunner(Protocol):
    """Contract for scenario evaluation helpers."""

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioEnvelope:
        """Evaluate single scenario."""
        ...
```

**Purpose**: Evaluates a single scenario on a portfolio
**Implementation**: Asset-specific evaluation logic
**Usage**: Called by engine for each scenario

##### **BaseStressEngine**
```python
@runtime_checkable
class BaseStressEngine(Protocol):
    """Base protocol for all stress testing engines."""

    def supports_portfolio(self, portfolio: BasePortfolio) -> bool:
        """Check if engine supports portfolio type."""
        ...

    def run_static_scenarios(
        self,
        portfolio: BasePortfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> StressResultEnvelope:
        """Run static scenario analysis."""
        ...

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioEnvelope:
        """Evaluate single scenario."""
        ...
```

**Purpose**: Main interface for stress testing engines
**Implementation Pattern**:
- Equity: `EquityStressEngine`
- Fixed Income: `FIStressEngine` (future)
- Future: Crypto, Commodity engines

### 2. Stress System (`stress/`)

#### **StressType** (`stress_types.py`)

```python
class StressType(Enum):
    """Type of stress to apply."""
    ABSOLUTE = "absolute"      # Add/subtract absolute value
    PERCENTAGE = "percentage"  # Apply percentage change
    VALUE = "value"            # Set to specific value

    def apply(self, current_value: float, stress_value: float) -> float:
        """Apply stress to current value."""
```

**Usage Examples**:
```python
# ABSOLUTE: Add 200bps to rate
StressType.ABSOLUTE.apply(0.05, 0.02)  # 0.07 (7%)

# PERCENTAGE: Reduce spot by 20%
StressType.PERCENTAGE.apply(100, -0.20)  # 80.0

# VALUE: Set rate to 5%
StressType.VALUE.apply(0.03, 0.05)  # 0.05 (5%)
```

#### **StressLevel** (`stress_types.py`)

```python
class StressLevel(Enum):
    """Level at which to apply stress."""
    PORTFOLIO = "portfolio"    # All positions
    UNDERLYING = "underlying"  # Specific underlying
    POSITION = "position"      # Specific position
```

**Purpose**: Granular control over stress application scope
**Examples**:
```python
# PORTFOLIO: All positions
Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO)

# UNDERLYING: Only AAPL
Stress("spot", StressType.PERCENTAGE, -0.25, StressLevel.UNDERLYING, "AAPL")

# POSITION: Specific position
Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.POSITION, "pos_123")
```

#### **Stress Applicator** (`stress/stress_applicator.py`)

```python
class StressApplicator:
    """
    Applies stresses to PricingEnvironment objects.

    Responsibilities:
    - Clone pricing environments
    - Apply stresses based on type and level
    - Support adapter hooks for custom logic
    - Generate stress summaries
    """

    @staticmethod
    def apply_scenario_to_portfolio(
        portfolio: BasePortfolio,
        scenario: Scenario,
    ) -> BasePortfolio:
        """Apply scenario to portfolio and return stressed portfolio."""
        # 1. Clone portfolio deep
        # 2. Filter stresses by level
        # 3. Apply each stress via adapter
        # 4. Return stressed portfolio
```

**Adapter System**:
```python
# Register custom stress handler
def apply_key_rate(env, stress):
    bucket = stress.metadata.get("tenor_bucket", "10Y")
    shift = stress.stress_type.apply(
        env.rate_curve.get_rate(bucket),
        stress.stress_value
    )
    # Custom FI logic

StressApplicator.register_adapter("key_rate", apply_key_rate)

# Default adapters registered for: spot, volatility, rate, dividend
```

### 3. Scenario System (`scenario/`)

#### **Stress Class** (`scenario/scenario.py`)

```python
@dataclass
class Stress:
    """
    Single parameter stress definition.

    Attributes:
        parameter: Name of parameter (e.g., "spot", "volatility", "rate")
        stress_type: Type of stress (ABSOLUTE, PERCENTAGE, VALUE)
        stress_value: Magnitude
        level: Application level (PORTFOLIO, UNDERLYING, POSITION)
        target: Optional target identifier
        description: Human-readable description
        metadata: Additional parameters (tenor_bucket, spread_curve, etc.)
    """
```

**Validation** (`__post_init__`):
- Parameter name required
- Target required for UNDERLYING/POSITION levels
- No target for PORTFOLIO level
- Metadata validation for specialized stresses

**Serialization**:
```python
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for YAML/JSON."""
    return {
        "parameter": self.parameter,
        "stress_type": self.stress_type.value,
        "stress_value": self.stress_value,
        "level": self.level.value,
        "target": self.target,
        "description": self.description,
        "metadata": self.metadata,
    }

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'Stress':
    """Create from dictionary."""
    return cls(
        parameter=data["parameter"],
        stress_type=StressType(data["stress_type"]),
        stress_value=data["stress_value"],
        level=StressLevel(data["level"]),
        target=data.get("target"),
        description=data.get("description"),
        metadata=data.get("metadata", {}),
    )
```

#### **Scenario Class** (`scenario/scenario.py`)

```python
@dataclass
class Scenario:
    """Container for multiple stresses with metadata."""

    name: str
    description: Optional[str] = None
    stresses: List[Stress] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stress(self, stress: Stress) -> 'Scenario':
        """Add stress and return self (fluent)."""
        self.stresses.append(stress)
        return self

    def get_stresses_for_level(self, level: StressLevel) -> List[Stress]:
        """Get stresses for specific level."""
        return [s for s in self.stresses if s.level == level]

    def get_stresses_for_target(self, target: str) -> List[Stress]:
        """Get stresses for specific target."""
        return [s for s in self.stresses if s.target == target]
```

**Purpose**: Group related stresses into coherent scenarios
**Usage**:
```python
scenario = Scenario(
    name="Market Crash",
    description="Significant market downturn",
    stresses=[
        Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO),
        Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.PORTFOLIO),
    ],
    metadata={"category": "equity", "severity": "high"}
)
```

#### **ScenarioBuilder** (`scenario/scenario_builder.py`)

```python
class ScenarioBuilder:
    """
    Fluent API for building scenarios.

    Example:
        scenario = (ScenarioBuilder()
            .name("Market Stress")
            .spot_stress(-0.20)
            .vol_stress(0.50)
            .rate_stress(0.02, stress_type=StressType.ABSOLUTE)
            .build())
    """

    def __init__(self):
        self._scenario = Scenario(name="", description="")

    def name(self, name: str) -> 'ScenarioBuilder':
        """Set scenario name."""
        self._scenario.name = name
        return self

    def description(self, desc: str) -> 'ScenarioBuilder':
        """Set description."""
        self._scenario.description = desc
        return self

    def spot_stress(
        self,
        stress_value: float,
        stress_type: StressType = StressType.PERCENTAGE,
        level: StressLevel = StressLevel.PORTFOLIO,
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'ScenarioBuilder':
        """Add spot stress."""
        stress = Stress(
            parameter="spot",
            stress_type=stress_type,
            stress_value=stress_value,
            level=level,
            target=target,
            metadata=metadata or {},
        )
        self._scenario.stresses.append(stress)
        return self

    def vol_stress(self, ...) -> 'ScenarioBuilder': ...

    def rate_stress(self, ...) -> 'ScenarioBuilder': ...

    def key_rate_stress(
        self,
        stress_value: float,
        tenor_bucket: str = "10Y",
        stress_type: StressType = StressType.ABSOLUTE,
    ) -> 'ScenarioBuilder':
        """Add key rate stress with tenor bucket."""
        stress = Stress(
            parameter="key_rate",
            stress_type=stress_type,
            stress_value=stress_value,
            level=StressLevel.PORTFOLIO,
            metadata={"tenor_bucket": tenor_bucket},
        )
        self._scenario.stresses.append(stress)
        return self

    def spread_stress(
        self,
        stress_value: float,
        spread_curve: str,
        stress_type: StressType = StressType.ABSOLUTE,
    ) -> 'ScenarioBuilder':
        """Add spread stress."""
        stress = Stress(
            parameter="spread",
            stress_type=stress_type,
            stress_value=stress_value,
            level=StressLevel.PORTFOLIO,
            metadata={"spread_curve": spread_curve},
        )
        self._scenario.stresses.append(stress)
        return self

    def build(self) -> Scenario:
        """Build and return scenario."""
        if not self._scenario.name:
            raise ValidationError("Scenario name is required")
        if not self._scenario.stresses:
            raise ValidationError("At least one stress is required")
        return self._scenario
```

**Key Features**:
- Fluent, chainable API
- Sensible defaults (percentage stress, portfolio level)
- Specialized helpers (key_rate_stress, spread_stress)
- Validation before build

#### **ScenarioLibrary** (`scenario/scenario_library.py`)

```python
class ScenarioLibrary:
    """Predefined scenarios for common use cases."""

    @staticmethod
    def market_crash() -> Scenario:
        """Standard market crash scenario."""
        return ScenarioBuilder() \
            .name("Market Crash") \
            .spot_stress(-0.20) \
            .vol_stress(0.50) \
            .description("20% equity drop, 50% vol spike") \
            .build()

    @staticmethod
    def market_rally() -> Scenario:
        """Market rally scenario."""
        return ScenarioBuilder() \
            .name("Market Rally") \
            .spot_stress(0.15) \
            .vol_stress(-0.30) \
            .description("15% equity rise, 30% vol crush") \
            .build()

    @staticmethod
    def vol_spike() -> Scenario:
        """Volatility spike scenario."""
        return ScenarioBuilder() \
            .name("Vol Spike") \
            .vol_stress(0.80) \
            .description("80% volatility spike") \
            .build()

    @staticmethod
    def rate_hike() -> Scenario:
        """Interest rate hike scenario."""
        return ScenarioBuilder() \
            .name("Rate Hike") \
            .rate_stress(0.02, stress_type=StressType.ABSOLUTE) \
            .description("200bps rate increase") \
            .build()

    @staticmethod
    def severe_downturn() -> Scenario:
        """Severe downturn scenario."""
        return ScenarioBuilder() \
            .name("Severe Downturn") \
            .spot_stress(-0.35) \
            .vol_stress(1.00) \
            .rate_stress(-0.01, stress_type=StressType.ABSOLUTE) \
            .description("35% drop, 100% vol spike, 100bps cut") \
            .build()
```

**Historical Scenarios**:
```python
@staticmethod
def black_monday_1987() -> Scenario:
    """Black Monday 1987 crash."""
    return ScenarioBuilder() \
        .name("Black Monday 1987") \
        .spot_stress(-0.226) \
        .description("22.6% drop on Oct 19, 1987") \
        .metadata({"date": "1987-10-19", "historical": True}) \
        .build()

@staticmethod
def financial_crisis_2008() -> Scenario:
    """2008 Financial Crisis."""
    return ScenarioBuilder() \
        .name("Financial Crisis 2008") \
        .spot_stress(-0.40) \
        .vol_stress(1.20) \
        .description("40% equity drop, 120% vol spike") \
        .metadata({"date": "2008-09-15", "historical": True}) \
        .build()

@staticmethod
def covid_crash_2020() -> Scenario:
    """COVID Crash 2020."""
    return ScenarioBuilder() \
        .name("COVID Crash 2020") \
        .spot_stress(-0.34) \
        .vol_stress(2.00) \
        .description("34% equity drop, 200% vol spike") \
        .metadata({"date": "2020-03-01", "historical": True}) \
        .build()
```

#### **ScenarioStorage** (`scenario/scenario_storage.py`)

```python
class ScenarioStorage:
    """Save and load scenarios from YAML/JSON."""

    @staticmethod
    def save_scenarios(scenarios: List[Scenario], filepath: str) -> None:
        """Save scenarios to YAML or JSON."""
        # Auto-detect format from extension
        # Serialize scenarios to dict
        # Write to file

    @staticmethod
    def load_scenarios(filepath: str) -> List[Scenario]:
        """Load scenarios from file."""
        # Read file
        # Detect format
        # Deserialize to Scenario objects
        # Return list

    @staticmethod
    def load_scenario(filepath: str, scenario_index: int = 0) -> Scenario:
        """Load single scenario from file."""
        scenarios = ScenarioStorage.load_scenarios(filepath)
        return scenarios[scenario_index]

    @staticmethod
    def generate_template(filepath: str) -> None:
        """Generate scenario template file."""
        # Create YAML template with examples
        # Useful for users to understand format
```

**File Format**:
```yaml
# scenarios.yaml
- name: Market Crash
  description: 20% drop, 50% vol spike
  stresses:
    - parameter: spot
      stress_type: percentage
      stress_value: -0.20
      level: portfolio
    - parameter: volatility
      stress_type: percentage
      stress_value: 0.50
      level: portfolio
  metadata:
    category: equity
    severity: high

- name: Rate Hike
  description: 200bps increase
  stresses:
    - parameter: rate
      stress_type: absolute
      stress_value: 0.02
      level: portfolio
  metadata:
    category: rates
```

### 4. Engine Layer

#### **EquityStressEngine** (`equity/engine.py`)

```python
class EquityStressEngine:
    """
    Main stress testing engine for equity portfolios.

    Orchestrates stress test execution:
    1. Calculate baseline (non-stressed) values
    2. For each scenario:
       - Apply stresses to portfolio
       - Calculate stressed values
       - Compute Greeks
       - Generate position-level details
    3. Aggregate results
    """

    def __init__(
        self,
        config: 'EquityStressTestConfig',
        metrics_adapter: Optional[StressMetricsAdapter] = None,
    ):
        self.config = config
        self.metrics_adapter = metrics_adapter or EquityStressMetricsAdapter()
        self.scenario_runner = EquityScenarioRunner(config)

    def supports_portfolio(self, portfolio: BasePortfolio) -> bool:
        """Check if supports portfolio type."""
        from portfolio.equity.portfolio import EquityPortfolio
        return isinstance(portfolio, EquityPortfolio)

    def run_static_scenarios(
        self,
        portfolio: BasePortfolio,
        scenarios: Sequence[Scenario],
        baseline_label: str = "Current Market",
    ) -> StressResultEnvelope:
        """Run static scenario analysis."""
        # 1. Calculate baseline
        baseline_value = portfolio.value()
        baseline_greeks = self._calculate_greeks(portfolio) if self.config.calculate_greeks else None

        # 2. Evaluate each scenario
        scenario_results = []
        for scenario in scenarios:
            result = self.evaluate_scenario(portfolio, scenario, baseline_value)
            scenario_results.append(result)

        # 3. Create envelope
        return StressResultEnvelope(
            baseline_value=baseline_value,
            baseline_greeks=baseline_greeks,
            scenario_results=scenario_results,
            total_execution_time=...,
            config_summary=self.config.get_summary(),
        )

    def evaluate_scenario(
        self,
        portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
    ) -> ScenarioEnvelope:
        """Evaluate single scenario."""
        # 1. Apply stresses
        stressed_portfolio = StressApplicator.apply_scenario_to_portfolio(
            portfolio, scenario
        )

        # 2. Calculate stressed value
        stressed_value = stressed_portfolio.value()

        # 3. Calculate Greeks if configured
        greeks = None
        if self.config.calculate_greeks:
            greeks = self._calculate_greeks(stressed_portfolio)

        # 4. Calculate position-level results
        position_results = self._calculate_position_results(
            portfolio, stressed_portfolio, scenario
        )

        # 5. Calculate underlying-level results
        underlying_results = self._calculate_underlying_results(
            portfolio, stressed_portfolio, scenario
        )

        # 6. Calculate asset-specific metrics
        extra_metrics = {}
        if self.metrics_adapter.supports(portfolio):
            extra_metrics = self.metrics_adapter.compute_metrics(
                portfolio, stressed_portfolio, scenario,
                baseline_value, stressed_value
            )

        # 7. Create envelope
        return ScenarioEnvelope(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=stressed_value - baseline_value,
            portfolio_pnl_pct=(stressed_value - baseline_value) / baseline_value,
            greeks=greeks,
            position_results=position_results,
            underlying_results=underlying_results,
            extra_metrics=extra_metrics,
        )
```

**Key Components**:
- Metrics adapter for asset-specific calculations
- Scenario runner for evaluation logic
- Position and underlying-level aggregation
- Configurable Greeks calculation

#### **FIStressEngine** (`fi/engine.py`)

Similar structure to equity engine but for fixed income:
- Portfolio type: `FIPortfolio`
- Metrics: DV01, convexity, duration (not Greeks)
- Stress parameters: rate_curve, key_rates, spreads

### 5. Configuration

#### **StressTestConfig** (`config.py`)

```python
@dataclass
class StressTestConfig:
    """Configuration for stress tests."""

    calculate_greeks: bool = True
    greeks_method: str = "analytical"  # or "numerical"
    export_formats: List[str] = field(default_factory=lambda: ['parquet'])
    output_dir: str = "./stress_results"
    save_detailed_results: bool = True
    parallel_execution: bool = False  # Future
    max_workers: int = 4  # Future
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "calculate_greeks": self.calculate_greeks,
            "greeks_method": self.greeks_method,
            "export_formats": self.export_formats,
            "output_dir": self.output_dir,
        }
```

#### **EquityStressTestConfig** (`equity/config.py`)

Equity-specific configuration:
- Extends base config
- Equity-specific options
- Greeks calculation settings

### 6. Results Management

#### **StressTestResults** (`results/stress_results.py`)

```python
class StressTestResults:
    """
    Container for complete stress test results.

    Provides query and analysis methods.
    """

    def __init__(self, envelope: StressResultEnvelope):
        self.envelope = envelope

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all scenarios."""
        # Calculate worst/best, avg, std, etc.
        pass

    def get_worst_scenario(self) -> ScenarioResult:
        """Get worst-performing scenario."""
        # Find scenario with max loss
        pass

    def get_best_scenario(self) -> ScenarioResult:
        """Get best-performing scenario."""
        # Find scenario with max gain
        pass

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Convert to summary DataFrame."""
        # Columns: name, pnl, pnl_pct, greeks_delta, etc.
        pass

    def to_position_dataframe(self, scenario_name: str) -> pd.DataFrame:
        """Get position-level results for scenario."""
        # Find scenario
        # Return position details DataFrame
        pass
```

#### **ResultAggregator** (`results/result_aggregator.py`)

```python
class ResultAggregator:
    """Analysis and comparison utilities."""

    @staticmethod
    def get_risk_summary(results: StressTestResults) -> Dict[str, Any]:
        """Calculate risk summary statistics."""
        return {
            "avg_pnl": ...,
            "max_loss": ...,
            "max_gain": ...,
            "volatility": ...,
            "max_drawdown_pct": ...,
        }

    @staticmethod
    def calculate_var_cvar(
        results: StressTestResults,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """Calculate VaR and CVaR from scenario P&Ls."""
        # Extract P&L distribution
        # Calculate percentile-based metrics
        pass

    @staticmethod
    def compare_scenarios(
        results: StressTestResults,
        metric: str = 'portfolio_pnl',
    ) -> pd.DataFrame:
        """Compare scenarios by metric."""
        # Create comparison DataFrame
        pass
```

#### **ResultExporter** (`results/result_exporter.py`)

```python
class ResultExporter:
    """Export results to various formats."""

    @staticmethod
    def export_to_parquet(
        results: StressTestResults,
        output_dir: str,
        include_positions: bool = True,
    ) -> None:
        """Export to Parquet format."""
        # Summary table
        # Position-level details
        # Greeks
        pass

    @staticmethod
    def export_to_csv(
        results: StressTestResults,
        output_dir: str,
    ) -> None:
        """Export to CSV format."""
        # Multiple CSV files
        pass

    @staticmethod
    def export_to_json(
        results: StressTestResults,
        filepath: str,
    ) -> None:
        """Export to JSON format."""
        # Complete structured data
        pass

    @staticmethod
    def export(
        results: StressTestResults,
        output_dir: str,
        formats: List[str],
        base_name: str = "stress_test",
    ) -> None:
        """Export to multiple formats."""
        # Batch export
        pass
```

### 7. Reporting & Visualization

#### **ReportGenerator** (`report/report_generator.py`)

```python
class ReportGenerator:
    """Generate comprehensive HTML reports."""

    def generate_report(
        self,
        results: StressTestResults,
        output_path: str,
        title: str = "Stress Test Report",
    ) -> str:
        """Generate HTML report."""
        # 1. Create HTML template
        # 2. Add executive summary
        # 3. Add scenario results table
        # 4. Add risk metrics
        # 5. Add visualizations
        # 6. Save to file
```

**Report Contents**:
- Executive summary (worst/best scenarios)
- Scenario results table
- Risk metrics dashboard
- Greeks comparison
- Position-level breakdown

#### **StressTestVisualizer** (`report/visualizer.py`)

```python
class StressTestVisualizer:
    """Create static and interactive visualizations."""

    def plot_pnl_waterfall(
        self,
        results: StressTestResults,
        output_path: str,
    ) -> plt.Figure:
        """Plot P&L waterfall chart."""
        # Show contribution of each scenario
        pass

    def plot_pnl_distribution(
        self,
        results: StressTestResults,
        output_path: str,
    ) -> plt.Figure:
        """Plot P&L distribution histogram."""
        # Show distribution of scenario P&Ls
        pass

    def plot_scenario_comparison(
        self,
        results: StressTestResults,
        output_path: str,
    ) -> plt.Figure:
        """Plot scenario comparison bar chart."""
        # Compare scenarios side-by-side
        pass

    def plot_greeks_comparison(
        self,
        results: StressTestResults,
        output_path: str,
    ) -> plt.Figure:
        """Plot Greeks comparison."""
        # Show Greeks under each scenario
        pass

    def create_interactive_dashboard(
        self,
        results: StressTestResults,
        output_path: str,
    ) -> go.Figure:
        """Create interactive Plotly dashboard."""
        # Multi-panel interactive visualization
        pass
```

### 8. Results Classes

#### **ScenarioResult** (`results/stress_results.py`)

```python
@dataclass
class ScenarioResult:
    """Individual scenario result."""
    scenario: Scenario
    portfolio_value: float
    portfolio_pnl: float
    portfolio_pnl_pct: float
    greeks: Optional[Dict[str, float]]
    position_results: List[Dict[str, Any]]
    underlying_results: Dict[str, Dict[str, Any]]
    extra_metrics: Dict[str, Dict[str, Any]]
```

#### **EquityScenarioResult** (`equity/results.py`)

Equity-specific result with Greeks breakdown

#### **FIScenarioResult** (`fi/results.py`)

FI-specific result with DV01/convexity/duration

### 9. Metrics Adapters

#### **EquityStressMetricsAdapter** (in `equity/engine.py`)

```python
class EquityStressMetricsAdapter:
    """Calculate equity-specific stress metrics."""

    def supports(self, portfolio: BasePortfolio) -> bool:
        from portfolio.equity.portfolio import EquityPortfolio
        return isinstance(portfolio, EquityPortfolio)

    def compute_metrics(
        self,
        original_portfolio: BasePortfolio,
        stressed_portfolio: BasePortfolio,
        scenario: Scenario,
        baseline_value: float,
        stressed_value: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate Greeks and equity-specific metrics."""
        # Calculate Greeks for stressed portfolio
        greeks = self._calculate_greeks(stressed_portfolio)

        # Calculate Greeks deltas
        greeks_delta = {
            greek: stressed_greeks[greek] - baseline_greeks[greek]
            for greek in stressed_greeks.keys()
        }

        return {
            "equity": {
                "greeks": greeks,
                "greeks_delta": greeks_delta,
            }
        }
```

#### **FIStressMetricsAdapter** (in `fi/engine.py`)

FI-specific metrics:
- DV01
- Convexity
- Duration
- Key rate DV01s

## Usage Patterns

### Pattern 1: Basic Stress Test

```python
from stresstest import StressTestEngine, StressTestConfig
from stresstest.scenario.scenario_library import ScenarioLibrary

# Configure
config = StressTestConfig(calculate_greeks=True)
engine = StressTestEngine(config)

# Use predefined scenarios
scenarios = [
    ScenarioLibrary.market_crash(),
    ScenarioLibrary.market_rally(),
    ScenarioLibrary.vol_spike(),
]

# Run stress test
results = engine.run_static_scenarios(portfolio, scenarios)

# View results
print(results.get_summary())
```

### Pattern 2: Custom Scenario Builder

```python
from stresstest.scenario.scenario_builder import ScenarioBuilder
from stresstest.stress.stress_types import StressType, StressLevel

# Build custom scenario
scenario = (ScenarioBuilder()
    .name("Custom Stress")
    .description("15% drop, 30% vol spike, 50bps hike")
    .spot_stress(-0.15)
    .vol_stress(0.30)
    .rate_stress(0.005, stress_type=StressType.ABSOLUTE)
    .build())

results = engine.run_static_scenarios(portfolio, [scenario])
```

### Pattern 3: Multi-Parameter Scenario

```python
# Complex scenario with multiple stresses
scenario = (ScenarioBuilder()
    .name("Severe Crisis")
    .spot_stress(-0.30)
    .vol_stress(1.50)
    .rate_stress(-0.02, stress_type=StressType.ABSOLUTE)  # Cut rates
    .key_rate_stress(0.01, tenor_bucket="5Y")  # Steepener
    .spread_stress(0.005, spread_curve="CDX HY")  # Credit spread shock
    .build())
```

### Pattern 4: Targeted Stress

```python
# Stress only AAPL
scenario = (ScenarioBuilder()
    .name("AAPL Shock")
    .spot_stress(-0.25, target="AAPL")
    .vol_stress(0.60, target="AAPL")
    .build())

# Other underlyings unchanged
```

### Pattern 5: Scenario Persistence

```python
from stresstest.scenario.scenario_storage import ScenarioStorage

# Save scenarios
ScenarioStorage.save_scenarios(scenarios, "my_scenarios.yaml")

# Load scenarios
loaded_scenarios = ScenarioStorage.load_scenarios("my_scenarios.yaml")

# Generate template
ScenarioStorage.generate_template("template.yaml")
```

### Pattern 6: Results Export

```python
from stresstest.results.result_exporter import ResultExporter

# Export to all formats
ResultExporter.export(
    results,
    output_dir="./output",
    formats=['parquet', 'csv', 'json'],
    base_name="Q4_2024_stress_test"
)

# Export specific format
ResultExporter.export_to_parquet(
    results,
    "./output",
    include_positions=True
)
```

### Pattern 7: Visualization

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
visualizer.plot_pnl_waterfall(results, "./plots/waterfall.png")
visualizer.plot_pnl_distribution(results, "./plots/distribution.png")

# Interactive dashboard
visualizer.create_interactive_dashboard(
    results,
    "./plots/dashboard.html"
)
```

### Pattern 8: Position-Level Analysis

```python
# Get worst scenario
worst = results.get_worst_scenario()

# Analyze position contributions
position_df = results.to_position_dataframe(worst.scenario.name)
print(position_df.sort_values('pnl', ascending=True))

# Position P&L contribution
for _, pos in position_df.iterrows():
    print(f"{pos['position_id']}: ${pos['pnl']:,.2f} ({pos['pnl_pct']:.2%})")
```

## Performance Considerations

### Engine Performance

| Scenario Type | 10 Positions | 100 Positions | 1000 Positions |
|---------------|--------------|---------------|----------------|
| Simple (1-2 stresses) | < 1s | 2-3s | 10-15s |
| Complex (5+ stresses) | 1-2s | 5-8s | 30-45s |
| With Greeks | +50% | +50% | +50% |
| Position-level details | +20% | +20% | +20% |

### Optimization Strategies

1. **Disable Greeks for Large Portfolios**:
   ```python
   config = StressTestConfig(calculate_greeks=False)
   ```

2. **Skip Detailed Results**:
   ```python
   config = StressTestConfig(save_detailed_results=False)
   ```

3. **Use Parquet for Large Datasets**:
   ```python
   config = StressTestConfig(export_formats=['parquet'])
   ```

4. **Filter Scenarios**:
   ```python
   # Only run relevant scenarios
   scenarios = [s for s in all_scenarios if s.metadata.get('category') == 'equity']
   ```

### Memory Usage

- **Scenario Result**: ~1KB per scenario
- **Position Details**: ~100 bytes per position per scenario
- **Greeks**: ~50 bytes per scenario

**Optimization**:
- Disable `save_detailed_results` for memory-constrained environments
- Use position-level aggregation instead of full details
- Export results immediately and discard in-memory

## Testing

### Test Structure

```
test/test_stress_test.py
├── TestStress                 # Stress validation
├── TestScenario               # Scenario creation
├── TestScenarioBuilder        # Builder API
├── TestScenarioLibrary        # Predefined scenarios
├── TestScenarioStorage        # YAML/JSON I/O
├── TestStressTypes            # Stress type application
├── TestStressApplicator       # Stress application
├── TestEquityStressEngine     # Engine execution
├── TestStressResults          # Results
├── TestResultAggregator       # Analysis
└── TestResultExporter         # Export
```

### Running Tests

```bash
# All stress test tests
python -m pytest test/test_stress_test.py -v

# Specific test
python -m pytest test/test_stress_test.py::TestScenarioBuilder -v

# With coverage
python -m pytest test/test_stress_test.py --cov=stresstest --cov-report=html
```

### Test Categories

1. **Unit Tests**: Individual components
   - Stress validation
   - Scenario builder
   - Scenario library
   - Storage I/O

2. **Integration Tests**: End-to-end workflows
   - Engine execution
   - Results generation
   - Export functionality

3. **Asset-Specific Tests**:
   - Equity engine and metrics
   - FI engine and metrics (future)

### Example Test

```python
def test_scenario_builder_fluent_api():
    """Test fluent API chaining."""
    scenario = (ScenarioBuilder()
        .name("Test Scenario")
        .spot_stress(-0.20)
        .vol_stress(0.50)
        .build())

    assert scenario.name == "Test Scenario"
    assert len(scenario.stresses) == 2
    assert scenario.stresses[0].parameter == "spot"
    assert scenario.stresses[0].stress_value == -0.20
```

## Error Handling

### Exception Hierarchy

```
QuantArkException (base)
├── ValidationError       # Invalid configuration/scenario
├── MarketDataError       # Missing/invalid market data
├── NumericalError        # Numerical issues
└── StressTestError       # General stress test errors
```

### Common Errors and Solutions

1. **ValidationError: "Scenario name is required"**
   - Call `.name()` before `.build()`
   - Check scenario builder flow

2. **ValidationError: "At least one stress is required"**
   - Add at least one stress before `.build()`
   - Use spot_stress(), vol_stress(), etc.

3. **ValidationError: "Target underlying is required"**
   - Set target for UNDERLYING level stresses
   - Use target="AAPL" parameter

4. **ValidationError: "tenor_bucket metadata required"**
   - For key_rate_stress, must specify tenor_bucket
   - Use `.key_rate_stress(value, tenor_bucket="5Y")`

### Debugging Tips

1. **Check Scenario Definition**:
   ```python
   # Print scenario details
   print(f"Name: {scenario.name}")
   print(f"Stresses: {len(scenario.stresses)}")
   for stress in scenario.stresses:
       print(f"  {stress.parameter}: {stress.stress_value}")
   ```

2. **Validate Stresses**:
   ```python
   # Check stress application
   stressed_portfolio = StressApplicator.apply_scenario_to_portfolio(
       portfolio, scenario
   )
   summary = StressApplicator.get_stress_summary(
       portfolio.pricing_environment,
       stressed_portfolio.pricing_environment
   )
   print(summary)
   ```

3. **Check Results**:
   ```python
   # Validate scenario results
   for result in results.envelope.scenario_results:
       print(f"Scenario: {result.scenario.name}")
       print(f"  P&L: ${result.portfolio_pnl:,.2f}")
       print(f"  Positions: {len(result.position_results)}")
   ```

## Integration with Other Modules

### Portfolio Module

```python
from portfolio import Portfolio
from portfolio.equity.portfolio import EquityPortfolio

# Equity portfolio
equity_portfolio = EquityPortfolio(positions=[...])

# Engine checks portfolio type
engine.supports_portfolio(equity_portfolio)  # True
```

### PriceEnv Module

```python
from priceenv import PricingEnvironment

# Stress applicator modifies PricingEnvironment
stressed_env = StressApplicator.apply_stress(
    original_env,
    stress
)
```

### Market Data Module

```python
from util.marketdata.adapter import MarketDataAdapter

# Market data used for baseline calculations
baseline_value = portfolio.value(market_data)
```

## Best Practices

### 1. Use Predefined Scenarios as Starting Point

```python
# Good: Start with library scenarios
scenarios = [
    ScenarioLibrary.market_crash(),
    ScenarioLibrary.vol_spike(),
]

# Customize as needed
custom = (ScenarioBuilder()
    .name("Modified Crash")
    .spot_stress(-0.15)  # Less severe than library
    .vol_stress(0.40)
    .build())
```

### 2. Document Scenarios with Metadata

```python
scenario = (ScenarioBuilder()
    .name("Custom Scenario")
    .description("Detailed description")
    .spot_stress(-0.20)
    .build())

# Add metadata
scenario.metadata = {
    "category": "equity",
    "severity": "high",
    "source": "Internal research",
    "approved_by": "Risk Committee"
}
```

### 3. Use Appropriate Stress Types

```python
# Good: Use PERCENTAGE for spot (relative change)
spot_stress = Stress("spot", StressType.PERCENTAGE, -0.20)

# Good: Use ABSOLUTE for rates (basis points)
rate_stress = Stress("rate", StressType.ABSOLUTE, 0.02)

# Good: Use VALUE to set specific levels
vol_stress = Stress("volatility", StressType.VALUE, 0.80)
```

### 4. Validate Scenarios Before Running

```python
# Validate individual stress
try:
    stress = Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO)
except ValidationError as e:
    print(f"Invalid stress: {e}")

# Validate scenario
scenario = ScenarioBuilder().name("Test").spot_stress(-0.20).build()
try:
    scenario.validate()
except ValidationError as e:
    print(f"Invalid scenario: {e}")
```

### 5. Export Results in Multiple Formats

```python
# Good: Export for different use cases
ResultExporter.export(
    results,
    output_dir="./output",
    formats=['parquet', 'csv', 'json'],
    base_name="stress_test"
)

# Parquet: Fast loading, small size
# CSV: Human-readable, Excel
# JSON: Complete structured data
```

### 6. Generate Reports for Stakeholders

```python
# Executive summary for management
ReportGenerator().generate_report(
    results,
    "./reports/executive_summary.html",
    title="Stress Test Results - Q4 2024"
)

# Detailed analysis for risk team
StressTestVisualizer().create_all_plots(
    results,
    output_dir="./plots",
    prefix="detailed_analysis"
)
```

## Common Pitfalls

### 1. Mismatched Stress Levels

```python
# Wrong: Position-level stress without target
stress = Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.POSITION)
# ValidationError: Target position_id required

# Correct: Specify target
stress = Stress(
    "spot",
    StressType.PERCENTAGE,
    -0.20,
    StressLevel.POSITION,
    target="pos_123"
)
```

### 2. Conflicting Stresses

```python
# Wrong: Multiple stresses on same parameter at same level
scenario = ScenarioBuilder()
scenario.spot_stress(-0.20)
scenario.spot_stress(-0.15)  # Last one wins
# Results in -15% stress

# Correct: Combine into single stress
scenario = ScenarioBuilder()
scenario.spot_stress(-0.20)  # Single stress
# Or use different levels
```

### 3. Missing Metadata for Specialized Stresses

```python
# Wrong: key_rate_stress without tenor_bucket
scenario = ScenarioBuilder()
scenario.key_rate_stress(0.01)  # ValidationError

# Correct: Specify tenor_bucket
scenario = ScenarioBuilder()
scenario.key_rate_stress(0.01, tenor_bucket="5Y")
```

### 4. Not Saving Scenarios

```python
# Wrong: Creating scenarios inline each time
results = engine.run_static_scenarios(
    portfolio,
    [(ScenarioBuilder().name("Test").spot_stress(-0.20).build())]
)

# Good: Save and load scenarios
ScenarioStorage.save_scenarios(scenarios, "scenarios.yaml")
loaded = ScenarioStorage.load_scenarios("scenarios.yaml")
```

## Future Enhancements (Potential TODOs)

1. **Dynamic Scenario Analysis**:
   - Time-series scenario evolution
   - Path-dependent scenarios
   - Hedging strategy integration

2. **Parallel Execution**:
   - Multi-core scenario processing
   - Distributed computing support

3. **Monte Carlo Scenarios**:
   - Stochastic scenario generation
   - Distribution-based stress testing

4. **Historical Replay**:
   - Actual historical market data replay
   - Correlation analysis

5. **Extended Asset Classes**:
   - Crypto stress testing
   - Commodity stress testing
   - FX stress testing

6. **Advanced Analytics**:
   - Scenario attribution analysis
   - Correlation impact analysis
   - What-if sensitivity analysis

7. **Web Interface**:
   - Browser-based scenario builder
   - Interactive results exploration

8. **Cloud Integration**:
   - Distributed stress testing
   - Cloud storage for scenarios and results

## API Reference

### Core Classes

```python
# Engine
class EquityStressEngine:
    def __init__(self, config: EquityStressTestConfig)
    def run_static_scenarios(...) -> StressResultEnvelope
    def evaluate_scenario(...) -> ScenarioEnvelope

# Configuration
@dataclass
class StressTestConfig:
    calculate_greeks: bool
    greeks_method: str
    export_formats: List[str]

# Scenario
@dataclass
class Scenario:
    name: str
    stresses: List[Stress]

@dataclass
class Stress:
    parameter: str
    stress_type: StressType
    stress_value: float
    level: StressLevel
    target: Optional[str]

# Results
class StressTestResults:
    def get_summary() -> Dict[str, Any]
    def get_worst_scenario() -> ScenarioResult
    def to_summary_dataframe() -> pd.DataFrame

# Builder
class ScenarioBuilder:
    def name(str) -> ScenarioBuilder
    def spot_stress(...) -> ScenarioBuilder
    def vol_stress(...) -> ScenarioBuilder
    def build() -> Scenario

# Storage
class ScenarioStorage:
    @staticmethod
    def save_scenarios(scenarios: List[Scenario], filepath: str)
    @staticmethod
    def load_scenarios(filepath: str) -> List[Scenario]
```

### Enums

```python
class StressType(Enum):
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    VALUE = "value"

class StressLevel(Enum):
    PORTFOLIO = "portfolio"
    UNDERLYING = "underlying"
    POSITION = "position"
```

## References

### Academic Papers

1. Basel Committee. "Principles for sound stress testing and supervision"
2. Glasserman, P. "Monte Carlo Methods in Financial Engineering"
3. Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"

### Regulatory Documents

1. Basel III: Framework for the measurement and monitoring of stress testing
2. CCAR: Comprehensive Capital Analysis and Review
3. EBA: Guidelines on Stress Testing

## Support and Resources

- **GitHub Issues**: QuantArk repository
- **Documentation**: QuantArk Docs
- **Email**: quantark-support@example.com
- **Internal Wiki**: [Internal Stress Test Documentation]

---

**Note**: This module is actively maintained. For significant changes or new features, create an OpenSpec proposal following the project guidelines.
