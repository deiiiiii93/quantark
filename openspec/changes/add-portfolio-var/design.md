# Design: Portfolio VaR Module

## Context

Value-at-Risk (VaR) is a standard risk metric that estimates the maximum potential loss over a given time horizon at a specified confidence level. This module provides three industry-standard VaR methodologies for both equity and fixed income portfolios, with comprehensive risk attribution capabilities.

### Stakeholders
- Risk managers requiring daily VaR reports
- Portfolio managers for position sizing
- Regulatory compliance (Basel III/IV requirements)

### Constraints
- Must integrate with existing portfolio structures without modification
- Must support both equity and FI portfolios
- Must handle portfolios with derivatives (non-linear payoffs)
- Performance: 1-day VaR should compute in < 5 seconds for typical portfolios

## Goals / Non-Goals

### Goals
- Implement Parametric, Historical, and Monte Carlo VaR methods
- Support equity risk factors (spot, vol, rate, div yield)
- Support FI risk factors (parallel shift, key-rate shifts)
- Provide component VaR and marginal VaR attribution
- Support configurable confidence levels and holding periods
- Accept historical data from MarketDataAdapter or DataFrame

### Non-Goals
- Real-time streaming VaR (batch calculation only)
- Credit VaR / counterparty risk
- Liquidity-adjusted VaR
- FX risk factors (future enhancement)
- Commodity risk factors (future enhancement)

## Decisions

### Decision 1: Module Structure

```
var/
├── __init__.py                 # Public API exports
├── config.py                   # VaRConfig, RiskFactorConfig, VaRMethod enum
├── base.py                     # VaREngine protocol, VaRResult dataclass
├── engines/
│   ├── __init__.py
│   ├── parametric.py           # ParametricVaREngine
│   ├── historical.py           # HistoricalVaREngine
│   └── monte_carlo.py          # MonteCarloVaREngine
├── results/
│   ├── __init__.py
│   ├── var_result.py           # VaRResult, ScenarioResult
│   └── var_report.py           # VaRReportGenerator
└── risk_factors/
    ├── __init__.py
    ├── base.py                 # RiskFactor protocol
    ├── equity_factors.py       # SpotReturn, VolChange, RateShift, DivYieldShift
    └── fi_factors.py           # ParallelShift, KeyRateShift
```

**Rationale**: Mirrors existing module patterns (stresstest/, backtest/) for consistency. Separates engines by method for maintainability.

### Decision 2: VaR Method Enum Pattern

Follow the existing `EngineType` pattern in `util/enum/engine_enums.py`:

```python
class VaRMethod(Enum):
    PARAMETRIC = auto()
    HISTORICAL = auto()
    MONTE_CARLO = auto()
```

**Rationale**: Consistent with project conventions. Allows future extension with method-specific sub-options.

### Decision 3: Configuration Dataclass

```python
@dataclass
class VaRConfig:
    confidence_level: float = 0.99          # 99% VaR default
    holding_period: int = 1                  # 1-day VaR
    lookback_days: int = 252                 # 1 year of history
    var_method: VaRMethod = VaRMethod.PARAMETRIC
    
    # Risk factors configuration
    equity_factors: Optional[EquityRiskFactorConfig] = None
    fi_factors: Optional[FIRiskFactorConfig] = None
    
    # Scaling
    scaling_method: str = "sqrt_t"           # sqrt(t) rule for multi-day
    
    # Monte Carlo specific
    mc_num_simulations: int = 10000
    mc_seed: Optional[int] = None
    
    # Output options
    calculate_component_var: bool = True
    calculate_marginal_var: bool = True
    calculate_factor_var: bool = True

@dataclass  
class EquityRiskFactorConfig:
    include_spot: bool = True
    include_vol: bool = True
    include_rate: bool = True
    include_div_yield: bool = False

@dataclass
class FIRiskFactorConfig:
    include_parallel_shift: bool = True
    include_key_rates: bool = False
    key_rate_tenors: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 30.0])
```

**Rationale**: Follows `EquityStressConfig` pattern. Separates asset-class-specific settings for clarity.

### Decision 4: VaR Engine Protocol

```python
@runtime_checkable
class VaREngine(Protocol):
    config: VaRConfig
    
    def calculate_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: Union[MarketDataSet, pd.DataFrame],
    ) -> VaRResult:
        """Calculate VaR for the portfolio."""
        ...
    
    def supports_portfolio(self, portfolio: Any) -> bool:
        """Check if this engine supports the portfolio type."""
        ...
```

**Rationale**: Follows `BaseStressEngine` protocol pattern. Unified interface across methods.

### Decision 5: Historical Data Input

Support two input modes:

1. **MarketDataAdapter** (structured): Uses existing `get_spot_history()`, `get_vol_history()`, etc.
2. **DataFrame** (flexible): Direct returns/prices DataFrame for ad-hoc analysis

```python
def calculate_var(
    self,
    portfolio: BasePortfolio,
    historical_data: Union[MarketDataSet, pd.DataFrame],
) -> VaRResult:
    if isinstance(historical_data, pd.DataFrame):
        # Expect columns: 'spot_return', 'vol_change', 'rate_shift', etc.
        scenarios = self._scenarios_from_dataframe(historical_data)
    else:
        # Extract from MarketDataSet
        scenarios = self._scenarios_from_market_data(historical_data)
```

**Rationale**: MarketDataAdapter provides consistency with backtest module. DataFrame provides flexibility for custom data sources.

### Decision 6: VaR Result Structure

```python
@dataclass
class VaRResult:
    # Core metrics
    var: float                              # Total portfolio VaR (positive number)
    cvar: float                             # Conditional VaR / Expected Shortfall
    confidence_level: float
    holding_period: int
    method: VaRMethod
    
    # Portfolio context
    portfolio_value: float
    var_as_pct: float                       # VaR / portfolio_value
    
    # Attribution (optional)
    component_var: Optional[Dict[str, float]] = None    # By position_id
    marginal_var: Optional[Dict[str, float]] = None     # By position_id
    factor_var: Optional[Dict[str, float]] = None       # By risk factor
    
    # Scenarios (for Historical/MC)
    scenarios: Optional[pd.DataFrame] = None
    worst_scenarios: Optional[List[Dict]] = None        # Top N worst scenarios
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    execution_time_seconds: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
```

**Rationale**: Comprehensive results for reporting. Optional fields keep simple cases lightweight.

### Decision 7: Parametric VaR Implementation

For portfolios with derivatives, use delta-gamma approximation:

```
Portfolio P&L ≈ Δ·δS + ½·Γ·(δS)² + ν·δσ + ρ·δr + ...

VaR = -μ + z_α · σ_portfolio
```

Where:
- μ = expected portfolio change (usually 0 for short horizons)
- z_α = standard normal quantile at confidence level
- σ_portfolio = sqrt(w'Σw) where w = factor exposures, Σ = covariance matrix

**Rationale**: Industry standard for linear/near-linear portfolios. Fast computation.

### Decision 8: Historical VaR Implementation

Full revaluation approach:

1. For each historical date t in lookback:
   - Extract risk factor changes: δS_t, δσ_t, δr_t, ...
   - Create stressed `PricingEnvironment` with shifted parameters
   - Reprice entire portfolio
   - Record P&L_t = V_stressed - V_base
2. Sort P&L scenarios
3. VaR = -percentile(P&L, 1 - confidence_level)
4. CVaR = -mean(P&L where P&L < -VaR)

**Rationale**: No distributional assumptions. Captures non-linear effects. Industry standard.

### Decision 9: Monte Carlo VaR Implementation

1. Fit multivariate distribution to historical risk factor changes
2. Generate `mc_num_simulations` correlated scenarios using Cholesky decomposition
3. For each simulated scenario:
   - Apply risk factor shocks to pricing environment
   - Full portfolio revaluation
   - Record P&L
4. VaR/CVaR from empirical distribution of simulated P&L

**Rationale**: Flexible distribution fitting. Handles fat tails with appropriate models. Standard approach.

### Decision 10: Multi-day VaR Scaling

Support two methods:

1. **sqrt_t** (default): `VaR_N = VaR_1 × √N`
   - Assumes i.i.d. returns
   - Fast, commonly used

2. **overlapping** (more accurate): Use N-day overlapping returns from history
   - Captures serial correlation
   - More computationally intensive

**Rationale**: sqrt_t is industry standard for regulatory VaR. Overlapping available for accuracy.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Full revaluation slow for large portfolios | Cache pricing where possible; parallel processing for MC |
| Parametric VaR inaccurate for options | Warn user; recommend Historical/MC for non-linear books |
| Historical data quality issues | Validate data completeness; handle missing values |
| Overfitting with limited history | Require minimum lookback (e.g., 250 days) |

### Decision 11: Stressed VaR

Include Stressed VaR (SVaR) as an additional output, computed using a crisis period window:

```python
@dataclass
class VaRConfig:
    # ... existing fields ...
    calculate_stressed_var: bool = True
    stressed_period_start: Optional[datetime] = None  # e.g., 2008-09-01
    stressed_period_end: Optional[datetime] = None    # e.g., 2009-03-31
    stressed_lookback_days: int = 252                 # 1 year stressed window
```

SVaR computation:
1. Identify a 12-month stressed period (highest volatility or user-specified)
2. Run Historical VaR using only stressed period scenarios
3. Report both regular VaR and SVaR

**Rationale**: Basel III requires SVaR for regulatory capital. Captures tail risk during crisis periods.

### Decision 12: VaR Backtesting

Include backtesting capabilities to validate VaR model accuracy:

```python
@dataclass
class VaRBacktestResult:
    num_observations: int
    num_exceptions: int                    # Days where loss > VaR
    exception_rate: float                  # num_exceptions / num_observations
    expected_exceptions: float             # (1 - confidence_level) * num_observations
    
    # Statistical tests
    kupiec_pof_statistic: float           # Proportion of Failures test
    kupiec_pof_pvalue: float
    kupiec_pof_pass: bool                 # p-value > 0.05
    
    christoffersen_statistic: float       # Independence + coverage test
    christoffersen_pvalue: float
    christoffersen_pass: bool
    
    # Basel traffic light
    basel_zone: str                        # "green", "yellow", "red"
    
    exceptions_dates: List[datetime]       # Dates of VaR breaches
```

Backtesting workflow:
1. For each day t in backtest period:
   - Compute VaR_{t-1} (predicted)
   - Observe actual P&L_t
   - Record exception if P&L_t < -VaR_{t-1}
2. Run Kupiec POF test (binomial test for exception frequency)
3. Run Christoffersen test (joint test for coverage + independence)
4. Assign Basel traffic light zone based on exception count

**Rationale**: Regulatory requirement. Essential for model validation and governance.

### Decision 13: Incremental VaR

Include Incremental VaR (IVaR) to measure impact of adding/removing positions:

```python
@dataclass
class IncrementalVaRResult:
    position_id: str
    current_var: float                     # Portfolio VaR with position
    var_without_position: float            # Portfolio VaR excluding position
    incremental_var: float                 # current_var - var_without_position
    incremental_var_pct: float             # As percentage of current VaR
```

Computation:
1. Compute portfolio VaR with all positions
2. For each position p:
   - Create portfolio copy excluding position p
   - Compute VaR of reduced portfolio
   - IVaR_p = VaR_full - VaR_without_p

Difference from Marginal VaR:
- **Marginal VaR**: Sensitivity of VaR to small change in position size (derivative)
- **Incremental VaR**: Discrete impact of fully adding/removing a position

**Rationale**: Critical for position sizing, risk budgeting, and trade approval workflows.

## Migration Plan

No migration required - this is a new module with no breaking changes to existing code.

