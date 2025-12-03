# VaR Module Implementation Plan

## Executive Summary

This implementation plan addresses the remaining tasks to complete the VaR (Value-at-Risk) module. The plan prioritizes critical blocking issues first, then proceeds with engine enhancements, attribution features, and documentation.

**Status**: 47% complete (31/66 tasks done)
**Critical Blocking Issues**: 2
**Estimated Implementation Time**: 5-7 days

---

## Phase 1: Critical Infrastructure (DAY 1) - UNBLOCKS ALL OTHER WORK

### Task 1.1: Create Missing `var/results/` Module [BLOCKING]
**Priority**: CRITICAL - Must complete before any other work
**Files**: `var/results/__init__.py`, `var/results/var_result.py`, `var/results/incremental_var_result.py`, `var/results/var_report.py`

**Implementation Approach**:
1. Create `var/results/` directory structure
2. Move `VaRResult` from `var/base.py` to `var/results/var_result.py` as per design
3. Create `IncrementalVaRResult` dataclass for incremental VaR calculations
4. Ensure `var/__init__.py` can import from the new location

**Dependencies**: None
**Complexity**: Low
**Testing**:
- Verify imports work correctly
- Test VaRResult creation and access
- Run existing tests to ensure no breakage

---

### Task 1.2: Fix VaRResult Import Paths
**Files**: `var/base.py`, `var/results/var_result.py`

**Implementation Approach**:
1. Move VaRResult class definition to `var/results/var_result.py`
2. Update `var/base.py` to import VaRResult from results module
3. Update all engines to import VaRResult from var.base
4. Update `var/__init__.py` exports

**Dependencies**: Task 1.1
**Complexity**: Low

---

## Phase 2: Historical VaR Engine Completion (DAY 1-2)

### Task 2.1: Implement MarketDataSet Support for Historical VaR
**Files**: `var/engines/historical.py`

**Implementation Approach**:
```python
def _scenarios_from_market_data(self, market_data: MarketDataSet) -> pd.DataFrame:
    """Extract scenarios from MarketDataSet."""
    # Get all available underlyings
    underlyings = market_data.get_underlyings()

    scenarios_data = {}
    for underlying in underlyings:
        # Get historical data for each underlying
        hist_data = market_data.get_underlying_data(underlying)
        scenarios_data[f"{underlying}_spot_return"] = hist_data['spot'].pct_change()

    # Combine all risk factors
    scenarios_df = pd.DataFrame(scenarios_data)
    scenarios_df = scenarios_df.dropna()

    return scenarios_df
```

**Dependencies**: None (market data adapters exist)
**Complexity**: Medium
**Testing**:
- Test with mock MarketDataSet
- Verify DataFrame format matches expected structure

---

### Task 2.2: Complete Multi-Day Historical VaR (Overlapping Returns)
**Files**: `var/engines/historical.py`

**Implementation Approach**:
```python
def _compute_overlapping_scenarios(
    self,
    scenarios: pd.DataFrame,
    lookback_days: int,
    holding_period: int
) -> pd.DataFrame:
    """Generate overlapping scenarios for multi-day VaR."""
    overlapping_scenarios = []

    for i in range(lookback_days - holding_period + 1):
        scenario_window = scenarios.iloc[i:i + holding_period]
        combined_return = (1 + scenario_window).prod() - 1
        overlapping_scenarios.append(combined_return)

    return pd.DataFrame(overlapping_scenarios)
```

Then update `calculate_var()` to use the scaling method from config:
```python
if self.config.holding_period > 1 and self.config.scaling_method == "overlapping":
    scenarios = self._compute_overlapping_scenarios(scenarios, len(scenarios), self.config.holding_period)
elif self.config.holding_period > 1 and self.config.scaling_method == "sqrt_t":
    # Keep single-day scenarios, scaled later
    pass
```

**Dependencies**: None
**Complexity**: High
**Testing**:
- Test with synthetic 5-day return scenario
- Verify overlap matches manual calculation

---

### Task 2.3: Enhance Stress Environment for Vol and Rate Shocks
**Files**: `var/engines/historical.py`, `var/engines/monte_carlo.py`

**Implementation Approach**:
For Historical Engine:
```python
def _create_stressed_environment(
    self,
    base_env: PricingEnvironment,
    scenario: pd.Series,
) -> PricingEnvironment:
    """Create stressed pricing environment from scenario."""
    from copy import deepcopy

    stressed_env = deepcopy(base_env)

    # Spot shock
    if "spot_return" in scenario.index:
        stressed_env.spot_quote.spot = base_env.spot * (1 + scenario["spot_return"])

    # Volatility shock (percentage point change)
    if "vol_change" in scenario.index:
        vol_surface = deepcopy(base_env.vol_surface)
        vol_shock = scenario["vol_change"]
        # Apply shock to all strikes/tenors
        new_vols = vol_surface.get_all_vols() * (1 + vol_shock)
        stressed_env.vol_surface = FlatVolSurface(new_vols)

    # Rate shock (basis points)
    if "rate_shift" in scenario.index:
        rate_curve = deepcopy(base_env.rate_curve)
        rate_shock = scenario["rate_shift"] / 10000  # Convert bp to decimal
        stressed_env.rate_curve = rate_curve.with_parallel_shift(rate_shock)

    return stressed_env
```

**Dependencies**: None
**Complexity**: Medium
**Testing**:
- Test vol surface cloning and modification
- Test rate curve shifting
- Test stress environment creation

---

## Phase 3: Monte Carlo VaR Engine Enhancement (DAY 2)

### Task 3.1: Implement MarketDataSet Support for Monte Carlo VaR
**Files**: `var/engines/monte_carlo.py`

**Implementation Approach**:
Similar to historical engine, add MarketDataSet support:
```python
def _extract_scenarios_from_market_data(self, market_data: MarketDataSet) -> pd.DataFrame:
    """Extract scenarios from MarketDataSet."""
    # Implementation similar to historical engine
    # Get risk factor returns for all underlyings
```

**Dependencies**: None
**Complexity**: Medium
**Testing**:
- Test scenario extraction
- Verify Cholesky decomposition works

---

### Task 3.2: Enhance Stress Environment for Monte Carlo Engine
**Files**: `var/engines/monte_carlo.py`

**Implementation Approach**:
Implement same stressed environment logic as historical engine.

**Dependencies**: Task 2.3
**Complexity**: Low
**Testing**:
- Reuse tests from historical engine

---

## Phase 4: VaR Attribution Implementation (DAY 2-3)

### Task 4.1: Implement Component VaR (Euler Allocation)
**Files**: `var/attribution.py` (new file)

**Implementation Approach**:
Create new module for attribution:
```python
class VaRAttributionCalculator:
    """Calculate VaR attribution components using Euler decomposition."""

    def calculate_component_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        total_var: float,
        var_engine: VaREngine,
        historical_data: Union[any, pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate component VaR using Euler method."""
        component_var = {}

        for position_id, position in portfolio.positions.items():
            # Remove position from portfolio
            temp_portfolio = portfolio.remove_position(position_id)

            # Calculate VaR without this position
            var_without = var_engine.calculate_var(temp_portfolio, historical_data).var

            # Component VaR = Total VaR - VaR without this position
            component_var[position_id] = total_var - var_without

        return component_var
```

**Dependencies**: None
**Complexity**: High
**Testing**:
- Verify sum(component_var) equals total VaR
- Test with single-position portfolio

---

### Task 4.2: Implement Marginal VaR
**Files**: `var/attribution.py`

**Implementation Approach**:
```python
def calculate_marginal_var(
    self,
    portfolio: Union[EquityPortfolio, FIPortfolio],
    total_var: float,
    var_engine: VaREngine,
    historical_data: Union[any, pd.DataFrame],
    position_id: str
) -> float:
    """Calculate marginal VaR for a specific position."""
    position = portfolio.positions[position_id]

    # Calculate derivative approximation
    small_increment = 0.01  # 1% of position
    temp_portfolio = portfolio.add_position(
        position_id,
        position.with_quantity(position.quantity + small_increment)
    )

    var_with_increment = var_engine.calculate_var(temp_portfolio, historical_data).var
    marginal_var = (var_with_increment - total_var) / small_increment

    return marginal_var
```

**Dependencies**: Task 4.1
**Complexity**: Medium
**Testing**:
- Test marginal VaR calculation
- Verify numerical stability

---

### Task 4.3: Add Attribution to VaRResult
**Files**: `var/results/var_result.py`

**Implementation Approach**:
Update VaRResult to include component and marginal VaR:
```python
@dataclass
class VaRResult:
    # ... existing fields ...
    component_var: Optional[Dict[str, float]] = None
    marginal_var: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate attribution adds to total VaR."""
        if self.component_var:
            total_component = sum(self.component_var.values())
            # Allow small tolerance for numerical differences
            assert abs(total_component - self.var) < self.var * 0.01, \
                f"Component VaR sum {total_component} != total VaR {self.var}"
```

**Dependencies**: None
**Complexity**: Low
**Testing**:
- Test validation logic

---

### Task 4.4: Write Unit Tests for Attribution
**Files**: `test/test_var_attribution.py` (new file)

**Testing Strategy**:
- Test component VaR sums to total VaR (within tolerance)
- Test marginal VaR numerical accuracy
- Test with single-position portfolio
- Test diversification scenarios (component VaR > individual positions)

**Dependencies**: Tasks 4.1-4.3
**Complexity**: Medium

---

## Phase 5: Stressed VaR Implementation (DAY 3-4)

### Task 5.1: Implement Auto-Detection of Stressed Period
**Files**: `var/stressed_var.py` (new file)

**Implementation Approach**:
```python
def find_stressed_period(
    self,
    historical_data: pd.DataFrame,
    lookback_days: int = 252,
) -> Tuple[datetime, datetime]:
    """Find highest volatility 12-month period."""
    # Calculate rolling volatility
    rolling_vol = historical_data['spot_return'].rolling(window=252).std()

    # Find period with maximum volatility
    max_vol_idx = rolling_vol.idxmax()
    period_end = max_vol_idx
    period_start = period_end - timedelta(days=365)

    return period_start, period_end
```

**Dependencies**: None
**Complexity**: Medium
**Testing**:
- Test with synthetic crisis period
- Verify period length (365 days)

---

### Task 5.2: Implement SVaR Calculation
**Files**: `var/stressed_var.py`, `var/engines/*.py`

**Implementation Approach**:
Update each engine to support stressed VaR:
```python
def calculate_stressed_var(
    self,
    portfolio: Union[EquityPortfolio, FIPortfolio],
    historical_data: pd.DataFrame,
    config: VaRConfig
) -> VaRResult:
    """Calculate Stressed VaR."""
    # Auto-detect or use user-specified period
    if config.stressed_period_start and config.stressed_period_end:
        stressed_period_data = historical_data[
            (historical_data.index >= config.stressed_period_start) &
            (historical_data.index <= config.stressed_period_end)
        ]
    else:
        # Auto-detect
        stress_start, stress_end = self._find_stressed_period(
            historical_data, config.stressed_lookback_days
        )
        stressed_period_data = historical_data[
            (historical_data.index >= stress_start) &
            (historical_data.index <= stress_end)
        ]

    # Calculate VaR using stressed period data
    result = self.calculate_var(portfolio, stressed_period_data)

    # Mark as stressed VaR
    result.stressed_var = result.var
    result.stressed_cvar = result.cvar
    result.stressed_period = {
        "start": stress_start,
        "end": stress_end
    }

    return result
```

**Dependencies**: Task 5.1
**Complexity**: High
**Testing**:
- Test with known crisis periods (e.g., 2008, 2020)
- Verify stressed VaR > regular VaR

---

### Task 5.3: Integrate Stressed VaR into Base Engines
**Files**: `var/engines/parametric.py`, `var/engines/historical.py`, `var/engines/monte_carlo.py`

**Implementation Approach**:
Add method to each engine:
```python
def calculate_stressed_var(
    self,
    portfolio: Union[EquityPortfolio, FIPortfolio],
    historical_data: pd.DataFrame,
) -> VaRResult:
    """Calculate Stressed VaR if enabled in config."""
    if not self.config.calculate_stressed_var:
        return None

    # Calculate stressed VaR
    stressed_result = self._compute_stressed_var(portfolio, historical_data)

    # Update main result with stressed values
    main_result = self.calculate_var(portfolio, historical_data)
    main_result.stressed_var = stressed_result.var
    main_result.stressed_cvar = stressed_result.cvar
    main_result.stressed_period = stressed_result.stressed_period

    return main_result
```

**Dependencies**: Task 5.2
**Complexity**: Medium

---

### Task 5.4: Write Unit Tests for Stressed VaR
**Files**: `test/test_stressed_var.py` (new file)

**Testing Strategy**:
- Test auto-detection of stressed period
- Test user-specified stressed period
- Test with known crisis data (if available)
- Verify stressed VaR calculation

**Dependencies**: Tasks 5.1-5.3
**Complexity**: Medium

---

## Phase 6: Incremental VaR Implementation (DAY 4-5)

### Task 6.1: Implement Incremental VaR Calculation
**Files**: `var/incremental_var.py` (new file)

**Implementation Approach**:
```python
def calculate_incremental_var(
    self,
    portfolio: Union[EquityPortfolio, FIPortfolio],
    historical_data: pd.DataFrame,
    var_engine: VaREngine,
    position_id: str
) -> float:
    """Calculate Incremental VaR for a position."""
    # Get total VaR
    total_var = var_engine.calculate_var(portfolio, historical_data).var

    # Calculate VaR without position
    portfolio_without = portfolio.remove_position(position_id)
    var_without = var_engine.calculate_var(portfolio_without, historical_data).var

    # Incremental VaR = Total VaR - VaR without position
    incremental_var = total_var - var_without

    return incremental_var

def calculate_all_incremental_var(
    self,
    portfolio: Union[EquityPortfolio, FIPortfolio],
    historical_data: pd.DataFrame,
    var_engine: VaREngine
) -> Dict[str, float]:
    """Calculate Incremental VaR for all positions."""
    incremental_vars = {}

    for position_id in portfolio.positions.keys():
        inc_var = self.calculate_incremental_var(
            portfolio, historical_data, var_engine, position_id
        )
        incremental_vars[position_id] = inc_var

    return incremental_vars
```

**Dependencies**: None
**Complexity**: Medium
**Testing**:
- Test single position calculation
- Test all positions calculation
- Verify values match manual calculation

---

### Task 6.2: Implement IncrementalVaRResult Class
**Files**: `var/results/incremental_var_result.py`

**Implementation Approach**:
```python
@dataclass
class IncrementalVaRResult:
    """Results from Incremental VaR calculation."""

    position_id: str
    incremental_var: float
    component_var: float
    marginal_var: float

    contribution_to_portfolio_var: float
    diversification_benefit: float

    base_portfolio_var: float
    portfolio_var_without_position: float

    portfolio_value: float
    position_value: float
```

**Dependencies**: None
**Complexity**: Low
**Testing**:
- Test result creation and fields

---

### Task 6.3: Add Incremental VaR to VaRResult
**Files**: `var/results/var_result.py`

**Implementation Approach**:
Update VaRResult to store incremental VaR results:
```python
@dataclass
class VaRResult:
    # ... existing fields ...
    incremental_var: Optional[Dict[str, float]] = None

    def add_incremental_var(self, incremental_var_dict: Dict[str, float]):
        """Add incremental VaR results."""
        self.incremental_var = incremental_var_dict
```

**Dependencies**: Tasks 6.1-6.2
**Complexity**: Low

---

### Task 6.4: Write Unit Tests for Incremental VaR
**Files**: `test/test_incremental_var.py` (new file)

**Testing Strategy**:
- Test incremental VaR calculation
- Test diversification scenarios
- Test empty portfolio edge case
- Verify values with known simple portfolios

**Dependencies**: Tasks 6.1-6.3
**Complexity**: Medium

---

## Phase 7: Parametric Engine MarketDataSet Support (DAY 5)

### Task 7.1: Implement MarketDataSet Support for Parametric VaR
**Files**: `var/engines/parametric.py`

**Implementation Approach**:
```python
def _extract_risk_factors_from_market_data(
    self, market_data: MarketDataSet, is_equity: bool
) -> pd.DataFrame:
    """Extract risk factors from MarketDataSet."""
    underlyings = market_data.get_underlyings()

    if is_equity:
        risk_factors_data = {}
        for underlying in underlyings:
            hist_data = market_data.get_underlying_data(underlying)

            # Extract each risk factor
            risk_factors_data[f"{underlying}_spot_return"] = hist_data['spot'].pct_change()

        return pd.DataFrame(risk_factors_data)
    else:
        # Similar for FI with rate data
        raise NotImplementedError("FI MarketDataSet not yet supported")
```

**Dependencies**: None
**Complexity**: Medium
**Testing**:
- Test with mock MarketDataSet
- Verify covariance matrix calculation

---

## Phase 8: Documentation and Validation (DAY 5-6)

### Task 8.1: Create var/README.md
**Files**: `var/README.md` (new file)

**Content Structure**:
1. Overview of VaR module
2. Quick start guide with examples
3. Configuration options
4. Engine comparison (parametric vs historical vs Monte Carlo)
5. Attribution explanation
6. Stressed VaR usage
7. API reference
8. Example scripts

**Example Usage**:
```python
from var import VaRConfig, HistoricalVaREngine

config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    lookback_days=252
)

engine = HistoricalVaREngine(config)
var_result = engine.calculate_var(portfolio, historical_data)

print(f"VaR: ${var_result.var:,.2f}")
print(f"CVaR: ${var_result.cvar:,.2f}")
```

**Dependencies**: All prior tasks
**Complexity**: Medium

---

### Task 8.2: Run Full Test Suite and Coverage
**Files**: All test files

**Actions**:
1. Run all VaR tests: `python -m pytest test/test_var*.py -v`
2. Check coverage: `python -m pytest --cov=var`
3. Fix any failing tests
4. Target coverage: >90%

**Commands**:
```bash
python -m pytest test/test_var_config.py test/test_var_backtest.py -v
python -m pytest --cov=var test/
python -m pytest --cov=var --cov-report=html
```

**Dependencies**: All prior tasks
**Complexity**: Low

---

### Task 8.3: Validate Against Benchmarks
**Files**: `test/benchmark/test_var_benchmarks.py` (new file)

**Validation Strategy**:
1. Parametric VaR for single call option (Black-Scholes)
2. Historical VaR for simple portfolio (known results)
3. Monte Carlo VaR convergence test
4. Compare with Bloomberg/Reuters benchmark values (if available)

**Example**:
```python
def test_parametric_var_bs_model():
    """Test parametric VaR matches Black-Scholes delta-gamma."""
    # Known portfolio: 1 ATM call, spot=100, vol=20%, r=5%
    # Expected VaR at 99% for 1-day ≈ $1.645 * delta_stdev
    # Calculate manually and compare

    # Assert within 5% tolerance
```

**Dependencies**: All prior tasks
**Complexity**: High

---

### Task 8.4: Validate Backtesting
**Files**: `test/validation/test_var_backtest_validation.py` (new file)

**Validation Strategy**:
1. Create synthetic exception scenarios
2. Verify Kupiec POF test accuracy
3. Verify Christoffersen test accuracy
4. Verify Basel traffic light zones

**Example**:
```python
def test_kupiec_pof_accuracy():
    """Test Kupiec POF test with known exception rates."""
    # 1% expected, 3 exceptions in 250 observations
    # Should pass (3 < 4.3 expected + 3*std)

    backtester = VaRBacktester(confidence_level=0.99)
    result = backtester.run_backtest(...)

    assert result.kupiec_pof_pass == True
```

**Dependencies**: All prior tasks
**Complexity**: High

---

## Phase 9: Final Integration and Documentation (DAY 6-7)

### Task 9.1: Update Portfolio Module Integration
**Files**: `portfolio/equity/portfolio.py`, `portfolio/fi/portfolio.py`

**Implementation Approach**:
Add convenience methods:
```python
def calculate_var(
    self,
    historical_data: pd.DataFrame,
    confidence_level: float = 0.99,
    holding_period: int = 1,
    method: VaRMethod = VaRMethod.HISTORICAL
) -> VaRResult:
    """Calculate VaR for portfolio."""
    from var import VaRConfig, HistoricalVaREngine, ParametricVaREngine, MonteCarloVaREngine

    config = VaRConfig(
        confidence_level=confidence_level,
        holding_period=holding_period,
        var_method=method
    )

    engine_map = {
        VaRMethod.HISTORICAL: HistoricalVaREngine,
        VaRMethod.PARAMETRIC: ParametricVaREngine,
        VaRMethod.MONTE_CARLO: MonteCarloVaREngine
    }

    engine = engine_map[method](config)
    return engine.calculate_var(self, historical_data)
```

**Dependencies**: All prior tasks
**Complexity**: Low

---

### Task 9.2: Create Example Scripts
**Files**: `example/var_examples.py` (new file)

**Examples to Include**:
1. Basic VaR calculation with all three methods
2. VaR attribution example
3. Stressed VaR calculation
4. Incremental VaR comparison
5. Portfolio backtesting
6. Multi-asset portfolio (equity + FI)

**Dependencies**: All prior tasks
**Complexity**: Medium

---

### Task 9.3: Performance Optimization
**Files**: All engine files

**Optimization Strategies**:
1. Vectorized portfolio revaluation where possible
2. Cache sensitivity calculations
3. Parallel scenario processing for historical/Monte Carlo
4. Optimize memory usage for large scenarios

**Testing**:
- Profile with 10,000 simulation Monte Carlo
- Profile with 1,000-day historical VaR
- Benchmark against baseline

**Dependencies**: All prior tasks
**Complexity**: Medium

---

## Critical Files for Implementation

Based on the analysis, the most critical files to implement are:

1. **`var/results/__init__.py`** - Creates missing module and exports (BLOCKING)
2. **`var/results/var_result.py`** - VaRResult class definition
3. **`var/results/incremental_var_result.py`** - IncrementalVaRResult class
4. **`var/results/var_report.py`** - VaRReportGenerator for reporting
5. **`var/attribution.py`** - Component and marginal VaR calculations

---

## Testing Strategy Summary

### Unit Tests Required:
- `test/test_var_attribution.py` - Component/marginal VaR
- `test/test_stressed_var.py` - Stressed VaR
- `test/test_incremental_var.py` - Incremental VaR
- `test/test_var_engines_enhanced.py` - Enhanced engine features
- `test/benchmark/test_var_benchmarks.py` - Benchmark validation
- `test/validation/test_var_backtest_validation.py` - Backtest validation

### Integration Tests:
- End-to-end VaR calculation with all engines
- VaR attribution with multi-asset portfolio
- Stressed VaR auto-detection
- Incremental VaR for diversified portfolio

### Performance Tests:
- Monte Carlo VaR with 100k simulations
- Historical VaR with 5-year lookback
- Multi-day VaR with overlapping returns

---

## Risk Assessment

### High-Risk Areas:
1. **Component VaR Euler decomposition** - Numerical stability issues
2. **Multi-day overlapping returns** - Complex window calculations
3. **Stressed VaR period detection** - Market-specific logic
4. **MarketDataSet integration** - Interface compatibility

### Mitigation Strategies:
1. Extensive unit tests with known values
2. Tolerance-based comparisons (within 1% of expected)
3. Fallback logic for edge cases
4. Validation with synthetic data

---

## Success Criteria

1. **All tasks in checklist completed** (66/66 tasks)
2. **Test coverage >90%** for var module
3. **All imports work correctly** (no missing modules)
4. **Engine methods fully implemented** (no NotImplementedError)
5. **Benchmark validation passes** (within tolerance)
6. **Documentation complete** (README with examples)
7. **Integration tests pass** (end-to-end scenarios)

---

## Timeline Summary

**Total Estimated Time**: 5-7 days
- **Phase 1 (Critical)**: 1 day
- **Phase 2-3 (Engines)**: 1-2 days
- **Phase 4-6 (Attribution)**: 2-3 days
- **Phase 7-9 (Integration/Docs)**: 1-2 days

**Parallelization Opportunities**:
- Tasks 2.1-2.3 can be done together
- Tasks 4.1-4.3 (attribution) can be done together
- Tasks 5.1-5.2 (stressed VaR) can be done together
- Documentation (8.1) can be done while tests run

---

## Dependencies Flow

```
Phase 1 (BLOCKING)
  ↓
Phase 2-3 (Engines)
  ↓
Phase 4 (Attribution) ← Can start after engines
  ↓
Phase 5 (Stressed VaR) ← Can start after attribution
  ↓
Phase 6 (Incremental VaR) ← Independent
  ↓
Phase 7-9 (Integration/Docs)
```

---

## Conclusion

This implementation plan provides a clear roadmap to complete the VaR module. The critical path is:
1. Create missing `var/results/` module (UNBLOCKS EVERYTHING)
2. Complete Historical VaR engine features
3. Implement VaR attribution
4. Implement Stressed VaR
5. Implement Incremental VaR
6. Documentation and validation

Total effort: 5-7 days with experienced developer. All tasks are achievable and well-defined. The plan prioritizes blocking issues first, ensuring steady progress.
