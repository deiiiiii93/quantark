# VaR Module - AI Agent Guide

## Purpose

This guide is specifically for AI agents working with the QuantArk VaR module. It provides targeted guidance on common tasks, patterns, and pitfalls to help you work effectively with the VaR implementation.

## Quick Start for AI Agents

### Understanding the VaR Module Structure

```
var/
├── base.py               # VaR engine protocol
├── engines/              # VaR engine implementations
│   ├── parametric.py     # Fastest (variance-covariance)
│   ├── historical.py     # Accurate (full revaluation)
│   └── monte_carlo.py    # Flexible (simulated scenarios)
├── config.py             # VaRConfig and related classes
├── attribution.py        # Component, Marginal, Incremental VaR
├── results/              # Result classes and reporting
│   ├── var_result.py
│   ├── incremental_var_result.py
│   └── var_report.py
├── risk_factors/         # Risk factor implementations
│   ├── base.py
│   ├── equity_factors.py
│   └── fi_factors.py
├── backtest/             # VaR backtesting framework
├── doc/                  # Implementation notes and examples
└── README.md             # User-facing documentation
```

### Common Imports

```python
# Main classes
from var import VaRConfig, VaRMethod
from var.engines import HistoricalVaREngine, ParametricVaREngine, MonteCarloVaREngine
from var.results import VaRResult, IncrementalVaRResult, VaRReportGenerator

# Attribution
from var.attribution import ComponentVaRCalculator, MarginalVaRCalculator

# Risk factors
from var.config import EquityRiskFactorConfig, FIRiskFactorConfig
```

## Task-Oriented Guidance

### Task 1: Adding a New VaR Engine

**When**: When you need to implement a new VaR calculation methodology (e.g., Extreme Value Theory, GARCH-based VaR).

**Steps**:

1. **Create engine class** in `var/engines/your_method.py`:

```python
from typing import Optional, Union
from var.base import VaREngine
from var.config import VaRConfig
from var.results import VaRResult, IncrementalVaRResult

class YourVaREngine(VaREngine):
    """
    Your custom VaR engine.

    Follow the VaREngine protocol.
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        """Initialize engine with configuration."""
        self.config = config if config is not None else VaRConfig()
        # Ensure method matches
        self.config.var_method = VaRMethod.YOUR_METHOD

    def calculate_var(
        self,
        portfolio: Any,
        historical_data: Union[Any, pd.DataFrame],
    ) -> VaRResult:
        """
        Calculate VaR following the protocol.

        Must implement all features from VaRResult:
        - var, cvar
        - component_var (if configured)
        - marginal_var (if configured)
        - factor_var (if configured)
        - incremental_var (if configured)
        - stressed_var (if configured)
        """
        # Implementation here
        pass

    def calculate_incremental_var(
        self,
        portfolio: Any,
        historical_data: Union[Any, pd.DataFrame],
    ) -> IncrementalVaRResult:
        """Calculate Incremental VaR."""
        # Implementation here
        pass

    def supports_portfolio(self, portfolio: Any) -> bool:
        """Check if portfolio type is supported."""
        return isinstance(portfolio, (EquityPortfolio, FIPortfolio))
```

2. **Add to exports** in `var/__init__.py`:

```python
from var.engines.your_method import YourVaREngine

__all__ = [
    # ... existing exports
    "YourVaREngine",
]
```

3. **Add to exports** in `var/engines/__init__.py`:

```python
from var.engines.your_method import YourVaREngine

__all__ = [
    "HistoricalVaREngine",
    "ParametricVaREngine",
    "MonteCarloVaREngine",
    "YourVaREngine",  # Add here
]
```

4. **Add VaRMethod enum value** in `var/config.py`:

```python
class VaRMethod(Enum):
    PARAMETRIC = auto()
    HISTORICAL = auto()
    MONTE_CARLO = auto()
    YOUR_METHOD = auto()  # Add new method
```

5. **Add tests** in `test/test_var_your_method.py`:

```python
import pytest
from var import VaRConfig, VaRMethod
from var.engines import YourVaREngine

def test_your_var_engine_basic():
    """Test basic VaR calculation."""
    config = VaRConfig(var_method=VaRMethod.YOUR_METHOD)
    engine = YourVaREngine(config=config)
    result = engine.calculate_var(portfolio, data)
    assert result.var >= 0
    assert result.cvar >= result.var

def test_your_var_engine_attribution():
    """Test attribution calculations."""
    config = VaRConfig(
        var_method=VaRMethod.YOUR_METHOD,
        calculate_component_var=True,
        calculate_marginal_var=True
    )
    engine = YourVaREngine(config=config)
    result = engine.calculate_var(portfolio, data)
    assert result.component_var is not None
    assert result.marginal_var is not None
```

**Key Requirements**:
- ✅ Follow the `VaREngine` protocol exactly
- ✅ Return complete `VaRResult` with all configured attribution
- ✅ Support both DataFrame and MarketDataSet inputs
- ✅ Handle both EquityPortfolio and FIPortfolio
- ✅ Include comprehensive tests
- ✅ Update all __init__.py exports
- ✅ Add to VaRMethod enum

### Task 2: Adding New Risk Factors

**When**: When you need to add a new risk factor (e.g., FX risk, commodity risk, credit spread risk).

**Steps**:

1. **Identify risk factor type**: Equity or Fixed Income?

2. **Create risk factor class** in appropriate file:

For equity (in `var/risk_factors/equity_factors.py`):

```python
class YourEquityRiskFactor(RiskFactor):
    """Your equity risk factor."""

    def __init__(self, name: str = "your_factor"):
        """Initialize with name."""
        self.name = name
        self._covariance_matrix = None

    def get_scenarios(
        self,
        market_data: Union[pd.DataFrame, MarketDataSet],
        lookback_days: int = 252,
    ) -> pd.Series:
        """
        Generate scenarios for this risk factor.

        Returns:
            Series of risk factor changes
        """
        # Implementation: extract from market data
        pass

    def get_covariance_matrix(
        self,
        market_data: Union[pd.DataFrame, MarketDataSet],
    ) -> pd.DataFrame:
        """Get covariance with other risk factors."""
        # Implementation
        pass
```

3. **Use in ParametricVaREngine** (in `var/engines/parametric.py`):

```python
from var.risk_factors.your_equity_factor import YourEquityRiskFactor

# In calculate_var method, add:
if self.config.equity_factors and self.config.equity_factors.include_your_factor:
    your_factor = YourEquityRiskFactor()
    # Add to risk factor list
    risk_factors.append(your_factor)
```

4. **Add to EquityRiskFactorConfig** (in `var/config.py`):

```python
@dataclass
class EquityRiskFactorConfig:
    include_spot: bool = True
    include_vol: bool = True
    include_rate: bool = True
    include_div_yield: bool = False
    include_your_factor: bool = False  # Add new factor
```

5. **Add tests** in `test/test_var_risk_factors.py`:

```python
def test_your_equity_risk_factor():
    """Test your risk factor."""
    factor = YourEquityRiskFactor()
    scenarios = factor.get_scenarios(market_data)
    assert len(scenarios) > 0
    # More tests...
```

**Common Risk Factors to Add**:
- FX Risk (currency exposure)
- Commodity Risk (precious metals, energy)
- Credit Spread Risk (credit default swaps)
- Inflation Risk (CPI-based instruments)

### Task 3: Adding New Attribution Methods

**When**: When you need to add new risk attribution beyond Component, Marginal, and Incremental VaR.

**Steps**:

1. **Add to VaRConfig** (in `var/config.py`):

```python
@dataclass
class VaRConfig:
    # ... existing fields
    calculate_component_var: bool = True
    calculate_marginal_var: bool = True
    calculate_factor_var: bool = True
    calculate_incremental_var: bool = False
    calculate_your_attribution: bool = False  # Add new attribution
```

2. **Implement calculator** (in `var/attribution.py`):

```python
class YourAttributionCalculator:
    """Calculate your attribution method."""

    @staticmethod
    def calculate(
        portfolio: Any,
        var_result: VaRResult,
        market_data: Any,
    ) -> Dict[str, float]:
        """
        Calculate your attribution.

        Returns:
            Dictionary mapping position/factor to attribution value
        """
        # Implementation
        pass
```

3. **Integrate into engines** (pick appropriate engine(s)):

```python
# In engine.calculate_var()
if self.config.calculate_your_attribution:
    your_attribution = YourAttributionCalculator.calculate(
        portfolio, var_result, market_data
    )
    result.your_attribution = your_attribution
```

4. **Add to VaRResult** (in `var/results/var_result.py`):

```python
@dataclass
class VaRResult:
    # ... existing fields
    component_var: Optional[Dict[str, float]] = None
    marginal_var: Optional[Dict[str, float]] = None
    factor_var: Optional[Dict[str, float]] = None
    incremental_var: Optional[Dict[str, float]] = None
    your_attribution: Optional[Dict[str, float]] = None  # Add new field
```

5. **Add tests** in `test/test_var_attribution.py`:

```python
def test_your_attribution():
    """Test your attribution calculation."""
    config = VaRConfig(calculate_your_attribution=True)
    engine = YourEngine(config=config)
    result = engine.calculate_var(portfolio, data)
    assert result.your_attribution is not None
    # Validate attribution properties
```

### Task 4: Fixing Bugs in VaR Calculations

**When**: When you encounter incorrect VaR calculations, attribution issues, or numerical errors.

**Debugging Steps**:

1. **Check input validation**:

```python
# Verify portfolio
assert portfolio.positions, "Portfolio must have positions"
assert portfolio.total_value > 0, "Portfolio value must be positive"

# Verify market data
assert len(market_data) >= config.lookback_days, "Insufficient data"
assert not market_data.isnull().any(), "Market data has missing values"
```

2. **Check configuration**:

```python
# Ensure method matches engine
assert config.var_method == VaRMethod.HISTORICAL, "Method mismatch"

# Ensure risk factors match portfolio
if isinstance(portfolio, EquityPortfolio):
    assert config.equity_factors is not None, "Missing equity factors"
elif isinstance(portfolio, FIPortfolio):
    assert config.fi_factors is not None, "Missing FI factors"
```

3. **Validate results**:

```python
result = engine.calculate_var(portfolio, data)

# Basic validation
assert result.var >= 0, f"VaR must be non-negative: {result.var}"
assert result.cvar >= result.var, f"CVaR must be >= VaR: {result.cvar}"

# Attribution validation
if result.component_var:
    total_component_var = sum(result.component_var.values())
    assert abs(total_component_var - result.var) < 0.01, \
        f"Component VaR must sum to total VaR: {total_component_var} vs {result.var}"

if result.marginal_var:
    # Marginal VaR values should be reasonable
    for pos_id, marg_var in result.marginal_var.items():
        assert abs(marg_var) <= result.var * 2, \
            f"Marginal VaR unreasonable: {marg_var}"
```

**Common Bugs and Solutions**:

1. **Bug**: Component VaR doesn't sum to total VaR
   - **Cause**: Incorrect Euler decomposition implementation
   - **Fix**: Check marginal VaR calculation and Euler allocation

2. **Bug**: Historical VaR slower than expected
   - **Cause**: Not using vectorized operations
   - **Fix**: Use pandas/numpy vectorization for portfolio revaluation

3. **Bug**: Monte Carlo VaR results vary between runs
   - **Cause**: Missing random seed
   - **Fix**: Set `mc_seed` in VaRConfig for reproducibility

4. **Bug**: Stressed VaR not calculated
   - **Cause**: `calculate_stressed_var=False` or insufficient data
   - **Fix**: Enable flag and ensure lookback_days >= 252

### Task 5: Adding Performance Optimizations

**When**: When VaR calculations are too slow for production use.

**Optimization Strategies**:

1. **Vectorization** (in engines):

```python
# Slow: Loop through positions
for pos_id, position in portfolio.positions.items():
    pnl[pos_id] = calculate_position_pnl(position, scenario)

# Fast: Vectorized operations
pnl = portfolio.revalue_portfolio_vectorized(scenarios)
```

2. **Caching** (for expensive calculations):

```python
from functools import lru_cache

class YourVaREngine:
    def __init__(self, config):
        self.config = config
        self._covariance_cache = None

    @property
    def covariance_matrix(self):
        """Cache covariance matrix calculation."""
        if self._covariance_cache is None:
            self._covariance_cache = self._calculate_covariance()
        return self._covariance_cache
```

3. **Parallel processing** (for scenario calculations):

```python
from concurrent.futures import ThreadPoolExecutor

def calculate_scenarios_parallel(scenarios, num_workers=4):
    """Calculate P&L in parallel."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_scenario, s) for s in scenarios]
        results = [f.result() for f in futures]
    return results
```

4. **Early termination** (for Incremental VaR):

```python
def calculate_incremental_var_efficient(self, portfolio, data):
    """Calculate IVaR with early termination."""
    # Sort positions by expected contribution
    sorted_positions = self._sort_positions_by_risk(portfolio)

    ivar_results = {}
    for pos_id in sorted_positions:
        # Calculate IVaR
        ivar = self._calculate_single_ivar(portfolio, data, pos_id)

        # Early termination if contribution is negligible
        if ivar < threshold:
            break

        ivar_results[pos_id] = ivar

    return ivar_results
```

### Task 6: Adding New Portfolio Types

**When**: When you need to support new asset classes (e.g., CryptoPortfolio, CommodityPortfolio).

**Steps**:

1. **Create portfolio class** (in appropriate location):

```python
from portfolio.base import PortfolioBase

class CryptoPortfolio(PortfolioBase):
    """Portfolio of cryptocurrency positions."""

    @property
    def positions(self) -> Dict[str, CryptoPosition]:
        """Get crypto positions."""
        # Implementation
        pass

    def get_sensitivities(self, risk_factors: List[RiskFactor]) -> Dict[str, Dict[str, float]]:
        """Get position sensitivities (delta, gamma, etc.)."""
        # Implementation
        pass

    def revalue_portfolio(self, scenarios: pd.DataFrame) -> pd.Series:
        """Revalue portfolio under scenarios."""
        # Implementation
        pass
```

2. **Add to engine** `supports_portfolio()`:

```python
def supports_portfolio(self, portfolio: Any) -> bool:
    """Check if engine supports portfolio type."""
    from portfolio.equity.portfolio import EquityPortfolio
    from portfolio.fi.portfolio import FIPortfolio
    from portfolio.crypto.portfolio import CryptoPortfolio  # Add new type

    return isinstance(portfolio, (EquityPortfolio, FIPortfolio, CryptoPortfolio))
```

3. **Add risk factors** for new asset class:

```python
# Create risk factors for crypto
class CryptoRiskFactor(RiskFactor):
    """Risk factor for crypto markets."""
    pass
```

4. **Add tests** in `test/test_var_new_portfolio.py`:

```python
def test_var_with_crypto_portfolio():
    """Test VaR with crypto portfolio."""
    portfolio = CryptoPortfolio(...)
    config = VaRConfig(confidence_level=0.99)
    engine = HistoricalVaREngine(config=config)
    result = engine.calculate_var(portfolio, crypto_data)
    assert result.var >= 0
```

### Task 7: Backtesting and Validation

**When**: When you need to validate VaR model accuracy.

**Steps**:

1. **Use VaRBacktester**:

```python
from var.backtest import VaRBacktester

backtester = VaRBacktester(confidence_level=0.99)

# Run Kupiec test
result = backtester.run_kupiec_test(
    var_estimates=var_series,
    actual_losses=loss_series
)

print(f"Violations: {result.violations}")
print(f"Violation Rate: {result.violation_rate:.3f}")
print(f"Expected Rate: {result.expected_rate:.3f}")
print(f"Kupiec Test: {'PASS' if result.kupiec_pass else 'FAIL'}")
print(f"P-value: {result.kupiec_pvalue:.4f}")
```

2. **Add new backtest metrics**:

```python
# In var/backtest/var_backtester.py
class VaRBacktestResult:
    # ... existing fields
    christoffersen_test: bool = False
    christoffersen_pvalue: float = 0.0

    def run_christoffersen_test(self):
        """Test independence of violations."""
        # Implementation
        pass
```

## Common Patterns and Anti-Patterns

### ✅ DO: Follow These Patterns

1. **Use the VaREngine protocol**:

```python
# Good: Implement the protocol
class YourEngine(VaREngine):
    def calculate_var(self, portfolio, data) -> VaRResult:
        # Implementation
        pass
```

2. **Validate inputs**:

```python
# Good: Validate before calculation
def calculate_var(self, portfolio, data):
    if not portfolio.positions:
        raise ValidationError("Portfolio has no positions")
    # ... proceed with calculation
```

3. **Return complete VaRResult**:

```python
# Good: Return all configured attribution
result = VaRResult(
    var=var_value,
    cvar=cvar_value,
    # ... include all configured attribution
)
```

4. **Use type hints**:

```python
# Good: Full type hints
from typing import Dict, Optional, Union
import pandas as pd

def calculate_var(
    self,
    portfolio: PortfolioBase,
    historical_data: Union[pd.DataFrame, MarketDataSet],
) -> VaRResult:
    pass
```

5. **Document complex logic**:

```python
# Good: Document why
def calculate_component_var(self):
    """
    Calculate Component VaR using Euler decomposition.

    Euler's homogeneous property ensures Component VaRs sum to total VaR:
    VaR(Σx_i) = Σ VaR_i(x_i)

    This is required for Basel compliance.
    """
    # Implementation
```

### ❌ DON'T: Avoid These Anti-Patterns

1. **Don't modify VaRConfig in place**:

```python
# Bad: Modifying config
def calculate_var(self, portfolio, data):
    self.config.calculate_component_var = True  # DON'T DO THIS

# Good: Create a copy or validate
config = self.config
if not config.calculate_component_var:
    return result_without_attribution
```

2. **Don't assume portfolio type**:

```python
# Bad: Assume EquityPortfolio
def calculate_var(self, portfolio, data):
    assert isinstance(portfolio, EquityPortfolio)  # Too restrictive

# Good: Support multiple types
def supports_portfolio(self, portfolio):
    return isinstance(portfolio, (EquityPortfolio, FIPortfolio))
```

3. **Don't ignore attribution configuration**:

```python
# Bad: Always calculate attribution
def calculate_var(self, portfolio, data):
    component_var = self._calculate_component_var(portfolio)  # Always calculate

# Good: Respect configuration
if self.config.calculate_component_var:
    component_var = self._calculate_component_var(portfolio)
else:
    component_var = None
```

4. **Don't use loops where vectorization is possible**:

```python
# Bad: Python loop
pnl = {}
for idx, scenario in scenarios.iterrows():
    pnl[idx] = sum(calculate_position_pnl(pos, scenario) for pos in portfolio.positions.values())

# Good: Vectorized
pnl = portfolio.revalue_portfolio_vectorized(scenarios)
```

5. **Don't forget error handling**:

```python
# Bad: No error handling
def calculate_var(self, portfolio, data):
    var = expensive_calculation(portfolio, data)
    return VaRResult(var=var, ...)

# Good: Comprehensive error handling
def calculate_var(self, portfolio, data):
    try:
        var = expensive_calculation(portfolio, data)
    except NumericalError as e:
        raise NumericalError(f"VaR calculation failed: {e}") from e
    return VaRResult(var=var, ...)
```

## Quick Reference

### Engine Selection Guide

| Portfolio Type       | Recommended Engine | Configuration                                    |
|---------------------|-------------------|--------------------------------------------------|
| Large equity (1000+) | Parametric        | `equity_factors=EquityRiskFactorConfig()`       |
| Options/derivatives  | Historical        | `calculate_component_var=True`                   |
| Fixed income (bonds) | Parametric        | `fi_factors=FIRiskFactorConfig()`                |
| Path-dependent       | Monte Carlo       | `mc_num_simulations=50000`                       |
| Real-time monitoring | Parametric        | `calculate_component_var=True`                   |

### Confidence Level Selection

| Use Case              | Confidence Level | Example                          |
|----------------------|------------------|----------------------------------|
| Basel regulatory     | 0.99 (99%)       | Bank capital adequacy            |
| Internal risk mgmt   | 0.95 (95%)       | Daily risk limits                |
| Stress testing       | 0.90 (90%)       | Crisis scenario analysis         |

### Attribution Use Cases

| Attribution Type   | When to Use                            | Cost    |
|-------------------|----------------------------------------|---------|
| Component VaR     | Basel capital allocation               | Low     |
| Marginal VaR      | Marginal capital allocation            | Low     |
| Factor VaR        | Factor hedging decisions               | Medium  |
| Incremental VaR   | Position-level risk decisions          | High    |

## Testing Strategy

### Test Categories

1. **Unit Tests** (test individual components)
   - Configuration validation
   - Risk factor calculations
   - Attribution methods

2. **Integration Tests** (test full workflows)
   - End-to-end VaR calculation
   - Multiple engine comparison
   - Attribution validation

3. **Performance Tests** (test scalability)
   - Large portfolio performance
   - Memory usage
   - Calculation time

4. **Accuracy Tests** (test correctness)
   - Known portfolio results
   - Attribution sum validation
   - Backtesting accuracy

### Example Test Structure

```python
import pytest
from var import VaRConfig, VaRMethod
from var.engines import HistoricalVaREngine

class TestHistoricalVaREngine:
    """Test suite for HistoricalVaREngine."""

    def test_basic_var_calculation(self):
        """Test basic VaR calculation."""
        # Arrange
        portfolio = create_test_portfolio()
        data = create_test_market_data()
        config = VaRConfig(confidence_level=0.99)

        # Act
        engine = HistoricalVaREngine(config=config)
        result = engine.calculate_var(portfolio, data)

        # Assert
        assert result.var >= 0
        assert result.cvar >= result.var
        assert result.confidence_level == 0.99
        assert result.method == VaRMethod.HISTORICAL

    def test_component_var_attribution(self):
        """Test Component VaR attribution."""
        # Arrange
        portfolio = create_diversified_portfolio()
        data = create_test_market_data()
        config = VaRConfig(
            confidence_level=0.99,
            calculate_component_var=True
        )

        # Act
        engine = HistoricalVaREngine(config=config)
        result = engine.calculate_var(portfolio, data)

        # Assert
        assert result.component_var is not None
        total_component_var = sum(result.component_var.values())
        assert abs(total_component_var - result.var) < 0.01

    def test_incremental_var_calculation(self):
        """Test Incremental VaR calculation."""
        # Arrange
        portfolio = create_test_portfolio()
        data = create_test_market_data()

        # Act
        engine = HistoricalVaREngine()
        ivar_result = engine.calculate_incremental_var(portfolio, data)

        # Assert
        assert ivar_result.portfolio_var >= 0
        assert len(ivar_result.position_ivari) == len(portfolio.positions)
        assert ivar_result.diversification_benefit >= 0
```

## Integration with Other Modules

### Portfolio Module

```python
from portfolio.equity.portfolio import EquityPortfolio
from portfolio.fi.portfolio import FIPortfolio

# VaR engines support these portfolio types
portfolio = EquityPortfolio(positions={...})
result = engine.calculate_var(portfolio, market_data)
```

### PriceEnv Module

```python
from priceenv import PricingEnvironment

# Historical VaR uses PricingEnvironment for revaluation
env = PricingEnvironment(
    spot=spot_data,
    vol=vol_surface,
    rate=rate_curve
)
result = engine.calculate_var(portfolio, env)
```

### Market Data Module

```python
from util.marketdata.models import MarketDataSet

# All engines support MarketDataSet
market_data = MarketDataSet(
    spot_data=spot_history,
    vol_data=vol_history,
    rate_data=rate_history
)
result = engine.calculate_var(portfolio, market_data)
```

## Working with Results

### VaRResult Analysis

```python
result = engine.calculate_var(portfolio, data)

# Basic metrics
print(f"VaR: ${result.var:,.2f}")
print(f"CVaR: ${result.cvar:,.2f}")
print(f"VaR as % of portfolio: {result.var_as_pct * 100:.2f}%")

# Attribution
if result.component_var:
    sorted_contrib = sorted(
        result.component_var.items(),
        key=lambda x: x[1],
        reverse=True
    )
    print("\nTop 5 Contributors:")
    for pos_id, comp_var in sorted_contrib[:5]:
        print(f"  {pos_id}: ${comp_var:,.2f}")

# Stressed VaR
if result.stressed_var:
    print(f"\nStressed VaR: ${result.stressed_var:,.2f}")
    print(f"SVaR/VaR Ratio: {result.stressed_var / result.var:.2f}x")

# Factor breakdown
if result.factor_var:
    print("\nFactor VaR:")
    for factor, factor_var in result.factor_var.items():
        print(f"  {factor}: ${factor_var:,.2f}")
```

### IncrementalVaRResult Analysis

```python
ivar_result = engine.calculate_incremental_var(portfolio, data)

# Diversification analysis
print(f"Portfolio VaR: ${ivar_result.portfolio_var:,.2f}")
print(f"Diversification Benefit: ${ivar_result.diversification_benefit:,.2f}")
print(f"Diversification Ratio: {ivar_result.get_diversification_ratio():.3f}")

# Position-level IVaR
top_contributors = ivar_result.get_top_contributors(10)
print("\nTop 10 IVaR Positions:")
for pos_id, ivar in top_contributors:
    print(f"  {pos_id}: ${ivar:,.2f}")
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now VaR engine will log detailed information
result = engine.calculate_var(portfolio, data)
```

### Check Calculation Time

```python
import time

start_time = time.time()
result = engine.calculate_var(portfolio, data)
elapsed = time.time() - start_time

print(f"Calculation time: {elapsed:.3f}s")
print(f"Expected time for this portfolio: ~{len(portfolio.positions) * 0.001:.3f}s")

if elapsed > expected_time * 2:
    print("WARNING: Calculation slower than expected")
```

### Validate Attribution

```python
def validate_attribution(result):
    """Validate VaR attribution results."""
    errors = []

    # Check Component VaR sum
    if result.component_var:
        total_component_var = sum(result.component_var.values())
        if abs(total_component_var - result.var) > 0.01:
            errors.append(
                f"Component VaR sum mismatch: {total_component_var} != {result.var}"
            )

    # Check Marginal VaR bounds
    if result.marginal_var:
        for pos_id, marg_var in result.marginal_var.items():
            if abs(marg_var) > result.var * 2:
                errors.append(
                    f"Marginal VaR out of bounds for {pos_id}: {marg_var}"
                )

    return errors

# Validate results
errors = validate_attribution(result)
if errors:
    print("VALIDATION ERRORS:")
    for error in errors:
        print(f"  - {error}")
```

## Common Issues and Solutions

### Issue: "Insufficient Historical Data"

**Error**:
```
MarketDataError: Need at least 252 days of data, got 100
```

**Solution**:
```python
# Reduce lookback_days
config = VaRConfig(lookback_days=100)

# Or provide more data
market_data = get_more_data(market_data)
```

### Issue: "Covariance Matrix is Singular"

**Error**:
```
NumericalError: covariance matrix is singular
```

**Solution**:
```python
# Add regularization to covariance matrix
cov_matrix += np.eye(len(cov_matrix)) * 1e-8

# Or remove redundant risk factors
risk_factors = remove_perfectly_correlated_factors(risk_factors)
```

### Issue: "Component VaR Doesn't Sum to Total VaR"

**Error**:
```
AssertionError: Component VaR sum 100.0 != Total VaR 95.0
```

**Solution**:
```python
# Check marginal VaR calculation
marginal_vars = calculate_marginal_vars(sensitivities, cov_matrix)
component_vars = {
    pos_id: marg_var * position_value
    for pos_id, marg_var in marginal_vars.items()
}

# Ensure Euler decomposition is correct
total_component_var = sum(component_vars.values())
assert abs(total_component_var - total_var) < tolerance
```

### Issue: "Slow Calculation for Large Portfolio"

**Problem**: 10,000 position portfolio takes > 30 seconds

**Solutions**:
1. **Use Parametric VaR** instead of Historical
2. **Disable Incremental VaR** (calculates n+1 VaRs)
3. **Reduce attribution flags** to essentials only
4. **Use MarketDataSet** for better performance
5. **Vectorize portfolio revaluation**

### Issue: "Monte Carlo Results Not Reproducible"

**Problem**: Same configuration gives different results

**Solution**:
```python
# Set random seed
config = VaRConfig(
    var_method=VaRMethod.MONTE_CARLO,
    mc_num_simulations=10000,
    mc_seed=42  # Set seed for reproducibility
)
```

## Performance Monitoring

### Benchmark Function

```python
def benchmark_var_engines(portfolio, market_data):
    """Benchmark all VaR engines."""
    methods = [
        (VaRMethod.PARAMETRIC, ParametricVaREngine),
        (VaRMethod.HISTORICAL, HistoricalVaREngine),
        (VaRMethod.MONTE_CARLO, MonteCarloVaREngine)
    ]

    results = {}
    for method, engine_class in methods:
        config = VaRConfig(
            confidence_level=0.99,
            var_method=method,
            mc_num_simulations=5000 if method == VaRMethod.MONTE_CARLO else None
        )

        engine = engine_class(config=config)

        start = time.time()
        result = engine.calculate_var(portfolio, market_data)
        elapsed = time.time() - start

        results[method.name] = {
            'var': result.var,
            'time': elapsed,
            'engine': engine_class.__name__
        }

    return results

# Run benchmark
benchmark_results = benchmark_var_engines(portfolio, data)
for method, metrics in benchmark_results.items():
    print(f"{method:15} VaR: ${metrics['var']:,.2f} "
          f"({metrics['time']:.3f}s)")
```

### Performance Regression Testing

```python
# test_performance.py
import time
import pytest

@pytest.mark.performance
def test_large_portfolio_performance():
    """Ensure VaR calculation completes within time limit."""
    portfolio = create_large_portfolio(10000)
    data = create_market_data()

    engine = ParametricVaREngine()
    start = time.time()
    result = engine.calculate_var(portfolio, data)
    elapsed = time.time() - start

    # Must complete within 5 seconds
    assert elapsed < 5.0, f"Calculation took {elapsed:.3f}s (>5s limit)"
```

## Documentation Guidelines

### For New Features

When adding new functionality, document:

1. **Purpose**: What problem does this solve?
2. **Usage**: How to use it (with examples)?
3. **Configuration**: What parameters control it?
4. **Performance**: What's the performance impact?
5. **Validation**: How is correctness verified?

### For Bug Fixes

Document:
1. **Root Cause**: What was the bug?
2. **Impact**: What was affected?
3. **Fix**: How was it corrected?
4. **Tests**: What tests prevent regression?

### Code Comments

Add comments for:

1. **Complex algorithms** (why, not just what)
2. **Non-obvious optimizations**
3. **Regulatory requirements** (Basel, etc.)
4. **Mathematical formulas** (with references)

```python
# Good: Explains why
# Apply Basel III capital requirement: multiply VaR by 3
# This accounts for model risk and estimation error
var_adjusted = var * 3.0

# Good: Explains optimization
# Vectorized calculation: 100x faster than loop
pnl_vectorized = np.dot(scenarios.values, sensitivities.values)

# Good: References regulatory requirements
# Required for Basel III - Stressed VaR calculation
# Uses 12-month rolling window with highest volatility
```

## Checklist for AI Agents

Before submitting code changes:

- [ ] All new engines follow `VaREngine` protocol
- [ ] All configuration fields validated in `_validate()`
- [ ] All attribution methods properly configured
- [ ] Results include all configured attribution
- [ ] Type hints added to all functions
- [ ] Unit tests added for new functionality
- [ ] Integration tests verify end-to-end workflows
- [ ] Performance impact assessed and documented
- [ ] Documentation updated (CLAUDE.md, AGENTS.md)
- [ ] Examples added to README.md if user-facing
- [ ] Backwards compatibility maintained
- [ ] Error handling comprehensive
- [ ] Code follows project style guidelines

## Summary

This guide provides AI agents with targeted guidance for working with the VaR module. Key takeaways:

1. **Always follow the VaREngine protocol** - Ensures consistency across engines
2. **Validate inputs and results** - Prevents silent errors
3. **Choose the right engine** - Parametric for speed, Historical for accuracy
4. **Enable only needed attribution** - Performance matters
5. **Test thoroughly** - VaR is critical for risk management
6. **Document complex logic** - Future maintainers will thank you

For detailed implementation guidance, see `var/CLAUDE.md`.
For user-facing documentation, see `var/README.md`.
