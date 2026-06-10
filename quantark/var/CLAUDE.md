# VaR Module - Developer Guide

## Overview

The VaR (Value-at-Risk) module provides production-grade portfolio risk metrics with three engines, extensive attribution, and Basel III/IV compliance.

## Architecture

### Module Structure

```
var/
├── __init__.py               # Public API exports
├── base.py                   # VaREngine protocol
├── config.py                 # VaRConfig, risk factor configs
├── attribution.py            # ComponentVaR, MarginalVaR, VaRAttributor
├── engines/                  # VaR calculation engines
│   ├── parametric.py        # ParametricVaREngine
│   ├── historical.py        # HistoricalVaREngine
│   └── monte_carlo.py       # MonteCarloVaREngine
├── risk_factors/             # Risk factor modeling
│   ├── base.py              # RiskFactor base class
│   ├── equity_factors.py    # SpotReturnFactor, VolChangeFactor
│   └── fi_factors.py        # ParallelShiftFactor, KeyRateFactor
├── results/                  # Results and reporting
│   ├── var_result.py        # VaRResult, IncrementalVaRResult
│   └── var_report.py        # VaRReportGenerator
├── backtest/                 # Backtesting framework
│   └── var_backtester.py    # VaRBacktester, Kupiec test
└── doc/                      # Module documentation
```

### Exports

```python
from var import (
    VaRConfig,
    VaRMethod,
    EquityRiskFactorConfig,
    FIRiskFactorConfig,
    VaRResult,
    IncrementalVaRResult,
    ParametricVaREngine,
    HistoricalVaREngine,
    MonteCarloVaREngine,
    VaRBacktester,
    VaRBacktestResult,
    VaRReportGenerator,
    ComponentVaRCalculator,
    MarginalVaRCalculator,
    VaRAttributor,
)
```

## VaR Engines

### Engine Comparison

| Engine | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| **ParametricVaREngine** | ⚡⚡⚡ Fastest | Good (linear) | Large equity portfolios, real-time |
| **HistoricalVaREngine** | ⚡⚡ Medium | Excellent | Options, derivatives, non-linear payoffs |
| **MonteCarloVaREngine** | ⚡ Slowest | Good | Path-dependent, limited historical data |

### ParametricVaREngine
Uses variance-covariance approach with sensitivities (delta, vega, rho).
- Formula: `VaR = z_score × √(s^T × Σ × s)`
- Scalable to 100,000+ positions
- Closed-form Component VaR via Euler decomposition

### HistoricalVaREngine
Full portfolio revaluation under historical scenarios.
- Captures gamma, vega, and non-linear effects
- No distributional assumptions
- Supports DataFrame and MarketDataSet inputs

### MonteCarloVaREngine
Simulated scenarios using stochastic processes.
- Configurable simulations (default: 10,000)
- Random seed for reproducibility
- Flexible scenario generation

## Configuration

### VaRConfig
```python
@dataclass
class VaRConfig:
    confidence_level: float = 0.99         # 99% regulatory, 95% internal
    holding_period: int = 1                # Days (1, 5, 10)
    lookback_days: int = 252               # Historical data window
    var_method: VaRMethod = VaRMethod.PARAMETRIC

    # Risk factors
    equity_factors: Optional[EquityRiskFactorConfig] = None
    fi_factors: Optional[FIRiskFactorConfig] = None

    # Attribution flags (performance impact)
    calculate_component_var: bool = True   # ~10% overhead
    calculate_marginal_var: bool = True    # ~15% overhead
    calculate_factor_var: bool = True      # ~20% overhead
    calculate_incremental_var: bool = False # ~100% overhead (n+1 calcs)
    calculate_stressed_var: bool = False   # Basel compliance

    # Monte Carlo
    mc_num_simulations: int = 10000
    mc_seed: Optional[int] = None

    # Multi-day scaling
    scaling_method: str = "sqrt_t"         # or "overlapping"
```

### Risk Factor Configs
```python
EquityRiskFactorConfig(
    include_spot=True,
    include_vol=True,
    include_rate=True,
    include_div_yield=True
)

FIRiskFactorConfig(
    include_parallel_shift=True,
    include_key_rates=True,
    key_rate_tenors=[2.0, 5.0, 10.0, 30.0]
)
```

## Results

### VaRResult
```python
@dataclass
class VaRResult:
    var: float                           # Value-at-Risk
    cvar: float                          # Conditional VaR (Expected Shortfall)
    confidence_level: float
    holding_period: int
    method: VaRMethod
    portfolio_value: float
    var_as_pct: float

    # Attribution (optional)
    component_var: Optional[Dict[str, float]]
    marginal_var: Optional[Dict[str, float]]
    factor_var: Optional[Dict[str, float]]
    incremental_var: Optional[Dict[str, float]]

    # Stressed VaR (Basel)
    stressed_var: Optional[float]
    stressed_cvar: Optional[float]
    stressed_period: Optional[Dict[str, datetime]]
```

### IncrementalVaRResult
```python
@dataclass
class IncrementalVaRResult:
    portfolio_var: float
    incremental_vars: Dict[str, float]    # Position-level IVaR
    diversification_benefit: float
    top_contributors: List[Tuple[str, float]]
```

## Attribution Classes

### ComponentVaRCalculator
Position-level risk contribution using Euler decomposition.
- Formula: `Component VaR_i = ∂VaR/∂x_i × x_i`
- Component VaRs sum to total VaR

### MarginalVaRCalculator
Marginal change in VaR per unit position change.
- Formula: `Marginal VaR_i = ∂VaR/∂x_i`

### VaRAttributor
High-level wrapper combining Component and Marginal VaR calculations.

## Usage Examples

### Basic VaR Calculation

```python
from var import VaRConfig, HistoricalVaREngine

config = VaRConfig(confidence_level=0.99)
engine = HistoricalVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)

print(f"VaR: ${result.var:,.2f}")
print(f"CVaR: ${result.cvar:,.2f}")
```

### Full Attribution Analysis

```python
from var import VaRConfig, ParametricVaREngine

config = VaRConfig(
    confidence_level=0.99,
    calculate_component_var=True,
    calculate_marginal_var=True,
    calculate_factor_var=True,
)

engine = ParametricVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)

for pos_id, comp_var in result.component_var.items():
    pct = comp_var / result.var * 100
    print(f"{pos_id}: ${comp_var:,.2f} ({pct:.1f}%)")
```

### Incremental VaR

```python
config = VaRConfig(confidence_level=0.99, calculate_incremental_var=True)
engine = HistoricalVaREngine(config=config)
ivar_result = engine.calculate_incremental_var(portfolio, market_data)

print(f"Diversification Benefit: ${ivar_result.diversification_benefit:,.2f}")
for pos_id, ivar in ivar_result.get_top_contributors(5):
    print(f"{pos_id}: ${ivar:,.2f}")
```

### Stressed VaR (Basel)

```python
config = VaRConfig(
    confidence_level=0.99,
    calculate_stressed_var=True,
    stressed_lookback_days=252  # 12 months
)

engine = HistoricalVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)

print(f"VaR: ${result.var:,.2f}")
print(f"SVaR: ${result.stressed_var:,.2f}")
print(f"SVaR/VaR Ratio: {result.stressed_var / result.var:.2f}x")
```

### Multi-Day VaR

```python
# Square root of time scaling
config = VaRConfig(holding_period=10, scaling_method="sqrt_t")

# Overlapping returns (more accurate)
config = VaRConfig(holding_period=10, scaling_method="overlapping")
```

### VaR Backtesting

```python
from var import VaRBacktester

backtester = VaRBacktester(confidence_level=0.99)
result = backtester.run_kupiec_test(var_estimates, actual_losses)

print(f"Violations: {result.num_violations} / {result.total_obs}")
print(f"P-value: {result.p_value:.4f}")
print(f"Test: {'PASS' if result.passed else 'FAIL'}")
```

## Data Formats

### DataFrame Input
```python
# Equity
df = pd.DataFrame({
    'spot_return': [0.01, -0.02, ...],      # Percentage change
    'vol_change': [0.001, -0.002, ...],     # Absolute change
    'rate_shift': [0.0001, -0.0002, ...],   # Absolute change
})

# Fixed Income
df = pd.DataFrame({
    'parallel_shift': [0.001, -0.002, ...], # Curve shifts
    'rate_2y': [0.04, 0.041, ...],          # Key rates
    'rate_5y': [0.05, 0.051, ...],
    'rate_10y': [0.055, 0.056, ...],
})
```

### MarketDataSet Input
```python
from util.marketdata.models import MarketDataSet

market_data = MarketDataSet(
    spot_data=spot_history,
    vol_data=vol_history,
    rate_data=rate_history,
)
```

## Testing

```bash
# All VaR tests
python -m pytest test/test_var*.py -v

# Specific test
python -m pytest test/test_var_attribution.py::test_component_var_equity -v

# With coverage
python -m pytest test/test_var*.py --cov=var
```

## Performance Guidelines

### Engine Selection
- **Parametric**: 100x faster for 1,000+ positions, use for linear portfolios
- **Historical**: Use for options/derivatives with non-linear payoffs
- **Monte Carlo**: Use for path-dependent products or limited data

### Attribution Overhead
| Flag | Overhead |
|------|----------|
| Component VaR | +10% |
| Marginal VaR | +15% |
| Factor VaR | +20% |
| Incremental VaR | +100% |

Enable only what's needed.

### Lookback Period
- 252 days (1 year): Standard
- 504 days (2 years): More stable
- 126 days (6 months): More responsive

## Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "confidence_level must be between 0 and 1" | Using 99 instead of 0.99 | Use decimal (0.99) |
| "portfolio must have positions" | Empty portfolio | Add positions first |
| "insufficient historical data" | lookback > data length | Reduce lookback_days |
| "covariance matrix is singular" | Perfectly correlated factors | Remove redundant factors |

## Integration Points

- **Portfolio**: `portfolio.Portfolio` (equity) or `portfolio.fi.FIPortfolio`
- **Market Data**: `util.marketdata.models.MarketDataSet` or `pd.DataFrame`
- **Risk Measures**: Uses `asset.equity.riskmeasures.GreeksCalculator` for sensitivities

## Summary

- **Engines**: 3 (Parametric, Historical, Monte Carlo)
- **Attribution**: Component, Marginal, Factor, Incremental VaR
- **Backtesting**: Kupiec test for model validation
- **Compliance**: Basel III/IV Stressed VaR
- **Risk Factors**: Equity (spot, vol, rate, div) + FI (parallel, key rate)
