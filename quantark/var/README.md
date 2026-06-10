# QuantArk Value-at-Risk (VaR) Module

## Overview

The QuantArk VaR module provides comprehensive Value-at-Risk calculation capabilities for financial portfolios, supporting multiple methodologies, risk attribution, and regulatory compliance (Basel III/IV). It is designed for professional quantitative finance applications with a focus on accuracy, performance, and flexibility.

## Features

### Core VaR Engines
- **Parametric VaR**: Variance-covariance approach using portfolio sensitivities (fastest)
- **Historical VaR**: Full portfolio revaluation using historical scenarios (most accurate)
- **Monte Carlo VaR**: Simulated scenarios with stochastic processes (most flexible)

### Advanced Analytics
- **Component VaR**: Position-level risk contribution using Euler decomposition
- **Marginal VaR**: Marginal impact of each position on portfolio VaR
- **Incremental VaR**: VaR difference when excluding individual positions
- **Factor VaR**: Risk attribution by underlying risk factors

### Stressed VaR (SVaR)
- **Automatic crisis period detection** using rolling volatility
- **12-month stressed period** identification
- **Basel-compliant** Stressed VaR calculations
- Configurable stress window and methods

### Portfolio Support
- **Equity Portfolios**: Options, stocks, derivatives with Greeks
- **Fixed Income Portfolios**: Bonds, swaps with DV01 and curve risk
- **Mixed Portfolios**: Multi-asset class portfolios

### Data Sources
- **DataFrame**: Direct pandas DataFrame input
- **MarketDataSet**: Time series from QuantArk market data module
- **Flexible formatting**: Multiple input formats supported

## Quick Start

### Basic VaR Calculation

```python
from var import VaRConfig, HistoricalVaREngine

# Configure VaR calculation
config = VaRConfig(
    confidence_level=0.99,      # 99% confidence level
    holding_period=1,           # 1-day VaR
    lookback_days=252,          # 1 year of history
    var_method=VaRMethod.HISTORICAL
)

# Create engine
engine = HistoricalVaREngine(config=config)

# Calculate VaR
result = engine.calculate_var(portfolio, market_data)

print(f"VaR: ${result.var:,.2f}")
print(f"CVaR: ${result.cvar:,.2f}")
print(f"Method: {result.method}")
```

### VaR with Attribution

```python
from var import VaRConfig, ParametricVaREngine

config = VaRConfig(
    confidence_level=0.99,
    calculate_component_var=True,    # Component VaR
    calculate_marginal_var=True,     # Marginal VaR
    calculate_factor_var=True,       # Factor VaR
    calculate_incremental_var=True   # Incremental VaR
)

engine = ParametricVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)

# Position-level risk contributions
for pos_id, comp_var in result.component_var.items():
    print(f"{pos_id}: ${comp_var:,.2f} ({comp_var/result.var*100:.1f}%)")

# Marginal VaR (marginal impact)
for pos_id, marg_var in result.marginal_var.items():
    print(f"{pos_id} Marginal VaR: ${marg_var:,.2f}")
```

### Incremental VaR (Standalone)

```python
from var import VaRConfig, HistoricalVaREngine

config = VaRConfig(
    confidence_level=0.99,
    calculate_incremental_var=True
)

engine = HistoricalVaREngine(config=config)

# Detailed IVaR analysis
ivar_result = engine.calculate_incremental_var(portfolio, market_data)

print(f"Portfolio VaR: ${ivar_result.portfolio_var:,.2f}")
print(f"Diversification Benefit: ${ivar_result.diversification_benefit:,.2f}")
print(f"Diversification Ratio: {ivar_result.get_diversification_ratio():.3f}")

# Top 5 risk contributors
top_5 = ivar_result.get_top_contributors(5)
for pos_id, ivar in top_5:
    print(f"{pos_id}: ${ivar:,.2f}")
```

### Stressed VaR (SVaR)

```python
from var import VaRConfig, HistoricalVaREngine

config = VaRConfig(
    confidence_level=0.99,
    calculate_stressed_var=True,    # Enable Stressed VaR
    stressed_lookback_days=252      # 12-month stressed period
)

engine = HistoricalVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)

print(f"VaR: ${result.var:,.2f}")
print(f"SVaR: ${result.stressed_var:,.2f}")
print(f"Stressed Period: {result.stressed_period}")
print(f"SVaR / VaR Ratio: {result.stressed_var / result.var:.2f}x")
```

## API Reference

### VaRConfig

Configuration class for VaR calculations.

```python
@dataclass
class VaRConfig:
    # Core parameters
    confidence_level: float = 0.99        # VaR confidence level
    holding_period: int = 1               # Holding period in days
    lookback_days: int = 252              # Historical lookback period
    var_method: VaRMethod = VaRMethod.PARAMETRIC

    # Risk factor configuration
    equity_factors: EquityRiskFactorConfig = None
    fi_factors: FIRiskFactorConfig = None

    # Advanced features
    calculate_component_var: bool = True
    calculate_marginal_var: bool = True
    calculate_factor_var: bool = True
    calculate_incremental_var: bool = False
    calculate_stressed_var: bool = False

    # Monte Carlo specific
    mc_num_simulations: int = 10000
    mc_seed: Optional[int] = None

    # Scaling method for multi-day VaR
    scaling_method: str = "sqrt_t"  # or "overlapping"
```

### VaRResult

Result object containing VaR metrics and attribution.

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

    # Attribution (optional, based on configuration)
    component_var: Optional[Dict[str, float]] = None
    marginal_var: Optional[Dict[str, float]] = None
    factor_var: Optional[Dict[str, float]] = None
    incremental_var: Optional[Dict[str, float]] = None

    # Stressed VaR
    stressed_var: Optional[float] = None
    stressed_cvar: Optional[float] = None
    stressed_period: Optional[Dict[str, datetime]] = None

    # Metadata
    scenarios: Optional[pd.DataFrame] = None
    worst_scenarios: Optional[List[Dict]] = None
    calculation_timestamp: datetime
    execution_time_seconds: float
    config_summary: Dict[str, Any]
```

### VaREngines

All VaR engines implement the `VaREngine` protocol.

#### HistoricalVaREngine

Uses historical market scenarios for VaR calculation.

**Advantages**:
- Most accurate (uses actual historical data)
- Captures non-linear effects
- No model assumptions required

**Use Cases**:
- Equity portfolios with options
- Portfolios with complex payoffs
- When historical data quality is high

```python
from var import HistoricalVaREngine

engine = HistoricalVaREngine(config=config)
result = engine.calculate_var(portfolio, historical_data)

# Supports both DataFrame and MarketDataSet
result = engine.calculate_var(portfolio, df_data)
result = engine.calculate_var(portfolio, market_data_set)
```

#### ParametricVaREngine

Uses variance-covariance approach with portfolio sensitivities.

**Advantages**:
- Fastest calculation
- Closed-form Greeks support
- Scalable to large portfolios

**Use Cases**:
- Large equity portfolios (delta, gamma, vega)
- Fixed income portfolios (DV01, convexity)
- Real-time risk monitoring

```python
from var import ParametricVaREngine

engine = ParametricVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)

# Supports FI risk factors
config = VaRConfig(
    fi_factors=FIRiskFactorConfig(
        include_parallel_shift=True,
        include_key_rates=True,
        key_rate_tenors=[2.0, 5.0, 10.0, 30.0]
    )
)
```

#### MonteCarloVaREngine

Uses simulated scenarios for VaR calculation.

**Advantages**:
- Flexible scenario generation
- Can model complex dependencies
- Handles path-dependent products

**Use Cases**:
- Path-dependent derivatives
- When historical data is limited
- Stress testing scenarios

```python
from var import MonteCarloVaREngine

engine = MonteCarloVaREngine(config=config)
result = engine.calculate_var(portfolio, market_data)
```

### IncrementalVaRResult

Detailed Incremental VaR analysis.

```python
@dataclass
class IncrementalVaRResult:
    portfolio_var: float                 # Total portfolio VaR
    position_ivari: Dict[str, float]     # IVaR by position
    diversification_benefit: float       # Diversification benefit

    portfolio_var_without_position: Optional[Dict[str, float]] = None
    ivari_method: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    # Methods
    def get_diversification_ratio(self) -> float
    def get_top_contributors(self, n: int = 5) -> List[tuple[str, float]]
    def get_summary_dict(self) -> Dict[str, Any]
```

## Configuration

### Equity Risk Factors

Configure equity risk factors for Parametric VaR.

```python
from var.config import EquityRiskFactorConfig

config = EquityRiskFactorConfig(
    include_spot=True,        # Delta (spot sensitivity)
    include_vol=True,         # Vega (volatility sensitivity)
    include_rate=True,        # Rho (rate sensitivity)
    include_div_yield=False   # Dividend yield sensitivity
)
```

### Fixed Income Risk Factors

Configure FI risk factors for Parametric VaR.

```python
from var.config import FIRiskFactorConfig

config = FIRiskFactorConfig(
    include_parallel_shift=True,      # Parallel curve shifts
    include_key_rates=False,          # Key rate exposures
    key_rate_tenors=[2.0, 5.0, 10.0, 30.0]  # Key rate points
)
```

## Data Formats

### DataFrame Format

Historical VaR and Monte Carlo VaR support DataFrame input.

**Equity DataFrame**:
```python
df = pd.DataFrame({
    'spot_return': [0.01, -0.02, 0.03, ...],      # Spot returns (pct change)
    'vol_change': [0.001, -0.002, 0.001, ...],    # Vol changes (absolute)
    'rate_shift': [0.0001, -0.0002, 0.0001, ...], # Rate changes (absolute)
    'div_yield_shift': [0.0001, 0, -0.0001, ...]  # Div yield changes
})
```

**Fixed Income DataFrame**:
```python
df = pd.DataFrame({
    'parallel_shift': [0.001, -0.002, 0.001, ...],  # Parallel shifts
    # OR
    'rate': [0.05, 0.051, 0.049, ...],              # Rate levels

    # Optional key rates
    'rate_2y': [0.04, 0.041, 0.039, ...],
    'rate_5y': [0.05, 0.051, 0.049, ...],
    'rate_10y': [0.055, 0.056, 0.054, ...],
    'rate_30y': [0.06, 0.061, 0.059, ...]
})
```

### MarketDataSet Format

All engines support MarketDataSet input.

```python
# MarketDataSet with time series
market_data = MarketDataSet(
    spot_data=spot_history,      # Time series of spot prices
    vol_data=vol_history,        # Time series of volatilities
    rate_data=rate_history,      # Time series of rates
    div_yield_data=div_history   # Time series of dividend yields (optional)
)

# Calculate VaR with MarketDataSet
result = engine.calculate_var(portfolio, market_data)
```

## Risk Attribution

### Component VaR

Decomposes portfolio VaR into position-level contributions using Euler decomposition.

**Interpretation**:
- Sum of Component VaRs equals total portfolio VaR
- Shows each position's risk contribution
- Useful for capital allocation

```python
result = engine.calculate_var(portfolio, data)
for pos_id, comp_var in result.component_var.items():
    pct = comp_var / result.var * 100
    print(f"{pos_id}: ${comp_var:,.2f} ({pct:.1f}%)")
```

### Marginal VaR

Measures the marginal impact of each position on portfolio VaR.

**Interpretation**:
- How much VaR changes when position size changes
- Different from Component VaR
- Used for marginal capital allocation

```python
result = engine.calculate_var(portfolio, data)
for pos_id, marg_var in result.marginal_var.items():
    print(f"{pos_id} Marginal VaR: ${marg_var:,.2f}")
```

### Incremental VaR

Measures VaR difference when position is excluded.

**Formula**:
```
IVaR_i = VaR(full portfolio) - VaR(portfolio without position i)
```

**Interpretation**:
- How much VaR would change if position was removed
- Useful for position-level risk management
- Identifies positions to hedge/remove

```python
# Integrated with VaRResult
result = engine.calculate_var(portfolio, data)
if result.incremental_var:
    for pos_id, ivar in result.incremental_var.items():
        print(f"{pos_id} IVaR: ${ivar:,.2f}")

# Standalone with IncrementalVaRResult
ivar_result = engine.calculate_incremental_var(portfolio, data)
top_contributors = ivar_result.get_top_contributors(10)
```

### Factor VaR

Decomposes VaR by underlying risk factors (spot, vol, rate, etc.).

**Interpretation**:
- Which risk factors drive portfolio risk
- Useful for factor hedging
- Regulatory reporting (Basel)

```python
result = engine.calculate_var(portfolio, data)
for factor, factor_var in result.factor_var.items():
    pct = factor_var / result.var * 100
    print(f"{factor}: ${factor_var:,.2f} ({pct:.1f}%)")
```

## Stressed VaR (SVaR)

Basel requirement for capital adequacy calculations.

### Automatic Detection

Automatically identifies the most volatile 12-month period.

```python
config = VaRConfig(
    calculate_stressed_var=True,
    stressed_lookback_days=252  # 12 months
)

result = engine.calculate_var(portfolio, data)
print(f"Stressed Period: {result.stressed_period}")
print(f"SVaR: ${result.stressed_var:,.2f}")
```

### Comparison with VaR

- **VaR**: Regular Value-at-Risk using entire history
- **SVaR**: Stressed VaR using crisis period
- SVaR typically higher than VaR
- Used for regulatory capital calculation

## Multi-Day VaR

Calculate VaR for holding periods > 1 day.

### Method 1: Square Root of Time

```python
config = VaRConfig(
    holding_period=10,         # 10-day VaR
    scaling_method="sqrt_t"    # Square root of time scaling
)

result = engine.calculate_var(portfolio, data)
# VaR scales with sqrt(10)
```

### Method 2: Overlapping Returns

```python
config = VaRConfig(
    holding_period=10,
    scaling_method="overlapping"  # Use overlapping scenarios
)

result = engine.calculate_var(portfolio, data)
# More accurate for short holding periods
```

## Validation and Error Handling

The VaR module includes comprehensive validation:

```python
try:
    result = engine.calculate_var(portfolio, data)
except ValidationError as e:
    print(f"Invalid input: {e}")
except MarketDataError as e:
    print(f"Market data issue: {e}")
except NumericalError as e:
    print(f"Numerical instability: {e}")
```

## Performance Considerations

### Engine Comparison

| Engine    | Speed   | Accuracy | Flexibility | Use Case |
|-----------|---------|----------|-------------|----------|
| Parametric| Fastest | Good     | Moderate    | Large portfolios, real-time |
| Historical| Medium  | Best     | Low         | Equity with options |
| Monte Carlo| Slowest | Good     | Highest     | Complex derivatives |

### Optimization Tips

1. **Use Parametric VaR** for large, linear portfolios
2. **Enable only needed attribution** (slower when enabled)
3. **Set appropriate lookback_days** (more data = slower)
4. **Use MarketDataSet** for better performance than DataFrame
5. **Monte Carlo**: Reduce `mc_num_simulations` for faster results

## Examples

### Equity Portfolio with Options

```python
from var import VaRConfig, HistoricalVaREngine
from var.config import EquityRiskFactorConfig

# Configure for equity options
config = VaRConfig(
    confidence_level=0.99,
    var_method=VaRMethod.HISTORICAL,
    equity_factors=EquityRiskFactorConfig(
        include_spot=True,
        include_vol=True,
        include_rate=True,
        include_div_yield=True
    ),
    calculate_component_var=True,
    calculate_marginal_var=True,
    calculate_incremental_var=True,
    calculate_stressed_var=True
)

engine = HistoricalVaREngine(config=config)

# Calculate with MarketDataSet
result = engine.calculate_var(equity_portfolio, market_data_set)

print(f"VaR: ${result.var:,.2f}")
print(f"CVaR: ${result.cvar:,.2f}")
print(f"SVaR: ${result.stressed_var:,.2f}")
print("\nTop 5 Contributors:")
for pos_id, comp_var in sorted(result.component_var.items(),
                               key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {pos_id}: ${comp_var:,.2f}")
```

### Fixed Income Portfolio

```python
from var import VaRConfig, ParametricVaREngine
from var.config import FIRiskFactorConfig

# Configure for fixed income
config = VaRConfig(
    confidence_level=0.99,
    var_method=VaRMethod.PARAMETRIC,
    fi_factors=FIRiskFactorConfig(
        include_parallel_shift=True,
        include_key_rates=True,
        key_rate_tenors=[2.0, 5.0, 10.0, 30.0]
    ),
    calculate_component_var=True,
    calculate_stressed_var=True
)

engine = ParametricVaREngine(config=config)

# Calculate with FI DataFrame
fi_data = pd.DataFrame({
    'parallel_shift': np.random.normal(0, 0.001, 300),
    'rate_2y': np.random.normal(0, 0.001, 300),
    'rate_5y': np.random.normal(0, 0.001, 300),
    'rate_10y': np.random.normal(0, 0.001, 300),
    'rate_30y': np.random.normal(0, 0.001, 300)
})

result = engine.calculate_var(fi_portfolio, fi_data)

print(f"VaR: ${result.var:,.2f}")
print(f"DV01 (approx): ${result.component_var['TOTAL']/0.0001:,.0f}")
```

### Portfolio Comparison

```python
from var import VaRConfig, HistoricalVaREngine, ParametricVaREngine, MonteCarloVaREngine

# Create historical data
dates = pd.date_range('2020-01-01', periods=500, freq='D')
data = pd.DataFrame({
    'spot_return': np.random.normal(0, 0.02, 500),
    'vol_change': np.random.normal(0, 0.01, 500),
    'rate_shift': np.random.normal(0, 0.001, 500)
}, index=dates)

# Run all three engines
methods = [
    (VaRMethod.PARAMETRIC, ParametricVaREngine),
    (VaRMethod.HISTORICAL, HistoricalVaREngine),
    (VaRMethod.MONTE_CARLO, MonteCarloVaREngine)
]

for method, engine_class in methods:
    config = VaRConfig(
        confidence_level=0.99,
        var_method=method,
        mc_num_simulations=1000 if method == VaRMethod.MONTE_CARLO else None
    )

    engine = engine_class(config=config)
    result = engine.calculate_var(portfolio, data)

    print(f"{method.name:15} VaR: ${result.var:,.2f} "
          f"({result.execution_time_seconds:.3f}s)")
```

## Testing

Run the comprehensive test suite:

```bash
# All VaR tests
python -m pytest test/test_var*.py -v

# Specific test files
python -m pytest test/test_var_attribution.py -v
python -m pytest test/test_stressed_var.py -v
python -m pytest test/test_incremental_var.py -v

# Run with coverage
python -m pytest test/test_var*.py --cov=var --cov-report=html
```

## Best Practices

1. **Use Historical VaR** for portfolios with options or non-linear payoffs
2. **Use Parametric VaR** for large, linear portfolios (equity, FI)
3. **Enable attribution** based on need (slower when enabled)
4. **Validate market data** before running VaR calculations
5. **Use appropriate confidence levels** (99% for regulatory, 95% for internal)
6. **Monitor stressed VaR** for regulatory capital
7. **Review attribution regularly** for risk management
8. **Use Incremental VaR** for position-level decisions

## References

- Basel Committee on Banking Supervision. "Basel III: The Liquidity Coverage Ratio and liquidity risk monitoring tools."
- Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"
- McNeil, A., Frey, R., Embrechts, P. "Quantitative Risk Management"

## Support

For issues, questions, or contributions:
- GitHub Issues: [QuantArk Issues]
- Documentation: [QuantArk Docs]
- Email: quantark-support@example.com

---

**QuantArk VaR Module** - Professional-grade Value-at-Risk calculations for quantitative finance.
