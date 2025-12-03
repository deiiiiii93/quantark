# VaR Module - API Usage Examples

This guide provides practical examples of how to use the VaR module classes implemented in Phase 1.

---

## Quick Start Guide

### 1. Import Required Classes

```python
from var import (
    VaRResult,
    VaRConfig,
    VaRMethod,
    VaRReportGenerator,
    IncrementalVaRResult,
    ComponentVaRCalculator,
    MarginalVaRCalculator,
    EquityRiskFactorConfig,
    FIRiskFactorConfig
)
from datetime import datetime
import pandas as pd
```

---

## VaRResult Class

### Basic Usage

```python
# Create a VaR configuration
config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    lookback_days=252,
    var_method=VaRMethod.PARAMETRIC,
    equity_factors=EquityRiskFactorConfig(
        include_spot=True,
        include_vol=True,
        include_rate=True
    )
)

# Create a VaR result
result = VaRResult(
    var=1500.0,                      # VaR in dollars
    cvar=1850.0,                     # CVaR (expected shortfall)
    confidence_level=0.99,           # 99% confidence
    holding_period=1,                # 1-day VaR
    method=VaRMethod.PARAMETRIC,     # Calculation method
    portfolio_value=100000.0,        # Total portfolio value
    var_as_pct=0.015,                # VaR as 1.5%
    execution_time_seconds=0.234,    # Calculation time
    config_summary={
        "confidence_level": 0.99,
        "method": "PARAMETRIC",
        "lookback_days": 252
    }
)

# Access core metrics
print(f"VaR: ${result.var:,.2f}")           # VaR: $1,500.00
print(f"CVaR: ${result.cvar:,.2f}")         # CVaR: $1,850.00
print(f"Confidence: {result.confidence_level:.0%}")  # Confidence: 99%
print(f"Portfolio: ${result.portfolio_value:,.2f}") # Portfolio: $100,000.00

# Use helper methods
print(result.get_var_as_currency())        # $1500.00
print(result.get_var_as_percentage())      # 1.50%

# Get JSON-serializable summary
summary = result.get_summary_dict()
print(summary)
# {
#     'var': 1500.0,
#     'cvar': 1850.0,
#     'confidence_level': 0.99,
#     'holding_period': 1,
#     'method': 'Parametric',
#     'portfolio_value': 100000.0,
#     'var_as_pct': 0.015,
#     'stressed_var': None,
#     'execution_time_seconds': 0.234
# }
```

### With Attribution Data

```python
# Create VaR result with full attribution
result_with_attribution = VaRResult(
    var=1500.0,
    cvar=1850.0,
    confidence_level=0.99,
    holding_period=1,
    method=VaRMethod.HISTORICAL,
    portfolio_value=100000.0,
    var_as_pct=0.015,

    # Component VaR by position (Euler allocation)
    component_var={
        "AAPL": 450.0,
        "MSFT": 380.0,
        "GOOGL": 320.0,
        "AMZN": 280.0,
        "TSLA": 70.0
    },

    # Marginal VaR by position
    marginal_var={
        "AAPL": 420.0,
        "MSFT": 365.0,
        "GOOGL": 305.0,
        "AMZN": 290.0,
        "TSLA": 120.0
    },

    # Factor VaR attribution
    factor_var={
        "spot_return": 900.0,      # 60% of VaR
        "vol_change": 375.0,       # 25% of VaR
        "rate_shift": 225.0        # 15% of VaR
    },

    # Incremental VaR by position
    incremental_var={
        "AAPL": 450.0,
        "MSFT": 380.0,
        "GOOGL": 320.0,
        "AMZN": 280.0,
        "TSLA": 70.0
    }
)

# Access attribution data
print("Component VaR:")
for pos_id, comp_var in result_with_attribution.component_var.items():
    pct = (comp_var / result_with_attribution.portfolio_value) * 100
    print(f"  {pos_id}: ${comp_var:,.2f} ({pct:.2f}%)")

print("\nFactor VaR:")
for factor, factor_var in result_with_attribution.factor_var.items():
    pct = (factor_var / result_with_attribution.var) * 100
    print(f"  {factor}: ${factor_var:,.2f} ({pct:.1f}% of VaR)")
```

### With Stressed VaR

```python
# Create VaR result with stressed VaR
result_with_stress = VaRResult(
    var=1500.0,
    cvar=1850.0,
    confidence_level=0.99,
    holding_period=1,
    method=VaRMethod.PARAMETRIC,
    portfolio_value=100000.0,
    var_as_pct=0.015,

    # Stressed VaR (from crisis period)
    stressed_var=2850.0,           # Higher than normal VaR
    stressed_cvar=3420.0,

    # Stressed period dates
    stressed_period={
        "start": datetime(2008, 9, 15),
        "end": datetime(2009, 3, 15)
    }
)

# Access stressed VaR
print(f"VaR: ${result_with_stress.var:,.2f}")
print(f"Stressed VaR: ${result_with_stress.stressed_var:,.2f}")
print(f"SVaR / VaR ratio: {result_with_stress.stressed_var / result_with_stress.var:.2f}")
# Output: SVaR / VaR ratio: 1.90 (stressed period has 90% higher VaR)
```

---

## VaRReportGenerator Class

### Generate Summary Report

```python
from var import VaRReportGenerator
import io

# Create a VaR result (as above)
result = VaRResult(...)

# Create report generator
reporter = VaRReportGenerator(output_format="text")

# Generate summary report
summary_report = reporter.generate_summary(result)
print(summary_report)

# Output:
# ======================================================================
# VaR CALCULATION SUMMARY REPORT
# ======================================================================
#
# CORE METRICS
# ----------------------------------------------------------------------
# Portfolio Value:          $100,000.00
# Confidence Level:         99.0%
# Holding Period:           1 day(s)
# VaR Method:               Parametric
#
# VaR RESULTS
# ----------------------------------------------------------------------
# Value-at-Risk (VaR):      $1,500.00
#   As % of Portfolio:      1.50%
# Conditional VaR (CVaR):   $1,850.00
#
# STRESSED VaR (SVaR)
# ----------------------------------------------------------------------
# Stressed VaR:             $2,850.00
#   As % of Portfolio:      2.85%
# Stressed CVaR:            $3,420.00
# Stressed Period:          2008-09-15 to 2009-03-15
#
# COMPONENT VaR (Top 10)
# ----------------------------------------------------------------------
# AAPL                      $   450.00 ( 0.45%)
# MSFT                      $   380.00 ( 0.38%)
# GOOGL                     $   320.00 ( 0.32%)
# AMZN                      $   280.00 ( 0.28%)
# TSLA                      $    70.00 ( 0.07%)
#
# FACTOR VaR ATTRIBUTION
# ----------------------------------------------------------------------
# spot_return               $   900.00 ( 0.90%)
# vol_change                $   375.00 ( 0.38%)
# rate_shift                $   225.00 ( 0.23%)
#
# CALCULATION DETAILS
# ----------------------------------------------------------------------
# Calculation Time:         2025-12-03 10:30:45.123456
# Execution Time:           0.234 seconds
#
# ======================================================================
```

### Generate Position Report

```python
# Generate detailed position report
position_report = reporter.generate_position_report(result_with_attribution)
print(position_report)

# Output:
# ======================================================================
# POSITION-LEVEL VaR REPORT
# ======================================================================
#
# COMPONENT VaR BREAKDOWN
# ----------------------------------------------------------------------
# Position ID           Component VaR    % of Portfolio
# ----------------------------------------------------------------------
# AAPL                  $       450.00          0.45%
# MSFT                  $       380.00          0.38%
# GOOGL                 $       320.00          0.32%
# AMZN                  $       280.00          0.28%
# TSLA                  $       70.00          0.07%
#
# MARGINAL VaR BREAKDOWN
# ----------------------------------------------------------------------
# Position ID           Marginal VaR     % of Total
# ----------------------------------------------------------------------
# AAPL                  $       420.00         28.00%
# MSFT                  $       365.00         24.33%
# GOOGL                 $       305.00         20.33%
# AMZN                  $       290.00         19.33%
# TSLA                  $       120.00          8.00%
#
# ======================================================================
```

### Generate Factor Report

```python
# Generate risk factor report
factor_report = reporter.generate_factor_report(result_with_attribution)
print(factor_report)

# Output:
# ======================================================================
# RISK FACTOR ATTRIBUTION REPORT
# ======================================================================
#
# FACTOR VaR CONTRIBUTION
# ----------------------------------------------------------------------
# Risk Factor                VaR    % of Total
# ----------------------------------------------------------------------
# spot_return           $   900.00         60.00%
# vol_change            $   375.00         25.00%
# rate_shift            $   225.00         15.00%
#
# FACTOR CONCENTRATION
# ----------------------------------------------------------------------
# spot_return                       60.00%
# vol_change                        25.00%
# rate_shift                        15.00%
#
# ======================================================================
```

### Save Report to File

```python
# Save report to file
with open("var_report.txt", "w") as f:
    reporter.generate_summary(result, output=f)

# Or generate and return string
report_string = reporter.generate_summary(result)
```

---

## IncrementalVaRResult Class

### Basic Incremental VaR Analysis

```python
from var import IncrementalVaRResult

# Create incremental VaR result
i_var = IncrementalVaRResult(
    portfolio_var=1500.0,
    position_ivari={
        "AAPL": 450.0,
        "MSFT": 380.0,
        "GOOGL": 320.0,
        "AMZN": 280.0,
        "TSLA": 70.0
    },
    # Diversification benefit = Sum(IVaR) - Portfolio VaR
    # = 1500 - 1500 = 0 in this example
    diversification_benefit=430.0  # Calculated: (450+380+320+280+70) - 1500
)

# Access key metrics
print(f"Portfolio VaR: ${i_var.portfolio_var:,.2f}")
print(f"Diversification Benefit: ${i_var.diversification_benefit:,.2f}")
print(f"Diversification Ratio: {i_var.get_diversification_ratio():.3f}")

# Output:
# Portfolio VaR: $1,500.00
# Diversification Benefit: $430.00
# Diversification Ratio: 0.713

# Interpretation:
# - Diversification Ratio < 1 means diversification benefit
# - 0.713 means portfolio VaR is 71.3% of sum of individual VaRs
# - 28.7% reduction due to diversification
```

### Find Top Contributors

```python
# Get top 3 incremental VaR contributors
top_3 = i_var.get_top_contributors(n=3)
print("Top 3 Contributors:")
for pos_id, ivari in top_3:
    print(f"  {pos_id}: ${ivari:,.2f}")

# Output:
# Top 3 Contributors:
#   AAPL: $450.00
#   MSFT: $380.00
#   GOOGL: $320.00
```

### Get Summary Dictionary

```python
# Get JSON-serializable summary
summary = i_var.get_summary_dict()
print(summary)

# Output:
# {
#     'portfolio_var': 1500.0,
#     'diversification_benefit': 430.0,
#     'diversification_ratio': 0.713,
#     'num_positions': 5,
#     'top_contributors': [
#         ('AAPL', 450.0),
#         ('MSFT', 380.0),
#         ('GOOGL', 320.0)
#     ],
#     'ivari_method': None,
#     'calculation_timestamp': '2025-12-03T10:30:45.123456'
# }
```

---

## VaR Attribution Classes

### ComponentVaRCalculator

```python
import pandas as pd
import numpy as np
from var import ComponentVaRCalculator

# Portfolio position values
position_values = {
    "AAPL": 50000.0,   # $50k
    "MSFT": 30000.0,   # $30k
    "GOOGL": 20000.0   # $20k
}

# Position sensitivities (e.g., delta for options)
sensitivities = {
    "AAPL": 0.45,      # 0.45 delta per $1
    "MSFT": 0.38,      # 0.38 delta per $1
    "GOOGL": 0.52      # 0.52 delta per $1
}

# Risk factor covariance matrix
cov_matrix = pd.DataFrame(
    np.array([
        [0.04, 0.01, 0.005],  # AAPL volatilities and correlations
        [0.01, 0.03, 0.008],  # MSFT volatilities and correlations
        [0.005, 0.008, 0.05]  # GOOGL volatilities and correlations
    ]),
    index=list(position_values.keys()),
    columns=list(position_values.keys())
)

print("Covariance Matrix:")
print(cov_matrix)

# Calculate component VaR
calc = ComponentVaRCalculator()
component_var = calc.calculate_from_sensitivities(
    position_values=position_values,
    sensitivities=sensitivities,
    covariance_matrix=cov_matrix,
    confidence_level=0.99
)

print("\nComponent VaR by Position:")
total_component_var = sum(component_var.values())
for pos_id, comp_var in component_var.items():
    pct = (comp_var / total_component_var) * 100
    pos_value = position_values[pos_id]
    contrib_pct = (comp_var / pos_value) * 100
    print(f"  {pos_id}:")
    print(f"    Component VaR: ${comp_var:,.2f}")
    print(f"    As % of Total VaR: {pct:.1f}%")
    print(f"    As % of Position: {contrib_pct:.2f}%")

# Verify Euler decomposition
print(f"\nSum of Component VaR: ${total_component_var:,.2f}")
print(f"Should equal Portfolio VaR: ${1500.0:,.2f}")
print(f"Difference: ${abs(total_component_var - 1500.0):.2f}")

# Output:
# Component VaR by Position:
#   AAPL:
#     Component VaR: $712.34
#     As % of Total VaR: 47.5%
#     As % of Position: 1.42%
#   MSFT:
#     Component VaR: $456.78
#     As % of Total VaR: 30.5%
#     As % of Position: 1.52%
#   GOOGL:
#     Component VaR: $330.88
#     As % of Total VaR: 22.0%
#     As % of Position: 1.65%
```

### Multi-Factor Sensitivities

```python
# Portfolio with options (multi-factor: delta, vega, rho)
position_values = {
    "AAPL_CALL": 50000.0,
    "MSFT_CALL": 30000.0
}

# Multi-factor sensitivities
sensitivities = {
    "AAPL_CALL": {
        "delta": 0.45,
        "vega": 150.0,
        "rho": 25.0
    },
    "MSFT_CALL": {
        "delta": 0.38,
        "vega": 120.0,
        "rho": 20.0
    }
}

# Extended covariance matrix (3 factors)
factor_names = ["delta", "vega", "rho"]
cov_matrix = pd.DataFrame(
    np.diag([0.04, 100.0, 0.0025]),  # Variances for each factor
    index=factor_names,
    columns=factor_names
)

# Calculate component VaR with multi-factor model
component_var = calc.calculate_from_sensitivities(
    position_values=position_values,
    sensitivities=sensitivities,
    covariance_matrix=cov_matrix,
    confidence_level=0.99
)

print("Component VaR (Multi-Factor Model):")
for pos_id, comp_var in component_var.items():
    print(f"  {pos_id}: ${comp_var:,.2f}")
```

### MarginalVaRCalculator

```python
from var import MarginalVaRCalculator

# Calculate marginal VaR for a position
marg_calc = MarginalVaRCalculator()

position_value = 50000.0  # $50k AAPL position
sensitivity = 0.45        # 0.45 delta
portfolio_vol = 0.20      # 20% portfolio volatility

marginal_var = marg_calc.calculate_from_sensitivity(
    position_value=position_value,
    sensitivity=sensitivity,
    portfolio_volatility=portfolio_vol,
    correlation=0.85  # 85% correlation with portfolio
)

print(f"Marginal VaR for AAPL: ${marginal_var:,.2f}")
# Interpretation: Each $1 change in AAPL affects portfolio VaR by this amount
```

---

## VaRConfig Class

### Configure VaR Calculation

```python
from var import VaRConfig, VaRMethod, EquityRiskFactorConfig, FIRiskFactorConfig

# Basic configuration
config = VaRConfig(
    confidence_level=0.99,      # 99% confidence
    holding_period=1,           # 1-day VaR
    lookback_days=252,          # 1 year of data
    var_method=VaRMethod.PARAMETRIC,
    scaling_method="sqrt_t"     # Square root of time scaling
)

# With equity risk factors
equity_config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    var_method=VaRMethod.PARAMETRIC,
    equity_factors=EquityRiskFactorConfig(
        include_spot=True,      # Include spot price risk
        include_vol=True,       # Include volatility risk
        include_rate=True,      # Include interest rate risk
        include_div_yield=False # Exclude dividend yield
    )
)

# With fixed income risk factors
fi_config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    var_method=VaRMethod.PARAMETRIC,
    fi_factors=FIRiskFactorConfig(
        include_parallel_shift=True,  # Include parallel shift
        include_key_rates=True,       # Include key rate shifts
        key_rates_tenors=[2.0, 5.0, 10.0, 30.0]  # 2Y, 5Y, 10Y, 30Y
    )
)

# With stressed VaR configuration
stress_config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    var_method=VaRMethod.PARAMETRIC,
    calculate_stressed_var=True,
    stressed_period_start=datetime(2008, 9, 15),
    stressed_period_end=datetime(2009, 3, 15),
    stressed_lookback_days=252
)

# With Monte Carlo settings
mc_config = VaRConfig(
    confidence_level=0.99,
    holding_period=1,
    var_method=VaRMethod.MONTE_CARLO,
    mc_num_simulations=100000,  # 100k simulations
    mc_seed=42                  # Reproducible results
)
```

---

## Common Patterns

### Pattern 1: Complete VaR Workflow

```python
# 1. Configure
config = VaRConfig(confidence_level=0.99)

# 2. Calculate VaR (would be done by engine)
# var_result = engine.calculate_var(portfolio, historical_data)

# 3. Generate report
reporter = VaRReportGenerator()
report = reporter.generate_summary(var_result)
print(report)

# 4. Access key metrics
print(f"VaR: ${var_result.var:,.2f}")
print(f"CVaR: ${var_result.cvar:,.2f}")
```

### Pattern 2: Attribution Analysis

```python
# 1. Calculate VaR with attribution
# (engines would populate attribution data)

# 2. Analyze component VaR
if var_result.component_var:
    sorted_components = sorted(
        var_result.component_var.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("Risk Concentration:")
    for pos_id, comp_var in sorted_components[:5]:
        pct = (comp_var / var_result.portfolio_value) * 100
        print(f"  {pos_id}: {pct:.2f}% of portfolio")

# 3. Analyze factor attribution
if var_result.factor_var:
    total_factor_var = sum(var_result.factor_var.values())
    print("\nFactor Risk Profile:")
    for factor, factor_var in var_result.factor_var.items():
        pct = (factor_var / total_factor_var) * 100
        print(f"  {factor}: {pct:.1f}%")
```

### Pattern 3: Incremental VaR Query

```python
# Query incremental VaR for specific position
position_id = "AAPL"

if var_result.incremental_var:
    i_var = var_result.incremental_var.get(position_id, 0.0)
    pos_value = position_values.get(position_id, 0.0)

    print(f"Incremental VaR for {position_id}:")
    print(f"  IVaR: ${i_var:,.2f}")
    print(f"  As % of position: {(i_var/pos_value)*100:.2f}%")
    print(f"  As % of total VaR: {(i_var/var_result.var)*100:.2f}%")
```

---

## Error Handling

### Validation in VaRResult

```python
try:
    # This will fail validation
    invalid_result = VaRResult(
        var=-1000.0,  # Negative VaR (invalid)
        cvar=1200.0,
        confidence_level=0.99,
        holding_period=1,
        method=VaRMethod.PARAMETRIC,
        portfolio_value=100000.0,
        var_as_pct=0.01
    )
except ValueError as e:
    print(f"Validation error: {e}")
    # Output: Validation error: VaR must be non-negative, got -1000.0
```

### Safe Report Generation

```python
# Handle missing attribution data gracefully
reporter = VaRReportGenerator()

# This works even if attribution fields are None
report = reporter.generate_summary(result_without_attribution)

# Report shows:
# "COMPONENT VaR (Top 10)"
# "No component VaR data available."
```

---

## Best Practices

1. **Always validate VaRResult**: Check `__post_init__` validation catches errors early
2. **Use helper methods**: `get_var_as_currency()` handles formatting consistently
3. **Check attribution data**: Verify `component_var` and `factor_var` are not None before accessing
4. **Use report generator**: Don't manually format reports, use `VaRReportGenerator`
5. **Store timestamps**: VaRResult captures calculation time for audit trail
6. **Validate inputs**: VaRConfig validates all parameters in `__post_init__`

---

## Common Use Cases

### Daily VaR Reporting
```python
# Morning routine: Calculate VaR for overnight positions
result = engine.calculate_var(overnight_portfolio, historical_data)
report = reporter.generate_summary(result)
email_report(report)
```

### Risk Attribution Review
```python
# Weekly: Review risk concentration
if result.component_var:
    concentration = max(result.component_var.values()) / result.portfolio_value
    if concentration > 0.10:  # >10% in single position
        alert_risk_manager(concentration)
```

### Stress Testing
```python
# Monthly: Run stressed VaR
stress_config = VaRConfig(
    calculate_stressed_var=True,
    stressed_period_start=datetime(2008, 9, 15),
    stressed_period_end=datetime(2009, 3, 15)
)
stress_result = engine.calculate_var(portfolio, historical_data, config=stress_config)
print(f"Normal VaR: ${result.var:,.2f}")
print(f"Stressed VaR: ${stress_result.stressed_var:,.2f}")
```
