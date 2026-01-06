# SIMM Module - Developer Guide

## Overview

The SIMM (Standard Initial Margin Model) module implements ISDA SIMM v2.6 for initial margin calculations. It provides sensitivity calculation, aggregation, reporting, CRIF support, and what-if analysis.

### Risk Classes
1. Interest Rate (IR) - Rate curves and volatility
2. Credit Qualifying (CreditQ) - Investment grade credit
3. Credit Non-Qualifying (CreditNQ) - Non-qualifying credit
4. Equity - Spot and volatility risk
5. Commodity - Price and volatility risk
6. FX - Foreign exchange risk

### Product Classes
1. RatesFX - Interest rate and FX products
2. Credit - Credit products
3. Equity - Equity products
4. Commodity - Commodity products

## Architecture

```
┌─────────────────────────────────────────┐
│           SIMM High-Level API           │
│     (SIMMCalculator, Report Generators) │
├─────────────────────────────────────────┤
│  Aggregation Layer                      │
│  (Bucket, Risk Class, Product Class     │
│   Aggregators, Add-On Calculator)       │
├─────────────────────────────────────────┤
│  Sensitivity Engine Layer               │
│  (IR, Equity, Credit, FX, Commodity     │
│   Sensitivity Engines)                  │
├─────────────────────────────────────────┤
│  Data Model Layer                       │
│  (Sensitivities, Taxonomy, CRIF)        │
├─────────────────────────────────────────┤
│  Configuration Layer                    │
│  (SIMMConfig, Calibration Data)         │
└─────────────────────────────────────────┘
```

## Module Structure

```
simm/
├── __init__.py                 # Public API exports
├── config.py                   # SIMMConfig, SIMMVersion
├── taxonomy.py                 # Risk classes, buckets, tenors, enums
├── sensitivity.py              # Sensitivity dataclasses & protocols
├── calibration/                # Risk weights, correlations, HVR
│   ├── ir.py                  # IR risk weights & correlations
│   ├── equity.py              # Equity risk weights & correlations
│   ├── credit_qualifying.py   # Credit Q risk weights
│   ├── credit_non_qualifying.py
│   ├── commodity.py           # Commodity risk weights
│   ├── fx.py                  # FX risk weights
│   ├── cross_risk.py          # Cross-risk-class correlations
│   └── accessors.py           # Calibration data accessors
├── engines/                    # Sensitivity calculation engines
│   ├── base.py                # Base engine classes & protocols
│   ├── factory.py             # Engine factory & registry
│   ├── portfolio_adapter.py   # Portfolio-to-sensitivity adapter
│   ├── result.py              # SIMMResult dataclass
│   ├── classification/        # Bucket classification
│   │   └── bucket_mapper.py
│   ├── risk_class/            # Risk-class-specific engines
│   │   ├── ir_engine.py       # IR engine
│   │   └── equity_engine.py   # Equity engine
│   └── aggregation/           # Margin aggregation
│       ├── simm_calculator.py # Main SIMM calculator
│       ├── concentration.py   # Concentration risk
│       ├── bucket_aggregator.py
│       ├── risk_class_aggregator.py
│       ├── product_class_aggregator.py
│       └── addon.py           # Add-on calculator
├── crif/                      # CRIF format handling
│   ├── models.py             # CRIFRecord, CRIFHeader
│   └── parser.py             # CSV parsing & validation
├── results/                   # Results & attribution
│   ├── simm_result.py        # SIMMResult, RiskClassMargin
│   ├── attribution.py        # SIMMAttribution
│   └── whatif.py            # What-if analysis
└── report/                    # Report generation
    ├── html_generator.py     # HTML report (Plotly)
    ├── excel_generator.py    # Excel report
    └── crif_export.py        # CRIF export utilities
```

## Implementation Status

| Component | Delta | Vega | Curvature | Notes |
|-----------|-------|------|-----------|-------|
| IR Engine | ✅ | ❌ TODO (line 117) | ❌ TODO (line 141) | Delta complete with tenor interpolation |
| Equity Engine | ✅ | ✅ | ❌ | Bucket classification complete |
| Credit Engine | ❌ | ❌ | ❌ | Not implemented |
| FX Engine | ❌ | ❌ | ❌ | Not implemented |
| Commodity Engine | ❌ | ❌ | ❌ | Not implemented |
| Aggregation | ✅ | ✅ | ✅ | Complete for all margin types |
| CRIF Support | ✅ | ✅ | ✅ | Parse/export complete |
| Reporting | ✅ | ✅ | ✅ | HTML + Excel complete |
| What-If | ✅ | ✅ | ✅ | Add/remove/marginal analysis |

**Overall**: ~60% sensitivity engines, 100% aggregation, 100% reporting

## Core Components

### SIMMConfig
```python
@dataclass
class SIMMConfig:
    version: SIMMVersion = SIMMVersion.V2_6
    calculation_currency: str = "USD"
    calculate_delta: bool = True
    calculate_vega: bool = True
    calculate_curvature: bool = True
    calculate_base_corr: bool = True
    include_attribution: bool = True
    include_bucket_detail: bool = True
```

### SIMMCalculator
```python
calculator = SIMMCalculator(config)
result = calculator.calculate_full_simm(sensitivities)
result.total_margin
result.by_margin_type[MarginType.DELTA]
result.by_risk_class[RiskClass.EQUITY]
result.attribution.top_contributors(5)
```

### Sensitivity Classes
- `IRDeltaSensitivity` - DV01 by tenor/sub-curve
- `IRVegaSensitivity` - IR volatility sensitivity
- `EquityDeltaSensitivity` - Equity delta by bucket
- `EquityVegaSensitivity` - Equity vega by expiry
- `CreditDeltaSensitivity` - CS01 by tenor/issuer
- `FXDeltaSensitivity` - FX spot sensitivity
- `CommodityDeltaSensitivity` - Commodity delta
- `CurvatureSensitivity` - CVR calculation
- `BaseCorrSensitivity` - Base correlation (BC01)

### Taxonomy Enums
```python
class RiskClass(Enum):
    INTEREST_RATE = "IR"
    CREDIT_QUALIFYING = "CreditQ"
    CREDIT_NON_QUALIFYING = "CreditNQ"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    FX = "FX"

class MarginType(Enum):
    DELTA = "Delta"
    VEGA = "Vega"
    CURVATURE = "Curvature"
    BASE_CORR = "BaseCorr"

class ProductClass(Enum):
    RATES_FX = "RatesFX"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
```

## Usage Examples

### Portfolio to SIMM

```python
from simm import SIMMConfig, SIMMVersion
from simm.engines import SIMMPortfolioAdapter
from simm.engines.aggregation import SIMMCalculator

config = SIMMConfig(version=SIMMVersion.V2_6, calculation_currency="USD")

# Convert portfolio to sensitivities
adapter = SIMMPortfolioAdapter(config)
sensitivities = adapter.convert_portfolio(portfolio)

# Calculate SIMM margin
calculator = SIMMCalculator(config)
result = calculator.calculate_full_simm(sensitivities)

print(f"Total SIMM: {result.total_margin:,.2f}")
print(f"Delta: {result.by_margin_type[MarginType.DELTA]:,.2f}")
```

### CRIF Input

```python
from simm.crif import parse_crif_csv, crif_to_sensitivities

records, warnings = parse_crif_csv("sensitivities.crif")
sensitivities = crif_to_sensitivities(records)

result = calculator.calculate_full_simm(sensitivities)
```

### What-If Analysis

```python
from simm.results.whatif import SIMMWhatIf

whatif = SIMMWhatIf(base_sensitivities=sensitivities, config=config)

# Impact of adding positions
add_result = whatif.impact_of_adding(new_sensitivities)
print(f"Incremental margin: {add_result.incremental_margin:,.2f}")

# Marginal SIMM for single position
marginal = whatif.marginal_simm(new_position)
```

### Generate Reports

```python
from simm.report import SIMMHTMLReportGenerator, SIMMExcelReportGenerator

# HTML with charts
html_gen = SIMMHTMLReportGenerator(include_charts=True)
html_gen.generate(result, "simm_report.html")

# Excel with multiple sheets
excel_gen = SIMMExcelReportGenerator()
excel_gen.generate(result, "simm_report.xlsx", include_crifs=True)
```

## CRIF Format

CRIF (Common Risk Interchange Format) is the standard format for exchanging sensitivities:

```python
# Parse CRIF CSV
records, warnings = parse_crif_csv("file.crif")

# Convert to sensitivities
sensitivities = crif_to_sensitivities(records)

# Export back to CRIF
crif_records = sensitivities_to_crif(sensitivities)
write_crif_csv(crif_records, "output.crif")
```

## Testing

```bash
# All SIMM tests
python -m pytest test/test_simm_*.py -v

# Specific component
python -m pytest test/test_simm_ir_sensitivity.py -v
python -m pytest test/test_simm_aggregation.py -v

# With coverage
python -m pytest test/test_simm_*.py --cov=simm
```

## Key TODOs

### IR Engine (`engines/risk_class/ir_engine.py`)
- Line 117: `calculate_vega_sensitivities()` - Swaption vol sensitivity
- Line 141: `calculate_curvature_sensitivities()` - CVR with SF(t) scaling

### Missing Engines (need to create)
- `credit_engine.py` - CS01, base correlation
- `fx_engine.py` - FX delta, translation risk
- `commodity_engine.py` - Commodity delta by bucket

## Integration Points

- **Portfolio**: `portfolio.Portfolio` or `portfolio.fi.FIPortfolio`
- **Pricing**: Uses QuantArk pricing engines for Greeks/DV01
- **Greeks**: `asset.equity.riskmeasures.GreeksCalculator`
- **Bond Risk**: `asset.bond.riskmeasures.BondGreeksCalculator`

## Summary

- **SIMM Version**: v2.6
- **Risk Classes**: 6 (IR, CreditQ, CreditNQ, Equity, Commodity, FX)
- **Product Classes**: 4 (RatesFX, Credit, Equity, Commodity)
- **Margin Types**: 4 (Delta, Vega, Curvature, BaseCorr)
- **Complete**: Aggregation, CRIF, Reporting, What-If
- **Partial**: Sensitivity engines (IR delta, Equity delta/vega complete)
