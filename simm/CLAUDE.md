# SIMM Module - Developer & AI Agent Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Structure](#module-structure)
4. [Current Implementation Status](#current-implementation-status)
5. [Key TODOs & Incomplete Features](#key-todos--incomplete-features)
6. [Development Guidelines](#development-guidelines)
7. [Testing](#testing)
8. [Integration Patterns](#integration-patterns)
9. [API Reference](#api-reference)
10. [Performance Considerations](#performance-considerations)

---

## Overview

The SIMM (Standard Initial Margin Model) module implements ISDA SIMM v2.6 for initial margin calculations. It's a comprehensive financial risk management library that provides:

- **Sensitivity Calculation**: Calculate delta, vega, and curvature sensitivities for positions
- **Aggregation**: Apply risk weights, correlations, and concentration adjustments
- **Reporting**: Generate HTML and Excel reports with detailed margin breakdowns
- **CRIF Support**: Import/export sensitivities via Common Risk Interchange Format
- **What-If Analysis**: Analyze the impact of adding/removing positions

### Risk Classes Supported
1. **Interest Rate (IR)** - Interest rate curves and volatility
2. **Credit Qualifying (CreditQ)** - Investment grade and high yield credit
3. **Credit Non-Qualifying (CreditNQ)** - Non-qualifying credit exposures
4. **Equity** - Equity spot and volatility risk
5. **Commodity** - Commodity price and volatility risk
6. **FX** - Foreign exchange spot and volatility risk

### Product Classes
1. **RatesFX** - Interest rate and FX products
2. **Credit** - Credit products
3. **Equity** - Equity products
4. **Commodity** - Commodity products

---

## Architecture

### Core Design Pattern

The SIMM module follows a **layered, modular architecture** with clear separation of concerns:

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

### Component Interaction Flow

```python
# Typical calculation flow
positions → Sensitivity Engines → Weighted Sensitivities
                                     ↓
Portfolio/CIF ← Aggregation Layer ← Bucket Aggregation
    ↓
Risk Class Aggregation
    ↓
Product Class Aggregation (SIMM Final)
    ↓
Add-On Calculation
    ↓
SIMM Result + Attribution
```

---

## Module Structure

```
simm/
├── __init__.py                 # Public API exports
├── config.py                   # SIMMConfig, SIMMVersion
├── taxonomy.py                 # Risk classes, buckets, tenors, enums
├── sensitivity.py              # Sensitivity dataclasses & protocols
│
├── calibration/                # Risk weights, correlations, HVR
│   ├── __init__.py
│   ├── version.py             # SIMM version constants
│   ├── ir.py                  # IR risk weights & correlations
│   ├── equity.py              # Equity risk weights & correlations
│   ├── credit_qualifying.py   # Credit Q risk weights
│   ├── credit_non_qualifying.py
│   ├── commodity.py           # Commodity risk weights
│   ├── fx.py                  # FX risk weights
│   ├── cross_risk.py          # Cross-risk-class correlations
│   └── accessors.py           # Calibration data accessors
│
├── engines/                    # Sensitivity calculation engines
│   ├── __init__.py
│   ├── base.py                # Base engine classes & protocols
│   ├── factory.py             # Engine factory & registry
│   ├── portfolio_adapter.py   # Portfolio-to-sensitivity adapter
│   ├── result.py              # SIMMResult dataclass
│   ├── classification/        # Bucket classification
│   │   └── bucket_mapper.py
│   ├── risk_class/            # Risk-class-specific engines
│   │   ├── __init__.py
│   │   ├── ir_engine.py       # IR: delta ✓, vega ✗, curvature ✗
│   │   └── equity_engine.py   # Equity: delta ✓, vega ✓, curvature ✗
│   └── aggregation/           # Margin aggregation engines
│       ├── __init__.py
│       ├── simm_calculator.py # Main SIMM calculator
│       ├── concentration.py   # Concentration risk calculator
│       ├── weighted_sensitivity.py
│       ├── bucket_aggregator.py
│       ├── risk_class_aggregator.py
│       ├── product_class_aggregator.py
│       └── addon.py           # Add-on calculator
│
├── crif/                      # CRIF format handling
│   ├── __init__.py
│   ├── models.py             # CRIFRecord, CRIFHeader
│   └── parser.py             # CSV parsing & validation
│
├── results/                   # Results & attribution
│   ├── __init__.py
│   ├── simm_result.py        # SIMMResult, RiskClassMargin
│   ├── attribution.py        # SIMMAttribution, PositionAttribution
│   └── whatif.py            # What-if analysis
│
└── report/                    # Report generation
    ├── __init__.py
    ├── html_generator.py     # HTML report generator (Plotly)
    ├── excel_generator.py    # Excel report generator
    ├── crif_export.py        # CRIF export utilities
    └── templates/            # HTML templates
```

---

## Current Implementation Status

### ✅ Completed Features

#### 1. Foundation Layer (100%)
- [x] Risk class taxonomy (6 risk classes)
- [x] Product class taxonomy (4 product classes)
- [x] Margin type taxonomy (Delta, Vega, Curvature, BaseCorr)
- [x] Sensitivity data models with protocols
- [x] CRIF v2.x parsing and export
- [x] SIMM configuration (v2.5, v2.6)
- [x] Currency volatility classification

#### 2. Calibration Layer (100%)
- [x] IR risk weights and correlations
- [x] Equity risk weights and correlations
- [x] Credit Q/NQ risk weights
- [x] Commodity risk weights
- [x] FX risk weights
- [x] Cross-risk-class correlations
- [x] Data accessors with SIMM version support

#### 3. Sensitivity Engines (Partial - 60%)
- [x] Base engine architecture and protocols
- [x] Portfolio adapter (equity + FI positions)
- [x] **IR Engine**: Delta calculation (DV01 bucketing)
- [x] **IR Engine**: Sub-curve classification (OIS, LIBOR, etc.)
- [x] **IR Engine**: Tenor interpolation
- [x] **Equity Engine**: Delta calculation (via GreeksCalculator)
- [x] **Equity Engine**: Vega calculation
- [x] **Equity Engine**: Bucket classification (size, region, sector)
- [x] Bucket classification framework
- [ ] **IR Engine**: Vega calculation (swaption vol sensitivity)
- [ ] **IR Engine**: Curvature calculation
- [ ] **Credit Engine**: Not implemented
- [ ] **FX Engine**: Not implemented
- [ ] **Commodity Engine**: Not implemented
- [ ] Cross-risk vega engine
- [ ] Cross-risk curvature engine

#### 4. Aggregation Layer (100%)
- [x] Concentration risk calculator (all margin types)
- [x] Weighted sensitivity calculator
- [x] Bucket aggregator (all risk classes)
- [x] Risk class aggregator (Delta, Vega, Curvature, BaseCorr)
- [x] Product class aggregator (inter-risk correlations)
- [x] Add-on calculator (fixed, factor, multiplicative scale)
- [x] Main SIMM calculator orchestration
- [x] CRIF input mode
- [x] Portfolio input mode
- [x] Calculation tracing

#### 5. Results & Reporting (100%)
- [x] SIMMResult dataclass with full breakdown
- [x] Attribution framework (position-level)
- [x] What-if analysis (add/remove/marginal impact)
- [x] HTML report generator with Plotly charts
- [x] Excel report generator with multiple sheets
- [x] CRIF export utilities
- [x] Summary statistics and diversification metrics

#### 6. Testing (Partial - 70%)
- [x] `test_simm_taxonomy.py` - All taxonomy tests
- [x] `test_simm_config.py` - Configuration tests
- [x] `test_simm_crif.py` - CRIF parsing tests
- [x] `test_simm_calibration_*.py` - Calibration tests
- [x] `test_simm_aggregation.py` - Aggregation tests (50+ tests)
- [x] `test_simm_ir_sensitivity.py` - IR engine tests
- [x] `test_simm_equity_sensitivity.py` - Equity engine tests
- [x] `test_simm_engines_base.py` - Base engine tests
- [x] `test_simm_results.py` - Results & attribution tests
- [x] `test_simm_reports.py` - Report generation tests
- [x] `test_simm_whatif.py` - What-if analysis tests
- [ ] `test_simm_credit_sensitivity.py` - Not created (credit engine missing)
- [ ] `test_simm_fx_sensitivity.py` - Not created (FX engine missing)
- [ ] `test_simm_vega_curvature.py` - Not created (engines missing)

**Total Test Coverage**: ~70% of module, ~200+ tests passing

---

## Key TODOs & Incomplete Features

### High Priority

#### 1. Complete IR Sensitivity Engine
**File**: `simm/engines/risk_class/ir_engine.py`

```python
# Line 117 - TODO
def calculate_vega_sensitivities(self, ...):
    # TODO: Implement full IR vega calculation
    # - Swaption volatility sensitivity
    # - Cap/floor volatility sensitivity
    # - Tenor mapping for option expiries
    # - HVR (Historical Volatility Ratio) integration
    pass

# Line 141 - TODO
def calculate_curvature_sensitivities(self, ...):
    # TODO: Implement full IR curvature calculation
    # - CVR (Curvature Risk) calculation
    # - Scaling function SF(t) = 0.5 × min(1, 14/t)
    # - HVR^(-2) scaling for IR curvature
    pass

# Line 157 - TODO
def _determine_sub_curve(self, ...):
    # TODO: Implement proper sub-curve determination based on product
    # - Map products to appropriate sub-curves
    # - Handle cross-currency basis swaps
    pass
```

**Implementation Notes**:
- Requires integration with QuantArk's swaption/capfloor pricing engines
- Needs volatility surface data access
- Should use calibration data from `simm.calibration.ir`

#### 2. Implement Missing Sensitivity Engines
**Missing Engines** (as per `openspec/changes/add-simm-sensitivity-engines/tasks.md`):

##### A. Credit Sensitivity Engine
**File to create**: `simm/engines/risk_class/credit_engine.py`
- [ ] Credit Q delta calculation (CS01 by tenor)
- [ ] Issuer/seniority classification
- [ ] Bucket assignment (12 buckets + residual)
- [ ] Base correlation sensitivity (BC01)
- [ ] Credit NQ delta calculation
- [ ] Credit vega calculation
- [ ] Credit curvature calculation

##### B. FX Sensitivity Engine
**File to create**: `simm/engines/risk_class/fx_engine.py`
- [ ] FX delta calculation (1% spot shock)
- [ ] FX translation risk from position values
- [ ] FX vega calculation (currency pair volatility)
- [ ] FX curvature calculation

##### C. Commodity Sensitivity Engine
**File to create**: `simm/engines/risk_class/commodity_engine.py`
- [ ] Commodity delta calculation
- [ ] Bucket classification (17 commodity types)
- [ ] Commodity vega calculation
- [ ] Commodity curvature calculation

##### D. Cross-Risk Engines
**File to create**: `simm/engines/cross_risk/`
- [ ] `vega_engine.py` - Vol-weighted vega across all risk classes
- [ ] `curvature_engine.py` - Curvature calculation with SF(t) scaling

#### 3. Enhance CRIF Integration
**File**: `simm/engines/portfolio_adapter.py`
- [ ] Add sensitivity netting by risk factor
- [ ] Implement CRIF round-trip validation tests
- [ ] Add batch CRIF processing for large portfolios

#### 4. Create Missing Test Suites
- [ ] `test/test_simm_credit_sensitivity.py`
- [ ] `test/test_simm_fx_sensitivity.py`
- [ ] `test/test_simm_vega_curvature.py`
- [ ] CRIF round-trip integration tests

### Medium Priority

#### 5. Performance Optimizations
- [ ] Add caching layer for repeated calibration lookups
- [ ] Vectorize bucket aggregation calculations
- [ ] Optimize correlation matrix operations
- [ ] Add parallel processing for large portfolios

#### 6. Enhanced Reporting
- [ ] Add PDF report generator (using WeasyPrint or similar)
- [ ] Create dashboard-style HTML report with D3.js
- [ ] Add interactive what-if analysis UI
- [ ] Implement real-time margin monitoring

#### 7. Additional Features
- [ ] SIMM-MVA (Margin Valuation Adjustment) calculation
- [] Historical backtesting framework
- [ ] Stress testing integration
- [ ] Portfolio optimization based on SIMM

---

## Development Guidelines

### Code Style
- **Formatting**: PEP 8, use `black` for formatting
- **Type Hints**: All public APIs must have type hints
- **Docstrings**: Google-style docstrings for all public classes/methods
- **Naming**:
  - Classes: PascalCase (e.g., `SIMMCalculator`)
  - Functions/methods: snake_case (e.g., `calculate_delta_sensitivities`)
  - Constants: UPPER_SNAKE_CASE (e.g., `IR_TENORS`)
  - Private: `_leading_underscore`

### Design Principles

#### 1. Protocol-Based Architecture
Use Python protocols for engine interfaces:

```python
from typing import Protocol, List
from simm.sensitivity import AnySensitivity

class SensitivityEngine(Protocol):
    """Protocol for all sensitivity engines."""

    @property
    def risk_class(self) -> RiskClass:
        """Return the risk class this engine handles."""
        ...

    def calculate_delta_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        """Calculate delta sensitivities."""
        ...

    def calculate_vega_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        """Calculate vega sensitivities."""
        ...
```

#### 2. Immutable Data Models
Sensitivity and result objects should be immutable:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class IRDeltaSensitivity:
    """Interest Rate Delta Sensitivity."""
    trade_id: str
    amount: float
    currency: str
    tenor: float
    sub_curve: IRSubCurve
```

#### 3. Duck Typing for Positions
Accept positions via duck typing, not strict inheritance:

```python
def calculate_sensitivities(self, positions: List[Any]) -> List[AnySensitivity]:
    """Accept any position with required methods."""
    for position in positions:
        # Use hasattr for duck typing
        if hasattr(position, 'get_dv01') and hasattr(position, 'underlying'):
            # Process position
            ...
```

#### 4. Error Handling
Use specific exceptions from `util.exceptions`:

```python
from util.exceptions import ValidationError, NumericalError, MarketDataError

# Validate inputs
if amount == 0:
    raise ValidationError(f"Sensitivity amount cannot be zero for {trade_id}")

# Handle numerical issues
try:
    result = calculate_risk_weight(sensitivity, config)
except ZeroDivisionError:
    raise NumericalError(f"Division by zero in risk weight calculation for {trade_id}")
```

### SIMM Formula Implementation

When implementing SIMM formulas, include formula references in comments:

```python
def calculate_weighted_sensitivity(
    self,
    sensitivity: BaseSensitivity,
    risk_weight: float,
    concentration_risk: float,
) -> float:
    """
    Calculate weighted sensitivity.

    Formula: WS = RW × s × CR
    Where:
        WS = Weighted Sensitivity
        RW = Risk Weight (from calibration data)
        s = Raw sensitivity
        CR = Concentration Risk

    Reference: ISDA SIMM v2.6, Section 3.2.1
    """
    return risk_weight * sensitivity.amount * concentration_risk
```

---

## Testing

### Test Organization

Tests are organized by module:
```
test/
├── test_simm_taxonomy.py              # Taxonomy tests
├── test_simm_config.py                # Configuration tests
├── test_simm_crif.py                  # CRIF parsing tests
├── test_simm_calibration_*.py         # Calibration data tests
├── test_simm_aggregation.py           # Aggregation engine tests
├── test_simm_ir_sensitivity.py        # IR engine tests
├── test_simm_equity_sensitivity.py    # Equity engine tests
├── test_simm_credit_sensitivity.py    # Credit engine tests (NOT CREATED)
├── test_simm_fx_sensitivity.py        # FX engine tests (NOT CREATED)
├── test_simm_engines_base.py          # Base engine tests
├── test_simm_results.py               # Results & attribution tests
├── test_simm_reports.py               # Report generation tests
└── test_simm_whatif.py                # What-if analysis tests
```

### Running Tests

```bash
# Run all SIMM tests
python -m pytest test/test_simm_*.py -v

# Run specific test file
python -m pytest test/test_simm_ir_sensitivity.py -v

# Run with coverage
python -m pytest test/test_simm_*.py --cov=simm --cov-report=html

# Run only passing tests (skip known failures)
python -m pytest test/test_simm_*.py -v --ignore=test/test_simm_credit_sensitivity.py
```

### Writing Tests

Follow these patterns:

#### 1. Sensitivity Engine Tests
```python
def test_calculate_delta_sensitivities():
    """Test IR delta sensitivity calculation."""
    # Arrange
    positions = [create_test_swap()]
    config = SIMMConfig(calculate_delta=True)
    engine = IRSensitivityEngine(config)

    # Act
    sensitivities = engine.calculate_delta_sensitivities(positions, pricing_envs)

    # Assert
    assert len(sensitivities) > 0
    assert all(isinstance(s, IRDeltaSensitivity) for s in sensitivities)
    assert sum(s.amount for s in sensitivities) == pytest.approx(100000.0, rel=1e-3)
```

#### 2. Aggregation Tests
```python
def test_bucket_aggregation():
    """Test bucket-level aggregation."""
    # Arrange
    sensitivities = create_test_sensitivities()
    calculator = BucketAggregator(config)

    # Act
    result = calculator.aggregate(sensitivities)

    # Assert
    assert isinstance(result, BucketResult)
    assert len(result.weighted_sensitivities) > 0
    assert result.margin > 0
```

#### 3. CRIF Round-Trip Tests
```python
def test_crif_round_trip():
    """Test CRIF export and re-import."""
    # Arrange
    sensitivities = create_test_sensitivities()

    # Act
    crif_records = sensitivities_to_crif(sensitivities)
    imported_sensitivities = crif_to_sensitivities(crif_records)

    # Assert
    assert len(imported_sensitivities) == len(sensitivities)
    # Verify key fields match
    for orig, imported in zip(sensitivities, imported_sensitivities):
        assert orig.trade_id == imported.trade_id
        assert orig.amount == pytest.approx(imported.amount, rel=1e-6)
```

---

## Integration Patterns

### Pattern 1: Portfolio to SIMM Result

```python
from simm.engines import SIMMPortfolioAdapter
from simm.engines.aggregation import SIMMCalculator
from simm.config import SIMMConfig

# Configure SIMM
config = SIMMConfig(
    version=SIMMVersion.V2_6,
    calculation_currency="USD",
    calculate_delta=True,
    calculate_vega=True,
    calculate_curvature=True,
)

# Create portfolio adapter
adapter = SIMMPortfolioAdapter(config)

# Convert positions to sensitivities
sensitivities = adapter.convert_portfolio(portfolio)

# Calculate SIMM margin
calculator = SIMMCalculator(config)
result = calculator.calculate_full_simm(sensitivities)

# Access results
print(f"Total SIMM Margin: {result.total_margin}")
print(f"Delta Margin: {result.by_margin_type[MarginType.DELTA]}")
print(f"Vega Margin: {result.by_margin_type[MarginType.VEGA]}")

# Get attribution
if result.attribution:
    print("\nTop 5 Contributors:")
    for contributor in result.attribution.top_contributors(5):
        print(f"  {contributor.trade_id}: {contributor.margin_contribution}")
```

### Pattern 2: CRIF Input to SIMM Result

```python
from simm.crif import parse_crif_csv, crif_to_sensitivities
from simm.engines.aggregation import SIMMCalculator

# Load CRIF file
records, warnings = parse_crif_csv("sensitivities.crif")
print(f"Loaded {len(records)} CRIF records")
print(f"Warnings: {warnings}")

# Convert to sensitivities
sensitivities = crif_to_sensitivities(records)

# Calculate SIMM
calculator = SIMMCalculator(config)
result = calculator.calculate_full_simm(sensitivities)
```

### Pattern 3: Generate Reports

```python
from simm.report import SIMMHTMLReportGenerator, SIMMExcelReportGenerator

# HTML Report
html_gen = SIMMHTMLReportGenerator(include_charts=True)
html_path = html_gen.generate(
    result=result,
    output_path="simm_report.html",
    include_charts=True,
    logo_path="company_logo.png"
)
print(f"HTML report generated: {html_path}")

# Excel Report
excel_gen = SIMMExcelReportGenerator()
excel_path = excel_gen.generate(
    result=result,
    output_path="simm_report.xlsx",
    include_crifs=True
)
print(f"Excel report generated: {excel_path}")
```

### Pattern 4: What-If Analysis

```python
from simm.results.whatif import SIMMWhatIf

# Create what-if analyzer
whatif = SIMMWhatIf(base_sensitivities=sensitivities, config=config)

# Analyze adding new positions
add_result = whatif.impact_of_adding(new_sensitivities)
print(f"Impact of adding positions: {add_result.incremental_margin}")

# Analyze removing positions
remove_result = whatif.impact_of_removing(positions_to_remove)
print(f"Margin reduction from removal: {remove_result.margin_saved}")

# Calculate marginal SIMM
marginal = whatif.marginal_simm(new_position)
print(f"Marginal margin for new position: {marginal}")
```

### Pattern 5: Custom Sensitivity Engine

```python
from simm.engines.base import BaseSensitivityEngine
from simm.taxonomy import RiskClass
from typing import List, Dict, Any

class CustomSensitivityEngine(BaseSensitivityEngine):
    """Custom sensitivity engine for specific risk class."""

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.INTEREST_RATE  # or other risk class

    def calculate_delta_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        # Implementation here
        ...

    def calculate_vega_sensitivities(
        self,
        positions: List[Any],
        pricing_environments: Dict[str, Any],
    ) -> List[AnySensitivity]:
        # Implementation here
        ...

# Register with factory
from simm.engines.factory import register_engine
register_engine(RiskClass.INTEREST_RATE, CustomSensitivityEngine)
```

---

## API Reference

### Core Classes

#### SIMMConfig
Configuration for SIMM calculations.

```python
@dataclass
class SIMMConfig:
    version: SIMMVersion = SIMMVersion.V2_6
    calculation_currency: str = "USD"

    # Margin type flags
    calculate_delta: bool = True
    calculate_vega: bool = True
    calculate_curvature: bool = True
    calculate_base_corr: bool = True

    # Product class multipliers
    ms_rates_fx: float = 1.0
    ms_credit: float = 1.0
    ms_equity: float = 1.0
    ms_commodity: float = 1.0

    # Add-ons
    addon_fixed: float = 0.0
    addon_factors: Dict[str, float] = field(default_factory=dict)

    # Result options
    include_attribution: bool = True
    include_bucket_detail: bool = True
```

#### SIMMCalculator
Main calculator orchestrating the full SIMM pipeline.

```python
class SIMMCalculator:
    def calculate_full_simm(
        self,
        sensitivities: List[AnySensitivity],
        config: Optional[SIMMConfig] = None,
    ) -> SIMMResult:
        """Calculate complete SIMM margin with full breakdown."""
        ...

    def calculate_by_risk_class(
        self,
        sensitivities: List[AnySensitivity],
        risk_class: RiskClass,
        config: Optional[SIMMConfig] = None,
    ) -> RiskClassResult:
        """Calculate margin for specific risk class."""
        ...

    def calculate_delta_margin(
        self,
        sensitivities: List[AnySensitivity],
        config: Optional[SIMMConfig] = None,
    ) -> float:
        """Calculate delta margin only."""
        ...
```

#### SIMMResult
Complete SIMM calculation results.

```python
@dataclass
class SIMMResult:
    """Complete SIMM calculation result."""
    total_margin: float

    # By margin type
    by_margin_type: Dict[MarginType, float]
    delta_margin: float
    vega_margin: float
    curvature_margin: float
    base_corr_margin: float

    # By product class
    by_product_class: Dict[ProductClass, ProductClassResult]

    # By risk class
    by_risk_class: Dict[RiskClass, RiskClassMargin]

    # Add-ons
    addon_total: float
    addon_breakdown: AddonBreakdown

    # Attribution (optional)
    attribution: Optional[SIMMAttribution] = None

    # Configuration used
    config: SIMMConfig = field(default_factory=SIMMConfig)
```

### Sensitivity Classes

#### IRDeltaSensitivity
```python
@dataclass(frozen=True)
class IRDeltaSensitivity(BaseSensitivity):
    """Interest Rate Delta Sensitivity (DV01)."""
    trade_id: str
    amount: float
    amount_currency: str
    currency: str          # Currency of the curve
    tenor: float          # Tenor in years
    sub_curve: IRSubCurve  # OIS, LIBOR, etc.
```

#### IRVegaSensitivity
```python
@dataclass(frozen=True)
class IRVegaSensitivity(BaseSensitivity):
    """Interest Rate Vega Sensitivity."""
    trade_id: str
    amount: float
    amount_currency: str
    currency: str
    tenor: float          # Option expiry tenor
```

#### EquityDeltaSensitivity
```python
@dataclass(frozen=True)
class EquityDeltaSensitivity(BaseSensitivity):
    """Equity Delta Sensitivity."""
    trade_id: str
    amount: float
    amount_currency: str
    ticker: str
    bucket: EquityBucket
    sub_class: EquitySubClass  # Size, region, sector
```

#### EquityVegaSensitivity
```python
@dataclass(frozen=True)
class EquityVegaSensitivity(BaseSensitivity):
    """Equity Vega Sensitivity."""
    trade_id: str
    amount: float
    amount_currency: str
    ticker: str
    bucket: EquityBucket
    expiry_tenor: float
```

### Enums

#### RiskClass
```python
class RiskClass(Enum):
    INTEREST_RATE = "IR"
    CREDIT_QUALIFYING = "CreditQ"
    CREDIT_NON_QUALIFYING = "CreditNQ"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    FX = "FX"
```

#### MarginType
```python
class MarginType(Enum):
    DELTA = "Delta"
    VEGA = "Vega"
    CURVATURE = "Curvature"
    BASE_CORR = "BaseCorr"
```

#### ProductClass
```python
class ProductClass(Enum):
    RATES_FX = "RatesFX"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
```

### Key Functions

#### CRIF Functions
```python
def parse_crif_csv(file_path: str) -> Tuple[List[CRIFRecord], List[str]]:
    """Parse CRIF CSV file."""
    ...

def crif_to_sensitivities(records: List[CRIFRecord]) -> SensitivityCollection:
    """Convert CRIF records to sensitivity objects."""
    ...

def sensitivities_to_crif(sensitivities: List[AnySensitivity]) -> List[CRIFRecord]:
    """Convert sensitivities to CRIF records."""
    ...

def write_crif_csv(records: List[CRIFRecord], file_path: str) -> None:
    """Write CRIF records to CSV file."""
    ...
```

#### Portfolio Adapter
```python
class SIMMPortfolioAdapter:
    def convert_portfolio(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
    ) -> SensitivityCollection:
        """Convert portfolio positions to sensitivities."""
        ...

    def convert_positions(
        self,
        positions: List[Union[EquityPosition, FIPosition]],
    ) -> SensitivityCollection:
        """Convert positions to sensitivities."""
        ...
```

---

## Performance Considerations

### Optimization Strategies

#### 1. Use NumPy for Vectorization
```python
import numpy as np

# Bad: Python loops
for sensitivity in sensitivities:
    weighted_sens[i] = sensitivity.amount * risk_weight[i]

# Good: NumPy vectorization
weighted_sens = np.array([s.amount for s in sensitivities]) * risk_weights
```

#### 2. Cache Calibration Data
```python
from functools import lru_cache

class CalibrationAccessor:
    @lru_cache(maxsize=128)
    def get_risk_weight(
        self,
        risk_class: RiskClass,
        bucket: Bucket,
        margin_type: MarginType,
        tenor: float,
        version: SIMMVersion,
    ) -> float:
        """Cached risk weight lookup."""
        ...
```

#### 3. Batch Process Sensitivities
```python
def calculate_all_sensitivities(
    self,
    positions: List[Position],
    pricing_environments: Dict[str, Any],
) -> SensitivityCollection:
    """Batch process all positions for better performance."""
    # Group by position type
    fi_positions = [p for p in positions if isinstance(p, FIPosition)]
    equity_positions = [p for p in positions if isinstance(p, EquityPosition)]

    # Process in batches
    fi_sensitivities = self._process_fi_batch(fi_positions, pricing_envs)
    equity_sensitivities = self._process_equity_batch(equity_positions, pricing_envs)

    return SensitivityCollection([...fi_sensitivities, ...equity_sensitivities])
```

#### 4. Lazy Evaluation for Attribution
```python
@dataclass
class SIMMResult:
    ...

    @property
    def attribution(self) -> Optional[SIMMAttribution]:
        """Lazy load attribution only when accessed."""
        if self._attribution is None and self._include_attribution:
            self._attribution = self._calculate_attribution()
        return self._attribution
```

### Benchmarking

Use the benchmark tests to monitor performance:

```bash
# Run benchmark tests
python -m pytest test/test_simm_aggregation.py::test_performance_benchmark -v

# Profile specific functions
python -m pytest test/test_simm_aggregation.py -k "profile" --profile
```

### Memory Management

- Use generators for large sensitivity lists
- Clear pricing environments after calculation
- Avoid storing large intermediate results

---

## Additional Resources

### Documentation
- **ISDA SIMM v2.6 Methodology**: Full SIMM specification
- **QuantArk Portfolio Module**: Integration with equity and fixed income portfolios
- **OpenSpec Changes**: See `openspec/changes/` for detailed implementation specs

### External Dependencies
- `plotly` - Interactive charts in HTML reports
- `openpyxl` - Excel report generation
- `pandas` - CRIF CSV processing (optional)
- `numpy` - Numerical computations

### Related Modules
- `portfolio.equity` - Equity position management
- `portfolio.fi` - Fixed income position management
- `var` - Value-at-Risk calculations
- `asset.equity` - Equity pricing engines
- `asset.bond` - Fixed income pricing engines

---

## Summary

The SIMM module is a comprehensive implementation of ISDA SIMM v2.6 with **strong foundation support** and **partial sensitivity engine coverage**. The **aggregation layer is complete and robust**, enabling end-to-end SIMM calculations for Interest Rate and Equity risk classes. The **reporting infrastructure is production-ready** with HTML, Excel, and CRIF export capabilities.

**Key Strengths**:
- Clean, modular architecture
- Complete aggregation logic
- Comprehensive reporting
- Strong test coverage for implemented features
- Full calibration data support

**Main Gaps**:
- Missing Credit, FX, and Commodity sensitivity engines
- Incomplete IR vega/curvature calculations
- Limited cross-risk sensitivity engines
- Missing test suites for unimplemented features

**Next Steps**:
1. Complete IR sensitivity engine (vega, curvature, sub-curve detection)
2. Implement Credit sensitivity engine
3. Implement FX sensitivity engine
4. Add cross-risk vega and curvature engines
5. Create comprehensive test suites for all engines

For questions or contributions, refer to the OpenSpec proposals in `openspec/changes/` for detailed implementation specifications.
