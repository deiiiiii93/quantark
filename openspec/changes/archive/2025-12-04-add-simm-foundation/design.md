# Design: SIMM Foundation Module

## Context

ISDA SIMM v2.6 is a complex regulatory model with multiple layers of taxonomy and data structures. This design document establishes the foundational architecture that all SIMM components will build upon. The design must support:

1. Six risk classes with distinct bucketing schemes
2. Four product classes with separate margin calculations
3. Multiple sensitivity types (Delta, Vega, Curvature, Base Correlation)
4. Industry-standard CRIF format for data interchange
5. Extensibility for future SIMM versions

### Stakeholders
- Risk managers calculating regulatory initial margin
- Compliance teams ensuring ISDA SIMM methodology adherence
- Trading desks managing margin requirements
- Systems integrators importing/exporting CRIF data

### Constraints
- Must align with ISDA SIMM v2.6 specification exactly
- Must integrate with existing QuantArk portfolio/position infrastructure
- Must support both auto-calculated and externally-provided sensitivities
- CRIF format must be compatible with industry standards

## Goals / Non-Goals

### Goals
- Define comprehensive enum taxonomy for all SIMM dimensions
- Create type-safe sensitivity data models
- Implement CRIF parser supporting CSV format
- Establish clear interfaces for downstream components
- Enable versioning of SIMM methodology

### Non-Goals
- Risk weight/correlation data (Change 2: add-simm-calibration)
- Sensitivity calculation logic (Change 3: add-simm-sensitivity-engines)
- Aggregation formulas (Change 4: add-simm-aggregation)
- Reporting and attribution (Change 5: add-simm-reporting)

## Decisions

### Decision 1: Enum-based Taxonomy

Use Python `Enum` classes for all categorical dimensions:

```python
class RiskClass(Enum):
    INTEREST_RATE = "IR"
    CREDIT_QUALIFYING = "CreditQ"
    CREDIT_NON_QUALIFYING = "CreditNQ"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    FX = "FX"

class ProductClass(Enum):
    RATES_FX = "RatesFX"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"

class MarginType(Enum):
    DELTA = "Delta"
    VEGA = "Vega"
    CURVATURE = "Curvature"
    BASE_CORR = "BaseCorr"
```

**Rationale**: Type safety, IDE autocompletion, prevents string typos. Consistent with existing `util/enum/` patterns.

### Decision 2: Bucket Definitions as Frozen Dataclasses

```python
@dataclass(frozen=True)
class IRBucket:
    currency: str
    
@dataclass(frozen=True)
class CreditQualifyingBucket:
    bucket_number: int
    credit_quality: str  # "IG" or "HY/NR"
    sector: str

@dataclass(frozen=True)
class EquityBucket:
    bucket_number: int
    size: str           # "Large", "Small", "All"
    region: str         # "Emerging markets", "Developed markets", "All"
    sector: str
```

**Rationale**: Frozen dataclasses are immutable, hashable, and self-documenting. Different risk classes have different bucket structures.

### Decision 3: Tenor Points as Tuple Constants

```python
# IR tenor vertices (in years, matching SIMM spec)
IR_TENORS = (
    0.0384,  # 2 weeks = 14/365
    0.0833,  # 1 month
    0.25,    # 3 months
    0.5,     # 6 months
    1.0,     # 1 year
    2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0
)

# IR tenor labels for display
IR_TENOR_LABELS = ("2w", "1m", "3m", "6m", "1yr", "2yr", "3yr", "5yr", "10yr", "15yr", "20yr", "30yr")

# Credit tenor vertices
CREDIT_TENORS = (1.0, 2.0, 3.0, 5.0, 10.0)
CREDIT_TENOR_LABELS = ("1yr", "2yr", "3yr", "5yr", "10yr")
```

**Rationale**: Immutable tuples for fixed tenor points. Separate labels for human-readable output.

### Decision 4: Sensitivity Protocol

```python
@runtime_checkable
class Sensitivity(Protocol):
    """Base protocol for all SIMM sensitivities."""
    
    @property
    def risk_class(self) -> RiskClass:
        """The risk class this sensitivity belongs to."""
        ...
    
    @property
    def margin_type(self) -> MarginType:
        """Delta, Vega, or Curvature."""
        ...
    
    @property
    def amount(self) -> float:
        """The sensitivity value in calculation currency."""
        ...
    
    @property
    def bucket(self) -> Any:
        """The bucket this sensitivity is assigned to."""
        ...
```

**Rationale**: Protocol-based interface allows different concrete implementations while ensuring type compatibility.

### Decision 5: CRIF Data Model

CRIF (Common Risk Interchange Format) is the industry standard for exchanging SIMM sensitivities.

```python
@dataclass
class CRIFRecord:
    """Single CRIF record representing one sensitivity."""
    trade_id: str
    valuation_date: date
    call_put: Optional[str]  # For options
    
    # SIMM classification
    risk_class: RiskClass
    risk_type: str           # "Risk_IRCurve", "Risk_FX", etc.
    qualifier: str           # Currency for IR, issuer for Credit, etc.
    bucket: str
    label1: str              # Tenor for IR/Credit, empty for others
    label2: str              # Sub-curve for IR, empty for others
    
    # Sensitivity value
    amount: float
    amount_currency: str
    amount_usd: Optional[float] = None
    
    # Optional fields
    im_model: str = "SIMM"
    product_class: Optional[ProductClass] = None
    post_regulations: Optional[str] = None
    collect_regulations: Optional[str] = None
```

**Rationale**: Matches ISDA CRIF specification v2.x. Supports all required fields plus common optional fields.

### Decision 6: Module Structure

```
simm/
├── __init__.py           # Public API exports
├── config.py             # SIMMConfig dataclass
├── taxonomy.py           # Enums, bucket definitions, tenors
├── sensitivity.py        # Sensitivity dataclasses and protocols
└── crif/
    ├── __init__.py
    ├── models.py         # CRIFRecord dataclass
    └── parser.py         # CSV parser, validation
```

**Rationale**: Follows existing module patterns (`var/`, `stresstest/`). CRIF as submodule for clean separation.

### Decision 7: Configuration Dataclass

```python
@dataclass
class SIMMConfig:
    """Configuration for SIMM calculation."""
    
    # SIMM version
    version: str = "2.6"
    
    # Calculation currency
    calculation_currency: str = "USD"
    
    # Which components to calculate
    calculate_delta: bool = True
    calculate_vega: bool = True
    calculate_curvature: bool = True
    calculate_base_corr: bool = True
    
    # Product class multipliers (default = 1.0)
    ms_rates_fx: float = 1.0
    ms_credit: float = 1.0
    ms_equity: float = 1.0
    ms_commodity: float = 1.0
    
    # Add-on configuration
    addon_fixed: float = 0.0
    addon_factors: Dict[str, float] = field(default_factory=dict)
    
    # Output options
    include_attribution: bool = True
    include_bucket_detail: bool = True
```

**Rationale**: Follows `VaRConfig` pattern. Supports all SIMM configuration options including add-ons.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| SIMM version changes break compatibility | Version field in config; calibration data versioned separately |
| CRIF format variations across counterparties | Flexible parser with validation; clear error messages |
| Enum values hardcoded | Comprehensive test coverage; version-specific calibration |
| Complex bucket structures | Frozen dataclasses provide clear structure |

## Migration Plan

No migration required - this is a new module.

## Open Questions

None - foundational design is well-defined by ISDA SIMM specification.

