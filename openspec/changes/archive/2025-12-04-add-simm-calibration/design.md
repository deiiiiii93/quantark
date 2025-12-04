# Design: SIMM Calibration Data

## Context

ISDA SIMM v2.6 specifies hundreds of calibration parameters across six risk classes. These parameters include risk weights, intra-bucket correlations, inter-bucket correlations, inter-risk-class correlations, and concentration thresholds. The parameters are updated annually by ISDA based on historical market data analysis.

### Stakeholders
- Quantitative developers implementing SIMM calculation
- Risk managers verifying margin calculations
- Compliance teams ensuring ISDA methodology adherence

### Constraints
- All parameters MUST exactly match ISDA SIMM v2.6 specification
- Parameters should be easily versioned for future SIMM updates
- Matrix operations should be efficient for large portfolios
- Must support lookup by risk class, bucket, tenor, currency

## Goals / Non-Goals

### Goals
- Provide complete SIMM v2.6 calibration parameters
- Enable efficient lookup by risk dimensions
- Support version identification (v2.6 effective Dec 2, 2023)
- Provide clear documentation of parameter sources

### Non-Goals
- Historical SIMM versions (can be added later)
- Parameter derivation/fitting (ISDA provides final values)
- Real-time parameter updates

## Decisions

### Decision 1: Module Structure

```
simm/calibration/
├── __init__.py           # Exports all parameter accessors
├── version.py            # SIMM version info
├── ir.py                 # Interest Rate parameters
├── credit_qualifying.py  # Credit Q parameters
├── credit_non_qualifying.py  # Credit NQ parameters
├── equity.py             # Equity parameters
├── commodity.py          # Commodity parameters
├── fx.py                 # FX parameters
└── cross_risk.py         # Inter-risk-class correlations
```

**Rationale**: Separating by risk class matches ISDA specification structure and enables focused testing.

### Decision 2: Risk Weight Storage

Use dictionaries with tuple keys for efficient lookup:

```python
# IR risk weights by (tenor_label, currency_group)
IR_RISK_WEIGHTS = {
    ("2w", "regular"): 109, ("1m", "regular"): 105, ...,
    ("2w", "low"): 15, ("1m", "low"): 18, ...,
    ("2w", "high"): 163, ("1m", "high"): 109, ...
}

# Equity risk weights by bucket
EQUITY_RISK_WEIGHTS = {
    1: 30, 2: 33, 3: 36, 4: 29, 5: 26, 6: 25,
    7: 34, 8: 28, 9: 36, 10: 50, 11: 19, 12: 19,
    "Residual": 50
}
```

**Rationale**: Dictionary lookup is O(1). Tuple keys provide type-safe multi-dimensional access.

### Decision 3: Correlation Matrix Storage

Use numpy arrays for correlation matrices to enable efficient matrix operations:

```python
import numpy as np

# IR tenor correlation matrix (12x12)
IR_TENOR_CORRELATIONS = np.array([
    [1.00, 0.77, 0.67, 0.59, 0.48, 0.39, 0.34, 0.30, 0.25, 0.23, 0.21, 0.20],
    [0.77, 1.00, 0.84, 0.74, 0.56, 0.43, 0.36, 0.31, 0.26, 0.21, 0.19, 0.19],
    # ... remaining rows
])

# Accessor function with symbolic indices
def get_ir_tenor_correlation(tenor1: str, tenor2: str) -> float:
    idx1 = IR_TENOR_LABELS.index(tenor1)
    idx2 = IR_TENOR_LABELS.index(tenor2)
    return IR_TENOR_CORRELATIONS[idx1, idx2]
```

**Rationale**: NumPy arrays enable vectorized correlation calculations. Accessor functions provide convenient symbolic access.

### Decision 4: Inter-Bucket Correlation Tables

For large inter-bucket correlation matrices (e.g., 12x12 for Credit, Equity):

```python
# Equity inter-bucket correlations
EQUITY_INTER_BUCKET_CORRELATIONS = np.array([
    #    1     2     3     4     5     6     7     8     9    10    11    12
    [0.00, 0.18, 0.19, 0.19, 0.14, 0.16, 0.15, 0.16, 0.18, 0.12, 0.19, 0.19],  # 1
    [0.18, 0.00, 0.22, 0.21, 0.15, 0.18, 0.17, 0.19, 0.20, 0.14, 0.21, 0.21],  # 2
    # ... remaining rows
])
```

**Rationale**: Full matrix storage with NumPy for correlation lookup in aggregation formulas.

### Decision 5: Concentration Thresholds

Store as dictionaries with clear units:

```python
# IR delta concentration thresholds (USD mm / bp)
IR_DELTA_CONCENTRATION_THRESHOLDS = {
    "high": 30,        # High volatility currencies
    "regular_well_traded": 330,  # USD, EUR, GBP
    "regular_less_traded": 130,  # AUD, CAD, etc.
    "low": 61          # JPY
}

# Equity delta concentration thresholds (USD mm / %)
EQUITY_DELTA_CONCENTRATION_THRESHOLDS = {
    (1, 2, 3, 4): 3,        # EM Large Cap
    (5, 6, 7, 8): 12,       # DM Large Cap
    9: 0.64,                # EM Small Cap
    10: 0.37,               # DM Small Cap
    (11, 12): 810,          # Indexes
    "Residual": 0.37
}
```

**Rationale**: Clear structure matching SIMM specification tables.

### Decision 6: Parameter Accessor Functions

Provide high-level accessor functions that handle all dimension lookups:

```python
def get_risk_weight(
    risk_class: RiskClass,
    bucket: Union[int, str],
    tenor: Optional[str] = None,
    currency: Optional[str] = None
) -> float:
    """Get risk weight for a given risk class and dimensions."""
    
def get_intra_bucket_correlation(
    risk_class: RiskClass,
    bucket: Union[int, str],
    risk_factor_1: Any,
    risk_factor_2: Any
) -> float:
    """Get correlation between two risk factors in same bucket."""
    
def get_inter_bucket_correlation(
    risk_class: RiskClass,
    bucket_1: Union[int, str],
    bucket_2: Union[int, str]
) -> float:
    """Get correlation between two buckets."""
    
def get_concentration_threshold(
    risk_class: RiskClass,
    bucket: Union[int, str],
    margin_type: MarginType = MarginType.DELTA
) -> float:
    """Get concentration threshold for a bucket."""
```

**Rationale**: Unified API hides per-risk-class complexity. Makes aggregation engine code cleaner.

### Decision 7: Version Information

```python
@dataclass(frozen=True)
class SIMMVersion:
    version: str = "2.6"
    base_version: str = "2.5.6"
    effective_date: date = date(2023, 12, 2)
    publication_date: date = date(2023, 8, 16)

CURRENT_VERSION = SIMMVersion()
```

**Rationale**: Clear tracking of which SIMM version is implemented.

## Parameter Tables

### Interest Rate Parameters

| Tenor | Regular Vol | Low Vol | High Vol |
|-------|-------------|---------|----------|
| 2w    | 109         | 15      | 163      |
| 1m    | 105         | 18      | 109      |
| 3m    | 90          | 9       | 87       |
| 6m    | 71          | 11      | 89       |
| 1yr   | 66          | 13      | 102      |
| 2yr   | 66          | 15      | 96       |
| 3yr   | 64          | 19      | 101      |
| 5yr   | 60          | 23      | 97       |
| 10yr  | 60          | 23      | 97       |
| 15yr  | 61          | 22      | 102      |
| 20yr  | 61          | 22      | 106      |
| 30yr  | 67          | 23      | 101      |

- Inflation risk weight: 61
- Cross-currency basis risk weight: 21
- HVR: 0.47
- VRW: 0.23
- Sub-curve correlation: 99.3%
- Inter-currency correlation: 32%

### Equity Parameters

| Bucket | Risk Weight | Intra-Bucket Corr |
|--------|------------|-------------------|
| 1      | 30         | 18%               |
| 2      | 33         | 20%               |
| 3      | 36         | 28%               |
| 4      | 29         | 24%               |
| 5      | 26         | 25%               |
| 6      | 25         | 36%               |
| 7      | 34         | 35%               |
| 8      | 28         | 37%               |
| 9      | 36         | 23%               |
| 10     | 50         | 27%               |
| 11     | 19         | 45%               |
| 12     | 19         | 45%               |
| Res    | 50         | 0%                |

- HVR: 60%
- VRW: 0.45 (bucket 12: 0.96)

### Inter-Risk-Class Correlations (ψ)

|            | IR   | CreditQ | CreditNQ | Equity | Commodity | FX   |
|------------|------|---------|----------|--------|-----------|------|
| IR         | -    | 4%      | 4%       | 7%     | 37%       | 14%  |
| CreditQ    | 4%   | -       | 54%      | 70%    | 27%       | 37%  |
| CreditNQ   | 4%   | 54%     | -        | 46%    | 24%       | 15%  |
| Equity     | 7%   | 70%     | 46%      | -      | 35%       | 39%  |
| Commodity  | 37%  | 27%     | 24%      | 35%    | -         | 35%  |
| FX         | 14%  | 37%     | 15%      | 39%    | 35%       | -    |

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Manual transcription errors | Comprehensive test suite comparing to ISDA spec |
| Future SIMM version changes | Version field enables multiple versions |
| Large correlation matrices | NumPy arrays for memory efficiency |

## Migration Plan

No migration required - new data module.

