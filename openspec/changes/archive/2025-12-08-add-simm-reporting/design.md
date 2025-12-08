# Design: SIMM Reporting Module

## Context

SIMM calculation produces a single margin number, but users need detailed attribution to understand margin drivers, validate calculations, and make informed decisions. This design establishes the result structures and reporting capabilities.

### Stakeholders
- Risk managers analyzing margin requirements
- Trading desks optimizing margin usage
- Compliance teams for regulatory reporting
- Operations for reconciliation

### Constraints
- Must provide attribution at multiple levels (product class, risk class, bucket, trade)
- Must support what-if analysis without full recalculation when possible
- Must integrate with existing reporting patterns in the codebase
- Reports must be suitable for both internal and regulatory use

## Goals / Non-Goals

### Goals
- Define comprehensive result dataclasses
- Provide attribution by all SIMM dimensions
- Enable what-if analysis (incremental SIMM)
- Generate HTML and Excel reports
- Support CRIF export of calculated sensitivities

### Non-Goals
- Real-time margin streaming
- Integration with external reporting systems
- Historical SIMM tracking (separate module)

## Decisions

### Decision 1: Module Structure

```
simm/results/
├── __init__.py           # Public exports
├── simm_result.py        # Main result dataclass
└── attribution.py        # Attribution breakdown

simm/report/
├── __init__.py
├── html_generator.py     # HTML report with charts
├── excel_generator.py    # Excel report with sheets
└── templates/            # HTML templates
```

**Rationale**: Follows existing `var/results/` pattern.

### Decision 2: Result Dataclass Hierarchy

```python
@dataclass
class SIMMResult:
    """Complete SIMM calculation result."""
    
    # Total and summary
    total_simm: float
    calculation_currency: str
    calculation_date: date
    simm_version: str
    
    # By product class
    product_class_simm: Dict[ProductClass, float]
    
    # By risk class (within each product class)
    risk_class_margin: Dict[ProductClass, Dict[RiskClass, RiskClassMargin]]
    
    # Add-ons
    addon_amount: float
    addon_details: Optional[AddonBreakdown] = None
    
    # Attribution
    attribution: Optional[SIMMAttribution] = None
    
    # Metadata
    execution_time_seconds: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class RiskClassMargin:
    """Margin breakdown for a single risk class."""
    risk_class: RiskClass
    product_class: ProductClass
    
    # Margin components
    delta_margin: float
    vega_margin: float
    curvature_margin: float
    base_corr_margin: float  # Credit Q only
    total_margin: float
    
    # Bucket-level detail
    bucket_detail: Dict[Union[int, str], BucketDetail]

@dataclass
class BucketDetail:
    """Detailed results for a single bucket."""
    bucket: Union[int, str]
    k_value: float              # Bucket-level K
    s_value: float              # Capped sum S_b
    ws_sum: float               # Sum of weighted sensitivities
    concentration_factor: float # CR for bucket
    
    # Sensitivity contributions
    sensitivities: List[SensitivityContribution]
```

**Rationale**: Hierarchical structure mirrors SIMM aggregation for natural navigation.

### Decision 3: Attribution Structure

```python
@dataclass
class SIMMAttribution:
    """Attribution of SIMM to various dimensions."""
    
    # By product class
    by_product_class: Dict[ProductClass, float]
    
    # By risk class
    by_risk_class: Dict[RiskClass, float]
    
    # By margin type
    by_margin_type: Dict[MarginType, float]
    
    # By bucket (within each risk class)
    by_bucket: Dict[RiskClass, Dict[Union[int, str], float]]
    
    # By position/trade (approximate contribution)
    by_position: Dict[str, PositionAttribution]
    
    # Top contributors
    top_contributors: List[ContributorInfo]

@dataclass
class PositionAttribution:
    """SIMM attribution to a single position."""
    position_id: str
    trade_id: Optional[str]
    underlying: str
    
    # Attributed margins
    delta_contribution: float
    vega_contribution: float
    curvature_contribution: float
    total_contribution: float
    
    # Percentage of total
    pct_of_total: float
    
    # Sensitivities
    sensitivities_count: int

@dataclass
class ContributorInfo:
    """Information about a top SIMM contributor."""
    identifier: str           # position_id, bucket, risk_class, etc.
    contribution_type: str    # "position", "bucket", "risk_class"
    amount: float
    pct_of_total: float
```

**Rationale**: Multiple attribution views for different analysis needs.

### Decision 4: What-If Analysis

```python
class SIMMWhatIf:
    """What-if analysis for SIMM impact."""
    
    def __init__(self, base_result: SIMMResult, calculator: SIMMCalculator):
        self.base_result = base_result
        self.calculator = calculator
    
    def impact_of_adding(
        self,
        new_sensitivities: SensitivityCollection,
    ) -> WhatIfResult:
        """Calculate SIMM impact of adding new sensitivities."""
        
    def impact_of_removing(
        self,
        position_ids: List[str],
    ) -> WhatIfResult:
        """Calculate SIMM impact of removing positions."""
        
    def marginal_simm(
        self,
        sensitivity: Sensitivity,
    ) -> float:
        """Calculate marginal SIMM for a single sensitivity."""

@dataclass
class WhatIfResult:
    """Result of what-if analysis."""
    base_simm: float
    new_simm: float
    delta_simm: float
    delta_pct: float
    
    # Breakdown of change
    delta_by_product_class: Dict[ProductClass, float]
    delta_by_risk_class: Dict[RiskClass, float]
```

**Rationale**: What-if analysis is critical for trading and margin optimization.

### Decision 5: HTML Report Generator

```python
class SIMMHTMLReportGenerator:
    """Generate HTML reports for SIMM results."""
    
    def generate(
        self,
        result: SIMMResult,
        output_path: str,
        include_charts: bool = True,
    ) -> str:
        """Generate HTML report."""
        
    def _render_summary_section(self, result: SIMMResult) -> str:
        """Render executive summary."""
        
    def _render_product_class_breakdown(self, result: SIMMResult) -> str:
        """Render product class waterfall chart."""
        
    def _render_risk_class_breakdown(self, result: SIMMResult) -> str:
        """Render risk class breakdown tables and charts."""
        
    def _render_bucket_detail(self, result: SIMMResult) -> str:
        """Render bucket-level detail tables."""
        
    def _render_top_contributors(self, result: SIMMResult) -> str:
        """Render top contributors table."""
```

Report sections:
1. Executive Summary (total SIMM, date, version)
2. Product Class Breakdown (waterfall chart)
3. Risk Class Analysis (per product class)
4. Margin Type Breakdown (Delta/Vega/Curvature)
5. Bucket Detail (expandable tables)
6. Top Contributors
7. Configuration and Warnings

**Rationale**: Follows existing `var/results/var_report.py` pattern.

### Decision 6: Excel Report Generator

```python
class SIMMExcelReportGenerator:
    """Generate Excel reports for SIMM results."""
    
    def generate(
        self,
        result: SIMMResult,
        output_path: str,
    ) -> str:
        """Generate Excel workbook."""
        
    def _create_summary_sheet(self, wb, result: SIMMResult):
        """Summary sheet with key metrics."""
        
    def _create_product_class_sheet(self, wb, result: SIMMResult):
        """Product class breakdown sheet."""
        
    def _create_risk_class_sheets(self, wb, result: SIMMResult):
        """One sheet per risk class with bucket detail."""
        
    def _create_attribution_sheet(self, wb, result: SIMMResult):
        """Position-level attribution."""
        
    def _create_crif_sheet(self, wb, result: SIMMResult):
        """CRIF format sensitivities."""
```

Sheets:
1. Summary
2. Product Class Breakdown
3. IR Detail
4. Credit Q Detail
5. Credit NQ Detail
6. Equity Detail
7. Commodity Detail
8. FX Detail
9. Position Attribution
10. CRIF Export

**Rationale**: Excel format commonly used for regulatory submissions.

### Decision 7: CRIF Export

```python
def export_sensitivities_to_crif(
    sensitivities: SensitivityCollection,
    trade_id: str,
    valuation_date: date,
    output_path: Optional[str] = None,
) -> Union[List[CRIFRecord], str]:
    """
    Export calculated sensitivities to CRIF format.
    
    Enables round-trip: Portfolio → Sensitivities → CRIF → External System
    """
```

**Rationale**: CRIF export enables data exchange with external systems.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Large reports slow to generate | Lazy loading, pagination |
| Attribution approximation for correlated positions | Document methodology |
| Excel file size for large portfolios | Limit detail rows, summarize |

## Migration Plan

No migration required - new reporting module.

