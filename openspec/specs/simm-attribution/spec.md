# simm-attribution Specification

## Purpose
TBD - created by archiving change add-simm-reporting. Update Purpose after archive.
## Requirements
### Requirement: SIMMAttribution Dataclass

The system SHALL provide a `SIMMAttribution` dataclass with margin attribution across dimensions:
- `by_product_class`: Dict[ProductClass, float]
- `by_risk_class`: Dict[RiskClass, float]
- `by_margin_type`: Dict[MarginType, float]
- `by_bucket`: Dict[RiskClass, Dict[Union[int, str], float]]
- `by_position`: Dict[str, PositionAttribution]
- `top_contributors`: List[ContributorInfo]

#### Scenario: Access attribution by risk class
- **GIVEN** a SIMMAttribution
- **WHEN** accessing by_risk_class
- **THEN** margin attributed to each risk class is returned

#### Scenario: Get bucket-level attribution
- **GIVEN** a SIMMAttribution
- **WHEN** accessing by_bucket[RiskClass.EQUITY]
- **THEN** margin attributed to each equity bucket is returned

### Requirement: PositionAttribution Dataclass

The system SHALL provide a `PositionAttribution` dataclass with:
- `position_id`: Position identifier
- `trade_id`: Optional trade identifier
- `underlying`: Underlying asset identifier
- `delta_contribution`: Attributed delta margin
- `vega_contribution`: Attributed vega margin
- `curvature_contribution`: Attributed curvature margin
- `total_contribution`: Total attributed margin
- `pct_of_total`: Percentage of total SIMM

#### Scenario: Access position contribution
- **GIVEN** a PositionAttribution for position P001
- **WHEN** accessing total_contribution
- **THEN** the position's approximate SIMM contribution is returned

#### Scenario: Calculate position percentage
- **GIVEN** position contribution = 50, total SIMM = 1000
- **WHEN** accessing pct_of_total
- **THEN** 5.0 (%) is returned

### Requirement: Top Contributors Identification

The system SHALL identify top SIMM contributors:
- Top positions by margin contribution
- Top buckets by margin contribution
- Top risk classes by margin contribution

#### Scenario: Get top 10 positions
- **GIVEN** a portfolio with 100 positions
- **WHEN** accessing top_contributors with filter type="position"
- **THEN** 10 positions with highest contribution are returned

#### Scenario: Get top buckets across risk classes
- **GIVEN** multiple risk classes with multiple buckets
- **WHEN** accessing top contributors by bucket
- **THEN** buckets are ranked by contribution amount

### Requirement: Attribution Methodology

The system SHALL calculate position-level attribution using the following methodology:
1. For each position, calculate its weighted sensitivities
2. Attribute bucket K proportionally by WS² contribution
3. Attribute risk class margin proportionally by bucket contribution
4. Sum to get total position attribution

Note: Due to correlation effects, sum of position attributions may not exactly equal total SIMM.

#### Scenario: Attribution sum approximation
- **GIVEN** position attributions calculated
- **WHEN** summing all position attributions
- **THEN** result approximately equals total SIMM (within 5%)

### Requirement: HTML Report Generator

The system SHALL provide an `SIMMHTMLReportGenerator` class that generates HTML reports with:
- Executive summary with total SIMM
- Product class waterfall chart
- Risk class breakdown tables
- Margin type distribution charts
- Bucket-level detail (expandable)
- Top contributors table
- Configuration and warnings section

#### Scenario: Generate HTML report
- **GIVEN** a SIMMResult
- **WHEN** calling generator.generate(result, "report.html")
- **THEN** an HTML file is created at the specified path
- **AND** the file contains all required sections

#### Scenario: Include interactive charts
- **GIVEN** include_charts=True
- **WHEN** generating HTML report
- **THEN** Plotly charts are embedded for interactive visualization

### Requirement: HTML Report Styling

The system SHALL produce professionally styled HTML reports:
- Clean, modern CSS styling
- Responsive layout for different screen sizes
- Consistent color scheme (matching QuantArk branding)
- Clear typography and hierarchy
- Print-friendly formatting

#### Scenario: Print HTML report
- **GIVEN** a generated HTML report
- **WHEN** printing to PDF
- **THEN** the output is properly formatted for printing

### Requirement: Excel Report Generator

The system SHALL provide an `SIMMExcelReportGenerator` class that generates Excel workbooks with sheets:
1. Summary - Key metrics and totals
2. Product Class - Breakdown by product class
3. IR Detail - Interest Rate bucket detail
4. Credit Q Detail - Credit Qualifying bucket detail
5. Credit NQ Detail - Credit Non-Qualifying bucket detail
6. Equity Detail - Equity bucket detail
7. Commodity Detail - Commodity bucket detail
8. FX Detail - FX detail
9. Position Attribution - Per-position breakdown
10. CRIF Export - Sensitivities in CRIF format

#### Scenario: Generate Excel report
- **GIVEN** a SIMMResult
- **WHEN** calling generator.generate(result, "report.xlsx")
- **THEN** an Excel workbook is created with all sheets

#### Scenario: Excel formatting
- **GIVEN** a generated Excel report
- **WHEN** opening the file
- **THEN** proper formatting is applied (headers, borders, number formats)

### Requirement: CRIF Export from Results

The system SHALL provide CRIF export functionality:
- Export calculated sensitivities to CRIF CSV format
- Include all sensitivity fields per CRIF specification
- Support filtering by risk class or product class

#### Scenario: Export sensitivities to CRIF
- **GIVEN** a SIMMResult with sensitivities
- **WHEN** calling export_to_crif(result, "output.csv")
- **THEN** a valid CRIF CSV file is created

#### Scenario: Round-trip CRIF
- **GIVEN** sensitivities exported to CRIF
- **WHEN** parsing the CRIF file back
- **THEN** sensitivities match the original values

### Requirement: Report Configuration

The system SHALL support report configuration options:
- `include_bucket_detail`: Include/exclude bucket tables
- `include_position_attribution`: Include/exclude position breakdown
- `max_positions_shown`: Limit positions in attribution table
- `decimal_places`: Number format precision
- `chart_style`: Chart styling options

#### Scenario: Generate summary-only report
- **GIVEN** include_bucket_detail=False, include_position_attribution=False
- **WHEN** generating HTML report
- **THEN** only summary sections are included

### Requirement: Diversification Benefit Calculation

The system SHALL calculate and report diversification benefit:
```
Diversification Benefit = Σ(standalone margins) - SIMM
```

#### Scenario: Calculate diversification benefit
- **GIVEN** standalone risk class margins summing to 1200
- **AND** aggregated SIMM = 1000
- **WHEN** calculating diversification benefit
- **THEN** benefit = 200 (16.7%)

### Requirement: Concentration Analysis

The system SHALL provide concentration analysis:
- Identify risk classes with highest concentration factors
- Flag positions exceeding concentration thresholds
- Report concentration impact on margin

#### Scenario: Report concentration impact
- **GIVEN** a currency with CR = 1.5
- **WHEN** generating concentration analysis
- **THEN** the currency is flagged as concentrated
- **AND** margin impact of concentration is calculated

### Requirement: Report Templates

The system SHALL support customizable HTML templates:
- Base template with standard sections
- Ability to override sections
- Custom CSS injection
- Logo/branding support

#### Scenario: Apply custom branding
- **GIVEN** custom logo and CSS
- **WHEN** generating report with branding config
- **THEN** custom branding appears in output

### Requirement: Batch Report Generation

The system SHALL support batch report generation:
- Generate reports for multiple dates
- Generate comparison reports across time periods
- Support parallel generation for performance

#### Scenario: Generate weekly reports
- **GIVEN** SIMM results for 5 business days
- **WHEN** calling generate_batch(results, output_dir)
- **THEN** 5 HTML files and 5 Excel files are created

### Requirement: Report Validation

The system SHALL validate generated reports:
- Verify all numbers match the source result
- Check that totals sum correctly
- Validate chart data matches tables

#### Scenario: Validate report accuracy
- **GIVEN** a generated report
- **WHEN** running validation
- **THEN** all displayed numbers match the SIMMResult

