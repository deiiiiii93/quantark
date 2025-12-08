# Tasks: Add SIMM Reporting Module

## 1. Module Structure Setup
- [x] 1.1 Create `simm/results/__init__.py` with public exports
- [x] 1.2 Create `simm/report/__init__.py` with public exports

## 2. Result Dataclasses
- [x] 2.1 Create `simm/results/simm_result.py`
- [x] 2.2 Implement SIMMResult dataclass
- [x] 2.3 Implement RiskClassMargin dataclass
- [x] 2.4 Implement BucketDetail dataclass
- [x] 2.5 Implement SensitivityContribution dataclass
- [x] 2.6 Implement AddonBreakdown dataclass
- [x] 2.7 Add result validation methods
- [x] 2.8 Add result serialization (to_dict, to_json)

## 3. Attribution Module
- [x] 3.1 Create `simm/results/attribution.py`
- [x] 3.2 Implement SIMMAttribution dataclass
- [x] 3.3 Implement PositionAttribution dataclass
- [x] 3.4 Implement ContributorInfo dataclass
- [x] 3.5 Implement attribution calculation from results
- [x] 3.6 Implement top contributor identification
- [x] 3.7 Add percentage contribution calculations

## 4. What-If Analysis
- [x] 4.1 Create `simm/results/whatif.py`
- [x] 4.2 Implement SIMMWhatIf class
- [x] 4.3 Implement impact_of_adding method
- [x] 4.4 Implement impact_of_removing method
- [x] 4.5 Implement marginal_simm method
- [x] 4.6 Implement WhatIfResult dataclass
- [x] 4.7 Add incremental calculation optimization

## 5. HTML Report Generator
- [x] 5.1 Create `simm/report/html_generator.py`
- [x] 5.2 Create `simm/report/templates/` directory
- [x] 5.3 Create base HTML template
- [x] 5.4 Implement executive summary section
- [x] 5.5 Implement product class waterfall chart (using Plotly)
- [x] 5.6 Implement risk class breakdown tables
- [x] 5.7 Implement bucket detail expandable sections
- [x] 5.8 Implement margin type pie charts
- [x] 5.9 Implement top contributors table
- [x] 5.10 Add CSS styling for professional appearance
- [x] 5.11 Add configuration/warnings section

## 6. Excel Report Generator
- [x] 6.1 Create `simm/report/excel_generator.py`
- [x] 6.2 Implement summary sheet
- [x] 6.3 Implement product class breakdown sheet
- [x] 6.4 Implement per-risk-class detail sheets (IR, CreditQ, etc.)
- [x] 6.5 Implement position attribution sheet
- [x] 6.6 Implement CRIF export sheet
- [x] 6.7 Add formatting (headers, borders, number formats)
- [x] 6.8 Add conditional formatting for key metrics

## 7. CRIF Export
- [x] 7.1 Implement export_sensitivities_to_crif function
- [x] 7.2 Map internal sensitivities to CRIF format
- [x] 7.3 Add file output option (CSV)
- [x] 7.4 Add DataFrame output option
- [x] 7.5 Validate CRIF output format

## 8. Summary Statistics
- [x] 8.1 Implement margin utilization metrics
- [x] 8.2 Implement diversification benefit calculation
- [x] 8.3 Implement concentration analysis
- [x] 8.4 Implement risk driver identification

## 9. Testing
- [x] 9.1 Create `test/test_simm_results.py`
- [x] 9.2 Add result dataclass tests
- [x] 9.3 Add attribution calculation tests
- [x] 9.4 Create `test/test_simm_whatif.py`
- [x] 9.5 Add what-if analysis tests
- [x] 9.6 Create `test/test_simm_reports.py`
- [x] 9.7 Add HTML report generation tests
- [x] 9.8 Add Excel report generation tests
- [x] 9.9 Add CRIF export round-trip tests

## 10. Documentation
- [x] 10.1 Add docstrings to all result classes
- [x] 10.2 Document attribution methodology
- [x] 10.3 Add example report generation code
- [x] 10.4 Document report customization options

