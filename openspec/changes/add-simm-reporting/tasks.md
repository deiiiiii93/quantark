# Tasks: Add SIMM Reporting Module

## 1. Module Structure Setup
- [ ] 1.1 Create `simm/results/__init__.py` with public exports
- [ ] 1.2 Create `simm/report/__init__.py` with public exports

## 2. Result Dataclasses
- [ ] 2.1 Create `simm/results/simm_result.py`
- [ ] 2.2 Implement SIMMResult dataclass
- [ ] 2.3 Implement RiskClassMargin dataclass
- [ ] 2.4 Implement BucketDetail dataclass
- [ ] 2.5 Implement SensitivityContribution dataclass
- [ ] 2.6 Implement AddonBreakdown dataclass
- [ ] 2.7 Add result validation methods
- [ ] 2.8 Add result serialization (to_dict, to_json)

## 3. Attribution Module
- [ ] 3.1 Create `simm/results/attribution.py`
- [ ] 3.2 Implement SIMMAttribution dataclass
- [ ] 3.3 Implement PositionAttribution dataclass
- [ ] 3.4 Implement ContributorInfo dataclass
- [ ] 3.5 Implement attribution calculation from results
- [ ] 3.6 Implement top contributor identification
- [ ] 3.7 Add percentage contribution calculations

## 4. What-If Analysis
- [ ] 4.1 Create `simm/results/whatif.py`
- [ ] 4.2 Implement SIMMWhatIf class
- [ ] 4.3 Implement impact_of_adding method
- [ ] 4.4 Implement impact_of_removing method
- [ ] 4.5 Implement marginal_simm method
- [ ] 4.6 Implement WhatIfResult dataclass
- [ ] 4.7 Add incremental calculation optimization

## 5. HTML Report Generator
- [ ] 5.1 Create `simm/report/html_generator.py`
- [ ] 5.2 Create `simm/report/templates/` directory
- [ ] 5.3 Create base HTML template
- [ ] 5.4 Implement executive summary section
- [ ] 5.5 Implement product class waterfall chart (using Plotly)
- [ ] 5.6 Implement risk class breakdown tables
- [ ] 5.7 Implement bucket detail expandable sections
- [ ] 5.8 Implement margin type pie charts
- [ ] 5.9 Implement top contributors table
- [ ] 5.10 Add CSS styling for professional appearance
- [ ] 5.11 Add configuration/warnings section

## 6. Excel Report Generator
- [ ] 6.1 Create `simm/report/excel_generator.py`
- [ ] 6.2 Implement summary sheet
- [ ] 6.3 Implement product class breakdown sheet
- [ ] 6.4 Implement per-risk-class detail sheets (IR, CreditQ, etc.)
- [ ] 6.5 Implement position attribution sheet
- [ ] 6.6 Implement CRIF export sheet
- [ ] 6.7 Add formatting (headers, borders, number formats)
- [ ] 6.8 Add conditional formatting for key metrics

## 7. CRIF Export
- [ ] 7.1 Implement export_sensitivities_to_crif function
- [ ] 7.2 Map internal sensitivities to CRIF format
- [ ] 7.3 Add file output option (CSV)
- [ ] 7.4 Add DataFrame output option
- [ ] 7.5 Validate CRIF output format

## 8. Summary Statistics
- [ ] 8.1 Implement margin utilization metrics
- [ ] 8.2 Implement diversification benefit calculation
- [ ] 8.3 Implement concentration analysis
- [ ] 8.4 Implement risk driver identification

## 9. Testing
- [ ] 9.1 Create `test/test_simm_results.py`
- [ ] 9.2 Add result dataclass tests
- [ ] 9.3 Add attribution calculation tests
- [ ] 9.4 Create `test/test_simm_whatif.py`
- [ ] 9.5 Add what-if analysis tests
- [ ] 9.6 Create `test/test_simm_reports.py`
- [ ] 9.7 Add HTML report generation tests
- [ ] 9.8 Add Excel report generation tests
- [ ] 9.9 Add CRIF export round-trip tests

## 10. Documentation
- [ ] 10.1 Add docstrings to all result classes
- [ ] 10.2 Document attribution methodology
- [ ] 10.3 Add example report generation code
- [ ] 10.4 Document report customization options

