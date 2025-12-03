# Change: Add SIMM Reporting Module

## Why

After SIMM margin is calculated, users need comprehensive results with detailed attribution, what-if analysis capabilities, and formatted reports for regulatory and internal purposes. This change provides the results dataclasses, attribution breakdown, and report generation capabilities that complete the SIMM module.

## What Changes

- **NEW** `simm/results/` submodule with result dataclasses
- **NEW** `SIMMResult` comprehensive result dataclass
- **NEW** `SIMMAttribution` for margin decomposition
- **NEW** What-if analysis (incremental SIMM, position impact)
- **NEW** `simm/report/` submodule for report generation
- **NEW** HTML report generator with charts
- **NEW** Excel report generator with detailed sheets
- **NEW** CRIF export from results

## Impact

- Affected specs: simm-results (new), simm-attribution (new)
- Affected code: New `simm/results/` and `simm/report/` submodules
- Dependencies:
  - simm-risk-taxonomy from add-simm-foundation
  - simm-margin-calculator from add-simm-aggregation
- Downstream dependencies: None (final layer of SIMM module)

