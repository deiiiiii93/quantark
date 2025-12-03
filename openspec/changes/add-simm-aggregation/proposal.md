# Change: Add SIMM Aggregation Engine

## Why

ISDA SIMM calculates initial margin by aggregating sensitivities through multiple levels: within buckets, across buckets within risk classes, across risk classes within product classes, and finally across product classes. This change implements the core SIMM margin calculation formulas including concentration risk, weighted sensitivities, and all aggregation steps per SIMM Sections B and 5-13.

## What Changes

- **NEW** `simm/engine/` submodule with aggregation logic
- **NEW** Concentration risk factor calculation (CR, VCR, g_bc)
- **NEW** Weighted sensitivity calculation (WS = RW × s × CR)
- **NEW** Bucket-level aggregation (K_b formula)
- **NEW** Risk class aggregation (DeltaMargin, VegaMargin, CurvatureMargin)
- **NEW** Base correlation margin (Credit Qualifying only)
- **NEW** Product class aggregation (SIMM_product formula)
- **NEW** Total SIMM calculation with product class summation
- **NEW** Add-on formulas (notional-based, multiplicative scales)
- **NEW** Main `SIMMCalculator` class as unified entry point

## Impact

- Affected specs: simm-margin-calculator (new)
- Affected code: New `simm/engine/` submodule
- Dependencies:
  - simm-risk-taxonomy from add-simm-foundation
  - simm-calibration-data from add-simm-calibration
  - simm-*-sensitivity from add-simm-sensitivity-engines
- Downstream dependencies: add-simm-reporting consumes SIMM results

