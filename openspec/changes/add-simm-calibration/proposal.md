# Change: Add SIMM Calibration Data

## Why

ISDA SIMM v2.6 requires specific calibration parameters (risk weights, correlations, concentration thresholds) to calculate initial margin. These parameters are published by ISDA and updated annually. This change provides a complete, versioned implementation of all SIMM v2.6 calibration data that the aggregation engine will use.

## What Changes

- **NEW** `simm/calibration/` submodule with per-risk-class parameter files
- **NEW** Interest Rate risk weights (12 tenors × 3 currency groups) and correlations
- **NEW** Credit Qualifying risk weights (12 buckets + residual) and correlations
- **NEW** Credit Non-Qualifying risk weights (2 buckets + residual) and correlations
- **NEW** Equity risk weights (12 buckets + residual) and correlations
- **NEW** Commodity risk weights (17 buckets) and correlations
- **NEW** FX risk weights and correlations
- **NEW** Inter-risk-class correlation matrix (ψ)
- **NEW** Delta concentration thresholds by risk class
- **NEW** Vega concentration thresholds by risk class
- **NEW** Historical Volatility Ratios (HVR) by risk class
- **NEW** Vega Risk Weights (VRW) by risk class

## Impact

- Affected specs: simm-calibration-data (new)
- Affected code: New `simm/calibration/` submodule
- Dependencies: Requires simm-risk-taxonomy from add-simm-foundation
- Downstream dependencies: add-simm-aggregation uses calibration data for margin calculation

