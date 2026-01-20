# Change: Add SIMM Sensitivity Engines

## Why

ISDA SIMM requires specific sensitivity calculations (Delta, Vega, Curvature) for each risk class. These sensitivities must be calculated according to SIMM methodology definitions (Section C of the spec) and can either be computed from portfolio positions using existing pricing engines or imported from external CRIF files. This change provides the sensitivity calculation layer that bridges portfolio positions to SIMM margin calculation.

## What Changes

- **NEW** `simm/sensitivity/` submodule with per-risk-class sensitivity engines
- **NEW** Interest Rate sensitivity engine (PV01, inflation, cross-currency basis)
- **NEW** Credit sensitivity engine (CS01 by issuer/seniority/tenor, base correlation)
- **NEW** Equity sensitivity engine (equity delta by bucket classification)
- **NEW** Commodity sensitivity engine (commodity delta by bucket)
- **NEW** FX sensitivity engine (FX delta by currency pair)
- **NEW** Vega sensitivity calculator (volatility sensitivities for all risk classes)
- **NEW** Curvature sensitivity calculator (CVR with scaling function)
- **NEW** Portfolio-to-sensitivity conversion integrating with existing Greeks calculator
- **NEW** CRIF sensitivity import integration

## Impact

- Affected specs: simm-ir-sensitivity, simm-credit-sensitivity, simm-equity-sensitivity, simm-commodity-sensitivity, simm-fx-sensitivity (all new)
- Affected code: New `simm/sensitivity/` submodule
- Dependencies: 
  - simm-risk-taxonomy from add-simm-foundation
  - simm-crif-format from add-simm-foundation
  - simm-calibration-data from add-simm-calibration
  - Existing `asset/equity/riskmeasures/greeks_calculator.py`
  - Existing `portfolio/equity/` and `portfolio/fi/` modules
- Downstream dependencies: add-simm-aggregation consumes sensitivities for margin calculation

