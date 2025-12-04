# Change: Add SIMM Foundation Module

## Why

ISDA SIMM (Standard Initial Margin Model) is the industry-standard methodology for calculating initial margin for non-cleared OTC derivatives. This foundational module establishes the core data structures, taxonomy, and formats required for SIMM calculation. It provides the building blocks that all subsequent SIMM components (calibration, sensitivity engines, aggregation, reporting) will depend upon.

## What Changes

- **NEW** `simm/` module directory structure
- **NEW** Risk class taxonomy (IR, CreditQ, CreditNQ, Equity, Commodity, FX)
- **NEW** Product class taxonomy (RatesFX, Credit, Equity, Commodity)
- **NEW** Margin type taxonomy (Delta, Vega, Curvature, BaseCorr)
- **NEW** Sensitivity protocols and base interfaces
- **NEW** CRIF (Common Risk Interchange Format) data model and parser
- **NEW** `SIMMConfig` configuration dataclass
- **NEW** Bucket definitions for each risk class
- **NEW** Tenor definitions for IR and Credit risk

## Impact

- Affected specs: simm-risk-taxonomy (new), simm-crif-format (new)
- Affected code: New `simm/` module with `__init__.py`, `config.py`, `taxonomy.py`, `crif/` submodule
- Dependencies: None (foundational module)
- Downstream dependencies: All subsequent SIMM changes depend on this foundation

