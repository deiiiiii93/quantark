# engine-enums Specification

## Purpose
TBD - created by archiving change add-pde-greeks-mode. Update Purpose after archive.
## Requirements
### Requirement: GreeksCalculationMode Enum

The system SHALL provide a `GreeksCalculationMode` enum in `util/enum/engine_enums.py` with three values:

- `ENGINE = "engine"`: Use engine's `calculate_greeks()` if available
- `BUMP = "bump"`: Use finite difference bump method
- `AUTO = "auto"`: Use engine method for PDE, bump otherwise

#### Scenario: Enum value access
- **GIVEN** `GreeksCalculationMode` enum is imported
- **WHEN** accessing `GreeksCalculationMode.BUMP.value`
- **THEN** returns `"bump"`
- **AND** accessing `GreeksCalculationMode.ENGINE.value` returns `"engine"`
- **AND** accessing `GreeksCalculationMode.AUTO.value` returns `"auto"`

#### Scenario: String conversion
- **GIVEN** `GreeksCalculationMode` enum is imported
- **WHEN** calling `str(GreeksCalculationMode.BUMP)`
- **THEN** returns `"bump"`
- **AND** calling `str(GreeksCalculationMode.ENGINE)` returns `"engine"`
- **AND** calling `str(GreeksCalculationMode.AUTO)` returns `"auto"`

### Requirement: Enum Export

The system SHALL export `GreeksCalculationMode` from `util/enum/__init__.py`.

#### Scenario: Import from util.enum
- **GIVEN** the enum is defined in `util.enum.engine_enums`
- **WHEN** importing `from util.enum import GreeksCalculationMode`
- **THEN** import succeeds
- **AND** enum can be used directly

