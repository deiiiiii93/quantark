# equity-greeks Specification

## Purpose
TBD - created by archiving change update-equity-greeks-observation-theta. Update Purpose after archive.
## Requirements
### Requirement: Numerical theta bump with observation schedules
The system SHALL compute numerical theta for barrier-like equity products that use an `ObservationSchedule` by advancing valuation time, excluding observation entries that are in the past at the bumped valuation, and leaving future observation data unchanged.

#### Scenario: Date-based observation records
- **WHEN** numerical theta advances the valuation date by a bump (e.g., +1 day) for a product with an `ObservationSchedule` containing `observation_date` entries
- **THEN** records with `observation_date` on or before the bumped valuation date SHALL be treated as already observed and excluded from the bumped pricing run
- **AND** remaining records SHALL keep their original `observation_date` values (no forward shift), with time-to-observation recalculated from the bumped valuation date

#### Scenario: Time-based observation records
- **WHEN** numerical theta applies a time bump Δt (e.g., 1/365) for a product with an `ObservationSchedule` containing `observation_time` entries measured from the valuation date
- **THEN** records with `observation_time` ≤ Δt SHALL be treated as already observed and excluded from the bumped pricing run
- **AND** remaining records SHALL keep their original observation data while their effective time-to-observation is reduced by Δt (i.e., resolved relative to the bumped valuation date, not shifted forward)

#### Scenario: Past observations locked
- **WHEN** past observation records are excluded during the theta bump
- **THEN** their barrier checks SHALL NOT be re-evaluated, and aggregation modes SHALL apply only to the remaining future records

#### Scenario: Legacy and continuous monitoring unaffected
- **WHEN** a product has no `ObservationSchedule` or uses continuous/expiry-only monitoring
- **THEN** numerical theta bump behavior SHALL remain unchanged from current implementation

