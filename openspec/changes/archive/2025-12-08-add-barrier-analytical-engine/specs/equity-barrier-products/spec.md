## MODIFIED Requirements
### Requirement: ObservationSchedule structure for barrier-like products
The system SHALL provide an `ObservationSchedule` that holds ordered `ObservationRecord` entries for discrete barrier monitoring across barrier-like products and SHALL support an additional `EXPIRY` observation type representing monitoring only at exercise.

#### Scenario: Per-date barrier data with validation
- **WHEN** a user defines an `ObservationSchedule` with observation time (year fraction or date), barrier inputs (single `barrier` or `upper_barrier`/`lower_barrier`), and payoff terms (rebate or return-rate amount)
- **THEN** the schedule SHALL accept the per-date values
- **AND** SHALL validate records are ordered by observation time and non-empty for discrete monitoring

#### Scenario: Reuse across product families
- **WHEN** the same schedule is attached to `BarrierOption`, `DoubleBarrierOption`, `OneTouchOption`, or `DoubleOneTouchOption`
- **THEN** the schedule SHALL be consumable without redefining the product type
- **AND** barrier inputs in each record SHALL align with the product’s barrier configuration (single vs upper/lower)

#### Scenario: Regular discrete monitoring readiness
- **WHEN** an analytical engine requires barrier shift for discrete monitoring
- **THEN** the schedule SHALL expose frequency inference/validation and fixed-payoff checks to ensure a regular grid with consistent payoffs

#### Scenario: Legacy fields without schedule
- **WHEN** a barrier-like product is created without an `ObservationSchedule`
- **THEN** discrete monitoring SHALL continue to use legacy `observation_dates` with a uniform barrier/rebate as today

#### Scenario: Schedule overrides legacy dates
- **WHEN** an `ObservationSchedule` is supplied alongside legacy `observation_dates`
- **THEN** the system SHALL use the schedule for discrete monitoring
- **AND** SHALL raise validation errors on inconsistent inputs (e.g., empty schedule with discrete type)

