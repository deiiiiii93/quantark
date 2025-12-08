## ADDED Requirements

### Requirement: ObservationSchedule structure for barrier-like products
The system SHALL provide an `ObservationSchedule` that holds ordered `ObservationRecord` entries for discrete barrier monitoring across barrier-like products.

#### Scenario: Per-date barrier data with validation
- **WHEN** a user defines an `ObservationSchedule` with `ObservationRecord` entries containing observation time (year fraction or date), barrier inputs (single `barrier` or `upper_barrier`/`lower_barrier`), and payoff terms (rebate or return-rate amount)
- **THEN** the schedule SHALL accept the per-date values
- **AND** SHALL validate records are ordered by observation time and non-empty for discrete monitoring

#### Scenario: Reuse across product families
- **WHEN** the same schedule is attached to `BarrierOption`, `DoubleBarrierOption`, `OneTouchOption`, or `DoubleOneTouchOption`
- **THEN** the schedule SHALL be consumable without redefining the product type
- **AND** barrier inputs in each record SHALL align with the product’s barrier configuration (single vs upper/lower)

### Requirement: Aggregation modes for discrete hits
The system SHALL support aggregation modes on `ObservationSchedule` to control payoff behavior when multiple observation dates are hit.

#### Scenario: Stop at first hit
- **WHEN** aggregation mode is `stop-first-hit`
- **AND** barrier conditions are evaluated in chronological order
- **THEN** the system SHALL apply the payoff from the first hit record and ignore later records

#### Scenario: Accumulate all hits
- **WHEN** aggregation mode is `accumulate`
- **THEN** the system SHALL sum the payoffs of all hit records across the schedule

#### Scenario: Best payoff across hits
- **WHEN** aggregation mode is `best`
- **THEN** the system SHALL take the maximum payoff across all hit records

#### Scenario: Worst payoff across hits
- **WHEN** aggregation mode is `worst`
- **THEN** the system SHALL take the minimum payoff across all hit records

### Requirement: Backward compatibility for discrete monitoring
The system SHALL retain existing discrete monitoring inputs (`observation_type`, `observation_dates`, uniform barrier/rebate) when no `ObservationSchedule` is supplied.

#### Scenario: Legacy fields without schedule
- **WHEN** a barrier-like product is created without an `ObservationSchedule`
- **THEN** discrete monitoring SHALL continue to use legacy `observation_dates` with a uniform barrier/rebate as today

#### Scenario: Schedule overrides legacy dates
- **WHEN** an `ObservationSchedule` is supplied alongside legacy `observation_dates`
- **THEN** the system SHALL use the schedule for discrete monitoring
- **AND** SHALL raise validation errors on inconsistent inputs (e.g., empty schedule with discrete type)

