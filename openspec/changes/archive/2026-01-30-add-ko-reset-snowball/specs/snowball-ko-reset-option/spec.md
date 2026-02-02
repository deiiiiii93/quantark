# snowball-ko-reset-option Specification

## Purpose
Define a KO-reset snowball option product where KO terms switch to a second schedule after a KI event, with support for absolute and rebased post-KI schedules, and Monte Carlo pricing support.

## ADDED Requirements

### Requirement: KO-reset product definition
The system SHALL provide a `KnockOutResetSnowballOption` product that:
- Uses a pre-KI KO schedule before KI is triggered.
- Switches to a post-KI KO schedule after KI is triggered.
- Uses the same KI logic and payoff structure as Snowball (V0/V1) aside from the KO schedule switch.

#### Scenario: Construct KO-reset snowball
- **GIVEN** pre-KI KO schedule, post-KI KO schedule, and KI barrier schedule
- **WHEN** a `KnockOutResetSnowballOption` is constructed
- **THEN** the product validates schedules and exposes both KO schedules and KI config

---

### Requirement: Post-KI schedule mode
The system SHALL support two post-KI KO schedule modes:
- `ABSOLUTE`: post-KI KO schedule uses fixed calendar times/dates; only observations after KI apply.
- `REBASED`: post-KI KO schedule uses offsets from the KI event time.

#### Scenario: Absolute mode switch
- **GIVEN** a KO-reset product in `ABSOLUTE` mode
- **AND** a KI event occurs at time t_ki
- **WHEN** KO is evaluated after KI
- **THEN** only post-KI schedule observations with time > t_ki are considered

#### Scenario: Rebased mode switch
- **GIVEN** a KO-reset product in `REBASED` mode with offsets [0.25, 0.5]
- **AND** a KI event occurs at time t_ki
- **WHEN** KO is evaluated after KI
- **THEN** KO is observed at times [t_ki + 0.25, t_ki + 0.5]

---

### Requirement: Rebasing constraint for MC
For Monte Carlo pricing, rebased post-KI schedules SHALL require discrete KI monitoring.

#### Scenario: Reject rebased with continuous KI
- **GIVEN** a KO-reset product with `REBASED` mode and continuous KI
- **WHEN** the product is validated for MC pricing
- **THEN** a ValidationError is raised indicating rebased mode requires discrete KI

---

### Requirement: Monte Carlo pricing support
The Monte Carlo engine SHALL price KO-reset snowball options by switching KO schedules after KI.

#### Scenario: MC pricing with KO-reset
- **GIVEN** a KO-reset product with pre-KI KO rate of 15% and post-KI KO rate of 3%
- **WHEN** priced with the MC engine
- **THEN** the price is computed without error and reflects the KO schedule switch

---

### Requirement: Event stats attribution
The Monte Carlo engine SHALL report KO probabilities for pre-KI and post-KI schedules along with overall V0/V1 probabilities.

#### Scenario: KO probabilities split by phase
- **GIVEN** a KO-reset product priced with the MC engine
- **WHEN** event stats are requested
- **THEN** the result includes pre-KI KO probability, post-KI KO probability, and total KO probability
