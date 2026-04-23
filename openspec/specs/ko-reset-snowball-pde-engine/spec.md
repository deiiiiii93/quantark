## ADDED Requirements

### Requirement: KO-reset Snowball PDE Solver
The system SHALL provide a `KOResetSnowballPDESolver` that prices `KnockOutResetSnowballOption` using the two-surface PDE method (V0/V1) consistent with SnowballPDESolver.

#### Scenario: Price KO-reset snowball via PDE solver
- **GIVEN** a `KnockOutResetSnowballOption` with discrete pre-KI KO observations and KI monitoring
- **WHEN** `KOResetSnowballPDESolver.price(product, pricing_env)` is called
- **THEN** the solver returns a finite price using two-surface recursion

### Requirement: Pre/Post KO Schedule Application
The system SHALL apply the pre-KI KO schedule to the V0 surface and the post-KI KO schedule to the V1 surface.

#### Scenario: V0 uses pre-KI schedule
- **GIVEN** pre-KI KO observations at times [0.25, 0.5]
- **WHEN** PDE steps backward across time 0.25
- **THEN** the KO jump is applied to V0 using the pre-KI barrier/rate

#### Scenario: V1 uses post-KI schedule
- **GIVEN** post-KI KO observations at times [0.75, 1.0]
- **WHEN** PDE steps backward across time 0.75
- **THEN** the KO jump is applied to V1 using the post-KI barrier/rate

### Requirement: Post-KO Mode Validation
The system SHALL support only `PostKOScheduleMode.ABSOLUTE` for PDE pricing and raise a `ValidationError` for `REBASED`.

#### Scenario: Reject rebased post-KO schedule
- **GIVEN** a `KnockOutResetSnowballOption` with `post_ko_mode=REBASED`
- **WHEN** `KOResetSnowballPDESolver.price(...)` is called
- **THEN** a `ValidationError` is raised indicating REBASED is unsupported

### Requirement: disable_ko_after_ki Compatibility
The system SHALL honor `disable_ko_after_ki=True` by suppressing post-KI KO jumps on V1.

#### Scenario: Suppress post-KI KO when disabled
- **GIVEN** a KO-reset snowball with `disable_ko_after_ki=True`
- **WHEN** a post-KI KO observation is reached
- **THEN** V1 is not overwritten by the KO payoff

### Requirement: PDEEngine Dispatch Integration
The system SHALL allow `PDEEngine` to dispatch `KnockOutResetSnowballOption` to `KOResetSnowballPDESolver`.

#### Scenario: KO-reset product via PDEEngine
- **GIVEN** a `KnockOutResetSnowballOption`
- **WHEN** `PDEEngine.price(product, pricing_env)` is called
- **THEN** the price is produced by `KOResetSnowballPDESolver`
