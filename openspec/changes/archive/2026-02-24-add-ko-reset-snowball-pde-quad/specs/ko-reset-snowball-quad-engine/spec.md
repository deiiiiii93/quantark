## ADDED Requirements

### Requirement: KO-reset Snowball Quadrature Engine
The system SHALL provide a `KOResetSnowballQuadEngine` that prices `KnockOutResetSnowballOption` using a two-surface quadrature recursion aligned with SnowballQuadEngine.

#### Scenario: Price KO-reset snowball via quadrature
- **GIVEN** a `KnockOutResetSnowballOption` with discrete pre-KI KO observations
- **WHEN** `KOResetSnowballQuadEngine.price(product, pricing_env)` is called
- **THEN** the engine returns a finite price using a two-surface recursion

### Requirement: Pre/Post KO Schedule Application
The system SHALL apply pre-KI KO observations to V_out and post-KI KO observations to V_in.

#### Scenario: V_out uses pre-KI schedule
- **GIVEN** pre-KI KO observations at times [0.25, 0.5]
- **WHEN** the recursion steps over time 0.25
- **THEN** V_out is updated using the pre-KI KO payoff

#### Scenario: V_in uses post-KI schedule
- **GIVEN** post-KI KO observations at times [0.75, 1.0]
- **WHEN** the recursion steps over time 0.75
- **THEN** V_in is updated using the post-KI KO payoff

### Requirement: Post-KO Mode Validation
The system SHALL support only `PostKOScheduleMode.ABSOLUTE` for quadrature pricing and raise a `ValidationError` for `REBASED`.

#### Scenario: Reject rebased post-KO schedule
- **GIVEN** a `KnockOutResetSnowballOption` with `post_ko_mode=REBASED`
- **WHEN** `KOResetSnowballQuadEngine.price(...)` is called
- **THEN** a `ValidationError` is raised indicating REBASED is unsupported

### Requirement: disable_ko_after_ki Compatibility
The system SHALL honor `disable_ko_after_ki=True` by suppressing post-KI KO jumps on V_in.

#### Scenario: Suppress post-KI KO when disabled
- **GIVEN** a KO-reset snowball with `disable_ko_after_ki=True`
- **WHEN** a post-KI KO observation is reached
- **THEN** V_in is not overwritten by the KO payoff
