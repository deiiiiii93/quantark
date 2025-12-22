# snowball-option-helpers Specification Delta

## MODIFIED Requirements

### Requirement: Airbag Snowball Helper

The system SHALL provide a `create_airbag_snowball()` function that creates a snowball with reduced participation below airbag barrier, and the resulting SnowballOption SHALL compute correct airbag payoffs.

#### Scenario: Airbag payoff when spot below airbag barrier
- **GIVEN** a SnowballOption created with `create_airbag_snowball(initial_price=100, strike=100, maturity=1.0, airbag_barrier=60, airbag_participation_rate=0.5)`
- **AND** a V1 state (KI triggered, no KO)
- **WHEN** `get_maturity_payoff_v1(spot=50)` is called with spot=50 (below airbag_barrier=60)
- **THEN** the payoff uses `airbag_participation_rate=0.5` instead of standard `participation_rate=1.0`
- **AND** the loss is reduced by 50% compared to standard snowball

#### Scenario: Standard payoff when spot above airbag barrier
- **GIVEN** a SnowballOption created with `create_airbag_snowball(initial_price=100, strike=100, maturity=1.0, airbag_barrier=60, airbag_participation_rate=0.5)`
- **AND** a V1 state (KI triggered, no KO)
- **WHEN** `get_maturity_payoff_v1(spot=70)` is called with spot=70 (above airbag_barrier=60)
- **THEN** the payoff uses the standard `participation_rate`

#### Scenario: Custom airbag strike
- **GIVEN** a SnowballOption with `airbag_strike=90` (different from strike=100)
- **AND** spot=50 (below airbag_barrier)
- **WHEN** `get_maturity_payoff_v1(spot=50)` is called
- **THEN** the airbag payoff calculation uses `airbag_strike=90` as the reference strike

#### Scenario: Airbag with reverse snowball
- **GIVEN** a reverse SnowballOption created with `create_airbag_snowball(initial_price=100, strike=100, maturity=1.0, is_reverse=True, ki_barrier=110, airbag_barrier=130, airbag_participation_rate=0.5)`
- **AND** a V1 state (KI triggered, no KO)
- **WHEN** `get_maturity_payoff_v1(spot=140)` is called with spot=140 (above airbag_barrier=130)
- **THEN** the payoff uses `airbag_participation_rate=0.5` instead of standard `participation_rate=1.0`
- **AND** the loss is reduced by 50% compared to reverse snowball without airbag
- **AND** when `get_maturity_payoff_v1(spot=120)` is called with spot=120 (at or below airbag_barrier=130), the payoff uses the standard `participation_rate`

#### Scenario: MC engine prices airbag snowball correctly
- **GIVEN** an airbag snowball and a standard snowball with identical parameters except airbag config
- **WHEN** both are priced with `SnowballMCEngine`
- **THEN** the airbag snowball price is higher (due to reduced downside participation)
