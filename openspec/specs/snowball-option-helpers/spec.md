# snowball-option-helpers Specification

## Purpose
TBD - created by archiving change add-snowball-option-helpers. Update Purpose after archive.
## Requirements
### Requirement: Standard Snowball Helper

The system SHALL provide a `create_standard_snowball()` function that creates a basic snowball with flat KO barrier and continuous KI monitoring.

#### Scenario: Create standard snowball with minimal parameters
Given initial_price=100, strike=100, maturity=1.0
When create_standard_snowball(initial_price, strike, maturity) is called
Then a SnowballOption is returned with:
  - ko_barrier = 103.0 (103% of initial_price)
  - ki_barrier = 75.0 (75% of initial_price)
  - ko_rate = 0.15 (15% annualized)
  - 12 monthly KO observations
  - Continuous KI monitoring
  - notional = 1,000,000
  - is_reverse = False

#### Scenario: Override default parameters
Given initial_price=100, strike=100, maturity=1.0, ko_barrier=105.0
When create_standard_snowball(initial_price, strike, maturity, ko_barrier=105.0) is called
Then a SnowballOption is returned with ko_barrier=105.0

#### Scenario: Override via kwargs
Given initial_price=100, strike=100, maturity=1.0
When create_standard_snowball(initial_price, strike, maturity, include_principal=True) is called
Then a SnowballOption is returned with payoff_config.include_principal=True

---

### Requirement: Step-Down Snowball Helper

The system SHALL provide a `create_stepdown_snowball()` function that creates a snowball where KO barrier decreases each observation period.

#### Scenario: Create step-down snowball with default stepdown
Given initial_price=100, strike=100, maturity=1.0
When create_stepdown_snowball(initial_price, strike, maturity) is called
Then a SnowballOption is returned with:
  - ko_barrier = [103.0, 102.5, 102.0, 101.5, 101.0, 100.5, 100.0, 99.5, 99.0, 98.5, 98.0, 97.5]
  - Barrier decreases by 0.5 (0.5% of initial_price) each period

#### Scenario: Custom stepdown rate
Given initial_price=100, strike=100, maturity=1.0, stepdown_rate=0.01
When create_stepdown_snowball(initial_price, strike, maturity, stepdown_rate=0.01) is called
Then a SnowballOption is returned with barriers decreasing by 1.0 each period

---

### Requirement: European Knock-In Snowball Helper

The system SHALL provide a `create_european_ki_snowball()` function that creates a snowball with KI only observed at maturity.

#### Scenario: Create European KI snowball
Given initial_price=100, strike=100, maturity=1.0
When create_european_ki_snowball(initial_price, strike, maturity) is called
Then a SnowballOption is returned with:
  - ki_observation_type = ObservationType.DISCRETE
  - ki_continuous = False
  - ki_observation_dates = [1.0] (only maturity)
  - ko_observation_dates = [0.083..., 0.167..., ...] (12 monthly)

---

### Requirement: Parachute Snowball Helper

The system SHALL provide a `create_parachute_snowball()` function that creates a snowball where the last KO barrier equals the KI barrier.

#### Scenario: Create parachute snowball with default barriers
Given initial_price=100, strike=100, maturity=1.0
When create_parachute_snowball(initial_price, strike, maturity) is called
Then a SnowballOption is returned with:
  - ko_barrier = [103.0, 103.0, ..., 75.0] (last barrier equals ki_barrier)
  - ki_barrier = 75.0
  - 12 monthly KO observations

#### Scenario: Custom KO and KI barriers
Given initial_price=100, strike=100, maturity=1.0, ko_barrier=105.0, ki_barrier=80.0
When create_parachute_snowball(initial_price, strike, maturity, ko_barrier=105.0, ki_barrier=80.0) is called
Then a SnowballOption is returned with:
  - ko_barrier = [105.0, 105.0, ..., 80.0] (last barrier equals ki_barrier)
  - ki_barrier = 80.0

---

### Requirement: Phoenix Snowball Helper

The system SHALL provide a `create_phoenix_snowball()` function that creates a snowball with periodic coupon payments when spot is above coupon barrier.

#### Scenario: Create phoenix snowball with default coupon barrier
Given initial_price=100, strike=100, maturity=1.0
When create_phoenix_snowball(initial_price, strike, maturity) is called
Then a SnowballOption is returned with:
  - Additional coupon_barrier = 80.0 (80% of initial_price)
  - Periodic coupon_rate = 0.01 (1% per period)

---

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

### Requirement: Observation Date Generator

The system SHALL provide a `generate_ko_observation_dates()` utility that generates evenly spaced observation dates.

#### Scenario: Generate monthly observations
Given maturity=1.0, frequency="monthly"
When generate_ko_observation_dates(maturity, frequency) is called
Then [1/12, 2/12, 3/12, ..., 12/12] is returned

#### Scenario: Generate quarterly observations with lock-out
Given maturity=1.0, frequency="quarterly", skip_first=1
When generate_ko_observation_dates(maturity, frequency, skip_first=1) is called
Then [0.5, 0.75, 1.0] is returned (first quarter skipped)

---

### Requirement: Step-Down Barrier Generator

The system SHALL provide a `generate_stepdown_barriers()` utility that generates decreasing barrier levels.

#### Scenario: Generate step-down barriers
Given initial_barrier=103.0, stepdown_rate=0.5, num_observations=4
When generate_stepdown_barriers(initial_barrier, stepdown_rate, num_observations) is called
Then [103.0, 102.5, 102.0, 101.5] is returned

#### Scenario: Step-down with floor
Given initial_barrier=103.0, stepdown_rate=2.0, num_observations=4, min_barrier=100.0
When generate_stepdown_barriers(initial_barrier, stepdown_rate, num_observations, min_barrier=100.0) is called
Then [103.0, 101.0, 100.0, 100.0] is returned (floor at 100)

---

### Requirement: Input Validation

All helper functions MUST validate input parameters before creating SnowballOption.

#### Scenario: Reject negative initial price
Given initial_price=-100
When any helper function is called with initial_price=-100
Then ValidationError is raised with message containing "initial_price"

#### Scenario: Reject zero maturity
Given maturity=0
When any helper function is called with maturity=0
Then ValidationError is raised with message containing "maturity"

#### Scenario: Reject negative notional
Given notional=-1000
When any helper function is called with notional=-1000
Then ValidationError is raised with message containing "notional"

---

### Requirement: Module Exports

The helper functions SHALL be exported from the option module.

#### Scenario: Import from option module
Given the asset.equity.product.option module
When importing create_standard_snowball
Then the function is accessible as `from asset.equity.product.option import create_standard_snowball`

#### Scenario: Import all helpers
Given the asset.equity.product.option.snowball_helpers module
When importing all helpers
Then all create_* functions and generate_* utilities are available

