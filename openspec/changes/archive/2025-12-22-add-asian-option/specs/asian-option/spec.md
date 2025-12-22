# Asian Option Specification

## ADDED Requirements

### Requirement: Asian Option Product Definition
The system SHALL provide an `AsianOption` class that represents path-dependent options where the payoff depends on the average price of the underlying asset over a specified observation period.

#### Scenario: Create fixed strike Asian call
- **WHEN** user creates an AsianOption with strike=100, option_type=CALL, asian_strike_type=FIXED, averaging_type=ARITHMETIC
- **THEN** the option is created with specified parameters
- **AND** the option inherits from BaseEquityOption

#### Scenario: Create floating strike Asian put
- **WHEN** user creates an AsianOption with option_type=PUT, asian_strike_type=FLOATING, averaging_type=ARITHMETIC
- **THEN** the option is created with floating strike configuration
- **AND** strike parameter may be zero for floating strike options

### Requirement: Averaging Type Support
The system SHALL support both arithmetic and geometric averaging methods for computing the average price.

#### Scenario: Arithmetic averaging
- **WHEN** averaging_type is ARITHMETIC
- **THEN** average is computed as sum(prices) / n

#### Scenario: Geometric averaging
- **WHEN** averaging_type is GEOMETRIC
- **THEN** average is computed as (product(prices))^(1/n)

### Requirement: Asian Strike Type Support
The system SHALL support both fixed strike (average price) and floating strike (average strike) variants.

#### Scenario: Fixed strike payoff (average price option)
- **WHEN** asian_strike_type is FIXED with strike K
- **THEN** call payoff is max(average - K, 0)
- **AND** put payoff is max(K - average, 0)

#### Scenario: Floating strike payoff (average strike option)
- **WHEN** asian_strike_type is FLOATING
- **THEN** call payoff is max(spot_at_maturity - average, 0)
- **AND** put payoff is max(average - spot_at_maturity, 0)

### Requirement: Observation Schedule
The system SHALL support specification of discrete observation times for computing the average.

#### Scenario: Discrete observations specified as year fractions
- **WHEN** observation_times is provided as list of floats [0.25, 0.5, 0.75, 1.0]
- **THEN** the option uses these times to determine when prices are sampled
- **AND** validation ensures all times are non-negative and sorted

#### Scenario: Discrete observations specified as dates
- **WHEN** observation_dates is provided as list of datetime objects
- **THEN** the option uses these dates for price sampling
- **AND** observation_times can be derived from dates given valuation_date

#### Scenario: Default uniform observations
- **WHEN** neither observation_times nor observation_dates is specified
- **THEN** the system generates uniform observations from start to maturity
- **AND** num_observations parameter controls the count (default: 12)

### Requirement: Average Computation
The system SHALL provide a method to compute the average from a price path or list of observed prices.

#### Scenario: Compute arithmetic average from prices
- **GIVEN** an AsianOption with ARITHMETIC averaging
- **WHEN** get_average([100, 105, 110, 108]) is called
- **THEN** returns (100 + 105 + 110 + 108) / 4 = 105.75

#### Scenario: Compute geometric average from prices
- **GIVEN** an AsianOption with GEOMETRIC averaging
- **WHEN** get_average([100, 105, 110, 108]) is called
- **THEN** returns (100 * 105 * 110 * 108)^(1/4)

### Requirement: Payoff Calculation
The system SHALL compute the correct payoff given observed prices and final spot price.

#### Scenario: Fixed strike call payoff
- **GIVEN** AsianOption(strike=100, option_type=CALL, asian_strike_type=FIXED)
- **WHEN** get_payoff(spot=110, observed_prices=[100, 105, 110, 115]) is called
- **THEN** returns max(107.5 - 100, 0) = 7.5

#### Scenario: Floating strike put payoff
- **GIVEN** AsianOption(option_type=PUT, asian_strike_type=FLOATING)
- **WHEN** get_payoff(spot=95, observed_prices=[100, 105, 110, 115]) is called
- **THEN** returns max(107.5 - 95, 0) = 12.5

### Requirement: Validation
The system SHALL validate all Asian option parameters at construction time.

#### Scenario: Invalid strike for fixed strike option
- **WHEN** user creates fixed strike AsianOption with strike <= 0
- **THEN** ValidationError is raised with descriptive message

#### Scenario: Invalid observation times
- **WHEN** observation_times contains negative values or unsorted values
- **THEN** ValidationError is raised with descriptive message

#### Scenario: Empty observation times
- **WHEN** observation_times is provided as empty list
- **THEN** ValidationError is raised with descriptive message
