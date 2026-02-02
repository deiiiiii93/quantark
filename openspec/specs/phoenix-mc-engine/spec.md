# phoenix-mc-engine Specification

## Purpose
TBD - created by archiving change add-phoenix-engines. Update Purpose after archive.
## Requirements
### Requirement: Monte Carlo Pricing for Phoenix Options
The system SHALL provide Monte Carlo pricing for Phoenix options using the `PhoenixMCEngine` class, supporting periodic coupon payments at each observation where the coupon barrier is hit.

#### Scenario: Price standard Phoenix with monthly coupons
- **GIVEN** a PhoenixOption with monthly observations and coupon barrier at 85%
- **WHEN** `PhoenixMCEngine.price(phoenix, pricing_env)` is called
- **THEN** the engine returns a price including expected coupon cashflows
- **AND** the price converges as num_paths increases

#### Scenario: Price Phoenix with memory coupon
- **GIVEN** a PhoenixOption with `memory_coupon=True`
- **WHEN** coupon barrier is missed at early observations but hit later
- **THEN** accumulated missed coupons are paid when the barrier is eventually hit

#### Scenario: Price Phoenix with step-down coupon barriers
- **GIVEN** a PhoenixOption with decreasing coupon barriers [85, 84, 83, ...]
- **WHEN** `PhoenixMCEngine.price(phoenix, pricing_env)` is called
- **THEN** the engine correctly applies different coupon barriers at each observation

### Requirement: Coupon Barrier Trigger Logic
The system SHALL check coupon barriers at each observation date and pay coupons when spot exceeds the coupon barrier.

#### Scenario: Coupon paid when barrier exceeded
- **GIVEN** spot is above coupon barrier at observation date
- **WHEN** the engine evaluates the observation
- **THEN** the coupon payment is added to the path cashflow

#### Scenario: Coupon not paid when barrier missed
- **GIVEN** spot is below coupon barrier at observation date
- **WHEN** the engine evaluates the observation
- **THEN** no coupon is paid (unless memory coupon applies)

#### Scenario: Memory coupon accumulation
- **GIVEN** a PhoenixOption with `memory_coupon=True` and missed coupons at t1, t2
- **WHEN** coupon barrier is hit at t3
- **THEN** all accumulated coupons (t1 + t2 + t3) are paid at t3

### Requirement: Day Count Convention Support
The system SHALL calculate coupon time fractions using the configured day count convention.

#### Scenario: ACT/365 convention
- **GIVEN** a PhoenixOption with `day_count_convention=DayCountConvention.ACT_365`
- **WHEN** coupon accrual is calculated
- **THEN** time fractions use actual days / 365

#### Scenario: 30/360 convention
- **GIVEN** a PhoenixOption with `day_count_convention=DayCountConvention.D30_360`
- **WHEN** coupon accrual is calculated
- **THEN** time fractions use 30/360 day count method

### Requirement: Coupon Payment Timing
The system SHALL support INSTANT and EXPIRY coupon payment timing with correct discounting.

#### Scenario: INSTANT coupon payment
- **GIVEN** a PhoenixOption with `coupon_pay_type=CouponPayType.INSTANT`
- **WHEN** coupon is triggered at time t
- **THEN** coupon is discounted from time t to valuation date

#### Scenario: EXPIRY coupon payment
- **GIVEN** a PhoenixOption with `coupon_pay_type=CouponPayType.EXPIRY`
- **WHEN** coupons are triggered throughout the life
- **THEN** all coupons are accumulated and paid at maturity (discounted from maturity)

### Requirement: KO and KI Barrier Handling
The system SHALL apply KO and KI barriers following the same logic as Snowball options.

#### Scenario: KO triggered before maturity
- **GIVEN** a PhoenixOption with KO barrier at 103%
- **WHEN** spot exceeds KO barrier at observation date
- **THEN** the option terminates with KO payoff plus any accumulated coupons

#### Scenario: KI triggered with continuous monitoring
- **GIVEN** a PhoenixOption with `ki_continuous=True`
- **WHEN** spot touches KI barrier at any time
- **THEN** the not-knocked-in state transitions to knocked-in state

#### Scenario: Coupon still paid after KI
- **GIVEN** a PhoenixOption in knocked-in state
- **WHEN** spot is above coupon barrier at observation
- **THEN** coupon is still paid (KI only affects maturity payoff)

### Requirement: Event Statistics API
The system SHALL provide per-observation event statistics for Phoenix options.

#### Scenario: Per-observation coupon probability
- **WHEN** `PhoenixMCEngine.calculate_event_stats(phoenix, pricing_env)` is called
- **THEN** the result includes coupon trigger probability for each observation

#### Scenario: Expected cashflow breakdown
- **WHEN** `PhoenixMCEngine.calculate_event_stats(phoenix, pricing_env)` is called
- **THEN** the result includes expected discounted cashflow from coupons, KO, and maturity

### Requirement: Two-Level Enum Method Selection
The system SHALL support the two-level enum pattern for method selection consistent with other QuantArk engines.

#### Scenario: Initialize with two-level enum
- **WHEN** engine is initialized with `EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)`
- **THEN** the engine extracts the method and configures QMC path generation

#### Scenario: Initialize with method enum directly
- **WHEN** engine is initialized with `MonteCarloMethod.QUASI` directly
- **THEN** the engine accepts it and configures QMC path generation

### Requirement: Consistency with Phoenix Product Methods
The system SHALL delegate coupon calculations to Phoenix product methods for consistency.

#### Scenario: Use product coupon check
- **WHEN** checking coupon triggers
- **THEN** the engine uses `product.is_coupon_triggered(spot, obs_index)`

#### Scenario: Use product coupon payoff
- **WHEN** calculating coupon value
- **THEN** the engine uses `product.get_coupon_payoff(obs_index, accumulated_periods)`

### Requirement: Standard Error Estimation
The system SHALL provide standard error estimates for MC convergence diagnostics.

#### Scenario: Standard error for normal MC
- **WHEN** normal MC is run with N paths
- **THEN** standard error is estimated as `std(payoffs) / sqrt(N)`

#### Scenario: Standard error with coupon variance
- **WHEN** Phoenix with many coupons is priced
- **THEN** standard error reflects variance from both coupon and terminal payoffs

### Requirement: Numerical Consistency
The system SHALL produce prices consistent with SnowballMCEngine for equivalent structures.

#### Scenario: Phoenix without coupons matches Snowball
- **GIVEN** a PhoenixOption with coupon_barrier set very high (never triggered)
- **WHEN** priced with PhoenixMCEngine
- **THEN** price matches SnowballMCEngine within standard error

#### Scenario: Factory helper configurations
- **GIVEN** a Phoenix created with `create_standard_phoenix()`
- **WHEN** priced with PhoenixMCEngine
- **THEN** pricing completes without error

