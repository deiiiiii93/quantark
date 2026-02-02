# phoenix-quad-engine Specification

## Purpose
TBD - created by archiving change add-phoenix-engines. Update Purpose after archive.
## Requirements
### Requirement: Regime-Switching Quadrature for Phoenix
The system SHALL implement a Phoenix quadrature engine that propagates value functions (knocked-in and not-knocked-in) within a single backward quadrature recursion, with coupon value accumulation.

#### Scenario: Price standard Phoenix with regime switching
- **GIVEN** a PhoenixOption with discrete KO observations and discrete KI monitoring
- **WHEN** `PhoenixQuadEngine.price(phoenix, pricing_env)` is called
- **THEN** the engine returns a price derived from backward recursion over V_in and V_out

#### Scenario: Price Phoenix with coupons
- **GIVEN** a PhoenixOption with coupon barriers at each observation
- **WHEN** `PhoenixQuadEngine.price(phoenix, pricing_env)` is called
- **THEN** coupon values are added to value surfaces at each observation where barrier is exceeded

### Requirement: Terminal Conditions for V_in and V_out
The system SHALL set terminal conditions using the Phoenix product payoff helpers.

#### Scenario: V_in and V_out initialization at maturity
- **GIVEN** a PhoenixOption and maturity time
- **WHEN** the recursion initializes terminal values
- **THEN** `V_in(S, T)` equals `get_maturity_payoff_v1(S)` and `V_out(S, T)` equals `get_maturity_payoff_v0(S)`

#### Scenario: Terminal coupon if barrier hit
- **GIVEN** a PhoenixOption at maturity
- **WHEN** spot is above coupon barrier at maturity
- **THEN** final coupon is included in terminal value

### Requirement: Discrete KO Observations
The system SHALL apply KO barrier checks only at discrete KO observation times.

#### Scenario: KO jump at observation
- **GIVEN** a PhoenixOption with KO barriers at observation dates
- **WHEN** the recursion steps over a KO observation date
- **THEN** both `V_in` and `V_out` are set to the KO payoff for `S` beyond the KO barrier

#### Scenario: KO payoff includes accumulated coupons
- **GIVEN** a PhoenixOption with coupons already paid
- **WHEN** KO is triggered
- **THEN** KO payoff includes principal plus any KO coupon (per coupon_pay_type)

### Requirement: Coupon Barrier Application at Observations
The system SHALL apply coupon payments at each observation where spot exceeds coupon barrier.

#### Scenario: Coupon added at observation
- **GIVEN** a PhoenixOption with coupon barrier at 85%
- **WHEN** recursion steps over observation with S > coupon_barrier
- **THEN** coupon value is added to both V_in and V_out for S > coupon_barrier

#### Scenario: Step-down coupon barriers
- **GIVEN** a PhoenixOption with different coupon barriers per observation
- **WHEN** recursion processes each observation
- **THEN** the correct coupon barrier for that observation is used

#### Scenario: Memory coupon accumulation
- **GIVEN** a PhoenixOption with `memory_coupon=True`
- **WHEN** coupon barrier is exceeded after previous misses
- **THEN** accumulated coupon value (missed + current periods) is added

### Requirement: Discrete KI Observations
The system SHALL apply discrete KI monitoring only at KI observation times.

#### Scenario: Discrete KI transition
- **GIVEN** a PhoenixOption with KI observation dates
- **WHEN** the recursion steps over a KI observation date
- **THEN** `V_out(S, t)` is set to `V_in(S, t)` for `S` beyond the KI barrier

### Requirement: Continuous KI via Brownian Bridge
The system SHALL support continuous KI monitoring by applying Brownian-bridge transition probabilities.

#### Scenario: Continuous KI transition
- **GIVEN** a PhoenixOption with `ki_continuous=True`
- **WHEN** the recursion steps over a time interval
- **THEN** `V_out` mixes into `V_in` using Brownian-bridge hit probabilities

### Requirement: KO Precedence at Coincident Observations
The system SHALL prioritize KO over KI and coupon when all are triggered at the same observation time.

#### Scenario: KO takes precedence
- **GIVEN** a PhoenixOption with KO, KI, and coupon barriers all breached at same observation
- **WHEN** the recursion applies observation logic
- **THEN** both `V_in` and `V_out` take the KO payoff (includes any KO coupon)

### Requirement: Validation for Supported Configurations
The system SHALL validate Phoenix configurations for quadrature compatibility.

#### Scenario: Supported standard Phoenix
- **GIVEN** a PhoenixOption with standard coupon configuration
- **WHEN** `PhoenixQuadEngine.price(...)` is called
- **THEN** the engine accepts the configuration without error

#### Scenario: Supported airbag Phoenix
- **GIVEN** a PhoenixOption with airbag features
- **WHEN** `PhoenixQuadEngine.price(...)` is called
- **THEN** the engine accepts the configuration and applies airbag payoff

#### Scenario: disable_ko_after_ki supported
- **GIVEN** a PhoenixOption with `disable_ko_after_ki=True`
- **WHEN** `PhoenixQuadEngine.price(...)` is called
- **THEN** the engine applies KO suppression after KI

### Requirement: Event Statistics Support
The system SHALL provide per-observation event statistics for Phoenix options.

#### Scenario: Per-observation probabilities
- **WHEN** `PhoenixQuadEngine.calculate_event_stats(phoenix, pricing_env)` is called
- **THEN** the result includes KO, KI, and coupon probabilities for each observation

#### Scenario: Expected cashflow decomposition
- **WHEN** `PhoenixQuadEngine.calculate_event_stats(phoenix, pricing_env)` is called
- **THEN** the result includes expected discounted cashflow breakdown

### Requirement: Numerical Consistency
The system SHALL produce prices consistent with PhoenixMCEngine and PhoenixPDESolver.

#### Scenario: QUAD vs MC consistency
- **GIVEN** a standard Phoenix configuration
- **WHEN** priced with both PhoenixQuadEngine and PhoenixMCEngine (100k paths)
- **THEN** prices agree within 0.5% relative error

#### Scenario: QUAD vs PDE consistency
- **GIVEN** a standard Phoenix configuration
- **WHEN** priced with both PhoenixQuadEngine and PhoenixPDESolver
- **THEN** prices agree within 0.5% relative error

