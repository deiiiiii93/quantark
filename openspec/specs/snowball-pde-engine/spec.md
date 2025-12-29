# snowball-pde-engine Specification

## Purpose
TBD - created by archiving change add-snowball-pde-engine. Update Purpose after archive.
## Requirements
### Requirement: Two-Surface PDE Solver for Snowball Options

The system SHALL provide a `SnowballPDESolver` class that implements the Two-Surface PDE method for pricing Snowball options, maintaining separate value grids for knocked-in (V1) and not-knocked-in (V0) states.

#### Scenario: Price standard snowball option

- **GIVEN** a SnowballOption with discrete KO observations and continuous KI monitoring
- **WHEN** `SnowballPDESolver.price(snowball, pricing_env)` is called
- **THEN** the solver returns a price within 0.5% of SnowballMCEngine with 100k paths

#### Scenario: Price reverse snowball option

- **GIVEN** a SnowballOption with `is_reverse=True`
- **WHEN** `SnowballPDESolver.price(snowball, pricing_env)` is called
- **THEN** the solver correctly handles UP KI barrier and DOWN KO barrier directions

#### Scenario: Handle already knocked-in state

- **GIVEN** a SnowballOption where current spot is below KI barrier
- **WHEN** `SnowballPDESolver.price(snowball, pricing_env)` is called
- **THEN** the solver returns the V1 surface value at current spot

#### Scenario: Handle already knocked-out state

- **GIVEN** a SnowballOption where current spot is above KO barrier at first observation
- **WHEN** `SnowballPDESolver.price(snowball, pricing_env)` is called
- **THEN** the solver returns the discounted KO payoff

### Requirement: Terminal Condition Handling

The system SHALL set correct terminal conditions for both V0 (not knocked-in) and V1 (knocked-in) surfaces at maturity.

#### Scenario: V0 terminal with fixed rebate

- **GIVEN** a SnowballOption with `call_rebate_enabled=False` and `rebate_rate=0.10`
- **WHEN** terminal conditions are set at maturity
- **THEN** V0 surface has payoff `principal + rebate_rate * notional * accrual_factor`

#### Scenario: V0 terminal with call rebate

- **GIVEN** a SnowballOption with `call_rebate_enabled=True` and `call_strike=105`
- **WHEN** terminal conditions are set at maturity
- **THEN** V0 surface has payoff `principal + call_participation * max(S - call_strike, 0) * N / S0`

#### Scenario: V1 terminal with standard participation

- **GIVEN** a SnowballOption with `participation_rate=1.0` and `protection_type=NONE`
- **WHEN** terminal conditions are set at maturity
- **THEN** V1 surface has payoff `principal + participation * min((S - K) / S0, 0) * N`

#### Scenario: V1 terminal with protection floor

- **GIVEN** a SnowballOption with `protection_type=PARTIAL` and `protection_rate=0.5`
- **WHEN** terminal conditions are set at maturity
- **THEN** V1 surface payoff is floored at `principal - protection_rate * notional`

#### Scenario: V1 terminal with airbag structure

- **GIVEN** a SnowballOption with `airbag_barrier=60` and `airbag_participation_rate=0.5`
- **WHEN** terminal conditions are set at maturity for spot < airbag_barrier
- **THEN** V1 surface uses reduced airbag_participation_rate instead of standard participation

### Requirement: Discrete KO Observation Handling

The system SHALL apply KO barrier checks and payoffs only at discrete observation times, using time-varying barriers and rates from the resolved schedule.

#### Scenario: KO triggered at observation date

- **GIVEN** a SnowballOption with KO barriers [103, 102, 101] at times [0.25, 0.5, 0.75]
- **WHEN** spot exceeds barrier at observation time
- **THEN** both V0 and V1 surfaces jump to the KO payoff value at that observation

#### Scenario: KO not triggered between observations

- **GIVEN** spot exceeds KO barrier at time 0.3 (between observations at 0.25 and 0.5)
- **WHEN** PDE steps backward through time 0.3
- **THEN** no KO jump is applied (discrete monitoring only)

#### Scenario: INSTANT coupon payment

- **GIVEN** a SnowballOption with `coupon_pay_type=CouponPayType.INSTANT`
- **WHEN** KO is triggered at time t with settlement_time=t
- **THEN** KO payoff is valued at current time (no additional discounting)

#### Scenario: EXPIRY coupon payment

- **GIVEN** a SnowballOption with `coupon_pay_type=CouponPayType.EXPIRY`
- **WHEN** KO is triggered at time t with settlement_time=T (maturity)
- **THEN** KO payoff is discounted from maturity to current time

### Requirement: KI Barrier Handling

The system SHALL support both discrete and continuous KI monitoring, applying the V0 → V1 jump when the KI barrier is breached.

#### Scenario: Continuous KI monitoring

- **GIVEN** a SnowballOption with `ki_continuous=True` and `ki_barrier=75`
- **WHEN** PDE steps backward through any time step
- **THEN** V0 is set to V1 for all grid points where S ≤ ki_barrier

#### Scenario: Discrete KI monitoring

- **GIVEN** a SnowballOption with discrete KI observation dates [0.5, 1.0]
- **WHEN** PDE steps backward through time 0.5
- **THEN** V0 is set to V1 for grid points where S ≤ ki_barrier only at that time

#### Scenario: KI does not disable KO (default)

- **GIVEN** a SnowballOption with `disable_ko_after_ki=False`
- **WHEN** both KI and KO are triggered at the same observation
- **THEN** KO takes precedence and both surfaces receive KO payoff

### Requirement: Grid Construction

The system SHALL construct spatial and temporal grids optimized for snowball barrier structures.

#### Scenario: Spatial grid includes barrier levels

- **GIVEN** a SnowballOption with KI barrier at 75 and KO barriers at [103, 102, 101]
- **WHEN** spatial grid is constructed
- **THEN** grid includes nodes at or near 75, 103, 102, 101, strike, and spot

#### Scenario: Time grid aligns with observations

- **GIVEN** a SnowballOption with KO observations at monthly intervals
- **WHEN** time grid is constructed
- **THEN** all observation times are exact grid points (no interpolation needed)

#### Scenario: Rannacher smoothing after observations

- **GIVEN** default PDEParams (or explicit `rannacher_steps > 0`)
- **WHEN** stepping backward from an observation event
- **THEN** 2-4 implicit Euler steps are applied for smoothing before resuming Crank-Nicolson

### Requirement: PDEEngine Integration

The system SHALL extend the unified `PDEEngine` to dispatch SnowballOption pricing to SnowballPDESolver.

#### Scenario: Snowball pricing via PDEEngine

- **GIVEN** a SnowballOption product
- **WHEN** `PDEEngine.price(snowball, pricing_env)` is called
- **THEN** the engine delegates to `SnowballPDESolver` and returns the option price

#### Scenario: Unsupported snowball configuration

- **GIVEN** a SnowballOption with non-scalar KI barrier and continuous KI monitoring
- **WHEN** `PDEEngine.price(snowball, pricing_env)` is called
- **THEN** a `ValidationError` is raised with message about continuous KI requiring scalar barrier

### Requirement: Greeks Calculation Integration

The system SHALL enable `GreeksCalculator.calculate_numerical_greeks()` to work with SnowballPDESolver for all supported configurations.

#### Scenario: Delta calculation via PDE

- **GIVEN** a SnowballOption and `PDEEngine`
- **WHEN** `GreeksCalculator.calculate_numerical_greeks()` is called
- **THEN** delta is computed by bumping spot and repricing via PDE

#### Scenario: Vega calculation via PDE

- **GIVEN** a SnowballOption and `PDEEngine`
- **WHEN** `GreeksCalculator.calculate_numerical_greeks()` is called
- **THEN** vega is computed by bumping volatility and repricing via PDE

#### Scenario: Gamma smoothness

- **GIVEN** a SnowballOption priced via PDE with default grid settings
- **WHEN** gamma is computed via second-order finite differences
- **THEN** gamma values are smooth (no oscillations near barriers)

### Requirement: Error Handling and Validation

The system SHALL validate snowball configurations for PDE compatibility and provide clear error messages.

#### Scenario: Reject non-scalar KI with continuous monitoring

- **WHEN** SnowballPDESolver receives a SnowballOption with `ki_barrier=[75, 74, 73]` and `ki_continuous=True`
- **THEN** a `ValidationError` is raised with message "Continuous KI monitoring requires scalar ki_barrier"

#### Scenario: Reject expired product

- **WHEN** SnowballPDESolver receives a SnowballOption with maturity ≤ 0
- **THEN** a `ValidationError` is raised or intrinsic value is returned

#### Scenario: Missing pricing environment

- **WHEN** `SnowballPDESolver.price(snowball, None)` is called
- **THEN** a `ValidationError` is raised indicating pricing environment is required

### Requirement: Numerical Consistency with Monte Carlo

The system SHALL produce prices that agree with SnowballMCEngine within acceptable tolerance for equivalent configurations.

#### Scenario: Standard snowball price consistency

- **GIVEN** a standard snowball with monthly KO, continuous KI
- **WHEN** priced with both SnowballPDESolver and SnowballMCEngine (100k paths)
- **THEN** prices agree within 0.5% relative error

#### Scenario: Step-down snowball price consistency

- **GIVEN** a step-down snowball with decreasing KO barriers
- **WHEN** priced with both SnowballPDESolver and SnowballMCEngine (100k paths)
- **THEN** prices agree within 0.5% relative error

#### Scenario: Airbag snowball price consistency

- **GIVEN** an airbag snowball with reduced participation below airbag barrier
- **WHEN** priced with both SnowballPDESolver and SnowballMCEngine (100k paths)
- **THEN** prices agree within 0.5% relative error

