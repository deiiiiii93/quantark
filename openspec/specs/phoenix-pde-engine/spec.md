# phoenix-pde-engine Specification

## Purpose
TBD - created by archiving change add-phoenix-engines. Update Purpose after archive.
## Requirements
### Requirement: Two-Surface PDE Solver for Phoenix Options
The system SHALL provide a `PhoenixPDESolver` class that implements the Two-Surface PDE method for pricing Phoenix options, maintaining separate value grids for knocked-in (V1) and not-knocked-in (V0) states with coupon tracking.

#### Scenario: Price standard Phoenix option
- **GIVEN** a PhoenixOption with discrete KO observations and continuous KI monitoring
- **WHEN** `PhoenixPDESolver.price(phoenix, pricing_env)` is called
- **THEN** the solver returns a price within 1% of PhoenixMCEngine with 100k paths

#### Scenario: Price reverse Phoenix option
- **GIVEN** a PhoenixOption with `is_reverse=True`
- **WHEN** `PhoenixPDESolver.price(phoenix, pricing_env)` is called
- **THEN** the solver correctly handles UP KI barrier and DOWN KO barrier directions

#### Scenario: Handle already knocked-in state
- **GIVEN** a PhoenixOption where current spot is below KI barrier
- **WHEN** `PhoenixPDESolver.price(phoenix, pricing_env)` is called
- **THEN** the solver returns the V1 surface value at current spot

### Requirement: Terminal Condition Handling
The system SHALL set correct terminal conditions for both V0 (not knocked-in) and V1 (knocked-in) surfaces at maturity.

#### Scenario: V0 terminal with accumulated coupons
- **GIVEN** a PhoenixOption that has not knocked out or knocked in
- **WHEN** terminal conditions are set at maturity
- **THEN** V0 surface includes principal plus any final coupon if barrier is hit

#### Scenario: V1 terminal with standard participation
- **GIVEN** a PhoenixOption with `participation_rate=1.0` and `protection_type=NONE`
- **WHEN** terminal conditions are set at maturity
- **THEN** V1 surface has payoff matching `product.get_maturity_payoff_v1()`

#### Scenario: V1 terminal with airbag structure
- **GIVEN** a PhoenixOption with `airbag_barrier` and `airbag_participation_rate`
- **WHEN** terminal conditions are set at maturity for spot < airbag_barrier
- **THEN** V1 surface uses reduced airbag_participation_rate

### Requirement: Discrete KO Observation Handling
The system SHALL apply KO barrier checks and payoffs only at discrete observation times.

#### Scenario: KO triggered at observation date
- **GIVEN** a PhoenixOption with KO barriers
- **WHEN** spot exceeds barrier at observation time
- **THEN** both V0 and V1 surfaces jump to the KO payoff value

#### Scenario: KO not triggered between observations
- **GIVEN** spot exceeds KO barrier between observations
- **WHEN** PDE steps backward through that time
- **THEN** no KO jump is applied (discrete monitoring only)

### Requirement: Coupon Barrier Jump Application
The system SHALL apply coupon payments at each observation where the coupon barrier is exceeded.

#### Scenario: Coupon added to grid at observation
- **GIVEN** a PhoenixOption with coupon barrier at 85%
- **WHEN** PDE steps backward over an observation where S > coupon barrier
- **THEN** the coupon value is added to both V0 and V1 surfaces for S > coupon barrier

#### Scenario: Step-down coupon barriers
- **GIVEN** a PhoenixOption with decreasing coupon barriers
- **WHEN** PDE processes observations
- **THEN** each observation uses its own coupon barrier from the schedule

#### Scenario: Memory coupon in PDE
- **GIVEN** a PhoenixOption with `memory_coupon=True`
- **WHEN** coupon barrier is exceeded after previous misses
- **THEN** accumulated coupon value is added to grid (encoding missed periods)

### Requirement: KI Barrier Handling
The system SHALL support both discrete and continuous KI monitoring.

#### Scenario: Continuous KI monitoring
- **GIVEN** a PhoenixOption with `ki_continuous=True` and `ki_barrier=75`
- **WHEN** PDE steps backward through any time step
- **THEN** V0 is set to V1 for all grid points where S ≤ ki_barrier

#### Scenario: Discrete KI monitoring
- **GIVEN** a PhoenixOption with discrete KI observation dates
- **WHEN** PDE steps backward through KI observation time
- **THEN** V0 is set to V1 for grid points where S ≤ ki_barrier only at that time

### Requirement: Grid Construction
The system SHALL construct spatial and temporal grids optimized for Phoenix barrier structures.

#### Scenario: Spatial grid includes coupon barrier levels
- **GIVEN** a PhoenixOption with coupon barriers [85, 84, 83, ...]
- **WHEN** spatial grid is constructed
- **THEN** grid includes nodes at or near all unique coupon barrier levels

#### Scenario: Time grid aligns with observations
- **GIVEN** a PhoenixOption with observations at specified times
- **WHEN** time grid is constructed
- **THEN** all observation times are exact grid points

### Requirement: PDEEngine Integration
The system SHALL extend the unified `PDEEngine` to dispatch PhoenixOption pricing to PhoenixPDESolver.

#### Scenario: Phoenix pricing via PDEEngine
- **GIVEN** a PhoenixOption product
- **WHEN** `PDEEngine.price(phoenix, pricing_env)` is called
- **THEN** the engine delegates to `PhoenixPDESolver` and returns the option price

### Requirement: Greeks Calculation Integration
The system SHALL enable `GreeksCalculator.calculate_numerical_greeks()` to work with PhoenixPDESolver.

#### Scenario: Delta calculation via PDE
- **GIVEN** a PhoenixOption and `PDEEngine`
- **WHEN** `GreeksCalculator.calculate_numerical_greeks()` is called
- **THEN** delta is computed by bumping spot and repricing via PDE

#### Scenario: Analytical Delta from grid
- **GIVEN** a PhoenixOption and `PhoenixPDESolver`
- **WHEN** price is computed
- **THEN** delta can be extracted from grid finite differences without repricing

### Requirement: Error Handling and Validation
The system SHALL validate Phoenix configurations for PDE compatibility.

#### Scenario: Reject non-scalar KI with continuous monitoring
- **WHEN** PhoenixPDESolver receives a PhoenixOption with list ki_barrier and `ki_continuous=True`
- **THEN** a `ValidationError` is raised

#### Scenario: Missing pricing environment
- **WHEN** `PhoenixPDESolver.price(phoenix, None)` is called
- **THEN** a `ValidationError` is raised

### Requirement: Numerical Consistency with Monte Carlo
The system SHALL produce prices that agree with PhoenixMCEngine within acceptable tolerance.

#### Scenario: Standard Phoenix price consistency
- **GIVEN** a standard Phoenix with monthly observations
- **WHEN** priced with both PhoenixPDESolver and PhoenixMCEngine (100k paths)
- **THEN** prices agree within 1% relative error

#### Scenario: Memory coupon price consistency
- **GIVEN** a Phoenix with `memory_coupon=True`
- **WHEN** priced with both PhoenixPDESolver and PhoenixMCEngine
- **THEN** prices agree within 1% relative error

