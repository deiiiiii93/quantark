# convertible-bond-pde-engine Specification Delta

## ADDED Requirements

### Requirement: Time-Dependent Parameters in PDE Solution
The system SHALL query interest rates and volatility at each time step during PDE backward induction, rather than using single values at maturity.

#### Scenario: Forward rate per time step
- **GIVEN** a `ConvertibleBondJumpDiffusionEngine` or `ConvertibleBondTFEngine` pricing a convertible bond
- **WHEN** the PDE time-stepping loop processes time step from $t$ to $t + \Delta t$
- **THEN** the engine uses `rate_curve.get_forward_rate(t, t + dt)` for the local interest rate

#### Scenario: Local volatility per time step
- **GIVEN** a PDE engine pricing a convertible bond
- **WHEN** the PDE time-stepping loop processes time $t$
- **THEN** the engine derives a per-step effective volatility `sigma_step(t, t+dt)` from implied vols via `pricing_env.get_vol(strike, time_to_maturity)` using total variance differences

#### Scenario: Non-flat rate curve produces different price
- **GIVEN** a convertible bond with 4-year maturity
- **AND** a stepped rate curve: 1% for years 0-2, 9% for years 2-4
- **WHEN** pricing with a PDE engine
- **THEN** the price differs from a flat 5% curve by more than 0.1% of face value

#### Scenario: Matrices built with time-local parameters
- **GIVEN** a PDE engine with Crank-Nicolson or implicit Euler scheme
- **WHEN** `_build_matrices()` is called for time step $n$
- **THEN** the method receives the forward rate and local volatility for that specific time step
