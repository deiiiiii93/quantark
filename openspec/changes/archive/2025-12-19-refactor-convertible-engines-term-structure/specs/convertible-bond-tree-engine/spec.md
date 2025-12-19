# convertible-bond-tree-engine Specification Delta

## ADDED Requirements

### Requirement: Time-Dependent Parameters in Trinomial Tree
The system SHALL query interest rates and volatility at each time step during trinomial tree backward induction, using maximum volatility for grid stability.

#### Scenario: Maximum volatility for grid spacing
- **GIVEN** a `ConvertibleBondTrinomialEngine` pricing a convertible bond
- **WHEN** the trinomial tree grid is constructed
- **THEN** the engine calculates the maximum volatility over the bond's life and uses it for the vertical grid spacing ($dx$)

#### Scenario: Local forward rate per time step
- **GIVEN** a `ConvertibleBondTrinomialEngine` performing backward induction
- **WHEN** processing time step $i$ corresponding to time $t = i \cdot \Delta t$
- **THEN** the engine uses `rate_curve.get_forward_rate(t, t + dt)` for the local interest rate

#### Scenario: Local volatility per time step
- **GIVEN** a `ConvertibleBondTrinomialEngine` performing backward induction
- **WHEN** processing time step $i$ corresponding to time $t$
- **THEN** the engine derives a per-step effective volatility `sigma_step(t, t+dt)` from implied vols via `pricing_env.get_vol(strike, time_to_maturity)` using total variance differences

#### Scenario: Transition probabilities recalculated per step
- **GIVEN** a trinomial tree with time-varying parameters
- **WHEN** backward induction processes each time step
- **THEN** the transition probabilities $(p_u, p_m, p_d)$ are recalculated using local rate and volatility

#### Scenario: Non-flat rate curve produces different price
- **GIVEN** a convertible bond with 4-year maturity
- **AND** a stepped rate curve: 1% for years 0-2, 9% for years 2-4
- **WHEN** pricing with `ConvertibleBondTrinomialEngine`
- **THEN** the price differs from a flat 5% curve by more than 0.1% of face value

### Requirement: Binomial Engine Non-Flat Curve Warning
The system SHALL warn users when the binomial engine is used with non-flat rate curves or volatility surfaces.

#### Scenario: Warning logged for non-flat rate curve
- **GIVEN** a `ConvertibleBondBinomialEngine`
- **AND** a `PricingEnvironment` with `InterpolatedRateCurve` (non-flat)
- **WHEN** `price()` or `price_with_details()` is called
- **THEN** a warning is logged: "Binomial GS engine approximates piecewise curves using a flat rate/vol to maturity. Use PDE or Trinomial engines for better accuracy."

#### Scenario: Warning logged for non-flat volatility surface
- **GIVEN** a `ConvertibleBondBinomialEngine`
- **AND** a `PricingEnvironment` with a non-`FlatVolSurface` volatility surface
- **WHEN** `price()` or `price_with_details()` is called
- **THEN** a warning is logged recommending PDE or Trinomial engines

#### Scenario: No warning for flat curves
- **GIVEN** a `ConvertibleBondBinomialEngine`
- **AND** a `PricingEnvironment` with `FlatRateCurve` and `FlatVolSurface`
- **WHEN** `price()` is called
- **THEN** no warning about curve approximation is logged

#### Scenario: Binomial engine unchanged mathematically
- **GIVEN** a `ConvertibleBondBinomialEngine` with flat curves
- **WHEN** `price()` is called before and after this change
- **THEN** the prices are identical (no mathematical changes)
