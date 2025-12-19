# Convertible Bond Facade Engine - Risk Metrics Delta

## ADDED Requirements

### Requirement: Floor Bond Price Calculation
The system SHALL provide a method to calculate the floor bond (straight bond) price for a convertible bond.

#### Scenario: Floor bond price calculation
- **GIVEN** a `ConvertibleBond` with coupon schedule, maturity, and credit spread
- **WHEN** `ConvertibleBondEngine.floor_bond_price(bond)` is called
- **THEN** the engine returns the present value of all bond cashflows discounted at the risky rate (risk-free + credit spread), ignoring conversion and call/put options

#### Scenario: Floor bond price with zero credit spread
- **GIVEN** a `ConvertibleBond` with `credit_spread=0.0`
- **WHEN** `floor_bond_price(bond)` is called
- **THEN** the floor bond is discounted using only the risk-free rate

#### Scenario: Floor bond price as lower bound
- **GIVEN** a `ConvertibleBond` with embedded options
- **WHEN** both `price(bond)` and `floor_bond_price(bond)` are called
- **THEN** `price >= floor_bond_price` (convertible is worth at least its floor value)

### Requirement: Floor Bond Risk Metrics
The system SHALL provide DV01, CS01, modified duration, and convexity calculations for the floor bond.

#### Scenario: Floor bond DV01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_dv01(bond)` is called
- **THEN** the engine returns the floor bond's price change per basis point rate move

#### Scenario: Floor bond CS01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_cs01(bond)` is called
- **THEN** the engine returns the floor bond's price change per basis point credit spread move

#### Scenario: Floor bond CS01 equals DV01
- **GIVEN** a `ConvertibleBond` with credit spread
- **WHEN** both `floor_bond_dv01(bond)` and `floor_bond_cs01(bond)` are called
- **THEN** the values are equal (since floor bond discounts at r+s, both have identical effect)

#### Scenario: Floor bond duration calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_duration(bond)` is called
- **THEN** the engine returns the floor bond's modified duration (weighted average time of cashflows)

#### Scenario: Floor bond convexity calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.floor_bond_convexity(bond)` is called
- **THEN** the engine returns the floor bond's convexity (weighted average time squared of cashflows)

#### Scenario: Floor bond metrics consistency
- **GIVEN** floor bond price, DV01, and duration
- **WHEN** metrics are computed
- **THEN** `floor_bond_dv01 ≈ floor_bond_duration × floor_bond_price × 0.0001` (within numerical tolerance)

### Requirement: Convertible Bond Interest Rate and Credit Risk Metrics
The system SHALL provide numerical DV01, CS01, duration, and convexity calculations for the full convertible bond.

#### Scenario: Convertible DV01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.dv01(bond)` is called
- **THEN** the engine returns the convertible's price change per basis point rate move, computed via numerical rate bumping

#### Scenario: Convertible CS01 calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.cs01(bond)` is called
- **THEN** the engine returns the convertible's price change per basis point credit spread move, computed via numerical spread bumping

#### Scenario: Convertible duration calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.modified_duration(bond)` is called
- **THEN** the engine returns the convertible's modified duration derived from DV01

#### Scenario: Convertible convexity calculation
- **GIVEN** a `ConvertibleBond`
- **WHEN** `ConvertibleBondEngine.convexity(bond)` is called
- **THEN** the engine returns the convertible's convexity via central difference rate bumping

#### Scenario: Interest rate bump isolation
- **WHEN** computing convertible DV01
- **THEN** only the risk-free rate is bumped, not the credit spread, isolating interest rate risk

#### Scenario: Credit spread bump isolation
- **WHEN** computing convertible CS01
- **THEN** only the credit spread is bumped, not the risk-free rate, isolating credit risk

## MODIFIED Requirements

### Requirement: Results Container
The system SHALL support a structured results container with price and optional additional outputs via `price_with_details()`.

#### Scenario: Extended results with risk metrics
- **WHEN** `price_with_details()` is called
- **THEN** the result includes `floor_bond_price`, `floor_bond_dv01`, `floor_bond_cs01`, `floor_bond_duration`, `floor_bond_convexity`, `dv01`, `cs01`, `modified_duration`, and `convexity` fields

#### Scenario: Risk metrics populated on demand
- **WHEN** `price_with_details(include_risk_metrics=True)` is called (default)
- **THEN** all risk metrics are computed and populated in the result

#### Scenario: Skip risk metrics for performance
- **WHEN** `price_with_details(include_risk_metrics=False)` is called
- **THEN** risk metrics fields are set to 0.0 to avoid computational overhead
