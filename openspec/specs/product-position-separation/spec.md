# product-position-separation Specification

## Purpose
TBD - created by archiving change refactor-product-position-separation. Update Purpose after archive.
## Requirements
### Requirement: Product Unit Representation

The system SHALL ensure all product classes represent exactly one unit of an instrument, with no embedded position sizing (notional or position quantity) attributes.

#### Scenario: Equity option product has no notional or quantity
- **GIVEN** any equity option product (EuropeanVanillaOption, SnowballOption, etc.)
- **WHEN** the product is inspected
- **THEN** it has no `notional` attribute
- **AND** it has no `quantity` attribute

#### Scenario: Equity option product has a contract multiplier
- **GIVEN** any equity option product
- **WHEN** the product is created without specifying contract_multiplier
- **THEN** `product.contract_multiplier` defaults to 1.0
- **AND** this represents the number of underlying units per contract

#### Scenario: Bond product has denominator attribute
- **GIVEN** a FixedBond product
- **WHEN** the product is created with denominator=1000
- **THEN** `product.get_denominator()` returns 1000.0
- **AND** this represents the minimum tradable notional

#### Scenario: Snowball product has no notional parameter
- **GIVEN** a SnowballOption is being created
- **WHEN** instantiated with initial_price=100, strike=100, maturity=1.0
- **THEN** the product is created successfully
- **AND** no notional parameter is accepted
- **AND** payoff calculations return per-contract values (scaled by initial_price and contract_multiplier)

---

### Requirement: Position as Source of Truth

The system SHALL ensure position classes are the single source of truth for quantity, with products representing instrument specifications only.

#### Scenario: Equity position quantity determines total value
- **GIVEN** an EquityPosition with product=option, quantity=100
- **AND** engine.price(product, env) returns 5.0 (per-contract)
- **WHEN** position.get_market_value(env) is called
- **THEN** the result is 5.0 * 100 = 500.0

#### Scenario: FI position quantity times denominator equals actual notional
- **GIVEN** an FIPosition with product=bond (denominator=1000), quantity=100
- **WHEN** position.get_actual_notional() is called
- **THEN** the result is 100 * 1000 = 100,000

#### Scenario: Position Greeks are scaled by quantity
- **GIVEN** an EquityPosition with quantity=10
- **AND** product delta = 0.5 (per-contract)
- **WHEN** position.get_greeks(env) is called
- **THEN** the returned delta is 0.5 * 10 = 5.0

---

### Requirement: Per-Unit Engine Pricing

The system SHALL ensure all pricing engines return per-contract prices, with position layer responsible for quantity scaling.

#### Scenario: BlackScholesEngine returns per-contract price
- **GIVEN** a EuropeanVanillaOption with strike=100, maturity=1.0, contract_multiplier=1.0
- **AND** a BlackScholesEngine
- **WHEN** engine.price(option, env) is called
- **THEN** the price is per-contract (not scaled by any position quantity)

#### Scenario: SnowballMCEngine returns per-contract payoff
- **GIVEN** a SnowballOption with initial_price=100, contract_multiplier=1.0, ko_rate=0.15
- **AND** KO triggered with accrual_fraction=0.5
- **WHEN** the payoff is calculated
- **THEN** payoff = 100 * 1.0 * 0.15 * 0.5 = 7.5 (per-contract)
- **NOT** payoff = position_notional * 0.15 * 0.5

#### Scenario: Contract multiplier scales per-contract payoff
- **GIVEN** a EuropeanVanillaOption with contract_multiplier=100.0
- **WHEN** intrinsic value is computed for spot=120, strike=100
- **THEN** intrinsic value equals 100.0 * (120 - 100) = 2000.0 (per-contract)

---

### Requirement: Bond Denominator Pattern

The system SHALL use the denominator pattern for fixed income products, where denominator is the minimum tradable notional and actual notional = quantity × denominator.

#### Scenario: FixedBond uses denominator
- **GIVEN** a FixedBond with denominator=1000, coupon_rate=0.05
- **AND** a FIPosition with quantity=100
- **WHEN** position.get_actual_notional() is called
- **THEN** the result is 100,000
- **AND** coupon payments are calculated based on this actual notional

#### Scenario: IRS uses time-based denominator
- **GIVEN** an InterestRateSwap with initial notional schedule
- **WHEN** swap.get_denominator(as_of_date) is called
- **THEN** the denominator for that date is returned
- **AND** position.quantity × denominator = actual notional

---

### Requirement: Removed NotionalQuantityPolicy

The system SHALL NOT have a NotionalQuantityPolicy enum or any reconciliation logic between notional and quantity at the product level.

#### Scenario: No reconciliation policy exists
- **GIVEN** the util.enum module
- **WHEN** enums are listed
- **THEN** NotionalQuantityPolicy is not present
- **AND** no product class has a reconciliation_policy parameter

#### Scenario: Product validation does not check notional-quantity consistency
- **GIVEN** any product class
- **WHEN** validate() is called
- **THEN** no notional-quantity reconciliation is performed
- **AND** no NotionalQuantityPolicy is consulted

