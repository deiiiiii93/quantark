# convertible-bond-product Specification

## Purpose
Defines the `ConvertibleBond` product class representing convertible bond contracts. A convertible bond is a corporate debt security that gives the holder the right to convert the bond into a predetermined number of shares of the issuer's common stock. The product captures all contract terms including conversion features, call/put provisions, coupon payments, and credit attributes.

## ADDED Requirements

### Requirement: Core Contract Terms
The system SHALL provide a `ConvertibleBond` dataclass that captures essential contract terms including face value, maturity date, coupon rate, coupon frequency, and day count convention.

#### Scenario: Basic convertible bond creation
- **WHEN** a `ConvertibleBond` is created with face_value=100, maturity_date, coupon_rate=0.05, coupon_frequency=2
- **THEN** the bond is initialized with the specified terms and inherits `BaseBondProduct` functionality

#### Scenario: Zero-coupon convertible
- **WHEN** a `ConvertibleBond` is created with coupon_rate=0
- **THEN** the bond represents a zero-coupon convertible with no periodic coupon payments

### Requirement: Conversion Features
The system SHALL support conversion ratio, conversion price, and conversion schedule (including lockout periods).

#### Scenario: Fixed conversion ratio
- **WHEN** a `ConvertibleBond` is created with conversion_ratio=10
- **THEN** the holder can convert each bond into 10 shares of common stock

#### Scenario: Conversion price calculation
- **WHEN** `conversion_price` property is accessed on a bond with face_value=100 and conversion_ratio=10
- **THEN** the conversion price is 10 (face_value / conversion_ratio)

#### Scenario: Conversion lockout period
- **WHEN** a `ConvertibleBond` is created with first_conversion_date after issue_date
- **THEN** conversion is not allowed before the first_conversion_date

#### Scenario: Parity calculation
- **WHEN** `parity(stock_price)` is called with stock_price=12 and conversion_ratio=10
- **THEN** the parity value is 120 (conversion_ratio * stock_price)

### Requirement: Call Provisions
The system SHALL support issuer call provisions with call schedule, call prices, and optional stock price triggers (provisional calls).

#### Scenario: Absolute call schedule
- **WHEN** a `ConvertibleBond` is created with call_schedule=[(date1, 110), (date2, 105), (date3, 100)]
- **THEN** the bond is callable at the specified prices on or after each date

#### Scenario: Provisional call with stock trigger
- **WHEN** a `ConvertibleBond` is created with provisional_call_trigger=1.30
- **THEN** the call can only be exercised if stock price exceeds 130% of conversion price

#### Scenario: Call protection period
- **WHEN** a `ConvertibleBond` is created with first_call_date after issue_date
- **THEN** the bond is not callable before the first_call_date

### Requirement: Put Provisions
The system SHALL support holder put provisions with put schedule and put prices.

#### Scenario: Put schedule
- **WHEN** a `ConvertibleBond` is created with put_schedule=[(date1, 100), (date2, 105)]
- **THEN** the holder can put the bond back to the issuer at the specified prices on each date

#### Scenario: No put provisions
- **WHEN** a `ConvertibleBond` is created with put_schedule=None or empty
- **THEN** the bond has no put provisions and cannot be put by the holder

### Requirement: Credit Attributes
The system SHALL support credit risk parameters including credit spread, hazard rate, and recovery rate for default modeling.

#### Scenario: Credit spread specification
- **WHEN** a `ConvertibleBond` is created with credit_spread=0.02 (200 bps)
- **THEN** the credit spread is available for credit-adjusted discounting

#### Scenario: Hazard rate for jump-diffusion
- **WHEN** a `ConvertibleBond` is created with hazard_rate=0.01 (1% annual default probability)
- **THEN** the hazard rate is available for jump-diffusion pricing

#### Scenario: Recovery rate
- **WHEN** a `ConvertibleBond` is created with recovery_rate=0.40
- **THEN** 40% of face value is recovered in the event of default

#### Scenario: Stock jump on default
- **WHEN** a `ConvertibleBond` is created with stock_jump_on_default=0.40
- **THEN** the stock price jumps down by 40% upon default event

### Requirement: Dividend Handling
The system SHALL support both continuous dividend yield and discrete dividend schedules for the underlying stock.

#### Scenario: Continuous dividend yield
- **WHEN** a `ConvertibleBond` is created with dividend_yield=0.02
- **THEN** the underlying stock is modeled with 2% continuous dividend yield

#### Scenario: Discrete dividend schedule
- **WHEN** a `ConvertibleBond` is created with discrete_dividends=[(date1, 0.50), (date2, 0.50)]
- **THEN** the underlying stock pays discrete dividends of $0.50 on each specified date

### Requirement: Cashflow Generation
The system SHALL generate coupon cashflows consistent with `BaseBondProduct.get_cashflows()` interface.

#### Scenario: Future coupon cashflows
- **WHEN** `get_cashflows(valuation_date)` is called on a convertible bond
- **THEN** all coupon payments after valuation_date are returned as Cashflow objects

#### Scenario: Principal at maturity
- **WHEN** `get_cashflows(valuation_date)` is called and bond has not been converted
- **THEN** the principal payment at maturity is included in cashflows

### Requirement: Accrued Interest Calculation
The system SHALL calculate accrued interest consistent with `BaseBondProduct.calculate_accrued_interest()` interface.

#### Scenario: Accrued interest calculation
- **WHEN** `calculate_accrued_interest(settlement_date)` is called
- **THEN** accrued interest is computed from last coupon date to settlement using day count convention

### Requirement: BaseBondProduct Interface Compliance
The system SHALL implement all abstract methods required by `BaseBondProduct` so that `ConvertibleBond` can be used anywhere a bond product is expected.

#### Scenario: Maturity date retrieval
- **WHEN** `get_maturity_date()` is called
- **THEN** the product returns the bond maturity date

#### Scenario: Issue date retrieval
- **WHEN** `get_issue_date()` is called
- **THEN** the product returns the bond issue date

#### Scenario: Notional retrieval
- **WHEN** `get_notional()` is called
- **THEN** the product returns the bond face value (par/notional)

### Requirement: Validation
The system SHALL validate input parameters and raise `ValidationError` for invalid configurations.

#### Scenario: Negative face value
- **WHEN** a `ConvertibleBond` is created with face_value=-100
- **THEN** a `ValidationError` is raised indicating face value must be positive

#### Scenario: Invalid conversion ratio
- **WHEN** a `ConvertibleBond` is created with conversion_ratio=0 or negative
- **THEN** a `ValidationError` is raised indicating conversion ratio must be positive

#### Scenario: Maturity before issue
- **WHEN** a `ConvertibleBond` is created with maturity_date before issue_date
- **THEN** a `ValidationError` is raised indicating maturity must be after issue date

#### Scenario: Invalid credit parameters
- **WHEN** a `ConvertibleBond` is created with recovery_rate > 1.0 or negative
- **THEN** a `ValidationError` is raised indicating recovery rate must be in [0, 1]

### Requirement: Representation
The system SHALL provide human-readable string representation for debugging and logging.

#### Scenario: String representation
- **WHEN** `__repr__` is called on a `ConvertibleBond`
- **THEN** a string showing key terms (face_value, maturity, conversion_ratio) is returned
