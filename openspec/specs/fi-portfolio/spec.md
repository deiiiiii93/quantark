# fi-portfolio Specification

## Purpose
TBD - created by archiving change add-fi-backtest. Update Purpose after archive.
## Requirements
### Requirement: Fixed Income Position

The system SHALL provide an `FIPosition` class for tracking Fixed Income positions (bonds, bond futures).

The `FIPosition` MUST:
- Support bond products (`BaseBondProduct` subclasses)
- Calculate position DV01, modified duration, and convexity
- Support bond pricing engines for valuation
- Track entry price/yield and current market value
- Handle accrued interest correctly

#### Scenario: Bond Position Creation

- **GIVEN** a `FixedBond` product with 5% coupon and 10-year maturity
- **WHEN** an `FIPosition` is created with quantity 1,000,000 notional
- **THEN** the position stores the product, quantity, and entry price
- **AND** the position can calculate DV01 and duration

#### Scenario: Bond Position Risk Measures

- **GIVEN** an `FIPosition` with a bond product
- **WHEN** `get_risk_measures()` is called with a pricing environment
- **THEN** it returns DV01, modified duration, convexity, and market value
- **AND** risk measures are scaled by position quantity

### Requirement: Fixed Income Portfolio

The system SHALL provide an `FIPortfolio` class for managing multiple Fixed Income positions.

The `FIPortfolio` MUST:
- Implement the `BasePortfolio` protocol
- Aggregate DV01, duration, and convexity across positions
- Support multiple bonds with different maturities
- Calculate portfolio-level yield and duration
- Support bond futures positions for hedging

#### Scenario: Portfolio Risk Aggregation

- **GIVEN** an `FIPortfolio` with three bond positions of different maturities
- **WHEN** `get_portfolio_risk_measures()` is called
- **THEN** it returns aggregated DV01, weighted-average duration, and total convexity
- **AND** each component is correctly summed across positions

#### Scenario: Portfolio with Hedge Positions

- **GIVEN** an `FIPortfolio` with bond positions and bond futures hedges
- **WHEN** the net DV01 is calculated
- **THEN** it includes both long bond DV01 and short futures DV01
- **AND** the net exposure reflects the hedged position

### Requirement: Fixed Income Pricing Environment Integration

The system SHALL integrate `FIPortfolio` with the existing `PricingEnvironment` class for rate curve-based valuation.

#### Scenario: Portfolio Valuation Update

- **GIVEN** an `FIPortfolio` with positions
- **WHEN** the rate curve in the pricing environment shifts by 10 basis points
- **THEN** the portfolio market value changes according to DV01
- **AND** the change approximates DV01 × 10 bps for small moves

