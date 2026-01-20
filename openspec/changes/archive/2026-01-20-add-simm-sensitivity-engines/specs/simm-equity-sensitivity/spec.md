# SIMM Equity Sensitivity

This capability provides Equity sensitivity calculation for SIMM, including equity delta, vega, and curvature sensitivities with bucket classification.

## ADDED Requirements

### Requirement: Equity Sensitivity Engine Protocol

The system SHALL provide an `EquitySensitivityEngine` class that calculates Equity delta, vega, and curvature sensitivities per SIMM Sections C and G.

#### Scenario: Calculate equity delta for option position
- **GIVEN** an EquityPosition with a call option on AAPL
- **AND** a PricingEnvironment with AAPL spot price
- **WHEN** calling `equity_engine.calculate_delta(position, pricing_env)`
- **THEN** a DeltaSensitivity with risk_class = EQUITY is returned

### Requirement: Equity Delta Calculation

The system SHALL calculate Equity delta as the change in value for a 1% relative change in equity price:

```
s_ik = V_i(EQ_k + 1% × EQ_k) - V_i(EQ_k)
```

#### Scenario: Calculate 1% equity delta
- **GIVEN** an equity option with BS delta = 0.5
- **AND** spot price = $100
- **WHEN** calculating SIMM equity delta
- **THEN** delta = 0.5 × $100 × 0.01 = $0.50 per contract

#### Scenario: Delta-one equity sensitivity
- **GIVEN** a spot equity position of 100 shares at $50
- **WHEN** calculating SIMM equity delta
- **THEN** delta = 100 × $50 × 0.01 = $50

### Requirement: Equity Bucket Classification

The system SHALL classify Equity sensitivities into buckets 1-12 or Residual based on:
- **Size**: Large (market cap >= $2B) vs Small (< $2B)
- **Region**: Emerging Markets vs Developed Markets
- **Sector**: Consumer, Telecom, Basic Materials, Financials, etc.

| Buckets | Size | Region | Sector Pattern |
|---------|------|--------|----------------|
| 1-4 | Large | Emerging | By sector |
| 5-8 | Large | Developed | By sector |
| 9 | Small | Emerging | All |
| 10 | Small | Developed | All |
| 11 | All | All | Indexes/ETFs |
| 12 | All | All | Vol Indexes |

#### Scenario: Classify US large-cap tech stock
- **GIVEN** AAPL with market cap > $2B, US (Developed), Technology sector
- **WHEN** classifying bucket
- **THEN** bucket = 8 (Large, Developed, Financials/Real Estate/Tech)

#### Scenario: Classify small-cap EM stock
- **GIVEN** a Brazilian stock with market cap < $2B
- **WHEN** classifying bucket
- **THEN** bucket = 9 (Small, Emerging Markets)

### Requirement: Equity Index Handling

For equity indexes, ETFs, and funds, delta sensitivities SHALL be assigned to bucket 11 (standard approach) unless bilaterally agreed to allocate to constituents.

#### Scenario: SPY ETF delta
- **GIVEN** a position in SPY ETF
- **WHEN** calculating delta (standard approach)
- **THEN** bucket = 11 (Indexes, Funds, ETFs)

### Requirement: Volatility Index Handling

Equity volatility index positions (e.g., VIX) SHALL be assigned to bucket 12. Curvature risk for bucket 12 SHALL be zero.

#### Scenario: VIX option sensitivity
- **GIVEN** a VIX call option position
- **WHEN** calculating sensitivities
- **THEN** delta and vega go to bucket = 12
- **AND** curvature sensitivity = 0

### Requirement: Equity Vega Calculation

The system SHALL calculate Equity vega using the vol-weighted formula:

```
VR_ik = HVR_c × Σ_j σ_kj × (∂V/∂σ)
```

where:
- HVR = 60% for Equity
- σ_kj = RW_k × sqrt(365/14) / α
- α = Φ^(-1)(99%)

#### Scenario: Calculate equity vega for option
- **GIVEN** an equity option with vega = $5 per 1% vol
- **AND** bucket 5 (RW = 26)
- **WHEN** calculating SIMM vega
- **THEN** σ = 26 × sqrt(26.07) / 2.326 = 57.1%
- **AND** VR = 0.60 × 57.1% × $5 = $1.71

### Requirement: Equity Vega Tenor Bucketing

Equity vega SHALL be mapped to option expiry tenor buckets matching IR tenors: 2w, 1m, 3m, 6m, 1yr, 2yr, 3yr, 5yr, 10yr, 15yr, 20yr, 30yr.

#### Scenario: Map option expiry to tenor
- **GIVEN** an option expiring in 4 months
- **WHEN** mapping to tenor bucket
- **THEN** assigned to "3m" or "6m" bucket (nearest or interpolated)

### Requirement: Equity Curvature Calculation

The system SHALL calculate Equity curvature using:

```
CVR_ik = Σ_j SF(t_kj) × σ_kj × (∂V/∂σ)
SF(t) = 0.5 × min(1, 14/t)
```

#### Scenario: Calculate curvature for 1-year option
- **GIVEN** an equity option expiring in 1 year (365 days)
- **WHEN** calculating curvature
- **THEN** SF = 0.5 × (14/365) = 0.0192

#### Scenario: Calculate curvature for 2-week option
- **GIVEN** an equity option expiring in 2 weeks (14 days)
- **WHEN** calculating curvature
- **THEN** SF = 0.5 × min(1, 14/14) = 0.5

### Requirement: Integration with GreeksCalculator

The system SHALL leverage the existing `GreeksCalculator` class to compute underlying Greeks (delta, vega) and convert them to SIMM format.

#### Scenario: Use analytical Greeks when available
- **GIVEN** a European vanilla option
- **WHEN** calculating SIMM sensitivities
- **THEN** analytical BS Greeks are used (faster, more accurate)

#### Scenario: Fall back to numerical Greeks
- **GIVEN** an American option or barrier option
- **WHEN** calculating SIMM sensitivities
- **THEN** numerical Greeks (bump-and-reprice) are used

### Requirement: Equity Qualifier Identification

The system SHALL use a consistent equity identifier (ticker, ISIN, or internal ID) as the qualifier field for all equity sensitivities.

#### Scenario: Consistent qualifier across margin types
- **GIVEN** delta and vega sensitivities for AAPL option
- **WHEN** generating sensitivities
- **THEN** both have qualifier = "AAPL" (or configured identifier)

### Requirement: Bespoke Basket Handling

For bespoke equity baskets, delta AND vega sensitivities SHALL be allocated back to individual underlying equities.

#### Scenario: Custom equity basket option
- **GIVEN** an option on a custom basket of 10 stocks
- **WHEN** calculating sensitivities
- **THEN** delta is allocated to each of the 10 underlying stocks
- **AND** vega is allocated to each of the 10 underlying stocks

