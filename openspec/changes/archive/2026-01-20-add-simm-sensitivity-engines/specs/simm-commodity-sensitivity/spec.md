# SIMM Commodity Sensitivity

This capability provides Commodity sensitivity calculation for SIMM, including commodity delta, vega, and curvature sensitivities with bucket classification.

## ADDED Requirements

### Requirement: Commodity Sensitivity Engine Protocol

The system SHALL provide a `CommoditySensitivityEngine` class that calculates Commodity delta, vega, and curvature sensitivities per SIMM Sections C and H.

#### Scenario: Calculate commodity delta
- **GIVEN** a commodity futures position on WTI Crude
- **AND** a PricingEnvironment with crude oil price
- **WHEN** calling `commodity_engine.calculate_delta(position, pricing_env)`
- **THEN** a DeltaSensitivity with risk_class = COMMODITY is returned

### Requirement: Commodity Delta Calculation

The system SHALL calculate Commodity delta as the change in value for a 1% relative change in commodity price:

```
s_ik = V_i(CTY_k + 1% × CTY_k) - V_i(CTY_k)
```

#### Scenario: Calculate crude oil delta
- **GIVEN** a crude oil forward position with notional $1M
- **WHEN** calculating SIMM commodity delta
- **THEN** delta = $1M × 0.01 = $10,000

### Requirement: Commodity Bucket Classification

The system SHALL classify Commodity sensitivities into buckets 1-17 based on commodity type:

| Bucket | Commodity Type |
|--------|---------------|
| 1 | Coal |
| 2 | Crude |
| 3 | Light Ends |
| 4 | Middle Distillates |
| 5 | Heavy Distillates |
| 6 | North America Natural Gas |
| 7 | European Natural Gas |
| 8 | North American Power |
| 9 | European Power and Carbon |
| 10 | Freight |
| 11 | Base Metals |
| 12 | Precious Metals |
| 13 | Grains and Oilseed |
| 14 | Softs and Other Agriculturals |
| 15 | Livestock and Dairy |
| 16 | Other |
| 17 | Indexes |

#### Scenario: Classify WTI crude
- **GIVEN** a WTI crude oil position
- **WHEN** classifying bucket
- **THEN** bucket = 2 (Crude)

#### Scenario: Classify gold
- **GIVEN** a gold position
- **WHEN** classifying bucket
- **THEN** bucket = 12 (Precious Metals)

#### Scenario: Classify S&P GSCI
- **GIVEN** a S&P GSCI commodity index position
- **WHEN** classifying bucket
- **THEN** bucket = 17 (Indexes)

### Requirement: Commodity Forward Curve Treatment

Risks to commodity forward prices SHALL be allocated back to spot price risks, assuming parallel curve shifts.

#### Scenario: Commodity forward sensitivity
- **GIVEN** a 6-month crude oil forward
- **WHEN** calculating delta
- **THEN** sensitivity is expressed as spot price risk
- **AND** bucket = 2 (Crude)

### Requirement: Commodity Index Handling

For commodity indexes, delta sensitivities SHALL be assigned to bucket 17 (standard approach) unless using advanced allocation to constituents.

#### Scenario: GSCI index delta
- **GIVEN** a position in GSCI commodity index
- **WHEN** calculating delta (standard approach)
- **THEN** bucket = 17 (Indexes)

### Requirement: Commodity Vega Calculation

The system SHALL calculate Commodity vega using the vol-weighted formula:

```
VR_ik = HVR_c × Σ_j σ_kj × (∂V/∂σ)
```

where:
- HVR = 74% for Commodity
- σ_kj = RW_k × sqrt(365/14) / α

For commodity index volatilities, use the risk weight of bucket 17 (Indexes).

#### Scenario: Calculate crude oil option vega
- **GIVEN** a crude oil call option with vega = $10 per 1% vol
- **AND** bucket 2 (RW = 29)
- **WHEN** calculating SIMM vega
- **THEN** σ = 29 × sqrt(26.07) / 2.326 = 63.7%
- **AND** VR = 0.74 × 63.7% × $10 = $4.71

### Requirement: Commodity Curvature Calculation

The system SHALL calculate Commodity curvature using the standard scaling function:

```
CVR_ik = Σ_j SF(t_kj) × σ_kj × (∂V/∂σ)
SF(t) = 0.5 × min(1, 14/t)
```

#### Scenario: Calculate curvature for 3-month option
- **GIVEN** a commodity option expiring in 3 months (91 days)
- **WHEN** calculating curvature
- **THEN** SF = 0.5 × (14/91) = 0.077

### Requirement: Commodity Qualifier Identification

The system SHALL use a consistent commodity identifier as the qualifier field. Common qualifiers include specific commodity names like "Coal Europe", "Precious Metals Gold", "Livestock Lean Hogs".

#### Scenario: Commodity qualifier for gold
- **GIVEN** a gold futures position
- **WHEN** generating sensitivity
- **THEN** qualifier = "Precious Metals Gold" or similar identifier
- **AND** bucket = 12

### Requirement: Regional Commodity Distinction

The system SHALL distinguish regional commodities (e.g., NA Natural Gas vs EU Natural Gas) as different risk factors in separate buckets.

#### Scenario: Henry Hub vs TTF natural gas
- **GIVEN** positions in Henry Hub (NA) and TTF (EU) natural gas
- **WHEN** calculating sensitivities
- **THEN** Henry Hub has bucket = 6 (NA Natural Gas)
- **AND** TTF has bucket = 7 (EU Natural Gas)

### Requirement: Bespoke Commodity Basket Handling

For bespoke commodity baskets, delta sensitivities SHALL be allocated back to individual commodities. Index vega sensitivities SHALL remain at the index level (bucket 17).

#### Scenario: Custom commodity basket
- **GIVEN** an option on a custom basket of commodities
- **WHEN** calculating sensitivities
- **THEN** delta is allocated to each underlying commodity
- **AND** vega is assigned to bucket 17 (Indexes)

