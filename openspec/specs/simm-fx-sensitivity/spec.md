# simm-fx-sensitivity Specification

## Purpose
TBD - created by archiving change add-simm-sensitivity-engines. Update Purpose after archive.
## Requirements
### Requirement: FX Sensitivity Engine Protocol

The system SHALL provide an `FXSensitivityEngine` class that calculates FX delta, vega, and curvature sensitivities per SIMM Sections C and I.

#### Scenario: Calculate FX delta
- **GIVEN** a position with EUR exposure and USD calculation currency
- **WHEN** calling `fx_engine.calculate_delta(position, pricing_env)`
- **THEN** a DeltaSensitivity with risk_class = FX is returned
- **AND** qualifier = "EUR"

### Requirement: FX Delta Calculation

The system SHALL calculate FX delta as the change in value for a 1% relative change in the exchange rate:

```
s_ik = V_i(FX_k + 1% × FX_k) - V_i(FX_k)
```

where FX_k is the spot exchange rate between currency k and the calculation currency.

#### Scenario: Calculate EUR/USD delta
- **GIVEN** a EUR-denominated position worth EUR 1,000,000
- **AND** calculation currency = USD
- **WHEN** calculating SIMM FX delta
- **THEN** delta = EUR 1,000,000 × EUR/USD rate × 0.01

#### Scenario: No FX delta for calculation currency
- **GIVEN** a USD position with USD calculation currency
- **WHEN** calculating FX delta
- **THEN** no FX sensitivity is generated (calculation currency excluded)

### Requirement: FX Translation Risk

The system SHALL include FX translation risk from the position's value into the calculation currency. All non-calculation-currency positions generate FX delta.

#### Scenario: FX translation from position value
- **GIVEN** a JPY-denominated bond position
- **AND** calculation currency = USD
- **WHEN** calculating sensitivities
- **THEN** FX delta to JPY is generated based on position market value

### Requirement: FX Single Bucket

All FX delta sensitivities SHALL be within a single bucket. Inter-bucket aggregation is not required for FX delta, but cross-bucket curvature calculations still apply.

#### Scenario: FX bucket assignment
- **GIVEN** FX sensitivities to EUR, GBP, JPY
- **WHEN** assigning buckets
- **THEN** all have bucket = "" (single FX bucket)

### Requirement: FX Vega Sensitivity

The system SHALL calculate FX vega for FX options using the vol-weighted formula:

```
VR_ik = HVR_c × Σ_j σ_kj × (∂V/∂σ)
```

where:
- HVR = 57% for FX
- σ_kj depends on the FX risk weights of both currencies in the pair

For FX vega, the risk weight to use is from the FX delta risk weight table based on volatility groups.

#### Scenario: Calculate EUR/USD option vega
- **GIVEN** a EUR/USD call option with vega = $1000 per 1% vol
- **AND** both EUR and USD are regular volatility currencies
- **AND** RW(regular, regular) = 7.4
- **WHEN** calculating SIMM FX vega
- **THEN** σ = 7.4 × sqrt(26.07) / 2.326 = 16.3%
- **AND** VR = 0.57 × 16.3% × $1000 = $92.91

#### Scenario: Calculate USD/BRL option vega
- **GIVEN** a USD/BRL option
- **AND** USD is regular, BRL is high volatility
- **AND** RW(regular, high) = 14.7
- **WHEN** calculating SIMM FX vega
- **THEN** higher σ is used due to high-vol currency

### Requirement: FX Vega Risk Factor

FX vega risk factors are currency pairs (e.g., EUR/USD, USD/JPY). The qualifier SHALL identify the currency pair, not individual currencies.

#### Scenario: FX vega qualifier
- **GIVEN** an option on EUR/USD
- **WHEN** generating vega sensitivity
- **THEN** qualifier = "EUR/USD" or similar pair identifier
- **AND** risk_type = RISK_FX_VOL

### Requirement: FX Curvature Sensitivity

The system SHALL calculate FX curvature using the standard scaling function:

```
CVR_ik = Σ_j SF(t_kj) × σ_kj × (∂V/∂σ)
SF(t) = 0.5 × min(1, 14/t)
```

FX vega/curvature risk factors are correlated at 50%.

#### Scenario: Calculate FX option curvature
- **GIVEN** an FX option expiring in 1 month (30 days)
- **WHEN** calculating curvature
- **THEN** SF = 0.5 × (14/30) = 0.233

### Requirement: FX Volatility Group Classification

The system SHALL classify currencies into FX volatility groups:
- **High volatility**: BRL, RUB, TRY
- **Regular volatility**: All other currencies

This classification affects risk weights and correlations.

#### Scenario: Classify BRL as high volatility
- **GIVEN** currency "BRL"
- **WHEN** determining volatility group
- **THEN** "high" is returned

#### Scenario: Classify EUR as regular volatility
- **GIVEN** currency "EUR"
- **WHEN** determining volatility group
- **THEN** "regular" is returned

### Requirement: FX Cross-Rate Sensitivity

For FX cross rates (e.g., EUR/GBP where calculation currency is USD), the system SHALL properly decompose into sensitivities to each currency versus the calculation currency.

#### Scenario: EUR/GBP cross sensitivity
- **GIVEN** an option on EUR/GBP with USD calculation currency
- **WHEN** calculating sensitivities
- **THEN** FX delta includes both EUR/USD and GBP/USD exposures
- **AND** vega is to EUR/GBP volatility

### Requirement: Calculation Currency Impact on Correlations

The system SHALL use different FX correlation tables based on the calculation currency's volatility group. When the calculation currency is regular volatility, one correlation matrix applies; when high volatility, a different matrix applies.

#### Scenario: Correlations with USD calc currency
- **GIVEN** calculation currency = USD (regular volatility)
- **WHEN** looking up FX correlations
- **THEN** the regular-calc-currency correlation table is used
- **AND** regular-regular correlation = 50%

#### Scenario: Correlations with BRL calc currency
- **GIVEN** calculation currency = BRL (high volatility)
- **WHEN** looking up FX correlations
- **THEN** the high-calc-currency correlation table is used
- **AND** regular-regular correlation = 88%

