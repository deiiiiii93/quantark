# simm-ir-sensitivity Specification

## Purpose
TBD - created by archiving change add-simm-sensitivity-engines. Update Purpose after archive.
## Requirements
### Requirement: IR Delta Sensitivity Protocol

The system SHALL provide an `IRSensitivityEngine` class that calculates Interest Rate delta sensitivities per SIMM Section C and D definitions.

#### Scenario: Calculate IR delta for a bond position
- **GIVEN** a FIPosition with a fixed-rate bond in USD
- **AND** a PricingEnvironment with USD rate curve
- **WHEN** calling `ir_engine.calculate_delta(position, pricing_env)`
- **THEN** a list of DeltaSensitivity objects is returned
- **AND** each sensitivity has risk_class = INTEREST_RATE
- **AND** sensitivities cover all 12 tenor vertices with non-zero exposure

### Requirement: IR PV01 Calculation

The system SHALL calculate IR delta (PV01) as the change in value for a 1 basis point parallel shift at each tenor vertex:

```
s(i, r_t) = V_i(r_t + 1bp, cs_t) - V_i(r_t, cs_t)
```

#### Scenario: Calculate PV01 at 5-year tenor
- **GIVEN** a 10-year bond with USD exposure
- **WHEN** bumping the 5-year rate by 1bp and repricing
- **THEN** the PV01 at 5yr tenor is returned
- **AND** the sign indicates price decrease for rate increase (negative for long bonds)

#### Scenario: PV01 tenor bucketing
- **GIVEN** a bond with cash flows at 3.5 years and 7 years
- **WHEN** calculating PV01
- **THEN** sensitivities are assigned to the nearest standard tenors (3yr, 5yr, 10yr)
- **OR** interpolated across adjacent tenors

### Requirement: IR Sub-Curve Decomposition

The system SHALL decompose IR sensitivities by sub-curve (OIS, Libor1m, Libor3m, Libor6m, Libor12m, Prime, Municipal) when the position references specific rate indices.

#### Scenario: Swap with 3M LIBOR leg
- **GIVEN** an interest rate swap paying 3M LIBOR
- **WHEN** calculating delta sensitivities
- **THEN** sensitivities include entries with label2 = "Libor3m"

#### Scenario: OIS swap
- **GIVEN** an OIS swap
- **WHEN** calculating delta sensitivities
- **THEN** sensitivities have label2 = "OIS"

### Requirement: IR Inflation Sensitivity

The system SHALL calculate inflation sensitivity for inflation-linked instruments as the change in value for a 1bp shock to the flat inflation rate.

#### Scenario: Calculate inflation sensitivity
- **GIVEN** an inflation-linked bond
- **WHEN** calling `ir_engine.calculate_inflation_delta(position, pricing_env)`
- **THEN** a DeltaSensitivity with risk_type = RISK_INFLATION is returned

### Requirement: IR Cross-Currency Basis Sensitivity

The system SHALL calculate cross-currency basis sensitivity for cross-currency swaps as the change in value for a 1bp shock to the basis spread.

#### Scenario: Calculate xccy basis sensitivity
- **GIVEN** a USD/EUR cross-currency swap
- **WHEN** calling `ir_engine.calculate_xccy_basis_delta(position, pricing_env)`
- **THEN** a DeltaSensitivity with risk_type = RISK_XCCY_BASIS is returned
- **AND** the qualifier is the non-USD currency

### Requirement: IR Vega Sensitivity

The system SHALL calculate IR vega sensitivity for instruments with IR optionality (swaptions, caps, floors):

```
VR_ik = VRW × (Σ_i VR_ik) × VCR_b
```

where VR_ik = Σ_j σ_kj × (∂V/∂σ)

#### Scenario: Calculate swaption vega
- **GIVEN** a 5Y into 10Y payer swaption
- **WHEN** calling `ir_engine.calculate_vega(position, pricing_env)`
- **THEN** a VegaSensitivity is returned
- **AND** risk_type = RISK_IR_VOL
- **AND** option_expiry corresponds to 5 years

#### Scenario: No vega for non-option instruments
- **GIVEN** a fixed-rate bond with no optionality
- **WHEN** calling `ir_engine.calculate_vega(position, pricing_env)`
- **THEN** an empty list is returned

### Requirement: IR Curvature Sensitivity

The system SHALL calculate IR curvature sensitivity using the scaling function:

```
CVR_ik = Σ_j SF(t_kj) × σ_kj × (∂V/∂σ)
SF(t) = 0.5 × min(1, 14 days / t days)
```

The final IR curvature margin MUST be multiplied by HVR_IR^(-2).

#### Scenario: Calculate swaption curvature
- **GIVEN** a swaption expiring in 1 year (365 days)
- **WHEN** calculating curvature
- **THEN** SF = 0.5 × (14/365) = 0.0192 is applied
- **AND** a CurvatureSensitivity is returned

### Requirement: IR Currency Bucket Assignment

The system SHALL assign IR sensitivities to currency-based buckets. Each currency constitutes its own bucket.

#### Scenario: USD and EUR sensitivities in separate buckets
- **GIVEN** a portfolio with USD and EUR rate exposure
- **WHEN** calculating sensitivities
- **THEN** USD sensitivities have bucket = "USD"
- **AND** EUR sensitivities have bucket = "EUR"

### Requirement: IR Tenor Interpolation

The system SHALL handle instruments with cash flows at non-standard tenors by interpolating sensitivities to adjacent standard tenor vertices.

#### Scenario: Interpolate 4-year maturity
- **GIVEN** a bond maturing in exactly 4 years
- **WHEN** calculating PV01
- **THEN** sensitivity is split between 3yr and 5yr tenors proportionally

