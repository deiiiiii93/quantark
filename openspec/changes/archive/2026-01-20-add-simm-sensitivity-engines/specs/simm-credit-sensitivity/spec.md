# SIMM Credit Sensitivity

This capability provides Credit sensitivity calculation for SIMM, including Credit Qualifying (CS01, base correlation) and Credit Non-Qualifying sensitivities.

## ADDED Requirements

### Requirement: Credit Sensitivity Engine Protocol

The system SHALL provide a `CreditSensitivityEngine` class that calculates Credit Qualifying and Non-Qualifying delta sensitivities per SIMM Sections E and F.

#### Scenario: Calculate Credit Q delta
- **GIVEN** a position in a corporate CDS
- **AND** issuer credit quality is Investment Grade
- **WHEN** calling `credit_engine.calculate_delta(position, pricing_env)`
- **THEN** a list of DeltaSensitivity objects is returned
- **AND** risk_class = CREDIT_QUALIFYING

### Requirement: CS01 Calculation

The system SHALL calculate Credit delta (CS01) as the change in value for a 1 basis point credit spread shift at each tenor vertex:

```
s(i, cs_t) = V_i(r_t, cs_t + 1bp) - V_i(r_t, cs_t)
```

Credit tenors are: 1yr, 2yr, 3yr, 5yr, 10yr

#### Scenario: Calculate CS01 for 5-year CDS
- **GIVEN** a 5-year CDS on an IG corporate
- **WHEN** calculating CS01
- **THEN** sensitivities at 1yr, 2yr, 3yr, 5yr tenors are returned
- **AND** the 5yr tenor has the largest sensitivity

### Requirement: Credit Issuer/Seniority Classification

The system SHALL classify Credit sensitivities by issuer and seniority. Sensitivities to the same issuer/seniority but different tenors or currencies are separate risk factors with 93% correlation.

#### Scenario: Same issuer different tenors
- **GIVEN** CS01 sensitivities to Company ABC at 3yr and 5yr
- **WHEN** classifying risk factors
- **THEN** both have qualifier = "ABC" but different label1 (tenor)

### Requirement: Credit Qualifying Bucket Assignment

The system SHALL assign Credit Q sensitivities to buckets 1-12 or Residual based on credit quality (IG vs HY/NR) and sector:

| Bucket | Quality | Sector |
|--------|---------|--------|
| 1 | IG | Sovereigns |
| 2 | IG | Financials |
| 3 | IG | Basic materials, energy, industrials |
| 4 | IG | Consumer |
| 5 | IG | Technology, telecommunications |
| 6 | IG | Health care, utilities, local government |
| 7-12 | HY/NR | Same sectors as 1-6 |
| Residual | - | Unclassified |

#### Scenario: Classify IG Financial issuer
- **GIVEN** a CDS on an Investment Grade bank
- **WHEN** assigning bucket
- **THEN** bucket = 2 (IG Financials)

#### Scenario: Classify HY Tech issuer
- **GIVEN** a CDS on a High Yield technology company
- **WHEN** assigning bucket
- **THEN** bucket = 11 (HY Technology/Telecom)

### Requirement: Credit Qualifying Index Handling

For Credit Qualifying indexes (CDX, iTraxx), delta sensitivities SHALL be computed to the underlying issuer/seniority risk factors. Vega sensitivities MAY remain at the index level.

#### Scenario: CDX IG index delta allocation
- **GIVEN** a CDX IG index position
- **WHEN** calculating delta sensitivities
- **THEN** sensitivities are allocated to underlying constituent issuers
- **AND** each constituent sensitivity has appropriate bucket assignment

### Requirement: Base Correlation Sensitivity (BC01)

The system SHALL calculate Base Correlation sensitivity for CDO tranches:

```
s_ik = V_i(BC_k + 1%) - V_i(BC_k)
```

where BC_k is the base correlation for index family k (CDX IG, iTraxx Main, etc.)

#### Scenario: Calculate BC01 for CDX tranche
- **GIVEN** a CDX IG tranche position
- **WHEN** calling `credit_engine.calculate_base_corr_delta(position, pricing_env)`
- **THEN** a DeltaSensitivity with risk_type = RISK_BASE_CORR is returned
- **AND** qualifier identifies the index family

### Requirement: Credit Non-Qualifying Classification

The system SHALL classify Credit Non-Qualifying sensitivities into buckets 1 (IG RMBS/CMBS) or 2 (HY/NR RMBS/CMBS) or Residual.

#### Scenario: Classify CMBS position
- **GIVEN** an IG-rated CMBS position
- **WHEN** assigning Credit NQ bucket
- **THEN** bucket = 1 (IG RMBS/CMBS)

### Requirement: Credit Non-Qualifying Securitization Rule

For non-qualifying securitizations, CS01 SHALL be calculated with respect to the spread of the instrument (not the underlying), unless it meets qualifying criteria.

#### Scenario: Calculate CS01 for non-qualifying securitization
- **GIVEN** a non-qualifying ABS position
- **WHEN** calculating CS01
- **THEN** sensitivity is to the tranche spread, not underlying credits

### Requirement: Credit Vega Sensitivity

The system SHALL calculate Credit vega for instruments with credit volatility exposure:

```
VR_k = VRW × (Σ_i VR_ik) × VCR_k
```

Index vega need not be allocated to underlying issuers.

#### Scenario: Calculate credit index vega
- **GIVEN** an option on CDX IG
- **WHEN** calculating vega
- **THEN** a VegaSensitivity with the CDX index as qualifier is returned
- **AND** bucket matches the CDX bucket classification

### Requirement: Credit Curvature Sensitivity

The system SHALL calculate Credit curvature using the standard scaling function SF(t).

#### Scenario: Calculate credit index option curvature
- **GIVEN** an option on iTraxx Main expiring in 6 months
- **WHEN** calculating curvature
- **THEN** SF(182.5 days) = 0.5 × (14/182.5) = 0.0384 is applied

### Requirement: Credit Payment Currency Handling

Credit sensitivities SHALL be distinguished by payment currency. Quanto CDS (non-local currency) and standard CDS (local currency) are treated as different risk factors with 93% correlation.

#### Scenario: Quanto CDS vs standard CDS
- **GIVEN** EUR-denominated CDS on USD issuer (Quanto)
- **AND** USD-denominated CDS on same issuer (Standard)
- **WHEN** calculating sensitivities
- **THEN** they have different amount_currency values
- **AND** they are not fully netted (93% correlated)

