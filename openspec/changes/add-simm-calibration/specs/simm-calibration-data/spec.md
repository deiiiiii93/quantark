# SIMM Calibration Data

This capability provides all ISDA SIMM v2.6 calibration parameters including risk weights, correlations, concentration thresholds, historical volatility ratios, and vega risk weights.

## ADDED Requirements

### Requirement: SIMM Version Information

The system SHALL provide version information for the implemented SIMM calibration:
- Version: "2.6"
- Base version: "2.5.6"
- Effective date: December 2, 2023
- Publication date: August 16, 2023

#### Scenario: Access SIMM version
- **GIVEN** the calibration module is imported
- **WHEN** accessing `SIMM_VERSION`
- **THEN** a SIMMVersion object with version="2.6" is returned

### Requirement: Interest Rate Risk Weights

The system SHALL provide IR delta risk weights for all 12 tenors and 3 currency volatility groups as specified in SIMM v2.6 Section D.1:

| Tenor | Regular Vol | Low Vol (JPY) | High Vol |
|-------|-------------|---------------|----------|
| 2w    | 109         | 15            | 163      |
| 1m    | 105         | 18            | 109      |
| 3m    | 90          | 9             | 87       |
| 6m    | 71          | 11            | 89       |
| 1yr   | 66          | 13            | 102      |
| 2yr   | 66          | 15            | 96       |
| 3yr   | 64          | 19            | 101      |
| 5yr   | 60          | 23            | 97       |
| 10yr  | 60          | 23            | 97       |
| 15yr  | 61          | 22            | 102      |
| 20yr  | 61          | 22            | 106      |
| 30yr  | 67          | 23            | 101      |

#### Scenario: Get IR risk weight for regular currency
- **GIVEN** tenor "5yr" and currency "USD"
- **WHEN** calling `get_ir_risk_weight("5yr", "USD")`
- **THEN** 60 is returned

#### Scenario: Get IR risk weight for low volatility currency
- **GIVEN** tenor "5yr" and currency "JPY"
- **WHEN** calling `get_ir_risk_weight("5yr", "JPY")`
- **THEN** 23 is returned

#### Scenario: Get IR risk weight for high volatility currency
- **GIVEN** tenor "5yr" and currency "BRL"
- **WHEN** calling `get_ir_risk_weight("5yr", "BRL")`
- **THEN** 97 is returned

### Requirement: Interest Rate Special Risk Weights

The system SHALL provide special IR risk weights:
- Inflation risk weight: 61 (for all currencies)
- Cross-currency basis risk weight: 21 (for all currencies)

#### Scenario: Get inflation risk weight
- **GIVEN** any currency
- **WHEN** calling `get_ir_inflation_risk_weight()`
- **THEN** 61 is returned

#### Scenario: Get cross-currency basis risk weight
- **GIVEN** any currency
- **WHEN** calling `get_ir_xccy_basis_risk_weight()`
- **THEN** 21 is returned

### Requirement: Interest Rate Tenor Correlations

The system SHALL provide the IR tenor correlation matrix (ρ_kl) as specified in SIMM v2.6 Section D.2. The matrix MUST be symmetric and positive semi-definite.

#### Scenario: Get IR tenor correlation
- **GIVEN** tenors "5yr" and "10yr"
- **WHEN** calling `get_ir_tenor_correlation("5yr", "10yr")`
- **THEN** 0.95 (95%) is returned

#### Scenario: IR tenor correlation symmetry
- **GIVEN** any two tenors t1 and t2
- **WHEN** calling `get_ir_tenor_correlation(t1, t2)` and `get_ir_tenor_correlation(t2, t1)`
- **THEN** both calls return the same value

### Requirement: Interest Rate Sub-Curve Correlation

The system SHALL provide the IR sub-curve correlation (φ_ij) = 99.3% between any two sub-curves of the same currency.

#### Scenario: Get sub-curve correlation
- **GIVEN** sub-curves "OIS" and "Libor3m"
- **WHEN** calling `get_ir_subcurve_correlation("OIS", "Libor3m")`
- **THEN** 0.993 is returned

### Requirement: Interest Rate Special Correlations

The system SHALL provide IR special correlations:
- Inflation to yield correlation: 24%
- Cross-currency basis to yield/inflation correlation: 4%
- Inter-currency correlation (γ_bc): 32%

#### Scenario: Get inflation correlation
- **GIVEN** inflation and any yield tenor
- **WHEN** calling `get_ir_inflation_correlation()`
- **THEN** 0.24 is returned

#### Scenario: Get inter-currency correlation
- **GIVEN** any two currencies
- **WHEN** calling `get_ir_inter_currency_correlation()`
- **THEN** 0.32 is returned

### Requirement: Interest Rate HVR and VRW

The system SHALL provide IR historical volatility ratio and vega risk weight:
- HVR: 0.47
- VRW: 0.23

#### Scenario: Get IR HVR
- **WHEN** calling `get_hvr(RiskClass.INTEREST_RATE)`
- **THEN** 0.47 is returned

#### Scenario: Get IR VRW
- **WHEN** calling `get_vrw(RiskClass.INTEREST_RATE)`
- **THEN** 0.23 is returned

### Requirement: Credit Qualifying Risk Weights

The system SHALL provide Credit Q delta risk weights by bucket as specified in SIMM v2.6 Section E.1:

| Bucket | Risk Weight |
|--------|-------------|
| 1 (IG Sovereigns) | 75 |
| 2 (IG Financials) | 90 |
| 3 (IG Basic materials) | 84 |
| 4 (IG Consumer) | 54 |
| 5 (IG Tech/Telecom) | 62 |
| 6 (IG Health/Utilities) | 48 |
| 7 (HY Sovereigns) | 185 |
| 8 (HY Financials) | 343 |
| 9 (HY Basic materials) | 255 |
| 10 (HY Consumer) | 250 |
| 11 (HY Tech/Telecom) | 214 |
| 12 (HY Health/Utilities) | 173 |
| Residual | 343 |

#### Scenario: Get Credit Q risk weight for IG bucket
- **GIVEN** bucket 1
- **WHEN** calling `get_credit_q_risk_weight(1)`
- **THEN** 75 is returned

#### Scenario: Get Credit Q risk weight for HY bucket
- **GIVEN** bucket 8
- **WHEN** calling `get_credit_q_risk_weight(8)`
- **THEN** 343 is returned

### Requirement: Credit Qualifying Correlations

The system SHALL provide Credit Q correlations as specified in SIMM v2.6 Section E.2:
- Same issuer/seniority, different vertex or currency: 93%
- Different issuer/seniority within bucket: 46%
- Residual bucket: 50% for all pairs
- Base correlation across index families: 29%

#### Scenario: Get Credit Q same-name correlation
- **GIVEN** same issuer, different tenors
- **WHEN** calling `get_credit_q_same_issuer_correlation()`
- **THEN** 0.93 is returned

#### Scenario: Get Credit Q different-name correlation
- **GIVEN** different issuers in same bucket
- **WHEN** calling `get_credit_q_different_issuer_correlation()`
- **THEN** 0.46 is returned

### Requirement: Credit Qualifying Inter-Bucket Correlations

The system SHALL provide the Credit Q inter-bucket correlation matrix (γ_bc) as a 12x12 matrix as specified in SIMM v2.6 Section E.2.

#### Scenario: Get Credit Q inter-bucket correlation
- **GIVEN** buckets 1 and 7
- **WHEN** calling `get_credit_q_inter_bucket_correlation(1, 7)`
- **THEN** 0.42 is returned

### Requirement: Credit Qualifying Base Correlation

The system SHALL provide Credit Q base correlation parameters:
- Risk weight: 10
- Inter-index family correlation: 29%

#### Scenario: Get base correlation risk weight
- **WHEN** calling `get_base_corr_risk_weight()`
- **THEN** 10 is returned

### Requirement: Credit Non-Qualifying Risk Weights

The system SHALL provide Credit NQ delta risk weights as specified in SIMM v2.6 Section F.1:

| Bucket | Risk Weight |
|--------|-------------|
| 1 (IG RMBS/CMBS) | 280 |
| 2 (HY RMBS/CMBS) | 1300 |
| Residual | 1300 |

#### Scenario: Get Credit NQ risk weight
- **GIVEN** bucket 1
- **WHEN** calling `get_credit_nq_risk_weight(1)`
- **THEN** 280 is returned

### Requirement: Credit Non-Qualifying Correlations

The system SHALL provide Credit NQ correlations as specified in SIMM v2.6 Section F.2:
- Same group name: 83%
- Different group name: 32%
- Residual bucket: 50%
- Inter-bucket (non-residual): 43%

#### Scenario: Get Credit NQ same-group correlation
- **WHEN** calling `get_credit_nq_same_group_correlation()`
- **THEN** 0.83 is returned

### Requirement: Equity Risk Weights

The system SHALL provide Equity delta risk weights by bucket as specified in SIMM v2.6 Section G.1:

| Bucket | Risk Weight |
|--------|-------------|
| 1 | 30 |
| 2 | 33 |
| 3 | 36 |
| 4 | 29 |
| 5 | 26 |
| 6 | 25 |
| 7 | 34 |
| 8 | 28 |
| 9 | 36 |
| 10 | 50 |
| 11 | 19 |
| 12 | 19 |
| Residual | 50 |

#### Scenario: Get Equity risk weight
- **GIVEN** bucket 5
- **WHEN** calling `get_equity_risk_weight(5)`
- **THEN** 26 is returned

### Requirement: Equity Intra-Bucket Correlations

The system SHALL provide Equity intra-bucket correlations (ρ_kl) as specified in SIMM v2.6 Section G.2:

| Bucket | Correlation |
|--------|-------------|
| 1 | 18% |
| 2 | 20% |
| 3 | 28% |
| 4 | 24% |
| 5 | 25% |
| 6 | 36% |
| 7 | 35% |
| 8 | 37% |
| 9 | 23% |
| 10 | 27% |
| 11 | 45% |
| 12 | 45% |
| Residual | 0% |

#### Scenario: Get Equity intra-bucket correlation
- **GIVEN** bucket 6
- **WHEN** calling `get_equity_intra_bucket_correlation(6)`
- **THEN** 0.36 is returned

### Requirement: Equity Inter-Bucket Correlations

The system SHALL provide the Equity inter-bucket correlation matrix (γ_bc) as a 12x12 matrix as specified in SIMM v2.6 Section G.2.

#### Scenario: Get Equity inter-bucket correlation
- **GIVEN** buckets 5 and 6
- **WHEN** calling `get_equity_inter_bucket_correlation(5, 6)`
- **THEN** 0.29 is returned

### Requirement: Equity HVR and VRW

The system SHALL provide Equity HVR and VRW:
- HVR: 60%
- VRW: 0.45 for all buckets except bucket 12
- VRW for bucket 12 (Volatility Indexes): 0.96

#### Scenario: Get Equity HVR
- **WHEN** calling `get_hvr(RiskClass.EQUITY)`
- **THEN** 0.60 is returned

#### Scenario: Get Equity VRW for standard bucket
- **GIVEN** bucket 5
- **WHEN** calling `get_equity_vrw(5)`
- **THEN** 0.45 is returned

#### Scenario: Get Equity VRW for volatility index bucket
- **GIVEN** bucket 12
- **WHEN** calling `get_equity_vrw(12)`
- **THEN** 0.96 is returned

### Requirement: Commodity Risk Weights

The system SHALL provide Commodity delta risk weights by bucket as specified in SIMM v2.6 Section H.1:

| Bucket | Commodity Type | Risk Weight |
|--------|---------------|-------------|
| 1 | Coal | 48 |
| 2 | Crude | 29 |
| 3 | Light Ends | 33 |
| 4 | Middle Distillates | 25 |
| 5 | Heavy Distillates | 35 |
| 6 | NA Natural Gas | 30 |
| 7 | EU Natural Gas | 60 |
| 8 | NA Power | 52 |
| 9 | EU Power/Carbon | 68 |
| 10 | Freight | 63 |
| 11 | Base Metals | 21 |
| 12 | Precious Metals | 21 |
| 13 | Grains/Oilseed | 15 |
| 14 | Softs/Ag | 16 |
| 15 | Livestock/Dairy | 13 |
| 16 | Other | 68 |
| 17 | Indexes | 17 |

#### Scenario: Get Commodity risk weight
- **GIVEN** bucket 2 (Crude)
- **WHEN** calling `get_commodity_risk_weight(2)`
- **THEN** 29 is returned

### Requirement: Commodity Correlations

The system SHALL provide Commodity intra-bucket and inter-bucket correlations as specified in SIMM v2.6 Sections H.2.

#### Scenario: Get Commodity intra-bucket correlation
- **GIVEN** bucket 2 (Crude)
- **WHEN** calling `get_commodity_intra_bucket_correlation(2)`
- **THEN** 0.97 is returned

#### Scenario: Get Commodity inter-bucket correlation
- **GIVEN** buckets 2 and 3
- **WHEN** calling `get_commodity_inter_bucket_correlation(2, 3)`
- **THEN** 0.92 is returned

### Requirement: Commodity HVR and VRW

The system SHALL provide:
- HVR: 74%
- VRW: 0.55

#### Scenario: Get Commodity HVR
- **WHEN** calling `get_hvr(RiskClass.COMMODITY)`
- **THEN** 0.74 is returned

### Requirement: FX Risk Weights

The system SHALL provide FX delta risk weights based on volatility groups as specified in SIMM v2.6 Section I.1:

| Currency Vol Group | Calc Currency Vol Group | Risk Weight |
|--------------------|------------------------|-------------|
| Regular | Regular | 7.4 |
| Regular | High | 14.7 |
| High | Regular | 14.7 |
| High | High | 21.4 |

#### Scenario: Get FX risk weight for regular-regular
- **GIVEN** currency "EUR" with calculation currency "USD"
- **WHEN** calling `get_fx_risk_weight("EUR", "USD")`
- **THEN** 7.4 is returned

#### Scenario: Get FX risk weight for high volatility
- **GIVEN** currency "BRL" with calculation currency "USD"
- **WHEN** calling `get_fx_risk_weight("BRL", "USD")`
- **THEN** 14.7 is returned

### Requirement: FX Correlations

The system SHALL provide FX correlations based on volatility groups as specified in SIMM v2.6 Section I.2:

**Regular calculation currency:**
| Vol Group | Regular | High |
|-----------|---------|------|
| Regular | 50% | 25% |
| High | 25% | -5% |

**High volatility calculation currency:**
| Vol Group | Regular | High |
|-----------|---------|------|
| Regular | 88% | 72% |
| High | 72% | 50% |

FX vega/curvature correlation: 50%

#### Scenario: Get FX correlation with regular calc currency
- **GIVEN** two regular vol currencies and USD calc currency
- **WHEN** calling `get_fx_correlation("EUR", "GBP", "USD")`
- **THEN** 0.50 is returned

### Requirement: FX HVR and VRW

The system SHALL provide:
- HVR: 57%
- VRW: 0.48

#### Scenario: Get FX HVR
- **WHEN** calling `get_hvr(RiskClass.FX)`
- **THEN** 0.57 is returned

### Requirement: Inter-Risk-Class Correlations

The system SHALL provide the inter-risk-class correlation matrix (ψ_rs) as specified in SIMM v2.6 Section K:

|            | IR   | CreditQ | CreditNQ | Equity | Commodity | FX   |
|------------|------|---------|----------|--------|-----------|------|
| IR         | -    | 4%      | 4%       | 7%     | 37%       | 14%  |
| CreditQ    | 4%   | -       | 54%      | 70%    | 27%       | 37%  |
| CreditNQ   | 4%   | 54%     | -        | 46%    | 24%       | 15%  |
| Equity     | 7%   | 70%     | 46%      | -      | 35%       | 39%  |
| Commodity  | 37%  | 27%     | 24%      | 35%    | -         | 35%  |
| FX         | 14%  | 37%     | 15%      | 39%    | 35%       | -    |

#### Scenario: Get inter-risk-class correlation
- **GIVEN** IR and Equity risk classes
- **WHEN** calling `get_inter_risk_class_correlation(RiskClass.INTEREST_RATE, RiskClass.EQUITY)`
- **THEN** 0.07 is returned

### Requirement: Delta Concentration Thresholds

The system SHALL provide delta concentration thresholds by risk class and bucket as specified in SIMM v2.6 Section J.

#### Scenario: Get IR delta concentration threshold
- **GIVEN** currency "USD" (regular volatility, well-traded)
- **WHEN** calling `get_delta_concentration_threshold(RiskClass.INTEREST_RATE, "USD")`
- **THEN** 330 (USD mm / bp) is returned

#### Scenario: Get Equity delta concentration threshold
- **GIVEN** bucket 5 (DM Large Cap)
- **WHEN** calling `get_delta_concentration_threshold(RiskClass.EQUITY, 5)`
- **THEN** 12 (USD mm / %) is returned

### Requirement: Vega Concentration Thresholds

The system SHALL provide vega concentration thresholds by risk class and bucket as specified in SIMM v2.6 Section J.

#### Scenario: Get IR vega concentration threshold
- **GIVEN** currency "USD" (regular volatility, well-traded)
- **WHEN** calling `get_vega_concentration_threshold(RiskClass.INTEREST_RATE, "USD")`
- **THEN** 4900 (USD mm) is returned

#### Scenario: Get Equity vega concentration threshold
- **GIVEN** bucket 5 (DM Large Cap)
- **WHEN** calling `get_vega_concentration_threshold(RiskClass.EQUITY, 5)`
- **THEN** 1300 (USD mm) is returned

### Requirement: Unified Parameter Accessor

The system SHALL provide a unified accessor function that retrieves any calibration parameter:

```python
def get_calibration_param(
    param_type: str,  # "risk_weight", "intra_corr", "inter_corr", etc.
    risk_class: RiskClass,
    **kwargs
) -> float
```

#### Scenario: Unified access to risk weight
- **GIVEN** param_type="risk_weight", risk_class=EQUITY, bucket=5
- **WHEN** calling `get_calibration_param("risk_weight", RiskClass.EQUITY, bucket=5)`
- **THEN** 26 is returned

### Requirement: Calibration Data Immutability

All calibration data SHALL be immutable (frozen dataclasses, tuples, or frozensets) to prevent accidental modification during SIMM calculation.

#### Scenario: Attempt to modify calibration data
- **GIVEN** any calibration data structure
- **WHEN** attempting to modify a value
- **THEN** an error is raised or the operation has no effect

### Requirement: Matrix Validity Checks

All correlation matrices SHALL be symmetric and positive semi-definite. The system SHALL provide validation functions to verify matrix properties.

#### Scenario: Verify correlation matrix symmetry
- **GIVEN** any correlation matrix
- **WHEN** calling `validate_correlation_matrix(matrix)`
- **THEN** the function returns True if symmetric, raises error otherwise

#### Scenario: Verify positive semi-definite
- **GIVEN** any correlation matrix
- **WHEN** checking eigenvalues
- **THEN** all eigenvalues are non-negative

