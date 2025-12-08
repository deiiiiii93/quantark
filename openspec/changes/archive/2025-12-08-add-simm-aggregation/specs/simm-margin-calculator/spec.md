# SIMM Margin Calculator

This capability provides the core SIMM margin calculation engine implementing all aggregation formulas from ISDA SIMM v2.6 Sections B and 5-13.

## ADDED Requirements

### Requirement: SIMM Calculator Protocol

The system SHALL provide a `SIMMCalculator` class that calculates total SIMM and all component margins from input sensitivities.

#### Scenario: Calculate SIMM from sensitivity collection
- **GIVEN** a SensitivityCollection with delta, vega, and curvature sensitivities
- **AND** a SIMMConfig with default settings
- **WHEN** calling `calculator.calculate(sensitivities)`
- **THEN** a SIMMResult is returned with total SIMM and breakdown by product class

#### Scenario: Calculate SIMM from CRIF records
- **GIVEN** a list of CRIFRecord objects
- **WHEN** calling `calculator.calculate_from_crif(crif_records)`
- **THEN** records are converted to sensitivities and SIMM is calculated

### Requirement: Concentration Risk Factor Calculation

The system SHALL calculate concentration risk factors (CR) per SIMM paragraphs 7-8.

**For Interest Rate (paragraph 7):**
```
CR_b = max(1, sqrt(|Σ_{k,i} s_{k,i}| / T_b))
```
where the sum is over all tenors k and sub-curves i in currency b, including inflation but excluding cross-currency basis.

**For Credit spread risk (paragraph 8):**
```
CR_k = max(1, sqrt(|Σ_j s_j| / T_b))
```
where j sums over all risk factors with same issuer and seniority as k.

**For Equity, Commodity, FX (paragraph 8):**
```
CR_k = max(1, sqrt(|s_k| / T_b))
```

#### Scenario: Calculate IR concentration risk
- **GIVEN** USD IR sensitivities totaling 500 million USD
- **AND** USD concentration threshold = 330 million USD
- **WHEN** calculating CR for USD bucket
- **THEN** CR = max(1, sqrt(500/330)) = 1.23

#### Scenario: Concentration risk floor at 1
- **GIVEN** sensitivities below threshold
- **WHEN** calculating CR
- **THEN** CR = 1.0 (not less than 1)

#### Scenario: Base correlation CR is 1
- **GIVEN** base correlation sensitivities
- **WHEN** calculating CR
- **THEN** CR = 1.0 (no concentration scaling for base corr)

### Requirement: Weighted Sensitivity Calculation

The system SHALL calculate weighted sensitivities as:
```
WS_k = RW_k × s_k × CR_k
```

For IR cross-currency basis sensitivities, concentration risk factor SHALL NOT be applied.

#### Scenario: Calculate weighted sensitivity
- **GIVEN** an equity sensitivity s = $1000, RW = 26, CR = 1.5
- **WHEN** calculating weighted sensitivity
- **THEN** WS = 26 × 1000 × 1.5 = $39,000

#### Scenario: IR xccy basis without CR scaling
- **GIVEN** a cross-currency basis sensitivity
- **WHEN** calculating weighted sensitivity
- **THEN** WS = RW × s (CR not applied)

### Requirement: Bucket-Level Aggregation

The system SHALL aggregate weighted sensitivities within each bucket using:
```
K_b = sqrt(Σ_k WS_k² + Σ_k Σ_{l≠k} ρ_kl × f_kl × WS_k × WS_l)
```

where:
```
f_kl = min(CR_k, CR_l) / max(CR_k, CR_l)
```

For IR risk class, f_kl = 1 (no concentration adjustment at intra-bucket level).

#### Scenario: Aggregate two sensitivities in same bucket
- **GIVEN** WS_1 = 100, WS_2 = 200, ρ = 0.5, CR_1 = CR_2 = 1
- **WHEN** calculating K_b
- **THEN** K_b = sqrt(100² + 200² + 2 × 0.5 × 1 × 100 × 200) = sqrt(60000) = 245

#### Scenario: Apply f_kl concentration adjustment
- **GIVEN** CR_1 = 1.5, CR_2 = 1.0
- **WHEN** calculating f_12
- **THEN** f_12 = min(1.5, 1.0) / max(1.5, 1.0) = 0.67

### Requirement: Risk Class Delta Margin Aggregation

The system SHALL aggregate across buckets within a risk class using:
```
DeltaMargin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × S_b × S_c) + K_residual
```

where:
```
S_b = max(min(Σ_k WS_k, K_b), -K_b)
```

For IR risk class, include g_bc factor:
```
DeltaMargin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × g_bc × S_b × S_c)
```
where:
```
g_bc = min(CR_b, CR_c) / max(CR_b, CR_c)
```

#### Scenario: Aggregate equity buckets
- **GIVEN** K values for buckets 5 and 6
- **AND** inter-bucket correlation γ_56 = 0.29
- **WHEN** calculating DeltaMargin
- **THEN** the formula with γ_bc is applied

#### Scenario: IR aggregation with g_bc
- **GIVEN** USD and EUR IR buckets with CR_USD = 1.2, CR_EUR = 1.0
- **WHEN** aggregating across currencies
- **THEN** g_bc = min(1.2, 1.0) / max(1.2, 1.0) = 0.83 is applied

#### Scenario: Residual bucket added separately
- **GIVEN** non-residual DeltaMargin = 1000
- **AND** K_residual = 200
- **WHEN** calculating total DeltaMargin
- **THEN** total = 1000 + 200 = 1200

### Requirement: Risk Class Vega Margin Aggregation

The system SHALL calculate Vega Margin using the same aggregation structure as Delta:
```
VegaMargin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × g_bc × S_b × S_c) + K_residual
```

where g_bc factors apply only for IR risk class.

#### Scenario: Calculate Vega Margin
- **GIVEN** vega sensitivities across multiple buckets
- **WHEN** calculating VegaMargin
- **THEN** aggregation follows same structure as Delta with vega-specific K_b values

### Requirement: Risk Class Curvature Margin Aggregation

The system SHALL calculate Curvature Margin using squared correlations per paragraph 11:

```
K_b = sqrt(Σ_k CVR_{b,k}² + Σ_k Σ_{l≠k} ρ_kl² × CVR_{b,k} × CVR_{b,l})
```

```
θ = min(Σ_{b,k} CVR_{b,k} / Σ_{b,k} |CVR_{b,k}|, 0)
λ = (Φ^(-1)(99.5%)² - 1)(1 + θ) - θ
```

```
CurvatureMargin_non-res = max(Σ CVR + λ × sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc² × S_b × S_c), 0)
```

Total CurvatureMargin = CurvatureMargin_non-res + CurvatureMargin_residual

For IR only, multiply final curvature margin by HVR_IR^(-2).

#### Scenario: Calculate θ for all positive CVRs
- **GIVEN** all CVR values are positive (net long gamma)
- **WHEN** calculating θ
- **THEN** θ = min(Σ CVR / Σ|CVR|, 0) = min(1, 0) = 0

#### Scenario: Calculate θ for mixed CVRs
- **GIVEN** CVR sum = -100, |CVR| sum = 500
- **WHEN** calculating θ
- **THEN** θ = min(-100/500, 0) = -0.2

#### Scenario: Apply squared correlations in curvature
- **GIVEN** intra-bucket correlation ρ = 0.5
- **WHEN** calculating curvature K_b
- **THEN** ρ² = 0.25 is used in the formula

#### Scenario: IR curvature HVR scaling
- **GIVEN** IR CurvatureMargin = 100, HVR_IR = 0.47
- **WHEN** applying scaling
- **THEN** final margin = 100 × (0.47)^(-2) = 453

### Requirement: Base Correlation Margin Calculation

The system SHALL calculate Base Correlation Margin for Credit Qualifying per paragraph 13:
```
BaseCorrMargin = sqrt(Σ_k WS_k² + Σ_k Σ_{l≠k} ρ_kl × WS_k × WS_l)
```

where WS_k = RW × s_k (no concentration risk).

#### Scenario: Calculate base correlation margin
- **GIVEN** BC01 sensitivities to CDX IG and iTraxx Main
- **AND** inter-index correlation = 29%
- **WHEN** calculating BaseCorrMargin
- **THEN** aggregation formula is applied with ρ = 0.29

### Requirement: Product Class Aggregation

The system SHALL aggregate across risk classes within a product class using:
```
SIMM_product = sqrt(Σ_r IM_r² + Σ_r Σ_{s≠r} ψ_rs × IM_r × IM_s)
```

where IM_r = DeltaMargin_r + VegaMargin_r + CurvatureMargin_r + BaseCorrMargin_r

#### Scenario: Aggregate equity product class
- **GIVEN** Equity IM = 1000, FX IM = 500
- **AND** ψ(Equity, FX) = 39%
- **WHEN** calculating SIMM_equity product class
- **THEN** SIMM = sqrt(1000² + 500² + 2 × 0.39 × 1000 × 500) = 1278

### Requirement: Total SIMM Calculation

The system SHALL calculate total SIMM as the sum of product class SIMMs:
```
SIMM = SIMM_RatesFX + SIMM_Credit + SIMM_Equity + SIMM_Commodity
```

#### Scenario: Sum product class SIMMs
- **GIVEN** SIMM_RatesFX = 100, SIMM_Credit = 200, SIMM_Equity = 150, SIMM_Commodity = 50
- **WHEN** calculating total SIMM
- **THEN** SIMM = 100 + 200 + 150 + 50 = 500

### Requirement: Add-On Calculation

The system SHALL support additional initial margin calculations per Section L:
```
Additional IM = AddOnFixed + Σ_p (AddOnFactor_p × Notional_p) + 
                (MS_RatesFX - 1) × SIMM_RatesFX + ... 
```

#### Scenario: Apply multiplicative scale
- **GIVEN** SIMM_Equity = 1000, MS_Equity = 1.2
- **WHEN** calculating with multiplier
- **THEN** contribution = 1.2 × 1000 = 1200

#### Scenario: Apply notional-based add-on
- **GIVEN** product p with notional = 10M, AddOnFactor = 5%
- **WHEN** calculating add-on
- **THEN** add-on contribution = 0.05 × 10M = 500K

### Requirement: Empty Bucket Handling

The system SHALL handle empty buckets (no sensitivities) gracefully by excluding them from aggregation.

#### Scenario: Portfolio with single bucket exposure
- **GIVEN** equity sensitivities only in bucket 5
- **WHEN** calculating DeltaMargin
- **THEN** only bucket 5 contributes to K and no cross-bucket terms exist

### Requirement: Numerical Precision

The system SHALL ensure numerical stability:
- Use max(0, x) before taking square root
- Handle very small sensitivities (< 1e-10) appropriately
- Use double precision for all calculations

#### Scenario: Negative value under square root
- **GIVEN** numerical errors result in negative variance
- **WHEN** calculating K_b
- **THEN** sqrt(max(0, variance)) is used

### Requirement: Calculation Traceability

The system SHALL provide detailed calculation trace including:
- Intermediate K_b values per bucket
- S_b values per bucket
- Risk class margins before product aggregation
- Concentration risk factors used

#### Scenario: Debug calculation
- **GIVEN** a SIMM calculation
- **WHEN** requesting trace
- **THEN** all intermediate values are accessible for validation

### Requirement: Risk Class Margin Components

The system SHALL calculate and report margin for each risk class as:
```
IM_X = DeltaMargin_X + VegaMargin_X + CurvatureMargin_X + BaseCorrMargin_X
```

#### Scenario: Report margin breakdown
- **GIVEN** an equity portfolio with options
- **WHEN** calculating SIMM
- **THEN** result includes separate Delta, Vega, Curvature margins for Equity risk class

### Requirement: Product Class Assignment

The system SHALL assign trades to product classes based on primary risk:
- IR derivatives, FX derivatives → RatesFX
- Credit derivatives → Credit
- Equity derivatives → Equity
- Commodity derivatives → Commodity

Each trade's sensitivities stay within its assigned product class.

#### Scenario: Equity option in Equity product class
- **GIVEN** an equity option position
- **WHEN** assigning product class
- **THEN** all sensitivities (equity delta, FX delta, IR delta) go to Equity product class

