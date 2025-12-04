# SIMM Risk Taxonomy

This capability defines the complete taxonomy for ISDA SIMM risk classification including risk classes, product classes, margin types, buckets, tenors, and currency classifications.

## ADDED Requirements

### Requirement: Risk Class Enumeration

The system SHALL provide a `RiskClass` enum with the following values representing the six SIMM risk classes:
- `INTEREST_RATE` (code: "IR")
- `CREDIT_QUALIFYING` (code: "CreditQ")
- `CREDIT_NON_QUALIFYING` (code: "CreditNQ")
- `EQUITY` (code: "Equity")
- `COMMODITY` (code: "Commodity")
- `FX` (code: "FX")

#### Scenario: Enumerate all risk classes
- **GIVEN** the SIMM taxonomy module is imported
- **WHEN** iterating over RiskClass enum
- **THEN** exactly 6 risk classes are returned
- **AND** each has a unique string code

#### Scenario: Risk class code lookup
- **GIVEN** a RiskClass enum value
- **WHEN** accessing its code via `.value`
- **THEN** the corresponding SIMM specification code is returned (e.g., "IR" for INTEREST_RATE)

### Requirement: Product Class Enumeration

The system SHALL provide a `ProductClass` enum with the following values representing the four SIMM product classes:
- `RATES_FX` (code: "RatesFX")
- `CREDIT` (code: "Credit")
- `EQUITY` (code: "Equity")
- `COMMODITY` (code: "Commodity")

#### Scenario: Enumerate product classes
- **GIVEN** the SIMM taxonomy module is imported
- **WHEN** iterating over ProductClass enum
- **THEN** exactly 4 product classes are returned

#### Scenario: Map risk class to product class
- **GIVEN** a RiskClass value
- **WHEN** looking up its corresponding ProductClass
- **THEN** IR and FX map to RATES_FX
- **AND** CreditQ and CreditNQ map to CREDIT
- **AND** Equity maps to EQUITY
- **AND** Commodity maps to COMMODITY

### Requirement: Margin Type Enumeration

The system SHALL provide a `MarginType` enum with the following values:
- `DELTA` - First-order sensitivity margin
- `VEGA` - Volatility sensitivity margin
- `CURVATURE` - Second-order volatility margin
- `BASE_CORR` - Base correlation margin (Credit Qualifying only)

#### Scenario: Enumerate margin types
- **GIVEN** the SIMM taxonomy module is imported
- **WHEN** iterating over MarginType enum
- **THEN** exactly 4 margin types are returned

### Requirement: Sensitivity Type Enumeration

The system SHALL provide a `SensitivityType` enum with the following CRIF risk type codes:
- `RISK_IR_CURVE` (code: "Risk_IRCurve")
- `RISK_IR_VOL` (code: "Risk_IRVol")
- `RISK_INFLATION` (code: "Risk_Inflation")
- `RISK_INFLATION_VOL` (code: "Risk_InflationVol")
- `RISK_XCCY_BASIS` (code: "Risk_XCcyBasis")
- `RISK_CREDIT_Q` (code: "Risk_CreditQ")
- `RISK_CREDIT_VOL` (code: "Risk_CreditVol")
- `RISK_CREDIT_NQ` (code: "Risk_CreditNonQ")
- `RISK_CREDIT_NQ_VOL` (code: "Risk_CreditVolNonQ")
- `RISK_EQUITY` (code: "Risk_Equity")
- `RISK_EQUITY_VOL` (code: "Risk_EquityVol")
- `RISK_COMMODITY` (code: "Risk_Commodity")
- `RISK_COMMODITY_VOL` (code: "Risk_CommodityVol")
- `RISK_FX` (code: "Risk_FX")
- `RISK_FX_VOL` (code: "Risk_FXVol")
- `RISK_BASE_CORR` (code: "Risk_BaseCorr")

#### Scenario: Map sensitivity type to risk class
- **GIVEN** a SensitivityType value
- **WHEN** determining its RiskClass
- **THEN** the correct risk class is returned (e.g., RISK_IR_CURVE -> INTEREST_RATE)

#### Scenario: Map sensitivity type to margin type
- **GIVEN** a SensitivityType value
- **WHEN** determining its MarginType
- **THEN** curve/delta types map to DELTA
- **AND** vol types map to VEGA
- **AND** base corr maps to BASE_CORR

### Requirement: Interest Rate Tenor Definitions

The system SHALL define IR tenor vertices as specified in SIMM v2.6:
- Tenors: 2w, 1m, 3m, 6m, 1yr, 2yr, 3yr, 5yr, 10yr, 15yr, 20yr, 30yr
- Numeric values in years: 0.0384 (14/365), 0.0833, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0

#### Scenario: Access IR tenor values
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `IR_TENORS`
- **THEN** a tuple of 12 numeric values in years is returned

#### Scenario: Access IR tenor labels
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `IR_TENOR_LABELS`
- **THEN** a tuple of 12 string labels ("2w", "1m", ..., "30yr") is returned

### Requirement: Credit Tenor Definitions

The system SHALL define Credit tenor vertices as specified in SIMM v2.6:
- Tenors: 1yr, 2yr, 3yr, 5yr, 10yr
- Numeric values in years: 1.0, 2.0, 3.0, 5.0, 10.0

#### Scenario: Access Credit tenor values
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `CREDIT_TENORS`
- **THEN** a tuple of 5 numeric values in years is returned

### Requirement: Interest Rate Sub-Curve Definitions

The system SHALL define IR sub-curves for rate curve decomposition:
- `OIS` - Overnight indexed swap curve
- `LIBOR_1M` - 1-month LIBOR
- `LIBOR_3M` - 3-month LIBOR
- `LIBOR_6M` - 6-month LIBOR
- `LIBOR_12M` - 12-month LIBOR
- `PRIME` - Prime rate (USD only)
- `MUNICIPAL` - Municipal rate (USD only)

#### Scenario: Enumerate sub-curves
- **GIVEN** the taxonomy module is imported
- **WHEN** iterating over `IRSubCurve` enum
- **THEN** all 7 sub-curve types are available

### Requirement: Currency Volatility Classification

The system SHALL classify currencies into volatility groups as per SIMM v2.6:
- **Low volatility**: JPY
- **Regular volatility**: USD, EUR, GBP, CHF, AUD, NZD, CAD, SEK, NOK, DKK, HKD, KRW, SGD, TWD
- **High volatility**: All other currencies

#### Scenario: Classify currency volatility
- **GIVEN** a currency code string
- **WHEN** calling `get_currency_volatility_group(currency)`
- **THEN** "low", "regular", or "high" is returned based on SIMM classification

#### Scenario: JPY is low volatility
- **GIVEN** currency code "JPY"
- **WHEN** calling `get_currency_volatility_group("JPY")`
- **THEN** "low" is returned

#### Scenario: USD is regular volatility
- **GIVEN** currency code "USD"
- **WHEN** calling `get_currency_volatility_group("USD")`
- **THEN** "regular" is returned

#### Scenario: BRL is high volatility
- **GIVEN** currency code "BRL"
- **WHEN** calling `get_currency_volatility_group("BRL")`
- **THEN** "high" is returned

### Requirement: Credit Qualifying Bucket Definitions

The system SHALL define Credit Qualifying buckets as per SIMM v2.6 Section E:
- Buckets 1-6: Investment Grade (IG) by sector
- Buckets 7-12: High Yield/Non-Rated (HY/NR) by sector
- Residual bucket for unclassified

| Bucket | Credit Quality | Sector |
|--------|---------------|--------|
| 1 | IG | Sovereigns including central banks |
| 2 | IG | Financials including government-backed financials |
| 3 | IG | Basic materials, energy, industrials |
| 4 | IG | Consumer |
| 5 | IG | Technology, telecommunications |
| 6 | IG | Health care, utilities, local government, government-backed corporates |
| 7 | HY/NR | Sovereigns including central banks |
| 8 | HY/NR | Financials including government-backed financials |
| 9 | HY/NR | Basic materials, energy, industrials |
| 10 | HY/NR | Consumer |
| 11 | HY/NR | Technology, telecommunications |
| 12 | HY/NR | Health care, utilities, local government, government-backed corporates |
| Residual | - | Unclassified |

#### Scenario: Access Credit Qualifying bucket definitions
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `CREDIT_QUALIFYING_BUCKETS`
- **THEN** a dictionary mapping bucket numbers (1-12, "Residual") to bucket metadata is returned

### Requirement: Equity Bucket Definitions

The system SHALL define Equity buckets as per SIMM v2.6 Section G:

| Bucket | Size | Region | Sector |
|--------|------|--------|--------|
| 1-4 | Large | Emerging Markets | By sector |
| 5-8 | Large | Developed Markets | By sector |
| 9 | Small | Emerging Markets | All sectors |
| 10 | Small | Developed Markets | All sectors |
| 11 | All | All | Indexes, Funds, ETFs |
| 12 | All | All | Volatility Indexes |
| Residual | - | - | Unclassified |

#### Scenario: Access Equity bucket definitions
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `EQUITY_BUCKETS`
- **THEN** a dictionary mapping bucket numbers (1-12, "Residual") to bucket metadata is returned

#### Scenario: Determine if equity is large or small cap
- **GIVEN** an equity with market cap >= USD 2 billion
- **WHEN** classifying by size
- **THEN** it is classified as "Large"

### Requirement: Commodity Bucket Definitions

The system SHALL define Commodity buckets as per SIMM v2.6 Section H:

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

#### Scenario: Access Commodity bucket definitions
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `COMMODITY_BUCKETS`
- **THEN** a dictionary mapping bucket numbers (1-17) to bucket metadata is returned

### Requirement: Credit Non-Qualifying Bucket Definitions

The system SHALL define Credit Non-Qualifying buckets as per SIMM v2.6 Section F:

| Bucket | Credit Quality | Sector |
|--------|---------------|--------|
| 1 | IG | RMBS/CMBS |
| 2 | HY/NR | RMBS/CMBS |
| Residual | - | Unclassified |

#### Scenario: Access Credit Non-Qualifying bucket definitions
- **GIVEN** the taxonomy module is imported
- **WHEN** accessing `CREDIT_NON_QUALIFYING_BUCKETS`
- **THEN** a dictionary mapping bucket numbers (1, 2, "Residual") to bucket metadata is returned

### Requirement: FX Volatility Category Classification

The system SHALL classify currencies into FX concentration threshold categories:
- **Category 1** (Significantly material): USD, EUR, JPY, GBP, AUD, CHF, CAD
- **Category 2** (Frequently traded): BRL, CNY, HKD, INR, KRW, MXN, NOK, NZD, RUB, SEK, SGD, TRY, ZAR
- **Category 3** (Others): All other currencies

#### Scenario: Classify FX category
- **GIVEN** a currency code string
- **WHEN** calling `get_fx_category(currency)`
- **THEN** 1, 2, or 3 is returned based on SIMM classification

### Requirement: FX High Volatility Classification

The system SHALL identify high FX volatility currencies as per SIMM v2.6 Section I:
- High FX volatility: BRL, RUB, TRY
- Regular FX volatility: All other currencies

#### Scenario: Check FX volatility group
- **GIVEN** a currency code string
- **WHEN** calling `get_fx_volatility_group(currency)`
- **THEN** "high" or "regular" is returned

### Requirement: Developed vs Emerging Market Classification

The system SHALL classify regions for equity bucket assignment:
- **Developed Markets**: Canada, US, Mexico, euro area, UK, Norway, Sweden, Denmark, Switzerland, Japan, Australia, New Zealand, Singapore, Hong Kong
- **Emerging Markets**: All other regions/countries

#### Scenario: Check market classification
- **GIVEN** a country or region identifier
- **WHEN** calling `is_developed_market(region)`
- **THEN** True is returned for developed markets, False otherwise

