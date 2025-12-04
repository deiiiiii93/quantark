# SIMM CRIF Format

This capability defines the CRIF (Common Risk Interchange Format) data model and parser for importing/exporting SIMM sensitivities in the industry-standard format.

## ADDED Requirements

### Requirement: CRIF Record Data Model

The system SHALL provide a `CRIFRecord` dataclass representing a single CRIF sensitivity record with the following fields:

**Required Fields:**
- `trade_id`: str - Unique trade identifier
- `valuation_date`: date - Valuation date for the sensitivity
- `risk_type`: str - CRIF risk type code (e.g., "Risk_IRCurve", "Risk_FX")
- `qualifier`: str - Risk factor qualifier (currency for IR, issuer for Credit, etc.)
- `bucket`: str - Bucket identifier
- `label1`: str - Primary label (tenor for IR/Credit)
- `label2`: str - Secondary label (sub-curve for IR)
- `amount`: float - Sensitivity amount
- `amount_currency`: str - Currency of the amount

**Optional Fields:**
- `amount_usd`: Optional[float] - USD equivalent of amount
- `im_model`: str - Initial margin model (default: "SIMM")
- `product_class`: Optional[ProductClass] - Product class assignment
- `post_regulations`: Optional[str] - Post regulations identifier
- `collect_regulations`: Optional[str] - Collect regulations identifier
- `end_date`: Optional[date] - Trade end/maturity date
- `call_put`: Optional[str] - Call/Put indicator for options
- `notional`: Optional[float] - Trade notional

#### Scenario: Create CRIF record from dictionary
- **GIVEN** a dictionary with CRIF field values
- **WHEN** creating a CRIFRecord from the dictionary
- **THEN** all fields are correctly populated
- **AND** optional fields default to None or their default values

#### Scenario: Validate required fields
- **GIVEN** a dictionary missing required fields
- **WHEN** attempting to create a CRIFRecord
- **THEN** a ValidationError is raised indicating the missing field

### Requirement: CRIF Risk Type Validation

The system SHALL validate that `risk_type` values match valid SIMM sensitivity types:
- Risk_IRCurve, Risk_IRVol, Risk_Inflation, Risk_InflationVol, Risk_XCcyBasis
- Risk_CreditQ, Risk_CreditVol, Risk_CreditNonQ, Risk_CreditVolNonQ
- Risk_Equity, Risk_EquityVol
- Risk_Commodity, Risk_CommodityVol
- Risk_FX, Risk_FXVol
- Risk_BaseCorr

#### Scenario: Valid risk type accepted
- **GIVEN** a CRIF record with risk_type "Risk_IRCurve"
- **WHEN** validating the record
- **THEN** validation passes

#### Scenario: Invalid risk type rejected
- **GIVEN** a CRIF record with risk_type "Invalid_Type"
- **WHEN** validating the record
- **THEN** a ValidationError is raised

### Requirement: CRIF CSV Parser

The system SHALL provide a `CRIFParser` class that can parse CRIF CSV files with the following methods:
- `parse_file(filepath: str) -> List[CRIFRecord]` - Parse a CRIF CSV file
- `parse_string(csv_content: str) -> List[CRIFRecord]` - Parse CRIF from string
- `parse_dataframe(df: pd.DataFrame) -> List[CRIFRecord]` - Parse from DataFrame

#### Scenario: Parse CRIF CSV file
- **GIVEN** a valid CRIF CSV file path
- **WHEN** calling `parser.parse_file(filepath)`
- **THEN** a list of CRIFRecord objects is returned
- **AND** each record corresponds to a row in the CSV

#### Scenario: Parse CRIF with standard column names
- **GIVEN** a CRIF CSV with columns: TradeId, ValuationDate, RiskType, Qualifier, Bucket, Label1, Label2, Amount, AmountCurrency
- **WHEN** parsing the CSV
- **THEN** columns are correctly mapped to CRIFRecord fields

#### Scenario: Parse CRIF with alternative column names
- **GIVEN** a CRIF CSV with columns: trade_id, valuation_date, risk_type, qualifier, bucket, label1, label2, amount, amount_currency
- **WHEN** parsing the CSV
- **THEN** columns are correctly mapped using case-insensitive matching

#### Scenario: Handle missing optional columns
- **GIVEN** a CRIF CSV without optional columns (e.g., AmountUSD)
- **WHEN** parsing the CSV
- **THEN** optional fields are set to None/default values
- **AND** parsing completes successfully

### Requirement: CRIF Record Grouping

The system SHALL provide methods to group CRIF records by various dimensions:
- `group_by_trade(records: List[CRIFRecord]) -> Dict[str, List[CRIFRecord]]`
- `group_by_risk_class(records: List[CRIFRecord]) -> Dict[RiskClass, List[CRIFRecord]]`
- `group_by_product_class(records: List[CRIFRecord]) -> Dict[ProductClass, List[CRIFRecord]]`

#### Scenario: Group records by trade
- **GIVEN** a list of CRIF records from multiple trades
- **WHEN** calling `group_by_trade(records)`
- **THEN** a dictionary mapping trade_id to list of records is returned

#### Scenario: Group records by risk class
- **GIVEN** a list of CRIF records with different risk types
- **WHEN** calling `group_by_risk_class(records)`
- **THEN** a dictionary mapping RiskClass to list of records is returned
- **AND** Risk_IRCurve records are under INTEREST_RATE key

### Requirement: CRIF to Sensitivity Conversion

The system SHALL provide functions to convert CRIF records to internal Sensitivity objects:
- `crif_to_delta_sensitivity(record: CRIFRecord) -> DeltaSensitivity`
- `crif_to_vega_sensitivity(record: CRIFRecord) -> VegaSensitivity`
- `crif_to_sensitivities(records: List[CRIFRecord]) -> SensitivityCollection`

#### Scenario: Convert IR curve CRIF to delta sensitivity
- **GIVEN** a CRIFRecord with risk_type "Risk_IRCurve", qualifier "USD", label1 "5yr", label2 "OIS"
- **WHEN** calling `crif_to_delta_sensitivity(record)`
- **THEN** a DeltaSensitivity is returned with:
  - risk_class = RiskClass.INTEREST_RATE
  - currency = "USD"
  - tenor = 5.0
  - sub_curve = IRSubCurve.OIS
  - amount = record.amount

#### Scenario: Convert FX CRIF to delta sensitivity
- **GIVEN** a CRIFRecord with risk_type "Risk_FX", qualifier "EUR"
- **WHEN** calling `crif_to_delta_sensitivity(record)`
- **THEN** a DeltaSensitivity is returned with:
  - risk_class = RiskClass.FX
  - currency = "EUR"
  - amount = record.amount

#### Scenario: Convert vol CRIF to vega sensitivity
- **GIVEN** a CRIFRecord with risk_type "Risk_EquityVol", qualifier "AAPL", label1 "1yr"
- **WHEN** calling `crif_to_vega_sensitivity(record)`
- **THEN** a VegaSensitivity is returned with:
  - risk_class = RiskClass.EQUITY
  - option_expiry = 1.0
  - amount = record.amount

### Requirement: Sensitivity to CRIF Export

The system SHALL provide functions to export internal Sensitivity objects to CRIF format:
- `sensitivity_to_crif(sensitivity: Sensitivity, trade_id: str, valuation_date: date) -> CRIFRecord`
- `sensitivities_to_crif(sensitivities: SensitivityCollection, trade_id: str, valuation_date: date) -> List[CRIFRecord]`

#### Scenario: Export delta sensitivity to CRIF
- **GIVEN** an IR DeltaSensitivity with currency="USD", tenor=5.0, sub_curve=OIS, amount=10000
- **WHEN** calling `sensitivity_to_crif(sensitivity, "TRADE001", date(2024,1,1))`
- **THEN** a CRIFRecord is returned with:
  - trade_id = "TRADE001"
  - valuation_date = 2024-01-01
  - risk_type = "Risk_IRCurve"
  - qualifier = "USD"
  - label1 = "5yr"
  - label2 = "OIS"
  - amount = 10000

### Requirement: CRIF CSV Writer

The system SHALL provide a `CRIFWriter` class to export CRIF records to CSV:
- `write_file(records: List[CRIFRecord], filepath: str) -> None`
- `write_string(records: List[CRIFRecord]) -> str`
- `write_dataframe(records: List[CRIFRecord]) -> pd.DataFrame`

#### Scenario: Write CRIF to CSV file
- **GIVEN** a list of CRIFRecord objects
- **WHEN** calling `writer.write_file(records, "output.csv")`
- **THEN** a CSV file is created with standard CRIF column headers
- **AND** each record is written as a row

#### Scenario: Round-trip CRIF data
- **GIVEN** a list of CRIFRecord objects
- **WHEN** writing to CSV and parsing back
- **THEN** the parsed records match the original records

### Requirement: CRIF Validation Rules

The system SHALL validate CRIF records according to SIMM rules:
- IR records MUST have valid tenor in label1
- IR records MUST have valid sub-curve in label2
- Credit records MUST have valid bucket (1-12 for Q, 1-2 for NQ, or Residual)
- Equity records MUST have valid bucket (1-12 or Residual)
- Commodity records MUST have valid bucket (1-17)
- FX records MUST have valid currency qualifier

#### Scenario: Validate IR record with valid tenor
- **GIVEN** an IR CRIF record with label1 = "5yr"
- **WHEN** validating the record
- **THEN** validation passes

#### Scenario: Reject IR record with invalid tenor
- **GIVEN** an IR CRIF record with label1 = "7yr"
- **WHEN** validating the record
- **THEN** a ValidationError is raised indicating invalid tenor

#### Scenario: Validate Credit bucket
- **GIVEN** a Credit Qualifying CRIF record with bucket = "3"
- **WHEN** validating the record
- **THEN** validation passes

### Requirement: CRIF Netting

The system SHALL provide functions to net CRIF records by risk factor:
- `net_crif_records(records: List[CRIFRecord]) -> List[CRIFRecord]`

Records are netted when they share the same:
- risk_type
- qualifier
- bucket
- label1
- label2
- amount_currency

#### Scenario: Net identical risk factors
- **GIVEN** two CRIF records with identical risk factor dimensions but amounts 100 and -50
- **WHEN** calling `net_crif_records(records)`
- **THEN** a single record with amount 50 is returned

#### Scenario: Keep distinct risk factors separate
- **GIVEN** two CRIF records with different tenors (label1 = "5yr" vs "10yr")
- **WHEN** calling `net_crif_records(records)`
- **THEN** both records are retained unchanged

### Requirement: CRIF File Header Metadata

The system SHALL support optional CRIF file header metadata:
- `valuation_date`: date - Common valuation date for all records
- `calculation_currency`: str - Calculation currency for SIMM
- `im_model`: str - Initial margin model identifier
- `regulations`: str - Applicable regulations

#### Scenario: Parse CRIF with header section
- **GIVEN** a CRIF file with header metadata before the data rows
- **WHEN** parsing the file
- **THEN** header metadata is extracted and available
- **AND** data rows are parsed correctly

### Requirement: CRIF Error Handling

The system SHALL provide clear error messages for CRIF parsing failures:
- Line number where error occurred
- Column/field that failed validation
- Expected vs actual value
- Suggested correction if applicable

#### Scenario: Report parsing error with context
- **GIVEN** a CRIF CSV with invalid data on line 15
- **WHEN** parsing fails
- **THEN** the error message includes "Line 15" and the specific issue
- **AND** the problematic data is included in the error message

