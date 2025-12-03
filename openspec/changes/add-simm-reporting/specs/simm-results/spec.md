# SIMM Results

This capability provides comprehensive result dataclasses for SIMM calculation including hierarchical margin breakdown and what-if analysis.

## ADDED Requirements

### Requirement: SIMMResult Dataclass

The system SHALL provide a `SIMMResult` dataclass containing the complete SIMM calculation output with the following structure:
- `total_simm`: Total SIMM amount (float)
- `calculation_currency`: Currency of calculation (str)
- `calculation_date`: Date of calculation (date)
- `simm_version`: SIMM version used (str, e.g., "2.6")
- `product_class_simm`: Breakdown by product class (Dict[ProductClass, float])
- `risk_class_margin`: Detailed margin by risk class within each product class
- `addon_amount`: Total add-on amount (float)
- `attribution`: Optional SIMMAttribution object
- `execution_time_seconds`: Calculation time (float)
- `warnings`: List of calculation warnings

#### Scenario: Create SIMMResult from calculation
- **GIVEN** a completed SIMM calculation
- **WHEN** creating a SIMMResult
- **THEN** all margin components are captured
- **AND** total_simm equals sum of product_class_simm values plus addon

#### Scenario: Access product class breakdown
- **GIVEN** a SIMMResult
- **WHEN** accessing product_class_simm
- **THEN** a dictionary mapping ProductClass to margin is returned
- **AND** all four product classes are present (with 0 for empty)

### Requirement: RiskClassMargin Dataclass

The system SHALL provide a `RiskClassMargin` dataclass with margin components:
- `delta_margin`: Delta margin amount (float)
- `vega_margin`: Vega margin amount (float)
- `curvature_margin`: Curvature margin amount (float)
- `base_corr_margin`: Base correlation margin, Credit Q only (float)
- `total_margin`: Sum of all components (float)
- `bucket_detail`: Breakdown by bucket

#### Scenario: Access margin components
- **GIVEN** a RiskClassMargin for Equity risk class
- **WHEN** accessing margin components
- **THEN** delta, vega, curvature margins are available
- **AND** total_margin = delta + vega + curvature

#### Scenario: Base correlation for Credit Q only
- **GIVEN** a RiskClassMargin for Interest Rate
- **WHEN** accessing base_corr_margin
- **THEN** value is 0 (not applicable)

### Requirement: BucketDetail Dataclass

The system SHALL provide a `BucketDetail` dataclass with bucket-level calculation details:
- `bucket`: Bucket identifier (int or str)
- `k_value`: Bucket-level K aggregation result (float)
- `s_value`: Capped sensitivity sum S_b (float)
- `ws_sum`: Sum of weighted sensitivities (float)
- `concentration_factor`: Concentration risk factor CR (float)
- `sensitivities`: List of contributing sensitivities

#### Scenario: Access bucket K value
- **GIVEN** a BucketDetail for Equity bucket 5
- **WHEN** accessing k_value
- **THEN** the aggregated bucket K is returned

#### Scenario: Verify S_b capping
- **GIVEN** a bucket with ws_sum = 500 and k_value = 300
- **WHEN** accessing s_value
- **THEN** s_value = 300 (capped at K_b)

### Requirement: Result Serialization

The system SHALL provide serialization methods for SIMMResult:
- `to_dict()`: Convert to nested dictionary
- `to_json()`: Convert to JSON string
- `from_dict(data)`: Create from dictionary (class method)

#### Scenario: Serialize result to JSON
- **GIVEN** a SIMMResult
- **WHEN** calling to_json()
- **THEN** valid JSON string is returned
- **AND** all nested structures are properly serialized

#### Scenario: Round-trip serialization
- **GIVEN** a SIMMResult
- **WHEN** converting to dict and back
- **THEN** the recreated result equals the original

### Requirement: Result Summary Methods

The system SHALL provide summary methods on SIMMResult:
- `get_margin_by_risk_class()`: Dict[RiskClass, float]
- `get_margin_by_margin_type()`: Dict[MarginType, float]
- `get_top_buckets(n: int)`: List of top n buckets by margin

#### Scenario: Get margin by risk class
- **GIVEN** a SIMMResult
- **WHEN** calling get_margin_by_risk_class()
- **THEN** aggregated margin per risk class is returned

#### Scenario: Get top buckets
- **GIVEN** a SIMMResult with multiple buckets
- **WHEN** calling get_top_buckets(5)
- **THEN** top 5 buckets by margin contribution are returned

### Requirement: What-If Analysis Class

The system SHALL provide a `SIMMWhatIf` class for impact analysis:
- `impact_of_adding(sensitivities)`: Calculate SIMM change from adding sensitivities
- `impact_of_removing(position_ids)`: Calculate SIMM change from removing positions
- `marginal_simm(sensitivity)`: Calculate marginal SIMM for single sensitivity

#### Scenario: Calculate impact of adding trade
- **GIVEN** a base SIMMResult and new sensitivities
- **WHEN** calling impact_of_adding(new_sensitivities)
- **THEN** a WhatIfResult is returned with delta_simm

#### Scenario: Calculate impact of removing position
- **GIVEN** a base result including position P001
- **WHEN** calling impact_of_removing(["P001"])
- **THEN** the SIMM impact of removing P001 is calculated

### Requirement: WhatIfResult Dataclass

The system SHALL provide a `WhatIfResult` dataclass with:
- `base_simm`: Original SIMM (float)
- `new_simm`: SIMM after change (float)
- `delta_simm`: Change in SIMM (float)
- `delta_pct`: Percentage change (float)
- `delta_by_product_class`: Change breakdown by product class
- `delta_by_risk_class`: Change breakdown by risk class

#### Scenario: Analyze what-if result
- **GIVEN** a WhatIfResult
- **WHEN** accessing delta_simm
- **THEN** new_simm - base_simm is returned

#### Scenario: Breakdown of change
- **GIVEN** a what-if result from adding EUR IR exposure
- **WHEN** accessing delta_by_risk_class
- **THEN** IR risk class shows the primary impact

### Requirement: Result Validation

The system SHALL validate result consistency:
- Total equals sum of product class margins plus addon
- Risk class margin equals sum of margin components
- All required fields are populated

#### Scenario: Validate result consistency
- **GIVEN** a SIMMResult
- **WHEN** calling validate()
- **THEN** True is returned if consistent
- **AND** ValidationError is raised if inconsistent

### Requirement: Calculation Warnings

The system SHALL capture calculation warnings in the result:
- Missing market data warnings
- Bucket classification fallbacks
- Numerical precision issues

#### Scenario: Access calculation warnings
- **GIVEN** a calculation with missing issuer classification
- **WHEN** accessing result.warnings
- **THEN** a warning about residual bucket assignment is present

### Requirement: Execution Metadata

The system SHALL capture execution metadata:
- Calculation timestamp
- Execution duration
- SIMM version
- Configuration used

#### Scenario: Access execution metadata
- **GIVEN** a SIMMResult
- **WHEN** accessing execution_time_seconds
- **THEN** the calculation duration in seconds is returned

### Requirement: Result Comparison

The system SHALL provide comparison methods:
- `compare(other: SIMMResult)`: Compare two results
- `diff(other: SIMMResult)`: Get detailed differences

#### Scenario: Compare two results
- **GIVEN** two SIMMResults from different dates
- **WHEN** calling result1.compare(result2)
- **THEN** a comparison summary with deltas is returned

