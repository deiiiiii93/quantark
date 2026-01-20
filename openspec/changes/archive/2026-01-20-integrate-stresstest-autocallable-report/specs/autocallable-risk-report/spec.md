## MODIFIED Requirements
### Requirement: Stress scenarios and conditional cashflows
The system SHALL generate a stress scenario table using the stresstest framework (ScenarioBuilder + EquityStressEngine), and SHALL accept either stresstest Scenario inputs or simple shock configs converted to scenarios.

#### Scenario: Stresstest-backed scenario table
- **GIVEN** a SnowballOption, PricingEnvironment, and stresstest scenarios
- **WHEN** the report generates the stress scenario table
- **THEN** the table reflects stresstest results and scenario metadata
