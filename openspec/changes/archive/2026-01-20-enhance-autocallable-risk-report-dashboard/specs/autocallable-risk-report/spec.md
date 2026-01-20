## ADDED Requirements
### Requirement: Barrier proximity and zoom risk
The system SHALL report barrier proximity in sigma terms for the next KO and KI levels and SHALL compute a barrier-zoom grid around KI/KO (±2%) to capture localized Greeks.

#### Scenario: Barrier proximity and zoom section
- **GIVEN** a SnowballOption with KO/KI barriers and a PricingEnvironment
- **WHEN** the report is generated
- **THEN** the report includes barrier distance metrics in sigma units and zoomed Greeks around the barriers

### Requirement: Advanced volatility risk surfaces
The system SHALL support skew/smile shock inputs and report Vanna (dDelta/dVol) and Volga (dVega/dVol) surfaces alongside existing PV/Greeks surfaces.

#### Scenario: Generate Vanna/Volga surfaces
- **GIVEN** a SnowballOption, a PricingEnvironment, and a skew/smile shock configuration
- **WHEN** the report computes volatility risk outputs
- **THEN** the report includes Vanna and Volga surfaces and labels the shock model used

### Requirement: Higher-order time Greeks
The system SHALL compute and report Charm (dDelta/dTime) and Color (dGamma/dTime) surfaces using consistent bump conventions.

#### Scenario: Compute time Greeks
- **GIVEN** a SnowballOption and a PricingEnvironment
- **WHEN** the report computes time Greek surfaces
- **THEN** the report includes Charm and Color with documented bump sizes

### Requirement: Executive dashboard and lifecycle context
The system SHALL provide an executive dashboard that summarizes PV and key Greeks, barrier watch metrics, and a status indicator, and SHALL adapt sections based on pre-KI vs post-KI state.

#### Scenario: Dashboard reflects lifecycle state
- **GIVEN** a SnowballOption with known KI status
- **WHEN** the report is generated
- **THEN** the dashboard highlights the appropriate lifecycle focus and status indicator

### Requirement: Stress scenarios and conditional cashflows
The system SHALL provide a stress scenario table and conditional cashflow projections that separate expected cashflows from conditional-on-KO-date cashflows.

#### Scenario: Stress and cashflow tables
- **GIVEN** a SnowballOption and a PricingEnvironment
- **WHEN** the report is generated
- **THEN** the report includes stress scenario PnL and conditional cashflow tables
