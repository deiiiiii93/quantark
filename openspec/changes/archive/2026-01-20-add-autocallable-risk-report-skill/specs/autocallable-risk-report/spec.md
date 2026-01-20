## ADDED Requirements
### Requirement: Snowball risk profile report generation
The system SHALL generate a Snowball risk profile report containing: trade/schedule summary, valuation assumptions,
local Greeks, risk surfaces with plots, risk-neutral event statistics, and historical risk statistics.

#### Scenario: Generate a Snowball report with plots
- **GIVEN** a SnowballOption, a PricingEnvironment, and an engine selection order (QUAD→PDE→MC)
- **WHEN** a risk profile report is generated
- **THEN** the output includes a Markdown report and plot artifacts for PV/Greeks and required surfaces

### Requirement: Dividend/basis risk surfaces for China index hedging
The system SHALL include dividend yield sensitivities as first-class outputs, including surfaces for `dividend_rho`
vs `Spot×Vol` and `Spot×Dividend`, and a basis mapping `basis_rho = -dividend_rho` under `b = r - q`.

#### Scenario: Produce dividend_rho surfaces
- **GIVEN** a SnowballOption and a flat dividend yield `q0`
- **WHEN** the report computes `dividend_rho` on `Spot×Vol` and `Spot×Dividend` grids
- **THEN** the report includes the surfaces and states their scaling conventions

### Requirement: Spot–dividend cross exposure reporting
The system SHALL compute and report mixed spot–dividend cross sensitivity as `∂²V/(∂S∂q)` and a derived delta surface
`Delta(Spot, Dividend)`.

#### Scenario: Compute cross sensitivity grid
- **GIVEN** a spot grid and dividend yield grid
- **WHEN** the report computes cross sensitivity outputs
- **THEN** the report includes `Delta(Spot, Dividend)` and `∂²V/(∂S∂q)` surfaces

### Requirement: Risk-neutral event probabilities and cashflow attribution
The system SHALL compute per-observation event probabilities and expected discounted cashflows under a pricing measure,
including `P(KO at i)`, `P(survive to i)`, `P(coupon at i)`, expected life, `P(KI before maturity)`, and a PV
reconciliation against discounted expected cashflows.

#### Scenario: RN cashflow reconciliation
- **GIVEN** a SnowballOption and a pricing-measure simulation/engine
- **WHEN** the report computes risk-neutral event statistics and expected discounted cashflows
- **THEN** the sum of expected discounted cashflows approximately reconciles to PV within tolerance

### Requirement: Historical replay and parametric scenario analysis
The system SHALL support historical replay and parametric scenario generation driven by user-supplied time series for
spot and dividend yield (flat q), producing PnL distributions and event-statistic distributions.

#### Scenario: Run historical replay with spot and dividend series
- **GIVEN** a spot time series and a dividend yield time series
- **WHEN** the report runs historical replay analysis
- **THEN** the report includes horizon PnL distributions and historical event statistics

### Requirement: Default report grid configuration
The system SHALL default to the following grids unless overridden: `Spot ∈ [0.60, 1.20] * S0`, `q ∈ q0 ± 500bp`,
and `σ ∈ σ0 * (1±5%)` with 11 vol nodes.

#### Scenario: Apply default grids
- **GIVEN** a request without custom grid parameters
- **WHEN** the report generation runs
- **THEN** the report uses the default spot/dividend/vol grids

### Requirement: Engine event stats API roadmap (optional enhancement)
The system SHALL define an engine-level event stats / cashflow decomposition API so QUAD/PDE engines can optionally
produce per-observation event statistics for autocallables without Monte Carlo simulation.

#### Scenario: Engine exposes event stats capability
- **GIVEN** a QUAD or PDE engine that implements the event stats API
- **WHEN** the report requests per-observation probabilities
- **THEN** the report uses engine-provided event stats where available and falls back to MC analysis otherwise

