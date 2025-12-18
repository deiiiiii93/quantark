# convertible-bond-pde-engine Specification (Delta)

## MODIFIED Requirements

### Requirement: Exact conversion probability output
The system SHALL compute and return an exact (within the PDE discretization) risk-neutral probability of eventual conversion for PDE-based convertible bond engines, consistent with the same optimal policy constraints used for pricing (conversion, call, put).

#### Scenario: Boundary cases
- **WHEN** the bond is configured so conversion is optimal immediately for all relevant stock prices
- **THEN** `conversion_probability` returned by the PDE engine is `1.0` (within numerical tolerance)
- **WHEN** conversion is never allowed during the valuation horizon
- **THEN** `conversion_probability` returned by the PDE engine is `0.0` (within numerical tolerance)

#### Scenario: Probability bounds
- **WHEN** a PDE engine computes conversion probability on a grid
- **THEN** the probability is in `[0, 1]`
