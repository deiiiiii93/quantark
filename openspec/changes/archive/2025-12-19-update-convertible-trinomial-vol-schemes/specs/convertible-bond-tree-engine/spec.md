## MODIFIED Requirements
### Requirement: Trinomial Tree Engine (Hull-White Model)
The system SHALL provide a `ConvertibleBondTrinomialEngine` implementing the Hull-White trinomial tree model with explicit default probability at each node and an explicit volatility scheme selection.

#### Scenario: Scheme selection for trinomial engine
- **WHEN** `ConvertibleBondTrinomialEngine` is initialized with a specific trinomial volatility scheme
- **THEN** the engine prices using the selected scheme and exposes it in its configuration

## ADDED Requirements
### Requirement: Trinomial Volatility Schemes
The system SHALL support multiple volatility schemes for the trinomial convertible bond tree.

#### Scenario: Constant-volatility trinomial scheme
- **WHEN** the constant-volatility scheme is selected
- **THEN** the tree uses a CRR-style fixed volatility grid and does not apply term-structure volatility

#### Scenario: Fixed-dx log-price scheme
- **WHEN** the fixed-dx log-price scheme is selected
- **THEN** the tree uses a constant log step and per-step probabilities that match the step-local variance

#### Scenario: Variable-dx log-price scheme with re-gridding
- **WHEN** the variable-dx log-price scheme is selected
- **THEN** the tree recomputes the log step per time interval and re-grids values to maintain a recombining lattice
