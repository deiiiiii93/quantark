# equity-digital-products Specification

## Purpose
TBD - created by archiving change add-equity-digital-option. Update Purpose after archive.
## Requirements
### Requirement: Cash-or-nothing European digital option product
The system SHALL provide a cash-or-nothing European digital option that pays a fixed cash amount when the terminal spot is on the paying side of the strike.

#### Scenario: Validate inputs and European exercise
- **WHEN** a digital option is created with non-positive strike or payout, missing maturity/exercise_date, or an exercise style other than European
- **THEN** the system SHALL raise `ValidationError` describing the invalid parameter

#### Scenario: Payoff definition for call and put
- **WHEN** evaluating payoff at maturity for a call and the spot is strictly above the strike
- **THEN** the payoff SHALL equal the fixed payout, otherwise 0
- **AND** for a put with spot strictly below the strike the payoff SHALL equal the payout, otherwise 0

