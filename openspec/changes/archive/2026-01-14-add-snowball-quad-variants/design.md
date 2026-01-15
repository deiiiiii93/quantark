# Design: Snowball Quad variants

## Overview
Extend the two-regime quadrature recursion to support airbag V1 payoff logic, call-rebate V0 payoff logic, and the `disable_ko_after_ki` interaction.

## Decisions
- **Airbag**: Use the existing `SnowballOption.get_maturity_payoff_v1` logic for terminal `V_in`. No additional barrier logic is required.
- **Call-rebate V0**: Use `SnowballOption.get_maturity_payoff_v0` for terminal `V_out`. No additional barrier logic is required.
- **disable_ko_after_ki**:
  - KO applies only to the not-knocked-in regime (`V_out`).
  - KI transitions still occur based on discrete observations or Brownian-bridge mixing.
  - If KO and KI occur at the same discrete observation, KO is suppressed (consistent with MC engine `ko_time < ki_time`).

## Recursion Updates
- Apply KO updates conditionally:
  - If `disable_ko_after_ki=False`: current behavior (KO overwrites both `V_in` and `V_out`).
  - If `disable_ko_after_ki=True`: KO overwrites only `V_out`, and KO masks exclude KI hits at the same observation.

## Validation
- Remove rejection checks for airbag and call-rebate.
- Keep existing constraints: discrete KO only; continuous KI scalar barrier only.
