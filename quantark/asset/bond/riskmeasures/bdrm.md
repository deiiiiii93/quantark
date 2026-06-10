# Bond Risk Measures

## Bond Component Risk Measures
- Modified Duration
- Macaulay Duration
- DV01 (Dollar Value of 01)
- Convexity

## Option Component Risk Measures (via BondGreeksCalculator)
- Delta: Sensitivity to forward bond price
- Gamma: Second-order sensitivity to forward price
- Vega: Sensitivity to volatility (per 1% change)
- Theta: Time decay (per day)
- Rho: Sensitivity to interest rates (per 1% change)
- Option DV01: Sensitivity to parallel rate shift
- Option Duration: Effective duration through the option
- Delta-Equivalent DV01: Delta-weighted underlying DV01

## Implementation
- `BondGreeksCalculator`: Main calculator class
  - `calculate_analytical_greeks()`: Black '76 closed-form Greeks
  - `calculate_numerical_greeks()`: Finite difference Greeks
  - `calculate_bond_sensitivities()`: Bond-specific risk measures