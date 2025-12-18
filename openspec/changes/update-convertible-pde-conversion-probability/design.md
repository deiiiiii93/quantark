# Design: Exact conversion probability for PDE convertible engines

## Goal
Compute the risk-neutral probability that the optimal policy results in conversion before (or at) maturity (“eventual conversion probability”), consistent with the same early exercise / call / put logic used in PDE pricing.

## Approach
Introduce an auxiliary state variable `P(S,t)` on the same PDE grid as the value solve:
- `P(S,t)` is the probability of eventual conversion, conditional on current stock price `S` and time `t`.
- Time stepping uses the same finite-difference operator and scheme as the pricing PDE, but with:
  - No coupon source term
  - No discounting by `r` (it is a probability, not a present value)

### Tsiveriotis–Fernandes (TF)
Solve `P` under the same stock diffusion as the value PDE:
- `P_t + (r-q)S P_S + 0.5 σ² S² P_SS = 0`

Apply the same policy constraints:
- If conversion is optimal at `(S,t)` → set `P=1` (conversion happens now)
- If put is exercised → set `P=0`
- If issuer calls and holder redeems (call) → set `P=0`
- If issuer calls and holder converts → set `P=1`
- Otherwise (“hold”) → keep PDE continuation probability

### Jump-Diffusion (hazard default)
Solve `P` under intensity `λ` with default terminating the contract without conversion:
- `P_t + (r-q-λ η)S P_S + 0.5 σ² S² P_SS - λ P = 0`

Coupon/recovery cashflows affect value but not the conversion-event probability.

## Boundary / Terminal Conditions
- Terminal at maturity uses the same tie-break rule as pricing terminal logic:
  - `P(T,S)=1` if conversion is optimal at maturity, else `0`.
- Boundary conditions follow the pricing engine’s far-field assumptions:
  - `P(t, S≈0)=0`
  - `P(t, S≈S_max)=1`

## Notes
This is “mathematically exact” relative to the PDE discretization and boundary assumptions of each engine (i.e., no heuristic post-processing from the value surface).
