# Double Barrier Option Analytical Engine Specification

## Overview
Implement an analytical pricing engine for double barrier options in the QuantArk library.

## Product
`DoubleBarrierOption` (or extension of existing `BarrierOption` if appropriate)

## Observation Types (Required)
1. **Continuous observation** - Standard Ikeda & Kuintomo (1992) formula
2. **Daily observation** - Apply barrier shift (e.g., Broadie-Glasserman-Kou adjustment or similar) to approximate discrete monitoring
3. **Expiry observation** - Price at expiration only (barriers checked only at maturity)

## Pricing Formula
Use the Ikeda and Kuintomo (1992) infinite series formula for double knock-out options.

**Call Up-and-Out-Down-and-Out** and **Put Up-and-Out-Down-and-Out** formulas are provided in the reference document:
- `double-barrier-option-price-formula.md`

Key components:
- Infinite series of weighted normal distribution functions
- Parameters: spot `S`, strike `X`, lower barrier `L`, upper barrier `U`, time `T`, risk-free rate `r`, cost of carry `b`, volatility `σ`
- Curvature parameters `δ₁` and `δ₂` (flat boundaries: `δ₁ = δ₂ = 0`)
- The series converges rapidly; typically 2-3 terms suffice for most cases

## Knock-In / Knock-Out Relationship
- Double knock-in call = long standard call - short double knock-out call
- Double knock-in put = long standard put - short double knock-out put

## Engine Interface
Follow QuantArk conventions:
- Engine location: `asset/equity/engine/analytical/double_barrier_option_engine.py`
- Base class: `BaseEngine`
- Method: `price(product, market_env)`
- Support `contract_multiplier` scaling
- Use `util.numerical` utilities (`safe_log`, `safe_exp`, `safe_sqrt`, `is_zero`, etc.)

## Validation Benchmarks (Continuous Observation)
Use Table 4-15 from the reference document as exact validation baselines.

Selected test cases (S=100, X=100, r=0.1, b=0.1):

| L | U | δ1 | δ2 | T=0.25 σ=0.15 | T=0.25 σ=0.25 | T=0.5 σ=0.15 | T=0.5 σ=0.25 |
|---|---|----|----|---------------|---------------|--------------|--------------|
| 50 | 150 | 0 | 0 | 4.3515 | 6.1644 | 6.9853 | 7.9336 |
| 80 | 120 | 0 | 0 | 3.7516 | 2.6387 | 3.5805 | 1.5098 |
| 90 | 110 | 0 | 0 | 1.2055 | 0.3098 | 0.5537 | 0.0441 |
| 50 | 150 | -0.1 | 0.1 | 4.3514 | 6.0997 | 6.8974 | 6.9821 |
| 90 | 110 | -0.1 | 0.1 | 0.5887 | 0.1016 | 0.0398 | 0.0002 |
| 50 | 150 | 0.1 | -0.1 | 4.3515 | 6.2040 | 7.0086 | 8.6080 |
| 90 | 110 | 0.1 | -0.1 | 1.9229 | 0.6451 | 1.7079 | 0.3038 |

Error tolerance for continuous cases: < 0.1% or within 1e-4 absolute.

## Daily Observation
Implement a barrier shift for discrete (daily) monitoring. For example, shift barriers by:
- `L_shift = L * exp(-0.5826 * σ * sqrt(dt))`
- `U_shift = U * exp(0.5826 * σ * sqrt(dt))`
where `dt = 1/252`.

Then price using the shifted barriers with the continuous formula.

## Expiry Observation
When observation is only at expiry, the option payoff is simply:
- Call: `max(S_T - X, 0) * 1_{L < S_T < U}`
- Put: `max(X - S_T, 0) * 1_{L < S_T < U}`
This can be priced using the vanilla Black-Scholes formula with truncated domain, or equivalently as a portfolio of vanilla options and digital options.

## Edge Cases
- `T = 0`: return intrinsic value subject to barrier condition
- `σ = 0`: deterministic payoff
- `S <= L` or `S >= U` for knock-out: return 0
- Strike outside barrier range: document limitation (Ikeda-Kuintomo requires strike inside barriers)

## Tests Required
- Unit tests in `test/test_double_barrier_option_engine.py`
- Tests must cover all Table 4-15 continuous cases
- Tests for daily observation (smoke tests, monotonicity checks)
- Tests for expiry observation
- Edge case tests (T=0, S at barriers, deep ITM/OTM)
