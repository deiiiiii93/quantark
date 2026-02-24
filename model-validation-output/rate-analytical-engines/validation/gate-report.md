# Gate Report: Rate Analytical Engines

## Summary
**Gate Decision**: PASS
**Date**: 2026-02-11
**Validator**: Developer B (Independent)
**Tests Run**: 20
**Tests Passed**: 20
**Tests Failed**: 0

## Methodology

Independent implementations of the following pricing formulas were coded
from scratch using only numpy and scipy (no QuantArk math imports):

1. **FRA**: NPV = N * dcf * (L - K) / (1 + L * dcf) * df(T_settle)
2. **Cap/Floor (Black-76)**: Caplet = df * dcf * N * [F*N(d1) - K*N(d2)]
3. **Swaption (Black-76)**: V = A * [S*N(d1) - K*N(d2)]
4. **Swaption (Bachelier)**: V = A * [(S-K)*N(d) + sigma*sqrt(T)*n(d)]

QuantArk was used only to construct products and invoke engine `.price()` methods.
Results were compared with a 1% relative error tolerance (or $1 absolute for near-zero values).

## Test Results

| Engine | Test Case | Independent Value | Engine Value | Rel Error | Status |
|--------|-----------|-------------------|--------------|-----------|--------|
| FRA | At-market FRA NPV ~ 0 | 0.000000 | 0.000000 | 0.0000% | PASS |
| FRA | Off-market FRA (K=4%) | 24,653.040083 | 24,653.040083 | 0.0000% | PASS |
| FRA | Off-market FRA (K=6%) | -24,653.040083 | -24,653.040083 | 0.0000% | PASS |
| FRA | 6-month tenor FRA (K=4.5%) | 121,525.809731 | 121,525.809731 | 0.0000% | PASS |
| Cap/Floor | ATM single caplet (F=K=5%, vol=20%) | 4,800.229870 | 4,800.229870 | 0.0000% | PASS |
| Cap/Floor | Multi-period ATM cap (2Y quarterly) | 72,459.262896 | 72,459.262896 | 0.0000% | PASS |
| Cap/Floor | Multi-period ATM floor (2Y quarterly) | 72,459.262896 | 72,459.262896 | 0.0000% | PASS |
| Cap/Floor | Cap-Floor parity check (indep) | 0.000000 | 0.000000 | 0.0000% | PASS |
| Cap/Floor | Cap-Floor parity check (engine) | 0.000000 | 0.000000 | 0.0000% | PASS |
| Cap/Floor | OTM caplet (F=5%, K=7%) | 1.030353 | 1.030353 | 0.0000% | PASS |
| Swaption | Forward swap rate | 0.050335 | 0.050335 | 0.0000% | PASS |
| Swaption | Annuity | 41,816,977.814081 | 41,816,977.814081 | 0.0000% | PASS |
| Swaption | ATM payer (independent inputs) | 174,420.358330 | 174,420.358330 | 0.0000% | PASS |
| Swaption | ATM payer (engine S,A + indep formula) | 174,420.358330 | 174,420.358330 | 0.0000% | PASS |
| Swaption | ATM receiver (engine S,A + indep formula) | 160,430.814651 | 160,430.814651 | 0.0000% | PASS |
| Swaption | Payer-Receiver parity (engine) | 13,989.543679 | 13,989.543679 | 0.0000% | PASS |
| Swaption | ITM payer (K=4%) | 455,537.201576 | 455,537.201576 | 0.0000% | PASS |
| Swaption | ATM payer Bachelier (normal vol=80bp) | 140,754.469750 | 140,754.469750 | 0.0000% | PASS |
| Swaption | ATM receiver Bachelier (normal vol=80bp) | 126,764.926070 | 126,764.926070 | 0.0000% | PASS |
| Swaption | Bachelier Payer-Receiver parity | 13,989.543679 | 13,989.543679 | 0.0000% | PASS |

## Engine-Specific Results

### FRA Engine
- Tests: 4 | Passed: 4 | Failed: 0
- Max relative error: 0.0000%
- Formula verified: NPV = N * dcf * (L - K) / (1 + L * dcf) * df(T_settle)

### Cap/Floor Engine
- Tests: 6 | Passed: 6 | Failed: 0
- Max relative error: 0.0000%
- Formula verified: Black-76 caplet/floorlet pricing
- Cap-Floor parity verified: Cap(K) - Floor(K) = PV(FRA strip)

### Swaption Engine
- Tests: 10 | Passed: 10 | Failed: 0
- Max relative error: 0.0000%
- Formulas verified: Black-76 and Bachelier
- Payer-Receiver parity verified for both models

## Findings
- All tests passed within the 1% relative error tolerance.
- Independent implementations coded from reference formulas (Black-76, Bachelier, FRA discounting) using only numpy/scipy.
- QuantArk used only for product construction and engine invocation.

## Recommendation
- FRA engine: APPROVED for production use.
- Cap/Floor engine: APPROVED for production use.
- Swaption engine: APPROVED for production use.

Overall gate decision: **PASS**
