# Model Validation Package: Range Accrual Analytical Engine

| Field | Value |
|-------|-------|
| Model | Range Accrual Analytical Engine (Digital Decomposition) |
| Engine File | `asset/equity/engine/analytical/range_accrual_analytical_engine.py` |
| Test File | `test/test_range_accrual_analytical_engine.py` |
| Date | 2026-02-10 |
| Status | **VALIDATED** |

---

## 1. Executive Summary

A closed-form analytical pricing engine for Range Accrual options has been developed, validated, and reviewed. The engine decomposes the Range Accrual payoff into a portfolio of digital options under Black-Scholes-Merton, leveraging linearity of expectation to price each observation independently.

**Key results:**
- Gate validation: **PASS** (machine-precision match with independent implementation)
- MC cross-validation: **PASS** (all 8 cases within 0.04% of QMC with 500K paths)
- Performance review: **PASS** (vectorized NumPy, 1000-10000x faster than MC)
- Security review: **PASS** (proper input validation, safe numerics)
- Code quality review: **PASS** (consistent with codebase patterns, 25 tests all passing)

---

## 2. Model Specification

### Pricing Formula

```
Price = exp(-r*T) * S_0 * M * c * tau * E[accrual_ratio]

E[accrual_ratio] = (1/W) * [past_in_range_weights + sum_i w_i * P_i]

P_i = N(d2_L) - N(d2_U)           (standard mode)
P_i = 1 - [N(d2_L) - N(d2_U)]    (reverse mode)

d2(K, t_i) = [ln(S/K) + (r - q - sigma_i^2/2) * t_i] / (sigma_i * sqrt(t_i))
```

### Key Insight

Under GBM, linearity of expectation allows pricing each observation independently as a digital call spread, regardless of the correlation between observations. The correlation affects hedging/risk but not the expected payoff.

### Supported Features

- Weighted observations (e.g., Friday=3 for weekend carry)
- Historical (past) observations with known outcomes
- Time-varying barriers (per-observation or scalar)
- Reverse mode (pay when outside range)
- Annualized or non-annualized accrual rates
- Per-observation vol term structure
- Analytical delta and gamma

---

## 3. Validation Results

### 3a. Developer B Gate Report

| Case | Abs Diff | Rel Diff | Status |
|------|----------|----------|--------|
| Standard 12M monthly | 1.33e-15 | 5.05e-16 | PASS |
| Reverse mode | 0.00 | 0.00 | PASS |
| Narrow range, low vol | 0.00 | 0.00 | PASS |
| Time-varying barriers | 0.00 | 0.00 | PASS |
| Past + future obs | 0.00 | 0.00 | PASS |

**Gate Decision: PASS** - Machine-precision agreement across all cases.

### 3b. MC Cross-Validation (500K QMC paths)

| Case | Analytical | MC Price | Rel Diff | Status |
|------|-----------|----------|----------|--------|
| Standard | 26230.89 | 26232.64 | 0.0067% | PASS |
| Low vol (10%) | 38304.87 | 38310.54 | 0.0148% | PASS |
| High vol (40%) | 14755.64 | 14749.77 | 0.0398% | PASS |
| Narrow [95,105] | 23435.51 | 23427.79 | 0.0330% | PASS |
| Wide [70,130] | 26796.57 | 26792.96 | 0.0135% | PASS |
| Reverse mode | 21330.58 | 21328.83 | 0.0082% | PASS |
| Step-down barriers | 26185.08 | 26188.65 | 0.0137% | PASS |
| Weighted obs | 24075.50 | 24080.67 | 0.0215% | PASS |

**Max relative difference: 0.04%** - well within MC sampling error.

---

## 4. Review Results

| Review | Result | Critical Issues | Notes |
|--------|--------|-----------------|-------|
| Performance | PASS | 0 | Vectorized NumPy; 1000-10000x faster than MC |
| Security | PASS | 0 | Proper validation, safe_log, is_zero utilities |
| Code Quality | PASS | 0 | Consistent patterns, comprehensive tests |

---

## 5. Test Coverage

- **25 tests** covering: basic pricing, digital decomposition correctness, MC comparison, historical observations, time-varying barriers, edge cases, vol sensitivity, analytical Greeks, error handling
- **All 25 tests pass** in 0.53 seconds
- **70 existing range accrual tests** remain passing (no regressions)

---

## 6. Files Delivered

| File | Purpose |
|------|---------|
| `asset/equity/engine/analytical/range_accrual_analytical_engine.py` | Production engine |
| `asset/equity/engine/analytical/__init__.py` | Updated exports |
| `test/test_range_accrual_analytical_engine.py` | Test suite (25 tests) |

---

## 7. Known Limitations

1. **GBM assumption**: Assumes log-normal dynamics. No jumps or stochastic volatility.
2. **Flat vol per observation**: Uses a single vol per observation time. Does not account for vol smile effects on digital option pricing.
3. **Single discount factor**: Applies exp(-rT) to entire payoff. For INSTANT pay type, per-observation discounting would be more accurate.
4. **No skew adjustment**: Digital options are sensitive to vol slope at barrier strikes; this is a known model risk under flat vol.

---

## 8. Final Recommendation

**APPROVED FOR PRODUCTION USE**

The Range Accrual Analytical Engine has been independently verified, cross-validated against Monte Carlo, and reviewed for performance, security, and code quality. All validation checks pass. The engine provides accurate, fast pricing suitable for production use under GBM assumptions.
