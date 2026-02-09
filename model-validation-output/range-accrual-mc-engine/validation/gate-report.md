# Range Accrual MC Engine - Validation Gate Report

**Date**: 2025-02-05
**Validator**: Developer B (Independent Validation)
**Subject**: Range Accrual Monte Carlo Engine Cross-Validation

---

## Executive Summary

| Decision | Status |
|----------|--------|
| **GATE DECISION** | **PASS** |

The Range Accrual Monte Carlo Engine (Developer A implementation) has been independently validated. All test cases pass within Monte Carlo sampling error (3-sigma tolerance). The implementations produce statistically consistent prices across a variety of market conditions and product configurations.

---

## Validation Methodology

### Approach

An independent Monte Carlo implementation was developed without reference to Developer A's code. The validation implementation:

1. **Simulates GBM paths** using standard Euler-Maruyama discretization
2. **Checks barrier conditions** at each observation time
3. **Computes weighted accrual ratio** (sum of in-range weights / total weights)
4. **Calculates payoff** using the Range Accrual formula:
   ```
   Payoff = initial_price * contract_multiplier * accrual_rate
            * accrual_ratio * year_fraction
   ```
5. **Discounts to present value** using the risk-free rate

### Key Implementation Details

- **Path Generation**: Standard GBM with log-space evolution
  - `S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)`
- **Barrier Check**: `lower_barrier <= spot <= upper_barrier`
- **Reverse Mode**: Accrue when OUTSIDE range (if `is_reverse=True`)
- **Same Seed**: Both implementations use identical random seeds for direct comparison

---

## Test Results

### Test Case 1: Basic Range Accrual

| Parameter | Value |
|-----------|-------|
| Initial Price | 100.0 |
| Barriers | [90, 110] |
| Accrual Rate | 5% (annualized) |
| Maturity | 1.0 year |
| Observations | 12 (monthly) |
| Volatility | 20% |
| Risk-Free Rate | 5% |
| Dividend Yield | 2% |

| Metric | Value |
|--------|-------|
| Validation Price | 2.636368 |
| Developer A Price | 2.640565 |
| Standard Error | 0.004270 |
| Difference | 0.004197 |
| 3-sigma Tolerance | 0.012810 |
| **Result** | **PASS** |

---

### Test Case 2: Low Volatility (High In-Range Probability)

| Parameter | Value |
|-----------|-------|
| Volatility | 10% |
| (Other params same as Test 1) | |

| Metric | Value |
|--------|-------|
| Validation Price | 3.927036 |
| Developer A Price | 3.925985 |
| Standard Error | 0.003533 |
| Difference | 0.001050 |
| 3-sigma Tolerance | 0.010600 |
| **Result** | **PASS** |

**Observation**: Lower volatility leads to higher prices (more observations in range), which is the expected behavior.

---

### Test Case 3: High Volatility (Low In-Range Probability)

| Parameter | Value |
|-----------|-------|
| Volatility | 40% |
| (Other params same as Test 1) | |

| Metric | Value |
|--------|-------|
| Validation Price | 1.470486 |
| Developer A Price | 1.472816 |
| Standard Error | 0.003326 |
| Difference | 0.002331 |
| 3-sigma Tolerance | 0.009977 |
| **Result** | **PASS** |

**Observation**: Higher volatility leads to lower prices (more observations outside range), confirming correct sensitivity.

---

### Test Case 4: Narrow Range

| Parameter | Value |
|-----------|-------|
| Barriers | [95, 105] |
| Accrual Rate | 8% |
| (Other params same as Test 1) | |

| Metric | Value |
|--------|-------|
| Validation Price | 2.347330 |
| Developer A Price | 2.356817 |
| Standard Error | 0.005314 |
| Difference | 0.009487 |
| 3-sigma Tolerance | 0.015941 |
| **Result** | **PASS** |

**Observation**: Narrow range has lower accrual ratio, partially offset by higher accrual rate. Largest variance among tests due to binary nature of narrow barrier hits.

---

### Test Case 5: Daily Observations (252)

| Parameter | Value |
|-----------|-------|
| Observations | 252 (daily) |
| Paths | 50,000 (reduced for performance) |
| (Other params same as Test 1) | |

| Metric | Value |
|--------|-------|
| Validation Price | 2.763222 |
| Developer A Price | 2.763739 |
| Standard Error | 0.005726 |
| Difference | 0.000517 |
| 3-sigma Tolerance | 0.017179 |
| **Result** | **PASS** |

**Observation**: Dense observation schedule produces very tight agreement between implementations. The small difference demonstrates correct handling of many observation points.

---

## Summary Statistics

| Test Case | Diff / Std Error | Within 3-sigma |
|-----------|-----------------|----------------|
| Basic Range Accrual | 0.98 | Yes |
| Low Volatility | 0.30 | Yes |
| High Volatility | 0.70 | Yes |
| Narrow Range | 1.79 | Yes |
| Daily Observations | 0.09 | Yes |

**All 5 test cases pass within the 3-sigma tolerance.**

---

## Validation Implementation

The independent implementation is located at:
```
model-validation-output/range-accrual-mc-engine/validation/independent-impl/range_accrual_mc_validator.py
```

Key functions:
- `price_range_accrual_mc()`: Core MC pricing with raw parameters
- `price_range_accrual_from_product()`: Wrapper that extracts params from product/env
- `run_validation_tests()`: Full validation test suite

---

## Issues Found

**None.** The Developer A implementation correctly implements the Range Accrual MC pricing logic as specified in the product definition.

---

## Recommendations

1. **Approved for Use**: The Range Accrual MC Engine is validated and ready for production use.

2. **Future Enhancements** (not blocking):
   - Time-varying barriers support in validation impl (currently raises NotImplementedError)
   - QMC variance reduction comparison (if Developer A uses it)
   - Greeks validation via bump-and-reprice

---

## Conclusion

**GATE DECISION: PASS**

The Range Accrual Monte Carlo Engine has been independently validated through cross-comparison with a clean-room implementation. Both implementations produce statistically identical prices across all test scenarios, confirming correctness of the pricing logic.

---

*Validated by Developer B on 2025-02-05*
