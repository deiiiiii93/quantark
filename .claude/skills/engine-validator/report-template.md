# Validation Report: {{ENGINE_NAME}}

**Generated**: {{DATE}}  
**Engine**: `{{ENGINE_PATH}}`  
**Reference**: `{{REFERENCE_PATH}}`  
**Validator**: engine-validator skill v1.0

---

## Executive Summary

| Check Type | Status | Pass Rate | Details |
|------------|--------|-----------|---------|
| Method Implementation | {{METHOD_STATUS}} | — | {{METHOD_NOTES}} |
| Boundary Checks | {{BOUNDARY_STATUS}} | {{BOUNDARY_RATE}} | {{BOUNDARY_NOTES}} |
| Benchmark Checks | {{BENCHMARK_STATUS}} | {{BENCHMARK_RATE}} | {{BENCHMARK_NOTES}} |
| User Cases | {{USER_STATUS}} | {{USER_RATE}} | {{USER_NOTES}} |

**Overall Status**: {{OVERALL_STATUS}}

### Key Findings

{{KEY_FINDINGS}}

### Recommendations

{{RECOMMENDATIONS}}

---

## 1. Method Description

### 1.1 Pricing Method Summary

{{METHOD_SUMMARY}}

### 1.2 Mathematical Formulation

{{MATH_FORMULAS}}

### 1.3 Reference Comparison

| Aspect | Reference | Implementation | Match |
|--------|-----------|----------------|-------|
| Core Pricing Formula | {{REF_FORMULA}} | {{IMPL_FORMULA}} | {{FORMULA_MATCH}} |
| Parameter Definitions | {{REF_PARAMS}} | {{IMPL_PARAMS}} | {{PARAMS_MATCH}} |
| Edge Case Handling | {{REF_EDGE}} | {{IMPL_EDGE}} | {{EDGE_MATCH}} |
| Numerical Methods | {{REF_NUMERICAL}} | {{IMPL_NUMERICAL}} | {{NUMERICAL_MATCH}} |
| Greeks Calculation | {{REF_GREEKS}} | {{IMPL_GREEKS}} | {{GREEKS_MATCH}} |

### 1.4 Implementation Issues Found

{{IMPLEMENTATION_ISSUES}}

---

## 2. Boundary Checks

**Script**: `{{BOUNDARY_SCRIPT_PATH}}`

### 2.1 Extreme Market Cases

| Test Case | Expected Behavior | Actual Result | Status |
|-----------|------------------|---------------|--------|
| Low volatility (σ → 0) | Value → intrinsic | {{LOW_VOL_RESULT}} | {{LOW_VOL_STATUS}} |
| Near expiry (T → 0) | Value → payoff | {{NEAR_EXPIRY_RESULT}} | {{NEAR_EXPIRY_STATUS}} |
| Deep ITM | Delta → ±1 | {{DEEP_ITM_RESULT}} | {{DEEP_ITM_STATUS}} |
| Deep OTM | Value → 0 | {{DEEP_OTM_RESULT}} | {{DEEP_OTM_STATUS}} |
| Very high volatility | Call → S, Put → K×e^(-rT) | {{HIGH_VOL_RESULT}} | {{HIGH_VOL_STATUS}} |
| Zero interest rate | No discounting | {{ZERO_RATE_RESULT}} | {{ZERO_RATE_STATUS}} |

### 2.2 Theoretical Relationships

| Relationship | Formula | Result | Status |
|--------------|---------|--------|--------|
| Put-Call Parity | C - P = S×e^(-qT) - K×e^(-rT) | {{PCP_RESULT}} | {{PCP_STATUS}} |
| Call Spread Monotonicity | C(K1) ≥ C(K2) when K1 < K2 | {{CALL_SPREAD_RESULT}} | {{CALL_SPREAD_STATUS}} |
| Butterfly Spread | C(K1) + C(K3) ≥ 2×C(K2) | {{BUTTERFLY_RESULT}} | {{BUTTERFLY_STATUS}} |
| Gamma Positivity | Γ ≥ 0 | {{GAMMA_RESULT}} | {{GAMMA_STATUS}} |
| Vega Positivity | V ≥ 0 | {{VEGA_RESULT}} | {{VEGA_STATUS}} |
| Delta Bounds | Call: [0,1], Put: [-1,0] | {{DELTA_RESULT}} | {{DELTA_STATUS}} |

### 2.3 Product-Specific Checks

{{PRODUCT_SPECIFIC_CHECKS}}

### 2.4 Boundary Check Summary

```
Total Tests:  {{BOUNDARY_TOTAL}}
Passed:       {{BOUNDARY_PASSED}} ({{BOUNDARY_PASS_PERCENT}}%)
Failed:       {{BOUNDARY_FAILED}}
Warnings:     {{BOUNDARY_WARNINGS}}
Skipped:      {{BOUNDARY_SKIPPED}}
```

---

## 3. Benchmark Comparison

**Benchmark Engine**: `{{MC_ENGINE_PATH}}`  
**Tolerance**: {{BENCHMARK_TOLERANCE}}%  
**MC Paths**: {{MC_PATHS}}  
**MC Steps**: {{MC_STEPS}}  
**Script**: `{{BENCHMARK_SCRIPT_PATH}}`

### 3.1 Test Results

| Case | Analytical | MC | Std Error | Rel Error | Status |
|------|-----------|-----|-----------|-----------|--------|
{{BENCHMARK_TABLE}}

### 3.2 Error Analysis

| Metric | Value |
|--------|-------|
| Mean Error | {{MEAN_ERROR}}% |
| Median Error | {{MEDIAN_ERROR}}% |
| Max Error | {{MAX_ERROR}}% |
| Min Error | {{MIN_ERROR}}% |
| Std Dev | {{STD_ERROR}}% |

### 3.3 Benchmark Summary

```
Total Cases:  {{BENCHMARK_TOTAL}}
Passed:       {{BENCHMARK_PASSED}} ({{BENCHMARK_PASS_PERCENT}}%)
Failed:       {{BENCHMARK_FAILED}}
Skipped:      {{BENCHMARK_SKIPPED}}
```

### 3.4 Benchmark Notes

{{BENCHMARK_NOTES_DETAIL}}

---

## 4. User Test Cases

{{USER_CASES_SECTION}}

---

## 5. Detailed Analysis

### 5.1 Numerical Stability

{{NUMERICAL_STABILITY_ANALYSIS}}

### 5.2 Performance Considerations

{{PERFORMANCE_ANALYSIS}}

### 5.3 Edge Case Handling

{{EDGE_CASE_ANALYSIS}}

---

## 6. Recommendations

### 6.1 Critical Issues (Must Fix)

{{CRITICAL_ISSUES}}

### 6.2 Improvements (Should Fix)

{{IMPROVEMENTS}}

### 6.3 Suggestions (Nice to Have)

{{SUGGESTIONS}}

---

## Appendix

### A. Test Environment

| Component | Version |
|-----------|---------|
| Python | {{PYTHON_VERSION}} |
| NumPy | {{NUMPY_VERSION}} |
| SciPy | {{SCIPY_VERSION}} |
| QuantArk | {{QUANTARK_VERSION}} |

### B. Script Execution Commands

```bash
# Run boundary checks
python {{BOUNDARY_SCRIPT_PATH}}

# Run benchmark checks
python {{BENCHMARK_SCRIPT_PATH}}

# Run all validation (if combined script exists)
python {{COMBINED_SCRIPT_PATH}}
```

### C. Test Data Used

```python
# Base market parameters
spot = {{BASE_SPOT}}
strike = {{BASE_STRIKE}}
rate = {{BASE_RATE}}
vol = {{BASE_VOL}}
dividend = {{BASE_DIV}}
maturity = {{BASE_MATURITY}}
```

### D. References

1. {{REFERENCE_1}}
2. {{REFERENCE_2}}
3. {{REFERENCE_3}}

---

## Validation History

| Date | Version | Changes | Validator |
|------|---------|---------|-----------|
| {{DATE}} | 1.0 | Initial validation | {{VALIDATOR}} |

---

*Report generated by engine-validator skill*
