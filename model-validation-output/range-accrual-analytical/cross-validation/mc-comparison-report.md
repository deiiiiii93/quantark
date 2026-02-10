# Range Accrual Analytical Engine vs Monte Carlo Cross-Validation Report

**Generated:** 2026-02-10 10:09:35

## Executive Summary

**Status:** PASS ✓

All test cases show relative differences < 1% between the analytical and Monte Carlo engines.

## Test Configuration

- **Monte Carlo Method:** Quasi-Monte Carlo (QMC)
- **Number of Paths:** 500,000
- **Seed:** 42
- **Antithetic Variates:** Disabled

## Test Cases

### Case 1: Standard

| Metric | Value |
|--------|-------|
| Analytical Price | $26,230.89 |
| Monte Carlo Price | $26,232.64 |
| MC Standard Error | $19.15 |
| Absolute Difference | $1.75 |
| Relative Difference | 0.0067% |
| **Status** | **PASS ✓** |

### Case 2: Low vol

| Metric | Value |
|--------|-------|
| Analytical Price | $38,304.87 |
| Monte Carlo Price | $38,310.54 |
| MC Standard Error | $16.63 |
| Absolute Difference | $5.67 |
| Relative Difference | 0.0148% |
| **Status** | **PASS ✓** |

### Case 3: High vol

| Metric | Value |
|--------|-------|
| Analytical Price | $14,755.64 |
| Monte Carlo Price | $14,749.77 |
| MC Standard Error | $14.84 |
| Absolute Difference | $5.87 |
| Relative Difference | 0.0398% |
| **Status** | **PASS ✓** |

### Case 4: Narrow range

| Metric | Value |
|--------|-------|
| Analytical Price | $23,435.51 |
| Monte Carlo Price | $23,427.79 |
| MC Standard Error | $23.73 |
| Absolute Difference | $7.72 |
| Relative Difference | 0.0330% |
| **Status** | **PASS ✓** |

### Case 5: Wide range

| Metric | Value |
|--------|-------|
| Analytical Price | $26,796.57 |
| Monte Carlo Price | $26,792.96 |
| MC Standard Error | $5.90 |
| Absolute Difference | $3.60 |
| Relative Difference | 0.0135% |
| **Status** | **PASS ✓** |

### Case 6: Reverse mode

| Metric | Value |
|--------|-------|
| Analytical Price | $21,330.58 |
| Monte Carlo Price | $21,328.83 |
| MC Standard Error | $19.15 |
| Absolute Difference | $1.75 |
| Relative Difference | 0.0082% |
| **Status** | **PASS ✓** |

### Case 7: Step-down barriers

| Metric | Value |
|--------|-------|
| Analytical Price | $26,185.08 |
| Monte Carlo Price | $26,188.65 |
| MC Standard Error | $20.33 |
| Absolute Difference | $3.58 |
| Relative Difference | 0.0137% |
| **Status** | **PASS ✓** |

### Case 8: Weighted obs

| Metric | Value |
|--------|-------|
| Analytical Price | $24,075.50 |
| Monte Carlo Price | $24,080.67 |
| MC Standard Error | $22.66 |
| Absolute Difference | $5.17 |
| Relative Difference | 0.0215% |
| **Status** | **PASS ✓** |

## Results Summary Table

| Case | Analytical | MC Price | MC StdErr | Abs Diff | Rel Diff % | Status |
|------|------------|----------|-----------|----------|------------|--------|
| Standard | $26,230.89 | $26,232.64 | $19.15 | $1.75 | 0.0067% | PASS ✓ |
| Low vol | $38,304.87 | $38,310.54 | $16.63 | $5.67 | 0.0148% | PASS ✓ |
| High vol | $14,755.64 | $14,749.77 | $14.84 | $5.87 | 0.0398% | PASS ✓ |
| Narrow range | $23,435.51 | $23,427.79 | $23.73 | $7.72 | 0.0330% | PASS ✓ |
| Wide range | $26,796.57 | $26,792.96 | $5.90 | $3.60 | 0.0135% | PASS ✓ |
| Reverse mode | $21,330.58 | $21,328.83 | $19.15 | $1.75 | 0.0082% | PASS ✓ |
| Step-down barriers | $26,185.08 | $26,188.65 | $20.33 | $3.58 | 0.0137% | PASS ✓ |
| Weighted obs | $24,075.50 | $24,080.67 | $22.66 | $5.17 | 0.0215% | PASS ✓ |

## Analysis

The cross-validation compares the newly implemented Range Accrual Analytical Engine against the established Monte Carlo (QMC) engine across 8 diverse test cases:

1. **Standard**: Baseline case with typical parameters
2. **Low vol**: Tests behavior in low volatility regime (σ=0.10)
3. **High vol**: Tests behavior in high volatility regime (σ=0.40)
4. **Narrow range**: Tighter barriers [95,105] with non-annualized rate
5. **Wide range**: Wider barriers [70,130] with non-annualized rate
6. **Reverse mode**: Accrues when outside the range
7. **Step-down barriers**: Time-varying barriers that tighten over time
8. **Weighted observations**: Non-uniform observation weights

### Key Observations

- **Maximum Relative Difference:** 0.0398%
- **Minimum Relative Difference:** 0.0067%
- **Average Relative Difference:** 0.0189%

All test cases demonstrate excellent agreement between the analytical and MC methods, with relative differences well below the 1% threshold. This validates the correctness of the analytical implementation.

### Computational Efficiency

The analytical engine provides instant pricing without statistical noise, while the Monte Carlo engine requires 500,000 paths to achieve comparable accuracy. This represents a significant computational advantage for the analytical method.

## Conclusion

The Range Accrual Analytical Engine successfully passes cross-validation against the Monte Carlo engine. The implementation is validated for production use.
