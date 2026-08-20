# Validation Report: Barrier Analytical Engine

**Generated**: 2025-12-25  
**Engine**: `asset/equity/engine/analytical/barrier_analytical_engine.py`  
**Reference**: `asset/equity/engine/docs/barrier_analytical_engine.md`  
**Benchmark**: `asset/equity/engine/mc/barrier_option_mc_engine.py`

---

## Executive Summary

| Check Type | Status | Pass Rate |
|------------|--------|-----------|
| Method Implementation | ✅ PASS | - |
| Boundary Checks | ✅ PASS | 100% (8/8) |
| Benchmark Checks | ⚠️ WARNINGS | 87.5% (7/8) |
| User Cases | ⏭️ SKIPPED | - |

**Overall Status**: ⚠️ WARNINGS

---

## 1. Method Description

### 1.1 Pricing Method Summary

The engine implements closed-form barrier pricing with three monitoring modes:

- **Continuous monitoring** uses Reiner-Rubinstein style formulas with `A/B/C/D` terms and cost-of-carry `b = r - q`.
- **Discrete monitoring** applies the Broadie-Glasserman-Kou barrier shift (regular observation grid required).
- **Expiry-only monitoring** decomposes payoff via lognormal probabilities at expiry.

Rebates are valued through the `OneTouchAnalyticalEngine` using one-touch or no-touch legs.

### 1.2 Reference Comparison

| Aspect | Reference | Implementation | Match |
|--------|-----------|----------------|-------|
| Continuous KO formula | `A/B/C/D` terms | `_price_knock_out_closed_form()` | ✅ |
| Barrier shift | Broadie-Glasserman-Kou | `apply_barrier_shift()` | ✅ |
| Rebate leg | One-touch / No-touch | `_price_rebate_leg()` | ✅ |
| Expiry monitoring | Vanilla/digital decomposition | `_price_expiry()` | ✅ |
| Immediate knock-in hit | Should price as vanilla | Prices vanilla via BS | ✅ |

### 1.3 Issues Found

No discrepancies identified after updating knock-in immediate-hit handling.

---

## 2. Boundary Checks

**Script**: `asset/equity/engine/validation/script/boundary_check_barrier_analytical.py`  
**Status**: ✅ PASS (8/8)

### Implemented Tests

- Near expiry intrinsic value (barrier far away)
- KO + KI parity (continuous monitoring)
- KO ≤ vanilla bound
- Down-and-out monotonicity in barrier
- Continuous vs discrete KO (no rebate)
- Immediate KO rebate (pay at hit)
- Immediate KO rebate (pay at expiry)
- KO + KI parity (expiry monitoring)

### 2.1 Results

| Test | Status |
|------|--------|
| Near expiry (T→0) | ✅ PASS |
| KO + KI = Vanilla (continuous) | ✅ PASS |
| KO ≤ Vanilla | ✅ PASS |
| Down-and-out monotonicity (barrier up) | ✅ PASS |
| Continuous KO ≤ Discrete KO | ✅ PASS |
| Immediate KO rebate (pay at hit) | ✅ PASS |
| Immediate KO rebate (pay at expiry) | ✅ PASS |
| KO + KI = Vanilla (expiry monitoring) | ✅ PASS |

---

## 3. Benchmark Comparison

**Benchmark Engine**: `asset/equity/engine/mc/barrier_option_mc_engine.py`  
**Tolerance**: 5%  
**MC Paths / Steps**: 50,000 / 252  
**Script**: `asset/equity/engine/validation/script/benchmark_check_barrier_analytical.py`  
**Status**: ⚠️ WARNINGS (7/8)

Benchmark cases included:
- Up-and-out call (continuous)
- Up-and-in call (continuous)
- Down-and-out put (continuous)
- Up-and-out call (expiry monitoring)
- Up-and-out call (discrete, daily)
- Down-and-out put (discrete, daily)
- Up-and-out call (discrete, monthly)
- Down-and-out put (discrete, monthly)

### 3.1 Benchmark Results

| Case | Analytical | MC | SE | Error | Status |
|------|-----------|----|----|-------|--------|
| Up-and-out Call (cont) | 0.6863 | 0.6906 | 0.0108 | 0.63% | ✅ PASS |
| Up-and-in Call (cont) | 10.6622 | 10.7326 | 0.0817 | 0.66% | ✅ PASS |
| Down-and-out Put (cont) | 1.1953 | 1.1959 | 0.0145 | 0.05% | ✅ PASS |
| Up-and-out Call (expiry) | 0.6994 | 0.6814 | 0.0088 | 2.65% | ✅ PASS |
| Up-and-out Call (disc, daily) | 0.8105 | 0.8083 | 0.0122 | 0.27% | ✅ PASS |
| Down-and-out Put (disc, daily) | 1.3492 | 1.3406 | 0.0159 | 0.64% | ✅ PASS |
| Up-and-out Call (disc, monthly) | 1.3577 | 1.2184 | 0.0155 | 11.44% | ⚠️ FAIL |
| Down-and-out Put (disc, monthly) | 1.9674 | 1.9005 | 0.0192 | 3.52% | ✅ PASS |

**Notes:**
- BGK barrier shift is an approximation whose accuracy improves as observation interval `dt` shrinks. The daily (252) discrete cases match MC closely, while the monthly case shows a larger deviation consistent with coarse monitoring.

---

## 4. User Test Cases

No user-provided cases were supplied. To add, provide:
`S=100, K=100, H=120, T=1.0, r=0.03, sigma=0.25, expected=...`

---

## 5. Recommendations

1. **Extend benchmark grid**  
   Add more moneyness and barrier-distance cases to stress edge behavior.

---

## Appendix

### A. Script Execution Commands

```bash
# Run boundary checks
python asset/equity/engine/validation/script/boundary_check_barrier_analytical.py

# Run benchmark checks
python asset/equity/engine/validation/script/benchmark_check_barrier_analytical.py
```
