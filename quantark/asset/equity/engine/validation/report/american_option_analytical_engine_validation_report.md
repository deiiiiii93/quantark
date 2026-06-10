# Validation Report: American Option Analytical Engine

**Generated**: 2025-02-14  
**Engine**: `asset/equity/engine/analytical/american_option_engine.py`  
**Reference**: `asset/equity/engine/docs/ameopt_analytical_engine.md`  
**Benchmark**: `asset/equity/engine/mc/american_option_mc_engine.py`

---

## Executive Summary

| Check Type | Status | Pass Rate |
|------------|--------|-----------|
| Method Implementation | ✅ PASS | - |
| Boundary Checks | ✅ PASS | 100% (8/8) |
| Benchmark Checks | ✅ PASS | 100% (12/12) |
| User Cases | ⚠️ NOT PROVIDED | - |

**Overall Status**: ✅ VALIDATED

---

## 1. Method Description

### 1.1 Pricing Method Summary

The engine implements three common approximation methods for American options:

1. **BS93** (Bjerksund-Stensland 1993) - Single-barrier approximation  
2. **BS02** (Bjerksund-Stensland 2002) - Two-barrier approximation  
3. **BAW** (Barone-Adesi-Whaley) - Quadratic approximation with critical price search  

Calls without dividends are short-circuited to European pricing, as early
exercise is not optimal in that case.

### 1.2 Reference Comparison

| Aspect | Reference | Implementation | Match |
|--------|-----------|----------------|-------|
| BS93 approximation | Bjerksund-Stensland 1993 | `_price_bs93()` | ✅ |
| BS02 approximation | Bjerksund-Stensland 2002 | `_price_bs02()` | ✅ |
| BAW approximation | Barone-Adesi-Whaley 1987 | `_price_baw()` | ✅ |
| Put-call transformation | Standard approach | `is_call` logic | ✅ |
| No-dividend call early exercise | Never optimal | Explicit shortcut | ✅ |
| Near-expiry handling | Return intrinsic | `T < MIN_MATURITY` | ✅ |

### 1.3 Implementation Quality

**Strengths:**
- Correct method dispatch across BS93/BS02/BAW
- Handles call early exercise constraints via explicit logic
- Clips extreme volatility/maturity for numerical stability

**Notes / Risks:**
- Numerical failures trigger a warning and fallback to European pricing, which
  can mask issues in edge cases. Consider logging more context if needed.

---

## 2. Boundary Checks

**Script**: `asset/equity/engine/validation/script/boundary_check_american_analytical.py`

Planned tests:
- Near expiry (T→0) → payoff
- Low volatility → intrinsic value
- High volatility → finite, positive
- Deep OTM / ITM checks
- American ≥ European (put)
- American call equals European when q=0
- Monotonicity in strike

**Result**: 8/8 passed (100%).

| Test | Status | Notes |
|------|:-------|-------|
| Near expiry (T→0) | ✅ | Price close to payoff |
| Low volatility (σ→0) | ✅ | Matches deterministic forward value |
| High volatility (σ=100%) | ✅ | Finite positive price |
| Deep OTM call | ✅ | Near zero |
| Deep ITM put | ✅ | Above intrinsic |
| American ≥ European (put) | ✅ | Dominance holds |
| Call (q=0) equals European | ✅ | Early exercise not optimal |
| Call monotonicity in strike | ✅ | Decreasing with strike |

---

## 3. Benchmark Comparison

**Benchmark Engine**: `asset/equity/engine/mc/american_option_mc_engine.py`  
**Tolerance**: 5%  
**Script**: `asset/equity/engine/validation/script/benchmark_check_american_analytical.py`

Planned cases:
- ATM call/put, ITM put, OTM call  
- Methods: BS93, BS02, BAW

**Result**: 12/12 passed (100%).

| Case | Analytical | MC | SE | Error | Status |
|------|-----------:|---:|---:|------:|:-------|
| BS93 ATM Call (q=2%) | 9.2270 | 9.0895 | 0.0382 | 1.51% | ✅ |
| BS93 ATM Put (q=2%) | 6.5736 | 6.6317 | 0.0238 | 0.88% | ✅ |
| BS93 ITM Put (q=2%) | 21.1629 | 21.1787 | 0.0328 | 0.07% | ✅ |
| BS93 OTM Call (q=0%) | 2.1364 | 2.0897 | 0.0185 | 2.23% | ✅ |
| BS02 ATM Call (q=2%) | 9.2270 | 9.0895 | 0.0382 | 1.51% | ✅ |
| BS02 ATM Put (q=2%) | 6.5507 | 6.6317 | 0.0238 | 1.22% | ✅ |
| BS02 ITM Put (q=2%) | 21.0796 | 21.1787 | 0.0328 | 0.47% | ✅ |
| BS02 OTM Call (q=0%) | 2.1364 | 2.0897 | 0.0185 | 2.23% | ✅ |
| BAW ATM Call (q=2%) | 9.2276 | 9.0895 | 0.0382 | 1.52% | ✅ |
| BAW ATM Put (q=2%) | 6.6722 | 6.6317 | 0.0238 | 0.61% | ✅ |
| BAW ITM Put (q=2%) | 21.1570 | 21.1787 | 0.0328 | 0.10% | ✅ |
| BAW OTM Call (q=0%) | 2.1364 | 2.0897 | 0.0185 | 2.23% | ✅ |

---

## 4. User Test Cases

No user test cases provided.  
Provide cases in the form: `S=100, K=100, T=1, r=0.05, σ=0.2, expected=10.5`

---

## 5. Recommendations

1. Maintain benchmark coverage for all three methods when tuning parameters.  
2. Add user cases if you need regression targets for specific market regimes.

---

## Appendix

### A. Script Execution Commands

```bash
python asset/equity/engine/validation/script/boundary_check_american_analytical.py
python asset/equity/engine/validation/script/benchmark_check_american_analytical.py
```
