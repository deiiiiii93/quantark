# Phoenix Engine Pricing Investigation - Root Cause Analysis

## Executive Summary

Investigated pricing discrepancies between Phoenix Quadrature, PDE, and Monte Carlo engines. Identified **one critical bug** (already fixed) and several algorithmic differences that explain the observed variations.

## Critical Bug Fixed ✓

**File:** `asset/equity/engine/mc/phoenix_mc_engine.py:755`
```python
# BEFORE (Bug):
if product.accrual_config.coupon_pay_type == CouponPayType.INSTANT:

# AFTER (Fixed):
if product.coupon_config.coupon_pay_type == CouponPayType.INSTANT:
```
The `AccrualConfig` class doesn't have `coupon_pay_type` - this attribute exists only in `CouponBarrierConfig`.

---

## Diagnostic Test Results

### Test Setup
- Product: Standard Phoenix
- Spot: 100, Strike: 100, KO: 103, KI: 75, Coupon: 95
- Vol: 20%, Rate: 3%, 12 monthly observations
- Comparison: Quad (401 grid), PDE (200x100), MC (50K paths)

### Results Summary

| Test Case | Quad | PDE | MC | Q-M Diff | P-M Diff |
|-----------|------|-----|----|----------|----------|
| **Baseline (all features)** | 25,533 | 23,934 | 25,553 | **0.1%** | **6.3%** |
| KI Disabled (KI=999) | -11,082 | -12,880 | -11,805 | 6.1% | 9.1% |
| Coupon Smooth (barrier=1) | 29,880 | 28,629 | 30,172 | 1.0% | 5.1% |
| KO Far (KO=150) | 93,088 | 89,407 | 90,436 | 2.9% | 1.1% |
| KO Always (KO=90) | 10,322 | 10,383 | 10,551 | 2.2% | 1.6% |
| **No Barriers** | -44,901 | -44,943 | -45,043 | **0.3%** | **0.2%** |

### Key Findings

1. **Core diffusion is correct** - With no barriers, all engines agree within 0.3%

2. **Quad FFT oscillation is real:**
   ```
   201 points: 19,742
   301 points: 26,580  ← +35% jump!
   401 points: 25,533
   501 points: 24,047
   601 points: 22,441
   801 points: 24,437  ← Oscillates back up
   1001 points: 23,658
   ```

3. **PDE-MC 6% baseline difference** comes from:
   - Grid time alignment for observation times
   - KO/KI jump application at grid points
   - When KO is moved far away, PDE-MC drops to 1.1%

4. **Coupon barrier has minimal impact** - Smoothing only changes difference by +0.9%

---

## Algorithmic Differences (By Design)

### Memory State Tracking

| Engine | Method | Complexity |
|--------|--------|------------|
| **Quad** | N+1 value surfaces | O(N × grid) |
| **PDE** | N+1 value surfaces | O(N × grid) |
| **MC** | Per-path accrued array | O(paths × obs) |

### Barrier Observation

| Engine | KO Timing | KI Timing |
|--------|-----------|-----------|
| **Quad** | Nearest grid point | Discrete or Brownian bridge |
| **PDE** | Nearest grid point | Discrete only |
| **MC** | Exact observation times | Discrete or Brownian bridge |

### Convergence Behavior

| Engine | Convergence | Notes |
|--------|-------------|-------|
| **Quad** | Non-monotonic | FFT boundary issues |
| **PDE** | Monotonic | Stable with grid refinement |
| **MC** | √N convergence | Statistical noise |

---

## Recommendations

### For Users

1. **Use MC for:**
   - Products with continuous KI (has Brownian bridge correction)
   - High volatility scenarios (>30%)
   - Final validation/p benchmarking

2. **Use PDE for:**
   - Products with many observations (up to 50)
   - Fast greeks calculation
   - Stable grid-based pricing

3. **Use Quad for:**
   - Quick calibration runs
   - Low volatility scenarios (<25%)
   - When speed is critical

### For Developers

1. **Fix Quad FFT oscillation:**
   - Investigate boundary treatment in `_diffuse_fft`
   - Consider adding damping or padding
   - Test with higher grid points for convergence

2. **Improve PDE time alignment:**
   - Ensure observation times align with grid points
   - Or implement interpolation for exact times

3. **Add convergence tests:**
   - MC: Test with increasing paths (verify 1/√N convergence)
   - PDE: Test with increasing grid (verify monotonic convergence)
   - Quad: Test with increasing points (verify bounded oscillation)

---

## Files Modified

1. `asset/equity/engine/mc/phoenix_mc_engine.py` - Bug fix (line 755)
2. `test/test_phoenix_engine_comparison.py` - New comprehensive test suite
3. `example/phoenix_diagnostic.py` - Diagnostic test for root cause analysis

---

## Conclusion

The pricing differences between Phoenix engines are primarily due to:
1. **FFT boundary effects in Quad** (non-monotonic convergence)
2. **Grid time alignment in PDE** (6% baseline difference from MC)
3. **Different KO/KI observation implementations** (design choices)

The core diffusion logic (Black-Scholes) is consistent across all engines, as demonstrated by the excellent agreement (0.3%) when all barriers are removed.
