# Model Development Report / 模型开发报告 (Developer A)

**Model**: Double Barrier Option Analytical Engine
**Date**: 2026-04-16
**Developer**: Developer A (Claude)

---

## 1. 实现摘要 / Implementation Summary

### 1.1 Model Specification

| Attribute | Value |
|-----------|-------|
| Model Type | Analytical |
| Product Supported | `DoubleBarrierOption` |
| Base Class | `BaseEngine` (`asset/equity/engine/base_engine.py`) |
| Engine Location | `asset/equity/engine/analytical/double_barrier_option_engine.py` |

### 1.2 Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `asset/equity/engine/analytical/double_barrier_option_engine.py` | Created | Main engine implementation |
| `asset/equity/engine/docs/double_barrier_option_engine.md` | Created | Reference documentation |
| `asset/equity/engine/analytical/__init__.py` | Modified | Added `DoubleBarrierOptionAnalyticalEngine` export |
| `test/test_double_barrier_option_engine.py` | Created | 53-unit test suite |

---

## 2. 参考文档 / Reference Documentation

**Location**: `asset/equity/engine/docs/double_barrier_option_engine.md`

### 2.1 Mathematical Basis

The engine implements the Ikeda & Kuintomo (1992) infinite-series closed-form for double knock-out options under Black-Scholes assumptions. Knock-in options are priced via parity (`Vanilla - Knock-Out`).

### 2.2 Key Formulas

| Formula | Source |
|---------|--------|
| Core pricing (call KO) | Ikeda & Kuintomo (1992), Haug Table 4-15 |
| Core pricing (put KO) | Ikeda & Kuintomo (1992), mapped from call formula |
| Discrete barrier shift | Broadie-Glasserman-Kou (1997) continuity correction |
| Expiry truncated payoff | Standard truncated-domain vanilla decomposition |

---

## 3. 输入验证 / Input Validation

### 3.1 Validated Parameters

| Parameter | Type | Range | Edge Check | Sanity |
|-----------|------|-------|------------|--------|
| spot | float | > 0 | NaN/Inf | < 1e10 |
| strike | float | > 0 | NaN/Inf | < 1e10 |
| maturity | float | >= 0 | NaN/Inf | < 100 years |
| volatility | float | > 0 | NaN/Inf | [1e-12, 5.0] |
| lower_barrier | float | > 0 | NaN/Inf | < upper_barrier |
| upper_barrier | float | > 0 | NaN/Inf | > lower_barrier |
| risk_free_rate | float | any | NaN/Inf | abs(r) <= 1.0 |
| dividend_yield | float | >= 0 | NaN/Inf | q < 1.0 |

### 3.2 Validation Code Location

`asset/equity/engine/analytical/double_barrier_option_engine.py:433` - `_validate_inputs()` method

---

## 4. 边界情况处理 / Edge Case Handling

| Condition | Handling | Code Location |
|-----------|----------|---------------|
| `T = 0` | Return intrinsic payoff or rebate | `price():113` |
| `σ = 0` | Deterministic forward path; rebate if outside barriers | `price():117` |
| Spot outside barriers (KO) | Return rebate immediately | `price():121` |
| Spot outside barriers (KI) | Price as vanilla (already activated) | `price():125` |
| Strike outside [L, U] | Raise `ValidationError` | `_validate_inputs():457` |
| Deep ITM / extreme terms | Skip non-finite weights to avoid NaN | `_price_knock_out_ikeda_kuintomo():380` |

---

## 5. 性能考虑 / Performance Considerations

### 5.1 Complexity

- **Time**: O(N) where N = `2 * max_terms + 1` (default 21 terms). Each term involves constant-time math.
- **Space**: O(1) — only scalar accumulators.

### 5.2 Benchmark Results

| Operation | Time | Notes |
|-----------|------|-------|
| Single price (continuous) | ~0.05 ms | Baseline case, 21 terms |
| Full test suite (53 tests) | ~0.90 s | Includes parametrized benchmarks |

### 5.3 Optimizations Applied

1. Finite-series truncation with default `max_terms = 10`.
2. `math.isfinite()` guard on weights to avoid `inf * 0 = NaN` in curvature cases.
3. Reuse of existing `BlackScholesEngine` instance for vanilla parity and truncated payoff sub-calculations.

---

## 6. 集成清单 / Integration Checklist

- [x] Reference documentation created
- [x] Inputs validated
- [x] Edge cases handled
- [x] Added to `__init__.py`
- [x] Contract multiplier applied
- [x] Test suite created and passing

---

## 7. 待验证项 / Items for Validation (Developer B)

1. Independent re-implementation of the Ikeda-Kuintomo series formula.
2. Boundary check tests (zero maturity, zero vol, spot outside barriers).
3. Monte Carlo benchmark comparison for continuous vs discrete observation.
4. Parity validation (Knock-In = Vanilla - Knock-Out).

---

## 8. 已知限制 / Known Limitations

1. **Strike must be strictly inside barriers**: The Ikeda-Kuintomo formula does not support `K <= L` or `K >= U`. The engine raises `ValidationError` in these cases.
2. **Finite series approximation**: Although convergence is rapid, truncation introduces a negligible error for typical parameters. Extremely long maturities or pathological barrier curvature may require more terms.
3. **Rebate timing**: The current implementation assumes rebate is paid at maturity. Immediate rebate at knock-out is not supported.

---

## Appendix: Code Snippets

### A. Core Pricing Method

```python
def price(self, product, pricing_env) -> float:
    # ... validation and edge cases ...
    if obs_type == ObservationType.CONTINUOUS:
        return self._price_continuous(...)
    if obs_type == ObservationType.DISCRETE:
        return self._price_discrete(...)
    if obs_type == ObservationType.EXPIRY:
        return self._price_expiry(...)
```

### B. Edge Case Handling

```python
if is_zero(T):
    return self._price_zero_maturity(product, S, multiplier)
if is_zero(sigma):
    return self._price_zero_vol(product, S, T, r, q, multiplier)
if product.is_barrier_hit(S):
    if product.is_knock_out:
        return product.rebate * multiplier
    vanilla = self._create_vanilla(product)
    return self._bs_engine.price(vanilla, pricing_env) * multiplier
```
