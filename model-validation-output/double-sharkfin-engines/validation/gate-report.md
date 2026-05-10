# 验证结果 / Validation Gate Report

## 1. Gate Result

**Status**: PASS_WITH_NOTES

## 2. Independent Checks

The focused test suite verifies:

- expiry call analytical price against direct truncated-lognormal expectation
- expiry put analytical price against direct truncated-lognormal expectation
- continuous no-rebate analytical price against the existing double-barrier option engine
- pay-at-hit rebate value greater than pay-at-expiry rebate value under positive rates
- discrete daily analytical pricing equivalence to BGK-shifted continuous pricing
- MC multiplier scaling
- MC discrete daily schedule execution
- MC two-level enum method selection
- product type rejection for both engines

## 3. Notes

The analytical continuous cash leg uses a survival-series plus quadrature
representation. This is mathematically traceable and deterministic, but it is
not an independent closed-form first-exit density implementation.
