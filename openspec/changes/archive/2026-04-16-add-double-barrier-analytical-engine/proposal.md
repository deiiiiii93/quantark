## Why

QuantArk currently supports single-barrier options but lacks a dedicated analytical pricer for double-barrier options. Double-barrier options are widely traded structured products and risk management tools. Adding an analytical engine based on the Ikeda & Kuintomo (1992) formula will provide fast, accurate pricing for continuous, daily, and expiry-monitored double-barrier instruments.

## What Changes

- Add `DoubleBarrierOptionAnalyticalEngine` in `asset/equity/engine/analytical/`
- Extend or reuse existing barrier product definitions to support double-barrier instruments
- Implement Ikeda & Kuintomo (1992) infinite series pricing for:
  - Double knock-out call
  - Double knock-out put
- Support three observation modes:
  - **Continuous** — direct analytical formula
  - **Daily** — barrier-shift adjustment for discrete monitoring
  - **Expiry** — barriers checked only at maturity
- Add comprehensive unit tests with Table 4-15 benchmark validation for continuous cases
- Add smoke and monotonicity tests for daily and expiry observation modes

## Capabilities

### New Capabilities
- `double-barrier-analytical-engine`: Analytical pricing engine for double-barrier options (call/put, knock-out/knock-in) with continuous, daily, and expiry observation support

### Modified Capabilities
- None (implementation-only addition; no existing spec requirements change)

## Impact

- New engine file: `asset/equity/engine/analytical/double_barrier_option_engine.py`
- New test file: `test/test_double_barrier_option_engine.py`
- Minor additions to `asset/equity/product/` if a dedicated `DoubleBarrierOption` product class is introduced
- No breaking changes to existing APIs
