# Change: Add Phoenix Option Product for Equity Asset

## Why

Phoenix options are autocallable structured products that differ from Snowball options by providing **periodic coupon payments** when spot price exceeds a coupon barrier at observation dates. While Snowball options only pay coupon on knock-out events, Phoenix options pay coupons at each observation where the coupon barrier condition is met, making them more attractive to income-seeking investors.

The current codebase has a placeholder `create_phoenix_snowball()` helper in snowball-option-helpers spec, but lacks a dedicated Phoenix option product class with proper coupon barrier logic and memory coupon support.

## What Changes

- **ADDED**: `PhoenixOption` product class inheriting from `BaseEquityOption`
- **ADDED**: `CouponBarrierConfig` configuration class for coupon barrier settings
- **ADDED**: Phoenix option helper functions (`create_standard_phoenix()`, etc.)
- **ADDED**: Day count convention integration for coupon accrual (ACT/365, 30/360)
- **ADDED**: Memory coupon feature (accumulates missed coupons)
- **ADDED**: Unit tests for Phoenix option product
- **ADDED**: Demo script for Phoenix option

Key Features:
- Coupon barrier with per-period coupon payments
- Memory coupon (accumulates missed coupons when barrier not hit)
- Day count conventions for coupon calculation (DayCountConvention enum)
- Coupon payment timing (CouponPayType.INSTANT or EXPIRY)
- All variants: standard, reverse, step-down
- Reuses existing configs: BarrierConfig, PayoffConfig, AccrualConfig, AirbagConfig

## Impact

- Affected specs: `equity-phoenix-option` (new)
- Affected code:
  - `asset/equity/product/option/phoenix_option.py` (new)
  - `asset/equity/product/option/phoenix_config.py` (new)
  - `asset/equity/product/option/phoenix_helpers.py` (new)
  - `asset/equity/product/option/__init__.py` (export)
  - `test/test_phoenix_option.py` (new)
  - `example/phoenix_option_demo.py` (new)
