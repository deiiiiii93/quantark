# Add Snowball Option Helper Functions

## Summary

Add a helper module with factory functions to simplify creation of common Snowball option structures. The `SnowballOption` class has ~36 parameters across 4 configuration objects (`BarrierConfig`, `PayoffConfig`, `AccrualConfig`, and core parameters), making it complex to instantiate correctly. These helpers provide pre-configured templates for common market structures.

## Motivation

Currently, creating a Snowball option requires:
1. Understanding 13+ barrier configuration parameters
2. Configuring 8+ payoff parameters correctly
3. Setting 5+ accrual parameters
4. Coordinating all parameters for a valid structure

Example from `example/snowball_mc_demo.py` shows 30+ lines of code just to create a basic snowball. For common structures like standard snowballs, step-down snowballs, or European knock-in snowballs, this complexity is unnecessary since most parameters follow predictable patterns.

## Proposed Solution

Create a `snowball_helpers.py` module in `asset/equity/product/option/` with factory functions for common structures:

1. **`create_standard_snowball()`** - Basic snowball with flat KO barrier and continuous KI
2. **`create_stepdown_snowball()`** - KO barrier decreases over time (common in China market)
3. **`create_european_ki_snowball()`** - KI only observed at maturity (European-style)
4. **`create_parachute_snowball()`** - Last KO barrier equals KI barrier (guaranteed exit if not knocked in)
5. **`create_phoenix_snowball()`** - Snowball with periodic coupons regardless of KO
6. **`create_airbag_snowball()`** - Snowball with participation-based KI protection

Each helper:
- Accepts minimal required parameters (initial_price, strike, maturity, notional)
- Provides sensible defaults for structure-specific parameters
- Allows override of any parameter via `**kwargs`
- Returns a fully configured `SnowballOption` instance

## Impact

- **New Files**: `asset/equity/product/option/snowball_helpers.py`
- **Modified Files**: `asset/equity/product/option/__init__.py` (export helpers)
- **Risk**: Low - additive change, no modification to existing classes
- **Backward Compatibility**: Fully compatible

## Alternatives Considered

1. **Builder Pattern**: More complex API, overkill for this use case
2. **Subclasses**: Would duplicate validation logic and complicate inheritance
3. **Class Methods**: Would clutter the main `SnowballOption` class

Factory functions are the simplest, most Pythonic approach that aligns with existing patterns in the codebase (e.g., `create_pricing_env`, `create_engine`).
