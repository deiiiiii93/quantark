# Change: Add PDE Greeks Calculation Mode

## Why

Currently `GreeksCalculator.calculate_numerical_greeks()` always uses the bump method (finite difference) for delta/gamma calculation. However, PDE engines have their own `calculate_greeks()` method that extracts Greeks directly from the PDE solution grid, which is:

- **More accurate**: Uses the actual solution surface instead of finite differences
- **More efficient**: Reuses the PDE solution already computed for pricing
- **Consistent**: Greeks are consistent with the PDE pricing method

The bump method with PDE engines also has numerical issues due to:
1. Re-solving the PDE multiple times with bumped spot prices
2. Grid interpolation artifacts when spot is between grid points

## What Changes

The system SHALL provide the following changes:

- **NEW** `GreeksCalculationMode` enum in `util/enum/engine_enums.py` with values:
  - `BUMP`: Always use finite difference bump method (default, maintains current behavior)
  - `ENGINE`: Use engine's `calculate_greeks()` if available
  - `AUTO`: Use engine method for PDE engines, bump otherwise

- **NEW** `engine_type` attribute on `BaseEngine` for easy engine type detection
  - Values: `ANALYTICAL`, `MONTE_CARLO`, `PDE`, `QUADRATURE`, `TREE`
  - The system SHALL set this on all concrete engine classes

- **MODIFIED** `GreeksCalculator`:
  - The system SHALL add a new `greeks_mode` parameter in `__init__()` (defaults to `BUMP` for backward compatibility)
  - The system SHALL add a new `_should_use_engine_greeks()` helper method
  - The system SHALL modify `calculate_numerical_greeks()` to conditionally use engine's `calculate_greeks()`

- **FIXED** `PDEEngine.calculate_greeks()`:
  - The system SHALL add delegation to appropriate PDE solver's grid-based method
  - Previously, calling `calculate_greeks()` on `PDEEngine` used base class bump method

## Impact

- Affected specs: None (new feature, backward compatible)
- Affected code:
  - `util/enum/engine_enums.py` - Added `GreeksCalculationMode`
  - `asset/equity/engine/base_engine.py` - Added `engine_type` attribute
  - All engine classes - Set `engine_type` class attribute
  - `asset/equity/riskmeasures/greeks_calculator.py` - Added `greeks_mode` parameter
  - `asset/equity/engine/pde_engine.py` - Added `calculate_greeks()` delegation

- Backward compatibility: Fully maintained (default is `BUMP` mode)
- Dependencies: None
- Downstream dependencies: None (opt-in feature)

## Alternatives Considered

1. **Hardcode PDE detection**: Could check if engine has `calculate_greeks()` method, but:
   - Less explicit and harder to understand
   - Doesn't allow user control over calculation method

2. **Always use engine method**: Could break existing code that relies on bump method
   - Bump method is more universally applicable
   - Some engines may have incomplete `calculate_greeks()` implementations

3. **Separate PDE-specific calculator**: Would duplicate code and be less ergonomic
