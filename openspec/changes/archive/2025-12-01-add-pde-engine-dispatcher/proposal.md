# Change: Add Unified PDE Engine for Greeks Calculation

## Why
The PDE solvers (`EuropeanPDESolver`, `AmericanPDESolver`, `BarrierPDESolver`, etc.) in `asset/equity/engine/pde/` are product-specific implementations that extend `BasePDESolver` but not the `BaseEngine` interface expected by `GreeksCalculator`. This prevents reuse of the general-purpose `GreeksCalculator.calculate_numerical_greeks()` method for PDE-based pricing.

A unified `PDEEngine` is needed to:
- Extend `BaseEngine` and satisfy the `price()` interface required by `GreeksCalculator`
- Automatically dispatch pricing requests to the appropriate PDE solver based on product type
- Enable seamless Greeks calculation for all PDE-priced products using the existing `GreeksCalculator` infrastructure

## What Changes
- Create new `PDEEngine` class in `asset/equity/engine/pde_engine.py` that extends `BaseEngine`
- Implement product-to-solver mapping and automatic dispatch logic
- Add support for all existing PDE solvers (European, American, Barrier, DoubleBarrier, OneTouch, DoubleOneTouch)
- Export `PDEEngine` from `asset/equity/engine/__init__.py`
- Update `util/enum/engine_enums.py` to include PDE method selection via two-level enum pattern: `EngineType.PDE(PDEMethod.CRANK_NICOLSON)`
- Ensure `PDEEngine` works seamlessly with `GreeksCalculator.calculate_numerical_greeks()`

## Impact
- **Affected specs**: `equity-pde-engine` (new capability)
- **Affected code**:
  - New file: `asset/equity/engine/pde_engine.py`
  - Modified: `asset/equity/engine/__init__.py` (export `PDEEngine`)
  - Modified: `util/enum/engine_enums.py` (add `PDEMethod` enum)
- **Dependencies**: Requires existing PDE solvers in `asset/equity/engine/pde/`
- **Benefits**:
  - Unified interface for all equity pricing engines
  - Greeks calculation via PDE methods for all supported products
  - Consistent API across analytical, Monte Carlo, and PDE engines
