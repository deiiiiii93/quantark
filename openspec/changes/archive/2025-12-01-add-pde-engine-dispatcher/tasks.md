# Implementation Tasks

## 1. Enum Definition
- [x] 1.1 Add `PDEMethod` enum to `util/enum/engine_enums.py` with methods: `CRANK_NICOLSON`, `EXPLICIT_EULER`, `IMPLICIT_EULER`
- [x] 1.2 Update `EngineType` enum to support `EngineType.PDE(PDEMethod.CRANK_NICOLSON)` pattern

## 2. PDEEngine Implementation
- [x] 2.1 Create `asset/equity/engine/pde_engine.py` with `PDEEngine` class extending `BaseEngine`
- [x] 2.2 Implement `__init__` accepting `PDEParams` and optional `method` parameter
- [x] 2.3 Implement product-to-solver mapping (European→EuropeanPDESolver, American→AmericanPDESolver, etc.)
- [x] 2.4 Implement `price()` method with automatic dispatch logic
- [x] 2.5 Add input validation (unsupported product types)
- [x] 2.6 Add comprehensive docstrings following project conventions

## 3. Integration
- [x] 3.1 Export `PDEEngine` from `asset/equity/engine/__init__.py`
- [x] 3.2 Add to `__all__` list in `asset/equity/engine/__init__.py`

## 4. Testing
- [x] 4.1 Write unit tests in `test/test_pde_engine_dispatcher.py` for European options
- [x] 4.2 Add tests for American options
- [x] 4.3 Add tests for barrier options (single and double)
- [x] 4.4 Add tests for one-touch options (single and double)
- [x] 4.5 Test Greeks calculation integration with `GreeksCalculator`
- [x] 4.6 Test error handling for unsupported product types
- [x] 4.7 Verify numerical results match direct solver usage

## 5. Examples
- [x] 5.1 Create `example/pde_engine_demo.py` demonstrating all features
- [x] 5.2 Run example to verify it works correctly

## 6. Validation
- [x] 6.1 Run `python -m pytest test/test_pde_engine_dispatcher.py -v` (23/23 tests passed)
- [x] 6.2 Run example script successfully
- [x] 6.3 Verify integration with existing test suite