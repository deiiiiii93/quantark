# Tasks: Add Convertible Bond Product and Pricing Engines

## 1. Product Implementation
- [x] 1.1 Add `EngineType.TREE` to `util/enum/engine_enums.py` (for two-level typing)
- [x] 1.2 Create `ConvertibleBondMethod` enum in `util/enum/engine_enums.py` (BINOMIAL_GS, TRINOMIAL_HW, JUMP_DIFFUSION, TF)
- [x] 1.3 Create `CallScheduleEntry` and `PutScheduleEntry` dataclasses for schedule representation
- [x] 1.4 Create `ConvertibleBond` dataclass in `asset/bond/product/convertible/convertible_bond.py`
- [x] 1.5 Implement core attributes (face_value, maturity, coupon, conversion_ratio)
- [x] 1.6 Implement call/put schedule handling
- [x] 1.7 Implement credit attributes (credit_spread, hazard_rate, recovery_rate)
- [x] 1.8 Implement dividend handling (continuous and discrete)
- [x] 1.9 Implement `get_cashflows()` method
- [x] 1.10 Implement `calculate_accrued_interest()` method
- [x] 1.11 Implement `parity(stock_price)` helper method
- [x] 1.12 Implement validation logic
- [x] 1.13 Write unit tests for ConvertibleBond product

## 2. Tree Engine Implementation
- [x] 2.1 Create `asset/bond/engine/tree/convertible/` directory structure
- [x] 2.2 Create `ConvertibleBondTreeParams` dataclass for tree configuration
- [x] 2.3 Implement `ConvertibleBondBinomialEngine` with GS credit-adjusted model
- [x] 2.4 Implement binomial tree construction with up/down moves
- [x] 2.5 Implement credit-adjusted discount rate calculation
- [x] 2.6 Implement conversion probability tracking at each node
- [x] 2.7 Implement early exercise handling (conversion, call, put)
- [x] 2.8 Implement coupon handling in tree
- [x] 2.9 Implement `ConvertibleBondTrinomialEngine` with HW default model
- [x] 2.10 Implement trinomial tree with default branch
- [x] 2.11 Implement recovery value on default
- [x] 2.12 Implement discrete dividend handling in tree
- [x] 2.13 Implement delta calculation from tree
- [x] 2.14 Write unit tests for tree engines

## 3. PDE Engine Implementation
- [x] 3.1 Create `asset/bond/engine/pde/convertible/` directory structure
- [x] 3.2 Create `ConvertibleBondPDEParams` dataclass for PDE configuration
- [x] 3.3 Implement `ConvertibleBondJumpDiffusionEngine` (Bloomberg OVCV)
- [x] 3.4 Implement jump-diffusion PDE formulation
- [x] 3.5 Implement hazard rate and stock jump terms
- [x] 3.6 Implement boundary conditions (maturity, high/low stock)
- [x] 3.7 Implement time stepping schemes (Crank-Nicolson, implicit Euler)
- [x] 3.8 Implement Rannacher smoothing
- [x] 3.9 Implement `ConvertibleBondTFEngine` (Tsiveriotis-Fernandes)
- [x] 3.10 Implement coupled PDE system for u and v
- [x] 3.11 Implement COCB boundary conditions
- [x] 3.12 Implement coupon handling as source term
- [x] 3.13 Implement Greeks calculation from PDE grid
- [x] 3.14 Implement convergence verification
- [x] 3.15 Write unit tests for PDE engines

## 4. Facade Engine Implementation
- [x] 4.1 Create `asset/bond/engine/convertible/` directory structure
- [x] 4.2 Implement `ConvertibleBondEngine` facade class
- [x] 4.3 Implement method-based dispatch logic
- [x] 4.4 Implement two-level enum method selection
- [x] 4.5 Implement parameter passthrough to specialized engines
- [x] 4.6 Implement `ConvertibleBondResult` container for `price_with_details()` (do not change `price()` return type)
- [x] 4.7 Implement Greeks calculation delegation
- [x] 4.8 Implement error handling and validation
- [x] 4.9 Write unit tests for facade engine

## 5. Integration and Testing
- [x] 5.1 Update `asset/bond/engine/__init__.py` exports
- [x] 5.2 Update `asset/bond/product/__init__.py` exports
- [x] 5.3 Create integration tests comparing methods
- [x] 5.4 Create benchmark tests against known values (from documentation examples)
- [x] 5.5 Verify numerical consistency between tree and PDE methods
- [x] 5.6 Test integration with GreeksCalculator
- [x] 5.7 Test PricingEnvironment integration

## 6. Documentation and Examples
- [x] 6.1 Create `example/convertible_bond_demo.py` demonstrating product creation and pricing
- [x] 6.2 Create example showing method comparison
- [x] 6.3 Add docstrings to all public classes and methods
- [x] 6.4 Update module-level docstrings

## Dependencies
- Tasks 2.x depend on 1.x completion (need product class)
- Tasks 3.x depend on 1.x completion (need product class)
- Tasks 4.x depend on 2.x and 3.x completion (facade wraps specialized engines)
- Tasks 5.x depend on 1.x-4.x completion
- Tasks 6.x can partially proceed in parallel

## Parallelizable Work
- 2.x (Tree engines) and 3.x (PDE engines) can be developed in parallel after 1.x
- 2.3-2.8 (Binomial) and 2.9-2.12 (Trinomial) can be developed in parallel
- 3.3-3.8 (Jump-Diffusion) and 3.9-3.12 (TF) can be developed in parallel
