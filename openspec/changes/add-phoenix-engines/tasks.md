# Tasks: Add Phoenix Pricing Engines (MC, PDE, QUAD)

## Overview

Implement three pricing engines for Phoenix (autocallable with periodic coupons) options following the established patterns from Snowball engines.

## Prerequisites

- [x] Existing `PhoenixOption` product with helper functions
- [x] Existing `CouponBarrierConfig` configuration
- [x] Existing `SnowballMCEngine` as reference implementation
- [x] Existing `SnowballPDESolver` as reference implementation
- [x] Existing `SnowballQuadEngine` as reference implementation
- [x] Existing `BasePDESolver` base class
- [x] Existing `QuadratureCore` quadrature infrastructure

## Phase 1: Monte Carlo Engine

### 1.1 Core MC Engine Implementation

- [ ] **1.1.1** Create `phoenix_mc_engine.py` with `PhoenixMCEngine` class skeleton
  - Inherit from `BaseEngine`
  - Define `PhoenixMCResult` dataclass with coupon breakdown
  - **Validation**: Class instantiates without error

- [ ] **1.1.2** Implement coupon barrier check logic
  - Check coupon barrier at each observation date
  - Handle step-down coupon barriers (list of barriers)
  - Track memory coupon accumulation
  - **Validation**: Coupons trigger correctly at barrier crossings

- [ ] **1.1.3** Implement day count convention for coupon accrual
  - Support ACT_365, ACT_360, 30_360 conventions
  - Calculate correct time fractions for coupon payments
  - **Validation**: Time fractions match expected values

- [ ] **1.1.4** Implement INSTANT vs EXPIRY coupon payment timing
  - INSTANT: Discount each coupon from observation date
  - EXPIRY: Accumulate coupons, pay at expiry
  - **Validation**: Payment timing affects price correctly

- [ ] **1.1.5** Implement `price()` method
  - Generate paths using QMC
  - Apply KO/KI/coupon barriers at observation dates
  - Calculate expected discounted payoff
  - **Validation**: Returns valid price for test cases

- [ ] **1.1.6** Implement `calculate_event_stats()` method
  - Per-observation coupon probabilities
  - KO/KI probabilities
  - Expected cashflow decomposition
  - **Validation**: Event stats match path-level analysis

### 1.2 MC Integration and Testing

- [ ] **1.2.1** Add `PhoenixMCEngine` to `mc/__init__.py` exports
  - **Validation**: Import from `asset.equity.engine.mc` works

- [ ] **1.2.2** Create `test/test_phoenix_mc.py` with unit tests
  - Coupon barrier trigger tests
  - Memory coupon accumulation tests
  - Day count convention tests
  - Payment timing tests
  - **Validation**: All unit tests pass

- [ ] **1.2.3** Add integration tests comparing Phoenix MC vs Snowball MC
  - Standard phoenix (monthly coupons)
  - Reverse phoenix
  - Memory vs non-memory coupons
  - **Tolerance**: 0.5% relative error for equivalent structures

## Phase 2: PDE Engine

### 2.1 Core PDE Solver Implementation

- [ ] **2.1.1** Create `phoenix_pde_solver.py` with `PhoenixPDESolver` class skeleton
  - Inherit from `BasePDESolver`
  - Add dual grid storage (`_grid_v0`, `_grid_v1`)
  - Add coupon state tracking
  - **Validation**: Class instantiates without error

- [ ] **2.1.2** Implement `_build_grids()` method for Phoenix-specific grids
  - Extract critical points: spot, strike, all barriers, coupon barriers
  - Build time grid aligned with all observation times
  - **Validation**: Grid nodes include coupon barrier levels

- [ ] **2.1.3** Implement terminal conditions for V0 and V1
  - Handle Phoenix-specific maturity payoffs
  - Apply accumulated coupon if memory coupon active
  - **Validation**: Terminal matches `product.get_maturity_payoff_v0/v1()`

- [ ] **2.1.4** Implement coupon barrier jump application
  - Add coupon value to grid when barrier hit
  - Handle memory coupon accumulation across observations
  - **Validation**: Coupon payments reflected in grid values

- [ ] **2.1.5** Implement `price()` method orchestrating backward induction
  - Initialize grids
  - Set terminal conditions
  - Step backward applying PDE, KO, KI, coupon jumps
  - **Validation**: Returns valid price for test cases

### 2.2 PDE Integration and Testing

- [ ] **2.2.1** Update `PDEEngine` to dispatch PhoenixOption
  - Add import and type check for PhoenixOption
  - Create PhoenixPDESolver and delegate
  - **Validation**: `PDEEngine.price(phoenix, env)` works

- [ ] **2.2.2** Add `PhoenixPDESolver` to `pde/__init__.py` exports
  - **Validation**: Import from `asset.equity.engine.pde` works

- [ ] **2.2.3** Create `test/test_phoenix_pde.py` with unit tests
  - Terminal condition tests
  - Coupon barrier jump tests
  - Grid construction tests
  - **Validation**: All unit tests pass

- [ ] **2.2.4** Add integration tests comparing PDE vs MC
  - Standard phoenix configurations
  - **Tolerance**: 1% relative error with 100k MC paths

## Phase 3: Quadrature Engine

### 3.1 Core QUAD Engine Implementation

- [ ] **3.1.1** Create `phoenix_quad_engine.py` with `PhoenixQuadEngine` class skeleton
  - Inherit from appropriate quad base
  - Add regime surfaces for KI state and coupon tracking
  - **Validation**: Class instantiates without error

- [ ] **3.1.2** Implement terminal conditions for quadrature
  - V_in and V_out initialization at maturity
  - Include terminal coupon accumulation
  - **Validation**: Terminal matches product payoffs

- [ ] **3.1.3** Implement discrete KO observation handling
  - Apply KO payoffs when barrier breached
  - Handle coupon at KO if applicable
  - **Validation**: KO jump applied correctly

- [ ] **3.1.4** Implement discrete KI and coupon barrier transitions
  - KI: V_out → V_in transition
  - Coupon: Add coupon value to surfaces when barrier hit
  - **Validation**: Transitions work correctly

- [ ] **3.1.5** Implement continuous KI via Brownian bridge (if applicable)
  - State-transition probabilities for continuous monitoring
  - **Validation**: Continuous KI matches discrete with fine grid

- [ ] **3.1.6** Implement `price()` method with backward recursion
  - Single pass over observation grid
  - FFT convolution for value propagation
  - **Validation**: Returns valid price for test cases

### 3.2 QUAD Integration and Testing

- [ ] **3.2.1** Update quad engine dispatcher for PhoenixOption
  - Add routing to PhoenixQuadEngine
  - **Validation**: Unified interface works

- [ ] **3.2.2** Add `PhoenixQuadEngine` to `quad/__init__.py` exports
  - **Validation**: Import from `asset.equity.engine.quad` works

- [ ] **3.2.3** Create `test/test_phoenix_quad.py` with unit tests
  - Terminal condition tests
  - Barrier observation tests
  - Coupon accumulation tests
  - **Validation**: All unit tests pass

- [ ] **3.2.4** Add integration tests comparing QUAD vs MC and PDE
  - Standard phoenix configurations
  - **Tolerance**: 0.5% relative error

## Phase 4: Documentation and Cleanup

- [ ] **4.1** Add docstrings to all public methods in new engines
  - Include Args, Returns, Raises sections
  - **Validation**: All public methods have docstrings

- [ ] **4.2** Update `asset/equity/CLAUDE.md` with Phoenix engine info
  - Usage examples for each engine
  - Engine comparison table
  - **Validation**: Documentation is complete

- [ ] **4.3** Run full test suite and fix any regressions
  - `python -m pytest test/test_phoenix_*.py -v`
  - **Validation**: All tests pass

## Verification Checklist

- [ ] All three engines produce consistent prices (within 0.5%)
- [ ] All Phoenix helper configurations work with each engine
- [ ] Greeks are finite and stable
- [ ] Performance targets met (MC < 2s, PDE < 1s, QUAD < 0.5s)
- [ ] No regressions in existing tests
- [ ] Documentation updated

## Notes

- Use `product.is_coupon_triggered()` for coupon barrier checks
- Use `product.get_coupon_payoff()` for coupon calculations
- Use `product.get_maturity_payoff_v0/v1()` for terminal payoffs
- Leverage existing Snowball engine patterns for KO/KI handling
- The `CouponBarrierConfig` provides all coupon-related configuration
