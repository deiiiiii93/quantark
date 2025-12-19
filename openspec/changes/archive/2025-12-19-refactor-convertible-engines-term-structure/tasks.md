# Tasks: Refactor Convertible Bond Engines for Term Structure Support

## Overview
Implement time-dependent rate and volatility support in Convertible Bond pricing engines.

## Task List

### 1. Update Jump-Diffusion PDE Engine
**File:** `asset/bond/engine/pde/convertible/jump_diffusion_engine.py`

- [x] 1.1 Modify `price_with_details()` to move rate/vol queries inside time loop
- [x] 1.2 Use `get_forward_rate(t, t + dt)` for local interest rate at each step
- [x] 1.3 Use `pricing_env.get_vol(strike, time_to_maturity)` at step endpoints and derive `sigma_step(t, t+dt)` from total variance
- [x] 1.4 Define a clear convention for time arguments (all times measured in years from valuation date)
- [x] 1.5 Handle edge case at `t=0` when computing total variance (`w(0)=0`)

**Validation:** Existing tests pass; new term structure test shows price difference.

---

### 2. Update TF PDE Engine
**File:** `asset/bond/engine/pde/convertible/tf_engine.py`

- [x] 2.1 Modify `price_with_details()` to move rate/vol queries inside time loop
- [x] 2.2 Use `get_forward_rate(t, t + dt)` for local interest rate at each step
- [x] 2.3 Use `pricing_env.get_vol(strike, time_to_maturity)` at step endpoints and derive `sigma_step(t, t+dt)` from total variance
- [x] 2.4 Define a clear convention for time arguments (all times measured in years from valuation date)
- [x] 2.5 Handle edge case at `t=0` when computing total variance (`w(0)=0`)

**Validation:** Existing tests pass; new term structure test shows price difference.

---

### 3. Update Trinomial Tree Engine
**File:** `asset/bond/engine/tree/convertible/trinomial_engine.py`

- [x] 3.1 Add `_calculate_max_vol_for_grid()` helper method
- [x] 3.2 Modify `price_with_details()` to use max_vol for grid spacing (u, d factors)
- [x] 3.3 Modify `_backward_induction()` to query local rate/vol at each time step
- [x] 3.4 Recalculate transition probabilities per time step with local parameters (rate forward + sigma_step)
- [x] 3.5 Ensure discount factor uses local forward rate

**Validation:** Existing tests pass; new term structure test shows price difference.

---

### 4. Add Warning to Binomial Engine
**File:** `asset/bond/engine/tree/convertible/binomial_engine.py`

- [x] 4.1 Add `_warn_if_non_flat_curves()` helper method
- [x] 4.2 Call warning method at start of `price_with_details()`
- [x] 4.3 Import `logging` and create logger for the module

**Validation:** Warning logged when using non-flat curves; no warning for flat curves.

---

### 5. Create Term Structure Tests
**File:** `test/test_convertible_bond_term_structure.py` (new)

- [x] 5.1 Create test fixture with sample convertible bond (4-year maturity)
- [x] 5.2 Create flat rate curve (5%)
- [x] 5.3 Create stepped rate curve (piecewise-constant: 1% for 0-2y, 9% for 2-4y) as a small test-only `RateCurve` subclass
- [x] 5.4 Test Jump-Diffusion engine: flat vs stepped produces different prices
- [x] 5.5 Test TF engine: flat vs stepped produces different prices
- [x] 5.6 Test Trinomial engine: flat vs stepped produces different prices
- [x] 5.7 Test Binomial engine: warning logged for non-flat curves
- [x] 5.8 Test Binomial engine: no warning for flat curves
- [x] 5.9 Verify all engines still produce same result for flat curves (backward compatibility)

**Validation:** All new tests pass; existing test suite passes.

---

### 6. Run Full Test Suite
- [x] 6.1 Run `python -m pytest test/` and verify all tests pass
- [x] 6.2 Run existing convertible bond tests specifically
- [x] 6.3 Verify no regressions in pricing accuracy for flat curve scenarios

**Validation:** Full test suite green.

---

## Dependencies
- Tasks 1, 2, 3, 4 can be done in parallel
- Task 5 depends on Tasks 1-4
- Task 6 depends on Task 5

## Estimated Complexity
- Tasks 1-2: Medium (similar changes to two files)
- Task 3: Medium-High (more complex due to grid stability considerations)
- Task 4: Low (simple warning addition)
- Task 5: Medium (comprehensive test coverage)
- Task 6: Low (running existing tests)
