# Tasks: Enhance Phoenix Engines with Memory Coupon Support

## Overview
Implement multi-state backward induction in PDE and QUAD engines to correctly price Phoenix options with `memory_coupon=True`.

## Phase 1: Quadrature Engine Enhancement
Quad is easier to iterate on for this logic structure.

- [ ] **1.1** Update `PhoenixQuadEngine.price` to handle vector states
    - [ ] If `memory_coupon=False`, keep existing fast path.
    - [ ] If `memory_coupon=True`:
        - [ ] Initialize list of $N+1$ value vectors at maturity.
        - [ ] In backward loop, manage list of vectors.
        - [ ] Implement state transition logic (fan-in) at observation times.
    - [ ] **Validation**: Run `example/phoenix_engine_compare_demo.py` (modified to enable memory).

- [ ] **1.2** Add safety checks
    - [ ] Raise `ValidationError` if `num_observations > 50` and `memory_coupon=True` (to prevent O(N^2) explosion).

## Phase 2: PDE Engine Enhancement
Apply the same logic to the PDE solver's grid structure.

- [ ] **2.1** Update `PhoenixPDESolver` state management
    - [ ] Change `grid_v0`, `grid_v1` from 2D arrays to lists of 2D arrays (or 3D arrays).
    - [ ] Update `_build_grids` to allocate sufficient space.

- [ ] **2.2** Implement Memory Transitions in PDE
    - [ ] Update `_set_terminal_condition_v0` to set values for all $k$ memory states.
    - [ ] Update `_apply_coupon_jump` to perform the "fan-in" reduction (mapping $N-i+1$ grids to $N-i$ grids).
    - [ ] Update `_apply_ko_jump` to respect the memory state of each grid layer.

## Phase 3: Validation

- [ ] **3.1** Create `test/test_phoenix_memory_pricing.py`
    - [ ] Test case: Standard Phoenix (Monthly) with Memory.
    - [ ] Compare MC vs Quad vs PDE.
    - [ ] Tolerance: < 1% relative error.

- [ ] **3.2** Verify Greeks
    - [ ] Ensure `calculate_greeks` still works (may need adaptation for 3D grid).
    - [ ] Verify Delta/Gamma smoothness.
