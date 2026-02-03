# PDE Solver Refactoring Plan

## 1. Investigation Findings

### 1.1 Code Duplication
The Crank-Nicolson time-stepping logic is triplicated across:
- `BasePDESolver._time_stepping` (Standard 1-grid)
- `SnowballPDESolver._time_stepping_two_surface` (2-grid V0/V1)
- `PhoenixPDESolver._time_stepping_vector_surface` (N-grid for memory coupons)

This makes maintenance difficult (e.g., changing Rannacher smoothing requires updates in 3 places) and increases the risk of bugs.

### 1.2 Performance Bottleneck
- **Tridiagonal Solve**: `BasePDESolver` uses `scipy.sparse.linalg.splu` (SuperLU), which is a general sparse solver. For 1D tridiagonal systems, `scipy.linalg.solve_banded` is significantly faster (O(N)) and more memory-efficient.
- **Python Loops in Phoenix**: `PhoenixPDESolver` iterates over memory states (`k=0..max_k`) in a Python loop inside the time-stepping loop.
- **Inefficiency**: Overhead of Python loops and multiple `solve` calls prevents lower-level optimization.
- **Opportunity**: Both `splu` and `solve_banded` support matrix RHS (`X = solve(A, B)` where B is `NxM`). This allows solving all memory states simultaneously in a single vectorized call.

### 1.3 Complexity
- **Event Logic**: KO, KI, and Coupon logic is hardcoded inside the time loops, making the solvers rigid and hard to test in isolation.
- **Event Stats**: `calculate_event_stats` reimplements the entire solver logic, leading to massive code duplication.
- **State Management**: The distinction between "Not Knocked In" (V0) and "Knocked In" (V1) is manually managed in the loops.

## 2. Proposed Architecture

### 2.1 Abstractions

#### `PDESystemState` (Abstract Base)
Encapsulates the grid data (1, 2, or N surfaces).
- **Methods**: `apply_boundary()`, `get_values()`, `set_values()`, `solve_step()`.
- **Implementations**:
    - `ScalarState`: Wraps a single `(N,)` array (European/American).
    - `RegimeState`: Wraps `(N, 2)` array (Snowball V0/V1).
    - `VectorState`: Wraps `(N, M)` array (Phoenix Memory). **Vectorized solve**.

#### `PDEEvent` (Interface)
Represents a discrete event that modifies the state.
- **Method**: `apply(state, time, grid_context)`
- **Implementations**:
    - `KnockOutEvent`: Sets values to rebate in breached region.
    - `KnockInEvent`: Copies values from V1 to V0.
    - `CouponEvent`: Shifts values (k -> k+1) or adds cashflows.

#### `UnifiedPDESolver`
A single, generic time-stepping engine using `solve_banded`.
```python
def solve(self, state: PDESystemState, events: List[PDEEvent]):
    # 1. Prepare banded matrices
    banded_lhs, banded_rhs = self.build_banded_matrices()
    
    for t in time_steps:
        # 2. Evolve diffusion (Vectorized solve_banded for all states)
        state.solve_step(banded_lhs, banded_rhs)
        
        # 3. Apply Events
        for event in active_events(t):
            event.apply(state, t)
```

### 2.2 Benefits
1.  **Conciseness**: Removes ~400 lines of duplicated code.
2.  **Performance**: 
    - **Base speedup**: 2-3x by switching from `splu` to `solve_banded`.
    - **Phoenix speedup**: 5-10x for deep memory (vectorized solve).
3.  **Accuracy**: Unified logic ensures improvements (like Rannacher smoothing) apply everywhere.
4.  **Extensibility**: New products (e.g., Target Redemption) just need new `Events` or `States`.

## 3. Implementation Strategy

1.  **Promote Banded Solver**: Update `BasePDESolver` to support `solve_banded` and multi-column RHS.
2.  **Create `asset/equity/engine/pde/core/`**: New module for `PDESystemState` and `PDEEvent`.
3.  **Implement `PDESystemState`**: Focus on vectorization using `solve_banded` with multi-column RHS.
4.  **Refactor `SnowballPDESolver`**: Use `RegimeState` (2-grid) and `KnockOutEvent`.
5.  **Refactor `PhoenixPDESolver`**: Use `VectorState` (N-grid) and `CouponEvent`.
6.  **Clean up**: Remove the old duplicated methods and consolidate logic.