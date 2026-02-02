# Proposal: Enhance Phoenix Engines with Memory Coupon Support

## Summary

Upgrade the `PhoenixPDESolver` and `PhoenixQuadEngine` to correctly price "Memory Coupon" features by implementing a multi-state backward induction algorithm. This resolves the current limitation where these engines must ignore memory to avoid double-counting.

## Why

The current deterministic engines (PDE/QUAD) price Phoenix options as if `memory_coupon=False`. While this provides a valid lower bound, it significantly undervalues standard products where the memory feature is active. To provide accurate pricing and Greeks for market-standard Phoenix options without relying on Monte Carlo (which is noisy for Greeks), we need deterministic engines that handle path-dependent memory accumulation.

## Problem Statement

Memory coupons introduce path dependency: the payoff at time $t_i$ depends on the number of previously missed coupons ($k$). 
- **Current State:** Engines assume a single state, effectively $k=0$ or "always pay everything", leading to incorrect pricing.
- **Required State:** At each observation $t_i$, the engine must know the value function for every possible accumulated memory state $k 
in \{0, \dots, i\}$.

## Proposed Solution: Vectorized State Backward Induction

We will implement a "Vector of Surfaces" approach. Since the number of observations $N$ is typically small (e.g., 12 for monthly), we can track the value function for all possible memory states.

### Algorithm

Let $V_i[k](S)$ be the value at observation $t_i$ given $k$ missed coupons.

1.  **Initialization (Maturity $t_N$):**
    *   Construct $N+1$ terminal value surfaces for $k \in \{0, \dots, N\}$.
    *   $V_N[k](S) = \text{MaturityPayoff}(S, \text{accumulated}=k \times c)$.

2.  **Backward Induction ($i = N-1 \to 0$):**
    *   **Input:** $N-i+1$ surfaces from step $i+1$ (diffused to $t_i$). Let's call them $U_i[k]$ for $k \in \{0, \dots, i+1\}$.
    *   **Transition:** For each incoming memory state $k \in \{0, \dots, i\}$:
        *   **Hit ($S$ in pay region):**
            *   Payoff: $C_i + (k \times C)$.
            *   Continuation: System resets to 0 missed coupons. Value is $U_i[0]$.
            *   Total: $C_i + kC + U_i[0]$.
        *   **Miss ($S$ in miss region):**
            *   Payoff: 0.
            *   Continuation: System moves to $k+1$ missed coupons. Value is $U_i[k+1]$.
            *   Total: $U_i[k+1]$.
    *   **Output:** $N-i$ surfaces ($V_i[k]$) ready for diffusion to $t_{i-1}$.

3.  **Final Step ($t_0$):**
    *   We end with a single surface $V_0[0]$, which is the option price.

### Complexity
*   The number of surfaces tracks linearly with remaining steps: $1, 2, \dots, N+1$.
*   Total diffusions $\approx N^2/2$.
*   For Monthly ($N=12$), ~78 diffusions (vs 12 currently). feasible.
*   For Daily ($N=252$), heuristic/limitations will be applied (e.g., cap memory depth or fallback to MC).

## What Changes

### Modified Files
- `asset/equity/engine/pde/phoenix_pde_solver.py`:
    - Replace single `grid_v0`/`grid_v1` with lists of grids.
    - Implement the "fan-out / fan-in" logic for memory states.
- `asset/equity/engine/quad/phoenix_quad_engine.py`:
    - Replace single `v_in`/`v_out` arrays with lists of arrays.
    - Implement the recursive state transition.

### New Tests
- `test/test_phoenix_memory_pricing.py`: Validate that PDE/QUAD match MC for `memory_coupon=True`.

## Scope
- Support `memory_coupon=True` for Standard and Reverse Phoenix.
- Optimize for discrete observations (Monthly/Quarterly).
- Add safeguards for high-frequency observations (e.g., raise error if $N > 50$).

## Success Criteria
1.  **Accuracy:** PDE and QUAD prices for `memory_coupon=True` match MC within 1%.
2.  **Consistency:** `memory_coupon=False` path remains efficient (O(N)).
3.  **Stability:** Greeks remain smooth.
