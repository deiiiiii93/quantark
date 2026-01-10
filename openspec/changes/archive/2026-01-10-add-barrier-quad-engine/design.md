## Context

The quadrature method for barrier options uses FFT-based convolution to solve the backward pricing PDE. This approach is based on the paper "A Simple and Efficient Numerical Method for Pricing Discretely Monitored Early-Exercise Options" by Huang and Luo.

The reference implementation exists in `docs/quad/ref_scripts/option_quad.py` with the `price_single_barrier()` function.

## Goals / Non-Goals

**Goals:**
- Provide deterministic barrier option pricing with spectral accuracy
- Support all 4 single barrier types (UP_IN, UP_OUT, DOWN_IN, DOWN_OUT)
- Handle discrete observation schedules efficiently
- Achieve O(N log N) complexity via FFT

**Non-Goals:**
- Double barrier support (future enhancement)
- Continuous barrier monitoring (use analytical engine)
- American-style early exercise

## Decisions

### Decision 1: FFT Convolution Approach

**What**: Use FFT-based convolution for backward recursion instead of direct integration.

**Why**: FFT reduces complexity from O(N²) to O(N log N) per time step, making the method practical for fine time grids.

**Reference**: `docs/quad/ref_scripts/option_quad.py:184-226`

### Decision 2: Factor-Based Payoff Decomposition

**What**: Decompose barrier option payoffs into 6 binary option components:
- `asset1`, `asset2`, `asset3`: Asset-or-nothing components
- `cash1`, `cash2`, `cash3`: Cash-or-nothing components

**Why**: This allows flexible payoff specification and reuse of the core convolution machinery for different barrier types.

**Reference**: `docs/quad/ref_scripts/option_quad.py:92-161`

### Decision 3: Simpson's Rule Integration

**What**: Use Simpson's rule with boundary corrections for numerical integration.

**Why**: Simpson's rule provides O(1/N⁴) convergence, significantly better than trapezoidal rule's O(1/N²).

**Reference**: `docs/quad/ref_scripts/option_quad.py:167-176`

### Decision 4: Log-Price Grid

**What**: Work in log-price space `x = ln(S/S₀)` rather than price space.

**Why**:
- Log-normal distribution becomes Gaussian in log-space
- Uniform grid provides better coverage of probability mass
- Barrier levels map to fixed grid boundaries

### Decision 5: Knock-In via Identity

**What**: Price knock-in options using the identity: KI = Vanilla - KO

**Why**: Avoids implementing separate knock-in logic; reuses knock-out pricing.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Numerical instability near barriers | Use adaptive grid with finer spacing near barrier levels |
| Memory usage for large grids | Reuse arrays across time steps; limit default grid size |
| Accuracy for continuous monitoring | Document that this engine is for discrete monitoring only |

## Migration Plan

No migration needed - this is a new capability that doesn't affect existing code.

## Open Questions

1. Should we support double barriers in this initial implementation?
   - **Decision**: No, defer to future enhancement. Single barriers cover most use cases.

2. Should we support continuous monitoring approximation?
   - **Decision**: No, use analytical engine for continuous monitoring. Quadrature is best for discrete.
