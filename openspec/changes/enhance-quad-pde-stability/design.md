## Context
Phoenix and Snowball quad pricing exhibit non-monotonic convergence around discontinuous event operators (KO/KI/coupon). PDE solvers require exact event alignment to avoid timing errors at discrete observation dates.

## Goals / Non-Goals
- Goals:
  - Add explicit quadrature stability controls (padding, filtering, barrier alignment, smoothing).
  - Ensure discrete-event PDE grids include observation times and raise when misaligned.
  - Preserve existing engine APIs and default behavior where possible.
- Non-Goals:
  - Rewriting core quad recursion or PDE schemes.
  - Changing product payoff definitions.

## Decisions
- Decision: Add `QuadParams` fields for FFT padding/filter, domain width (`num_std_devs`), and event-step smoothing.
  - Rationale: Centralized knobs enable consistent stabilization across quad engines.
- Decision: Implement event-step smoothing as a local weighting around barrier boundaries.
  - Rationale: Damp Gibbs ringing without altering diffusion or payoff logic.
- Decision: Enforce exact event-time alignment in PDE solvers.
  - Rationale: Discrete observation products require exact timing for KO/KI/coupon jumps.

## Risks / Trade-offs
- Smoothing can slightly bias prices near barriers; mitigate by making it configurable and defaulting to a conservative width.
- Stricter PDE alignment may raise errors for custom grids; mitigate with clear error messages and guidance to use event-aligned grids.

## Migration Plan
1. Add new quad parameters with safe defaults.
2. Update quad math utilities and engines to honor new parameters.
3. Enforce event alignment in PDE solvers and update docs/examples.
4. Validate with existing tests and Phoenix comparison demo.

## Open Questions
- Should reverse Phoenix default to zero smoothing while standard uses one cell?
- Should barrier alignment priority be configurable per product?
