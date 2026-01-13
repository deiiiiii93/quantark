## Context
The discrete quadrature engine currently contains product-specific wiring for barrier and one-touch options. The team expects additional product types (snowball, phoenix) to reuse the same discrete quadrature core, which makes a product-agnostic adapter layer the safer long-term architecture.

## Goals / Non-Goals
- Goals:
  - Keep `DiscreteQuadEngine` product-agnostic.
  - Introduce a consistent adapter interface for building `QuadCoreInputs`.
  - Preserve pricing behavior and numerical characteristics.
  - Improve internal type safety for factor plumbing.
- Non-Goals:
  - Changing pricing formulas or core recursion.
  - Adding new product types or payoffs in this change.
  - Altering public product APIs.

## Decisions
- Decision: Add a `QuadInputAdapter` interface with per-product implementations.
  - Rationale: isolates product logic and scales to new product types without modifying the core.
- Decision: Add `_price_discrete_instrument(adapter)` to unify common pricing steps.
  - Rationale: reduces duplication and makes consistency easier to maintain.
- Decision: Replace untyped factor dictionaries with a dataclass or NamedTuple.
  - Rationale: improves clarity and reduces key mismatch risks.
- Decision: Vectorize tail-integral computation when it does not materially increase memory usage.
  - Rationale: improves performance without changing results.

## Risks / Trade-offs
- Risk: Subtle behavior changes during refactor.
  - Mitigation: parity tests against existing products and key boundary conditions.
- Risk: Memory blow-up from full broadcasting on large grids.
  - Mitigation: guard or chunked fallback when grid sizes exceed thresholds.

## Migration Plan
1. Introduce adapter layer and update existing barrier/one-touch engines to use it.
2. Move shared flow into `_price_discrete_instrument(adapter)`.
3. Convert factors to a typed container with a small compatibility shim.
4. Confirm pricing parity with existing tests and add coverage for zero-barrier handling.

## Open Questions
- Should the adapter registry be centralized in the quad engine module or live alongside product engines?
