# Change: Refactor quad engine with adapters

## Why
Barrier and one-touch wiring currently lives inside the discrete quad engine, which makes it harder to add new products like snowball and phoenix without touching core pricing flow.

## What Changes
- Introduce a quad input adapter layer to build `QuadCoreInputs` per product type.
- Add a unified discrete pricing flow that consumes adapters and remains product-agnostic.
- Migrate factor passing to a typed container for clarity and safety.
- Optionally vectorize tail-integral calculations where safe for performance.

## Impact
- Affected specs: `specs/equity-quad-engine/spec.md`
- Affected code: `asset/equity/engine/quad/*`, product-specific quad engines, and related tests/examples
