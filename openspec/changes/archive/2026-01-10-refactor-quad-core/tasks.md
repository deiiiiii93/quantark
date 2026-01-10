## 1. Implementation
- [x] Create a product-agnostic quadrature core module implementing Eq. 3.5 with explicit `K^-`, `K^+`, `a^-`, `b^-`, `a^+`, `b^+`, `a_M`, `b_M` inputs.
- [x] Add adapter classes/functions for barrier and one-touch products that map product parameters and schedules to core inputs.
- [x] Refactor `BarrierQuadEngine` and `OneTouchQuadEngine` to use adapters + core (no direct factor manipulation).
- [x] Update quad docs to describe the core/adapter split and observation-date time stepping.

## 2. Validation
- [x] Add regression tests for barrier/one-touch quad engines to ensure parity and rebate timing cases.
- [x] Update demos to reflect adapter usage and verify numerical parity.

## 3. Verification
- [x] Run relevant demo scripts and targeted pytest cases.
