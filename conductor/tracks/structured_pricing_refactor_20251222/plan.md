# Plan: Consolidate and Refactor Structured Product Pricing

## Phase 1: Core Interface and Shared Logic
- [ ] Task: Define `BaseStructuredProduct` class with shared properties (KO, KI, Airbag).
- [ ] Task: Implement shared payoff calculation logic for V0 (never KO/KI) and V1 (KI occurred).
- [ ] Task: Write unit tests for the base class and shared logic.
- [ ] Task: Conductor - User Manual Verification 'Core Interface and Shared Logic' (Protocol in workflow.md)

## Phase 2: Product Refactoring
- [ ] Task: Refactor `SnowballOption` to inherit from `BaseStructuredProduct`.
- [ ] Task: Update factory functions in `snowball_helpers.py` to support the new structure.
- [ ] Task: Write tests for refactored `SnowballOption` and helpers.
- [ ] Task: Conductor - User Manual Verification 'Product Refactoring' (Protocol in workflow.md)

## Phase 3: Engine Refactoring
- [ ] Task: Update `SnowballMCEngine` to use the standardized `BaseStructuredProduct` interface.
- [ ] Task: Update PDE and Analytical engines for compatibility.
- [ ] Task: Implement comprehensive integration tests comparing pricing results across engines.
- [ ] Task: Conductor - User Manual Verification 'Engine Refactoring' (Protocol in workflow.md)
