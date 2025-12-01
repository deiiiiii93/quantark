# Design: Unified PDE Engine

## Context
QuantArk has implemented multiple PDE solvers for different equity option types (European, American, Barrier, etc.). These solvers extend `BasePDESolver` which provides common finite difference infrastructure but are not directly compatible with the `BaseEngine` interface required by `GreeksCalculator`.

The `GreeksCalculator.calculate_numerical_greeks()` method expects engines that implement `BaseEngine.price(product, pricing_env)`, enabling automatic Greeks calculation via finite difference bumping. Currently, users must call PDE solvers directly and cannot leverage the general-purpose Greeks calculator.

**Stakeholders**: Quantitative analysts using PDE methods for Greeks calculation, developers maintaining pricing infrastructure.

**Constraints**:
- Must not break existing PDE solver implementations
- Must follow project's two-level enum pattern (per AGENTS.md)
- Must maintain numerical stability for all product types
- Must support all existing PDE solvers

## Goals / Non-Goals

### Goals
- Create a unified `PDEEngine` that implements `BaseEngine.price()` interface
- Automatically dispatch to appropriate PDE solver based on product type
- Enable Greeks calculation via `GreeksCalculator` for PDE-priced products
- Support all existing product types (European, American, Barrier, DoubleBarrier, OneTouch, DoubleOneTouch)
- Follow project's two-level enum pattern: `EngineType.PDE(PDEMethod.CRANK_NICOLSON)`

### Non-Goals
- Modifying existing PDE solver implementations
- Adding new PDE methods or solvers
- Optimizing PDE solver performance
- Supporting products beyond existing PDE solver coverage

## Decisions

### Decision 1: Dispatcher Pattern
**What**: `PDEEngine` acts as a facade/dispatcher that delegates to product-specific solvers.

**Why**:
- Preserves existing well-tested PDE solver implementations
- Provides a unified interface consistent with `BlackScholesEngine` and future MC engines
- Enables seamless integration with `GreeksCalculator`
- Follows Single Responsibility Principle (dispatch logic separate from numerical methods)

**Alternatives considered**:
1. **Modify all PDE solvers to extend BaseEngine**: Rejected due to code duplication and violates DRY principle
2. **Multiple inheritance**: Rejected due to Python MRO complexity and maintenance burden
3. **Factory pattern**: Considered but dispatcher is simpler and more explicit

### Decision 2: Product-to-Solver Mapping
**What**: Use dictionary mapping from product class types to solver classes.

**Implementation**:
```python
PRODUCT_SOLVER_MAP = {
    EuropeanVanillaOption: EuropeanPDESolver,
    AmericanOption: AmericanPDESolver,
    BarrierOption: BarrierPDESolver,
    DoubleBarrierOption: DoubleBarrierPDESolver,
    OneTouchOption: OneTouchPDESolver,
    DoubleOneTouchOption: DoubleOneTouchPDESolver,
}
```

**Why**:
- Simple, explicit, and maintainable
- Easy to extend for new product types
- Fast O(1) lookup
- Type-safe validation at runtime

**Alternatives considered**:
1. **String-based mapping**: Rejected due to lack of type safety
2. **if-elif chain**: Rejected due to poor scalability
3. **Dynamic discovery via introspection**: Rejected due to complexity and implicit dependencies

### Decision 3: Method Selection via Enum
**What**: Follow project's two-level enum pattern for method selection.

**Implementation**:
```python
class PDEMethod(Enum):
    CRANK_NICOLSON = "crank_nicolson"
    EXPLICIT_EULER = "explicit_euler"
    IMPLICIT_EULER = "implicit_euler"

# Usage: EngineType.PDE(PDEMethod.CRANK_NICOLSON)
```

**Why**:
- Consistent with `AmericanAnalyticalMethod` pattern in AGENTS.md
- Type-safe method selection
- IDE autocomplete support
- Backward compatible with string-based selection

**Note**: The `method` parameter maps to PDE solver schemes. Currently all PDE solvers use the same schemes, so the method is passed to `PDEParams.scheme`.

### Decision 4: Parameter Handling
**What**: `PDEEngine.__init__` accepts `PDEParams` (not generic `EngineParams`).

**Why**:
- PDE solvers require specific parameters (grid sizes, schemes, smoothing)
- Type safety and IDE support
- Explicit documentation of required configuration

### Decision 5: Error Handling
**What**: Raise `ValidationError` for unsupported product types with clear message indicating supported products.

**Why**:
- Fail fast with actionable error messages
- Consistent with project exception hierarchy
- Helps users understand PDE engine limitations

## Risks / Trade-offs

### Risk 1: Product Type Coupling
**Risk**: `PDEEngine` is tightly coupled to specific product classes.

**Mitigation**:
- Document supported products clearly in docstrings
- Provide helpful error messages listing supported types
- Design allows easy extension via `PRODUCT_SOLVER_MAP` updates

**Trade-off**: Accepted coupling for simplicity and type safety over complex dynamic dispatch.

### Risk 2: Parameter Propagation
**Risk**: Different PDE solvers might require different parameters in the future.

**Mitigation**:
- All solvers currently use `PDEParams`, standardized in project
- If divergence occurs, can extend `PDEParams` with optional solver-specific fields
- Alternative: Add solver-specific kwargs passthrough if needed

**Trade-off**: Standardization over flexibility; revisit if future solvers need custom params.

### Risk 3: Maintenance Burden
**Risk**: Adding new PDE solver requires updating `PDEEngine` mapping.

**Mitigation**:
- Clear documentation in code comments
- Error messages guide developers to update mapping
- Simple one-line addition to dictionary

**Trade-off**: Explicit mapping (requires update) over implicit discovery (hidden dependencies).

## Migration Plan

### Phase 1: Implementation
1. Add `PDEMethod` enum to `util/enum/engine_enums.py`
2. Implement `PDEEngine` in `asset/equity/engine/pde_engine.py`
3. Export from `asset/equity/engine/__init__.py`

### Phase 2: Testing
1. Write comprehensive unit tests
2. Verify numerical agreement with direct solver usage
3. Test Greeks calculator integration

### Phase 3: Rollout
- No breaking changes; purely additive feature
- Existing code using PDE solvers directly continues to work
- New code can use `PDEEngine` for unified interface

### Rollback
- Simply remove `PDEEngine` exports; no dependent code in current codebase
- No data migration required

## Open Questions
None. All design decisions finalized based on existing architecture patterns.
