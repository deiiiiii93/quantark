## 1. Preset registry and definitions
- [x] 1.1 Define preset profiles for QUAD and PDE (`fast`, `balanced`, `accurate`, `barrier_sensitive`, `reverse_sensitive`).
- [x] 1.2 Implement a registry that maps engine type to preset dictionaries.
- [x] 1.3 Validate that presets only include known parameter keys.

## 2. Factory helpers
- [x] 2.1 Implement `make_quad_params(profile, product=None, reverse=False, **overrides)`.
- [x] 2.2 Implement `make_pde_params(profile, product=None, reverse=False, **overrides)`.
- [x] 2.3 Add convenience wrapper for batch use (e.g., `make_engine_params(engine, ...)`).

## 3. Config loaders
- [x] 3.1 Implement `QuadParams.from_config(...)` and `PDEParams.from_config(...)` supporting dict/YAML/JSON.
- [x] 3.2 Validate unknown keys and unknown profiles with clear errors.
- [x] 3.3 Document config schema with examples for batch pipelines.

## 4. Documentation and examples
- [x] 4.1 Add a "Choosing Params" guide with a decision table.
- [x] 4.2 Add a minimal batch example (YAML/JSON + factory usage).

## 5. Tests
- [x] 5.1 Unit tests for preset mapping and overrides.
- [x] 5.2 Unit tests for config loading and validation.
