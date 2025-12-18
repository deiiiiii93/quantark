# convertible-bond-facade-engine Specification (Delta)

## MODIFIED Requirements

### Requirement: Extended results
`ConvertibleBondEngine.price_with_details()` SHALL include a `conversion_probability` that is mathematically consistent with the underlying method:
- Tree methods: computed directly from the lattice optimal policy
- PDE methods: computed from an auxiliary PDE for the conversion event indicator under the same optimal policy constraints

#### Scenario: PDE method probability is not a heuristic
- **WHEN** `ConvertibleBondEngine(method=ConvertibleBondMethod.TF)` or `ConvertibleBondMethod.JUMP_DIFFUSION` is used
- **THEN** `conversion_probability` is produced by the PDE engine and propagated through the facade result
