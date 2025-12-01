# Design: European Monte Carlo Engine

## Context
QuantArk has a comprehensive path generation infrastructure (`GBMPathGenerator`, Sobol QMC, RQMC driver) but no pricing engine that leverages these capabilities. The Monte Carlo engine needs to:
- Support multiple randomization strategies (pseudo, quasi, randomized-quasi)
- Integrate seamlessly with the existing `BaseEngine` interface
- Handle European option payoffs efficiently
- Provide error estimates for convergence diagnostics

## Goals / Non-Goals

**Goals:**
- Implement production-ready MC pricing for European vanilla options
- Support all three MC methods: normal, QMC (Sobol), RQMC
- Follow QuantArk's two-level enum pattern for method selection
- Integrate with existing path generators (`GBMPathGenerator`)
- Provide standard error estimation

**Non-Goals:**
- Path-dependent options (barriers, Asians) - future work
- Multi-asset options - future work
- American exercise - different engine needed
- GPU acceleration - future optimization

## Decisions

### Decision 1: Method Selection Pattern
Follow QuantArk's two-level enum pattern:
```python
# Preferred usage
engine = EuropeanMCEngine(
    method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI)
)

# Alternative
engine = EuropeanMCEngine(method=MonteCarloMethod.QUASI)

# Backward compatibility
engine = EuropeanMCEngine(method="quasi")
```

**Rationale:** Consistency with existing engines like `AmericanOptionAnalyticalEngine`

**Alternatives considered:**
- Single enum only - less flexible for future extensions
- Keyword arguments - less type-safe

### Decision 2: Path Generator Ownership
The engine creates and owns the `GBMPathGenerator` instance, configured from `MCParams`.

**Rationale:**
- Encapsulation: pricing parameters should drive path generation
- Simplicity: users don't need to understand path generator internals
- Consistency: matches how other engines handle their computational backends

**Alternatives considered:**
- Inject path generator as dependency - more flexible but more complex API
- Factory pattern - overkill for current needs

### Decision 3: Payoff Calculation
Calculate payoffs from terminal prices only (European options):
```python
terminal_prices = paths[:, -1]  # Shape: (num_paths,)
payoffs = np.maximum(terminal_prices - strike, 0)  # Call
price = discount * payoffs.mean()
```

**Rationale:**
- European options only need terminal values
- Efficient: no need for full path storage
- Clear separation: path generation vs. payoff calculation

### Decision 4: RQMC Integration
Use `run_rqmc` driver for RQMC method with adaptive stopping.

**Rationale:**
- Proven implementation with Welford's algorithm for variance estimation
- Adaptive batching improves efficiency
- Standard error estimation built-in

**Trade-offs:**
- Slightly more complex than simple MC loop
- Requires batch-based thinking
- Worth it for superior convergence properties

### Decision 5: Variance Reduction
Support variance reduction through `VarianceReductionConfig` in path generator:
- Antithetic variates (MC mode)
- Control variates (optional, future)

**Rationale:**
- Infrastructure already exists in `GBMPathGenerator`
- Significant efficiency gains (factor of 2+ for antithetic)
- Transparent to end users through `MCParams`

## Risks / Trade-offs

**Risk: Memory Usage for Large Path Counts**
- Mitigation: Document recommended path counts, support batching in RQMC mode

**Risk: Performance for High-Dimensional Time Grids**
- Mitigation: Use sensible defaults (time_steps=252), document trade-offs

**Risk: Numerical Stability for Extreme Parameters**
- Mitigation: Leverage validation in `BSMProcess` and `GBMPathGenerator`

## Migration Plan
This is a new engine, no migration needed. Integration points:
1. Import `EuropeanMCEngine` in `asset/equity/engine/mc/__init__.py`
2. Add to engine selection documentation
3. Add examples comparing analytical vs. MC pricing

## Open Questions
- Should we support custom payoff functions for flexibility? → No, keep focused on vanilla Europeans
- Should we expose path generator directly? → No, encapsulate through MCParams
- Should we support Greeks via pathwise derivatives? → Future work, use finite differences for now
