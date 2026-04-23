## Context

QuantArk already has single-barrier option support via `BarrierOption` and `BarrierOptionAnalyticalEngine`. Double-barrier options introduce a second boundary and require the Ikeda & Kuintomo infinite-series solution. The existing barrier product can be extended (or a new `DoubleBarrierOption` created) to hold two barrier levels. The engine layer follows the established pattern: product holds specs, engine holds pricing logic, `BaseEngine` defines the interface.

## Goals / Non-Goals

**Goals:**
- Implement a production-grade analytical engine for double-barrier options
- Support continuous, daily (discrete), and expiry-only observation modes
- Match Table 4-15 benchmark values within 1e-4 for continuous cases
- Follow QuantArk conventions (type hints, input validation, safe math, `BaseEngine` interface)

**Non-Goals:**
- Greeks calculations (can be added later via finite differences)
- American-style double-barrier options
- PDE or MC double-barrier engines (out of scope for this change)
- Support for strike outside the barrier range (document limitation)

## Decisions

1. **Product Class**: Create a new `DoubleBarrierOption` dataclass in `asset/equity/product/double_barrier_option.py` rather than overloading the existing `BarrierOption`. This keeps the type system clean and allows two barrier fields (`lower_barrier`, `upper_barrier`) without confusing optional single-barrier semantics.
   - *Alternative considered*: Add `second_barrier` to `BarrierOption`. Rejected because it complicates validation and single-barrier engines would need to handle an unused field.

2. **Engine Naming**: `DoubleBarrierOptionAnalyticalEngine` in `asset/equity/engine/analytical/double_barrier_option_engine.py`.
   - Follows the `*AnalyticalEngine` naming convention used for `BarrierOptionAnalyticalEngine`.

3. **Observation Mode Representation**: Use an enum `DoubleBarrierObservationType` with values `CONTINUOUS`, `DAILY`, `EXPIRY`.
   - Daily observation applies a barrier shift (`exp(±0.5826 * σ * sqrt(1/252))`) before calling the continuous formula.
   - Expiry observation computes the closed-form truncated-domain vanilla payoff.

4. **Series Truncation**: Use a configurable `max_series_terms` parameter defaulting to 10. The Ikeda & Kuintomo series converges extremely fast; 3 terms is usually enough, but 10 provides a wide safety margin with negligible performance cost.

5. **Safe Math**: All `log`, `exp`, `sqrt`, and division operations MUST use `util.numerical.safe_math` utilities to prevent crashes at boundary conditions.

## Risks / Trade-offs

- **Risk**: Daily barrier shift is an approximation (Broadie-Glasserman-Kou style). It is not exact for double barriers. → Document this in docstrings and tests; users needing exact discrete pricing should use MC or tree methods.
- **Risk**: Strike outside `[L, U]` is not supported by the Ikeda-Kuintomo formula. → Raise `ValidationError` with a clear message when `strike <= lower_barrier` or `strike >= upper_barrier`.
- **Risk**: Deep ITM/OTM or very wide barriers can cause numerical underflow/overflow in the infinite-series weights. → Use `safe_exp` and cap terms when weights become machine-zero.

## Migration Plan

No migration needed. This is a pure addition with no breaking changes.

## Open Questions

None.
