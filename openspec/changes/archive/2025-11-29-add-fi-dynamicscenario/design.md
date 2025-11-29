## Context

The current `DynamicScenarioEngine` is equity-centric:
- It works with `Portfolio` (equity) and assumes access to spot/vol parameters and equity Greeks.
- Path patterns in `PathLibrary` focus on spot and volatility changes.
- Results and visualizations only emit delta, gamma, vega, theta.

After `add-fi-backtest`, we have `FIPortfolio`, DV01 calculators, and FI hedging strategies, but none of that can be simulated through dynamic day-by-day scenarios because the module lacks FI-aware contracts. Portfolio managers need to see how duration gaps and DV01 exposure evolve through multi-day rate scenarios, ideally sharing infrastructure with equity dynamic scenarios.

## Goals

- Introduce asset-agnostic dynamic scenario protocols so equity and FI engines share the same entry points.
- Support FI-specific risk measures (DV01, convexity, key-rate durations, modified duration) throughout day-by-day simulation.
- Provide reusable FI path patterns (rate hike cycles, parallel shifts, curve twists, steepeners/flatteners) analogous to equity patterns.
- Enable FI hedging simulation using `DV01NeutralStrategy` and bond futures.
- Maintain backward compatibility for existing equity dynamic scenarios.

## Non-Goals

- Supporting credit derivatives or FX dynamic scenarios beyond rate/curve changes.
- Real-time trading or external market data integration.
- Replacing the existing equity visualization layouts.

## Decisions

### Decision 1: Protocol-Based Dynamic Scenario Architecture
**What**: Create `BaseDynamicScenarioEngine`, `BaseScenarioResults`, and `RiskMetricsAdapter` protocols in `dynamicscenario/base.py`.  
**Why**: Mirrors the successful pattern from `add-fi-backtest` and `add-fi-stresstest`; allows new engines (equity, FI, future asset classes) to plug in without modifying callers.  
**Alternatives Considered**: Class inheritance (couples implementations), dynamic type checks (fragile).

### Decision 2: Split Implementations by Asset Class
**What**: Move existing code into `dynamicscenario/equity/` and build `dynamicscenario/fi/` for new logic. Root package re-exports `DynamicScenarioEngine` pointing to the equity implementation for backward compatibility.  
**Why**: Keeps asset-specific dependencies isolated, simplifies testing, and mirrors the `backtest/` and `stresstest/` layout.  
**Risk**: Short-term import churn; mitigated with alias exports.

### Decision 3: Extended Path Components for FI
**What**: Extend `ParameterChange` and `PathBuilder` to support rate curve shocks (parallel shifts, key-rate bumps at specific tenors, curve twists).  
**Why**: FI scenarios require more than flat-rate bumps; need to express multi-tenor curve movements.  
**Implementation**: Add `rate_shock` parameter type with tenor specification.

### Decision 4: FI Path Library
**What**: Create `FIPathLibrary` with FI-specific patterns:
- `parallel_shift(bps)` - uniform rate change across all tenors
- `curve_steepener(short_bps, long_bps)` - short rates up/down, long rates opposite
- `curve_flattener(short_bps, long_bps)` - inverse of steepener
- `rate_hike_cycle(days, total_bps)` - gradual rate increases
- `historical_fed_tightening()` - modeled on historical tightening cycles

**Why**: FI-specific patterns are fundamentally different from equity patterns; rate curve dynamics drive FI P&L.

### Decision 5: FI Results with DV01/Duration Tracking
**What**: Create `FIDayResult` and `FIDynamicScenarioResults` that track:
- DV01 evolution (pre-hedge and post-hedge)
- Convexity evolution
- Modified duration evolution
- Key-rate DV01 vectors (optional)
- Rate curve state at each step

**Why**: These are the primary risk measures for FI portfolios; portfolio managers need to see how they evolve through scenarios.

### Decision 6: Hedging Integration
**What**: Support `DV01NeutralStrategy` from backtest module for FI hedging during dynamic scenarios.  
**Why**: Reuses existing FI hedging infrastructure; shows hedge effectiveness through rate shocks.

## Component Architecture

```
dynamicscenario/
├── __init__.py              # MODIFIED: Re-export from submodules with aliases
├── base.py                  # NEW: protocols (BaseDynamicScenarioEngine, etc.)
├── config.py                # Keep as-is (general config)
├── equity/                  # NEW: Equity implementations (refactored from root)
│   ├── __init__.py
│   └── engine.py            # MOVED: EquityDynamicScenarioEngine (was engine.py)
├── fi/                      # NEW: Fixed Income implementations
│   ├── __init__.py
│   ├── engine.py            # FIDynamicScenarioEngine
│   ├── config.py            # FIDynamicScenarioConfig
│   └── results.py           # FIDayResult, FIDynamicScenarioResults
├── path/
│   ├── day_path.py          # MODIFIED: Support rate curve changes
│   ├── path_builder.py      # MODIFIED: Add rate shock methods
│   ├── path_library.py      # Keep equity patterns
│   └── fi_path_library.py   # NEW: FI-specific path patterns
├── results/
│   ├── dynamic_results.py   # MODIFIED: Add base class/protocol
│   └── result_exporter.py   # MODIFIED: Support FI exports
└── report/
    ├── dynamic_report.py    # MODIFIED: Support FI reports
    └── visualizer.py        # MODIFIED: Add FI visualizations
```

## Risk Measures Mapping

| Equity | Fixed Income | Description |
|--------|--------------|-------------|
| Delta | DV01 | First-order price sensitivity |
| Gamma | Convexity | Second-order price sensitivity |
| Vega | Rate Vega | Volatility sensitivity |
| Theta | Carry/Roll | Time decay |

## Data Flow

1. Caller instantiates `BaseDynamicScenarioEngine` via factory (equity default, FI when fed `FIPortfolio` or explicit config).
2. `DayPath` contains rate curve changes expressed via `ParameterChange` with `rate` or `rate_curve` parameters.
3. Engine steps through days, applying rate curve updates to `FIPortfolio` pricing environments.
4. `RiskMetricsAdapter` computes FI risk measures (DV01, convexity, duration) at each step.
5. Optional hedging via `DV01NeutralStrategy` executes futures trades.
6. `FIDayResult` captures day state; `FIDynamicScenarioResults` aggregates full path.
7. Exporters and reporters render FI-specific outputs.

## Migration Plan

1. **Protocol scaffolding**: Add `dynamicscenario/base.py`, refactor existing engine into `dynamicscenario/equity/` with re-exports; ensure tests still pass.
2. **Extended path components**: Add rate curve support to `ParameterChange`, `PathBuilder`; create `FIPathLibrary`.
3. **FI implementation**: Build config/engine/results for FI; integrate with `FIPortfolio` and hedging strategies.
4. **Visualization & reporting**: Extend visualizer and report generator with FI-specific plots and sections.
5. **Documentation & validation**: Create example, update README, run `openspec validate add-fi-dynamicscenario --strict`.

## Risks / Trade-offs

- **Complexity creep**: Protocols and adapters increase abstraction layers. Mitigated by mirroring existing patterns from `add-fi-backtest` and `add-fi-stresstest`.
- **Path component changes**: Extending `ParameterChange` may affect existing equity paths. Mitigated via backward-compatible additions (new parameter types, not modifications).
- **Dependency ordering**: Requires `add-fi-backtest` landing first. We'll document the dependency and guard imports until available.

## Open Questions

1. Should key-rate DV01 tracking (per tenor) be included in v1 or is aggregate DV01 sufficient?
   - **Proposed answer**: Include optional key-rate DV01 since rate paths already parametrize tenor-specific changes.

2. Should we support multi-curve scenarios (e.g., different curves for discount vs. projection)?
   - **Proposed answer**: Not in v1; use single curve for simplicity, design for future extensibility.

