## Context

The current `StressTestEngine` is monolithic and equity-centric:
- It instantiates `portfolio.Portfolio` and assumes access to spot/vol/greeks.
- Scenario application supports only parallel spot/vol/rate shifts with flat surfaces.
- Results/visualizations only emit equity Greeks and P&L.

After `add-fi-backtest`, we have `FIPortfolio`, DV01 calculators, and FI hedging data, but none of that can be stressed because the stress module lacks FI-aware contracts. Portfolio managers need to see how duration gaps behave under historical or hypothetical rate shocks, ideally sharing infrastructure with equity stress tests.

## Goals
- Introduce asset-agnostic stress protocols so equity and FI engines share the same entry points.
- Support FI-specific risk measures (DV01, convexity, key-rate durations, carry) throughout scenario evaluation, storage, export, and visualization.
- Provide reusable FI scenario templates (parallel shifts, steepeners, spread shocks) and validation logic for their parameters.
- Maintain backward compatibility for existing equity stress workflows.

## Non-Goals
- Supporting credit derivatives or FX stress workflows beyond rate/curve shocks.
- Implementing the dynamic (time-dependent) stress API.
- Replacing the existing equity report layouts.

## Decisions

### Decision 1: Protocol-Based Stress Architecture
**What**: Create `BaseStressEngine`, `ScenarioRunner`, and `StressMetricsAdapter` protocols in `stresstest/base.py`.  
**Why**: Mirrors the successful pattern from `add-fi-backtest`; allows new engines (equity, FI, future asset classes) to plug in without modifying callers.  
**Alternatives Considered**: Class inheritance hierarchy (couples implementations), dynamic type checks (fragile).

### Decision 2: Split Implementations by Asset Class
**What**: Move existing code into `stresstest/equity/` and build `stresstest/fi/` for new logic. Root package re-exports `StressTestEngine` pointing to the equity implementation for backward compatibility.  
**Why**: Keeps asset-specific dependencies isolated, simplifies testing, and mirrors the `portfolio/` + `backtest/` layout.  
**Risk**: Short-term import churn; mitigated with alias exports and deprecation warnings.

### Decision 3: Rate Shock Abstraction Layer
**What**: Extend `StressApplicator` with pluggable adapters (`RateShockAdapter`, `SpreadShockAdapter`) so FI engines can stress multi-node curves (parallel, key-rate, twist) and credit spreads.  
**Why**: FI stresses require more than a single flat-rate bump and must record metadata (bp shift per tenor).  
**Alternatives**: Teach `StressApplicator` about every asset class (would bloat core logic).

### Decision 4: Results Schema with Asset-Specific Extensions
**What**: Introduce a `StressResultEnvelope` that always contains P&L plus optional `extra_metrics` dict keyed by asset class (`equity`, `fi`). FI engines populate DV01/convexity vectors, which downstream reporters understand.  
**Why**: Maintains a unified API while surfacing FI-only insights without breaking equity consumers.

## Component Architecture

```
stresstest/
├── base.py                     # protocols: BaseStressEngine, ScenarioRunner, StressMetricsAdapter
├── equity/                     # existing implementation (moved)
│   ├── engine.py               # EquityStressEngine (alias StressTestEngine)
│   ├── config.py
│   ├── results.py
│   ├── report/
│   └── visualizer.py
└── fi/                         # NEW
    ├── engine.py               # FIStressEngine implements BaseStressEngine
    ├── config.py               # FIStressConfig with DV01 thresholds, curve adapters
    ├── metrics.py              # FIMetricsCalculator (DV01 tracking, convexity)
    ├── results.py              # FIStressResults extends base schema
    ├── exporter.py             # Writes DV01 series, curve shocks
    ├── scenario_library.py     # FI templates (parallel, steepener, spread)
    └── examples/fi_rate_shocks.py
```

Shared utilities (`StressApplicator`, `results/result_aggregator.py`, `report/`) gain adapter hooks so both engines can publish enriched data without diverging code paths.

## Data Flow
1. Caller instantiates `BaseStressEngine` via factory (equity default, FI when fed `FIPortfolio` or explicit config).
2. `ScenarioRunner` builds stressed pricing environments using adapters registered per asset class.
3. `StressMetricsAdapter` reads stressed positions and computes asset-specific metrics (DV01, convexity) alongside P&L.
4. `StressResultEnvelope` stores base + asset metrics; exporters/reporters read from the envelope to create tables, charts, and files.

## Migration Plan
1. **Protocol scaffolding**: add `stresstest/base.py`, refactor existing engine into `stresstest/equity` with re-exports; ensure tests still pass.
2. **Adapter-enabled stress application**: extend `StressApplicator` with hooks for rate/credit stresses (still default to current behavior for equities).
3. **FI implementation**: build config/engine/metrics/results/export/report stacks plus scenario library + example; integrate with `FIPortfolio`.
4. **Documentation & validation**: update README, add FI guide, write pytest coverage for DV01 aggregation, run `openspec validate add-fi-stresstest --strict`.

## Risks / Trade-offs
- **Complexity creep**: Protocols and adapters increase abstraction layers. Mitigated by mirroring existing `add-fi-backtest` patterns and keeping defaults simple.
- **Data duplication**: FI results may store large DV01 vectors. Mitigated via opt-in `save_dv01_series` flag in `FIStressConfig`.
- **Dependency ordering**: Requires `add-fi-backtest` landing first. We'll document the dependency and guard imports until available.

## Open Questions
1. Do we need key-rate DV01 (per tenor) in v1 or is aggregate DV01 sufficient? (Assumed: include optional key-rate vector since scenario shocks already parametrize tenor buckets.)
2. Should FI stresses support credit spread bumps separate from rate shifts? (Assumed: yes, represent as `spread` parameter handled by adapters.)


