# QuantArk execution framework

`quantark.execution` is the composable execution kernel for QuantArk's MC and
PDE engines: one immutable plan per request, explicit resource budgets,
deterministic reproduction manifests, and session-level batch / adaptive /
prepared-artifact / scenario execution — while every direct legacy call keeps
its exact historical behavior (framework contract v1, no deprecations).

```python
from quantark.execution import PricingSession

with PricingSession() as session:
    pv = session.price(engine, product, pricing_env)   # == engine.price(...)
```

Spec of record:
`docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`.

## Documents

- [Capability matrix](capability-matrix.md) — generated per-engine
  batch/adaptive/prepared adoption states with rationales
  (`python -m quantark.execution.capability_matrix`; CI-enforced freshness).
- [Policy guide](policy-guide.md) — precedence, environment variables,
  budgets, backend × capability matrix.
- [Internals and legacy duplicates](internals-and-legacy.md) — what is
  unified, what deliberately remains duplicated and under which removal
  preconditions (spec §17.3).
- [Reproducibility schemas](schemas/) — JSON Schemas for the WorkerSpec,
  scenario-cell, manifest, and normalized-economics payloads, validated
  against live payloads in CI.
- Migration examples — `example/execution_session_demo.py` and
  `example/execution_scenarios_demo.py` (runnable; exercised by
  `test/execution/test_examples.py`).

## Performance snapshots

Dev-machine (Apple Silicon, macOS) measurements from the Phase 6 exit rerun,
2026-07-18, via `test/execution/benchmark_phase{2,4,5}.py`. These are
**documentation, not gate passes** — the spec §20 release gates run on the
controlled host with production-sized workloads (see the pre-tag checklist
below).

- **Fixed-batch MC** (DCN, 2^17 paths, 16 batches, cold draws): session
  threads 2.29x @4 workers (PASS vs the 1.5x dev floor); 2.13x @8 workers —
  host-limited (the direct legacy `num_workers=8` path is no faster on this
  machine; ≥2.5x@8 needs the controlled host). Framework adds no threading
  overhead vs engine-internal threads (worst ratio 1.04). CRN draw reuse
  2.04x (PASS vs 2x). Serial overhead −0.2%.
- **PDE prepared artifacts** (CRN x10 LV-snowball, 58 ms solves): session
  1.29x vs uncached direct — march-dominated small fixture; the ≥2x gate is
  sized for production grids on the controlled host. Warm-session single
  European dispatch overhead: −9.7% (session faster) at a 0.8 ms solve.
- **Scenario processes** (9-cell surface-shock menu, 4096 paths, 4 spawn
  workers): median 1.63x vs serial with **194/194 normalized fields
  matching in every rep**; tiny cells put spawn+import fixed costs and the
  heaviest cell on the critical path — the ≥2.5x gate runs on >10s serial
  grids on the controlled host.

## Before tagging v0.3.0

This phase produces **release-preparation evidence only** — pushing a tag IS
publishing (`.github/workflows/release.yml` is tag-triggered). Outstanding
hard prerequisites before any `v0.3.0` tag:

1. **Controlled-host performance gates** (spec §20): ≥2x fixed-batch MC at 4
   workers on >10s serial workloads, ≥2x PDE CRN-sweep production gate, and
   ≥2.5x scenario/process gate — measured on the controlled multi-core host,
   not this dev machine (dev snapshots below are documentation, not gate
   passes).
2. **Full-suite green**: resolve or explicitly quarantine (with written
   rationale) the pre-existing
   `test_snowball_quad_flat_identity_golden` failure — it predates the
   execution-framework program and reproduces on unmodified main, but a tag
   must not ship a red suite silently.
3. **Wheel-artifact compatibility**: the otc-price-adapter suite green
   against the installed candidate wheel at the tagged commit.
