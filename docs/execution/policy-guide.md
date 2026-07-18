# QuantArk execution framework — policy guide

How execution policy is declared, resolved, and enforced. The framework never
changes direct legacy calls; everything here applies to `PricingSession` /
`PricingRunContext` execution only. Spec of record:
`docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`
(§12, §17.2).

## Precedence (spec §17.2)

New-framework resolution is **field-by-field**, highest precedence first:

1. explicit `PricingRunContext` / `PricingSession` setting;
2. existing or later-approved constructor-level execution setting;
3. legacy engine-specific environment variable or flag;
4. generic QuantArk execution configuration/environment variable;
5. historical default.

Unset fields do not shadow lower levels. **Resolution occurs once in the
parent**; children (spawn/Dask workers) receive resolved values inside the
`WorkerSpec` and never re-resolve the host environment. Direct legacy paths
retain their historical resolution time (e.g. `DCNMCEngine(num_workers=None)`
reads `QUANTARK_DCN_MC_WORKERS` at engine construction; the QMC cache reads
`QUANTARK_QMC_CACHE_MB` at `quantark.montecarlo.qmc_sobol` import).

## Environment variables

Generic aliases (level 4), read once when the session resolves its context:

| Variable | Field | Notes |
|---|---|---|
| `QUANTARK_EXEC_BATCH_BACKEND` | `policy.batch.backend` | `serial` / `threads` |
| `QUANTARK_EXEC_BATCH_WORKERS` | `policy.batch.workers` | |
| `QUANTARK_EXEC_SCENARIO_BACKEND` | `policy.scenario.backend` | `serial` / `threads` / `processes` / `dask` |
| `QUANTARK_EXEC_SCENARIO_WORKERS` | `policy.scenario.workers` | |
| `QUANTARK_EXEC_MEMORY_MB` | `budget.total_memory_bytes` | |
| `QUANTARK_EXEC_CACHE_MB` | `budget.artifact_cache_bytes` | |
| `QUANTARK_EXEC_DRAW_CACHE_MB` | `budget.draw_cache_bytes` | |
| `QUANTARK_EXEC_MAX_THREADS` | `budget.max_threads` | |
| `QUANTARK_EXEC_MAX_PROCESSES` | `budget.max_processes` | |
| `QUANTARK_EXEC_MAX_IN_FLIGHT` | `budget.max_in_flight` | hard admission bound |

Invalid text falls back to the default and records `env_invalid_default` in
the resolution sources (visible in diagnostics).

Legacy engine-specific variables (level 3) that WIN over the generic aliases
for their engines when no explicit setting exists:

- `QUANTARK_DCN_MC_WORKERS` — DCN engine internal worker count (resolved at
  engine construction; invalid text → 1; nonpositive clamps to 1).
- `QUANTARK_QMC_CACHE_MB` — requested legacy QMC draw-cache ceiling, subject
  to an explicit parent session budget (import-time resolution; invalid text
  → 2048 MiB; negative clamps to 0).

## Policy and budget fields

`ExecutionPolicy` — `batch` / `scenario` (each an `ExecutorSelection`:
`backend`, `workers`, `max_in_flight`, `may_shrink`, `fallback_order`),
`nested_execution` (default **False** — spawned children run inner-serial),
`fail_fast` (default True; `collect_errors=True` on the call site flips
per-item error collection), `retries` (process-backend infrastructure
retries: identical payloads resubmitted on a fresh pool for positions whose
results are missing after a `WorkerInfrastructureError`).

`DeterminismPolicy` — `require_manifest`, `changed_plan_profile`
(default `"reject"`), `mismatch_raises`.

`ResourceBudget` — `max_processes`, `max_threads`, `total_memory_bytes`,
`draw_cache_bytes`, `artifact_cache_bytes`, `max_in_flight`. Byte budgets
divide across workers when a `WorkerSpec` is built; children never re-read
the host environment.

**Auto-budget upgrade rule:** a session upgrades DEFAULT-sourced
`max_processes` / `max_threads` / `max_in_flight` to machine-derived values;
explicit or env-provided values are never upgraded. Admission clamps
(process window = `min(policy.max_in_flight or workers, budget.max_in_flight)`)
are recorded in diagnostics.

## Backend × capability matrix

| Capability | serial | threads | processes | dask |
|---|---|---|---|---|
| `price` / `price_many` | yes | — | — | — |
| fixed-batch MC (`BatchPlan`) | yes | yes | — | — |
| adaptive RQMC (compat mode) | yes | — | — | — |
| PDE prepared artifacts | yes | yes (batch bodies stay engine-bound) | — | — |
| scenarios (`run_scenarios`) | yes | yes (`request/v1` runner) | yes | yes |

Process/Dask scenario execution requires runners registered with
`value_kind="float"`; the default `request/v1` runner returns native result
objects and is therefore serial/threads-only. Explicitly requesting `dask`
with dask missing raises `CapabilityError` (no silent fallback); the legacy
engine-level `use_dask` flag keeps its historical warn-and-fall-back
behavior. Adaptive plans preserve the sequential stopping sequence and
support the serial backend only. Nested execution stays off by default:
children run with inner-serial policy and divided budgets.

## Diagnostics

Every outcome carries a `ReproducibilityManifest`
(`execution-manifest/0`; see `docs/execution/schemas/`) and immutable
diagnostics recording resolved policy values with their sources, admission
clamps, cache/lease events, retries, and scenario dedupe counts.
