# Execution Framework Phase 5 — Scenario, Portfolio, Process, and Dask

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec Phase 5 of `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`: typed scenario planning (`ScenarioPlan`), spawn-safe `WorkerSpec` process execution, a Dask backend over the same plans/reducers, the full surface-risk cell-menu port (no worker globals, no name parsing), `price_many` grouping, and the complete-payload comparison validator.

**Architecture:** A new `quantark/execution/scenario/` subpackage adds transformer/factory/runner registries (importable by string ID — the spawn-safety primitive), a planner that normalizes `ScenarioSpec`s into an immutable `ScenarioPlan` with verified mutation footprints, and a runner that executes the same plan on serial, threads, processes (spawn), or Dask backends with caller-order reassembly. Scenario **cells** are executed by registered *runners*: `request/v1` dispatches a transformed `PricingRequest` through the existing kernel; `equity-surface-shock/v1` runs `run_surface_shock_pipeline` from typed cell parameters. Process/Dask workers reconstruct everything from a `WorkerSpec` (registered factory IDs + canonical JSON payloads + expected dependency fingerprints) — never from closures, live objects, worker globals, or environment mutation.

**Tech Stack:** Python stdlib `concurrent.futures.ProcessPoolExecutor` with the **spawn** context; `dask.distributed.LocalCluster` (new dev-extra dependency); existing `quantark.execution` kernel/policy/lease/fingerprint infrastructure.

## Kickoff decisions (2026-07-17, user-selected)

1. **Dask:** full adapter + install `dask[distributed]` into the dev venv (tests skip gracefully when absent). Legacy `use_dask` engine paths untouched.
2. **Surface-risk:** full cell menu — LV/Heston × frozen/recalibrate × global/tenor-bucket/moneyness-bucket — as typed transformers driving quantark's `run_surface_shock_pipeline`.
3. **Gates:** CI-enforced spawn round-trip, complete-payload serial-vs-process equality, fault isolation, child-budget enforcement. Speed benchmarked on the dev machine and documented; the ≥2.5x@4-processes gate (spec §20 gate 6) stays a controlled-host production gate, like Phases 2/4.
4. **price_many:** ships now, on the same grouping planner.

## Global Constraints

- Nothing is pushed to origin; merge target is **local main** only. 0.3.0 release remains the user's manual decision.
- Worktree tests: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` so worktree source shadows the editable install.
- `test_snowball_quad_flat_identity_golden` is a known pre-existing failure — out of scope.
- Registry rule (program memory): `exact=True` whenever an adapter bypasses `engine.price()` or reconstructs engines from a fixed signature. (Phase 5 registers no new engine adapters; runners/factories/transformers use their own string-ID registries.)
- Spec hard invariants (§3.3): no mutable global/thread-local run context; explicit new-framework backend requests never silently fall back; nested parallelism disabled unless explicitly selected and budgeted; failed outcomes never enter aggregates; operational metadata never participates in economic equality.
- Direct legacy behavior unchanged: `run_surface_shock_pipeline`, DCN engines, legacy `use_dask` — all keep exact current semantics. Phase 5 only ADDS framework routes.
- Spawn-safety contract (§12.3): `WorkerSpec` payloads are canonical JSON-serializable primitives + registered IDs; a contract check at registration/plan time rejects closures, lambdas, non-module-level callables, and non-JSON payloads.
- No mutation of `os.environ` anywhere in the new code (the solution script's `QUANTARK_DCN_MC_WORKERS` env mutation is exactly what §2 bans); engine worker counts travel as explicit factory payload parameters.
- Code style: PEP 8, dataclasses + type hints, `quantark.util.numerical` for float comparisons, commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `docs/` files are gitignored on main — plan file needs `git add -f`.

## File Structure

```
quantark/execution/scenario/
    __init__.py            # public exports
    contracts.py           # ScenarioPlan, ScenarioCell, WorkerSpec, BaseInputsRef,
                           #   ScenarioItemOutcome helpers, SCENARIO_SCHEMA_VERSION
    registries.py          # TransformerRegistration/register_transformer,
                           #   register_runner, register_factory (+ lookups, contract checks)
    planner.py             # plan_scenarios(): normalize -> verify footprint -> group -> ScenarioPlan
    runner.py              # run_plan(): serial/threads/processes/dask dispatch + caller-order reassembly
    worker.py              # WorkerSpec build/verify + run_worker_cell (spawn/dask entry point)
    validate.py            # compare_scenario_outcomes(): complete-payload equality + ScenarioComparisonReport
quantark/execution/backends/processes.py   # bounded spawn ProcessPoolExecutor driver
quantark/execution/backends/dask_backend.py # same cells as Dask futures; CapabilityError when absent
quantark/asset/equity/riskmeasures/surface_shock_scenarios.py
                           # typed cell menu + transformer + runner + economics schema
quantark/execution/api.py  # run_scenarios implementation; price_many grouping
quantark/execution/policy.py  # (no changes needed — scenario selection already resolves)
test/execution/test_scenario_contracts.py
test/execution/test_scenario_planner.py
test/execution/test_scenario_runner.py
test/execution/test_worker_spec.py
test/execution/test_processes_backend.py
test/execution/test_dask_backend.py
test/execution/test_surface_shock_scenarios.py
test/execution/benchmark_phase5.py
pyproject.toml             # dev extra: dask[distributed]
```

Design notes locked here (referenced by tasks):

- **Runner registry, not engine adapters.** A scenario cell = (transformer output, runner). Runners are registered module-level callables `runner(cell, resolved_inputs, child_context) -> (value, normalized_economics)`. `request/v1` wraps kernel dispatch; workflow runners (surface shock) run multi-step pipelines. This is how the solution's per-cell job becomes typed instead of name-parsed.
- **Callable references, not bare IDs (plan-gate finding 1, 2026-07-17).** Registries are process-local dicts; a fresh spawn/Dask worker has NOT imported the registering modules, so ID lookup alone cannot work. Every registration therefore records a versioned `CallableRef(kind, ref_id, module, qualname, schema_version)`, and `WorkerSpec` carries the complete tuple of `CallableRef`s the plan needs (base factory, transformers, runners, engine factory) plus canonical constructor payloads. `run_worker_cell` IMPORTS each ref's module first (executing that module's `register_*` calls), then resolves the ID from the now-populated registry and verifies `getattr(import_module(ref.module), ref.qualname)` **is** the registered object — an unknown ID after import, a mismatched object, or an import failure raises `CapabilityError` before any numerical work. Engine parameters (num_paths, seed, engine worker counts) always travel as explicit canonical payload values, never as ambient state. Tests must exercise a genuinely clean worker (spawn children import nothing but `run_worker_cell`'s module) and non-default engine parameters.
- **Two base-input shapes for `run_scenarios(base_request, specs, engine_factory)`:** a live `PricingRequest` (serial/threads only — same trust model as the rest of the framework) or a `BaseInputsRef(factory_id, payload)` naming a registered factory that rebuilds base inputs (required for processes/dask; the child calls the factory, mirroring the solution's `_init_worker` but registered and explicit). A live-request plan that resolves to a process/dask backend raises `CapabilityError` (explicit requests never silently fall back).
- **Mutation-footprint verification (spec §10.2, hardened by plan-gate finding 2):** each transformer registration declares `components`: an ordered tuple of `(tag, extractor)` pairs. The planner fingerprints each component of the base and transformed snapshots via `try_fingerprint`, AND fingerprints the whole transformed snapshot. Enforcement: (a) changed component tags must be a subset of BOTH the spec's `mutation_tags` and the registration's `allowed_tags`, else `ValidationGateError`; (b) if the whole-snapshot fingerprint changed but no declared component changed, the mutation escaped the component schema — that is under-declaration and raises `ValidationGateError` (a change the schema cannot attribute is never silently accepted); (c) any `Uncanonicalizable` component or whole-snapshot fingerprint conservatively marks the cell `invalidate_all=True` (no artifact reuse; spec: "conservatively invalidated in full") instead of failing. Tests must cover: mutation of a field outside the component schema (b), and a spec declaring a tag the registration's `allowed_tags` does not permit (a).
- **Child contexts are inner-serial** (spec §12.5): the `WorkerSpec` child policy is serial/serial with `nested_execution=False`; child budgets are explicit numbers computed by dividing the parent budget across workers — children never re-read `QUANTARK_EXEC_*`.
- **Worker verification before preparation (spec §12.3):** the child compares `WorkerSpec.expected` (schema version, quantark/numpy/scipy versions, dtype tag) against locally computed values; mismatch raises `CapabilityError`/`DeterminismViolation` before any numerical work.
- **Complete-payload equality (spec §13.4):** `normalized_cell_payload` = normalized economics + plan-dependent numerical diagnostics, excluding operational metadata. The validator traverses the full payload tree, reports scenarios-vs-fields counts separately, missing/extra fields, and the first mismatching path; `all_scenarios_match` is computed, never accepted as input.
- **Dask scope:** the Dask backend runs **ScenarioPlan** cells (same `run_worker_cell` entry, same reducer/reassembly). BatchPlan-over-Dask is explicitly out of scope for Phase 5: fixed-batch bodies close over live prepared engines by design (Phase 2 contract), so they are not process-serializable; scenario-level parallelism is the productized Dask surface. Legacy Snowball/Phoenix `use_dask` is untouched (spec §12.4 keeps it non-bit-identical legacy).
- **`price_many` grouping (spec §13.3):** a pure planning helper groups items by `(engine class path, resolved adapter id)` and executes group-by-group so session artifact/draw caches see contiguous compatible work; results return in caller order; semantics (fail-fast, native values, `collect_errors`) unchanged. No parallelism is added to `price_many` in Phase 5.
- **Base-valuation-once (spec §13.3):** the planner deduplicates request-runner cells whose *base* dispatch fingerprint matches, so the base valuation for a bump family is computed once per plan on serial backends; workflow runners reuse base artifacts through the child-local artifact cache instead (per-worker, documented).

---

### Task 1: Scenario contracts and registries

**Files:**
- Create: `quantark/execution/scenario/__init__.py`
- Create: `quantark/execution/scenario/contracts.py`
- Create: `quantark/execution/scenario/registries.py`
- Test: `test/execution/test_scenario_contracts.py`

**Interfaces produced:**
- `SCENARIO_SCHEMA_VERSION = "scenario/v1"`
- `BaseInputsRef(factory_id: str, payload: tuple)` frozen dataclass — payload is a sorted `(key, value)` pair tuple of JSON-able primitives.
- `ScenarioCell(scenario_id, position, transformer_id, runner_id, parameters, mutation_tags, changed_tags, invalidate_all, cell_fingerprint, group_key, est_bytes)` frozen dataclass.
- `ScenarioPlan(plan_id, schema_version, base_kind ("request"|"inputs_ref"), base_fingerprint, engine_factory_id, cells: tuple, groups: tuple, backend_independent: bool = True)` frozen dataclass.
- `CallableRef(kind, ref_id, module, qualname, schema_version)` frozen dataclass — the spawn-reconstructable reference every registration records (see the design note above); `registries.callable_ref(kind, ref_id) -> CallableRef` builds one from a registration.
- `WorkerSpec(schema_version, base_ref: BaseInputsRef, callable_refs: tuple[CallableRef, …], child_policy_values: tuple, child_budget_values: tuple, expected: tuple)` frozen dataclass, all fields JSON-able (`callable_refs` covers base factory, engine factory, every transformer, and every runner the plan uses).
- `registries.register_transformer(transformer_id, fn, *, allowed_tags: frozenset, components: tuple, schema_version="1")` / `get_transformer(transformer_id)`
- `registries.register_runner(runner_id, fn, *, value_kind="native")` / `get_runner(runner_id)` — `value_kind in ("float", "native")`; only `"float"` runners are eligible for process/dask backends (Task 4/5 value contract)
- `registries.register_factory(factory_id, fn)` / `get_factory(factory_id)`
- `registries.check_worker_payload(obj)` — raises `ValidationGateError` unless `json.dumps(obj)` round-trips.

Registration contract (enforced in `register_*`): the callable must be resolvable by import — `getattr(importlib.import_module(fn.__module__), fn.__qualname__, None) is fn` — so spawn children can rebuild it by ID; lambdas, closures, and instance methods are rejected with `ValidationGateError`. Registries are module-level dicts with an idempotent-same-object rule: re-registering the same ID with the same function object is a no-op; a different object raises `ValidationGateError` (protects against test-order surprises without allowing silent replacement).

- [ ] **Step 1: Write failing tests** — `test/execution/test_scenario_contracts.py`:

```python
"""Scenario contracts and registries (spec sections 12.3, 13.1)."""
import json

import pytest

from quantark.execution.errors import ValidationGateError
from quantark.execution.scenario.contracts import (
    SCENARIO_SCHEMA_VERSION,
    BaseInputsRef,
    ScenarioCell,
    ScenarioPlan,
    WorkerSpec,
)
from quantark.execution.scenario import registries


def _module_level_fn(base, parameters):
    return base


def test_schema_version_present():
    assert SCENARIO_SCHEMA_VERSION == "scenario/v1"


def test_worker_spec_is_json_serializable():
    from quantark.execution.scenario.contracts import CallableRef
    spec = WorkerSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        base_ref=BaseInputsRef(factory_id="f", payload=(("a", 1),)),
        callable_refs=(
            CallableRef("factory", "f", "some.module", "build", "1"),
            CallableRef("runner", "toy/v1", "some.module", "toy_runner", "1"),
        ),
        child_policy_values=(("scenario.backend", "serial"),),
        child_budget_values=(("max_threads", 1),),
        expected=(("numpy", "2.0"),),
    )
    from quantark.execution.scenario.worker import worker_spec_to_payload
    payload = worker_spec_to_payload(spec)
    assert json.loads(json.dumps(payload)) == payload


def test_register_transformer_requires_importable_function():
    with pytest.raises(ValidationGateError):
        registries.register_transformer(
            "t-lambda", lambda b, p: b, allowed_tags=frozenset(), components=()
        )


def test_register_transformer_and_lookup():
    registries.register_transformer(
        "t-test/v1", _module_level_fn,
        allowed_tags=frozenset({"vol_surface"}), components=(),
    )
    reg = registries.get_transformer("t-test/v1")
    assert reg.fn is _module_level_fn
    assert reg.allowed_tags == frozenset({"vol_surface"})
    # idempotent same-object re-registration
    registries.register_transformer(
        "t-test/v1", _module_level_fn,
        allowed_tags=frozenset({"vol_surface"}), components=(),
    )
    # different object under the same id is rejected
    with pytest.raises(ValidationGateError):
        registries.register_transformer(
            "t-test/v1", test_schema_version_present,
            allowed_tags=frozenset(), components=(),
        )


def test_unknown_ids_raise_capability_error():
    from quantark.execution.errors import CapabilityError
    with pytest.raises(CapabilityError):
        registries.get_transformer("nope")
    with pytest.raises(CapabilityError):
        registries.get_runner("nope")
    with pytest.raises(CapabilityError):
        registries.get_factory("nope")


def test_check_worker_payload_rejects_non_json():
    with pytest.raises(ValidationGateError):
        registries.check_worker_payload({"x": object()})
    registries.check_worker_payload({"x": [1, 2.5, "s", None, True]})
```

(Note: `test_worker_spec_is_json_serializable` imports from `worker.py` which arrives in Task 5 — mark it `pytest.importorskip("quantark.execution.scenario.worker")` for now and drop the skip in Task 5.)

- [ ] **Step 2: Run tests, verify they fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement `contracts.py` and `registries.py`.** `TransformerRegistration` is a frozen dataclass `(transformer_id, fn, allowed_tags, components, schema_version)`. Keep three separate module-level dicts and a shared `_register(kind, table, key, value)` helper implementing the idempotency rule. `check_worker_payload` does `json.loads(json.dumps(obj))` inside try/except → `ValidationGateError`.
- [ ] **Step 4: Run the new tests — PASS** (with the one importorskip).
- [ ] **Step 5: Commit** `feat(execution): scenario contracts and importable registries (Phase 5)`.

### Task 2: Planner — normalization, footprint verification, grouping

**Files:**
- Create: `quantark/execution/scenario/planner.py`
- Test: `test/execution/test_scenario_planner.py`

**Interfaces produced:**
- `plan_scenarios(base, scenario_specs, engine_factory, *, context) -> ScenarioPlan` where `base` is a `PricingRequest` or `BaseInputsRef`, `engine_factory` is a callable or a registered factory id (str). Raises `ValidationError` on duplicate scenario ids; `ValidationGateError` on footprint under-declaration; `CapabilityError` on unknown transformer/runner ids.
- `resolve_base(base, context)` — returns `(base_kind, resolved_base_inputs, base_fingerprint)`; for `BaseInputsRef` it calls the registered factory ONCE in the parent for verification/planning (children call it again, worker-side).
- Each `ScenarioSpec.parameters` (already a sorted pair-tuple per `contracts.ScenarioSpec`) is passed to the transformer as a dict; the transformer must return a NEW object (identity check vs base: `result is not base_inputs` when it claims changes) — the planner verifies the base snapshot fingerprint is unchanged after every transform (transformer purity, spec §13.1).

Planner algorithm (implement exactly):
1. Validate unique `scenario_id`s, record caller `position`.
2. For each spec: resolve transformer registration; apply `fn(base_inputs, dict(parameters))`.
3. Footprint verification: for each `(tag, extractor)` in `registration.components`, compute `try_fingerprint(extractor(base_inputs))` and `try_fingerprint(extractor(transformed))`. Changed set = tags whose fingerprints differ (both non-None). Any `None` fingerprint on either side → `invalidate_all=True` for that cell (conservative full invalidation, spec §10.2) and the tag counts as *potentially changed* only for subset checking when the base-side fingerprint existed. `changed_tags - spec.mutation_tags` non-empty → `ValidationGateError` naming the under-declared tags.
4. Verify base purity: fingerprint of every base-side component unchanged after transform (compare against pre-transform values) — a mutating transformer raises `ValidationGateError`.
5. `runner_id` resolution: taken from spec `required_capabilities` entry `"runner:<id>"` if present, else default `"request/v1"`.
6. `cell_fingerprint = try_fingerprint((transformer_id, parameters, base_fingerprint, runner_id))`.
7. `group_key = (runner_id, transformer_id)`; `groups` = tuple of `(group_key, cell positions)` in first-appearance order.
8. Base-once dedupe marker: plan carries `base_fingerprint`; the runner uses it to evaluate the base valuation a single time for `request/v1` cells that request PnL-vs-base (Phase 5 `request/v1` returns the cell's own outcome only — dedupe applies when the same *transformed* fingerprint repeats: identical cells share one execution, recorded in diagnostics).

- [ ] **Step 1: Write failing tests** — cover: caller-order positions preserved; duplicate ids raise; under-declared footprint raises `ValidationGateError` (transformer that changes a component not in `mutation_tags`); spec declaring a tag outside the registration's `allowed_tags` raises `ValidationGateError`; a transformer mutating a field OUTSIDE the component schema (whole-snapshot fingerprint changes, no component changes) raises `ValidationGateError`; over-declaration is fine; uncanonicalizable component → `invalidate_all=True` but plan succeeds; mutating transformer (modifies base in place) raises; unknown transformer id raises `CapabilityError`; identical parameters → identical `cell_fingerprint`. Use a tiny fake "base inputs" dataclass with two float fields and transformers registered at module level in the test file:

```python
import dataclasses

from quantark.execution.scenario import registries
from quantark.execution.scenario.planner import plan_scenarios
from quantark.execution.contracts import ScenarioSpec


@dataclasses.dataclass(frozen=True)
class FakeInputs:
    spot: float
    vol: float


def bump_spot(base, parameters):
    return dataclasses.replace(base, spot=base.spot + parameters["ds"])


def bump_both(base, parameters):
    return dataclasses.replace(
        base, spot=base.spot + parameters["ds"], vol=base.vol + 0.01
    )


registries.register_transformer(
    "fake-bump-spot/v1", bump_spot,
    allowed_tags=frozenset({"spot"}),
    components=(("spot", lambda b: b.spot), ("vol_surface", lambda b: b.vol)),
)
```

(Extractors given as lambdas are fine: workers never unpickle them — a child resolves the whole registration by transformer id AFTER importing the registering module, which re-creates the lambdas at import time. The importability contract therefore applies to the registered `fn` only; state this explicitly in the `registries.py` module docstring.)

- [ ] **Step 2: Run tests — fail.**
- [ ] **Step 3: Implement `planner.py` per the algorithm above.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** `feat(execution): scenario planner with verified mutation footprints (Phase 5)`.

### Task 3: Serial runner, `run_scenarios`, outcomes, fault semantics

**Files:**
- Create: `quantark/execution/scenario/runner.py`
- Modify: `quantark/execution/api.py` (replace the `run_scenarios` CapabilityError stub; keep `price_many` untouched until Task 4)
- Test: `test/execution/test_scenario_runner.py`

**Interfaces produced:**
- `runner.run_plan(plan, base_inputs, engine_factory, context, *, collect_errors=False) -> list[ScenarioOutcome | PricingFailure]` — caller order always.
- Registered default runner `request/v1` (in `runner.py`, module level): `def run_request_cell(cell, resolved, child_context)` where `resolved` is a small namespace `(base_inputs, transformed, engine)`; it builds a `PricingRequest` from the transformed inputs (for request-kind plans the transformed object IS a `PricingRequest`) and calls `ExecutionKernel.dispatch(engine, request, child_context)`; returns `(outcome.value, outcome.normalized_economics, manifest_fp)`.
- `PricingSession.run_scenarios(base_request, scenario_specs, engine_factory)` — plans via Task 2, executes on the resolved `context.execution_policy.scenario.backend`: `"serial"` in this task; `"threads"` reuses the serial cell loop through a bounded ThreadPool ONLY when every cell's runner is `request/v1` and the engine factory produces a fresh engine per cell (each cell gets its own engine → no shared-instance reentrancy problem); `"processes"`/`"dask"` raise `CapabilityError("…arrives in Task 5/6…")` until those tasks land, then are wired in.
- Fail-fast default: first cell failure stops submission and raises the native exception; `collect_errors=True` (new keyword on `run_scenarios`) converts each failure to `PricingFailure(item_id=scenario_id, …)` and continues.
- Cancellation: between cells, `context.cancellation_token` (if not None and `.cancelled()` truthy) stops with `TaskExecutionError("cancelled")` — matches spec §15 "between scenarios".
- Identical-cell dedupe (plan-gate finding 3): cells with equal `cell_fingerprint` execute once, but only the scenario-INDEPENDENT execution payload `(value, economics, manifest_fp, diagnostics)` is cached — every cell still gets its OWN `ScenarioOutcome` constructed with that cell's `scenario_id` and position, so caller identity and validator pairing survive. Diagnostics record `scenario:deduped=<n>`. The dedupe test asserts both distinct scenario_ids on the outcomes AND equal economics.
- Each `ScenarioOutcome` = `(scenario_id, value, normalized_economics, diagnostics, manifest_fingerprint)`; scheduling metadata (elapsed seconds, worker id) goes ONLY into `diagnostics` (operational), never economics.

- [ ] **Step 1: Write failing tests** using a real engine but a cheap one — `BlackScholesEngine` + `EuropeanVanillaOption` (matrix-fixture style):

```python
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
```

Register (module level in the test) a `PricingRequest`-level transformer `euro-spot-bump/v1` whose components extract `(("spot", lambda r: r.pricing_env.spot_quote.spot),)` and which returns a NEW request wrapping a rebuilt `PricingEnvironment` (no mutation of the base env). Tests:
  - ordered outcomes: 3 bump sizes → 3 `ScenarioOutcome`s in caller order, each `economics_mapping(outcome)["pv"]` equal to a DIRECT `engine.price` call on an identically bumped env (exact float equality — same code path);
  - fail-fast: a transformer parameter that produces an invalid env (negative vol) raises out of `run_scenarios`;
  - `collect_errors=True`: same setup → `PricingFailure` at the failing position, valid `ScenarioOutcome`s elsewhere;
  - dedupe: two identical specs (different scenario_ids, same parameters) → both outcomes carry the same PV, `outcome_a.scenario_id != outcome_b.scenario_id` (each cell keeps its own identity), and diagnostics record `scenario:deduped=1`;
  - cancellation: a token object with `cancelled()` returning True after the first cell → `TaskExecutionError`, fewer executions than cells;
  - `session.run_scenarios` no longer raises the Phase 0 `CapabilityError`.

- [ ] **Step 2: Run tests — fail** (stub raises CapabilityError).
- [ ] **Step 3: Implement `runner.py` + wire `api.py`.** Session context flows through `context.child()` per cell group; the serial path holds the session's dispatch semantics unchanged (each cell dispatch behaves exactly like `session.execute`).
- [ ] **Step 4: Tests pass. Also run** `test_session_parity.py` and `test_matrix_parity.py` — unchanged behavior elsewhere.
- [ ] **Step 5: Commit** `feat(execution): serial scenario runner and session.run_scenarios (Phase 5)`.

### Task 4: Complete-payload validator + `price_many` grouping

**Files:**
- Create: `quantark/execution/scenario/validate.py`
- Modify: `quantark/execution/api.py` (`price_many`)
- Test: extend `test/execution/test_scenario_runner.py` (validator section) and `test/execution/test_session_parity.py` (price_many grouping regression)

**Interfaces produced:**
- `normalized_cell_payload(outcome) -> tuple` — THE canonical comparison payload (spec §13.4, plan-gate finding 4): complete `normalized_economics` (which every runner must populate with both economic fields and `numerical.`-prefixed plan-dependent diagnostics — the tier marking is part of the runner contract, enforced for `equity-surface-shock/v1` in Task 7) PLUS the outcome's native `value` under a reserved `value.` path (float compared exactly; `None` vs float is a mismatch — the cross-backend public value contract is validated, not assumed). Operational metadata (elapsed, PIDs, worker ids) lives only in diagnostics and never enters the payload.
- `compare_scenario_outcomes(left: Sequence, right: Sequence) -> ScenarioComparisonReport` with fields: `scenarios_compared`, `scenarios_matching`, `fields_compared`, `fields_matching`, `missing_fields: tuple`, `extra_fields: tuple`, `first_mismatch_path: str | None`, `all_scenarios_match: bool` (computed). Traversal: pair outcomes by `scenario_id`; a `PricingFailure` on either side = scenario mismatch; payload = `normalized_cell_payload` traversed recursively (dict/tuple/list nodes → dotted/indexed paths, floats compared for exact equality via `==` on the canonical tree so NaN handling matches `canonical_tree`); counts report FIELD leaves, never conflated with scenario counts (spec §2 table). Tests must include a perturbation of ONLY a `numerical.*` diagnostic field (must be reported as a mismatch) and a serial-vs-process `value` type/content comparison.
- `plan_price_groups(items) -> tuple` in `validate.py`? No — put it in `planner.py`: `plan_price_groups(items)` groups `(engine, request)` pairs by `(type(engine).__module__ + "." + type(engine).__qualname__)` preserving first-appearance group order and intra-group caller order; returns `((group_key, (original_indices…)), …)`.
- `price_many` executes group-by-group but fills a results list indexed by original position; `collect_errors` semantics identical; fail-fast raises on FIRST failure **in caller order** — to preserve exact legacy semantics, when `collect_errors=False` run in pure caller order (grouping applies only to the `collect_errors=True` path where partial results are the contract, OR simpler: fail-fast path also groups but on failure re-raises immediately; the items after the failing one may or may not have run — spec §15 requires stop-submission-on-first-failure, which grouped execution satisfies). **Decision: fail-fast path keeps pure caller order (zero behavior change); `collect_errors=True` path uses grouping.** Document in the docstring.

- [ ] **Step 1: Write failing tests:**
  - validator: two identical outcome lists → all match, `fields_compared > scenarios_compared`; perturb one PV → `all_scenarios_match=False`, `first_mismatch_path == "<scenario_id>:pv"`, `scenarios_matching == n-1`; drop a field from one side → reported in `missing_fields`;
  - a `PricingFailure` on one side → scenario counted, not matching;
  - price_many grouping: interleaved engines A,B,A,B with `collect_errors=True` → results in caller order, values equal to the ungrouped Phase-0 behavior (compare against direct loop);
  - price_many fail-fast unchanged: failing second item raises before third executes (instrument with a counting engine wrapper).
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit** `feat(execution): complete-payload scenario validator and price_many grouping (Phase 5)`.

### Task 5: WorkerSpec, worker verification, spawn process backend

**Files:**
- Create: `quantark/execution/scenario/worker.py`
- Create: `quantark/execution/backends/processes.py`
- Modify: `quantark/execution/scenario/runner.py` (wire `"processes"`)
- Modify: `quantark/execution/api.py` (auto-budget: upgrade `max_processes` like `max_threads` when default-sourced)
- Test: `test/execution/test_worker_spec.py`, `test/execution/test_processes_backend.py`

**Interfaces produced:**
- `worker.build_worker_spec(plan, context) -> WorkerSpec`: child policy = serial/serial + `nested_execution=False`; child budget = explicit values `(max_processes=1, max_threads=1, artifact_cache_bytes=parent_artifact_bytes // workers, draw_cache_bytes=parent_draw_bytes // workers, max_in_flight=1)`; `expected = (("schema", SCENARIO_SCHEMA_VERSION), ("quantark", <version>), ("numpy", np.__version__), ("scipy", scipy.__version__), ("dtype", "float64"))`.
- `worker.worker_spec_to_payload(spec) -> dict` / `worker.payload_to_worker_spec(payload) -> WorkerSpec` (JSON round-trip; drop the Task 1 importorskip now).
- `worker.verify_worker_environment(spec) -> None` — recomputes expected values in-process; schema/version mismatch → `CapabilityError`; dtype/dependency fingerprint mismatch → `DeterminismViolation`.
- `worker.run_worker_cell(spec_payload: dict, cell_payload: dict) -> dict` — TOP-LEVEL function (spawn target): rebuild `WorkerSpec`, `verify_worker_environment`, build the child context from the spec's explicit policy/budget values (`PricingRunContext` with fresh owned leases/cache per process — cached per-process via a module-level `functools.lru_cache` keyed by the spec fingerprint: this is per-process *immutable service construction*, not a mutable worker global; document why it does not violate §3.3), resolve base inputs via the registered factory, resolve the transformer/runner by ID, execute, and return a JSON-able outcome payload `{"scenario_id", "value_pv", "economics": [[k, v]…], "manifest_fingerprint", "elapsed_seconds", "error": None}` or `{"error": {"type", "message"}, …}` — the parent maps these back to `ScenarioOutcome`/`PricingFailure`. **Cross-backend value contract (plan-gate finding 4):** native value objects do not cross the process boundary in v1, so a runner eligible for process/dask backends must return a `float` (or `None`) value on EVERY backend — the planner rejects (CapabilityError) process/dask plans whose runners do not declare `value_kind="float"` at registration (`register_runner(..., value_kind="float"|"native")`; `request/v1` is `"native"` and therefore serial/threads-only, which it already is by the live-request rule). The Task 4 validator compares `value` across backends, so the contract is enforced, not assumed. Surface-shock cells return `pnl` as the float value and their full field mapping inside economics.
- `backends/processes.iter_ordered(cells, spec_payload, workers, window, mp_context="spawn", fail_fast=True) -> iterator of (position, outcome_payload)`: `ProcessPoolExecutor(max_workers=…, mp_context=multiprocessing.get_context("spawn"))`, index-ordered submission with the same bounded-window/buffered-reassembly discipline as `backends/threads.py` (copy the pattern, not the lease plumbing — parent-side admission uses task slots via `AdmissionLeases` only if a lease manager is passed; keep parent-side per-cell slot + est-bytes lease identical to threads). **Fail-fast with error payloads (plan-gate finding 5):** worker failures arrive as SUCCESSFUL futures carrying an error payload, so `future.result()` never raises for them — the completion loop must inspect each completed payload for `"error"` IMMEDIATELY, before buffering it or submitting more cells; when `fail_fast` is set it then stops submission, cancels pending futures, drains, and raises `TaskExecutionError` carrying the child error type/message — even when the failing cell's position is LATER than unfinished earlier cells. A dedicated out-of-order-failure test (failing cell finishes first via a slow-sleep toy runner on the earlier positions) asserts no new cells were admitted after the failure completed.
- Runner wiring: `"processes"` backend requires `plan.base_kind == "inputs_ref"` and a registered (string) engine factory — else `CapabilityError` with an explanatory message. Workers = `min(policy.scenario.workers, budget.max_processes, n_cells)` with clamp records.

- [ ] **Step 1: Write failing tests.** `test_worker_spec.py`: round-trip JSON; verification passes in-process; corrupt expected numpy version → `DeterminismViolation`; corrupt schema version → `CapabilityError`; child budget values divide the parent budget; child policy is serial + `nested_execution=False`. `test_processes_backend.py` (guard `pytest.mark.skipif(sys.platform == "emscripten")` not needed — macOS/linux fine): register at module level in a **helper module** `test/execution/scenario_process_helpers.py` (importable by spawn children through the inherited `sys.path`):

```python
"""Spawn-importable scenario fixtures for process/dask backend tests."""
import dataclasses

from quantark.execution.scenario import registries


@dataclasses.dataclass(frozen=True)
class ToyInputs:
    spot: float
    vol: float


def build_toy_inputs(payload):
    return ToyInputs(spot=payload["spot"], vol=payload["vol"])


def toy_bump(base, parameters):
    return dataclasses.replace(base, spot=base.spot + parameters["ds"])


def toy_runner(cell, resolved, child_context):
    t = resolved.transformed
    pv = t.spot * t.vol  # deterministic, cheap, float
    return pv, (("pv", pv), ("spot", t.spot)), None


def toy_runner_failing(cell, resolved, child_context):
    if resolved.transformed.spot > 100.0:
        raise ValueError("boom")
    return toy_runner(cell, resolved, child_context)


registries.register_factory("toy-inputs/v1", build_toy_inputs)
registries.register_transformer(
    "toy-bump/v1", toy_bump,
    allowed_tags=frozenset({"spot"}),
    components=(("spot", lambda b: b.spot), ("vol_surface", lambda b: b.vol)),
)
registries.register_runner("toy/v1", toy_runner)
registries.register_runner("toy-failing/v1", toy_runner_failing)
```

  Tests: (a) 6 cells on 2 spawn workers → outcomes equal (exact floats) to the serial run of the same plan, caller order — this is intrinsically the CLEAN-WORKER reconstruction test (spawn children import only `worker.py`'s module; every registration is reached via `CallableRef` imports), and the toy factory takes a non-default parameter (`vol`) proving constructor payloads travel; (b) fault: `toy-failing/v1` with one poisoned cell + `collect_errors=True` → one `PricingFailure`, five successes; fail-fast → raises `TaskExecutionError` (parent wraps the child error payload) and cancels pending; (b2) out-of-order failure: failing cell at a LATE position finishes first (earlier toy cells sleep) → no new submissions after the failure completes (assert via submitted-count instrumentation), `TaskExecutionError` raised; (c) child-budget: a runner that asserts its `child_context.resource_budget.max_processes == 1` and `max_threads == 1` (nested off by default) and returns them in economics — parent asserts; (d) explicit `"processes"` with a live `PricingRequest` base → `CapabilityError`, no silent fallback; (d2) explicit `"processes"` with a `value_kind="native"` runner → `CapabilityError`; (e) WorkerSpec expected-mismatch injected via monkeypatched builder → child raises before running (assert error type surfaces in the failure payload); (f) unknown `CallableRef` (id registered under a different object in the child's import, or unimportable module) → `CapabilityError` before numerical work.
- [ ] **Step 2: Run — fail.**
- [ ] **Step 3: Implement `worker.py`, `backends/processes.py`, runner wiring, session auto-budget `max_processes` upgrade.**
- [ ] **Step 4: Tests pass — including on this macOS host (spawn is the default and only supported start method here, satisfying the spec §12.3 spawn/macOS gate).**
- [ ] **Step 5: Commit** `feat(execution): spawn WorkerSpec process backend with child budgets (Phase 5)`.

### Task 6: Dask backend

**Files:**
- Modify: `pyproject.toml` (dev extras: `"dask[distributed]"`)
- Create: `quantark/execution/backends/dask_backend.py`
- Modify: `quantark/execution/scenario/runner.py` (wire `"dask"`)
- Test: `test/execution/test_dask_backend.py`

**Interfaces produced:**
- `dask_backend.available() -> bool`; `dask_backend.iter_ordered(cells, spec_payload, workers, client=None)` — submits the SAME `run_worker_cell` (imported by the dask worker by module path) as dask futures with deterministic keys `f"{plan_id}-{position}"`, gathers, yields `(position, payload)` in caller order. When `client is None`, creates a short-lived `distributed.LocalCluster(n_workers=workers, threads_per_worker=1, processes=True)`; a caller-supplied client is borrowed (not closed).
- Explicit `"dask"` request with dask missing → `CapabilityError` (spec §12.4) — tested with `monkeypatch.setitem(sys.modules, "distributed", None)` style import guard.
- Same `WorkerSpec` verification runs inside dask workers (it is inside `run_worker_cell`).
- NOT in scope: BatchPlan-over-Dask (documented rationale in the module docstring — fixed-batch bodies are engine-bound by the Phase 2 contract); legacy `use_dask` untouched (grep-guard test: `snowball_mc_engine.py` unmodified this phase).

- [ ] **Step 1: `.venv/bin/pip install "dask[distributed]"` and add to `[project.optional-dependencies] dev`.** Record installed version in the commit message.
- [ ] **Step 2: Write failing tests** (`pytest.importorskip("distributed")` at module scope, plus one non-skipped test for the missing-dask CapabilityError using an import-guard monkeypatch): same toy plan as Task 5 → dask outcomes exactly equal serial outcomes, caller order; fault isolation with `collect_errors=True`; explicit-dask-unavailable raises.
- [ ] **Step 3: Run — fail; implement; pass.** Use one module-scoped LocalCluster fixture to keep the suite fast.
- [ ] **Step 4: Full execution-test sweep** `PYTHONPATH=$PWD .venv/bin/python -m pytest test/execution -n0 -q` (serial: process/dask tests manage their own workers).
- [ ] **Step 5: Commit** `feat(execution): Dask scenario backend over the same plans and reducers (Phase 5)`.

### Task 7: Surface-risk workflow port — typed full cell menu

**Files:**
- Create: `quantark/asset/equity/riskmeasures/surface_shock_scenarios.py`
- Test: `test/execution/test_surface_shock_scenarios.py`
- Test helper: `test/execution/surface_shock_process_helpers.py` (spawn-importable fixture factory built on `test/dcn_fixtures.synthetic_cleaned_set`)

**Interfaces produced:**
- `SurfaceShockCell(model: str, mode: str, dsigma: float, tenor_bucket: tuple | None, moneyness_bucket: tuple | None)` frozen dataclass — the TYPED replacement for the solution's name parsing; `cell_scenario_id(cell)` renders a stable id (rendering only — NEVER parsed back).
- `build_surface_shock_cells(tenors: Sequence[float], moneyness_buckets: Sequence[tuple], dsigma: float) -> tuple[SurfaceShockCell, …]` — **the solution's exact production menu** (plan-gate finding 6 scope clarification): `lv_frozen`, `lv_recalibrate`, `heston_frozen`, `heston_recalibrate` global cells + one `lv_recalibrate` cell per tenor (`tenor_bucket=(t-1e-6, t+1e-6)`) + one per moneyness bucket (4 + n_tenors + n_buckets cells; 13 for the solution's 2-tenor/7-bucket data — data-driven, not hardcoded). "Full cell menu" from kickoff means **port parity with the solution's 13-cell workflow**, NOT the 40-cell model×mode×bucket Cartesian product the solution never ran. The typed `SurfaceShockCell`, however, spans the whole Cartesian space with no name-parsing constraint — proven by a test pricing one combination the solution could not express (`heston` × `recalibrate` × tenor bucket) through the same transformer/runner.
- `cells_to_scenario_specs(cells) -> tuple[ScenarioSpec, …]` with `transformer_id="equity-surface-shock/v1"`, `runner_id` carried as `required_capabilities={"runner:equity-surface-shock/v1"}`, `mutation_tags=frozenset({"vol_surface"})`, parameters = the typed cell fields as sorted pairs (buckets as lists → tuples).
- Registered transformer `equity-surface-shock/v1`: input snapshot is a `SurfaceShockInputs(cleaned, spot, rate_curve, carry_curve, engine_settings: tuple)` frozen dataclass; the transformer applies `shock_cleaned_ivs(cleaned, dsigma, tenor_bucket, moneyness_bucket)` and returns a new `SurfaceShockInputs` (pure; base untouched). Components: `(("vol_surface", lambda s: _cleaned_iv_tree(s.cleaned)), ("rate_curve", …), ("spot", lambda s: s.spot))` where `_cleaned_iv_tree` renders the slices into a canonicalizable nested tuple `(expiry, ((log_moneyness, iv), …))` — write it here, since `CleanedQuoteSet` may not canonicalize as-is.
- Registered runner `equity-surface-shock/v1`: rebuilds env-builder + engine factory from `engine_settings` (num_paths, seed, engine workers — EXPLICIT payload numbers; no env mutation) and calls `run_surface_shock_pipeline(product=…, model=cell.model, mode=SurfaceShockMode(cell.mode), …)` with the BASE cleaned set and the cell's bucket/dsigma parameters (the pipeline applies the shock internally — the transformer's shocked snapshot exists for footprint verification; the runner passes base inputs + typed parameters so the pipeline's own auditable shock path stays the single implementation). Returns `(result.pnl, economics, None)` where economics = the marked field mapping below.
- Economics schema (spec §13.4 explicit marking): economic = `pv_base, pv_shocked, pnl, no_arb_passed, mode, model, shock.*, notes`; numerical-plan = `calibration.{base,shocked}.{rmse_iv, success}` and `artifact_diagnostics.*`; operational = elapsed (diagnostics only). Encode as `economics = (("pv_base", …), …, ("numerical.calibration.base.rmse_iv", …))` — a flat mapping with a `numerical.` prefix marking the plan-dependent tier, so the Task 4 validator's complete-payload comparison covers every field while the tiers stay distinguishable.
- The product travels via a registered factory too: the spawn helper registers `surface-shock-test-inputs/v1` returning `(SurfaceShockInputs, product)` deterministically from `synthetic_cleaned_set()` + `make_dcn(DCN_A)`; payload = `{"num_paths": 4096, "seed": 42}`.

- [ ] **Step 1: Write failing tests:**
  - menu construction: solution-shaped inputs (2 tenors, 7 buckets) → 13 cells, ids unique, every id renders from typed fields;
  - Cartesian expressiveness (plan-gate finding 6): a `heston` × `recalibrate` × tenor-bucket cell — a combination the solution's name grammar could not express — prices through the same transformer/runner (serial, small paths) and its footprint verifies;
  - transformer footprint: planner accepts the specs (changed set = {vol_surface}); an over-eager variant that also bumps spot (register a test-local broken transformer) → `ValidationGateError`;
  - **serial equality vs direct:** for the 4 global cells (paths=4096 — the existing `test_surface_shock_pipeline.py` budget), `run_scenarios` serial economics == a direct `run_surface_shock_pipeline` call, field-for-field exact (same seeds, same code path);
  - **process equality gate (the Phase 5 headline):** run the SAME plan serial and on 2 spawn processes; `compare_scenario_outcomes` → `all_scenarios_match=True`, `fields_compared` reported and > cells count (spec §2: never label field comparisons as cells). Use a REDUCED menu for runtime (global lv_frozen + lv_recalibrate + one tenor + one moneyness cell, heston cells excluded from the process gate for time) — the FULL 13-cell equality run goes into `benchmark_phase5.py` instead, and the test asserts the reduced set covers every transformer code path (frozen/recalibrate/tenor/moneyness);
  - fault isolation on real cells: poison one cell (`max_calibration_rmse_iv=0.0` heston cell via engine_settings override → `NumericalError` in-child) with `collect_errors=True` → that cell is a `PricingFailure`, the other cells' economics still exactly match their serial values;
  - no-name-parsing guard: `grep`-style assertion that `surface_shock_scenarios.py` source contains no `.startswith(` / `.split(` on scenario ids (read the file in the test; cheap and honest).
- [ ] **Step 2: Run — fail; implement; pass.** Watch runtimes: keep heston cells only in serial tests (calibration ~seconds each), mark the process gate `@pytest.mark.slow` if it exceeds ~60s but leave it in the default run (`-n auto` absorbs it in CI).
- [ ] **Step 3: Full-suite regression** on the touched equity module: `python -m pytest test/test_surface_shock_pipeline.py test/execution -q`.
- [ ] **Step 4: Commit** `feat(equity): typed surface-shock scenario port on the execution framework (Phase 5)`.

### Task 8: Benchmark, inventory/docs, full suite

**Files:**
- Create: `test/execution/benchmark_phase5.py`
- Modify: `quantark/execution/__init__.py` (docstring → Phases 0-5; export `scenario` names)
- Modify: `quantark/execution/inventory.py` ONLY IF a CI gate references scenario capability (none planned — scenario capability is session-level, not per-engine; add a module-docstring note instead)
- Test: `test/execution/test_inventory.py` untouched unless imports break

**Deliverables:**
- `benchmark_phase5.py` (dev-machine evidence, spec §20 gate 6 attribution): full 13-cell surface-shock menu, `paths=4096` — serial wall vs 4 spawn processes wall, median of ≥3 reps, plus the complete-payload equality check on every rep (a speedup with a mismatch is a FAIL). Docstring records the dev-host numbers and restates that the ≥2.5x release gate runs on the controlled host with production-sized cells (serial >10s).
- `__init__.py` exports: `ScenarioPlan`, `BaseInputsRef`, `plan_scenarios`, `compare_scenario_outcomes`, `SurfaceShockCell` stays in the equity module (asset code never imported by the kernel — keep the one-way dependency).
- Full suite: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` → only the known quad-golden failure.

- [ ] **Step 1: Write and run the benchmark; paste the numbers into its docstring.**
- [ ] **Step 2: Update `__init__.py`; run the execution suite.**
- [ ] **Step 3: Full test suite; verify only the known failure.**
- [ ] **Step 4: Commit** `feat(execution): Phase 5 benchmark and exports; docs`.

## Plan-Gate Findings Applied (Codex Tier 1, 1 iteration, 2026-07-17)

1. **[high] Spawn workers can't reconstruct registered callables** → `CallableRef(kind, ref_id, module, qualname, schema_version)` recorded at registration; `WorkerSpec.callable_refs` replaces bare IDs; `run_worker_cell` imports each ref module (populating the registry), then verifies the imported object IS the registered one; clean-worker + non-default-parameter tests added (Task 5a/f).
2. **[high] Footprint verification incomplete** → enforcement is now: changed ⊆ `mutation_tags` AND ⊆ registration `allowed_tags`; whole-snapshot fingerprint detects mutations escaping the component schema → `ValidationGateError`; uncanonicalizable → conservative `invalidate_all`. Tests added (Task 2).
3. **[high] Dedupe corrupted scenario identity** → cache the scenario-independent execution payload only; construct a distinct `ScenarioOutcome` per cell with its own scenario_id; test asserts distinct ids (Task 3).
4. **[high] Validator missed diagnostics and the value contract** → `normalized_cell_payload` = complete economics (incl. `numerical.*` tier) + native `value` under a reserved path; `register_runner(value_kind=…)` gates process/dask eligibility to float-valued runners; diagnostic-only-perturbation and cross-backend value tests added (Tasks 4/5).
5. **[high] Error payloads defeated fail-fast** → the processes completion loop inspects payloads on completion (before buffering/submission), cancels pending, and raises `TaskExecutionError`; out-of-order failure test added (Task 5b2).
6. **[medium] Menu scope ambiguity** → "full cell menu" is pinned to port parity with the solution's exact 13-cell workflow (not the 40-cell Cartesian product it never ran); the typed cell type spans the full Cartesian space, proven by pricing a combination the solution could not express (Task 7).

## Self-Review (checked while writing)

- **Spec coverage:** §13.1 typed spec/transformer (T1/T2), §13.2 plan contents (T2), §13.3 execution behavior incl. price_many + base-once/dedupe (T3/T4), §13.4 outcome/equality + report shape (T4/T7), §12.3 WorkerSpec/spawn/verification/child budgets (T5), §12.4 Dask + unavailable→CapabilityError + legacy untouched (T6), §12.5 nested-off default (T5c), §15 fail-fast/collect_errors/cancellation/typed errors (T3/T5), §10.2 footprint verification + conservative invalidation (T2), Phase 5 exit gates: spawn (T5), child-budget (T5c), complete-payload (T7), fault (T5b/T7), scenario speed measured (T8), nested off (T5c).
- **Known trims (deliberate, reviewable):** BatchPlan-over-Dask out (rationale in T6); native value objects don't cross process boundaries (economics payload is the contract; T5); per-worker (not global) base-artifact reuse for workflow cells; threads backend for scenarios limited to fresh-engine request cells.
- **Type consistency:** `ScenarioSpec`/`ScenarioOutcome` reuse the Phase 0 contracts; `parameters` is a sorted pair-tuple everywhere; runner return shape `(value, economics, manifest_fp)` is uniform across T3/T5/T7.
