# Execution Framework Phase 6 — Cleanup and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the `quantark.execution` program: parity-gated removal of the one
genuinely duplicated legacy loop, publish the four spec-mandated deliverables
(capability matrix, policy guide, reproducibility schema, migration examples), prove
downstream compatibility (otc-price-adapter), and prepare the 0.3.0 release without
pushing.

**Architecture:** Spec `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`
Phase 6 (§21) + §17.3 removal rule + §22 versioning + §24 definition of done.
Kickoff decisions (user-selected): parity-gated unification only; generated + tested
docs; prep 0.3.0 with no push; local gate rerun with documented controlled-host
deferrals; **plus** an explicit compatibility gate for `/Users/fuxinyao/otc-price-adapter`.

**Tech Stack:** Python 3.11+, pytest, dask[distributed] (dev extra), `jsonschema`
(new dev extra), hatchling/`python -m build`.

## Scouting results this plan is built on (2026-07-17/18 audit)

- **Already shared (no work):** single RQMC stopping loop (`run_rqmc_traced`; the
  bsm `qmc_rqmc_driver` is a 4-line re-export shim); one Dupire session-prep helper
  (`execution/prep/dupire.py`, all three adapter families import it); one fingerprint
  utility; no legacy multiprocessing/joblib helper exists; phoenix_vol_mc_engines
  **imports** `_qmc_normals`/`_ArrayPathGenerator`/etc. from snowball_vol_mc_engines
  (shared, not duplicated); DCN vol engines use the shared `qmc_draws` layer.
- **The one parity-plausible duplicate:** the legacy Dask batch loop is triplicated
  line-for-line: `SnowballMCEngine._price_parallel` (snowball_mc_engine.py:1921),
  `SnowballMCEngine._price_ko_reset_parallel` (:2034), and
  `PhoenixMCEngine._price_parallel` (phoenix_mc_engine.py:1073). Identical batch-size
  split, identical `delayed(...)`/`compute(*...)` fan-out, identical 8-key sum/sum²
  accumulation, identical stderr formula, identical error messages. The vol-model
  subclasses inherit these methods (no fourth copy).
- **Keep-with-rationale (document, do NOT remove):** DCN legacy thread-batch loop in
  `price_detailed` vs the Phase 2 adapter plan/execute/reduce (arithmetic already
  shared via `_LegAccumulator`/`_finalize_dcn_result`; §17.1 keeps
  `QUANTARK_DCN_MC_WORKERS`/`num_workers`/direct path); two Sobol draw-cache scopes
  (process-global `QMCDrawCache` vs session `DrawRepository` — two scopes BY DESIGN,
  generator shared); legacy `QMCDrawCache` LRU vs `backends/admission.py` (different
  scopes, not duplicates).
- **Compat baseline (already measured):** otc-price-adapter (`quantark==0.2.5`
  pinned, own venv) passes **54/54** against its pinned wheel AND **54/54** against
  current quantark main via `PYTHONPATH=/Users/fuxinyao/quant-ark` shadow. Each full
  run takes ~2h55m — schedule reruns as background merge gates, never inner-loop.
- **Release facts:** origin/main == v0.2.5 (03f9b10, 2026-07-03); local main is 239
  commits ahead; pyproject says 0.2.6 (bumped in 318009e, never tagged — the frozen
  quant-mini-project wheel); CHANGELOG's newest entry is 0.2.2 (0.2.3–0.2.5 tagged
  without entries); `.github/workflows/release.yml` is tag-triggered publishing.
- **Inventory:** `InventoryRecord` (inventory.py:126) exposes all capability fields;
  `docs/` has zero execution-framework docs today.
- `jsonschema` is NOT installed; `build` IS importable in `.venv`.

## Global Constraints

- **No public-surface change** (spec §17.1): `use_dask`, `num_batches`, warnings,
  error messages, result classes, env-var timing all byte-preserved. **No
  deprecation warnings** (§17.3). Removal only where both call sites use the same
  internal reducer with pre-refactor goldens proving bitwise identity.
- Canonical `quantark.*` imports; `quantark.util.numerical` helpers (never raw
  tolerances); PEP 8; docstrings on public classes/functions.
- Worktree tests: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest`
  (editable install must be shadowed). Serial debugging: `-n0`.
- New `docs/**` files need `git add -f` (gitignore); `example/*.py` files do not.
- Known pre-existing failure `test/test_snowball_quad_engine.py::test_snowball_quad_flat_identity_golden`
  is OUT OF SCOPE — everything else must be green.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Nothing is pushed to origin; no tags are created. 0.3.0 push/tag/publish is the
  user's manual decision.

---

### Task 1: Pre-refactor legacy-Dask goldens

Freeze bitwise goldens of the CURRENT legacy Dask path before touching it, so
Task 2's extraction is provably behavior-preserving (same discipline as Phase 4's
`pde_phase4_goldens.json`).

**Files:**
- Modify: `test/execution/freeze_goldens.py` (add `phase6_dask` mode)
- Create: `test/execution/goldens/legacy_dask_phase6_goldens.json` (generated)
- Create: `test/execution/test_legacy_dask_goldens.py`

**Interfaces:**
- Consumes: `test/execution/matrix_fixtures.py` — `_snowball()`, `_phoenix()`,
  `_eq_flat_env()`, `_eq_grid_env()`, `_mcp(**kw)` (MCParams builder), and
  `create_ko_reset_snowball` from `quantark.asset.equity.product.option`.
- Produces: golden JSON rows keyed `snowball-dask`, `ko-reset-dask`, `phoenix-dask`,
  `lv-snowball-dask`, each `{"price": repr-float, "std_error": ..., "num_paths": int,
  "ko_probability": ..., "v0_probability": ..., "v1_probability": ...,
  "avg_ko_time": float|null, "batches_used": int}`.

- [ ] **Step 1: Add the `phase6_dask` freeze mode**

In `freeze_goldens.py`, following its existing mode pattern, add a function that
builds four engine/product pairs, all with `use_dask=True, num_batches=3` and
`MCParams(seed=42, num_paths=20_000, ...)` via `_mcp`:

1. `SnowballMCEngine` on `_snowball()` + `_eq_flat_env()`
2. `SnowballMCEngine` on `create_ko_reset_snowball(...)` (mirror the construction
   used by `matrix_fixtures._build_equity_mc`'s `ko_reset()` builder at
   matrix_fixtures.py:680 — reuse the same arguments so the product is valid and
   NOT already knocked out)
3. `PhoenixMCEngine` on `_phoenix()` + `_eq_flat_env()`
4. `LocalVolSnowballMCEngine` on `_snowball()` + `_eq_grid_env()` (proves the
   inherited `_price_parallel` path for vol subclasses)

For each, call `engine.price_detailed(product, env)` if that is what returns the
`SnowballMCResult`/`PhoenixMCResult` — otherwise `engine.price(...)` returns the
result object on these engines; check the engine and use the call that yields the
full result object with `std_error`/`batches_used` (the existing use_dask tests at
`test/test_snowball_mc_engine.py:691` show the call shape). Serialize every float
with `repr()` so equality is bitwise.

- [ ] **Step 2: Generate and inspect the goldens**

Run: `.venv/bin/python test/execution/freeze_goldens.py phase6_dask`
Expected: writes `test/execution/goldens/legacy_dask_phase6_goldens.json` with 4
rows; verify `batches_used == 3` and `avg_ko_time` is present (a null is fine only
if the fixture genuinely never knocks out — prefer fixtures with nonzero
ko_probability so the stat-aggregation paths are exercised).

- [ ] **Step 3: Write the golden-assertion test**

`test/execution/test_legacy_dask_goldens.py`: for each row, rebuild the same
engine/product/env, price, and assert **exact** equality (`==` on floats after
`float(repr_value)`, `==` on ints, `is None` for null). Skip module with
`pytest.importorskip("dask")`.

- [ ] **Step 4: Run the test (must pass on UNCHANGED code)**

Run: `.venv/bin/python -m pytest test/execution/test_legacy_dask_goldens.py -n0 -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add test/execution/freeze_goldens.py test/execution/goldens/legacy_dask_phase6_goldens.json test/execution/test_legacy_dask_goldens.py
git commit -m "test(execution): freeze legacy Dask-path goldens before Phase 6 dedup"
```

---

### Task 2: Extract the shared legacy Dask batch reducer

**Files:**
- Create: `quantark/asset/equity/engine/mc/autocallable_dask_batch.py`
- Modify: `quantark/asset/equity/engine/mc/snowball_mc_engine.py` (`_price_parallel`
  :1921-2032, `_price_ko_reset_parallel` :2034-2136)
- Modify: `quantark/asset/equity/engine/mc/phoenix_mc_engine.py` (`_price_parallel`
  :1073-1176)
- Test: `test/execution/test_legacy_dask_goldens.py` (unchanged — the gate)

**Interfaces:**
- Produces:

```python
"""Shared legacy Dask batch fan-out/reduction for autocallable MC engines.

One implementation of the batch-size split, delayed fan-out, and sum/sum²
reduction that was previously triplicated across SnowballMCEngine (vanilla +
KO-reset) and PhoenixMCEngine. Spec §17.3: both legacy routes now use the same
internal reducer; behavior (messages, arithmetic order, result fields) is
byte-preserved and gated by test_legacy_dask_goldens.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional


@dataclass(frozen=True)
class DaskBatchTotals:
    price: float
    std_error: float
    num_paths: int
    ko_probability: float
    v0_probability: float
    v1_probability: float
    avg_ko_time: Optional[float]
    batches_used: int


def run_autocallable_dask_batches(
    *,
    num_batches: int,
    total_paths: int,
    batch_fn: Callable[..., Dict[str, float]],
    batch_kwargs: Mapping[str, object],
) -> DaskBatchTotals:
    ...
```

The body is a verbatim consolidation of the triplicated code: `ValidationError`
with the exact message `f"num_batches must be positive, got {num_batches}"`
raised FIRST (preserving the lazy-validation timing — this function is only
reached when the Dask path is entered); the `base`/`remainder` split; the
skip-empty-batch loop calling
`delayed(batch_fn)(batch_id=batch_id, batch_num_paths=batch_num_paths, **batch_kwargs)`;
`results = compute(*batch_results)`; the 8-accumulator loop in the same
iteration order with the same `int()`/`float()` coercions and `.get()` defaults;
`PricingError("Dask parallel pricing produced zero simulated paths")` on
`total_n <= 0`; the identical `sample_var`/`std_error` arithmetic; and the
probability/avg_ko_time computation. Import `from dask import compute, delayed`
**inside the function** (dask is optional; this path is only reached when the
engine already verified availability). Import `ValidationError`/`PricingError`
from `quantark.util.exceptions`.

- [ ] **Step 1: Write `autocallable_dask_batch.py`** with the exact code above
  (full body transplanted from snowball_mc_engine.py:1939-2021 — that segment is
  the canonical copy; keep the arithmetic statement-for-statement).

- [ ] **Step 2: Rewire the three call sites.** Each `_price_parallel*` keeps: its
  grid build (`_build_time_grid` / `_build_time_grid_ko_reset`), then calls

```python
totals = run_autocallable_dask_batches(
    num_batches=self.num_batches,
    total_paths=int(self.params.num_paths),
    batch_fn=self._price_single_batch,   # or _price_single_batch_ko_reset
    batch_kwargs=dict(product=product, pricing_env=pricing_env, S=S, T=T,
                      r=r, q=q, sigma=sigma, all_times=all_times,
                      dt_array=dt_array, ko_indices=ko_indices,
                      ki_indices=ki_indices),  # ko-reset passes grid=grid instead
)
return SnowballMCResult(  # or PhoenixMCResult
    price=totals.price, std_error=totals.std_error, num_paths=totals.num_paths,
    ko_probability=totals.ko_probability, v0_probability=totals.v0_probability,
    v1_probability=totals.v1_probability, avg_ko_time=totals.avg_ko_time,
    batches_used=totals.batches_used,
)
```

Delete the now-dead inline loops. Do NOT touch `_price_single_batch*` kernels,
`use_dask` resolution, the `DASK_AVAILABLE` fallback warning, or dispatch logic.
Remove `from dask import compute, delayed` from the two engine modules ONLY if
nothing else in the module uses them (check: snowball_mc_engine.py:51,
phoenix_mc_engine.py:39-40).

- [ ] **Step 3: Golden gate + affected suites**

Run: `.venv/bin/python -m pytest test/execution/test_legacy_dask_goldens.py test/test_snowball_mc_engine.py test/test_phoenix_option.py test/execution/test_session_parity.py -v`
Expected: all pass; goldens bitwise-identical.

- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/mc/autocallable_dask_batch.py quantark/asset/equity/engine/mc/snowball_mc_engine.py quantark/asset/equity/engine/mc/phoenix_mc_engine.py
git commit -m "refactor(mc): single legacy Dask batch reducer for autocallables (spec §17.3)"
```

---

### Task 3: Capability matrix — generator, document, freshness gate

**Files:**
- Create: `quantark/execution/capability_matrix.py`
- Create: `docs/execution/capability-matrix.md` (generated; `git add -f`)
- Modify: `quantark/execution/__init__.py` (export `render_capability_matrix`)
- Test: `test/execution/test_capability_matrix.py`

**Interfaces:**
- Produces: `render_capability_matrix() -> str` — pure function over
  `ENGINE_INVENTORY` (no asset imports, mirroring inventory.py's constraint), and a
  CLI entry `python -m quantark.execution.capability_matrix [output_path]` that
  writes the file (default stdout).

- [ ] **Step 1: Write the generator.** Markdown document with:
  - Header: title, "GENERATED by `python -m quantark.execution.capability_matrix`
    — do not edit by hand; CI enforces freshness", schema/date-free (deterministic:
    NO timestamps, so regeneration is reproducible).
  - Summary counts by `adoption_state`, `batch_state`, `adaptive_state`,
    `prepared_state`.
  - One table per `asset_family`+`engine_type` group with columns:
    Engine | Model | Products | Planning | Adoption | Batch | Adaptive | Prepared |
    Validation profile. Rationale strings rendered as footnotes below each table
    (deduplicated) so tables stay readable.
  - A "Session-level capabilities" section stating scenario/process/Dask is
    session-level (quote the inventory module docstring's Phase 5 note).
- [ ] **Step 2: Generate `docs/execution/capability-matrix.md`** via the CLI.
- [ ] **Step 3: Freshness + content tests** in `test_capability_matrix.py`:
  - `test_checked_in_matrix_is_fresh`: `render_capability_matrix()` == the file's
    text exactly (read with `encoding="utf-8"`).
  - `test_matrix_covers_every_inventory_row`: every `InventoryRecord.name` appears.
  - `test_matrix_is_deterministic`: two calls return identical strings.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest test/execution/test_capability_matrix.py -v` → 3 passed.
- [ ] **Step 5: Commit** (`git add -f docs/execution/capability-matrix.md` plus the rest):
  `docs(execution): generated capability matrix with CI freshness gate`

---

### Task 4: Reproducibility schemas + validation tests

**Files:**
- Create: `docs/execution/schemas/worker-spec.v1.schema.json`
- Create: `docs/execution/schemas/scenario-cell.v1.schema.json`
- Create: `docs/execution/schemas/execution-manifest.v0.schema.json`
- Create: `docs/execution/schemas/normalized-economics.v1.schema.json`
- Modify: `pyproject.toml` (dev extras += `"jsonschema>=4.0"`)
- Test: `test/execution/test_reproducibility_schemas.py`

**Interfaces:**
- Consumes: `worker_spec_to_payload` / `_cell_payload` (scenario/worker.py:160/:335),
  `ReproducibilityManifest` + `MANIFEST_SCHEMA_VERSION = "execution-manifest/0"`
  (manifest.py), `normalized_cell_payload` (scenario/validate.py:50).
- Produces: four Draft 2020-12 JSON Schemas, each with `additionalProperties: false`
  at the top level, `required` listing every field, and a `const` pin on the
  embedded schema-version string (`"scenario/v1"`, `"execution-manifest/0"`).

- [ ] **Step 1: `pip install` the new dev extra** into `.venv`
  (`.venv/bin/pip install "jsonschema>=4.0"`) and add it to `pyproject.toml` dev
  extras.
- [ ] **Step 2: Author the four schemas** — derived from the EMITTED payloads, not
  from this plan's memory (Codex plan-gate finding 3: guessed fields are wrong):
  - worker-spec: `schema_version` (const scenario/v1), `base_ref` {factory_id:
    string, payload: pairs-array}, `callable_refs`: array of 5-element
    [kind, ref_id, module, qualname, schema_version] string arrays,
    `child_policy_values`/`child_budget_values`/`expected`: pairs arrays,
    `import_paths`: string array. Define a shared `$defs/pairs` (array of
    2-element arrays whose first element is a string; values recursive
    scalar/pairs/array — match `_pairs_to_lists` output).
  - scenario-cell: EXACTLY the keys `_cell_payload` (scenario/worker.py:335)
    emits: `scenario_id` (string), `position` (integer), `transformer_id`
    (string), `runner_id` (string), `parameters` (pairs), `invalidate_all`
    (boolean), `cell_fingerprint` (string or null). NOTE: the cell payload
    carries NO embedded schema_version by design — it never travels alone; every
    `run_worker_cell(spec_payload, cell_payload, ...)` call pairs it with the
    WorkerSpec payload whose `scenario/v1` version gates execution first. State
    this in the schema's `description` so the versioning story (spec §22) is
    explicit rather than implied.
  - execution-manifest: the 10 `ReproducibilityManifest` fields; versions and
    resolved_policy as pairs arrays; nullable fingerprints
    (`"type": ["string", "null"]`).
  - normalized-economics: object requiring `value.native` with the FULL leaf
    union the normalizer can emit — `["number", "string", "boolean", "null"]`
    (non-float natives normalize to fingerprints/strings; economics fields can
    be booleans and strings, not just numbers). All other properties are
    scalar-or-null leaves keyed by dotted paths (`patternProperties` on `.*`
    with the same union, `additionalProperties` false via the pattern).
- [ ] **Step 3: Validation tests** in `test_reproducibility_schemas.py`
  (`pytest.importorskip("jsonschema")`):
  - Build a REAL WorkerSpec via the toy fixtures in
    `test/execution/scenario_process_helpers.py` (same construction as
    `test_worker_spec.py`), real cell payloads from a planned toy scenario, a real
    manifest from a `PricingSession.price` outcome (see how existing tests obtain
    outcomes/manifests, e.g. test_session_parity), and real normalized economics
    payloads from `normalized_cell_payload` — one from a float-value runner AND
    one from a `request/v1` native-value outcome (non-numeric `value.native`),
    plus at least one economics payload containing a boolean and a string leaf.
  - `jsonschema.validate(payload, schema)` passes for all of the above.
  - Negative tests: deleting a required key AND adding an unknown top-level key
    each raise `ValidationError` (schema strictness both directions).
  - Schema-version pin tests: mutating the worker-spec/manifest `schema_version`
    fails validation (the `const`), mirroring the reader-side rejection contract
    (spec §22) — and cross-check the reader side by asserting
    `verify_worker_environment` raises `CapabilityError` on the same mutated
    spec, so schema and reader stay in lockstep.
- [ ] **Step 4: Run** the new test file → all pass.
- [ ] **Step 5: Commit** (`git add -f docs/execution/schemas/*.json` + rest):
  `docs(execution): reproducibility JSON Schemas validated against live payloads`

---

### Task 5: Policy guide, internals rationale, docs index

**Files:**
- Create: `docs/execution/policy-guide.md`
- Create: `docs/execution/internals-and-legacy.md`
- Create: `docs/execution/README.md`
- Modify: `CLAUDE.md` (Reference Documentation list: add `docs/execution/`)

All hand-written; content requirements (every bullet must appear):

- [ ] **Step 1: `policy-guide.md`** — (a) the §17.2 precedence list verbatim
  (5 levels, field-by-field, resolve-once-in-parent); (b) env-var table:
  `QUANTARK_EXEC_BATCH_BACKEND/BATCH_WORKERS/SCENARIO_BACKEND/SCENARIO_WORKERS/`
  `MEMORY_MB/CACHE_MB/MAX_IN_FLIGHT/MAX_PROCESSES` + legacy
  `QUANTARK_DCN_MC_WORKERS` (wins for DCN absent explicit setting) +
  `QUANTARK_QMC_CACHE_MB` (requested QMC ceiling under parent budget) — with
  resolution timing (session vs engine-construction vs import); (c)
  `ExecutionPolicy`/`ResourceBudget` field reference incl. `retries`, `fail_fast`,
  `collect_errors`, auto-budget upgrade rule (default-sourced values only); (d)
  backend×capability matrix (serial/threads/processes/dask vs price/price_many/
  batch/adaptive/prepared/scenario) noting `value_kind="float"` requirement for
  process/dask and adaptive = serial-only; (e) nested execution off by default,
  children inner-serial with divided budgets.
- [ ] **Step 2: `internals-and-legacy.md`** — the §17.3 story: what was unified
  (Task 2's shared Dask reducer; the Phase 3 single RQMC loop; the shared Dupire
  prep helper) and the keep-with-rationale table (DCN thread loop, dual draw-cache
  scopes, admission scopes — reasons from the audit, citing §17.1 surfaces that
  must survive: `use_dask`, `num_workers`, `QUANTARK_DCN_MC_WORKERS`, DCN
  `_prepare_simulation`/`_resolve_surface` hooks pinned by the frozen 0.2.6
  downstream). Mark every kept duplicate explicitly TEMPORARY with its removal
  precondition ("removable only once this route uses the kernel's plan/reducer,
  under a separate deprecation spec"). State explicitly: Task 2's consolidation
  is intra-legacy, not kernel convergence; no v1 deprecation warnings; any
  removal needs a separate spec (§17.3).
- [ ] **Step 3: `README.md`** — index: what the framework is (one paragraph),
  quick-start snippet (`PricingSession().price(...)`), links to the matrix, policy
  guide, schemas, internals doc, the spec, and the migration examples; a
  "Performance snapshots" section that Task 8 fills in (leave the section header
  with `<!-- filled by Task 8 -->`).
- [ ] **Step 4: Commit** (`git add -f docs/execution/*.md`):
  `docs(execution): policy guide, legacy-internals rationale, docs index`

---

### Task 6: Migration examples

**Files:**
- Create: `example/execution_session_demo.py`
- Create: `example/execution_scenarios_demo.py`
- Test: `test/execution/test_examples.py`

- [ ] **Step 1: `execution_session_demo.py`** — runnable in <60s, prints as it goes:
  (1) direct `engine.price()` vs `PricingSession().price()` equality on a European
  MC engine; (2) `price_many` over a small book; (3) a threads-backend batch run on
  a DCN engine showing bit-identical PV vs serial; (4) reading the outcome's
  manifest + diagnostics. Use small `num_paths` (e.g. 20_000).
- [ ] **Step 2: `execution_scenarios_demo.py`** — migrating a worker-globals-style
  scenario sweep to typed scenarios: register a factory/transformer/runner
  (module-level, importable — this file doubles as the spawn-import module), run
  serial then `backend="processes"` with 2 workers, compare via
  `compare_scenario_outcomes`, print the report counts. Guard the executable part
  with `if __name__ == "__main__":` (spawn re-imports the module).
- [ ] **Step 3: smoke test** `test_examples.py`: run each script via
  `subprocess.run([sys.executable, path], ...)` asserting returncode 0
  (env: `PYTHONPATH` includes repo root so the worktree source is used); mark the
  scenarios demo `pytest.importorskip("dask")`-free (it uses processes, not dask)
  and give both a `@pytest.mark.timeout`-free generous design (fast params).
- [ ] **Step 4: Run** the smoke tests → 2 passed. Also run both demos manually once
  and eyeball the output.
- [ ] **Step 5: Commit:** `docs(execution): runnable migration examples + smoke tests`

---

### Task 7: Release prep — version 0.3.0, CHANGELOG, wheel

**Files:**
- Modify: `pyproject.toml` (version `0.2.6` → `0.3.0`)
- Modify: `CHANGELOG.md`
- Modify: `quantark/execution/__init__.py` + `quantark/execution/inventory.py`
  docstrings ("Phases 0-6"; inventory `_MILESTONE` stays — check nothing asserts on
  it before touching)

- [ ] **Step 1: CHANGELOG.** Derive content from
  `git log --oneline --no-merges v0.2.5..HEAD` grouped by prefix. Structure:
  - Backfill stubs for `[0.2.3]`, `[0.2.4]`, `[0.2.5]` (one-two lines each from
    their bump commits' surrounding history; check
    `git log v0.2.2..v0.2.3 --oneline` etc. for the headline items).
  - `[0.3.0] - 2026-07-18` with sections:
    **Added** — `quantark.execution` framework (session/kernel/adapters, batch
    backends serial/threads, adaptive RQMC session mode, PDE prepared artifacts,
    typed scenarios + spawn processes + Dask backend, capability inventory +
    generated matrix, reproducibility manifests/schemas, policy resolution + env
    aliases, migration examples); DCN product + MC/PDE/vol engines + SVI/curve
    layers; implied futures carry (IndexFuturesCurve, futures-tenor buckets);
    volmodels improvement program phases 4-5 (Krylov/TR-BDF2 opt-in, QE-M,
    concentrated grids, LV Rannacher); PDE event-stats API; equity TRS risk-stack
    integration; SA-CVA module. **Changed** — DCN engine speedups (draw cache,
    batch threads, LV build hoist); ADI unification + Craig–Sneyd corrector fix;
    vectorized Lewis Heston calibration. **Fixed** — PDE/QUAD audit fixes (13
    bugs); barrier-smoothing and KO-reset fixes — verify each bullet against the
    actual log; drop anything that landed before v0.2.5.
  - Honesty rule (spec §22): the 0.3.0 entry must state per-capability adoption is
    inventory-driven (`temporary_legacy` rows remain) and scaling gates ≥2x/≥2.5x
    are documented as controlled-host-deferred — no universal-support claim.
  - **Pre-tag checklist** (Codex plan-gate finding 4: deferred gates must be
    structural blockers on tagging, not prose): this phase produces
    RELEASE-PREPARATION evidence only — no tag is created. Add a
    "Before tagging v0.3.0" checklist to `docs/execution/README.md` (and
    reference it from the CHANGELOG entry) listing the outstanding hard
    prerequisites: (1) controlled-host ≥2x batch / ≥2x PDE CRN / ≥2.5x scenario
    gates (spec §20) measured and passing; (2) the pre-existing
    `test_snowball_quad_flat_identity_golden` failure resolved or explicitly
    quarantined with a written rationale (it predates this program and
    reproduces on unmodified main — it is not framework acceptance debt, but a
    tag must not ship a red suite silently); (3) the wheel-artifact compat run
    (Task 8 Step 1) green at the tagged commit. The tag-triggered
    `release.yml` means pushing a tag IS publishing — the checklist is the gate.
- [ ] **Step 2: Bump version** in pyproject.toml to `0.3.0`.
- [ ] **Step 3: Build + install-check the wheel from a CLEAN candidate tree**
  (Codex plan-gate finding 1: gates must target the artifact, not a dirty source
  tree). This step runs AFTER all other Phase 6 commits, so the wheel is built at
  the exact candidate commit:

```bash
cd <worktree>
git status --porcelain            # MUST be empty — abort the step if not
/Users/fuxinyao/quant-ark/.venv/bin/python -m build --wheel --outdir dist/
shasum -a 256 dist/quantark-0.3.0-*.whl   # record hash in the release notes
python3 -m venv /tmp/qa-wheel-check && /tmp/qa-wheel-check/bin/pip install dist/quantark-0.3.0-*.whl
/tmp/qa-wheel-check/bin/python -c "import quantark; from quantark.execution import PricingSession; print(quantark.__version__)"
/tmp/qa-wheel-check/bin/python -c "import asset"   # compat .pth shim must be installed by the wheel
```

Expected: prints `0.3.0`; the flat-import check emits a `DeprecationWarning` but
imports (proves `quantark_compat.pth` ships in the wheel — a PYTHONPATH shadow
never validates this). Record the wheel sha256 + candidate commit hash in the
release notes. (`/tmp/qa-wheel-check` is throwaway; `dist/` stays untracked.)
- [ ] **Step 4: Run** `test/execution/test_inventory.py` + any version-asserting
  tests (`grep -rn "0\.2\.6\|__version__" test/ quantark/ --include="*.py"` first;
  update stale assertions).
- [ ] **Step 5: Commit:** `chore(release): prepare 0.3.0 — changelog + version bump`

---

### Task 8: Exit-gate verification — benchmarks, compat rerun, full suites

**Files:**
- Modify: `docs/execution/README.md` (fill "Performance snapshots")
- No other source changes expected.

- [ ] **Step 1: Kick off the otc-price-adapter compat rerun in the BACKGROUND
  against the INSTALLED 0.3.0 wheel** (Codex plan-gate finding 1: the adapter
  must exercise the exact artifact — metadata, package layout, compat shim — not
  a PYTHONPATH source shadow). Requires Task 7 Step 3's wheel. It takes ~2h55m —
  start it first, collect it last:

```bash
python3 -m venv /tmp/qa-adapter-compat
/tmp/qa-adapter-compat/bin/pip install <worktree>/dist/quantark-0.3.0-*.whl \
    pandas numpy openpyxl tabulate pytest pytest-xdist
cd /Users/fuxinyao/otc-price-adapter && /tmp/qa-adapter-compat/bin/python -m pytest -q
```

(The adapter's `quantark==0.2.5` pin is deliberately overridden by installing
the candidate wheel directly instead of the adapter package; its
`pythonpath = ["."]` pytest config resolves the local adapter modules.)
Expected (when collected before merge): `54 passed`. Baselines already recorded
2026-07-17/18: 54/54 on the pinned 0.2.5 wheel, 54/54 on main source.
- [ ] **Step 2: Benchmark reruns** on this machine, recording numbers:
  `benchmark_phase2.py`, `benchmark_phase4.py`, `benchmark_phase5.py` (run each via
  `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python test/execution/benchmark_<n>.py`).
  Fill the README "Performance snapshots" section with: measured numbers + date +
  host caveat + the explicit controlled-host deferral for the ≥2x (Phase 2 threads),
  ≥2x (Phase 4 CRN production gate), ≥2.5x (Phase 5 scenario) targets, citing spec
  §20. Do NOT tune anything based on these runs; they are documentation.
- [ ] **Step 3: Execution suite** — `PYTHONPATH=$PWD ... -m pytest test/execution -q`
  → everything passes (450+ tests incl. the new files).
- [ ] **Step 4: Full worktree suite** — `PYTHONPATH=$PWD ... -m pytest -q` →
  only the known quad golden failure.
- [ ] **Step 5: Commit:** `docs(execution): Phase 6 exit-gate snapshot (benchmarks, gates)`
- [ ] **Step 6: Before merge (Stage 7):** collect Step 1's background run — merge is
  BLOCKED unless it reports 54 passed.

---

## Plan-Gate Findings Applied (Codex, 1 iteration)

1. **Release gates must target the artifact** — Task 7 Step 3 now builds from a
   verified-clean candidate tree, records the wheel sha256 + commit, and validates
   the installed layout (incl. the `quantark_compat.pth` shim); Task 8 Step 1 runs
   the otc-price-adapter suite against the INSTALLED wheel in an isolated venv
   (pin overridden by direct wheel install), not a PYTHONPATH shadow.
2. **No kernel-convergence claim** — Task 2 is INTRA-legacy consolidation (three
   copies of one legacy loop → one legacy helper), NOT the §17.3 legacy-vs-kernel
   duplicate removal. The plan, `internals-and-legacy.md`, and the CHANGELOG must
   say exactly that: legacy-vs-kernel duplicates (DCN thread loop, legacy Dask
   route vs session backends) REMAIN by design, marked temporary, removable only
   under a future deprecation spec once both routes share the kernel's plan and
   reducer. Phase 6's §21 removal clause is satisfied by removing nothing that
   requires kernel routing — the kickoff decision (parity-gated unification, §17.1
   + frozen 0.2.6 downstream pins) makes the full facade migration out of scope.
   [Architectural trade-off accepted at kickoff; documented, not actioned.]
3. **Schemas derived from emitted payloads** — Task 4 now pins the scenario-cell
   schema to `_cell_payload`'s literal 7 keys (no invented footprint fields),
   documents the no-embedded-version-by-design envelope story, widens
   `value.native`/economics leaves to the full number/string/boolean/null union,
   and adds native-value, boolean/string-leaf, and reader-lockstep test cases.
4. **Deferred gates become structural pre-tag blockers** — Task 7 adds the
   "Before tagging v0.3.0" checklist (controlled-host gates, quad-golden
   resolution, wheel-compat run) referenced from the CHANGELOG; the phase output
   is framed as release-preparation/RC evidence, never release-ready.

## Self-review notes

- Spec Phase 6 line 1 ("remove duplicate scaffolding only after legacy entry points
  use the kernel"): per plan-gate finding 2, Task 2 is intra-legacy consolidation
  gated by bitwise goldens (Task 1); NO legacy-vs-kernel duplicate is removed and
  no kernel-convergence is claimed. Kernel-routing of `use_dask`/DCN direct paths
  is deliberately NOT done: §17.1 + the frozen 0.2.6 downstream pin make that a
  separate-spec deprecation concern; `internals-and-legacy.md` documents exactly
  this (Task 5).
- Spec Phase 6 exit "no exported engine missing / every advertised capability has
  tests": already CI-enforced by `test_inventory.py` discovery gates + the per-phase
  gates; Task 3 adds the matrix freshness gate on top; Task 8 re-runs everything.
- All four §21 deliverables land: matrix (T3), schemas (T4), policy guide (T5),
  migration examples (T6). §22 versioning honesty lands in T7. The extra user
  requirement (otc-price-adapter) is T8 Step 1/6 with baseline already recorded.
- No placeholders: every doc task lists its exact required content; code tasks carry
  the real extracted signatures. Two verify-against-source instructions remain by
  design (scenario-cell payload keys, changelog bullets) — they instruct reading the
  authoritative source rather than trusting this plan's memory.
