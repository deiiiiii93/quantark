# `quantark.modelvalidation` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine-release certification module: stochastic RQMC benchmarks certify deterministic PDE/QUAD engines through a framework-owned pipeline with banked evidence, amendments, and CI anchors.

**Architecture:** One typed core object (`CertificationStudy`) produced by two front doors (YAML + builder registry, or Python directly); a framework-owned pipeline (reference → candidate → gates → decisions → evidence → report → anchors) that no study can customize away. All gate math is pure-data; all pricing goes through injected builders.

**Tech Stack:** Python 3 dataclasses/Protocols, PyYAML (already a dependency: `pyyaml>=6.0.0` in pyproject.toml), pytest, existing quantark engines (`SnowballMCEngine`, `SnowballPDESolver`, `SnowballQuadEngine`, `BlackScholesEngine`).

**Spec:** `docs/superpowers/specs/2026-08-14-modelvalidation-design.md` — read it first; this plan implements it section by section.

## Global Constraints

- All new library code under `quantark/modelvalidation/`; canonical `quantark.*` imports only (no flat legacy imports).
- Use `quantark.util.numerical` (`is_zero`, `is_close`, `safe_divide`, `validate_positive`) — never raw float compares or hardcoded tolerances.
- Exceptions: reuse `quantark.util.exceptions` (`ValidationError`, `NumericalError`, `PricingError`, `QuantArkException`) — no new exception tree.
- Evidence schema starts at `SCHEMA_VERSION = 1`; the sha256 covers a *projection* excluding volatile fields.
- All output writes are atomic (write-temp-then-rename) and never under `/tmp`.
- Frozen dataclasses for all config/spec types; engines/builders are the only stateful parts.
- Tests live in `test/modelvalidation/`, named `test_<module>.py`. Run serially-safe (no shared global state between tests; use `tmp_path`).
- Run tests from the worktree with the shadowed path so the worktree source is tested, per repo convention:
  `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/modelvalidation -n0 -q`
- Commit after every task with a conventional-commit message ending in the Co-Authored-By line:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

**Two deliberate spec-layout amendments** (record here so the spec and code don't silently diverge):
1. Orchestration lives in `pipeline.py` (spec §11 lists no home for `certify`/`amend` bodies; `__init__.py` only re-exports).
2. The reference arm is banked **per case** (one paired batch set yields all quantities at once), not per cell; gates still operate per cell = case × quantity. The reference identity hash includes the quantity list, sampling policy, and bump width, and still excludes candidates so one bank serves PDE and QUAD.

## File Structure

```
quantark/modelvalidation/
├── __init__.py          re-exports: certify, amend, load_study, CertificationStudy, register_builder
├── study.py             CaseSpec, GateBounds, SamplingPolicy, EconomicScale, HedgeContractScale, CertificationStudy
├── registry.py          register_builder / get_builder / list_builders
├── gates.py             evaluate_cell_gate, evaluate_aggregate_gate (pure data)
├── decisions.py         Verdict, Decision, decide_cell, decide_candidate
├── stopping.py          StopDecision, should_stop (gate-driven)
├── evidence.py          canonical_json, evidence_projection, projected_sha256, identity_hash,
│                        atomic_write_json, CheckpointStore
├── reference.py         BatchResult, ReferenceEstimate, ReferenceBuilder protocol, run_reference
├── candidate.py         LadderRung, CandidateResult, CandidateEvaluator protocol
├── pipeline.py          certify(), Certificate (amend() added in amendment task)
├── report.py            render_markdown
├── anchors.py           extract_anchors, assert_anchors, machine_fingerprint
├── amendment.py         validate_parent, amend()
├── yaml_loader.py       load_study (YAML → registry → CertificationStudy)
├── cli.py               main(argv) for {run, amend, list, anchors}
├── __main__.py          python -m quantark.modelvalidation
└── builders/
    ├── __init__.py      importing this package registers all builtin builders
    ├── european_selftest.py   flat-BSM European: analytical candidate + tiny-MC reference
    └── equity_snowball.py     snowball flat-BSM: MC reference + PDE/QUAD candidates

example/modelvalidation/
├── european_selftest.yaml     the CI self-test study (also used by integration tests)
├── snowball_flat_bsm.yaml     the demo/acceptance study
└── README.md                  how to run quick/full, where evidence lands

test/modelvalidation/          one test file per module (see tasks)
docs/modelvalidation/RELEASE_PROCEDURE.md
```

Engine APIs used (verified in this worktree):
- `SnowballMCEngine(params: MCParams, method=MonteCarloMethod.RANDOMIZED_QUASI).price(product, env)`; `MCParams(seed=…, num_paths=…, rqmc_min_batches=1, rqmc_max_batches=1, rqmc_paths_mode="per_batch")` makes one call = one batch.
- `SnowballPDESolver(params: PDEParams).calculate_greeks(product, env) -> {"price","delta","gamma"}`.
- `SnowballQuadEngine(params: QuadParams).price(product, env) -> float`.
- `BlackScholesEngine().price(product, env)` for the self-test candidate.
- `PricingEnvironment(spot_quote=SpotQuote(spot=…), vol_surface=FlatVolSurface(volatility=…), rate_curve=FlatRateCurve(rate=…), div_yield=ContinuousDividendYield(div_yield=…), valuation_date=datetime(…))`.
- Snowball product: `SnowballOption` + `BarrierConfig`/`PayoffConfig`/`AccrualConfig` from `quantark.asset.equity.product.option.snowball_config` (copy the construction pattern from `example/snowball_mc_demo.py`).

---

### Task 1: Core types (`study.py`)

**Files:**
- Create: `quantark/modelvalidation/__init__.py`, `quantark/modelvalidation/study.py`
- Test: `test/modelvalidation/test_study.py` (and empty `test/modelvalidation/__init__.py` if the repo pattern uses one — check `test/`; most test dirs have none, so skip it)

**Interfaces:**
- Produces (all frozen dataclasses unless noted):
  - `CaseSpec(name: str, environment_params: Mapping[str, Any] = {}, product_params: Mapping[str, Any] = {})` — mappings stored as immutable `MappingProxyType` via `__post_init__` conversion is NOT needed; store plain dicts but never mutate (convention used repo-wide).
  - `GateBounds(cell: float, mean_signed_bias: float, se_budget_fraction: float = 0.25, interval_k: float = 2.0, envelope_fraction: float = 0.5)` — validates all positive.
  - `SamplingPolicy(paths_per_batch: int, min_batches: int, max_batches: int, seed: int, bump: float = 0.01)` — `bump` is the *relative* spot bump for CRN greeks; validates `min_batches >= 2` (SE needs ddof=1), `max_batches >= min_batches`.
  - `class EconomicScale(Protocol): def to_economic(self, quantity: str, raw: float) -> float`
  - `HedgeContractScale(hedge_multiplier: float, hedge_inception_spot: float, notional: float)` with property `delta_quantum = hedge_multiplier * hedge_inception_spot / notional` and conversions (this resolves the spec §16 open item — these formulas are now the written definition):
    - `delta`: `raw / delta_quantum`
    - `gamma`: `raw * (0.01 * hedge_inception_spot) / delta_quantum` (contracts of delta drift per 1% spot move)
    - `pv`: `raw / (delta_quantum * 0.01 * hedge_inception_spot)` (PV error equated to the P&L c contracts make on a 1% move)
  - `CertificationStudy(name, schema: int, cases: tuple[CaseSpec, ...], quantities: tuple[str, ...], bounds: GateBounds, scale: EconomicScale, reference: "ReferenceBuilder", candidates: tuple["CandidateEvaluator", ...], sampling: SamplingPolicy, source_text: str | None = None)` — validates `schema == 1`, quantities ⊆ `{"pv","delta","gamma"}`, non-empty cases/candidates, unique case names.
  - `QUANTITIES = ("pv", "delta", "gamma")` module constant.

- [ ] **Step 1: Write failing tests**

```python
# test/modelvalidation/test_study.py
import pytest
from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.study import (
    CaseSpec, GateBounds, SamplingPolicy, HedgeContractScale,
)

def test_hedge_contract_scale_formulas():
    scale = HedgeContractScale(hedge_multiplier=200.0,
                               hedge_inception_spot=100.0,
                               notional=50_000_000.0)
    dq = 200.0 * 100.0 / 50_000_000.0            # 4e-4 delta per contract
    assert scale.delta_quantum == pytest.approx(dq)
    assert scale.to_economic("delta", 2 * dq) == pytest.approx(2.0)
    assert scale.to_economic("gamma", 1.0) == pytest.approx(0.01 * 100.0 / dq)
    assert scale.to_economic("pv", dq * 0.01 * 100.0) == pytest.approx(1.0)

def test_gate_bounds_validation():
    with pytest.raises(ValidationError):
        GateBounds(cell=-0.5, mean_signed_bias=0.1)

def test_sampling_policy_validation():
    with pytest.raises(ValidationError):
        SamplingPolicy(paths_per_batch=1024, min_batches=1, max_batches=8, seed=7)
    with pytest.raises(ValidationError):
        SamplingPolicy(paths_per_batch=1024, min_batches=4, max_batches=2, seed=7)

def test_case_spec_defaults():
    c = CaseSpec(name="ordinary")
    assert c.environment_params == {} and c.product_params == {}
```

Also a `CertificationStudy` validation test — construct with stub reference/candidates (plain objects, the study does not type-check them in v1) and assert `ValidationError` on `schema=2`, on duplicate case names, and on `quantities=("vega",)`.

- [ ] **Step 2: Run** `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/modelvalidation/test_study.py -n0 -q` — expect FAIL (module missing).

- [ ] **Step 3: Implement `study.py`** — frozen dataclasses with `__post_init__` validation raising `ValidationError`; `HedgeContractScale.to_economic` dispatches on quantity name and raises `ValidationError` for unknown quantities. `__init__.py` starts as just `from quantark.modelvalidation.study import CertificationStudy  # noqa: F401`.

- [ ] **Step 4: Run tests — PASS.**

- [ ] **Step 5: Commit** `feat(modelvalidation): core study types and hedge-contract economic scale`

### Task 2: Builder registry (`registry.py`)

**Files:**
- Create: `quantark/modelvalidation/registry.py`
- Test: `test/modelvalidation/test_registry.py`

**Interfaces:**
- Produces:
  - `register_builder(name: str, kind: str)` — decorator; `kind` ∈ `{"product","environment","reference","candidate","economic_scale"}`, else `ValidationError` at decoration time. Duplicate `(kind, name)` → `ValidationError`.
  - `get_builder(name: str, kind: str) -> Callable` — unknown name → `ValidationError` whose message contains the kind and the sorted registered names for that kind.
  - `list_builders() -> dict[str, tuple[str, ...]]` — kind → sorted names.
  - `clear_registry()` — **test-only** helper (documented as such) so tests can isolate; production code never calls it.

- [ ] **Step 1: Failing tests** — register a dummy builder, fetch it, assert unknown-name error message lists registered names, assert duplicate registration raises, assert bad kind raises. Use a `pytest.fixture(autouse=True)` that snapshots and restores the registry dict around each test.

```python
def test_unknown_builder_lists_available():
    @register_builder("flat_bsm_x", kind="environment")
    def make_env(params): return {"env": params}
    with pytest.raises(ValidationError) as exc:
        get_builder("nope", kind="environment")
    assert "flat_bsm_x" in str(exc.value) and "environment" in str(exc.value)
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — module-level `_REGISTRY: dict[tuple[str, str], Callable]`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): per-family builder registry`

### Task 3: Gate arithmetic (`gates.py`)

**Files:**
- Create: `quantark/modelvalidation/gates.py`
- Test: `test/modelvalidation/test_gates.py`

**Interfaces:**
- Consumes: `GateBounds`, `EconomicScale` from `study.py`.
- Produces (frozen dataclasses):
  - `CellGateResult(signed_err_c: float, se_c: float, interval_c: float, se_budget_met: bool, interval_within_bound: bool, envelope_c: float | None, envelope_within_bound: bool, passed: bool)`
  - `evaluate_cell_gate(candidate_raw: float, reference_raw: float, reference_se_raw: float, quantity: str, scale: EconomicScale, bounds: GateBounds, envelope_raw: float | None = None) -> CellGateResult` — everything is converted with `scale.to_economic(quantity, …)` (errors and SEs convert linearly because `to_economic` is linear per quantity); `interval_c = abs(signed_err_c) + bounds.interval_k * se_c`; `se_budget_met = se_c <= bounds.se_budget_fraction * bounds.cell`; `envelope_within_bound = envelope_c is None or envelope_c <= bounds.envelope_fraction * bounds.cell`; `passed = se_budget_met and interval_within_bound and envelope_within_bound`.
  - `AggregateGateResult(mean_signed_bias_c: float, se_of_mean_c: float, within_bound: bool, se_adequate: bool, passed: bool)`
  - `evaluate_aggregate_gate(signed_errs_c: Sequence[float], ses_c: Sequence[float], bounds: GateBounds) -> AggregateGateResult` — `mean_signed_bias_c = mean(signed_errs_c)`; `se_of_mean_c = sqrt(sum(se**2 for se in ses_c)) / len(ses_c)`; `within_bound = abs(mean) + bounds.interval_k * se_of_mean <= bounds.mean_signed_bias`; `se_adequate = se_of_mean_c <= bounds.se_budget_fraction * bounds.mean_signed_bias`; `passed = within_bound and se_adequate`. Empty input → `ValidationError`.

- [ ] **Step 1: Failing tests** — use `HedgeContractScale(200, 100, 50e6)` so conversions are non-trivial:

```python
def _scale():
    return HedgeContractScale(200.0, 100.0, 50_000_000.0)

def test_cell_gate_pass_and_fail():
    b = GateBounds(cell=0.5, mean_signed_bias=0.1)
    dq = _scale().delta_quantum
    # candidate off by 0.2 contracts of delta, SE 0.05 contracts -> interval 0.3 <= 0.5
    r = evaluate_cell_gate(candidate_raw=0.2 * dq, reference_raw=0.0,
                           reference_se_raw=0.05 * dq, quantity="delta",
                           scale=_scale(), bounds=b)
    assert r.passed and r.signed_err_c == pytest.approx(0.2)
    # SE over budget (0.2 > 0.25*0.5) -> not passed even though interval fits
    r2 = evaluate_cell_gate(0.0, 0.0, 0.2 * dq, "delta", _scale(), b)
    assert not r2.se_budget_met and not r2.passed

def test_cell_gate_envelope():
    b = GateBounds(cell=0.5, mean_signed_bias=0.1, envelope_fraction=0.5)
    dq = _scale().delta_quantum
    r = evaluate_cell_gate(0.0, 0.0, 0.01 * dq, "delta", _scale(), b,
                           envelope_raw=0.3 * dq)   # 0.3 > 0.25 -> violation
    assert not r.envelope_within_bound and not r.passed

def test_aggregate_gate():
    b = GateBounds(cell=0.5, mean_signed_bias=0.1)
    r = evaluate_aggregate_gate([0.02, -0.01, 0.03], [0.005, 0.005, 0.005], b)
    assert r.passed
    r2 = evaluate_aggregate_gate([0.2, 0.2], [0.001, 0.001], b)
    assert not r2.within_bound and not r2.passed
```

- [ ] **Step 2: Run — FAIL.**  
- [ ] **Step 3: Implement `gates.py`** — pure functions, no numpy needed beyond `math`; comparisons via plain arithmetic (bounds are user config, not float-noise territory; do NOT wrap in `is_close`).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): pure cell and aggregate gate arithmetic`

### Task 4: Verdicts and decisions (`decisions.py`)

**Files:**
- Create: `quantark/modelvalidation/decisions.py`
- Test: `test/modelvalidation/test_decisions.py`

**Interfaces:**
- Consumes: `CellGateResult`, `AggregateGateResult` from `gates.py`.
- Produces:
  - `class Verdict(str, Enum): PASS, FAIL, ERROR, UNRESOLVED` and `class Decision(str, Enum): ADMITTED, REJECTED, INCONCLUSIVE` (str-Enums so they JSON-serialize as their value).
  - `decide_cell(gate: CellGateResult | None, error: bool) -> Verdict` — `error=True` → `ERROR`; gate with `se_budget_met=False` → `UNRESOLVED`; else `PASS`/`FAIL` by `gate.passed`.
  - `decide_candidate(cell_verdicts: Sequence[Verdict], aggregates: Sequence[AggregateGateResult]) -> Decision` implementing spec §6 exactly:
    - all cells PASS and all aggregates passed → `ADMITTED`
    - any FAIL cell (FAIL implies SE budget was met, so it is a *confident* failure), or any aggregate with `se_adequate and not within_bound` → `REJECTED`
    - otherwise → `INCONCLUSIVE` (covers ERROR, UNRESOLVED, aggregate-SE-inadequate)

- [ ] **Step 1: Failing tests** — table-driven over the decision lattice: all-pass → ADMITTED; one FAIL → REJECTED; one ERROR (rest pass) → INCONCLUSIVE; one UNRESOLVED → INCONCLUSIVE; confident aggregate failure → REJECTED; aggregate SE inadequate → INCONCLUSIVE; ERROR present *and* FAIL present → REJECTED (a confident failure dominates).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): verdict lattice and candidate decisions`

### Task 5: Gate-driven stopping (`stopping.py`)

**Files:**
- Create: `quantark/modelvalidation/stopping.py`
- Test: `test/modelvalidation/test_stopping.py`

**Interfaces:**
- Consumes: `GateBounds`, `SamplingPolicy`, `EconomicScale`.
- Produces:
  - `StopDecision(stop: bool, reason: str, batches: int)` frozen dataclass; `reason` ∈ `{"below_min_batches", "se_budget_met", "max_batches"}`.
  - `should_stop(std_errors_raw: Mapping[str, float], batches: int, scale: EconomicScale, bounds: GateBounds, policy: SamplingPolicy) -> StopDecision` — never stop before `policy.min_batches`; stop with `"se_budget_met"` when **every** quantity's economic SE ≤ `bounds.se_budget_fraction * bounds.cell`; stop with `"max_batches"` at `policy.max_batches` regardless.

- [ ] **Step 1: Failing tests** — three cases: below min (stop=False, reason `below_min_batches`); all SEs under budget at batches ≥ min (stop=True, `se_budget_met`); one SE over budget at `max_batches` (stop=True, `max_batches`). Use `HedgeContractScale` so raw→economic conversion is exercised.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): gate-driven sequential stopping`

### Task 6: Evidence primitives and checkpoints (`evidence.py`)

**Files:**
- Create: `quantark/modelvalidation/evidence.py`
- Test: `test/modelvalidation/test_evidence.py`

**Interfaces:**
- Produces:
  - `SCHEMA_VERSION = 1`
  - `canonical_json(obj) -> str` — `json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)` where `_json_default` handles numpy scalars (`.item()`) and raises `TypeError` otherwise. Floats serialize with full `repr` precision (json default).
  - `VOLATILE_KEYS = frozenset({"wall_clock_seconds", "timestamp", "started_at", "finished_at", "host_load", "projected_sha256"})`
  - `evidence_projection(obj)` — recursive copy of dict/list trees dropping any dict key in `VOLATILE_KEYS`.
  - `projected_sha256(payload: dict) -> str` — sha256 hex of `canonical_json(evidence_projection(payload))`.
  - `identity_hash(mapping: Mapping) -> str` — sha256 hex of `canonical_json(mapping)`.
  - `atomic_write_json(path: Path, payload: dict) -> None` — writes `path.with_suffix(path.suffix + ".tmp")` then `os.replace`. Refuses (ValidationError) any path under `/tmp` or `tempfile.gettempdir()`.
  - `atomic_write_text(path: Path, text: str) -> None` — same rename discipline (for the markdown report).
  - `class CheckpointStore(root: Path)`:
    - `save(kind: str, key: str, identity: Mapping, payload: dict)` → writes `root / kind / f"{key}.json"` with `{"identity": …, "identity_hash": identity_hash(identity), "payload": …}` atomically.
    - `load(kind: str, key: str, identity: Mapping) -> dict | None` — returns `payload` only when the stored `identity_hash` equals `identity_hash(identity)`; a mismatch returns `None` **and** renames the stale file to `*.stale` (never silently reuses, never silently deletes).
    - Keys are filesystem-safe: `key` must match `[A-Za-z0-9._-]+` else `ValidationError`.

- [ ] **Step 1: Failing tests**

```python
def test_projection_drops_volatile_and_hash_stable():
    payload = {"a": 1, "wall_clock_seconds": 3.2,
               "nested": [{"timestamp": "x", "b": 2}]}
    proj = evidence_projection(payload)
    assert "wall_clock_seconds" not in proj and "timestamp" not in proj["nested"][0]
    h1 = projected_sha256(payload)
    payload["wall_clock_seconds"] = 99.0
    assert projected_sha256(payload) == h1          # volatile change -> same hash
    payload["a"] = 2
    assert projected_sha256(payload) != h1          # real change -> new hash

def test_atomic_write_refuses_tmp(tmp_path):
    import tempfile
    bad = Path(tempfile.gettempdir()) / "x.json"
    with pytest.raises(ValidationError):
        atomic_write_json(bad, {})

def test_checkpoint_identity_gate(tmp_path):
    store = CheckpointStore(tmp_path)
    ident = {"case": "ordinary", "seed": 7}
    store.save("reference", "ordinary", ident, {"v": 1})
    assert store.load("reference", "ordinary", ident) == {"v": 1}
    assert store.load("reference", "ordinary", {"case": "ordinary", "seed": 8}) is None
    assert (tmp_path / "reference" / "ordinary.json.stale").exists()
```

(Note: `tmp_path` is pytest's fixture in the repo checkout, not `/tmp` — on macOS it resolves under `/private/var/folders`, so the `/tmp` guard does not trip it. In the guard implementation compare against `tempfile.gettempdir()` and the literal `/tmp` prefix after `Path.resolve()`.)

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `evidence.py`.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): evidence hashing, atomic writes, identity-gated checkpoints`

### Task 7: Reference arm (`reference.py`)

**Files:**
- Create: `quantark/modelvalidation/reference.py`
- Test: `test/modelvalidation/test_reference.py`

**Interfaces:**
- Consumes: `CaseSpec`, `SamplingPolicy`, `EconomicScale`, `GateBounds`, `should_stop`, `CheckpointStore`.
- Produces:
  - `BatchResult(index: int, seed: int, values: Mapping[str, float])` — one paired batch, all quantities at once, raw units.
  - `ReferenceEstimate(values: Mapping[str, float], std_errors: Mapping[str, float], batches: int, seeds: tuple[int, ...], stopped_reason: str)` — `values[q] = mean(batch values)`, `std_errors[q] = std(batch values, ddof=1)/sqrt(B)`.
  - `class ReferenceBuilder(Protocol):`
    - `def identity(self, case: CaseSpec) -> Mapping[str, Any]` — MUST include builder name+params, case name+params, quantities, sampling policy fields, and bump width (spec §7).
    - `def run_batch(self, case: CaseSpec, batch_index: int) -> BatchResult`
  - `run_reference(builder, case, quantities, scale, bounds, policy, store: CheckpointStore | None, resume: bool) -> ReferenceEstimate`:
    - checkpoint key = case.name, kind `"reference"`, identity = `builder.identity(case)`.
    - on `resume` with a valid checkpoint: reload banked `BatchResult`s, **validate the bank against the stopping decision** — replay `should_stop` over the banked batch prefix; if the recorded `stopped_reason`/count could not have been produced by the policy (e.g. more batches than `max_batches`, or fewer than `min_batches`), raise `ValidationError` (spec §5 soundness rule). If the bank stopped early only because of a *previous interrupt* (no stop reason recorded), continue batching from `len(banked)`.
    - loop: `run_batch`, append, recompute estimate, `should_stop(...)`; after every batch, `store.save` the full bank (batches list + current stop state) — durability per batch, not per cell.
    - seeds: `policy.seed + batch_index` (recorded in each BatchResult; the *builder* must use `BatchResult.seed`, `run_reference` just checks returned `seed == policy.seed + index`).

- [ ] **Step 1: Failing tests** — use a deterministic fake builder (no MC): batch value = `mu + noise[index]` from a fixed table, so SE shrinks predictably.

```python
class FakeBuilder:
    def __init__(self, series):        # series: list of dicts per batch
        self.series = series
    def identity(self, case):
        return {"builder": "fake", "case": case.name}
    def run_batch(self, case, i):
        return BatchResult(index=i, seed=100 + i, values=self.series[i])
```

Tests: (a) stops with `se_budget_met` once enough tight batches accumulate; (b) hits `max_batches` on a noisy series and reports `stopped_reason="max_batches"`; (c) resume path: run with `store`, interrupt by truncating (simulate: run once with max_batches=3, then rerun with resume=True and max_batches=6 → `ValidationError` because the bank's recorded policy differs — identity mismatch catches it via sampling-policy fields in identity); (d) wrong seed from builder → `ValidationError`. Fix the fake's seed convention to `policy.seed + i` in passing tests.

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `reference.py`** — statistics via numpy (`np.mean`, `np.std(ddof=1)`); guard `batches >= 2` before computing SE (below that SE is `inf` so stopping can never fire before min_batches anyway).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): banked paired-batch reference arm with sound resume`

### Task 8: Candidate arm (`candidate.py`)

**Files:**
- Create: `quantark/modelvalidation/candidate.py`
- Test: `test/modelvalidation/test_candidate.py`

**Interfaces:**
- Produces:
  - `LadderRung(axis: str, level: str, values: Mapping[str, float])` — `level` ∈ `{"coarse", "medium", "target"}`.
  - `CandidateResult(values: Mapping[str, float], ladders: tuple[LadderRung, ...])`
  - `class CandidateEvaluator(Protocol):`
    - `def name(self) -> str`
    - `def params(self) -> Mapping[str, Any]` — feeds identity/evidence
    - `def evaluate(self, case: CaseSpec) -> CandidateResult` — target-grid values for every study quantity, plus one ladder rung per configured axis at each level.
  - `envelope_from_ladders(ladders: Sequence[LadderRung], quantity: str) -> float | None` — per axis, `|target − medium|`; envelope = **sum over axes** (conservative). Returns `None` when no ladder has both `target` and `medium` rungs for the quantity.
  - `candidate_identity(evaluator: CandidateEvaluator, case: CaseSpec) -> Mapping` — `{"candidate": name, "params": params, "case": …}` for candidate checkpoints.

- [ ] **Step 1: Failing tests** — `envelope_from_ladders` with two axes (`n_x`: target 1.000/medium 1.004; `n_t`: target 1.000/medium 0.998) → envelope `0.006`; missing medium rung → `None`; single axis works.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): candidate evaluator protocol and ladder envelopes`

### Task 9: Pipeline (`pipeline.py`)

**Files:**
- Create: `quantark/modelvalidation/pipeline.py`
- Modify: `quantark/modelvalidation/__init__.py` (re-export `certify`)
- Test: `test/modelvalidation/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces:
  - `Certificate(payload: dict, path: Path)` frozen dataclass.
  - `certify(study: CertificationStudy, out_dir: str | Path, quick: bool = False, resume: bool = False) -> Certificate`.
  - `runtime_environment() -> dict` — `{"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0], "numpy": np.__version__, "quantark_git_sha": <git rev-parse HEAD or None>}` (git via `subprocess.run`, swallow failure → `None`).
  - `quick_policy(policy: SamplingPolicy) -> SamplingPolicy` — deterministic shrink: `paths_per_batch=max(128, policy.paths_per_batch // 8)`, `min_batches=2`, `max_batches=min(4, policy.max_batches)`, same seed/bump.

**Behavior (spec §6, §7, §10):**
1. `out = Path(out_dir) / study.name`; `store = CheckpointStore(out / "checkpoints")`.
2. For each case: `run_reference(...)` (quick mode swaps in `quick_policy(study.sampling)`); an exception from the reference arm marks **every cell of that case** `ERROR` with the traceback string, and processing continues.
3. For each candidate × case: `evaluate(case)` inside try/except (`PricingError`, `NumericalError`, `ValidationError` → cell `ERROR`, traceback recorded, continue). Candidate results are checkpointed too (`kind="candidate"`, key `f"{candidate.name()}-{case.name}"`, identity from `candidate_identity`).
4. For each candidate × case × quantity: `evaluate_cell_gate(...)` with `envelope_from_ladders`; `decide_cell`.
5. Per candidate × quantity: `evaluate_aggregate_gate` over non-ERROR cells (an ERROR cell excludes its case from the aggregate but caps the decision via the verdict lattice); `decide_candidate`.
6. Assemble payload:

```python
payload = {
  "schema": SCHEMA_VERSION,
  "study": {"name": study.name, "source_text": study.source_text,
            "quantities": list(study.quantities),
            "bounds": asdict(study.bounds), "sampling": asdict(sampling_used),
            "quick": quick},
  "runtime": runtime_environment(),
  "references": {case.name: {"estimate": …, "identity_hash": …,
                             "stopped_reason": …, "batches": …, "seeds": […]}},
  "cells": [ {"candidate": …, "case": …, "quantity": …,
              "reference": {"value": …, "se": …},
              "candidate_value": …, "gate": asdict(gate) or None,
              "verdict": verdict.value, "error": traceback_or_None} ],
  "aggregates": [ {"candidate": …, "quantity": …, **asdict(agg), "passed": …} ],
  "decisions": {candidate_name: decision.value},
  "wall_clock_seconds": …,
}
payload["projected_sha256"] = projected_sha256(payload)
```

7. `validate_payload(payload)` (same module): checks schema==1, every cell references a known case/candidate/quantity, every verdict/decision is a valid enum value, and recomputed `projected_sha256` matches — raise `ValidationError` otherwise. Run it **before** writing.
8. `atomic_write_json(out / "certificate.json", payload)`; return `Certificate`.

- [ ] **Step 1: Failing test — full pipeline over fakes** (no pricing): `FakeBuilder` (Task 7 test) as reference; a `FakeCandidate` returning constant values + a tiny ladder; assert: decisions ADMITTED when the fake candidate equals the reference mean; a second candidate off by `2 × cell bound` → REJECTED; certificate file exists; `projected_sha256` recomputes; rerunning `certify(..., resume=True)` reuses checkpoints (assert the fake's `run_batch` call-count did not grow — count calls on the fake).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement `pipeline.py`.** Keep it under ~250 lines: the heavy lifting already lives in Tasks 3–8; this file is a loop and a dict assembly.
- [ ] **Step 4: Run — PASS.** Also run the whole suite: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/modelvalidation -n0 -q`.
- [ ] **Step 5: Commit** `feat(modelvalidation): framework-owned certification pipeline`

### Task 10: Markdown report (`report.py`)

**Files:**
- Create: `quantark/modelvalidation/report.py`
- Modify: `quantark/modelvalidation/pipeline.py` (write `report.md` next to `certificate.json` via `atomic_write_text`)
- Test: `test/modelvalidation/test_report.py`

**Interfaces:**
- Produces: `render_markdown(payload: dict) -> str` — sections: title + study name + date-free header (no timestamps in the body — the report is part of projected evidence), runtime block, per-candidate decision table, per-cell table (candidate, case, quantity, ref±SE, candidate value, err_c, interval_c, verdict), aggregates table, and an "Errors" section listing tracebacks (first line only, full traceback stays in the JSON).

- [ ] **Step 1: Failing test** — feed the Task 9 fake payload; assert the decision line (`| fake-good | ADMITTED |`), a cell row containing the verdict, and that an ERROR traceback's first line appears when present.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement + wire into `certify` (report written after certificate; report render must not mutate payload).**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): markdown certification report`

### Task 11: Anchors (`anchors.py`)

**Files:**
- Create: `quantark/modelvalidation/anchors.py`
- Test: `test/modelvalidation/test_anchors.py`

**Interfaces:**
- Consumes: certificate payload (Task 9 shape), `yaml_loader.load_study` (Task 13 — see note), registry.
- Produces:
  - `machine_fingerprint() -> dict` — `{"machine": platform.machine(), "system": platform.system()}`.
  - `extract_anchors(payload: dict, study: CertificationStudy, rel_tol: float = 1e-12) -> dict` — for every candidate × case with a non-ERROR verdict on every quantity: `{"candidate", "case", "values": {quantity: candidate_value}}`; plus top-level `{"schema": 1, "study_source_text": payload["study"]["source_text"], "fingerprint": machine_fingerprint(), "rel_tol": rel_tol}`. Raise `ValidationError` if the study has no `source_text` (anchors must be re-runnable from YAML alone).
  - `assert_anchors(anchor_path: str | Path) -> None` — loads the file, reconstructs the study via `load_study_text(source_text)`, re-evaluates each anchored candidate × case, compares: exact `==` when `machine_fingerprint()` matches the stored one; else `abs(a-b) <= rel_tol * max(1, abs(a))`. Raises `AssertionError` with a per-value diff table on failure (AssertionError, not ValidationError — this is the pytest-facing API).
  - **Ordering note:** this task lands before the YAML loader; implement `extract_anchors` + `machine_fingerprint` fully, and implement `assert_anchors` against the loader's *interface* (`load_study_text(text) -> CertificationStudy`, defined in Task 13). The unit test for `assert_anchors` uses a monkeypatched `load_study_text` returning a fake study, so the test is loader-independent.

- [ ] **Step 1: Failing tests** — (a) `extract_anchors` from the Task 9 fake payload (give the fake study a `source_text="stub: true"`): anchors contain the good candidate's values, skip ERROR cells; (b) `assert_anchors` round-trip on same machine passes with exact values, and a perturbed value (1e-9 relative) fails; (c) with a *mismatched* stored fingerprint, a 1e-13 perturbation passes under `rel_tol=1e-12` and 1e-9 fails.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): deterministic anchors with cross-arch tolerance policy`

### Task 12: Amendments (`amendment.py`)

**Files:**
- Create: `quantark/modelvalidation/amendment.py`
- Modify: `quantark/modelvalidation/__init__.py` (re-export `amend`)
- Test: `test/modelvalidation/test_amendment.py`

**Interfaces:**
- Consumes: pipeline internals (refactor shared assembly into `pipeline._assemble_and_write(...)` if needed — small, mechanical), evidence primitives.
- Produces:
  - `validate_parent(parent_path: Path) -> dict` — loads JSON, `validate_payload`, recomputed hash must equal stored `projected_sha256`, schema must be 1; returns payload.
  - `amend(study: CertificationStudy, parent: str | Path, out_dir, reason: str, quick=False, resume=False) -> Certificate`:
    - `validate_parent`.
    - Compute the reference identity per case and candidate identity per candidate×case for the *current* study; compare with identities recorded in the parent (add identity hashes to the parent payload in Task 9's `references` and cell entries — `identity_hash` fields; if Task 9 already landed without them, add them now and update its test).
    - Cells whose identities match the parent are **carried forward**: copied into the new payload with `"carried_from": parent_projected_sha256`. Changed cells are re-run (reference re-banked only for changed cases).
    - Carried-forward requires a *matching* identity; if a case exists in the study but not the parent → it is new → run it. If a parent case is absent from the study → `ValidationError` (an amendment cannot silently drop coverage; shrinking scope is a new certification).
    - Payload gains `"amendment": {"parent": str(parent_path), "parent_projected_sha256": …, "reason": reason, "replaced_cells": […], "carried_cells": […]}`; decisions recomputed over the merged cell set.
- [ ] **Step 1: Failing tests** — build a parent with fakes (Task 9 helper), then: (a) amend with identical study → everything carried forward, zero `run_batch` calls, decisions unchanged; (b) amend with one candidate's params changed → only that candidate's cells replaced; (c) tampered parent file (flip one value) → `ValidationError` from hash check; (d) study missing a parent case → `ValidationError`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): hash-chained amendment flow`

### Task 13: YAML loader (`yaml_loader.py`)

**Files:**
- Create: `quantark/modelvalidation/yaml_loader.py`
- Modify: `quantark/modelvalidation/__init__.py` (re-export `load_study`)
- Test: `test/modelvalidation/test_yaml_loader.py`

**Interfaces:**
- Consumes: registry (`get_builder`), study types.
- Produces:
  - `load_study_text(text: str) -> CertificationStudy` (the function `assert_anchors` calls) and `load_study(path: str | Path) -> CertificationStudy` (reads file, delegates, stores the verbatim text as `source_text`).
  - YAML schema (validated with explicit checks, no external schema lib):

```yaml
study: <name>                    # required, str
schema: 1                        # required, == 1
quantities: [pv, delta, gamma]   # required, non-empty subset
bounds: {cell: 0.5, mean_signed_bias: 0.1}          # + optional se_budget_fraction/interval_k/envelope_fraction
sampling: {paths_per_batch: 65536, min_batches: 8, max_batches: 64, seed: 20260814, bump: 0.01}
economic_scale: {builder: hedge_contracts, params: {...}}
environment:    {builder: flat_bsm, params: {...}}   # base env; cases override via environment_params
product:        {builder: equity.snowball, params: {...}}
reference:      {builder: equity.snowball.mc_rqmc, params: {...}}
candidates:
  - {builder: equity.snowball.pde,  params: {...}}
  - {builder: equity.snowball.quad, params: {...}}
cases:
  - {name: ordinary}
  - {name: near_ko, environment: {spot: 102.5}}      # -> CaseSpec.environment_params
  - {name: low_vol, environment: {vol: 0.12}, product: {ko_barrier: 101.0}}
```

  - Loader resolution order: `economic_scale` builder → `EconomicScale`; `environment`/`product` param dicts are **not** built here — they are passed to the `reference`/`candidate` builders, which receive `(env_spec, product_spec, sampling, quantities, params)` and return a `ReferenceBuilder` / `CandidateEvaluator`. (Builders own object construction per the spec; the loader only resolves names and validates structure.) Exact builder call signatures:
    - reference builder: `fn(environment_params: dict, product_params: dict, sampling: SamplingPolicy, quantities: tuple[str, ...], params: dict) -> ReferenceBuilder`
    - candidate builder: `fn(environment_params: dict, product_params: dict, quantities: tuple[str, ...], params: dict) -> CandidateEvaluator`
    - economic_scale builder: `fn(params: dict) -> EconomicScale`
  - Every structural violation → `ValidationError` naming the YAML path (e.g. `"candidates[1].builder"`).

- [ ] **Step 1: Failing tests** — register fake builders (env-independent), then: (a) a minimal valid YAML loads; `source_text` round-trips verbatim; (b) `schema: 2` → `ValidationError`; (c) unknown builder name → `ValidationError` mentioning registered names; (d) missing `bounds.cell` → `ValidationError` naming the path; (e) duplicate case names → `ValidationError`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** (`yaml.safe_load` only — never `yaml.load`).
- [ ] **Step 4: Run — PASS. Also un-monkeypatch nothing: go back to `test_anchors.py` and add one integration assertion that `assert_anchors` works against the real `load_study_text` with fake builders registered.**
- [ ] **Step 5: Commit** `feat(modelvalidation): YAML study loader over the builder registry`

### Task 14: CLI (`cli.py`, `__main__.py`)

**Files:**
- Create: `quantark/modelvalidation/cli.py`, `quantark/modelvalidation/__main__.py`
- Test: `test/modelvalidation/test_cli.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int` with subcommands:
  - `run <study.yaml> [--quick] [--resume] [--out DIR]` (default `--out output/modelvalidation`) → prints decisions + certificate path; exit 0 always when the run completes (a REJECTED decision is a *successful certification run*; exit 1 only on exceptions).
  - `amend <study.yaml> --parent <certificate.json> --reason TEXT [--quick] [--resume] [--out DIR]`
  - `anchors <certificate.json> [--out FILE]` — needs the study; reads `study.source_text` from the certificate, extracts, writes anchor JSON (default: next to the certificate as `anchors.json`).
  - `list` — prints `list_builders()` kinds/names and any YAML studies found in `example/modelvalidation/`.
  - `__main__.py`: `sys.exit(main())`.
  - `cli.py` imports `quantark.modelvalidation.builders` at the top so builtin builders are always registered for CLI users.
- [ ] **Step 1: Failing tests** — call `main([...])` directly (no subprocess): `run` on a fake-builder YAML in `tmp_path` produces a certificate; `list` output contains a registered fake name; bad subcommand returns nonzero from argparse (assert `SystemExit.code == 2`).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** (note: `builders/` package does not exist until Task 15 — create the empty `builders/__init__.py` in THIS task so the import holds).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `feat(modelvalidation): universal CLI (run, amend, anchors, list)`

### Task 15: European flat-BSM self-test study (builders + integration test)

**Files:**
- Create: `quantark/modelvalidation/builders/european_selftest.py`, `example/modelvalidation/european_selftest.yaml`
- Modify: `quantark/modelvalidation/builders/__init__.py` (import the module)
- Test: `test/modelvalidation/test_selftest_study.py`

**Interfaces:**
- Consumes: `BlackScholesEngine`, `EuropeanVanillaOption`, `PricingEnvironment` et al. (exact imports in the File Structure section), plus everything above.
- Produces (registered builders):
  - `flat_bsm` (environment kind): `params {spot, vol, rate, div_yield}` → returns the params dict enriched with defaults (the *reference/candidate* builders build the actual `PricingEnvironment`, applying case `environment_params` overrides: `spot`, `vol`, `rate`, `div_yield` keys replace).
  - `hedge_contracts` (economic_scale): `params {hedge_multiplier, hedge_inception_spot, notional}` → `HedgeContractScale`.
  - `equity.european.analytical` (candidate): builds `EuropeanVanillaOption(strike=…, option_type=OptionType.CALL|PUT, maturity=…)` per product params `{strike, maturity, option_type}`; `evaluate` prices with `BlackScholesEngine().price(product, env)` at base / down / up spot (relative bump `params.get("bump", 0.01)`) → pv/delta/gamma central differences; ladders: a single axis `"analytic"` where medium == target (envelope 0 — exact engine).
  - `equity.european.mc` (reference): `run_batch(case, i)` prices the same product with `SnowballMCEngine`? **No** — European MC engine is `quantark.asset.equity.engine.mc.euro_mc_engine` (class name: check `grep "^class" quantark/asset/equity/engine/mc/euro_mc_engine.py` — use whatever `EuroMCEngine`-like class it exposes with `MCParams`); one batch = one engine call with `MCParams(seed=policy.seed + i, num_paths=policy.paths_per_batch, use_qmc=True)` and `method=MonteCarloMethod.RANDOMIZED_QUASI` with `rqmc_min_batches=1, rqmc_max_batches=1`. CRN pairing: the same seed for the down/base/up prices of batch *i*.
- The YAML: strike 100 call, maturity 1.0, flat 20% vol, r 2.5%, q 3%; 3 cases (`atm`, `itm` spot 110, `otm` spot 90); quantities pv/delta/gamma; small sampling (paths_per_batch 8192, min 4, max 16); bounds loose enough that the analytical candidate ADMITS with tiny MC noise (`cell: 1.0, mean_signed_bias: 0.5` in hedge-contract units with multiplier 100, spot 100, notional 1_000_000 — tune once at implementation; the acceptance assertion is ADMITTED, so pick bounds with ≥5× margin from a hand run).
- [ ] **Step 1: Write the integration test first** (it is the failing test):

```python
def test_selftest_study_end_to_end(tmp_path):
    study = load_study("example/modelvalidation/european_selftest.yaml")
    cert = certify(study, out_dir=tmp_path, quick=True, resume=False)
    assert cert.payload["decisions"]["equity.european.analytical"] == "ADMITTED"
    # resume reproduces the projected hash (spec §13 acceptance criterion 2, CI-scale)
    cert2 = certify(study, out_dir=tmp_path, quick=True, resume=True)
    assert cert2.payload["projected_sha256"] == cert.payload["projected_sha256"]
    # amendment carries everything forward on an unchanged study
    cert3 = amend(study, parent=cert.path, out_dir=tmp_path / "amended",
                  reason="no-op amendment test", quick=True)
    assert cert3.payload["amendment"]["replaced_cells"] == []
    # anchors round-trip on this machine
    anchors = extract_anchors(cert.payload, study)
    p = tmp_path / "anchors.json"; atomic_write_json(p, anchors)
    assert_anchors(p)
```

- [ ] **Step 2: Run — FAIL** (builders unregistered).
- [ ] **Step 3: Implement builders + YAML.** First run `grep -n "^class" quantark/asset/equity/engine/mc/euro_mc_engine.py` and use the real class/params. Tune bounds per the hand-run rule above.
- [ ] **Step 4: Run — PASS** (budget: the quick self-test must finish under ~60 s; shrink `paths_per_batch` in the YAML if it doesn't).
- [ ] **Step 5: Commit** `feat(modelvalidation): European flat-BSM self-test study and end-to-end integration test`

### Task 16: Snowball flat-BSM demo study (builders + YAML + smoke test)

**Files:**
- Create: `quantark/modelvalidation/builders/equity_snowball.py`, `example/modelvalidation/snowball_flat_bsm.yaml`
- Modify: `quantark/modelvalidation/builders/__init__.py` (import the module)
- Test: `test/modelvalidation/test_snowball_study.py`

**Interfaces:**
- Consumes: `SnowballOption` + configs, `SnowballMCEngine`/`MCParams`/`MonteCarloMethod`, `SnowballPDESolver`/`PDEParams`, `SnowballQuadEngine`/`QuadParams`, plus the framework.
- Produces (registered builders):
  - `equity.snowball` (product kind): params `{ko_barrier, ki_barrier, ko_rate, coupon_rate, months, maturity, ko_observation_dates?}` → a `_make_snowball(product_params, overrides)` helper building `SnowballOption` with `BarrierConfig`(monthly discrete KO dates `[i/12 for i in range(1, months+1)]` unless given, continuous KI) / `PayoffConfig` / `AccrualConfig` following the `example/snowball_mc_demo.py` pattern. Case `product_params` override individual keys (e.g. `ko_barrier`).
  - `equity.snowball.mc_rqmc` (reference): `run_batch(case, i)`:

```python
def run_batch(self, case, i):
    seed = self.sampling.seed + i
    h = self.sampling.bump * self._base_spot(case)
    prices = {}
    for tag, spot in (("dn", s0 - h), ("base", s0), ("up", s0 + h)):
        env = self._make_env(case, spot=spot)
        engine = SnowballMCEngine(
            params=MCParams(seed=seed, num_paths=self.sampling.paths_per_batch,
                            rqmc_min_batches=1, rqmc_max_batches=1,
                            rqmc_paths_mode="per_batch"),
            method=MonteCarloMethod.RANDOMIZED_QUASI)
        prices[tag] = engine.price(self._product(case), env)   # same seed per arm -> CRN
    return BatchResult(index=i, seed=seed, values={
        "pv": prices["base"],
        "delta": (prices["up"] - prices["dn"]) / (2 * h),
        "gamma": (prices["up"] - 2 * prices["base"] + prices["dn"]) / (h * h)})
```

    (One engine instance per arm per batch — engine instances are not safe for concurrent reuse per the engine contract, and a fresh instance guarantees no term-context bleed.)
  - `equity.snowball.pde` (candidate): `evaluate(case)` calls `SnowballPDESolver(params=PDEParams(accuracy=params.get("accuracy", "standard"))).calculate_greeks(product, env)` → `{"pv": r["price"], "delta": r["delta"], "gamma": r["gamma"]}`. Ladders: axis `"accuracy"` with `level="medium"` = one profile coarser than target (`fast` when target is `standard`, `standard` when target is `high`) re-run of `calculate_greeks`; envelope from `envelope_from_ladders`.
  - `equity.snowball.quad` (candidate): prices with `SnowballQuadEngine(QuadParams(grid_points=params.get("grid_points", 1001)))` at base/dn/up spots (same relative bump as sampling policy) for pv/delta/gamma; ladder axis `"grid_points"` with `medium = (grid_points - 1) // 2 + 1` (nested odd grid, e.g. 501 for 1001).
- The YAML (initial values; §16 of the spec says these get pinned by evidence, so treat as start point):

```yaml
study: snowball-flat-bsm
schema: 1
quantities: [pv, delta, gamma]
bounds: {cell: 0.5, mean_signed_bias: 0.1}
sampling: {paths_per_batch: 65536, min_batches: 8, max_batches: 64, seed: 20260814, bump: 0.01}
economic_scale: {builder: hedge_contracts,
                 params: {hedge_multiplier: 200, hedge_inception_spot: 100.0, notional: 50000000}}
environment: {builder: flat_bsm, params: {spot: 100.0, vol: 0.22, rate: 0.025, div_yield: 0.03}}
product: {builder: equity.snowball,
          params: {ko_barrier: 103.0, ki_barrier: 85.0, ko_rate: 0.15, coupon_rate: 0.15,
                   months: 24, maturity: 2.0}}
reference: {builder: equity.snowball.mc_rqmc, params: {}}
candidates:
  - {builder: equity.snowball.pde,  params: {accuracy: standard}}
  - {builder: equity.snowball.quad, params: {grid_points: 1001}}
cases:
  - {name: ordinary}
  - {name: near_ko,     environment: {spot: 102.5}}
  - {name: near_ki,     environment: {spot: 86.5}}
  - {name: low_vol,     environment: {vol: 0.12}}
  - {name: near_expiry, product: {months: 3, maturity: 0.25}}
```

- [ ] **Step 1: Failing smoke test** — quick-mode, 2 cases only, tiny paths, asserting *soundness not outcome*:

```python
def test_snowball_study_quick_smoke(tmp_path):
    study = load_study("example/modelvalidation/snowball_flat_bsm.yaml")
    small = dataclasses.replace(study, cases=study.cases[:2],
                                sampling=dataclasses.replace(study.sampling,
                                    paths_per_batch=2048, min_batches=2, max_batches=3))
    cert = certify(small, out_dir=tmp_path, quick=False, resume=False)
    for cand in ("equity.snowball.pde", "equity.snowball.quad"):
        assert cert.payload["decisions"][cand] in ("ADMITTED", "REJECTED", "INCONCLUSIVE")
    assert not any(c["verdict"] == "ERROR" for c in cert.payload["cells"])
```

  (Tiny sampling will usually land INCONCLUSIVE — that is correct behavior; the smoke test asserts the machinery runs the real engines without ERROR, not that the engines pass. Keep the whole test under ~3 minutes; shrink further if not.)
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement builders + YAML.** Copy the exact `SnowballOption` construction from `example/snowball_mc_demo.py` (BarrierConfig/PayoffConfig/AccrualConfig kwargs) — do not guess field names; read the file.
- [ ] **Step 4: Run — PASS.** Run the full test dir too.
- [ ] **Step 5: Commit** `feat(modelvalidation): snowball flat-BSM demo study with PDE and QUAD candidates`

### Task 17: Docs, example README, CLAUDE.md row, acceptance-run instructions

**Files:**
- Create: `docs/modelvalidation/RELEASE_PROCEDURE.md`, `example/modelvalidation/README.md`
- Modify: `CLAUDE.md` (add one row to the Supporting Modules table: `| Model Validation | quantark/modelvalidation/ | Engine-release certification: RQMC benchmarks, gates, evidence, CI anchors; see docs/modelvalidation/ |`)
- Test: none (docs) — but `git add -f` the docs files (repo gitignore requires it for new docs).

**RELEASE_PROCEDURE.md contents** (write fully, not as an outline):
1. **When to certify** — new engine or numerical method → full certification (`run`); deliberate numerics change → `amend` with `--reason`; pure refactor with bitwise proof (byte-compare on a detached worktree) → anchors only.
2. **How to run** — the three CLI invocations with real paths; quick mode is for wiring checks only, never bankable.
3. **Where evidence lives** — `output/modelvalidation/<study>/certificate.json` + `report.md` + `checkpoints/`; banked certificates are committed under `docs/modelvalidation/certificates/<study>/<date>/` (JSON + report only, never checkpoints).
4. **Anchors in CI** — extract with `anchors`, commit the anchor file next to the banked certificate, add a test calling `assert_anchors(<path>)`; tolerance policy: exact on the banking machine (fingerprint match), `rel_tol=1e-12` cross-arch (CI is x86_64 Linux; evidence banks on ARM64 macOS — see `test/golden_compare.py` precedent).
5. **Amendment rules** — parent hash must validate; scope may grow, never silently shrink; `--reason` is mandatory and lands in the payload.
6. **Adding a new engine family** — write builders (few lines), write the YAML, `run --quick`, run full, bank, extract anchors, wire the anchor test. Reference the two existing builder modules as templates.
7. **Sign-off checklist** — reviewer confirms: decisions match the report; SE budgets met (no UNRESOLVED cells in an ADMITTED claim); envelope columns populated for grid engines; runtime block matches the claimed machine; certificate committed with its report.

**README.md**: how to run the two studies (quick + full), expected runtimes, where output lands, one-paragraph pointer to the procedure doc.

- [ ] **Step 1: Write both docs + CLAUDE.md row.**
- [ ] **Step 2: Verify** `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m quantark.modelvalidation list` shows all registered builders and both YAML studies.
- [ ] **Step 3: Commit** `docs(modelvalidation): release procedure, example README, CLAUDE.md row` (with `git add -f` for docs paths).

### Task 18 (offline, human-triggered): full demo acceptance run

Not a coding task — the acceptance gate from spec §13, run by the user (hours-scale, machine-dependent):

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m quantark.modelvalidation \
    run example/modelvalidation/snowball_flat_bsm.yaml --out output/modelvalidation
# kill mid-run once, then:
#   ... run ... --resume        -> projected_sha256 must equal an uninterrupted run's
# then: amend with a changed candidate grid; then: anchors + assert_anchors
```

Acceptance criteria are spec §13 items 1–5 verbatim. Findings (including a FAIL/INCONCLUSIVE decision with sound evidence) go in the banked report — never tuned away silently.

---

## Self-Review (completed at write time)

**Spec coverage:** §2 architecture → Tasks 9/13/14; §3 registry → Task 2; §4 types → Tasks 1/7/8; §5 statistical policy → Tasks 5/7/16 (paired CRN in builders, pilot-frozen allocation is trivially satisfied in v1 — per-case batch size is fixed by `SamplingPolicy`, nothing adapts across cells; noted so a future adaptive allocator knows the constraint); §6 gates/decisions → Tasks 3/4; §7 evidence/checkpoints → Tasks 6/9; §8 amendments → Task 12; §9 anchors → Task 11; §10 error handling → Tasks 6/9 (ERROR cells, atomic writes, /tmp guard); §11 layout → File Structure (two recorded amendments); §12 testing tiers → unit (Tasks 1–14), integration (Task 15), acceptance (Task 18); §13 demo → Tasks 16/18; §14 procedure doc → Task 17; §15 non-goals respected (no universal serializer — builders own construction; no stage-16 imports anywhere); §16 open items → resolved in Task 1 (economic formulas), Task 16 (case params, ladder axes, sampling), Task 11 (anchor rule: every non-ERROR candidate×case).

**Placeholder scan:** the only deliberately deferred lookups are two `grep`-then-use steps (Euro MC class name in Task 15, snowball config kwargs in Task 16) — each names the exact file and command; no TBDs remain.

**Type consistency:** `BatchResult`/`ReferenceEstimate` (Task 7) consumed by Tasks 9/16; `CellGateResult` fields used by `decide_cell` (Task 4) and payload assembly (Task 9); `load_study_text` name pinned in Task 13 and referenced by Task 11; `Certificate(payload, path)` used by Tasks 12/15. `should_stop` signature identical in Tasks 5 and 7.

## Execution

Plan complete. Execute task-by-task with fresh-context workers per task (subagent-driven) or inline with checkpoints; every task ends green and committed before the next begins.


