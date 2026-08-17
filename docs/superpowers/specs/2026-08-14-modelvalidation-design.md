# `quantark.modelvalidation` — Engine Release Certification Module

**Date:** 2026-08-14 · **Status:** approved design, pre-implementation · **Branch:** `worktree-modelvalidation`

A standard procedure for releasing complex pricing engines: statistically-controlled
stochastic benchmarks certify deterministic PDE/QUAD engines, with banked evidence
packages and fast CI anchors guarding the result afterwards.

The method is the one proven by the ADI Greek certification (stage-16, branch
`adi-greek-certification`). That work is **prior art only**: this module generalizes
the method and reuses none of stage-16's code, schemas, or banked artifacts. The
module is validated by a fresh demo study (§13), not by replaying stage-16.

---

## 1. Decisions (fixed points)

| Question | Decision |
|---|---|
| Deliverable | Library module **+** written standard release-procedure doc |
| v1 scope | Full method: references, gates, decisions, evidence, **and** parent-certificate/amendment flow |
| Validation | Fresh **snowball / flat-BSM** demo study; no stage-16 material |
| Demo scope | Certify **PDE + QUAD** snowball engines on **PV + delta + gamma** in economic units vs a paired-RQMC MC benchmark |
| Release model | Offline certification banks evidence; module emits **deterministic anchors** that become cheap CI tests |
| Location | `quantark/modelvalidation/` — peer of `var/`, `simm/`, `saccr/` |
| Authoring | **YAML-first** over a per-family builder registry, Python-defined studies as escape hatch |

## 2. Architecture: two front doors, one narrow waist

The framework core consumes exactly one typed object — `CertificationStudy` — and
owns everything downstream of it (the pipeline: reference building, candidate
evaluation, gates, decisions, evidence, report, anchors). Nothing below the waist
can be customized away per study; that is what makes the procedure a standard.

Two front doors produce a `CertificationStudy`:

1. **YAML study file** (primary). Declarative, diffable, deliverable. It names
   *registered builders* plus their params; it never serializes rich pricing
   objects. The resolved YAML text is embedded verbatim in the certificate and
   covered by its sha256, so a certificate ships its own re-runnable definition.
2. **Python study module** (escape hatch). Constructs `CertificationStudy`
   directly, for anything YAML can't express yet. Bypasses the registry, never
   the pipeline: flexibility in authoring, never in evidence or gates.

One universal CLI drives every study:

```
python -m quantark.modelvalidation run     <study.yaml | registered-name> [--quick] [--resume] [--out DIR]
python -m quantark.modelvalidation amend   <study> --parent <certificate.json>
python -m quantark.modelvalidation anchors <certificate.json> [--out FILE]
python -m quantark.modelvalidation list
```

Public Python API (same operations, importable):

```python
from quantark.modelvalidation import certify, amend, load_study, CertificationStudy

certificate = certify(study, out_dir=..., quick=False, resume=False)
certificate = amend(study, parent=..., out_dir=...)
```

## 3. Builder registry

`@register_builder("equity.snowball", kind="product")` registers a small function
that turns YAML params into real Python objects. Kinds: `product`, `environment`,
`reference`, `candidate`, `economic_scale`. Rules:

- The registry **grows per engine family, only when a real study needs it** —
  never speculatively. v1 registers only what §13–14 need.
- Unknown builder name or bad params → `ValidationError` naming the builder and
  the valid registered set, before any pricing starts.
- Builders are the only place YAML meets quantark objects. The loader
  (`yaml_loader.py`) validates structure; builders validate semantics.

## 4. Core types

```python
@dataclass(frozen=True)
class CaseSpec:            # one market/product scenario ("ordinary", "near_ko", ...)
    name: str
    environment_params: Mapping[str, Any]   # overrides applied to the study environment
    product_params: Mapping[str, Any]       # overrides applied to the study product

@dataclass(frozen=True)
class GateBounds:
    cell: float                 # per-cell |err| bound, economic units
    mean_signed_bias: float     # per-quantity aggregate bound, economic units
    se_budget_fraction: float = 0.25   # SE must be ≤ this × cell bound
    interval_k: float = 2.0            # gate uses |err| + k·SE ≤ cell bound

class EconomicScale(Protocol):      # raw quantity → economic units ("c")
    def to_economic(self, quantity: str, raw: float) -> float: ...

class ReferenceBuilder(Protocol):   # stochastic arm
    def identity(self, cell) -> Mapping[str, Any]        # → reference identity hash
    def run_batches(self, cell, batch_range) -> BatchSet # paired base/bumped batches

class CandidateEvaluator(Protocol): # deterministic arm
    def name(self) -> str
    def evaluate(self, cell) -> CandidateResult          # target-grid values + ladders

@dataclass(frozen=True)
class CertificationStudy:
    name: str
    cases: tuple[CaseSpec, ...]
    quantities: tuple[str, ...]         # subset of {"pv", "delta", "gamma"}
    bounds: GateBounds
    scale: EconomicScale
    reference: ReferenceBuilder
    candidates: tuple[CandidateEvaluator, ...]
    sampling: SamplingPolicy            # §5
    source_text: str | None             # verbatim YAML when loaded from file
```

A **cell** is one case × one quantity. References are banked **per cell**, keyed
by reference identity (§7), and are shared across candidates: one banked RQMC
reference serves both the PDE and QUAD arms.

## 5. Statistical policy (reference arm)

- **Paired CRN design.** Base and bumped environments consume the same RQMC
  randomization per batch. Delta/gamma are paired central differences at bump
  width `h` (recorded in evidence); pairing collapses difference variance.
- **Batch-mean SE.** With B batches, `SE = std(batch_means, ddof=1) / sqrt(B)`.
- **Gate-driven stopping.** Sampling continues until `SE ≤ se_budget_fraction ×
  cell_bound` for every quantity of the cell, subject to `min_batches` /
  `max_batches`. Hitting `max_batches` without meeting budget does not fail the
  cell; it caps the achievable verdict (§6).
- **Pilot-frozen allocation.** Any cross-cell allocation choice (batch sizing per
  cell) is frozen from a pilot round, never adapted from selected interim results
  — avoids the selected-τ bias failure mode.
- **Soundness rules** (from the stage-16 gate-driven work): every banked batch
  records its batch index range and seed; on resume, banked batches are
  validated against the recorded stopping decision (count and order), and a
  mismatch invalidates the bank rather than silently reusing it.
- **Engines.** The reference builder for the demo wraps the existing equity
  snowball MC engine with RQMC (Sobol + Brownian bridge) from
  `quantark/montecarlo/`. The module itself is engine-agnostic.

## 6. Gates, verdicts, decisions

All gate arithmetic is pure-data (`gates.py`), in economic units:

- **Cell gate:** `|cand − ref| + k·SE ≤ cell_bound` (k = `interval_k`), plus the
  SE budget check, plus the **ladder envelope** check: per-axis refinement
  ladders at the target grid must show the candidate's own discretization
  increment within a configured fraction of the cell bound.
- **Aggregate gate (per quantity, per candidate):** `|mean signed (cand − ref)|
  ≤ mean_signed_bias`, with its own SE check across cells.

Cell verdicts: `PASS` · `FAIL` · `ERROR` (engine raised) · `UNRESOLVED` (SE
budget unmet at `max_batches`).

Decision per candidate engine:

- **ADMITTED** — every cell `PASS`, every aggregate gate passes.
- **REJECTED** — at least one *confident* failure: a `FAIL` cell whose reference
  met its SE budget (the failure is not sampling noise), or a failed aggregate
  gate with adequate SE.
- **INCONCLUSIVE** — anything else (`ERROR` cells, `UNRESOLVED` cells, envelope
  violations without a confident failure). An `ERROR` cell makes `ADMITTED`
  unreachable.

Numerical comparisons use `quantark.util.numerical` (no raw float compares);
failures raise the standard hierarchy (`ValidationError`, `NumericalError`,
`PricingError`) — the module adds no parallel exception tree.

## 7. Evidence, schema, checkpointing

- **Schema.** The module starts at `schema: 1`, versioned independently of
  stage-16. `validate_payload` checks every certificate before write and after
  load.
- **Payload.** Verbatim study YAML + resolved config; runtime environment
  (platform, Python/NumPy/quantark versions, git sha); per-cell: reference
  estimate (mean, SE, batch count, seeds, stopping decision), candidate values +
  ladders, gate numbers, verdict; per-candidate decisions; wall-clock metadata.
- **Projected sha256.** The evidence hash covers a *projection* that excludes
  volatile fields (timestamps, durations, host load), so re-runs on the same
  machine reproduce the hash. The projection function is part of the schema.
- **Atomic writes.** Certificate and checkpoints are written
  write-temp-then-rename, always under the run's output directory — never `/tmp`
  (a `/tmp` crash has already cost this repo a banked evidence set once).
- **Checkpointing / resume.** Each cell's banked reference batches and each
  candidate evaluation are durably checkpointed as they complete. `run --resume`
  reuses a checkpoint only when its **identity hash** matches: sha256 of the
  canonical JSON of (schema, case spec, quantity, environment/product params,
  reference builder name+params, sampling policy, economic scale). Candidate
  checkpoints add the candidate builder name+params. Reference identity
  deliberately excludes candidates, so one reference bank serves all candidates.

## 8. Amendments

An amendment re-certifies a subset of cells/candidates after a deliberate change:

- The amendment names its **parent certificate** by path + sha256 and validates
  the parent (schema, hash, manifest) before any pricing.
- It records exactly which cells it replaces and why (free-text reason +
  changed-config diff); untouched cells carry forward by reference to the parent
  hash — lineage is a hash chain, and `validate_payload` walks it.
- Carried-forward cells must have matching identity hashes in the parent; a
  mismatch (the "unchanged" thing actually changed) is a `ValidationError`, not
  a silent re-use.

## 9. Anchors and CI

- `anchors.py` extracts a small set of **deterministic anchor values** from a
  certificate: pinned (case, candidate engine, config) → exact engine outputs.
  Anchors are cheap (seconds) because they re-run only the deterministic engine
  at pinned configs — never the MC reference.
- **Tolerance policy** (explicit, per anchor file): `exact` when the executing
  machine fingerprint (arch/platform) matches the banking machine; `rel_tol`
  (default 1e-12, configurable) cross-arch. CI is x86_64 Linux while evidence
  banks on ARM64 macOS; bitwise `==` fails cross-arch at ~1e-14 ULP — same rule
  as `test/golden_compare.py`.
- A pytest helper (`quantark.modelvalidation.anchors.assert_anchors(path)`) makes
  an anchor file a one-line CI test.

## 10. Error handling

- Study/YAML problems → `ValidationError` before any pricing (fail fast at load).
- During a run, a cell whose engine raises is recorded `ERROR` with the traceback
  in evidence, and the run continues — one broken cell must not destroy an
  hours-scale run. `ERROR` caps the decision at `INCONCLUSIVE`.
- Reference-arm numerical failures follow the same rule.
- A keyboard interrupt leaves completed checkpoints valid (atomic writes) —
  `--resume` continues.

## 11. Module layout

```
quantark/modelvalidation/
├── __init__.py      public API: certify, amend, load_study, CertificationStudy
├── study.py         CertificationStudy · CaseSpec · GateBounds · SamplingPolicy · EconomicScale
├── registry.py      @register_builder — per-family, grows on demand
├── yaml_loader.py   YAML study → CertificationStudy (text embedded in evidence)
├── reference.py     ReferenceBuilder protocol · PairedRQMCReference
├── candidate.py     CandidateEvaluator protocol · target grid + ladders
├── stopping.py      gate-driven sequential stopping policy
├── gates.py         pure gate arithmetic in economic units
├── decisions.py     cell verdicts → per-candidate decisions
├── evidence.py      schema-versioned payload · sha256 projections · atomic writes · checkpoints
├── amendment.py     parent-certificate validation · amendment flow
├── anchors.py       anchor extraction · tolerance policy · pytest helper
├── report.py        markdown report renderer
└── cli.py           python -m quantark.modelvalidation {run · amend · list · anchors}
```

Studies live in `example/modelvalidation/` (YAML + any Python-study modules).

## 12. Testing strategy

1. **Unit tier (fast, no pricing).** `gates.py`, `decisions.py`, `stopping.py`,
   `evidence.py` (hash projection, atomic write, checkpoint identity),
   `registry.py`, `yaml_loader.py`, `amendment.py` lineage validation — all
   pure-data with synthetic inputs. Tests in `test/modelvalidation/`.
2. **Integration tier (CI, seconds).** A **European vanilla / flat-BSM self-test
   study**: candidate = analytical engine (closed-form truth), reference = tiny
   MC. Runs the entire pipeline end-to-end including evidence write, resume, an
   amendment, and anchor extraction. If the framework cannot admit an
   analytically exact engine, the framework is wrong — this is the module's own
   calibration check, permanently in CI.
3. **Acceptance tier (offline).** The snowball flat-BSM demo study (§13) run in
   full; its banked evidence feeds the first real anchor tests.

## 13. Demo study: snowball, flat BSM (acceptance gate)

- Product: equity snowball (absolute barrier levels); environment: flat BSM
  (flat vol, flat r/q).
- Cases: `ordinary`, `near_ko`, `near_ki`, `low_vol`, `near_expiry` (final list
  pinned during implementation; must include at least one KO-stressed and one
  KI-stressed case).
- Quantities: `pv`, `delta`, `gamma`. Economic scale: hedge contracts
  (`hedge_multiplier × hedge_inception_spot / notional` per contract).
- Reference: paired-RQMC snowball MC (Sobol + Brownian bridge).
- Candidates: snowball **PDE** engine and snowball **QUAD** engine at their
  production-default configurations.
- Acceptance criteria for the module:
  1. The study runs end-to-end from YAML via the CLI, quick mode and full mode.
  2. Kill-and-resume mid-run reproduces the uninterrupted certificate's
     projected sha256 (same machine).
  3. An amendment run (e.g. a changed candidate grid) validates lineage and
     replaces only the amended cells.
  4. Anchors extracted from the certificate pass `assert_anchors` on the banking
     machine, and the demo report renders.
  5. The decision for each candidate is whatever the evidence says — a `FAIL` is
     an acceptable demo outcome **if the evidence is sound**; known definitional
     gaps (e.g. QUAD's `ki_probability` semantics) are documented in the report,
     not papered over.

## 14. Procedure doc

`docs/modelvalidation/RELEASE_PROCEDURE.md`, written against the working module:

- **Triggers:** new engine or new numerical method → full certification; a
  deliberate numerics change → amendment; a refactor with bitwise proof →
  anchors only.
- Where evidence banks live and how they are named; how certificates are
  referenced from release notes.
- Sign-off checklist (who reviews the report, what a reviewer checks).
- How to add a new engine family: write builders, write the YAML study, run
  quick mode, run full, bank, extract anchors, wire the anchor test.

## 15. Non-goals (v1)

- No universal YAML serializer for arbitrary quantark objects — builders only.
- No replay of stage-16 artifacts; no schema compatibility with stage-16.
- No distributed/multi-host execution (single-machine, resumable runs).
- No Heston/SLV demo study (the flat-BSM demo validates the framework; vol-model
  studies are the natural second user).
- No GUI/HTML report (markdown only; HTML can come later).

## 16. Open items to pin during implementation

- Exact demo case parameters (barriers, coupon, tenor) and sampling sizes.
- Per-quantity conversion formulas for the hedge-contract scale (delta quantum is
  standard; the PV and gamma mappings into "c" units must be written down).
- Ladder axes per candidate family (PDE: n_x/n_t; QUAD: nodes/steps) and the
  envelope fraction default.
- Anchor selection rule (how many, which cells).
- `SamplingPolicy` fields (paths per batch, min/max batches, pilot round size).
