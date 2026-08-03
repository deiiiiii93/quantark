# Design: Snowball Study Progress Dashboard

**Date:** 2026-08-03
**Status:** revised after adversarial review; ready for implementation planning
**Branch:** `fix/snowball-rebaseline-7a4-engine-fixes` (or a descendant)

**Goal.** A single page that answers *where is the snowball vol-model study*, replacing six
disconnected sources with one read-only view over the artifacts that already exist.

Related documents:

- Study spec: `docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md`
  (all `§` references below point there unless stated otherwise)
- Gate plan: `docs/superpowers/plans/2026-07-31-snowball-rebaseline-gates.md`
- Original plan: `docs/superpowers/plans/2026-07-23-snowball-volmodel-backtest.md`

---

## 1. Problem

Progress on the study is real but unreadable. It is spread across six places, two of which
actively mislead.

| Source | Knows | Reliable |
|---|---|---|
| Study spec §5.5–§5.8, §7A.10–§7A.12 | measured findings, in prose | yes, but ~1500 lines |
| Gate plan, 77 checkboxes | task decomposition | **no — 0 of 77 checked, yet Tasks 0–9 are landed in git** |
| `git log` | what landed, when | yes; the de-facto ledger |
| `output/*.json` | gate verdicts, run counts, calibration status | yes, but unversioned against code |
| `output/*.log` | run narration | ad-hoc names, no index |
| Memory file | narrative state | point-in-time, already stale |

Three failures are load-bearing and motivate the design.

**1.1 Verdicts carry no provenance, and a live one is already invalid.**
`output/` holds `gate_decision_prepolicy.json`, `gate_decision_pre_task9.json` and
`gate_decision_pre_pdefix.json` beside the live `pde_convergence_gate/gate_decision.json`. Those
names are a hand-maintained staleness ledger that nothing enforces.

It has already failed once. The live G2 decision was written 2026-08-03 14:39. At 15:17 the same
day, commit `3fbbf21` added study spec §5.8, which opens: *"This invalidates the delta half of
every §5.5 route decision."* The MC delta reference carries σ ≈ 0.41–0.51 futures contracts of
noise against a 0.1-contract bias bound — 4.6× over, unresolvable in principle. **The live
artifact's delta verdicts are void, and nothing on disk says so.**

That this invalidation arrived as a *documentation* commit, not a code commit, is the central
design constraint: freshness cannot be a function of source directories alone.

**1.2 The fleet run tree holds orphaned cells that no tool counts.**
`run_manifest.json` records only its last invocation (`config.variants: ["flat_bsm"]`,
`counts.runs_completed: 27`). Walking `runs/<inception>/<variant>/run_summary.json` finds **35**:

| variant | cells | mtime | status |
|---|---|---|---|
| `flat_bsm` | 27 | Aug 2–3 | current |
| `ts_bsm` | 4 | Jul 27 | **void** — predates §7A.4 (`41f2117`, Jul 31 10:13) |
| `localvol` | 4 | Jul 27 | **void** — predates §7A.4 |

`13_aggregate_and_report.py:694 aggregate()` iterates `manifest["runs"]`, so it does **not** pick
these up — an earlier draft of this spec claimed it would, and that was wrong. The real hazard is
the opposite: the eight cells are *orphaned*. Neither the manifest, nor `aggregate()`, nor
`verify_fleet_completeness()` sees them. They occupy the tree, are indistinguishable by eye from
current work, and would be silently adopted by any future invocation that re-lists those
(inception, variant) pairs without recomputing them.

This creates a genuine **denominator divergence** the dashboard must state rather than hide:
Panel 2 reports what `aggregate()` sees (manifest-scoped), Panel 3 reports what exists on disk
(tree-scoped). Those numbers legitimately differ. A page showing both without explaining the
difference is worse than showing one.

**1.3 Stage 13 does not know the study has six variants.**
`13_aggregate_and_report.py:41 VARIANT_ORDER` lists five, omitting `flat_bsm_quad` — the engine
control added by gate-plan Task 3. `aggregate()` sorts any unlisted variant to index 99. The
dashboard must not inherit this list; it sources variants from stage 12 (§5.3). Fixing stage 13
is out of scope here but is flagged as a defect.

## 2. Scope

**In.** A read-only dashboard over existing artifacts: three panels (program status, results,
fleet coverage/monitor), a snapshot mode, and an optional local server whose fleet panel polls.

**Out.** Any change to how gates or fleets run. The dashboard never writes into `output/`, never
triggers a run, never repairs data. It does not fix stage 13's `VARIANT_ORDER` (§1.3).
Opportunistic `provenance` stamping of gate artifacts is deferred to whenever those scripts are
next touched (§6.4).

**Explicitly not claimed.** The dashboard is a *viewer*, not a gate. It reports state and its own
confidence in that state. It never certifies that a verdict is valid — only that it has, or has
not, found evidence against it (§6.3).

## 3. Architecture

One collection pass produces one versioned `payload` dict; both modes render from it.

```
dashboard.yaml (registry) ─┐
output/*.json ─────────────┤
output/*/runs/**/ ─────────┼─→ collect() ─→ payload{v1} ─┬─→ render.html  (snapshot: inlined)
git log / git status ──────┤                             └─→ /api/*       (serve: re-collect)
output/*.log (tails) ──────┘
```

### 3.1 Layout

A thin CLI over a small package. `example/mo_volmodels/11_pde_convergence_gate.py` is already
~1600 lines and the gate plan forbids restructuring it; a ~1000-line `16_dashboard.py` would
recreate that problem in a new file.

```
example/mo_volmodels/
  16_dashboard.py            argparse + wiring only (~60 lines)
  dashboard.yaml             the registry (contract A, §5)
  dashboard/
    provenance.py            freshness rule (contract B, §6) — pure
    gates.py                 G1/G4/G2/G5 artifacts → status rows
    fleet.py                 registry + runs/ walk → 6 x 27 cell grid
    results.py               gate evidence · backtest · calibration
    payload.py               assemble, stamp schema_version
    render.py                payload → self-contained HTML
    serve.py                 stdlib http.server; re-collects per poll
test/mo_volmodels/
  test_dashboard.py          pure functions + one real-artifact fixture (§8)
```

Each collector is a pure function of (paths, registry) returning a plain dict. No collector
imports another; `payload.py` is the only composition point.

### 3.2 Modes

- **Snapshot** (default): `payload` is JSON-inlined into one self-contained HTML at
  `output/snowball_dashboard_latest.html`. Opens over `file://`, archivable, diffable. A
  `file://` page cannot `fetch()` a sibling JSON, so inlining is required, not a preference.
- **Serve** (`--serve`, default port 8765, bound to `127.0.0.1`): same collectors behind
  `/api/gates`, `/api/results`, `/api/fleet`, `/api/live`. Panel 3 polls every 10 s; panels 1–2
  every 60 s.

### 3.3 Failure policy

Read-only and fail-soft-but-loud. A missing or unparseable artifact yields a row with
`status: "unreadable"` carrying the exception text, rendered in place. Never a silent zero, never
an omitted row, never an exception that blanks the page. A declared dependency path that no
longer exists is an error row, not a skipped check — a renamed engine directory must not silently
turn every verdict green.

### 3.4 Data model

```python
payload = {
  "schema_version": 1,
  "generated_at": "<ISO 8601>",
  "mode": "snapshot" | "serve",
  "git": {"branch", "head", "head_subject", "dirty_paths": [...]},
  "cohort": {"asof", "n_admitted", "n_excluded", "excluded": [...], "inceptions": [...]},
  "gates": [GateRow, ...],
  "chain": {"nodes": [...], "next_action": {"node", "why", "confidence"}},
  "fleet": {"expected_cells", "variants", "inceptions", "grid", "run_dirs": [RunDir, ...],
            "counts": {"fresh", "stale", "void", "failed", "running", "unreadable", "missing"}},
  "results": {"gate_evidence": {...}, "backtest": {...}, "calibration": {...}},
  "live": {"active": [...], "log_tails": {...}},   # serve mode only; absent in snapshot
  "errors": [{"source", "path", "message"}, ...],
}
```

`GateRow = {id, title, artifact_path, artifact_mtime, status, headline, facets}` where
`facets: {<facet>: Provenance}` — G2 carries `pv` and `delta` separately (§6.2); other gates
carry a single `all` facet.

`Provenance = {mode: "exact"|"inferred", freshness: "fresh"|"stale"|"void",
invalidated_by: <commit>|null, superseded_by: [...], dirty_deps: [...], missing_deps: [...]}`.

`Cell = {inception, variant, state, mtime, provenance, run_dir}` with `state` from the exhaustive
machine in §4.3.

## 4. Panels

### 4.1 Panel 1 — Program status

One row per gate, with per-facet provenance. Artifacts are fixed and known:

| Gate | Artifact | Headline |
|---|---|---|
| G1 surface admission | `output/gate_g1_admission.json` | `n_verified`/`n_admitted`, failures, `min_expiries_seen` |
| G4 fair coupon | `output/volmodel_backtest/inceptions.json` | count with `coupon_solution.solved`, coupon range |
| G2 engine admission | `output/pde_convergence_gate/gate_decision.json` | per-variant route + **separate PV and delta verdicts** |
| G5 grid pre-flight | *(none on disk)* | `NOT RUN`, rendered as a state |

G4's artifact is `inceptions.json`, **not** the run manifest. In the 2026-08-01 invocation the
coupon solve succeeded 27/27 while every replay in the same process failed on the `PDEEngine`
event-stats defect (later fixed by `b6b97f0`). Gate status and run status are independent axes
and must not be collapsed into one row.

**The chain.** Nodes `G1 → G4 → G2 → G5 → fleet → aggregate`, declared as a DAG in code. G5 is
mandatory before fleet work: study spec §9 requires a grid-resolution sweep over every operating
point, and `fdf3a70` made under-resolution a fail-closed `ValidationError`. An earlier draft
omitted G5 and would have recommended fleet work while a mandatory pre-flight was absent.

Node predicates, all enumerated so two implementers cannot differ:

| Node | Satisfied when |
|---|---|
| G1 | artifact readable, `failures == []`, `n_verified == n_admitted`, every facet non-void |
| G4 | every cohort inception has `coupon_solution.solved == true`, every facet non-void |
| G2 | every study variant has a route, **and both `pv` and `delta` facets non-void** |
| G5 | artifact exists and reports zero under-resolved operating points |
| fleet | `fresh` cells == `expected_cells` |
| aggregate | aggregate artifact exists and is newer than the last fresh cell |

`next_action` is the first unsatisfied node, reported with the reason and the confidence of the
evidence behind it (`exact` / `inferred`). It is labelled **next action**, not "blocker" — the
dashboard advises, it does not gate.

Also on this panel: cohort pin (`COHORT_ASOF`, admitted/excluded counts), branch, HEAD, and the
**dirty working-tree file list** — currently invisible everywhere despite
`quantark/volmodels/calibration.py` and `quantark/volmodels/heston/calibration.py` being
modified-uncommitted.

### 4.2 Panel 2 — Results

Three stacked blocks, each from a distinct source, each stamped with the exact cell or variant
set it included.

**Gate evidence** — from `gate_decision.json` / `gate_evidence.json`: six variants × route
(`pde`/`quad`/`mc`), the coarse/medium/fine ladder, and **PV and delta reported as separate
columns with separate provenance**. Per study spec §5.8 the delta column currently renders
`VOID — reference noise σ≈0.46 ct vs a 0.1 ct bound (§5.8)`. Feller-regime conditioning per
§7A.11; the `ratio > 10` band labelled **EXCLUDE — never average** (§7A.10(3)).

**Backtest outcomes** — from `13_aggregate_and_report.py aggregate()`, which is
**manifest-scoped** (§1.2). The block states its denominator explicitly — "27 runs listed in
`volmodel_backtest/run_manifest.json`" — and, when Panel 3's tree walk disagrees, renders the
difference as an explicit reconciliation line rather than leaving two numbers to be compared by
eye. Carries §8's caveat inline: KO dates collapse onto ~13 days, 2024-10-08 kills 7, so
effective sample size is far below 27.

**Calibration health** — from `output/mo_daily_calibration/status.json` and
`calibration_manifest.json`: Feller-ratio distribution, the 6.6 % sigma-collapse band, fit cost in
bp of IV (§7A.10: median 8.4 / p90 29.3 / max 217), bound-hits, and whether the launchd job
(`com.quantark.mo-daily-calibration`) last completed.

### 4.3 Panel 3 — Fleet coverage and monitor

A 6 × 27 grid over the dimensions defined in §5.3. Cells come from walking
`runs/<inception>/<variant>/run_summary.json` in registry-declared fleet dirs — never from
`run_manifest.json` counts (§1.2).

**Attempt identity** is `(run_dir, inception, variant)`. The latest *attempt* status and the
latest *successful artifact* are tracked separately, because a persisted `run_summary.json` can
coexist with a failure entry for the same pair in a later overwritten manifest.

States are exhaustive and resolved by strict precedence, highest first:

| Precedence | State | Condition |
|---|---|---|
| 1 | `unreadable` | cell dir exists but `run_summary.json` cannot be parsed |
| 2 | `running` | serve mode only: dir exists, no `run_summary.json`, mtime within the poll window |
| 3 | `failed` | pair appears in the dir's `run_manifest.json` `failures[]` |
| 4 | `void` | artifact present, but a scoped invalidation applies (§6.2) |
| 5 | `stale` | artifact present, a declared dependency moved after it |
| 6 | `fresh` | artifact present, no invalidation and no newer dependency |
| 7 | `missing` | no cell directory |

Only `fresh` counts toward coverage. Run dirs are labelled by registry role (`fleet` / `probe`);
a dir on disk but absent from the registry renders in an **unclassified** strip. That failure
mode is not hypothetical: `output/volmodel_smoke_gated` was created 2026-08-03 and was missed
during this design's own survey of `output/`. Six run dirs exist; a naive sum of their
`runs_completed` gives 38, against 27 genuinely admitted cells.

In serve mode this panel adds in-flight rows and a tail of the active log. In snapshot mode those
are omitted rather than rendered stale.

### 4.4 Presentation

Inherits the house style from `example/simm_portfolio_demo.py` / `simm_portfolio_dashboard.html`:
dark paper/ink palette (`--paper: #111110`, `--ink: #f5f2e8`), monospace numerics, flat-bordered
cards. Plotly is loaded from CDN there; this dashboard uses inline SVG and CSS for its grid and
histograms so a snapshot stays readable offline.

## 5. Contract A — the registry

`example/mo_volmodels/dashboard.yaml`. Hand-maintained; `pyyaml>=6.0.0` is already declared
(`pyproject.toml:39`). It states only what code cannot derive.

### 5.1 Run dirs

```yaml
schema_version: 1

fleet:
  - dir: output/volmodel_backtest      # 0.4.0 re-baseline fleet
probes:
  - dir: output/volmodel_smoke         # 1 inception x 6 variants, censored at data_end
  - dir: output/volmodel_smoke_gated   # 1 inception x 3 variants, post-f97fba3, censored
  - dir: output/timing_on              # §7.4 event-stats timing, --quick
  - dir: output/timing_off             # §7.4 control, --quick
  - dir: output/volmodel_backtest_g3   # dead G3 probe, 1 failed run
```

Missing file: every dir renders `unclassified` plus one error row. Never a crash.

### 5.2 Scoped invalidations

An invalidation must declare **what it invalidates**. An unscoped list is not merely imprecise —
applied uniformly to the current artifacts it voids G1 (Aug 1 11:35), G4 (Aug 3 01:55) and all
27 `flat_bsm` cells (≤ Aug 3 01:55) on the strength of `f97fba3` (Aug 3 13:39), a 2D-PDE Heston
delta fix that touches none of them, leaving zero admitted cells. An earlier draft of this spec
did exactly that and contradicted its own success criteria.

```yaml
invalidations:
  - commit: 41f2117
    landed: 2026-07-31T10:13:27+08:00
    spec: "§7A.4"
    applies_to: {scopes: [G2, G4, FLEET], variants: "*", facets: "*"}
    reason: "enforce_feller + degenerate_pde + QE-M changed what every engine computes"

  - commit: f97fba3
    landed: 2026-08-03T13:39:19+08:00
    spec: "§5.6"
    applies_to: {scopes: [G2, FLEET], variants: [heston, heston_slv], facets: "*"}
    reason: "2D PDE Heston delta grid; 1D routes untouched"

  - commit: 3fbbf21
    landed: 2026-08-03T15:17:23+08:00
    spec: "§5.8"
    applies_to: {scopes: [G2], variants: "*", facets: [delta]}
    reason: "MC delta reference sigma 0.41-0.51 contracts against a 0.1 contract bias bound;
             the delta half of every route decision is void. The PV half stands."
```

`scopes` are gate ids plus the pseudo-scope `FLEET` (run cells). `variants` and `facets` accept
`"*"`. This is what replaces the `_pre_*` renaming convention, and — crucially — it lets a
**documentation** commit invalidate a **numeric** artifact, which §1.1 shows is a live
requirement, not a hypothetical.

`b6b97f0` is deliberately absent. Before it, `PDEEngine` failed closed and produced no output, so
no surviving pricing summary contains numbers it changed. It is nonetheless covered by the
dependency table (§6.5), which includes the engine facade files it touched, so an artifact
predating it reads `stale` — flagged for a re-run — rather than being silently certified.

### 5.3 Fleet dimensions

Derived from executable definitions, never restated here:

- **Variants** — `VARIANTS` from `12_snowball_volmodel_backtest.py:143`, the canonical 6-tuple.
  Not stage 13's `VARIANT_ORDER`, which lists 5 (§1.3).
- **Inceptions** — `schedule_inceptions()` from `12_snowball_volmodel_backtest.py:452`. It is
  **not** in `cohort.py`, which exposes only `COHORT_ASOF`, `admitted_dates()` and
  `excluded_records()`; an earlier draft named a function that does not exist. It requires four
  arguments — `calendar`, `data_start`, `data_end`, `first_admitted_surface` — supplied as
  `data_end = COHORT_ASOF`, `data_start` and `first_admitted_surface` from
  `cohort.admitted_dates()`, and the study calendar. The resulting 6 × 27 grid is fixtured
  exactly (§8), so a drift in either source fails a test rather than silently changing a
  denominator.

## 6. Contract B — freshness

### 6.1 The rule

```python
freshness(artifact, scope, facet, deps, invalidations) -> Provenance
    voided = [i for i in invalidations
              if scope in i.scopes and variant_matches(i) and facet_matches(i)
              and i.landed > artifact_mtime]
    newer  = commits touching deps with committer time > artifact_mtime
    dirty  = deps modified-uncommitted with file mtime > artifact_mtime
    missing = deps whose path does not exist
    -> "void"  if voided
    -> "stale" if newer or dirty
    -> "fresh" otherwise
```

Scope, variant and facet filtering is the whole point: without it the rule is not conservative,
it is wrong in both directions at once — voiding unrelated artifacts while saying nothing about
the one artifact that is actually invalid.

### 6.2 Facets

A verdict is not atomic. G2 decides PV admission and delta admission from different evidence with
different failure modes, and §5.8 voids exactly one of them. Facets are `pv` and `delta` for G2,
and a single `all` elsewhere. Each facet carries its own `Provenance` and renders its own badge;
the gate's overall status is the worst facet.

### 6.3 Stale versus void, and what inference can prove

*Stale* means a dependency moved: re-run to be sure. *Void* means a declared invalidation applies:
the study spec says this output is not comparable. Collapsing them would let void output read as
merely old — which is how it gets averaged into a result.

Neither verdict is proof. Inferred freshness rests on wall-clock ordering, which is **necessary,
not sufficient**: an artifact that was copied, restored, or `touch`ed reads `fresh` while
containing stale numbers; a dependency edited *before* the artifact is ignored even though the
producing run may not have loaded it. The dashboard therefore:

- never renders a bare `PASS` — always `PASS (exact)` or `PASS (inferred)`;
- carries a standing header caveat that inferred freshness is evidence against invalidity, not
  evidence of validity;
- reports `next_action` with the confidence of the evidence behind it, so an inferred-fresh node
  is visibly weaker ground than an exact-fresh one.

Making inferred-fresh non-satisfying was considered and rejected: on day one every artifact is
inferred, so the chain would report the same blocker regardless of state and carry no
information. Honest labelling beats a uniformly-blocked page.

### 6.4 Exact versus inferred

When an artifact carries an embedded `provenance.commit` block, comparison switches to
`git rev-list <commit>..HEAD -- <deps>` and the badge reads `exact`. Otherwise §6.1 applies and
the badge reads `inferred`. Both readers ship; no gate script is modified by this work, so every
artifact reads `inferred` on day one. Stamping is added opportunistically when those scripts are
next edited.

### 6.5 Dependency table

Declared in `dashboard/provenance.py`. Dependencies are **not code-only** — §1.1 shows a docs
commit invalidating a numeric artifact, so the study spec is a first-class dependency of every
gate that cites it.

| Key | Dependencies |
|---|---|
| `G1` | `13_gate_g1_surface_admission.py`, `cohort.py`, `data/history/surface_manifest.json` |
| `G4` | `12_snowball_volmodel_backtest.py`, `snowball_option.py`, `ENGINE_PATHS`, `STUDY_SPEC` |
| `G2` | `11_pde_convergence_gate.py`, `ENGINE_PATHS`, `STUDY_SPEC` |
| `G5` | `11_pde_convergence_gate.py`, `ENGINE_PATHS` |
| `FLEET` | `12_snowball_volmodel_backtest.py`, `ENGINE_PATHS`, `STUDY_SPEC` |

```
ENGINE_PATHS = [
  "quantark/asset/equity/engine/",     # includes the facade files pde_engine.py,
                                       # event_stats.py, base_engine.py, capabilities.py,
                                       # localvol_greeks.py — b6b97f0 touched pde_engine.py,
                                       # which a narrower engine/pde/ glob would have missed
  "quantark/volmodels/",
  "quantark/backtest/replay/",
]
STUDY_SPEC = "docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md"
```

Untracked data dependencies (`surface_manifest.json`, `data/history/`) have no git history, so
only the mtime arm of §6.1 applies to them. That limit is stated on the page beside any row whose
freshness rests on one.

## 7. CLI

```
16_dashboard.py [--out PATH] [--registry PATH] [--serve] [--port 8765] [--open]
```

Default writes `output/snowball_dashboard_latest.html`. `--serve` starts the local server and
does not write a file. `--open` launches a browser.

## 8. Testing

`test/mo_volmodels/test_dashboard.py` — pure functions, following `test_gate_scope.py`.

1. `freshness` over all three outcomes × scope/variant/facet filtering, driven by a synthetic
   commit list; no real `git` in unit tests. Includes the §5.2 regression: an unscoped `f97fba3`
   must **not** void G1, G4 or `flat_bsm` cells.
2. Registry parsing: dir on disk absent from YAML becomes `unclassified`; missing YAML yields
   all-unclassified plus one error row, never an exception.
3. Cell state machine: precedence is total and every input lands in exactly one state; the
   `failed`+`run_summary.json` overlap resolves to `failed`.
4. Fleet dimensions: the exact 6 × 27 grid is fixtured, so drift in stage 12's `VARIANTS` or
   `schedule_inceptions` fails a test rather than moving a denominator silently.
5. Collector soft-fail: unreadable JSON yields `status: "unreadable"` with the message.
6. Payload contract: `schema_version` and required top-level keys present.

**Integration fixture (the check that would have caught the §5.2 defect).** One test asserts the
whole expected dashboard state against the real artifacts as of 2026-08-03, pinned by mtime in
the fixture rather than read live: G1 `fresh`; G4 `stale`; G2 `pv` present and `delta` **void by
`3fbbf21`**; 27 `fresh` cells; the 8 Jul-27 `ts_bsm`/`localvol` cells `void` by `41f2117`;
coverage 27/162; `next_action` = G2. It skips when `output/` is absent, matching the existing
convention for tests depending on the uncommitted history cache.

Render and serve get one smoke assertion each — non-empty HTML containing all three panel ids,
and `/api/fleet` returning valid JSON — consistent with `13_aggregate_and_report.py` not being
HTML-tested.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Registry goes stale, hiding real work | Unregistered dirs render in a visible `unclassified` strip (§4.3) |
| Invalidation scoped too narrowly — invalid output reads fresh | Scopes are declared with a spec citation; the integration fixture (§8) pins expected state against real artifacts |
| Invalidation scoped too broadly — everything void | Same fixture catches it; §5.2 records the concrete failure this replaces |
| Inferred freshness mistaken for proof | Never a bare `PASS`; standing header caveat; confidence on `next_action` (§6.3) |
| Dep table too narrow — a real change reads fresh | Directory-level deps incl. engine facades; `STUDY_SPEC` covers doc-borne invalidations; missing path is an error row |
| Panel 2 and Panel 3 denominators diverge | Divergence is expected and rendered as an explicit reconciliation line (§1.2, §4.2) |
| Snapshot mistaken for live | `generated_at` and `mode` in the header; `live` block absent in snapshots |
| Dashboard drifts from study definitions | Variants and inceptions imported from stage 12 and fixtured (§5.3) |

## 10. Success criteria

1. One command produces a page stating, without reading any other file: each gate's verdict per
   facet with its freshness and confidence; fleet coverage as fresh cells out of 162; and the
   next action with the reason behind it.
2. **G2's `delta` facet renders `void` by `3fbbf21` (§5.8)** — the live invalidation that exists
   today and appears nowhere on disk. This is the criterion that distinguishes this dashboard
   from a file listing.
3. The eight Jul-27 `ts_bsm`/`localvol` cells render `void`, not as progress.
4. Coverage reads 27/162 — not the 35 cells on disk, and not the 38 a naive sum of
   `runs_completed` across all six run dirs would give.
5. G1 and the 27 `flat_bsm` cells are **not** voided by `f97fba3`, demonstrating scoping works.
6. The dirty working tree is visible on the page.
7. `--serve` updates the fleet panel within ~10 s of a cell completing.
8. The dashboard never writes to `output/` outside its own HTML file.
