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
  mo_dashboard.yaml          the registry (contract A, §5)
  mo_dashboard/
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

The package is `mo_dashboard`, not `dashboard`: tests put `example/mo_volmodels/` on `sys.path`,
where a top-level module named `dashboard` would be a collision hazard.

Paths are absolutised with `os.path.normpath`, never `Path.resolve()`. Resolving follows
symlinks, and `output/` is a symlink in a worktree checkout — registry entries would resolve into
the main repository while a scan of the same symlink stays in worktree space, so the two sides
could never match and every displayed path would be absolute.

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

**Absent and corrupt are different states and must not share a return value.** A reader that
answers `None` to both lets a truncated `run_manifest.json` render as "0 runs completed" — a
legitimate-looking result produced by a parse failure. Every artifact reader returns a structured
result carrying `missing` versus `unreadable` plus the exception text, and every `unreadable`
propagates into `payload["errors"]`. A collector that always returns an empty error list is a
contract violation, not a convenience.

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

`GateRow = {id, title, artifact_path, artifact_mtime, status, headline, facets, by_variant}`.
`facets: {<facet>: Provenance}` is the gate-level roll-up — G2 carries `pv` and `delta`
separately (§6.2), other gates a single `all`. G2 additionally carries
`by_variant: {<variant>: {<facet>: Provenance}}`, because its invalidations are variant-scoped
and a facet-only key makes them unreachable; each gate-level facet is the worst across variants.

`Provenance = {mode: "inferred", freshness: "fresh"|"stale"|"void", invalidated_by: <commit>|null,
invalidation_reason: str, superseded_by: [...], dirty_deps: [...], missing_deps: [...]}`.
`mode` is always `"inferred"` — see §6.4 for why `"exact"` is deferred rather than half-built.

`Read = {state: "ok"|"missing"|"unreadable", doc: Any|None, message: str}` — the return of every
artifact reader, so absent and corrupt never collapse into the same value (§3.3).

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
| G5 | artifact readable **and schema-complete** (positive `n_operating_points` and an explicit `under_resolved` list) with zero under-resolved points |
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
§7A.11; the `ratio > 10` band labelled **EXCLUDE (provisional)**. The label is deliberately not
"never average": §7A.10(3) established the exclusion, but §5.9 (`ec20db9`, 2026-08-03) supersedes
§7A.11's attribution — those dates fail on *discretisation*, not calibration (Péclet ≈ 5,872
against a monotonicity bound of 2, with 81 % of medium-grid steps accidentally damping the bad
modes), and are fixable. The panel renders the citation alongside the band so the exclusion reads
as a current numerical limitation rather than a property of the model.

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

**Coverage counts `fresh + stale`, and shows the split.** An earlier draft counted `fresh` only,
which is wrong for the reason the stale/void distinction exists: *stale* means a dependency moved
and the cell should be re-run to be certain, not that the work is absent. Under the fresh-only
rule today's real state reads **0/162** — every `flat_bsm` cell predates `f97fba3`, `3fbbf21` and
`ec20db9` — which is not a useful statement about a fleet that has 27 completed cells. `void`,
`failed` and `missing` are what disqualify. The headline therefore reads
`27/162 admitted (0 fresh · 27 stale)`, never a bare 27.

Run dirs are labelled by registry role (`fleet` / `probe`);
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

**The inlined payload must escape `<` as `<`.** `json.dumps` passes `</script>` through
untouched (verified), and the payload carries log tails, exception text and git subjects — all
attacker-adjacent only in the sense that they are arbitrary text from disk, which is enough. An
unescaped `</script>` terminates the `application/json` element and everything after it becomes
markup. Cheap to prevent, and the snapshot is a file people forward.

## 5. Contract A — the registry

`example/mo_volmodels/mo_dashboard.yaml`. Hand-maintained; `pyyaml>=6.0.0` is already declared
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

Two commits are deliberately absent, and the reasons are recorded so a later reader does not
mistake omission for oversight.

`b6b97f0` — before it, `PDEEngine` failed closed and produced no output, so no surviving pricing
summary contains numbers it changed. It is nonetheless covered by the dependency table (§6.5),
which includes the engine facade files it touched, so an artifact predating it reads `stale` —
flagged for a re-run — rather than being silently certified.

`ec20db9` (§5.9) — a **reattribution**, not an invalidation. It supersedes §7A.11's *explanation*
of why σ-collapse dates fail the PDE gate (discretisation, not calibration) while leaving the
measured failures themselves intact. No artifact's numbers become non-comparable, so voiding
would be wrong. Its consequence is a label change in Panel 2 (§4.2), not a freshness verdict.
It does reach freshness by another route: `STUDY_SPEC` is a declared dependency of G2, G4 and
FLEET (§6.5), so artifacts predating it read `stale`.

### 5.3 Fleet dimensions

Derived from executable definitions, never restated here:

- **Variants** — `VARIANTS` from `12_snowball_volmodel_backtest.py:143`, the canonical 6-tuple.
  Not stage 13's `VARIANT_ORDER`, which lists 5 (§1.3).
- **Inceptions** — read at runtime from the G4 artifact
  `output/volmodel_backtest/inceptions.json`, which is the authoritative list of what the fleet
  *is*. **The collector must not call `schedule_inceptions()`.** That function lives in
  `12_snowball_volmodel_backtest.py:452` (not `cohort.py`, which exposes only `COHORT_ASOF`,
  `admitted_dates()` and `excluded_records()` — an earlier draft named a function that does not
  exist), and reaching it means executing stage 12, which imports the whole pricing and backtest
  stack. That violates §2's no-pricing-code rule, is slow on every page render, and fails outright
  in a read-only environment where matplotlib cannot write its font cache — degrading the grid to
  zero cells silently.

  The definition is still enforced, just in the test rather than the collector: a unit test
  asserts the artifact's inception list equals
  `schedule_inceptions(calendar=…, data_start=…, data_end=COHORT_ASOF, first_admitted_surface=…)`
  with `data_start` from the spot cache's first row and `first_admitted_surface` from
  `cohort.admitted_dates()[0]`, mirroring `test_cohort.py::test_data_end_pin_governs_the_
  inception_count`. Drift fails a test; it never silently changes a denominator.

  When the G4 artifact is absent the fleet dimensions are unknown, so the grid is not drawn and
  the panel says so. That is the honest reading: with no coupon solve there is no defined fleet.

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

**G2's provenance is keyed by (variant, facet), not facet alone.** `f97fba3` is scoped to
`heston` and `heston_slv`; if the collector evaluates G2 with `variant=None`, that invalidation
can never reach it — the scoping mechanism would be dead code for the one gate it was written
for. G2 therefore produces a provenance entry per variant per facet, and the gate-level facet
badge is the worst across variants.

**Route is the decision; the comparison flags are evidence.** `delta_pass: false` on `heston`
does not mean G2 failed — it is *why* `heston` is routed to `mc` rather than `pde`. Reading
`delta_pass` as the gate predicate would leave G2 permanently unsatisfiable no matter what the
study does, since the routes that exist are precisely the ones chosen because a comparison did
not pass. The predicate is route presence plus non-void facets (§4.1); `medium_pass`,
`fine_pass`, `delta_pass` and the bias blocks are rendered as supporting evidence.

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

### 6.4 Exact versus inferred — inferred only, for now

Every verdict this dashboard renders is `inferred`. An `exact` mode — reading an artifact's
embedded `provenance.commit` and comparing with `git rev-list <commit>..HEAD -- <deps>` — is
**deferred until artifacts actually carry stamps**, and is deliberately not half-built.

An earlier draft shipped a `stamped_commit` parameter that set the badge to `exact` while still
deciding freshness from mtime and never validating the SHA. A badge that says *exact* on
unvalidated evidence is worse than no badge: it is the one label a reader would trust without
checking, and it would have been wrong for every artifact, including ones stamped with a commit
that does not exist. The parameter is removed rather than left as a lie.

The dashboard consequently carries a single confidence level, and §6.3's honesty requirements
apply to all of it. When gate scripts are eventually stamped, `exact` arrives together with the
`rev-list` comparison and tests covering an old stamp, a current stamp and an invalid one.

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

Two details that a naive implementation gets wrong, both verified against this repository:

- **`git status --porcelain` collapses untracked trees to a parent.** It reports
  `?? example/mo_volmodels/data/history/`, never the `surface_manifest.json` inside it, so a
  containment test that only asks "is the reported path at or below my declared dep" misses every
  change. Declared untracked deps are therefore `stat`ed directly rather than discovered through
  `git status`.
- **A missing dependency is an error and a non-fresh verdict**, not metadata attached to a green
  row. §3.3 already requires the error row; freshness must agree with it, or a renamed engine
  directory silently turns every verdict green — the exact failure the error row exists to
  prevent.

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

7. **Vacuity guards**, each written because the corresponding check would otherwise pass without
   exercising anything:
   - G2 provenance is keyed by variant — a `heston`-scoped invalidation reaches G2's `heston`
     row and not its `flat_bsm` row. Without this, §5.2's scoping is dead code for the one gate
     it exists for.
   - `headline_g5({})` and `headline_g5({"n_operating_points": 3})` are **not** satisfied.
   - A manifest `failures[]` entry whose cell directory was never created renders `failed` **at
     the collector level**, not merely in a hand-built `cell_state` call.
   - Artifact readers distinguish missing from corrupt, and a corrupt `run_manifest.json`
     produces a `payload["errors"]` entry rather than "0 runs completed".
   - Rendering with a payload containing `</script><script>` yields a document where that text
     does not terminate the payload element.
8. **Dimensions are read-only**: collecting the fleet grid imports no pricing module. Asserted by
   checking `sys.modules` gains no `quantark.asset` entry across a `collect()` call.

**Integration fixture (the check that would have caught the §5.2 defect).** One test asserts the
whole expected dashboard state against the real artifacts as of 2026-08-03: G2 `pv` present and
`delta` **void by `3fbbf21`**; G1 not void; **27 `stale` cells and 0 `fresh`** — every
`flat_bsm` cell predates `f97fba3`, `3fbbf21` and `ec20db9`, which is why coverage counts
`fresh + stale` (§4.3); the 8 Jul-27 `ts_bsm`/`localvol` cells `void` by `41f2117`; admitted
27/162; `next_action` = G2. It skips when `output/` is absent, matching the existing convention
for tests depending on the uncommitted history cache.

The fixture asserts *computed* state, not hand-copied numbers: if it disagrees with the design,
the design or the collector is wrong — the assertion is not the thing to adjust.

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
| Dashboard drifts from study definitions | Variants from stage 12's `VARIANTS`; inceptions read from the G4 artifact and asserted equal to `schedule_inceptions()` in a test (§5.3) |
| Rendering the page executes the pricing stack | Dimensions come from an artifact, never from executing stage 12; a test renders in a read-only environment (§5.3, criterion 9) |
| A confidence badge that does not check anything | `exact` mode removed until stamps exist and are validated (§6.4) |
| A gate reads satisfied on a partial artifact | G5 requires a complete schema; incomplete is `unreadable`, not satisfied (§4.1) |
| An early failure leaves no directory and vanishes | Cells are the union of the pinned grid, the walked tree, and manifest `failures[]` (criterion 10) |
| Arbitrary artifact text breaks out of the inlined payload | `<` escaped as `<` in the embedded JSON (§4.4) |

## 10. Success criteria

1. One command produces a page stating, without reading any other file: each gate's verdict per
   facet with its freshness; fleet coverage as admitted (`fresh + stale`) cells out of 162 with
   the split shown; and the next action with the reason behind it.
2. **G2's `delta` facet renders `void` by `3fbbf21` (§5.8)** — the live invalidation that exists
   today and appears nowhere on disk. This is the criterion that distinguishes this dashboard
   from a file listing.
3. The eight Jul-27 `ts_bsm`/`localvol` cells render `void`, not as progress.
4. Coverage reads `27/162 admitted (0 fresh · 27 stale)` — not the 35 cells on disk, not the 38
   a naive sum of `runs_completed` across all six run dirs would give, and not the 0 a
   fresh-only rule would give.
5. G1 and the 27 `flat_bsm` cells are **not** voided by `f97fba3`, demonstrating scoping works.
6. The dirty working tree is visible on the page.
7. `--serve` updates the fleet panel within ~10 s of a cell completing.
8. The dashboard never writes to `output/` outside its own HTML file.
9. Rendering the page imports no pricing or backtest code and completes in a read-only
   environment — no stage-12 execution, no matplotlib font cache write (§5.3).
10. A cell listed in a manifest's `failures[]` whose directory was never created still renders
    as `failed`, not `missing`. An execution failure must not disappear because it failed early
    enough to leave no trace on disk.
