# Design: Snowball Study Progress Dashboard

**Date:** 2026-08-03
**Status:** approved, ready for implementation planning
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
| Study spec §5.5–§5.7, §7A.10–§7A.12 | measured findings, in prose | yes, but ~1400 lines |
| Gate plan, 77 checkboxes | task decomposition | **no — 0 of 77 checked, yet Tasks 0–9 are landed in git** |
| `git log` | what landed, when | yes; the de-facto ledger |
| `output/*.json` | gate verdicts, run counts, calibration status | yes, but unversioned against code |
| `output/*.log` | run narration | ad-hoc names, no index |
| Memory file | narrative state | point-in-time, already 3 days stale |

Two failures are load-bearing and motivate most of this design.

**1.1 Verdicts carry no code provenance.** `output/` holds `gate_decision_prepolicy.json`,
`gate_decision_pre_task9.json` and `gate_decision_pre_pdefix.json` beside the live
`pde_convergence_gate/gate_decision.json`. Those names are a hand-maintained staleness ledger:
each records "a code change landed, so the previous verdict is not comparable". Nothing
enforces it, nothing surfaces it, and it is not applied to any other artifact.

**1.2 `run_manifest.json` under-reports coverage, and the run tree holds void work.**
The manifest at `output/volmodel_backtest/run_manifest.json` records only the last invocation
(`config.variants: ["flat_bsm"]`, `counts.runs_completed: 27`). Walking
`runs/<inception>/<variant>/run_summary.json` finds **35** cells:

| variant | cells | mtime | status |
|---|---|---|---|
| `flat_bsm` | 27 | Aug 2–3 | current |
| `ts_bsm` | 4 | Jul 27 | **void** — predates §7A.4 (landed Jul 31) |
| `localvol` | 4 | Jul 27 | **void** — predates §7A.4 |

`13_aggregate_and_report.py:694 aggregate()` scans that root. Pointed at it today, it averages
eight cells computed on pre-§7A.4 engines into results from post-fix engines.

## 2. Scope

**In.** A read-only dashboard over existing artifacts, with three panels: program status,
results, fleet coverage/monitor. Two run modes: archivable snapshot, and a local server whose
fleet panel polls.

**Out.** Any change to how gates or fleets *run*. The dashboard never writes into `output/`,
never triggers a run, and never repairs data. Opportunistic `provenance` stamping of gate
artifacts is deferred to whenever those scripts are next touched for other reasons (§6.2).

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
  test_dashboard.py          pure functions (§8)
```

Each collector is a pure function of (paths, registry) returning a plain dict. No collector
imports another; `payload.py` is the only composition point.

### 3.2 Modes

- **Snapshot** (default): `payload` is JSON-inlined into one self-contained HTML at
  `output/snowball_dashboard_latest.html`. Opens over `file://`, archivable, diffable.
  A `file://` page cannot `fetch()` a sibling JSON, so inlining is required, not a preference.
- **Serve** (`--serve`, default port 8765, bound to `127.0.0.1`): same collectors behind
  `/api/gates`, `/api/results`, `/api/fleet`, `/api/live`. Panel 3 polls every 10 s; panels 1–2
  refresh every 60 s.

### 3.3 Failure policy

Read-only and fail-soft-but-loud. A missing or unparseable artifact yields a row with
`status: "unreadable"` carrying the exception text, rendered in place. Never a silent zero,
never an omitted row, never an exception that blanks the page. This mirrors the study's
fail-closed rule (`gate plan, Global Constraints`) as applied to a viewer: the dashboard's
failure mode is *reporting that it cannot read*, not guessing.

### 3.4 Data model

```python
payload = {
  "schema_version": 1,
  "generated_at": "<ISO 8601>",
  "mode": "snapshot" | "serve",
  "git": {"branch", "head", "head_subject", "dirty_paths": [...]},
  "cohort": {"asof", "n_admitted", "n_excluded", "excluded": [...], "n_inceptions"},
  "gates": [GateRow, ...],
  "fleet": {"expected_cells", "variants", "inceptions", "grid", "run_dirs": [RunDir, ...]},
  "results": {"gate_evidence": {...}, "backtest": {...}, "calibration": {...}},
  "live": {"active": [...], "log_tails": {...}},   # serve mode only; absent in snapshot
  "errors": [{"source", "path", "message"}, ...],
}
```

`GateRow = {id, title, artifact_path, artifact_mtime, status, headline, provenance}`.
`provenance = {mode: "exact"|"inferred", freshness: "fresh"|"stale"|"void", superseded_by: [...],
dirty_deps: [...], invalidated_by: <commit>|null}`.

## 4. Panels

### 4.1 Panel 1 — Program status

One row per gate. Artifacts are fixed and known:

| Gate | Artifact | Headline |
|---|---|---|
| G1 surface admission | `output/gate_g1_admission.json` | `n_verified`/`n_admitted`, failures, `min_expiries_seen` |
| G4 fair coupon | `output/volmodel_backtest/inceptions.json` | count with `coupon_solution.solved`, coupon range |
| G2 engine admission | `output/pde_convergence_gate/gate_decision.json` | per-variant verdict, ladder, delta criterion |
| G5 grid pre-flight | *(none on disk)* | `NOT RUN`, rendered as a state |

G4's artifact is `inceptions.json`, **not** the run manifest. In the 2026-08-01 invocation the
coupon solve succeeded 27/27 while every replay in the same process failed on the `PDEEngine`
event-stats defect (later fixed by `b6b97f0`). Gate status and run status are independent axes
and must not be collapsed into one row.

Below the gates: **next blocking action**, derived rather than written. The chain
`G1 → G4 → G2 → fleet → aggregate` is declared in code; the blocker is the first link that is
not `(PASS and fresh)`. As of 2026-08-03 that resolves to G2, because `f97fba3` touched a PDE
dependency after the live decision was written.

Also on this panel: cohort pin (`COHORT_ASOF`, admitted/excluded counts), branch, HEAD, and the
**dirty working-tree file list**. The last is currently invisible everywhere despite
`quantark/volmodels/calibration.py` and `quantark/volmodels/heston/calibration.py` being
modified-uncommitted — a fact that bears on every verdict on the page.

### 4.2 Panel 2 — Results

Three stacked blocks, each from a distinct source.

**Gate evidence** — from `gate_decision.json` / `gate_evidence.json`: the six-variant table with
route (`pde`/`quad`/`mc`), the coarse/medium/fine ladder, `max_abs_diff_pct` against
`max(2·mc_se_pct, 0.25)`, delta agreement expressed in IM contracts (§5.3), reported per Feller
regime (§7A.11, plan Task 7). The `ratio > 10` band is labelled **EXCLUDE — never average**
(§7A.10(3): those fits satisfy Feller by collapsing sigma to its bound).

**Backtest outcomes** — from `13_aggregate_and_report.py aggregate()`: fair coupon, hedge P&L
(% notional), max drawdown, KO/KI counts per variant, and paired variant-vs-variant diffs.
Carries §8's caveat inline — KO dates collapse onto ~13 days, 2024-10-08 kills 7, so effective
sample size is far below 27. A 27-column table without that note invites over-reading.

**Calibration health** — from `output/mo_daily_calibration/status.json` and
`calibration_manifest.json`: Feller-ratio distribution, the 6.6 % sigma-collapse band, fit cost
in bp of IV (§7A.10 measured median 8.4 / p90 29.3 / max 217), parameter bound-hits, and whether
the launchd job (`com.quantark.mo-daily-calibration`) last completed.

### 4.3 Panel 3 — Fleet coverage and monitor

A 6 x 27 grid: six variants (§4 of the study spec) by the inceptions from
`cohort.schedule_inceptions(data_end=COHORT_ASOF)`. Cells come from walking
`runs/<inception>/<variant>/run_summary.json` in registry-declared fleet dirs — **never** from
`run_manifest.json` counts, per §1.2.

Four cell states:

| State | Meaning |
|---|---|
| fresh | `run_summary.json` present, mtime after every declared invalidation |
| void | present, but mtime precedes an invalidation commit (§6.1) |
| failed | listed in the dir's `run_manifest.json` `failures[]` |
| missing | no cell directory |

Run dirs are labelled by registry role (`fleet` / `probe`). A dir found on disk but absent from
the registry renders in an **unclassified** strip — the stale-registry failure mode is made
visible rather than silent. This is not hypothetical: `output/volmodel_smoke_gated` was created
on 2026-08-03 and was missed during this design's own survey of `output/`. Six run dirs exist;
a naive sum of their `runs_completed` gives 38, against 27 genuinely admitted cells.

In serve mode this panel adds in-flight rows and a tail of the active log. In snapshot mode
those are omitted rather than rendered stale.

### 4.4 Presentation

Inherits the existing house style from `example/simm_portfolio_demo.py` /
`simm_portfolio_dashboard.html`: dark paper/ink palette (`--paper: #111110`, `--ink: #f5f2e8`),
monospace numerics, flat-bordered cards. Plotly is loaded from CDN there; this dashboard uses
inline SVG and CSS for its grid and histograms instead, so a snapshot stays readable offline.

## 5. Contract A — the registry

`example/mo_volmodels/dashboard.yaml`. Hand-maintained; `pyyaml>=6.0.0` is already a declared
dependency (`pyproject.toml:39`). It states only what code cannot derive. The six variants come
from the study spec §4 and the inceptions from `cohort.py`; neither is duplicated here.

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

invalidations:
  - commit: 41f2117
    landed: 2026-07-31T10:13:27+08:00
    spec: "§7A.4"
    reason: "enforce_feller + degenerate_pde + QE-M; prior engine output is not comparable"
  - commit: f97fba3
    landed: 2026-08-03T13:39:19+08:00
    spec: "§5.6"
    reason: "PDE Heston delta grid stabilisation"
```

`invalidations` replaces the `_pre_*` renaming convention with a rule applied uniformly to every
artifact and every fleet cell. It is what mechanically marks the eight Jul-27 `ts_bsm` /
`localvol` cells void.

`b6b97f0` (`PDEEngine forwards event stats`) is deliberately **not** an invalidation. Before it,
the engine failed closed and produced no output; it unblocked runs rather than changing numbers
any surviving artifact contains.

Missing file: every dir renders `unclassified` and the page shows one error row. Never a crash.

## 6. Contract B — freshness

### 6.1 The rule

```python
freshness(artifact_mtime, dep_paths, invalidations) -> Freshness
    newer = commits touching dep_paths whose committer time > artifact_mtime
    dirty = dep_paths modified-uncommitted whose file mtime > artifact_mtime
    voided = invalidations whose commit time > artifact_mtime
    -> "void"  if voided
    -> "stale" if newer or dirty
    -> "fresh" otherwise
```

Distinguishing the two amber states is the point. *Stale* means a dependency moved and the
verdict should be re-run. *Void* means the study spec declares the output not comparable —
§7A.4 changed what the engines compute, so pre-Jul-31 cells are not out-of-date numbers, they
are numbers from a different model. One shared badge would let void cells read as merely old,
which is exactly how they would get averaged into a result.

Applied at two granularities by the same function: gate artifacts (dep set per gate) and fleet
cells (dep set = the engine paths).

### 6.2 Exact vs inferred

When an artifact carries an embedded `provenance.commit` block, comparison switches to
`git rev-list <commit>..HEAD -- <deps>` and the badge reads `exact`. Otherwise the mtime rule
above applies and the badge reads `inferred`. Both readers ship; no gate script is modified by
this work, so every artifact reads `inferred` on day one. Stamping is added opportunistically
when those scripts are next edited for other reasons. The badge is always visible, so an
inferred verdict is never mistaken for an exact one.

### 6.3 Dependency table

Declared in `dashboard/provenance.py`, one entry per gate plus one shared entry for fleet cells:

| Key | Paths |
|---|---|
| `G1` | `example/mo_volmodels/13_gate_g1_surface_admission.py`, `example/mo_volmodels/cohort.py` |
| `G4` | `example/mo_volmodels/12_snowball_volmodel_backtest.py`, `quantark/asset/equity/product/option/snowball_option.py`, `ENGINE_PATHS` |
| `G2` | `example/mo_volmodels/11_pde_convergence_gate.py`, `ENGINE_PATHS` |
| `CELL` | `ENGINE_PATHS` |

`ENGINE_PATHS = ["quantark/asset/equity/engine/pde/", "quantark/asset/equity/engine/mc/",
"quantark/asset/equity/engine/quad/", "quantark/volmodels/", "quantark/backtest/replay/"]`.

A path listed here that no longer exists is an error row on the page, not a skipped check — a
renamed engine directory must not silently turn every verdict green.

## 7. CLI

```
16_dashboard.py [--out PATH] [--registry PATH] [--serve] [--port 8765] [--open]
```

Default writes `output/snowball_dashboard_latest.html`. `--serve` starts the local server and
does not write a file. `--open` launches a browser.

## 8. Testing

`test/mo_volmodels/test_dashboard.py`, pure functions only — the pattern
`test/mo_volmodels/test_gate_scope.py` already uses for gate logic.

1. `freshness` over all four outcomes, driven by a synthetic commit list; no real `git`
   invocation in unit tests.
2. Registry parsing: dir on disk absent from YAML becomes `unclassified`; missing YAML yields
   all-unclassified plus one error row, never an exception.
3. Cell grid versus manifest, fixtured on the real discrepancy from §1.2 — 35 cells on disk,
   27 in the manifest — asserting the tree walk wins and the 8 Jul-27 cells resolve to `void`.
4. Collector soft-fail: unreadable JSON yields `status: "unreadable"` with the message.
5. Payload contract: `schema_version` and required top-level keys present.

Render and serve get one smoke assertion each — non-empty HTML containing all three panel ids,
and `/api/fleet` returning valid JSON — consistent with `13_aggregate_and_report.py` not being
HTML-tested. Checks that need real artifacts skip when `output/` is absent, matching the
existing convention for tests that depend on the uncommitted history cache.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Registry goes stale, hiding real work | Unregistered dirs render in a visible `unclassified` strip (§4.3) |
| Dep table too narrow — a real change reads as fresh | Directory-level deps, not file-level; missing path is an error row (§6.3) |
| Dep table too broad — everything permanently stale | Deps are scoped to engine/gate paths, excluding tests and docs |
| Snapshot mistaken for live | `generated_at` and `mode` rendered in the header; `live` block absent in snapshots |
| Dashboard drifts from the study's own definitions | Variants and inceptions are imported from `cohort.py` and the spec, never restated in config (§5) |

## 10. Success criteria

1. One command produces a page that states, without reading any other file: each gate's verdict
   and whether it is fresh, stale or void; fleet coverage as admitted cells out of 162; and the
   next blocking action.
2. The eight Jul-27 `ts_bsm`/`localvol` cells render as **void**, not as progress.
3. Fleet coverage reads 27/162 — not the 35 cells the fleet dir holds on disk, and not the 38
   a naive sum of `runs_completed` across all six run dirs would give.
4. The dirty working tree is visible on the page.
5. `--serve` updates the fleet panel within ~10 s of a cell completing.
6. The dashboard never writes to `output/` outside its own HTML file.
