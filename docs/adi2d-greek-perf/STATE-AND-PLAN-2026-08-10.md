# 2D PDE Engines → Production: Consolidated State & Plan

**Date:** 2026-08-10 · **Branch:** `codex/adi-greek-certification` (worktree `.worktrees/adi-greek-certification`) · **Companion docs:** [`README.md`](README.md) (measured research), [`implementation-spec.md`](implementation-spec.md) (WS-A..G), [`RECOVERY.md`](RECOVERY.md) (2026-08-10 data loss)

## 1. Program goal

Make the 2D ADI PDE engines (Heston + Heston-SLV snowball) the production route for
Greeks: **certified** (statistically equivalent to MC references under explicit economic
bounds), **fast enough** for the backtest fleet and desk hedging, and **admitted** into
the G4/G1/G2 snowball-backtest gates.

## 2. Where the work stands (the reorg)

Three threads were running in parallel; the 2026-08-10 crash cut across all of them.

### 2.1 Certification framework — committed, safe (Codex, `ee0d5f0..14234fd`)

~15.7k lines on this branch: the Stage-16 per-cell Greek certification harness
(7 regime cells × heston/heston_slv, PDE vs banked MC references, schema'd evidence with
checkpoints), Stage-17 aggregate-only SLV certification with a **frozen, hash-pinned
production allocation** (4096 primary / 256 middle batches, Heston control weight 0.7),
`quantark/validation/greek_certification.py` (equivalence verdicts: PASS/FAIL/INCONCLUSIVE
with uncertainty budgets), MC variance-reduction infrastructure
(`quantark/montecarlo/{conditional_snowball,qmc_qe_coupling,qmc_rqmc_driver}`), the
`v_drift_scheme` engine knob, and ~3.7k lines of tests.

### 2.2 Certification verdicts — schema-11 (evidence files lost, verdicts recovered)

| Route | Per-cell | Aggregate mean signed delta bias (bound ±0.10) | Verdict |
|---|---|---|---|
| Heston PDE | 7/7 PASS delta+gamma | −0.0479, interval [−0.0843, −0.0115] | **ADMITTED** |
| Heston-SLV PDE | 7/7 PASS delta+gamma | −0.0527, interval [−0.1394, +0.0340] | **INCONCLUSIVE** → excluded |

The INCONCLUSIVE decomposes into two separable problems (2026-08-07 analysis):

1. **Noise (63% of interval width):** MC reference noise concentrated in the three cells
   never given the variance-reduction treatment (`ordinary_decayed` 37.5%,
   `ordinary_full` 36.4%, `sigma_collapse` 15.8%). `near_ki` — with multilevel
   Heston-control + bridge — has batch SD 0.19 vs 1.08 for untreated cells (~33×
   variance reduction, proven machinery).
2. **Structural bias (the estimate):** concentrated in `sigma_collapse`
   (heston −0.112 ± 0.010, 11σ) and `low_feller` (−0.107 ± 0.031), same signs in SLV
   (−0.140, −0.159); `sigma_collapse`'s n_v axis not converged at production floors
   (first-order signature).

### 2.3 Perf & precision research — WS-A..G (recovered into this package)

22 standalone demos, all measured, none merged (see `README.md` §1–§14 and
`implementation-spec.md`): C Thomas kernel 2–2.7× bitwise (A), regime-aware floors
4.4× (B), semi-Lagrangian v-transport killing the σ-collapse first-order v-axis error
(C), KI-band mesh 1.8× (D), time-Richardson pairs (E), desk grid profile ~4 s/position
+ `SnowballLadderCache` sub-ms intraday marks (G).

### 2.4 In-flight at crash time — both relaunched/replannable

- **Codex production certification run** (Stage-17, frozen allocation) — killed at PID
  6864; needs schema-11 parent evidence regenerated first (Codex's own assessment).
- **Delta-bias attribution probes** (Claude session `3b8441f4`) — four processes killed
  ~15 min before completion; the decisive `heston/sigma_collapse` probe was **relaunched
  2026-08-10** from `probes/probe_delta_attribution.py` (results in §3).

### 2.5 Lost in the crash

Banked stage-16 MC reference checkpoints, schema-11 evidence JSON/report, stage-17
pilots/projections/checkpoints, package `logs/`. Inventory + prevention policy in
`RECOVERY.md`. Source code lost: none.

## 3. The decisive experiment — 2026-08-10 probe results

`heston/sigma_collapse`, production grid 300×135×4800 unless stated; gaps in futures
contracts of delta vs the recovered schema-11 MC reference (SE 0.010 contracts):

| Row | Grid / scheme | Delta gap (contracts) | Reading |
|---|---|---|---|
| target (fidelity) | 300×135×4800, adaptive_upwind | −0.11236, **bitwise = banked** | recovery + reproducibility proven |
| centered | same grid, centered | **+0.00593** | the entire gap is upwind truncation |
| nv270 | n_v 270, adaptive_upwind | −0.05238 (predicted −0.053) | first-order law confirmed |
| nv540 | n_v 540, adaptive_upwind | −0.02227 (predicted −0.023) | first-order law confirmed |
| joint2x | 600×270×9600, adaptive_upwind | −0.05223 | vs nv270's −0.05238: n_x/n_t cross-terms ≤ 0.0002 — the n_v axis owns the error |

**Conclusion:** the σ-collapse bias is a PDE v-axis discretization artifact with a
clean first-order law `E(n_v) ≈ 16/n_v`, exactly as hypothesized.

`heston/low_feller` movement probe (same day; no banked fidelity anchor — movements are
exact, absolute gaps anchored on the recorded −0.107 ± 0.031):

| Row | Grid / scheme | Δdelta vs target (contracts) | Implied gap |
|---|---|---|---|
| centered | 300×135×3200, centered | **+0.00000 (exactly)** | −0.107 |
| nv270 | n_v 270 | +0.02558 | −0.081 |
| joint2x | 600×270×6400 | +0.03325 | −0.074 |

The adaptive upwind scheme already selects centered weights everywhere in this regime —
**low_feller's bias is not scheme-driven.** Only ~30% is grid truncation; the ~−0.074
residual points at the v=0 boundary treatment in this strongly Feller-violated regime
(compare volmodels-P5 WS-C3 `degenerate_pde`: 4.4× error cut in Feller-violated cells)
or at the MC reference side. It needs its own diagnosis (P1.1b).

**Aggregate arithmetic:** removing the SLV sigma_collapse scheme bias (−0.140 → ≈0)
moves the SLV aggregate estimate by ≈ +0.020; low_feller's grid component adds ≈ +0.005
→ estimate ≈ −0.028. With the schema-12 noise treatment halving the statistical
interval, the aggregate lands well inside ±0.10 even if low_feller's residual is real —
the two fixes together are sufficient for admission; the low_feller residual then
becomes a quality item, not a blocker.

**Consequence for sequencing:** regenerating days of MC evidence *before* fixing the
scheme would certify a knowingly-biased PDE and likely land INCONCLUSIVE again. Scheme
fix first, then regenerate references once, then certify.

## 4. The plan

### Phase 0 — Secure the ground (done 2026-08-10)

- ✅ Research package + probe + recovered anchors committed (`0515025`).
- ✅ Decisive probe re-run; upwind hypothesis confirmed (§3).
- ✅ Policy: all long-lived artifacts under `/Users` (repo `output/`), never `/private/tmp`;
  archive certification checkpoints as dated tarballs post-run.

### Phase 1 — Kill the structural bias (engine work, this branch)

1. **P1.1 `low_feller` attribution probe** (~20 min compute): same row matrix. If it
   also collapses under centered/SL → same fix covers it; if not, it needs its own
   diagnosis (Feller-boundary treatment) before P1.4.
2. **P1.2 WS-C semi-Lagrangian v-transport** as opt-in `v_drift_scheme="semi_lagrangian"`
   per `implementation-spec.md` WS-C (prototype `scripts/lever2_sl_vtransport.py`),
   gates C-G1..C-G5, both families. SL rather than bare `centered` as the *default
   candidate* because centered loses the M-matrix/monotonicity guarantee the upwind
   scheme exists for; SL gets flatness-in-n_v with monotone interpolation.
   *(Interim lever available immediately: `sigma_collapse`-regime runs at n_v 270/540
   quantify remaining bias by the measured 16/n_v law — useful for any urgent re-run,
   not the production answer.)*
3. **P1.3 Schema-12 variance amendment**: extend the proven multilevel Heston-control
   estimator from `near_ki` to `ordinary_decayed`, `ordinary_full`, `sigma_collapse`
   (63% of the interval width). Framework exists; this is configuration + banking compute.
4. **P1.4 Regenerate references once, at the corrected scheme** (this also rebuilds the
   lost banked fixtures for every later re-cert gate): stage-16 full run → schema-12
   evidence; then **Stage-17 production aggregate run** with the frozen allocation.
   Target: Heston-SLV aggregate PASS → **both PDE routes admitted**.

### Phase 2 — Perf program to production (WS sequence, per implementation-spec)

Backtest consumer: **A1 → B → E1 → A2 → C-G6 → D**; desk consumer: **A1 → A2 → G1 → G2**.
A1 (bitwise numpy Thomas, 1.3×) can start immediately and speeds up P1.4 itself; every
re-cert gate (B-G1, C-G6, G1-G1) consumes the Phase-1 regenerated references.

### Phase 3 — Consume

G4/G1/G2 snowball-backtest re-run with the admitted engine set (the 0.4.0 re-baseline
consumer); dashboard reflects admission state.

## 5. Decision register (user checkpoints)

| ID | Decision | Recommendation | When |
|---|---|---|---|
| D-0 | Sequencing: scheme fix before MC regeneration | Yes — probe data (§3) makes this clear-cut | now (this doc) |
| D-1 | WS-A2 packaging: pure wheel + local C build vs platform wheels | pure wheel + local build | before A2 merge |
| C-G6 | Flip auto classifier to SL for `path_focused` regime (default change) | separate PR after C-G1..G5 + full re-cert | after P1.4 |
| D-2 | Ladder API: public `build_greek_ladder` engine method vs serving-side class | public engine method | before G2 |
| D-3 | Should Stage-17 production run wait for P1.2+P1.3, or run a pilot on the interim n_v-law lever | wait; pilot only if admission is urgent | Phase 1 |

## 6. Risks

Beyond the `implementation-spec.md` risk register: (i) `low_feller` bias may *not* be
scheme-driven — P1.1 resolves this before it can mislead P1.4; (ii) regenerated
references shift all downstream gate anchors — regenerate exactly once, after the scheme
decision; (iii) the recovered demo scripts reference the dead `/private/tmp` path — set
the repo root per `RECOVERY.md` before reproducing research numbers.
