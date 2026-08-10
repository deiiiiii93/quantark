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

1. **P1.1 `low_feller` attribution probe — DONE, and it is not scheme-driven.**
   The cell has zero non-monotone centered rows at production grids (max local
   Péclet 0.543), so `adaptive_upwind` already *is* `centered` there and no
   v-axis scheme can move it. Demoted to quality item P1.1b; not a P1.4 blocker.
   See §7.
2. **P1.2 WS-C semi-Lagrangian v-transport — DONE 2026-08-10** (commit 46d8d63).
   Shipped as opt-in `v_drift_scheme="semi_lagrangian"` in `adi_core` plus the
   solver-layer validator; 18 tests; measured evidence in
   `docs/adi2d-greek-perf/logs/wsc_semi_lagrangian.log`.
   Measured `n_v` refinement ratios: **2.03 / 1.98** for `adaptive_upwind`
   (first-order, exactly as the probe predicted) vs **113 / 153** for SL, both
   schemes agreeing at fine `n_v`. SL at `n_v=60` beats upwind at `n_v=240` on
   accuracy while running 2.6× faster; equal-grid overhead is 1.30×. Default path
   bitwise unchanged (104,960 bytes of full-march surfaces; 136 PDE tests + 28
   replay goldens green).
   C-G1/C-G2/C-G3 in their *certification* form (against banked MC references) and
   the C-G6 default-flip decision remain open — they need P1.4's references.
3. **P1.3 MC variance amendment — BUILT 2026-08-10**, superseded in detail by
   `docs/superpowers/specs/2026-08-10-mc-reference-convergence-design.md` and its plan.
   Delivered: `bridge8` treatment shipped on all three variance-dominant cells
   (measured unbiased at 0.28–0.65σ, free, 2.14× / 2.62× / 1.49× in SE²·seconds),
   schema 11 → 13 with per-cell `reference_treatment` descriptors, cross-fitted control
   weights (`quantark.montecarlo.control_weights`), estimate-blind Neyman allocation and
   precision stopping (`quantark.validation.adaptive_allocation`), and stage-17
   `--adaptive` sizing with the frozen 4096/256 counts retained as a floor.
   Measured feasibility: the 0.02-contract target needs ~3.8 h wall-clock at measured
   rates (7.6 h at a pessimistic ×2 contention), 7 cells concurrent at ~33 GiB of 48 —
   **spec goal G1 holds**.
4. **P1.4 Regenerate references once, at the corrected scheme** (this also rebuilds the
   lost banked fixtures for every later re-cert gate): stage-16 full run → schema-13
   evidence; then **Stage-17 production aggregate run** (`--adaptive`, target 0.02,
   cap 12 h). Target: Heston-SLV aggregate PASS → **both PDE routes admitted**.

   **Blocked until this runs** (recorded 2026-08-10): 8 stage-17 tests fail on a clean
   tree because the crash destroyed `output/adi_greek_certification_schema*/` — they
   need a parent evidence + decision pair. For the same reason the adaptive path could
   not be rehearsed end-to-end; `run_production_amendment` refuses without
   `--parent-evidence`/`--parent-decision`. The adaptive machinery is unit-tested (5
   tests incl. the S-G1 blindness and S-G3 determinism gates) but has never executed
   against a real parent. First live use must therefore be treated as a rehearsal:
   run stage-16 first, then stage-17 with a loose `--precision-target` before the real
   one.

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
| D-0 | Sequencing: scheme fix before MC regeneration | **RESOLVED yes** — and it had to be, see §7 | done 2026-08-10 |
| D-1 | WS-A2 packaging: pure wheel + local C build vs platform wheels | pure wheel + local build | before A2 merge |
| C-G6 | Flip auto classifier to SL for `path_focused` regime (default change) | **RESOLVED: shipped as `v_drift_scheme="auto"`** (`908588c`), not a blanket SL flip | done 2026-08-10 |
| D-2 | Ladder API: public `build_greek_ladder` engine method vs serving-side class | public engine method | before G2 |
| D-3 | Should Stage-17 production run wait for P1.2+P1.3, or run a pilot on the interim n_v-law lever | wait; pilot only if admission is urgent | Phase 1 |

## 7. The scheme decision, resolved 2026-08-10

D-0 and C-G6 turned out to be the **same** decision, not a sequence. The
certification pins its controls dict equal to stage-11's and stage-12's
(`test_stage16_parent_controls_match_stage11_and_schema12_router`), so one scheme
cannot be certified while another ships. And `IMPLEMENTATION_INPUTS` covers both
`adi_core.py` and the stage-16 script, so any later scheme change invalidates
every checkpoint and forces a second full MC regeneration. The scheme therefore
had to be final *before* P1.4, not after it.

The 42-solve cross-scheme matrix and the operator diagnostics
(`probes/results-2026-08-10/SCHEME-CELL-MATRIX.md`) settled it. Only
`sigma_collapse` disagrees between schemes (+0.1149 heston, +0.1119 SLV, against
a recorded bias of −0.112 ± 0.010 — disjoint machinery agreeing to 3%). Every
other cell moves ≤0.003 contracts, and there `adaptive_upwind` is *already*
`centered` to every printed digit, so uniform SL would have been slightly
**less** accurate on 12 of 14 cells.

The reason is that the donor-cell fallback engages for two unrelated causes: a
coordinate singularity at v→0 (2 nodes at v ≤ 1.4e-05, Péclet diverging at any
resolution, no probability mass) versus genuine convection dominance across the
live domain (132/133 nodes, v = 0.00129…0.482, containing θ and v0). Shipped
`v_drift_scheme="auto"` separates them on *where* the non-monotone rows sit
relative to `min(v0, θ)` — clearing by 2857× and 158×, so nothing is tuned.

At production grids: **12 of 14 cells bitwise unchanged, both `sigma_collapse`
cells on transport, 1.00× cost on normal books.**

`low_feller` has **zero** non-monotone rows, so no v-axis scheme can move it.
That closes P1.1: its −0.107/−0.159 is the v=0 boundary treatment or the MC
reference side, a quality item (P1.1b), not an admission blocker.

**P1.4 must run `--full-recertification`**: WS-C legitimately moved the PDE
surface, so the schema-9 carry path is closed (`5884fb2`).

## 6. Risks

Beyond the `implementation-spec.md` risk register: (i) `low_feller` bias may *not* be
scheme-driven — P1.1 resolves this before it can mislead P1.4; (ii) regenerated
references shift all downstream gate anchors — regenerate exactly once, after the scheme
decision; (iii) the recovered demo scripts reference the dead `/private/tmp` path — set
the repo root per `RECOVERY.md` before reproducing research numbers.
