# MC Reference-Stack Convergence — Variance Reduction + Adaptive Stopping

**Date:** 2026-08-10 · **Status:** APPROVED (design) · **Branch:** `codex/adi-greek-certification`
**Companions:** `docs/adi2d-greek-perf/STATE-AND-PLAN-2026-08-10.md` (program plan; this spec refines P1.3/P1.4 mechanics), `docs/adi2d-greek-perf/implementation-spec.md` (ADI-side WS-A..G), `docs/adi2d-greek-perf/RECOVERY.md` (2026-08-10 evidence loss)

## 1. Problem

The Heston-SLV admission gate is starved by MC reference noise, and the machine's RAM
caps brute force. Schema-11 measured (2026-08-07 analysis, recovered):

- Aggregate signed-delta verdict INCONCLUSIVE: estimate −0.0527, total uncertainty
  0.0867 vs the 0.0473 needed at that estimate; **63% of the interval width is MC
  statistical noise**.
- That noise is concentrated in three cells that never received the variance-reduction
  treatment: `ordinary_decayed` (37.5% variance share), `ordinary_full` (36.4%),
  `sigma_collapse` (15.8%). Treated `near_ki` (multilevel Heston-control + bridge 8×8 +
  16 substeps) has batch SD **0.19 vs 1.08** for untreated cells (~33× variance
  reduction) — variance follows treatment, not regime difficulty.
- The frozen production allocation (4096 primary / 256 middle batches, control weight
  0.7) was sized to brute-force untreated noise, and runs only 4+2 workers on the
  48 GiB / 14-core machine — days of wall-clock.

The 2026-08-10 crash destroyed all banked references; they must be regenerated anyway
(after the WS-C v-drift scheme fix, so regeneration happens exactly once). This is the
moment to make the reference estimators converge fast instead of long.

## 2. Goals

- **G1 (decisive overnight):** schema-12 reference regeneration + Stage-17 aggregate
  run completes in ≤ 12 h on this machine with aggregate statistical half-width
  ≤ 0.02 contracts (pre-registered precision target; decisive at the expected ~−0.03
  estimate against the ±0.10 bound). "Statistical half-width" keeps the schema-11
  convention: t₀.₉₇₅,df × SE under the cohort-mean CLT.
- **G2 (unbiased by construction):** every estimator change ships with a matched-budget
  unbiasedness demonstration against the plain estimator; adaptive machinery never
  reads the estimate.
- **G3 (evidence semantics preserved):** the final verdict is the unchanged
  fixed-confidence equivalence test in `quantark/validation/greek_certification.py`;
  cohort bookkeeping keeps `sum_of_independent_cohort_means`.

## 3. Non-goals

- No memory/dtype/streaming work (descoped by user 2026-08-10; revisit only if the
  overnight target misses).
- No importance sampling (nothing in the variance budget indicates rare-event tails).
- No group-sequential / alpha-spending early-verdict stopping (precision-based rule
  chosen instead — see §5).
- No changes to fleet/backtest MC engines this cycle (reference stack only; wins port
  later under their own spec).
- No PDE-side changes (owned by the ADI program).

## 4. WS-V — Variance reduction (per-path efficiency)

### V1 Treatment extension to the untreated cells

Extend the proven stack — conditional-snowball Rao-Blackwellization
(`quantark/montecarlo/conditional_snowball.py`: analytic integration of KO/KI intervals
and the terminal short put on the scalar conditioning normal Z) + RQMC bridge — from
`near_ki` to `ordinary_full`, `ordinary_decayed`, `sigma_collapse`, for both `heston`
and `heston_slv` variants.

All certification cells are `SnowballOption`s built by `make_snowball`, which is the
estimator's supported product family; its deliberate fail-closed validation is
exercised per cell by the V-G1 demo (a cell that fails closed falls back to the plain
estimator and is reported as such, never silently approximated).

**Gates**
- V1-G1 unbiasedness (per cell × variant): |treated − plain| ≤ 2 × combined SE at a
  matched pilot budget; fail-closed guards exercised in the demo.
- V1-G2 efficiency: measured SD-reduction factor ≥ 4× per treated cell (planning
  number is 8×; 33× is the measured ceiling on near_ki). Record the factor in the
  decision matrix whatever it is.
- V1-G3 cost honesty: per-batch wall-clock of the treated estimator recorded alongside
  (Rao-Blackwellization adds per-path analytic work; the metric that matters is
  SE² × seconds, not SD alone).

### V2 Per-cell control weights (`heston_slv` variant only)

The Heston control exists only on the `heston_slv` side (the control *is* the Heston
analog; `heston` cells have no outer control). Replace its frozen global weight 0.7
with per-cell weights estimated by cross-fitting: batches split into halves, weight
fitted on each half applied to the other, so the combined estimator stays unbiased.
Weights and their fit diagnostics are recorded in the evidence.

**Gates**
- V2-G1: cross-fitted estimator agrees with fixed-0.7 within combined SE (unbiasedness).
- V2-G2: variance not worse than fixed-0.7 on any cell; report the per-cell gain.

### V3 Stratification (measure-first, optional)

Only if the post-V1/V2 pilot still shows one cell dominating the aggregate variance:
stratify on the bridge terminal coordinate for that cell. Ships only with its own
V1-style unbiasedness + efficiency demo. Explicitly cut from the critical path.

## 5. WS-S — Adaptive allocation & precision-based stopping

Two-stage, pre-registered, **estimate-blind**. Chosen over group-sequential designs
because a stopping rule that reads only achieved precision (never the estimate) adds
no optional-stopping bias and needs no boundary machinery.

- **S1 pilot:** fixed 32 batches per cell × variant under the WS-V estimators, seeds
  pre-registered in the schema. Yields per-cell batch variance and per-batch cost.
- **S2 allocation:** cost-weighted Neyman allocation over cells — minimize projected
  aggregate SE per wall-clock second, computed from the pilot only, frozen and recorded
  in the evidence before the main run starts. (The pilot batches are kept: they are the
  first cohort of the main run, not discarded.)
- **S3 main run + monitor:** banking proceeds per allocation; a monitor extending
  `qmc_rqmc_driver`'s stopping logic recomputes projected aggregate SE per completed
  cohort and halts when it reaches the precision target (0.02 contracts) or the budget
  cap (12 h). The monitor's inputs are per-cell SEs and costs only.
- **S4 verdict:** computed once, after stopping, by the unchanged equivalence test on
  everything banked.

**Gates**
- S-G1 blindness: the monitor's implementation takes SEs/costs as inputs; a test
  asserts the estimate is not reachable from the stopping path (interface-level, not
  convention-level).
- S-G2 allocation sanity: on a synthetic 3-cell fixture with known variances, the
  Neyman allocation reproduces the analytic optimum within discreteness.
- S-G3 reproducibility: pilot + allocation + stopping trigger fully determined by the
  recorded seeds and schema fields — a re-run reproduces the same allocation decision.
- S-G4 fallback: if the monitor is disabled, the run degrades to the frozen 4096/256
  allocation (which remains hash-pinned as the budget-cap fallback).

## 6. Evidence & schema (schema-12)

New recorded fields: per-cell treatment descriptor (estimator id + parameters +
fail-closed outcome), cross-fitted control weights, pilot cohort (seeds, variances,
costs), frozen allocation decision, precision target, stopping trigger
(target-reached | budget-cap), achieved per-cell and aggregate SE. The allocation
decision is hash-pinned exactly as the frozen allocation was. Cohort structure remains
`sum_of_independent_cohort_means` with per-cohort seeds.

## 7. Delivery order (standalone-demo-first, house rule)

1. **D1** demos: V1 per cell (6 demos: 3 cells × 2 variants) + V2 — each producing the
   three numbers (unbiasedness, SD factor, sec/batch) against the recovered anchors
   (`docs/adi2d-greek-perf/recovered/`, e.g. ordinary_full reference delta/SE).
2. **User decision matrix**: pick which treatments ship per cell (a cell whose demo
   underdelivers can stay plain — the allocation step absorbs it).
3. **D2** harness integration: stage-16/17 wiring + schema-12 fields + S1–S4, gated
   S-G1..G4, unit tests beside the existing certification tests.
4. **D3** production sequence (unchanged from STATE-AND-PLAN): WS-C scheme fix lands →
   schema-12 regeneration with this machinery → Stage-17 aggregate run → admission
   decision.

Effort: V1 M (the estimator exists; per-cell wiring + demos dominate), V2 S, S1–S4 M,
schema fields S.

## 8. Feasibility arithmetic

Untreated, a 0.02-contract aggregate half-width needs ~950 batches per noisy cell
(SE ∝ 1/√n from the 0.0545 @ 128-batch measurement). At the planning 8× SD reduction,
the same contribution needs ~15 batches — the 32-batch pilot alone nearly saturates
treated cells, and the budget concentrates on whatever the pilot exposes as slowest.
The overnight target holds with ≥ 2× margin against V1 underdelivering. If treatments
underdeliver badly enough that target and cap conflict, the cap wins: the run stops at
12 h, records `stopping_trigger = budget-cap`, and the verdict uses whatever precision
was achieved — honestly reported, never silently extended.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Conditional estimator fails closed on a cell's product features | that cell stays plain | V1-G1 demo surfaces it early; allocation absorbs plain cells |
| Rao-Blackwellization cost eats its variance win | wall-clock target slips | V1-G3 gates on SE²×seconds, not SD |
| Pilot variance estimates noisy at 32 batches | suboptimal allocation | Neyman is flat near the optimum; monitor corrects by stopping on achieved SE |
| Estimate leaks into stopping | biased verdict | S-G1 interface-level blindness test |
| Superseding the hash-pinned frozen allocation | evidence-continuity questions | frozen allocation kept as recorded fallback; supersession reasoning recorded here and in the evidence |

## 10. Decisions log

- 2026-08-10 scope: certification reference stack first (user).
- 2026-08-10 target: overnight ≤ 12 h, decisive interval (user).
- 2026-08-10 focus: variance reduction + adaptive stopping; memory/dtype work descoped (user).
- 2026-08-10 stopping semantics: precision-based, estimate-blind; no alpha-spending (user).
