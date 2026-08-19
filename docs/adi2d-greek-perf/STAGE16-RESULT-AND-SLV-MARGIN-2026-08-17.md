# Stage-16 certification result, and why the SLV admission does not hold up

2026-08-17. Covers `output/p17_fixed` (published 13:09) and a defect found while
preparing the stage-17 follow-on.

## 1. What was published

Schema 13, profile `production`, `certification_mode: full_recertification`.

```
heston      route=pde   sampling_complete=True  evidence_complete=True
heston_slv  route=pde   sampling_complete=True  evidence_complete=True
```

14/14 cells PASS, 28/28 greek verdicts PASS, every cell at its full declared
allocation (Heston 1024 scrambles, `near_ki` 2048; SLV 128, `low_feller` 512).
`implementation c7142c38b5b64874`, `numerical b791e0dbfc7e1d4f`,
`evidence_sha256 a5c934062386ec44`. ~51 h wall: Heston half 20.6 h,
`heston_slv/low_feller` alone 24.5 h.

Aggregate signed delta bias, economic bound ±0.10 contracts:

| variant | estimate | interval | total unc. | margin to bound | scrambles |
|---|---|---|---|---|---|
| heston | −0.03272 | [−0.06297, −0.00248] | 0.03024 | 0.03703 (37.0%) | 1024 |
| heston_slv | −0.02888 | [−0.09526, +0.03750] | 0.06638 | **0.00474 (4.7%)** | 128 |

The Heston interval tightened 1.88× against the earlier gate-driven run and now
excludes zero: the signed bias is statistically real and economically
irrelevant at 3% of the bound. That is the certification doing its job.

The SLV row is the problem. It passes by 4.7% of the bound, and the estimate's
own standard error (0.01616) is **3.4× the remaining margin**.

## 2. The defect: the aggregate and the cell verdict disagree

Stage-16's aggregate gate aligns the seven regime cells by **truncating** each
one to the common scramble count (`16_adi_greek_certification.py`, in
`make_decisions`):

```python
row["batch_difference_contracts"]["delta"][:common_batches]
```

Two cells were deliberately allocated more than the common count, because the
pilot-frozen Neyman allocation identified them as variance-dominant:
`heston_slv/low_feller` banked 512 against a common 128, and `heston/near_ki`
banked 2048 against a common 1024. Truncation means those extra rows never
reach the aggregate gate — 384 rows on the single most expensive cell in the
fleet, 75% of a 24.5-hour computation.

The same payload therefore reports two different means of the same quantity for
the same cell:

| quantity | value | rows used |
|---|---|---|
| `heston_slv/low_feller` **cell verdict** difference | −0.159096 | all 512 |
| that cell's contribution to the **aggregate** | −0.048367 | first 128 |

Both are exact — verified by recomputation against the banked arrays. The cell
verdict uses all the evidence; the aggregate gate, which is what decides
admission, uses a quarter of it.

Re-deriving the aggregate from each cell's **full** banked evidence shifts it by
exactly `(−0.159096 + 0.048367)/7 = −0.015818`, entirely attributable to
`low_feller`:

| alignment | estimate | interval | total unc. | margin | status |
|---|---|---|---|---|---|
| truncated (published) | −0.028883 | [−0.095264, +0.037498] | 0.066381 | +0.00474 | PASS |
| pooled (all 512 rows) | −0.044701 | [−0.101487, +0.012084] | 0.056785 | −0.00149 | **INCONCLUSIVE** |

Pooling *lowers* the uncertainty (0.0664 → 0.0568, as more evidence should) and
still fails, because it also moves the point estimate to the more accurate
value.

### It is alignment luck, not a defect in the rows

`low_feller`'s four disjoint 128-scramble blocks each give a different verdict:

| block | rows | block mean | aggregate estimate | margin | status |
|---|---|---|---|---|---|
| 0 | 0–127 | −0.048367 | −0.028883 | +0.00474 | PASS ← published |
| 1 | 128–255 | −0.126185 | −0.040000 | −0.01681 | INCONCLUSIVE |
| 2 | 256–383 | −0.203314 | −0.051018 | −0.02040 | INCONCLUSIVE |
| 3 | 384–511 | −0.258518 | −0.058904 | −0.03360 | INCONCLUSIVE |

The published certificate landed on the one block of four that passes.

The monotone look of those block means prompted a check for structure in the
reference rows — a trend or autocorrelation would invalidate the outer standard
error itself. It does not survive: across all fourteen cells, OLS trend and
lag-1 autocorrelation produce two marginal p-values (`heston_slv/low_feller`
p=0.023, `heston_slv/sigma_collapse` p=0.025) and one marginal lag-1 z
(`heston/near_ki` z=−2.74, and *negative* autocorrelation makes the SE
conservative, not anti-conservative). Two or three marginal statistics out of
fourteen tests is what independence predicts. **The rows are sound; the gate is
simply marginal.**

### Which estimator is right

Truncation is not a bug in the narrow sense: it is declared, deterministic, and
unbiased, and `validate_payload` checks the declared `aggregate_common_scrambles`.
Its stated purpose is to keep the aggregate common-random-number aligned — every
cell contributes the *same* scramble index, so cross-case covariance is
preserved before the outer standard error is taken. Pooling breaks that
alignment for the over-allocated cell (it is also unbiased, keeps the outer rows
independent, and empirically has lower variance).

The real finding is that these two design choices are **incompatible**:

* per-cell Neyman allocation gives the variance-dominant cell more scrambles;
* a CRN-aligned common-scramble aggregate cannot use them.

So the allocation that cost 24.5 hours buys precision for `low_feller`'s own
verdict and, by construction, nothing for the aggregate gate that actually
decides admission. Resolving this needs an aggregate that combines unequal
per-cell allocations while accounting for the partial CRN overlap — a
statistical design change, not a one-line fix.

On the Heston side the same re-derivation *widens* the margin (37.0% → 39.9%),
so **the Heston admission is robust either way**. Only the SLV admission is at
issue.

## 3. Where the SLV uncertainty actually lives

Decomposition of the published SLV total uncertainty (0.066381):

| component | value | share |
|---|---|---|
| statistical half-width of the delta estimate | 0.036654 | 55% |
| reference substep bias envelope | 0.020577 | 31% |
| PDE discretization envelope | 0.009150 | 14% |

The substep envelope is `|measured mean| + its own half-width` =
`0.000408 + 0.020169`. The measured substep bias is essentially **zero**; 98% of
that envelope is the uncertainty in measuring it. So **86% of the SLV
uncertainty is statistical** and shrinks as 1/√n, against 55% for Heston at
1024 scrambles.

Brute force is therefore effective but expensive. To reach a Heston-like
comfortable margin (total ≈ 0.030, interval ≈ [−0.075, −0.015], 25% margin) the
statistical part must fall ~2.25×, i.e. roughly 5× more SLV scrambles — about
5× the SLV half of the fleet, on the order of 150 h. Not affordable.

## 4. Why stage-17 cannot run as written — and why it is needed after all

`17_adi_slv_aggregate_certification.py` exists to do exactly this job: cut the
reference variance with control variates instead of paths, at a declared
`--precision-target 0.02` statistical half-width inside a 12 h budget. A 0.02
statistical half-width implies total ≈ 0.030 — precisely the comfortable target
above.

Two things block it:

1. **Its pinned parent no longer exists.** It hard-refuses any parent whose file
   digest is not `3d4cb66b8fface3a…`; that artifact — a schema-13
   `incremental_amendment` from commit `b5463093` — is not on disk (lost in the
   2026-08-10 crash). Verified by hashing every `adi_greek_certification.json`
   under the repo.
2. **Its parent contract demands an unresolved gate.** It requires
   `certification_mode == "incremental_amendment"`,
   `profile == "production incremental amendment"`, and
   `decisions.heston_slv.route == "excluded_greek_unresolved"`
   (lines 594–655), and it exists to flip that route to `pde` (line 1177). The
   published parent says `route=pde` already, so the amendment refuses it.

Blocker 2 is the script being right for the wrong reason: against the
*published* aggregate there is nothing to resolve, and against the *correct*
aggregate there is.

## 5. Recommended sequence

1. **Decide the aggregate estimator.** Either keep CRN-aligned truncation and
   accept that per-cell over-allocation cannot help the aggregate, or design an
   unequal-allocation aggregate. This is the decision that gates everything
   else.
2. **Re-publish stage-16 from the banked checkpoints** under whichever estimator
   is chosen. Note the cost: `make_decisions` is currently **inside** the
   numerical projection (`NON_NUMERICAL_SYMBOLS` exempts `validate_payload` and
   `make_amendment_decisions` but not `make_decisions`), so editing it moves
   `numerical_implementation_sha256` and invalidates all fourteen banked cells —
   a 51 h re-run for a publish-time change. Classifying `make_decisions` as
   non-numerical (category 1, "runs after the numbers exist") makes this a
   seconds-long re-publish instead. That widens the exemption list and is a
   provenance judgment call, not a mechanical fix.
3. **Then stage-17**, re-pinned to the resulting parent, to resolve the SLV gate
   with control variates in ~12 h rather than ~150 h of extra paths.

## 6. Status of the certification

* **Heston: admitted, robust.** 37–40% margin under either alignment.
* **Heston-SLV: not established.** The published `route=pde` rests on the one
  128-scramble alignment out of four that passes; the best estimate from all
  banked evidence puts the interval across the bound. It should not be relied on
  until the aggregate estimator is settled and the gate re-run.

## Reproducing

```bash
PYTHONPATH=$PWD .venv/bin/python \
  docs/adi2d-greek-perf/probes/probe_aggregate_pooling_headroom.py
```

The probe reproduces the published gate bit-for-bit before reporting any
alternative, and fails if it cannot. Output: `output/aggregate_pooling_headroom/`.

## 7. Resolution (later the same day): strided pooling, re-published

The §5 sequence was executed on 2026-08-17 (commits `258fd7e`, `ae460f8`).

**Estimator decision: strided pooling.** Stage-17's consecutive grouping was
measured first (`probe_crn_strided_alignment.py`) and found to have its own
defect: with cross-case CRN coupling, outer row j carries the common cells at
scramble j while the over-allocated cell's scramble-j row lands in outer row
floor(j/g), so the coupling straddles outer-row boundaries and the empirical
SE estimates the variance as if the over-cell covariance were zero. Grouping by
STRIDE instead — output row j averages scrambles {j, j+m, j+2m, ...} — keeps
every same-scramble coupling inside one output row, leaves the rows i.i.d.,
and its empirical variance matches the partial-CRN-overlap plug-in formula.
Measured on the banked fleet: the coupling is real and negative (corr −0.43 on
`heston/low_feller × near_ki`, −20.5% of the heston aggregate variance);
consecutive matches its as-if-zero prediction to 0.4%, strided matches the
full formula to 2.7%. Same point estimate as pooling; only the SE construction
differs, and strided is the one that is exactly right.

**Reclassification and re-publish.** `make_decisions` moved into
`NON_NUMERICAL_SYMBOLS` (category 1 — it reads banked evidence and renders
verdicts). `probe_numerical_projection_equivalence` proved the base and edited
trees project identically under the new list; `restamp_p18_strided.py` then
re-keyed a copy of the fleet after rebuilding all fourteen banked identities
bit-for-bit and replaying the live configuration. The resume re-publish
(`output/p18_strided`) took seconds, resumed all 15 checkpoints, and every
gate component is bitwise-equal to the probe's validated numbers. The
alignment is declared in the artifact (`aggregate_alignment: strided_pooled`,
per-case banked counts) and enforced by `validate_payload`.

| variant | estimate | interval | total unc. | margin | route |
|---|---|---|---|---|---|
| heston | −0.032170 | [−0.059049, −0.005290] | 0.026879 | 41.0% | **pde** |
| heston_slv | −0.044701 | [−0.102191, +0.012788] | 0.057489 | −2.2% | **excluded_greek_unresolved** |

**Stage-17 re-pinned.** Parent constants now point at `output/p18_strided`
(commit `258fd7e`); the full-recertification parent carries pinned per-cell
identities in place of the amendment-only cell_provenance/auxiliary_controls/
aggregate_cohorts projections, and `_group_delta_rows` uses the same strided
grouping as the parent aggregate. Parent validation verified green. The
parent's unresolved SLV route is now genuinely the gate stage-17 exists to
flip.

**Still open before a production stage-17 run.**

1. `PRODUCTION_ALLOCATION_FROZEN = True` carries an allocation frozen from a
   development pilot run against the *lost* parent. The physical variances it
   measured did not change, but the freeze's lineage did: either re-run the
   non-admissive development pilot against the new parent, or record an
   explicit waiver. This is the §5-style decision that gates the production
   run.
2. The production run itself (~12 h) with `--precision-target 0.02`; if the
   true aggregate bias is near the current best estimate (−0.045), it admits
   SLV at ≈25% margin; if the truth sits near ±0.07, the honest conclusion is
   exclusion.

## 8. Closure (2026-08-19): SLV ADMITTED — program complete

Both open items resolved.

**Re-pilot (2026-08-17, 3h50m)** — and it was substantively necessary, not just
auditability: reconstruction against the pinned hashes proved the lost
`b5a5243` pilots measured **pre-bridge8** estimators for
`ordinary_full`/`ordinary_decayed`/`sigma_collapse` (bridge dims 1 → 8 shipped
after they ran; `low_feller`, whose config never changed, reproduced
bit-for-bit). The old frozen allocation was stale on three of six control
cells. Fresh pilots + the strided-parent projection re-froze an **8× smaller**
plan (commit `0f13d93`): primary 512 × 1024, middle 32 × 8192, smooth Heston
weight 1.0, 8,388,608 unique paths; projected interval [−0.081, −0.008],
guarded [−0.092, +0.003], both inside the bound.

**Production amendment (launched 2026-08-17 21:37, published 2026-08-19 03:22,
28.6 h)** — held-out seeds 20260811/12 opened for the first time, fixed
allocation, no optional stopping, 1.16 × 10⁹ conditional path valuations.
Verdict:

- D + S = −0.029994 ± 0.024242, D − S = −0.027270 ± 0.018171
  (paired within each seed family, independent-cohort Welch-t across families,
  ≥95% simultaneous coverage)
- simultaneous interval **[−0.063387, +0.003398]**, bound ±0.10 →
  **PASS at 36.6% margin**
- `heston_slv route = pde`; Heston carried at `pde`. Artifact:
  `output/p18_slv_amendment` (evidence `84a16fca…`, parent `5c2bd579…`
  byte-linked, nothing rerun).

The held-out estimate (≈ −0.029) landed inside the development guard and
better-centered than projected. Final state of the certification: **both
variants admitted** — Heston at 41.0% margin (strided stage-16 re-publish,
`output/p18_strided`), Heston-SLV at 36.6% margin (this amendment) — with the
full lineage (parent, pilots, allocation freeze, seeds) hash-chained and
re-derivable end to end.
