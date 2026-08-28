# Pilot results — snowball-localvol-1d certification

Gates the certification run for `LocalVolSnowballPDESolver`. Nothing is banked
until every control here passes.

Machine: `arm64` / macOS — Python 3.11.8, NumPy 2.4.6.
Spec: `docs/superpowers/specs/2026-08-28-localvol-1d-pde-certification-design.md` §7.

---

## Controls 3 and 5 — 2026-08-28

```
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py
```

### Control 5 — economic scale on both surfaces

The two surfaces sit at different index levels while `economic_scale` is a
single study-level block. Uncorrected, every calm-surface error would be
overstated by `6207.268 / 4993.105 = 1.243` — which *inflates* a measured error
and so risks a false `REJECTED`, not a merely conservative pass.

```
delta_quantum = 1.000000000   (must be 1.0)

           surface         s0  contract_mult   reported       true     ratio
------------------------------------------------------------------------------
  crash 2024-02-08   4993.105       1.000000   1.000000   1.000000  1.000000
  calm  2023-11-15   6207.268       0.804397   0.804397   0.804397  1.000000
```

**PASS.** The computed `contract_multiplier = REFERENCE_SPOT / s0` converts a
known raw delta to the intended contract count on both surfaces, to 1e-6. The
correction is structural — derived in `resolve_product_spec` from the artifact's
own `s0` — so it cannot be mistyped in the study YAML.

### Control 3 — flat-surface collapse

Flatten the crash surface to its longest ATM pillar and the local-vol PDE must
collapse onto the flat-BSM PDE. This separates "the input is wrong" from "the
formula is wrong", and is the control that settled the original diagnosis.

```
flat vol = 0.294545   (longest ATM pillar)

quantity               LV              BSM          rel
--------------------------------------------------------
   price    4788.76486857    4788.76486857    1.083e-13
   delta       0.56359637       0.56359637    2.486e-13
   gamma      -0.00067065      -0.00067065    7.106e-12
```

**PASS, and far more strongly than the criterion required.** The pass bound was
`delta rel < 1e-3`, allowing for the two solvers aligning their grids
differently. The measured agreement is **1e-13** — float round-off. On a flat
Dupire surface the two code paths produce the same numbers to machine
precision, so the local-vol machinery contributes no discretization difference
of its own; it reuses the identical grid and time-stepping.

Worth recording because of what it rules out: any residual disagreement on a
*real* surface cannot come from the solver skeleton. It has to come from the
surface read or from the reference — which is exactly where
FINDING-2026-08-26 found it.

---

## Surface steepness — RECONCILED, 2026-08-28

Spec §10's open item. `FINDING-2026-08-26` reports `dσ/dlnS = −0.371` and
`dσ/dt = −0.082` for 2024-02-08; direct measurement reproduced neither. Sweeping
the definition space settles it.

```
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_steepness.py
```

**The FINDING's two slopes are measured on two different surfaces.**

| slope | reconciled definition | measured | FINDING | err |
|---|---|---|---|---|
| `dσ/dlnS` | **Dupire local vol**, at the long end (`t ≈ max_listed_T = 0.866`), narrow window (±0.02 log-moneyness) | **−0.3689** | −0.371 | 0.0021 |
| `dσ/dt` | **implied** ATM, least squares over 0.05 → 1.00 y | **−0.0772** | −0.082 | 0.0048 |

Mixing conventions is why no single surface reproduced both. The Dupire term
slope over the same range is `−0.2535` — nowhere near `−0.082` — and the implied
skew at the long end is far shallower than `−0.371`.

The other reason direct measurement missed: **the Dupire skew is strongly
maturity-dependent here.** At `t = 0.5` it is `−0.056`; at `t = 0.866` it is
`−0.369`, a 6.6× steepening across the listed range. Any skew figure quoted for
this surface is meaningless without its slice.

### Does this invalidate the calm-surface choice?

The contrast surface was selected under the *wrong* definition (Dupire at
`t = 0.5`, width 0.15), so the cohort was re-ranked under the reconciled one:

```
date            LV skew @maxT,w=.02    lv level
2024-10-10                   0.5646      0.3277
2024-02-08                  -0.3689      0.1629   <- crash, this study
2025-01-13                  -0.1537      0.1403
2023-05-15                   0.0828      0.1645
2025-04-09                   0.0783      0.0960
2023-11-15                   0.0672      0.1277   <- calm, this study
2026-07-15                   0.0539      0.2629
2024-06-14                  -0.0439      0.1562
```

**No — 2023-11-15 stands, and the reasoning is now stronger.** It is third
flattest of eight on the reconciled slope metric and **5.5× flatter than the
crash surface**, which is the contrast the study needs. It is *not* the single
flattest, and that is deliberate: the two flattest by slope are poor contrasts
on the evidence that matters. `2024-06-14` has the flattest skew yet was the
**second-worst** cell in the FINDING (`−0.6006` contracts), and `2026-07-15`
read `+0.4623`. `2023-11-15` is the best *joint* choice — third flattest by
slope and third calmest empirically (`+0.2614`).

That divergence is itself a caution worth banking: **slope-flatness and
empirical-calmness rank the cohort differently.** The crash/calm contrast in
this study is therefore justified primarily by the empirical per-cell gaps
(`−1.2726` vs `+0.2614`) and only secondarily by the slope metric.

---

## Controls 1 and 2 — 2026-08-28

```
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_reference.py
```

Raw output: `out_reference.txt`. Six batches of 65,536 paths per rung,
`lv_time_sampling="integrated"` throughout.

### Control 1 — the reference's own convergence, demonstrated

```
--- ordinary   PDE delta = 0.591502 (0.5915 contracts)
 substeps    gap (contracts)     +/- SE   sigma     secs
        4             0.0002     0.0030    0.08      218
        8             0.0047     0.0033    1.46      589
       16             0.0043     0.0056    0.76     1196

--- near_ki   PDE delta = 0.874040 (0.8740 contracts)
 substeps    gap (contracts)     +/- SE   sigma     secs
        4            -0.0003     0.0053    0.06      199
        8            -0.0003     0.0028    0.09      549
       16             0.0028     0.0049    0.56     1179
```

**PASS — the estimate has stopped moving at 8.** Reading the pair that decides
it, 8 against 16:

| cell | substeps 8 | substeps 16 | shift | |
|---|---|---|---|---|
| `ordinary` | +0.0047 ± 0.0033 | +0.0043 ± 0.0056 | 0.0004 ± 0.0065 | **0.06σ** |
| `near_ki` | −0.0003 ± 0.0028 | +0.0028 ± 0.0049 | 0.0031 ± 0.0056 | **0.55σ** |

`substeps = 8` is adopted, unchanged from the declared value.

**Note that substeps = 4 would also have passed** (4 vs 8 is 1.0σ on `ordinary`,
0.0σ on `near_ki`) and costs 2.7× less. It was not adopted: the reading is
borderline on one cell, 8 is the level `FINDING-2026-08-26` §5 demonstrated, and
this entire exercise exists because a reference was once run under-resolved. The
cheaper rung is not worth re-opening that question.

### The gaps are an order of magnitude smaller than the FINDING's

The worst gap here is **+0.0047 contracts against a 0.5 bound — under 1% of it.**
`FINDING-2026-08-26` measured `+0.2128 ± 0.0869` on its worst cell, 42% of the
bound. Two things differ: this is a **1-year** trade rather than 3-year, and
`lv_time_sampling="integrated"` attacks the σ_loc freeze from the time axis
instead of the step count, removing a bias the FINDING could only chase with
substeps.

### Control 2 — estimator choice, measured

SE budget = `0.25 × 0.5` = **0.125 contracts**, at substeps 8, `near_ki`:

| estimator | pv SE | delta SE | gamma SE | wall clock |
|---|---|---|---|---|
| `plain` | 0.01223 | 0.00279 | 0.00685 | 549 s |
| `one_step_survival` | 0.01154 | 0.00110 | 0.00334 | 285 s |

**`plain` meets the budget on every quantity, so `plain` is adopted** — the
pre-registered tiebreak. The reference stays a single estimator at a single
discretization, the same shape as the flat-BSM study.

`one_step_survival` is tighter (2.5× on delta, 2.1× on gamma) *and* 1.9× faster,
which is worth recording for any future amendment that needs sharper Greeks. It
was **not** adopted here: the decision rule was written before the measurement,
and switching to the better-looking arm after seeing the numbers is the post-hoc
selection the release procedure exists to prevent.

**A free cross-check fell out of it.** The two estimators are independent
constructions, and their deltas agree:

```
plain  delta = 0.873789      OSS delta = 0.872427
difference 0.00136 +/- 0.00300  ->  0.45 sigma
```

That is a `plain`-vs-OSS agreement control in the shape of `RESULTS.md`'s D1,
obtained at this study's own scale rather than inherited.

### Gamma turned out to be the easy quantity

Its SE is **18× inside budget** (0.00685 against 0.125), which inverts the
concern that motivated considering a split reference at all. The reason is
structural and was predicted in spec §8: `gamma_economic = raw × 0.01 × s0`, so
a raw gamma of `−0.00037` is only `−0.037` contracts.

### Cost consequence for the real run

Delta SE at six batches is `0.0028` against a `0.125` budget — 45× inside — so
the sequential stopping rule will terminate at `min_batches = 4`. At ~98 s per
batch that is ~6.5 minutes per cell, putting the 16-cell run near **2 hours**,
the bottom of the spec's 2–14 h estimate.

---

### Correction filed

Slope figures quoted in the study YAML and in spec §4 were written under the
unreconciled definition and have been corrected to the reconciled one. A note
recording which convention each of the FINDING's two slopes uses has been added
to the FINDING; its conclusions are unaffected.
