# Finding: the 1-D local-vol snowball PDE engine was never the problem

**Answers** `REQUEST-2026-08-26-localvol-1d-pde.md`.

**Verdict — the request's first branch.** `mc_ref` migrates toward the PDE and
the residual collapses. The engine is sound: at a converged reference its delta
disagrees by **+0.2128 +/- 0.0869 contracts** on the worst surface in the
sample, upper 95% bound 0.387 against a 0.5-contract desk bound — demonstrably
inside it. No certification of `LocalVolSnowballPDESolver` is warranted.

**The reference was under-resolved in two distinct ways**, and the second was
invisible until the first was fixed:

1. `substeps=1` instead of the declared 4, biasing **both** PV and delta. This
   is the defect the request predicted.
2. `substeps=4` is still unconverged for **delta**, worth a further 0.69
   contracts, while PV was already flat from substeps=2. No price-based
   diagnostic could have caught this.

A third defect compounded them: the delta admission rule carried no
reference-uncertainty term, so the reference's own sampling noise was charged
to the engine. All three are fixed here.

---

## 1. The cheap test, and its answer

The request asked: hold the PDE at `accuracy: standard` and refine the
reference instead; read which side moves.

The probe first reproduces the banked cells exactly — proof it is running the
gate's own arithmetic and not a lookalike:

| date | banked residual | probe at substeps=1 | banked delta gap | probe at substeps=1 |
|---|---|---|---|---|
| 2024-02-08 | −0.09902% | −0.09902% | −1.2726 | −1.2726 |
| 2024-06-14 | −0.06465% | −0.06465% | −0.6006 | −0.6006 |

One rung of reference refinement collapses both symptoms, on both dates:

| substeps | 2024-02-08 PV | delta (contracts) | 2024-06-14 PV | delta (contracts) |
|---|---|---|---|---|
| **1** | **−0.09902%** | **−1.2726** | **−0.06465%** | **−0.6006** |
| 2 | −0.00116% | −0.0712 | +0.00287% | −0.0853 |
| 4 | +0.01884% | −0.4628 | +0.02950% | +0.1483 |
| 8 | +0.03375% | −0.2062 | +0.05691% | −0.2223 |
| 16 | −0.00721% | +0.5738 | −0.03533% | +0.1071 |

Meanwhile the PDE does not move. Across its **entire** accuracy ladder
(`fast` → `standard` → `high`) the delta spread is **0.0079 contracts** — 63×
tighter than the 0.5-contract bound it was failing — and the price moves
0.0004% of notional.

### The control that settles it

Flatten the local-vol surface and the disagreement disappears. Same solver,
same barriers, same product, same discrete KI schedule:

| surface | dσ/dlnS | dσ/dt | residual at substeps=1 |
|---|---|---|---|
| real Dupire, 2024-02-08 | −0.371 | −0.082 | −0.09902% = **−2.88σ** |
| flat at 29.45% | 0.0 | 0.0 | −0.01128% = −0.23σ |

> **Convention note, added 2026-08-28.** The two slopes above are measured on
> two *different* surfaces, which is why neither reproduces if you assume one:
> `dσ/dlnS = −0.371` is the **Dupire local-vol** skew at the long end
> (`t ≈ max_listed_T = 0.866`) in a narrow ±0.02 log-moneyness window
> (reproduced: −0.3689), while `dσ/dt = −0.082` is the **implied** ATM term
> slope over 0.05 → 1.00 y (reproduced: −0.0772). The Dupire term slope over
> that range is −0.2535, nowhere near −0.082.
>
> Note also that the Dupire skew here is strongly maturity-dependent: −0.056 at
> `t = 0.5` against −0.369 at `t = 0.866`, a 6.6× steepening. A skew figure for
> this surface means nothing without its slice. The conclusions above are
> unaffected. See `docs/modelvalidation/pilot-localvol-1d/RESULTS.md`.

A defective PDE would not be rescued by flattening the surface. A one-step
log-Euler MC would be: `LocalVolSnowballMCEngine.simulate()` freezes
`vol = lv.local_vol(spot, t)` at each step's left endpoint, and a constant
σ_loc is exactly the case where that freezing costs nothing. The bias is
proportional to surface steepness — which is why 2024-02-08, the crash bottom,
is the worst cell.

Ten independent Sobol scrambles put it beyond doubt:

| reference | delta gap vs PDE | |
|---|---|---|
| substeps=1 | −0.8590 ± 0.1497 | **5.7σ — a real bias** |
| substeps=4 | −0.2130 ± 0.1774 | 1.2σ — consistent with zero |

---

## 2. Root cause A — the reference never received its declared resolution

`GATE_PAIRS["localvol"].build_reference` constructed
`LocalVolSnowballMCEngine` without `substeps_per_interval`.
`_make_mc_params()` cannot carry it (it is not an `MCParams` field — it is an
engine constructor kwarg), and `_SubstepRefinementMixin.substeps_per_interval`
defaults to **1**. The `heston`/`heston_slv` builders thread it through
`_make_mc_engine(..., MC_FULL["substeps_per_interval"])`. localvol did not.

So the gate ran one Euler step per daily observation against a Dupire surface,
while `gate_decision.json`'s top-level `mc_reference` block declared
`substeps_per_interval: 4` for the whole study. `variants.localvol.mc_params`
had no such key at all; `heston` and `heston_slv` both had `4`.

**The justification was a false belief, written down in three places:**

> *"Heston-only QE knobs; the gate's localvol entry has neither
> (LocalVolSnowballMCEngine accepts neither kwarg)"* — `12_snowball_volmodel_backtest.py`
>
> *"QE substeps_per_interval is a Heston/Heston-SLV discretization knob"* — `11_pde_convergence_gate.py`
>
> and the docstring of `test_localvol_gets_no_heston_only_options`.

`_SubstepRefinementMixin` sits on `_VolModelSnowballMCBase`, the shared base of
every vol MC engine. `LocalVolSnowballMCEngine._create_path_generator` calls
`self._refined_dt_array(dt_array)` exactly as the QE engines do. The repo's own
**green** test `test_mc_substeps_per_interval.py::test_pseudo_mc_equivalence[lv-snowball]`
has been proving this the whole time. Only `scheme` is genuinely Heston-only;
the comment generalized from the one knob where the claim was true.

**It reached production.** Stage 12 gates the fleet's MC config on what the
decision recorded, and its filter drops keys the decision omits — so the
localvol MC route the fleet currently runs also sits at `substeps=1`, carrying
the same bias. This is the exact failure stage 12's own comment says it exists
to prevent: *"Running the fleet on any other config computes a delta the gate
never certified."*

---

## 3. Root cause B — a noisy reference judged by a noise-blind rule

Fixing A alone does not make G2 pass.

The PV rule widens its tolerance by the reference's sampling error:
`gate_tolerance_pct` = `max(2 × mc_se, TOL_ABS)`. The delta rule did not —
`delta_cell_passed(abs_diff, s0)` compared against a fixed 0.5 contracts, and
no standard error was computed for the delta at all.

That is correct for a deterministic reference and wrong for a Monte-Carlo one.
The measured noise, by two independent methods that agree:

- sd across 10 independent Sobol scrambles: **0.5611 contracts**
- within-run batch spread over 16 batches (`PairedRQMCGreeksResult.delta_std_error`): **0.5939 contracts**

Against a 0.5-contract bound. At substeps=4, with the bias gone, **4 of 10
scrambles still failed** — on noise alone, with a PDE converged to 0.008
contracts.

It silently disabled the bias check too. `detect_delta_bias` tests the mean
signed gap over 8 cells against 0.1 contracts; that mean inherits SE/√8 =
**0.20 contracts**, twice its own bound. The banked `mean_signed = −0.0551`
passing was luck — and it looked innocuous only because the σ_loc-freezing bias
changes sign with the local skew and term slope (−1.27, −0.60, +0.70, −0.31,
+0.26, +0.13, +0.18, +0.46), so a genuine 5.7σ per-cell bias averaged to
almost nothing.

### Why localvol alone

| | deterministic reference | MC reference |
|---|---|---|
| **noise-blind rule (stage 11)** | `flat_bsm`, `flat_bsm_quad`, `ts_bsm` — valid | **`localvol` — the gap** |
| **uncertainty-aware (stage 16)** | — | `heston`, `heston_slv` — valid |

`decide_route`'s own docstring calls Stage 16 the *"paired, uncertainty-aware
delta/gamma certificate"*, and Stage 16 is built on `run_paired_rqmc_greeks`.
The machinery existed; stage 11 just never used it for the one variant that
needed it.

---

## 4. What changed

Both defects are fixed in `11_pde_convergence_gate.py` (schema 1 → 2).

| | change |
|---|---|
| A | the localvol reference is built with `substeps_per_interval`; `_make_mc_engine` covers localvol |
| A | `_reference_params_block` records substeps for **every** MC reference; `scheme`/`martingale_correction` stay Heston-only |
| A | the substeps sensitivity diagnostic covers every MC reference, not just the ADI pair |
| B | `delta_tolerance_per_unit(se, s0)` = `max(2 × SE, 0.5 contracts)`, mirroring `gate_tolerance_pct`; `None` SE keeps the flat desk bound |
| B | `_bumped_mc_delta` returns `(delta, std_error)` via `run_paired_rqmc_greeks` — the returned delta is **bit-identical** to the two-price CRN bump it replaces (0.5554460169 on 2024-02-08/full) |
| B | load-bearing MC deltas get their own reference at `MC_DELTA_SUBSTEPS = 8` and `MC_DELTA_PATH_FACTOR = 4` — see §5, where substeps and not paths turns out to be the binding knob |
| both | `validate_delta_rows` fails closed on an MC row with no SE; schema-1 evidence can no longer be rescored under the new rule |
| both | the three false "Heston-only" claims corrected |

### One behavioural change to record

`run_paired_rqmc_greeks` takes a `homogeneous` fast path when an engine
declares `rqmc_homogeneous_spot_scaling`: one path set, rescaled for the
down/base/up legs. `QESnowballMCEngine` declares it (Heston is
spot-homogeneous, so rescaling is exact); `HestonSLVQESnowballMCEngine` and
`LocalVolSnowballMCEngine` correctly do not, because `L(S,t)` and
`sigma_loc(S,t)` depend on absolute spot. So `heston`'s delta diagnostic gets
*cheaper* and lower-variance, and its value will differ slightly from the
schema-1 run. It is a `require_delta=False` diagnostic row, and this is the
construction Stage 16 already uses.

### A third defect, found on the way

`GATE_PAIRS` closed over `MC_FULL` directly, so `--quick` reported `MC_QUICK`
in its payload and then executed `MC_FULL` — a 64× overrun in the mode whose
purpose is to be cheap, and the same declared-is-not-executed shape as defect
A. Threading the active `mc` config into the reference builders fixes it:
`test_quick_end_to_end`, previously a multi-hour hang that had to be
deselected, now completes the whole gate in **3m23s**.

### Contract changes that ripple into existing tests

Two changes tighten contracts that existing tests encoded, and those tests were
updated rather than the contracts loosened:

- reference builders take a third argument (the active `mc` config), so
  `test_pde_convergence_gate.py`'s `_stub_gate_pairs` lambda gained it;
- `validate_gate_payload` pins `schema_version` and requires `deltas`, so the
  synthetic payloads in the same file moved to schema 2 with `"deltas": []`.

`detect_delta_bias` additionally now reports `mean_signed_se_contracts`
(= sqrt(sum(se^2))/n) and `bias_bound_is_resolvable`. Under the schema-1
configuration those read 0.20 contracts and `False` — the statistic's own error
bar was twice the 0.1-contract bound it was tested against, which is why
`delta_biased: false` carried no information while a 5.7-sigma per-cell bias
sat in the sample. They are `None`, never `0.0`, where any contributing
reference is deterministic.

### Both knobs were measured, not derived

SE in contracts (2024-02-08/full, paired RQMC, 16 batches):

| paths | substeps=4 | substeps=8 |
|---|---|---|
| 131,072 | 0.5939 | — |
| 262,144 | 0.4364 | 0.3228 |
| 524,288 | 0.3275 | **0.1832** ← adopted |
| 1,048,576 | 0.2146 | 0.1314 |

The path direction fits ≈ N^−0.43, not N^−0.5: a snowball's KO/KI indicators
are jump discontinuities in path space, so randomized QMC loses most of its
edge, and a bumped delta is carried by exactly the paths that flip barrier
status. A √N extrapolation would have mis-sized the factor.

The substeps direction is §5's subject — it removes bias *and* variance.

---

## 5. What the fix uncovers: a residual delta gap on the steepest surface

With the reference quietened to 1,048,576 paths, `2 x SE` falls under the desk
bound on both probed dates (0.429 and 0.378), so the desk bound binds again —
the path factor did its job. What it exposes is a disagreement the noise had
been hiding:

| date | SE (contracts) | gap (contracts) | verdict |
|---|---|---|---|
| 2024-06-14 | 0.189 | +0.188 | pass, comfortably |
| 2024-02-08 | 0.215 | **-0.579** | fail |

Pooling three independent estimates of that 2024-02-08 gap (10 scrambles at
x1; single runs at x4 and x8 — all different Sobol point sets):

```
POOLED   -0.3854 +/- 0.1262 contracts   (3.1 sigma from zero)
```

So the true gap is **real but inside the 0.5 desk bound**. The x8 draw read
-0.579 because `max(2 x SE, bound)` compares a POINT ESTIMATE against the bound
with no allowance for measurement error once the bound dominates: a single x8
measurement of this cell reads outside 0.5 about **30%** of the time.

### The gap is the reference's, not the engine's — delta converges later than PV

Moving the reference's discretization while holding paths at x8 settles it:

| reference | gap vs PDE (contracts) |
|---|---|
| substeps=4 (pooled over 10 seeds @x1, 1 @x4, 1 @x8) | **-0.3854 +/- 0.1262** |
| substeps=8 @x8 | **+0.3042 +/- 0.1314** |
| **shift** | **+0.6896 +/- 0.1822 = 3.8 sigma** |

The gap does not shrink — it **flips sign and moves 0.69 contracts**, more than
the 0.5-contract desk bound itself. The reference's DELTA is still materially
unconverged at substeps=4, the level this study declares.

**PV converged at substeps=2; delta had not converged at 4.** PV is measured at
the base spot, where the sigma_loc-freezing error largely cancels; the delta
differences two bumped prices, where it does not. Reading "PV is flat under
refinement" as "the reference is converged" is exactly the inference this
request was raised to correct — it recurs one level down, on the same engine,
for the same reason.

Two consequences:

- **Paths were the wrong lever to reach for first.** More paths shrink the SE
  but do nothing to a discretization bias; at substeps=4 the x8 reference
  simply measured a biased number more precisely, which is why it produced a
  confident-looking FAIL.
- **`MC_DELTA_FULL` must not inherit `substeps=4`.** The delta reference needs
  its own demonstrated substeps level, and `substeps=8` is not yet it
  (+0.3042 +/- 0.1314 is still 2.3 sigma from zero).

### Substeps is the better-value knob, in both currencies

Refining substeps does not merely remove bias — it also *reduces* the sampling
error at unchanged paths:

| paths | SE at substeps=4 | SE at substeps=8 | improvement |
|---|---|---|---|
| x2 | 0.4364 | 0.3228 | 1.35x |
| x8 | 0.2146 | 0.1314 | 1.63x |

With one step per observation the integrand's dependence on a Sobol point is
dominated by discrete barrier flips — jump discontinuities that destroy QMC's
advantage. Finer stepping resolves each crossing more gradually and hands some
of the low-discrepancy benefit back. So doubling substeps buys ~1.6x SE
reduction AND removes bias, where doubling paths buys 1.35x (N^-0.43) and no
bias reduction, at comparable cost.

This means `MC_DELTA_PATH_FACTOR = 8` was calibrated in the wrong regime: it
was measured at substeps=4, where both the bias and the noise were larger than
they need to be. Paths and substeps are not independent knobs here.

Pooling the substeps=8 measurements gives **+0.2154 +/- 0.1217 contracts
(1.8 sigma from zero)** — comfortably inside the 0.5-contract desk bound.

**A hard ceiling applies.** `_qmc_normals` requests a Sobol sequence of
dimension `732 * substeps`, and SciPy caps at 21201 — so `substeps <= 28`. If
the delta does not converge by then, this RQMC construction cannot reach it by
refinement alone.

### Convergence demonstrated, and the engine is provably inside the bound

The request required the reference's own convergence to be shown, not assumed.
Refining substeps at fixed paths until the estimate stops moving:

| substeps | gap vs PDE (contracts) |
|---|---|
| 1 | -0.8590 +/- 0.1497 |
| 4 | -0.3854 +/- 0.1262 |
| **8** | **+0.2151 +/- 0.1014** |
| **16** | **+0.2065 +/- 0.1689** |

substeps 8 and 16 differ by **0.04 sigma** — it has stopped moving, and 8 is
the level recorded. Note the ladder CROSSES zero rather than decaying to it, so
a one-sided reading at any single level mis-signs the error; that is exactly
how substeps=1 made the PDE look 1.27 contracts wrong in the other direction.

Pooling substeps 8 and 16 gives the converged disagreement:

```
+0.2128 +/- 0.0869 contracts     upper 95% bound 0.387  <  0.5 desk bound
```

So the LV PDE's delta is **demonstrably** inside the desk bound on the worst
surface in the sample — not merely inside in expectation. There is a small real
difference of ~0.21 contracts, at 42% of the tolerance the desk actually set.

### A caution worth banking

At substeps=4 with x8 paths the same cell read **-0.579 +/- 0.215, a confident
2.7-sigma FAIL**. That number was precise and wrong: the extra paths measured a
biased estimator more sharply. Precision without accuracy is the most
convincing way to be wrong, and the tell was available — the estimate moved
0.69 contracts when the discretization changed, which no amount of sampling
error could explain.

## 6. Result of the re-run

G2 re-run on the frozen cohort (8 dates, `--workers 2`, 4446s):
**all six variants route to PDE**, and every localvol delta cell passes.

| date | was | now | SE | tolerance |
|---|---|---|---|---|
| 2024-02-08 (crash bottom) | **-1.2726** | **+0.2142** | 0.183 | 0.500 |
| 2025-01-13 | +0.6998 | -0.0876 | 0.202 | 0.500 |
| 2024-06-14 | -0.6006 | -0.1495 | 0.259 | 0.517 |
| 2026-07-15 | +0.4623 | +0.2317 | 0.169 | 0.500 |
| 2024-10-10 | -0.3079 | -0.0572 | 0.222 | 0.500 |
| 2023-11-15 | +0.2614 | +0.2228 | 0.138 | 0.500 |
| 2025-04-09 | +0.1826 | -0.3445 | 0.153 | 0.500 |
| 2023-05-15 | +0.1337 | +0.2352 | 0.186 | 0.500 |

The three worst cells improve most, which is the signature of a bias
proportional to surface steepness rather than a coincidental fix. 2024-02-08
also flips sign, exactly as the substeps ladder predicted, and its `+0.2142`
reproduces the probe value for that configuration.

`delta_pass: true`, `delta_biased: false`, `mean_signed = +0.0331`.

**One honest caveat in the banked evidence.** `bias_bound_is_resolvable: false`
— the mean's SE is 0.068 against the 0.1-contract bias bound, so `2 x SE`
(0.136) still exceeds it. The bias check passes, but it is not yet a fully
powered test: it is now within a factor of 1.4 of being one, against a factor
of 4 before. That field exists so no reader takes `delta_biased: false` for
more than it is. Only 2024-06-14's tolerance (0.517) is set by the noise term;
every other cell is bound by the desk.

Note `variants.localvol.mc_params` records `substeps=4, paths=8192` — correct,
because that is the **PV** reference the route decision concerns. The delta
reference (substeps 8, x4 paths) is a separate, quieter configuration, and each
delta row carries its own SE and tolerance.

## 7. What lands

Done: G2 re-run and schema-2 evidence banked to
`output/pde_convergence_gate/` (the schema-1 artifact is gitignored and existed
only on disk; it was copied aside first). Schema-1 evidence can no longer be
rescored — `validate_gate_payload` refuses it, because those delta rows carry
no reference standard error and their localvol reference ran at `substeps=1`.

Still outstanding for whoever picks this up:

- **The fleet's existing localvol numbers must be discarded.** G2 routed localvol to
  MC, and stage 12 forwards only the keys the decision recorded — which
  omitted `substeps_per_interval`, so `engine_factory` fell through to the
  engine default of 1. Every localvol MC price and delta the fleet has
  produced under this routing carries the bias measured here.
- **The 40.6 CPU-hour localvol cost projection needs re-measuring.** It was
  taken under `route=pde`, which the arm has now been restored to — but the
  gate itself is dearer: the localvol delta reference costs ~524s per cell, and
  the whole re-run took 4446s at `--workers 2`.
- **Stage 16's checkpoints will not resume.** It fingerprints
  `11_pde_convergence_gate.py` in `IMPLEMENTATION_INPUTS` precisely so that
  checkpoints from a different source state are rejected rather than mixed.
  That is the mechanism working; any in-flight ADI Greek run needs a fresh
  start.

## 8. Evidence

Probe scripts and raw JSON are in the session scratchpad:
`probe_lv_substeps.py` (refinement ladder, both dates),
`probe_lv_flat_control.py` (flat-surface control),
`probe_lv_delta_noise.py` (10-scramble replication),
`probe_paired_delta.py` (bit-identity of the paired estimator),
`probe_delta_paths.py` (SE-vs-paths ladder),
`probe_residual_gap.py` (substeps-vs-real-gap test).
