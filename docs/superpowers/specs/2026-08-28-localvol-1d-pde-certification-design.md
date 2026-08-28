# Certifying the 1-D local-vol snowball PDE engine

**Date** 2026-08-28
**Engine** `LocalVolSnowballPDESolver`
(`quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py:69`)
**Study** `snowball-localvol-1d`
**Branch** `worktree-localvol-1d-certification`
**Procedure** `docs/modelvalidation/RELEASE_PROCEDURE.md`

---

## 1. Why this is being built

`REQUEST-2026-08-26-localvol-1d-pde.md` asked for this certification because
Gate G2 routed `localvol` to Monte Carlo — alone among the study's six variants
— confounding an engine difference with the model difference the study exists to
measure.

`FINDING-2026-08-26-localvol-1d-pde.md` then answered the request's own cheap
test and concluded the opposite: **the PDE was never wrong.** At a converged
reference its delta disagrees by `+0.2128 ± 0.0869` contracts on the worst
surface in the sample, upper 95% bound 0.387 against a 0.5-contract desk bound.
The reference had been under-resolved twice over — `substeps=1` where 4 was
declared, and still delta-unconverged at 4 — and the delta admission rule
carried no reference-uncertainty term.

So this study is **not** being built to settle a correctness question. That
question is settled. It is being built for three things the G2 route decision
cannot deliver:

1. **Banked, schema-versioned evidence.** A gate decision is a routing choice
   recomputed per run. A certificate is a durable record with a digest, a
   projected identity hash, and a parent chain.
2. **`delta_authority` delegation.** `heston` and `heston_slv` delegate to
   banked stage-16 evidence. `localvol` has nothing to delegate to, so stage 11
   must re-derive its own MC authority every run — the expensive path that
   produced the original defect.
3. **CI anchors.** The deterministic arm costs ~1s per case. Once banked,
   `assert_anchors` re-runs it on every commit and fails the moment the released
   solver stops producing the certified numbers. There is currently no such
   guard on any local-vol engine.

This motivation is recorded explicitly because a reader who finds the FINDING
first will otherwise reasonably ask why a certification was run against its
recommendation.

## 2. Scope

A certificate covers **only the configurations its YAML names**. That lesson is
banked from the 2026-08-19 variant-surface amendment, which found three defects
hiding in configurations the original certificate never enumerated.

**Covered:** two real calibrated CSI1000 Dupire surfaces; snowball maturities
0.25–1.0 years; continuous, discrete and European knock-in; flat and step-down
knock-out; spot at inception and adjacent to both barriers.

**Not covered, and not to be inferred:**

- the flat-BSM product variants — `parachute`, `airbag`, `protection_*`,
  `reverse`, `call_rebate`, `participation`, `disable_ko_after_ki`,
  `coupon_pay_type`, `is_annualized`, `ko_rate_step`. These exercise payoff code
  inherited from `SnowballPDESolver` and already certified under flat BSM
  (`snowball-flat-bsm/2026-08-19-*`); what local vol adds is the surface read on
  the `(S,t)` mesh, not the payoff assembly.
- surfaces other than the two named artifacts.
- maturities beyond 1.0 years, where this surface is majority-extrapolated
  (see §4).
- the quadrature engine. There is no local-vol quadrature engine.

## 3. Reproducibility: the surface must be committed

`example/mo_volmodels/data/history` is excluded through `.git/info/exclude`
line 71 — a **per-clone** file that is never pushed. The 787 surface artifacts
there exist only on the machine that built them.

This is not a preference to weigh; it is a hard constraint. `assert_anchors`
re-runs the **deterministic candidate** in CI, the candidate cannot build
without its local-vol surface, so a study reading from `data/history` would bank
a certificate whose anchors fail on every machine except one.

Two artifacts are therefore copied into the study's own directory and committed:

```
example/modelvalidation/data/
  iv_surface_20240208.json    # crash bottom  — s0 4993.105, sha b0e63653a774b5b3
  iv_surface_20231115.json    # calm contrast — s0 6207.3
```

Verified not caught by any ignore rule. ~33 KB each.

**The builder must not re-smooth.** The stored `iv_grid` is already
`sabr_calendar_projected` (`target_smoothing.method`). Dupire differentiates
total variance twice in strike and once in maturity, so applying
`sabr_smoothed_surface` again would silently certify a different surface from
the one the artifact names.

`IvSurfaceArtifact` (`quantark/param/vol/surface_history.py`) is a **library**
class, so the builder needs no example-code import. Its `sha256` goes into the
environment's identity, which pins the exact surface bytes into the certificate
and invalidates checkpoints if the artifact ever changes.

## 4. The surfaces, and what they constrain

```
2024-02-08   s0 4993.105   listed expiries 0.022 0.099 0.195 0.367 0.616 0.866
             ATM term  0.284 → 0.429 → 0.386 → 0.328 → 0.316   (humped, steep)
             IV range  0.281 .. 0.533                           (crash bottom)
2023-11-15   s0 6207.3     max listed T 0.849
             LV slopes  dσ/dlnS −0.020, dσ/dt −0.017            (flattest in cohort)
             LV level   0.135                                   (≈ half of the above)
```

`max_listed_T ≈ 0.87` with `extrapolation_policy = flat_total_variance` beyond
it. A 3-year trade on this surface would be **71% extrapolated** and would
certify the extrapolation policy rather than the calibrated skew. A 1-year trade
is ~87% live, which is the longest defensible maturity here — hence the scope
limit in §2 and the paired `inside_listed_grid` / `ordinary` cases in §6.

The calm contrast was selected by measurement, not by eye: all eight cohort
dates were ranked by **Dupire local-vol** steepness — the surface the PDE mesh
and the MC steps actually read — and 2023-11-15 is flattest on both axes at
roughly half the vol level. This makes the pair a direct test of the FINDING's
central mechanism, that the discretization bias is proportional to surface
steepness.

## 5. Architecture

One new module, `quantark/modelvalidation/builders/equity_snowball_localvol.py`,
registered from `builders/__init__.py`.

| kind | name | responsibility |
|---|---|---|
| environment | `equity.snowball.localvol_market` | artifact → `GridVolSurface` + `TermStructureDividendYield`; records `sha256`, `trade_date`, `max_listed_T`, `extrapolation_policy` |
| candidate | `equity.snowball.localvol_pde` | `LocalVolSnowballPDESolver` at a declared `accuracy`, with a `standard → fast` refinement rung |
| reference | `equity.snowball.localvol_mc` | `LocalVolSnowballMCEngine`, per-batch seed for independent scrambles |

The **product** builder is reused unchanged (`equity.snowball` +
`make_snowball` from `builders/equity_snowball.py`). Its existing keys —
`ki_monitoring`, `ko_stepdown`, `months`, `maturity` — already express every
case in §6. Re-implementing certified product construction is how two arms of
the same study drift apart.

The environment builder accepts a `spot` override so the `near_ko` / `near_ki`
cases can move spot without rebuilding the surface: local vol is a function of
absolute `(S,t)`, so moving spot moves the trade through a **fixed** surface,
which is exactly the intent.

### Reference design

One discretization for **all three quantities**:

```yaml
substeps_per_interval: 8
lv_time_sampling: integrated
```

`substeps=8` is the level FINDING §5 demonstrated the estimate stops moving at
(8 and 16 differ by 0.04σ). `integrated` replaces the left-endpoint σ freeze
with the closed-form per-step time-averaged variance — exact on time-only
surfaces, and the measured removal of a −1.26c daily-grid bias at zero per-step
cost (`docs/lv-mc-scheme-demos/RESULTS.md`).

The market is a flat `r = 0.02` (`FLAT_RATE`, the gate's production rate
channel) with the dividend/carry term structure implied from each artifact's own
per-expiry parity pillars via `term_structure_dividend_yield(rate)`. The carry
therefore comes from the same quotes the surface was calibrated against, rather
than from a number chosen independently of it.

Pinning one discretization across quantities is deliberate and is the design's
central correctness commitment. Running PV at one substep level and Greeks at
another estimates `P(h)` and `P(h/2)` — different numbers at finite `h` — so the
certified delta would not be the derivative of the certified price. The gate
does this today for a routing decision, which FINDING §6 had to write a
paragraph to justify; a certificate should not need that paragraph.

The **estimator** may differ between PV and Greeks, and only if §7's pilot shows
it must. **Tiebreak: if `plain` meets the standard-error budget on every
quantity, `plain` is adopted** — one estimator, fewer moving parts, and the same
shape as `equity.snowball.mc_rqmc`. `one_step_survival` is introduced only for a
quantity `plain` cannot resolve.

- `plain` paired-RQMC central difference, mirroring `equity.snowball.mc_rqmc`; or
- `one_step_survival` for Greeks, which `RESULTS.md` measured at ~50× tighter
  gamma SE with a bump-stable mean.

Splitting the estimator is safe where splitting the discretization is not: both
estimators target the same discretized expectation — OSS is a
Rao–Blackwellization, equal by the tower property — and their agreement is
already measured (`RESULTS.md` control D1, `Δ = −0.005c ± 0.039`). The split's
costs are a shared batch counter (`should_stop` waits for the slowest
quantity), two configurations to declare, and OSS's refusal of
`disable_ko_after_ki`, which §2 excludes anyway.

`estimator="one_step_survival"` rejects `RANDOMIZED_QUASI`, so an OSS arm runs
`QUASI`. This was checked rather than assumed: `_qmc_normals` always constructs
`qmc.Sobol(scramble=True, seed=base_seed + batch_id)`, so batches remain
independent scrambles and the batch-to-batch standard error stays valid. Had
`QUASI` been unscrambled, every batch would draw identical points, the SE would
collapse toward zero, and `should_stop` would fire `SE_BUDGET_MET` immediately —
a false `ADMITTED`, the one outcome this framework exists to prevent.

## 6. Cases — 8 per surface × 2 surfaces = 16

| case | what it stresses |
|---|---|
| `ordinary` | T=1.0, continuous KI, spot at inception; ~13% extrapolated |
| `inside_listed_grid` | T=0.75, entirely within listed expiries; isolates the engine from the extrapolation policy |
| `near_ko` | spot just under the KO barrier; the discontinuity sits inside the bump stencil |
| `near_ki` | spot just above the KI barrier, where the barrier-local `σ_loc(B)` coefficients bind |
| `discrete_ki` | drops the Brownian-bridge crossing correction and the per-step first-passage transfer |
| `european_ki` | the two-surface dynamic programme collapses to one terminal test |
| `stepdown_ko` | twelve distinct barrier levels competing for grid alignment on a *skewed* surface |
| `near_expiry` | T=0.25, in the steepest part of the term structure (ATM 0.284 → 0.429 within three weeks) |

Levels are pinned as fractions of each surface's own `s0`, so the pair is
economically comparable across two different index levels:

```
ko_barrier  1.03 × s0      ki_barrier  0.85 × s0     strike  1.00 × s0
ko_rate     0.15           rebate_rate 0.15          months  12  (KO monthly)
near_ko     spot 1.025 × s0            near_ki     spot 0.86 × s0
stepdown_ko 0.005 of initial price per observation  (1.03 → 0.975)
near_expiry months 3, maturity 0.25
```

These mirror `snowball_flat_bsm.yaml` exactly at `s0 = 100`, so a cell-by-cell
comparison against the flat-BSM certificate isolates the effect of the surface
and nothing else.

## 7. Phase 1 — the pilot, which gates everything after it

The FINDING's entire story is a reference trusted without its own convergence
being demonstrated. None of the following is optional, and no full run starts
until all four have passed.

1. **Reference convergence, demonstrated not inherited.** A substeps ladder
   4/8/16 on the two hardest cells (`near_ki` and `ordinary` on 2024-02-08),
   confirming the estimate has stopped moving at 8. FINDING §5 demonstrated
   `substeps=8` for a *3-year* trade at mo-study scale. This is a *1-year* trade
   at a different notional; it does not inherit that result.
2. **Estimator choice, measured not derived.** Per-quantity standard error for
   `plain` versus `one_step_survival` at study scale. Whichever meets the
   `0.25 × cell` budget is adopted, and the measurement is recorded in the
   certificate. Note §8's arithmetic makes gamma the *cheap* quantity at this
   scale, so `plain` may well suffice — which would keep the reference a single
   estimator and dissolve the split entirely.
3. **Flat-surface control.** Flatten one surface to its ATM level; the LV PDE
   must collapse onto the flat-BSM PDE, and the LV MC onto the BSM MC. This is
   the control that settled the original diagnosis, and it separates
   "input is wrong" from "formula is wrong".
4. **`--quick` wiring check.** Explicitly not bankable evidence; it proves the
   plumbing runs.
5. **Economic-scale verification.** A known raw delta must convert to the
   intended contract count on *both* surfaces, confirming the
   `contract_multiplier` correction of §8. A silent 1.243× error here would
   move every calm-surface cell against a bound that cannot detect it.

## 8. Bounds, economic scale, sampling

```yaml
quantities: [pv, delta, gamma]
bounds:  {cell: 0.5, mean_signed_bias: 0.1}
economic_scale:
  builder: hedge_contracts
  params: {hedge_multiplier: 200.0, hedge_inception_spot: 4993.105,
           notional: 998621.0}
sampling: {paths_per_batch: 65536, min_batches: 4, max_batches: 32,
           seed: 20260828, bump: 0.01}
```

`hedge_multiplier: 200` is the CSI1000 index-future multiplier, so one contract
is a real hedge instrument. `notional = 200 × 4993.105` makes
`delta_quantum = hedge_multiplier × hedge_inception_spot / notional` **exactly
1.0**, matching the flat-BSM study's normalization (`200 × 100 / 20000 = 1.0`).
Raw delta then reads directly as contracts and the two certificates are
numerically comparable — without inventing a rescaling of the surface.

### The two-surface scale correction

`economic_scale` is a single **study-level** block, but the two surfaces sit at
different index levels (`4993.105` and `6207.268`). Left uncorrected, the calm
surface's cells would be converted on the crash surface's basis and every error
there would be **overstated by `6207.268 / 4993.105 = 1.243`** — which, because
it inflates rather than shrinks a measured error, risks a false `REJECTED`, not
merely a conservative pass. A certificate must not make that claim wrongly.

The correction uses an existing product key rather than a new mechanism: each
surface's cases declare

```
contract_multiplier = 4993.105 / s0(surface)
    2024-02-08 → 1.000000
    2023-11-15 → 0.804397
```

`pv`, `delta` and `gamma` are all linear in `contract_multiplier`, so this scales
every raw quantity by exactly the factor the conversion needs, and it does so by
making both trades **the same economic notional** (998,621) expressed at their
own index level — 200 units at 4993.105, 160.9 units at 6207.268 — which is what
a desk trading a fixed notional actually holds.

**This is a derivation, so the pilot verifies it numerically** (§7 item 5)
rather than trusting the algebra: a known raw delta must convert to the intended
contract count on *both* surfaces before any evidence is banked.

A consequence worth recording, because it inverts the expected difficulty:
`gamma_economic = raw × 0.01 × s0`, so a measured raw gamma of `−0.00069` is
`−0.034` contracts, and its SE budget of 0.125 contracts corresponds to a raw
gamma SE of `0.0025` — over three times the gamma value itself. Gamma is the
*easy* quantity at this scale, not the hard one.

**Bounds are fixed.** Per the release procedure, widening a bound to convert an
`INCONCLUSIVE` or `REJECTED` into an `ADMITTED` inverts the purpose of the
exercise. If the study returns `INCONCLUSIVE`, the response is more sampling or
a fixed engine.

## 9. Phase 2–3 — run, bank, guard

- Full run (~2–14 h by the cost probe: PDE 1.08 s/case, MC ~102 s/batch,
  4–32 batches/case).
- Bank `certificate.json`, `report.md`, `report.html` and a `README.md`
  recording provenance to
  `docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/`.
  Never `checkpoints/`.
- Extract anchors; add `test_localvol_pde_matches_its_certification` alongside
  the existing anchor tests in `test/modelvalidation/`. The deterministic arm is
  ~1 s per case, so the CI cost is ~30 s including ladder rungs.
- Record in the certificate README that stage 11 may now delegate
  `delta_authority` for `localvol` to this evidence, and that the fleet's
  localvol cost projection needs re-measuring (FINDING §7 leaves both
  outstanding).

## 10. Open item

The FINDING records `dσ/dlnS = −0.371, dσ/dt = −0.082` for 2024-02-08. Measuring
the Dupire surface directly gives `−0.056` and `−0.269` at `t = 0.5y`; measuring
the *implied* surface gives `−0.031` and `−0.081`. The implied-vol term slope
reproduces the FINDING almost exactly while the skew is off by an order of
magnitude, which points to a definitional mismatch (which surface, which slice,
which moneyness window) rather than a data problem.

This does not change the case list — 2024-02-08 is unambiguously the worst cell
*empirically* (`−1.2726` contracts, `−2.88σ`), which is the ground truth that
selected it. But "bias ∝ surface steepness" is the mechanism the whole diagnosis
rests on, and the calm-surface contrast in §4 is chosen on that metric. It
should be resolved during the pilot, or explicitly recorded as unreconciled
before evidence is banked.

## 11. References

- `docs/modelvalidation/RELEASE_PROCEDURE.md`
- `docs/modelvalidation/REQUEST-2026-08-26-localvol-1d-pde.md`
- `docs/modelvalidation/FINDING-2026-08-26-localvol-1d-pde.md`
- `docs/lv-mc-scheme-demos/RESULTS.md`
- `example/modelvalidation/snowball_flat_bsm.yaml` (the model this mirrors)
- `docs/modelvalidation/certificates/snowball-flat-bsm/2026-08-19-3/`
