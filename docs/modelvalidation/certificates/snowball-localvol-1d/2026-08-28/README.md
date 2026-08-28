# snowball-localvol-1d — 2026-08-28

**Engine** `LocalVolSnowballPDESolver`
(`quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py:69`)
**Decision** **ADMITTED**
**Evidence digest** `931345b9ae5e684910b1e85be5f3376522af2b19ecc4548ee918764631eae06c`
**Machine** `arm64` / macOS — Python 3.11.8, NumPy 2.4.6, quantark `94b8e29`
**Study** `example/modelvalidation/snowball_localvol_1d.yaml`
**Wall clock** 4050 s (67.5 min), 16 cells × pv/delta/gamma = 48 gated cells, 0 errors

---

## Why this certification exists

`REQUEST-2026-08-26-localvol-1d-pde.md` asked for it because Gate G2 routed
`localvol` to Monte Carlo, alone among six variants, confounding an engine
difference with the model difference the study measures.

`FINDING-2026-08-26-localvol-1d-pde.md` then answered the request's own cheap
test and concluded **the PDE was never wrong** — the reference had been
under-resolved twice over. This certification was run anyway, and deliberately,
for three things a routing decision cannot deliver:

1. **banked, schema-versioned evidence** with a digest and a projected identity
   hash, rather than a decision recomputed per run;
2. **`delta_authority` delegation** — `heston` / `heston_slv` delegate to banked
   stage-16 evidence; `localvol` had nothing to delegate to, so stage 11 had to
   re-derive its own MC authority every run, which is the expensive path that
   produced the original defect;
3. **CI anchors** — the deterministic arm costs ~1 s per case, and nothing
   previously guarded any local-vol engine.

## Result

| quantity | mean signed bias (c) | SE of mean (c) | bound | powered? |
|---|---|---|---|---|
| pv | +0.0104 | 0.0040 | 0.1 | yes |
| delta | −0.0083 | 0.0012 | 0.1 | yes |
| gamma | +0.0049 | 0.0022 | 0.1 | yes |

All 48 cells PASS. Budget consumed: **median 5.6%, max 40.8%** — the release
procedure's warning sign is a study full of passes at 90%+, and only one cell
(`calm_european_ki` pv) exceeds 30%.

**The bias check here is a fully powered test.** `FINDING-2026-08-26` §6 had to
file `bias_bound_is_resolvable: false` on its G2 re-run — the mean's SE was
0.068 against the 0.1 bound, so the check passed without carrying information.
Here delta's mean SE is 0.0012, clearing the bound by a factor of 42.

## What is covered

Two real calibrated CSI1000 Dupire surfaces, **committed** under
`example/modelvalidation/data/` and pinned into the identity hash by sha256:

| artifact | trade date | s0 | sha256 (16) | max listed T |
|---|---|---|---|---|
| `iv_surface_20240208.json` | 2024-02-08 (crash bottom) | 4993.105 | `b0e63653a774b5b3` | 0.8658 |
| `iv_surface_20231115.json` | 2023-11-15 (calm contrast) | 6207.268 | `a7917303394e114f` | 0.8493 |

They are **not** read from `example/mo_volmodels/data/history`, which is excluded
through `.git/info/exclude` — a per-clone file that is never pushed. A study
reading from there would bank a certificate whose CI anchors fail on every
machine but the one that built them.

Eight case shapes per surface: `ordinary`, `inside_listed_grid`, `near_ko`,
`near_ki`, `discrete_ki`, `european_ki`, `stepdown_ko`, `near_expiry`.

## What is NOT covered — do not infer it

A certificate covers **only the configurations its YAML names**. That lesson is
banked from the 2026-08-19 variant-surface amendment, which found three defects
hiding in configurations the original certificate never enumerated.

- the flat-BSM product variants — `parachute`, `airbag`, `protection_*`,
  `reverse`, `call_rebate`, `participation`, `disable_ko_after_ki`,
  `coupon_pay_type`, `is_annualized`, `ko_rate_step`. These exercise payoff code
  inherited from `SnowballPDESolver`, certified under `snowball-flat-bsm`;
- surfaces other than the two above;
- maturities beyond 1.0 y, where these surfaces are majority-extrapolated
  (`max_listed_T ≈ 0.87`, `extrapolation_policy = flat_total_variance`);
- the quadrature engine — there is no local-vol quadrature engine.

## The reference, and how its configuration was chosen

```
LocalVolSnowballMCEngine   randomized_quasi   plain estimator
substeps_per_interval = 8  lv_time_sampling = integrated
paths_per_batch = 65536    paired central difference (CRN)
```

**One discretization serves pv, delta and gamma.** Splitting it would estimate
`P(h)` and `P(h/2)` — different numbers at finite `h` — so the certified delta
would not be the derivative of the certified price.

Every value was **measured before the run, not inherited**
(`docs/modelvalidation/pilot-localvol-1d/RESULTS.md`):

| control | result |
|---|---|
| 1. convergence ladder 4/8/16 | 8→16 shifts 0.06σ (`ordinary`), 0.55σ (`near_ki`) — stopped moving |
| 2. estimator choice | `plain` meets the 0.125 c SE budget on all three quantities → adopted by pre-registered tiebreak |
| 3. flat-surface collapse | LV PDE ≡ flat-BSM PDE to **1e-13** relative |
| 5. economic scale | contract ratio `1.000000` on both surfaces |

`FINDING-2026-08-26` §5 demonstrated `substeps=8` for a *three-year* trade; this
is a one-year trade at a different notional and does not inherit that, so the
ladder was walked again here. `substeps=4` would also have passed at 2.7× less
cost and was **not** adopted — the reading is borderline on one cell (1.0σ), and
this whole exercise exists because a reference once ran under-resolved.

`one_step_survival` measured 2.5× tighter on delta and 1.9× faster. It was **not**
swapped in: the decision rule was written before the numbers, and adopting the
better-looking arm afterwards is the post-hoc selection the release procedure
exists to prevent. It is banked in `RESULTS.md` for a future amendment. As a
by-product the two independent estimators' deltas agree to **0.45σ**, a
`plain`-vs-OSS control at this study's own scale.

## Surface-slope convention

`FINDING-2026-08-26`'s two reported slopes are measured on two *different*
surfaces, which is why neither reproduced directly. `dσ/dlnS = −0.371` is the
**Dupire** skew at the long end (`t ≈ 0.866`, ±0.02 window); `dσ/dt = −0.082` is
the **implied** ATM term slope over 0.05→1.00 y. The Dupire skew is also strongly
maturity-dependent (−0.056 at `t = 0.5` vs −0.369 at `t = 0.866`). A convention
note has been added to the FINDING; its conclusions are unaffected.

The calm contrast is 5.5× flatter than the crash surface on the reconciled
metric, but is *third* flattest rather than flattest — deliberately, because
slope-flatness and empirical-calmness rank the cohort differently
(`2024-06-14` has the flattest skew yet was the second-worst cell in the
FINDING). The contrast therefore rests primarily on the empirical per-cell gaps.

## Consequences for callers

- **Stage 11 may now delegate `delta_authority` for `localvol` to this
  evidence**, as `heston` / `heston_slv` do for stage 16.
- **The fleet's existing localvol numbers must be discarded.** G2 routed
  `localvol` to MC and stage 12 forwards only the keys the decision recorded,
  which omitted `substeps_per_interval` — so `engine_factory` fell through to
  the engine default of 1. Every localvol MC price and delta produced under that
  routing carries the bias `FINDING-2026-08-26` measured. (Outstanding from
  FINDING §7; not addressed here.)
- **The 40.6 CPU-hour localvol cost projection needs re-measuring.** It was taken
  under `route=pde`. (Outstanding from FINDING §7; not addressed here.)

## Anchors

`anchors.json` holds 16 anchor groups. No new test was needed:
`test/modelvalidation/test_banked_certificates.py` globs
`certificates/*/*/anchors.json`, so this certificate is guarded on every commit
automatically, at roughly one second per case.
