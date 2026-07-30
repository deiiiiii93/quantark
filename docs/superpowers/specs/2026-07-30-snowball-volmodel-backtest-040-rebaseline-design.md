# Design: Snowball Vol-Model Backtest — 0.4.0 Engine Re-baseline

Status: requirements locked with user on 2026-07-30.

**Amends** `docs/superpowers/specs/2026-07-23-snowball-volmodel-backtest-design.md`
and re-opens Phases 3 and 6 of
`docs/superpowers/plans/2026-07-23-snowball-volmodel-backtest.md`.

The original study is unchanged in objective, term sheet, and data plan. What
changed is the pricing library underneath it: quantark 0.4.0 replaced the PDE
grid construction layer outright and reworked the autocallable quadrature
engines. This amendment records what that invalidates, what it makes newly
possible, and the revised gates, run matrix, and cost model.

---

## 1. Trigger: what 0.4.0 changed

`CHANGELOG.md` 0.4.0 (2026-07-27) plus `fdf3a70` (2026-07-28):

- **Deleted** `time_grid.py`, `spatial_grid.py`, `event_projection.py`, the
  class-level shared grid cache, and 14 `PDEParams` knobs (`auto_grid`,
  `time_grid_type`, `grade_exponent`, `event_min_steps_per_interval`,
  `log_dx_target`, `include_spot_in_critical_points`, `frozen_critical_points`,
  `barrier_refine_log_width/levels`, `barrier_domain_expand`, `adaptive_grid`,
  `event_steps_per_day`, `max_time_steps`, `max_grid_size`).
- **Added** the declarative `quantark.asset.equity.engine.pde.grid` layer:
  `GridRequest` → `GridBinder` → shareable `SpatialLayout`, accuracy profiles
  `fast`/`standard`/`high`, one time builder, one spatial builder,
  four-stage `EventSchedule`.
- **Added** `HestonSLVADICore(x_nodes=...)`: the 2D solvers now take their
  S-axis from the *same* spatial builder as the 1D solvers
  (`snowball_vol_pde_solvers.py::_layer_x_nodes`, bound at `num_std=8`).
- **Changed** the quadrature autocallable engines: `filter_unreachable_barriers`
  (default on), `event_projection=CELL_AVERAGE` (default), `integration_rule`
  `"trapezoid"` (default; phase-stable after discontinuous events), and
  `auto_converge` with `convergence_rel_tol`/`convergence_abs_tol`/
  `max_convergence_grid_points`.
- **Stated in the changelog:** all PDE prices reprice.

### 1.1 Verified against this study's code

| Check | Result |
|---|---|
| Do stages 11/12/13 use any deleted knob? | **No.** They pass bare `PDEParams()`; `n_x/n_v/n_t` go only to the 2D vol-model solvers, where those knobs remain live. |
| Does `quantark/backtest/otc/` pass a rejected knob? | **No.** `engine_factory.py` uses `PDEParams()` defaults throughout. |
| Does the framework still run end-to-end on 0.4.0? | **Yes.** `12_snowball_volmodel_backtest.py --quick --max-inceptions 1` completes Gate G4 and replays `flat_bsm` and `ts_bsm` with no code change. |
| Do prices move? | **Yes, materially.** The 2023-05-04 fair coupon solves to **15.0707%** on 0.4.0 versus **15.0975%** recorded on 0.3.0 — a 2.7 bp shift in the *contract terms*, not merely the valuation. |

The last row is the decisive one: pre- and post-0.4.0 runs price different
contracts and cannot be pooled.

---

## 2. Status of the Phase-A fleet output

`output/volmodel_backtest/run_manifest.json`, elapsed 6.43 h:

- `runs_expected: 81`, `runs_completed: 12`, `runs_failed: 69`.
- All 69 failures are one `BrokenProcessPool` — the pool was killed, not a
  pricing error.
- The 12 survivors are 4 inceptions × 3 variants, priced on 0.3.0.

**Decision: discard.** Per §1.1 they are a different pricing function, and the
solved coupon differs. `output/volmodel_backtest/` is re-created from scratch.
The 554-entry `calibration_cache/` is retained — it is keyed by surface sha and
holds Heston/leverage calibrations that are engine-independent.

---

## 3. What stands, what is void

| Element | Status |
|---|---|
| Objective, §2 term sheet, §4 data plan, locked decisions 1–5 | **stand unchanged** |
| Daily SABR surfaces, thin-surface exclusions, extrapolation policy | **stand** |
| Gate G1 (surface arbitrage), Gate G3 (accounting sanity), Gate G4 (coupon) | **stand**, re-execute on 0.4.0 |
| §3.6 "PDE preferred for ALL variants" | **superseded** by §4 routing |
| Gate G2 decision `route=mc` | **void** — measured against deleted code |
| Plan Task 6.1 cost model (`3.2·T²`, 344 CPU-h) | **wrong** — see §7 |
| §6 run matrix, 5 variants | **superseded** — 6 variants, §4 |
| Phase-A output (12 runs) | **discarded**, §2 |

Gate G4's *algorithm* survives and was re-verified: PV is exactly affine in the
coupon because KO/KI triggers do not depend on it, so Illinois false position
lands the root in one iteration (`1 iters, |PV|=0.00` on 0.4.0). The algorithm
is intact; every number it produces is different.

---

## 4. Revised run matrix — six variants

Measured on the production term sheet (3Y, 34 KO observations, 726 daily-discrete
KI observations, real surfaces), price only, per solve:

| remaining | PDE `standard` | QUAD 1001 pts | speedup | PV spread |
|---|---|---|---|---|
| 3.01 y | 15.06 s | 0.68 s | 22× | 0.0036% of notional |
| 2.00 y | 7.51 s | 0.45 s | 17× | 0.004% |
| 1.00 y | 2.65 s | 0.15 s | 18× | 0.015% |
| 0.25 y | 1.16 s | 0.02 s | 58× | 0.0004% |

Quadrature agrees with the PDE inside 0.015% of notional and is 17–22× cheaper.
It is already constructed by `otc/engine_factory.py:108`, so routing is a
configuration change, not new plumbing. Its limit is structural: the FFT
convolution kernel must be spatially homogeneous, so quadrature supports
time-varying vol (`build_quad_term_params` supplies per-step `(rate, div, vol)`)
but **cannot** represent Dupire local vol or Heston.

| variant | engine | vol input | role |
|---|---|---|---|
| `flat_bsm` | PDE 1D | ATM IV at remaining maturity | baseline for `localvol` |
| `flat_bsm_quad` | QUAD | identical to `flat_bsm` | **engine control (new)** |
| `ts_bsm` | QUAD | ATM pillar term structure | vs `flat_bsm_quad` |
| `localvol` | PDE 1D | Dupire off the SABR grid | vs `flat_bsm` |
| `heston` | gate-decided | per-day Lewis calibration | vs `flat_bsm` |
| `heston_slv` | gate-decided | + FP leverage surface | vs `flat_bsm` |

Every headline comparison is same-engine:

- `localvol − flat_bsm` — both PDE 1D
- `ts_bsm − flat_bsm_quad` — both QUAD
- `heston{,_slv} − flat_bsm` — cross-engine, bounded by the §5 gate tolerance

`flat_bsm − flat_bsm_quad` is a **pure engine difference**: identical inputs,
identical solved terms, identical market path. Its spread across 27 inceptions
is the noise floor below which no model edge is interpretable. The original
5-variant matrix had no way to produce this number; it cost 3.5 CPU-hours to add.

---

## 5. Gate G2, re-scoped as the engine admission gate

G2 stops being "PDE or MC for Heston/SLV" and becomes: **is each variant's
production engine admissible, in PV and in delta, against an independent
method.** Structure is unchanged — same 8 sample dates × {full, decayed} cells,
same fail-closed refinement ladder, same `gate_decision.json` +
`evidence_sha256` contract that stage 12 already consumes via `GateRouting`.

### 5.1 Reference per route

| variant | production | independent reference |
|---|---|---|
| `flat_bsm` | PDE 1D | QUAD at high `grid_points` |
| `flat_bsm_quad` | QUAD | PDE 1D `accuracy="high"` |
| `ts_bsm` | QUAD | PDE 1D `accuracy="high"` |
| `localvol` | PDE 1D | `LocalVolSnowballMCEngine` (RQMC) |
| `heston` | gate-decided | `QESnowballMCEngine` (RQMC-QE) |
| `heston_slv` | gate-decided | `HestonSLVQESnowballMCEngine` |

`flat_bsm` and `flat_bsm_quad` reference each other. This is not circular: it is
one PDE-vs-QUAD comparison serving as the admission test for both routes and as
the study's engine control. Every route now has an independent reference of a
genuinely different numerical method — finite differences, FFT regime-switching
quadrature, and quasi-Monte-Carlo — which was not guaranteed:
`LocalVolSnowballMCEngine` (`snowball_vol_mc_engines.py:195`) is what makes the
`localvol` route checkable rather than self-referential.

### 5.2 PV tolerance — unchanged, with one corrected detail

0.25% of notional, plus 2σ of MC standard error where the reference is MC, plus
the existing bias detector (`bias_sign_fraction` 0.9, `bias_median_fraction_of_tol`
0.5) and the `medium→fine` drift bound.

**Correction:** the bias detector must be evaluated **within maturity buckets**,
not pooled across them. §7A measures a discretization error whose sign flips
between short and long remaining maturity; pooled, that scores a 0.75 sign
fraction and reads as unbiased, which is how the original G2 recorded
`biased: false` at 0.533 while a systematic per-maturity bias was present. Within
any single maturity the sign is unanimous.

### 5.3 Delta tolerance — derived from the hedge instrument

The study is a *hedging* study: PnL is driven by model-consistent delta, not PV.
The original G2 gated PV only. The tolerance is derived from what the hedge can
physically express rather than chosen as a percentage.

Position size is `contract_multiplier = notional / S₀` index units
(`12_snowball_volmodel_backtest.py:461`); IM futures carry
`FUTURES_MULTIPLIER = 200` index points per contract. So the hedge moves in
quanta of `200 · S₀ / notional` per-unit delta. Worked for the 2023-05-04
inception (`S₀ = 6733.97`, `contract_multiplier = 7425.5`):

```
1 IM contract = 0.0269 per-unit delta = 13,468 CNY exposure per 1% spot move
½ IM contract = 0.0135 per-unit delta =  6,734 CNY per 1% spot move
              = 1.35% of full-delta notional exposure (500,000 CNY per 1%)
```

`S₀` is the inception spot and differs per inception — **4,532.52** (2024-09-02)
to **6,733.97** (2023-05-04) across the 27 — so the delta quantum spans
`0.01813` to `0.02694` per contract, a factor of 1.49. The threshold is
therefore defined **in contracts** and the per-unit delta and cash equivalents
are computed per inception, never fixed at the figures above.

- **Per-cell admission:** `|Δ_production − Δ_reference| ≤ ½ IM contract`. Below
  this, contract rounding provably absorbs the disagreement and it cannot change
  a single trade.
- **Bias bound:** mean signed delta difference ≤ **0.1 IM contract**, because
  rounding can still let a small systematic bias accumulate over ~700
  rebalances. Mirrors the PV bias detector.

Both are expressed in cash per 1% spot move — the same unit stage 13 reports
hedge quality in (`RMS of post-hedge residual delta`) — so gate error and
measured result are directly comparable rather than living in incommensurate
scales.

### 5.4 Tolerances are relative to notional, never to PV

At inception the fair-coupon solver drives PV to ≈0 by construction, so any
PV-relative tolerance is unsatisfiable there. Measured: `QuadParams(auto_converge=True)`
raises `NumericalError: convergence was not reached by
max_convergence_grid_points=64001; last successive PV difference was 0.00128707`
at T=3 — a residual of **1.9e-7 of notional**, rejected only because the
tolerance is read against a near-zero PV.

Where `auto_converge` is used, `convergence_abs_tol` is set from notional and
the relative component is disabled. The existing G2 already respects this
(`tol_abs_pct_notional`); the new quadrature path must be made to.

---

## 6. Replay termination at knock-out

**Current behaviour is a defect for this study.** The 2023-05-04 trade knocked
out on 2025-09-04 (life 2.34 y) yet the run priced all 726 trading days to the
2026-05-06 maturity, where 570 would have sufficed — **156 trading days of a
terminated contract**, with records to match.

**Revised rule:** the replay ends on the later of the KO observation date and
its resolved settlement time, taken from the product's KO records
(`snowball_option.py:1041`, `rec.settlement_time`; `CouponPayType.EXPIRY` would
defer it to maturity). The KO cash must land in the ledger before the run stops.
Under the current term sheet this resolves to T+0, i.e. the KO observation date;
the rule is stated generally so adding a settlement lag does not silently change
the window.

The run manifest records `termination_reason ∈ {ko, ki_maturity, maturity,
data_end}` and both `days_replayed` and `days_in_contract`, so the truncation is
visible in the output rather than inferred. Stage 13's completeness check
(`verify_run_completeness`) validates against `days_replayed`.

Unchanged: KI does **not** terminate — a knocked-in trade runs to maturity.

---

## 7. Corrected cost model

The plan's model (`3.2·T²` s/day, 344 CPU-h, Phase A 8.2 h wall) integrated cost
over the trade's *life*, while the runner replayed the full *contract calendar*.
That is the whole of the ~2× miss. Mean realized trade life over the 27
inceptions is **0.78 y** — the plan's 0.77 y figure was correct; it was applied
to the wrong window.

### 7.1 What a replay day actually costs, and why it cannot be extrapolated

Per-solve timings do **not** compose into per-day cost, for two measured reasons:

1. **Layout reuse.** 0.4.0's `GridBinder` caches spatial layouts (LRU,
   `bind_shared`) and bump contexts reuse the base layout by object identity, so
   a warm solve costs far less than the cold single-solve figure.
2. **The greeks path differs per engine.** `_replay.calculate_greeks`
   (`otc/_replay.py:264–267`) branches on whether the engine overrides
   `BaseEngine.calculate_greeks`. Every PDE solver does
   (`snowball_pde_solver.py:1067`), so it takes the **native** path — one extra
   `self._solve()` with delta/gamma read off the grid. MC engines inherit the
   base method and take the **central-bump** path — two extra full prices.

So a PDE-priced day is `price` + one native-greeks solve = **2 solve-equivalents**;
an MC-priced day is **3 MC prices**. Measured confirmation: 29.56 s/day for
`flat_bsm` at T≈3 against a 15.06 s cold solve is a ratio of **1.96**.

Two hypotheses were tested and falsified, and are recorded so they are not
re-proposed:

- **Auxiliary engines are a major cost.** They are not. Measured with daily event
  probabilities ON: 29.56 s/day; OFF: 29.60 s/day — **−0.2%, zero within noise**.
  The event-stats engine rides the cached layout. Consequently the option of
  re-routing the auxiliary engines to quadrature is void: there is nothing to
  save. **Daily event probabilities stay ON** (owner decision 2026-07-30,
  confirming the 2026-07-25 decision on measured evidence).
- **Per-day cost follows a power law in `T`.** It does not, because per-day fixed
  overhead (surface load, calibration cache read, hedge accounting, CSV writes)
  floors the cost at short remaining maturity while the KI schedule shrinks more
  slowly than `T²`.

### 7.2 Figures that survive validation

| quantity | value | basis |
|---|---|---|
| PDE-priced variant, full 3 y contract calendar | **5.67–5.81 h** | 4 completed fleet runs, measured |
| PDE-priced variant, T≈3 | **29.56 s/day** | 25-day stage-12 smoke, measured |
| Replay-days, KO-terminated vs not | **5,039 vs 12,995** | realized KO dates, path arithmetic |
| Cold Heston + SLV calibration | **2.76 s/date**, 780 dates = **0.60 CPU-h** once, shared | measured |

Realized KO dates are pure path arithmetic — realized spot against 103% of each
inception spot on the monthly schedule — so all 27 are derivable without pricing.
KO termination removes **61% of replay-days**, and its benefit is strongly
inception-dependent: small on the earliest inceptions (the removed tail is the
cheap low-`T` end) and large on the latest (the removed span is the expensive
`T≈3` head). It must be computed per inception, never scaled.

### 7.3 What is deliberately not estimated

Fleet totals for the QUAD-priced and MC-priced variants are **not stated here**.
No such routing has ever been run end-to-end — it does not exist in the code yet
— and §7.1 shows why single-solve extrapolation is unsound. Earlier drafts of
this section carried figures (`flat_bsm_quad` 28.3 CPU-h, `heston@mc` 52.5,
a 311.3 CPU-h matrix total, a "48% aux-routing saving") derived from a solve
decomposition that measurement has since falsified; they are withdrawn.

Plan **Task 6.1** already prescribes the correct instrument: time one inception
across the variants, then extrapolate. That step is now load-bearing rather than
a formality, and the fleet total is set there — from a measured per-day curve
across remaining maturity, validated against the 29.56 s/day and 5.67–5.81 h
anchors in §7.2.

**Staging is retained as a review gate, not a resource gate.** The Phase A/B
split existed to manage a 39 h+ run. Whatever Task 6.1 measures, the checkpoint
after the 1D block stays — inspect the engine-control spread and the 1D results
before committing to the 2D block.

---

## 7A. The 2D PDE–MC disagreement: cause identified

**Conclusion: the 2D ADI scheme is sound. The disagreement measured on the
production sheet is the `v=0` boundary treatment in the Feller-violated regime,
reachable only because the production Heston calibration is pinned at its bounds.**

This section supersedes two earlier wrong diagnoses of mine, both recorded so they
are not revisited: a "systematic PDE scheme bias", and "time under-resolution of a
dense event schedule". Neither survived measurement.

### 7A.1 Owner's independent controlled case

An owner-authored controlled case (`S₀=K=100`, `T=1`, `r=2%`, `q=1%`, Heston
`v0=θ=0.04, κ=2, σ=0.30, ρ=−0.50`, monthly KO at 103 @ 12%, monthly discrete KI
at 75) found 2D ADI converging cleanly into the QE-M RQMC 95% interval:

| resolution | PV | successive move |
|---|---|---|
| 64×24×96 | 1.170888 | — |
| 96×36×192 | 1.189080 | 0.018192 |
| 144×54×384 | 1.194053 | 0.004973 |
| **216×81×768** | **1.196041** | **0.001988** |
| QE-M RQMC, 1,048,576 paths | 1.208693 | SE 0.006915 |

Extra-fine PDE is inside `[1.195139, 1.222248]`; gap **1.27 bp of notional**,
1.83 MC standard errors. Increments contract at ratios 3.7 and 2.5 against a
refinement factor of 1.5 — a **convergent, roughly second-order scheme**.

### 7A.2 Two candidate explanations, both refuted

The production sheet showed gaps of 185–404 bp — ~150× the controlled case. Two
hypotheses were tested at T=1 on the production sheet:

- **Dense event schedule.** The controlled case has 24 discrete events against
  `n_t=768` (~32 steps per event interval); the production sheet has 254
  events/year against `n_t=ceil(400·T)` (~1.6). **Refuted:** refining `n_t`
  400→3200 (steps/event 1.57→12.60) moved the PDE only −38,719 cash *in total*,
  and **away** from MC. Successive moves contracted cleanly (−23,217, −12,782,
  −2,720) — the PDE was converging, to the wrong value.
- **Biased MC reference.** `QESnowballMCEngine` defaults to
  `martingale_correction=False` (scheme `QUADEXP`), and the committed gate
  decision records `QUADEXP` with `substeps_per_interval=1`, while QE-M exists to
  remove coarse-step martingale bias. **Refuted:** gate config vs QE-M/8-substeps
  differs by only **+0.078% of notional** — real, but not a 2.5% explanation.

### 7A.3 The cause, with a control

The production calibration for 2025-05-06 returns `κ=3.000` and `σ=0.700`, which
are **exactly the `mo_frozen` upper bounds** `(0.5, 3.0, 0.5, 0.7, 0.0)`. That
gives `2κθ/σ² = 0.540` — **Feller violated**, so variance reaches zero.
`HestonSLVADICore` supports `v0_boundary="degenerate_pde"` for precisely that
regime, but defaults to `"neumann"` and `snowball_vol_pde_solvers.py` never passes
it, so the production route cannot reach it.

Forcing the boundary at core level (T=1, production sheet, 200×60, MC reference
QE-M / 8 substeps / 32 batches):

| case | `2κθ/σ²` | `v0_boundary` | n_t=400 | n_t=1600 | inside MC 95% CI |
|---|---|---|---|---|---|
| production | 0.540 (violated) | `neumann` | −2.467% | −2.539% | no |
| production | 0.540 (violated) | `degenerate_pde` | +0.334% | +0.553% | no |
| **control** (σ 0.700→0.200) | 6.617 (satisfied) | `neumann` | +0.104% | **+0.034%** | **yes** |
| **control** | 6.617 (satisfied) | `degenerate_pde` | +0.106% | +0.036% | yes |

The boundary switch moves the violated case by ~3% of notional and cuts the gap
**4.6×**; in the satisfied case the two treatments agree to **0.002% of notional**
— the flag matters exactly where theory says it must and nowhere else. The control
also reproduces the owner's result **on the production sheet**, with its 254
events/year, confirming event density was never the issue.

Residual: even with `degenerate_pde` the violated case sits +0.33%/+0.55% off and
drifts *away* from MC under refinement. The boundary fix removes the dominant
error, not all of it — which is why §7A.4 constrains the calibration instead of
relying on the boundary alone.

### 7A.4 Decisions

1. **`enforce_feller=True`** in the `mo_frozen` Heston preset (owner decision
   2026-07-30; currently `False` with only a soft `regularize_feller=0.05`
   penalty). No date may then produce a degenerate parameter set, both engines
   agree, and the PDE route reopens. Cost: a worse smile fit on the dates that
   previously violated Feller — this **must** be reported as per-date calibration
   RMSE, since it changes the model being tested.
   Cache safety verified: the heston fingerprint embeds the full preset contents
   (`vol_calibrators.py:542`) and `heston_slv` chains it, so the key changes
   automatically — no stale hits and no `_CACHE_SCHEMA_VERSION` bump. The 552
   cached `localvol-` entries stay valid; their fingerprint excludes the preset.
2. **Plumb `v0_boundary` and default it to `degenerate_pde`** for the snowball and
   phoenix vol PDE solvers. **Mandatory, not belt-and-braces** — see §7A.6. An
   earlier draft called this redundant after (1); the Feller-boundary sweep
   disproves that. `enforce_feller=True` lands constrained fits *on* the boundary
   at `2κθ/σ² ≈ 1.0`, which is exactly where `neumann` fails (−0.540% of
   notional). The two fixes are complementary: (1) removes the deep-violation
   regime, (2) is required for the marginal regime that (1) produces.
3. **G2 records `2κθ/σ²` and bound-hit flags per date**, and evaluates its verdict
   **conditioned on the Feller ratio** rather than pooling regimes. §7A.3 shows a
   uniform verdict would average a 0.03% regime with a 2.5% one.
4. **MC reference upgraded** to `martingale_correction=True` with
   `substeps_per_interval` ≥ 4. The measured reference bias is only 0.078% of
   notional, but it is free to remove and the gate's tolerance is 0.25%.

### 7A.6 Feller-boundary sweep and accuracy-matched performance

`enforce_feller=True` constrains the fit to `2κθ ≥ σ²`; a constrained optimiser
pushed by the data sits **on** that boundary, at ratio ≈ 1.0, not comfortably
inside it. §7A.3's passing control was at 6.617, so it did not test the regime the
constraint actually produces. Sweeping σ to hit target ratios at T=1 on the
production sheet (κ, θ, v0, ρ held at the calibration; PDE 200×60×1600; MC QE-M,
8 substeps, 32 batches = 262,144 paths):

| `2κθ/σ²` | σ | `neumann` | verdict | `degenerate_pde` | verdict | MC | PDE | speedup |
|---|---|---|---|---|---|---|---|---|
| **1.000** | 0.5145 | **−0.540%** | **FAIL** | **+0.156%** | **PASS** | 39.7 s | 8.6 s | **4.6×** |
| 1.250 | 0.4601 | −0.242% | PASS (marginal) | +0.112% | PASS | 46.3 s | 8.8 s | 5.2× |
| 1.500 | 0.4200 | −0.076% | PASS | +0.101% | PASS | 49.5 s | 9.0 s | 5.5× |
| 3.000 | 0.2970 | +0.070% | PASS | +0.051% | PASS | 48.3 s | 8.8 s | 5.5× |
| 6.617 | 0.2000 | +0.034% | PASS | +0.036% | PASS | — | — | — |

Verdicts apply the gate criterion `max(2·mc_se_pct, 0.25%)`, which is the 0.25%
floor throughout (`mc_se` ≈ 0.023% of notional).

Two conclusions:

1. **`degenerate_pde` is required.** It holds +0.10 … +0.16% across the whole
   regime range and passes everywhere, while `neumann` swings from −0.540% (FAIL
   at the boundary) to +0.034%. A treatment whose error is insensitive to regime is
   the correct default; one that fails precisely where `enforce_feller` lands is not.
2. **Accuracy-matched, the 2D PDE is 4.6–5.5× faster than MC** at T=1, measured at
   the configurations that pass. This reverses an earlier claim in this spec that
   MC was the cheaper route — that comparison had put PDE at a passing resolution
   against MC at the gate's *failing* configuration.

For a hedging study the margin is wider than the table shows: daily deltas come
from bumped re-prices, so MC pays its cost three times per day (§7.1) and
differences two noisy numbers, whereas the PDE reads delta and gamma off the same
solve (`snowball_pde_solver.py:1124`) at no extra cost.

**Two caveats bounding this result.** Measured exponents are `pde2d ≈ T^1.78`
against `mc ≈ T^1.04–1.20`, so the advantage narrows with maturity — an
extrapolation to T=3 gives ~92 s vs ~136 s, thin enough that it must be measured,
not assumed. And every fixed-configuration number here is `heston`;
`heston_slv` adds a leverage surface on the same ADI core and is **not** assumed to
inherit the result. Both are G2's job.

**Prediction on the record:** with (1) and (2) applied, G2 admits the PDE route for
both 2D variants. Recorded as a prediction so the gate can falsify it.

### 7A.5 Method note

The first version of this evidence was rejected by the owner for using cases whose
|PV| was too small to support a conclusion. That objection was correct and is what
led here. The replacement design sweeps the KO coupon (0.05/0.15/0.30) to move
|PV| widely **at a fixed state**, separating conditioning from error, and records
MC standard error so the gate's real criterion
`max(2·mc_se_pct, 0.25%)` (`11_pde_convergence_gate.py:532`) can be applied.
Measured `mc_se` is 0.003–0.076% of notional, so the tolerance is the 0.25% floor
throughout and MC is a sound arbiter.

One property of that probe worth preserving in G2: the PDE error **changes sign
with maturity** (positive at T=0.25, negative at T≥1). Pooled, that scores a 0.75
sign fraction — below the 0.9 threshold — which is how the original G2 recorded
`biased: false` at 0.533 while a systematic per-maturity bias was present. The
bias test must be evaluated within maturity buckets (§5.2).

For the record, the 24-cell probe run under the **unfixed** configuration
(`v0_boundary="neumann"`, bound-pinned calibration) measured:

| T | gap % notional | gap % \|PV\| | spatial drift (200×60 → 300×90) |
|---|---|---|---|
| 0.25 | +0.001 … +0.075% | 0.2–2.6% | ~0.000% |
| 1.00 | −1.85 … −4.04% | 15–198% | 0.40–0.86% |
| 2.00 | −0.74 … −0.85% | 3.6–6.1% | 0.115–0.150% |
| 3.01 | −1.05 … −3.13% | 24–178% | not measured |

Two properties of that table are worth keeping, because they are what made the
diagnosis findable: the gaps do **not** shrink when |PV| is large (T=2.00 /
coupon 0.05 has PV at 20% of notional and still fails by 3.3×), and at T=2 they
**survive spatial convergence** (drift 0.115–0.150%, passing). A gap immune to
both PV magnitude and spatial refinement is not a conditioning artifact and not
under-resolution — which is what pointed at the boundary condition.

Note this probe refined **space only**, holding `n_t = ceil(400·T)` fixed at both
ladder levels; the `n_t` sweep in §7A.2 supplied the time axis separately.

**Routing is no longer predetermined.** Earlier drafts concluded
`heston`/`heston_slv` would stay on `route=mc`. With §7A.4 (1) and (2) applied the
PDE route is viable again — the control reaches +0.034% of notional, inside the MC
95% interval — so G2 decides on fresh evidence, per Feller regime, and the 2D cost
question in §7.3 reopens with it.

---

## 8. Outcome concentration — a stronger caveat than the plan carried

The 27 realized KO dates collapse onto ~13 distinct days, and **2024-10-08
terminates seven inceptions at once**; 2025-09-01/03/04 accounts for four more.
Overlapping inceptions do not merely correlate — the trades share terminal
events.

Consequences the report must state, not bury:

1. Effective sample size is far below 27. Stage 13's two-standard-error
   significance test will substantially overstate confidence.
   `_outcome_concentration_caveat` must report the KO-date histogram and an
   effective-sample-size estimate, derived from run data as the existing
   caveat is.
2. **All 27 inceptions knock out.** None reaches maturity knocked-in — the state
   where snowball model risk peaks and the seller absorbs equity downside. Any
   null result is therefore "no measurable edge *on knock-out paths*", which is
   a narrower claim than the study's title suggests, and must be worded that way.

This is a property of the 2023-05 → 2026-07 CSI 1000 path, not a defect. It
bounds the conclusion and is reported as a data-derived finding.

---

## 9. Pre-flight grid-resolution sweep (new)

`fdf3a70` promoted the spatial-grid resolution check from `logger.warning` to a
fail-closed `ValidationError` (`grid/space.py:288–303`): a solve now raises when
`max/min` spacing ratio exceeds `_MAX_DX_RATIO = 100` or achieved spacing
exceeds `2 × eps_crit` (0.003 for `fast`/`standard`, 0.002 for `high`).

This is correct behaviour and closes the defect that let `accuracy="standard"`
ship a +116% sign-flipped price. But it converts a silent accuracy loss into a
mid-fleet exception. Stage 12 already contains the blast radius — a failed run is
recorded in `run_manifest.json` with its traceback rather than killing the fleet
(plan Task 4.2) — so the residual risk is not a wrong number, it is discovering
after 143 CPU-hours that some inceptions were never priceable.

**Requirement:** before the fleet launches, sweep every (inception, remaining
maturity) operating point and *build the grid only* — no solve — at the
configured accuracy profile. Cost is negligible next to the fleet. Any operating
point that raises is resolved by profile or `GridConfig` choice before
committing compute, and the sweep result is recorded alongside the gate
decision.

---

## 10. Sequencing

0. **Secure the framework first.** Phases 1–5 are entirely uncommitted: 519
   lines across `quantark/backtest/otc/` plus untracked `vol_calibrators.py` and
   `vol_history.py`, stages 11/12/13, and 17 test files — none in HEAD, in a
   tree through which another session pushed 41 engine commits. Commit this
   scope, and only this scope, on a feature branch. The 13 modified
   option-product files and other sessions' WIP are left untouched.
1. **Apply the §7A.4 engine and calibration fixes first** — `enforce_feller=True`
   in the `mo_frozen` preset, `v0_boundary` plumbed through the snowball/phoenix
   vol PDE solvers, MC reference on `martingale_correction=True` with
   `substeps_per_interval` ≥ 4. G2 must not be re-run before these land, or it
   will certify the configuration §7A just disproved.
2. Re-execute Gate G4 (coupon solve) and Gate G1 (surface admission) on 0.4.0.
3. Re-run and re-scope Gate G2 per §5, with the verdict conditioned on the Feller
   regime and `2κθ/σ²` recorded per date; emit a fresh `gate_decision.json` and
   evidence hash. The 2D route is genuinely open — §7A.3's control reaches +0.034%
   of notional, inside the MC 95% interval.
4. Implement §6 termination, the `flat_bsm_quad` variant, and §9 pre-flight.
4. Gate G3 on one inception, 0.4.0.
5. **Task 6.1 timing run** — measure the per-day cost curve across remaining
   maturity for every route actually configured, validated against the §7.2
   anchors. The fleet total is set here (§7.3), not before.
6. Run the 1D block (4 variants × 27 inceptions). Review checkpoint.
7. Run the 2D block (2 variants × 27 inceptions) per the §5 routing.
8. Aggregate and report, with §8 caveats.

---

## 11. Validation gates, restated

- **G1** — every daily surface passes arbitrage checks after SABR smoothing;
  fail-closed and logged. *Unchanged.*
- **G2** — engine admission for all six variants, in PV (§5.2) and delta (§5.3),
  against an independent method (§5.1), with tolerances relative to notional
  (§5.4) and the bias test evaluated within maturity buckets (§5.2).
  *Re-scoped and re-run.*
- **G3** — one inception end-to-end accounting sanity via `sanity_check_run`:
  PnL decomposition, portfolio and cash identities, cost reconciliation,
  lifecycle monotonicity, hedge effectiveness, NaN screening. *Unchanged, re-run.*
- **G4** — coupon solver converges for every inception; solved coupon recorded.
  *Unchanged, re-run — the roots move.*
- **G5 (new)** — pre-flight grid-resolution sweep over every operating point
  (§9) before the fleet launches.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| G2 re-run admits Heston/SLV to PDE, changing 2D cost and results | Expected, not feared: §7A.3's control reaches +0.034% of notional inside the MC 95% interval once the §7A.4 fixes land. Gate decides on fresh evidence per Feller regime; 2D timing is measured in the Task 6.1 run before the 2D block |
| Fleet cost is unknown until Task 6.1, so the run cannot be scheduled in advance | Accepted deliberately (§7.3). Single-solve extrapolation is unsound here; a wrong estimate is worse than a deferred one. The measured anchors in §7.2 bound the PDE-priced variants |
| 2D uniform ADI time grid carries a ~0.8%-of-notional bias at T=2 (§7A) | Out of scope — the MC route avoids it. Recorded as a quantified engine defect with evidence, so event-aligned 2D time can be prioritised on data rather than intuition |
| Engine-control spread turns out large enough to swamp model edges | That *is* the finding — report it; it invalidates cross-engine comparisons honestly rather than silently |
| Quadrature cannot price some operating point (unreachable-barrier filtering, dense-KI refinement) | G5 pre-flight covers QUAD routes as well as PDE |
| Heston weakly identified on CFFEX settlement data (κ/σ bound hits ~half of sampled dates) | **Escalated.** §7A shows this has a pricing consequence, not only a parameter-stability one: bound-pinned κ/σ violate Feller and drive a 2.5%-of-notional PDE–MC gap. Mitigated by `enforce_feller=True` (§7A.4), per-date Feller ratio and bound-hit flags in the calibration records, and a G2 verdict conditioned on regime |
| `enforce_feller=True` degrades the smile fit on previously-violating dates | Report per-date calibration RMSE before/after. This changes the model being tested and must be stated as such, not buried — a constrained Heston is a different model from a free one |
| The `degenerate_pde` boundary leaves a +0.33–0.55% residual on violated dates that grows under refinement | `enforce_feller=True` removes the regime entirely; the boundary flag is belt-and-braces. If violated dates somehow persist, they are flagged per date rather than silently priced |
| MO surface ends ~1 y against a 3 y trade | Explicit flat-total-variance extrapolation, stated in the report. *Carried forward.* |
| Another session's WIP in the shared tree is disturbed | §10 step 0 commits a strict file scope on a branch |
| Fleet killed again mid-run | Stop via process group, never `pkill -f`; per-run output isolation means completed runs survive |
