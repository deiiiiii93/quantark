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

### 5.2 PV tolerance — unchanged

0.25% of notional, plus 2σ of MC standard error where the reference is MC, plus
the existing bias detector (`bias_sign_fraction` 0.9, `bias_median_fraction_of_tol`
0.5) and the `medium→fine` drift bound.

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

Fitted to the measured 0.4.0 per-solve timings and calibrated against the four
completed fleet runs (5.67–5.81 h each):

```
PDE  cost/solve = 1.717·T^1.91 + 1.005 s
QUAD cost/solve = 0.208·T^1.13 s
effective solves per replay day = 4.87   (price + central-bump greeks + daily event stats)
```

Realized KO dates are pure path arithmetic — realized spot against 103% of each
inception spot on the monthly schedule — so all 27 are derivable without pricing.
Fleet totals, 27 inceptions:

| configuration | per variant | 4 approved 1D variants | wall, 12 workers |
|---|---|---|---|
| PDE, no termination (old basis) | 138.0 CPU-h | 552.1 CPU-h | 46.0 h |
| PDE, KO-terminated | 68.3 CPU-h | — | — |
| QUAD, KO-terminated | 3.5 CPU-h | — | — |
| **approved mix** (2 PDE + 2 QUAD) | — | **143.4 CPU-h** | **12.0 h** |

A 3.85× reduction from two independent sources: termination removes 51% of PDE
cost, and quadrature takes two of four variants from 68.3 to 3.5 CPU-hours.

Termination's benefit is strongly inception-dependent and had to be computed per
inception rather than scaled: 4% on the earliest (3.55 → 3.39 h, because the
removed tail is the cheap low-`T` end) against 67% on the latest
(2.33 → 0.77 h, because the removed span is the expensive `T≈3` head).

The 2D variants are carried forward from the plan's 0.3.0-era MC measurements
(Heston RQMC-QE 7.1 s at 3.0 y, SLV 11.9 s) at roughly 180 CPU-h / ~15 h wall,
and must be re-timed once §5 fixes their routing.

**Staging is no longer a resource decision.** The Phase A/B split existed to
manage a 39 h+ run. At 12 h wall for the 1D block, the checkpoint after it is
retained as a *review* gate — inspect the engine-control spread and the 1D
results before committing to the 2D block — not as a resource gate.

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
1. Re-execute Gate G4 (coupon solve) and Gate G1 (surface admission) on 0.4.0.
2. Re-run and re-scope Gate G2 per §5; emit a fresh `gate_decision.json` and
   evidence hash. Re-time the 2D routes.
3. Implement §6 termination, the `flat_bsm_quad` variant, and §9 pre-flight.
4. Gate G3 on one inception, 0.4.0.
5. Run the 1D block (4 variants × 27 inceptions, ~12 h wall). Review checkpoint.
6. Run the 2D block (2 variants × 27 inceptions) per the §5 routing.
7. Aggregate and report, with §8 caveats.

---

## 11. Validation gates, restated

- **G1** — every daily surface passes arbitrage checks after SABR smoothing;
  fail-closed and logged. *Unchanged.*
- **G2** — engine admission for all six variants, in PV (§5.2) and delta (§5.3),
  against an independent method (§5.1), with tolerances relative to notional
  (§5.4). *Re-scoped and re-run.*
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
| G2 re-run admits Heston/SLV to PDE, changing 2D cost and results | Gate decides on evidence; timing re-measured before the 2D block |
| Engine-control spread turns out large enough to swamp model edges | That *is* the finding — report it; it invalidates cross-engine comparisons honestly rather than silently |
| Quadrature cannot price some operating point (unreachable-barrier filtering, dense-KI refinement) | G5 pre-flight covers QUAD routes as well as PDE |
| Heston weakly identified on CFFEX settlement data (κ/σ bound hits ~half of sampled dates) | Frozen bounds config, per-day diagnostics in calibration records, reported honestly. *Carried forward unchanged.* |
| MO surface ends ~1 y against a 3 y trade | Explicit flat-total-variance extrapolation, stated in the report. *Carried forward.* |
| Another session's WIP in the shared tree is disturbed | §10 step 0 commits a strict file scope on a branch |
| Fleet killed again mid-run | Stop via process group, never `pkill -f`; per-run output isolation means completed runs survive |
