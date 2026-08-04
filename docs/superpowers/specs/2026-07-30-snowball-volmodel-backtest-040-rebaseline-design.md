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
| Does the replay backtest pass a rejected knob? | **No.** `replay/engine_factory.py` uses `PDEParams()` defaults throughout. |
| Does the framework still run end-to-end on 0.4.0? | **Yes.** `12_snowball_volmodel_backtest.py --quick --max-inceptions 1` completes Gate G4 and replays `flat_bsm` and `ts_bsm` with no code change. |
| Do prices move? | **Yes, materially.** The 2023-05-04 fair coupon solves to **15.0707%** on 0.4.0 versus **15.0975%** recorded on 0.3.0 — a 2.7 bp shift in the *contract terms*, not merely the valuation. |

The last row is the decisive one: pre- and post-0.4.0 runs price different
contracts and cannot be pooled.

### 1.2 Backtest replay consolidation (landed 2026-07-30)

A second platform change arrived after this spec was first written:
`quantark.backtest.otc` was consolidated into `quantark.backtest.replay` — ONE
multi-product daily loop, with `otc/` kept as a deprecated shim until 0.5.0. It
resolves three of this spec's requirements and moves several of its citations.

**Delivered, so no longer work items:**

| This spec asked for | Delivered by |
|---|---|
| §6 KO termination at settlement | `terminate_on_lifecycle_end=True` (default) + pending-settlement semantics, `7455484` |
| Framework committed and secured | consolidation merged; stages 11/12/13 already migrated to canonical imports |
| Per-day IV-surface channel | `replay/market.py:147` `surface_history` |

**Relocations** (all citations in this spec updated):

| was | now |
|---|---|
| `backtest/otc/vol_calibrators.py` | `quantark/volmodels/calibration.py` |
| `backtest/otc/vol_history.py` | `quantark/param/vol/surface_history.py` |
| `backtest/otc/engine_factory.py` | `backtest/replay/engine_factory.py` |
| `backtest/otc/_replay.py` | `backtest/replay/product_replay.py` |

**Newly available, and relevant:**

- `replay/schema.py` — typed row schemas as the single source of truth for record
  columns (`STATE_COLUMNS`, `GREEK_COLUMNS`, `CALIBRATION_RECORD_KEYS`, …). Stage
  13's `REQUIRED_CATEGORIES` currently duplicates column lists by hand; deriving
  them from `schema.py` would make the §6.2 completeness gate impossible to drift
  from the writer. Worth doing while stage 13 is being touched anyway.
- `event_stats_fallback: Literal["none","mc"] = "none"` (`replay/config.py:133`) —
  fail-closed by default. Consistent with this study's no-silent-fallback rule.
- `futures_ledger.py` — shared `FuturesHedgePosition` / `FuturesRollPolicy`,
  replacing per-engine hedge bookkeeping.
- Greeks now **fail closed**: the silent `delta = 0.0` fallback is gone (see §7.1).

**Landed 2026-07-31:** all four §7A.4 decisions, with the cohort-wide measurements
they implied recorded in §7A.10. `enforce_feller=True` and the `v0_boundary`
plumbing are live; the MC reference runs QE-M at 4 substeps; the calibration
record now carries `feller_ratio`. Two consequences of the enforcement that were
*not* anticipated by §7A.4 are recorded in §7A.10 and §12 — they change what the
`heston`/`heston_slv` rows of the study mean on a minority of dates.

**Amended 2026-08-01:** a concurrent workstream put the surface history under a
live daily scheduler and added three calibration fields to the config the replay
uses. The cohort is no longer frozen and the fleet grows from 27 to 28
inceptions once `data_end` crosses 2026-08-01. §7A.12 records the drift, pins
`COHORT_ASOF = "20260731"`, and confirms the gates still cover what production
runs.

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
It is already constructed by `replay/engine_factory.py:135`, so routing is a
configuration change, not new plumbing. Its limit is structural: the FFT
convolution kernel must be spatially homogeneous, so quadrature supports
time-varying vol (`build_quad_term_params` supplies per-step `(rate, div, vol)`)
but **cannot** represent Dupire local vol or Heston.

| variant | engine | vol input | role |
|---|---|---|---|
| `flat_bsm` | PDE 1D | ATM IV at remaining maturity | baseline for `localvol` |
| `flat_bsm_quad` | QUAD | identical to `flat_bsm` | **engine control (new)** |
| `ts_bsm` | PDE 1D | ATM pillar term structure | vs `flat_bsm` |
| `localvol` | PDE 1D | Dupire off the SABR grid | vs `flat_bsm` |
| `heston` | gate-decided | per-day Lewis calibration | vs `flat_bsm` |
| `heston_slv` | gate-decided | + FP leverage surface | vs `flat_bsm` |

Every headline comparison is same-engine:

- `localvol − flat_bsm` — both PDE 1D
- `ts_bsm − flat_bsm` — both PDE 1D (**corrected 2026-08-02**; this pair was
  previously documented as `ts_bsm − flat_bsm_quad`, both QUAD. `ts_bsm` ships on
  the PDE default, so the same-engine vol-input comparison runs against
  `flat_bsm`. Owner decision: keep the landed engine, correct the document. The
  comparison is unchanged in kind — engine held constant, flat vs term-structure
  vol isolated — only the partner variant differs.)
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
| `ts_bsm` | PDE 1D | QUAD at high `grid_points` |
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

### 5.5 G2 outcome — measured 2026-08-02

Run: 8 sample dates × 2 cases × 6 variants × 3 ladder levels, cohort pinned at
`COHORT_ASOF = 2026-07-31`, warm calibration cache (715 entries seeded from the
daily pipeline, 3 distinct fingerprints). Wall clock **2397.8 s**.
Evidence `output/pde_convergence_gate/pde_convergence_gate.json`,
`evidence_sha256 = 4ac62ab4d820cbba57807c5edbe68f94f5fc5562b19c9d244f1a0988726df9d1`.
Calibration policy stamped into the decision: `heston_preset="mo_frozen"`,
`enforce_feller=true`, `heston_temporal_regularization=0.0`,
`slv_heston_override=null`.

| variant | route | production params | decided by |
|---|---|---|---|
| `flat_bsm` | **pde** | `pde_1d`, accuracy=standard | clean on all four criteria |
| `flat_bsm_quad` | **pde** | `quad`, grid_points=4096 | clean on all four criteria |
| `ts_bsm` | **pde** | `pde_1d`, accuracy=standard | clean — but see the caveat below |
| `localvol` | **mc** | `lv_mc_rqmc`, 8192×16 RQMC | **delta alone** |
| `heston` | **mc** | `qe_m_rqmc`, QUADEXP_M, 4 substeps | PV (1 cell) + delta + delta bias |
| `heston_slv` | **mc** | `qe_m_rqmc`, QUADEXP_M, 4 substeps | PV (1 cell) + delta + delta bias |

**PV vs delta diverge sharply.** Median |diff| in the `full` bucket against the
0.25% notional tolerance: `flat_bsm` 0.0005%, `localvol` 0.021%, `heston` 0.101%,
`heston_slv` 0.103% — every one comfortably inside. Delta, in IM futures
contracts (production − reference), against a 0.5-contract per-cell bound and a
0.1-contract bias bound:

| variant | mean signed | max abs | cells > 0.5 | biased |
|---|---|---|---|---|
| `flat_bsm` / `ts_bsm` | +0.006 | 0.029 | 0 | no |
| `localvol` | −0.050 | **1.272** | 3 | no |
| `heston` | **−0.471** | 1.209 | 5 | yes |
| `heston_slv` | **−0.338** | 1.145 | 4 | yes |

`localvol`'s PV agrees to 0.021% while its delta (~0.6 per unit) is off by ~0.034
— about 5%, a **200–300× amplification** of the relative error. Differentiating
the O(h²) grid-error field costs roughly a factor of the length scale over which
it varies, and for a snowball that scale is set by barrier proximity
(KO 1.03·S₀, KI 0.75·S₀), not by the domain width. **A PV tolerance cannot see
this**, which is why §5.3's delta criterion is load-bearing rather than
confirmatory: it is the sole reason `localvol` routes to MC.

Two distinct delta failure modes, and the pair of criteria separates them:
`localvol` is **dispersion without drift** (mean inside the bias bound, individual
cells at 1.27 contracts); `heston`/`heston_slv` are **genuinely one-signed**. A
persistent −0.47-contract gap is ~1.3% of the 50M notional left unhedged at every
one of ~750 daily rebalances, one-signed — and the backtest's P&L *is* accumulated
hedge error.

#### The delta criterion covers only the ~3Y regime

`_evaluate_case` gates the whole delta block on `if case_name == "full":`
(line 1416), so the `decayed` (~1Y) case never produces a delta row. Measured
in the evidence: **48 delta rows, every one `full`; zero `decayed`** — against
126 `decayed` PV cells. Every failing cell listed above is therefore `/full`
by construction, not by outcome.

This is pre-existing and was not introduced by §5.3's criterion, but it is
conceptually the *same* blind spot §5.2 just closed for PV bias: maturity-
dependent error hiding in an unrepresentative sample. And it is not obviously
benign in the conservative direction. On PV the decayed regime is far easier
(`heston` median 0.003% decayed vs 0.101% full), which invites the assumption
that delta follows — but a snowball's delta is most violent near a barrier as
T → 0, where gamma steepens, so the 1Y regime could plausibly be *worse* on
delta rather than better. It is simply unmeasured.

Consequence for the routes: the three MC routes are conservative — they can
only become more justified if the ungated regime is worse. The three **PDE**
routes are the exposure, since `flat_bsm`, `flat_bsm_quad` and `ts_bsm` were
admitted partly on a delta criterion that never examined half the maturity
range they will be priced over. Their measured `full`-case margins are large
(max 0.0210 contracts against a 0.5 bound, ~24×), which is reassuring but not
a substitute for measurement. Extending the delta block to the `decayed` case
is the natural follow-up and is cheap — the decayed envs and models are
already built.

**§7A.8 is falsified.** It predicted both 2D variants would be admitted to PDE at
200×60 "on delta stability more than speed". Delta stability is precisely what
rejected them, at 4.7× and 3.4× the bias bound.

**Feller conditioning localises the PV failure.** For both Heston variants the
`boundary` bucket is 13/13 passing at max 0.18%, and the single failing cell
(`2025-04-09/full`, |diff| 0.313% / 0.329%) carries Feller ratio **1901.4** — the
`degenerate` bucket. PV degrades exactly where the calibration collapses, as
§7A.10 predicted from the calibration side. That bucket holds 2 cells, below the
4-cell floor, so it is marked `skipped=true` and cannot formally flag: visible
ignorance rather than a false clean bill. `violated` is empty (n=0) — `enforce_feller`
is doing its job.

**`heston_slv`'s `full` PV bucket has sign fraction 1.0** (8/8 same direction) yet
`biased=false`, because the median stays under the 0.5×TOL threshold. This is
§7A.11's "biased but currently small" caveat, now observed at maximum
one-sidedness. It is not disqualifying today; it would become so under any
tolerance tightening.

#### Caveat: `ts_bsm` was not independently gated — **RESOLVED 2026-08-03**

*Fixed by Task 9 (`7109868`) and re-gated. The finding below is kept because it
is the evidence that motivated the fix; the resolution follows it in §5.6.*

`ts_bsm` and `flat_bsm` returned **bitwise identical** results — all 15 PV cells
and every delta, to 16 significant digits. Cause: stage 12 distinguishes them by
`surface_vol_mode` (`flat_atm_remaining` vs `term_structure`), but stage 11's
`build_pricing_env` hands every variant `artifact.grid_vol_surface()` — i.e.
`full_grid` — and never reads `surface_vol_mode`. The gate ran four distinct
computations and reported six rows.
`test_gate_covers_every_study_variant` asserts set equality of variant names, so
it cannot detect that two of them are the same computation.

Impact is bounded and in the conservative direction: `full_grid` carries more
structure (smile *and* term) than either `flat_atm_remaining` or `term_structure`,
so PDE-vs-QUAD agreement on the simpler surface should be at least as good. The
three calibrated variants genuinely use `full_grid`, so the three that *failed*
are gated on the data the fleet will use. But that conservatism is incidental, not
designed, and `ts_bsm`'s `route=pde` currently rests on no independent evidence.
Resolving this is a prerequisite for admitting `ts_bsm` to the fleet.

---

### 5.6 Re-gate on the corrected surfaces — measured 2026-08-03

Task 9 made `build_pricing_env` honour `surface_vol_mode`, mirroring
`ProductReplay._vol_and_dividend`. G2 was then re-run in full: 2442.4 s, same
cohort pin, same warm cache,
`evidence_sha256 = bbfa8f5543c2a3fc6dd0af37ef255989e603ad96ddb016a1eee17763880a238a`.

**The re-run carries its own control.** The three `full_grid` variants already
priced the right surface, so Task 9 must not have moved them; the three BSM
variants must move. Measured, comparing every cell across all three ladder
levels:

| group | variants | cells | moved |
|---|---|---|---|
| control (`full_grid`) | `localvol`, `heston`, `heston_slv` | 135 | **0** |
| treatment (BSM) | `flat_bsm`, `flat_bsm_quad`, `ts_bsm` | 135 | **135** |

And the defect itself: `ts_bsm` vs `flat_bsm` went from **45/45 cells identical**
to **0/45**. The two are now separate computations.

**All six routes are unchanged**, and every rationale string is identical to
§5.5's. The BSM variants' delta agreement improved slightly on their correct
surfaces:

| variant | mean signed (before → after) | max abs (before → after) |
|---|---|---|
| `flat_bsm` | 0.0058 → **0.0047** | 0.0286 → **0.0185** |
| `flat_bsm_quad` | 0.0008 → **0.0014** | 0.0208 → **0.0210** |
| `ts_bsm` | 0.0058 → **0.0044** | 0.0286 → **0.0184** |

All still two orders of magnitude inside the 0.5-contract bound. §5.5's
substantive findings — the delta criterion deciding `localvol`, the −0.471 and
−0.338 contract Heston biases, the Feller localisation of the single PV failure
— are untouched, because they belong to the control group.

**`ts_bsm` vs `flat_bsm` is a large effect — corrected 2026-08-03.**

*An earlier draft of this paragraph claimed the opposite: that `ts_bsm`
"tests a feature this data mostly lacks" and that a small difference should be
read as thin market structure. That was inferred from comparing vol LEVELS at
T ≥ 1, where the two curves do coincide (§5.7), and wrongly generalised to
pricing impact. Measured, the pricing impact is large.*

Comparing the two variants' own PV and delta levels in the re-gate evidence —
a variant-vs-variant comparison, not the gate's engine-vs-reference metric:

| quantity | mean | max abs | n |
|---|---|---|---|
| PV diff, `full` (T≈3) | **−0.5403%** of notional | **1.3400%** | 8 |
| PV diff, `decayed` (T≈1) | −0.0036% | 0.0171% | 7 |
| delta diff | **−1.143 contracts** | **4.033** | 8, **all negative** |

Against G2's tolerances (0.25% of notional, 0.5 contracts) that is **5.4×** on
PV and **8.1×** on delta. The two variants are genuinely different products
from a hedging standpoint.

**Why, despite the curves agreeing above `max_listed_T`.** A snowball's value
is dominated by early knock-out probability, not terminal vol. The first KO
observation is at 0.25y and they are monthly thereafter, so ~9 of the 34
observations fall inside the window where the curves differ — and those early
observations carry the most probability mass of terminating the trade.
`ts_bsm` sees up to 3.3 vol points *more* than `flat_bsm` across exactly that
window (§5.7). The signs agree: higher near-term vol raises both early-KO and
KI probability, `ts_bsm`'s PV is lower on 6 of 8 dates, and its delta is lower
on 8 of 8.

The `decayed` near-zeros are a property of that sample, not of maturity —
those states sit at |PV| ≈ 0.1–0.45% of notional, close to deterministic,
where no vol assumption moves much.

**This is the study's signal-to-noise check, and it passes.** What the study
measures between variants (1.34% of notional, 4.03 contracts) is 5–8× what its
numerics can resolve (0.25%, 0.5 contracts). Had the ordering been reversed the
design would be unsound.

---

### 5.7 The far-end vol is an extrapolation convention, not market data

Found 2026-08-03 while explaining why `ts_bsm` tracks `flat_bsm`. This
conditions every number in §5.5 and §5.6 and every result the fleet will
produce, so it is recorded here rather than in a gate section.

**CSI 1000 lists 11 months of options; the product is 3 years.** At
2026-07-15 the artifact carries five ATM pillars and
`max_listed_T = 0.9260y`:

| T (y) | expiry | ATM vol | implied q |
|---|---|---|---|
| 0.1014 | 2026-08-21 | 0.29720 | 0.06144 |
| 0.1781 | 2026-09-18 | 0.29573 | 0.08193 |
| 0.4274 | 2026-12-18 | 0.28650 | 0.10762 |
| 0.6767 | 2027-03-19 | 0.26965 | 0.10392 |
| 0.9260 | 2027-06-18 | 0.26448 | 0.09872 |

Both curves are genuinely sloped — 3.3 vol points and ~470 bp of q across
11 months. Beyond 0.9260y both clamp to the last pillar. **69% of the
maturity axis is extrapolated at inception**, and the product's own maturity
exceeds `max_listed_T` for roughly 91% of the replay days.

That also explains why `ts_bsm` tracks `flat_bsm` so closely (§5.6):
`flat_atm_remaining` samples the ATM curve at the *remaining* maturity, which
at inception is 3.0y — already in the clamped zone — so it returns 0.26448 and
wraps it flat. The two differ only below 0.9260y.

**The declared policy misdescribes the behaviour.**
`03_build_iv_surface_history.py:104` hardcodes
`EXTRAPOLATION_POLICY = "flat_total_variance"`, stamped into all 768 artifacts.
Measured, the surfaces extrapolate **flat volatility**:

| T | measured | flat-vol predicts | flat-total-variance predicts |
|---|---|---|---|
| 1.5 | 0.264476 | **0.264476** | 0.207803 |
| 3.0 | 0.264476 | **0.264476** | 0.146939 |

The string is never branched on. Its only consumer is
`product_replay.py:242`, which copies it into
`last_surface_provenance["surface_extrapolation"]` — so it is a pure label,
and it is recorded into every run's provenance as a description of what that
run did. `product_replay.py:206` repeats the error in prose ("the vol surfaces
clamp flat (matching the artifact's `flat_total_variance` policy)"); clamping
flat in vol is the *opposite* claim about forward variance.

**The behaviour is the defensible one and should not change.** Flat total
variance would assert that no further variance accrues after 11 months — the
index frozen for the last two years of a three-year trade — and would price
the far end at 0.147 instead of 0.264. The label is what is wrong.

**Consequences for reading this study.** The gate is unaffected: both sides of
every G2 pair see the same extrapolation, so §5.5's and §5.6's comparisons
stand. What is affected is interpretation — the single most consequential vol
assumption for a 3Y snowball here was not chosen, it is the native edge
behaviour of the surface classes. Any conclusion about long-dated snowball
hedging inherits it, and a sensitivity run against an alternative convention
would be the honest way to bound that. Deliberately not attempted here.

**Not fixed in this branch.** `03_build_iv_surface_history.py` belongs to the
concurrently-landed daily-pipeline workstream; renaming the constant and
regenerating artifact metadata needs to be coordinated with it.

---

### 5.8 G2's delta criterion cannot resolve its own bound — measured 2026-08-03

**This invalidates the delta half of every §5.5 route decision.** The criterion
is sound in its economics and unusable as implemented: it polices a
0.1-contract bias bound with a reference carrying ~0.46 contracts of noise.

#### What prompted it

The owner's 2D PDE delta fix (`f97fba3`, merged `ae7e06c`) cut the measured
bias by 46% (`heston`, −0.4715 → −0.2561) and 66% (`heston_slv`, −0.3379 →
−0.1160), removed the one-sided tilt (residuals went from 7/8 negative to
mixed-sign), and tightened the BSM variants 24×. **No route changed.** A fix
that large moving no verdict is itself evidence about the instrument.

#### The attribution experiment

G2 reports `|delta_PDE − delta_MC|`, which is a *disagreement* and cannot say
which side is wrong. Perturb each estimator in a way that must not change the
answer — re-draw the MC path set (vary the seed), refine the PDE grid — and
the noisier one is the one that moves.

| variant | date | gate score | MC σ (5 seeds) | PDE medium→fine | **PDE − MC(mean)** |
|---|---|---|---|---|---|
| `heston` | 2023-05-15 | −0.932 | 0.438 | 0.036 | **−0.218** |
| `heston` | 2024-02-08 | +0.844 | 0.510 | 0.001 | **+0.166** |
| `heston_slv` | 2023-11-15 | −0.717 | 0.409 | 0.026 | **−0.083** |
| `localvol` | 2024-06-14 | −0.601 | 0.478 | 0.001 | **+0.173** |

All in futures contracts. **Every cell G2 failed is inside the 0.5-contract
bound once the reference is de-noised.**

#### The finding

The MC reference's delta noise is **σ = 0.41–0.51 contracts**, remarkably
stable across variants and dates — the intrinsic floor of a bumped RQMC delta
at 8192×16 paths for this product, not date-specific bad luck. Against
Task 6's bounds:

- vs the **0.5** per-cell bound: the reference alone fails cells at roughly
  coin-flip odds, whatever the engine does
- vs the **0.1** bias bound: **4.6×** over — unresolvable in principle
- averaged over 8 dates: σ/√8 ≈ 0.16, still **1.6×** the bias bound

In all four cells the fixed seed 20260723 drew an extreme value (highest of
five, lowest, highest, highest). That is not luck; it is what one draw from a
σ=0.46 distribution does against a 0.5 threshold.

Meanwhile the PDE moves **0.001–0.036 contracts** under refinement. On two
cells the reference is over **1000×** less stable than the engine it judges.

`localvol` is the control that closes the argument: a **1D** LV PDE, untouched
by `f97fba3`, whose failure signature was dispersion rather than bias — a
different mechanism from the Heston pair. It shows the identical pattern. The
common factor across all four cells is the reference, not any engine.

#### What is and is not established

Established: the PDE delta is *converged* and is statistically consistent with
the MC mean (|PDE − MC(mean)| ≤ 0.218, against an SE of the 5-seed mean of
≈0.20). G2's delta verdicts for `localvol`, `heston` and `heston_slv` are
artifacts of reference noise.

NOT established: that the PDE is *correct*. Convergence under refinement shows
stability, not accuracy — both grids can share a systematic error. Bounding
accuracy properly needs a reference materially better than the current one.

#### The corrective

Fix the reference, not the bounds. Task 6's thresholds are derived from the
hedge instrument — half a futures contract genuinely is where two hedges
diverge, and 0.1 contracts of persistent drift genuinely does accumulate over
~750 rebalances — so they should stand. Average the MC reference delta over N
seeds: N=16 gives σ ≈ 0.115 contracts, at last commensurate with the bias
bound. Cost is N× the reference delta only, which is a small part of the gate.

Consequential for the fleet: §5.5's MC routes carry a ~6× cost penalty
(§5.9) *and* inject engine-artifact churn into the hedge P&L the study
measures — a σ=0.46-contract delta driving daily `round_contracts`
rebalancing crosses integer boundaries constantly, and only for the MC-routed
variants. That is a confound in a study whose question is which vol model
hedges better.

**Not acted on yet.** A separate signal is unresolved: after `f97fba3` both
Heston variants' PV at `2025-04-09/full` (the Feller-degenerate cell, ratio
1901) *diverges* under refinement — medium −0.2336 (pass), fine −0.3801
(fail), where before the fix it converged (+0.3132 → +0.2071). A convergent
scheme cannot do that, so it may be a real engine defect confined to the
σ-collapse regime. Root-cause investigation in flight. Routing the Heston pair
to PDE on delta grounds while that is open would trade a measured problem for
an unmeasured one.

---

### 5.9 σ-collapse dates fail on DISCRETISATION, not calibration — root cause 2026-08-03

Root-caused by Codex (gpt-5.6-sol, max effort) on the `2025-04-09/full` cell.
Full transcript `output/codex_rootcause.log`.

**This supersedes §7A.11's attribution.** §7A.11 observed that a σ-collapse date
fails the PDE gate under *both* calibration policies (+0.579% unconstrained,
+0.277% enforced) and concluded "no calibration flag fixes them", implying the
model itself is unusable there. The observation is right and the inference is
wrong: it is not a calibration problem. It is a **numerical** one, and it is
fixable.

#### Root cause

The variance operator is `A_v U = ½σ²v·U_vv + κ(θ−v)·U_v`. At σ-collapse
(2025-04-09: σ = 0.00311, κ = 3.0, θ = 0.00306, v0 = 0.14027) diffusion is
negligible while drift is O(0.4) — a **convection-dominated** equation
discretised with a **centered** stencil (`adi_core.py:380`, `:523`).

| forward time | variance | Pe medium | Pe fine |
|---|---|---|---|
| 0 | 0.14027 | **5,872** | 4,031 |
| 0.5y | 0.03368 | 2,358 | 1,547 |
| 2.0y+ | ≈0.0034 | 62 | 35 |

Monotonicity needs `Pe ≤ 2`. Every one of the 28 medium / 42 fine interior
nodes between θ and v0 carries a **negative generator off-diagonal**: the
discrete operator is not an M-matrix and has no maximum principle.

#### Why refining time diverges

Two fully-implicit Douglas/Rannacher steps are marked after each event
(`snowball_vol_pde_solvers.py:28`). With 760 KI + 34 KO observations:

| | medium `n_t=1203` | fine `n_t=2406` |
|---|---|---|
| damped steps | 972 | 1518 |
| **damped fraction** | **80.8%** | **63.1%** |

**The medium grid only looks acceptable because 81% of its steps accidentally
suppress the bad modes.** Refining time dilutes that damping and exposes the
defect — the crutch is removed faster than resolution is added. Per-axis:
`n_v` 60→90 *improves* (−0.2336 → −0.1436), `n_x` is neutral (−0.2423), `n_t`
alone is −0.5581.

Proven three ways: fine-time with **four** event steps (restoring 80.8%
coverage) recovers −765.45; events disabled is stable but badly wrong
(−846.82 → −846.70); events disabled **plus a monotone donor-cell v stencil**
is stable *and* correct (−748.72 → −748.50).

#### The true value, and the verdict on both ladders

Four independent methods agree on ≈ **−750**:

| method | price |
|---|---|
| QE-M RQMC, 524,288 paths / 64 batches | −749.984 ± 1.104 (95% CI [−752.15, −747.82]) |
| high 1D PDE (σ→0 deterministic limit) | −750.038 |
| QUAD 8,192 / 16,384 | −750.313 / −750.266 |

| ladder | medium err | fine err |
|---|---|---|
| legacy `v_grid_power=0` | +0.289% | **+0.183%** |
| graded default 2.5 | −0.257% | **−0.404%** |
| monotone donor-cell (diagnostic) | +0.051% | **+0.024%** |

**Neither centered-drift ladder is admissible.** The pre-fix apparent
convergence was compensating error. `v_grid_power` did not introduce the
defect — it changed the error's sign and magnitude, which made a long-standing
problem visible. The monotone stencil prices the *same calibration* to
+0.024%, so these dates are perfectly priceable; the scheme was the problem.

#### Cleared, and incidental findings

- **`create_bump_context` is exonerated for PV**: base price bitwise identical
  (−764.5382710535866 both ways). It touches only the delta path.
- **`V_max` is not collapsing**: `max(5θ, 0.5, 2v0) = 0.5`. The stochastic tube
  is [0.00279, 0.14027]; at maturity variance is 0.003081 ± 0.0000706, so zero
  is **44σ** away. The power mesh targets a near-zero singularity that does not
  exist here — it spends 8/60 nodes below θ and makes the v0 cell *wider* than
  legacy (0.00989 vs 0.00698). Mismatched to this regime, but not the root
  defect.
- **Latent landmine**: `v_grid_power=2.5` with `v0_boundary="neumann"` blew up
  to −7.55e12. Not the default, but `neumann` stays selectable for cross-checks.
- **The `coarse` nulls are unrelated**: the S-axis binder reaches 1.063% local
  spacing against a 0.6% guard (`space.py:295`), and 760 KI dates collapse into
  97 time keys — 663 collisions.
- **Correction to §5.5's reading of the `decayed` case**: that row *recalibrates*
  on the decay surface (gate line 1711) — σ = 0.5785, Feller 1.00003,
  T = 1.6959y. Its stability isolates the *calibration regime*, not maturity.

#### Recommended fix (implemented 2026-08-03; production certification pending)

1. Monotone v-generator when local Pe is large: use centered drift only when
   both neighbour generator coefficients stay non-negative, else an
   M-matrix-preserving upwind or, preferably, exponentially fitted
   (Scharfetter–Gummel).
2. Use identical fitted coefficients in `_A2` **and** `_tri_V` — changing one
   alone breaks ADI consistency.
3. Regime-aware variance grid: power grading for low-Feller cases,
   CIR-quantile / path-focused nodes with θ and v0 explicitly represented for
   σ-collapse.
4. Retain `degenerate_pde`. Do **not** paper over this by scaling Rannacher
   steps or by reverting `v_grid_power` — both merely restore compensating
   damping.
5. Add a σ-collapse gate requiring monotone v coefficients plus **separate**
   `n_x` / `n_v` / `n_t` refinement, since the pooled ladder hid this.

All five implementation requirements now exist on the standalone
`codex/adi-greek-certification` worktree branch:

- `HestonSLVADICore` builds one cached V-generator coefficient set. Centered
  drift is retained only where both neighbour coefficients are non-negative;
  other rows use directionally correct donor-cell drift. `_A2` and `_tri_V`
  consume that exact set, and roundoff-negative accepted coefficients are
  projected to zero.
- `variance_grid_mode="auto"` keeps power grading for ordinary/low-Feller
  regimes and selects a path-focused grid with exact θ/v0 nodes for
  sigma-collapse/deterministic-variance regimes. Explicit legacy/power/path
  modes remain diagnostic controls.
- `v0_boundary="degenerate_pde"` remains the Snowball default. Event and
  terminal Rannacher counts are unchanged; neither damping nor a legacy-grid
  reversion is used as a repair.
- `variance_operator_diagnostics()` emits the local-Péclet and M-matrix
  evidence required by the gate. The execution-session clone now preserves
  both new constructor controls, closing the same explicit-kwargs hazard that
  previously dropped `v0_boundary`.
- Stage 11 serializes `variance_grid_mode="auto"`,
  `v_drift_scheme="adaptive_upwind"`, and `v0_boundary="degenerate_pde"` as
  explicit production controls. Stage 12 rejects a stale decision that still
  injects `v_grid_power=2.5`; that legacy override would otherwise disable the
  path-focused sigma-collapse grid after certification.
- `16_adi_greek_certification.py` holds `n_x`, `n_v`, and `n_t` fixed in turn,
  adds a separate 2%/1%/0.5%/0.25% bump ladder, and covers ordinary/full,
  decayed, near-barrier, low-Feller, sigma-collapse, and near-expiry cases. Each
  axis must contract (apart from an explicitly immaterial fraction of the
  economic bound); a non-convergent deterministic ladder fails the cell.

The correctness gate is deliberately stronger than the old G2 point estimate.
For each QE-M RQMC scramble it prices `S-h`, `S`, and `S+h` using the same Sobol
point set, forms delta and gamma *inside the batch*, and estimates uncertainty
across independent scrambles. Heston analytically integrates the independent
terminal Brownian-bridge spot factor conditional on each QE variance/residual
path. SLV uses the corresponding exact conditional expectation as a control;
the closer control freezes the leverage path at the factor-zero proxy while the
target retains fully state-dependent leverage, so the estimator remains
unbiased. Target/fine QE draws are projections of one finest Sobol set. The
production profile is 8,192 paths × 1,024 common scrambles for Heston, with the
near-KI cell alone extended to 2,048 scrambles, and 1,024 paths × 128 scrambles
for SLV. The Heston production Greek grid uses at least 300 spot nodes, 135
variance nodes, and 1,600 ADI steps/year; when the finite-bump stencil straddles
a dense KI schedule it uses at least 600 spot nodes and 16 exactly aligned ADI
steps per schedule tick. Near KI, Heston also crosses four inner points over
the eight leading residual Brownian-bridge coordinates after integrating the
terminal factor. Only the first 1,024 common scramble IDs enter the
cross-regime signed-bias gate. The stronger grid and common count were selected
on the 20260806 pilot family; schema 9 production evidence uses the untouched
20260807 scramble family.
SLV crosses four randomized terminal-factor strata and
their antithetic shifts with eight randomized midpoint Brownian-bridge strata;
variance and QE-branch streams are Brownian-bridge ordered too. Equal-cost
ordinary/full pilots selected this 8×8 allocation over terminal-only and 16×4
alternatives. The sampling profiles, including the exact Heston case map, and
the full conditional profile are serialized and rechecked by Stage 12 before a
`pde` route is accepted.

The equivalence interval is `(PDE-fine-reference) ± fine-reference Student-t
uncertainty ± paired target-to-fine substep-bias upper bound ± separate PDE
n_x/n_v/n_t envelopes`. The two stochastic components each use 97.5% coverage,
giving at least 95% simultaneous coverage by Bonferroni. It is tri-state:
wholly inside the hedge-derived bound is PASS, wholly outside is FAIL, and
overlap is INCONCLUSIVE. Delta retains the 0.5-contract cell and 0.1-contract
mean-signed-bias bounds. For the mean-bias gate, signed axis corrections and
paired substep batches are aggregated by scramble before uncertainty is taken;
absolute per-cell envelopes are not incorrectly averaged into a signed bias.
Gamma is expressed as the change in hedge contracts for a 1% spot move. Near
discontinuities these are finite-bump hedge exposures, not purported classical
derivatives. The smaller-bump ladder diagnoses that semantic choice and is not
counted as PDE discretization error.

Four deterministic reductions precede the stochastic matrix: Heston vanilla
against the semi-analytical engine; constant-variance Snowball against 1-D BSM
PDE and QUAD; unit-leverage SLV against Heston; and deterministic variance
against a time-dependent 1-D local-vol PDE. The JSON preserves raw batch
estimates and covariance, while the Markdown and decision artifact expose a
fail-closed route. A quick run is never admissive. A production-sized run may
emit `pde` only if every required anchor, cell, and signed-bias gate passes;
otherwise it emits `excluded_greek_unresolved` rather than replacing a stable
PDE estimator with a noisy daily MC delta. Anchors and individual regime cells
are atomically checkpointed; `--resume` reuses them only when the complete run
configuration and every numerical source input match their SHA-256 fingerprints.
Independent scrambles may execute concurrently, but results are reduced in
deterministic batch-id order.

The compact routing decision is self-hashed and embeds the full run
configuration plus Python/NumPy/SciPy/platform identity. Stage 12 recomputes
the live Stage-16 implementation fingerprint (including the Stage-11/12
routing seam) and rejects a stale source, runtime, or edited decision before
constructing a production engine.

Stage 12 consumes that decision for `heston`/`heston_slv`. A 2-D variant enters
the replay fleet only when Stage 11 admits its PDE PV **and** Stage 16 admits
its Greeks. A Stage-11 MC route or a Stage-16 unresolved route excludes the
variant and is written to the fleet manifest; neither condition can select a
daily MC Greek path.

Implementation smoke command (non-production):

```bash
.venv/bin/python example/mo_volmodels/16_adi_greek_certification.py --quick
```

The unqualified command is the production evidence run. Landing the machinery
does **not** itself claim that either 2-D variant has passed that run; admission
is carried only by the generated, hashed decision artifact.

#### Consequence for the study

The risk table's line on the 50 σ-collapse dates (6.6%) should be re-read:
"stage 13 must flag or exclude them" was the right mitigation for a defect
believed to be unfixable. It is fixable. Exclusion remains the correct
*interim* posture, but the durable answer is the monotone stencil, after which
those dates become ordinary.

---

## 6. Replay termination at knock-out — **DELIVERED by the library**

*Status changed 2026-07-30: this section specified a requirement; the backtest
replay consolidation implemented it. It is recorded here as satisfied, not as
work.*

The defect was real: the 2023-05-04 trade knocked out on 2025-09-04 (life 2.34 y)
yet the run priced all 726 trading days to the 2026-05-06 maturity, where 570
would have sufficed — **156 trading days of a terminated contract**, with records
to match, and (per §7.2) 61% of fleet replay-days wasted across the 27 inceptions.

`7455484 feat(backtest): pending-settlement KO termination — replay ends when
terminal cash lands` implements exactly the rule this section asked for, including
the settlement semantics: the replay ends once the terminal cash has landed in the
ledger, not on the observation date. It is exposed as
`terminate_on_lifecycle_end: bool = True` on both `AutocallableBacktestConfig`
(`replay/config.py:257`) and `ReplayBacktestConfig` (`replay/config.py:325`).

**Stage 12 inherits it automatically** — it constructs
`AutocallableBacktestConfig` without passing the flag, so the `True` default
applies. No stage-12 change is required.

Unchanged and still correct: KI does **not** terminate — a knocked-in trade runs
to maturity.

**Residual requirement.** The library does not emit a `termination_reason` field.
Stage 12's own manifest already records `lifecycle`
(`censored_at_data_end`/`knocked_in`/`knocked_out`/`ko_date`/`matured`) plus
`n_days` and `last_date`, which is sufficient to distinguish KO / maturity /
data-end termination after the fact. Deriving an explicit `termination_reason`
from those flags during aggregation is a stage-13 nicety, not a blocker.

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
2. **The greeks path differs per engine.** The 1D BSM/LocalVol Snowball PDE
   solvers take the **native** path (`snowball_pde_solver.py:1067`) and read
   delta/gamma from one solved surface. The 2D Heston/Heston-SLV Snowball PDE
   solvers explicitly delegate to `BaseEngine.calculate_greeks`, so they take a
   deterministic **central-bump** path. As of 2026-08-03 that base method resolves
   `create_bump_context` once, and all base/up/down prices reuse the base-market
   spatial layout. MC engines also take a central-bump path, but each reprice is a
   full simulation.

   *Updated 2026-07-30 for the replay consolidation.* The branch used to live in
   the replay layer (`otc/_replay.py:264–267`, which hand-rolled the bumps).
   `ProductReplay.calculate_greeks` (`replay/product_replay.py:282`) now delegates
   unconditionally to `engine.calculate_greeks`, so the native-vs-bump decision is
   entirely engine-side. The **cost accounting above is unchanged** — 2 solves/day
   for PDE-priced variants, 3 prices/day for MC-priced — but the mechanism moved,
   and the consolidation also removed a silent `delta = 0.0` fallback on engine
   failure. That fallback matters to this study specifically: a zero-delta day
   manufactured phantom unwind trades, which would have contaminated exactly the
   cost-drag and hedge-turnover figures the study reports. Greeks now fail closed.

So a PDE-priced day is `price` + one native-greeks solve = **2 solve-equivalents**;
an MC-priced day is **3 MC prices**. Measured confirmation: 29.56 s/day for
`flat_bsm` at T≈3 against a 15.06 s cold solve is a ratio of **1.96**.

Two hypotheses were tested and falsified, and are recorded so they are not
re-proposed:

- ~~**Auxiliary engines are a major cost.** They are not. Measured with daily event
  probabilities ON: 29.56 s/day; OFF: 29.60 s/day — **−0.2%, zero within noise**.
  The event-stats engine rides the cached layout. Consequently the option of
  re-routing the auxiliary engines to quadrature is void: there is nothing to
  save. **Daily event probabilities stay ON** (owner decision 2026-07-30,
  confirming the 2026-07-25 decision on measured evidence).~~
  **RETRACTED 2026-08-02 — the measurement was void.** See §7.4.
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

### 7.4 Event statistics cost +55%, not −0.2% — the earlier measurement was void

§7.1 recorded "auxiliary engines are a major cost" as a *falsified* hypothesis on
measured evidence, and an owner decision to keep daily event probabilities ON
rested on it. Both are withdrawn.

**Why the measurement was void.** `PDEEngine` is a dispatch facade that never
overrode `calculate_event_stats`, so it inherited `BaseEngine`'s `return None`
("unsupported") and silently discarded the exact statistics `SnowballPDESolver`
computes. Every ON-vs-OFF comparison therefore compared *doing nothing* against
*doing nothing*, and −0.2% is exactly the null result that produces. The
supporting claim that "the event-stats engine rides the cached layout" described
an engine that was never invoked. Fixed in `b6b97f0`; see §7A.13.

**Re-measured on an idle machine**, one inception (2023-05-04), `flat_bsm`,
`--quick`, `--max-inceptions 1`, both arms sharing an identical 44 s coupon solve
that is excluded below:

| daily event probabilities | replay wall-clock |
|---|---|
| OFF | 711.3 s |
| ON | 1102.9 s |
| **marginal cost** | **+391.6 s = +55%** |

**What this does and does not explain.** The `flat_bsm` fleet re-run reached
4/27 in 19 h 47 m across 4 workers — roughly 17 h per replay against §7.2's
measured 5.67–5.81 h. A +55% uplift takes 5.7 h to about 8.8 h, so it accounts
for well under half the overrun. The remainder is most likely CPU contention:
two full pytest suites and a large review agent ran concurrently on the same
machine. That is a hypothesis, not a measurement, and it is recorded as such.

**Consequences.**

1. §7.2's `5.67–5.81 h` PDE-priced fleet-run figure was measured on the same
   inert path and is a **lower bound**, not a current estimate. Any fleet
   scheduling built on it under-provisions by at least 55%.
2. Keeping event probabilities ON is now a real cost decision rather than a free
   one, and it should be re-taken by the owner on this evidence. Note that
   nothing in stage 13's reporting consumes them — `13_aggregate_and_report.py`
   contains no reference to KO/KI probability columns — so `--no-event-probabilities`
   is available at no loss to the study's published output. The counter-argument
   is that the frames are the study's only per-day record of barrier proximity,
   which is diagnostic value the aggregate report does not capture.
3. The re-routing-auxiliary-engines-to-quadrature option, closed in §7.1 as
   having "nothing to save", is **reopened**. There is now 55% to save.

This is a second-order lesson worth stating plainly: a silent no-op does not
merely produce absent output, it produces confident measurements *of itself*
that then enter planning documents as settled facts. §7.1 read as one of this
spec's better-supported claims precisely because someone did run the experiment.

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

### 7A.4 Decisions — all four landed 2026-07-31 (see §7A.10 for measured cost)

1. **`enforce_feller=True`** in the `mo_frozen` Heston preset (owner decision
   2026-07-30; currently `False` with only a soft `regularize_feller=0.05`
   penalty). No date may then produce a degenerate parameter set, both engines
   agree, and the PDE route reopens. Cost: a worse smile fit on the dates that
   previously violated Feller — this **must** be reported as per-date calibration
   RMSE, since it changes the model being tested.
   Cache safety verified: the heston fingerprint embeds the full preset contents
   (`quantark/volmodels/calibration.py:528`) and `heston_slv` chains it, so the key changes
   automatically — no stale hits and no `_CACHE_SCHEMA_VERSION` bump. The 552
   cached `localvol-` entries stay valid; their fingerprint excludes the preset.
2. **Plumb `v0_boundary` and default it to `degenerate_pde`** for the snowball and
   phoenix vol PDE solvers. **Mandatory, not belt-and-braces** — see §7A.6. An
   earlier draft called this redundant after (1); the Feller-boundary sweep
   disproves that. `enforce_feller=True` lands constrained fits *on* the boundary
   at `2κθ/σ² ≈ 1.0`, which is exactly where `neumann` fails (−0.540% of
   notional). The two fixes are complementary: (1) removes the deep-violation
   regime, (2) is required for the marginal regime that (1) produces.

   **The default is unconditional, and deliberately so** (owner, 2026-07-31).
   It lives on the *solver*, not on the calibration, so it does not depend on
   (1) being in force. Feller violation is the normal state of this data —
   **86% of unconstrained fits violate it** (§7A.10) — and the same is likely on
   any equity index cohort with a steep short-dated smile. A solver whose
   correctness is contingent on an upstream calibration flag is one preset
   change, one hand-supplied `HestonParams`, or one new data cohort away from
   silently mis-pricing. `degenerate_pde` costs nothing when Feller holds
   (§7A.6: +0.037% vs +0.034% at ratio 6.617) and is the difference between
   −0.540% and +0.156% when it does not. There is no regime in which `neumann`
   is the better default, so it is retained only as an explicit cross-check.
3. **G2 records `2κθ/σ²` and bound-hit flags per date**, and evaluates its verdict
   **conditioned on the Feller ratio** rather than pooling regimes. §7A.3 shows a
   uniform verdict would average a 0.03% regime with a 2.5% one.
   *Recording half landed:* `feller_ratio` joins `feller_margin`,
   `feller_satisfied` and `bound_hits` in the calibration record, which the gate
   already passes through verbatim (`11_pde_convergence_gate.py:_calibration_record`).
   The margin alone could not support the conditioning — it is a difference, so
   1e-3 is comfortable at σ=0.03 and vanishing at σ=0.6. *Verdict-conditioning
   logic is deliberately NOT landed:* it is gate scoring, not an engine fix, and
   belongs with the G2 re-run so the buckets are chosen against the regimes the
   re-run actually produces (§7A.10: 80% at ratio ≈ 1.0, 6.6% above 10).
4. **MC reference upgraded** to `martingale_correction=True` with
   `substeps_per_interval` ≥ 4. The measured reference bias is only 0.078% of
   notional, but it is free to remove and the gate's tolerance is 0.25%.
   *Landed* as `MC_MARTINGALE = True` and `MC_FULL[substeps_per_interval] = 4`;
   `mc_reference.scheme` in `gate_decision.json` now reads `QUADEXP_M`. `MC_QUICK`
   stays at 1 substep — it is a plumbing smoke already marked non-production-valid.
   **This changes gate cost:** 4 substeps multiply the reference MC time grid
   fourfold, and §7A.6 measured MC at 40–50 s per case against PDE at ~9 s. The
   G2 re-run is the expensive step in §10, not the fleet.

### 7A.6 Feller-boundary sweep

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
2. The PDE column above used `n_t=1600`, which §7A.7 shows is **unnecessary**.

### 7A.7 Cheapest passing configuration, per replay day

Speed claims are only meaningful between the *cheapest configuration of each
engine that meets tolerance*, in the units the fleet pays (§7.1: MC = 3 prices/day
via the bump path, PDE = 2 solves/day via native greeks). Measured at T=1, Feller
ratio 1.0, `degenerate_pde`, against the QE-M/8/32 reference:

| engine | PV gap | gate | 1 call | **per replay day** | delta (cash / 1% spot) |
|---|---|---|---|---|---|
| MC `QE` sub=1 bat=16 | +0.071% | PASS | 2.0 s | 6.1 s | 485,747 |
| MC `QE-M` sub=1 bat=16 | +0.071% | PASS | 2.4 s | 7.1 s | 485,747 |
| MC `QE-M` sub=2 bat=16 | −0.001% | PASS | 4.9 s | 14.8 s | 478,306 |
| MC `QE-M` sub=4 bat=16 | +0.016% | PASS | 10.3 s | 30.9 s | 480,146 |
| **PDE degen 200×60×400** | **+0.159%** | **PASS** | **2.1 s** | **4.2 s** | 482,792 |
| PDE degen 200×60×800 | +0.139% | PASS | 3.9 s | 7.9 s | 482,850 |
| PDE degen 200×60×1600 | +0.156% | PASS | 8.6 s | 17.3 s | 481,745 |

**The gate's existing 200×60×`ceil(400·T)` grid passes** once `degenerate_pde` is
in place; refining to `n_t=1600` does not improve it (+0.156% vs +0.159%), so the
residual is not time resolution and the extra cost buys nothing. Cheapest passing
PDE is **4.2 s/day** against MC's **6.1 s/day** — PDE ~1.45× cheaper.

**Two superseded claims, recorded so they are not re-cited.** This spec has twice
stated a speed conclusion from a mismatched pair: first that MC was cheaper
(PDE at a passing resolution vs MC at the gate's *failing* configuration), then
that PDE was 4.6–5.5× cheaper (over-resolved PDE vs MC at an *arbiter-grade*
configuration no production run would use). Both are withdrawn. The margin is
**1.45×**, and it is the weaker half of the case.

### 7A.8 Delta stability — the deciding factor

The study trades on delta, not PV. MC delta comes from differencing two noisy
prices; the PDE reads delta and gamma off the same solve
(`snowball_pde_solver.py:1124`). Measured at the cheapest passing MC config,
central 1% bump, three seeds:

```
seed 20260723 : 485,747      seed 11111 : 488,989      seed 99999 : 492,256
spread 6,509    stdev 3,254   (cash per 1% spot move)
```

§5.3 sets the delta admission threshold at half an IM contract = **6,734** cash per
1% move, derived from the hedge's own granularity. **MC's seed-to-seed delta noise
is 0.97 contracts wide — it consumes essentially the whole tolerance before any
engine-vs-engine disagreement is counted.** The PDE's delta is deterministic.

This does not average away over a ~700-day daily rebalance: every spurious contract
crossing is a real trade at 0.5 bp commission + 1 bp spread, so MC would
manufacture hedge turnover that is a numerical artifact — and cost drag and hedge
slippage are exactly what the study measures.

**Two caveats bounding all of §7A.6–7A.8.** Measured exponents are
`pde2d ≈ T^1.78` against `mc ≈ T^1.04–1.20`, so the cost advantage narrows with
maturity; the T=3 pair is **extrapolated, not measured**. And every
fixed-configuration number is `heston` — `heston_slv` adds a leverage surface on
the same ADI core and is **not** assumed to inherit the result. Both are G2's job.

**Prediction on the record:** with `enforce_feller=True` and
`v0_boundary=degenerate_pde`, G2 admits the PDE route for both 2D variants at the
existing 200×60×`ceil(400·T)` grid, on delta stability more than on speed.
Recorded as a prediction so the gate can falsify it.

### 7A.9 Method note

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

### 7A.10 What landing §7A.4 actually cost — measured on all 762 surfaces

All four decisions are implemented (commit below). §7A.4(1) required the fit cost
be reported; the sweep runs every surface in
`example/mo_volmodels/data/history/iv_surface/` through both policies, which also
validates the iteration budget the constrained solver needs.

> **Basis: the 762 surfaces admitted as of 2026-07-24.** A daily calibration
> pipeline has since extended the history — 766 admitted at 2026-07-31, growing
> each weekday. Read every denominator below as "of 762 as-of 2026-07-24", not
> "of the cohort". §7A.12 has the drift, the pin, and why the four added dates
> change no ratio here.

| quantity | result |
|---|---|
| surfaces converging, both policies | **762 / 762** — no date fails closed |
| soft-policy fits violating Feller | **658 / 762 (86%)** |
| hard fits landing in `2κθ/σ² ∈ [0.999, 1.001]` | **611 / 762 (80%)** |
| fit-RMSE degradation **vs the SABR-smoothed target** | median **8.4 bp**, p90 **29.3 bp**, p99 **73.8 bp**, max **217 bp** |
| hard-path `nfev` | median 94, p99 196, max **344** |
| one-off cohort calibration cost | 37 min single-core, then cached |

**What that RMSE is measured against.** Heston is *not* calibrated to raw CFFEX
quotes. `_heston_nodes` consumes `artifact.per_expiry[*].points`, which are
already SABR-smoothed — uniformly `method="sabr_calendar_projected"`, `beta=1.0`,
per-expiry `alpha/rho/nu`, with a calendar-slope projection. Two error layers
stack, and only the second is what the table above reports:

```
raw CFFEX quotes ──SABR──► smoothed target ──Heston──► model
                  median 41.4 bp            median 8.4 bp degradation
                  p90 79.4, max 236.8       p90 29.3, max 217
```

Consequences worth stating in the report rather than leaving implicit:

- The Feller constraint's *median* fit cost is roughly **one fifth** of the
  smoothing residual that every variant already carries. On typical dates it is
  a minor perturbation of an already-approximated target. In the tail the two
  are comparable (217 bp vs 237 bp), which is where it stops being minor.
- This is nonetheless the correct target for the study: G1 admits surfaces
  *after* SABR smoothing, so the smoothed surface **is** this study's market, and
  all six variants price against the same one. The figures are fit-to-target, not
  fit-to-market, and must be labelled that way.
- Zero of the 762 surfaces required calendar-arbitrage adjustment
  (`calendar_adjusted_nodes == 0` throughout), so the smoothing is a per-slice
  SABR fit in practice, not a reshaping of the term structure.
- No SVI anywhere in this pipeline. The SVI layer in the codebase belongs to the
  DCN work.

**Weak evidence on the mechanism.** SABR's `nu` is the vol-of-vol analogue, and
Heston carries a single global `σ` against a term structure of it, so high
short-dated `nu` is the shape that should push `σ` into Feller violation. The
data agrees only directionally: `corr(log nu_max, log unconstrained ratio) =
−0.311` (n=762), median `nu_max` 2.17 on violated dates against 1.64 on
satisfied ones. Right sign, ~10% of the variance. Recorded as a hypothesis with
its strength stated, not a finding.

Three things follow, two of them new.

1. **§7A.4(2) is confirmed as mandatory, on production data.** The spec argued
   from a synthetic sweep that `enforce_feller=True` lands fits *on* the boundary
   where `neumann` fails at −0.540% of notional. It does: **80% of the cohort now
   sits within ±0.1% of ratio 1.0.** Had only fix (1) shipped, the 2D PDE route
   would have been wrong on four dates in five.

2. **The budget is safe, but not for the reason the number suggests.**
   `heston_max_nfev=200` maps to SLSQP's `maxiter`, i.e. 200 *major iterations* at
   ~6 function evaluations each — an effective budget ~6× the nominal figure. The
   worst observed date used 344 evaluations (~57 major iterations, 29% of budget).
   The fleet will not die on an iteration limit. The same mapping is why
   `test_otc_vol_calibrators.py`'s fast fixture had to move 15 → 40: it was tuned
   to the unconstrained `least_squares` branch, where `max_nfev` really does count
   evaluations.

3. **NEW — vol-of-vol collapse on 50 dates (6.6%).** Not anticipated by §7A.4.
   On these the optimizer satisfies `2κθ ≥ σ²` not by raising `κθ` but by driving
   `σ` to its lower bound (`σ < 0.01`, ratio > 10, up to 1.7e5). Heston with
   `σ ≈ 0` is a *deterministic-variance* model: no vol-of-vol, no smile dynamics,
   and `heston_slv`'s leverage surface then carries essentially the whole smile.
   On those dates the `heston` variant is not testing what the study says it
   tests. This is a reporting obligation, not a bug — `feller_ratio` is in the
   calibration record precisely so stage 13 can identify and exclude them. The
   worst RMSE degradations cluster here (2023-12-08 at 217 bp, 2023-07-13 at
   212 bp): a model that cannot flex its vol-of-vol cannot fit the smile.

The honest summary: the constraint buys a well-posed PDE on 93% of dates and
substitutes a degenerate model on the remaining 7%. Both must appear in the
report; neither is visible without the per-date record.

**A defect the flip exposed.** `enforce_feller=True` did not work at all before
this change. SLSQP satisfies constraints only to about its own accuracy tolerance,
but the feasibility buffer was a constant `1e-8` — adequate at `ftol=1e-8`
(observed slack ~1e-9) and too small at the preset's `ftol=1e-6` (observed slack
~1.6e-7). Every constrained fit tripped the strict post-check and failed closed.
The buffer now scales as `max(1e-8, 10·ftol)`; the strict check is unchanged, so
it still fails closed if that is ever insufficient. Fix and regression test:
`quantark/volmodels/heston/calibration.py`,
`test_heston_feller_calibration.py::test_hard_feller_margin_survives_a_loose_optimizer_tolerance`.

**A second defect the flip exposed — and a standing hazard for this study.**
`Heston2DAutocallableSessionAdapter._clone_engine`
(`pde_execution_adapters.py:444`) rebuilds the 2D autocallable engines from an
explicit kwargs list. `v0_boundary` was not in it, so a session or
prepared-adapter run reverted to the default while the direct call honoured the
constructor — **silently**, with the two paths differing by 0.66% of notional in
the Feller-violating regime. Caught by `test_matrix_parity`'s direct-vs-session
equality only because the golden fixtures pin `neumann`; with both paths on the
default it would have stayed invisible.

The hazard is structural, not specific to this argument: *every* constructor
argument on these four solvers must be added to that list by hand, and omission
is silent. Anything the G2 re-run or the fleet configures on a 2D autocallable
engine — and stage 12 routes through the execution layer — must be checked
against that list. `test_vol_pde_v0_boundary.py::test_session_clone_preserves_v0_boundary`
guards this one; the general case wants the clone derived from the signature
rather than transcribed, which is recorded here as a follow-up, not done.

### 7A.11 Is `degenerate_pde` enough on its own? — measured, and no

The last assumption in §7A's chain was that `enforce_feller=True` is *needed*:
the +0.33/+0.55% residual that justified it came from one synthetic control, so
"drop enforcement and rely on the boundary fix" was never actually tested. It is
now, on the production configuration — the gate's own product, environment,
engines and tolerance — with both policies run on the **same** dates, so the
comparison is paired. Full 3Y case, medium grid 200×60×`ceil(400·T)`, QE-M
reference at 4 substeps, tolerance the 0.25% floor throughout (`mc_se` 0.033 …
0.092% of notional).

| date | unconstrained ratio | diff | | enforced ratio | diff | |
|---|---|---|---|---|---|---|
| 2024-01-12 | 0.197 | **+0.579%** | FAIL | 7.194 | **+0.277%** | **FAIL** |
| 2023-11-17 | 0.392 | **+0.486%** | FAIL | 1.0001 | +0.227% | PASS |
| 2023-05-04 | 0.477 | +0.205% | PASS | 1.0001 | +0.115% | PASS |
| 2026-07-17 | 0.493 | **+0.317%** | FAIL | 1.0000 | +0.163% | PASS |
| 2023-06-29 | 0.504 | +0.226% | PASS | 1.0001 | +0.105% | PASS |
| 2023-08-22 | 0.794 | +0.105% | PASS | 1.0001 | +0.078% | PASS |
| 2024-06-26 | 0.805 | +0.179% | PASS | 1.0001 | +0.163% | PASS |
| 2025-09-02 | 0.930 | +0.125% | PASS | 1.0000 | +0.108% | PASS |
| 2025-11-03 | 0.972 | +0.069% | PASS | 1.0000 | +0.060% | PASS |
| 2024-08-19 | 0.991 | +0.128% | PASS | 1.0000 | +0.113% | PASS |
| 2026-04-23 | 1.144 | +0.022% | PASS | 1.0000 | +0.000% | PASS |
| 2025-09-29 | 1.262 | +0.040% | PASS | 1.1062 | +0.065% | PASS |
| 2026-05-25 | 1.557 | +0.112% | PASS | 1.7729 | +0.122% | PASS |

**Decision: `enforce_feller=True` stays.** Four findings support it.

1. **The synthetic control was right.** It predicted +0.33/+0.55% at deep
   violation; real unconstrained fits give **+0.486% at ratio 0.392** and
   **+0.579% at 0.197**. §7A.3's reasoning survives contact with production data.
2. **The error is monotone in violation depth** — +0.02% at ratio 1.14 rising
   smoothly to +0.58% at 0.197 — so the ratio is a genuine predictor of PDE
   error, which is what makes the §7A.4(3) conditioning meaningful.
3. **Enforcement is load-bearing, and only for the tail.** Unconstrained fails
   3/13, all at ratio ≤ 0.50; enforced passes 12/13, and is strictly closer to MC
   on 11/13. But from ratio ~0.5 upward the boundary fix alone already passes.
   **16.4% of the cohort sits below 0.50** (7.2% below 0.35) — a minority, but far
   too large to leave failing.
4. **This does not rescue the σ-collapse dates.** 2024-01-12 fails *both* ways
   (+0.579% → +0.277%). Enforcement halves its error and still misses the
   tolerance. Those 6.6% need their own treatment (§7A.10(3)); no calibration
   flag fixes them.

**Two cautions for G2, both new.**

- **Every one of the 26 cells has the PDE above MC — sign fraction 1.0.** Run
  through the gate's own `detect_systematic_bias`, the unconstrained set flags
  biased (median 0.128% ≥ the 0.125% threshold) and the enforced set does *not*
  (median 0.113%) — by 0.012 points. A unanimous sign with a median just under
  the bar is not "unbiased"; it is "biased but currently small". G2 should read
  that verdict as marginal, not clean, and the bias is one-directional so it
  accumulates rather than cancels over a hedging run.
- **These are medium-grid verdicts.** The gate escalates a medium failure to the
  fine ladder, so "FAIL" here means "fails at 200×60", not "route rejected".
  The fine level may clear 2026-07-17 and 2023-11-17; it is unlikely to clear a
  degenerate-σ date, where the problem is the model rather than the grid.

Reproduction: `unconstrained_pde_vs_mc.py` (scratchpad), which imports
`11_pde_convergence_gate.py` and calls its helpers rather than re-implementing
them — the same discipline that the per-day cost model needed after two failed
attempts to model the call sequence instead of replicating it.

---

### 7A.12 The cohort is no longer frozen — a daily pipeline now extends it

Everything in §7A.10 and §7A.11 was measured against a surface history that a
concurrent workstream has since put under a **live scheduler**. This section
records what landed, what it moves, and the one decision it forces on the gates.

#### What landed

Three additions to `VolModelCalibrationConfig`
(`quantark/backtest/replay/config.py`) — the same config the replay engine and
stage 12 construct:

| field | default | effect |
|---|---|---|
| `heston_temporal_reference` | `None` | prior `(v0,κ,θ,σ,ρ)`; also becomes the solver's initial guess |
| `heston_temporal_regularization` | `0.0` | weight λ on a structural penalty toward that prior |
| `slv_heston_override` | `None` | explicit Heston vector for SLV leverage calibration, bypassing the calibrated fit |

Consumed in `quantark/volmodels/calibration.py`; the penalty itself is
`0.5·λ·Σⱼ((θⱼ − priorⱼ)/bound_spanⱼ)²` over `{κ,θ,σ,ρ}` only — `v0` is never
penalized. Driven by two new stages, `14_daily_calibration_pipeline.py` and
`15_calibration_stability_report.py`, documented in `DAILY_PIPELINE.md`.

**All of this is uncommitted working-tree state on
`fix/snowball-rebaseline-7a4-engine-fixes`.** The 92 tests covering it pass
alongside the §7A.4 work, but the gates below will execute on top of code that
has not been reviewed or committed. Land it first, or accept that the gate
evidence is keyed to an unversioned tree.

#### The gates still cover production — verified, not assumed

The installed launchd job `com.quantark.mo-daily-calibration` runs at 18:30 and
20:30 Asia/Shanghai, Mon–Fri, and its `ProgramArguments` carry **no
`--temporal-smoothing`**. Production therefore runs independent daily
calibration at λ=0 — the identical policy these gates certify. Two consequences:

- No re-gating is needed today.
- Enabling `--temporal-smoothing` (on the scheduler or by hand) puts production
  on a calibration policy **no gate in this spec has evaluated**. That switch is
  a re-gate trigger, not a tuning knob.

Default-valued, the new fields are inert: the temporal keys enter the Heston
cache fingerprint only when a reference is set, and the SLV fingerprint keeps
its `"heston"` component unless an override is set. The existing calibration
cache is not invalidated.

#### Measured drift

| quantity | §7A.10/§7A.11 basis | now (`data_end` 2026-07-31) | next scheduler run (2026-08-03) |
|---|---|---|---|
| admitted surfaces | 762 | **766** | 767+ |
| snowball inceptions | 27 | **27** | **28** |

The inception count is the one that bites. `schedule_inceptions` admits a
monthly start only when `inception + MIN_OBSERVABLE_MONTHS(12) ≤ data_end`, and
`data_end` is the last row of the spot CSV the daily job refreshes. The
2025-08-01 inception needs `data_end ≥ 2026-08-01`; today's 2026-07-31 misses it
by **one day**. Monday's run clears it and the fleet becomes 28.

Every "27" in this spec and in the gate plan — §8's outcome concentration, the
KO-date collapse, the G2 cell count — is therefore correct only against a pinned
window. Measured, not projected: `schedule_inceptions` returns 27 at `data_end`
2026-07-24 and 2026-07-31, and 28 at 2026-08-03.

#### Decision: pin the cohort, do not chase it

The gates run against a frozen `COHORT_ASOF = "20260731"`:

- G1 scans the manifest records with `date ≤ COHORT_ASOF`, not the directory.
- G2 and stage 12 pass `data_end = COHORT_ASOF` explicitly rather than reading
  the last spot row, so a mid-run scheduler tick cannot change the fleet.
- Re-running any gate later reproduces the same cell set.

762 → 766 does not invalidate §7A.10's ratios: the four additions are
2026-07-27/29/30/31, all beyond every inception's KO, so they enter no replay.
They do change the denominator, and §7A.10's table should be read as
"762 surfaces, as-of 2026-07-24" rather than "the cohort".

Widening the study to 28 inceptions is a legitimate future choice. It is not
this re-baseline, because it would re-open G4 (a 28th coupon solve) and shift
§8's concentration statistics.

#### 720 calibrations are already paid for

The daily pipeline's cache at `output/mo_daily_calibration/calibration_cache/`
holds 240 dates (2025-07-31 → 2026-07-31) × `{localvol, heston, heston_slv}`.
Verified empirically rather than by inspection: the stored `config_fingerprint`
on all 720 entries is byte-identical to what stage 12's full-quality
`VolModelCalibrationConfig(slv_n_steps=40, slv_n_x=161, slv_n_z=81)` computes —
`240/240` match on every variant. Since the cache key is
`sha256(surface_sha | variant | fingerprint)`, seeding the fleet's cache from
this directory is a pure hit, and it covers the final year of the replay window
where the SLV leverage solves are most of the cost.

#### One thing that is safe, and is not obvious

`enforce_feller=True` parks 80% of fits at `2κθ/σ² ≈ 1.0` (§7A.10), so it is
worth asking whether the EWMA in the temporal scheme can average two feasible
vectors into an infeasible one. It cannot. Written as `2κθ − σ² ≥ 0` the
constraint looks indefinite — that form's Hessian in `(κ,θ)` is `[[0,2],[2,0]]`.
But the equivalent `√(2κθ) − σ ≥ 0` is a **concave** function on `κ,θ,σ > 0`
(the geometric mean is concave), so the Feller region is a convex set and any
convex combination of feasible vectors stays feasible — strictly interior unless
the two are proportional. A smoothed vector therefore lands at ratio ≥ 1.0:
still inside the regime where §7A.11 measured `neumann` failing at −0.540% of
notional. The temporal scheme does not weaken the `degenerate_pde` requirement;
it lands squarely in the regime that motivated it.

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

*Revised 2026-07-31. Three prior steps are complete: "secure the framework" (it is
committed and consolidated), "implement §6 termination" (the library delivers it),
and the §7A.4 engine and calibration fixes (landed; §7A.10).*

~~1. Apply the §7A.4 engine and calibration fixes.~~ **DONE 2026-07-31.**
`enforce_feller=True`; `v0_boundary` plumbed and defaulted to `degenerate_pde` on
the snowball and phoenix 2D solvers; MC reference on QE-M at 4 substeps;
`feller_ratio` in the calibration record. Cohort evidence and two unanticipated
consequences in §7A.10. A latent defect that made `enforce_feller=True` unusable
at the preset's `ftol` was found and fixed in the same pass.

1. Re-execute Gate G4 (coupon solve) and Gate G1 (surface admission).
2. Re-run and re-scope Gate G2 per §5. Two things must be built here rather than
   assumed: the **verdict conditioning** deferred from §7A.4(3) — with buckets
   chosen against the measured regime split (80% at ratio ≈ 1.0, 6.6% above 10),
   not invented in advance — and the **within-maturity** bias evaluation of §5.2.
   Emit a fresh `gate_decision.json` and evidence hash. The 2D route is genuinely
   open: §7A.7 shows the *existing* 200×60×`ceil(400·T)` grid passing at +0.159%.
   Budget for this step went up, not down — the QE-M reference at 4 substeps
   quadruples the reference MC time grid (§7A.4(4)).
3. Add the `flat_bsm_quad` variant and the §9 pre-flight sweep. (§6 termination
   needs no work — stage 12 inherits `terminate_on_lifecycle_end=True`.)
4. Gate G3 on one inception.
5. **Task 6.1 timing run** — measure the per-day cost curve across remaining
   maturity for every route actually configured, validated against the §7.2
   anchors. The fleet total is set here (§7.3), not before.
6. Run the 1D block (4 variants × 27 inceptions). Review checkpoint.
7. Run the 2D block (2 variants × 27 inceptions) per the §5 routing.
8. Aggregate and report, with §8 caveats — including the §7A.10(3) degenerate-σ
   dates, which must be identified and excluded or flagged, not averaged in.

**Warm-cache note for step 1.** The 552 `localvol-` cache entries stay valid
(their fingerprint excludes the preset). Every `heston-` and `heston_slv-` entry
is now a miss, as designed. Recalibrating the full 762-surface cohort costs ~37
minutes single-core, once (§7A.10).

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
  Two consolidation changes make this re-run more than a formality: the replay now
  terminates at KO-plus-settlement (§6), so the cash and cost identities must
  reconcile at a **truncated** window with terminal cash landing on the final day,
  and `f1506ff` altered P&L to be receivable-inclusive with an honest `data_end` —
  both squarely inside what `sanity_check_run` asserts.
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
| `enforce_feller=True` degrades the smile fit on previously-violating dates | **Quantified (§7A.10):** median 8.4 bp, p90 29.3 bp, max 217 bp of IV across all 762 surfaces. Report per-date. This changes the model being tested and must be stated as such, not buried — a constrained Heston is a different model from a free one |
| ~~The `degenerate_pde` boundary is belt-and-braces once `enforce_feller` lands~~ **This was wrong.** | Retracted on measurement. §7A.10: **80% of constrained fits land in `2κθ/σ² ∈ [0.999, 1.001]`** — precisely the regime where `neumann` mis-prices by −0.540% of notional. The two fixes are complementary, and shipping (1) without (2) would have been worse than shipping neither, because it *concentrates* the cohort on the failing point |
| **NEW — `enforce_feller` satisfies the constraint by collapsing vol-of-vol on 50 dates (6.6%)** | σ driven to its lower bound (`σ < 0.01`, ratio up to 1.7e5), i.e. deterministic-variance Heston with no smile dynamics; these carry the worst fit degradations (212–217 bp). **§7A.11 shows such a date fails the PDE gate under *both* calibration policies** (+0.579% unconstrained, +0.277% enforced), so no calibration flag fixes them. Detect via `feller_ratio`; stage 13 must flag or exclude them, never average them into a `heston` result |
| **NEW — the 2D PDE sits above the MC reference on every cell measured (sign fraction 1.0, n=26)** | §7A.11. The enforced set escapes the gate's bias detector by 0.012 points of median, not by a clean margin. One-directional error accumulates over ~700 rebalances rather than cancelling, so §5.3's delta bias bound (0.1 IM contract) is the binding check, not the PV bias flag. G2 must read a narrow `biased: false` as marginal |
| MO surface ends ~1 y against a 3 y trade | Explicit flat-total-variance extrapolation, stated in the report. *Carried forward.* |
| `backtest.otc` shim is removed in 0.5.0, breaking the study mid-flight | Stages 11/12/13 already import canonical `quantark.backtest.replay` and `quantark.param.vol.surface_history` (§1.2) — verified, no shim dependency |
| Stage 13's hand-copied column lists drift from what the replay writer emits | Derive `REQUIRED_CATEGORIES` from `replay/schema.py` (§1.2) rather than maintaining a parallel list |
| Fleet killed again mid-run | Stop via process group, never `pkill -f`; per-run output isolation means completed runs survive |
| **NEW — a 2D-autocallable engine setting is silently dropped between the direct and session paths** | `_clone_engine` transcribes constructor arguments by hand (§7A.10). `v0_boundary` was already missing and priced 0.66% of notional apart. Before the fleet, diff the kwargs list against the four solvers' signatures; stage 12 routes through the execution layer, so a dropped setting would mis-price every day of every run without erroring |
| **NEW — the surface cohort and the inception fleet grow on a schedule** | §7A.12. A live launchd job extends the history every weekday; admitted surfaces went 762 → 766, and `data_end` crossing 2026-08-01 admits a **28th inception**. Unpinned, two runs of the same gate compare different cell sets and §8's concentration statistics silently shift. Mitigated by freezing `COHORT_ASOF = "20260731"` and passing `data_end` explicitly instead of reading the last spot row |
| **NEW — a calibration policy exists that no gate has evaluated** | §7A.12. `heston_temporal_regularization` > 0 changes the fitted Heston vector, and `slv_heston_override` replaces the Heston that SLV leverage calibrates against. Both default off and the installed scheduler does not enable them, so today's gates do cover production. Enabling `--temporal-smoothing` is a **re-gate trigger** — record it in the run manifest so a future reader cannot mistake the λ=0 evidence for coverage of λ>0 |
| **NEW — the gates will execute on uncommitted third-party working-tree state** | §7A.12. The daily-pipeline workstream (config fields, calibrator plumbing, stages 14/15) is unversioned on this branch. Its 92 tests pass alongside the §7A.4 work, but gate evidence keyed to an unversioned tree is not reproducible. Land it before Phase C, or stamp the tree hash into `gate_decision.json` and say so |
