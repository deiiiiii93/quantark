# QUAD event-stats follow-ups: KI-walk fusion + forward-density distribution

**Date:** 2026-08-24. **Status:** approved design, pre-implementation.
**Predecessors:** `docs/autocall-engine-perf/FINDINGS-2026-08-24.md`,
`docs/autocall-engine-perf/SOLUTIONS-2026-08-24.md`, commit `76471b8`
(A1 token memo, B1 zero-row skip, B2 bridge-kernel cache, F1 streams
contract).

## 1. Motivation and target

After `76471b8`, the 2026-06-30 full-book QUAD batch (97 rows, 7 greek-bump
cells each, 8 workers) runs in 569 s vs PDE 111 s and MC 230 s. The residual
cost is structural: one `price_with_events` cell still walks the time loop
three times (stacked KO-indicator recursion with one row per KO observation,
a separate KI-probability recursion, and the internal `price()` pass), and
the stacked rows are the dominant term (~50 diffused surfaces on a book row).

**Decisions (user, 2026-08-24):**

| Question | Decision |
|---|---|
| Acceptance target | Full power: QUAD book ≈ PDE (~100–150 s) |
| npv source under forward mode | `npv = price()` (backward) — prices bit-identical, only the distribution changes |
| Rollout | Opt-in flag `QuadParams.event_stats_mode`, default stays `stacked`; default flip is a later, separate decision |
| Validation bar | Full battery (§6) |
| Approach | A: Track 1 bitwise fusion for the default path + Track 2 flagged forward-density mode |

## 2. Goals / non-goals

**Goals**

- G1. Fuse the KI-probability walk into the stacked recursion (Track 1) —
  bit-identical output, default path.
- G2. A `forward_density` event-stats mode (Track 2) producing the same
  `AutocallableEventStats` / `PhoenixEventStats` schema from a forward
  transition-density march: 2–3 surfaces total instead of ~2×(n_ko+extras)+4.
- G3. `npv` bit-identical across modes (always the backward `price()`).
- G4. Streams contract (§11.1) honored in both modes.
- G5. Validation battery banked as evidence for the later default flip.

**Non-goals**

- Flipping the default mode (separate decision on banked evidence).
- KO-reset forward mode (`KOResetSnowballQuadEngine` has its own stacked
  stats implementation; the flag is ignored there — a permission, not an
  obligation, same language as streams).
- Any MC/PDE engine change; any adapter (`otc-price-adapter`) change.
- Changing `price()` numerics in any way.

## 3. Track 1 — fuse the KI-probability walk (bitwise)

**Where:** `SnowballQuadEngine._compute_event_stats`
(`quantark/asset/equity/engine/quad/snowball_quad_engine.py`).

**Change:** when the KI walk runs today (`product.has_ki_barrier and want_ki
and not knocked_in_at_valuation`), append four rows to the stacked arrays
instead of running the second loop:

| Row | Today | Fused position |
|---|---|---|
| `v_in_ki`  | 1-D, plain FFT | `v_in[n_base]` |
| `v_in_ever`| 1-D, plain FFT | `v_in[n_base+1]` |
| `v_out_ki` | 1-D, bridged against `v_in_ki` | `v_out[n_base]` |
| `v_out_ever`|1-D, bridged against `v_in_ever` | `v_out[n_base+1]` |

(`n_base = n_ko + extra coupon rows`.) The stacks already share the same
steps, `omega_array`, prefactor, and bridge kernels, so diffusion is one
batched call. Event applications keep each row's **exact current
treatment** via per-row exceptions (the PDE solver already structures its
fused columns this way):

- KO at an observation: smoothed `ko_w` scaling applies to rows
  `[0:n_base)` only; the ki rows keep their **hard** `ko_mask` zeroing
  (`v_out_ki[ko_mask]=0`, `v_in_ki[ko_mask]=0` unless
  `disable_ko_after_ki`); the ever rows are **exempt** (save/restore around
  any whole-stack operation, mirroring the PDE `ever0/ever1` pattern). Same
  row-slicing care in the `_use_cell_average_events` projection branch.
- Discrete KI: value/indicator rows keep `_blend_ki_transition`; the ki row
  keeps the hard `ki_mask & ~ko_mask` copy; the ever row keeps the raw
  `ki_mask` copy.
- Readout: `pv_ki_no_ko` / `pv_ki_ever` interpolate the fused rows at x=0,
  exactly as today.

**Gate:** bitwise. Batched pocketfft transforms rows independently and the
bridge correction is elementwise per row, so per-row bit-identity is
expected; it is *verified*, not assumed: pre-change golden capture (every
stats field, price, pwe npv, snowball + phoenix), the
`docs/autocall-engine-perf/demos/` harness pattern, full suite, and a book
spot-check. If bit-identity does not hold on some platform path, Track 1 is
re-evaluated (it is an optimization of a mode that forward-density
supersedes; correctness bar stays absolute).

**Expected effect:** removes one of the three walks in the default mode
(~25% off the stats pass measured on the demo product); keeps benefiting
KO-reset never covered by Track 2, and the default mode until the flip.

## 4. Track 2 — forward-density event stats

### 4.1 Mode selection

- `QuadParams.event_stats_mode: str = "stacked"`, validated in
  `__post_init__` against `{"stacked", "forward_density"}` (pattern:
  existing `cache_strategy` validation).
- `SnowballQuadEngine._compute_event_stats` dispatches on the flag after
  the shared preamble (validation, records resolution, streams flags).
  `PhoenixQuadEngine` inherits; `KOResetSnowballQuadEngine` documents the
  flag as ignored.
- `price_with_events` is unchanged (76471b8 already forwards `streams`;
  `npv = stats.pv = self.price(...)` in both modes).

### 4.2 Forward march (math)

State: the existing log-space grid (same `QuadratureMath`, same
`_event_stats_alignment_log` alignment so barriers sit on nodes; same
`_resolve_grid_points`). March the transition density forward over the same
merged `times` grid the stacked recursion uses.

- **Step operator:** `p_{k+1}(y) = Σ_x w(x) · K(y−x) · p_k(x)` with `K` the
  Gaussian of mean `μ·dt = (r−q−σ²/2)dt` and variance `σ²dt`. Same FFT
  convolution machinery; two adjustments versus the backward value step:
  the kernel orientation flips (backward is a correlation in `z = y−x`;
  forward is a convolution — implemented as a sign flip on the `α` term of
  `omega_array`), and **no discounting inside the kernel**
  (`β_fwd = α²`; the backward `β` carries `+2r/σ²`). Discount factors are
  applied only at readout. Simpson weights apply to the source density
  exactly as they apply to the value surface today.
- **No `_tail_correction`:** the backward tail term extrapolates *value*
  beyond the grid edge; a density is ≈0 there by construction
  (`num_std_devs = 10`). The forward pass instead tracks `Σ w·p` as a mass
  diagnostic; if edge leakage proves material in the battery, edge mass is
  accumulated into an explicit "beyond-grid" bucket (never silently
  renormalized — no fabricated mass).
- **Continuous-KI bridge:** with `C(y) = Σ_x w(x)·K(y−x)·p_hit(x,y)·p_out(x)`
  (the same banded `p_hit` kernel; the formula `exp(−2·d0·d1/denom)` is
  symmetric in source/target, so the landed `_bridge_kernels` cache serves
  the forward direction via offset reversal of the drift factor):
  - `p_out' = T[p_out] − C` (mass that did not touch the barrier),
  - `p_in'  = T[p_in] + C` (previously-in mass plus newly touched mass),
  - one banded correction per step serves both surfaces.
- **Surfaces:** `p_out`, `p_in` always; one `p_notouch` (bridge survival,
  **no KO absorption**) only when a KI stream is requested —
  `ki_ever = 1 − Σ w·p_notouch(T)`. Discrete-KI products need no bridge at
  all (the merged times grid is just the observation dates).

### 4.3 Initial condition

Analytic first step — no discrete delta on the grid (x=0 is generally not a
node after barrier alignment): after the first interval `Δt₁` the density
is the closed-form Gaussian evaluated on the grid; for continuous KI the
first-interval survival multiplies by the point-source bridge factor
(`p_hit` with the source pinned at spot). Valuation-date observations keep
the existing t=0 short-circuit conventions (they resolve before the density
starts).

Seasoned states: `knocked_in_at_valuation` starts all mass in `p_in` and
latches the ki scalar fields at 1.0 exactly as today.

### 4.4 Events at observation dates

Order at a shared date mirrors the MC/stacked reference: coupon readout
(alive before KO) → KO absorption → KI transition (with the
`disable_ko_after_ki` narrowing analog).

- **KO at `t_i`:** `absorbed_i = Σ w·ko_w·p_out` (+ the `p_in` term unless
  `disable_ko_after_ki`), then both scale by `(1−ko_w)`, using the same
  smoothed `_event_weight` (and the `_project_quad_event` weights in the
  cell-average mode). `ko_probability_i = absorbed_i` **directly** — no
  discount-factor division. `ed_ko_cf_i = absorbed_i · payoff_i ·
  df(observation→settlement)` via the existing `_ko_discount` / `df_local`.
- **Discrete KI:** smoothed transfer `p_out → p_in`. Design decision: the
  forward mode uses **smoothed events uniformly** (consistent with
  `price()` and the stacked KO rows); the stacked ki-indicator's legacy
  hard mask is a finite-h definitional difference that vanishes under
  refinement and is quantified by battery gate (a). Per-KI-date transfer
  masses fill the `ki_times` / `ki_event_probability` /
  `ki_survival_probability` arrays.
- **Phoenix coupons:** `coupon_probability_i = Σ w·pay_w·(p_out+p_in)` at
  `t_i`, read before KO absorption (coupon on a simultaneous KO is paid,
  matching MC's `alive_before`). No extra surfaces. Implemented through a
  small overridable readout hook (`_forward_extra_readouts`, no-op for
  Snowball) mirroring `_extract_extra_quad_stats`.
- **Terminal:** maturity-no-KI mass `= Σ w·p_out(T)`, with-KI
  `= Σ w·p_in(T)`; legacy `ki_probability = Σ w·p_in(T)` ("settles
  knocked-in": KI'd and never KO'd);
  `ki_survive_knocked_in_probability = ki_probability`. A terminal KO
  observation absorbs first, as today.
- **Residual convention retained:**
  `expected_discounted_maturity_cf = pv − Σ ed_ko_cf − Σ coupon_cf` with
  `pv = self.price(...)` — the distribution reconciles to engine PV by
  construction; forward-vs-stacked differences are confined to the split.

### 4.5 Streams and cost

- `want_ki` false ⇒ no `p_notouch`, ki fields zero (same pruned contract as
  76471b8); `want_coupon` false ⇒ coupon readout skipped.
- Cost per `price_with_events` cell ≈ `price()` + one 2–3-surface forward
  pass ≈ 2× `price()`, versus ~4.3× today post-76471b8 and the ~50-surface
  stacked walk on book rows. Book projection: the 100–150 s zone; measured,
  not promised, by gate (e).

## 5. Code touch points

| File | Change |
|---|---|
| `quantark/asset/equity/param/engine_params.py` | `QuadParams.event_stats_mode` + validation + docstring |
| `quantark/asset/equity/engine/quad/snowball_quad_engine.py` | Track 1 fusion; `_compute_event_stats` dispatch; `_compute_event_stats_forward`; forward step/bridge helpers (reusing `_diffuse_fft` internals and `_bridge_kernels`); `_forward_extra_readouts` hook |
| `quantark/asset/equity/engine/quad/phoenix_quad_engine.py` | `_forward_extra_readouts` coupon readout |
| `quantark/asset/equity/engine/quad/ko_reset_snowball_quad_engine.py` | docstring: flag ignored |
| `test/` | fusion bitwise tests; forward-density battery (§6) |
| `docs/` | CHANGELOG entry; evidence doc under `docs/autocall-engine-perf/` |

## 6. Validation battery (admission evidence)

New test module(s) (`test/test_quad_forward_density_stats.py`, plus the
Track 1 bitwise tests) and a banked evidence document
(`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-<date>.md`).

Cross-arch rule for checked-in tests: bitwise assertions compare two paths
**computed in the same test run** (same-machine invariant — safe on x86 CI),
never a live value against a frozen constant; frozen-golden comparisons in
CI go through the `test/golden_compare.py` tolerance machinery. The
dev-time bitwise golden captures stay in the demo harness, not in `test/`.

- **(a) Grid-refinement agreement.** Product matrix: standard snowball,
  reverse, phoenix, discrete-KI, seasoned mid-schedule,
  knocked-in-at-valuation, `disable_ko_after_ki`, settlement-delayed.
  Every stats field: `|forward − stacked|` under simultaneous grid/step
  refinement must contract toward zero; production-grid (1001) tolerances
  are **set from pilot convergence measurements and banked**, not invented
  upfront.
- **(b) MC cross-check.** Snowball/Phoenix MC event stats (QMC, large-N):
  forward fields within 3σ.
- **(c) Benchmark-free identities.** No-barrier density: mass = 1, mean and
  variance match the analytic lognormal, `Σ w·p·payoff` equals the
  undiscounted Black–Scholes forward value (this gate must pass before any
  event code lands — it pins the kernel orientation). Continuous-KI:
  forward `ki_ever` vs the closed-form constant-vol Brownian-bridge
  first-passage probability.
- **(d) r = q = 0 bisection** on the matrix of (a) to isolate carry bugs.
- **(e) Full-book adapter A/B** (same detached-worktree + PYTHONPATH
  protocol as 2026-08-24; the flag is set by a driver wrapper when
  constructing engines — no adapter change): npv column **exactly equal**
  between modes; leg-PV and greek diffs quantified against the position
  materiality the desk uses; wall time vs the ≈PDE target; timings in one
  window, anomalous arms re-run.

## 7. Phasing

1. Track 1 fusion — bitwise gates, land independently.
2. Forward core, no events — gate (c) density identities.
3. KO absorption + discrete KI + terminal split — gates (a)(d) on the
   discrete subset.
4. Continuous-KI bridge + `p_notouch` — gate (c) first-passage + (a).
5. Phoenix coupons, settlement delays, seasoned states — full (a)(b)(d).
6. Flag + streams pruning + docs + CHANGELOG; run (e); bank evidence.
7. *(out of scope)* default-flip decision on the banked evidence.

Each phase lands only with its gates green; discoveries that break the
design (e.g. bit-identity failing in Track 1, material mass leakage in
Track 2) stop the phase and reopen the relevant section here.

## 8. Risks and mitigations

- **Kernel orientation/sign errors** — caught structurally by gate (c)
  before any event code exists.
- **Smoothed-vs-hard KI definitional deltas at production grid** — gate (a)
  quantifies; if leg-PV-material at grid 1001, revisit (option: hard-mask
  the forward KI transfer to match the stacked definition field-by-field).
- **Edge mass leakage without tail correction** — mass diagnostic + explicit
  beyond-grid bucket if material; never renormalize silently.
- **Negative density lobes from FFT spectral filtering** — intermediate
  negativity is acceptable for linear functionals and vanishes under
  refinement; watched in (a)/(c), never clipped.
- **`_use_cell_average_events` forward analog** — reuses the same projection
  weights; covered by (a) run in both event-projection modes.
- **Shared-host timing pollution** — (e) follows the one-window protocol;
  anomalous arms re-run before being believed (lesson re-learned
  2026-08-24).

## 9. Acceptance criteria

- Track 1: bit-identical to pre-change goldens (price, pwe npv, every stats
  field, snowball + phoenix), full suite green, demo harness shows the
  walk-2 elimination.
- Track 2: battery (a)–(e) green with tolerances banked; npv bit-equal
  across modes on the full book; default behavior (`stacked`) bit-identical
  to pre-change; book wall time with the flag on lands in (or credibly
  near) the 100–150 s zone, measured by protocol (e).
- Documentation: CHANGELOG, `QuadParams` docstring, evidence doc, and an
  update to `docs/autocall-engine-perf/SOLUTIONS-2026-08-24.md` marking F2/F3
  delivered.
