# Implementation Spec — 2D ADI Greek Performance & Precision Program

**Status:** DRAFT for review · **Date:** 2026-08-06 · **Prereq reading:** [`README.md`](README.md) (all numbers cited here are measured there) · **Prototype code:** `scripts/` in this package — every workstream below has a working, validated prototype; "implementation" means promoting a demonstrated design into the engine behind the stated gates, not inventing one.

## 0. Goals, non-goals, principles

**Goals**
- G1: cut the per-mark cost of certified 2D snowball Greeks by ≥ 5× on ordinary cells and ≥ 3× on the special regimes, without weakening any certification bound.
- G2: remove the σ-collapse regime's first-order v-axis penalty at its root (scheme, not resolution).
- G3: keep every default-path change **bitwise-identical**; anything that changes numerics ships as an opt-in mode (house rule: exact semantics by default).

**Non-goals**
- No changes to the MC reference stack, the certification harness semantics, or the equivalence bounds.
- No GPU, no new heavyweight dependencies (JAX/PyTorch/Numba), no MC-inside-PDE hybrids.
- The COS/CF engine idea stays a scoping note (WS-F3), not a commitment.

**Principles**
- One workstream = one PR-sized unit with its own gates; later workstreams must not be blocked on earlier *optional* ones.
- Every gate that says "bitwise" means `np.array_equal` on full-march surfaces/readouts, not a tolerance.
- Re-certification runs are PDE-only against the **banked** Stage-16 MC references (checkpoints are the fixture; no MC re-runs).

---

## WS-A Kernel replacement

### WS-A1 `numpy+` hoisted-guard Thomas (default swap)

**Change:** rewrite `quantark/util/numerical/tridiag.py::solve_tridiag_batch` internals per `scripts/boosted_tridiag_np.py`: store denominators, run the sweep without per-iteration `np.any`, single guard check post-sweep under `np.errstate` suppression; same exception, same message.

**Gates**
- A1-G1 bitwise: full-march PV + one-march delta/gamma readout equal (`==`) to stock on the three demo cells (near_ko, sigma_collapse, near_expiry) at target grids, **plus one `heston_slv` march** (SLV bitwise measured TRUE in round 3, `slv_desk_demo.py` Part 0 — the kernel is model-agnostic; the gate pins that).
- A1-G2 guard parity: the two constructed trip cases (first-pivot zero; interior exact-IEEE-zero denominator) raise the identical `NumericalError`; add both as unit tests.
- A1-G3 perf: ≥ 1.25× on the near_ko target march.
- A1-G4 full test suite + replay goldens untouched (bitwise ⇒ expected to pass without re-baseline).

**Effort:** S. **Risk:** none identified.

### WS-A2 Compiled C accelerator (opt-in, auto-detected)

**Change:** ship `thomas_kernel.c` (transposed variants only) + a ctypes loader as an *optional accelerator*: at import, try to load a prebuilt `thomas_kernel` shared library; on success route `solve_tridiag_batch` through it, else fall back to WS-A1 silently. Provide `scripts`-style build entry (`python -m quantark.util.numerical.build_thomas_kernel` or a Makefile target) — **decision D-1 (packaging)** below governs whether wheels ever carry the binary.

**Design constraints (from the research):**
- compile with `-O3 -ffp-contract=off -fPIC`; FMA contraction breaks bit-identity (measured).
- transposed `(n, n_sys)` layout; stride-0 broadcast detection routes the V-sweep to `thomas_multi_rhs_t`.
- pivot guard inside the kernel at the same 1e-14 threshold, mapped to the same `NumericalError`.

**Gates**
- A2-G1 bitwise vs WS-A1 (hence vs stock) on the full 4-arm demo matrix; guard-parity tests pass against the compiled path.
- A2-G2 perf: ≥ 2× near_ko target march vs stock; ≥ 8× on the tridiag micro-bench.
- A2-G3 fallback: absence of the binary must be observationally identical to WS-A1 (CI job runs the suite both ways).
- A2-G4 an introspection hook (`tridiag_backend() -> "numpy" | "c"`) so runs record which kernel produced them.

**Decision D-1 (user):** keep quantark a pure-Python wheel with the accelerator as a local build (recommended; zero packaging risk), or move to platform wheels (cibuildwheel) so PyPI users get it. Default assumption: pure wheel + local build.

**Effort:** M. **Risk:** packaging only; numerics risk retired by A2-G1.

### WS-A3 SciPy LAPACK route — contingency only

Only if D-1 rejects any compiled artifact *and* 1.3× (A1) is insufficient. Requires: golden re-baseline behind `test/golden_compare.py` tolerances, explicit sign-off that near-zero-pivot systems now solve instead of raising (demonstrated behavior), and a post-solve finiteness guard. **Not recommended** while A2 exists. Effort M; semantic-risk M.

---

## WS-B Regime-aware Greek floors (config)

**Change:** split the production floor constants on `HestonSnowballPDESolver` (and the SLV twin) by variance regime, resolved from the existing auto classifier at `greek_time_grid_policy` time:
- `power` regime (classifier does not select `path_focused`): `greek_min_n_v 135→90`, `greek_min_steps_per_year 1600→800`, `greek_min_n_x 300→200`.
- `path_focused` regime: keep 300/135/1600 **until WS-C lands**, then revisit (expected 300/60/800).
- dense-KI floors: unchanged (proximity-gated; near_ki joint-coarse margin loss measured 10×, not worth it).

**Gates**
- B-G1 Stage-16 PDE-only re-certification against banked references at the new floors: every cell PASS with per-cell |Δ| ≤ 0.10 c (tighter than the 0.5 bound — locks in the measured margins: ordinary cells ≤ 0.057 c at joint-coarse), mean signed bias within budget *computed per regime as well as pooled* (guard against the cancellation flattering measured in the research).
- B-G2 schema bump + `production_greek_grid_policy` records the regime and the floor set used.
- B-G3 cost telemetry: all-cell Greek-mark total ≤ 90 s stock kernel / ≤ 35 s with WS-A2 (measured joint-coarse: 55.7 s with the C kernel, but B keeps near_ki + path_focused at target).

**Depends on:** nothing (WS-A accelerates the re-cert but is not required). **Effort:** S code, S re-cert. **Risk:** low; bounded by B-G1. **Both model families:** B-G1 runs on `heston` *and* `heston_slv` cells (SLV desk-A sweep measured 7/7 in round 3; 2 of 7 SLV MC refs are INCONCLUSIVE — for those, certify on drift vs the certified target-grid PDE value).

---

## WS-C Semi-Lagrangian v-transport (opt-in scheme; the one real engine addition)

*(Round 3: the banked SLV ladders show the same first-order n_v signature — fitted p = 1.18 vs Heston's 1.19 — so the motivation carries to SLV; the v-generator is leverage-independent. C-gates run on both families.)*

**Change:** add `v_drift_scheme="semi_lagrangian"` to `adi_core` (ctor already validates scheme names), implementing exactly the prototype (`scripts/lever2_sl_vtransport.py::SLCore`):
1. `_build_v_generator_coefficients`: scheme branch producing the diffusion-only generator (existing `_vv` weights; `diag = -(sub+sup)`; monotone flags all-true, fallback all-false → diagnostics stay meaningful).
2. `_A2`: skip the degenerate-v0 drift add-on under this scheme (advection owns all drift).
3. `_tri_V`: degenerate v0 row becomes implicit identity; Neumann mode unchanged; top row unchanged.
4. `_advect_v(U, dt_sub)` + per-`dt` weight cache: exact CIR feet `θ + (v−θ)e^{−κ·dt}` (feet provably interior — mean reversion contracts), 4-point cubic-Lagrange weights on the non-uniform grid, linear at edges, **linear-bracket clipping** for monotonicity; one dense (n_v×n_v) matmul per half-step.
5. `_douglas_step` / `_cs_step`: Strang wrap (advect dt/2 → parent step → advect dt/2 → `_bc`).

**Composition rules (all verified in the prototype or by construction):**
- grids: composes with `legacy|power|path_focused|auto` (prototype ran on auto→path_focused);
- SLV leg: leverage multiplies *spot* diffusion only — the v-process (CIR) is untouched, so the scheme applies verbatim; add one SLV smoke gate;
- Rannacher initial halves and event-damped Douglas steps inherit the wrap (dt/2 and dt cache keys);
- S-boundary rows are v-constant and interpolation weights sum to 1 ⇒ boundaries survive advection exactly.

**Gates**
- C-G1 σ-collapse flatness: at n_x=300, n_t=2400, delta errors across n_v ∈ {45,60,90,135} all within 0.06 c of the banked reference and max−min spread ≤ 0.02 c (measured: −0.048…−0.036, spread 0.012).
- C-G2 PV de-biasing: |PV − RQMC reference| ≤ 2 SE at n_v=60 (measured +0.9 SE); reference value pinned in the test fixture.
- C-G3 n_t contraction at n_v=60 (600→1200→2400 moves strictly shrinking; measured −0.051 → −0.029).
- C-G4 no-regression: ordinary_full delta within 0.01 c of upwind at equal grid (measured 0.003 c); full-suite pass with scheme default **unchanged** (`adaptive_upwind` stays the default — SL is opt-in until C-G6).
- C-G5 perf: ≤ 30% march overhead at equal grid (measured 15–25%); with n_v 135→60 the net mark cost must drop ≥ 1.5×.
- C-G6 (separate follow-up PR): flip the auto classifier to select SL for the `path_focused` regime + WS-B floor relax for that regime (135→60, 1600→800) + full Stage-16 re-cert. Default flip is its own decision point with the user.
- C-G7 diagnostics: `variance_operator_diagnostics()` reports `scheme="semi_lagrangian"`, Péclet 0, plus feet-displacement stats (max cells traversed) for observability.

**Depends on:** none (WS-A/B independent). **Effort:** M–L (the prototype is the design; engine plumbing + tests dominate). **Risk:** M — mitigated by C-G1..G4 and opt-in default; known theoretical edge: repeated linear-edge interpolation is diffusive, so keep cubic interior weights (prototype) and gate on C-G1.

---

## WS-D KI-band mesh density (grid-layer option)

**Change:** add a band-density option to the declarative S-grid layer (spec 4.6 builder) used by `_layer_x_nodes`: parameters `ki_band_halfwidth_log` (default 0.12) and `outside_thin_factor` (default 2), active only when the dense-KI policy fires. Node count then *derives* from the band content instead of the global `barrier_greek_min_n_x=600`.

**Gates**
- D-G1 near_ki reproduction: thinned mesh within 0.01 c delta / 0.01 c gamma of the n_x=600 certified row (measured −0.0213/+0.1511 vs −0.0245/+0.1507 — note gamma matched to 0.0004 c).
- D-G2 bump-clone stability: the frozen S-axis reuse across the bump stencil (resolve_bound_layout identity) must hold for banded meshes; add the misuse-fails-closed test.
- D-G3 perf: dense-KI mark ≥ 1.6× cheaper (measured 1.8×).
- D-G4 alignment guard: document (do **not** "fix") the x-axis oscillation; the gate is reproduction of the certified row, never beating it — n_x cherry-picking is explicitly out of bounds.

**Depends on:** none. **Effort:** M. **Risk:** low-M (grid-layer surface area).

---

## WS-E Time-Richardson pair mode (opt-in estimator)

**Change:** two stages.
1. **Driver-level recipe now (no engine change):** a documented helper for the backtest — run (n_t/2, n_t) or (n_t/4, n_t/2) marches, combine `(4·U_f − U_c)/3` on price/delta/gamma (linear readouts commute with extrapolation), guarded.
2. **Optional engine mode later:** `greek_time_richardson=True` on the solver → `calculate_greeks` runs the pair internally and reports both members + the extrapolant in its diagnostics.

**Mandatory guard (from the near_ki finding):** accept the extrapolant only if (i) both members are inside the certification cell bound, and (ii) `|U_f − U_c|/3` (the correction magnitude) ≤ a configured cap (default 0.15 c); otherwise fall back to the fine member and flag. Rationale: the raw n_t/4 member at near_ki measured +0.75 c while the pair was valid — but a pair whose members disagree wildly has left the asymptotic range.

**Semantics note to carry in the docstring:** the extrapolant targets the PDE's h→0 limit, which differs from the MC reference by the estimator-semantics floor (~0.05–0.1 c); at KI-adjacent states it will look *worse* against MC than an unconverged lucky row. That is honesty, not regression; certification bounds absorb it (near_ki converged offset −0.08 c ≪ 0.5 c).

**Gates:** E-G1 reproduce the lever-1 table within noise; E-G2 guard triggers on the near_ki n_t/4 pair; E-G3 opt-in only (house rule).

**Depends on:** none; pairs get 2× cheaper with WS-A2. **Effort:** S (stage 1) / S-M (stage 2).

---

## WS-F Deferred / documentation-only items

- **F1 event-damping guidance:** document the measured trade (delta ~4× vs near-KI gamma insurance) in the engine param guide; keep defaults. Optional later: state-aware `event_theta` (softer when no live KI within the stencil) — needs its own gamma-oscillation study before any default change.
- **F2 explicit-op fusion:** post-WS-A2 the march is ~35% explicit ops + grid build; a C fusion of `_A1/_A0/_A2` bounds at ≲ 2×. Do only if backtest wall-clock still binds after WS-A/B/C.
- **F3 COS/CF backward-induction engine (pure Heston):** spectral accuracy = the asymptotic "coarse grid, same precision"; adjacent to the vectorized Lewis CF infrastructure. Scope: new engine, KO/KI/coupon backward induction, KI-state doubling; SLV keeps the PDE. Write a separate spec before any commitment.
- **F4 grid-ODE build cost** (1.2 s/mark; ~20% post-A2): vectorize or coarsen-and-interpolate the concentrated-grid ODE integration. Round 2 sharpened the motivation: at desk-A grids the per-solve Python overhead (~2 s of grid/event construction) already exceeds the march itself.

---

## WS-G Desk serving layer (README §14; prototypes `desk_profile_floors.py`, `desk_ladder_cache.py`)

The desk quality contract (~1.0 c per Greek per position, refresh on triggers rather than ticks) is weaker than the certification contract and is served by two additions that touch **no solver numerics**.

### WS-G1 Desk grid profile (config; rides WS-B)

A named profile `desk` = 0.80/0.67/0.75 of the joint-coarse rung per axis, exposed through the same regime-aware floor mechanism as WS-B (e.g. `greek_grid_profile="certification" | "desk"`). Measured: **all 7 cells ≤ 1.0 c (worst near_ki +0.515 c), 27.9 s total single-core, ~4 s/position average**.

Constraints baked into the profile table:
- `sigma_collapse` keeps n_x ≥ 160 — the S-grid eps_crit spacing guard correctly raises below that (measured at 120/100); the guard is the safety net, not the sizing rule.
- The cheapest-passing regime mix (20.1 s) is **not** the shipped profile: its near_ki row (desk-C, +0.909 c) has 0.09 c margin against ±0.12 c alignment noise.
- **G1-G1**: PDE-only re-cert of the desk profile vs banked refs, per-cell ≤ 1.0 c with ≥ 0.3 c margin on near-barrier cells, mean |bias| ≤ 0.15 c. (Measured PASS at desk-A except margin on near_ki = 0.485 c — passes.)

### WS-G2 `SnowballLadderCache` (serving layer; independent of every other WS)

One cached live-surface march per position; marks are stencil lookups; re-solves only on triggers. The prototype consumes four engine internals read-only (`_valuation_state_signature`, `create_bump_context`, `_solve_live_surface`, `core.interpolate`). Productization should instead add **one public engine method** — `build_greek_ladder(product, env) -> GreekLadder` — returning a self-contained readout object (grids + surface + stencil config + build fingerprint), so the internals stay private and the cache layer (a small module under the serving/backtest side, not the engine) holds only `GreekLadder`s plus the trigger logic.

Trigger contract (all measured firing):
1. **Recalibration** — κ/θ/σ/ρ in the fingerprint, **plus the leverage-surface identity (`time_grid`/`strike_grid`/`leverage_grid`) for SLV** (measured firing as its own trigger in round 3); **v0 is a readout argument, not a fingerprint field** (surface is v0-independent; measured remark agreement ≤ 0.015 c on both families). SLV caveat: the free v0 remark holds *given a fixed L* — a real SLV v0 remark usually re-calibrates L, which correctly fires this trigger. A v0 outside the cached V-grid span is a range invalidation, not a remark.
2. **Date roll** — valuation-date + T in the fingerprint; any roll invalidates (the surface is a t=0 snapshot).
3. **Lifecycle** — delegate to `_valuation_state_signature` (covers `_otc_lifecycle_knocked_in`, continuous-KI breach, immediate-KO, terminal). Knocked-in rebuilds are single-V1 marches (measured 3.15 s vs 4.84 s).
Plus the same stencil-range guard production `calculate_greeks` applies (bump stencil inside the cached S-grid).

Gates:
- **G2-G1 (parity)**: lookup at build spot bitwise-equal to `calculate_greeks`. (Measured TRUE on both cells; holds by construction while the readout arithmetic is shared.)
- **G2-G2 (transport)**: lookup vs fresh re-solve at moved spots within the desk bound; report the near-KI gamma alignment band honestly. (Measured: ≤ 0.016 c ordinary ±5%; worst +0.39 c gamma at the KI kink.)
- **G2-G3 (triggers)**: unit tests for all three triggers + range guards; a stale-surface read must be impossible without an explicit `is_valid` bypass.
- **G2-G4 (v0 remark)**: remark agreement vs fresh solve ≤ 0.05 c across the cached V-span.
- **D-2 (decision)**: API shape — public `build_greek_ladder` engine method (recommended) vs a serving-side class reaching into engine internals. Also where the cache module lives (serving/risk layer vs `engine/pde`).

Effort: G1 small (config + re-cert run); G2 medium (one engine method + one cache module + tests). Value: intraday marks stop costing solves at all — the highest-leverage desk item in this package.

---

## Ordering, dependencies, effort

```
WS-A1 (S, bitwise)  ──►  WS-A2 (M, D-1 decision)          [independent of B–E]
WS-B  (S + re-cert) ──►  (with WS-C) path_focused floor relax
WS-C  (M–L, opt-in) ──►  C-G6 default-flip decision + re-cert
WS-D  (M)                                             [independent]
WS-E1 (S, driver)   ──►  WS-E2 (S–M, engine mode)          [independent]
WS-G1 (S, config)        rides WS-B (same floor mechanism)
WS-G2 (M, serving)  ──►  D-2 API-shape decision        [independent of A–E]
```

Recommended sequence for the G4/G1/G2 backtest consumer: **A1 → B → E1 (driver pairs where time-limited) → A2 → C → D**, with C-G6 and the D-1 packaging decision as the two explicit user checkpoints. A1+B alone already take an ordinary 3y mark from 45 s to ~10 s; A2+C+D complete the §1 stack (~6 s ordinary, ~5–7 s collapse, ~17 s near-KI).

For the **desk-hedging consumer** the sequence is shorter: **A1 → A2 → G1 → G2** (B/C/D/E improve batch sweeps but are not on the intraday path). That yields: trigger-driven re-marks ~4 s/position single-core (~1 min per 100 positions on 8 workers) and intraday spot/v0 marks as sub-millisecond ladder lookups.

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| FMA/fast-math flags sneak into the C build | silent bit-drift | build script owns flags; A2-G1 bitwise gate in CI |
| Packaging drift (D-1) | wheel platform matrix | default = pure wheel + local accelerator build |
| SL edge behavior (extreme κ·dt, edge-linear interp diffusion) | accuracy in unseen regimes | C-G1/C-G3 gates + feet-stats diagnostics (C-G7); opt-in default |
| Floor relax meets a regime the 7 cells don't span | bound breach in production | B-G1 tightened to 0.10 c per cell; regime-resolved floors keep path_focused/dense-KI conservative until C-G6 |
| Richardson misuse outside asymptotic range | wrong Greek shipped | WS-E mandatory guard; opt-in |
| Cross-family agreement gate (MC/QUAD/PDE) after C-G6 | gate churn | re-run gate in the C-G6 PR; SL changes PDE within existing tolerances (PV moves *toward* MC) |

## Acceptance-gate summary

| Gate | One-line criterion |
|---|---|
| A1-G1 / A2-G1 | full-march bitwise equality vs stock |
| A1-G2 | guard-trip parity (two constructed cases) |
| A2-G3 | binary-absent ⇒ identical to A1 |
| B-G1 | PDE-only re-cert, per-cell ≤ 0.10 c, per-regime bias budgets |
| C-G1..G5 | σ-collapse flat-in-n_v ≤ 0.06 c & spread ≤ 0.02 c; PV ≤ 2 SE; n_t contraction; ordinary regression ≤ 0.01 c; overhead ≤ 30% |
| C-G6 | default flip + full Stage-16 re-cert (separate decision) |
| D-G1 | near_ki ≡ 600-row within 0.01 c / 0.01 c |
| E-G2 | pair guard rejects the near_ki n_t/4 member |
| G1-G1 | desk profile ≤ 1.0 c all cells, ≥ 0.3 c near-barrier margin |
| G2-G1..G4 | ladder parity bitwise; transport ≤ desk bound; all triggers tested; v0 remark ≤ 0.05 c |

---
*Prototypes, measurements, and raw logs: this package. Certification fixtures: `output/adi_greek_certification/checkpoints/` (banked MC references, schema 9).*
