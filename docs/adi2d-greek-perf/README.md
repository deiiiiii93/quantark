# 2D ADI Snowball Greeks — Performance & Precision Research

**Date:** 2026-08-05 · **Branch:** `codex/adi-greek-certification` · **Status:** research complete, nothing merged — every result below was produced by standalone scripts that patch the loaded engine at runtime; no engine source was modified.

This package contains the complete evidence base for making the certified 2D Heston/SLV ADI snowball Greek engine fast enough for production/backtest use, plus the measured answer to "can a coarse grid keep the same precision?". The companion document [`implementation-spec.md`](implementation-spec.md) turns these findings into a workstream plan with acceptance gates.

**Package map**

```
docs/adi2d-greek-perf/
├── README.md               this document (findings)
├── implementation-spec.md  future-work spec (workstreams WS-A … WS-F)
├── scripts/                all 20 research scripts, path-fixed to run from here
│   └── thomas_kernel.c     compiled C kernel (build line in §14)
└── logs/                   raw outputs of every run cited below
```

Units: **"c" = hedge contracts**, the certification's economic unit — `delta_quantum_per_contract = hedge_multiplier × hedge_inception_spot / study_notional = 200 × 4532.52 / 50,000,000 = 0.01813008` delta per contract. Certification bounds: **0.5 c per cell**, **0.1 c mean signed bias** across cells. All errors are quoted against the banked paired-RQMC-QE MC references from the finished Stage-16 Heston checkpoints (7/7 PASS).

---

## 1. Executive summary

| Regime | Certified cost (stock kernel, target grid) | Demonstrated stack | Accuracy |
|---|---|---|---|
| ordinary (5 of 7 cells) | 45.2 s (`ordinary_full`) | joint-coarse grid + C kernel ≈ **6.4 s** | equal (≤ 0.07 c) |
| σ-collapse regime | 53.0 s | semi-Lagrangian v-transport @ n_v≈45–60 + C kernel ≈ **5–7 s** | **better** (−0.04 c vs −0.11 c; PV bias eliminated) |
| near-KI dense | 73.2 s | KI-band thinned mesh (341 nodes) + C kernel ≈ **17 s** | equal (−0.021 c vs −0.025 c, gamma identical) |

Per-lever verdicts:

1. **Kernel** — the march is interpreter-bound, not FLOP-bound. A compiled Thomas kernel with identical IEEE operation order is **bitwise-equal** to the engine over full marches and cuts the sweep cost ~9× (2–2.7× at march level). A pure-NumPy hoisted-guard variant gives 1.3× bitwise for free. The SciPy/LAPACK route is fast but **changes numerics and pivot-guard semantics** — demonstrated, not hypothesized.
2. **Floors (config only)** — every cell passes the certification bounds at joint-coarse grids (all axes coarse at once); the certified floors buy accuracy the 0.5 c bound never consumes, except in the `path_focused` regime.
3. **Time axis** — clean second-order expansions survive event damping, so **Richardson pairs work**: target-or-better delta from (n_t/4, n_t/2) solves at 60–100% of the target-row cost.
4. **v axis** — the donor-cell upwind fallback is measured first-order (p = 1.19). **Semi-Lagrangian transport along exact CIR characteristics is flat in n_v (converged at n_v = 45)** and removes the σ-collapse PV bias entirely.
5. **x axis** — near-KI spot-axis error is **alignment noise, not convergence** (±0.12 c swings across n_x 300→750). Node *placement* fixes what node *count* cannot: a 341-node KI-band mesh reproduces the certified 600-node row at 1.8× less cost.
6. **Event damping** — a real accuracy/robustness trade, now quantified: softening damping cuts near_ko delta time-error ~4×, but damping is gamma-oscillation insurance at near_ki.
7. **Precision floor** — beyond ~0.05–0.1 c, the PDE−MC gap is estimator semantics (finite-bump h-width exposure vs MC pairing), not discretization. No grid technique can pass that floor; five of seven cells already sit on it at *coarse* grids.
8. **Desk serving (round 2, §14)** — the desk quality contract (~1 c) is a different contract from certification (0.5 c / 0.1 c bias): a desk-A grid profile passes all 7 cells at ≤ 1 c in **27.9 s total** (regime mix 20.1 s), and a **ladder cache** — one cached live-surface march per position, re-read through the production stencil readout — makes intraday spot marks and v0 remarks **sub-millisecond lookups** (bitwise-identical to `calculate_greeks` at the build spot), with re-solves only on recalibration / date roll / lifecycle triggers.
9. **SLV carry-over (round 3, §14.3)** — the desk levers re-measured on the `heston_slv` cells once their references banked: kernel bitwise TRUE on a full SLV march, desk-A **7/7 in 26.3 s**, ladder parity bitwise, same time-axis p≈2 / oscillatory-x / n_v p≈1.18 structure (the axis pathologies live in the shared core). Two SLV-specific contract items: the recalibration fingerprint must include the **leverage surface**, and the free v0 remark holds only **given a fixed L**.

---

## 2. Baseline

Stage-16 certification (schema 9): Heston **7/7 PASS**, mean signed delta bias −0.0481 c of the 0.1 budget, worst cell |delta| 0.1124 c of 0.5, worst interval edge 0.212. Only `sigma_collapse` selects `path_focused` (132/133 donor-cell rows, max Péclet 1.09e4); the other six run the `power` grid with ≤ 2/133 fallback rows.

Production Greek floors (engine class constants): `n_x ≥ 300`, `n_v ≥ 135`, `≥ 1600 steps/yr`; dense-KI (proximity-gated: only when the bump stencil straddles a live KI barrier — solver `greek_time_grid_policy`, barrier-straddle test): `n_x ≥ 600`, `16 steps/tick`.

Certified per-cell Greek-mark cost, stock kernel, 1% one-march readout: `ordinary_full` 45.2 s · `ordinary_decayed` 23.3 s · `near_ko` 16.3 s · `near_ki` 73.2 s · `low_feller` 30.7 s · `sigma_collapse` 53.0 s · `near_expiry` 4.9 s — **246.6 s** for all seven.

> **Timing caveat:** all wall-clock numbers in this document were measured while two SLV certification cells (~5 cores of 14) ran concurrently. Ratios are trustworthy; absolute seconds carry 10–30% load noise.

## 3. Where the time goes

cProfile of one production march (`near_ko`, 300×135×1600; 19.5 s unprofiled, 27.7 s profiled; `scripts/profile_march.py`, `logs/profile.log`):

| Component | Profiled time | Share |
|---|---|---|
| `solve_tridiag_batch` (10,792 calls) | 17.8 s cum | **64%** |
| — of which `np.any` pivot checks (2.35M calls) | ~4.1 s | 15% |
| `_A1` / `_A0` / `_A2` explicit operators | 3.2 s | 12% |
| concentrated-grid ODE build (`_legacy_ode_f`, 377k calls) | 1.8 s | 6% |
| orchestration, projections, misc | ~4.9 s | 18% |

20.2M Python function calls per price. Nominal `n_t=1600` became **3,202 actual steps** (2,194 Craig–Sneyd + 1,008 damped Douglas from event restarts); each CS step does two sweeps per direction → 5,396 solves per direction per march. The arithmetic content of the entire march is ~8 GFLOP — sub-second in compiled code. The engine pays interpreter overhead, not FLOPs: `solve_tridiag_batch` runs the Thomas recurrence as a Python loop along the solve axis (300 iterations per S-sweep, 135 per V-sweep, ~5 small NumPy ops plus one `np.any` each).

## 4. Kernel study

Four drop-in variants for `solve_tridiag_batch` (same signature, same full-length convention). Scripts: `boosted_tridiag_np.py`, `boosted_tridiag.py` (SciPy), `thomas_kernel.c` + `boosted_tridiag_c.py`; unit gate `c_kernel_unit.py`; march A/B `boosted_march_demo_c.py`.

| Variant | Sweep cost/march† | March speedup (3 cells) | Bitwise vs stock | Pivot-guard contract | Adoption cost |
|---|---|---|---|---|---|
| stock (pure NumPy, per-iteration guards) | 17.0 s | 1.0× | — | — | — |
| **numpy+** (hoisted guards) | 12.0 s | 1.29–1.37× | **BITWISE** | preserved (raises post-sweep) | none; ~15-line diff |
| **scipy** (block-concat + multi-RHS `gtsv`) | 5.0 s | 1.81–2.26× | no (~1e-15/solve, ~1e-14 PV) | **changed** | golden re-baseline + sign-off |
| **C** (transposed, `-ffp-contract=off`) | **1.9 s** | 1.95–**2.71×** | **BITWISE** | **preserved exactly** | C toolchain; packaging decision |

† micro-bench at the real shapes (S: 135 distinct systems × N=300; V: 1 system × 135 × 300 RHS; 5,396 solves each per near_ko march). A quieter-machine re-run from this package measured stock 12.8 / numpy+ 9.2 / scipy 4.1 / **C 1.3** s/march — same ordering, better ratios.

**Full-march agreement** (four arms, identical engines, kernel patched into `adi_core`'s namespace per arm): `numpy+` and `C` produce **bit-for-bit identical PVs and Greeks** on all three demo cells (near_ko incl. the production one-march delta/gamma readout; sigma_collapse; near_expiry) — goldens and the certification remain valid with no re-run. SciPy differs by 3.6e-15–9.1e-15 absolute PV.

**Guard parity, demonstrated:** on two constructed trip cases (first-pivot zero; an interior denominator that cancels *exactly* in IEEE arithmetic), stock/`numpy+`/`C` raise the identical `NumericalError("zero pivot … (refine grid)")`, while **SciPy silently solves both to finite values** — a zero Thomas pivot is not a singular matrix, and LAPACK's row pivoting survives it. That is the concrete semantic change any LAPACK adoption must sign off.

Mechanics worth recording:
- **Block concatenation** (S-sweep): the full-length convention's ignored entries (`sub[:,0]`, `sup[:,-1]`) become the zero seams that exactly decouple 135 systems concatenated into one size-40,500 tridiagonal system → one `dgtsv` call. Partial pivoting cannot cross a seam (|0| never wins a pivot comparison).
- **The V-sweep is natively multi-RHS** (the variance operator is x-independent; the engine already `broadcast_to`s it) — stride-0 detection routes it to one `gtsv` with 300 RHS.
- **The C kernel's transposed layout** `(n, n_sys)` vectorizes the recurrence *across systems* (NEON `fdiv.2d` pipelines independent divisions), keeping each system's operation order — hence bit-identity — while beating the system-major variant 1.9× on S and 6.9× on V.
- `-ffp-contract=off` is mandatory: FMA contraction rounds once where NumPy rounds twice and silently breaks bit-identity.
- Ruled out: Numba (not installed; C matches it), JAX/PyTorch (dependency weight; no bit-stability), GPU batched tridiag (wrong hardware, wrong shapes), cached `gttrf` factorization (back-substitution dominates; ≤0.15 ms extra on S, slower on V).

## 5. Grid-floor headroom (config lever)

Two independent evidence sets:

**(a) Per-axis certification ladders** (banked; `scripts/ladder_headroom.py`): every coarse row of every axis in every cell passes the 0.5 c bound with >2× margin. The certified floors move ordinary-cell deltas by ≤ 0.04 c — refinement the bound never consumes.

**(b) Joint-coarse live run** (all axes coarse at once — the point the one-axis-at-a-time ladders never tested; `scripts/relaxed_floors_demo.py`, C kernel):

| Cell | Joint-coarse grid | Δ err (at target) | Γ err (at target) | Cost | |
|---|---|---|---|---|---|
| ordinary_full | 200×90×2400 | −0.044 c (−0.038) | +0.020 c (+0.018) | 6.4 s | PASS |
| ordinary_decayed | 200×90×1200 | −0.051 c (−0.013) | +0.040 c (−0.001) | 5.5 s | PASS |
| near_ko | 200×90×800 | −0.057 c (−0.017) | +0.000 c (−0.004) | 5.8 s | PASS |
| near_ki | 450×90×2016 | +0.284 c (−0.025) | −0.020 c (+0.151) | 13.5 s | PASS |
| low_feller | 200×90×1600 | −0.148 c (−0.107) | +0.103 c (+0.059) | 8.1 s | PASS |
| sigma_collapse | 200×90×2400 | −0.205 c (−0.112) | −0.032 c (−0.011) | 13.9 s | PASS |
| near_expiry | 200×90×200 | −0.069 c (−0.025) | −0.006 c (+0.012) | 2.5 s | PASS |

All seven cells: **55.7 s vs 246.6 s certified (4.4× with the C kernel)**. Mean signed bias −0.041 c — but that mean is flattered by near_ki's +0.28 cancelling the other cells' negatives; per-cell margin is the honest metric, and near_ki/sigma_collapse each gave up ~10× of theirs. **Recommendation: relax regime-aware, not globally** — coarse for the five `power`-regime cells (errors ≤ 0.07 c), keep `path_focused` cells at 135/1600 until WS-C (semi-Lagrangian) retires that need, keep the dense-KI floors (proximity-gated; rarely paid).

## 6. Richardson extrapolation: viability by axis

From the banked three-point ladders (`scripts/richardson_analysis.py`), fitting `U(h) ≈ U∞ + C·hᵖ`:

| Axis | Verdict | Evidence |
|---|---|---|
| **n_t** | **clean p = 1.96–2.66 on every monotone cell** | pair recipe (coarse+target, p=2) beats the *fine* row: decayed −0.047→−0.001 c, near_ko −0.054→−0.005 c, near_expiry −0.070→−0.010 c |
| **n_x** | **oscillatory in 5/7 cells → invalid** | e.g. near_ki +0.118 → −0.025 → +0.029 c; node-vs-kink alignment shifts with n_x; no smooth expansion exists |
| **n_v** | p ≈ 2–4 where `power` runs; **p = 1.19 on sigma_collapse** (donor-cell, measured in production data) | fitted-p extrapolation −0.112→−0.009 c, but the p=2 recipe under-corrects (−0.061 c) — wrong-exponent risk is real; the engine's own `variance_operator_diagnostics()` (donor-cell fallback fraction) is the regime signal for choosing p |

Why the time axis stays clean **despite** 31% damped Douglas steps: the damped-step count scales with the *event count*, not n_t, so their O(dt²) local errors sum to a fixed-count, still-second-order global contribution.

**Live confirmation** (`scripts/lever1_time_richardson.py`, all 7 cells, three marches each at n_t/4, n_t/2, n_t on the target spatial grid, production one-march readout, C kernel):

| Cell | Target row | Cheap pair (n_t/4, n_t/2) | Quality pair (n_t/2, n_t) |
|---|---|---|---|
| ordinary_full | −0.0382 c @ 24.4 s | −0.0357 c @ 17.8 s | −0.0360 c @ 36.9 s |
| ordinary_decayed | −0.0125 c @ 16.3 s | **−0.0048 c @ 16.9 s** | −0.0010 c @ 28.4 s |
| near_ko | −0.0171 c @ 14.6 s | **−0.0085 c @ 8.7 s** | −0.0047 c @ 19.9 s |
| near_ki | −0.0245 c @ 37.2 s | −0.0802 c @ 31.3 s | −0.0754 c @ 56.7 s |
| low_feller | −0.1072 c @ 18.8 s | −0.0982 c @ 20.3 s | −0.1020 c @ 31.9 s |
| sigma_collapse | −0.1124 c @ 22.9 s | −0.1074 c @ 24.1 s | −0.1047 c @ 37.2 s |
| near_expiry | −0.0249 c @ 3.0 s | −0.0184 c @ 3.3 s | −0.0100 c @ 4.9 s |

Gamma never degrades under the pairs (near_ki's grows +0.151→+0.174 c, still far inside the bound). Two caveats are part of the finding:
- **Richardson converges to the PDE's own h→0 limit, not to the MC reference.** near_ki's pair lands at ≈ −0.08 c because that *is* the converged PDE−MC offset; the certified row's −0.0245 c is flattered by unconverged time error cancelling it.
- The raw n_t/4 member at near_ki is +0.75 c — outside the certification bound on its own. Pairs need an asymptotic-range guard before any productionization (spec WS-E).

## 7. Semi-Lagrangian v-transport (the σ-collapse fix)

**Design** (demo subclass in `scripts/lever2_sl_vtransport.py`; ~150 lines): the shared v-generator keeps *only* diffusion (monotone by construction, local Péclet ≡ 0, no upwind fallback and no centered wiggle); drift is transported along the **exact CIR characteristics** `v_foot = θ + (v−θ)·e^{−κ·dt}` by precomputed clipped-cubic-Lagrange interpolation (mean reversion contracts, so feet are always interior — no outflow BC); Strang split (advect dt/2 → drift-free ADI step → advect dt/2); the degenerate v=0 row (pure drift) becomes advection + implicit identity. Advection is one cached (n_v×n_v) matrix product per half-step — ~15–25% march overhead at equal grid.

**Results, sigma_collapse** (n_x=300, n_t=2400 = 800/yr; delta vs banked reference, PV vs the session RQMC-QE reference +1.511243 ± 0.011054):

| n_v | shipped upwind | semi-Lagrangian |
|---|---|---|
| 45 | −0.394 c · PV +12.8 SE | **−0.048 c · PV +1.0 SE** |
| 60 | −0.288 c · +9.5 SE | −0.043 c · +0.9 SE |
| 90 | −0.199 c · +6.6 SE | −0.036 c · +0.8 SE |
| 135 | −0.135 c · +4.6 SE | −0.039 c · +0.9 SE |

**Flat in n_v — converged at 45 nodes**; the first-order PV bias is *eliminated*, not reduced. n_t refinement contracts cleanly (PV +8.2 → +3.6 → +0.9 SE at 600/1200/2400 — no Rannacher-crutch divergence). `ordinary_full` sanity: SL −0.0515 c vs shipped −0.0486 c — no regression. For context, matching upwind's accuracy by refinement alone was measured to need n_v ≈ 1200 (~13×) in the accuracy-phase study (`scripts/nv_order.py`, `logs/nv.log`).

Alternatives rejected on evidence: Scharfetter–Gummel (the measured Péclet distribution at n_v=90 is bimodal — median 1157, **zero rows in the 2–50 band** where SG differs from upwind); explicit deferred-correction of the central flux (anti-diffusive correction CFL ≈ 2.6 > 1 in the path-focused tube → unstable exactly where it's needed).

## 8. Event-damping economics (config knob today)

`PDEParams` already exposes `event_rannacher_steps` (default 2), `event_theta` (1.0), `rannacher_at_events`. Measured at target grids (`scripts/lever3_event_damping.py`):

| Config | near_ko Δ | near_ko Γ | sigma_collapse Δ | near_ki Δ | near_ki Γ |
|---|---|---|---|---|---|
| production (eks=2, θ=1.0) | −0.0171 c | −0.0035 | −0.1124 | −0.0245 | +0.1507 |
| eks=1 | −0.0106 | −0.0041 | −0.1083 | −0.0540 | +0.1721 |
| eks=4 | −0.0303 | −0.0022 | −0.1206 | +0.0348 | **+0.1049** |
| damping OFF | −0.0045 | −0.0047 | −0.1045 | −0.0834 | +0.1924 |
| event_theta=0.5 | **−0.0037** | −0.0046 | −0.1045 | −0.0827 | +0.1947 |

The damping constant is a real, monotone delta cost (~4× at near_ko) — **and** real gamma-oscillation insurance at the KI kink (damping off degrades near_ki gamma +0.151→+0.192; heavier damping improves it to +0.105). near_ki deltas under softer damping move *toward* the honest converged −0.08 c (§6). Keep production defaults globally; flip only in delta-centric, non-KI-adjacent contexts.

## 9. Spot axis: alignment, not count

near_ki sweep at n_v=135, n_t=4032, auto grid focus (= KI) (`scripts/lever4_x_mesh.py`):

| n_x | Δ err | Γ err | cost |
|---|---|---|---|
| 300 | +0.005 c | −0.077 | 15.0 s |
| 400 | +0.096 | −0.306 | 20.2 s |
| 450 | +0.118 | +0.056 | 24.2 s |
| **600 (certified)** | −0.025 | +0.151 | 30.1 s |
| 750 | +0.029 | +0.130 | 41.8 s |

Non-convergent: ±0.12 c delta / ±0.2 c gamma swings with no systematic trend — node-vs-kink **alignment noise**. (n_x=300's +0.005 c is luck; do not cherry-pick it.) The placement fix: keep the 600-row's density inside a ±12% log-band around the KI barrier, thin to every other node outside → **341 nodes reproduce the certified row in both Greeks** (−0.0213 c / +0.1511 c vs −0.0245 / +0.1507) at **16.6 s vs 30.1 s**. Same solution cheaper — not different-luck. (Demo subclass overrides `_layer_x_nodes`; the core already accepts injected non-uniform `x_nodes`.)

## 10. The estimator-semantics floor

Across every lever, a residual PDE−MC offset of ~0.05–0.1 c survives all refinement and extrapolation: the ordinary cells' flat ~−0.04 c across all nine ladder rows, and near_ki's converged −0.08 c. Its origin is not discretization: the harness's barrier-adjacent values are declared **h-width finite-bump exposures** (`finite_bump_exposure: true`), while the MC reference measures its own paired estimator, plus reference SE (0.0015–0.031 c/cell) and substep bias. **Grid techniques cannot cross this floor.** Precision beyond it is readout-semantics work, not resolution.

## 11. Techniques evaluated and excluded

| Technique | Verdict |
|---|---|
| Scharfetter–Gummel exponential fitting | ≡ upwind on 98.9% of rows here (Péclet bimodal; empty 2–50 band) |
| Explicit deferred-correction (central) in v | anti-diffusion CFL ≈ 2.6 in the tube → unstable where needed |
| 4th-order compact stencils; sparse-grid combination | kink/alignment sensitivity (§9 is the direct evidence); 2D gains marginal |
| MC-assisted PDE hybrids | excluded by house rule (deterministic engines stay deterministic) |
| GPU batched tridiag (cuSPARSE), JAX `tridiagonal_solve`, PyTorch | wrong hardware / heavy deps / no bit-stability |
| COS / CF backward induction (pure Heston) | legitimate spectral endgame, but a new engine; SLV still needs the PDE — scoped as deferred (spec WS-F) |

## 12. Combined stack and backtest projection

Kernel (×2–2.7) and grids (×2–4, regime-aware) multiply; SL transport turns the worst regime into an ordinary one; the KI-band mesh halves the dense-cell cost. Net per-mark: **45 s → ~6 s** (ordinary 3y), **53 s → ~5–7 s with better accuracy** (collapse), **73 s → ~17 s** (near-KI). For the pending G4/G1/G2 snowball backtest re-run (27 paths × ~750 marks), that moves the PDE leg from weeks-scale to **hours** on a 12-worker box.

## 13. What requires what

| Lever | Touches | Bitwise-safe |
|---|---|---|
| numpy+ kernel | `quantark/util/numerical/tridiag.py` internals | yes |
| C kernel | optional accelerator + packaging decision | yes |
| scipy kernel | same file; **contract change** | no |
| floors, event damping, grid focus | config / class constants | n/a (grid choice) |
| time-Richardson | driver recipe (no engine) or a `calculate_greeks` mode | n/a (new estimator, opt-in) |
| SL v-transport | `adi_core` opt-in scheme (the one real engine addition) | n/a (new scheme, opt-in) |
| KI-band density | grid-layer option | n/a |

## 14. Desk-grade serving: tolerance profile + ladder cache (round 2)

Round 2 answers "is this engine usable for desk hedging?" The desk works to ~1 c per position, not the certification contract (0.5 c/cell, 0.1 c mean bias) — and no desk re-solves a PDE per tick. Two demos, run after the round-1 stack.

### 14.1 Desk grid profiles (`scripts/desk_profile_floors.py`, `logs/desk_floors.log`)

Profiles are fractions of each cell's joint-coarse rung (§5), C kernel patched in, production one-march readout, banked MC references. Desk bound 1.0 c per Greek per cell.

| Profile (f_x/f_v/f_t) | 7-cell total | worst \|dErr\| | worst \|gErr\| | mean delta bias | desk pass |
|---|---|---|---|---|---|
| coarse 1/1/1 (§5 baseline) | 39.0 s | 0.284 c | 0.103 c | −0.041 c | 7/7 (7/7 also ≤ cert 0.5 c) |
| **desk-A 0.80/0.67/0.75** | **27.9 s** | **0.515 c** (near_ki) | 0.216 c | −0.058 c | **7/7** (6/7 ≤ cert bound) |
| desk-B 0.60/0.50/0.50 | — | grid guard trips on `sigma_collapse` | | | |
| desk-C 0.50/0.36/0.33 | — | guard + `low_feller` FAIL (−1.18 c) | | | |

- **desk-A is the recommended desk profile**: every cell ≤ 1 c with margin, ~4 s/position average; the cheapest-passing regime mix reaches **20.1 s for all 7 cells** but rides a 0.09 c near_ki margin (desk-C +0.909 c of 1.0) — not robust against the ±0.12 c alignment noise (§9), so the mix is quoted, not recommended.
- The engine **self-protects**: below ~160 x-nodes the σ-collapse S-grid spacing guard raises (`achieved spacing 0.00795 exceeds 2x target eps_crit`) instead of returning silently degraded Greeks.
- At desk grids the march no longer dominates — per-solve Python overhead (grid/event construction, ~2 s) does. `ordinary_full` desk-B has ~7× fewer march ops than coarse yet runs only 13% faster. Further *solve* speedups need overhead work, not kernel work — or no solve at all, which is §14.2.

### 14.2 Ladder cache (`scripts/desk_ladder_cache.py`, `logs/desk_ladder.log`)

The production `calculate_greeks` already prices its ±1% bump stencil by bilinear interpolation of **one** solved surface (`_solve_live_surface` → `core.interpolate`). The prototype `SnowballLadderCache` keeps that `(core, surface)` pair alive per position and re-reads it at any (spot, v0) through the identical stencil arithmetic; re-solves happen only on three triggers.

Measured on desk-A grids (ladder lookup vs full fresh re-solve at the moved spot):

| Check | ordinary_full (build 3.2 s) | near_ki (build 6.0 s) |
|---|---|---|
| parity at build spot | **bitwise-identical** to `calculate_greeks` | **bitwise-identical** |
| spot moves | ±0.5…±5%: \|dDiff\| ≤ 0.016 c, \|gDiff\| ≤ 0.015 c | ±0.5…±3%: worst +0.13 c delta, +0.39 c gamma at the KI kink (mesh-alignment noise, §9) — all PASS at 1 c |
| v0 remark (0.04→0.03/0.06) | ≤ 0.002 c delta, ≤ 0.015 c gamma — the surface is v0-independent; only the readout line moves | n/a |
| lookup cost | 0.6–0.8 ms (validity check dominates) | ~0.2 ms |
| re-solve cost | 3.7–4.9 s | 4.6–5.5 s |

All three invalidation triggers fire with correct reasons and rebuild cleanly:

1. **Recalibration** — κ/θ/σ/ρ change (v0 deliberately excluded: it is a readout argument, not a surface parameter). Rebuild 4.8 s, post-rebuild parity bitwise.
2. **Date roll** — the surface is a t=0 snapshot for one valuation date.
3. **Lifecycle** — the engine's own `_valuation_state_signature` is the oracle: the overnight `_otc_lifecycle_knocked_in` flag flips it (discrete-KI product), and a continuous-KI variant invalidates on an intraday breach (lookup at 74.6 below KI 75.0 → INVALID). The knocked-in rebuild is a **single V1 march** (3.15 s vs 4.84 s two-march build).

Desk arithmetic for a ~100-position book: intraday spot/v0 marks are **lookups (≪1 s for the whole book)**; a trigger-driven full re-mark at desk-A is ~4 s/position single-core ≈ **~1 minute on 8 workers** — recalibration cadence, not tick cadence. The estimator floor (§10) still applies: the ladder reproduces the *production estimator*, so its agreement with a fresh re-solve (≤0.02 c away from kinks) is tighter than either's agreement with MC truth (~0.05–0.1 c).

### 14.3 SLV carry-over (`scripts/slv_desk_demo.py`, `slv_richardson_analysis.py`; `logs/slv_desk.log`, `slv_richardson.log`)

Rounds 1–2 were measured on the Heston cells only (the SLV certifications were still running). Once the SLV references banked, the desk levers were re-measured on `heston_slv__*` through `cert16.make_pde_engine` (the exact certified SLV construction):

| Claim | Heston (rounds 1–2) | SLV (round 3) |
|---|---|---|
| C kernel bitwise over a full march | TRUE | **TRUE** (1.75× at desk-A grid — lower than the 2–2.7× march ratio because leverage/coefficient rebuild is a larger share, and desk grids are overhead-dominated) |
| desk-A profile, 7 cells @ 1 c | 7/7, 27.9 s | **7/7, 26.3 s** (worst conclusive-ref err 0.387 c; worst target-grid drift 0.509 c, near_ki) |
| ladder parity at build spot | bitwise | **bitwise** (both cells) |
| ladder vs re-solve, ordinary ±3% | ≤ 0.016 c | **≤ 0.016 c** |
| ladder vs re-solve, near-KI kink | worst +0.39 c gamma | **worst +0.34 c gamma** (same alignment-noise band) |
| v0 remark via readout line | ≤ 0.015 c | **≤ 0.015 c given the same L** (see caveat) |
| time axis Richardson-clean (banked ladders) | p ≈ 2 | **p = 2.02 / 1.96 / 2.37** where above ref noise |
| x axis oscillatory (Richardson invalid) | yes | **yes** |
| σ-collapse n_v first-order upwind signature | p = 1.19 | **p = 1.18** — the pathology lives in the shared core, so the WS-C motivation carries |

SLV-specific caveats (now in the WS-G contract):

- **The recalibration fingerprint must include the leverage surface** (`time_grid`/`strike_grid`/`leverage_grid` identity), not just κ/θ/σ/ρ — measured firing as its own trigger.
- **The "free v0 remark" is conditional on L**: the PDE surface is v0-independent *given a fixed leverage surface* (measured ≤ 0.015 c), but a real SLV v0 remark normally re-calibrates L — which correctly fires trigger 1 and forces a rebuild. Heston has no such coupling.
- **2 of 7 SLV MC references are INCONCLUSIVE** (`near_ki` half-widths ±0.63/±1.41 c, `low_feller` ±0.30 c) — desk verdicts on those cells ride the drift-vs-certified-target-grid column, which is MC-noise-free.
- Not re-measured on SLV (mechanism shared, numbers pending if promoted): SL v-transport itself, KI-band mesh (extra caveat: thinning outside the KI band also thins where the leverage surface has structure), event-damping trade.

## 15. Reproduction

All scripts run from this directory against the repo root (paths are computed relative to `__file__`; they were originally executed from the session scratchpad — logs in `logs/` are those original runs). The A/B pattern: build the stock engine, run; rebind `adi_core.solve_tridiag_batch` (or `solver_mod.HestonSLVADICore`) to the variant; run fresh engines; restore in `finally`.

```bash
cd docs/adi2d-greek-perf/scripts
cc -O3 -ffp-contract=off -shared -fPIC -o thomas_kernel.dylib thomas_kernel.c
<venv>/bin/python c_kernel_unit.py            # bitwise + guard parity + micro-bench
<venv>/bin/python boosted_march_demo_c.py     # 4-arm full-march A/B
<venv>/bin/python richardson_analysis.py      # banked-ladder axis analysis (no solves)
<venv>/bin/python lever1_time_richardson.py   # … lever2/3/4 likewise

# round 2 (desk serving): these import worktree-only symbols through the
# certification module, so the repo root MUST be on PYTHONPATH — the compat
# .pth of an editable install binds `quantark` at interpreter startup, before
# any in-script sys.path.insert can run.
PYTHONPATH=<repo-root> <venv>/bin/python desk_profile_floors.py
PYTHONPATH=<repo-root> <venv>/bin/python desk_ladder_cache.py

# round 3 (SLV carry-over; needs the completed heston_slv checkpoints)
PYTHONPATH=<repo-root> <venv>/bin/python slv_desk_demo.py
<venv>/bin/python slv_richardson_analysis.py   # banked SLV ladders, no solves
```

Requires the Stage-16 Heston checkpoints under `output/adi_greek_certification/checkpoints/` (banked MC references) and, for `sigma_collapse_reference.py`, ~20 min to rebuild the RQMC PV reference. Environment of record: arm64 macOS, Apple clang 15, scipy 1.17.1, numba absent, 14-core/48 GB, two SLV cert cells running concurrently during all timings.
