# Vol Models (Dupire LV / Heston / SLV) Improvement Program — Design

**Date:** 2026-07-04
**Status:** Draft — findings confirmed against source 2026-07-04; ready for review
**Scope:** `quantark/volmodels/` — `localvol/` (Dupire builder, surface, LV MC/PDE), `heston/` (analytical CF, MC, ADI PDE, calibration), `slv/` (SLV MC, backward ADI PDE, leverage, Fokker-Planck calibration), `black_scholes.py`
**Origin:** full-module math + code review (all ~3,400 lines read); 28 findings F1–F28
**Out of scope:** `volmodels/risk/` (scenario contracts), engine-layer wrappers in `asset/*/engine/`

---

## 1. Problem

The vol-models core is mathematically sound — Gatheral implied-variance Dupire form, little-trap CF branch, Andersen QE with the exact K0–K4 drift decomposition, mass-conserving finite-volume FP operators — and round-trip test coverage exists. The review nevertheless found:

1. **Two behavioral defects.** The SLV MC on-the-fly leverage clip (`L ≤ √10 ≈ 3.16`) contradicts the FFP calibration clip (`L ≤ 20`), so the two documented calibration routes are *not* cross-checks of each other in high-leverage regimes (F16). The Dupire builder's `safe_divide(1.0, w)` returns 0 (not ∞) for near-zero total variance, inverting the failure direction of the butterfly check (F1).
2. **Large performance headroom.** Every PDE solver runs pure-Python Thomas loops (F10, F26); the Heston calibrator re-runs `scipy.integrate.quad` per option per residual evaluation despite the CF being strike-independent (F9); the FP solver refactorizes a 20k×20k operator every step (F20).
3. **Accuracy headroom.** `−rU` is fully explicit in both ADI solvers, degrading Craig-Sneyd to O(dt) (F5); uniform (x, v) grids waste nodes (F6); central-difference FP convection is the root cause of the tolerated negative probability mass (`tol_neg=0.5`) (F17); the 1D LV CN solver has no Rannacher damping (F25).
4. **Duplication and drift.** `_HestonSLVADI` is ~80% a copy of `_HestonADI` (F22); three separate Thomas implementations exist in the module (F22b); the SLV MC module docstring claims a QE variance scheme the code deliberately does not use, and `FpCalibrationConfig.rannacher_steps`/`.scheme` are validated but never consumed (F23).

### Findings traceability

| # | Finding (file:line at review time) | Severity | Workstream |
|---|---|---|---|
| F1 | `safe_divide(1.0, w)` zeroes `1/w`, `y²/w²` terms; butterfly check bypassed as w→0 (`localvol/dupire.py:142`) | Behavioral edge | WS-A1 |
| F2 | `LocalVolSurface` interpolates in (t, S); `LeverageSurface` in (t, ln S); vol not variance in t (`localvol/surface.py:51`) | Consistency | WS-D3 |
| F3 | Butterfly/calendar checks use one-sided edge stencils → false rejections possible at grid boundary (`localvol/dupire.py:150-164`) | Robustness | WS-D6 |
| F4 | `_fd1`/`_fd2` general non-uniform stencils belong in `util/numerical` (`localvol/dupire.py:31-87`) | Reuse | WS-D2 |
| F5 | `−rU` fully explicit in ADI predictor only → CS formally O(dt) in discount term (`heston/pde_kernel.py:227,234`; `slv/slv_pde_kernel.py:161,168`) | Accuracy | WS-C1 |
| F6 | Uniform x and v grids; `V_max = max(5θ, 0.5, 2v0)` floor makes dV coarse near v0 (`heston/pde_kernel.py:52-63`) | Accuracy | WS-C2 |
| F7 | v=0 boundary is plain Neumann `U[:,0]=U[:,1]`; inaccurate when Feller violated (`heston/pde_kernel.py:93,172`) | Accuracy | WS-C3 |
| F8 | QE lacks Andersen martingale correction (QE-M): E[S_T] ≠ F bias at coarse steps / high vol-of-vol (`heston/mc_kernel.py:73-128`) | Accuracy (opt-in) | WS-C5 |
| F9 | Calibration runs `quad` per option per residual eval; CF is strike-independent; FD Jacobian multiplies by ~6 (`heston/calibration.py:131-149`) | Performance (largest) | WS-B1 |
| F10 | Python-loop Thomas: N_V solves of size N_S per step, nested Python loops (`heston/pde_kernel.py:97-110,191-222`) | Performance | WS-B2 |
| F11 | Dense mode rebuilds time-independent S-tridiagonals every solve call; only sparse mode caches (`heston/pde_kernel.py:199`) | Performance | WS-B2 |
| F12 | MC pregenerates 3×(paths×steps) arrays (memory at scale); `u_var` drawn even when unused (`heston/mc_kernel.py:172-182`) | Performance | WS-B4 |
| F13 | Lewis/Gatheral/Weber duplicate the CF core algebra; `log(f/k)` recomputed inside integrand (`heston/analytical_kernel.py:63-155`) | Reuse | WS-D4 |
| F14 | Dead `else 0.2` / `else 0.0` fallbacks in calibration target construction read as real defaults (`heston/calibration.py:110,118`) | Clarity | WS-A3 |
| F15 | `NumericalError` from CF pricer at extreme trial params aborts `least_squares` mid-run; no guidance (`heston/calibration.py`) | Robustness | WS-D5 |
| F16 | Leverage clip mismatch: MC binning `sigma_hat² ∈ [1e-8, 10]` (L ≤ 3.16) vs FFP `leverage_clip (0.05, 20)` (`slv/slv_mc_kernel.py:63,71` vs `fokkerplanck/config.py:30`) | **Behavioral** | WS-A2 |
| F17 | Central-difference FP convection → negative density; `tol_neg=0.5` tolerates up to 50% negative mass (`fokkerplanck/fp_operators.py:37-58,61-85`) | Accuracy (highest math value) | WS-C4 |
| F18 | Dirac seed at nearest node → O(h) placement bias for the whole calibration (`fokkerplanck/fp_solver.py:53-60`) | Accuracy | WS-C4 |
| F19 | Backward-Euler FP march is O(dt); TR-BDF2 keeps L-stability at second order (`fokkerplanck/fp_solver.py:76-108`) | Accuracy (opt-in) | WS-C6 |
| F20 | Full `splu` refactorization of the coupled operator every step (`fokkerplanck/fp_solver.py:93`) | Performance | WS-B3 |
| F21 | `bin_conditional` re-sorts per step; O(bins×n) mask loop for means (`slv/leverage.py:27-70`) | Performance | WS-B4 |
| F22 | `_HestonSLVADI` ≈ `_HestonADI` with L≢1; `_thomas`/`_bc`/`_terminal`/`_solve_V`/`interpolate`/`price_delta_gamma` near-verbatim duplicates; 3 Thomas impls in module | Consolidation | WS-D1 |
| F23 | Stale docstring (`slv_mc_kernel.py:5` claims QE variance; code is full-truncation Euler) and dead config (`FpCalibrationConfig.rannacher_steps`, `.scheme`; comment references nonexistent loop) | Doc/dead code | WS-A3 |
| F24 | API parity: SLV MC lacks `use_antithetic`; SLV PDE lacks `grid_spot` pin | Parity | WS-D3 |
| F25 | LV 1D CN solver: no Rannacher damping, uniform S-grid from 0, kinked payoff → gamma oscillation (`localvol/pde_kernel.py`) | Accuracy | WS-C7 |
| F26 | `_thomas_solve` Python loop; `scipy.linalg.solve_banded` is a drop-in (`localvol/pde_kernel.py:19-50`) | Performance | WS-B2 |
| F27 | LV PDE has no `price_delta_gamma` reader (Heston/SLV do) | Parity | WS-D3 |
| F28 | None of the three MC kernels use the shared `quantark/montecarlo` QMC layer (Sobol/BB/RQMC) | Feature (opt-in) | WS-D7 |

---

## 2. Research grounding

- **Heston CF pricing / calibration (Lewis 2000; Albrecher et al. 2007; Kahl-Jäckel 2005).** The `(b−d)/(b+d)` branch already implemented is the stable "little-trap" form. The CF φ(u; T, params) is strike-independent: per maturity, evaluating φ once on a fixed quadrature rule and pricing all strikes vectorized is the standard calibration formulation. Fixed Gauss-Legendre on a truncated/transformed domain (or Gauss-Laguerre) with 64–128 nodes reproduces adaptive `quad` to ~1e-10 for non-pathological parameters; node count is a config, and disagreement with the adaptive reference on the validation set is a gate.
- **ADI for Heston (in 't Hout & Foulon 2010; in 't Hout & Welfert 2007).** Douglas/CS with θ ≥ ½ is unconditionally stable in 2D with mixed terms. The reaction term belongs inside the implicit unidirectional operators (conventionally in A1, or split ½/½) — explicit treatment degrades the CS scheme to first order in `r·dt`. Sinh-stretched grids concentrated at the strike (x) and at v=0/v0 (v) are the reference grid design; the repo already has `concentrated_grid` (Tavella-Randall) in `fokkerplanck/coordinates.py`. At v=0 the PDE degenerates; the reference treatment inserts the degenerate PDE with one-sided differences rather than a Neumann condition.
- **QE simulation (Andersen 2008).** The implemented K0–K4 decomposition with γ1=γ2=½ is Andersen's plain QE. §4.2's martingale-corrected K0* ("QE-M") enforces E[S_{t+Δt}] = forward exactly; the correction is closed-form per branch (quadratic/exponential).
- **Positivity-preserving Fokker-Planck (Chang & Cooper 1970; Scharfetter-Gummel 1969).** Exponential fitting of the face fluxes yields an M-matrix for each directional convection-diffusion operator: positivity-preserving regardless of cell Péclet number, reducing the negative mass to the (small) mixed-term contribution. Reduces smoothly to central differencing as Péclet → 0, so the flat-vol/zero-drift regression stays intact.
- **TR-BDF2 (Bank et al. 1985; used for Dirac-seeded forward PDEs).** L-stable, second-order one-step method; keeps backward-Euler's damping of the singular seed while removing the O(dt) march error.
- **QMC (repo's own `quantark/montecarlo`).** Sobol + Brownian bridge for terminal-payoff kernels is the documented shared-infrastructure intent (CLAUDE.md).

---

## 3. Goals / non-goals

**Goals**

- Make the MC-binning and FFP leverage calibration routes true cross-checks (shared clip convention, shared diagnostics) — F16.
- Fix the Dupire near-zero-variance failure direction — F1.
- Cut Heston calibration wall-clock by ≥10× (target 50×) with a strike-vectorized fixed-quadrature CF pricer validated against the adaptive reference — F9.
- Cut ADI PDE and LV PDE wall-clock ≥10× via batched tridiagonal solves and operator caching, bit-identical results — F10, F11, F26.
- Second-order-in-time ADI (implicit reaction term), concentrated grids, and a positivity-preserving FP discretization, each gated on convergence benchmarks — F5, F6, F17.
- One shared ADI core; one shared Thomas/banded utility; no stale docs or dead config — F22, F23.
- API parity across the three model families (antithetic, `grid_spot`, `price_delta_gamma`) — F24, F27.

**Non-goals**

- No change to product/engine-layer semantics or to `volmodels/risk/`.
- No new smile dynamics, no jumps, no rough-vol.
- No silent default flips: any change that moves prices beyond convergence tolerance ships behind an explicit opt-in or a deliberate, reviewed golden update (repo convention: exact semantics by default; approximations opt-in; benchmark the exact path first).
- No MC inside deterministic engines (standing constraint).
- Numba/Cython/JIT dependencies — vectorized NumPy only.

---

## 4. Workstream A — Correctness & consistency (behavioral)

### WS-A1: Dupire near-zero total variance (F1)

**File:** `localvol/dupire.py`

Replace `inv_w = safe_divide(1.0, w)` with an explicit guard: if any `w <= W_FLOOR` **raise `NumericalError`** naming the offending (T, K) nodes — a total variance that small means the input grid has a degenerate front row and no finite-difference Dupire read-off is meaningful there. No silent fallback (repo convention). Above the floor, plain division.

`W_FLOOR` is an **absolute module constant `1e-12`** on total variance `w = iv²·T` (dimensionless-variance units; deliberately *not* derived from `Tolerance.ZERO = 1e-10`, which would reject legitimate short-dated low-vol rows such as `iv=0.03, T=1e-4 → w=9e-8 ≫ 1e-12` while `1e-12` still catches genuinely degenerate rows). Not maturity- or spot-scaled.

*Acceptance:* boundary unit tests on both sides of the floor — a grid whose front-row `w` is just **below** `1e-12` (e.g. `iv=1e-5, T=1e-3 → w=1e-13`) asserts `NumericalError` naming the node, and one just **above** (e.g. `iv=1e-4, T=0.5 → w=5e-9`) builds successfully; the flat-surface machine-precision round-trip test is unchanged.

### WS-A2: Unified leverage clip convention (F16)

**Files:** `slv/slv_mc_kernel.py`, `slv/fokkerplanck/config.py`, `slv/leverage.py`

- Single source of truth: named constant **`DEFAULT_LEVERAGE_CLIP: Tuple[float, float] = (0.05, 20.0)`** defined in `slv/leverage.py` (the module both routes already import). `FpCalibrationConfig.leverage_clip` defaults to it (no literal duplication); `price_european_slv_mc` and `_calibrate_mc_binning` gain a `leverage_clip=DEFAULT_LEVERAGE_CLIP` parameter threaded to `_simulate_slv`.
- `_simulate_slv` clips `sigma_hat = clip(sigma_lv/sqrt(econd), lo, hi)` — i.e. clip **L**, not L², so the band is directly comparable across routes.
- **Clip scope — calibration-generated leverage only.** The clip applies to on-the-fly binned leverage and to recorded calibration nodes. A user-supplied precomputed `LeverageSurface` (the `leverage_surface=` path in MC, the `lev_surface` argument of the backward PDE) is consumed **as-is** — validated finite and strictly positive by the dataclass, never re-clipped. Rationale: a precomputed surface is a reproducibility artifact; silently re-clipping it would change frozen model risk conventions.
- Recorded leverage nodes use the identical clip (preserving the existing invariant that the recorded surface matches the effective in-simulation leverage).
- MC-binning gains clip diagnostics: `LeverageSurface.diagnostics = {"method": "mc_binning", "n_clipped": …}` (currently `None`), where `n_clipped` is a **scalar total across all recorded (t, S) nodes**, matching the FFP convention.
- The lower bound of the old band (`sigma_hat² ≥ 1e-8` → L ≥ 1e-4) widens to L ≥ 0.05; the upper widens from 3.16 to 20. **This changes MC-binning results in high/low-leverage regimes by design.** Goldens that exercised the saturated region are updated deliberately with a note.

*Acceptance (as amended during implementation — see rationale below):*
1. Unit test (deterministic, no simulation): recording path with a fixture where `σ_LV=0.4` and the binned `E[v|S]=0.01` (wants L=4) — recorded leverage now 4.0, previously 3.162.
2. Cross-route gate (extends `test_slv_calibration_spec_method.py`): Heston `v0=θ=0.04, κ=1.5, σ=0.5, ρ=−0.7`, inline skewed LV fixture, `T=1.0` at 48 steps; MC-binning with `seed=42, num_paths=200_000, num_bins=30` vs FFP defaults; metric = max abs **relative** leverage difference evaluated at 11 interior times over S within **±2 conditional stds per time** (`±2√(v_ref·t)`); threshold **0.10**.
3. High-leverage fixture that previously saturated: same comparison with `v0=θ=0.01, σ=0.2` and flat `σ_LV=0.40` (true L ≈ 4 in the bulk) — passes the same 0.10 gate; the test also asserts `max(leverage_grid) > 3.163` (the old cap would have bound) and `n_clipped == 0`.
4. `n_clipped` present in diagnostics for both routes (scalar).
5. Extreme-regime guard: at the originally-specified `σ=0.8`, both routes are O(dt)-dominated at coarse steps (a single full-truncation Euler shock pins v at 0), so a fixed-step 0.10 gate would measure time-discretization error, not route agreement (measured: 0.55 at 48 steps → 0.26 at 96 → converging; ≈400 steps to reach 0.10). A convergence-trend test asserts the 48→96-step gap ratio < 0.65 instead.

**Amendments rationale (2026-07-05):** (a) the original "±2 total-vol stds" window read ±7 *conditional* stds at t=1/12 — deep tails where the two routes differ *by design* (MC flat-extrapolates tail-bin means; FFP blends to the unconditional CIR mean), so the window is per-time; (b) σ=0.8 vol-of-vol moved to the convergence-trend guard (see 5); (c) fixture 3's stated intent ("true L≈4") requires the variance to remain identifiable near v0 — at σ=0.8 full-truncation dynamics pin v at 0 and L is nowhere near 4, so fixture 3 uses σ=0.2.

### WS-A3: Stale docs, dead config, dead fallbacks (F23, F14)

- `slv/slv_mc_kernel.py` module docstring: replace "The variance follows the QE scheme" with the actual full-truncation-Euler description (the `_simulate_slv` docstring already explains the deliberate choice; hoist it).
- `FpCalibrationConfig`: delete `rannacher_steps` and `scheme` (validated, never consumed; the calibration comment references a `n < rannacher_steps` loop that does not exist — delete the comment too). Frozen dataclass, so removal is API-visible: grep confirms no internal consumers. **Callers passing the removed kwargs get the dataclass's natural `TypeError`** ("unexpected keyword argument") — accepted and documented; no custom `__init__` shim to convert it to `ValidationError` (the repo treats constructor-signature errors as programming errors, and a shim would have to special-case removed names forever). `test_fp_config.py` cases covering the removed fields are updated to assert `TypeError`.
- `heston/calibration.py:110,118`: replace the dead `else 0.2` / `else 0.0` branches with direct attribute access after the existing validation (`opt.iv` / `opt.price` are provably non-None in those branches).

*Acceptance:* grep-clean; `test_fp_config.py` updated; no behavior change (covered by existing suites).

---

## 5. Workstream B — Performance

### WS-B1: Strike-vectorized Heston CF pricing for calibration (F9)

**Files:** `heston/analytical_kernel.py` (new function), `heston/calibration.py`

- New `heston_call_prices_vectorized(s0, strikes, T, params, r, carry, *, n_nodes=96, u_max=None) -> np.ndarray` implementing the Lewis single-integral form on a fixed Gauss-Legendre rule over a transformed domain (`u = tan` map or truncation at an adaptive `u_max` derived from the CF decay rate `|φ(u)| ~ e^{-c·u}`; `u_max` estimated from params, never hardcoded): evaluate φ(u_j) **once**, then price all strikes as a matrix-vector product against `e^{-i u_j y_k}`.
- Existing per-option adaptive-`quad` functions are untouched and remain the reference implementation (and the default for one-off `heston_call_price` calls).
- `calibrate_heston` groups options by maturity and uses the vectorized pricer inside `residuals`; falls back to nothing — if the fixed rule disagrees with the reference beyond tolerance on the validation set, that is a gate failure to fix by raising `n_nodes`, not a runtime fallback.
- The `target="iv"` path benefits automatically (one CF sweep, then per-option Brent inversions — inversions are cheap relative to quad).

*Acceptance:* (i) vectorized vs adaptive reference: max abs price error < 1e-8·s0 across the standard validation grid (moneyness 0.5–2.0, T 0.05–5.0, Feller-satisfied and violated parameter sets, |ρ| up to 0.95); (ii) calibration regression: same fixture converges to the same parameters (component-wise < 1e-6) as the current implementation; (iii) benchmark script records ≥10× calibration speedup (target 50×).

### WS-B2: Batched tridiagonal solves + operator caching (F10, F11, F26)

**Files:** new `volmodels/tridiag.py` (or `util/numerical/tridiag.py` — see WS-D2), `heston/pde_kernel.py`, `slv/slv_pde_kernel.py`, `localvol/pde_kernel.py`

- New `solve_tridiag_batch(sub, diag, sup, rhs)` where inputs are `(n_sys, N)` arrays: forward/backward sweeps as a Python loop over N with vectorized ops across `n_sys` (the standard loop-order swap). Zero-pivot detection preserved (`NumericalError` when any system's pivot magnitude is below the same threshold each current implementation uses).
- New `solve_tridiag(sub, diag, sup, rhs)` single-system wrapper delegating to `scipy.linalg.solve_banded(check_finite=False)` for the 1D LV solver. LAPACK's arithmetic order differs from the sequential Thomas, so the LV path's acceptance gate is **numerical equivalence, not bit-identity**: max relative price difference ≤ 1e-12 across the existing `test_local_vol_pde_kernel.py` fixtures. Pivot semantics defined explicitly: `scipy.linalg.LinAlgError` (singular matrix) is caught and re-raised as `NumericalError` with the existing message intent; the current `denom == 0.0` pre-check is superseded by LAPACK's own singularity detection. (An `n_sys=1` batched Thomas was considered and rejected: it keeps bit-identity but Python-loops over N, forfeiting the F26 speedup.)
- The ADI batch path (`solve_tridiag_batch`) keeps the strict gate: identical sweep order within each system ⇒ bit-identical to the sequential Thomas.
- `_solve_S`: build all N_V systems' coefficients as `(N_V, N_S)` arrays (fully vectorized over V since coefficients are `c2(V_j)`, `c1(V_j)` outer products) → one `solve_tridiag_batch` call. `_solve_V` symmetric.
- Dense-mode coefficient caching: Heston S/V tridiagonal coefficient arrays keyed on `(dt_step, theta_loc)` exactly like the existing sparse `_ensure_S_lus` cache (operators are time-independent). SLV S-coefficients depend on `t_mid` via L and stay per-step, but are built vectorized.
- ADI results must be **bit-identical** to the sequential Thomas within each system (same arithmetic order along the sweep): exact equality on small grids and ≤1e-13 relative on production grids. The LV 1D path uses the relaxed ≤1e-12 gate defined above.

*Acceptance:* (i) equality gates vs current implementation on the full `test_heston_pde_kernel.py` / `test_slv_pde_kernel.py` fixtures (bit-identical/1e-13) and `test_local_vol_pde_kernel.py` (1e-12); (ii) benchmark: ≥10× on the default 200×100×100 Heston solve; (iii) `use_sparse` path retained and still consistent.

### WS-B3: Lagged-factorization Krylov option for the FP march (F20)

**File:** `slv/fokkerplanck/fp_solver.py`, `config.py`

- Opt-in `FpCalibrationConfig.linear_solver: {"direct", "krylov_lagged"} = "direct"`. Krylov mode: BiCGStab on `(I − dt·A) f_new = f`, preconditioned by the `splu` factorization refreshed every `refactor_every` steps (config, default 5) or on convergence failure (which triggers an immediate refactor + retry once; a second failure raises `NumericalError` — no silent degradation).
- Direct mode stays the default until the acceptance benchmark demonstrates identical calibration output (mass/negativity diagnostics and leverage surface within 1e-10) and a measured speedup; flipping the default is a separate reviewed change.

*Acceptance:* leverage surface from Krylov mode matches direct mode < 1e-10 max abs on the standard FFP fixtures (including |ρ|=0.9); calibration wall-clock benchmark recorded.

### WS-B4: MC kernel micro-performance (F12, F21)

- `heston/mc_kernel.py`: draw `u_var` only for QUADEXP; document that the z-stream is unchanged for EULER/EULERLOG (u draws happen after z draws — verify and pin with a seed-stability test). Optional `chunk_paths` is **not** added now (European kernels; M is small); instead a docstring note warns about the `3·paths·M` memory footprint.
- `slv/leverage.py::bin_conditional`: single `argsort`, boundaries by index arithmetic (already), bin means via `np.add.reduceat` over the sorted variance array with the existing empty-bin neighbor-fill preserved. **Tie convention pinned to the current mask semantics:** bin 0 is `[b_0, b_1]` (both edges inclusive), bin k>0 is `(b_k, b_{k+1}]` — i.e. a sample exactly equal to an interior boundary belongs to the **left** bin. On the sorted array this is exactly `split_k = np.searchsorted(s_sorted, boundaries[k], side="right")` for interior boundaries `k = 1..num_bins−1`, with segments `[split_k, split_{k+1})`; `side="right"` places boundary-tied samples below the split, i.e. in the left bin, for **both** bin methods (equal-weighted boundaries are sample values, so boundary ties are common there and this is the case that matters).

*Acceptance:* exact-output equality tests vs the current mask implementation on random clouds (both bin methods) **plus** dedicated boundary-tie cases: clouds with repeated S values placed exactly on equidistant boundaries, and equal-weighted clouds with heavy ties (e.g. quantized spots) so multiple samples equal `boundaries[k]`; empty-bin neighbor-fill cases; seed-stability test for the Heston MC schemes (u_var drawn after z draws ⇒ z-stream unchanged when u_var becomes QUADEXP-only — asserted by comparing EULERLOG prices before/after).

---

## 6. Workstream C — Numerical accuracy

Repo convention applied throughout: benchmark first, then either (a) the change is a pure accuracy upgrade at unchanged cost → default flip with deliberate golden update, or (b) it changes cost/semantics trade-offs → opt-in parameter.

### WS-C1: Implicit reaction term in ADI (F5)

**Files:** `heston/pde_kernel.py`, `slv/slv_pde_kernel.py` (via the WS-D1 shared core)

Fold `−rU` into A1 (in 't Hout–Foulon convention): `_A1 += −r·U` on interior nodes; `_tri_S` diagonal gains `+θ_loc·dt·r`; the predictor's explicit `− dt·r·U` term is removed (it is now inside A1's explicit application). Douglas and CS correctors then treat the reaction implicitly in the S-direction. This restores formal O(dt²) for CS and improves stability margin.

*Acceptance:* (i) convergence study: CS error vs `n_t` on the analytical-kernel reference shows second-order slope (currently first-order-contaminated for r=5%); (ii) prices move only within discretization tolerance at default resolutions — goldens updated deliberately; (iii) cross-family gate (analytical vs PDE vs MC) stays green.

### WS-C2: Concentrated (x, v) grids for ADI (F6)

**Files:** `heston/pde_kernel.py`, `slv/slv_pde_kernel.py`, reusing `fokkerplanck/coordinates.concentrated_grid`

- x-grid: sinh-concentrated around `ln K` (payoff kink), extent logic unchanged. v-grid: sinh-concentrated around `min(v0, θ)` with extent `[0, V_max]`; drop the hardcoded `0.5` floor in `V_max` in favor of a CIR-quantile-based extent (reuse `z_extents` logic on the variance scale, quantile config default `1e-5`), keeping the `max(…, 2·v0, 5θ·guard)` envelope as a lower bound on coverage.
- Non-uniform grids require the general non-uniform second-order stencils — reuse `_fd1`/`_fd2` coefficient logic (WS-D2) for the tridiagonal builders and explicit operators; the mixed term uses the standard non-uniform cross stencil.
- Ship as `grid_style: {"uniform", "concentrated"} = "uniform"` initially; flip the default in a follow-up once the convergence benchmark (equal-node accuracy, equal-accuracy node count) is recorded and goldens updated.

*Acceptance:* equal-accuracy benchmark shows ≥4× node reduction (or equal-node error reduction ≥5×) vs uniform on the Hout-Foulon test set (their four parameter cases) and on the repo's Feller-violated fixture; `grid_spot` pinning still yields clean bump Greeks.

### WS-C3: Degenerate v=0 boundary (F7)

**Files:** shared ADI core (WS-D1)

At `v=0`, replace the Neumann row with the degenerate PDE row: `U_t = (r−q)·U_x·δ + κθ·U_v(one-sided) − r·U` discretized with a second-order one-sided (forward) `U_v` stencil, applied inside the V-direction implicit solve (row becomes a 3-band row with entries at j=0,1,2 → still tridiagonal-compatible using a 2-point first-order forward difference, or handle the 3-point row via the batch solver's first row folded by elimination — implementation picks the 2-point version first, documented as O(dV) local error at one boundary row, and validates whether the 3-point version is needed).
Opt-in `v0_boundary: {"neumann", "degenerate_pde"} = "neumann"` until benchmarked; the Feller-violated fixture decides the default flip.

*Acceptance:* Feller-violated case (2κθ ≪ σ²) vs analytical kernel: PDE error at default resolution reduced measurably (record before/after); Feller-satisfied cases unchanged within tolerance.

### WS-C4: Positivity-preserving FP fluxes + seed splitting (F17, F18)

**Files:** `fokkerplanck/fp_operators.py`, `fp_solver.py`, `config.py`

- **Chang–Cooper / exponential fitting** on the x and z directional face fluxes: face weight `δ = 1/P − 1/(e^P − 1)` with local Péclet `P = μ_f·h/D_f` (guarded `expm1` form, exact central limit as P→0). Mass conservation is preserved (still a face-flux telescoping scheme). Mixed operator unchanged (inherently sign-indefinite, but its negativity contribution is the small one).
- **Dirac seed bilinear splitting**: distribute unit mass over the ≤4 nodes surrounding `(ln s0, ln v0)` with bilinear weights divided by quadrature weights (preserves total mass exactly and the seed's mean location to O(h²)).
- Config: `flux_scheme: {"central", "chang_cooper"} = "central"` initially; after the acceptance benchmark, flip default and **tighten `tol_neg` from 0.5 to 0.05** (the documented purpose of the change).

*Acceptance:* (i) on the standard FFP fixtures (incl. |ρ|=0.9), max negative mass drops ≥10× vs central; (ii) flat-LV + η=1 SLV reprices Heston within the existing cross-family tolerance (Chang–Cooper reduces to central in the resolved-Péclet limit, so well-resolved results move within discretization tolerance only); (iii) mass conservation ≤ existing `mass_tol`; (iv) `test_fp_acceptance.py` gates green in both modes during the transition.

### WS-C5: QE-M martingale correction (F8)

**File:** `heston/mc_kernel.py`, `util/enum/engine_enums.py`

New enum member `HestonMCScheme.QUADEXP_M`: Andersen §4.2 K0* correction — per-path closed form: for the quadratic branch `K0* = −ln(M_a)/… ` per Andersen's Prop. 4.1 (exact E[e^{corr}] normalization using the known MGF of `a(b+Z)²` and of the exponential/Bernoulli mixture), replacing the drift-consistency term so that `E[S_{t+Δt} | F_t] = S_t·e^{drift·Δt}` exactly. Existing `QUADEXP` untouched (it is the documented cross-check baseline).

*Acceptance:* martingale test: `E[S_T]·DF_carry/fwd − 1` at 8 steps/year, σ=1.0, ρ=−0.9 — QUADEXP_M ≤ 3 stderr of 0 where QUADEXP shows measurable bias; European prices agree with the analytical kernel within MC error.

### WS-C6: TR-BDF2 FP march (F19)

**File:** `fokkerplanck/fp_solver.py`, `config.py`

Opt-in `time_scheme: {"backward_euler", "tr_bdf2"} = "backward_euler"`. TR-BDF2 with γ=2−√2: trapezoidal substep then BDF2 substep, both against the same per-step operator (L frozen at the step as today — the leverage read-off order stays unchanged); two factorizations per step in direct mode (or two Krylov solves in WS-B3 mode). L-stable → keeps Dirac damping.

*Acceptance:* time-refinement study on the standard fixture: leverage-surface error vs a fine-dt reference shows second-order slope; first backward-Euler step retained as start-up (Rannacher-style) to damp the seed; negativity/mass diagnostics no worse than backward Euler.

### WS-C7: LV PDE Rannacher + strike-aware grid (F25)

**File:** `localvol/pde_kernel.py`

- `rannacher: bool = True` parameter (default on — matches Heston/SLV ADI convention): first step replaced by two implicit half-steps (θ_loc=1).
- Strike placed mid-cell: shift the uniform grid by up to `ds/2` so K falls halfway between nodes (standard kink-averaging; s0 read-off already interpolates). Behind the same change since both alter node placement semantics; goldens updated deliberately.
- Optional follow-up (not this program): sinh concentration around K.

*Acceptance:* gamma profile near strike (via dense read-off) free of CN oscillation on the short-dated fixture; price convergence order unchanged or improved; cross-check vs LV MC gate green.

---

## 7. Workstream D — Consolidation, parity, API

### WS-D1: One ADI core (F22)

**Files:** new `volmodels/adi_core.py`; `heston/pde_kernel.py` and `slv/slv_pde_kernel.py` become thin wrappers

`_HestonADI` ≡ `_HestonSLVADI` with `L ≡ 1, η = 1`. Extract a single `HestonSLVADICore` parameterized by an optional leverage provider `L(t) -> (nx_int,)` (constant-1 default) and `sig_eff = η·σ`:

- Shared verbatim members: `_thomas` (→ WS-B2 batch util), `_bc`, `_terminal`, `_s_boundary_rhs`, `_solve_V`, `_A2`, `interpolate`, `price_delta_gamma`, `solve`.
- `_A1`/`_A0`/`_solve_S` take the leverage row (Heston passes ones — the compiler-visible special case keeps the Heston coefficient caching from WS-B2, keyed on `(dt_step, theta_loc)` only when L is constant).
- Heston-only features preserved: `grid_spot` pinning, sparse-LU path, `_SIGMA_MIN` deterministic branch. SLV gains `grid_spot` for free (F24).
- Public entry points (`price_european_heston_pde`, `price_european_slv_pde`, …) keep their exact signatures and semantics.

*Acceptance:* full existing PDE test suites pass unchanged; line count of the two kernel files drops ~200+; a parameter sweep asserts Heston-via-core ≡ SLV-via-core with `LeverageSurface ≡ 1, η=1` to 1e-13.

### WS-D2: Shared numerical utilities (F4, part of F22)

- Move `_fd1`/`_fd2` to `quantark/util/numerical/finite_difference.py` (public `fd1_nonuniform`, `fd2_nonuniform`; Dupire imports them; docstrings and exactness-for-quadratics tests move along).
- Tridiagonal utilities land in `quantark/util/numerical/tridiag.py` (WS-B2) — one implementation for the whole repo; the equity-engine PDE solvers are *not* migrated in this program (separate consolidation, already tracked) but the utility is placed where they can adopt it.

### WS-D3: Surface/API parity (F2, F24, F27)

- `LocalVolSurface`: add `interp: {"linear_s", "linear_logs"} = "linear_s"` — log-strike bilinear as opt-in, matching `LeverageSurface`; default flip only after the cross-family agreement gate is re-benchmarked under both (behavior change, goldens). Time interpolation stays linear-in-vol in this program; linear-in-variance is noted as a follow-up with its own benchmark (it interacts with the calendar-arbitrage guarantees).
- `price_european_slv_mc` / `_simulate_slv`: `use_antithetic` with pair-averaged stderr, mirroring the Heston MC implementation exactly (same half/concat/1−u convention for any future uniform draws — currently z-only).
- `localvol/pde_kernel.py`: new `price_delta_gamma_european_lv_pde(...)` reading delta/gamma off the final grid with `np.gradient(edge_order=2)`, mirroring the ADI readers; the price read-off switches from `np.interp` to the same v-curve convention for internal consistency (F-note from review: `interpolate` vs `price_delta_gamma` read-off mismatch in ADI is also normalized in WS-D1 — both use the x-grid bilinear).

### WS-D4: Shared CF core (F13)

`heston/analytical_kernel.py`: extract `_cf_core(u_complex, b0, T, params) -> (A, B)` computing the shared `d, g, e^{−dT}, log((1−g·e^{−dT})/(1−g))` algebra once; Lewis/Gatheral/Weber express their integrands through it. Hoist `log(f/k)` out of integrands. Pure refactor: bit-identical integrand values asserted in tests.

### WS-D5: Calibration robustness & docs (F15, F3)

- `calibrate_heston`: validate `lower ≤ x0 ≤ upper` up front with a clear `ValidationError` (currently a raw scipy error). Document in the docstring that `NumericalError` from the CF pricer at extreme trial parameters propagates (by design — no fallback residuals) and that the remedy is tighter bounds; the WS-B1 fixed-quadrature pricer with the no-arb clamp makes this substantially rarer (quad tail failures were the main source).
- `dupire.py` (F3): edge-row arbitrage rejections get an explicit error-message hint ("boundary node — one-sided stencil; consider extending the input grid one strike/maturity") and the validation tolerance on **edge rows only** uses `10×Tolerance.PRECISION`. Interior rows unchanged (no loosening of the real check).

### WS-D6 *(folded into D5 — kept as its own ID for traceability of F3)*

### WS-D7: Opt-in QMC for the MC kernels (F28)

New optional `sampler` parameter on `price_european_lv_mc`, `price_european_heston_mc`, `price_european_slv_mc` accepting the shared `quantark/montecarlo` Sobol/RQMC generator (normal + uniform streams; QE consumes uniforms for the variance inverse-CDF, which is the QMC-friendly formulation Andersen designed for). Default remains `default_rng(seed)` — pseudo-MC is the documented baseline. Dimension layout documented (variance draws first per step, then spot, matching current column order so BB reordering is possible later).

*Acceptance:* RMSE-vs-paths study on the European fixtures shows the expected QMC convergence improvement; pseudo-MC path bit-identical to today.

---

## 8. Phasing & dependencies

| Phase | Content | Depends on | Risk |
|---|---|---|---|
| **0** | WS-A1, WS-A2, WS-A3 (correctness + hygiene) | — | Low; A2 changes MC-binning goldens deliberately |
| **1** | WS-B2 (batched tridiag + caching), WS-D2 (utils), WS-B4 | — | Low (equality-gated) |
| **2** | WS-B1 (vectorized calibration), WS-D4 (CF core), WS-D5 | D4 before B1 (shared core) | Medium (quadrature validation gate) |
| **3** | WS-D1 (ADI unification), WS-D3 (parity), WS-C1 (implicit −rU) | Phase 1 | Medium (goldens move within tolerance for C1) |
| **4** | WS-C4 (Chang–Cooper + seed), WS-B3 (Krylov), WS-C6 (TR-BDF2) | Phase 0 | Medium-high (FP acceptance gates in both modes) |
| **5** | WS-C2 (concentrated grids), WS-C3 (v=0 boundary), WS-C5 (QE-M), WS-C7 (LV Rannacher), WS-D7 (QMC) | Phase 3 | Medium (each behind opt-in until benchmarked) |

Each phase is a separate worktree branch with the standard three-tier review gate (Codex → ZenMux GPT-5.5 → reviewer subagent), merged independently. Default flips (C1 immediately; C2/C3/C4-tol_neg/C7 after benchmarks) are explicit commits with golden updates and a benchmark artifact in `quantark/asset/equity/engine/validation/report/` style.

---

## 9. Validation strategy (cross-cutting)

1. **Equality gates** (Phases 1, WS-D4): new code paths must reproduce old outputs to 1e-13/bit-identical before any accuracy change lands on top.
2. **Reference gates** (WS-B1, WS-C5): fixed quadrature vs adaptive quad; QE-M martingale test vs analytical forward.
3. **Cross-family gates** (existing house pattern): analytical ↔ PDE ↔ MC European agreement for Heston; LV MC ↔ LV PDE ↔ input-IV round-trip for Dupire; SLV(flat LV, η=1) ↔ Heston; SLV(η→0) ↔ LV. All must stay green at every phase boundary.
4. **Convergence-order studies** (WS-C1, C4, C6, C7): recorded scripts (validation/report convention), slope assertions in tests where stable.
5. **Benchmarks**: wall-clock before/after per workstream, recorded in the phase's design-doc addendum; the F9 (≥10×) and F10 (≥10×) targets are acceptance criteria, not aspirations.

---

## 10. Open questions (to resolve at phase kickoff, not blockers)

1. **WS-A2 golden policy:** are any downstream engine goldens (equity/FX SLV engines) sensitive to the widened clip band? Survey `test_heston_slv_*_engines.py` fixtures at Phase 0 kickoff; if none saturate, the change is invisible outside the new unit tests.
2. **WS-C2 default flip timing:** concentrated grids change every PDE golden slightly; batch the flip with WS-C1's golden update to avoid two churn events, or keep separate for bisectability? (Recommendation: separate — bisectability wins.)
3. **WS-D3 log-strike default:** flip `LocalVolSurface` to `linear_logs` after re-benchmarking, or keep `linear_s` indefinitely for continuity? (Recommendation: flip, with the cross-family gate as arbiter — consistency with `LeverageSurface` reduces long-term surprise.)
4. **`tol_neg` tightening magnitude** after Chang–Cooper: 0.05 proposed; set from the measured |ρ|=0.9 fixture, not a priori.
