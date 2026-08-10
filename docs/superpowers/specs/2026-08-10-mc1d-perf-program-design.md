# 1D MC Performance & Estimator Program — Design

**Date:** 2026-08-10
**Status:** Draft — Phase-1 candidates measured and bitwise-proven by standalone demos; ready for review
**Scope:** `quantark/asset/equity/engine/mc/` (BSM 1D engines), `quantark/asset/equity/process/bsm/qmc_path_generator.py`, `quantark/volmodels/localvol/surface.py`, new `quantark/montecarlo/gbm_kernels.py`
**Origin:** 2D engine-perf program migration (evidence: `docs/mc1d-perf/DECISION-2026-08-10.md`) + literature survey (`docs/mc1d-perf/LITERATURE-2026-08-10.md`)
**Out of scope (deferred to their own specs):** MLMC estimator, adjoint/Vibrato Greeks, Philox parallel substreams, `MultiAssetGBMPathGenerator` kernel extension, LV step-loop Numba fusion, phoenix / KO-reset one-step survival

---

## 1. Problem

Profiling (`docs/mc1d-perf/prof_baseline.py`, this host, 2026-08-10) put the
1D MC engines' time in three places:

| Run | Dominant cost |
|---|---|
| Euro MC pseudo 200k×252 | path build (exp + cumprod + temporaries) ≈ 72% |
| Snowball MC 100k×252 continuous-KI | bridge-KI check 59%, path build 38% |
| LV euro kernel 100k×252 | `LocalVolSurface.local_vol` **82%** |

Separately, autocallable MC Greeks are noise-limited: bump-revaluation
differentiates a KO **indicator**, so delta/gamma near barriers carry Bernoulli
variance that path count buys down only as 1/√n. The literature's answer
(one-step survival conditioning) removes that variance class analytically
(published: ~3× on price, >500× on first-order Greeks — Alm & Harrach).

Two candidates are already measured, bitwise-proven, and runtime-patched in
`docs/mc1d-perf/` (no engine code touched yet):

- **C1** LV scalar-`t` lookup fast path: 1.46× LV euro / 1.31× LV barrier end-to-end.
- **C2** Numba-fused GBM path build: 1.33× euro engine end-to-end, reaches every
  BSM 1D engine through the shared `GBMPathGenerator`.

One candidate was disproven and is recorded as skip-on-evidence (bridge-KI
transcendental hoist: 0.81× at 100k paths — memory-bandwidth regression).

## 2. Program contracts

These are project rules restated as gates; every workstream below is bound by
them.

**2.1 Bitwise gate (default-path changes).** Any change on a default path must
be byte-identical: same outputs to `.tobytes()` / `float.hex()` on the demo
sweeps AND a detached-worktree byte-compare (`git worktree add --detach <tmp>
HEAD~1`, run the same probe in both trees, `cmp` the dumps — the 2D program's
protocol). Goldens are not re-frozen.

**2.2 Optional-accelerator contract** (established by `qe_kernels.py` /
`tridiag.py`): a NumPy reference implementation always exists and is the
no-dependency default; the accelerator is opt-in via `pip install
quantark[accel]`; `njit(cache=True, fastmath=False)` — fastmath licenses
reassociation and is forbidden; the live backend is self-reporting
(`*_backend() -> "numba" | "numpy"`) so certification evidence can record what
produced a run; bit-identity of the accelerated path is **asserted in tests,
not assumed** (NumPy 2.x SIMD transcendentals need not equal scalar libm; on
this host they do, twice confirmed).

**2.3 Exact semantics by default.** Estimator changes (WS-4) are opt-in modes
behind an explicit flag; defaults and goldens untouched. No documented
approximations: one-step survival is exact (unbiased), not an approximation,
but it changes the estimator distribution, so it can never silently replace
the indicator estimator.

**2.4 Demo-first.** Unmeasured candidates (WS-3) get a standalone
runtime-patched demo with bitwise + speed evidence and an explicit
success gate before any engine code changes. Failed gates are recorded in
`DECISION-2026-08-10.md`, not silently dropped.

## 3. Workstreams

### WS-1 — Scalar-time fast path in `LocalVolSurface.local_vol` (Phase 1)

**Change.** `local_vol(spot, t)` gains a scalar-`t` branch: compute the time
bracket (`iT`, `wT`) once as Python scalars, gather the two bracketing grid
rows as 1-D views, and interpolate with the SAME arithmetic order as the
general path (strike interpolation per row first, then the time blend —
the 2D program measured that blending time rows before the strike interp is
NOT bitwise, 4.4e-16). Vector-`t` calls route through the existing code
unchanged; the single-time-grid branch is unchanged.

**Invariant** (mirrors `test_leverage_lookup_fastpath.py` from `b98a8d9`):
`local_vol(spots, t)` must equal `local_vol(spots, np.full(n, t))` **exactly**
— broadcasting the scalar must be redundant. Plus: node reproduction, clamp
edges (spot below/above strike grid, t below/above time grid), both interp
modes (`linear_s`, `linear_logs`), scalar-spot-scalar-t float return type.

**Beneficiaries.** `price_european_lv_mc`, `price_barrier_lv_mc`,
`LocalVolMCEngine`, `LocalVolSnowballMCEngine` (`snowball_vol_mc_engines.py:258`
calls `lv.local_vol(spot, t)` per step), FX LV engines via the same surface.

**Proven implementation:** `docs/mc1d-perf/demo_lv_scalar_t.py::local_vol_fast`.

### WS-2 — Shared GBM path-build kernel with optional Numba backend (Phase 1)

**Change.** New module `quantark/montecarlo/gbm_kernels.py` following the
`qe_kernels.py` contract:

- `gbm_path_tail_numpy(dW, drift_dt, vol_vec, s0) -> paths` — verbatim the
  current `generate_paths` tail (allocate, `exp(drift + vol·dW)`, `cumprod`,
  scale by `s0`), so numba-absent environments are byte-stable by
  construction.
- Numba kernel: per-path fused loop `c = c * exp(drift_dt[k] + vol[k]*dW[p,k])`,
  `out[p, k+1] = s0 * c` — exactly `np.cumprod`'s left-to-right fold with the
  `s0` multiply applied AFTER the fold (never seeded into the accumulator: FP
  multiplication is not associative). Eliminates all four (n_paths, n_steps)
  temporaries.
- `gbm_backend() -> "numba" | "numpy"`; dispatcher `gbm_path_tail(...)`.

`GBMPathGenerator.generate_paths` replaces its tail with one call to
`gbm_path_tail`. Normals generation, variance reduction, and Brownian-bridge
increments are untouched. `generate_terminal_values_qmc` is untouched (no
per-step loop there).

**Placement rationale:** `quantark/montecarlo/` next to `qe_kernels.py`; the
`[accel]` extra already carries numba (cherry-picked from `365c555`). The
known pre-existing circular-import hazard (`import quantark.montecarlo` before
`quantark.asset`) is not worsened: `qmc_path_generator` already imports
sibling `quantark.montecarlo`-backed modules at module level.

**Proven implementation:** `docs/mc1d-perf/demo_gbm_numba_fusion.py`
(`_gbm_build_kernel`, `numba_tail`).

**Host caveat (risk R2):** bit-identity of numba scalar `exp` vs NumPy SIMD
`exp` is a per-host empirical fact (holds on this arm64 host, confirmed twice).
The equality test asserts it wherever numba is installed; on a host where it
fails, the accelerator must not be used for reference runs — which the
self-reporting backend makes auditable. This mirrors the QE kernel's existing
posture exactly.

### WS-3 — Fused single-pass path+scan spike for discrete-observation snowballs (Phase 2, gated)

**Hypothesis.** The C3 failure showed the bandwidth regime: at production path
counts, materializing (n_paths, n_steps) matrices is the enemy. A numba kernel
that generates each path AND scans discrete KO/KI observations in one pass
needs no path matrix at all — per path it keeps only the running cumprod
state, first-KO index, discrete-KI flag/min, and the spot at KO/maturity.
Payoff assembly stays in NumPy on the per-path outputs.

**Scope.** Discrete-KO + discrete-KI snowball only. Continuous-KI (bridge)
stays outside the kernel: its `rng.random(idx.size)` draws are data-dependent
and cannot be ported to numba bit-compatibly (established in the C3 analysis).

**Gate.** Standalone demo (`demo_fused_snowball_kernel.py`): byte-identical
`(ko_triggered, first_ko_idx, ki_flags, payoffs, price)` vs the shipped
pipeline on the same draws, AND ≥1.5× end-to-end on a 100k×252 discrete-KI
snowball. Fails → record in `DECISION-2026-08-10.md`, no engine change.
Passes → a follow-up plan wires it as an internal fast path (same optional-
accelerator contract; NumPy pipeline remains the reference).

### WS-4 — One-step survival estimator for `SnowballMCEngine` (Phase 3, opt-in)

**Estimator** (Glasserman & Staum 2001; Alm & Harrach). Daily grid
`t_0 < … < t_N = T`, KO dates `{T_1..T_M} ⊂ {t_j}` with barriers `B_k`.
Per-step BSM transition `S_j = S_{j-1}·exp(m_j + v_j·Z_j)` with
`m_j = (r_j − q_j − σ_j²/2)Δ_j`, `v_j = σ_j√Δ_j`.

Each path carries a survival weight `w` (init 1). At a step ending on KO date
`T_k`, with step-start spot `s`:

- Let `z* = (ln(B_k/s) − m_j)/v_j` and `c = Φ(z*)`.
- Standard (up-KO): survival means `Z < z*`, so `p = c`, KO prob `q = 1 − c`.
  Reverse (down-KO): survival means `Z > z*`, so `p = 1 − c`, `q = c`.
- **KO leg, integrated exactly:** `contribution += w · q · DF(T_k) · pay_k`
  where `pay_k` is the engine's existing per-date KO payoff
  (`_compute_ko_schedule_payoffs`). Phase-3 scope covers KO payoffs that
  depend only on the date (standard coupon/rebate snowballs). Configurations
  whose KO payoff depends on `S_{T_k}` (call participation legs) raise
  `ValidationError` under this estimator until the partial-expectation
  extension (Black-type `E[S·1{S≥B}] = s·e^{(r−q)Δ}·Φ(v − z*)`) ships.
- **Conditioned draw:** with `u ~ U(0,1)`:
  up-KO: `z = Φ⁻¹(u·c)`; down-KO: `z = Φ⁻¹(c + u·(1−c))`. Then
  `S_j = s·exp(m_j + v_j·z)` is on the survival side a.s. Update `w ← w·p`.
- **Underflow guard:** if `p < 1e-300`, set `w = 0` and retire the path (all
  mass already booked to the KO leg). This is an exactness guard, not an
  approximation.
- Non-KO steps draw unconditionally. In OSS mode ALL steps map uniforms
  through `ndtri` (inverse CDF) so pseudo and Sobol layouts share one stream
  design; this differs from the default estimator's ziggurat normals, which
  is immaterial because OSS is a distinct estimator with its own SE.
- KI monitoring (discrete or bridge) runs on the conditioned path unchanged;
  by the tower property the weighted estimator stays unbiased.

Per-path PV: `Σ_k w_{k-1}·q_k·DF(T_k)·pay_k + w_M·DF(T)·maturity_pay(path)`;
price = mean, SE from per-path PVs (pair-averaged under antithetic).

**Config.** `SnowballMCEngine(..., estimator="indicator" | "one_step_survival")`
(enum-backed, default `"indicator"`). Defaults, goldens, and all existing
tests untouched. PSEUDO and QUASI/RQMC supported; KO-reset and phoenix
explicitly rejected with `ValidationError` (deferred).

**Validation gates.**
1. Fixture grid (spot ladder × {standard, reverse} × {discrete-KI,
   continuous-KI, no-KI}): OSS price within 3× joint SE of the indicator
   estimator.
2. Flat-vol discrete-KI configs vs the QUAD oracle within existing tolerance.
3. Measured SE ratio (target: ≥2× on price for near-barrier spots; report
   actuals).
4. Measured bump-delta noise ratio across a spot ladder (literature order is
   10²; report actuals, no hard gate).

### WS-5 — Measured spikes: terminal stratification & QMC orderings (Phase 4, demo-only)

Two half-day experiments on existing scaffolding, `docs/mc1d-perf/` demos,
adopt-only-on-evidence: (a) stratified/LHS terminal coordinate on the
existing bridge construction (Glasserman Ch. 4) for euro/sharkfin/DCN-style
payoffs; (b) Brownian-bridge vs PCA vs natural ordering for snowball QMC —
the discontinuity literature (Wang–Sloan) and our own bridge8 result
(2.14–2.62× where 8× was hoped) both warn orderings can disappoint; measure,
then decide. No engine changes in this workstream.

## 4. Phasing

| Phase | Content | Gate | Deliverable |
|---|---|---|---|
| 1 | WS-1 + WS-2 | bitwise (2.1/2.2) | engine changes + tests, plan `2026-08-10-mc1d-perf-phase1.md` |
| 2 | WS-3 spike | ≥1.5× AND bitwise | demo + DECISION row; wiring plan only if green |
| 3 | WS-4 | validation gates 1–4 | opt-in estimator + tests, own plan after spec approval |
| 4 | WS-5 | evidence only | demos + DECISION rows |

Phases 1–2 carry no estimator risk and can merge independently of 3–4.

## 5. Risks

- **R1 (WS-1):** a future caller passing vector-`t` with identical values gets
  the slow path — acceptable; the invariant test documents scalar-`t` as the
  fast contract.
- **R2 (WS-2):** numba/NumPy `exp` bit-agreement is per-host (see WS-2 caveat);
  CI (x86_64) runs NumPy-only unless numba is installed there — the goldens'
  cross-arch tolerance story is unchanged.
- **R3 (WS-2):** numba `cache=True` writes `__pycache__` `.nbc/.nbi` artifacts;
  they are platform artifacts and must stay untracked (bit us once already —
  `git add -f` on docs dirs sweeps them).
- **R4 (WS-4):** stream design differs from the default estimator by
  construction; any comparison is statistical (gate 1), never bitwise.
- **R5 (WS-4):** payoff-variant coverage — the `ValidationError` guard keeps
  unsupported KO-payoff shapes out rather than approximating them.

## 6. References

Evidence: `docs/mc1d-perf/{DECISION,LITERATURE}-2026-08-10.md` and demos.
Literature: Glasserman & Staum (2001); Alm & Harrach (StableDiffs); Rakhmonov
(IJTAF 2019); Glasserman, *MC Methods in Financial Engineering* (2004), Ch. 4;
Wang & Sloan (Mgmt. Sci. 2013). 2D ancestors: commits `b98a8d9`, `365c555`,
`66f7b0d` on `codex/adi-greek-certification`.
