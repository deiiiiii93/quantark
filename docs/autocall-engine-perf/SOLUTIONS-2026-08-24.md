# Autocallable engine perf — confirmation and fix candidates (decision matrix)

**Date:** 2026-08-24. **Follows:** `FINDINGS-2026-08-24.md`. **Code:** repo source
at 0.4.7rc2 (identical to the wheel the findings measured). **Host:** same arm64
machine; every baseline/candidate pair ran back-to-back in one process. Demo
scripts: `demos/demo_pde_cache_key.py`, `demos/demo_quad_event_stats.py` —
runtime-patched subclasses only, **no engine code changed**.

## Confirmation

Both issues reproduce on repo source with the mechanisms the findings named.

**Issue 1 (PDE).** cProfile of one 1.9y monthly-KO snowball solve (standard
accuracy): `_observation_cache_key` → `_product_cache_token` →
`SnowballOption.cache_key()` + `_freeze_cache_value` = 0.835 s of the 1.001 s
stepping loop (~83%), 1,890 calls (one per step via `_set_boundary_conditions_v0`
→ `_get_ko_payoff_at_time`), 1.58 M `_freeze_cache_value` frames. Root cause:
the OrderedDict *lookup* is O(1) but the *key build* re-serializes the whole
product (every KO/KI schedule record) each call; default
`cache_strategy="standard"` takes the `cache_key()` path
(`base_pde_solver.py:742`). A daily discrete-KI schedule (~480
`ObservationRecord`s — what real book rows carry) reproduces the findings'
magnitude: 5.96 s/solve, matching the reported ~5.5 s.

**Issue 2 (QUAD).** `price_with_events()` = 7.8× `price()` (293 ms vs 38 ms) on
the same snowball; PDE control ~parity. Cost decomposition (cProfile): **not the
FFT** (`_raw_fft` 0.085 s). 80% sits in `_diffuse_with_bridge`
(`snowball_quad_engine.py:1508`): a Python loop over ~2·band+1 ≈ 127 offsets
rebuilding identical `np.where/clip/exp` kernels per offset per call. Per
`calculate_event_stats` call the engine walks the time loop **three times** —
stacked KO-indicator recursion (2 stacks × n_ko rows), a separate KI-probability
recursion (`v_*_ki` / `v_*_ever`), and an internal `self.price()` for `stats.pv`
— rebuilding the same kernels in each. `streams` is dropped on the floor: the
base `price_with_events` (`base_engine.py:151`) never forwards it, and only the
PDE solver overrides with a streams-aware single-pass path
(`snowball_pde_solver.py:734`).

Additional structural defect found while confirming: KO indicator row *i* is
identically zero until the backward sweep crosses observation *i*, yet all rows
are diffused (FFT + bridge) at every step. Diffusion, tail correction, and the
bridge correction are all linear, so zero rows produce exact zeros — the work is
provably wasted.

## Candidates measured (all bitwise-identical to baseline)

### Issue 1 — `demos/demo_pde_cache_key.py`

| Arm | What | Light case (cont. KI) | Book-like case (daily KI schedule) | Bitwise (PV + all greeks) |
|---|---|---|---|---|
| baseline | — | 1.20 s/solve | 5.96 s/solve | — |
| **A1** | solver-level memo of `_product_cache_token` keyed `(id(product), strategy)` | **1.29×** | **5.29×** (→1.13 s) | OK |
| A2 | memo `SnowballOption.cache_key()` on the product | 1.09× | 1.42× | OK |

- A1 covers both the serialization *and* the deep-freeze; A2 only the former
  (the freeze is roughly half the tax). A1 also serves every other key builder
  on the solver (grid cache etc.) and, via inheritance, Phoenix/KO-reset.
- Engine edit for A1 must clear the memo at each public solve entry
  (`price` / `price_with_events` / `calculate_event_stats`) so a GC'd product
  cannot alias a reused `id()` across solves. Within one solve the caller holds
  the product, so the id is stable — exactness by construction, same precedent
  as the existing `"strict"` strategy.
- A2 additionally risks staleness if a product is mutated after first pricing
  (products are documented mutable + re-`validate()`); not recommended as-is.
- Findings' fix candidate 1 (hoist records resolution out of the loop) was not
  separately demoed: A1 makes the remaining key-build cost 2 dict hits per
  solve, so the hoist's incremental value is ~zero and it touches more
  signatures.

**Book projection (87 autocallables, 7 cells):** PDE full-book 540 s → ~100 s,
restoring the expected PDE < MC (230 s) ordering. Remaining per-solve cost is
grid build + tridiagonal algebra (~1.1 s on the book-like case).

### Issue 2 — `demos/demo_quad_event_stats.py`

Snowball (23 KO obs, continuous KI, grid 1001):

| Arm | price() | price_with_events() | pwe speedup | pwe/price | Bitwise (price, npv, all stats fields) |
|---|---|---|---|---|---|
| baseline | 37.8 ms | 293.5 ms | 1.00× | 7.8× | — |
| B1 zero-row skip | 37.4 ms | 184.6 ms | 1.59× | 4.9× | OK |
| B2 kernel cache | 20.0 ms | 192.9 ms | 1.52× | 9.7× | OK |
| **B1+B2** | 20.3 ms | **87.8 ms** | **3.34×** | 4.3× | OK |

Phoenix (23 obs, coupon 85 / KI 75):

| Arm | price() | price_with_events() | pwe speedup | Bitwise |
|---|---|---|---|---|
| baseline | 472 ms | 887 ms | 1.00× | — |
| **B1+B2** | 169 ms | **263 ms** | **3.38×** | OK |

- **B1 (zero-row skip):** in `_diffuse_fft` (2D path) and on the bridge's
  `delta`, reduce to rows with any nonzero before diffusing, scatter back
  exact zeros. Bitwise because pocketfft transforms rows independently and all
  three operators are linear with non-negative kernels (0·k = +0.0).
- **B2 (bridge-kernel cache):** cache the per-offset `omega·w·p_hit` arrays
  keyed on `(grid.size, h, grid[0], band, tau_step, alpha, vol²·dt,
  log_barrier, is_reverse)`. All three per-call walks (stats, KI loop, internal
  price) hit the same keys. Memory observed: 7 entries / 6.8 MB on a flat env;
  a term-structure env holds ~one entry per step — the engine edit should clear
  the cache per public solve entry (also required for correctness across
  engine-param changes it does not key on: `num_std_devs` feeds `band`, which
  IS in the key, so params are covered; the clear is for memory bounding).
- B2 alone also speeds plain `price()` 1.9–3.1× (the price pass has its own
  bridge walk) — it helps QUAD rows even outside the greek runner.

**Book projection:** QUAD full-book 924 s → ~275 s. Near MC (230 s) but not
below it; the remaining 4.3× pwe/price is real work (the stacked rows), for
which the structural options below exist.

## Not fixed by the above / follow-up options (need a decision, none demoed)

1. **QUAD streams contract (findings' fix 1):** mirror the PDE override —
   `price_with_events(streams=...)` → `calculate_event_stats(..., streams=)`,
   skip the KI-probability recursion when no leg requests
   KI/MATURITY_WITH_KI, skip Phoenix coupon rows when COUPON is unrequested,
   and reuse the internal price() as npv is already done (base impl uses
   `stats.pv`). Exact for the requested streams; saving depends on the book's
   leg mix (typical `TRUNCATE_AT_KO` funding legs still need the full KO
   stream, so the n_ko stacked rows stay).
2. **Fuse the three walks:** carry the KI-probability surfaces and the value
   surfaces as extra rows of the one stacked recursion. Removes two of the
   three per-call walks; with B2 in place, mostly saves the FFT/setup of the
   extra walks. Moderate invasiveness, bitwise achievable in principle
   (row-independent batching), needs care around `_diffuse_with_bridge`'s
   two-argument contract.
3. **Forward-density event stats:** replace the n_ko backward indicator rows
   with one forward density pair (KI-in/out), reading KO mass off each
   observation — O(rows) → O(1). This is the structural fix that would put
   QUAD clearly under MC, but it is a different (equally exact-in-the-limit)
   discretization: NOT bitwise, needs a modelvalidation-style gate vs current
   stats + MC. Larger effort.
4. **Reusing the base event distribution across bump cells (findings' fix 2):
   rejected as a default.** Leg PVs are valued off the *bumped* distribution
   per cell; freezing the base distribution zeroes the legs' sensitivity to
   KO-probability shifts and changes greeks. Opt-in approximation at most.

## Decision matrix

| # | Candidate | Speedup (measured) | Numerical impact | Effort / risk | Recommendation |
|---|---|---|---|---|---|
| A1 | PDE solver token memo | 5.3× book-like solve; book 540→~100 s | none (bitwise) | small; clear-per-solve discipline | **take** |
| A2 | product cache_key memo | 1.4× | none (bitwise) | mutation-staleness hazard | skip (A1 subsumes) |
| B1 | QUAD zero-row skip | ⊂ 3.35× combined | none (bitwise) | small | **take** |
| B2 | QUAD bridge-kernel cache | ⊂ 3.35× combined; +2-3× plain price | none (bitwise) | small; bounded-memory clear | **take** |
| F1 | QUAD streams contract | book-dependent | none (contract-exact) | medium | next, needs leg-mix data |
| F2 | fuse 3 walks | ≤ ~1.5× residual | none if done right | medium | after F1 |
| F3 | forward-density stats | potentially ≥5× residual | different discretization | large + validation gate | separate program |
| F4 | freeze base distribution across cells | ~7× per row | **changes greeks** | — | reject as default |

## Landed 2026-08-24 (same day, after user selection)

User selected **A1 + B1+B2 + F1**; all three are now engine code:

- **A1** — `SnowballPDESolver._product_cache_token` memo keyed
  `(id(product), strategy)`, cleared at `_price_with_solution` and
  `_compute_event_stats` entry (every solve on the hierarchy passes one of
  them; "strict" stays unmemoized). Covers Phoenix/KO-reset by inheritance.
- **B1** — zero-row skip in `SnowballQuadEngine._diffuse_fft` (2D path) and on
  the bridge `delta` rows.
- **B2** — `SnowballQuadEngine._bridge_kernels` cache on the engine instance,
  keyed `(grid.size, h, grid[0], band, tau_step, alpha, vol²·dt, log_barrier,
  is_reverse)`; clear-all bound at 64 entries
  (`_BRIDGE_KERNEL_CACHE_MAX_ENTRIES`), so a term-structure env degrades
  gracefully to uncached cost instead of growing.
- **F1** — QUAD streams contract mirroring PDE §11.1:
  `SnowballQuadEngine.price_with_events` override forwards `streams` →
  `calculate_event_stats(..., streams=)` → `_compute_event_stats` prunes the
  KI-probability recursion (`want_ki`) and the Phoenix coupon rows
  (`want_coupon`). KO-reset accepts and ignores `streams` (pruning is a
  permission, not an obligation). Measured on top of B1+B2 with KO-only
  streams: another 1.24× (snowball) / 1.30× (phoenix) on the stats pass;
  `pv`/`ko_probability`/`survival` bit-identical, KI/coupon fields pruned.

Post-edit verification:

- The pre-edit bitwise golden capture (PDE PV + all greeks on both cases;
  QUAD/Phoenix price, pwe npv, every stats field) passes `--check` against
  the edited engines — the default paths are bit-identical.
- Post-fix clean re-measurement (quiet machine): PDE book-like solve
  5.96 s → **1.15 s** (5.2×); QUAD snowball `price_with_events`
  293.5 → **88.8 ms** (3.3×), Phoenix 887.4 → **276.3 ms** (3.2×), QUAD plain
  `price()` 37.8 → 20.6 ms / 472 → 157 ms. The demo scripts now measure their
  patch arms against the fixed engine and read ~1.00× — the runtime patches
  have nothing left to save.
- Full pytest suite (`--deselect
  test/mo_volmodels/test_pde_convergence_gate.py::test_quick_end_to_end`, the
  known history-gated 4h+ gate): **6,534 passed**, 21 skipped, 3 failed — all
  3 in `test/mo_volmodels/`, failing on the content of the untracked local
  artifact `output/pde_convergence_gate/gate_decision.json` ("variant 'heston'
  does not delegate ADI Greek admission to Stage 16"), a pre-existing
  volmodel-backtest gate state unrelated to these engines.

## Full-book A/B through the adapter (2026-08-24, after landing)

Setup: two detached worktrees at HEAD — `wt-base` pristine, `wt-fixed` = HEAD +
only the four engine files — run through the adapter's own venv
(`otc-price-adapter/.venv`, quantark 0.4.7rc2 wheel PYTHONPATH-shadowed by each
tree), `otc_quantark_pricer_v047.py --as-of-date 2026-06-30 --workers 8`,
97-row book, arms back-to-back in one window. Isolating on HEAD keeps the
unrelated signed-div-yield WIP out of both arms.

| Engine | base (HEAD) | fixed | Speedup | Values (19 PV/greek cols × 97 rows) |
|---|---|---|---|---|
| PDE | 545 s | **111 s** | **4.9×** | all 1,843 cells exactly equal |
| QUAD | 934 s (re-run; 957 s first window; findings 924 s) | **569 s** | **1.64×** | all 1,843 cells exactly equal |

- Status identical both arms (95 ok / 2 error — the 2 errors are
  arm-independent, pre-existing). Fixed-arm results are bit-stable across
  repeated runs.
- **PDE < MC restored**: 111 s vs the findings' MC 230 s.
- QUAD's book-level gain (1.64×) is smaller than the synthetic demo's 3.3×:
  real rows spend a larger share of cell time outside the stats recursion
  (trade-state build, env setup, shorter/mixed schedules; single-row probes
  showed 1.3–1.7×). QUAD remains above MC; F2 (fuse the three walks) / F3
  (forward-density stats) are the levers if QUAD must get under it.
- Timing hazard note: the first fixed-QUAD window read 5,519 s — external
  contention on this shared host (single-row probes were FASTER on the fixed
  tree, RSS/swap flat, baseline stable). The interleaved re-run with a
  resource sampler gave the 569 s above. Lesson re-confirmed: re-run anomalous
  arms in a fresh window before believing them; value equality was unaffected
  (deterministic engines).

## F2 / F3 delivered (2026-08-24, follow-up program)

Spec `docs/superpowers/specs/2026-08-24-quad-forward-density-event-stats-design.md`;
evidence `docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`;
battery `test/test_quad_forward_density_stats.py` (21 tests).

- **F2 (Track 1, fuse the walks)** — the KI-probability recursion's four
  surfaces now ride the main stacked time loop (single loop, shared step
  kernels), **bitwise** (`capture_stats_goldens.py --check` hex-exact).
  Finding: numpy's complex multiply is not bitwise-commutative, and the 1D
  convolution path computes `omega_fft * u_fft` while the batched path
  computes `u_fft * omega_fft` — so the fused surfaces must keep their 1D
  diffusion calls; batching them into the 2D stack can never be
  bit-identical. Consequence: the fusion is structural (one loop, one event
  pass) and wall-time-neutral (demo pwe 89.6 ms snowball / 271 ms phoenix,
  within noise of pre-fusion).
- **F3 (Track 2, forward-density mode)** — opt-in
  `QuadParams.event_stats_mode="forward_density"`: 2–3 forward density
  surfaces replace the ~n_ko-row stacked recursion; `npv` stays the backward
  `price()` (hex-equality asserted). Validated: analytic density identities,
  closed-form first-passage (1.1e-5 @2001), 500k-QMC MC cross-check,
  full product matrix vs stacked, mass diagnostic. Default is still
  `"stacked"`; the flip is a later decision on the banked evidence.
- Modelvalidation note: `event_stats_mode` is excluded from the QUAD candidate
  identity hash (`_QUAD_NON_NUMERIC`) — certificates bank `price()`-derived
  quantities only, which both modes share by construction.

## Verification status

- All demo numbers: best-of-3, one process, arms interleaved with fresh engine
  instances; bitwise gates = `float.hex()` / `ndarray.tobytes()` equality on
  PV, every greek, npv, and every `AutocallableEventStats` field.
- Full-book A/B for the forward-density mode: see
  `FORWARD-DENSITY-EVIDENCE-2026-08.md` (appended by the Task 11 run).
