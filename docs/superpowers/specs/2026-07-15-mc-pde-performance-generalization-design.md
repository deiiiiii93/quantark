# Generalizing the DCN Performance Layer into quant-ark Core

**Date:** 2026-07-15
**Status:** Draft for review
**Baseline:** quant-ark `ed0863f` (event-stats API commit, on top of perf commit `318009e`)

## 1. Goal

Commit `318009e` gave the DCN engines four measured speedups: a byte-budgeted
LRU Sobol draw cache, a thread-parallel batch loop, a once-per-call local-vol
surface build, and an in-place uniform→normal transform. All but the last are
wired to DCN-specific code paths. This design promotes them to first-class
quant-ark infrastructure so every MC and PDE product engine — equity snowball,
phoenix, barrier, asian, sharkfin, accumulator, range-accrual, FX TARF/TARN/
barrier, and future products — benefits with little or no per-engine code.

Chosen approach (of three considered): **narrow-waist infrastructure with
opt-in adoption**. Shared capability moves to the lowest layer all engines
already pass through; engines adopt incrementally; no public API breaks.
A common `BatchedMCEngine` base class (larger refactor) and forcing all
engines through the `qmc_draws` adapter (dependency inversion, draw-alignment
risk) were considered and rejected; the base class remains a possible future
refactor on top of this work.

## 2. Hard constraints

1. **Bit-identity.** With default settings, every engine must produce
   byte-identical results before and after this change. Parallel modes must
   be byte-identical to serial modes. This is proven by golden tests, not
   asserted by review.
2. **Additive public API.** No existing constructor signature, method
   signature, or env-var contract changes meaning. `use_dask=True` keeps
   working. `QUANTARK_DCN_MC_WORKERS` and `QUANTARK_QMC_CACHE_MB` keep
   working.
3. **Library only.** No new vendored wheel; the quant-mini-project deliverable
   stays pinned to wheel 0.2.6 / commit `318009e`. Its frozen evidence chain
   is untouched.
4. **Tree hygiene.** The uncommitted bucketed-greeks example changes in the
   working tree are never staged. The phoenix/snowball PDE solver files are
   clean as of `ed0863f` and may be edited.

## 3. Architecture overview

Four independent units, each shippable and revertable on its own:

| Unit | New home | Who benefits |
|---|---|---|
| 1. Generator-level draw cache | `quantark/montecarlo/qmc_sobol.py` | Every Sobol user, present and future (equity, FX, processes) |
| 2. Batch runner | `quantark/montecarlo/batch_runner.py` (new) | All batched MC engines; unifies threads (DCN) and Dask (snowball/phoenix) |
| 3. Local-vol surface memo | `quantark/volmodels/localvol/surface_memo.py` (new) | All `*_vol_mc_engines` and `*_vol_pde_solvers` (LV, SLV, Heston-calibration paths that build Dupire) |
| 4. Scenario-grid runner | `quantark/util/parallel/scenario_runner.py` (new) | Any per-cell-independent risk grid: surface shocks, bump Greeks, scenario ladders — MC and PDE alike |

Dependency direction is strictly downward: engines → units; units depend only
on stdlib/NumPy (unit 2's Dask backend via optional import, matching the
existing `DASK_AVAILABLE` pattern).

## 4. Unit 1 — Draw cache moves into the generator

**Today.** `QMCDrawCache` lives in `montecarlo/qmc_sobol.py`, but only the
equity adapter `asset/equity/engine/mc/qmc_draws.py` consults it. Snowball,
phoenix, and the three FX engines construct `SobolNormalGenerator` directly
and bypass the cache.

**Why it is safe to push down.** `SobolNormalGenerator` is stateless between
calls: `normal(n_paths, dim, batch_id)` builds a fresh scrambled
`qmc.Sobol(d=dim, seed=base_seed + batch_id)` engine every call. Each block is
a pure function of `(kind, base_seed, n_paths, dim, batch_id)` — exactly the
key `QMCDrawCache` already uses. Caching at this level returns the same bits
the generator would have produced.

**Change.**
- `SobolNormalGenerator.normal()` and `.uniform()` gain a
  `writable: bool = True` parameter. On every call they consult the shared
  `QMCDrawCache`:
  - hit → return the cached read-only block (`writable=False`) or a fresh
    copy of it (`writable=True`);
  - miss → generate, `put()` a read-only master into the cache, return as
    above.
- **Default `writable=True` at the generator level.** Direct callers
  (snowball, phoenix, FX variance-reduction pipelines) were written when
  blocks were freshly allocated and may mutate them; a copy-on-hit default
  preserves that contract with zero behavior risk while still eliminating
  the expensive Sobol + `ndtri` regeneration. Hot paths that are audited
  mutation-free pass `writable=False` to also skip the copy — the equity
  adapter already does this.
- `asset/equity/engine/mc/qmc_draws.py` keeps its public signatures
  (`qmc_normals`, `qmc_uniforms`, `writable=False` default) but becomes a
  thin delegation to the generator, deleting its private `_cached_block`
  duplication.
- `PseudoRandomNormalGenerator` is untouched: it is a stateful stream and
  must not be cached.
- Cache budget, env knob `QUANTARK_QMC_CACHE_MB`, LRU eviction, and the
  "block larger than budget passes through" rule are unchanged.

**Sizing note.** The cache now sees more distinct keys (multiple products,
multiple seeds). LRU eviction already handles this; the design adds a
`get_qmc_draw_cache().stats()` accessor (hits, misses, evictions,
bytes-in-use) so users can size the budget empirically.

## 5. Unit 2 — Shared batch runner

**Today.** Three parallel conventions coexist: DCN's
`ThreadPoolExecutor` loop (`318009e`), snowball/phoenix's opt-in Dask
`delayed/compute` path, and plain serial loops everywhere else.

**Change.** New `quantark/montecarlo/batch_runner.py`:

```python
def run_batches(fn, num_batches, *, workers=None, backend=None) -> list:
    """Run fn(batch_id) for batch_id in range(num_batches).

    Always returns results ordered by batch_id (deterministic reduction).
    backend: "serial" | "threads" | "dask"; None resolves from
    QUANTARK_MC_BACKEND, defaulting to "threads" when workers > 1
    else "serial". workers: None resolves from QUANTARK_MC_WORKERS
    (default 1). Dask backend requires dask installed; import error
    is raised eagerly with a clear message.
    """
```

- Ordered results are the bit-identity mechanism: engines keep their own
  accumulation code and reduce over the returned list exactly as their serial
  loop would.
- Uneven batch sizes are the engine's business: `fn(batch_id)` closes over
  whatever per-batch path counts the engine computed (snowball's
  remainder-splitting continues to work unchanged).
- `workers` is clamped to `min(workers, num_batches)`; `workers == 1` or
  `num_batches == 1` short-circuits to the serial path with no executor.

**Adoption.**
- **DCN** (`dcn_mc_engine.py`): `price_detailed` batch loop replaced by
  `run_batches`. `num_workers` ctor arg and `QUANTARK_DCN_MC_WORKERS` remain;
  the DCN default resolves as `QUANTARK_DCN_MC_WORKERS` if set, else
  `QUANTARK_MC_WORKERS`, else 1 (back-compat alias).
- **Snowball/phoenix**: serial batch loops and the `_price_parallel` Dask
  scaffolding both route through `run_batches`; `use_dask=True` maps to
  `backend="dask"`. Their existing "parallel matches serial" accumulators are
  kept verbatim.
- Other engines (asian, barrier, sharkfin, …) adopt opportunistically later;
  nothing forces them.

**GIL note.** The threads backend pays off because batch bodies are dominated
by GIL-releasing NumPy/SciPy ufuncs (measured 3.5× at 8 threads on DCN). The
docstring states this so future non-NumPy payloads know to choose
`backend="dask"` (processes) instead.

## 6. Unit 3 — Local-vol surface memo

**Today.** DCN's fix (`_prepare_simulation` + `_resolve_surface` with an
`is pricing_env` identity check) is private to `dcn_vol_mc_engines.py`.
Snowball's LV MC engine still calls `_build_surface(env)` inside
`_create_path_generator` — a per-batch Dupire rebuild, the exact disease DCN
had. PDE vol solvers build once per price call (`_with_surface`) but rebuild
on every repricing of a CRN Greeks loop or shock grid.

**Change.** New `quantark/volmodels/localvol/surface_memo.py`:

```python
class SurfaceMemo:
    """Memoize an expensive surface build keyed on PricingEnvironment identity.

    get(env, builder) returns the memoized surface if env is the SAME object
    as the last call (identity, not equality), else calls builder(env),
    stores, and returns. Holds the env by weakref so a memoized engine never
    extends a dead environment's lifetime. Not thread-safe by design; one
    memo per engine instance, and engines are not shared across threads
    mid-call.
    """
```

- Identity keying (`is`), not equality: two environments that compare equal
  but differ by construction may embed different curve objects; identity is
  the only key that can never produce a stale surface. This is the pattern
  proven in DCN's `_resolve_surface`.
- A prebuilt surface passed to the ctor bypasses the memo entirely
  (existing engines' `self._prebuilt` contract unchanged).

**Adoption.**
- `snowball_vol_mc_engines.py`, `phoenix_vol_mc_engines.py`,
  `barrier_vol_mc_engines.py`: `_build_surface` call sites route through a
  per-engine `SurfaceMemo`. Snowball's per-batch rebuild collapses to one
  build per env — the same ~16× build-count reduction DCN measured.
- `dcn_vol_mc_engines.py`: internals swap to `SurfaceMemo` (behavior
  identical; deletes the bespoke `_active_surface`/`_active_surface_env`
  pair). The frozen mini-project subclass `SurfaceAwareLVDCNEngine` inherits
  unchanged because the `_prepare_simulation`/`_resolve_surface` hook names
  and semantics are preserved.
- PDE vol solvers (`snowball_vol_pde_solvers.py`, `phoenix_vol_pde_solvers.py`,
  `dcn_vol_pde_solvers.py`, `barrier_vol_pde_solvers.py`): `_with_surface`
  consults the memo before building, making repeated pricings against the
  same env (CRN Greeks, shock grids with per-cell envs reused across
  solvers) build Dupire once.

## 7. Unit 4 — Parallel scenario-grid runner

**Today.** The 5.2× surface-risk parallelization lives in the solution repo
(`surface_risk_parallel.py`) as a bespoke script: ProcessPoolExecutor +
per-worker initializer that rebuilds curves/products per process.

**Change.** New `quantark/util/parallel/scenario_runner.py`:

```python
def run_scenario_grid(cells, run_cell, *, workers=None, initializer=None,
                      initargs=(), ordered=True) -> list:
    """Map run_cell over independent scenario cells with a process pool.

    cells: sequence of picklable cell descriptors.
    run_cell: module-level callable (picklable) executed in workers.
    initializer/initargs: per-worker setup (fitters, curve caches, env vars).
    workers: None resolves from QUANTARK_SCENARIO_WORKERS, default
    min(cpu_count, len(cells)); workers == 1 runs inline (no pool), which
    keeps debugging and coverage simple.
    Results are returned in cell order.
    """
```

- Process-based (not threads) because scenario cells re-price entire
  products — large working sets, surface builds, and per-cell state that
  benefit from isolation; this generalizes the measured 26-cell/5.2× result.
- Engine-level thread parallelism composes: the initializer may set
  `QUANTARK_MC_WORKERS` for within-cell threading, exactly as the solution
  script sets per-worker env today.
- Works for PDE and MC alike — a cell is just "build env, price, return
  numbers". This is the PDE-facing deliverable of this design: PDE solvers
  gain grid-level parallelism without touching solver internals.
- Serial fallback (`workers=1`) is the default when env/knobs are absent,
  preserving current behavior everywhere.

## 8. Env-var and versioning summary

| Knob | Meaning | Default | Notes |
|---|---|---|---|
| `QUANTARK_QMC_CACHE_MB` | Draw-cache budget (0 disables) | 2048 | unchanged |
| `QUANTARK_MC_WORKERS` | Batch-loop workers, all MC engines | 1 | new |
| `QUANTARK_DCN_MC_WORKERS` | DCN override | unset | kept; wins over the global for DCN |
| `QUANTARK_MC_BACKEND` | `serial`/`threads`/`dask` | threads when workers>1 | new |
| `QUANTARK_SCENARIO_WORKERS` | Scenario-grid processes | 1 | new |

Version: `0.2.6 → 0.3.0` (minor bump: new public modules and knobs, no
breaking changes). No wheel is vendored anywhere.

## 9. Testing and verification

1. **Golden bit-identity harness** (new `test/test_perf_generalization.py`):
   for each adopted engine (DCN GBM/LV/Heston, snowball GBM/LV, phoenix,
   barrier LV, FX TARF/TARN/barrier QMC), capture PV + stderr + diagnostics
   at the baseline commit, then assert byte-equality after each unit lands:
   - defaults (serial, cache on) == baseline;
   - cache disabled (`QUANTARK_QMC_CACHE_MB=0`) == cache enabled;
   - `workers=8` == `workers=1`, for both threads and (if installed) dask
     backends;
   - `use_dask=True` == legacy dask path outputs (snowball/phoenix).
2. **Unit tests per module**: cache stats/eviction under multi-key load;
   `writable=True` copies are independent and `writable=False` blocks are
   read-only; `run_batches` ordering under a permuted-completion executor
   stub; `SurfaceMemo` identity semantics, weakref expiry, prebuilt bypass;
   scenario runner ordering, inline mode, initializer propagation.
3. **Build-count regression tests**: counting-builder engines assert one
   Dupire build per env for snowball/phoenix/barrier MC and PDE (the DCN
   `_CountingLVEngine` pattern, generalized).
4. **Mutation tripwire**: full library suite runs with a temporary test
   fixture forcing `writable=False` generator returns, to enumerate any
   caller that mutates draw blocks; such callers stay on `writable=True`
   and are listed in the module docstring.
5. **Full suite**: `test/` green except the pre-existing, environment-
   sensitive `test_snowball_quad_flat_identity_golden` last-ulp failure
   (documented as out of scope; exists at `0645969` before any perf work).

## 10. Non-goals

- No `BatchedMCEngine` base-class refactor (future work, now cheaper).
- No PDE solver-internal optimization (operators, grids) — PDE gains come
  from the scenario runner and surface memo only.
- No changes to `PseudoRandomNormalGenerator` or any stateful stream.
- No changes to the quant-mini-project deliverable, wheel, manifests, or
  published results.
- No removal of the legacy Dask code paths in this pass beyond routing them
  through `run_batches`; deleting dead scaffolding can follow once adoption
  is proven.

## 11. Rollout order

Each step lands with its tests green and is independently revertable:

1. Unit 1 (generator cache) + adapter thinning + golden harness bootstrap.
2. Unit 3 (surface memo) + MC adoptions (snowball/phoenix/barrier/DCN).
3. Unit 3 PDE adoptions (vol PDE solvers).
4. Unit 2 (batch runner) + DCN adoption, then snowball/phoenix adoption.
5. Unit 4 (scenario runner) + a worked example under `example/` showing a
   PDE shock grid and an MC shock grid on the same runner.
6. Version bump + CHANGELOG/docs (`quantark/asset/equity/engine/docs/`
   performance page describing knobs and adoption status per engine).
