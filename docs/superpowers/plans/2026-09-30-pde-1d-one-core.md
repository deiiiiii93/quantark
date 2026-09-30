# PDE 1D One Core (sub-project 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six hand-written 1D θ-step loops, three coefficient builders, and six preparation paths with one `prepare → march → readout` core. Product families plug in as adapters over column groups. Move the four scheme settings into a `CrankNicolson` stepper. Add `engine.explain(...)`. Every number stays bit-identical and no banked certificate identity moves.

**Architecture:** A new package `quantark/asset/equity/engine/pde/core/` holds the core:
- typed stages: `prepare` builds a frozen `PreparedProblem`, `march` returns a `MarchResult`, `read` returns the outputs;
- two private arithmetic backends (sparse LU and banded) behind one `FactorizationCache`;
- a sampler/stencil split (`MarketSamples` → `CentralDifference`);
- one adapter module per product family.

Adapters own the product knowledge by calling each family's existing product helpers (terminal payoffs, boundaries, event jumps). The helpers stay on the solver classes because the 2D Heston/SLV solvers inherit them. What gets deleted is every loop and every duplicated preparation. Families move one at a time. Each migration is followed by the extended byte book, which is a manifest-checked `float.hex()` gate run against the merged sub-project 1 commit. The scheme relocation, the evidence record, and the release come last.

**Tech Stack:** Python 3.10–3.13, NumPy/SciPy (`scipy.sparse.linalg.splu`, `scipy.linalg.solve_banded`), pytest (`-n auto` by default; `-n0` for `-s`/pdb), frozen dataclasses, `quantark.execution.cache.fingerprint`.

**Spec:** `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`. Read it next to this plan; § numbers below refer to it. **The spec and this plan are git-ignored files in the sub-project 1 worktree** (`.claude/worktrees/pde-1d-surface-cleanup/docs/superpowers/…`, status `!!`). Before that worktree is removed, copy both into the main checkout and `git add -f` them (user's call). Otherwise `git worktree remove` deletes them.

## Global Constraints

- **Prerequisite:** sub-project 1 is merged to `main`. `git merge-base --is-ancestor 302fa2ed main` must succeed, or a rebased equivalent of that commit must be on `main`. The byte-book base is the `main` commit this plan's worktree branches from. Line numbers quoted below are taken at `302fa2ed`. If a line has shifted, re-locate it by symbol name.
- Release **0.7.0**, clean break. Removed names raise `ValidationError`, and the message names the replacement from §4.4. Nothing is deprecated.
- **Bitwise, every task:** run `compare.py --manifest` (Task 1) against the one base capture. It must report `0 unexpected diffs` and `0 manifest violations`. No `--allow` prefix is used anywhere in this plan.
- **No banked identity moves:** `test_banked_cells_keep_their_identity` (`test/modelvalidation/test_variant_case_builders.py:172`) and every `assert_anchors` test pass unchanged after every task. The banked prefixes are snowball `c5b6a27b…`, phoenix `6a9aae02…`, and ko-reset `f2c9bd52…`.
- **Backends are fixed per family (decision 9):**
  - European, knock-out barriers, touches, American: sparse LU.
  - Snowball, KO-reset, LV snowball: one banded two-column group, or LU when the interior has ≤ 2 nodes.
  - Phoenix, LV phoenix: banded per column (1-D right-hand side), or LU per column.
  - Statistics: banded only (raises otherwise).
- **Factorization key:** `(backend, set index, round(dt, 12), θ)`.
  - θ is exact (decision 6).
  - `round(dt, 12)` is applied to `dt` exactly as each path passes it today: a `np.float64` on the value marches, a Python `float` on the statistics march.
  - The LU store is unbounded and cleared per march, as `_matrix_cache` is today.
  - The banded store is LRU, bounded by `BANDED_CACHE_MAX_ENTRIES = 512` (`move_to_end` on a hit, `popitem(last=False)` on overflow).
  - A session pack is consulted first. With `cache_enabled=False` every lookup builds fresh after the pack check.
- Cache constants are unchanged: `GRID_CACHE_MAX_ENTRIES = 128`, `BANDED_CACHE_MAX_ENTRIES = 512`, session pack cap `64`. `_BUILDER_VERSION = "pde-prep/v2"` (Task 10).
- **Scheme defaults:**
  - `CrankNicolson(theta=0.5, event_theta=1.0, terminal_damping_steps=None, event_damping_steps=None)`.
  - `theta` must be in [0, 1] and `event_theta` in (0, 1]. Counts must be non-negative ints.
  - `None` means the route default: (1, 2) on every core route and on the local-vol and 2D routes; (2, 2) on DCN.
  - Terminal damping always uses θ = 1.0. Terminal damping wins on overlap.
- `GridConfig` ends with nine fields in this order: `points, steps_per_day, time_steps, eps_crit, num_std, bounds, max_points, max_steps, day_count`. `ACCURACY_PROFILES` is unchanged. `_SHARED_DEFAULTS` loses only the two damping keys. `DCN_ACCURACY_PROFILES` loses only its two damping entries.
- Canonical `quantark.*` imports. Float comparisons go through `quantark.util.numerical` (`is_close`, `is_zero`, `Tolerance`). Never import or instantiate an MC engine from PDE code. Never add a fallback that invents semantics: if something cannot be implemented exactly, stop and ask.
- CI floor is Python 3.10 (`dataclass(kw_only=True)` is fine; no `match`-on-types tricks that need 3.11). Run the suite on ≥ 3.10 and 3.12 before merge.
- **Worktree:** create it with the native `EnterWorktree` tool, then verify that `git rev-parse HEAD` equals `main`. Run tests as `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …`. The worktree Bash refuses `source`, so never activate a venv.
- **Commits:**
  - Commit by explicit pathspec. Never use `git add -A` or `git add example/`.
  - `docs/` is excluded by `.git/info/exclude`, so a tracked docs file needs `git add -f`.
  - Never commit a `CLAUDE.md`.
  - End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Timing (Task 14):** run both arms back to back in one window on the same machine. An unstable measurement is inconclusive, not a pass.

## Review Focus

These five inputs are implied by the spec but no spec test exercises them. Each has a pinned test in the task that owns the code.

1. **Deep-copying or pickling a solver after it has priced.** The intraday route deep-copies the engine, and process pools pickle it. Measured at `302fa2ed`: both `copy.deepcopy` and `pickle` fail after one `EuropeanPDESolver.price` with `TypeError: cannot pickle 'SuperLU' object`, because `_matrix_cache` lives on the solver. Remove that and the next failure is `mappingproxy`: `TimeLayout.step_of` inside the binder's layout cache. After this plan the factorizations live on the `PreparedProblem` and `TimeLayout` reduces to plain values. Expected: both operations succeed after a `price`, and the copy prices bit-identically. Pinned in Task 5, Step 1 (base family), and extended to Snowball in Task 6 and to Phoenix in Task 8.
2. **American Greeks on a fresh solver, for a product whose exercise needs real dates.** Such a product has a settlement lag in business or calendar days, so `american_exercise_requires_dates(product)` returns True (`settlement_support.py:113`). The base `calculate_greeks` never runs `_prepare_solve_state`, so today it raises `PricingError("American exercise dates must be resolved…")`, or reads state left over from an earlier `price`. After this plan every operation runs the one `prepare`. Expected: fresh-solver Greeks equal Greeks-after-price, bit for bit. Pinned in Task 5, Step 1.
3. **Event statistics on a reused solver after a solve on a different market.** Today `_compute_event_stats` never clears the solver's `_banded_cache`, and the cache key `(set, dt, θ)` carries no coefficient identity. `PDEEngine` caches one solver per product type, so this is the normal usage. Measured at `302fa2ed`: after `s.price(p, env(vol=0.20))`, `s.calculate_event_stats(p, env(vol=0.25))` returns KO probabilities that differ from a fresh solver's by up to **0.0407**. The spatial grids differ but have the same size, so the stale matrices are silently accepted. After this plan the cache belongs to each prepared problem and is cleared per march. Expected: the statistics equal a fresh solver's, bit for bit. The book's rows use fresh engines, so the fix is bitwise there. It is a user-visible correction, and it goes in the changelog. Pinned in Task 9, Step 1.
4. **A `PricingSession` running two products with identical geometry and different economics** (KO coupon 15% vs 30%). The artefact keys are deliberately economics-free and shared (decision 13). Expected: each outcome equals direct pricing of that product, and the second product reuses the grid artefact. Pinned in Task 10, Step 1.
5. **`cache_enabled=False` against `True`.** The `PDEParams` docstring says "Same numbers either way". Because of the `round(dt, 12)` key, a cached run shares one factorization across intervals whose `dt` differ by an ULP, while an uncached run builds each one from its own `dt`. Expected: the two agree to `test/golden_compare.py` tolerance, not to the bit, and the docstring says so. Pinned in Task 4, Step 1. The docstring is corrected in Task 12.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `test/pde_bitwise/capture_book.py` | Adds the §9.1 rows; the `Row` registry declares `None` outputs and expected failures; `--write-manifest`; drops `__trace__` from the JSON it compares | 1 |
| `test/pde_bitwise/compare.py` | `--manifest` gate (decision 20) | 1 |
| `test/pde_bitwise/manifest.json` (new) | Every row id, its declared `None` outputs, and its declared error type | 1 |
| `quantark/asset/equity/engine/pde/base_pde_solver.py` | Exact θ in `_get_matrices` (2); value-keyed memos (3); legacy delegates (4); base march deleted (5); transient machinery deleted (10) | 2–5, 9, 10 |
| `quantark/asset/equity/engine/pde/snowball_pde_solver.py` | Exact θ in the banded key (2); observation keys (3); snowball march, statistics and preparation deleted (6, 9) | 2, 3, 6, 9 |
| `quantark/util/calendar/business_calendar.py` | `Calendar.identity()` | 3 |
| `quantark/asset/equity/engine/pde/grid/binder.py` | `Layout.fingerprint`; `validate_external_layout` compares `extra_times` | 3, 4 |
| `core/__init__.py`, `core/groups.py`, `core/backends.py`, `core/samples.py`, `core/coefficients.py`, `core/components.py`, `core/identity.py` (new) | Core types, backends, cache, sampler, stencil, stepper, readout, identities | 4 |
| `core/prepare.py`, `core/march.py`, `core/readout.py` (new) | The three stages | 5 |
| `core/adapters/base.py`, `european.py`, `american.py`, `barrier.py` (new) | Base-family adapters, knock-in composite | 5 |
| `core/adapters/snowball.py` (new) | Snowball and LV snowball; life surface | 6 |
| `core/adapters/ko_reset.py` (new) | KO-reset | 7 |
| `core/adapters/phoenix.py` (new) | Phoenix and LV phoenix | 8 |
| `core/statistics.py` (new) | The statistics-legacy column group (§6) | 9 |
| `pde_session_prep.py`, `pde_execution_adapters.py` | Sessions through `prepare(session=…)` | 5 (packs), 10 |
| `core/plan.py` (new), `quantark/intraday/engines/pde.py` | `PDEPlan` / `explain`; intraday reads the plan | 11 |
| `quantark/asset/equity/param/engine_params.py`, `param/pde_removed_names.py`, `grid/config.py`, `grid/time.py`, `route_applicability.py`, `dcn_pde_solver.py`, `snowball_vol_pde_solvers.py`, `phoenix_vol_pde_solvers.py`, `local_vol_pde_solver.py`, `barrier_vol_pde_solvers.py` | Scheme relocation and readers | 12 |
| `grid/config.py` (`resolved_grid_evidence`), `quantark/modelvalidation/builders/equity_{snowball,phoenix,ko_reset,snowball_localvol}.py`, `intraday_snowball.py` | Evidence record and candidate wiring | 13 |
| `test/pde_core/` (new): `test_groups.py`, `test_march_step.py`, `test_components.py`, `test_prepare.py`, `test_market_memos.py`, `test_identity.py`, `test_factorization_cache.py`, `test_review_focus.py` | New tests (§9.3 plus Review Focus) | 2–13 |
| `docs/engine_param_guide.md`, `docs/pde-core.md` (new), `README.md`, `CHANGELOG.md`, `pyproject.toml`, `quantark/__init__.py`, `docs/execution/capability-matrix.md`, `docs/modelvalidation/RELEASE_PROCEDURE.md` | Release | 14 |

All `core/…` paths are under `quantark/asset/equity/engine/pde/`.

---

### Task 0: Start

- [ ] **Step 1: Check the prerequisite.**
  ```bash
  cd /Users/fuxinyao/quant-ark && git checkout main && git log --oneline -3
  git merge-base --is-ancestor 302fa2ed main && echo MERGED || git log --oneline main | grep -m1 "0.6.0 — PDE 1D surface cleanup"
  ```
  Expected: `MERGED`, or the rebased 0.6.0 release-note commit is on `main`. If neither holds, **stop**. The plan's base does not exist yet.
- [ ] **Step 2: Carry the spec and plan.** Copy `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md` and this plan from the sub-project 1 worktree into the main checkout, if they are not already there. Committing them (`git add -f`) is the user's call. Ask.
- [ ] **Step 3: Create the worktree** with `EnterWorktree` (name `pde-1d-one-core`). Then run `git rev-parse HEAD` and `git rev-parse main`. They must be equal. `EnterWorktree`'s `baseRef` is a settings value, so verify rather than assume.
- [ ] **Step 4: Baseline suite.** `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q -x`. Expected: green. Record the counts in the task log.

---

### Task 1: Byte book for sub-project 2, manifest gate, base capture

**Files:**
- Modify: `test/pde_bitwise/capture_book.py`, `test/pde_bitwise/compare.py`
- Create: `test/pde_bitwise/manifest.json` (generated, then committed)

**Interfaces:**
- Produces:
  - `capture_book.py --out PATH [--only SUBSTR] [--workers N]` (unchanged).
  - `capture_book.py --write-manifest PATH`.
  - `compare.py BASE BRANCH --manifest test/pde_bitwise/manifest.json` exits 0 iff there are no violations and no diffs.
  - The `Row` registry: `Row(fn, none=frozenset[str](), error: Optional[str]=None)`.
- Every later task's "byte gate" is Step 5's branch capture plus `compare.py --manifest` against the same `$SCR/base.json`.

The script runs on the base (sub-project 1) commit and on every branch commit. Rows whose construction changes pick their API per row by feature detection:
- `"stepper" in _fields(PDEParams)` for the scheme rows;
- `hasattr(cls, "explain")` for the budget row.

- [ ] **Step 1: Registry, trace-free errors, manifest writer.** In `capture_book.py`:

```python
@dataclasses.dataclass(frozen=True)
class Row:
    """A book row: its thunk, outputs declared None, and a declared failure type."""
    fn: object
    none: frozenset = frozenset()
    error: Optional[str] = None   # exception class name the row must raise on both sides


def _row(fn, *, none=(), error=None):
    return Row(fn, frozenset(none), error)
```

- `book()` keeps returning `{row_id: thunk}` for the existing rows. `rows()` returns `{row_id: Row}` by wrapping every legacy thunk in `Row(fn)` and adding the new rows (Step 2).
- `run_row` records only `{"__error__": f"{type(exc).__name__}: {exc}"}`. `__trace__` goes to stderr, so books from worktrees at different paths compare equal.
- Add `--write-manifest PATH`. It writes `{row_id: {"none": sorted(row.none), "error": row.error}}` for every row, sorted by key, then exits.

- [ ] **Step 2: The §9.1 rows.** Add them to `rows()`. Each helper below goes above `rows()`.

```python
def _stats_fields(stats, prefix="stats_"):
    out = {}
    for name, val in sorted(vars(stats).items()):
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            out[f"{prefix}{name}"] = _hex(val)
        elif isinstance(val, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(val, dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            out[f"{prefix}{name}"] = [x.hex() for x in arr.tolist()]
    return out


def stats_only(solver, product, env, streams=None):
    """Direct calculate_event_stats, no npv: the march without a price (§5.10)."""
    stats = solver.calculate_event_stats(product, env, streams=streams)
    return {"stats": None} if stats is None else _stats_fields(stats)


def pwe(solver, product, env, streams=None):
    res = solver.price_with_events(product, env, streams=streams)
    out = {"npv": _hex(res.npv)}
    dist = res.event_distribution
    if dist is not None:
        out["dist_times"] = [float(x).hex() for x in np.asarray(dist.event_times, float).ravel()]
        out["dist_survival"] = [float(x).hex() for x in np.asarray(dist.survival_probability, float).ravel()]
        for et in sorted(dist.probabilities, key=str):
            out[f"dist_{et}"] = [float(x).hex() for x in np.atleast_1d(np.asarray(dist.probabilities[et], float)).ravel()]
    return out


def p_stepper(**kw):
    """Branch after Task 12: the stepper; before: the 0.6.0 home of the same four settings."""
    if "stepper" in _fields(PDEParams):
        from quantark.asset.equity.engine.pde.core.components import CrankNicolson
        return PDEParams(stepper=CrankNicolson(**kw))
    grid_kw = {k: kw.pop(k) for k in ("terminal_damping_steps", "event_damping_steps") if k in kw}
    return PDEParams(grid=GridConfig(**grid_kw) if grid_kw else None, **kw)


def phoenix_plain():
    return make_phoenix(dict(PHOENIX, memory_coupon=False))
```

Update the existing scheme helpers so they work on base and on every branch commit:
- `p_theta1()` returns `p_stepper(theta=1.0)`.
- `p_damping_0_1()` returns `p_stepper(terminal_damping_steps=0, event_damping_steps=1)`. Keep its base-only `use_rannacher` branch first.

Update `budget_row` to read the step count as follows:

```python
    if hasattr(cls, "explain"):
        n = int(cls(params=PDEParams()).explain(product, env).time.actual_steps)
    else:
        probe = cls(params=PDEParams()); probe.price(product, env)
        n = int(probe._active_layout.time.actual_steps)
```

New rows. Build the solver fresh inside every thunk, and import heavy modules lazily inside the thunks, as `vol_rows` does.

| Row id pattern | Thunk | Declared |
|---|---|---|
| `phoenix_plain\|{flat,term}\|{acc}` | `facade_outputs(phoenix_plain(), env, p_default(acc), event_stats=True)` | — |
| `stats\|{snowball_cont,snowball_monthly,snowball_daily,snowball_knocked_in,phoenix_memory,phoenix_plain,ko_reset}\|{flat,term}\|{acc}` | `stats_only(<Solver>(params=p_default(acc)), product, env)`, where `<Solver>` is `SnowballPDESolver` / `PhoenixPDESolver` / `KOResetSnowballPDESolver` | — |
| `pwe\|<same seven>\|{flat,term}\|standard` | `pwe(<Solver>(params=p_default()), product, env)` | — |
| `stats\|snowball_cont\|flat\|standard\|streams_ko`, `pwe\|…\|streams_ko` | same with `streams=frozenset({EventType.KO})` | — |
| `stats\|snowball_monthly\|flat\|nodal`, `pwe\|snowball_monthly\|flat\|nodal` | `p_nodal()` | — |
| `session_{phoenix,ko_reset}` | the `session_row` pattern with `PricingRequest(product=…, pricing_env=env_flat(), operation=PricingOperation.PRICE_DETAILED, outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS}))`; outputs are `price` = `outcome.value.pv`, `_stats_fields(outcome.value.event_stats)`, and `adapter` | — |
| `life_surface\|snowball` | `PDEEngine(params=PDEParams()).solve_life_surface(snowball(), env_flat(), extra_times=(0.25, 0.5, 0.75))`; outputs are hex lists of `sol.v0[:, 0]`, `sol.v1[:, 0]`, `sol.v0[:, -1]`, `sol.t0_readout`, `sol.t`; ints `actual_steps` and `requested_steps` | — |
| `{heston,slv}_snowball\|stats`, `heston_phoenix\|stats` | `stats_only(<2D solver as in vol_rows>, product, env_flat())` | — |
| `scheme\|{european,snowball_cont}\|theta1`, `…\|damp01` | `facade_outputs(m(), env_flat(), p_theta1() / p_damping_0_1(), event_stats=is_snowball)` | — |
| `terminal\|expired\|{snowball,phoenix,barrier}\|{price,greeks,pwe,stats}` | see below | the `stats` operation declares `none={"stats"}`; the barrier has no `pwe` / `stats` rows |
| `terminal\|immediate_ko\|snowball\|{price,greeks,pwe,stats}` | `env = env_flat()` with `SpotQuote(spot=130.0)` (the `test_price_with_events_single_pass.py::test_npv_matches_price_immediate_ko` fixture) | — |
| `terminal\|immediate_ko\|snowball\|life_surface` | `solve_life_surface(…)` | `error="ValidationError"` |
| `bump\|frozen_vs_fresh\|snowball` | `ctx = SnowballPDESolver(params=PDEParams()).create_bump_context(p, env_flat())`; bump the spot to 101 with the env's `with_spot` / `SpotQuote` replacement used in `GreeksCalculator`; outputs are `frozen = ctx.price(p, env101)` and `fresh = SnowballPDESolver(params=PDEParams()).price(p, env101)` | — |
| `heston_snowball\|greeks` | `solver_price_greeks(HestonSnowballPDESolver(hp, params=PDEParams(), n_x=80, n_v=32, n_t=40), snowball(), env_flat())` (as `vol_rows` builds it). This exercises the 2D `calculate_greeks` → `_bump_clone` path (Task 10) | — |
| `lv_snowball\|prebuilt` | `surf = LocalVolSnowballPDESolver(params=PDEParams())._build_surface(env_grid_vol())`, then `solver_price_greeks(LocalVolSnowballPDESolver(params=PDEParams(), local_vol_surface=surf), snowball(), env_grid_vol())` | — |
| `knock_in\|{up_in_call,down_in_put}\|{flat,term}\|standard`, `knock_in\|double_in\|…` | `facade_outputs(product, env, p_default())`. Products: `BarrierOption(strike=100, maturity=1.0, option_type=CALL, barrier=130, barrier_type=UP_IN, observation_type=CONTINUOUS)`; the put with `barrier=80, barrier_type=DOWN_IN`; `DoubleBarrierOption(100.0, OptionType.CALL, 120.0, 80.0, DoubleBarrierType.KNOCK_IN, maturity=1.0)` | — |
| `knock_in\|…\|knocked_in` | the same three products priced with spot beyond the barrier (spot 131, 79, 121) | — |
| `lu_fallback\|{snowball,phoenix}` | `solver_price_greeks(<Solver>(params=PDEParams(grid=GridConfig(points=4, eps_crit=1.0))), product, env_flat())`. The interior has two nodes, so the value march takes its LU branch (measured: a 4-point snowball prices at `302fa2ed`) | — |
| `lu_fallback\|snowball\|stats` | `stats_only(SnowballPDESolver(params=PDEParams(grid=GridConfig(points=4, eps_crit=1.0))), snowball(), env_flat())` | `error="ValidationError"` (the statistics march is banded only) |
| `settlement_lag\|{snowball,phoenix}` | `facade_outputs(_snowball(_convention()), _env(), p_default(), event_stats=True)` and the phoenix equivalent. Import `_snowball`, `_phoenix`, `_convention`, `_env` from `test_structured_numerical_settlement`; `test/` is on `sys.path` | — |

**Terminal-row products.** Define them as module functions named `expired_snowball()`, `expired_phoenix()`, `expired_barrier()`, `env_after_maturity(product)` (a `flat_env` dated ten days after `product.exercise_date`) and `immediate_ko_env()`. Task 11's tests import them. Expired rows use `dated_snowball(cal, T0)` and `dated_phoenix(cal, T0)` from `intraday.conftest`, with `cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))` and `T0 = datetime(2026, 3, 16)`. Price them in `flat_env(<their exercise_date> + timedelta(days=10))`. The expired barrier is `BarrierOption(…, exercise_date=datetime(2026, 6, 1))` priced at valuation `datetime(2026, 7, 1)`.

**Operation outputs:**
- `price`: `{"price": hex}`.
- `greeks`: the `grid_*` keys from `solver_price_greeks`.
- `pwe`: `pwe(...)`.
- `stats`: `stats_only(...)`.

- [ ] **Step 3: Manifest gate in `compare.py`.** Replace `main` with:

```python
def _finite_hex(v) -> bool:
    try:
        vals = v if isinstance(v, list) else [v]
        return all(isinstance(x, str) and math.isfinite(float.fromhex(x)) for x in vals)
    except (TypeError, ValueError):
        return False


def _manifest_violations(book: dict, manifest: dict, side: str) -> list:
    out = []
    for row, decl in manifest.items():
        if row not in book:
            out.append(f"{side}: missing row {row}")
            continue
        got = book[row]
        err = got.get("__error__")
        if decl["error"] is not None:
            if err is None or not err.startswith(decl["error"] + ":"):
                out.append(f"{side}: {row} must raise {decl['error']}, got {err or 'no error'}")
            continue
        if err is not None:
            out.append(f"{side}: {row} raised undeclared {err}")
            continue
        for key, val in got.items():
            if key in decl["none"]:
                if val is not None:
                    out.append(f"{side}: {row}.{key} declared None, got {val!r}")
            elif isinstance(val, (int, str)) and key in ("adapter", "steps", "actual_steps", "requested_steps"):
                continue  # declared integer / identifier outputs
            elif not _finite_hex(val):
                out.append(f"{side}: {row}.{key} is not a finite hex: {val!r}")
    extra = sorted(set(book) - set(manifest))
    out += [f"{side}: row {r} not in manifest" for r in extra]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("branch")
    ap.add_argument("--allow", action="append", default=[])
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)
    base = {k: {kk: vv for kk, vv in v.items() if kk != "__trace__"} for k, v in json.load(open(a.base)).items()}
    branch = {k: {kk: vv for kk, vv in v.items() if kk != "__trace__"} for k, v in json.load(open(a.branch)).items()}
    violations = []
    if a.manifest:
        manifest = json.load(open(a.manifest))
        violations = _manifest_violations(base, manifest, "base") + _manifest_violations(branch, manifest, "branch")
    # … the existing diff loop over base/branch, unchanged …
    for v in violations:
        print(f"MANIFEST {v}")
    print(f"{len(base)} base rows, {len(branch)} branch rows, {len(bad)} unexpected diffs, "
          f"{len(allowed)} allowed diffs, {len(violations)} manifest violations")
    return 1 if (bad or violations) else 0
```

Add `import math` at the top. Also accept `"dcn"`-style `steps` integers, and the `pv` key the DCN rows emit. Both are finite hexes or declared ints.

- [ ] **Step 4: Write and review the manifest.**
  ```bash
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python test/pde_bitwise/capture_book.py --write-manifest test/pde_bitwise/manifest.json
  ```
  Open it. The only declarations should be:
  - `none: ["stats"]` on the three `terminal|expired|…|stats` rows;
  - `error: "ValidationError"` on `terminal|immediate_ko|snowball|life_surface` and on `lu_fallback|snowball|stats`.

- [ ] **Step 5: Capture base, capture this commit, compare.**
  ```bash
  BASE=$(git merge-base HEAD main)
  SCR=/private/tmp/claude-501/-Users-fuxinyao-quant-ark/pde-one-core   # outside the repo
  mkdir -p $SCR && git worktree add --detach $SCR/base $BASE
  ( cd $SCR/base && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
      "$OLDPWD/test/pde_bitwise/capture_book.py" --out $SCR/base.json )
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python test/pde_bitwise/capture_book.py --out $SCR/branch.json
  /Users/fuxinyao/quant-ark/.venv/bin/python test/pde_bitwise/compare.py $SCR/base.json $SCR/branch.json --manifest test/pde_bitwise/manifest.json
  ```
  The detached base worktree is a byte-compare fixture, so plain `git worktree add --detach` is right here. `EnterWorktree` is for the branch you work in.

  Expected: `0 unexpected diffs, 0 manifest violations`.

  **Measure; do not assume.** Every terminal row must actually take its short-circuit. For each one, check the reason on base:
  - The expired `stats` rows are `{"stats": null}`.
  - The immediate-KO `price` equals `SnowballPDESolver._get_immediate_ko_payoff`.
  - The expired `price` rows equal the terminal value.

  If a constructor above does not produce the intended state on base, fix the constructor, not the manifest.

  Also confirm the knock-in rows took the composition: on base, instrument `BarrierPDESolver._price_vanilla` with a counter in a scratch script.
- [ ] **Step 6: Negative control.** Hand-edit one hex digit in a copy of `branch.json` and confirm `compare.py` fails. Then write `{"__error__": "X: y"}` into one row on both sides and confirm the manifest reports it as an undeclared failure. Record both outputs in the task log.
- [ ] **Step 7: Commit.**
  ```bash
  git add test/pde_bitwise/capture_book.py test/pde_bitwise/compare.py test/pde_bitwise/manifest.json
  git commit -m "test(pde): sub-project 2 byte book - statistics, terminal, knock-in, session, life-surface rows; manifest gate (spec 2 §9.1, decision 20)"
  ```

**Byte gate (used by every later task):**
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python test/pde_bitwise/capture_book.py --out $SCR/branch.json
/Users/fuxinyao/quant-ark/.venv/bin/python test/pde_bitwise/compare.py $SCR/base.json $SCR/branch.json --manifest test/pde_bitwise/manifest.json
```
Expected: `0 unexpected diffs, … 0 manifest violations`. Byte compares are immune to machine load, so the base is not re-captured per task. **Identity gate:** `… -m pytest test/modelvalidation/test_variant_case_builders.py test/modelvalidation/test_banked_certificates.py test/modelvalidation/test_anchors.py -q` → PASS.

---

### Task 2: Exact θ in the factorization keys (decision 6)

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py:1341` (`_get_matrices` key)
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py:2701,2726,2754` (banded key, pack key, `_pack_banded_entry`)
- Modify: `base_pde_solver.py:1656-1729` (`_session_factorization_packs` builds the same keys)
- Create: `test/pde_core/__init__.py` (empty), `test/pde_core/test_factorization_cache.py`

**Interfaces:**
- Produces: the key tuples `(coeff_key, round(dt, 12), theta)` with `theta` a Python `float`, on both backends. Task 4 lifts them into `FactorizationCache.key`.

- [ ] **Step 1: Write the failing tests.**

```python
"""Factorization-cache keys: θ exact, dt rounded as today (spec 2 decision 6, §5.7)."""
import numpy as np

from quantark.asset.equity.engine.pde import EuropeanPDESolver, SnowballPDESolver
from quantark.asset.equity.param import PDEParams


def _lcu(n=8):
    l = np.full(n, 1.0); c = np.full(n, -2.5); u = np.full(n, 1.2)
    return l, c, u


def test_two_thetas_agreeing_to_six_decimals_get_two_lu_factorizations():
    import scipy.sparse as sp
    s = EuropeanPDESolver(PDEParams())
    l, c, u = _lcu()
    A = s._build_operator_matrix(l, c, u, 8)
    I = sp.eye(6, format="csc")
    s._matrix_cache.clear()
    s._get_matrices(I, A, 0.01, 0.5, coeff_key=0)
    s._get_matrices(I, A, 0.01, 0.5000001, coeff_key=0)
    assert len(s._matrix_cache) == 2


def test_two_thetas_agreeing_to_twelve_decimals_get_two_banded_systems():
    s = SnowballPDESolver(PDEParams())
    l, c, u = _lcu()
    s._banded_cache.clear()
    s._get_banded_system(l, c, u, 0.01, 0.5, coeff_key=0)
    s._get_banded_system(l, c, u, 0.01, 0.5 + 1e-13, coeff_key=0)
    assert len(s._banded_cache) == 2


def test_dt_values_within_an_ulp_share_one_factorization():
    s = SnowballPDESolver(PDEParams())
    l, c, u = _lcu()
    dt = np.float64(1.0 / 12.0)
    s._banded_cache.clear()
    s._get_banded_system(l, c, u, dt, 0.5)
    s._get_banded_system(l, c, u, np.nextafter(dt, 1.0), 0.5)
    assert len(s._banded_cache) == 1
```

- [ ] **Step 2: Run the tests.** `… -m pytest test/pde_core/test_factorization_cache.py -v`. Expected: the first two FAIL (1 entry each) and the third PASSES.
- [ ] **Step 3: Implement.**
  - In `_get_matrices`, change the key to `(coeff_key, round(dt, 12), float(theta))`.
  - In `_get_banded_system` and `_pack_banded_entry`, change it to `(coeff_key, round(dt, 12), float(theta))`.
  - In `_session_factorization_packs`, make the keys it writes identical. The function passes `theta` through `_get_matrices` / `_pack_banded_entry`, so check that no second `round(theta, …)` exists: `grep -n "round(theta" quantark/` must return nothing.
  - Add one comment at each key, verbatim: `# θ exact (spec 2 decision 6); dt rounded: ULP-different intervals share one factorization, as every banked number was computed.`
- [ ] **Step 4: Run the tests, then the byte gate and the identity gate.** Expected: PASS; 0 diffs, 0 violations.
- [ ] **Step 5: Commit.** `git add` both solver files and `test/pde_core/`. Message: `fix(pde): exact θ in the factorization keys - two θ agreeing to 6 (LU) or 12 (banded) decimals no longer share a matrix (spec 2 decision 6)`.

---

### Task 3: Value-keyed market memos, `Calendar.identity()`, effective settlement calendar, `extra_times` validation

**Files:**
- Modify: `quantark/util/calendar/business_calendar.py`: `Calendar.identity()` (new method).
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py:58-84, 1064-1168, 1489-1547`: the memos.
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py:3210-3222`: `_observation_cache_key`.
- Modify: `quantark/asset/equity/engine/pde/grid/binder.py:40-80`: `validate_external_layout`.
- Modify: `quantark/intraday/engines/pde.py:143-159, 251`: delete `_release_market_memo` and both of its calls.
- Test: `test/pde_core/test_market_memos.py` (new); `test/intraday/test_route_pde.py` (drop the two `_ENV_STEP_COEFF_MEMO` assertions that test the pop, and keep the pricing assertions).

**Interfaces:**
- Produces `Calendar.identity() -> tuple`, of the form `(name, tuple(sorted(d.isoformat() for d in holidays)), tuple(sorted(weekend_days)))`.
- Produces `effective_settlement_calendar(product, env)`, defined in `quantark/asset/equity/settlement.py`, next to the resolver. It returns `(product.settlement_convention.calendar if product.settlement_convention else None) or getattr(env, "calendar", None)`. `SettlementResolver._resolve_payment:392` is rewritten to call it, so one rule serves both the resolver and the keys. Beside it, add `calendar_identity(cal) -> Optional[tuple]`, which returns `None` for `None` and `cal.identity()` otherwise.
- Produces `market_fingerprints(env, *, spot=True) -> dict[str, Optional[str]]`, defined in `base_pde_solver.py`. It returns `{"rate": try_fingerprint(env.rate_curve), "div": try_fingerprint(env.div_yield), "vol": try_fingerprint(env.vol_surface), "spot": float(env.spot).hex()}`.
- Task 4 moves these keys into `core/identity.py` unchanged.

- [ ] **Step 1: Write the failing tests.** Put them in `test/pde_core/test_market_memos.py`.

```python
"""Market memos key on values, not object identity (spec 2 decisions 14, 18, 19)."""
from datetime import datetime, timedelta

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.modelvalidation.builders.equity_snowball import make_snowball

SB = dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
          rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0)


def _env(rate=0.03):
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
                              rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=datetime(2026, 1, 5))


def test_mutating_the_rate_after_a_solve_resamples():
    p = make_snowball(SB)
    env = _env(0.03)
    s = SnowballPDESolver(PDEParams())
    s.price(p, env)
    env.rate_curve.rate = 0.07
    live = s.price(p, env)
    fresh = SnowballPDESolver(PDEParams()).price(p, _env(0.07))
    assert live == fresh


def test_discount_memo_follows_the_live_rate():
    s = SnowballPDESolver(PDEParams())
    env = _env(0.03)
    s._df_between_times(env, 0.0, 1.0)
    env.rate_curve.rate = 0.07
    assert s._df_between_times(env, 0.0, 1.0) == SnowballPDESolver(PDEParams())._df_between_times(_env(0.07), 0.0, 1.0)


class _Opaque:
    """A market object with no value fingerprint: never memoized."""
    def __init__(self, inner):
        self._inner = inner
    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_unfingerprintable_market_object_is_sampled_fresh_every_time():
    p = make_snowball(SB)
    env = _env()
    env.vol_surface = _Opaque(FlatVolSurface(volatility=0.2))
    s = SnowballPDESolver(PDEParams())
    a = s.price(p, env)
    env.vol_surface._inner = FlatVolSurface(volatility=0.3)
    b = s.price(p, env)
    assert a != b


def test_a_holiday_added_to_the_environment_calendar_re_resolves_settlement():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    env = _env(); env.calendar = cal
    p = make_snowball(SB)
    s = SnowballPDESolver(PDEParams())
    k1 = s._observation_cache_key(env, p, "ko")
    cal.holidays.add(datetime(2026, 2, 5))
    assert s._observation_cache_key(env, p, "ko") != k1


def test_calendar_identity_names_holidays_and_weekend():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    ident = cal.identity()
    cal.holidays.add(datetime(2026, 2, 5))
    assert cal.identity() != ident
```

Also add a spot-reading vol-surface double. It is a `FlatVolSurface` subclass whose `get_vol(strike, t, spot=None)` returns `0.2 + 1e-3 * (spot or 0) / 100`. Assert that a spot change re-samples: the prices at spots 100 and 101 differ from the prices you get by reusing the memo. Check `TermCoefficients.from_env`'s `get_vol` call for the exact keyword it passes the spot as, and match it.

- [ ] **Step 2: Run the tests to verify they fail.** Expected:
  - the rate test FAILS (it returns memoized coefficients);
  - the discount-factor test FAILS;
  - the calendar-key test FAILS (the key has no calendar);
  - `identity` FAILS with `AttributeError`.
- [ ] **Step 3: Implement.**
  1. Add `Calendar.identity()` as specified under Interfaces. Give it a docstring stating that the tuple is the calendar's value identity: `repr` shows only a name and a holiday count, so it is not an identity.
  2. In `_build_step_coefficients`:
     - Replace the `_per_object_memo(rate_curve)` outer lookup and the `is` checks with one module dict, `_ENV_STEP_COEFF_MEMO: "OrderedDict[tuple, StepCoefficients]"`.
     - Key it on `(fps["rate"], fps["div"], fps["vol"], fps["spot"], float(ref_strike), t_vec.tobytes(), dx_vec.tobytes(), int(num_x))`, where `fps = market_fingerprints(env)`.
     - If any of the first four entries is `None`, skip the memo: build fresh and store nothing.
     - Keep FIFO eviction at `_STEP_COEFF_MEMO_MAX_ENTRIES`.
  3. For `_df_between_times` and `_carry_df_between_times`:
     - The outer key is the fingerprint of `env.rate_curve` (DF) or of `env.div_yield` (carry).
     - Keep the inner keys unchanged (`(round(s,12), round(e,12))` and `("q", …)`).
     - `None` means no memo.
     - Keep the `_DF_MEMO_MAX_ENTRIES` clear-when-full rule.
     - Delete `_per_object_memo` and its weakref callback.
     - **Performance guard:** a fingerprint costs one sha256 per market object. Compute it once per solve and pass it down. Do this through a solver-level `self._market_fp` set at the top of `_solve` / `_compute_event_stats` and cleared in their `finally`. Do not recompute it per DF call. Task 5 moves this into `prepare`.
  4. Rewrite `_observation_cache_key` to return `(kind, strategy, class path, product token, valuation_date, day_count_convention, bus_days_in_year, calendar_identity(effective_settlement_calendar(product, env)))`.
  5. In `validate_external_layout`, also require `tuple(frozen.request.extra_times) == tuple(request.extra_times)`. On a mismatch, raise the same `PricingError` family it raises for event times. `resolve_bound_layout` already rebinds when events differ, so extend its "same tau and events" test to "same tau, events and extras".
  6. In the intraday route, delete `_release_market_memo` and its two call sites. The pop existed to work around the identity key.
- [ ] **Step 4: Run the tests.** Run `test/pde_core/test_market_memos.py test/intraday/test_route_pde.py test/pde_grid -q`. Expected: PASS.
- [ ] **Step 5: Run the byte gate and the identity gate.** Expected: 0 diffs, 0 violations. The flat environments' objects are dataclasses, so they fingerprint and stay memoized. If `make_term_env("kinked")`'s curve is not a dataclass, it now samples fresh every call. That is bitwise the same, but slower. Note it for Task 14's timing gate.
- [ ] **Step 6: Commit.** Message: `fix(pde): market memos keyed by value fingerprints; observation keys carry the effective settlement calendar; frozen layouts compare extra_times (spec 2 decisions 14, 19; §5.2 step 5)`.

---

### Task 4: Core scaffolding. Groups, backends, the factorization cache, sampler and stencil, the stepper, readout, identities, and legacy delegates

In this task every legacy path starts running on the new arithmetic objects, but every loop stays where it is. The legacy methods become one-line delegates, so from here on the byte gate proves the objects bitwise before any loop moves.

**Files:**
- Create: `quantark/asset/equity/engine/pde/core/__init__.py`, `groups.py`, `backends.py`, `samples.py`, `coefficients.py`, `components.py`, `identity.py`.
- Modify: `base_pde_solver.py`. `_calculate_coefficients`, `_build_step_coefficients`, `_flat_exact_step_coefficients`, `_build_operator_matrix`, `_operator_matrix_for_set`, `_get_matrices`, and `_theta_schedule_from_layout` become delegates. `_matrix_cache` becomes a read-only property.
- Modify: `snowball_pde_solver.py`. `_get_banded_system` and `_pack_banded_entry` become delegates. `_banded_cache` becomes a read-only property.
- Modify: `snowball_vol_pde_solvers.py:139-208` and `phoenix_vol_pde_solvers.py:115-188`. The LV `_build_step_coefficients` and `_local_vol_coefficients` become delegates.
- Modify: `grid/binder.py`. Add `Layout.fingerprint`.
- Test: `test/pde_core/test_march_step.py`, `test/pde_core/test_factorization_cache.py` (extend), `test/pde_core/test_review_focus.py` (new; Review Focus 5), `test/pde_grid/test_time_builder.py` (a schedule-equality test).

**Interfaces (produced; later tasks use these exact names):**

```python
# core/groups.py
class Backend(Enum): SPARSE_LU = "sparse_lu"; BANDED = "banded"
class BoundaryPolicy(Enum): DIRICHLET = "dirichlet"; CARRY = "carry"
@dataclass(frozen=True) class GroupPolicy: boundary: BoundaryPolicy; zero_diffusion_theta_one: bool; zero_slope_after_events: bool
VALUE: GroupPolicy          # DIRICHLET, override on, no zero-slope
VALUE_NO_ZERO_DIFFUSION_OVERRIDE: GroupPolicy   # Phoenix (§8)
STATISTICS_LEGACY: GroupPolicy                  # CARRY, override off, zero-slope
@dataclass(frozen=True) class ColumnGroup: name: str; surfaces: tuple[str, ...]; backend: Backend; policy: GroupPolicy; one_dimensional_rhs: bool = False
@dataclass(frozen=True, eq=False) class GroupState: group: ColumnGroup; v_next: np.ndarray  # (num_x, n_cols), read-only
# core/backends.py
class FactorizationCache(enabled=True, banded_max_entries=BANDED_CACHE_MAX_ENTRIES, lu_seed=None, banded_seed=None)
    .key(set_index, dt, theta) -> tuple ; .get(backend, set_index, dt, theta, build) ; .clear(backend=None) ; .store(backend) -> Mapping
def operator_matrix(l, c, u, num_x) -> sp.csc_matrix
def sparse_lu_factor(A, dt, theta) -> tuple[sp.csc_matrix, SuperLU]
def banded_factor(l, c, u, dt, theta) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def sparse_lu_advance(state: GroupState, s: "StepSet", dt, theta, boundaries) -> np.ndarray   # (n_int, n_cols)
def banded_advance(state: GroupState, s: "StepSet", dt, theta, boundaries) -> np.ndarray      # (n_int, n_cols) or (n_int,)
# core/samples.py
class SampleKind(Enum): SCALAR = "scalar"; NODAL = "nodal"
@dataclass(frozen=True, eq=False) class MarketSamples: kind; r_tau; q_tau; sigma_tau; t; fwd_rates; fwd_carry; step_vols; vol_rows; provenance
def sample_bsm(env, t_vec, *, ref_strike, r_tau, q_tau, sigma_tau) -> MarketSamples
def sample_local_vol(env, t_vec, s_vec, surface, *, ref_strike, r_tau, q_tau, sigma_tau) -> MarketSamples
# core/coefficients.py
class StepCoefficients(NamedTuple)          # MOVED here from base_pde_solver (re-exported there)
@dataclass(eq=False) class CoefficientsInForce: flat; sets; num_x  .set_at(j) .lcu(k) .is_zero_diffusion(k) .operator(k) .step_set(k, cache)
@dataclass(frozen=True, eq=False) class StepSet: index; l; c; u; operator: Callable; factorizations: FactorizationCache
# core/components.py
class OperatorFactory(Protocol); class TimeStepper(Protocol); class Readout(Protocol)
@dataclass(frozen=True) class CentralDifference: supports = frozenset({SCALAR, NODAL})
    .assemble(samples, layout) ; .assemble_arrays(samples, dx_vec, num_x) ; .flat(samples, dx_vec, num_x) ; .evidence() -> {}
@dataclass(frozen=True, kw_only=True) class CrankNicolson: theta=0.5; event_theta=1.0; terminal_damping_steps=None; event_damping_steps=None
    .resolved(damping_defaults: tuple[int, int]) -> CrankNicolson ; .schedule(layout) -> np.ndarray ; .step(state, s, dt, theta, boundaries) -> np.ndarray ; .evidence() -> dict
@dataclass(frozen=True) class StandardReadout: .price(column, spot, layout) ; .delta_gamma(column, spot, layout) ; .evidence() -> {}
def scheme_stepper(params, config) -> CrankNicolson        # 0.6.0 home of the four settings; replaced in Task 12
def scalar_stencil(r, q, sigma, dx_vec, num_x) ; def nodal_stencil(r, q, sigma_vec, dx_vec, num_x) ; def flat_exact(sc, flat) -> StepCoefficients
def validate_schedule(schedule, layout, component) ; validate_coefficients(sc, n_steps, num_x, component) ; validate_advanced(out, group, n_int, component) ; validate_readout(value, component)
# core/identity.py
def layout_fingerprint(layout) -> str ; def component_fingerprint(obj) -> Optional[str]   # = try_fingerprint
def market_fingerprints(env) -> dict[str, Optional[str]] ; def coefficient_key(layout_fp, market_fps, ref_strike, operator_fp, samples_fp) -> Optional[tuple]
# grid/binder.py
Layout.fingerprint  (functools.cached_property -> layout_fingerprint(self))
```

- [ ] **Step 1: Write the failing tests.**

`test/pde_core/test_march_step.py`. The kernels are checked against the legacy formulas, which are written out inside the test. The comparison is at `golden_compare` tolerance and uses no frozen arrays. The exact gate is the byte book.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from golden_compare import assert_close
from quantark.asset.equity.engine.pde.core.backends import (
    FactorizationCache, banded_advance, operator_matrix, sparse_lu_advance)
from quantark.asset.equity.engine.pde.core.coefficients import StepSet
from quantark.asset.equity.engine.pde.core.groups import (
    STATISTICS_LEGACY, VALUE, Backend, ColumnGroup, GroupState)

RNG = np.random.default_rng(7)
N = 12


def _lcu():
    l = 0.8 + 0.1 * RNG.random(N); c = -2.0 - 0.1 * RNG.random(N); u = 0.9 + 0.1 * RNG.random(N)
    return l, c, u


def _set(l, c, u):
    return StepSet(index=0, l=l, c=c, u=u, operator=lambda: operator_matrix(l, c, u, N),
                   factorizations=FactorizationCache())


def _legacy_banded(l, c, u, dt, theta, v_next, lower, upper):
    lower_d = -theta * dt * l[2:-1]; main = 1.0 - theta * dt * c[1:-1]; upper_d = -theta * dt * u[1:-2]
    ab = np.zeros((3, len(main))); ab[0, 1:] = upper_d; ab[1, :] = main; ab[2, :-1] = lower_d
    lower1 = (1.0 - theta) * dt * l[2:-1]; main1 = 1.0 + (1.0 - theta) * dt * c[1:-1]; upper1 = (1.0 - theta) * dt * u[1:-2]
    cols = []
    for k in range(v_next.shape[1]):
        vn = v_next[1:-1, k]
        rhs = main1 * vn; rhs[1:] += lower1 * vn[:-1]; rhs[:-1] += upper1 * vn[1:]
        rhs[0] += dt * ((1.0 - theta) * l[1] * v_next[0, k] + theta * l[1] * lower[k])
        rhs[-1] += dt * ((1.0 - theta) * u[-2] * v_next[-1, k] + theta * u[-2] * upper[k])
        cols.append(solve_banded((1, 1), ab, rhs))
    return np.column_stack(cols)


def test_two_column_banded_group_matches_the_legacy_step():
    l, c, u = _lcu(); v = RNG.random((N, 2)); lo = RNG.random(2); up = RNG.random(2)
    g = ColumnGroup("value", ("alive", "ki"), Backend.BANDED, VALUE)
    out = banded_advance(GroupState(g, v), _set(l, c, u), np.float64(0.01), 0.5, (lo, up))
    assert_close(out, _legacy_banded(l, c, u, 0.01, 0.5, v, lo, up))


def test_statistics_group_matches_the_legacy_step():
    l, c, u = _lcu(); v = RNG.random((N, 6))
    g = ColumnGroup("stats", ("stats_v0", "stats_v1"), Backend.BANDED, STATISTICS_LEGACY)
    out = banded_advance(GroupState(g, v), _set(l, c, u), 0.01, 1.0, (v[0], v[-1]))
    assert_close(out, _legacy_banded(l, c, u, 0.01, 1.0, v, v[0], v[-1]))


def test_single_column_lu_matches_the_legacy_step():
    l, c, u = _lcu(); v = RNG.random((N, 1)); lo = RNG.random(1); up = RNG.random(1)
    g = ColumnGroup("value", ("value",), Backend.SPARSE_LU, VALUE)
    out = sparse_lu_advance(GroupState(g, v), _set(l, c, u), 0.01, 0.5, (lo, up))
    A = operator_matrix(l, c, u, N); I = sp.eye(N - 2, format="csc")
    M1 = I + 0.5 * 0.01 * A; lu = spla.splu(I - 0.5 * 0.01 * A)
    rhs = M1 @ v[1:-1, 0]
    rhs[0] += 0.01 * (0.5 * l[1] * v[0, 0] + 0.5 * l[1] * lo[0])
    rhs[-1] += 0.01 * (0.5 * u[-2] * v[-1, 0] + 0.5 * u[-2] * up[0])
    assert_close(out[:, 0], lu.solve(rhs))
```

Extend `test/pde_core/test_factorization_cache.py`:

```python
from quantark.asset.equity.engine.pde.core.backends import FactorizationCache
from quantark.asset.equity.engine.pde.core.groups import Backend


def test_cache_keys_theta_exact_and_dt_rounded():
    fc = FactorizationCache()
    calls = []
    build = lambda: calls.append(1) or object()
    fc.get(Backend.BANDED, 0, np.float64(1 / 12), 0.5, build)
    fc.get(Backend.BANDED, 0, np.nextafter(np.float64(1 / 12), 1.0), 0.5, build)
    fc.get(Backend.BANDED, 0, np.float64(1 / 12), 0.5000001, build)
    fc.get(Backend.SPARSE_LU, 0, np.float64(1 / 12), 0.5, build)
    assert len(calls) == 3


def test_banded_store_is_lru_bounded_and_lu_store_unbounded():
    fc = FactorizationCache(banded_max_entries=2)
    for k in range(3):
        fc.get(Backend.BANDED, k, 0.1, 0.5, object)
        fc.get(Backend.SPARSE_LU, k, 0.1, 0.5, object)
    assert list(fc.store(Backend.BANDED)) == [fc.key(1, 0.1, 0.5), fc.key(2, 0.1, 0.5)]
    assert len(fc.store(Backend.SPARSE_LU)) == 3


def test_seed_is_consulted_first_even_when_disabled():
    sentinel = object()
    fc = FactorizationCache(enabled=False, lu_seed={FactorizationCache.key(0, 0.1, 0.5): sentinel})
    assert fc.get(Backend.SPARSE_LU, 0, 0.1, 0.5, object) is sentinel
    assert fc.get(Backend.SPARSE_LU, 1, 0.1, 0.5, object) is not fc.get(Backend.SPARSE_LU, 1, 0.1, 0.5, object)


def test_monthly_snowball_five_dt_floats_resolve_to_one_key():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    from quantark.modelvalidation.builders.equity_snowball import make_snowball
    from term_structure_benchmarks import make_term_env
    sb = make_snowball(dict(initial_price=100.0, strike=100.0, ko_barrier=103.0, ki_barrier=85.0, ko_rate=0.15,
                            rebate_rate=0.15, months=12, maturity=1.0, contract_multiplier=1.0))
    s = SnowballPDESolver(PDEParams()); s.price(sb, make_term_env("flat"))
    dts = {float(d) for d in s._active_layout.time.dt}
    keys = {FactorizationCache.key(0, np.float64(d), 0.5)[1] for d in dts}
    assert len(dts) >= 2 and len(keys) < len(dts)   # measured §14: 5 floats -> 1 key
```

Add to `test/pde_grid/test_time_builder.py`. The stepper's schedule must equal today's `TimeLayout` damping sets on every fixture:

```python
@pytest.mark.parametrize("term,event", [(1, 2), (0, 1), (3, 0), (40, 40)])
def test_stepper_schedule_equals_the_time_layout_schedule(term, event):
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson
    req = GridRequest(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0,),
                      event_times=(0.25, 0.5, 0.75), extra_times=(0.6,))
    cfg = resolve_config("standard", GridConfig(time_steps=20, terminal_damping_steps=term, event_damping_steps=event))
    time = build_time(req, cfg)
    layout = SimpleNamespace(time=time, request=req)
    got = CrankNicolson(theta=0.5, event_theta=0.7, terminal_damping_steps=term, event_damping_steps=event).schedule(layout)
    want = np.full(time.actual_steps, 0.5)
    for k in time.event_damping_steps: want[k] = 0.7
    for k in time.terminal_damping_steps: want[k] = 1.0
    assert np.array_equal(got, want)
```

`test/pde_core/test_review_focus.py` (Review Focus 5):

```python
def test_cache_disabled_agrees_with_cache_enabled_to_golden_tolerance():
    from golden_compare import assert_close
    p = make_snowball(SB); env = make_term_env("kinked")
    on = SnowballPDESolver(PDEParams()).price(p, env)
    off = SnowballPDESolver(PDEParams(cache_enabled=False)).price(p, env)
    assert_close(off, on)   # not ==: the rounded dt key shares ULP-different intervals (§5.7)
```

Here `SB` is the dict from `test_market_memos.py`. Import it, or repeat it as a module constant.

- [ ] **Step 2: Run the tests.** `… -m pytest test/pde_core test/pde_grid/test_time_builder.py -v`. Expected: `ModuleNotFoundError` for `…pde.core`. The review-focus test passes already. It pins the current behaviour.

- [ ] **Step 3: Implement `core/groups.py`.**

```python
"""Column groups and their policies (spec 2 §5.4, §5.5, decision 7).

A column group is exactly what one solve call advances today; the grouping,
the backend and the operation order inside a step decide bit-identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["Backend", "BoundaryPolicy", "GroupPolicy", "ColumnGroup", "GroupState",
           "VALUE", "VALUE_NO_ZERO_DIFFUSION_OVERRIDE", "STATISTICS_LEGACY"]


class Backend(Enum):
    SPARSE_LU = "sparse_lu"
    BANDED = "banded"


class BoundaryPolicy(Enum):
    DIRICHLET = "dirichlet"   # the adapter writes rows 0/-1 of step j before the solve
    CARRY = "carry"           # rows 0/-1 of step j are step j+1's (statistics-legacy)


@dataclass(frozen=True)
class GroupPolicy:
    boundary: BoundaryPolicy
    zero_diffusion_theta_one: bool    # θ=1 on zero-diffusion coefficient sets
    zero_slope_after_events: bool     # v[0]=v[1], v[-1]=v[-2] after the step's events


VALUE = GroupPolicy(BoundaryPolicy.DIRICHLET, True, False)
VALUE_NO_ZERO_DIFFUSION_OVERRIDE = GroupPolicy(BoundaryPolicy.DIRICHLET, False, False)
STATISTICS_LEGACY = GroupPolicy(BoundaryPolicy.CARRY, False, True)


@dataclass(frozen=True)
class ColumnGroup:
    """``surfaces`` name the state arrays this group advances, in right-hand-side column order."""
    name: str
    surfaces: tuple
    backend: Backend
    policy: GroupPolicy
    one_dimensional_rhs: bool = False   # Phoenix banded: a 1-D right-hand side (nrhs=1)


@dataclass(frozen=True, eq=False)
class GroupState:
    """What a stepper sees: the group and its columns at step j+1 (read-only)."""
    group: ColumnGroup
    v_next: np.ndarray
```

- [ ] **Step 4: Implement `core/backends.py`.** Every arithmetic line is copied from the legacy source cited in its comment. **Do not reorder any operand.**

```python
"""The two private arithmetic backends and the factorization cache (spec 2 §5.7).

Keys: (set index, round(dt, 12), θ) per backend. θ is exact (decision 6). dt
is rounded exactly as each caller passes it (np.float64 on the value marches,
float on the statistics march): intervals of nearly equal length differ by
ULPs and share one factorization, first build wins -- every banked number was
computed that way. LU store unbounded, banded store LRU-bounded, as today.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Mapping, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from quantark.util.exceptions import NumericalError

from .groups import Backend, GroupState

BANDED_CACHE_MAX_ENTRIES = 512


class FactorizationCache:
    def __init__(self, *, enabled: bool = True, banded_max_entries: int = BANDED_CACHE_MAX_ENTRIES,
                 lu_seed: Optional[Mapping] = None, banded_seed: Optional[Mapping] = None):
        self.enabled = enabled
        self._banded_max = banded_max_entries
        self._stores = {Backend.SPARSE_LU: {}, Backend.BANDED: OrderedDict()}
        self._seeds = {Backend.SPARSE_LU: lu_seed or {}, Backend.BANDED: banded_seed or {}}

    @staticmethod
    def key(set_index, dt, theta) -> tuple:
        return (int(set_index), round(dt, 12), float(theta))

    def get(self, backend: Backend, set_index, dt, theta, build: Callable):
        key = self.key(set_index, dt, theta)
        hit = self._seeds[backend].get(key)
        if hit is not None:
            return hit
        if not self.enabled:
            return build()
        store = self._stores[backend]
        if key in store:
            if backend is Backend.BANDED:
                store.move_to_end(key)
            return store[key]
        value = build()
        store[key] = value
        if backend is Backend.BANDED:
            while len(store) > self._banded_max:
                store.popitem(last=False)
        return value

    def clear(self, backend: Optional[Backend] = None) -> None:
        for b, store in self._stores.items():
            if backend is None or b is backend:
                store.clear()

    def store(self, backend: Backend) -> Mapping:
        return self._stores[backend]


def operator_matrix(l, c, u, num_x):
    # base_pde_solver.py:1231-1238 (_build_operator_matrix), verbatim
    return sp.diags([l[2:-1], c[1:-1], u[1:-2]], offsets=[-1, 0, 1],
                    shape=(num_x - 2, num_x - 2), format="csc")


def sparse_lu_factor(A, dt, theta):
    # base_pde_solver.py:1320-1364 (_get_matrices build branch), verbatim
    I = sp.eye(A.shape[0], format="csc")
    M1 = I + (1.0 - theta) * dt * A
    M2 = I - theta * dt * A
    try:
        return M1, spla.splu(M2)
    except Exception as exc:  # the legacy message and type
        raise NumericalError(f"PDE LU factorization failed: {exc}") from exc


def banded_factor(l, c, u, dt, theta):
    # snowball_pde_solver.py:2690-2742 (_get_banded_system build branch), verbatim
    lower = -theta * dt * l[2:-1]
    main = 1.0 - theta * dt * c[1:-1]
    upper = -theta * dt * u[1:-2]
    banded = np.zeros((3, len(main)))
    banded[0, 1:] = upper
    banded[1, :] = main
    banded[2, :-1] = lower
    lower1 = (1.0 - theta) * dt * l[2:-1]
    main1 = 1.0 + (1.0 - theta) * dt * c[1:-1]
    upper1 = (1.0 - theta) * dt * u[1:-2]
    return banded, lower1, main1, upper1


def _inject(rhs, l1, u2, vn_lo, vn_up, lo, up, dt, theta):
    # base_pde_solver.py:1297-1307 (_inject_boundary_contributions): the formula and its operand order
    rhs[0] += dt * ((1.0 - theta) * l1 * vn_lo + theta * l1 * lo)
    rhs[-1] += dt * ((1.0 - theta) * u2 * vn_up + theta * u2 * up)


def sparse_lu_advance(state: GroupState, s, dt, theta, boundaries):
    """Each column in order: rhs = M1 @ v, inject, solve (base march; snowball/phoenix LU fallback)."""
    M1, M2_lu = s.factorizations.get(Backend.SPARSE_LU, s.index, dt, theta,
                                     lambda: sparse_lu_factor(s.operator(), dt, theta))
    v = state.v_next
    lo, up = boundaries
    out = np.empty((v.shape[0] - 2, v.shape[1]))
    for k in range(v.shape[1]):
        rhs = M1 @ v[1:-1, k]
        if v.shape[0] > 2:
            _inject(rhs, s.l[1], s.u[-2], v[0, k], v[-1, k], lo[k], up[k], dt, theta)
        out[:, k] = M2_lu.solve(rhs)
    return out


def banded_advance(state: GroupState, s, dt, theta, boundaries):
    """One solve_banded call for the whole group (decision 7)."""
    banded, lower1, main1, upper1 = s.factorizations.get(
        Backend.BANDED, s.index, dt, theta, lambda: banded_factor(s.l, s.c, s.u, dt, theta))
    v = state.v_next
    lo, up = boundaries
    if state.group.one_dimensional_rhs:
        # phoenix_pde_solver.py:750-778 (_step_grids banded branch)
        vn = v[1:-1, 0]
        rhs = np.empty(len(vn))
        np.multiply(main1, vn, out=rhs)
        rhs[1:] += lower1 * vn[:-1]
        rhs[:-1] += upper1 * vn[1:]
        if v.shape[0] > 2:
            _inject(rhs, s.l[1], s.u[-2], v[0, 0], v[-1, 0], lo[0], up[0], dt, theta)
        return solve_banded((1, 1), banded, rhs, overwrite_b=True, check_finite=False)[:, None]
    # snowball_pde_solver.py:2592-2630 (value, two columns) and 1071-1107 (statistics)
    vn = v[1:-1, :]
    rhs = np.empty(vn.shape)
    rhs[:] = main1[:, None] * vn
    rhs[1:, :] += lower1[:, None] * vn[:-1, :]
    rhs[:-1, :] += upper1[:, None] * vn[1:, :]
    if v.shape[0] > 2:
        _inject(rhs, s.l[1], s.u[-2], v[0, :], v[-1, :], lo, up, dt, theta)
    return solve_banded((1, 1), banded, rhs, overwrite_b=True, check_finite=False)
```

**Bit-identity notes for the reviewer.**
- The legacy two-column path writes each column with `np.multiply(main1, v0_next, out=rhs_v0)`. The statistics path writes `rhs_v0[:] = main1[:, None] * v0n`. Both perform the same elementwise IEEE operations.
- The legacy statistics path casts `float(l[1])` and `float(dt)`. That is the same value as the `np.float64` operands used here.
- `overwrite_b` does not change the result.
- The value two-column group is still one `(n_int, 2)` call, and statistics is still one `(n_int, 2·n_cols)` call (decision 7).
- If the byte gate ever shows a diff attributable to a kernel, first suspect a changed operand order against the cited lines.

- [ ] **Step 5: Implement `core/samples.py`, `core/coefficients.py`, and `core/components.py`.**

`samples.py`:
- `MarketSamples` is a frozen dataclass with fields:
  - `kind: SampleKind`
  - `r_tau`, `q_tau`, `sigma_tau: float` (the scalar-stencil inputs at maturity)
  - `t: np.ndarray`
  - `fwd_rates`, `fwd_carry: np.ndarray` (shape `(n_steps,)`)
  - `step_vols: Optional[np.ndarray]` (SCALAR)
  - `vol_rows: Optional[np.ndarray]` (NODAL, shape `(n_steps, num_x)`)
  - `lv_surface: Optional[object]`
  - `provenance: Mapping[str, Optional[str]]`
- `sample_bsm`:
  - `tc = TermCoefficients.from_env(env, np.asarray(t_vec, dtype=float), ref_strike=float(ref_strike))`. This is `base_pde_solver.py:1092` verbatim.
  - `provenance = market_fingerprints(env)`.
- `sample_local_vol`:
  - `TermCoefficients.from_env` for rates and carries.
  - For each step `j`: `t_mid = t_vec[j] + 0.5 * dt_j`, `vol_rows[j] = surface.local_vol(s_vec, t_mid)`. This is `snowball_vol_pde_solvers.py:139-172` verbatim. Take the local-vol validation lines from `_local_vol_coefficients:177-190` and run them per row here.
  - `provenance = {**market_fingerprints(env), "lv_surface": try_fingerprint(surface)}`.
- `MarketSamples.first_passage(product, barrier) -> (mu, sig2)`:
  - SCALAR kind: the body of `SnowballPDESolver._first_passage_step_coefficients:587-609`, using `self.fwd_rates`, `self.fwd_carry`, `self.step_vols`.
  - NODAL kind: the body of `LocalVolBarrierCrossingMixin._first_passage_step_coefficients` (`vol_continuous_ki.py:28-61`), using `self.lv_surface`.
  - Verify first that `_first_passage_step_coefficients` reads `TermCoefficients.from_env(ref_strike=product.strike)` on the same `t`. The step-coefficient sample uses `ref_strike=strike` too, so reusing its arrays is bitwise. If the two `ref_strike`s differ on any route, keep a second `TermCoefficients` call inside `first_passage` rather than reuse the arrays.

`coefficients.py`:
- Move `StepCoefficients` here from `base_pde_solver.py:87-104`, and re-export it from `base_pde_solver` (tests import it there).
- `CoefficientsInForce`:

```python
@dataclass(eq=False)
class CoefficientsInForce:
    """The stencil a march uses: the scalar stencil when one set remains (today's
    ``step_coeffs = None if sc.n_unique == 1``), else the per-step sets."""
    flat: tuple                      # (l, c, u) from scalar_stencil(r_tau, q_tau, sigma_tau)
    sets: Optional[StepCoefficients]
    num_x: int
    _operators: dict = field(default_factory=dict)

    def set_at(self, j: int) -> int:
        return 0 if self.sets is None else int(self.sets.set_index[j])

    def lcu(self, k: int) -> tuple:
        return self.flat if self.sets is None else self.sets.lcu_sets[k]

    def is_zero_diffusion(self, k: int) -> bool:
        return self.sets is not None and k in self.sets.zero_diffusion_sets

    def operator(self, k: int):
        if k not in self._operators:
            self._operators[k] = operator_matrix(*self.lcu(k), self.num_x)
        return self._operators[k]

    def step_set(self, k: int, cache: FactorizationCache) -> StepSet:
        l, c, u = self.lcu(k)
        return StepSet(k, l, c, u, lambda: self.operator(k), cache)
```

This one rule also reproduces the local-vol single-step oddity bit for bit: with a one-step layout, `n_unique == 1` and the march uses the scalar Black-Scholes stencil. Today it does the same (`snowball_vol_pde_solvers.py` `_flat_exact` returns `sc` unchanged, and `_solve` then discards it). §8 does not list this oddity. It is recorded in this plan's self-review notes.

`components.py`:
- `scalar_stencil`: the body of `_calculate_coefficients:1197-1229`, verbatim.
- `nodal_stencil`: the body of `_local_vol_coefficients:177-208`, verbatim, minus the validation lines that moved to the sampler.
- `flat_exact(sc, flat)`: the body of `_flat_exact_step_coefficients:1170-1191`, verbatim (it drops `zero_diffusion_sets`, as today).
- `CentralDifference.assemble_arrays(samples, dx_vec, num_x)`:
  - SCALAR: lines `1093-1162` of `_build_step_coefficients` from `triples = …` to the `StepCoefficients(...)` return, reading `samples.fwd_rates`, `samples.fwd_carry` and `samples.step_vols`. Then return `flat_exact(sc, scalar_stencil(samples.r_tau, samples.q_tau, samples.sigma_tau, dx_vec, num_x))`.
  - NODAL: `StepCoefficients(lcu_sets=[nodal_stencil(samples.fwd_rates[j], samples.fwd_carry[j], samples.vol_rows[j], dx_vec, num_x) for j in range(n)], set_index=np.arange(n), n_unique=n)`, unchanged by `flat_exact`.
- `CentralDifference.assemble(samples, layout)` calls `assemble_arrays(samples, layout.spatial.dx, len(layout.spatial.x))`.
- `CentralDifference.flat(...)` calls `scalar_stencil(...)`.
- The protocols follow §5.8, with one change: `TimeStepper.step(...)` **returns** the advanced interior columns (`np.ndarray`), because §4.1 says so. The `-> None` in §5.8's code block contradicts §4.1 and the "advanced columns" validation.
- `CrankNicolson`:

```python
@dataclass(frozen=True, kw_only=True)
class CrankNicolson:
    """The built-in stepper: θ-weighted time stepping with terminal and event damping.

    ``None`` damping counts are the route's defaults (1 and 2 on the core routes,
    2 and 2 on DCN), resolved once by ``prepare``.
    """
    theta: float = 0.5
    event_theta: float = 1.0
    terminal_damping_steps: Optional[int] = None
    event_damping_steps: Optional[int] = None

    def __post_init__(self):
        if not 0.0 <= float(self.theta) <= 1.0:
            raise ValidationError(f"theta must be in [0, 1], got {self.theta}")
        if not 0.0 < float(self.event_theta) <= 1.0:
            raise ValidationError(f"event_theta must be in (0, 1], got {self.event_theta}")
        for name in ("terminal_damping_steps", "event_damping_steps"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValidationError(f"{name} must be a non-negative int or None, got {v!r}")

    def resolved(self, damping_defaults: tuple) -> "CrankNicolson":
        term, event = damping_defaults
        return replace(self,
                       terminal_damping_steps=term if self.terminal_damping_steps is None else self.terminal_damping_steps,
                       event_damping_steps=event if self.event_damping_steps is None else self.event_damping_steps)

    def _damping_sets(self, layout) -> tuple:
        # grid/time.py:148-161 (build_time), verbatim: damping follows EVENT nodes only
        if self.terminal_damping_steps is None or self.event_damping_steps is None:
            raise ValidationError("CrankNicolson.schedule needs resolved damping counts; call resolved() first")
        step_of = layout.time.step_of
        event_damp = frozenset(
            step_of[e] - j
            for e in layout.request.event_times
            for j in range(1, int(self.event_damping_steps) + 1)
            if step_of[e] - j >= 0
        )
        n_steps = len(layout.time.dt)
        term_damp = frozenset(
            n_steps - j
            for j in range(1, int(self.terminal_damping_steps) + 1)
            if n_steps - j >= 0
        )
        return event_damp, term_damp

    def schedule(self, layout) -> np.ndarray:
        # base_pde_solver.py:651-664 (_theta_schedule_from_layout), verbatim order
        event_damp, term_damp = self._damping_sets(layout)
        theta = np.full(layout.time.actual_steps, float(self.theta))
        for k in event_damp:
            theta[k] = float(self.event_theta)
        for k in term_damp:
            theta[k] = 1.0  # terminal wins on overlap
        return theta

    def step(self, state, s, dt, theta, boundaries):
        if state.group.backend is Backend.BANDED:
            return banded_advance(state, s, dt, theta, boundaries)
        return sparse_lu_advance(state, s, dt, theta, boundaries)

    def evidence(self) -> dict:
        out = {"theta": float(self.theta), "event_theta": float(self.event_theta)}
        for name in ("terminal_damping_steps", "event_damping_steps"):
            if getattr(self, name) is not None:
                out[name] = int(getattr(self, name))
        return out
```

`build_time` iterates `request.event_times` (`time.py:74`), and `layout.request` is that same request object, so these two sets equal `TimeLayout`'s frozensets by construction. Step 1's test proves it. Iterating a frozenset while assigning into `theta` is order-independent, because every index gets the same value.

- `scheme_stepper(params, config)` returns `CrankNicolson(theta=float(params.theta), event_theta=float(getattr(params, "event_theta", 1.0)), terminal_damping_steps=int(config.terminal_damping_steps), event_damping_steps=int(config.event_damping_steps))`. Its docstring says: "0.6.0 home of the four settings; Task 12 replaces it with `params.stepper.resolved(route.damping_defaults)`."
- `StandardReadout`:
  - `price` returns `float(np.interp(np.log(spot), layout.spatial.x, column))`, which is `_interpolate_price` with `x_target = np.log(spot)`.
  - `delta_gamma` is the body of `_calculate_delta_gamma:1411-1482`, verbatim, with `x_vec = layout.spatial.x` and `x_target = np.log(spot)`.
  - Both are also exposed as module functions `interpolate_price(v, x, x_target)` and `delta_gamma(v, x, x_target, spot)`, which the legacy methods delegate to.
- The validators, each raising `NumericalError(f"{type(component).__name__}: …")`:
  - `validate_schedule`: length equals `actual_steps`, finite, and every value is in [0, 1].
  - `validate_coefficients`:
    - `lcu_sets[k]` arrays have shape `(num_x,)` and are finite;
    - `set_index` has an integer dtype and length `n_steps`;
    - `0 <= set_index < n_unique`;
    - `zero_diffusion_sets ⊆ range(n_unique)`.
  - `validate_advanced`: the shape is `(n_int, n_cols)` (or `(n_int, 1)` for a 1-D group) and finite.
  - `validate_readout`: the value is finite.

- [ ] **Step 6: Implement `core/identity.py` and `Layout.fingerprint`.**

```python
def layout_fingerprint(layout) -> str:
    """sha256 over what the march consumes (decision 17): x, dx, t, dt, the event and
    extra node indices, config_key."""
    step_of = layout.time.step_of
    return fingerprint((
        np.asarray(layout.spatial.x), np.asarray(layout.spatial.dx),
        np.asarray(layout.time.t), np.asarray(layout.time.dt),
        tuple(sorted(int(step_of[e]) for e in layout.request.event_times)),
        tuple(sorted(int(step_of[e]) for e in layout.request.extra_times)),
        tuple(layout.config_key),
    ))


def component_fingerprint(obj) -> Optional[str]:
    return try_fingerprint(obj)   # decision 8: the one identity function


def market_fingerprints(env) -> dict:
    return {"rate": try_fingerprint(env.rate_curve), "div": try_fingerprint(env.div_yield),
            "vol": try_fingerprint(env.vol_surface), "spot": float(env.spot).hex()}


def coefficient_key(layout_fp, market_fps, ref_strike, operator_fp, lv_surface_fp=None):
    parts = (layout_fp, market_fps["rate"], market_fps["div"], market_fps["vol"],
             market_fps["spot"], float(ref_strike).hex(), operator_fp, lv_surface_fp)
    return None if any(p is None for p in parts[:7]) else parts
```

- `market_fingerprints` moves here from Task 3's `base_pde_solver` helper. The helper becomes an import.
- In `grid/binder.py`, `Layout` gets `fingerprint = functools.cached_property(lambda self: layout_fingerprint(self))`, importing lazily inside the lambda's body to avoid an import cycle. `Layout` is a frozen dataclass without `__slots__`, so `cached_property` writes straight to `__dict__`.
- If `config_key` contains a `None` inside a tuple, `canonical_tree` accepts it. Verify by pricing the book's `snowball_cont|flat|standard` in a scratch script and printing `layout.fingerprint`.

- [ ] **Step 7: Turn the legacy methods into delegates.**
  - `_calculate_coefficients` → `scalar_stencil`.
  - `_flat_exact_step_coefficients` → `flat_exact(sc, scalar_stencil(r, q, sigma, dx_vec, num_x))`.
  - `_build_operator_matrix` → `operator_matrix`.
  - `_interpolate_price` / `_calculate_delta_gamma` → the module functions.
  - `_build_step_coefficients`: keep Task 3's value-keyed memo. On a miss, run `sample_bsm(...)` and then `CentralDifference().assemble_arrays(samples, dx_vec, num_x)` **before** `flat_exact`. The legacy method returned the pre-flat-exact object and `_solve` applied flat-exact afterwards, so store and return the pre-flat-exact `StepCoefficients`. Expose that as `CentralDifference.assemble_sets(...)`, and have `assemble_arrays` call it and then apply `flat_exact`.
  - LV `_build_step_coefficients` → `sample_local_vol` + `CentralDifference().assemble_sets`. LV `_local_vol_coefficients` → `nodal_stencil` plus its validation.
  - Solver cache:
    - `BasePDESolver.__init__` creates `self._factorizations = FactorizationCache(enabled=self._cache_enabled)`.
    - `_get_matrices(I, A, dt, theta, coeff_key)` returns `self._factorizations.get(Backend.SPARSE_LU, coeff_key, dt, theta, lambda: sparse_lu_factor(A, dt, theta))`, with the session pack passed as `lu_seed` whenever `_session_matrix_pack` is set. To do that, rebuild `self._factorizations` in the `_session_matrix_pack` setter; make that attribute a property.
    - `_get_banded_system` is the same, with `Backend.BANDED` and `banded_seed=_session_banded_pack`.
    - `self._matrix_cache.clear()` / `self._banded_cache.clear()` become `self._factorizations.clear(Backend.SPARSE_LU / BANDED)`.
    - `_matrix_cache` / `_banded_cache` become read-only properties returning `self._factorizations.store(...)`, so Task 2's tests still see them.
    - `_bump_clone` resets `_factorizations` (instead of `_matrix_cache`).
  - `_theta_schedule_from_layout(layout)` returns `scheme_stepper(self.params, self.grid_binder.config).schedule(layout)`.
- [ ] **Step 8: Run the tests and the gates.** Run `… -m pytest test/pde_core test/pde_grid test/test_pde_term_coefficients.py test/test_pde_zero_diffusion_step.py test/execution -q`. Then run the byte gate and the identity gate. Expected: PASS; 0 diffs, 0 violations.
- [ ] **Step 9: Commit.** Message: `refactor(pde): core arithmetic objects - groups, backends, factorization cache, sampler/stencil, CrankNicolson schedule, readout, layout fingerprint; legacy methods delegate (spec 2 §5.7, §5.8, decision 17)`.

---

### Task 5: `prepare → march → read` for the base family and American; the knock-in composite

**Files:**
- Create: `core/prepare.py`, `core/march.py`, `core/readout.py`, `core/adapters/__init__.py`, `core/adapters/base.py`, `core/adapters/european.py`, `core/adapters/american.py`, `core/adapters/barrier.py`.
- Modify: `base_pde_solver.py`. Delete `_time_stepping` (1240-1295), `_solve` (804-886) and `_price_with_solution` (901-920), and the base bodies of `price` / `calculate_greeks` (922-975). `price` and `calculate_greeks` become `run(prepare(…))`. Add the `adapter` property.
- Modify: `european_pde_solver.py`, `american_pde_solver.py` (delete the `_time_stepping` override 284-362; its body moves), `barrier_pde_solver.py` (delete `_price_vanilla`, `_price_knock_out`, `_calculate_greeks_vanilla`, `_calculate_greeks_knock_out`, and the `_build_grids` override), `double_barrier_pde_solver.py`, `one_touch_pde_solver.py`, `double_one_touch_pde_solver.py`.
- Modify: `grid/time.py`. Add `TimeLayout.__reduce__`, so layouts deep-copy and pickle (Review Focus 1).
- Test: `test/pde_core/test_groups.py` (new), `test/pde_core/test_prepare.py` (new), `test/pde_core/test_review_focus.py` (extend).

**Interfaces:**
- Consumes (Task 4): `FactorizationCache`, `Backend`, `ColumnGroup`, `GroupState`, `VALUE`, `CoefficientsInForce`, `CentralDifference`, `StandardReadout`, `scheme_stepper`, `sample_bsm`, `validate_*`, `layout_fingerprint`, `coefficient_key`, `market_fingerprints`.
- Produces:

```python
# core/prepare.py
class Operation(Enum): PRICE; GREEKS; PRICE_WITH_EVENTS; EVENT_STATS; LIFE_SURFACE; SESSION_OUTPUTS
@dataclass(frozen=True) class OutputRequest: operation: Operation; price=True; greeks=False; event_stats=False; streams=None; grid=False; life_surface=False; extra_times=(); external_pv=None
@dataclass(frozen=True) class TerminalOutcome: reason: str; outputs: Any
@dataclass(frozen=True, eq=False) class SessionArtefacts: layout=None; coefficients=None; lu_pack=None; banded_pack=None
@dataclass(frozen=True, eq=False) class PreparedProblem: (fields below)
@dataclass(frozen=True, eq=False) class CompositeProblem: parts: tuple[PreparedProblem | CompositeProblem, ...]; compose: Callable[[tuple], Any]; request: OutputRequest; description: str
def prepare(solver, product, env, request, *, lifecycle_state=None, frozen=None, session=None) -> PreparedProblem | CompositeProblem
def run(problem) -> Any
# core/march.py
@dataclass(eq=False) class MarchResult: states: dict; captures: dict
def march(prepared: PreparedProblem) -> MarchResult
# core/readout.py
def read(prepared, result) -> Any          # dispatches to prepared.adapter.read
# core/adapters/base.py
class FamilyAdapter  (hooks listed in Step 4)
# on BasePDESolver
solver.adapter -> FamilyAdapter            # built once in __init__ via self._adapter_class(self)
solver._run(product, env, request, *, lifecycle_state=None) -> Any
```

`PreparedProblem` fields:
- `solver`, `adapter`, `product`, `env`, `request`, `lifecycle_state`
- `tau: float`
- `terminal: Optional[TerminalOutcome]`
- `solve_state: Mapping[str, Any]`
- `market: MarketSnapshot`, `grid_request: GridRequest`, `layout: Layout`
- `samples: Callable[[], MarketSamples]` (lazy, cached)
- `coefficients: CoefficientsInForce`
- `stepper`, `theta_schedule: np.ndarray`
- `groups: tuple[ColumnGroup, ...]`
- `factorizations: FactorizationCache`
- `readout`
- `events: tuple` (evidence rows)
- `identity: Mapping[str, Optional[str]]`: `layout`, `coefficient_key` (Task 5), and `problem` (Task 11)

When `terminal` is set, the geometry fields are `None`.

- [ ] **Step 1: Write the failing tests.**

`test/pde_core/test_groups.py`. The literal table of §9.3 is filled in family by family; this task adds the base rows:

```python
import pytest

from quantark.asset.equity.engine.pde import (AmericanPDESolver, BarrierPDESolver, DoubleBarrierPDESolver,
                                             DoubleOneTouchPDESolver, EuropeanPDESolver, OneTouchPDESolver)
from quantark.asset.equity.engine.pde.core.groups import VALUE, Backend
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare
from quantark.asset.equity.param import PDEParams

from pde_bitwise.capture_book import (american_tenor, barrier, double_barrier, double_one_touch,
                                      env_flat, european, one_touch)
from quantark.util.enum import ObservationType

# (solver, product) -> [(group name, surfaces, backend, policy)] -- the historical solve calls
BASE_ROWS = [
    (EuropeanPDESolver, european, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (AmericanPDESolver, american_tenor, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (BarrierPDESolver, lambda: barrier(ObservationType.CONTINUOUS), [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (DoubleBarrierPDESolver, double_barrier, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (OneTouchPDESolver, one_touch, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
    (DoubleOneTouchPDESolver, double_one_touch, [("value", ("value",), Backend.SPARSE_LU, VALUE)]),
]


@pytest.mark.parametrize("cls,make,want", BASE_ROWS)
def test_group_declaration_equals_the_historical_solve_calls(cls, make, want):
    p = prepare(cls(PDEParams()), make(), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == want
```

`test/pde_core/test_prepare.py`:

```python
def test_knock_in_prepares_two_problems_on_their_own_domains():
    ki = BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=120.0,
                       barrier_type=BarrierType.UP_IN, observation_type=ObservationType.CONTINUOUS)
    comp = prepare(BarrierPDESolver(PDEParams()), ki, env_flat(), OutputRequest(Operation.PRICE))
    assert isinstance(comp, CompositeProblem) and len(comp.parts) == 2
    vanilla, ko = comp.parts
    assert vanilla.layout.spatial.bounds[1] > 200.0          # automatic envelope (measured 222.554…)
    assert ko.layout.spatial.bounds[1] == 120.0              # the barrier's hard edge


def test_knock_in_composition_is_vanilla_minus_knock_out_for_price_and_greeks():
    ki = BarrierOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL, barrier=120.0,
                       barrier_type=BarrierType.UP_IN, observation_type=ObservationType.CONTINUOUS)
    env = env_flat()
    ko = replace_barrier_type(ki, BarrierType.UP_OUT)   # helper: dataclasses.replace or the constructor
    van = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g_ki = BarrierPDESolver(PDEParams()).calculate_greeks(ki, env)
    g_ko = BarrierPDESolver(PDEParams()).calculate_greeks(ko, env)
    g_v = EuropeanPDESolver(PDEParams()).calculate_greeks(van, env)
    for k in ("price", "delta", "gamma"):
        assert g_ki[k] == ki.participation_rate * g_v[k] - g_ko[k]


def test_expired_product_has_a_terminal_plan_and_no_layout():
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                      barrier_type=BarrierType.UP_OUT, observation_type=ObservationType.CONTINUOUS,
                      exercise_date=datetime(2026, 6, 1))
    env = flat_env(datetime(2026, 7, 1))
    prep = prepare(BarrierPDESolver(PDEParams()), p, env, OutputRequest(Operation.PRICE))
    assert prep.terminal is not None and prep.layout is None
    assert run(prep) == BarrierPDESolver(PDEParams()).price(p, env)
```

`flat_env` is imported from `intraday.conftest`. `replace_barrier_type` is a two-line test helper: it rebuilds the `BarrierOption` with the same keyword arguments and a different `barrier_type`.

`test/pde_core/test_review_focus.py`, Review Focus 1 and 2:

```python
@pytest.mark.parametrize("cls,make", [(EuropeanPDESolver, european), (AmericanPDESolver, american_tenor),
                                      (BarrierPDESolver, lambda: barrier(ObservationType.DISCRETE))])
def test_solver_deep_copies_and_pickles_after_pricing(cls, make):
    s = cls(PDEParams()); p = make(); env = env_flat()
    a = s.price(p, env)
    assert copy.deepcopy(s).price(p, env) == a
    assert pickle.loads(pickle.dumps(s)).price(p, env) == a


def test_american_greeks_on_a_fresh_solver_equal_greeks_after_price():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    conv = SettlementConvention(lag=2, lag_unit=SettlementLagUnit.BUSINESS_DAYS)
    p = AmericanOption(100.0, OptionType.PUT, exercise_date=datetime(2027, 3, 15), settlement_convention=conv)
    env = flat_env(datetime(2026, 3, 16)); env.calendar = cal
    assert american_exercise_requires_dates(p)
    fresh = AmericanPDESolver(PDEParams()).calculate_greeks(p, env)
    s = AmericanPDESolver(PDEParams()); s.price(p, env)
    after = s.calculate_greeks(p, env)
    assert fresh == after
```

Check `AmericanOption`'s constructor for how it accepts a `settlement_convention` keyword (`test/test_american_option_settlement.py:83-95` builds one). Mirror that fixture exactly.

- [ ] **Step 2: Run the tests to verify they fail.**
  - The `core.prepare` import fails.
  - The Review Focus tests fail: with `TypeError: cannot pickle 'SuperLU'`, and with `PricingError: American exercise dates must be resolved before building the PDE grid`.
- [ ] **Step 3: Implement `core/prepare.py`.**

```python
def prepare(solver, product, env, request, *, lifecycle_state=None, frozen=None, session=None):
    """The one preparation path (spec 2 §5.2). explain() is this function without the march."""
    adapter = solver.adapter
    adapter.validate(product, env, request)                              # the facades' raising checks
    composite = adapter.composite(product, env, request, lifecycle_state)
    if composite is not None:                                             # decision 21
        return composite
    tau = float(product.get_maturity(env))
    terminal = adapter.terminal_outcome(product, env, request, lifecycle_state, tau)   # step 0
    if terminal is not None:
        return PreparedProblem.terminal_only(solver, adapter, product, env, request, lifecycle_state, tau, terminal)
    adapter.check_route(solver.params)                                    # step 1
    fps = market_fingerprints(env)                                        # decision 14: once per prepare
    solver._market_fp = fps
    solve_state = adapter.solve_state(product, env, tau)                  # step 2
    market = solver.market_snapshot(product, env, tau)                    # step 3
    adapter.market_checks(product, env, tau)
    grid_request = adapter.grid_request(product, market, tau, request)    # step 4
    if session is not None and session.layout is not None:                # step 5
        layout = session.layout
    else:
        layout = resolve_bound_layout(solver.grid_binder, frozen, grid_request, market)
    solver._active_layout = layout        # read by helpers, the intraday route (until Task 11) and tests
    adapter.observation_maps(product, env, layout, tau)                   # step 6
    events = adapter.prepare_events(product, env, layout, tau, request)   # step 7
    samples = _lazy(lambda: adapter.samples(product, env, layout, tau))   # step 8
    coefficients = adapter.coefficients(product, env, layout, tau, samples, session)
    stepper = adapter.stepper(solver.params)                              # step 9
    schedule = stepper.schedule(layout)
    validate_schedule(schedule, layout, stepper)
    groups = adapter.groups(product, layout, request)                     # step 10
    factorizations = FactorizationCache(                                  # step 11 (retention: full, value groups)
        enabled=solver._is_cache_enabled(),
        lu_seed=None if session is None else session.lu_pack,
        banded_seed=None if session is None else session.banded_pack)
    identity = {"layout": layout.fingerprint,                             # step 12 (problem id: Task 11)
                "coefficient_key": adapter.coefficient_key(layout, fps, samples)}
    return PreparedProblem(solver=solver, adapter=adapter, product=product, env=env, request=request,
                           lifecycle_state=lifecycle_state, tau=tau, terminal=None, solve_state=solve_state,
                           market=market, grid_request=grid_request, layout=layout, samples=samples,
                           coefficients=coefficients, stepper=stepper, theta_schedule=schedule,
                           groups=groups, factorizations=factorizations, readout=StandardReadout(),
                           events=events, identity=identity)


def run(problem):
    if isinstance(problem, CompositeProblem):
        return problem.compose(tuple(run(p) for p in problem.parts))
    if problem.terminal is not None:
        return problem.terminal.outputs
    return read(problem, march(problem))
```

`_lazy(fn)` returns a zero-argument callable that evaluates `fn` once and caches the result.

`FamilyAdapter.coefficients` is where the step coefficients are built:
- If `session` holds coefficients, use them.
- Otherwise use `solver._build_step_coefficients(env, strike, layout.time.t, layout.spatial.dx, num_x)`. This is Task 3/4's memoized delegate, which returns the pre-flat-exact sets, so the memo is shared with the session prep.
- Apply `flat_exact(sc, scalar_stencil(r_tau, q_tau, sigma_tau, …))` for SCALAR routes. The LV adapters skip `flat_exact`, as their `_flat_exact_step_coefficients` override does today.
- Wrap the result in `CoefficientsInForce(flat=scalar_stencil(r_tau, q_tau, sigma_tau, dx, num_x), sets=None if sc.n_unique == 1 else sc, num_x=num_x)`.
- `validate_coefficients(sc, …, component=CentralDifference())`. Task 12 switches this to `params.operator`.
- `r_tau`, `q_tau` and `sigma_tau` are exactly `_solve:825-829`: `env.get_rate(tau)`, `env.get_div_yield(tau)`, `env.get_vol(strike, tau)` with `strike = getattr(product, "strike", spot)`.

- [ ] **Step 4: Implement `core/march.py`, `core/readout.py` and `core/adapters/base.py`.**

```python
def march(prepared) -> MarchResult:
    """One loop over the layout's steps, last to first (spec 2 §5.3). No product branches."""
    adapter, layout = prepared.adapter, prepared.layout
    cache = prepared.factorizations
    cache.clear()
    states, captures = adapter.allocate(prepared), {}
    adapter.terminal(prepared, states, captures)
    coeffs, schedule, stepper = prepared.coefficients, prepared.theta_schedule, prepared.stepper
    n_int = len(layout.spatial.x) - 2
    dt_vec = layout.time.dt
    for j in range(len(layout.time.t) - 2, -1, -1):
        dt = dt_vec[j]
        k = coeffs.set_at(j)
        for group in prepared.groups:
            theta = float(schedule[j])
            if group.policy.zero_diffusion_theta_one and coeffs.is_zero_diffusion(k):
                theta = 1.0  # upwind advection is monotone only fully implicit
            adapter.boundaries(prepared, group, states, j)
            v_next = np.column_stack([states[n][:, j + 1] for n in group.surfaces])
            v_next.setflags(write=False)
            lower = np.array([states[n][0, j] for n in group.surfaces])
            upper = np.array([states[n][-1, j] for n in group.surfaces])
            out = stepper.step(GroupState(group, v_next), coeffs.step_set(k, cache), dt, theta, (lower, upper))
            validate_advanced(out, group, n_int, stepper)
            for i, name in enumerate(group.surfaces):
                states[name][1:-1, j] = out[:, i]
            adapter.after_solve(prepared, group, states, j)
        adapter.step_events(prepared, states, captures, j)
    return MarchResult(states=states, captures=captures)
```

Bit-identity check for the base family:
- The legacy step computed the factorization, set the boundaries, formed `rhs = M1 @ v`, injected, solved, and then ran `_apply_step_modifications`.
- The march does exactly this, apart from when the factorization is looked up. The lookup has no side effects.
- `states["value"][0, j]` is read after `boundaries`, which matches `_inject_boundary_contributions` reading `grid[0, j]`.

`read(prepared, result)` returns `prepared.adapter.read(prepared, result)`.

`FamilyAdapter` hook table. Each default body reproduces the legacy base-solver call.

| Hook | Default body (legacy source) |
|---|---|
| `validate(product, env, request)` | no-op |
| `composite(...)` | `None` |
| `terminal_outcome(product, env, request, lifecycle_state, tau)` | `tau <= 0` gives `TerminalOutcome("expired", …)`. PRICE: `solver._calculate_intrinsic(product, spot)` (`_price_with_solution:901-920`). GREEKS: `{"price": intrinsic, "delta": _intrinsic_delta, "gamma": 0.0}` (`calculate_greeks:946-975`). |
| `check_route(params)` | `check_route_applicability(params, self.route)` when the solver declares a route. Otherwise a no-op; every shared-grid field is honoured. |
| `solve_state(product, env, tau)` | `solver._prepare_solve_state(product, env)`; returns `{}` |
| `market_checks(product, env, tau)` | `guard_constant_vol(env.vol_surface, type(solver).__name__, strict=getattr(solver.params, "strict_smile", False))` (`_solve:830`) |
| `grid_request(product, market, tau, request)` | `solver.grid_request(product, market, tau)` |
| `observation_maps(product, env, layout, tau)` | `solver._populate_observation_maps(product, env, layout, tau)` |
| `prepare_events(...)` | `()` |
| `samples(...)` | `sample_bsm(env, layout.time.t, ref_strike=strike, r_tau=…, q_tau=…, sigma_tau=…)` |
| `coefficients(...)` | as in Step 3 |
| `stepper(params)` | `scheme_stepper(params, solver.grid_binder.config)` |
| `coefficient_key(layout, fps, samples)` | `coefficient_key(layout.fingerprint, fps, strike, component_fingerprint(CentralDifference()))` |
| `groups(...)` | `(ColumnGroup("value", ("value",), Backend.SPARSE_LU, VALUE),)` |
| `allocate(prepared)` | `{"value": np.zeros((num_x, num_t))}` (`_solve:835`) |
| `terminal(prepared, states, captures)` | `solver.set_terminal_condition(states["value"], x, s, product, env)` |
| `boundaries(prepared, group, states, j)` | `solver.set_boundary_conditions(states["value"], x, s, j, prepared.tau - t[j], product, env)` |
| `after_solve(prepared, group, states, j)` | `solver._apply_step_modifications(states["value"], x, s, j, prepared.tau - t[j], product, env)` |
| `step_events(...)` | no-op |
| `read(prepared, result)` | PRICE: `readout.price(result.states["value"][:, 0], spot, layout)`. GREEKS: `{"price": that, "delta", "gamma"}` from `readout.delta_gamma(same column, spot, layout)`, each passed through `validate_readout`. |

Base-solver facade:

```python
def price(self, product, pricing_env):
    return self._run(product, pricing_env, OutputRequest(Operation.PRICE))

def calculate_greeks(self, product, pricing_env):
    return self._run(product, pricing_env, OutputRequest(Operation.GREEKS, greeks=True))

def _run(self, product, env, request, *, lifecycle_state=None):
    return run(prepare(self, product, env, request, lifecycle_state=lifecycle_state,
                       frozen=self._frozen_base_layout, session=SessionArtefacts.from_solver(self, product, env)))
```

- `SessionArtefacts.from_solver` returns `None` unless `_session_grids`, `_session_step_coefficients`, `_session_matrix_pack` or `_session_banded_pack` is set. When one is set, it builds the layout with the existing `_layout_from_session_grids` and carries the packs. It is a temporary bridge, removed in Task 10.
- `BaseEngine.__init_subclass__` still wraps these methods with the `lifecycle_state` keyword. Solvers whose `price` declares `lifecycle_state` pass it through to `_run`.

- [ ] **Step 5: Move each base-family solver onto its adapter.** Each move is verbatim, with `self.` rewritten to `self.solver.` where the code now runs inside the adapter.

| Adapter | `terminal_outcome` / `validate` / `composite` (moved from) | Other hooks |
|---|---|---|
| `EuropeanAdapter` | `european_pde_solver.py:61-131`: the type check and `validate_settlement_capability` become `validate`. The `terminal_lifecycle_pv` short-circuit and the `maturity <= 0` branches become `terminal_outcome` for PRICE and GREEKS. | `read(PRICE)` adds `pending_receivable_pv(product, env)` exactly where `price` added it after `super().price`. |
| `AmericanAdapter` | `american_pde_solver.py:98-144`. PRICE `tau <= 0`: `payoff(spot) * resolve_american_exercise_timings(..., [0.0]).delay_dfs[0]`. GREEKS `tau <= 0`: the base intrinsic with no delay factor. The GREEKS branch is kept as it is today; the characterized oddity is recorded in the self-review notes. | `solve_state` becomes `solver._prepare_solve_state` on **every** operation (Review Focus 2). `grid_request` is the solver's override (55-66). `prepare_events` takes the body of the deleted `_time_stepping` override (284-362): resolve the timings, set `solver._exercise_timings` and `solver._exercise_boundary_delay_dfs`, and read `t_vec` from `layout.time.t`. `read(PRICE)` returns `max(value, product.intrinsic_value(spot) * timings.delay_dfs[0])`. `read(GREEKS)` is the base read with no obstacle max. |
| `BarrierAdapter` (Barrier, DoubleBarrier, OneTouch, DoubleOneTouch; each solver sets `_adapter_class`) | From each solver's `price` / `calculate_greeks`: the EXPIRY raise becomes `validate`; the hit branches (rebate PV, 0, vanilla) become `terminal_outcome("barrier_hit" / "touched")` per operation, kept verbatim, including the double barrier's undiscounted Greeks rebate. The knock-in-not-hit branch becomes `composite`. | `observation_maps` takes Barrier's `_build_grids` override body (610-638): after the layout, `_setup_observation_indices(product, env, tau, layout.time.t, resolve_kwargs=…)`. The other three solvers keep `_populate_observation_maps`. |

The knock-in composite (decision 21):

```python
def composite(self, product, env, request, lifecycle_state):
    if not self._is_knock_in_not_hit(product, env):          # today's branch condition, verbatim
        return None
    vanilla_solver = EuropeanPDESolver(self.solver.params)   # fresh on every call, as today: no frozen layout
    vanilla = self._vanilla_product(product)                 # body of _price_vanilla's product construction
    ko = self._knock_out_product(product)                    # body of _price_knock_out's product construction
    pr = self._participation(product)                        # product.participation_rate; 1.0 for DoubleBarrier
    v_part = prepare(vanilla_solver, vanilla, env, request)
    k_part = prepare(self.solver, ko, env, request, frozen=self.solver._frozen_base_layout,
                     session=SessionArtefacts.from_solver(self.solver, ko, env))
    if request.operation is Operation.GREEKS:
        compose = lambda outs: {k: pr * outs[0][k] - outs[1][k] for k in ("price", "delta", "gamma")}
    else:
        compose = lambda outs: pr * outs[0] - outs[1]
    return CompositeProblem(parts=(v_part, k_part), compose=compose, request=request,
                            description="knock-in = participation × vanilla − knock-out")
```

- **Operation order.** The legacy expression was `product.participation_rate * solver.price(vanilla, env)`, and the difference was then taken as `vanilla - knock_out`. `pr * outs[0] - outs[1]` performs the same two operations in the same order.
- **Double barrier.** The legacy `_price_vanilla` does not multiply by a participation rate. Set `pr = None` and compose with `outs[0] - outs[1]`; do **not** multiply by `1.0`.
- **Greek keys.** Before writing the Greeks composition, read `_calculate_greeks_vanilla` and `_calculate_greeks_knock_out` for the exact keys and the order they iterate.

- [ ] **Step 6: Make layouts copyable.** Add `TimeLayout.__reduce__`:

```python
    def __reduce__(self):
        # mappingproxy does not pickle or deep-copy; rebuild from plain values.
        state = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        state["step_of"] = dict(self.step_of)
        return (_rebuild_time_layout, (state,))
```

`_rebuild_time_layout` is a module-level function. It wraps `step_of` back into a `MappingProxyType`, marks `t` and `dt` read-only, and constructs the layout.

Check whether `SpatialLayout`, `Layout` and `GridBinder` pickle as they are. If any of them fails, give it the same `__reduce__` treatment, and name the fix in the commit message.

- [ ] **Step 7: Delete the legacy code and run the tests.**
  - Delete `_time_stepping`, `_solve` and `_price_with_solution` in the base solver, and the overrides listed under Files.
  - Keep `_get_matrices`, `_build_grids` and `_grids_via_layer`. Snowball, Phoenix and the session prep still call them.
  - Run: `grep -rn "_time_stepping\b\|\._solve(" quantark/asset/equity/engine/pde/{base,european,american,barrier,double_barrier,one_touch,double_one_touch}_pde_solver.py`. Expected: empty.
  - Run: `… -m pytest test/pde_core test/test_pde_engine.py test/test_pde_engine_dispatcher.py test/test_barrier_family_settlement.py test/test_american_option_settlement.py test/test_european_settlement_matrix.py test/pde_grid test/execution -q`. Expected: PASS.
  - Four tests call `engine._solve` or `_bump_*` on a base-family solver: `test/test_bump_grid_steadiness.py:64-102`, `test/execution/test_pde_session_seams.py:39`, `test/pde_grid/test_frozen_bump_layouts.py`, and `test/test_pde_fixed_bump_grid.py`. Rewrite them against `prepare`/`march`/`read`, keeping the same assertions. For example, replace `engine._solve(p, env).solution_vec` with `march(prepare(engine, p, env, OutputRequest(Operation.PRICE))).states["value"][:, 0]`.
- [ ] **Step 8: Run the byte gate, the identity gate, and the full suite.** Expected: 0 diffs and 0 violations. Pay particular attention to the `knock_in|*` rows, the `american_dated|*` rows, and `european|flat|theta1`.
- [ ] **Step 9: Commit.** Message: `refactor(pde)!: base family and American march on the core; knock-in is a composite of two prepared problems (spec 2 §5.1-§5.3, decision 21)`.

---

### Task 6: Snowball, the local-vol snowball and the life surface on the core

**Files:**
- Create: `core/adapters/snowball.py` (`SnowballAdapter`, `LocalVolSnowballAdapter`).
- Modify: `snowball_pde_solver.py` and `snowball_vol_pde_solvers.py`. KO-reset still calls the legacy loop (Task 7), so the following are deleted here:
  - `_solve` (363-518);
  - `_price_with_solution` (646-688);
  - the price paths of `price`, `calculate_greeks` and `solve_life_surface`;
  - the `_build_grids` override (1884-1963, the nearest-match rebuild);
  - `_extra_time_nodes`, `_capture_branch_columns` and `_branch_cols` (class and instance flags);
  - `_reset_t0_readout_state` usage;
  - `event_schedule` (2299-2424; the body moves).
- Modify: `snowball_vol_pde_solvers.py`. The LV snowball `price`, `calculate_greeks` and `_session_outputs` wrappers go through the core.
- Test: `test/pde_core/test_groups.py`, `test/pde_core/test_review_focus.py`, `test/intraday/test_pde_zero_diffusion_two_surface.py` (rewritten), `test/test_snowball_life_surface.py:65`.

**Interfaces:**
- Consumes (Task 5): `prepare`, `run`, `march`, `FamilyAdapter`, `OutputRequest`, `Operation`.
- Produces:
  - `SnowballAdapter(FamilyAdapter)`. `PhoenixAdapter` (Task 8) and `KOResetAdapter` (Task 7) subclass it.
  - `OutputRequest(Operation.LIFE_SURFACE, life_surface=True, extra_times=nodes)`.
  - `captures["t0_pre_event"]`, `captures["branch"]`.
  - `SnowballAdapter.value_groups(product, layout) -> tuple[ColumnGroup, ...]`.

- [ ] **Step 1: Write the failing tests.**

Extend `test/pde_core/test_groups.py`:

```python
SNOWBALL_ROWS = [
    (SnowballPDESolver, lambda: snowball("continuous"),
     [("value", ("alive", "ki"), Backend.BANDED, VALUE)]),
    (LocalVolSnowballPDESolver, lambda: snowball("continuous"),
     [("value", ("alive", "ki"), Backend.BANDED, VALUE)]),
]


def test_snowball_with_two_interior_nodes_falls_back_to_two_lu_groups():
    # measured at 302fa2ed: points=4, eps_crit=1.0 prices (4 nodes, 2 interior); points=3 is rejected
    s = SnowballPDESolver(PDEParams(grid=GridConfig(points=4, eps_crit=1.0)))
    p = prepare(s, snowball("continuous"), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == [
        ("alive", ("alive",), Backend.SPARSE_LU, VALUE), ("ki", ("ki",), Backend.SPARSE_LU, VALUE)]
```

Extend `test_review_focus.py`. Parametrize the deepcopy/pickle test over `(SnowballPDESolver, lambda: snowball("continuous"))` and `(LocalVolSnowballPDESolver, …, env_grid_vol)`.

Rewrite `test/intraday/test_pde_zero_diffusion_two_surface.py:39-56`. It spies on `_time_stepping_two_surface`, `_get_banded_system` and `_get_matrices` to prove that zero-diffusion steps take θ=1 on the two-surface march. The new version spies on `CrankNicolson.step` with `monkeypatch.setattr(CrankNicolson, "step", spy)`, where `spy` records `(group.name, theta)` and calls through. Assert the same thing: on the zero-diffusion steps of the fixture, the value group received θ=1.0.

- [ ] **Step 2: Run the tests to verify they fail.**
- [ ] **Step 3: Implement `SnowballAdapter`.** The legacy source is cited line by line. Keep every operation order.

| Hook | Body |
|---|---|
| `validate` | the type check (`_check_product_type` 611-625) and `_validate_product` |
| `terminal_outcome` | `_price_with_solution:646-688`: clear `_product_token_memo` on **every** operation (§5.9). `tau <= 0` gives `_calculate_terminal_value`. KO at valuation gives `_get_immediate_ko_payoff`. GREEKS follow `calculate_greeks:1425-1496`'s own short-circuits verbatim. LIFE_SURFACE follows `solve_life_surface:1498-1520`'s raises. EVENT_STATS gives `None` (Task 9 fills it). |
| `solve_state` | `solver._prepare_solve_state(product, env)`, and stash `solver._grid_request_env = env` (`_prepare_for_request:2228-2235`). Returns `{"knocked_in": solver._knocked_in_at_valuation, "reverse": solver._is_reverse, "ki_continuous": solver._ki_continuous, "ki_barrier": solver._ki_barrier}` for identity and explain. |
| `grid_request` | `solver.grid_request(product, market, tau)`, with `extra_times=request.extra_times` passed through (replacing `self._extra_time_nodes` at 2225). Add an `extra_times=()` keyword to `SnowballPDESolver.grid_request`. |
| `observation_maps` | `solver._populate_observation_maps` (2237-2297). This is the only pass. The `_build_grids` rebuild with `_aligned_time_index` is deleted (§8: same nodes; the byte rows prove it). |
| `prepare_events` | Build the interior event table: the body of `event_schedule:2299-2424`, as an `EventSchedule`. Store it as `self._schedule` on the prepared problem's event slot (`PreparedProblem.events` carries `schedule.evidence()`: one row per step with kind, barrier, cash amount, settlement time and projection). Add a small `EventSchedule.evidence()` built from the lists the constructor already receives. Also call `solver._prepare_continuous_ki_correction(product, env, layout.time.t)` (2533). The first-passage coefficients read `prepared.samples().first_passage(...)` through the solver helper, which now delegates. |
| `value_groups` / `groups` | `use_banded = (num_x - 2) > 2` (2537). If true: `(ColumnGroup("value", ("alive", "ki"), BANDED, VALUE),)`. Otherwise: `(ColumnGroup("alive", ("alive",), SPARSE_LU, VALUE), ColumnGroup("ki", ("ki",), SPARSE_LU, VALUE))`. |
| `allocate` | `{"alive": zeros, "ki": zeros}`, and set `solver._grid_v0` and `solver._grid_v1` to those two arrays, as today (the v0 boundary and tests read them). |
| `terminal` | `_solve:423-467`: the v0 and v1 terminal conditions, then the terminal events (KO unless `ko_last`, KI at `num_t-1`, KO if `ko_last`). If `request.life_surface`, allocate `captures["branch"]` as `(empty_like(alive), empty_like(ki))` and copy the terminal column (2549-2555). |
| `boundaries` | For the two-column group: `_set_boundary_conditions_v0(alive, …, j, tau - t[j], …)` then `_set_boundary_conditions_v1(ki, …)` (2583-2588). For the LU groups: the one matching the group. |
| `after_solve` | no-op |
| `step_events` | `_time_stepping_two_surface:2657-2686`, verbatim, in this order: (1) branch capture if present; (2) `j == 0 and solver._t0_has_events(product)` gives `captures["t0_pre_event"] = (alive[:, 0].copy(), ki[:, 0].copy())`; (3) `j > 0`: `schedule.apply` then `schedule.continuous`, writing back; otherwise `solver._apply_step_modifications_two_surface(...)` at `j == 0`. |
| `samples` | BSM. `LocalVolSnowballAdapter` overrides it with `sample_local_vol(env, t, layout.spatial.s, solver._active_lv_surface, …)`. |
| `coefficients` | the base hook. `LocalVolSnowballAdapter` skips `flat_exact`. |
| `read` | `_compose_t0_readout(sel)` logic (1679-1689), where `sel = 1 if knocked_in_at_valuation else 0`: `readout_vec` = the t0 capture of `sel` if present, else the final column 0 of `sel`'s surface; `readout_override` is `solver._t0_readout_values[sel]` if set (Phoenix). PRICE returns `override`, else `readout.price(readout_vec)`, the priority `_price_with_solution` used. GREEKS follow `calculate_greeks:1425-1496`. LIFE_SURFACE returns `LifeSurfaceSolution(t=layout.time.t, x=layout.spatial.x, s=layout.spatial.s, v0=states["alive"], v1=captures["branch"], step_of=layout.time.step_of, t0_readout=readout_vec, knocked_in_at_valuation=…, requested_steps=…, actual_steps=…, fill_scaled=…)`, with the same field sources as 1498-1544. |

**Capture order.** §5.3 lists the branch capture *after* the events (item 4). The legacy code (2657-2661) captures branch columns **before** the step's events, and `LifeSurfaceSolution.v1` is that pre-event branch. Follow the code. The book's `life_surface|snowball` row proves it.

`SnowballPDESolver` facade:
- `price` becomes `self._run(p, env, OutputRequest(Operation.PRICE))`.
- `calculate_greeks` becomes `OutputRequest(Operation.GREEKS, greeks=True)`.
- `solve_life_surface(p, env, *, extra_times)` validates `extra_times` exactly as 1500-1520 does, then runs `OutputRequest(Operation.LIFE_SURFACE, life_surface=True, extra_times=tuple(nodes))`.
- `_session_outputs` and `price_with_events` still call the legacy statistics path until Task 9: `price` through the core, then `calculate_event_stats(npv=…)`.

The two-dimensional (Heston/SLV) snowball subclasses do not use the 1D march. They keep calling the inherited helpers (`_set_boundary_conditions_v0/_v1`, `_get_cached_ko_records`, …), which are **unchanged**. They override `price`, so the new facade does not reach them.

`KOResetSnowballPDESolver` still overrides `_solve` and calls `_time_stepping_two_surface`. **Keep `_time_stepping_two_surface` and `_get_banded_system` in this task.** KO-reset's `price` must keep working, so give it a private `_legacy_price_with_solution` that is a verbatim copy of the deleted `_price_with_solution` (646-688). Task 7 deletes all three. This deviates from the spec order (§10 step 2 deletes the loop here) so that `main` stays green; see the self-review notes.

- [ ] **Step 4: Run the tests, the byte gate and the identity gate.** Run `… -m pytest test/pde_core test/test_snowball_pde.py test/test_snowball_life_surface.py test/test_snowball_pde_date_schedule.py test/test_snowball_pde_knocked_in_grid.py test/test_pde_continuous_ki_first_passage.py test/test_pde_event_projection.py test/intraday test/test_local_vol_equity_engines.py -q`. Then run both gates.
  - Every `snowball_*` row must be 0 diffs, along with `lv_snowball*`, `life_surface|snowball`, `session_snowball`, `intraday_snowball` and `bump|frozen_vs_fresh|snowball`.
  - The statistics rows go through the legacy sweep, which now reads the observation maps from the single pass. The `stats|*` rows are the first proof of §8's "nearest-match lands on the same node" claim for the rebuilt maps.
  - **If a `snowball_daily|*` or `stats|*` row diffs**, the removed `_aligned_time_index` pass (1e-6 absolute tolerance) mapped some observation differently from `step_at` (1e-9 relative tolerance). Stop, measure which record moved, and report it. Do not restore the rebuild. It would be a second map pass, and the spec's claim would then be false, so the user decides.
- [ ] **Step 5: Commit.** Message: `refactor(pde)!: snowball, local-vol snowball and life surface march on the core; one observation-map pass (spec 2 §5.6, §8)`.

---

### Task 7: KO-reset on the core; the two-surface loop deleted

**Files:**
- Create: `core/adapters/ko_reset.py` (`KOResetAdapter(SnowballAdapter)`).
- Modify: `ko_reset_snowball_pde_solver.py`. Delete the `_solve` (365-518) and `_build_grids` (299-315) overrides. Delete `_legacy_price_with_solution` and `snowball_pde_solver._time_stepping_two_surface` (2500-2688).
- Test: `test/pde_core/test_groups.py` (a KO-reset row: `[("value", ("alive", "ki"), BANDED, VALUE)]`); `test/test_ko_reset_snowball_pde.py` (unchanged, must pass).

**Interfaces:**
- Consumes (Task 6): `SnowballAdapter`.
- Produces `KOResetAdapter`.

- [ ] **Step 1: Add the group row and run it.** It fails, because KO-reset has no adapter yet.
- [ ] **Step 2: Implement `KOResetAdapter`.** It differs from `SnowballAdapter` only as follows. The legacy source is `ko_reset_snowball_pde_solver.py`.
  - `terminal_outcome`: `product.require_alive(...)` (365-380) is a raise, so it goes in `validate`. The remaining short-circuits are those of the inherited `_price_with_solution`.
  - `solve_state`: `_prepare_solve_state` (inherited).
  - `observation_maps`: `solver._populate_observation_maps`, which is KO-reset's own override (250-257). It already snapshots the pre maps from this single pass. The ordering hazard in the map report (a snapshot taken before the rebuild) is gone, because the rebuild no longer exists.
  - `prepare_events`: no `EventSchedule`, because KO-reset's `_active_schedule` was `None`. Every step takes the inline path. Also run `_prepare_continuous_ki_correction`.
  - `coefficients`: KO-reset called `_build_step_coefficients` directly and bypassed session injection. It now takes the shared hook, which honours an injected session's coefficients. The values are identical (§8), and Task 10's session rows prove it.
  - `terminal`: `_solve:430-487`, verbatim. This covers `_v0_seed_values` / `_v0_seed_idx` staging, and the terminal order: pre KO on v0 unless `pre_ko_last`, then post KO on v1 unless `disable_ko_after_ki`, then KI unless staged, then pre KO if `pre_ko_last`. KO-reset does **not** call `_reset_t0_readout_state`. Keep that: do not clear `solver._t0_*` in this adapter.
  - `step_events`: branch capture and t0 capture as in `SnowballAdapter`, then always `solver._apply_step_modifications_two_surface(...)` (KO-reset's override 520-582). There is no schedule path.
  - `read`: KO-reset returned `PDESolutionResult` **without** `readout_vec` / `readout_override`. PRICE therefore interpolates the post-event column 0 of the selected surface, ignoring any t0 capture. GREEKS take delta and gamma from that same column. Keep this; it is a characterized oddity (see the self-review notes).
- [ ] **Step 3: Delete the legacy code.** Delete `_time_stepping_two_surface`, `_legacy_price_with_solution`, and KO-reset's `_solve` and `_build_grids`. Then `grep -rn "_time_stepping_two_surface" quantark test` must be empty.
- [ ] **Step 4: Run the tests and the gates.** Run `… -m pytest test/test_ko_reset_snowball_pde.py test/pde_core -q`, then the byte gate and the identity gate. The `ko_reset|*` and `stats|ko_reset|*` rows must show 0 diffs (the latter still go through QUAD plus the price delta), and so must `pwe|ko_reset|*`.
- [ ] **Step 5: Commit.** Message: `refactor(pde)!: KO-reset marches on the core; the two-surface loop is deleted`.

---

### Task 8: Phoenix and the local-vol phoenix on the core

**Files:**
- Create: `core/adapters/phoenix.py` (`PhoenixAdapter(SnowballAdapter)`, `LocalVolPhoenixAdapter`).
- Modify: `phoenix_pde_solver.py`. Delete `_solve` (317-469), `_time_stepping_vector_surface` (641-739), `_step_grids` (741-778), and the dead `_build_grids` override (514-530).
- Modify: `phoenix_vol_pde_solvers.py`. The LV wrappers go through the core.
- Modify: `base_pde_solver.py`. `_get_matrices` loses its last march caller. Keep it only if `_session_factorization_packs` still calls it; that goes in Task 10.
- Test: `test/pde_core/test_groups.py` (Phoenix rows), `test/pde_core/test_review_focus.py` (Phoenix in the copy test).

**Interfaces:**
- Consumes (Task 6): `SnowballAdapter`.
- Produces `PhoenixAdapter` and the per-column group names `v0[k]`, `v1[k]`, `w0`, `w1`.

- [ ] **Step 1: Write the group rows.** The literal table:

```python
def _phoenix_groups(max_k, termination, banded=True):
    b = Backend.BANDED if banded else Backend.SPARSE_LU
    g = [(f"v0[{k}]", (f"v0[{k}]",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE) for k in range(max_k + 1)]
    g += [(f"v1[{k}]", (f"v1[{k}]",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE) for k in range(max_k + 1)]
    if termination:
        g += [("w0", ("w0",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE), ("w1", ("w1",), b, VALUE_NO_ZERO_DIFFUSION_OVERRIDE)]
    return g


def test_phoenix_memory_groups():
    p = prepare(PhoenixPDESolver(PDEParams()), make_phoenix(PHOENIX), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == _phoenix_groups(12, False)
    assert all(g.one_dimensional_rhs for g in p.groups)


def test_phoenix_non_memory_groups():
    p = prepare(PhoenixPDESolver(PDEParams()), phoenix_plain(), env_flat(), OutputRequest(Operation.PRICE))
    assert [(g.name, g.surfaces, g.backend, g.policy) for g in p.groups] == _phoenix_groups(0, False)
```

`max_k` is `num_obs` with memory coupons, per `_solve:317-469`. Confirm the value 12 for the book's `PHOENIX` (12 observations) by reading the `max_k` line before you freeze it. Also add an EXPIRY-coupon Phoenix whose `termination` is `True` (`_termination_surface_active`). Build it from `make_phoenix(dict(PHOENIX, coupon_pay_type="expiry"))` if the builder accepts that key. Otherwise build the `CouponBarrierConfig(coupon_pay_type=CouponPayType.EXPIRY, …)` directly, as `test_phoenix_expiry_coupon_settlement.py` does.

- [ ] **Step 2: Run the tests to verify they fail.**
- [ ] **Step 3: Implement `PhoenixAdapter`.** The legacy source is `phoenix_pde_solver.py`.
  - `terminal_outcome`: as for the snowball. The immediate-KO and terminal-value overrides are Phoenix's own methods (252, 279), and they are reached through `solver.`.
  - `observation_maps`: `solver._populate_observation_maps`. Phoenix's override registers coupons with `step_at` (613-622).
  - `prepare_events`: `_prepare_continuous_ki_correction`. Then `terms = solver._prepare_termination_surfaces(product, env, layout.time.t, num_x)` (867-915), stored on the solver as today (`_term_w0`, …). No `EventSchedule`.
  - `groups`: `use_banded = (num_x - 2) > 2`. Return one group per column in the legacy stepping order: `v0[0..K]`, then `v1[0..K]`, then `w0`, `w1` when the termination surfaces are active. Every group has policy `VALUE_NO_ZERO_DIFFUSION_OVERRIDE` and `one_dimensional_rhs=use_banded`. **Phoenix keeps its missing zero-diffusion override (§8).**
  - `allocate`: `v0[k]` and `v1[k]` for `k = 0..max_k`, each `np.zeros((num_x, num_t))`, plus `w0` / `w1` taken from `terms`. The 50-observation raise moves to `validate`. Do **not** set `solver._grid_v1`. Phoenix never set it, so under continuous KI the v0 lower boundary read `None` and fell back to `0.0`. Keep that.
  - `terminal`: `_solve:430-455`: the terminal vector conditions, then coupon at the terminal index, then KI for each k, then terminal KO.
  - `boundaries`: `w0` / `w1` use `solver._set_termination_boundary(grid, j)`. `v1[k]` uses `_set_boundary_conditions_v1`. `v0[k]` uses `_set_boundary_conditions_v0`. This matches the dispatch in `_step_grids`.
  - `step_events`: `_time_stepping_vector_surface:720-739`, in order. At `j == 0` and `_t0_has_events`, take the t=0 pre-event capture of the level-0 columns. Then `solver._apply_step_modifications_vector_surface(...)` (780-843).
  - `read`: the snowball read. The t=0 coupon capture sets `solver._t0_readout_cols` / `_t0_readout_values` during the events, so `read` finds them as `_compose_t0_readout` did.
  - `LocalVolPhoenixAdapter`: `sample_local_vol`; no `flat_exact`.

One factorization per step is shared by every column group. With the core, every group at step `j` asks the cache for the same `(k, dt, θ)` key. The first group builds the factorization and the rest hit it, which is bitwise the same as the legacy single lookup per step.

- [ ] **Step 4: Delete the legacy code** listed under Files. Then `grep -rn "_time_stepping_vector_surface\|_step_grids" quantark test` must be empty.
- [ ] **Step 5: Run the tests and the gates.** Run `… -m pytest test/test_phoenix_pde.py test/test_phoenix_expiry_coupon_settlement.py test/pde_core -q`, then the byte gate and the identity gate. The `phoenix_*`, `lv_phoenix`, `stats|phoenix_*`, `pwe|phoenix_*`, `settlement_lag|phoenix` and `session_phoenix` rows must show 0 diffs.
- [ ] **Step 6: Commit.** Message: `refactor(pde)!: phoenix and local-vol phoenix march on the core; the vector-surface loop is deleted`.

---

### Task 9: Event statistics as a column group of the price march (Level 1); the 2D statistics-only march

**Files:**
- Create: `core/statistics.py` (`StatisticsLegacy`).
- Modify: `core/groups.py`. `ColumnGroup` gains `rolling: bool = False`. `GroupPolicy` gains `dt_as_python_float: bool` (`STATISTICS_LEGACY` sets it `True`; the other two policies set it `False`).
- Modify: `core/march.py`. Handle rolling groups and the statistics hooks.
- Modify: `core/prepare.py`. Step 10 adds the statistics group when `request.event_stats`.
- Modify: `snowball_pde_solver.py`. Delete `_compute_event_stats` (824-1423), `_get_banded_system` (2690-2742), `_pack_banded_entry`, `_pack_uses_banded`, and `_stats_layout`. `calculate_event_stats`, `price_with_events` and `_session_outputs` become `_run(...)`.
- Modify: `base_pde_solver.py`. Delete the legacy coefficient delegates that now have no caller (`_calculate_coefficients`, `_flat_exact_step_coefficients`, `_build_operator_matrix`, `_operator_matrix_for_set`) and `_get_matrices`. Delete `_session_factorization_packs` only if Task 10 has already landed. Otherwise leave it for Task 10, which deletes it. Check with `grep`.
- Modify: `snowball_vol_pde_solvers.py:583-636` and `phoenix_vol_pde_solvers.py:419-477`. The 2D `calculate_event_stats` / `_session_outputs` become the statistics-only request.
- Test: `test/pde_core/test_review_focus.py` (Review Focus 3); `test/pde_core/test_groups.py` (statistics rows); `test/test_event_stats_pruning.py:135-183` and `test/test_trade_stream_wiring.py:86-92` (rewritten); `test/pde_core/test_march_step.py` (already covers the kernel).

**Interfaces:**
- Consumes (Tasks 4–8): the march, the adapters, and `banded_advance`.
- Produces:
  - `StatisticsLegacy(adapter, prepared_parts)` with `.group`, `.allocate(states)`, `.terminal(states)`, `.step_events(states, captures, j)`, `.read(states, captures, pv)`.
  - `OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True, external_pv=pv, streams=…)`, used by the 2D solvers.
  - `OutputRequest(Operation.PRICE_WITH_EVENTS, event_stats=True, streams=…)`.
  - `OutputRequest(Operation.SESSION_OUTPUTS, event_stats=want_events, grid=want_grid, streams=streams)`, which returns `PDESessionOutputs`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_statistics_on_a_reused_solver_ignore_an_earlier_market():
    # Review Focus 3. Measured at 302fa2ed: KO probabilities off by up to 0.0407.
    p = snowball("discrete")
    s = SnowballPDESolver(PDEParams())
    s.price(p, _flat(vol=0.20))
    stale = s.calculate_event_stats(p, _flat(vol=0.25))
    fresh = SnowballPDESolver(PDEParams()).calculate_event_stats(p, _flat(vol=0.25))
    assert list(stale.ko_probability) == list(fresh.ko_probability)
```

`_flat(vol)` is the flat `PricingEnvironment` from `test_market_memos.py`, with the vol surface parameterised.

Statistics rows for `test_groups.py`:

```python
def test_statistics_group_is_one_banded_group_of_both_surfaces():
    p = prepare(SnowballPDESolver(PDEParams()), snowball("discrete"), env_flat(),
                OutputRequest(Operation.PRICE_WITH_EVENTS, event_stats=True))
    names = [(g.name, g.surfaces, g.backend, g.policy, g.rolling) for g in p.groups]
    assert names == [("value", ("alive", "ki"), Backend.BANDED, VALUE, False),
                     ("stats", ("stats_v0", "stats_v1"), Backend.BANDED, STATISTICS_LEGACY, True)]


def test_statistics_only_request_has_no_value_group():
    p = prepare(SnowballPDESolver(PDEParams()), snowball("discrete"), env_flat(),
                OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True, external_pv=1.0))
    assert [g.name for g in p.groups] == ["stats"]
```

Rewrite `test_event_stats_pruning.py`:
- It wraps `_compute_event_stats` to observe the `streams` argument. Wrap `core.statistics.StatisticsLegacy.__init__` instead, and assert that the recorded `want_ki` / `want_coupon` flags match what the old test asserted about the pruned columns.

Rewrite `test_trade_stream_wiring.py` the same way.

- [ ] **Step 2: Run the tests to verify they fail.** The Review Focus test fails with the stale-cache difference.
- [ ] **Step 3: Implement `core/statistics.py`.** The source is `snowball_pde_solver.py:_compute_event_stats`. Move each block verbatim, with `self.` rewritten to `solver.`.

| `StatisticsLegacy` part | Legacy lines | Change |
|---|---|---|
| `__init__(prepared)`: validation, `already_knocked_in`, `knocked_in_at_valuation`, `ko_records` from `product.resolve_ko_observations(env)` filtered by tau, the column layout (`n_ko`, the extra columns via `solver._n_extra_event_cols`, `ki_col`, `ki_ever_col`, `want_ki`, `want_coupon`), and `t0_capture` | 850-948, 1045-1052 | **KO indices:** replace the nearest-match loop (903-916) with `0` if `is_close(t, 0)`, `num_t - 1` if `is_close(t, tau)`, else `layout.time.step_at(t)`, keeping its `ValidationError` for an unmatched time (§8). The inline solve-state copy is dropped; `prepare` step 2 already ran `_prepare_solve_state` (§5.2). Raise `ValidationError("Event stats PDE currently requires banded solver path.")` when `(num_x - 2) <= 2`, as at 1030. |
| `group` | — | `ColumnGroup("stats", ("stats_v0", "stats_v1"), BANDED, STATISTICS_LEGACY, rolling=True)` |
| `allocate(states)` | 953-957 | `stats_v0` and `stats_v1` are `(num_x, n_cols)` zeros; `stats_v1[:, ki_col] = stats_v1[:, ki_ever_col] = 1` when `want_ki` |
| `terminal(states)` | 959-1015 | verbatim |
| `step_events(states, captures, j)` | 1113-1191 | verbatim, in order: t0 capture, KO, extra indicators, KI plus first-passage, zero-slope. It operates on `states["stats_v0"]` / `states["stats_v1"]`, which the march has just replaced with the advanced columns. |
| `read(states, captures, pv)` | 1196-1423 | verbatim. `pv` comes from the caller: `external_pv`, or the value readout of the same march. |

**The march with rolling groups and the statistics hooks** (`core/march.py`). This replaces Task 5's `march` in full:

```python
def march(prepared) -> MarchResult:
    """One loop over the layout's steps, last to first (spec 2 §5.3). No product branches."""
    adapter, layout, stats = prepared.adapter, prepared.layout, prepared.statistics
    cache = prepared.factorizations
    cache.clear()
    value = prepared.request.price           # False only for a statistics-only request (2D, external pv)
    states, captures = (adapter.allocate(prepared) if value else {}), {}
    if stats is not None:
        stats.allocate(states)
    if value:
        adapter.terminal(prepared, states, captures)
    if stats is not None:
        stats.terminal(states)
    coeffs, schedule, stepper = prepared.coefficients, prepared.theta_schedule, prepared.stepper
    n_int = len(layout.spatial.x) - 2
    dt_vec = layout.time.dt
    for j in range(len(layout.time.t) - 2, -1, -1):
        dt = dt_vec[j]
        k = coeffs.set_at(j)
        for group in prepared.groups:
            theta = float(schedule[j])
            if group.policy.zero_diffusion_theta_one and coeffs.is_zero_diffusion(k):
                theta = 1.0  # upwind advection is monotone only fully implicit
            if group.rolling:                                   # statistics-legacy: carry the boundary rows
                v_next = np.hstack([states[n] for n in group.surfaces])
                lower, upper = v_next[0, :].copy(), v_next[-1, :].copy()
            else:
                adapter.boundaries(prepared, group, states, j)
                v_next = np.column_stack([states[n][:, j + 1] for n in group.surfaces])
                lower = np.array([states[n][0, j] for n in group.surfaces])
                upper = np.array([states[n][-1, j] for n in group.surfaces])
            v_next.setflags(write=False)
            gdt = float(dt) if group.policy.dt_as_python_float else dt
            out = stepper.step(GroupState(group, v_next), coeffs.step_set(k, cache), gdt, theta, (lower, upper))
            validate_advanced(out, group, n_int, stepper)
            if group.rolling:
                col = 0
                for name in group.surfaces:
                    width = states[name].shape[1]
                    new = states[name].copy()                   # legacy: v_cur = v_next.copy()
                    new[1:-1, :] = out[:, col:col + width]
                    states[name] = new
                    col += width
            else:
                for i, name in enumerate(group.surfaces):
                    states[name][1:-1, j] = out[:, i]
                adapter.after_solve(prepared, group, states, j)
        if value:
            adapter.step_events(prepared, states, captures, j)
        if stats is not None:
            stats.step_events(states, captures, j)
    return MarchResult(states=states, captures=captures)
```

- `PreparedProblem` gains `statistics: Optional[StatisticsLegacy]`, which is `None` unless `request.event_stats` is set and the adapter supports the group.
- The statistics group is appended to `groups` after the value groups.
- With `request.price` false, the value groups are omitted and the adapter's `allocate` / `terminal` / `step_events` are skipped (the `value` guard). The solver state that the statistics read (`_ki_fp`, the observation maps, the KI regime) is set in `prepare` steps 2, 6 and 7, never in those hooks. That is why skipping them is safe.

**Bit-identity for the reviewer.** In the legacy code, `v_cur = v_next.copy()` carried the boundary rows forward, and the injection read `v_cur[0]` (equal to `v_next[0]`). Here `lower` and `upper` are copies of those same rows. The statistics key is rounded on a Python `float` `dt` (legacy 1057), so the key flavour is preserved by `dt_as_python_float`. The value and statistics groups share one `FactorizationCache`. At step `j` they ask for `(k, round(dt, 12), θ)` on the same `dt`, so a shared entry is always first built on the same step, from the same inputs, as each march would have built it alone.

**If a `stats|*` or `pwe|*` row diffs**, measure whether a statistics key has a value-group twin that was built at a different step. That can happen only if NumPy and Python `round` split an interval's keys differently. If so, give the statistics group its own namespace in the cache (add `group.policy` to the key), re-run, and record the finding. This is a namespacing choice that preserves semantics; it is not a fallback.

**Phoenix and KO-reset.**
- The Phoenix extra columns go through `solver._n_extra_event_cols`, `_set_extra_event_indicators`, `_extract_extra_event_stats` and `_t0_extra_indicator_overrides`, which are unchanged.
- KO-reset keeps its QUAD delegation (`ko_reset_snowball_pde_solver.py:70-122`). `KOResetAdapter.supports_statistics_group = False`. For KO-reset, `price_with_events` / `_session_outputs` run the value march and then call the solver's own `calculate_event_stats(npv=pv, streams=…)`, as today.

**Facades** (`snowball_pde_solver.py`):
- `calculate_event_stats(p, env, *, npv=None, streams=None)` keeps the type check and returns `None` when `env is None`. With `npv` given, it runs `OutputRequest(EVENT_STATS, price=False, event_stats=True, external_pv=npv, streams=streams)`. Without it, it runs `OutputRequest(EVENT_STATS, price=True, event_stats=True, streams=streams)`: one march, and the statistics read with `pv` equal to the value readout. `calculate_event_stats` no longer re-prices.
- `EVENT_STATS` terminal outcomes follow §5.2 step 0: `tau <= 0` gives `None` before any grid. A product knocked out at valuation has **no** short-circuit and marches.
- `price_with_events`:
  - It runs `OutputRequest(PRICE_WITH_EVENTS, event_stats=True, streams=streams)`.
  - The terminal outcome is `(terminal price, EventDistribution.trivial(max(tau, 0)))`, as in 742-772.
  - Otherwise the result is the price plus the statistics, converted with the existing post-processing code in `price_with_events`, unchanged.
- `_session_outputs(p, env, want_events=False, want_grid=False, streams=None)` returns `PDESessionOutputs` from `OutputRequest(SESSION_OUTPUTS, …)`. It keeps its signature, because `quantark/execution/registry.py:188` and the adapters call it.
- The 2D solvers (`snowball_vol_pde_solvers.py:583-602`, `phoenix_vol_pde_solvers.py:419-443`) replace `SnowballPDESolver.calculate_event_stats(self, …, npv=pde_pv, streams=streams)` with `self._run(product, env, OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True, external_pv=pde_pv, streams=streams))`. The adapter on a 2D instance is `SnowballAdapter` / `PhoenixAdapter` with BSM samples, which is today's 1D implied-vol statistics (§8).

- [ ] **Step 4: Delete the legacy code and run the tests.** Delete `_compute_event_stats`, `_get_banded_system`, `_pack_banded_entry`, `_pack_uses_banded`, the unused coefficient delegates, and `_get_matrices`. Then run:

```bash
grep -rn "_compute_event_stats\|_get_banded_system\|_get_matrices\|_calculate_coefficients\|_flat_exact_step_coefficients" quantark test
```

The only allowed hits are the QUAD engines' own `_compute_event_stats` methods (`snowball_quad_engine.py`, `ko_reset_snowball_quad_engine.py`), which are different methods, plus tests you have not rewritten yet. Rewrite those tests now:
- `test_pde_term_coefficients.py` and `test_pde_term_structure_solvers.py` move to `CentralDifference().assemble_arrays` / `scalar_stencil` / `flat_exact`, keeping the same assertions.
- `test/helpers_pde_zero_diffusion.py` moves to `CentralDifference().assemble_sets`.

Then run `… -m pytest test/pde_core test/test_event_stats_pruning.py test/test_trade_stream_wiring.py test/test_pde_engine_event_stats.py test/test_quad_pde_event_stats.py test/test_price_with_events_single_pass.py test/test_settlement_event_stats_reconciliation.py test/test_heston_pde_engines.py test/test_heston_slv_pde_engines.py -q`.
- [ ] **Step 5: Run the byte gate and the identity gate.** Every `stats|*`, `pwe|*`, `*|stats` (2D), `terminal|*|stats` (declared `None`), `settlement_lag|*` and `lu_fallback|snowball|stats` (declared `ValidationError`) row must show 0 diffs, and so must every facade row with `stats_*` outputs.
- [ ] **Step 6: Commit.** Message: `refactor(pde)!: event statistics are a column group of the price march (Level 1, bitwise); calculate_event_stats no longer re-prices; stale statistics factorizations fixed (spec 2 §6, decision 2)`.

---

### Task 10: One preparation path for sessions and bump contexts

**Files:**
- Modify:
  - `pde_session_prep.py` (all three artefact states and `_BUILDER_VERSION`);
  - `pde_execution_adapters.py:175-299, 404-434`;
  - `base_pde_solver.py`, which loses:
    - `create_bump_context` (the new body is below);
    - `_bump_clone`, `_bump_transient_attrs`, `_session_grids`, `_session_step_coefficients`, `_session_matrix_pack`, `_layout_from_session_grids`, `_session_factorization_packs`;
    - `_critical_points_cache`, `_critical_points_cache_key`, `_get_cached_critical_points`, `_params_cache_key`;
    - the `_matrix_cache` / `_banded_cache` properties and the solver-level `_factorizations`;
    - `_build_grids`, `_grids_via_layer`, `_bound_layout_for_solve`;
  - `snowball_pde_solver.py`, which loses `_session_banded_pack` and `_prepare_for_request`;
  - `core/prepare.py`, which loses `SessionArtefacts.from_solver`.
- Create: `quantark/asset/equity/engine/pde/_legacy_bump.py` (`LegacyBumpCloneMixin`, for the 2D solvers only).
- Modify: `snowball_vol_pde_solvers.py` and `phoenix_vol_pde_solvers.py`. The 2D bases take `LegacyBumpCloneMixin` first in their MRO.
- Test:
  - Rewrite `test/execution/test_pde_artifact_reuse.py`, `test_pde_prepared_adapters.py` and `test_pde_session_seams.py`.
  - Rewrite `test/pde_grid/test_frozen_bump_layouts.py` and `test_solver_seam.py`.
  - Rewrite `test/test_pde_fixed_bump_grid.py`, `test/test_snowball_pde_date_schedule.py:102` and `test/test_bump_grid_steadiness.py`.
  - Extend `test/pde_core/test_review_focus.py` (Review Focus 4).
  - Update `test/pde_core/test_factorization_cache.py`: the Task 2 tests move to `FactorizationCache`.

**Interfaces:**
- Consumes: `prepare(…, frozen=, session=)`, `FactorizationCache`, `layout_fingerprint`, `coefficient_key`.
- Produces:
  - `bind_layout(solver, product, env, request, *, frozen=None) -> Optional[Layout]` in `core/prepare.py`. It runs `prepare` steps 0–5 and returns `None` when the request is terminal. `prepare` itself calls it.
  - `warm_factorizations(prepared, max_entries) -> dict[Backend, Mapping]` in `core/march.py`.
  - `solver._session_artefacts: Optional[SessionArtefacts]`.
  - `solver._constructor_kwargs() -> dict`: base `{"params": self.params}`; LV adds `local_vol_surface=self._prebuilt`.

- [ ] **Step 1: Write the failing tests.**

Review Focus 4:

```python
def test_session_runs_two_products_with_shared_geometry_and_different_coupons():
    env = env_flat()
    a = make_snowball(dict(SB, ko_rate=0.15, rebate_rate=0.15))
    b = make_snowball(dict(SB, ko_rate=0.30, rebate_rate=0.30))
    engine = SnowballPDESolver(PDEParams())
    with PricingSession() as session:
        va = session.execute(engine, PricingRequest(product=a, pricing_env=env)).value
        vb = session.execute(engine, PricingRequest(product=b, pricing_env=env)).value
    assert va == SnowballPDESolver(PDEParams()).price(a, env)
    assert vb == SnowballPDESolver(PDEParams()).price(b, env)
    assert va != vb   # measured §14: 97.741… vs 100.566…
```

Add the grid-reuse assertion the way `test/execution/test_pde_artifact_reuse.py` asserts reuse today. Read it first; it inspects the session's artefact cache or the `_session_*` attributes. Assert that product `b` hit the `"pde-grid"` artefact.

Session tests:
- the fingerprint moves when the stepper configuration moves. Until Task 12, that is `PDEParams(theta=1.0)`; Task 12 changes this line to `stepper=CrankNicolson(theta=1.0)`.
- the stored `Layout` is the one execute binds: `prepared.layout is artefact.value`.
- `pde_session_prep._BUILDER_VERSION == "pde-prep/v2"`.
- Phoenix and KO-reset executed through a session honour the injected coefficients. Spy on `CentralDifference.assemble_sets` and assert zero calls on execute.

- [ ] **Step 2: Run the tests to verify they fail.**
- [ ] **Step 3: Sessions.**
  - `grid_state`: the fingerprint tuple is **unchanged** (§5.9 grid artefact key). The value is now `bind_layout(engine, product, env, OutputRequest(Operation.PRICE))`, a `Layout`, not the 5-tuple. `_grids_nbytes` measures the layout's arrays.
  - `step_coefficients_state`:
    - The fingerprint is `("pde-step-coeffs", class path, *coefficient_key(layout.fingerprint, market_fingerprints(env), strike, component_fingerprint(operator), lv_surface_fp))`.
    - `None` means build fresh and store no artefact.
    - The value is the pre-flat-exact `StepCoefficients`, built through the adapter's `coefficients` hook with `session=None`. Expose the hook's pre-flat-exact object as `adapter.step_sets(...)`.
  - `factorization_state`:
    - The fingerprint is `("pde-fact", class path, coefficient fp, component_fingerprint(stepper), tuple(g.backend.value for g in groups), fingerprint(per-group θ schedules))`. The per-group schedule is `schedule` with θ=1 substituted on zero-diffusion sets for groups with the override (decision 17).
    - The value is `warm_factorizations(prepared, max_entries=_PACK_MAX_ENTRIES)`, which walks the march's step order and groups. It **does** apply the zero-diffusion override; the legacy builder did not, and missed. It fills a fresh `FactorizationCache` until `max_entries` entries exist, then returns `{backend: MappingProxyType(dict(cache.store(backend)))}`.
    - Warmed entries are built at their first march-order occurrence from the same inputs as the march, so they are bitwise the same. Keep the `_EMPTY_PACKS` budget rule.
  - `_BUILDER_VERSION = "pde-prep/v2"`.
  - `pde_execution_adapters._prepare_artifacts`: delete the `_prepare_solve_state` call and the four `_session_*` assignments. It sets `clone._session_artefacts = SessionArtefacts(layout=grid.value, coefficients=coeff.value, lu_pack=packs.get(Backend.SPARSE_LU), banded_pack=packs.get(Backend.BANDED))`. The LV variant keeps arming `_active_lv_surface` around preparation. `_execute` calls `clone.price(...)`, or `clone._session_outputs(...)` plus `_grid_projection_from_solution`, unchanged.
  - `BasePDESolver._run` passes `session=self._session_artefacts`.
- [ ] **Step 4: Bump contexts.**

```python
def create_bump_context(self, product, pricing_env):
    """A fresh instance with the parent's constructor arguments, holding the frozen base layout (§5.10)."""
    layout = bind_layout(self, product, pricing_env, OutputRequest(Operation.PRICE))
    if layout is None:          # terminal: bumping re-prices the short-circuit, as today (returned self)
        return self
    clone = type(self)(**self._constructor_kwargs())
    clone._frozen_base_layout = layout
    return clone
```

  - Legacy `create_bump_context` returned `self` when `tau <= 0`, but ran `_prepare_for_request` and `bind` for an immediate-KO product. `bind_layout` returns `None` for **any** terminal outcome of `Operation.PRICE`. That includes immediate KO, which the legacy path bound anyway.
  - A frozen layout is never read by a terminal price, so the bumped numbers are unchanged. The `terminal|immediate_ko|*` rows exercise the bump Greeks through `facade_outputs` and prove it.
  - If they diff, a bumped spot must have left the KO region (the bump moved the product out of the terminal state). In that case keep binding for immediate KO: return `None` only for `tau <= 0`, and bind otherwise. That is the legacy rule exactly. Record which one you shipped.
  - `PDEEngine.create_bump_context` is unchanged.
  - Knock-in composites: the knock-out part is prepared with `frozen=self._frozen_base_layout`, and the vanilla part is fresh on every call (decision 21).
- [ ] **Step 5: The 2D legacy clone.** Move `create_bump_context` (base 365-392), `_bump_clone`, `_bump_transient_attrs` (base 394-430) and `_prepare_for_request` (snowball 2228-2235) into `_legacy_bump.LegacyBumpCloneMixin`, verbatim.
  - Its docstring: "Deep-copy bump clone kept for the 2D Heston/SLV solvers only: their `calculate_greeks` (`snowball_vol_pde_solvers.py:654`) and 2D S-axis freezing rely on it. Sub-project 3 moves the 2D solvers onto the core and deletes this."
  - The 2D bases (`_Heston2DSnowballPDEBase`, `_Heston2DPhoenixPDEBase`) inherit it first. Their existing `_bump_transient_attrs` / `create_bump_context` overrides keep calling `super()`.
  - `_bump_transient_attrs` names attributes that no longer exist on the 1D base: `_session_grids`, `_matrix_cache`, `_critical_points_cache`. The mixin's list keeps only names that still exist, and checks each with `hasattr`, so the deep copy still drops the binder and layouts.
  - The book's `heston_snowball|greeks` row (Task 1) proves this step bitwise.
- [ ] **Step 6: Delete the legacy code and rewrite the tests.**
  - Delete everything listed under Files. Then run:
    ```bash
    grep -rn "_session_grids\|_session_step_coefficients\|_session_matrix_pack\|_session_banded_pack\|_bump_clone\|_bump_transient_attrs\|_prepare_for_request\|_layout_from_session_grids\|_session_factorization_packs\|_params_cache_key\|_get_cached_critical_points\|_critical_points_cache\|_build_grids\|_grids_via_layer\|_matrix_cache\|_banded_cache" quantark test
    ```
    The only allowed hits are `_legacy_bump.py`, the 2D modules, and `grid/binder.py:183`: `GridLayerMixin._prepare_grid_request` uses `getattr(self, "_prepare_for_request", None)`; leave it, because DCN never defines it.
  - Rewrite every test hit with the same assertions against `prepare` / `bind_layout` / `create_bump_context` / `solver._session_artefacts` / `FactorizationCache`.
- [ ] **Step 7: Run the tests and the gates.** Run `… -m pytest test/execution test/pde_grid test/pde_core test/test_pde_fixed_bump_grid.py test/test_bump_grid_steadiness.py test/test_snowball_pde_date_schedule.py test/test_heston_pde_engines.py test/test_heston_slv_pde_engines.py -q`, then the byte gate and the identity gate. The `session_*`, `bump|*`, every `bump_*` output and `heston_snowball|greeks` must show 0 diffs. Then run `test/execution/test_pde_pre_refactor_goldens.py` and `test_pde_convergence_gate.py`. `pde_phase4_goldens.json` names no artefact kind and no `pde-prep/v1` (verified), so it is **not** re-frozen.
- [ ] **Step 8: Commit.** Message: `refactor(pde)!: one preparation path - sessions store the Layout (pde-prep/v2), bump contexts are fresh instances with a frozen layout; transient-attribute machinery deleted (2D keeps a legacy clone) (spec 2 §5.10)`.

---

### Task 11: `explain`, `PDEPlan`, the problem identity; the intraday route reads the plan

**Files:**
- Create: `core/plan.py`.
- Modify:
  - `core/prepare.py`: step 12, the problem identity;
  - `base_pde_solver.py`: `explain`; delete `_theta_schedule_from_layout`, and delete `self._active_layout` / `solver._active_layout = layout` in `prepare`;
  - `pde_engine.py`: `explain` delegates;
  - `quantark/intraday/engines/pde.py:29-140, 178-214, 264-269`.
- Test: `test/pde_core/test_prepare.py` and `test/pde_core/test_identity.py` (new); rewrite `test/intraday/test_route_pde.py:174,196`, `test/pde_grid/test_frozen_bump_layouts.py:71,191`, `test/test_pde_fixed_bump_grid.py:176`, `test/pde_grid/test_solver_seam.py:157`, `test/test_pde_event_projection.py:1049` and `test/test_pde_quad_audit_regressions.py:113`.

**Interfaces:**
- Produces:
  - `solver.explain(product, env, *, lifecycle_state=None) -> PDEPlan`, on every core solver and on `PDEEngine`.
  - `PDEPlan`, a frozen dataclass with sections `route`, `settings`, `space`, `time`, `events`, `scheme`, `outputs`, `identity`, `terminal`, a `layout_bound: bool`, plus `parts: tuple[PDEPlan, ...]` for a composite. It has `.to_dict()` and `__str__`. Each section is a `Mapping`. `plan.time.actual_steps` is attribute access through a `SimpleNamespace` view of the time section, which the budget row uses.
  - `problem_identity(prepared) -> Optional[str]`.
  - `factorization_key(prepared) -> Optional[str]`.

- [ ] **Step 1: Write the failing tests.**

`test/pde_core/test_prepare.py`, following §9.3:

```python
def test_explain_reports_the_cap_on_a_five_year_standard_snowball():
    p = make_snowball(dict(SB, months=60, maturity=5.0))
    plan = SnowballPDESolver(PDEParams()).explain(p, env_flat())
    assert (plan.time.requested_steps, plan.time.actual_steps, plan.time.fill_scaled) == (5040, 5000, True)


def test_explain_reports_the_cap_on_a_three_year_high_snowball():
    p = make_snowball(dict(SB, months=36, maturity=3.0))
    plan = SnowballPDESolver(PDEParams(accuracy="high")).explain(p, env_flat())
    assert (plan.time.requested_steps, plan.time.actual_steps) == (6048, 5000)


def test_explain_raises_where_price_would():
    bad = make_snowball(SB)
    s = SnowballPDESolver(PDEParams(grid=GridConfig(points=3, eps_crit=1.0)))   # measured: price raises
    with pytest.raises(ValidationError):
        s.price(bad, env_flat())
    with pytest.raises(ValidationError):
        s.explain(bad, env_flat())


def test_terminal_plan_on_an_expired_and_a_knocked_out_product():
    product = expired_snowball()
    plan = SnowballPDESolver(PDEParams()).explain(product, env_after_maturity(product))
    assert plan.layout_bound is False
    assert plan.terminal == {"price": "expired", "greeks": "expired", "price_with_events": "expired",
                             "event_stats": "expired", "life_surface": "raises ValidationError"}
    plan = SnowballPDESolver(PDEParams()).explain(snowball(), immediate_ko_env())
    assert plan.terminal["price"] == "knocked_out_at_valuation"
    assert plan.terminal["event_stats"] == "marches"          # decision 16: no short-circuit
    assert plan.terminal["life_surface"] == "raises ValidationError"


def _record_problem_identities(monkeypatch):
    import quantark.asset.equity.engine.pde.core.prepare as prep_mod
    seen, real = [], prep_mod.march
    def spy(prepared):
        seen.append(prepared.identity["problem"])
        return real(prepared)
    monkeypatch.setattr(prep_mod, "march", spy)
    return seen


def test_every_entry_point_reaches_the_explained_problem_identity(monkeypatch):
    p, env = make_snowball(SB), env_flat()
    want = SnowballPDESolver(PDEParams()).explain(p, env).identity["problem"]
    assert want is not None
    seen = _record_problem_identities(monkeypatch)
    SnowballPDESolver(PDEParams()).price(p, env)                                   # seen[0]
    SnowballPDESolver(PDEParams()).create_bump_context(p, env).price(p, env)       # seen[1]: same spot, same layout
    with PricingSession() as session:                                              # seen[2]: session execute
        session.execute(SnowballPDESolver(PDEParams()), PricingRequest(product=p, pricing_env=env))
    SnowballPDESolver(PDEParams()).calculate_greeks(p, env)                        # seen[3]
    assert seen[:3] == [want, want, want]
    assert seen[3] != want      # the output request is part of the problem identity
```

- `expired_snowball()`, `env_after_maturity()` and `immediate_ko_env()` are the Task 1 fixtures. Task 1 must define them under exactly these names in `capture_book.py`.
- `plan.layout_bound` is `False` for a terminal plan.
- `run` resolves `march` through the module global of `core.prepare`, so the spy sees every entry point.

Economic-split tests:
- two snowballs that differ only in KO coupon share `plan.identity["grid_artefact"]`, `["coefficient_key"]` and `["factorization"]`, and differ in `["problem"]`;
- the same holds for knocked-in against alive (`_otc_lifecycle_knocked_in`), and for a settlement lag against none.

`test/pde_core/test_identity.py`, following §9.3:
- **Budget with no interior node vs a readout node vs an event node.** Three `GridRequest`s with `GridConfig(time_steps=3)`, `tau=1.0`: no events; `extra_times=(2/3,)`; `event_times=(2/3,)`. Bind each with a `GridBinder`. Assert that `t` is bitwise equal across all three, that `layout.fingerprint` differs, and that `CrankNicolson().resolved((1, 2)).schedule(layout)` differs between the readout and event cases. That schedule enters `factorization_key`. This is the measured example from §14.
- **Holiday on a product-supplied settlement calendar.** Add a holiday and assert that the observation key and the problem identity both move.
- **Frozen bump vs fresh at the same bumped spot.** The layout fingerprint, coefficient key and problem identity all differ (the book's `bump|frozen_vs_fresh` prices differ).
- **LV prebuilt surfaces.** Two `LocalVolSnowballPDESolver`s with prebuilt 15% and 35% surfaces (built from two `GridVolSurface`s via `_build_surface`) have different coefficient keys.
- **A bump instance of an LV solver carries the prebuilt surface.** Assert `clone._prebuilt is solver._prebuilt`.

The operator and readout identity tests need the component slots (Task 12), so they go in Task 13.

- [ ] **Step 2: Run the tests to verify they fail.**
- [ ] **Step 3: Problem identity (step 12).** Implement `core/identity.py`:

```python
def factorization_key(prepared) -> Optional[str]:
    parts = (prepared.identity["coefficient_key"], component_fingerprint(prepared.stepper),
             tuple(g.backend.value for g in prepared.groups),
             tuple(_group_schedule(prepared, g) for g in prepared.groups))
    return None if parts[0] is None or parts[1] is None else fingerprint(parts)


def problem_identity(prepared) -> Optional[str]:
    fk = factorization_key(prepared)
    rd = component_fingerprint(prepared.readout)
    if fk is None or rd is None:
        return None
    env, product = prepared.env, prepared.product
    return fingerprint((
        fk, rd,
        prepared.solver._product_cache_token(product, "standard"),
        _plain(prepared.solve_state), prepared.events,
        (env.valuation_date, env.day_count_convention, int(env.bus_days_in_year)),
        calendar_identity(effective_settlement_calendar(product, env)),
        prepared.solver.params.event_projection, _plain(dataclasses.asdict(prepared.request)),
    ))
```

- `_group_schedule` returns the θ array with the zero-diffusion override substituted for override-policy groups.
- `_plain` converts enums, frozensets and datetimes to canonicalizable values: frozensets become sorted tuples. `canonical_tree` rejects `set` and non-string-keyed `dict`s, so every event-evidence row must be a tuple of plain values. `EventSchedule.evidence()` and the Phoenix/KO-reset `prepare_events` rows follow that rule.
- A product token that falls back to `id(...)` makes the identity process-local. That is acceptable, because the identity keys nothing persistent.

- [ ] **Step 4: `PDEPlan` and `explain`.**
  - `explain` = `build_plan(prepare(self, product, env, OutputRequest(Operation.PRICE), lifecycle_state=…))`. The plan describes the **price** request, so its `identity["problem"]` is the one `price` marches with. It has no march.
  - `build_plan` adds `terminal`, which evaluates `adapter.terminal_outcome` for each of PRICE, GREEKS, PRICE_WITH_EVENTS, EVENT_STATS and LIFE_SURFACE, and reports the reason, "marches", or "raises <Type>".
  - A composite gives a plan with `parts`, one per part, and `route["composition"]` set to the description.

  Section contents come from the `PreparedProblem`. Never recompute them.

  | Section | Source |
  |---|---|
  | `route` | `type(solver).__name__`, `type(adapter).__name__`, `samples().kind`, `solve_state` |
  | `settings` | per `PDEParams` field and per `GridConfig` field: requested (params/overlay), effective (`solver.grid_binder.config`), source (`preset`/`overlay`/`route`), applicability (from `route_applicability.ROUTE_RULES` for the solver's route; `honoured` on shared-grid routes) |
  | `space` | `len(x)`, `layout.spatial.bounds`, the rule per side (`request.hard_lower/upper` gives "hard edge", `config.bounds` gives "bounds", else "envelope"), `achieved_eps`, `active_critical_prices`, `ignored_critical_prices` |
  | `time` | the basis in force (`time_steps` or `steps_per_day`) and its value, `day_count` when the density basis is in force, `requested_steps`, `actual_steps`, `fill_scaled`, `cap_exceeded`, mandatory nodes (event and extra times) |
  | `events` | `prepared.events` |
  | `scheme` | `stepper.evidence()` (resolved), the counts of terminal- and event-damped steps, `{θ: count}`, the zero-diffusion step count, and `theta_by_step` (array, `to_dict` only) |
  | `outputs` | per group: name, surfaces, column count, backend; retention (`full` for value groups, `rolling` for statistics); readout mode (`t0 capture` / `final column` / `override`) |
  | `identity` | `layout`, `grid_artefact` (the `pde_session_prep` grid fingerprint tuple, hashed), `coefficient_key` (hashed), `factorization`, `problem`, `fingerprintable: bool` |

  `__str__` prints each section as one aligned block.

  **Routes without `explain` (§4.7).**
  - DCN and the local-vol vanilla and barrier solvers do not inherit `BasePDESolver`. Pin that with `assert not hasattr(DCNPDEEngine, "explain")`, and the same for `LocalVolDCNPDEEngine`, `LocalVolPDESolver` and `LocalVolBarrierPDESolver`, in `test_prepare.py`.
  - The 2D Heston/SLV snowball and phoenix bases *do* inherit it from the 1D solvers, and there it would describe the 1D implied-vol problem, not the 2D solve. Override it in `_Heston2DSnowballPDEBase` and `_Heston2DPhoenixPDEBase` to raise `PricingError("explain covers the 1D core routes; the 2D Heston/SLV solvers move onto the core in sub-project 3")`, and test that it raises.
  - This decision is beyond the spec. It is listed in the self-review notes. Node lists and arrays appear only in `to_dict()`.
- [ ] **Step 5: The intraday route reads the plan.**
  - `_layout_numbers(solver, spot)` becomes `_layout_numbers(plan, spot)`, reading `plan.to_dict()["space"]` / `["time"]` and the `x` array from `to_dict()`.
  - `_barrier_placement` and `_dx_min` read the plan in the same way.
  - `_time_status` reads `plan.to_dict()["scheme"]["theta_by_step"]` and the `t` array.
  - `_time_fill_for` keeps probing `build_time(plan_request, …)` ("the refined request is built as today", §5.10). The request comes from `plan.to_dict()["time"]["request"]`: add the `GridRequest` there.
  - `PDERoute.price` calls `solver.explain(product, env)` once after pricing. Delete the `_active_layout = None` resets. `_method` reads `params.stepper` in Task 12; for now it keeps `resolved().terminal_damping_steps`.
- [ ] **Step 6: Delete `_active_layout` and `_theta_schedule_from_layout`, and rewrite the tests.** Run `grep -rn "_active_layout\|_theta_schedule_from_layout" quantark test`. The only allowed hits are the 2D modules' own `_active_layout` uses, if they set their own; check them. Rewrite the six test hits against `explain` (for example, `solver._active_layout.time.actual_steps` becomes `solver.explain(p, env).time.actual_steps`), and `_theta_schedule_from_layout(layout)` becomes `prepared.theta_schedule`. The book's `budget_row` already detects `explain`.
- [ ] **Step 7: Run the tests and the gates.** Run `… -m pytest test/pde_core test/intraday test/pde_grid test/test_pde_fixed_bump_grid.py test/test_pde_event_projection.py test/test_pde_quad_audit_regressions.py -q`, then the byte gate (`intraday_snowball`, `*|budget_eq_density`) and the identity gate. The intraday schema-2 certificate's implementation digest may move; that is by design, as in 0.6.0 (§9.2). Its anchors must pass.
- [ ] **Step 8: Commit.** Message: `feat(pde): explain() - the resolved plan from the one preparation; problem identity; intraday route reads the plan (spec 2 §4.7, §5.9)`.

---

### Task 12: The scheme relocation. `stepper`, `operator` and `readout` on `PDEParams`; `GridConfig` down to nine fields

**Files:**
- Modify:
  - `quantark/asset/equity/param/engine_params.py`: the `PDEParams` fields, `__post_init__`, `from_config`, and the docstring;
  - `quantark/asset/equity/param/pde_removed_names.py`: the four new entries, five re-pointed entries, `REMOVED_GRID_NAMES`, and a per-name version;
  - `grid/config.py`: nine fields; `_SHARED_DEFAULTS` and `DCN_ACCURACY_PROFILES` each lose two keys; `_validate_resolved` loses the damping checks; add `DAMPING_DEFAULTS`;
  - `grid/time.py`: `TimeLayout` loses `event_damping_steps` and `terminal_damping_steps`, and `build_time` stops reading the counts;
  - `route_applicability.py`: re-keyed rules, the `stepper.` prefix, and the component rejections;
  - `core/components.py`: delete `scheme_stepper`; add `resolve_stepper`;
  - `core/adapters/base.py`: `stepper()`, `coefficients()` and the readout come from `params`;
  - `dcn_pde_solver.py:270-298`;
  - `snowball_vol_pde_solvers.py:34-61, 993-1024` and `phoenix_vol_pde_solvers.py:525-565`;
  - `local_vol_pde_solver.py:107-108` and `barrier_vol_pde_solvers.py:196`;
  - `quantark/intraday/engines/pde.py:264-269`;
  - `quantark/asset/equity/engine/pde_engine.py:75` (the docstring);
  - `quantark/asset/equity/engine/validation/script/{boundary,benchmark}_check_{snowball,barrier}_pde_solver.py`.
- Test:
  - `test/test_pde_params_surface.py`;
  - `test/pde_grid/test_time_builder.py`, `test_grid_config.py`, `test_grid_precedence.py`, `test_solver_seam.py`;
  - `test/execution/matrix_fixtures.py:81-96, 750`;
  - every test on the Task-12 list in Step 5.

**Interfaces:**
- Consumes (Task 4): `CrankNicolson`, `CentralDifference`, `StandardReadout`.
- Produces:
  - `PDEParams.stepper`, `.operator` and `.readout`. They are declared `Optional[...] = None` and normalised to the built-ins in `__post_init__`, so after construction `PDEParams().stepper == CrankNicolson()`.
  - `resolve_stepper(params, damping_defaults) -> TimeStepper`.
  - `DAMPING_DEFAULTS = {"core": (1, 2), "DCN": (2, 2)}` in `grid/config.py`.
  - `REMOVED_GRID_NAMES`.

The field defaults are `None`, not `CrankNicolson()` as §4.1's code block shows. `param` must not import `engine.pde` at module load: importing `…pde.core.components` executes `engine/pde/__init__.py`, which imports the solvers, which import `param`, which is a cycle. `grid` already uses the same lazy pattern. The observable contract (`PDEParams().stepper == CrankNicolson()`) is the spec's.

- [ ] **Step 1: Write the failing tests** in `test/test_pde_params_surface.py`.

```python
@pytest.mark.parametrize("kw,fragment", [
    ({"theta": 1.0}, "stepper=CrankNicolson(theta=...)"),
    ({"event_theta": 0.7}, "stepper=CrankNicolson(event_theta=...)"),
])
def test_removed_scheme_names_on_pdeparams_name_the_stepper(kw, fragment):
    with pytest.raises(ValidationError, match=re.escape(fragment)):
        PDEParams(**kw)


@pytest.mark.parametrize("name", ["terminal_damping_steps", "event_damping_steps"])
def test_removed_damping_counts_on_gridconfig_name_the_stepper(name):
    with pytest.raises(ValidationError, match=re.escape(f"PDEParams(stepper=CrankNicolson({name}=N))")):
        GridConfig(**{name: 1})


def test_removed_in_070_names_say_070():
    with pytest.raises(ValidationError, match="removed in 0.7.0"):
        PDEParams(theta=1.0)


def test_default_stepper_is_crank_nicolson_and_round_trips_from_config():
    assert PDEParams().stepper == CrankNicolson()
    p = PDEParams.from_config({"stepper": {"theta": 1.0, "terminal_damping_steps": 0}})
    assert p.stepper == CrankNicolson(theta=1.0, terminal_damping_steps=0)


@pytest.mark.parametrize("key", ["operator", "readout"])
def test_components_cannot_be_named_from_a_config_file(key):
    with pytest.raises(ValidationError, match=key):
        PDEParams.from_config({key: {}})


def test_gridconfig_key_has_nine_entries():
    assert [f.name for f in dataclasses.fields(GridConfig)] == [
        "points", "steps_per_day", "time_steps", "eps_crit", "num_std", "bounds",
        "max_points", "max_steps", "day_count"]
    assert len(GridConfig().key) == 9


def test_crank_nicolson_validates_its_four_settings():
    for bad in ({"theta": 1.5}, {"event_theta": 0.0}, {"terminal_damping_steps": -1},
                {"event_damping_steps": 1.5}, {"event_damping_steps": True}):
        with pytest.raises(ValidationError):
            CrankNicolson(**bad)


def test_stepper_is_honoured_on_dcn_as_counts_and_rejected_as_a_class():
    DCNPDEEngine(params=PDEParams(stepper=CrankNicolson(terminal_damping_steps=3)))   # honoured
    class Custom(CrankNicolson): pass
    with pytest.raises(ValidationError, match="sub-project 3"):
        DCNPDEEngine(params=PDEParams(stepper=Custom()))
    with pytest.raises(ValidationError, match="sub-project 3"):
        LocalVolPDESolver(params=PDEParams(operator=ShiftedOperatorStub()))
```

`ShiftedOperatorStub` is a frozen dataclass subclass of `CentralDifference` with one float field. It lives in `test/pde_core/components_fixtures.py`, which Task 13 extends. `Custom(CrankNicolson)` counts as "not a `CrankNicolson`" by an exact-type check: `type(stepper) is CrankNicolson`.

- [ ] **Step 2: Run the tests to verify they fail.**
- [ ] **Step 3: Implement.**
  1. **The removal map.**
     - Add `REMOVED_IN: Mapping[str, str]`. It maps `theta`, `event_theta`, `terminal_damping_steps` and `event_damping_steps` to `"0.7.0"`. Every other name defaults to `"0.6.0"`.
     - `removed_name_error` renders `was removed in {REMOVED_IN.get(name, '0.6.0')}`.
     - Add these entries. Each text is §4.4 verbatim:
       - `REMOVED_PDE_NAMES["theta"]`: `"stepper=CrankNicolson(theta=...); 0.5 is Crank-Nicolson, 1.0 backward Euler"`;
       - `REMOVED_PDE_NAMES["event_theta"]`: `"stepper=CrankNicolson(event_theta=...)"`;
       - `REMOVED_GRID_NAMES["terminal_damping_steps"]`: `"PDEParams(stepper=CrankNicolson(terminal_damping_steps=N)); 0 disables"`;
       - `REMOVED_GRID_NAMES["event_damping_steps"]`: the same for `event_damping_steps`.
     - `_TERMINAL` and `_EVENT` become the stepper texts, which re-points `use_rannacher`, `rannacher_steps`, `rannacher_at_events` and `event_rannacher_steps`. `REMOVED_DCN_NAMES["rannacher_steps"]` becomes `"PDEParams(stepper=CrankNicolson(terminal_damping_steps=N, event_damping_steps=N)) (DCN applies both in its half-step form)"`.
     - The `method` text becomes `"... stepper=CrankNicolson(theta=1.0) gives backward Euler."`.
     - `GridConfig`'s decorator becomes `@keyword_only_init(REMOVED_GRID_NAMES)`.
  2. **`PDEParams`.**
     - Delete `theta` and `event_theta`. Add `stepper: Optional[object] = None`, `operator: Optional[object] = None` and `readout: Optional[object] = None`.
     - `__post_init__` imports the built-ins lazily and replaces `None` with `CrankNicolson()`, `CentralDifference()` and `StandardReadout()`.
     - It then type-checks duck-typed conformance: `stepper` must have `schedule`, `step` and `evidence`; `operator` must have `assemble`, `evidence` and `supports`; `readout` must have `price`, `delta_gamma` and `evidence`. A missing method raises `ValidationError`.
     - `from_config`:
       - A `stepper` mapping becomes `CrankNicolson(**mapping)`.
       - An `operator` or `readout` key raises `ValidationError(f"PDEParams.from_config: {key!r} cannot be named in a config file (custom components are constructed in code; the built-ins have no settings in this release)")`.
       - A top-level `theta` / `event_theta` reaches the constructor and so goes through the removal map. The damping counts under `grid` reach `GridConfig(**grid)` and go through `REMOVED_GRID_NAMES`.
     - Docstring:
       - Replace the `theta` / `event_theta` bullets with the §4.1 `stepper`, `operator` and `readout` rows ("when to change it / what happens").
       - Correct `cache_enabled` to "Same numbers to round-off: a cached run shares one factorization across intervals whose dt differ by an ULP (Review Focus 5)."
  3. **`grid/config.py`.**
     - Delete the two `GridConfig` fields, and delete their keys from `_SHARED_DEFAULTS` and `DCN_ACCURACY_PROFILES`.
     - Delete the non-negativity checks for the two counts from `_validate_resolved`, because they move into `CrankNicolson`.
     - Add `DAMPING_DEFAULTS = {"core": (1, 2), "DCN": (2, 2)}`, with the comment "the built-in stepper's route defaults; a `None` count on `CrankNicolson` resolves to these".
  4. **`grid/time.py`.**
     - Delete the two `TimeLayout` fields and their construction (148-161).
     - Rewrite the module docstring's "damping schedules are derived here" line to "the damping schedule is the stepper's (`CrankNicolson.schedule`)".
     - `CrankNicolson._damping_sets` already derives from `request.event_times`, so nothing else changes.
     - Delete Task 4's equality test in `test_time_builder.py`, and replace it with a frozen expectation. Take the four parametrised cases, capture today's schedule arrays (by running the old test's `want` construction once at the Task 11 commit), and paste them as literal lists. They are small integer-index sets, so they are not cross-architecture floats. Also add `assert not hasattr(time, "event_damping_steps")`.
  5. **`components.resolve_stepper(params, damping_defaults)`.** It returns `params.stepper.resolved(damping_defaults)` when `type(params.stepper) is CrankNicolson`, and `params.stepper` otherwise. Delete `scheme_stepper`. `FamilyAdapter.stepper(params)` returns `resolve_stepper(params, DAMPING_DEFAULTS["core"])`.
  6. **Adapters.**
     - `FamilyAdapter.coefficients` uses the built-in memo only when `type(params.operator) is CentralDifference`.
     - Otherwise it calls `params.operator.assemble(self.samples(...), layout)`, runs `validate_coefficients`, and sets no memo and no session artefact when `component_fingerprint(params.operator)` is `None`.
     - `prepare` passes `readout=params.readout`.
     - `check_route` also checks that the samples' kind is in `params.operator.supports` (§5.8).
  7. **`route_applicability.py`.**
     - Re-key: `grid.terminal_damping_steps` becomes `stepper.terminal_damping_steps`, `grid.event_damping_steps` becomes `stepper.event_damping_steps`, and `event_theta` becomes `stepper.event_theta`.
     - `_explicit`:
       - For a `stepper.` prefix, return `None` when `type(params.stepper) is CrankNicolson` and the attribute equals `CrankNicolson()`'s default. Otherwise return the value.
       - For a `component.` prefix, return `None` when the component is the built-in instance (compare with `==`), and `type(component).__name__` otherwise.
     - Add a rule to `"local-vol vanilla"`, `"local-vol barrier"` and `"DCN"`: `Rule("component.stepper_class", REJECTED, "no core seam until sub-project 3")`. `_explicit` computes it as `None if type(params.stepper) is CrankNicolson else type(params.stepper).__name__`.
     - Add the same kind of rule for `component.operator` and `component.readout`.
  8. **Readers.**
     - DCN (`dcn_pde_solver.py:270-298`): `st = resolve_stepper(self.params, DAMPING_DEFAULTS["DCN"])`. Then `theta, event_theta = st.theta, st.event_theta`, `rann = int(st.terminal_damping_steps)` and, after events, `rann = int(st.event_damping_steps)`.
     - The 2D solvers:
       - `event_damped_step_keys(stepper, event_maps, n_t)` reads `stepper.event_damping_steps`.
       - The two `core.solve` calls pass `theta=float(st.theta)`, `rannacher=st.terminal_damping_steps > 0` and `damped_step_theta=float(st.event_theta)`, with `st = resolve_stepper(self.params, DAMPING_DEFAULTS["core"])`.
       - The docstring at 38 says `stepper.event_theta`.
     - Local-vol vanilla: `theta=float(st.theta)`, `rannacher=st.terminal_damping_steps > 0`. Local-vol barrier: `theta=float(st.theta)`.
     - Intraday `_method` reads `st.terminal_damping_steps > 0` and `st.theta`.
     - `pde_engine.py:75` shows `PDEParams(stepper=CrankNicolson(theta=1.0))`.
     - The validation scripts get the same two rewrites.
  9. **`test/execution/matrix_fixtures.py:_pdep`.** It becomes `PDEParams(stepper=CrankNicolson(event_damping_steps=1), event_projection="nodal", …)`, with its other fields unchanged. `pde_phase4_goldens.json` is **not** re-frozen: the numbers must match.
  10. **The historical `grid` record.** Without this item the identity gate fails, because `resolve_config(...).evidence()` has lost the two damping keys. Add the following to `grid/config.py`:

```python
def resolved_grid_evidence(params) -> dict:
    """The historical ten-key ``grid`` record certificates hash (spec 2 §4.6): the nine
    resolved GridConfig fields plus the built-in stepper's two resolved damping counts,
    under their 0.6.0 keys. It is the numerical fact, not a serialization of GridConfig."""
    from quantark.asset.equity.engine.pde.core.components import CrankNicolson, resolve_stepper
    record = dict(resolve_config(params.accuracy, params.grid).evidence())
    st = resolve_stepper(params, DAMPING_DEFAULTS["core"])
    if type(st) is not CrankNicolson:
        raise ValidationError("resolved_grid_evidence: only the built-in stepper is certifiable")
    record["terminal_damping_steps"] = int(st.terminal_damping_steps)
    record["event_damping_steps"] = int(st.event_damping_steps)
    return record
```

      In each of the five PDE candidate builders, replace `"grid": engine_config(resolve_config(accuracy, None))` with `"grid": resolved_grid_evidence(PDEParams(accuracy=accuracy))`.
      - First confirm that `engine_config` passes a plain `dict` of plain values through unchanged. Check `engine_config.py:27-68`. If it transforms dicts, apply the same transform here.
      - `canonical_json` sorts keys, so key order is irrelevant. Confirm where the certificate hash is taken.

- [ ] **Step 4: Run the grep gate** (§13 risk row):
  ```bash
  grep -rn "terminal_damping_steps\|event_damping_steps\|params\.theta\b\|params\.event_theta\|\.event_theta\b" quantark | grep -v "core/components.py\|pde_removed_names.py\|grid/config.py:.*DAMPING_DEFAULTS"
  ```
  Expected: only `stepper.`-qualified reads. Every hit must go through a resolved stepper (`st.` / `stepper.`). Record the hit list in the task log. `model_params.theta` / `HestonParams.theta` is the Heston mean variance, which is unrelated and allowed.
- [ ] **Step 5: Rewrite every test that passes the old names.** Rewrite each call with the same values in the new home. For example, `PDEParams(theta=1.0)` becomes `PDEParams(stepper=CrankNicolson(theta=1.0))`, and `GridConfig(..., event_damping_steps=1)` becomes `PDEParams(grid=GridConfig(...), stepper=CrankNicolson(event_damping_steps=1))`. The files:
  - `test/test_pde_params_surface.py`, `test/test_pde_engine.py`;
  - `test/test_local_vol_equity_engines.py`, `test/test_barrier_vol_pde_engines.py`;
  - `test/test_dcn_pde_solver.py`, `test/test_heston_pde_engines.py`;
  - `test/test_pde_time_grid_decoupling.py`, `test/test_pde_grid_convergence_gate.py`;
  - `test/test_pde_event_projection.py`, `test/test_pde_quad_audit_regressions.py`;
  - `test/pde_grid/test_solver_seam.py`, `test_grid_config.py`, `test_grid_precedence.py`;
  - `test/execution/matrix_fixtures.py`.

  Then confirm the sweep is empty:
  ```bash
  grep -rln "theta=\|terminal_damping_steps\|event_damping_steps" test | xargs grep -ln "PDEParams\|GridConfig"
  ```
  After the rewrite this must list no file that still passes the old names. Check every listed file by eye, and remember that `HestonParams(theta=…)` is legitimate.
- [ ] **Step 6: Run the tests and the gates.** Run the full suite: `… -m pytest -q`. Then run the byte gate: the `scheme|*` rows now construct through `p_stepper`'s `stepper` branch, and they must be 0 diffs against base's 0.6.0 construction. Then run the identity gate, which must **PASS**. Step 3 item 10 keeps the ten-key `grid` record, so no banked hash moves.
- [ ] **Step 7: Commit.** Message: `feat(pde-params)!: the built-in scheme's four settings live on PDEParams(stepper=CrankNicolson(...)); operator/readout component slots; GridConfig has nine fields (spec 2 §4.1-§4.5, decisions 1, 4, 5)`.

---

### Task 13: The evidence record, certification wiring, and component conformance

**Files:**
- Modify: `grid/config.py` (`resolved_grid_evidence`); `engine_params.py` (`PDEParams.evidence`); `quantark/modelvalidation/engine_config.py` (serializing `PDEParams` through `evidence()` already works via `_fields_of`; verify).
- Modify: `quantark/modelvalidation/builders/equity_snowball.py:453-490`, `equity_phoenix.py:332-343`, `equity_ko_reset.py:377-387`, `equity_snowball_localvol.py:288-306`, and `intraday_snowball.py:310-333`.
- Create: `test/modelvalidation/test_candidate_wiring.py`; `test/pde_core/components_fixtures.py`; `test/pde_core/test_components.py`.
- Extend: `test/pde_core/test_identity.py` (the operator and readout identities).

**Interfaces:**
- Produces:
  - `resolved_grid_evidence(params) -> dict`: the historical ten-key `grid` record.
  - `PDEParams.evidence() -> dict`.
  - The candidate builders construct `PDEParams.from_config(declared)` in `__init__`.

- [ ] **Step 1: Write the failing tests.**

`test/modelvalidation/test_candidate_wiring.py`:

```python
def _banked_pde_grid(cert_path):
    d = json.loads(Path(cert_path).read_text())
    for cand in d["study"]["candidates"]:
        if str(cand["params"].get("engine", "")).endswith("PDESolver"):
            return cand["params"]["grid"]
    raise AssertionError(f"no PDE candidate in {cert_path}")


@pytest.mark.parametrize("study", ["snowball-flat-bsm", "phoenix-flat-bsm", "ko-reset-flat-bsm"])
def test_resolved_grid_evidence_is_the_banked_record(study):
    certs = sorted((REPO_ROOT / "docs" / "modelvalidation" / "certificates" / study).glob("*/certificate.json"))
    banked = _banked_pde_grid(certs[-1])
    # JSON round trip: bounds is a tuple in memory and a list in the certificate
    assert json.loads(json.dumps(resolved_grid_evidence(PDEParams(accuracy="standard")))) == banked


def _arm(params):
    return SnowballPDECandidate(environment_params={}, product_params={}, quantities=("pv",), params=params)


def test_a_declared_stepper_is_executed_and_recorded():
    cand = _arm({"accuracy": "standard", "stepper": {"theta": 1.0}})
    assert cand._engine("standard").params.stepper == CrankNicolson(theta=1.0)
    assert cand.params()["stepper"] == {"theta": 1.0}


def test_a_key_from_config_rejects_fails_the_study_at_load():
    with pytest.raises(ValidationError):
        _arm({"accuracy": "standard", "operator": {}})
```

- `REPO_ROOT` is the one `test_variant_case_builders.py` uses; import it from there.
- The banked `snowball-flat-bsm/2026-09-28` record is the ten keys `bounds, day_count, eps_crit, event_damping_steps, max_points, max_steps, num_std, points, steps_per_day, terminal_damping_steps`. That was verified while writing this plan.
- `_engine(accuracy) -> solver` is a new method on every PDE candidate (Step 3). `_greeks` calls it instead of constructing the solver inline.

`test/pde_core/components_fixtures.py`:

```python
@dataclass(frozen=True)
class BackwardEuler:
    """Fully implicit: θ=1 everywhere, through the built-in arithmetic."""
    def schedule(self, layout):
        return np.ones(layout.time.actual_steps)
    def step(self, state, s, dt, theta, boundaries):
        return CrankNicolson().step(state, s, dt, 1.0, boundaries)
    def evidence(self):
        return {"kind": "backward_euler"}


@dataclass(frozen=True)
class ShiftedOperator(CentralDifference):
    shift: float = 0.0
    def assemble(self, samples, layout):
        sc = super().assemble(samples, layout)
        return sc._replace(lcu_sets=[(l, c + self.shift, u) for (l, c, u) in sc.lcu_sets])


@dataclass(frozen=True)
class NearestNodeReadout:
    def price(self, column, spot, layout):
        return float(column[int(np.argmin(np.abs(layout.spatial.s - spot)))])
    def delta_gamma(self, column, spot, layout):
        return StandardReadout().delta_gamma(column, spot, layout)
    def evidence(self):
        return {}


@dataclass(frozen=True)
class NaNStepper(BackwardEuler):
    def step(self, state, s, dt, theta, boundaries):
        out = super().step(state, s, dt, theta, boundaries); out[0] = np.nan
        return out


@dataclass(frozen=True)
class WrongShapeStepper(BackwardEuler):
    def step(self, state, s, dt, theta, boundaries):
        return super().step(state, s, dt, theta, boundaries)[:-1]


class PlainStepper:          # not a dataclass: no value fingerprint
    schedule = BackwardEuler.schedule; step = BackwardEuler.step; evidence = BackwardEuler.evidence


ShiftedOperatorStub = ShiftedOperator
```

`test/pde_core/test_components.py`:
- `BackwardEuler`, `ShiftedOperator(shift=0.0)` and `NearestNodeReadout` each price a European.
- `CrankNicolson(theta=1.0, terminal_damping_steps=0, event_damping_steps=0)` and `BackwardEuler()` agree via `is_close(..., rel_tol=Tolerance.RELATIVE)`.
- `NaNStepper` / `WrongShapeStepper` raise `NumericalError` whose message contains the class name.
- `PlainStepper()`:
  - prices;
  - `prepare(...).identity["problem"] is None`;
  - a `PricingSession` builds no PDE artefact for it (assert through the session's artefact cache, as in `test_pde_artifact_reuse.py`);
  - `SnowballPDECandidate` refuses it. A study candidate cannot be built from a `PDEParams` object at all, and `from_config` cannot name a custom component, so assert that the builder's constructor rejects a non-mapping params argument with `ValidationError`.
- **Read-only views** (the ownership rule of §5.8): a stepper that writes into `state.v_next` raises `ValueError: assignment destination is read-only`. It raises from the stepper, and a fresh price afterwards is unaffected.

Extend `test/pde_core/test_identity.py`:
- `ShiftedOperator(0.0)` and `ShiftedOperator(1e-9)` both return `evidence() == {}`, but they have different fingerprints and different coefficient keys.
- `StandardReadout()` against `NearestNodeReadout()` gives different problem identities and the same factorization key.

- [ ] **Step 2: Run the tests to verify they fail.**
- [ ] **Step 3: Wire the candidates.** `resolved_grid_evidence` already exists (Task 12, Step 3, item 10).
- Each PDE candidate builder:
  - `__init__` sets `self._pde_params = PDEParams.from_config(self._params)`, so a bad key fails at load. Because `_SnowballArm.__init__` is shared with the MC and QUAD arms, override `__init__` in the PDE candidate classes only.
  - New method `_engine(accuracy)`: `SnowballPDESolver(params=dataclasses.replace(self._pde_params, accuracy=accuracy))`. Use the matching solver class in each builder. `_greeks` calls it.
  - `params()` records `{**self._params, "engine": …, "grid": resolved_grid_evidence(dataclasses.replace(self._pde_params, accuracy=accuracy))}`.
  - The intraday builder's `_explicit_engine` becomes `PDEParams.from_config(self._params, grid=GridConfig(points=…, steps_per_day=…, max_points=…, max_steps=10**8))`.
- `PDEParams.evidence()`: `accuracy`; `grid` (`self.grid.evidence()` when set); `stepper` (`CrankNicolson.evidence()` when built in, else `component_fingerprint(...) or "uncertifiable"`); `operator` / `readout` only when not the built-in; `event_projection` (value or `None`); `cache_enabled`; and the inherited `EngineParams` fields exactly as `engine_config` serialises them today.
- [ ] **Step 4: Run the tests, then the identity gate.** Run `… -m pytest test/modelvalidation test/pde_core -q`. `test_banked_cells_keep_their_identity` must pass with the historical `c5b6a27b…`, `6a9aae02…` and `f2c9bd52…` prefixes, and every `assert_anchors` test must pass. Then run the byte gate.
- [ ] **Step 5: Commit.** Message: `feat(modelvalidation): candidates execute the params they record (from_config); the grid record composes the stepper's damping counts under their historical keys (spec 2 §4.6, decisions 1, 15)`.

---

### Task 14: Docs, release 0.7.0, and the timing gate

**Files:**
- Modify:
  - `docs/engine_param_guide.md`;
  - `quantark/asset/equity/CLAUDE.md` ("Engine Parameters"). Edit this local copy, but **never commit it**; CLAUDE.md files are untracked;
  - `README.md` (the PDE snippet);
  - `docs/execution/capability-matrix.md`;
  - `docs/modelvalidation/RELEASE_PROCEDURE.md`;
  - `CHANGELOG.md`;
  - `pyproject.toml`, `quantark/__init__.py` (`0.7.0`);
  - `quantark/asset/equity/engine/pde/grid/binder.py:120-125` (the `GridLayerMixin` docstring).
- Create: `docs/pde-core.md`; `test/pde_bitwise/timing.py`.

- [ ] **Step 1: Docs.**
  - **`engine_param_guide.md`:**
    - the component slots;
    - the `CrankNicolson` table with §4.1's "when to change it" lines;
    - an `explain` example. Paste real `str(plan)` output from a standard snowball in the flat env; generate it, do not hand-write it;
    - the removal additions, rendered from `REMOVED_PDE_NAMES` / `REMOVED_GRID_NAMES`.
  - **`docs/pde-core.md`:**
    - the three stages and `PreparedProblem`;
    - column groups and policies (§5.4/§5.5 tables);
    - adapters;
    - the protocols, with the `BackwardEuler` fixture as the worked custom component;
    - the fingerprint rule ("covers configuration and class, not code; only built-ins are certifiable");
    - the evidence record;
    - the characterized oddities of §8 plus this plan's additions (self-review notes).
  - **`capability-matrix.md`:** PDE session adapters now go through `prepare(session=…)`.
  - **`RELEASE_PROCEDURE.md`:** one sentence stating that the `grid` record composes the nine grid fields with the stepper's two counts.
  - **`GridLayerMixin` docstring:** "used by DCN only".
- [ ] **Step 2: `CHANGELOG.md`, section `## [0.7.0]`.**
  - **Added:** `stepper` / `operator` / `readout`, `explain`, `resolved_grid_evidence`, `OutputRequest`, `Calendar.identity()`.
  - **Removed:** the §4.4 table.
  - **Changed:**
    - the exact θ in the factorization key;
    - statistics computed in the price march;
    - `calculate_event_stats` no longer re-prices;
    - sessions store the layout (`pde-prep/v2`);
    - market memos are keyed by value, so mutating a market object after a solve is no longer silently stale;
    - certificate candidates are built from their declared params.
  - **Fixed:**
    - event statistics on a reused solver no longer reuse another market's factorizations (Review Focus 3, with the measured 0.041 KO-probability error);
    - American Greeks on a fresh solver for exercise with a settlement lag no longer raise;
    - solvers deep-copy and pickle after pricing.
  - **Sub-project 3:** the §12 list.
- [ ] **Step 3: Set the version.** `pyproject.toml` and `quantark/__init__.py` both become `0.7.0`.
- [ ] **Step 4: Timing gate (§9.5).** Write `test/pde_bitwise/timing.py`:
  - It takes `--out` and `--reps 5`.
  - Per family (European, American, Barrier, Snowball, Phoenix, KO-reset, LV snowball, statistics), with the book's `standard` flat product, it measures:
    - **cold:** a new solver and `price`, after clearing `base_pde_solver._ENV_STEP_COEFF_MEMO`;
    - **warm:** the same solver, a second `price`.
  - It takes five repetitions after one warm-up each, and writes the medians.
  - Peak memory per family is measured in a subprocess with `resource.getrusage(RUSAGE_SELF).ru_maxrss`, one family per process.

  Run base (the detached worktree) and branch **back to back in one window**, wrapped in `caffeinate -i`. Expected: no family's median is more than 10% slower, and no family's peak memory is more than 10% higher.
  - If two consecutive runs of the same arm disagree by more than 10%, the measurement is inconclusive. Re-run on a quiet machine; do not call it a pass.
  - If a family regresses, profile it (`python -m cProfile -o …`). The likely suspects are the per-step `np.column_stack` and `validate_advanced`.
  - Fix the regression without changing arithmetic, re-run the byte gate, then re-time.
- [ ] **Step 5: Commit.** Commit the docs by explicit pathspec, with `git add -f` for `docs/`. Message: `docs(release): 0.7.0 - PDE 1D one core`.

---

### Task 15: Final gate

- [ ] **Step 1: Byte gate, back to back, same machine.** Re-capture base (the Task 1 detached worktree) and the branch in one window. Then run:
  ```bash
  … compare.py $SCR/base.json $SCR/branch.json --manifest test/pde_bitwise/manifest.json | tee $SCR/byte_gate.txt
  ```
  Expected: `0 unexpected diffs, 0 allowed diffs, 0 manifest violations`.
- [ ] **Step 2: Anchors and identities.**
  ```bash
  … -m pytest test/modelvalidation/test_anchors.py test/modelvalidation/test_banked_certificates.py test/modelvalidation/test_variant_case_builders.py test/modelvalidation/test_candidate_wiring.py -q
  ```
  Expected: PASS, with the three banked prefixes unchanged.
- [ ] **Step 3: Full suite on two Pythons.** Run the main venv (3.12) and a 3.10 interpreter if one exists (`ls ~/.pyenv/versions`, `uv python list`). CI is x86_64 on 3.10–3.13, so state which versions actually ran. A green run on three versions proves nothing about the fourth.
- [ ] **Step 4: Grep sweep.**
  ```bash
  grep -rn "_time_stepping\|_time_stepping_two_surface\|_time_stepping_vector_surface\|_step_grids\|_compute_event_stats\|_get_banded_system\|_get_matrices\|_bump_transient_attrs\|_session_factorization_packs\|_theta_schedule_from_layout\|_active_layout\|pde-prep/v1\|round(theta" quantark test | grep -v "_legacy_bump.py\|quad_engine.py"
  ```
  Expected: empty. The 2D modules' `_bump_*` hits are through `_legacy_bump.py` only.
- [ ] **Step 5: Tear down.** Run `git worktree remove $SCR/base`. Hand off to `superpowers:finishing-a-development-branch`. The merge commit message carries `byte_gate.txt`, the timing table, and the Review Focus 3 measurement. `main` is ff-only; if `main` moved, rebase and re-run Steps 1–2.

---

## Self-review notes (for the reviewer)

**Where the plan decides beyond, or against, the spec.** Each item is flagged in its task.

1. **Adapters delegate to the kept product helpers, and per-solve state stays on the solver** (Architecture; Tasks 5–9). The 2D Heston/SLV solvers inherit `_set_boundary_conditions_v0/_v1`, the KO/KI record caches, the event jumps and the terminal payoffs. Moving that state onto `PreparedProblem` would touch the 2D solvers, which the spec keeps out of scope (§7). The march itself holds no product knowledge. Every loop and every duplicated preparation is deleted.
2. **Retention is `full` for value groups** (Task 5), not "rolling unless needed" (§5.3). Every family helper indexes surfaces by step, so rolling value columns would mean rewriting the helpers. `explain` reports retention, so sub-project 3 can change it.
3. **The branch capture happens before the step's events** (Task 6). §5.3 lists it after them. The legacy code (2657-2661) captures pre-event branches, and `LifeSurfaceSolution.v1` is that. The book row proves it.
4. **The spec's migration order changes in three places, to keep `main` green:**
   - KO-reset moves right after Snowball, and `_time_stepping_two_surface` is deleted with KO-reset (Task 7), not in §10 step 2.
   - `_get_matrices` / `_get_banded_system` stay as delegates until their last caller moves (Tasks 8–9).
   - `explain` (Task 11) lands before the scheme relocation (Task 12), because the intraday route (§10 step 6) needs it.
5. **`TimeStepper.step` returns the advanced columns** (Task 4). §5.8's code block says `-> None`, but §4.1 and the validation rules say it returns them.
6. **`OutputRequest` carries an `operation`** (Task 5). The per-operation terminal table (decision 16) needs it, and the spec's field list omits it.
7. **`PDEParams.stepper` / `operator` / `readout` default to `None`** and are normalised in `__post_init__` (Task 12). This avoids an import cycle. The observable default is the spec's.
8. **`_bump_clone` has a production caller the spec missed** (`snowball_vol_pde_solvers.py:654`, the 2D Greeks). It moves, verbatim, to `_legacy_bump.LegacyBumpCloneMixin` for the 2D solvers only (Task 10). A 2D Greek row is added to the book.
9. **The march still reads discount factors through the value-keyed memo** (Task 5). Decision 14's "the march consumes sampled arrays" holds for the coefficients. Discount factors are fetched per step by the boundary helpers, now keyed by a fingerprint taken once in `prepare`. The mutation hazard the decision targets is closed, and the helpers are not rewritten.
10. **Two defect fixes land as structural side effects, bitwise on the book:**
    - the stale statistics factorization (Review Focus 3, measured 0.0407; Task 9);
    - American Greeks on a fresh solver raising `PricingError` for exercise with a settlement lag (Review Focus 2; Task 5).

    Both go in the changelog under Fixed. Neither moves a book row, because every row uses fresh engines.
11. **The 2D Heston/SLV solvers raise from `explain`** (Task 11). They inherit it through the 1D classes, but a plan of the 1D statistics problem would misdescribe a 2D price.
12. **The session factorization warm-up now applies the zero-diffusion θ=1 override** (Task 10). The legacy pack builder did not, and missed on those steps and built lazily. Both paths build at the first march-order occurrence from the same inputs, so the result is bitwise.

**Characterized oddities found while planning and not listed in §8.** They are preserved bit for bit and should be added to `docs/pde-core.md`:

- A **single-step local-vol layout** marches on the scalar Black-Scholes stencil, because `n_unique == 1` selects `flat` (`CoefficientsInForce`, Task 4). It is unreachable at every preset.
- **American Greeks** skip the valuation-date exercise max that `price` applies, and the expired-Greeks short-circuit omits the delay factor that `price` applies (Task 5).
- The **double barrier's knocked-out Greeks** return the undiscounted rebate, while `price` discounts it (Task 5).
- **KO-reset** never composes its t=0 pre-event readout. It interpolates the post-event column 0 for price and Greeks (Task 7).
- **Interior KO cash** is discounted from `t_obs` on the snowball's schedule path, but from `total_tau − tau_remaining` on the inline paths. Interior projections use `layout.x` on the schedule path and `np.log(s_vec)` inline (Task 6, verbatim moves).
- **`cache_enabled=False`** differs from `True` at round-off (Review Focus 5). The docstring is corrected in Task 12.

**Where the spec's claims are proven, not assumed:**

| Claim | Proof |
|---|---|
| "Nearest-match lands on the same node" (§8) | `stats\|*` rows (Task 6 Step 4 names the stop condition) |
| "Phoenix and KO-reset honouring injection is identical" | `session_{phoenix,ko_reset}` rows plus the spy test (Task 10) |
| "The goldens need no re-freeze" | verified by grep: no artefact kinds and no `pde-prep/v1` in `pde_phase4_goldens.json`; only `_pdep`'s constructor changes (Task 12) |
| "Every banked candidate declares `{accuracy: standard}`" | `test_candidate_wiring.py` (Task 13) |
