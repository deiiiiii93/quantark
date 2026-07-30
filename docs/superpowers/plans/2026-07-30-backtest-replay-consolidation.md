# Backtest Replay Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `quantark/backtest/otc/` with a first-class `quantark/backtest/replay/` package with ONE daily loop, fail-closed semantics, shared futures ledger/metrics, relocated calibration infrastructure, and pending-settlement KO termination — all gated on exact-equality goldens.

**Architecture:** Behavior-frozen structural moves land first (shims at every old path), then the single engine collapses into the book engine (book-of-one, byte-identical), then behavior changes (fail-closed greeks, opt-in event-stats fallback, KO termination) land last as isolated diffs. Spec: `docs/superpowers/specs/2026-07-30-backtest-replay-consolidation-design.md`.

**Tech Stack:** Python 3.11, pandas, numpy, pytest (`-n auto` default; use `-n0` for debugging), git worktree via native EnterWorktree.

## Global Constraints

- Canonical imports only: `quantark.*` (CLAUDE.md).
- Numerical comparisons via `quantark.util.numerical` (`is_zero`, `is_close`) — never raw float compares (CLAUDE.md).
- No MC imports inside PDE code paths; deterministic engines stay deterministic.
- Fail-closed: no silent fallbacks; approximations/fallbacks are opt-in config.
- Every old `quantark.backtest.otc.*` import path keeps working via a `DeprecationWarning` shim until 0.5.0. Public class names keep working.
- Golden gate: `assert_frame_equal(check_exact=True)` + column-name/order equality on every deterministic frame; wall-clock fields (`pricing_seconds`, `calibration_seconds`) excluded by name.
- Worktree testing: the editable install resolves `quantark` to the main repo — always run `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` from inside the worktree so worktree source shadows it.
- Commit after every task. Conventional commits. Do not touch files outside each task's list (the shared tree has other sessions' WIP).

---

### Task 0: Commit the study framework scope, create the worktree

The consolidation must build on the uncommitted study framework (study spec §10 step 0). Commit exactly that scope on `main`, nothing else.

**Files:** (commit-only task; scope determined from `git status` at execution time)
- All tracked-modified and untracked files under `quantark/backtest/otc/`
- Untracked stage scripts `example/mo_volmodels/*.py` and untracked `example/mo_volmodels/data` additions belonging to stages 01/03/08–13
- Modified/untracked `test/mo_volmodels/*.py`, `test/test_otc_*.py`, `test/test_book_backtest.py`, `test/test_equity_lifecycle_trackers.py`, `test/test_ppp_dki_snowball_case_study.py`
- Modified `example/ppp_dki_snowball_backtest_case_study.py` (consumes `backtest.otc`)

- [ ] **Step 1: Enumerate and sanity-check the scope**

Run: `git status --porcelain | grep -E "backtest/otc|mo_volmodels|test_otc|test_book_backtest|test_equity_lifecycle|ppp_dki"`
The scope is a **coherent set**, not a stage subset: ALL modified/untracked
`example/mo_volmodels/*.py` (including the stage-04 script and
`_heston_diagnostics.py` — the modified `test_stage04_heston.py` depends on
both; committing the test without its script produces a knowingly broken
baseline), ALL `test/mo_volmodels/*.py`, all `quantark/backtest/otc/` files,
the `test_otc_*`/`test_book_backtest`/`test_equity_lifecycle_trackers`/
`ppp_dki` files, plus any `example/mo_volmodels/data/` file that a committed
test actually reads (check with `grep -rn "data/" test/mo_volmodels | grep -o "data/[^\"']*" | sort -u`).
EXCLUDE anything under `quantark/asset/` (13 modified option-product files
are another session's WIP), `quantark/volmodels/`, `example/fx_volmodels/`,
`.codegraph/`, `example/bucketed_greeks*`, `example/data/*`,
`example/snowball_volmodel_backtest/` (run output).

- [ ] **Step 2: Verify the scoped tree passes its own tests — including the staged study tests**

Run against the exact staged tree (stash-everything-else check):

```bash
git stash push --keep-index -u -m "task0-scope-check" || true   # only if unstaged noise interferes
.venv/bin/python -m pytest test/test_otc_autocallable_backtest.py test/test_book_backtest.py \
  test/test_otc_vol_calibrators.py test/test_otc_vol_history_env.py \
  test/test_backtest_interface.py test/mo_volmodels -q
git stash pop || true
```
Expected: all pass (this is the baseline being frozen). A failure here means
the scope is incoherent — fix the scope, not the tests.

- [ ] **Step 3: Commit on main**

```bash
git add <the enumerated files only>
git commit -m "feat(backtest): commit snowball-study framework scope (vol calibrators, surface history, stages 11-13, tests)"
```

- [ ] **Step 4: Create the isolated worktree**

Use the native `EnterWorktree` tool (branch name `backtest-replay-consolidation`). EnterWorktree bases on `origin/main`, not local HEAD — inside the worktree run:

```bash
git merge --ff-only main
```

Verify `git log --oneline -3` shows the study-scope commit and the two spec commits.

---

### Task 1: Golden capture — synthetic fixtures, three configurations

Freeze current behavior as committed fixtures BEFORE any refactor. Three goldens: scalar-BSM single, book-of-two, **calibrated `localvol`/PDE single** (the bsm goldens never touch `VolModelCalibrator` — spec §9.1).

**Files:**
- Create: `test/replay_golden/__init__.py` (empty)
- Create: `test/replay_golden/fixtures.py`
- Create: `test/replay_golden/capture.py`
- Create: `test/test_replay_goldens.py`
- Create (generated, committed): `test/replay_golden/data/*.csv`, `test/replay_golden/data/*_summary.json`

**Interfaces:**
- Produces: `fixtures.make_scalar_bsm_config() -> AutocallableBacktestConfig`, `fixtures.make_book_config() -> BookAutocallableBacktestConfig`, `fixtures.make_localvol_config(tmp_dir: Path) -> AutocallableBacktestConfig`, `fixtures.result_frames(results) -> dict[str, pd.DataFrame]`, `fixtures.result_summary(results) -> dict`, `capture.write_goldens(golden_dir: Path) -> None`
- `result_frames` keys: `states, greeks, rebalances, trades, actions, surfaces, daily_event_summary, event_probabilities` (+ `calibration_records` for the localvol case, timing fields dropped).

- [ ] **Step 1: Write `fixtures.py`**

Reuse the synthetic builders already proven in tests:
- Scalar BSM: copy the product/market construction from `test/test_otc_autocallable_backtest.py::_snowball_product/_market_data` (~60 trading days, deterministic arrays, `EngineType.PDE` defaults, `ZeroCostModel`).
- Book: two products (the snowball + the phoenix from `_phoenix_product`), `HedgeSpec(kind="futures")`, same market data.
- Localvol: build a synthetic `VolSurfaceHistory` on disk exactly as `test/test_otc_vol_history_env.py::_artifact_payload/_write_history` does (reuse by import if the helpers are importable; otherwise copy them into `fixtures.py`), then:

```python
def make_localvol_config(tmp_dir):
    history = VolSurfaceHistory(_write_history(tmp_dir, dates, payloads))
    market = _market_data(surface_history=history)
    engine_config = AutocallableEngineConfig(
        pricing_engine_type=EngineType.PDE,
        vol_source="surface",
        vol_model="localvol",
        vol_model_solver="pde",
        vol_model_calibration=VolModelCalibrationConfig(cache_dir=None),
    )
    return AutocallableBacktestConfig(
        product=_snowball_product(), market_data=market,
        engine_config=engine_config, transaction_cost_model=ZeroCostModel(),
        calculate_surfaces=False, calculate_event_probabilities=True,
    )
```

`result_frames` normalization: sort no rows (order is part of the contract), drop `pricing_seconds`/`calibration_seconds` columns/keys wherever present, `reset_index()` so the `date` column round-trips CSV exactly via `df.to_csv(path, index=False)` / `pd.read_csv(path, parse_dates=["date"], ...)` — **write floats with `float_format=None`** (pandas full repr) and compare after `read_csv` with `check_exact=True`.

- [ ] **Step 2: Write `capture.py`**

```python
def write_goldens(golden_dir: Path) -> None:
    for name, results in _run_all():          # runs the three configs via CURRENT engines
        for frame_name, df in result_frames(results).items():
            df.to_csv(golden_dir / f"{name}_{frame_name}.csv", index=False)
        (golden_dir / f"{name}_summary.json").write_text(
            json.dumps(_json_safe(result_summary(results)), indent=2, sort_keys=True, default=str))
```

For the localvol case also assert inside `_run_all`: `len(results.calibration_records) == number of priced days`, every record's `variant == "localvol"`, and `surface_date` is non-decreasing (calibrate-before-pricing ordering evidence).

- [ ] **Step 3: Write `test/test_replay_goldens.py`**

One parametrized test per (config, frame): re-run engine, load golden CSV, compare with `pd.testing.assert_frame_equal(actual, expected, check_exact=True)` **after** asserting `list(actual.columns) == list(expected.columns)`. Summary compared as dicts (timing keys stripped). CSV round-trip note: compare `read_csv(actual_written)` vs `read_csv(golden)` — i.e., write the actual frames to a tmp CSV first so float formatting is symmetric.

- [ ] **Step 4: Capture and verify**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "from pathlib import Path; from test.replay_golden.capture import write_goldens; write_goldens(Path('test/replay_golden/data'))"
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_replay_goldens.py -q
```
Expected: all golden tests PASS (trivially, same code).

- [ ] **Step 5: Real-data anchor (manual gate, not committed)**

If `example/mo_volmodels/12_snowball_volmodel_backtest.py --quick --max-inceptions 1` runs on this machine (it needs the study data files), run it and copy its output directory to `output/volmodel_backtest_golden_pre_consolidation/`. If data is unavailable, record that in the commit message and rely on the synthetic goldens.

- [ ] **Step 6: Commit**

```bash
git add test/replay_golden test/test_replay_goldens.py
git commit -m "test(backtest): freeze pre-consolidation goldens (bsm single, book, calibrated localvol)"
```

---

### Task 2: `quantark/util/io.py` — atomic JSON writer

**Files:**
- Create: `quantark/util/io.py`
- Modify: `quantark/backtest/otc/vol_calibrators.py` (delegate `_atomic_write_json`)
- Modify: `quantark/backtest/otc/results.py:12` (import from new home)
- Test: `test/test_util_io.py`

**Interfaces:**
- Produces: `quantark.util.io.atomic_write_json(path: Path, payload: Any) -> None` (public name; behavior identical to current `_atomic_write_json`: tmp file + `os.replace`, parent dir must exist).

- [ ] **Step 1: Write failing test** — `test_util_io.py`: writes a dict, asserts file exists with exact JSON; asserts no `*.tmp` residue; asserts concurrent-safe replace (write twice, last wins).
- [ ] **Step 2: Run** `pytest test/test_util_io.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** — move the body of `vol_calibrators._atomic_write_json` verbatim into `quantark/util/io.py::atomic_write_json`; keep `_atomic_write_json = atomic_write_json` alias in `vol_calibrators.py`; change `results.py` to `from quantark.util.io import atomic_write_json`.
- [ ] **Step 4: Run** `pytest test/test_util_io.py test/test_otc_vol_calibrators.py test/test_replay_goldens.py -q` → PASS.
- [ ] **Step 5: Commit** `git commit -m "refactor(util): extract atomic_write_json to quantark.util.io"`

---

### Task 3: Relocate `vol_history.py` → `quantark/param/vol/surface_history.py`

**Files:**
- Create: `quantark/param/vol/surface_history.py` (via `git mv quantark/backtest/otc/vol_history.py …`)
- Create (shim): `quantark/backtest/otc/vol_history.py`
- Modify: importers found by `grep -rln "otc.vol_history\|from .vol_history" quantark/ test/ example/`
- Test: existing `test/test_otc_vol_history_env.py` (unchanged, exercises the shim) + new import assertions in Task 11's compat test

- [ ] **Step 1: `git mv`, fix the moved module's own imports** (it imports nothing from `otc`, verify with `grep "^from \.\|^from quantark.backtest" quantark/param/vol/surface_history.py`).
- [ ] **Step 2: Write the shim** (this exact pattern is reused by every later shim):

```python
"""Deprecated shim — moved to quantark.param.vol.surface_history (0.5.0 removes this)."""
import warnings
from quantark.param.vol.surface_history import *          # noqa: F401,F403
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory  # noqa: F401

warnings.warn(
    "quantark.backtest.otc.vol_history moved to quantark.param.vol.surface_history; "
    "this alias is removed in 0.5.0",
    DeprecationWarning, stacklevel=2)
```

- [ ] **Step 3: Update in-repo importers to the canonical path** (`quantark/backtest/otc/market.py`, `vol_calibrators.py`, stages, `test/mo_volmodels/*`), EXCEPT `test/test_otc_vol_history_env.py` which stays on the old path (it becomes shim coverage) — add `filterwarnings` handling if the suite errors on DeprecationWarning.
- [ ] **Step 4: Run** `pytest test/test_otc_vol_history_env.py test/test_replay_goldens.py test/test_otc_vol_calibrators.py -q` → PASS.
- [ ] **Step 5: Commit** `git commit -m "refactor(param): relocate IV-surface history to quantark.param.vol.surface_history"`

---

### Task 4: Relocate `vol_calibrators.py` → `quantark/volmodels/calibration.py`

**Files:**
- Create: `quantark/volmodels/calibration.py` (via `git mv`)
- Create (shim): `quantark/backtest/otc/vol_calibrators.py` (Task-3 shim pattern; re-export `CalibratedVolModel`, `VolModelCalibrator`, `HESTON_PRESETS`, `VOL_MODEL_*`, `_atomic_write_json`)
- Modify: importers (`otc/__init__.py`, `otc/config.py`, `otc/engine.py`, `otc/engine_factory.py`, `otc/results.py`, stages, `test/mo_volmodels/*`)
- Test: `test/test_otc_vol_calibrators.py` stays on old path (shim coverage); add cache-key invariance test there

- [ ] **Step 1: `git mv` + fix moved module's imports** (it imports `vol_history` → now `quantark.param.vol.surface_history`; volmodels kernels already canonical).
- [ ] **Step 2: Shim + update in-repo importers to canonical path.**
- [ ] **Step 3: Cache-key invariance test** (append to `test/test_otc_vol_calibrators.py`):

```python
def test_cache_key_survives_module_relocation(tmp_path):
    # key = sha256(f"{surface_sha}|{variant}|{fingerprint}") — no module paths.
    calib = VolModelCalibrator(VolModelCalibrationConfig(cache_dir=str(tmp_path)))
    # Build the artifact with the same synthetic surface-history helper Task 1
    # uses: test.replay_golden.fixtures builds a VolSurfaceHistory on disk; take
    # history.surface_for(<first admitted date>) as the artifact.
    artifact = _make_synthetic_artifact(tmp_path)   # thin wrapper over those fixtures
    first = calib.calibrate("localvol", artifact)
    from quantark.volmodels.calibration import VolModelCalibrator as NewV
    second = NewV(VolModelCalibrationConfig(cache_dir=str(tmp_path))).calibrate("localvol", artifact)
    assert second.record["cache_hit"] is True     # same on-disk key found
```

- [ ] **Step 4: Run** `pytest test/test_otc_vol_calibrators.py test/test_replay_goldens.py -q` → PASS.
- [ ] **Step 5: Commit** `git commit -m "refactor(volmodels): relocate vol-model calibration service to quantark.volmodels.calibration"`

---

### Task 5: Extract `quantark/backtest/futures_ledger.py`

**Files:**
- Create: `quantark/backtest/futures_ledger.py`
- Modify: `quantark/backtest/otc/state.py` (keep only `AutocallableDeltaHedgeStrategy` + re-exports), `quantark/backtest/otc/config.py` (import `FuturesRollPolicy` from ledger)
- Test: `test/test_futures_ledger.py`

**Interfaces:**
- Produces: `quantark.backtest.futures_ledger.FuturesHedgePosition` and `FuturesRollPolicy` — class bodies moved VERBATIM from `otc/state.py:60-121` and `otc/config.py:20-60`. Old import sites keep working (`otc.state.FuturesHedgePosition`, `otc.config.FuturesRollPolicy` re-export).

- [ ] **Step 1: Failing test** — `test_futures_ledger.py`: average-cost accounting (open 2 @100, add 1 @106 → avg 102), partial close realizes PnL, flip resets avg to trade price, cross-contract trade raises `ValidationError`, `mark_to_market` includes realized; `FuturesRollPolicy.select_contract` keeps current contract until `roll_days_before_expiry`, rolls to next otherwise (build a 2-contract `futures_slice` DataFrame inline).
- [ ] **Step 2:** FAIL (module missing). **Step 3:** Move classes verbatim; `otc/state.py` and `otc/config.py` import + re-export them (no DeprecationWarning here yet — these files themselves become shims in Task 6).
- [ ] **Step 4: Run** `pytest test/test_futures_ledger.py test/test_otc_autocallable_backtest.py test/test_book_backtest.py test/test_replay_goldens.py -q` → PASS.
- [ ] **Step 5: Commit** `git commit -m "refactor(backtest): extract shared futures ledger (position + roll policy)"`

---

### Task 6: Create `backtest/replay/` package; `otc/` becomes shims

Pure move + rename. NO logic changes. Canonical aliases: `ReplayBacktestEngine = BookAutocallableBacktestEngine`-successor, `ReplayBacktestConfig`, `ReplayProduct`, `ReplayBacktestResults`.

**Files:**
- Create: `quantark/backtest/replay/__init__.py`
- `git mv`: `otc/config.py→replay/config.py`, `otc/market.py→replay/market.py`, `otc/engine_factory.py→replay/engine_factory.py`, `otc/_replay.py→replay/product_replay.py`, `otc/results.py→replay/results.py`, `otc/dashboard.py→replay/dashboard.py`, `otc/book_engine.py→replay/engine.py`, `otc/engine.py→replay/single.py`, `otc/state.py→replay/strategy_state.py`
- Create shims: `otc/__init__.py`, `otc/config.py`, `otc/engine.py`, `otc/book_engine.py`, `otc/market.py`, `otc/engine_factory.py`, `otc/state.py`, `otc/results.py`, `otc/dashboard.py`, `otc/_replay.py` (Task-3 pattern; each re-exports its full old surface — copy the name lists from the pre-move files' imports/`__all__`)
- Modify: `quantark/backtest/base.py:287-292` (factory imports → `quantark.backtest.replay.config` / `.single`), `quantark/backtest/__init__.py` (re-export `Replay*` canonical names alongside the existing `Autocallable*` ones)
- Modify: in-repo consumers to canonical paths: `example/ppp_dki_snowball_backtest_case_study.py`, `example/otc_autocallable_backtest_demo.py`, `example/mo_volmodels/11_pde_convergence_gate.py`, `example/mo_volmodels/12_snowball_volmodel_backtest.py`, `test/mo_volmodels/test_stage12_backtest_runner.py`, `test/mo_volmodels/test_surface_admission_dupire.py`, `test/test_ppp_dki_snowball_case_study.py`, `test/test_equity_lifecycle_trackers.py`
- Keep on old paths (shim coverage): `test/test_otc_autocallable_backtest.py`, `test/test_otc_autocallable_dashboard.py`, `test/test_book_backtest.py`, `test/test_backtest_interface.py`

**Interfaces:**
- Produces (in `replay/__init__.py` and `quantark.backtest`): `ReplayBacktestEngine`, `ReplayBacktestConfig`, `ReplayProduct`, `ReplayBacktestResults`, `HedgeSpec` — for now simple aliases: `ReplayBacktestEngine = BookAutocallableBacktestEngine` (class renamed with alias kept), `ReplayBacktestConfig = BookAutocallableBacktestConfig` (alias), `ReplayProduct = BookProduct` (alias), `ReplayBacktestResults = BookBacktestResults` (alias). Renames-in-place happen in Tasks 7–8; aliases guarantee both names always resolve.

- [ ] **Step 1: `git mv` + create shims + fix intra-package relative imports** (`from .config import` etc. move cleanly; `replay/single.py` imports `from .product_replay import ProductReplay`).

- [ ] **Step 1b: Relocate the book classes to the canonical modules.** The live
`BookAutocallableBacktestConfig`, `BookProduct`, and `HedgeSpec` are defined in
`book_engine.py` (→ now `replay/engine.py`) and `BookBacktestResults` at its
bottom. Move the class definitions: `BookAutocallableBacktestConfig`,
`BookProduct`, `HedgeSpec` → `replay/config.py`; `BookBacktestResults` →
`replay/results.py`; `replay/engine.py` imports them from there. This is what
makes `from quantark.backtest.replay.config import ReplayBacktestConfig` (Task
6 dispatch, Task 8/12/14 edits to `config.py`/`results.py`) actually resolve —
without it, later tasks would edit the wrong modules. The `otc/book_engine.py`
shim re-exports all four names regardless.
- [ ] **Step 2: Update `base.py`, `backtest/__init__.py`, canonical-path consumers.**

In `get_backtest_engine` (`base.py:260`), point the existing `AutocallableBacktestConfig` branch at `quantark.backtest.replay` and ADD the canonical dispatch ahead of it:

```python
from quantark.backtest.replay.config import ReplayBacktestConfig
if isinstance(config, ReplayBacktestConfig):
    from quantark.backtest.replay.engine import ReplayBacktestEngine
    return ReplayBacktestEngine(config)
```

(`AutocallableBacktestConfig` is not a `ReplayBacktestConfig` subclass, so order between the two branches is safe either way; keep Replay first as the canonical path.) Add a `test/test_backtest_interface.py` case: `get_backtest_engine(ReplayBacktestConfig(...)) → ReplayBacktestEngine` and the existing `AutocallableBacktestConfig` case still dispatches.
- [ ] **Step 3: Run the full affected suite:**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest \
  test/test_otc_autocallable_backtest.py test/test_book_backtest.py \
  test/test_otc_autocallable_dashboard.py test/test_backtest_interface.py \
  test/test_otc_vol_calibrators.py test/test_otc_vol_history_env.py \
  test/test_replay_goldens.py test/test_backtest_lifecycle.py -q
```
Expected: PASS, with DeprecationWarnings from shim-covered tests only.

- [ ] **Step 4: Commit** `git commit -m "refactor(backtest): promote product replay to quantark.backtest.replay; otc/ becomes deprecation shims"`

---

### Task 7: Strategy → `strategy/futures_delta_strategy.py`, subclassing `BaseStrategy`

**Files:**
- Create: `quantark/backtest/strategy/futures_delta_strategy.py`
- Modify: `quantark/backtest/replay/strategy_state.py` (re-export only), `quantark/backtest/strategy/__init__.py` (export), `otc/state.py` shim (unchanged — re-exports through `strategy_state`)
- Test: `test/test_futures_delta_strategy.py`

**Interfaces:**
- Produces: `AutocallableDeltaHedgeStrategy(BaseStrategy)` — constructor kwargs UNCHANGED (`delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0, round_contracts=True`); replay-facing methods UNCHANGED (`target_contracts(*, product_delta, product_quantity, futures_multiplier) -> float`, `should_rebalance(current_contracts, target_contracts) -> bool`, `get_parameters() -> dict`); NEW protocol methods `should_hedge(current_time, portfolio_greeks, market_data, **kw) -> bool` and `calculate_hedge_size(...) -> float` implemented as adapters (`portfolio_greeks["delta"]` = net position delta, `market_data["futures_multiplier"]` required, `ValidationError` if absent).

- [ ] **Step 1: Failing test** — construct with defaults; `target_contracts` rounding/hedge-ratio math matches current values (port the assertions style from `test_otc_autocallable_backtest.py`); `isinstance(strategy, BaseStrategy)`; `should_hedge`/`calculate_hedge_size` adapters agree with `target_contracts`/`should_rebalance` on the same inputs; validation errors preserved (`delta_threshold<0`, `hedge_ratio∉[0,1]`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement as a plain class extending `BaseStrategy` (`name="AutocallableDeltaHedge"`, `asset_class=AssetClass.EQUITY`, `hedging_target=HedgingTarget.DELTA`, `hedge_instrument="futures"`); body of the three existing methods copied verbatim; dataclass decorator dropped (BaseStrategy has its own `__init__`).
- [ ] **Step 4: Run** strategy test + Task-6 suite → PASS (engines construct the strategy by default — byte-identity via goldens proves the adapter didn't disturb sizing).
- [ ] **Step 5: Commit** `git commit -m "refactor(backtest): futures delta strategy joins the BaseStrategy hierarchy"`

---

### Task 8: Unify — port calibrator into `ReplayBacktestEngine`

Book engine gains the single engine's calibration capability. The single engine is NOT yet a wrapper (that is Task 9); this task only makes the book engine calibration-capable and byte-equal on book-of-one localvol.

**Files:**
- Modify: `quantark/backtest/replay/engine.py` (class renamed `ReplayBacktestEngine`, alias `BookAutocallableBacktestEngine = ReplayBacktestEngine`)
- Modify: `quantark/backtest/replay/config.py` (Task 6 Step 1b put the book config classes here: rename `BookAutocallableBacktestConfig`→`ReplayBacktestConfig` and `BookProduct`→`ReplayProduct`, old names kept as aliases; add the vol-model admissibility validation below)
- Modify: `quantark/backtest/replay/results.py` (book results — canonical home per Task 6 Step 1b: add `calibration_records` param + property + `export_calibration_records`, same as single results)
- Test: extend `test/test_book_backtest.py`-adjacent new file `test/test_replay_engine_unified.py`

**Interfaces:**
- Consumes: `VolModelCalibrator` (Task 4 canonical path), `create_vol_model_engine` (existing).
- Produces: `ReplayBacktestEngine._calibrate_day(date) -> dict | None` — calibrates ONCE per day (shared artifact), then rebuilds each alive replay's day-engine from the same `CalibratedVolModel`. Engine resolution order per day (MUST mirror `replay/single.py:130-166` exactly): build env → calibrate (if `vol_model != "bsm"` and (any alive or initial value pending)) → initial book value → lifecycle → price/greeks.

- [ ] **Step 1: Failing test** — `test_replay_engine_unified.py::test_book_of_one_localvol_matches_single`: run `fixtures.make_localvol_config` through the (still separate) single engine AND the same product wrapped as a book-of-one `ReplayBacktestConfig` with `vol_model="localvol"`; compare per-frame with the golden contract (`assert_frame_equal(check_exact=True)` on states/greeks/trades/actions; calibration_records equal after dropping timing keys). Add rejection tests: a Phoenix book-of-one with `vol_model="localvol"` raises `ValidationError` at config construction; a mixed Snowball+Phoenix book with `vol_model="localvol"` raises; the same mixed book with `vol_model="bsm"` constructs fine.
- [ ] **Step 2:** FAIL (`ReplayBacktestConfig` rejects/ignores vol_model → book run diverges or errors).
- [ ] **Step 3: Implement** — in `ReplayBacktestEngine.__init__`, mirror `single.py:71-81` (validator + `VolModelCalibrator` construction, `ValidationError` when `surface_history` missing). In `run()`, insert after env build:

```python
day_calibration_record = None
if self._calibrator is not None and (
    any(r.lifecycle.alive for r in self._replays) or self._initial_book_value is None
):
    day_calibration_record = self._calibrate_day(date)
```

(The condition mirrors `single.py:131-133`: calibrate while anything is alive, and always on the first day so the initial price is model-consistent.)

`_calibrate_day` calibrates once, then for each replay sets that day's engine (delivered to the replay calls — until Task 10 lands, assign `replay.pricing_engine = day_engine` for each replay, exactly like the single engine does today). Timing: wrap the per-day pricing block with `time.perf_counter()` into `pricing_seconds` as `single.py:158-166` does.

**Product/model admissibility (fail-closed):** `create_vol_model_engine` builds
ONLY Snowball vol-model engines (`LocalVolSnowball*`, `HestonSnowball*`,
`HestonSLVSnowball*`) and takes no product argument, while replay books accept
Phoenix and European products. Guard at config construction: when
`engine_config.vol_model != "bsm"`, `ReplayBacktestConfig.__post_init__` (and
the single config via the wrapper) raises `ValidationError` unless EVERY
`ReplayProduct.product` is a `SnowballOption`. Wiring the existing Phoenix
vol-model solvers into the factory is explicitly deferred (YAGNI — the study
prices snowballs); the rejection makes the gap loud instead of silently
pricing a Phoenix with a Snowball engine.

- [ ] **Step 4: Run** unified test + Task-6 suite + goldens → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(backtest): ReplayBacktestEngine gains per-day vol-model calibration (book-of-one == single, exact)"`

---

### Task 9: Single engine becomes a book-of-one wrapper

**Files:**
- Modify: `quantark/backtest/replay/engine.py` — per-product recording: `_record_day` computes per-product state/greek rows FIRST (exact single-engine schema, columns copied from `single.py:428-494`: states `date, portfolio_value, product_mtm, hedge_mtm, cash, cashflows, transaction_costs, product_pnl, hedge_pnl, total_pnl, spot, volatility, rate, basis_yield, implied_q, pricing_q, active_contract, futures_price, futures_ttm, futures_multiplier, futures_contracts, alive, knocked_in, knocked_out, matured` [+ surface-provenance keys when present]; greeks `date, price, delta, gamma, product_delta, product_gamma, product_position_delta, product_position_gamma, pre_hedge_contracts, post_hedge_contracts, futures_multiplier, pre_hedge_futures_delta, post_hedge_futures_delta, pre_hedge_delta, post_hedge_delta, pre_hedge_gamma, post_hedge_gamma, pre_hedge_delta_cash_1pct, post_hedge_delta_cash_1pct, pre_hedge_gamma_cash_1pct, post_hedge_gamma_cash_1pct, delta_cash_1pct, gamma_cash_1pct, vega, theta, rho, dividend_sensitivity, basis_sensitivity`), then derives the existing book-level row from them (book columns unchanged).
- Rewrite: `quantark/backtest/replay/single.py` → thin wrapper (~80 lines): keeps `AutocallableBacktestConfig` dataclass (public API), builds `ReplayBacktestConfig(products=[ReplayProduct(product=cfg.product, quantity=cfg.product_quantity, position_id=0, has_lifecycle=True, initial_price=cfg.initial_product_price)], …)`, runs `ReplayBacktestEngine`, and builds `AutocallableBacktestResults` (property-based API, `results.py` class unchanged) from the per-product sinks of the sole product.
- Test: `test/test_replay_goldens.py` (unchanged — THE gate), extend `test_replay_engine_unified.py` with a permanent `test_book_of_one_is_byte_identical_to_single` for all three fixture configs.

**Interfaces:**
- Consumes: Task 8's unified engine.
- Produces: `AutocallableBacktestEngine(config).run() -> AutocallableBacktestResults` — signature, results class, and every frame byte-identical to the goldens. `BookBacktestResults` keeps its **callable** accessors (`trades_df()`, `states_df()`, …) and `products_meta` constructor kwarg exactly as today.

- [ ] **Step 1:** Run goldens against the wrapper (they now exercise the unified loop) → expect FAIL initially; iterate until `pytest test/test_replay_goldens.py -q` is fully green. Every mismatch is a bug in the port — never touch the golden files.
- [ ] **Step 2:** Delete the old single-engine loop body from `single.py` (the wrapper replaces it). `grep -rn "class AutocallableBacktestEngine" quantark/` must show exactly one definition.
- [ ] **Step 3:** Full suite: Task-6 list + `test/test_ppp_dki_snowball_case_study.py` + `test/mo_volmodels/test_stage12_backtest_runner.py` → PASS.
- [ ] **Step 4: Commit** `git commit -m "refactor(backtest): single autocallable engine is now a book-of-one wrapper — one daily loop"`

---

### Task 10: Explicit engine flow (kill the mutable swap) + BumpConfig greeks

**Files:**
- Modify: `quantark/backtest/replay/product_replay.py` — `calculate_greeks(product, env, price, *, engine)`, `record_surfaces(..., *, engine)`, initial pricing call sites take `engine`; DELETE the mutable `pricing_engine` attribute and the ~70-line manual bump block (`product_replay.py:260-337`); DELETE the `except Exception → zeros` fallback (fail-closed).
- Modify: `quantark/backtest/replay/engine.py` — resolves `day_engine` once per day per product; passes it explicitly; `single.py` wrapper unchanged.
- Modify: `quantark/asset/equity/param/engine_params.py` — `BumpConfig` gains `gamma_spot_bump: Optional[float] = None` (falls back to `spot_bump`); `quantark/asset/equity/engine/base_engine.py::calculate_greeks` honors it for the second-difference bump.
- Modify: `quantark/backtest/replay/engine_factory.py` — `create_pricing_engine`/`create_vol_model_engine` accept `delta_bump_size`/`gamma_bump_size` and set `params.bump_config = BumpConfig(spot_bump=delta_bump, gamma_spot_bump=gamma_bump)` when provided.
- Test: `test/test_replay_greeks_failclosed.py` + BumpConfig unit test appended to the existing engine-params test module (locate via `grep -rl "BumpConfig" test/`)

**Interfaces:**
- Produces: `ProductReplay.calculate_greeks(product, env, price, *, engine) -> dict` — ALWAYS `engine.calculate_greeks`; raises on engine failure. `BumpConfig(gamma_spot_bump=…)` engine-side.

- [ ] **Step 1: Failing tests** — (a) a raising stub engine propagates (`pytest.raises`), no zero-delta dict; (b) `BumpConfig(spot_bump=0.01, gamma_spot_bump=0.02)`: `base_engine.calculate_greeks` prices at ±1% for delta and ±2% for gamma (assert via a spy engine recording env spots); (c) config `delta_bump_size=0.005` flows into the constructed engine's `params.get_effective_bump_config().spot_bump`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** Goldens MUST stay green: the engines' `get_effective_bump_config()` default (1% spot bump) equals the old replay fallback path only when the old code took the native/engine path — if a golden diff appears, the old manual-bump numbers differ from engine-side bumping; STOP and reconcile by setting the factory's BumpConfig from the same sources the old code read (`params.get_effective_bump_config().spot_bump` / legacy `bump_size`), not by re-adding replay-side bumping.
- [ ] **Step 5:** Run new tests + goldens + Task-6 suite → PASS. **Step 6: Commit** `git commit -m "refactor(backtest): engines flow explicitly through the replay; greeks via engine BumpConfig, fail-closed"`

---

### Task 11: Opt-in event-stats fallback + surface logging + behavioral compat test

**Files:**
- Modify: `quantark/backtest/replay/config.py` — `AutocallableEngineConfig.event_stats_fallback: Literal["none","mc"] = "none"` (validated in `__post_init__`).
- Modify: `quantark/backtest/replay/product_replay.py` — `_calculate_event_stats`: `"none"` → exceptions propagate, **and a `None` return from the primary engine (the `BaseEngine.calculate_event_stats` "unsupported" signal) raises `PricingError`** when event probabilities were requested — today `None` is a silent no-op that leaves the requested frames empty, which is exactly the fail-open path being removed; `"mc"` → on exception OR `None`, fallback via `create_mc_event_stats_engine`, `logger.warning`, and every row appended in that day's `record_event_probabilities`/`daily_event_sink` gains `"event_stats_engine": "mc_fallback"` (primary path writes `"primary"`). `record_surfaces` failure branch logs `logger.warning("surface node failed", exc_info=True)` — NaN row kept.
- Test: `test/test_replay_event_stats_fallback.py`, and `test/test_backtest_otc_compat.py` (the §9.4 behavioral compat test)

**Interfaces:**
- Produces: new column `event_stats_engine` in `daily_event_summary` and `event_probabilities` frames. Goldens: REGENERATE these two frame families ONLY (the new column with constant `"primary"`), via a targeted re-capture run; every other golden file must remain bit-identical (`git diff --stat` on `test/replay_golden/data` shows only `*_daily_event_summary.csv` / `*_event_probabilities.csv`).

- [ ] **Step 1: Failing tests** — default `"none"`: a raising stub event-stats engine propagates AND a `None`-returning stub raises `PricingError`; `"mc"`: for both a raising stub and a `None`-returning stub, the run completes, warning logged (`caplog`), provenance column present with `mc_fallback`.
- [ ] **Step 2–3:** Implement; regenerate the two golden families; verify `git diff --stat`.
- [ ] **Step 4: Behavioral compat test** (`test_backtest_otc_compat.py`): via OLD paths only (`quantark.backtest.otc.*`): construct both configs, run both engines end-to-end on the scalar fixture, touch EVERY public accessor — single results properties (`states_df, greeks_df, rebalance_df, trades_df, actions_df, surfaces_df, daily_event_summary_df, event_probability_df, calibration_records, get_summary, get_total_pnl, get_total_return, get_pnl_series, get_value_series, get_hedge_trades, get_lifecycle_events`), book results **methods** (`states_df(), greeks_df(), rebalances_df(), trades_df(), actions_df(), daily_event_summary_df(), event_probability_df(), surfaces_df(), get_summary()`) and `BookBacktestResults(..., products_meta=[…])` constructor kwarg; assert `quantark.backtest` re-exports (`AutocallableBacktestEngine`, `get_backtest_engine` dispatch on both config types); assert `pytest.warns(DeprecationWarning)` on shim imports (use `importlib.reload` to re-trigger).
- [ ] **Step 5:** Run everything → PASS. **Step 6: Commit** `git commit -m "feat(backtest): opt-in MC event-stats fallback with provenance; behavioral otc compat suite"`

---

### Task 12: Metrics split — `CorePerformanceMetrics`

**Files:**
- Create: `quantark/backtest/metrics.py`
- Modify: `quantark/backtest/equity/metrics.py` (subclass, equity-only metrics stay), `quantark/backtest/replay/results.py` (`.metrics` property), `quantark/backtest/__init__.py` (export `CorePerformanceMetrics`)
- Test: `test/test_core_metrics.py`

**Interfaces:**
- Produces: `CorePerformanceMetrics(results)` consuming ONLY `get_pnl_series()/get_value_series()/get_hedge_trades()/get_total_pnl()/get_total_return()` — methods moved from equity `PerformanceMetrics`: `returns_series, total_pnl, total_return, sharpe_ratio, max_drawdown, max_drawdown_duration, win_rate, profit_factor, value_at_risk, conditional_var, volatility, skewness, kurtosis, calculate_all_metrics (core fields only), to_dataframe`. Equity `PerformanceMetrics(CorePerformanceMetrics)` keeps `hedge_frequency, average_hedge_cost, total_hedge_cost_ratio, delta_tracking_error, average_absolute_delta, delta_rebalance_efficiency` and overrides `calculate_all_metrics` to add them — public API unchanged.

- [ ] **Step 1: Failing contract test** — instantiate `CorePerformanceMetrics` on an `AutocallableBacktestResults` from the scalar fixture; call EVERY core metric (no exception, finite or defined-zero returns). Instantiate equity `PerformanceMetrics` on an equity `BacktestResults` stub (build minimal stub exposing the extra attributes) and call EVERY public metric.
- [ ] **Step 2–4:** Implement split; run new test + `pytest -k "metrics" -q` + `test/test_backtest.py` → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(backtest): cross-asset CorePerformanceMetrics; replay results gain .metrics"`

---

### Task 13: `replay/schema.py` — typed row contracts

**Files:**
- Create: `quantark/backtest/replay/schema.py`
- Modify: `replay/engine.py`, `replay/product_replay.py`, `replay/results.py` — build rows via schema constants
- Test: `test/test_replay_schema.py`

**Interfaces:**
- Produces: `TypedDict`s `StateRow, GreekRow, TradeRow, RebalanceRow, ActionRow, SurfaceRow, DailyEventRow, EventProbRow, CalibrationRecord` + column-order tuples `STATE_COLUMNS, GREEK_COLUMNS, …` (values copied verbatim from Task 9's lists + Task 11's provenance column). Row-builder call sites reference the TypedDicts in annotations; results `_frame` reindexes to the schema order when all schema columns are present.

- [ ] **Step 1: Failing test** — for each frame of the scalar fixture run: `list(df.columns) == list(SCHEMA_ORDER)` (allowing the optional surface-provenance suffix set for surface-mode states); TypedDict keys == schema order sets.
- [ ] **Step 2–3:** Implement; goldens must remain bit-identical (schema encodes the existing order — any golden diff is a transcription bug in schema.py, fix schema.py).
- [ ] **Step 4:** Run schema test + goldens → PASS. **Step 5: Commit** `git commit -m "refactor(backtest): declare replay record schemas as the single source of truth"`

---

### Task 14: Pending-settlement KO termination

**Files:**
- Modify: `quantark/asset/equity/lifecycle/state.py` — `AutocallableLifecycleState` gains `pending_settlement_cashflow: float = 0.0`, `settlement_date: Optional[datetime] = None`, `settled: bool = False`; `mark_ko(date, cashflow, settlement_date=None)`: when `settlement_date` is later than `date`, park cashflow in `pending_settlement_cashflow` instead of `realized_cashflows`; new `settle(date)` moves it to `realized_cashflows` and sets `settled=True`. When `settlement_date is None` (or equal to `date`) behavior is EXACTLY today's (immediate realization, `settled=True`) — the T+0 degenerate case. **`mark_maturity` sets `settled=True` unconditionally** (maturity settlement is immediate) — without this, clean-maturity and KI-maturity runs never satisfy the all-settled termination predicate and default-on runs would continue to data end with a wrong `termination_reason`.
- Modify: `quantark/asset/equity/lifecycle/autocallable.py` — `_scheduled_records` carries `settlement_time` when the product's schedule records expose it (`getattr(rec, "settlement_time", None)` on `ko_observation_schedule.records`; `None` → observation date); `observe()` KO branch resolves `settlement_date = self._date_resolver(obs_date_for(settlement_time))` and passes it to `mark_ko`; KO `LifecycleEvent.metadata` gains `settlement_date`.
- Modify: `quantark/backtest/replay/config.py` — `terminate_on_lifecycle_end: bool = True` on BOTH configs (`ReplayBacktestConfig`, `AutocallableBacktestConfig`).
- Modify: `quantark/backtest/replay/engine.py` — daily: after lifecycle, call `replay.settle_pending_if_due(date)` (new `ProductReplay` passthrough to `lifecycle.settle`); portfolio value adds the discounted receivable `pending_settlement_cashflow * env.get_discount_factor(tau_settle)` where `tau_settle = (settlement_date - date).days / 365.0` (**the live `PricingEnvironment` API is `get_discount_factor`, not `get_df`**; when settlement ≤ date the receivable is zero); loop breaks when `terminate_on_lifecycle_end` and every replay is `settled` (dead AND cash posted) — AFTER recording that day's row. `get_summary()` gains `termination_reason` (`"ko"` if any KO'd, `"ki_maturity"` if matured knocked-in, `"maturity"` if matured clean, `"data_end"` otherwise), `days_replayed`, `days_in_contract` (calendar length of `_backtest_dates()`).
- Modify: `test/replay_golden/fixtures.py` — **every fixture config gains `terminate_on_lifecycle_end=False`** in the same commit that introduces the flag. The goldens were frozen before the flag existed and the scalar fixture KOs mid-path; without this edit the default-on truncation fails every golden. The golden DATA files are untouched — only the fixture configs pin legacy semantics.
- Test: `test/test_replay_termination.py`

- [ ] **Step 1: Failing tests** — (a) **T+0 degenerate:** scalar fixture with a path forcing KO; flag ON → last state row is the KO date, cash posted that day, `termination_reason=="ko"`, `days_replayed < days_in_contract`; flag OFF → frames equal the golden (goldens run flag OFF — assert against the same golden files); (b) **T+5 synthetic:** monkeypatch the product's KO schedule records with `settlement_time = obs_time + 5/365`; KO cash absent from `realized_cashflows` on obs date, portfolio value includes discounted receivable, cash posts on resolved settlement date, replay ends there; (c) **EXPIRY-paid:** settlement_time = maturity; run continues (dead, unpriced) to maturity, cash posts at maturity; (d) book: two products, one KO'd — run continues until BOTH settled; (e) **clean maturity:** no-KO path, flag ON → run ends on the maturity settlement date with `termination_reason=="maturity"`, not `data_end`; (f) **KI maturity:** KI'd no-KO path → ends at maturity with `termination_reason=="ki_maturity"`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement per file list. **Step 4:** Run termination tests + goldens (flag OFF path) + `test/test_equity_lifecycle_trackers.py` + `test/test_dynamic_scenario*.py -k lifecycle` (the tracker is shared with dynamicscenario — its callers pass no `settlement_date`, so behavior there is unchanged; verify) → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat(backtest): pending-settlement KO termination — replay ends when terminal cash lands"`

---

### Task 15: Efficiency — shallow env copies, pre-grouped futures

**Files:**
- Modify: `quantark/backtest/replay/product_replay.py` — add module-level `_env_with(env, *, spot=None, div_yield=None)` constructing a NEW `PricingEnvironment` sharing `vol_surface`/`rate_curve`/`basis_yield` objects and replacing only `spot_quote`/`div_yield`/`valuation_date`; `record_surfaces` uses it (replaces `deepcopy`).
- Modify: `quantark/backtest/replay/market.py` — `AutocallableMarketDataSet` builds `self._futures_by_date: dict[pd.Timestamp, pd.DataFrame]` lazily on first `get_futures_slice` (groupby once); `get_futures_slice` returns `self._futures_by_date[date].copy()` (`ValidationError` on missing date, same message).
- Test: `test/test_replay_env_helpers.py`

- [ ] **Step 1: Failing tests** — `_env_with` returns env whose `vol_surface is env.vol_surface` (shared identity) but `spot` replaced; mutation of the copy's spot does not affect the original; `get_futures_slice` equals the previous filter result (`assert_frame_equal`) and still raises on missing dates.
- [ ] **Step 2–3:** Implement. **Step 4:** goldens bit-identical (shared-object envs are semantically identical because market objects are immutable by convention) → PASS. **Step 5: Commit** `git commit -m "perf(backtest): shallow bump envs and pre-grouped futures chain"`

---

### Task 16: Docs, changelog, final verification

**Files:**
- Modify: `quantark/backtest/CLAUDE.md` (module tree, imports, replay section replaces otc section), root `CLAUDE.md` (backtest row + layout table), `CHANGELOG.md` (unreleased: consolidation, deprecations with old→new mapping table, `terminate_on_lifecycle_end`, `event_stats_fallback`, `gamma_spot_bump`)
- Verify-only: full suite + real-data anchor

- [ ] **Step 1:** Write docs/changelog.
- [ ] **Step 2:** Full test suite: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q` → PASS (record count).
- [ ] **Step 3:** Real-data anchor (if Task 1 Step 5 captured it): re-run stage-12 quick with `terminate_on_lifecycle_end=False` patched into the runner config (one-line temporary edit, reverted after) and `diff -r` the CSV outputs against `output/volmodel_backtest_golden_pre_consolidation/` ignoring manifest/timing files. Document the result in the commit message.
- [ ] **Step 4: Commit** `git commit -m "docs(backtest): document the replay package, deprecations, and new flags"`

---

## Execution order & gates

Task 0 → 1 (goldens) → 2 → 3 → 4 → 5 → 6 → 7 (all behavior-frozen; goldens green after EVERY task) → 8 → 9 (unification; goldens are the gate) → 10 → 11 (behavior; goldens regenerated ONLY where the spec says) → 12 → 13 → 14 → 15 → 16.

After Task 16: Stage-6 code review (feature-flow), then merge via `superpowers:finishing-a-development-branch`.
