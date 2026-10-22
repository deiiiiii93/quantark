# Project 4: reorganise `test/` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user chose native execution). Steps use checkbox syntax.

**Goal:** The 546 root-level `test_*.py` files move into subfolders that mirror the `quantark/` package tree, and numeric-correctness tests gather in a `test/golden/` division, with the collected test set unchanged.

**Architecture:** One mapping (file → folder) applied with `git mv` (Part A, pure moves), then three mechanical fix passes: repo-root/data path arithmetic, imports of sibling test modules, and CI/doc paths. Proof is exact: the multiset of collected test names (`basename::function[params]`) is identical before and after, and the full suite equals the baseline. Part B then curates the golden division from the ones worth moving.

**Tech Stack:** git, Python 3 (standard library) for the move scripts, pytest (main venv with `PYTHONPATH` shadow).

**Spec:** `docs/superpowers/specs/2026-10-22-example-and-test-reorg-design.md` (project 4)

## Global Constraints

- Run pytest as `cd <worktree> && env PYTHONPATH=$PWD:$PWD/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` for EVERY run.
- Baseline (main `bac43a05`): 18076 passed, 1108 skipped, 1 xfailed, 0 failed, 0 errors. End state: identical totals and an identical multiset of collected test names.
- Moves are `git mv`; the move commit has no content edits. Fixes go in separate commits.
- **Folder-name rule (found in recon):** every test folder is a package (has `__init__.py`, like the existing ones), and `test/` is on `sys.path`, so a folder's name becomes a top-level import name. `quantark/_compat.py` aliases 12 legacy roots (`asset backtest cashleg dynamicscenario param portfolio priceenv rfq simm stresstest util var`); a test folder with one of those names would shadow `from var import …`. Those folders therefore get a `test_` prefix (precedent: `test_cashleg`), e.g. `test_var/`, `test_asset/`. All other folders use the package name.
- `test/conftest.py` and `pytest.ini` (`testpaths = test`, `-n auto --dist worksteal`) are unchanged. Shared helper modules (`*_fixtures.py`, `helpers_*.py`, `golden_compare.py`, `mc_se_*`, `term_structure_benchmarks.py`) stay at `test/` root, which keeps them importable by their current names.
- `test/<legacy>` `__init__.py` files must not collide with study or example names; checked by the full run.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Layout (rulings are mine)

A root test file goes to the folder of the `quantark` package it imports most, ignoring `util`, `param` and `priceenv` unless nothing else is imported. Measured distribution: `asset.equity` 258, `volmodels` 52, `backtest` 44, `asset.fx` 35, `simm` 20, `sacva` 19, `util` 18, `var` 15, `asset.bond` 13, `param` 11, `pnlexplain` 10, `montecarlo` 9, `stresstest` 7, `asset.rate` 7, `asset.credit` 6, `validation` 4, `saccr` 4, the rest ≤ 2.

- `test/test_asset/{fx,bond,rate,credit}/` and `test/test_asset/equity/<layer>/` where layer is the sub-package the file imports most: `product` (split further by the first filename token into families when the folder would exceed ~40 files: `snowball`, `phoenix`, `options` for barrier/asian/american/european/digital/vanilla, `dcn`, `trs`, `accumulator`, `other`), `engine_pde`, `engine_analytical`, `engine_mc`, `engine_quad`, `lifecycle`, `market`, `riskmeasures`, `report`, `process`, `param`, `other`.
- `test/test_backtest/`, `test_var/`, `test_simm/`, `test_stresstest/`, `test_util/`, `test_param/`, `test_dynamicscenario/`, `test_portfolio/`, `test_priceenv/`, `test_rfq/`, plus plain names `volmodels/`, `sacva/`, `saccr/`, `pnlexplain/`, `montecarlo/`, `validation/`, `execution/`, `volcalibration/`. Files importing no package go to `test/misc/`.
- Existing topic folders (`pde_core`, `pde_grid`, `pde_routes`, `pde_statistics`, `pde_bitwise`, `pde_3a`, `accrual_time`, `lifecycle_ko_reset`, `theta_boundary`, `simulation`, `replay_golden`, `intraday`, `modelvalidation`, `mo_volmodels`, `fx_volmodels`, `execution`, `volcalibration`, `volmodels`, `test_cashleg`) stay where they are; renaming them would rewrite every `from pde_3a…` style import for no gain. `test/README.md` states the rules.

## Review Focus

- A moved test that locates the repo root, `test/data` or a fixture from `__file__` still finds it (depth changes by 1–3).
- A test that imports another test module by its old top-level name (38 root tests do) still imports.
- No moved folder shadows a legacy flat import (`from var import …`, `from util import …`); the demo tests still pass.
- CI (`.github/workflows/tests.yml` names `test/pde_grid`, `test/pde_core`, `test/modelvalidation/test_anchors.py`, …) still selects the same tests.
- Collected-name multiset unchanged; skip count unchanged.

---

### Task 1: Baseline and mapping

- [ ] **Step 1:** Collect names at `HEAD` (`pytest test --collect-only -q`) to `ids_before.txt` as `basename::name`; confirm the count equals the baseline run's collected total.
- [ ] **Step 2:** Write `mapping.tsv` (scratchpad) with the rules above. Expected: every root `test_*.py` appears once, no collisions, no folder over ~60 files, no folder named like a legacy root without the prefix.
- [ ] **Step 3:** Record every `__file__`-based path (`parents[k]`, `.parent` chains) and every `from test_<x> import` in the root tests to `before.txt`.

### Task 2: Move

- [ ] **Step 1:** Create the folders with `__init__.py`; `git mv` per the mapping. Expected: `git status` shows renames only; test file count under `test/` unchanged.
- [ ] **Step 2:** Commit `refactor(test): mirror the package tree`.

### Task 3: Fix passes (one commit each)

- [ ] **Step 1: Depth.** For each moved file with old depth 1 and new depth D: `Path(__file__)…parents[1]` → `parents[D]`; `.parent.parent` chains reaching the repo root gain `D−1` hops; a bare `.parent` meaning "test/" (data next to the test) becomes the right ancestor. Expected: a script printing `Path(…).resolve()` of each computed root before/after matches (spot-check 10).
- [ ] **Step 2: Sibling test imports.** Rewrite `from test_<x> import …` / `import test_<x>` to the destination dotted path (`from test_asset.equity.product.test_<x> import …`). Expected: `git grep` for bare `test_<x>` imports returns nothing.
- [ ] **Step 3: Literal paths.** Fix strings naming `test/test_<x>.py`, `test/<file>` locations of moved tests in code, READMEs, `.github/workflows/tests.yml`, `docs/` (not historical specs/plans/certificates).
- [ ] **Step 4:** Compile all test files; run `pytest --collect-only` and diff against `ids_before.txt`. Expected: identical multiset, zero collection errors.

### Task 4: Index

- [ ] **Step 1:** `test/README.md`: the layout rules, the legacy-name `test_` prefix and why, where helper modules live, what is golden and how golden tests treat cross-architecture tolerance (`golden_compare.py`), how to run a folder.
- [ ] **Step 2:** Commit.

### Task 5: Verify Part A

- [ ] **Step 1:** Full suite. Expected: totals equal the baseline exactly.
- [ ] **Step 2:** Ledger; if green, Part B.

### Task 6 (Part B): golden division

- [ ] **Step 1:** Create `test/golden/` as the home for tests that fail when a number moves against a frozen value. Candidates, each confirmed by reading: the banked-certificate and anchor tests (`test/modelvalidation/test_banked_certificates.py`, `test_anchors.py`), `test_replay_goldens.py` with `replay_golden/`, `pde_bitwise/`, `test_pde_term_structure_solvers::test_snowball_pde_flat_identity_golden`, `test_quad_v2_engine` independent-lognormal checks, the localvol study pins, and the three rejected-for-now candidates only if their numbers prove to be quantark-produced.
- [ ] **Step 2:** Move with import rewrites and update `.github/workflows/tests.yml`'s explicit paths in the same commit. Expected: the CI selection (those named paths) still collects the same tests.
- [ ] **Step 3:** Re-run the collected-name diff and the full suite. Ledger; report. Landing is the user's call.
