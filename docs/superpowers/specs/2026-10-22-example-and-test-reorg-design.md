# Example and test folder reorganisation — design

Date: 2026-10-22. Status: draft for approval. Repo: quant-ark (`quantark` package).

## Why

`example/` holds two different kinds of material in one flat heap, and `test/` has grown the same way.

- `example/` has 113 loose scripts at the top level (plus 4 loose support files) next to 13 case-study folders (~20 MB tracked). Finding the demo for a module means scanning ~120 filenames. Some case studies are hiding among the loose scripts (`ppp_dki_snowball_backtest_case_study.py`, the `ai_index_*` group, the eight-file `snowball_rfq_ko_rate_*` / `calculate_*` / `generate_*` / `render_*` workflow).
- `test/` has 898 test files, 550 of them flat at the root. Roughly 100 of the files under `test/` exist only to re-run study pipelines (`test/mo_volmodels` 35, `test/modelvalidation` 42, `test/volcalibration` 22, `test/fx_volmodels` 5, plus ~30 root files that run example scripts). Those tests import scripts by file path and run whole pipelines as subprocesses.
- The concrete pain that started this: `test/mo_volmodels/test_suite_smoke.py` regenerates `*_sample.*` files in the repo tree on every run. Two of them were tracked, so the working tree was permanently dirty (commit `cc975cac` untracked them). The same coupling caused the history-gated suite hang and the xdist shared-artifact race.
- `quantark/modelvalidation/cli.py` hard-codes `STUDY_DIR = Path("example/modelvalidation")`, and the `equity_snowball_localvol` builder reads committed surfaces from `example/modelvalidation/data/` and hashes them into its identity. The library depends on a folder named "example".

## Goals

1. **How-to examples** (how to use quantark) live in a tree that **mirrors the `quantark/` package tree** (e.g. `example/asset/bond/`, `example/var/`, `example/saccr/`), so a user finds the demo by the module they are using.
2. **Case studies** live apart from the how-tos, **one self-contained subfolder per study** (README, scripts, committed inputs). `mo_volmodels` is the flagship: it is finished work and the reference pattern for users who want to calibrate vol models for a new market. Its README says so.
3. **`test/` is reorganised the same way**: root-level `test_*.py` files grouped into subfolders mirroring the package, extending what `pde_core/`, `execution/`, `intraday/` already do.
4. **The library never reads from `example/`.** Anything the package needs at runtime lives inside the package or a package-owned location.
5. **Studies are not re-run by the unit test suite.** `mo_volmodels` keeps exactly one end-to-end test (it is a user-facing pattern a user will copy, so it must keep working), and that test writes to a temp directory, never into the repo.
6. Generated output stays out of git (the rule applied in `cc975cac`), as does `__pycache__`.

## Non-goals

- No change to any pricing, calibration or model behaviour. This is moves, deletions of redundant tests, and reference fixes only.
- No rewriting of example scripts beyond what a move forces (paths, imports, `HERE`-relative data lookups).
- No new documentation beyond a short README per top-level area and the updated references.
- Not touching `docs/modelvalidation` certificate *contents*; only the harness paths they cite, and only as far as the move in project 1 requires (see Open points).

## Anti-goals (a failure even if it technically works)

- A reference left pointing at a path that no longer exists (README, package docstring, certificate, test, CI).
- A moved test that silently stops being collected, or a library-behaviour test deleted along with the study plumbing.
- Any golden / bitwise / certificate gate changing value because of a move.
- Study results being "re-generated to make tests pass".

## Order of work (five projects, one at a time, each landed before the next starts)

### 0. Clean up all branches first

Decided by the user: every existing branch is resolved before any file moves, because mass `git mv` conflicts with anything unmerged that touches `example/` or `test/`. Local branches today: `accrual-time`, `afv-default-payoff`, `codex/equity-option-settlement-date`, `fix-cb-default-drift`, `handoff-small-fixes`, `hazard-from-spread`, `ko-reset-lifecycle-v2`, `ko-reset-lifecycle-v3`, `range-accrual-tenor`, two `worktree-agent-*` branches and `worktree-ship-tf-tree`, plus any remote branches and stale worktrees.

For each branch the builder reports: commits ahead of `main`, whether it is already contained in `main`, files touched under `example/` and `test/`, and last activity. Then one of: **already merged** (delete), **land** (ff-only onto `main`, after the full suite), **rebase later** (kept, but the user accepts rebasing over the reorg), or **abandon** (user's explicit word). Nothing unmerged is deleted without the user naming that branch. Landing a branch is the user's decision per branch; the builder never pushes.


### 1. Take `modelvalidation` out of `example/`

The package reads `example/modelvalidation` at runtime. Move the study definitions (YAMLs) and the committed surface data it depends on to a **new top-level repo folder that is not part of the wheel** (decided by the user; the 124 KB must not ship in the release), update `STUDY_DIR`, the builders' data lookups and the identity-hash inputs, and update the references in `docs/modelvalidation` (~89) and `docs/` generally.

Constraint: identity hashes and banked certificates must stay valid. Moving a file must not change a sha256 that enters an identity hash; if hashing includes the path string, that is a decision to surface, not to bury (see Open points).

### 2. Reorganise `example/`

- Classify every top-level script as how-to or case study. Classification is by reading, not by filename; the script-to-package table below is the starting point (measured by which `quantark.*` packages each script imports).
- How-tos: `git mv` into the mirror tree. Where a package has no demo today, no folder is created.
- Case studies: each gets its own folder with a README; multi-file workflows (`snowball_rfq_ko_rate_*`) are gathered into one study folder. Existing study folders move under the case-study area.
- Fix every reference: root `README.md` (~12 scripts), seven package READMEs and docstrings, `docs/`, test paths, `.gitignore`.
- Add a README per top-level area explaining the split; the `mo_volmodels` README leads with its role as the vol-calibration reference pattern.
- Remove `example/__pycache__` and decide `example/data/` by what reads it.

Starting mapping of the 113 loose scripts (importers' dominant package; the builder finalises this by reading the scripts):

| Destination | Scripts (by name) |
| --- | --- |
| `asset/equity/…` (largest group, ~60: snowball, phoenix, vanilla/american/asian/barrier, accumulator, DCN, PDE/QUAD/MC engine demos) | needs a sub-split, see Left to the builder |
| `asset/bond/` | `fixed_bond_demo`, `bond_option_demo`, `bond_forward_futures_demo`, `frn_demo`, `convertible_bond_demo`, `convertible_bond_term_structure_demo` |
| `asset/rate/` | `irs_demo` |
| `asset/credit/` + `simm`/`stresstest` | `credit_portfolio_risk_demo` (spans three packages; builder picks) |
| `asset/fx/` | the 13 `fx_*` scripts (several span `stresstest`/`backtest`/`dynamicscenario`/`portfolio`; primary asset class wins unless the demo is really about the risk module) |
| `var/`, `stresstest/`, `portfolio/` | `parametric_var`, `historical_var`, `monte_carlo_var`, `portfolio_var`, `var_backtest`, `stress_test`, `stresstest_fi_rate_shocks`, `portfolio_demo` |
| `backtest/`, `dynamicscenario/`, `execution/`, `intraday/`, `cashleg/`, `pnlexplain/`, `saccr/`, `sacva/`, `simm/` | the like-named demos |
| `param/`, `volmodels/` | `sabr_smile_demo`, `vanna_volga_smile_demo`, `trading_clock_demo`; `vol_models_demo`, `fp_*`, `adi_cs_convergence`, `heston_calibration_benchmark`, `volmodels_phase5_*` |
| `montecarlo/` | `qmc_mc_convergence_gbm` |
| case studies | `ppp_dki_snowball_backtest_case_study`, the `ai_index_*` group, the `snowball_rfq_ko_rate` workflow (8 files), `generate_snowball_greeks_spot_report_quad_1001` (needs reading) |

### 3. Triage the study tests

For each of the ~100 study-scoped test files, decide by reading it:

- **Library behaviour tested through study code** (e.g. `test_snowball_q_term_structure_study.py`, `test_adi_greek_certification.py`, `test_mo_frozen_feller.py` are candidates): move to the matching library test folder and stop depending on the study script. Where the behaviour lives in script code that should be library code, that is flagged, not silently promoted.
- **Study plumbing only** (stage-N output shape, report rendering, scheduler): delete, or fold into the single end-to-end test.
- **`mo_volmodels`**: one end-to-end offline test run from the committed sample snapshot, outputs in a temp directory. This replaces `test_suite_smoke.py`.
- Every deletion is listed with its reason in the PR/commit message; the count of library-behaviour tests must not drop without a stated reason.

### 4. Reorganise `test/`

Group the 550 root-level `test_*.py` files into subfolders mirroring the package, using `git mv` only. Handle `conftest.py`, shared fixture modules (`*_fixtures.py`, `helpers_*.py`, `golden_compare.py`), relative data paths and `test/data`. Must preserve: collection count, golden/bitwise gates, CI invocation.

Do 4 last: it should not run while test paths are still moving for other reasons.

## Constraints that bind every project

- CI is x86_64, `TZ=UTC`, Python 3.10–3.13; run the full suite, not a sample, before declaring any project landed. Four mo/otc tests are known to fail on main since 0.7.0; they stay known-failing, not newly broken.
- Moves are `git mv` so history follows; the commit that moves files contains no content edits other than path fixes.
- Each project is its own branch/worktree and lands ff-only; the user decides when to push.
- Anything user-visible (new folder layout) is described in the root `README.md`; `CLAUDE.md` and `openspec` are not tracked, so shipping guidance goes in `README.md`.
- No desk data enters the public repo as part of a move.

## Decided by the user

- Two example kinds: how-to (organised like `test/`, mirroring the package tree) and case study (one subfolder each).
- `test/` is reorganised too.
- `mo_volmodels` is finished work and a reference pattern for users calibrating vol models; it is not live tooling.
- Order: clean up all branches, then modelvalidation out of `example/`, then `example/`, then study tests, then `test/`.
- `modelvalidation` data goes to a new top-level folder that stays out of the wheel.
- All branches are cleaned up before any file moves.

## Open points (my calls, flagged for the user to check)

- **Name of the new folder** (project 1). Decided by the user: top-level, outside the wheel. My call on the name: `modelvalidation_studies/` at the repo root (the CLI's `STUDY_DIR` is cwd-relative today, so this keeps working from a checkout). The builder confirms the wheel `include` list in `pyproject.toml` does not pick it up.
- **Names of the two example areas.** My call: the package-mirror folders sit directly under `example/` (`example/asset/bond/`, `example/var/`, …) and the studies sit side by side with them under `example/case_studies/<study>/`.
- **Identity hashes and paths** (project 1). Checked: the `equity_snowball_localvol` identity covers the surface *content* sha256 (`surface_sha256`), not the file path, so moving the two surface JSONs leaves it unchanged. The builder verifies the other builders the same way, and if any hash covers a path, it stops and reports rather than re-banking.
- **End-to-end `mo_volmodels` test** (project 3). My call: it runs the same offline stages 02–10 the old smoke test ran, on a *copy* of the committed sample snapshot in a temp directory, with all outputs in that directory (no shared artifact, so no xdist race), and with the old smoke test's runtime as its budget. It must not use `--quick` or any MC_FULL gate.
- **Library behaviour that only works by importing a script** (project 3). My call: such a test is kept, never deleted; it moves beside the library tests and imports the script explicitly. The script is not promoted to library code in this program. Each case is logged as a follow-up item.
- **Interim handling of tests that run example scripts.** My call: project 2 fixes their paths in the same commit as the script move (a script move is not complete until its referencing tests pass); project 3 triages; project 4 moves.
- **Test infrastructure changes** (project 4). My call: adding `__init__.py` files or adjusting `conftest.py` / `pytest` config is allowed where moves require it, but each such change is listed and the baseline collection count (taken with `pytest --collect-only` before project 4 starts and recorded) must be unchanged afterwards.
- **`example/data/` and the loose support files.** My call: each goes next to its only reader; if both a how-to and a study read one, it is copied, not shared, so each study stays self-contained.

## Left to the builder

- The exact placement of the ~60 equity scripts inside `asset/equity/` (by product such as `snowball/`, `phoenix/`, `vanilla/`, or flat).
- Final classification of ambiguous scripts (how-to vs case study; which package a multi-package demo belongs to).
- How to detect all references mechanically (grep patterns, a link-check script) and prove none dangle.
- The exact set of surviving study tests in project 3, per the rules above.
- The `test/` subfolder names in project 4, following whatever the package tree and the already-existing subfolders suggest.
