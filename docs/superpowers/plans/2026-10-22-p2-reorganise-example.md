# Project 2: reorganise `example/` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user chose native execution). Steps use checkbox syntax.

**Goal:** Turn the flat `example/` heap into (a) how-to examples laid out like the `quantark/` package tree and (b) one self-contained folder per case study under `example/case_studies/`, with every reference still pointing at something that exists.

**Architecture:** One mapping (script/folder → destination), applied with `git mv` by a throwaway script kept in the scratchpad (not in the repo). A second pass fixes (1) path arithmetic inside moved scripts (`parents[N]`, `.parent` chains) by the depth change, (2) textual references to old paths in code, tests, docs and `.gitignore`. Verification is by compiling every moved file, running every test that touches `example/`, and a final full suite.

**Tech Stack:** git, Python 3 (standard library for the move script), pytest (main venv with `PYTHONPATH` shadow).

**Spec:** `docs/superpowers/specs/2026-10-22-example-and-test-reorg-design.md` (project 2)

## Global Constraints

- Run pytest as `cd <worktree> && env PYTHONPATH=$PWD:$PWD/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` for EVERY run.
- Moves are `git mv` only; the move commit has no content edits. Path fixes go in separate commits so rename detection survives.
- Do not edit `docs/modelvalidation/certificates/**`, `docs/superpowers/**` (historical specs/plans) or any banked evidence.
- Generated output stays out of git; do not add `__pycache__`, `.DS_Store` or sample outputs.
- No desk data enters the public repo.
- The user's untracked work in the main checkout (`example/intraday_ko_pin_risk/`, `intraday_pin_risk/`, `snowball_ftp_study/`, `standalone_rho/`, `intraday_snowball_near_k[io]_demo.py`, `test/intraday/test_ko_pin_risk_study.py`) and the 72 MB untracked `example/mo_volmodels/data/history/` are NOT touched by this branch; the landing step moves them only with the user's yes.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Layout (rulings are mine unless marked)

How-to examples mirror the package tree at the top of `example/`: `asset/{bond,credit,fx,rate}/`, `asset/equity/{accumulator,dcn,deltaone,engines,greeks,options,phoenix,snowball,trs}/` (the equity package is organised by layer, not product, so the equity depth stops at product families), `backtest/`, `cashleg/`, `dynamicscenario/`, `execution/`, `intraday/`, `modelvalidation/`, `param/`, `pnlexplain/`, `portfolio/`, `saccr/`, `sacva/`, `simm/`, `stresstest/`, `var/`, `volcalibration/`, `volmodels/`.

Case studies live in `example/case_studies/<study>/`: the 10 existing study folders (`mo_volmodels`, `fx_volmodels`, `snowball_q_term_structure`, `gaussian_quad_comparison`, `snowball_simulated_paths`, `quad_v2_validation`, `quad_reference_comparison`, `quad_v2_feasibility`, `snowball_volmodel_backtest`, `intraday_benchmark`) plus six new ones gathered from loose scripts: `ai_index_strategy` (5), `snowball_rfq_ko_rate` (8 scripts + its HTML template), `snowball_greeks_quad_1001` (1), `ppp_dki_snowball_backtest` (1), `volmodels_improvement_program` (10: `adi_cs_convergence`, `fp_*`, `volmodels_phase5_*`, `heston_calibration_benchmark`), `phoenix_engine_diagnostics` (2), and `engine_benchmarks` (5: cash-leg folding, dask, PDE term-structure, PDE perf, QMC convergence). Ruling: benchmarks and diagnostics are analyses with a conclusion, so they count as case studies.

`example/modelvalidation/` and `example/volcalibration/` stay at the top (they mirror the packages). `example/data/*_lecture_latest.html` move beside the script that writes them (`asset/equity/greeks/data/`, `pnlexplain/data/`).

## Review Focus

- A moved script that computes the repo root from `__file__` still finds the repo root (depth changed by 1–3).
- Cross-study references (`volcalibration` → `fx_volmodels/data`, `snowball_q_term_structure` → `mo_volmodels/data/history`, `snowball_simulated_paths` → `snowball_q_term_structure`) point at the new locations.
- Tests that import example modules via `sys.path` + module name, or by `spec_from_file_location`, still collect and run (a silently uncollected test is a failure).
- `.gitignore` entries naming old example paths (`/example/mo_volmodels/data/history`, `example/snowball_q_term_structure/data/bucket_hedge_v2/`) are updated, or the 72 MB data would show as untracked at its new path.
- READMEs (root, seven package READMEs) name scripts that exist.

---

### Task 1: Mapping and baseline

- [ ] **Step 1:** Write `mapping.tsv` (scratchpad) from the rules above. Expected: every tracked file under `example/` (excluding `modelvalidation/`, `volcalibration/`) appears exactly once; no destination collides; `??` count is zero.
- [ ] **Step 2: Baseline:** run every test file that mentions `example/` (list from `git grep -lI "example" -- test`) plus `test/mo_volmodels test/fx_volmodels test/volcalibration`; record pass/skip/fail counts in the ledger.
- [ ] **Step 3:** Record every path-arithmetic site: `git grep -nE "parents\[[0-9]\]|\.parent(\.parent)+|\"example\"" -- example` to the scratchpad as `before.txt`.

### Task 2: Move

- [ ] **Step 1:** Apply the mapping with `git mv` (create directories first). Expected: `git status` shows only renames; file count under `example/` unchanged.
- [ ] **Step 2:** Commit `refactor(example): move scripts into package-mirror and case_studies layout`.

### Task 3: Fix path arithmetic and references (one commit per class)

- [ ] **Step 1: Depth fix.** For each moved `.py`, with old depth `D_old` and new depth `D_new` (directories between the repo root and the file): `parents[k]` → `parents[k + (D_new − D_old)]` when `k == D_old − 1` (it names the repo root); `.parent` chains of length `D_old` → length `D_new`; `Path(__file__).parent` forms that name the study directory are left alone. Expected: a script that printed its repo root before prints the same after (spot-check 10 scripts by importing them with `python -c`).
- [ ] **Step 2: Textual references.** Replace old `example/<name>` with the new location in code, tests, READMEs, docstrings and docs (not historical specs/certificates), both slash form and `"example" / "<name>"` join form. Expected: a second `git grep` for old names returns only intentionally frozen strings.
- [ ] **Step 3: `.gitignore`.** Update the two example-path rules.
- [ ] **Step 4:** Compile every moved file with `python -m py_compile`. Expected: no errors.
- [ ] **Step 5:** Commit each class separately.

### Task 4: Indexes

- [ ] **Step 1:** Add `example/README.md` (how the two areas work, one line per package folder, pointer to `case_studies/`) and `example/case_studies/README.md` (one line per study: what it asks, how to run). Update the root `README.md` repo-layout line and its script paths.
- [ ] **Step 2:** Commit.

### Task 5: Verify

- [ ] **Step 1:** Re-run the Task 1 baseline set. Expected: identical to baseline.
- [ ] **Step 2:** Run the full suite (`-n 4`, background). Expected: failures only if they also fail on `main`; otherwise fix and re-run the failing tests.
- [ ] **Step 3:** Run 5 representative moved scripts end to end (one per family: a bond demo, an FX demo, a snowball demo, a var demo, a case-study entry point with `--help`). Expected: exit 0.
- [ ] **Step 4:** Ledger; report to the user. Landing on `main` and moving the untracked data in the main checkout are the user's calls.
