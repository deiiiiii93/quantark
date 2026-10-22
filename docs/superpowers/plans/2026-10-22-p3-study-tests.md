# Project 3: triage the study tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (the user chose native execution). Steps use checkbox syntax.

**Goal:** `test/` stops depending on study code for anything that is really library behaviour, keeps one end-to-end guard for the `mo_volmodels` reference pattern, and drops (or relocates) tests that only exercise a study's own plumbing, without losing a single library-behaviour assertion.

**Architecture:** Every study-coupled test file (78 files, 1213 tests) was classified by reading it (five read-only agents plus my spot checks, see the verdict table). Tier 1 (library behaviour) is decoupled from `example/` by copying the small fixtures it needs under `test/` and moved next to the library tests. Tier 2 (patterns) is consolidated into one end-to-end `mo_volmodels` test that works in a temp directory, plus the demo smoke tests that already exist. Tier 3 (plumbing) is removed, with the keep/delete split decided below.

**Tech Stack:** pytest (main venv with `PYTHONPATH` shadow), git.

**Spec:** `docs/superpowers/specs/2026-10-22-example-and-test-reorg-design.md` (project 3)

## Global Constraints

- Run pytest as `cd <worktree> && env PYTHONPATH=$PWD:$PWD/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` for EVERY run.
- Baseline: the clean full-suite run on this branch (after the dotted-import fix `466a3b49`) is recorded in the ledger; the end state must equal it minus exactly the tests the ledger lists as removed, plus the tests it lists as added. Collected-ID diff (`pytest --collect-only -q`) is the proof.
- The count of library-behaviour assertions must not drop: every Tier 1 test is moved or rewritten, never deleted. A test removed as Tier 3 gets one ledger line with its reason.
- No test may write into tracked or shared repo paths; the new end-to-end test works on a temp copy.
- A fixture added under `test/` is public market or synthetic data only; no desk data.
- Commits end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Verdicts (from reading; counts are test functions)

**Tier 1, library behaviour (keep, decouple from `example/`):**
- `test/volcalibration/*` (14 files, ~134 tests): copy `mo_snapshot_sample.json`, `mo_settlement_snapshot_20260430.json`, `cfets_usdcny_snapshot_{20260430,20260720}.json` (~150 KB) into `test/volcalibration/data/`; point the tests at them. Exceptions kept pointing at the real example config on purpose: `test_cli.py::test_the_shipped_mo_config_loads_with_the_frozen_fingerprint` and `::test_list_finds_the_shipped_config` (they guard the shipped example). `test_mo_common_shim.py` and the `_mo_common` comparison in `test_surface_smoothing.py` guard the example's legacy shim and move with Tier 3.
- Root tests that only mention an example in a comment: `test_quad_v2_engine`, `test_pde_term_structure_solvers`, `test_vol_pde_reverse_continuous_ki`, `test_pde_engine_event_stats` (no change needed beyond the comment).
- `test_futures_carry_recorder.py` (drop its one test that loads study `_common.py`), `test_otc_vol_history_env.py`, `test_otc_vol_calibrators.py` (42) and `test_mo_frozen_feller.py` (4): they read one untracked artifact, `mo_iv_surface_20230504.json`, so ~46 tests are silently skipped everywhere but the author's machine. Fix: commit that one artifact under `test/` (decision below).
- From `mo_volmodels`: 3 engine-constructor tests in `test_gate_scope.py`, `test_iv_inversion.py` (the sample-smile recovery has no library twin), 2 `VolSurfaceHistory` tests in `test_surface_admission_dupire.py` (rewrite on a synthetic history dir; currently all skipped), the 3 synthetic validator tests in `test_build_iv_surface_history.py`.
- From `fx_volmodels`: the flat-surface Dupire test, the FP leverage test and the published-forward environment test (inline the construction).
- `test/simulation/test_measures.py`: keep 3 of 4 (drop the one that loads the study).
- `test/modelvalidation/test_localvol_study.py`: already independent of `example/`; untouched (project 4).

**Tier 2, patterns (consolidate or keep):**
- `mo_volmodels` stage tests `test_stage02`…`test_stage09` and `test_suite_smoke.py` collapse into ONE end-to-end test: copy the committed sample snapshot into `tmp_path`, run the offline stages 02–07, 06 and 10 as the smoke test does, plus the 04 contract assertions and the 03/07 numeric thresholds; stages 08/09 stay out (expensive, ADI) unless the user wants them. Outputs go to `tmp_path`; the stages must accept an output-directory argument or environment override (see Task 3).
- Kept as they are: `test/execution/test_examples.py` (2), `test/volcalibration/test_example_demo.py` (2), `test_cash_leg_folding_benchmark.py`, `test_fx_integration_demo.py`, the VaR demo tests. Flag: `test_var_backtest_demo.py::test_main_execution` is `pass`; it asserts nothing.

**Tier 3, plumbing (decided by the user: move, do not delete):** the tests that only exercise a study's own machinery move into that study's own `tests/` folder, so each study is self-contained and CI's default run (`pytest test`) no longer depends on it: `example/case_studies/mo_volmodels/tests/` (stage 12 and 13, dashboard, the rest of gate_scope, pde_convergence_gate, cohort, grid_preflight, explainer, daily pipeline and scheduler, bulk_fetch, refresh_market_cache, settlement_cross_date, calibration_stability, gate_g1, the shim tests, stage 06, and the certification-harness tests `test_adi_greek_certification`, `test_adi_slv_aggregate_certification`, `test_certificate_transfer`), `example/case_studies/fx_volmodels/tests/`, `example/case_studies/snowball_q_term_structure/tests/` (`test_bucket_hedge_analysis`, `test_bucket_hedge_study`, `test_bucket_hedge_validation_cli`, `test_snowball_q_term_structure_study`) and `example/case_studies/snowball_simulated_paths/tests/`. They are run with `PYTHONPATH=.:test pytest example/case_studies/<study>/tests` (documented in each study README). Pure near-duplicates of library tests (`test_parity.py`, `test_otm_filter.py`) are the only deletions, because `test/volcalibration/test_listed_*` already cover the same assertions.

**Tier 4, golden (decided by the user: migrate the key numeric tests into `test/golden/`):** a test is golden if it fails when a number produced by `quantark` code moves, compared against a frozen value (a pinned constant, a banked certificate, or an independent reference), not synthetic arithmetic inside a study's own statistics. Candidates named by the reading, to be confirmed one by one by the executor: `test_heston_identification::test_latest_market_fixture_jacobian_is_finite_bound_aware_and_full_shape` (condition number 20.17 on a committed fixture; cross-arch, so it must use `test/golden_compare.py` tolerance), `test_bucket_hedge_validation_cli::test_the_tail_stress_reproduces_the_designs_49_8752_bp`, `test_certificate_transfer::test_the_certified_envelope_matches_stage_16s_own_case_definitions` and `test_the_envelope_is_the_measured_extremes_of_those_cells`, `test_stage12::test_cost_model_uses_the_agreed_cffex_parameters`, `test_gate_scope::test_delta_quantum_matches_the_worked_example`, `test_pde_convergence_gate::test_build_snowball_product_from_terms`, and the study-versus-engine agreement checks `test_snowball_q_term_structure_study::test_static_dividend_matches_the_engine` and `test_bucket_hedge_study::test_the_static_context_and_the_replay_dividend_agree`. A golden test may still load the study module it checks by path (that is the number under guard), and says so in its docstring. Moving modelvalidation's banked-certificate tests into golden is project 4.

## Review Focus

- Every Tier 1 test still exists, runs and is NOT skipped after the move (compare skip counts before and after; a `skipif` on a fixture that no longer exists must fail loudly).
- The new end-to-end test leaves `git status` clean and passes under `-n 4` alongside the rest of the suite.
- Deleting a test must not make a study script uncollectable somewhere else (stage 13 loads stages 11/12; cert-chain tests load script 16).
- `QUANTARK_MO_HISTORY`/absolute-path assumptions (`test_artifact_reproduction.py`) either keep working locally or are replaced by the committed artifact.

---

### Task 1: Baseline and fixtures

- [ ] **Step 1:** Record the clean full-suite baseline (`pytest test -q -n 4`) and the collected IDs (`--collect-only -q`) in the ledger.
- [ ] **Step 2:** Create `test/volcalibration/data/` with the four snapshot JSONs and repoint `test/volcalibration/*` (one `FIXTURES` constant per file or a shared `conftest` fixture). Expected: the suite passes with `example/` temporarily renamed (`mv example example.off`), then restore.
- [ ] **Step 3:** Commit.

### Task 2: Revive the silently skipped library tests

- [ ] **Step 1:** Add `test/data/mo_iv_surface_20230504.json` (public CSI 1000 option surface, 20 KB) and make `test_mo_frozen_feller.py`, `test_otc_vol_calibrators.py` and `test_otc_vol_history_env.py::TestRealArtifactSmoke` read it. Expected: their skip count drops to zero in a clean checkout.
- [ ] **Step 2:** Run those files; Expected: all pass (these were never run on a clean checkout, so failures here are real findings: report them, do not weaken assertions).
- [ ] **Step 3:** Commit.

### Task 3: One end-to-end test for the `mo_volmodels` pattern

- [ ] **Step 1: Write the failing test** `test/golden/…` is NOT its home (it is a pattern guard, not a numeric golden): `test/mo_volmodels/test_pattern_end_to_end.py::test_offline_pipeline_runs_in_a_temp_directory`. It copies `example/case_studies/mo_volmodels` data inputs to `tmp_path`, runs the stages with outputs there, asserts every stage exits 0, the 04 output contract (sha256 surface provenance, jacobian svd keys, bootstrap counts) and the 03/07 thresholds from the old tests, and asserts `git status --porcelain example` is empty afterwards.
- [ ] **Step 2:** Run it. Expected: FAIL (stages write beside their scripts today).
- [ ] **Step 3:** Give the offline stages an output-directory override (smallest change: honour an environment variable such as `MO_VOLMODELS_DATA_DIR`, default unchanged). Document it in the study README.
- [ ] **Step 4:** Run it. Expected: PASS, and the repo tree unchanged.
- [ ] **Step 5:** Delete `test_suite_smoke.py` and `test_stage02…09` (their coverage is now this test); ledger the mapping from old test to new assertion.
- [ ] **Step 6:** Commit.

### Task 4: Move the remaining Tier 1 tests

- [ ] **Step 1:** Rewrite the listed mo/fx/simulation Tier 1 tests to call `quantark` directly (no study import), move them to `test/volcalibration/`, `test/volmodels/` or `test/simulation/` as appropriate, and delete the originals. Expected: moved tests pass; the original files either vanish or keep only their Tier 3 remainder.
- [ ] **Step 2:** Commit.

### Task 5: Relocate the plumbing tests into their studies

- [ ] **Step 1:** `git mv` each Tier 3 file into `example/case_studies/<study>/tests/` (add `conftest.py` only if a moved test needs the repo root on `sys.path`); delete the two duplicate shim files. Expected: `pytest example/case_studies/<study>/tests` runs them (with `PYTHONPATH=.:test`) and their results equal their pre-move results.
- [ ] **Step 2:** Add the run command to each study README; note that these tests are outside CI's default run.
- [ ] **Step 3:** Commit per study.

### Task 5b: Migrate the key numeric tests to `test/golden/`

- [ ] **Step 1:** Read each Tier 4 candidate and confirm or reject it against the golden rule; record the decision in the ledger.
- [ ] **Step 2:** `git mv` confirmed tests (the single test function, not the file, so split files) to `test/golden/<area>/`; cross-arch-sensitive ones use `test/golden_compare.py` tolerance, same-machine invariants stay exact.
- [ ] **Step 3:** Run them; Expected: pass. Commit.

### Task 6: Verify

- [ ] **Step 1:** Full suite. Expected: failures only if they also fail on the Task 1 baseline.
- [ ] **Step 2:** Collected-ID diff against the baseline equals exactly the removed/added tests in the ledger.
- [ ] **Step 3:** Skip count: the ~46 revived tests are passes, not skips. `git status` clean after the run.
- [ ] **Step 4:** Ledger; report to the user. Landing is the user's call.
