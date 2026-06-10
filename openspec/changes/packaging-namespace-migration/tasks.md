# Tasks: Packaging & Namespace Migration

## 1. Setup & Repo Hygiene (Phase A)

- [x] 1.1 Create a git worktree for this change (per D8 — the main checkout
  backs the consumer's editable install; do not implement on it)
  *(worktree at `.claude/worktrees/packaging-namespace-migration`, branch
  `worktree-packaging-namespace-migration`, base `85f379b`)*
- [x] 1.2 Untrack run outputs and agent-tool state with `git rm -r --cached`:
  `output/`, `dynamic_results/`, `model-validation-output/`,
  `stress_results/`, `tmp/`, `.gemini/`, `.playwright-mcp/`, root screenshots
  and one-off HTML files (`openspec/` stays tracked)
  *(deviation: `.codex/` and `.claude/` stay tracked — inspection showed they
  hold the project's shared skills/commands/hooks, not run state; also
  untracked additional run-output dirs found during survey: `logs/`, `plots/`,
  `reports/`, `risk_metric_analysis/`, `stress_scenarios/`, plus
  `example/output/` stragglers — 449 files total)*
- [x] 1.3 Extend `.gitignore` to cover everything untracked in 1.2 plus
  `.venv/`; commit hygiene as its own commit *(commit `281c4ec`; `.venv` was
  already ignored)*
- [x] 1.4 Create `.venv/` from `requirements.txt` and verify
  `.venv/bin/python -m pytest --collect-only` works *(deviation: deleting the
  old `quantark/` venv happens in the main checkout at merge time — task 5.4
  — because the untracked venv dir would collide with the incoming tracked
  `quantark/` package, and the main checkout stays untouched until gates
  pass)*
  *(Baseline at `85f379b`: 1910 passed, 4 skipped, 7m18s. Pre-existing
  failures on main, all tied to missing symbols in
  `example/generate_snowball_rfq_ko_rate_demo.py`:
  `test_snowball_rfq_ko_rate_engine_consistency.py` (collection error),
  `test_generate_snowball_rfq_ko_rate_demo.py` (1F),
  `test_generate_snowball_rfq_ko_rate_demo_quad_1001.py` (3F) — out of scope,
  flagged for a separate bug fix.)*

## 2. Namespace Move (Phase C, part 1)

- [x] 2.1 `git mv` the 12 flat packages (`asset`, `backtest`, `cashleg`,
  `dynamicscenario`, `param`, `portfolio`, `priceenv`, `rfq`, `simm`,
  `stresstest`, `util`, `var`) under a new `quantark/` package with an
  `__init__.py`; commit the moves separately from any content edits
  *(commit `93a7855`, 452 renames; also drops the stale `quantark/`
  .gitignore entry)*
- [x] 2.2 Codemod all internal imports under `quantark/` from flat names to
  `quantark.*` (both `import X` and `from X import` forms); verify with a
  grep pass that zero flat imports remain in library code
  *(commit `fc0620f`, 412 files; tree had zero bare `import <root>` lines;
  one mock.patch string target fixed in test_simm_reports.py)*
- [x] 2.3 Codemod `test/` (~131 files) imports to `quantark.*`; verify with
  the same grep pass (leave `example/` on flat imports as live shim
  exercisers) *(same commit as 2.2)*
- [x] 2.4 Replace the repo-root path math in
  `quantark/util/calendar/business_calendar.py::_load_holidays_from_csv`
  with `importlib.resources.files("quantark.util")`-based resolution
  *(commit `c04e579`)*
- [x] 2.5 Run the full pytest suite from the worktree venv (editable install
  of the moved package) and fix fallout until green *(fallout was 19
  state-dependent failures, root-caused to 20 library files whose
  module-level `sys.path.insert` "repo root" computations landed on the
  `quantark/` package dir after the move — fixed in `55bb0fa`, plus shim
  hardening in `8ec844e`)*

## 3. Compatibility Shim (Phase C, part 2)

- [x] 3.1 Implement `quantark/_compat.py`: a `MetaPathFinder` mapping the 12
  legacy roots (and submodules) to existing `quantark.*` module objects via a
  loader whose `create_module` returns the canonical module
  *(commits `fc0620f` + `f912f46`; deviation from design: the finder
  is **prepended**, not appended — an appended finder never sees submodule
  imports because PathFinder resolves them through the aliased parent's
  `__path__` into duplicate modules; precedence for real distributions is
  explicit logic instead. The loader also restores `__spec__`/`__loader__`
  clobbered by `_init_module_attrs`, or importlib.resources breaks.
  design.md D3 amended.)*
- [x] 3.2 Emit one `DeprecationWarning` per legacy root per process, naming
  the `quantark.*` replacement
- [x] 3.3 Add `quantark_compat.pth` (containing `import quantark._compat`) to
  the distribution so the finder registers at interpreter startup
  *(hatchling force-include; landed with the packaging commit)*
- [x] 3.4 Add shim tests: deep submodule legacy import, module identity
  (`old is new` both import orders), enum member identity across spellings,
  legacy-import-first in a fresh subprocess, warn-once behavior, and
  installed-package-precedence (fake `param` on `sys.path` wins over alias)
  *(test/test_legacy_import_compat.py — 9 tests, all passing; the identity
  tests are what caught the append-order design flaw)*

## 4. Packaging (Phase B)

- [x] 4.1 Audit real runtime dependencies: grep third-party imports across
  `quantark/`, classify hard deps vs. lazy/optional (plotly, seaborn,
  openpyxl, pyarrow suspects in dashboard/exporters); decide extras per D5
  *(all top-level imports are unguarded → hard deps: numpy, scipy, pandas,
  matplotlib, seaborn, plotly, pyarrow, pyyaml, python-docx, plus
  python-dateutil which was previously only transitive; openpyxl/kaleido
  declared hard as runtime-dynamic deps of pandas-Excel/plotly-export;
  pytest moves to the `dev` extra. No extras split — nothing is lazily
  guarded except one simm plotly import.)*
- [x] 4.2 Write `pyproject.toml` (PEP 621): metadata, `requires-python =
  ">=3.10"`, audited dependencies, wheel limited to `quantark`, holiday CSVs
  as package data, `.pth` shim file; delete `setup.py` *(deviation: hatchling
  backend instead of setuptools — setuptools has no pyproject-only way to
  place a `.pth` at the site-packages root; hatchling force-include covers
  regular AND editable wheels. design.md D4 amended.)*
- [x] 4.3 Fresh-venv smoke test (non-editable): wheel contains only
  `quantark/` + `quantark_compat.pth` + dist-info, both CSVs included, no
  stray flat packages; from `cwd=/`: `import asset` first → resolves to
  `quantark.asset`, dashboard import ok, CSV resolves via
  `importlib.resources`, enum identity across spellings holds
- [x] 4.4 Repeat the smoke test with an editable install (`pip install -e`)
  to confirm `.pth` shim registration works in both modes *(worktree `.venv`
  is the editable install; legacy-first import from neutral cwd passes)*

## 5. Acceptance Gates & Docs

- [x] 5.1 Gate 1: full quant-ark pytest suite green on canonical imports in
  the worktree *(1920 passed, 4 skipped, 0 failed — baseline 1910 + 10 new
  shim tests; same 3 pre-broken example-dependent modules excluded as in the
  baseline)*
- [x] 5.2 Gate 2: run open-otc-trading's full suite **unchanged** against an
  editable install of the migrated worktree *(isolated venv
  `/tmp/otc_gate2_venv` with consumer deps + editable worktree quantark,
  `QUANTARK_PATH=<worktree>/quantark`: 1805 passed, 2 failed — byte-identical
  to the consumer's own-venv baseline against un-migrated main
  (`/tmp/otc_baseline.log` vs `/tmp/otc_gate2.log`); the 2 failures are
  pre-existing consumer bugs unrelated to quant-ark)*
- [x] 5.3 Update `CLAUDE.md`, `AGENTS.md`, `README.md`: venv path
  (`quantark/` → `.venv/`), install instructions, canonical import examples,
  legacy-name deprecation note *(commit `922e7b4`)*
- [x] 5.4 Merge the worktree branch to `main`, re-point the consumer's
  editable install at the migrated `main`, and confirm the consumer app
  boots (consumer's own import flip is Phase D in its repo, out of scope)
  *(Done 2026-06-10: main had moved (`73b7107` book engines) — merged main
  into the branch first, resolved two import conflicts by re-codemod, suite
  green (1921 passed); then deleted the old `quantark/` venv in the main
  checkout, fast-forwarded main, restored the 449 untracked output files
  from backup, removed `__pycache__`-only flat dir remnants, created
  `.venv/` with editable install (smoke green). Consumer: its venv is
  uv-managed and had NO quantark install (pure sys.path injection) —
  installed editable quantark via uv (shim `.pth` deployed), set
  `QUANTARK_PATH=/Users/fuxinyao/quant-ark/quantark` in its `.env` (interim
  config until Phase D: keeps the CSV-by-path lookup working; the package-
  dir sys.path insert is neutralized by the shim's leak guard). Consumer
  suite in its own venv: 1876 passed, 2 failed — the same two pre-existing
  failures as baseline, zero new. If the consumer app is currently running,
  restart it to pick up the new install.)*
