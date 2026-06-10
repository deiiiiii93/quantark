# Tasks: Packaging & Namespace Migration

## 1. Setup & Repo Hygiene (Phase A)

- [ ] 1.1 Create a git worktree for this change (per D8 — the main checkout
  backs the consumer's editable install; do not implement on it)
- [ ] 1.2 Untrack run outputs and agent-tool state with `git rm -r --cached`:
  `output/`, `dynamic_results/`, `model-validation-output/`,
  `stress_results/`, `tmp/`, `.codex/`, `.claude/`, `.gemini/`,
  `.playwright-mcp/`, root screenshots and one-off HTML files
  (`openspec/` stays tracked)
- [ ] 1.3 Extend `.gitignore` to cover everything untracked in 1.2 plus
  `.venv/`; commit hygiene as its own commit
- [ ] 1.4 Recreate the `quantark/` virtualenv as `.venv/` from
  `requirements.txt`, delete the old venv directory, and verify
  `.venv/bin/python -m pytest --collect-only` works

## 2. Namespace Move (Phase C, part 1)

- [ ] 2.1 `git mv` the 12 flat packages (`asset`, `backtest`, `cashleg`,
  `dynamicscenario`, `param`, `portfolio`, `priceenv`, `rfq`, `simm`,
  `stresstest`, `util`, `var`) under a new `quantark/` package with an
  `__init__.py`; commit the moves separately from any content edits
- [ ] 2.2 Codemod all internal imports under `quantark/` from flat names to
  `quantark.*` (both `import X` and `from X import` forms); verify with a
  grep pass that zero flat imports remain in library code
- [ ] 2.3 Codemod `test/` (~131 files) imports to `quantark.*`; verify with
  the same grep pass (leave `example/` on flat imports as live shim
  exercisers)
- [ ] 2.4 Replace the repo-root path math in
  `quantark/util/calendar/business_calendar.py::_load_holidays_from_csv`
  with `importlib.resources.files("quantark.util")`-based resolution
- [ ] 2.5 Run the full pytest suite from the worktree venv (editable install
  of the moved package) and fix fallout until green

## 3. Compatibility Shim (Phase C, part 2)

- [ ] 3.1 Implement `quantark/_compat.py`: a `MetaPathFinder` appended to
  `sys.meta_path` that maps the 12 legacy roots (and submodules) to existing
  `quantark.*` module objects via a loader whose `create_module` returns the
  canonical module and whose `exec_module` is a no-op
- [ ] 3.2 Emit one `DeprecationWarning` per legacy root per process, naming
  the `quantark.*` replacement
- [ ] 3.3 Add `quantark_compat.pth` (containing `import quantark._compat`) to
  the distribution so the finder registers at interpreter startup
- [ ] 3.4 Add shim tests: deep submodule legacy import, module identity
  (`old is new` both import orders), enum member identity across spellings,
  legacy-import-first in a fresh subprocess, warn-once behavior, and
  installed-package-precedence (fake `param` on `sys.path` wins over alias)

## 4. Packaging (Phase B)

- [ ] 4.1 Audit real runtime dependencies: grep third-party imports across
  `quantark/`, classify hard deps vs. lazy/optional (plotly, seaborn,
  openpyxl, pyarrow suspects in dashboard/exporters); decide extras per D5
- [ ] 4.2 Write `pyproject.toml` (PEP 621, setuptools backend): metadata,
  `requires-python = ">=3.10"`, audited dependencies, `packages.find`
  limited to `quantark*`, holiday CSVs as package data, `.pth` shim file;
  delete `setup.py`
- [ ] 4.3 Fresh-venv smoke test (non-editable): build + install, then in a
  new interpreter run `import asset` as the first import, and
  `python -c "from quantark.backtest.otc import
  AutocallableBacktestDashboard"`; verify the two CSVs resolve via
  `importlib.resources`; inspect the wheel for stray top-level packages
- [ ] 4.4 Repeat the smoke test with an editable install (`pip install -e`)
  to confirm `.pth` shim registration works in both modes

## 5. Acceptance Gates & Docs

- [ ] 5.1 Gate 1: full quant-ark pytest suite green on canonical imports in
  the worktree
- [ ] 5.2 Gate 2: run open-otc-trading's full suite **unchanged** against an
  editable install of the migrated worktree (its flat imports exercise the
  shim; its cross-channel equivalence tests catch module-identity bugs)
- [ ] 5.3 Update `CLAUDE.md`, `AGENTS.md`, `README.md`: venv path
  (`quantark/` → `.venv/`), install instructions, canonical import examples,
  legacy-name deprecation note
- [ ] 5.4 Merge the worktree branch to `main`, re-point the consumer's
  editable install at the migrated `main`, and confirm the consumer app
  boots (consumer's own import flip is Phase D in its repo, out of scope)
