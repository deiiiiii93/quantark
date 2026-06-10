# Design: Packaging & Namespace Migration

## Context

quant-ark is consumed by open-otc-trading via a hardcoded checkout path,
manual `pip install -e`, and runtime `sys.path` injection. `setup.py` uses
`find_packages()`, which installs 12 flat top-level packages (`asset`,
`backtest`, `cashleg`, `dynamicscenario`, `param`, `portfolio`, `priceenv`,
`rfq`, `simm`, `stresstest`, `util`, `var`) into `site-packages`. `param`
collides with HoloViz `param` on PyPI; the other generic names are one shared
venv away from silent shadowing. Full findings: `docs/packaging-migration.md`.

Constraints discovered during analysis:

- The repo's pre-configured virtualenv lives at `quantark/` — the exact
  directory the migration needs for the package root.
- The holiday CSV loader (`util/calendar/business_calendar.py:411`) computes a
  repo root from `Path(__file__).resolve().parents[2]` — a path-based lookup
  that must become resource-based.
- The consumer passes `util.enum` members straight into quant-ark
  constructors and its tests pin byte-identical pricing across channels, so
  any module-identity split (one class imported under two names) surfaces as
  pricing/enum bugs, not import errors.
- The consumer's venv holds an editable install pointing at this checkout;
  breaking it mid-flight breaks a running app.

## Goals / Non-Goals

**Goals:**

- One canonical top-level import package: `quantark.*`.
- `pip install` from a fresh venv yields a working library (deps declared,
  CSVs bundled).
- The 12 legacy flat names keep working, resolving to the *same module
  objects* as `quantark.*`, until the consumer migrates (Phase D).
- quant-ark's own tests exercise canonical imports; the consumer's untouched
  suite exercises the shim.

**Non-Goals:**

- Consumer-side changes (Phase D lives in open-otc-trading).
- Hosting/publishing decisions (GitHub vs. Gitee, PyPI), history scrubbing.
- Renaming private RFQ APIs (`RFQService._normalize_request`,
  `_evaluate_candidate`) the consumer calls.
- Upstreaming the consumer's sharkfin registry registrations.
- Removing the shim (Phase E, after consumer migration).

## Decisions

### D1. Physical move under `quantark/`, venv relocated to `.venv/`

`git mv` each of the 12 packages into `quantark/` (preserves history), add
`quantark/__init__.py`. The existing virtualenv at `quantark/` moves first:
recreate it as `.venv/` (venvs are not relocatable — recreate from
`requirements.txt` rather than rename), add `.venv/` to `.gitignore`, update
`CLAUDE.md`/`README.md`/`AGENTS.md` command examples.

*Alternative considered*: keep the flat layout and only add a `quantark`
facade package re-exporting the 12. Rejected: the flat packages would still
install into `site-packages`, so the collision problem (the actual motivation)
remains.

### D2. Internal import rewrite is mechanical and total

All `import <flat>` / `from <flat>... import` statements inside library code
are rewritten to `quantark.*` with a scripted codemod (regex over the 12 known
roots is sufficient — the names are unambiguous at line start; verify with a
post-pass grep that finds zero flat imports under `quantark/` and `test/`).
`test/` migrates in the same pass so the suite validates canonical imports.
`example/` scripts are left on flat imports deliberately — they become live
exercisers of the shim (migrating them is cheap follow-up work).

### D3. Compat shim: meta-path finder, registered via `.pth`, prepended with explicit precedence

A module `quantark._compat` defines a `MetaPathFinder` whose `find_spec`
handles any module name whose root is one of the 12 legacy names: it imports
the corresponding `quantark.<name>` module and returns a spec whose loader
hands back that **existing module object** (`create_module` returns it,
`exec_module` is a no-op), so `sys.modules["asset.equity"] is
sys.modules["quantark.asset.equity"]`. This is the only pattern that keeps
module identity for arbitrary-depth submodule imports
(`import asset.equity.engine.mc`).

*Alternative considered*: duplicate re-export package tree (old dirs with
`from quantark.x import *`). Rejected per the brief: submodule imports load
files twice → two class/enum identities → `isinstance` and enum comparisons
fail in ways that look like pricing bugs.

Registration: a `quantark_compat.pth` file shipped with the distribution
contains an `import quantark._compat` line, so site initialization installs
the finder before any user code runs — import order independent.
(*Alternative*: install the finder from `quantark/__init__.py`. Rejected:
only works if `quantark` happens to be imported before a flat name.)

Precedence: the finder is **prepended** to `sys.meta_path` with explicit
yield rules. *(Amended during implementation: the original "append" design
failed in testing — once a legacy root is aliased, its `__path__` points into
`quantark/`, so `PathFinder` resolves submodule imports like
`import util.enum` through that path into fresh duplicate modules before an
appended finder is ever consulted: the exact identity split the shim
prevents.)* The prepended finder defers explicitly: for root names it returns
`None` when `PathFinder` locates a real installed distribution (e.g. HoloViz
`param`; namespace placeholders without a loader don't count), and for dotted
names it only aliases when the root module is itself our alias. Net effect is
unchanged: real distributions win, collisions are deterministic instead of
silently shadowed, and a venv needing both HoloViz `param` and legacy flat
imports is exactly the situation the migration exists to eliminate.

Each first import of a legacy root emits one `DeprecationWarning` naming the
`quantark.*` replacement.

### D4. Packaging: `pyproject.toml` with hatchling backend

PEP 621 metadata with the **hatchling** build backend. *(Amended during
implementation: the original choice was setuptools for least churn, but
setuptools has no pyproject-only mechanism to place a `.pth` file at the
site-packages root. Hatchling's `force-include` maps an arbitrary source file
to the wheel root, and per the hatchling changelog (≥1.4.0) editable wheel
targets respect `force-include` by default — so the shim's `.pth` works for
both regular and editable installs.)* Wheel packages limited to `quantark`;
the holiday CSVs ride along as in-package files (hatchling includes
non-Python files inside selected packages by default; the smoke test pins
this). `requires-python = ">=3.10"` (matches the project's documented
support; `setup.py` said 3.8 — flagged in Open Questions). `setup.py` is
deleted, not kept alongside.

### D5. Dependency audit before declaring

Current `install_requires` is numpy/pandas/matplotlib/scipy. Audit by
grepping third-party imports across the moved tree; anything imported at
module-import time becomes a hard dependency; heavy optional imports (plotly,
seaborn, openpyxl — the dashboard HTML writer and exporters are suspects)
either become extras (`quantark[viz]`, `quantark[export]`) if their imports
are lazy/guarded, or hard deps if not. Acceptance is behavioral: fresh-venv
install + `python -c "from quantark.backtest.otc import
AutocallableBacktestDashboard"`.

### D6. Holiday CSVs via `importlib.resources`

`_load_holidays_from_csv` switches from `__file__`-relative repo-root math to
`importlib.resources.files("quantark.util") / "calendar" / "holidayfile" /
f"{name}.csv"`. Behavior (name matching, fallback when missing) is unchanged;
only resolution changes. This works for editable installs, wheels, and
zipped installs alike.

### D7. Repo hygiene is a separate, first commit

`git rm -r --cached` the run outputs and agent-tool state (`output/`,
`dynamic_results/`, `model-validation-output/`, `stress_results/`, `tmp/`,
`.codex/`, `.claude/`, `.gemini/`, `.playwright-mcp/`, root screenshots and
one-off HTML), extend `.gitignore`. `openspec/` **stays tracked** — it is the
project's working spec system, not tool state. Doing this first keeps the
namespace-move diff reviewable.

### D8. Work happens in a git worktree

Per the process notes: concurrent agent sessions have committed onto shared
checkouts, and the consumer's editable install points at this checkout. All
implementation happens in a worktree; `main` only moves when both acceptance
gates pass.

## Risks / Trade-offs

- [Module-identity split slips through] → identity is unit-tested directly
  (`import util.enum; import quantark.util.enum; assert one is the other`,
  enum member identity across spellings), and the consumer's cross-channel
  equivalence suite is the second gate — it pins byte-identical pricing.
- [Shim yields to an installed `param`/`util`-named distribution, breaking
  legacy flat imports in that venv] → intentional (deterministic beats silent
  shadowing); the DeprecationWarning and Phase D migration are the way out.
  Documented in README migration notes.
- [`.pth` not processed in some environments (isolated mode,
  `PYTHONNOUSERSITE`, unusual launchers)] → smoke test: fresh venv,
  non-editable install, new interpreter, `import asset` as the *first*
  import must succeed; editable-install variant tested too (pip's modern
  editable installs process `.pth` via the same site machinery).
- [Hidden runtime deps surface only on exotic code paths] → dependency audit
  is grep-based plus the fresh-venv smoke import; exporters/dashboards get
  explicit import smoke tests since they bit the consumer before (pyarrow).
- [Consumer's editable install breaks during implementation] → worktree
  isolation (D8); the consumer venv keeps pointing at an untouched `main`
  until gates pass.
- [Pickles or serialized artifacts referencing flat module paths] → shim
  resolves old paths at unpickle time as long as it ships; flagged as a
  consumer consideration for Phase E (shim removal), not this change.
- [Large mechanical diff obscures real changes] → hygiene commit first (D7),
  then `git mv` commits separate from the import-rewrite commit, then shim +
  packaging commits; each step keeps the suite green.

## Migration Plan

1. **Hygiene commit** (D7) — untrack outputs/tool state, extend `.gitignore`.
2. **Venv relocation** — recreate `.venv/`, update docs; coordinate so the
   consumer's editable install keeps working against `main` (it points at the
   main checkout, which is why implementation lives in a worktree).
3. **Move + rewrite** — `git mv` the 12 packages under `quantark/`, codemod
   internal + test imports, fix the CSV loader (D6).
4. **Shim** — add `quantark/_compat.py` + `.pth`, with identity tests.
5. **Packaging** — `pyproject.toml`, dependency audit, package data, delete
   `setup.py`.
6. **Gate 1**: quant-ark suite green on canonical imports (worktree venv,
   editable install).
7. **Gate 2**: open-otc-trading full suite green **unchanged** against an
   editable install of the migrated worktree.
8. Merge to `main`; re-point the consumer's editable install; consumer Phase D
   proceeds in its own repo.

**Rollback**: revert the merge; the shim means no consumer-visible API ever
disappeared, so rollback risk is confined to this repo.

## Open Questions

- `requires-python`: bump to `>=3.10` (documented support) or keep the old
  `>=3.8` floor from `setup.py`? Design assumes 3.10; cheap to relax.
- Shim lifetime: one transition window or indefinite-with-warnings?
  (Maintainer decision from the brief; does not block implementation.)
- Are matplotlib/seaborn/plotly hard deps or extras? Resolved by the D5 audit
  (depends on whether imports are lazy).
- Hosting (GitHub vs. Gitee) and history review for publication — maintainer
  decisions, out of scope for implementation.
