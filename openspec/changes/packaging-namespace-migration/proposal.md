# Packaging & Namespace Migration

## Why

quant-ark cannot be consumed as a normal Python dependency: its main consumer
(open-otc-trading) reaches it via a hardcoded absolute path, a manual
`pip install -e` doc step, and runtime `sys.path` injection — so the consumer
cannot be cloned and run by anyone but the author. Worse, `find_packages()`
installs **12 flat top-level packages** (`asset`, `backtest`, `cashleg`,
`dynamicscenario`, `param`, `portfolio`, `priceenv`, `rfq`, `simm`,
`stresstest`, `util`, `var`) directly into `site-packages`; `param` collides
with the HoloViz `param` package on PyPI, and generic names like `util`,
`portfolio`, `var`, `backtest` are silently shadowed (not errored) when any
other package in a shared venv claims them. This blocks the "shareable MVP"
milestone (see `docs/packaging-migration.md`, 2026-06-10).

## What Changes

- **Repo hygiene**: untrack run outputs and agent-tool state (`output/`,
  `dynamic_results/`, `model-validation-output/`, `stress_results/`, `tmp/`,
  `.codex/`, `.claude/`, `.gemini/`, `.playwright-mcp/`, root screenshots/HTML);
  extend `.gitignore`. (`openspec/` stays tracked — it is the project's
  spec-driven workflow, not disposable tool state.)
- **Packaging**: replace `setup.py` with `pyproject.toml`; audit and declare
  real runtime dependencies; declare the two holiday CSVs
  (`china.csv`, `china_sse.csv`) as package data.
- **BREAKING — Namespace move**: relocate the 12 flat top-level packages under
  a single top-level package `quantark/` (`quantark.asset`, `quantark.util`, …)
  and rewrite all internal imports to the canonical `quantark.*` form.
- **Venv relocation (prerequisite)**: the repo's pre-configured virtualenv
  currently lives at `quantark/` — exactly the target package root. It moves to
  `.venv/` (untracked); `CLAUDE.md`/docs references update accordingly.
- **Legacy import compatibility shim**: a meta-path finder (registered via a
  `.pth` file installed with the package) intercepts imports of the 12 old flat
  names and aliases them to the *same module objects* as `quantark.*`,
  emitting a `DeprecationWarning` once per top-level name. This keeps
  open-otc-trading's ~40 flat imports working unchanged, with module identity
  preserved (enum members and `isinstance` checks must not split identities).
- **Holiday CSV loading**: resolve holiday files via package resources
  (`importlib.resources`) instead of repo-relative paths, so installed (non-
  editable) copies work.
- **Test migration**: migrate `test/` (~131 files) imports to `quantark.*` so
  the suite validates the canonical path while the shim covers stragglers.

## Capabilities

### New Capabilities
- `package-distribution`: quant-ark installs as a standard pip package
  (`pyproject.toml`, declared runtime dependencies, bundled package data);
  a fresh-venv install can import and use the library without the repo checkout.
- `quantark-namespace`: a single top-level `quantark` package contains all
  library code as subpackages; internal code imports only canonical
  `quantark.*` names.
- `legacy-import-compat`: the 12 historical flat top-level names remain
  importable and resolve to the identical module objects as their `quantark.*`
  counterparts, with deprecation warnings.

### Modified Capabilities
- `calendar-holiday-files`: holiday CSV resolution changes from the
  repo-relative path `util/calendar/holidayfile/` to package-resource lookup
  inside `quantark.util` (CSVs ship as package data; name-matching rules are
  unchanged).

## Impact

- **All 12 top-level packages** move; every internal import line is rewritten
  (mechanical, large diff). `test/` imports migrate; `example/` (50+ scripts)
  keeps working via the shim and can migrate later.
- **`setup.py` removed**, `pyproject.toml` added; `.gitignore` extended;
  large numbers of tracked run-output files untracked (history retained).
- **Developer workflow**: venv path changes `quantark/` → `.venv/`;
  `CLAUDE.md`, `AGENTS.md`, `README.md` command examples update.
- **Consumer (open-otc-trading)**: must keep passing *unchanged* against an
  editable install of the migrated library (acceptance gate). Its own
  migration to `quantark.*` imports, `importlib.resources` CSV access, and
  dependency declaration is Phase D in the consumer repo — out of scope here.
  Its editable install points at this checkout, so implementation happens in a
  git worktree to avoid breaking the running app mid-flight.
- **Private API constraint**: `RFQService._normalize_request` /
  `_evaluate_candidate` are called by the consumer — not renamed in this
  change.
- **Out of scope**: hosting decision (GitHub vs. Gitee), history scrub for
  public release, upstreaming the consumer's sharkfin registry registrations,
  shim removal (Phase E).
