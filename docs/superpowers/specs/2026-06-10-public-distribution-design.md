# Public Distribution of QuantArk — Design

**Date:** 2026-06-10
**Status:** Approved (design); implementation plan to follow
**Decisions made by maintainer:** open-source library as the goal · Apache-2.0
license · fresh public git history · GitHub becomes the primary development
home · staged launch (Approach A)

## Goal

Turn quant-ark into a real open-source library: publicly visible on GitHub,
installable from PyPI as `quantark`, with CI, releases, and a credible
public face — while open-otc-trading (the existing private consumer) keeps
working at every step.

## Context (verified 2026-06-10)

- Packaging migration is complete and merged: PEP 621 `pyproject.toml`
  (hatchling), single top-level `quantark` package, flat-import compat shim
  via `quantark_compat.pth`, version 0.1.0.
- Remote is a personal Gitee repo; no LICENSE file exists anywhere in the
  tree or history.
- The names `quantark` and `quant-ark` are both unclaimed on PyPI
  (checked 2026-06-10).
- Tracked tree still contains non-library content: `openspec/` (289 files),
  agent-tool state (`.codex/`, `.claude/`, `.cursor/`, `.serena/`,
  `.qoder/`, `conductor/`, `commands/`), and `docs/` binaries including a
  structured-product regulatory filing (`敲出重置雪球期权结构报备.docx`)
  that must not be published.

## Non-goals

- No code/feature changes to pricing logic as part of the launch.
- No scrubbing or publishing of the existing git history — it stays private
  on Gitee permanently.
- Dependency slimming and the docs site are roadmap items (Stage 3), not
  launch blockers.

## Design

### 1. Public tree curation

The public repo starts from a curated export of current `main`, committed
as a single clean initial commit. The export is allowlist-based (copy what
is named below), not denylist-based, so nothing slips through by omission.

**Included:**

- `quantark/` — the entire library package, including `rfq` (legitimate
  library code) and the in-package holiday CSVs.
- `test/` (full suite), `example/` (all demos).
- `pyproject.toml`, `pytest.ini`, `quantark_compat.pth`, `README.md`.
- New files: `LICENSE` (Apache-2.0 text), `NOTICE` (copyright line),
  `CHANGELOG.md`, a fresh lean `.gitignore`.
- `docs/` — **markdown files only**: theory docs (English and Chinese `.md`
  both kept), engine parameter guide, implementation notes that read as
  library documentation.

**Excluded:**

- `openspec/` and all agent-tool state: `.codex/`, `.claude/`, `.cursor/`,
  `.serena/`, `.qoder/`, `conductor/`, `commands/`.
- `external/`, `debug/`, `tmp/`, `dist/`, and all run outputs
  (`output/`, `dynamic_results/`, `stress_results/`,
  `model-validation-output/`, `risk_metric_analysis/`, `reports/`,
  `plots/`, `logs/`, `portfolio/`, `stress_scenarios/`).
- Root one-offs: screenshots, `model-orchestrator-explainer.html`,
  `PROJECT_INDEX.md`, `QODER.md`, `GEMINI.md`, `QUAD_PDE_IMPROVEMENTS.md`,
  `package.json`, `package-lock.json`, `risk-report-wizard.skill`,
  `requirements.txt` (redundant with `pyproject.toml`).
- `docs/` binaries: all `.docx`/`.pdf`, including the regulatory filing.
- `CLAUDE.md` and `AGENTS.md` (root and module-level) — they reference the
  internal OpenSpec workflow, personal paths, and the private consumer. A
  public-appropriate `AGENTS.md` may be added later as a separate task.

**Curation-driven code changes:**

- `example/` scripts flip from flat imports to canonical `quantark.*`
  imports — examples are what adopters copy-paste. The shim coverage the
  examples provided moves into a dedicated regression test
  (e.g., `test/test_compat_shim.py`) that imports a representative set of
  flat names and asserts module identity with the `quantark.*` modules.
- Pre-push sweep of the curated tree: grep for personal paths
  (`/Users/fuxinyao`), email addresses, internal hostnames, and consumer
  references; fix or remove hits.

### 2. Stage 1 — GitHub launch

- Create public repo `github.com/<owner>/quantark` (name matches the
  import and PyPI name). Push the curated tree as the initial commit.
- Gitee repo is frozen as the private archive. The local
  `/Users/fuxinyao/quant-ark` checkout re-points `origin` to GitHub; the
  directory path and package layout are unchanged, so open-otc-trading's
  editable install keeps working untouched.
- `pyproject.toml` metadata polish: `license = "Apache-2.0"`, project URLs
  (homepage, repository, issues), trove classifiers (Python versions,
  Intended Audience :: Financial and Insurance Industry, Topic :: Office/
  Business :: Financial), keywords, real author name + email.
- README polish: badges (CI, PyPI version, Python versions, license),
  quickstart that works from a bare `pip install quantark`, supported
  products/engines matrix, and a standard disclaimer (research/educational
  use, not investment advice, no warranty).
- CI via GitHub Actions, two workflows:
  1. **tests** — pytest matrix on Python 3.10–3.13 (ubuntu-latest),
     `pip install -e ".[dev]"`, run with `-m "not slow"` on PRs; full suite
     on main.
  2. **packaging smoke** — build sdist + wheel, `twine check`, install the
     wheel into a fresh venv, run the import smoke test
     (`from quantark.backtest.otc import AutocallableBacktestDashboard`
     plus a vanilla pricing round-trip).
- Branch protection on `main`; changes land via PR.

### 3. Stage 2 — PyPI release

- Release workflow using **PyPI Trusted Publishing** (OIDC from GitHub
  Actions; no long-lived API tokens), triggered by `v*` tags.
- Dry-run against TestPyPI first; then publish `quantark==0.1.0` promptly
  to claim the name.
- Versioning: semver; 0.x signals the public API may still move.
  `CHANGELOG.md` in Keep-a-Changelog format, updated per release.

### 4. Stage 3 — adoption infrastructure and the road to 1.0

- **Dependency slimming (target 0.2):** core install narrows to
  numpy/scipy/pandas; heavy deps move to extras —
  `quantark[report]` (plotly, kaleido, openpyxl, python-docx, pyarrow),
  `quantark[viz]` (matplotlib, seaborn). Requires lazy imports in report/
  dashboard modules with a clear error message naming the extra to install.
- **Shim sunset:** the `.pth` flat-import shim ships through 0.x (existing
  `DeprecationWarning` stands); removed at 1.0 once open-otc-trading is
  fully migrated to `quantark.*` imports (its Phase D).
- Docs site: mkdocs-material on GitHub Pages, porting `docs/*.md`.
- Community files: `CONTRIBUTING.md`, issue templates; `CITATION.cff`
  optional.
- **1.0 criteria:** public API reviewed and frozen, dependencies slimmed,
  shim removed, docs site live.

## Error handling / failure modes

- **Leak prevention:** allowlist export + pre-push grep sweep; the private
  history never leaves Gitee, so a curation mistake at worst exposes one
  reviewable tree, not years of history.
- **Consumer breakage:** the only hard constraint. Stage 1 preserves the
  local path and layout exactly; Stages 2–3 only *add* installation
  channels. The shim is not removed until the consumer no longer needs it.
- **PyPI name squatting:** mitigated by publishing 0.1.0 promptly after
  the GitHub launch.

## Testing / acceptance

- Stage 1: CI green on the public repo (test matrix + packaging smoke);
  `pip install git+https://github.com/<owner>/quantark` works in a fresh
  venv; open-otc-trading suite still passes against the local checkout.
- Stage 2: `pip install quantark` from PyPI in a fresh venv passes the
  import smoke test.
- Curation: grep sweep returns no hits for personal/internal strings; a
  manual review of the file list before the initial push.
