# QuantArk Public Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish QuantArk as an open-source library: curated fresh-history repo at `github.com/deiiiiii93/quantark` with CI, then `quantark==0.1.0` on PyPI via trusted publishing.

**Architecture:** All preparation commits land on the current private `main` first (they improve the private repo too). The public tree is then created as an **orphan branch** in the same checkout — `git checkout --orphan` plus an allowlist `git add` — which never touches the working directory, so the consumer's editable install at `/Users/fuxinyao/quant-ark` keeps working throughout. The orphan branch becomes `main` on GitHub; the old history stays on the `gitee` remote and a local `private-main` branch.

**Tech Stack:** git orphan branches, `gh` CLI (already authenticated as `deiiiiii93`), GitHub Actions, hatchling (PEP 621/639), PyPI Trusted Publishing (OIDC).

**Spec:** `docs/superpowers/specs/2026-06-10-public-distribution-design.md`

**Verified preconditions (2026-06-10):**
- `gh auth status` → logged in as `deiiiiii93`, scopes include `repo`, `workflow`.
- PyPI names `quantark` and `quant-ark` both return 404 (unclaimed).
- `test/test_legacy_import_compat.py` exists (10 tests) — shim coverage survives the examples flip.
- 55 example files use `from <flatname> import …` (no plain `import <flatname>` form); 51 manipulate `sys.path`.
- 61 `.md` files tracked inside `quantark/` (module CLAUDE/AGENTS + engine notes) — all excluded from the public tree and wheel.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `example/*.py` | Modify (55 files) | Canonical `quantark.*` imports, no `sys.path` hacks |
| `CLAUDE.md`, `AGENTS.md` (root, private) | Modify | Update the "examples keep flat imports" note |
| `LICENSE` | Create | Apache-2.0 full text |
| `NOTICE` | Create | Copyright line |
| `CHANGELOG.md` | Create | Keep-a-Changelog, 0.1.0 entry |
| `pyproject.toml` | Modify | PEP 639 license, URLs, classifiers, authors, `.md` excludes |
| `README.md` | Modify | Badges, PyPI install, remove `docs/` reference, disclaimer |
| `.gitignore` | Replace | Lean public version |
| `.github/workflows/tests.yml` | Create | Test matrix + packaging smoke |
| `.github/workflows/release.yml` | Create | Tag-triggered PyPI trusted publishing |
| `.git/info/exclude` | Modify (local only) | Hide private leftovers from `git status` on the public branch |

---

### Task 1: Flip examples to canonical imports and drop sys.path hacks

**Files:**
- Modify: `example/*.py` (55 files with flat imports; 51 with sys.path)
- Modify: `CLAUDE.md` (root), `AGENTS.md` (root)

- [ ] **Step 1: Confirm the starting state (acts as the failing test)**

```bash
grep -lE "^from (asset|backtest|cashleg|dynamicscenario|param|portfolio|priceenv|rfq|simm|stresstest|util|var)\b" example/*.py | wc -l
grep -l "sys.path" example/*.py | wc -l
```

Expected: `55` and `51` (or close — must be nonzero).

- [ ] **Step 2: Rewrite flat imports to canonical**

```bash
perl -pi -e 's/^from (asset|backtest|cashleg|dynamicscenario|param|portfolio|priceenv|rfq|simm|stresstest|util|var)([.\s])/from quantark.$1$2/' example/*.py
```

- [ ] **Step 3: Strip sys.path manipulation**

```bash
.venv/bin/python - <<'EOF'
import re
from pathlib import Path

path_comment = re.compile(r"^\s*#\s*Add .*path", re.IGNORECASE)
if_guard = re.compile(r"^\s*if str\(ROOT(_DIR)?\) not in sys\.path:")
path_line = re.compile(r"sys\.path\.(insert|append)")
root_assign = re.compile(r"^(ROOT|ROOT_DIR)\s*=.*(__file__|dirname)")

for f in sorted(Path("example").glob("*.py")):
    text = f.read_text()
    lines = [
        l for l in text.splitlines(keepends=True)
        if not (path_comment.match(l) or if_guard.match(l) or path_line.search(l))
    ]
    # drop ROOT/ROOT_DIR assignment only if the name is now unused
    out = []
    for l in lines:
        m = root_assign.match(l)
        if m:
            name = m.group(1)
            others = "".join(x for x in lines if x is not l)
            if not re.search(rf"\b{name}\b", others):
                continue
        out.append(l)
    f.write_text("".join(out))
print("done")
EOF
```

- [ ] **Step 4: Verify no flat imports or path hacks remain**

```bash
grep -nE "^from (asset|backtest|cashleg|dynamicscenario|param|portfolio|priceenv|rfq|simm|stresstest|util|var)\b" example/*.py || echo "imports clean"
grep -n "sys.path" example/*.py || echo "sys.path clean"
```

Expected: `imports clean` and `sys.path clean`. If any `sys.path` line survives (an unanticipated pattern), remove it by hand with Edit and re-run.

- [ ] **Step 5: Run representative examples (the passing test)**

```bash
.venv/bin/python example/european_option_demo.py
.venv/bin/python example/snowball_mc_demo.py
.venv/bin/python example/parametric_var_demo.py
.venv/bin/python example/fixed_bond_demo.py
.venv/bin/python example/irs_demo.py
```

Expected: each exits 0 and prints pricing output. If one fails with `NameError` on `ROOT`/`Path`, the cleanup over-removed a still-used assignment — restore that one line by hand.

- [ ] **Step 6: Update the stale shim note in CLAUDE.md and AGENTS.md**

In root `CLAUDE.md`, replace the sentence

```
`example/` scripts intentionally keep flat imports as a live exerciser of the shim.
```

with

```
`example/` scripts use canonical `quantark.*` imports; the shim is exercised by `test/test_legacy_import_compat.py`.
```

Run `grep -n "flat imports" AGENTS.md` and apply the same replacement to any matching sentence there.

- [ ] **Step 7: Run the shim regression test (proves coverage survived)**

```bash
.venv/bin/python -m pytest test/test_legacy_import_compat.py -v
```

Expected: 10 passed (some may skip if quantark isn't pip-installed in the venv — editable install means none should skip).

- [ ] **Step 8: Commit**

```bash
git add example CLAUDE.md AGENTS.md
git commit -m "refactor(example): canonical quantark.* imports, drop sys.path hacks"
```

---

### Task 2: LICENSE, NOTICE, CHANGELOG

**Files:**
- Create: `LICENSE`, `NOTICE`, `CHANGELOG.md`

- [ ] **Step 1: Fetch the Apache-2.0 text verbatim**

```bash
curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
head -2 LICENSE
wc -l LICENSE
```

Expected: first lines contain "Apache License" / "Version 2.0, January 2004"; ~202 lines.

- [ ] **Step 2: Create NOTICE**

```
QuantArk
Copyright 2026 fuxinyao (https://github.com/deiiiiii93)
```

- [ ] **Step 3: Create CHANGELOG.md**

```markdown
# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

## [0.1.0] - 2026-06-10

### Added
- First public release.
- Equity derivatives: European/American/Asian vanilla options, barrier,
  one-touch, digital, sharkfin, and autocallable products (snowball,
  phoenix, KO-reset snowball, range accrual) with analytical, Monte
  Carlo, PDE, quadrature, and tree engines.
- Fixed income: fixed bonds, FRNs, bond options, bond forwards/futures,
  convertible bonds, interest rate swaps.
- Market data layer (`quantark.param`, `quantark.priceenv`), Greeks
  calculators, portfolio VaR (parametric/historical/Monte Carlo),
  ISDA SIMM v2.6, stress testing, multi-day scenario simulation, and a
  hedging backtest framework.
- Legacy flat-import compatibility shim (`asset`, `util`, …) with
  `DeprecationWarning`; slated for removal in 1.0.
```

(Update the date in the `[0.1.0]` heading if Task 13 happens on a later day.)

- [ ] **Step 4: Commit**

```bash
git add LICENSE NOTICE CHANGELOG.md
git commit -m "chore: add Apache-2.0 LICENSE, NOTICE, CHANGELOG"
```

---

### Task 3: pyproject metadata polish and wheel hygiene

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace `pyproject.toml` content**

Keep the dependency list exactly as it is today (slimming is Stage 3). Full new content:

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "quantark"
version = "0.1.0"
description = "Modular derivatives pricing and risk library: options, autocallables, bonds, VaR, SIMM"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
authors = [{ name = "fuxinyao", email = "ianchris0113@gmail.com" }]
keywords = [
    "quantitative-finance", "derivatives", "option-pricing", "monte-carlo",
    "pde", "autocallable", "snowball", "var", "risk-management", "simm",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Developers",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Office/Business :: Financial :: Investment",
]
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
    "kaleido>=0.2.1",
    "pyarrow>=12.0.0",
    "openpyxl>=3.0.0",
    "pyyaml>=6.0.0",
    "python-docx>=1.0.0",
    "python-dateutil>=2.8.0",
]

[project.urls]
Homepage = "https://github.com/deiiiiii93/quantark"
Repository = "https://github.com/deiiiiii93/quantark"
Issues = "https://github.com/deiiiiii93/quantark/issues"
Changelog = "https://github.com/deiiiiii93/quantark/blob/main/CHANGELOG.md"

[tool.hatch.build.targets.wheel]
packages = ["quantark"]
exclude = ["quantark/**/*.md"]

# Registers the legacy flat-import shim (quantark/_compat.py) at interpreter
# startup. Hatchling applies force-include to editable wheels too, so the
# shim is active for both `pip install` and `pip install -e`.
[tool.hatch.build.targets.wheel.force-include]
"quantark_compat.pth" = "quantark_compat.pth"

[tool.hatch.build.targets.sdist]
include = ["quantark", "quantark_compat.pth", "README.md", "LICENSE", "NOTICE", "CHANGELOG.md"]
exclude = ["quantark/**/*.md"]
```

Notes: PEP 639 SPDX `license` string requires hatchling ≥ 1.27 (hence the build-system pin). No `License ::` classifier — PEP 639 deprecates mixing them.

- [ ] **Step 2: Build and check**

```bash
.venv/bin/pip install build twine
rm -rf dist && .venv/bin/python -m build
.venv/bin/twine check dist/*
```

Expected: sdist + wheel built; `twine check` reports PASSED for both.

- [ ] **Step 3: Verify wheel contents**

```bash
unzip -l dist/quantark-0.1.0-py3-none-any.whl | grep -c "\.md" || echo "no .md in wheel"
unzip -l dist/quantark-0.1.0-py3-none-any.whl | grep -E "china(_sse)?\.csv|quantark_compat\.pth"
unzip -l dist/quantark-0.1.0-py3-none-any.whl | grep -iE "licen|notice"
```

Expected: `no .md in wheel` (or `0`); both holiday CSVs and the `.pth` present; LICENSE and NOTICE in `*.dist-info/licenses/`.

- [ ] **Step 4: Editable reinstall still works (consumer safety)**

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -c "import quantark; import asset" 2>&1 | head -3
```

Expected: imports succeed (the `asset` line may print a DeprecationWarning — that is correct).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(packaging): Apache-2.0 metadata, project URLs, exclude in-package .md"
```

---

### Task 4: README polish

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add badges directly under the title**

Replace the first line block:

```markdown
# QuantArk - Professional Financial Derivatives Pricing Library
```

with:

```markdown
# QuantArk — Financial Derivatives Pricing & Risk Library

[![tests](https://github.com/deiiiiii93/quantark/actions/workflows/tests.yml/badge.svg)](https://github.com/deiiiiii93/quantark/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/quantark)](https://pypi.org/project/quantark/)
[![Python](https://img.shields.io/pypi/pyversions/quantark)](https://pypi.org/project/quantark/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
```

- [ ] **Step 2: Rewrite the Installation section**

Replace the existing Installation section body (the paragraph and code block starting "QuantArk is a standard pip-installable package…") with:

````markdown
## Installation

```bash
pip install quantark
```

From source / latest development version:

```bash
pip install git+https://github.com/deiiiiii93/quantark
```

For development (editable install with test tooling):

```bash
git clone https://github.com/deiiiiii93/quantark && cd quantark
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```
````

Keep the existing "Migration note" paragraph about the flat-import shim — it is accurate and useful.

- [ ] **Step 3: Remove the private-docs reference**

Line ~172 reads: ``See `docs/engine_param_guide.md` for the preset decision table, config schema, and examples.`` Replace that sentence with:

```markdown
Engine parameter presets accept either preset names or explicit config objects; see the docstrings in `quantark/asset/equity/param/` for the full schema.
```

Then verify nothing else points at the private docs tree:

```bash
grep -n "docs/" README.md || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Append Disclaimer and License sections at the end of the file**

```markdown
## Disclaimer

QuantArk is provided for research and educational purposes. It is not
investment advice, and no warranty is made as to the correctness of any
price, risk figure, or model output. Validate independently before any
production or trading use. See the LICENSE file for the full terms.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): badges, PyPI install, disclaimer, drop private docs reference"
```

---

### Task 5: Public .gitignore

**Files:**
- Replace: `.gitignore`

- [ ] **Step 1: Replace `.gitignore` with the lean public version**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
build/
dist/
.pytest_cache/
.coverage
htmlcov/
.ipynb_checkpoints/
.DS_Store
```

(Private-only clutter — `docs/`, `openspec/`, agent dirs, run outputs — is deliberately NOT listed here; it gets hidden via the local-only `.git/info/exclude` in Task 10 so the public file stays generic.)

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: lean public .gitignore"
```

---

### Task 6: CI workflows

**Files:**
- Create: `.github/workflows/tests.yml`
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create `.github/workflows/tests.yml`**

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: pip install -e ".[dev]"
      - name: Run tests
        run: |
          if [ "${{ github.event_name }}" = "pull_request" ]; then
            python -m pytest -m "not slow"
          else
            python -m pytest
          fi

  package-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build
        run: |
          pip install build twine
          python -m build
          twine check dist/*
      - name: Install wheel in clean venv and smoke test
        run: |
          python -m venv /tmp/smoke
          /tmp/smoke/bin/pip install dist/*.whl
          cd /tmp
          /tmp/smoke/bin/python - <<'EOF'
          from quantark.backtest.otc import AutocallableBacktestDashboard
          from quantark.asset.equity.product.option import EuropeanVanillaOption
          from quantark.asset.equity.engine.analytical import BlackScholesEngine
          from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
          from quantark.priceenv import PricingEnvironment
          from quantark.util.enum import OptionType

          env = PricingEnvironment(
              spot_quote=SpotQuote(spot=100.0),
              vol_surface=FlatVolSurface(volatility=0.20),
              rate_curve=FlatRateCurve(rate=0.05),
              div_yield=ContinuousDividendYield(div_yield=0.02),
          )
          opt = EuropeanVanillaOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
          price = BlackScholesEngine().price(opt, env)
          assert 8.0 < price < 11.0, f"BS price out of range: {price}"
          print(f"smoke OK: {price:.6f}")
          EOF
```

(The `cd /tmp` matters: it proves the wheel works without the repo on `sys.path`. The price assertion brackets the known BS value ≈ 9.2 for S=K=100, T=1, σ=0.20, r=0.05, q=0.02.)

- [ ] **Step 2: Create `.github/workflows/release.yml`**

```yaml
name: release

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build distributions
        run: |
          pip install build twine
          python -m build
          twine check dist/*
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Publish to PyPI (trusted publishing)
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 3: Validate YAML syntax locally**

```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/tests.yml')); yaml.safe_load(open('.github/workflows/release.yml')); print('YAML OK')"
```

Expected: `YAML OK`.

- [ ] **Step 4: Commit**

```bash
git add .github
git commit -m "ci: test matrix, packaging smoke, tag-triggered PyPI release"
```

---

### Task 7: Full local gate

**Files:** none (verification only)

- [ ] **Step 1: Run the whole suite**

```bash
.venv/bin/python -m pytest
```

Expected: all tests pass (slow tests included — this is the pre-publication gate; expect several minutes). Any failure blocks the launch: debug it before proceeding, do not skip.

- [ ] **Step 2: Fresh-venv wheel smoke (mirror of the CI job)**

```bash
rm -rf dist && .venv/bin/python -m build
python3 -m venv /tmp/qa-smoke
/tmp/qa-smoke/bin/pip install dist/quantark-0.1.0-py3-none-any.whl
cd /tmp   # run from outside the repo so the wheel, not the source tree, is imported
/tmp/qa-smoke/bin/python -c "
from quantark.backtest.otc import AutocallableBacktestDashboard
from quantark.asset.equity.engine.analytical import BlackScholesEngine
print('wheel smoke OK')
"
cd /Users/fuxinyao/quant-ark
```

Expected: `wheel smoke OK`.

---

### Task 8: Leak sweep over the public allowlist

**Files:** any file with a finding

- [ ] **Step 1: Sweep the exact paths that will go public**

```bash
grep -rnE "fuxinyao|/Users/|gitee|open-otc|ianchris" \
  .github .gitignore CHANGELOG.md NOTICE README.md pyproject.toml \
  pytest.ini quantark_compat.pth quantark test example 2>/dev/null \
  | grep -vE "^quantark/.*\.md:" | sort
```

(The filter drops only the in-package `.md` notes, which are not exported; README.md and CHANGELOG.md remain in scope.)

- [ ] **Step 2: Triage hits**

Allowed (intentional) hits — exactly these and nothing more:
- `NOTICE`: the copyright line (`fuxinyao`).
- `pyproject.toml`: the `authors` entry (`fuxinyao`, `ianchris0113@gmail.com`).

Anything else (a hardcoded `/Users/fuxinyao/...` path in library code, a gitee URL, an open-otc-trading reference in a test) must be fixed: replace absolute paths with `Path(__file__)`-relative logic or temp dirs, and remove consumer references. The in-package `.md` files are not exported, so hits inside them (filtered above) can be ignored.

- [ ] **Step 3: Re-run the sweep until only the allowed hits remain, then commit any fixes**

```bash
git add -A && git commit -m "chore: remove internal paths/references from public surface" || echo "nothing to fix"
```

---

### Task 9: Archive the private history to Gitee

**Files:** none

- [ ] **Step 1: Push current main (full private history, all prep commits) to Gitee**

```bash
git push origin main
```

Expected: succeeds. Gitee is now the complete private archive including every prep commit.

---

### Task 10: Build the orphan public branch

**Files:** git only — the working tree is never modified in this task

- [ ] **Step 1: Confirm clean state, then create the orphan branch**

```bash
git status --porcelain | head   # expect empty
git checkout --orphan public-main
git rm -r --cached -q .
```

(`--orphan` + `rm --cached` empties the index without touching a single file on disk — the consumer's editable install is unaffected.)

- [ ] **Step 2: Stage the allowlist**

```bash
git add .github .gitignore CHANGELOG.md LICENSE NOTICE README.md \
        pyproject.toml pytest.ini quantark_compat.pth quantark test example
git ls-files | grep -E '^quantark/.*\.md$' | tr '\n' '\0' | xargs -0 git rm --cached -q
```

- [ ] **Step 3: Review the manifest (the test for this task)**

```bash
git ls-files | sort > /tmp/public-manifest.txt
wc -l /tmp/public-manifest.txt
grep -E "CLAUDE|AGENTS|GEMINI|QODER|^docs/|^openspec/|^\.claude|^\.codex|^\.cursor|^\.serena|^\.qoder|^conductor|^commands|\.docx|\.pdf|\.png" /tmp/public-manifest.txt || echo "manifest clean"
git ls-files example/ | grep -v '\.py$'
```

Expected: ~600 files (sanity range 580–620); `manifest clean`; the example non-`.py` survivors are exactly `engine_params_pde.json`, `engine_params_quad.yaml`, `templates/snowball_rfq_ko_rate_dashboard.html` — open the HTML template and skim for personal strings (`grep -nE "fuxinyao|/Users/" example/templates/*.html` → expect nothing).

If anything unexpected appears in the manifest, `git rm --cached` it and re-check before committing.

- [ ] **Step 4: Create the initial public commit**

```bash
git commit -m "QuantArk 0.1.0 — initial public release

Modular derivatives pricing and risk library: vanilla and exotic equity
options, autocallables (snowball, phoenix), fixed income, portfolio VaR,
ISDA SIMM, stress testing, and hedging backtests."
```

---

### Task 11: Create the GitHub repo and cut over

**Files:** git remotes/branches; `.git/info/exclude` (local only)

- [ ] **Step 1: Create the public repo and push**

```bash
gh repo create deiiiiii93/quantark --public \
  --description "Modular derivatives pricing & risk library: options, autocallables, bonds, VaR, SIMM"
git remote rename origin gitee
git remote add origin https://github.com/deiiiiii93/quantark.git
git push -u origin public-main:main
```

Expected: repo created; push succeeds; GitHub default branch is `main` (first pushed branch).

- [ ] **Step 2: Rearrange local branches — GitHub is now primary**

```bash
git branch -m main private-main
git branch -m public-main main
git branch --set-upstream-to=origin/main main
git branch -vv
```

Expected: `main` tracks `origin/main` (GitHub); `private-main` still tracks gitee.

- [ ] **Step 3: Hide private leftovers from git status (local-only excludes)**

Append to `.git/info/exclude` (NOT the public `.gitignore`):

```gitignore
# private working files not part of the public tree
docs/
openspec/
.claude/
.codex/
.cursor/
.serena/
.qoder/
.gemini/
.playwright-mcp/
conductor/
commands/
external/
debug/
tmp/
output/
dynamic_results/
stress_results/
model-validation-output/
risk_metric_analysis/
reports/
plots/
logs/
portfolio/
stress_scenarios/
CLAUDE.md
AGENTS.md
GEMINI.md
QODER.md
PROJECT_INDEX.md
QUAD_PDE_IMPROVEMENTS.md
requirements.txt
package.json
package-lock.json
risk-report-wizard.skill
/*.png
/*.html
```

Then verify:

```bash
git status --porcelain | head
```

Expected: empty (or only files you are actively editing).

- [ ] **Step 4: Protect main against force-push/deletion**

```bash
gh api -X PUT repos/deiiiiii93/quantark/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  --input - <<'EOF'
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

(Deliberately light for a solo maintainer: blocks history rewrites without forcing every change through a PR. Tighten later when contributors arrive.)

---

### Task 12: Verify the public launch

**Files:** none (verification only)

- [ ] **Step 1: CI green on GitHub**

```bash
gh run list --repo deiiiiii93/quantark --limit 5
gh run watch --repo deiiiiii93/quantark
```

Expected: the `tests` workflow triggered by the push completes successfully (all 4 Python versions + package-smoke). If a matrix leg fails on a Python version never tested locally (e.g., 3.13), debug it: it is a real compatibility bug, fix on `main` via a normal commit.

- [ ] **Step 2: Fresh-venv install straight from GitHub**

```bash
python3 -m venv /tmp/gh-smoke
/tmp/gh-smoke/bin/pip install "git+https://github.com/deiiiiii93/quantark"
cd /tmp && /tmp/gh-smoke/bin/python -c "
from quantark.backtest.otc import AutocallableBacktestDashboard
print('github install OK')
" && cd /Users/fuxinyao/quant-ark
```

Expected: `github install OK`.

- [ ] **Step 3: Consumer regression — open-otc-trading still works**

In your open-otc-trading checkout, run its test suite the way you normally do. Its venv holds an editable install pointing at `/Users/fuxinyao/quant-ark`, whose path, package layout, and working tree this plan never altered — expect a full pass. If anything fails here, stop and investigate before the PyPI release.

---

### Task 13: PyPI trusted publishing and the 0.1.0 release

**Files:** possibly `CHANGELOG.md` (date)

- [ ] **Step 1: Create the `pypi` environment on the GitHub repo**

```bash
gh api -X PUT repos/deiiiiii93/quantark/environments/pypi
```

- [ ] **Step 2 (MANUAL — maintainer): Register the pending trusted publisher on PyPI**

On https://pypi.org (logged in to your PyPI account; create one if needed, with 2FA):
*Your account → Publishing → "Add a new pending publisher"* with exactly:
- PyPI project name: `quantark`
- Owner: `deiiiiii93`
- Repository name: `quantark`
- Workflow name: `release.yml`
- Environment name: `pypi`

This cannot be scripted — PyPI has no API for it. Say "done" when complete.

- [ ] **Step 2b (optional but per spec — recommended): TestPyPI dry run**

If you want the full dry run, repeat Step 2 on https://test.pypi.org (separate account) and add a parallel publish job pointed at TestPyPI, or trigger a one-off manual upload:

```bash
rm -rf dist && .venv/bin/python -m build
.venv/bin/twine upload --repository testpypi dist/*   # prompts for a TestPyPI API token
python3 -m venv /tmp/testpypi-smoke
/tmp/testpypi-smoke/bin/pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ quantark==0.1.0
/tmp/testpypi-smoke/bin/python -c "import quantark; print('testpypi OK')"
```

Expected: `testpypi OK`. If you skip this, the `twine check` + wheel-install smoke in CI covers the same packaging risks except the index round-trip itself.

- [ ] **Step 3: Confirm CHANGELOG date matches today, then tag and push**

```bash
grep -n "0.1.0" CHANGELOG.md   # if the date is stale, edit + commit first
git tag v0.1.0
git push origin v0.1.0
gh run watch --repo deiiiiii93/quantark
```

Expected: the `release` workflow runs; `build` then `publish` succeed; PyPI shows `quantark 0.1.0`. A trusted-publishing error here means the Step 2 registration fields don't match — fix on PyPI and re-run the workflow (`gh run rerun <id>`).

- [ ] **Step 4: The final acceptance check**

```bash
python3 -m venv /tmp/pypi-smoke
/tmp/pypi-smoke/bin/pip install quantark==0.1.0
cd /tmp && /tmp/pypi-smoke/bin/python -c "
from quantark.backtest.otc import AutocallableBacktestDashboard
from quantark.asset.equity.engine.analytical import BlackScholesEngine
print('pip install quantark works')
" && cd /Users/fuxinyao/quant-ark
```

Expected: `pip install quantark works`. QuantArk is publicly distributed.

- [ ] **Step 5: Sync the private archive one last time**

```bash
git push gitee main:public-main-snapshot
```

(Pushes the public history to Gitee under a separate branch name so the private archive also records exactly what was published.)

---

## Out of scope (Stage 3 — separate plans later)

Dependency slimming into `[report]`/`[viz]` extras, docs site, CONTRIBUTING.md and issue templates, shim removal at 1.0, upstreaming the consumer's sharkfin registry entries.
