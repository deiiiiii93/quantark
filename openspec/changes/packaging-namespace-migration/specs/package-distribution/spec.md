## ADDED Requirements

### Requirement: Standard package metadata
The project SHALL declare its build and distribution metadata in
`pyproject.toml` (PEP 621) with the setuptools build backend, and the legacy
`setup.py` SHALL be removed. The distribution name SHALL be `quantark`.

#### Scenario: Build from source
- **WHEN** `python -m build` (or `pip install .`) runs against the repository
- **THEN** a `quantark` distribution builds successfully using only
  `pyproject.toml` metadata

### Requirement: Fresh-environment installability
Installing the distribution into a fresh virtual environment SHALL yield a
working library without the repository checkout, manual path injection, or
undeclared dependencies.

#### Scenario: Fresh venv smoke test
- **WHEN** `pip install <quantark source or wheel>` runs in a freshly created
  venv and `python -c "from quantark.backtest.otc import
  AutocallableBacktestDashboard"` is executed
- **THEN** the import succeeds with no `ModuleNotFoundError` for either
  quantark modules or third-party dependencies

### Requirement: Declared runtime dependencies
The project SHALL declare, as runtime dependencies in `pyproject.toml`, every
third-party package imported by quantark library code at module-import time.
Third-party packages imported lazily inside optional features MAY be declared
as extras instead, and their absence SHALL produce a clear error naming the
missing dependency and extra.

#### Scenario: No undeclared hard dependency
- **WHEN** all `quantark.*` modules are imported in a fresh venv containing
  only the declared runtime dependencies
- **THEN** no import fails due to a missing third-party package

#### Scenario: Missing optional extra is diagnosable
- **WHEN** an optional feature is used in an environment without its extra
  installed
- **THEN** the raised error names the missing package and the extra that
  provides it

### Requirement: Single top-level package in site-packages
The distribution SHALL install exactly one top-level import package,
`quantark` (plus the compatibility `.pth` hook); the 12 legacy flat packages
SHALL NOT be installed as separate top-level directories.

#### Scenario: Wheel contents
- **WHEN** the built wheel's contents are listed
- **THEN** all Python modules live under `quantark/`, and no top-level
  `asset/`, `backtest/`, `cashleg/`, `dynamicscenario/`, `param/`,
  `portfolio/`, `priceenv/`, `rfq/`, `simm/`, `stresstest/`, `util/`, or
  `var/` directory is present

### Requirement: Bundled package data
The built distribution SHALL include the non-Python data files required at
runtime: the holiday CSVs `calendar/holidayfile/china.csv` and
`calendar/holidayfile/china_sse.csv` under `quantark/util/`.

#### Scenario: Package data present in installed copy
- **WHEN** the distribution is installed non-editably into a fresh venv
- **THEN** `importlib.resources.files("quantark.util") / "calendar" /
  "holidayfile" / "china_sse.csv"` refers to an existing readable resource
