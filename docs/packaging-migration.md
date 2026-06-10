# Packaging & Namespace Migration Brief

**Date:** 2026-06-10
**Origin:** handoff from an open-otc-trading session (analysis of the quant-ark
integration ahead of the "shareable MVP" milestone). This document carries the
findings from that session so work here can start warm.

## Goal

Make quant-ark a properly distributable Python package so that
open-otc-trading (the main consumer) can declare it as a normal dependency and
be cloned + run by someone who is not the author. Today that is impossible:
the consumer reaches quant-ark via a hardcoded `/Users/fuxinyao/quant-ark`
path, a manual `pip install -e` doc step, and runtime `sys.path` injection.

## Current state (verified 2026-06-10, main @ `85f379b`)

- `setup.py`: `name="quantark"`, version `0.1.0`, `find_packages()`,
  `install_requires = numpy, pandas, matplotlib, scipy`.
- `find_packages()` exposes **12 flat top-level packages**:
  `asset`, `backtest`, `cashleg`, `dynamicscenario`, `param`, `portfolio`,
  `priceenv`, `rfq`, `simm`, `stresstest`, `util`, `var`.
- In-package data files (tracked, non-`.py`): only
  `util/calendar/holidayfile/china.csv` and
  `util/calendar/holidayfile/china_sse.csv` (plus many `.md` docs inside
  `asset/`). The CSVs are load-bearing — see "Consumer contact surface".
- Repo hygiene: working tree is ~949 MB but the git pack is only 8.3 MB.
  Of 1,549 *tracked* files, large chunks are run outputs and agent-tool
  state, not library code: `openspec/` (281), `output/` (187),
  `dynamic_results/` (86), `model-validation-output/` (57),
  `stress_results/` (28), `tmp/` (14), `.codex/` (48), `.claude/` (45),
  `.gemini/` (14), `.playwright-mcp/` (15), plus screenshots and one-off
  HTML files at the root.
- Remote: `gitee.com/fuxinyao/quant-ark` (personal). Tests live in `test/`
  (~131 files, flat-name imports), `pytest.ini` defines a `slow` marker.

## Why the flat namespace must go

The 12 top-level names are installed directly into `site-packages`:

- `param` **collides with the HoloViz `param` package on PyPI** (a common
  transitive dependency of plotting stacks).
- `util`, `portfolio`, `var`, `backtest` are generic enough that collision
  with some other package in a shared venv is a matter of time — and Python
  resolves the collision by *silent shadowing*, not an error.

Target layout: one top-level package, `quantark/`, containing the 12 as
subpackages (`quantark.asset`, `quantark.util`, …).

## Consumer contact surface (open-otc-trading)

Everything below must keep working through the migration. Importing files:
`backend/app/services/{quantark,backtest_bridge,scenario_test_bridge,risk_engine,rfq}.py`,
`backend/app/services/domains/{backtest,scenario_catalog,scenario_test}.py`,
`scripts/check_delta_value.py`, `tests/test_scenario_catalog.py`.

Modules imported (key symbols):

| Module | Symbols used by the consumer |
|---|---|
| `util.calendar` | `CalendarType`, `DayCountConvention`, `create_calendar`, `calculate_year_fraction` |
| `util.enum` | `OptionType`, `BarrierType`, `DoubleBarrierType`, `BarrierDirection`, `TouchType`, `ObservationType`, `ObservationAggregation`, `ObservationFrequency`, `CouponPayType`, `ProtectionType`, `TenorEnd`, `DeltaOneType` |
| `util.enum.engine_enums` | `EngineType` |
| `rfq.models` | `RFQRequest`, `RFQEngineSpec`, `RFQInputMode`, `RFQTarget`, `RFQTargetLabel`, `RFQTermsheetInput`, `RFQUnknownSpec` |
| `rfq.builders` | `build_product_from_termsheet`, `build_engine_from_termsheet`, `build_pricing_env_from_market_kwargs` |
| `rfq.service` | `quote_rfq`, `RFQService` |
| `rfq.registry` | `ENGINE_BUILDERS`, `PRODUCT_BUILDERS`, `QuoteableFieldAdapter`, `register_unknown_adapter`, `resolve_unknown_adapter` |
| `asset.equity.product.option` | `ObservationRecord`, `ObservationSchedule`, all option product classes incl. sharkfins |
| `asset.equity.engine.analytical` / `.mc` | sharkfin engines |
| `asset.equity.riskmeasures.greeks_calculator` | `GreeksCalculator` |
| `stresstest` (+ `scenario.*`, `results.*`, `report`, `stress.stress_types`) | `StressTestEngine`, `StressTestConfig`, `ScenarioStorage`, `ScenarioLibrary`, `ScenarioBuilder`, `ResultAggregator`, `ResultExporter`, `ReportGenerator`, `StressType` |
| `backtest.otc` | `BookAutocallableBacktestEngine`, `BookAutocallableBacktestConfig`, `BookProduct`, `HedgeSpec`, `FuturesRollPolicy`, `AutocallableBacktestDashboard`, `AutocallableDashboardConfig` |
| `portfolio` | `EquityPortfolio` |

Not imported by the consumer at all: `cashleg`, `dynamicscenario`, `param`,
`priceenv`, `simm`, `var` (they are still imported *internally* by quant-ark
code, so the compat story must cover all 12 names, not just the consumer's 6).

Beyond imports, three non-obvious contact points:

1. **Data file reached by path, not import**: the consumer's
   `backtest_market_history.py` opens
   `util/calendar/holidayfile/china_sse.csv` directly via a `QUANTARK_PATH`
   env var and sibling-directory guesses. After migration the CSVs must ship
   as package data so the consumer can switch to
   `importlib.resources.files("quantark.util") / "calendar/holidayfile/china_sse.csv"`.
2. **Runtime registry patching**: the consumer registers sharkfin products
   and engines into `rfq.registry` (`PRODUCT_BUILDERS.register(...)`,
   `ENGINE_BUILDERS.register(...)`, `register_unknown_adapter(...)`) at
   startup because quant-ark does not register them itself. Optional
   follow-up: upstream those registrations here so consumers don't patch.
3. **Private API usage**: the consumer calls
   `RFQService._normalize_request` and `RFQService._evaluate_candidate` to
   produce a friendly "target not bracketed" diagnostic. Don't rename those
   in this migration; optionally expose a public bracketing-diagnosis helper
   later.

## The module-identity trap (most important technical constraint)

During the transition, both spellings (`asset.…` and `quantark.asset.…`) will
be importable. They **must resolve to the same module objects**. If the old
and new names load the same files as *separate* modules, every enum and class
gets two identities — `isinstance` checks and enum comparisons fail in ways
that look like data bugs (the consumer passes `util.enum` members directly
into quant-ark constructors).

Consequences for the shim design:

- A simple duplicate package tree (old dirs re-exporting from new) is **not
  safe** for `import asset.equity.engine.mc`-style submodule imports — the
  submodules get imported twice.
- The robust pattern is a **meta-path finder** that intercepts any
  `asset.*`/`util.*`/… import, imports the corresponding `quantark.*` module,
  and registers the *same object* in `sys.modules` under the old name.
  Register the finder via a `.pth` file installed with the package (the way
  setuptools/editable installs do) so it works regardless of import order;
  emit a `DeprecationWarning` once per top-level name.

## Suggested phases

**Phase A — repo hygiene** (independent, do first):
untrack run outputs and agent-tool state (`output/`, `dynamic_results/`,
`model-validation-output/`, `stress_results/`, `tmp/`, `openspec/` if it's
tooling state, `.codex/`, `.claude/`, `.gemini/`, `.playwright-mcp/`, root
screenshots/HTML), extend `.gitignore` accordingly. Decide what happens to
`docs/` (theory docs incl. Chinese `.docx`/`.pdf` — fine to keep, but review
before any public release).

**Phase B — packaging**:
replace `setup.py` with `pyproject.toml`; declare the two holiday CSVs as
package data; **audit real runtime dependencies** — `install_requires`
currently lists numpy/pandas/matplotlib/scipy, but the backtest dashboard
HTML writer and exporters likely need more (plotly? openpyxl? pyarrow has
bitten the consumer before). A fresh-venv `pip install quantark` +
`python -c "from quantark.backtest.otc import AutocallableBacktestDashboard"`
smoke test is the acceptance check.

**Phase C — namespace move + compat shim**:
move the 12 packages under `quantark/`; rewrite *internal* flat imports to
`quantark.*` (mechanical, large); add the meta-path-finder shim for the old
names; migrate `test/` imports to `quantark.*` (the tests then validate the
canonical path while the shim covers stragglers).

**Acceptance gates for Phase C** (both must pass):

1. quant-ark's own suite passes with canonical imports.
2. **open-otc-trading's full suite passes *unchanged*** against an editable
   install of the migrated quant-ark — its flat imports exercise the shim,
   and its cross-channel equivalence tests pin byte-identical pricing, which
   is exactly what catches module-identity bugs.

**Phase D — consumer migration** (happens in open-otc-trading, not here):
flip the ~40 imports to `quantark.*`, delete `ensure_quantark_path()` /
`QUANTARK_PATH` plumbing, read the holiday CSV via `importlib.resources`,
declare `quantark` in `pyproject.toml`. Only after that lands can the shim be
removed (Phase E, optional).

## Open decisions (maintainer)

- [ ] **Hosting**: public GitHub vs. private repo with collaborator access.
  This decides whether open-otc-trading can use a plain
  `quantark @ git+https://…` dependency. (Gitee may be fine for CN-local
  collaborators; GitHub travels better.)
- [ ] **History**: the git pack is small (8.3 MB) so keeping history is cheap,
  but review it for anything proprietary/sensitive before going public.
- [ ] **Shim lifetime**: keep old flat names for one transition window only,
  or indefinitely with warnings?
- [ ] **Upstream the sharkfin registry entries** out of the consumer?

## Process notes

- Concurrent agent sessions have previously committed onto shared checkouts
  of both repos — do this work in a **git worktree**, not on the main
  checkout (the consumer's venv has an editable install pointing at
  `/Users/fuxinyao/quant-ark`; breaking that mid-flight breaks the running
  app).
- The consumer's hygiene pass (removing its hardcoded paths, fail-fast boot
  check, conftest guard) is a separate work item in open-otc-trading and does
  not depend on any phase here.
