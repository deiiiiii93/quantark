# Run Fixings (W6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record a range-accrual / Asian fixing once, at the spot of the run day it was live, when a multi-day run's clock moves past it, in the equity and FX backtests, both dynamic scenarios, the replay and PnL explain; numeric twins age exactly as their dated twins.

**Architecture:** A new asset-neutral, duck-typed module `quantark/util/run_fixings.py` (`RunFixingRecorder`, the stateless core `step_fixings`, `participates`, `run_step_dt`, `RecordedFixing`, `RunFixingDay`) drives a product protocol that extends W5b's (`missing_fixing_fills`, `with_fixings_recorded`) with `with_time_elapsed(dt)`, `open_fixings(env)` and `has_run_fixings`. Each run engine calls `register` at its first valuation (it IS W5b's freeze) and `advance` after every later market update; the replay calls `step_fixings` per date; PnL explain decides the fixings that pass inside its step at S0 and reports any disagreement with the t1 snapshot in the existing `FIXING_REFILL` row.

**Tech Stack:** Python 3.10–3.13, NumPy, pandas, pytest (+xdist). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-09-run-fixings-design.md` (rev 2, user-approved 2026-10-09) + evidence `docs/superpowers/specs/2026-10-09-run-fixings-evidence/`. Binding; §9 D1–D7 are user rulings. Executors read the spec and this plan.

## Prerequisites (must hold before Task 1)

1. **`w5b-fills-unbumped` (at `20a02bc9`, on main `33201993`) has landed on `main`.** It is under full-suite test now. The W6 implementation branch is cut from `main` AFTER it lands (never from `w5b-fills-unbumped` or from `w6-run-fixings-spec`). Controller check, before Task 1:
   ```bash
   git -C /Users/fuxinyao/quant-ark merge-base --is-ancestor 20a02bc9 main && echo W5B_ON_MAIN
   ```
   Expected: `W5B_ON_MAIN`. If not printed: STOP, do not start.
2. The W5b interfaces this plan consumes, quoted exactly as they exist at `20a02bc9` (verify they still read the same on `main`; if any differs, STOP and report):
   - `quantark/util/fixings.py`: `class MissingFixingWarning(UserWarning)`; `is_history(observation_time: float) -> bool` (`t < 0 and not is_zero(t)`); `fixing_label(index: int, record) -> str`; `warn_filled_fixings(product_name: str, field: str, labels: Sequence[str], *, stacklevel: int = 3) -> None`; `freeze_missing_fixings(product, env, *, warn: bool = True)` (returns `product.with_missing_fixings_filled(env, warn=warn)` or `product`); `freeze_portfolio_fixings(portfolio)`.
   - Product protocol on `RangeAccrualOption`, `AsianOption`, `FxRangeAccrualOption`: `missing_fixing_fills(env) -> Dict[int, outcome]` (strict, `is_history` only), `with_fixings_recorded(fills: Mapping[int, outcome]) -> copy` (a record already carrying an outcome keeps it; returns `self` when nothing changes), `with_missing_fixings_filled(env, *, warn: bool = True) -> copy`.
   - `quantark/pnlexplain/base.py`: `Factor.FIXING_REFILL` (value `"fixing_refill"`); `quantark/pnlexplain/equity/lifecycle.py::fixing_refill_row(cache, level)`; `ScenarioCache(..., snap1_carried=...)` with `value_t1()` / `value_t1_carried()`.
3. Cut the branch and baseline (controller):
   ```bash
   # native worktree tool (EnterWorktree) from main; then verify:
   git -C <WT> log --oneline -1   # must be main's HEAD, which contains 20a02bc9
   ```
   `<WT>` below is the implementation worktree's absolute path.

## Global Constraints

- Canonical `quantark.*` imports only; never flat imports.
- Numerics via `quantark/util/numerical` (`is_zero`, `is_close`, `Tolerance`); no hardcoded tolerances in library code (test tolerances are explicit `is_close(..., rel_tol=..., abs_tol=...)` arguments).
- QuantArk exceptions (`ValidationError`, `NumericalError`, `MarketDataError`, `PricingError`).
- Products stay engine-agnostic; parameter objects immutable; a booked product is never mutated (every `with_*` returns a copy or `self`).
- Never invent fallback semantics: if the math or the rule is not specified, write `# TODO(run-fixings): <what>` and STOP; report to the controller.
- **Standing rule:** a numeric contract shows NO difference from its dated twin. Every behaviour test runs in both forms.
- **History doesn't move:** a recorded or filled past fixing never takes a bumped or shocked spot. t = 0 is live. In runs, a fixing is fixed at the spot of the path instant it was live when the clock moves past it (R3), or at the previous path instant's spot when no instant fell on it (R4, `gap=True`).
- Test command for EVERY run (implementers at most `-n 2`):
  `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest <paths> -n 2 -q`
  The controller runs the full suite at `-n 6`; any full-suite step is the controller's.
- A path that does not exist voids the pytest run ("no tests ran"): check paths before trusting a green run.
- Existing expected values that move: list old -> new, with the reason, in `docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt` (created in Task 1; `git add -f`) for user sign-off before landing. Allowed reasons are only the spec §8 rows and the rulings below; any other mover: STOP and report.
- After any suite run that touches mo_volmodels: `git -C <WT> checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`; never `git add example/`.
- CHANGELOG entries under `## [Unreleased]` (`### Added` / `### Changed`). `docs/` is gitignored: `git add -f` for docs files.
- Commit by pathspec (`git -C <WT> add <paths>`; never `git add -A`). Every commit message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push or merge.
- The equity `BacktestEngine` writes a `logs/` directory: tests pass `results_path=str(tmp_path)`.

## Spec drift found while writing this plan (code at `3d0c3e06` = `20a02bc9` + spec commits)

Every `file:line` the spec cites was checked; all TODO sites (§0 table), `range_accrual_option.py:404-439/529-534/549`, `asian_option.py:410-415/437`, `fx_range_accrual_option.py:270-275/360/383`, `base_equity_option.py:649-669`, `manager.py:181-198`, `backtest/equity/engine.py:253/254-260/264/272/300`, `backtest/fx/engine.py:45/50/64`, `dynamicscenario/engine.py:117/119/159/182/187/192`, `dynamicscenario/fx/engine.py:71/86/92/95`, `product_replay.py:210-225/535-552`, `replay/engine.py:43-55/409/439`, `autocallable.py:219-265`, `explain.py:50-80/83/91-94`, `lifecycle.py:215/227`, `fingerprints.py:178-185/312-368` match. Drift / gaps:

1. **The replay cannot price a range accrual.** `backtest/replay/engine_factory.py` has no `RangeAccrualOption` route on any engine type (analytical: vanilla, Asian, barrier family; MC/QUAD: vanilla + autocallables). X1/X2/X3i in the replay run on the Asian only (as the spec's own E3 probe did). The replay code change is generic and covers a range accrual the day a route exists.
2. **A `DayPath` cannot have a missing day.** `DayPath.__post_init__` requires `day_index == i` and `get_date_for_day` is `start_date + day_index days`. X3b's "DayPath with a missing day" is replaced by: (a) the equity backtest on a weekly (`W-MON`) adapter, (b) the FX backtest on a `pd.bdate_range` path with a Saturday-dated record, (c) a numeric record between two dynamic-scenario days.
3. **The equity backtest environment is always `CALENDAR_DAYS`** (`_initialize` builds it) and the equity dynamic scenario's `_clone_portfolio` drops `env.calendar` (pre-existing), so X2 "on a BUSINESS_DAYS environment" runs on the FX backtest and FX dynamic scenario, and at recorder level (Task 4) with an equity environment.
4. **Gap detection needs the record times at the previous instant**, which the spec's three-method protocol cannot answer (`missing_fixing_fills` only lists `t < 0`). This plan adds `open_fixings(env) -> Dict[int, Tuple[float, str]]` (time and W5b label of every record without an outcome) and the instance flag `has_run_fixings` (a legacy-schedule Asian has the class methods but does not take part, §5.2). See Ruling 1.
5. **`RunFixingDay` needs a `pending_pv` field** for spec §6's "Σ quantity × pending_pv(env)"; the §3.1 dataclass lacks it. Added with default `0.0` (Task 5).
6. **`AsianOption` theta emits `MissingFixingWarning` on a fixing day** (base `time_shift` moves only `maturity`/the valuation date, so a 1-day theta bump at a dated Asian's fixing day leaves that record strictly past and open; pricing fills it at the held spot and warns). Spec §3.4 and X1c ("no warning after the first valuation") need it silent. Task 2 adds a value-neutral `AsianOption.time_shift` override (Ruling 3). Numeric record Asians' theta does not move record times (pre-existing; out of scope, reported, not changed).
7. **PnL explain's roll check refuses every aged numeric participant**: `observation_records` is fingerprinted with its `observation_time` leaves and `accrued_offset` is a public float, so once the recorder ages a numeric range accrual in the equity backtest, the explain recorder's pair (today `contract_roll_days=0`, the unrolled-float route) would raise "contract replacement is not a time step". The spec's E5 only relaxes outcome fields. Task 10 extends E5: a participant's alive contract is compared with `product_t0.with_time_elapsed(decrement)` (clock leaves within `ROLL_TOL`) — Ruling 5.
8. **E2's pair decisions must override a t1-recorded outcome on the alive/carried contracts** for §4.4's P2b numbers (pv_alive_t1 4.649441, re-fill −1.905067 although t1 records the fixing OUT). `with_fixings_recorded` keeps existing outcomes, so the pair contract is built from the t0 contract (aged), not from the t1 contract — Ruling 6.

## Rulings made in this plan (each: what — why — cost if wrong)

1. **Protocol = W5b's two + `with_time_elapsed(dt)` + `open_fixings(env)` + `has_run_fixings` (property).** — Gap (`gap=True`) must be decided from the record's time at the previous instant, identically for a numeric twin and its dated contract (X2 compares `gap` bitwise); only a time query answers it; `has_run_fixings` excludes legacy-schedule Asians per instance. — If wrong: one extra read-only method per product (W4b's TARN must implement it too); no value impact.
2. **`gap` = "the record was not live (t = 0) at the previous path instant"**, computed from `open_fixings` times, not from calendar dates. On `CALENDAR_DAYS`/ACT this equals the spec's "no path instant fell on its date". On `BUSINESS_DAYS` a Saturday record resolves to t = 0 on the Friday (`calculate_year_fraction(Fri, Sat)` counts 0 business days), so it is live on Friday and recorded at Friday's spot with `gap=False`; its numeric twin (same business-day time as Friday) does the same. — X2 requires twin-identical `gap`. — If wrong: only the `gap` flag of non-business-day records on a business-day clock changes; outcomes and spots are the same.
3. **`AsianOption.time_shift` records, at the held spot, every record the bump leaves strictly past without a price** (dated in practice). — Value-neutral (pricing filled the same spot), removes the theta-bump warning (drift 6), matches the range accrual's theta rule the spec cites (§3.4, R4). — If wrong: revert one override; X1c fails for the Asian on its fixing days.
4. **A fixing that was live is decided at the spot of the FIRST instant it was live (t = 0); a gap fixing (never live) at the previous instant's spot** (USER APPROVED 2026-10-09: "the spot of the last path day on or before its date"). On `CALENDAR_DAYS`/ACT a record is live at exactly one instant, so this is R3 unchanged. On a `BUSINESS_DAYS` environment with weekend path rows a Friday fixing stays t = 0 through Saturday and Sunday and is recorded on Monday at FRIDAY's spot; its numeric twin (same business-day time) went live on the same Friday, so twins stay identical. The recorder keeps, per position, `index -> (date, spot)` of the instant each open record first went live; a same-date `advance` (Ruling 9) refreshes the spot of the entries that went live on that date. `step_fixings` takes the map as `live_spots` (index -> spot) and decides each new fill in it at its own spot. — Measured design choice signed by the user; time-based, so X2 holds. — If wrong: only weekend-row paths on a business-day clock move. Weekend VALUATIONS price the live Friday record at the weekend row's spot (it is t = 0 there); USER RULING 2026-10-09: "asset price will not move during the weekend, at least for equity and fx products" — weekend rows carry Friday's spot, so the Monday recording equals what the weekend priced and no jump arises. No guard is added for paths that move the spot on weekend rows.
5. **E5 for a participant compares the alive contract with `product_t0.with_time_elapsed(decrement)`**: exact on every term except numeric clock leaves (`observation_time`, `accrued_offset`) within `ROLL_TOL` and an outcome open at t0 and set at t1 on a record strictly past at t1. `decrement` = `run_step_dt(t0 date, env_t1)` (R6's clock) when the transition rolls the full step. — Without it the equity backtest explain regresses on numeric range accruals (drift 7). — If wrong: explain refuses or accepts some hand-built numeric pairs differently; runs are unaffected.
6. **The pair contract for a participant is `product_t0.with_fixings_recorded(t0_fills).with_time_elapsed(dt).with_fixings_recorded(step_fills)`**, used as `product_alive_t1` and as the carried t1 snapshot's product. — Reproduces §4.4 (P1, P2, P2b, P3, P3b) and is bitwise the run's own t1 contract in the equity backtest (same operations), so X3i's re-fill is exactly 0.0. — If wrong: P2b's alive value would follow t1's record instead of S0.
7. **R8 (time-of-day refusal) is checked in `advance` (both instants of the step) and in `step_fixings`, not in `register`.** — A one-instant run passes nothing; an undated dynamic scenario (R7) never advances, so a time-stamped environment there keeps today's behaviour (X3d). An `H` backtest and a 15:00 FX path still refuse at their first advance. — If wrong: a single-instant time-stamped run is accepted instead of refused.
8. **A participating position first seen at `advance` (added mid-run) is registered there** (its first valuation: W5b fill + one warning, live set from that instant). — Same "first sight wins" rule as `FloatMaturityRoller`. — If wrong: such a position would be refused or aged from the run start instead.
9. **`advance` on the same date as the last instant only refreshes that instant's spot** (no ageing, no recording). — The FX backtest's first row repeats the registration date, and a dynamic scenario's day 0 repeats the baseline date with the day's market applied; R3's "that day's spot" is the last market seen for the date. — If wrong: a fixing dated the first day would be fixed at the baseline environment's spot instead of day 0's market spot.
10. **`register`'s `RunFixingDay.fixings` is empty**: R5 fills are W5b fills (warned), not clock recordings. — The TARN plan books nothing before the baseline either. — If wrong: consumers wanting the R5 fills read them from the product.
11. **The W4b hooks (`matured_before`, `lifecycle_events`, `pending_pv`) are built here, generic and stub-tested** (Task 5), because spec §6 assigns them to the recorder. — W4b Tasks 9–11 then only implement product methods. — If wrong: Task 5 is dead code until W4b; delete it.
12. **The replay keeps the tracker's maturity floor (1e-8) for participants** after `with_time_elapsed(elapsed)`. — Existing replay behaviour and the explain check's `MATURITY_FLOOR`. — If wrong: a numeric participant on its last replay day prices at 1e-8 instead of 0.0 (as today).
13. **Test files**: product/recorder tests in `test/test_run_fixings.py`, run-path tests in `test/test_run_fixings_runs.py` (the spec names one file; two keep task ownership clean), explain tests in `test/test_missing_fixings_followup.py` (as the spec says), shared fixtures in `test/run_fixings_fixtures.py` (from the evidence `_fixtures.py`).

## Review Focus

Inputs the spec implies but no acceptance test exercises, most likely to bite first; each line's test lives in the named task.

1. **Two underlyings in one equity book** (two range accruals on `IDX_A`/`IDX_B` stepping together): each fixing must be recorded at ITS underlying's previous spot, never the other's. Test: Task 4 `test_each_underlying_records_at_its_own_previous_spot`.
2. **A numeric contract aged to exactly its maturity** inside a run: maturity must snap to exactly `0.0` and price as its dated twin on the expiry date. Test: Task 1 `test_ageing_to_maturity_snaps_and_prices_as_the_dated_twin`.
3. **A participating position added mid-run** (a user strategy adding a range accrual on day 3): first-sight registration (Ruling 8), one warning, recorded thereafter. Test: Task 4 `test_a_position_first_seen_at_advance_is_registered_there`.
4. **A `BUSINESS_DAYS` environment with weekend path rows**: Friday's fixing is live through the weekend and recorded on Monday at FRIDAY's spot (Ruling 4, user-approved), twins identical; a same-date refresh on the day it went live moves that spot. Tests: Task 8 `test_business_day_clock_on_a_calendar_path_records_at_the_last_live_instant` and Task 4 `test_a_fixing_live_for_several_instants_takes_the_first_live_instants_spot`.
5. **An explain pair whose t1 is exactly a fixing date**: that fixing is live at t1 (D1) — not decided at S0, no re-fill row from it; pv_alive_t1 equals t1's own value. Test: Task 10 `test_a_fixing_dated_t1_stays_live_in_the_pair`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `quantark/util/fixings.py` (modify) | add `snap_time(t)` (an `is_zero` time becomes exactly 0.0) | 1 |
| `quantark/asset/equity/product/option/range_accrual_option.py` (modify) | `has_run_fixings`, `open_fixings`, `with_time_elapsed`; t = 0 fill silent | 1 |
| `quantark/asset/equity/product/option/asian_option.py` (modify) | same; `time_shift` override | 1, 2 |
| `quantark/asset/fx/product/option/fx_range_accrual_option.py` (modify) | same for FX (subclasses inherit) | 3 |
| `quantark/util/run_fixings.py` (create) | `RecordedFixing`, `RunFixingDay`, `participates`, `run_step_dt`, `at_spot`, `step_fixings`, `RunFixingRecorder` | 4, 5 |
| `quantark/asset/equity/lifecycle/manager.py` (modify) | `_roll_untracked` skips participants | 6 |
| `quantark/backtest/equity/engine.py` (modify) | `register`/`advance` | 6 |
| `quantark/dynamicscenario/engine.py`, `quantark/dynamicscenario/fx/engine.py` (modify) | `register` on the working copy, `advance` per dated day | 7, 8 |
| `quantark/backtest/fx/engine.py` (modify) | `register`/`advance` | 8 |
| `quantark/asset/equity/lifecycle/autocallable.py`, `quantark/backtest/replay/product_replay.py`, `quantark/backtest/replay/engine.py` (modify) | replay ageing + `step_fixings`; carry TODOs | 9 |
| `quantark/pnlexplain/equity/{explain,fingerprints,lifecycle}.py` (modify) | E2/E4/E5 | 10 |
| `test/run_fixings_fixtures.py` (create) | shared fixture (spec §2.1) | 1, 3 |
| `test/test_run_fixings.py` (create) | protocol + recorder tests | 1–5 |
| `test/test_run_fixings_runs.py` (create) | per-path X1/X2/X3/X6 | 6–10 |
| `test/test_missing_fixings_followup.py` (modify) | X5 explain | 10 |
| `test/test_missing_fixings.py` (modify) | `test_near_expiry_fills` 1 -> 0 | 1 |
| `docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt` (create, `-f`) | old -> new for sign-off | 1, 11 |
| `CHANGELOG.md` (modify) | `[Unreleased]` entries | each task |

---

### Task 1: Equity product protocol — `with_time_elapsed`, `open_fixings`, `has_run_fixings`, silent t = 0 fill

Spec §3.2, §5.1, X4, R6.

**Files:**
- Modify: `quantark/util/fixings.py`, `quantark/asset/equity/product/option/range_accrual_option.py:487-561`, `quantark/asset/equity/product/option/asian_option.py:390-450`
- Create: `test/run_fixings_fixtures.py`, `test/test_run_fixings.py`, `docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt`
- Modify (moved value): `test/test_missing_fixings.py:191-206`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: W5b `is_history`, `fixing_label`, `warn_filled_fixings`.
- Produces:
  - `quantark.util.fixings.snap_time(t: float) -> float` — `0.0` if `is_zero(t)` else `t`.
  - `RangeAccrualOption.has_run_fixings -> bool` (property, always True); `AsianOption.has_run_fixings -> bool` (`observation_records is not None`).
  - `open_fixings(pricing_env) -> Dict[int, Tuple[float, str]]` — `{index: (resolve_time(env), fixing_label(index, record))}` for every record without an outcome.
  - `with_time_elapsed(dt: float) -> Self` — dated contract (`exercise_date` set) or `is_zero(dt)`: `self`; numeric: a deep copy whose numeric record times, `maturity` (range accrual: plus `accrued_offset += dt`) move by `dt`, each through `snap_time`; a `num_observations`/`observation_times` schedule is written out as records first. Records nothing. `AsianOption.with_time_elapsed` on a legacy schedule raises `ValidationError`.
  - `resolve_observations` warns only for `is_history` fills (a t = 0 fill is decided at the spot silently).
  - Fixture module `test/run_fixings_fixtures.py` (names below) used by every later task.

- [ ] **Step 1: Create the fixture module**

Create `test/run_fixings_fixtures.py` (spec §2.1; equity part of the evidence `_fixtures.py`, plus helpers):

```python
"""Shared fixture for the run-fixings tests (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md §2.1).

Daily path from START: spot 105 (inside [90, 110]) through day 4, then +1 a day (out of the band from day 11).
Fixings on day 4, day 18, day 120, day 240, day 365. Numeric twins: the same records with time = days / 365.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd

from quantark.asset.equity.engine.analytical import AsianOptionAnalyticalEngine, RangeAccrualAnalyticalEngine
from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.fixings import MissingFixingWarning
from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData

START = datetime(2024, 1, 1)          # a Monday
DAYS = 25
FIX_DAY = 4
FIX2_DAY = 18
VOL, RATE, DIV = 0.25, 0.05, 0.02
UNDERLYING = "EQUITY_INDEX"
ENGINES = {"ra": RangeAccrualAnalyticalEngine, "asian": AsianOptionAnalyticalEngine}


def spot_on(day: int) -> float:
    return 105.0 if day <= FIX_DAY else 105.0 + (day - FIX_DAY)


def fixing_dates() -> List[datetime]:
    return [START + timedelta(days=d) for d in (FIX_DAY, FIX2_DAY, 120, 240, 365)]


def day_of(d: datetime) -> int:
    return (d - START).days


def _range_config() -> RangeAccrualConfig:
    return RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10)


def ra_dated(outcomes: Optional[Dict[int, bool]] = None, dates: Optional[List[datetime]] = None,
             initial_date: datetime = START) -> RangeAccrualOption:
    outcomes = outcomes or {}
    return RangeAccrualOption(
        initial_price=100.0, range_config=_range_config(), exercise_date=START + timedelta(days=365),
        initial_date=initial_date,
        observation_records=[RangeAccrualObservationRecord(observation_date=d, observed_in_range=outcomes.get(i))
                             for i, d in enumerate(dates or fixing_dates())],
    )


def ra_numeric(outcomes: Optional[Dict[int, bool]] = None, times: Optional[List[float]] = None,
               maturity: float = 1.0) -> RangeAccrualOption:
    outcomes = outcomes or {}
    times = times if times is not None else [day_of(d) / 365.0 for d in fixing_dates()]
    return RangeAccrualOption(
        initial_price=100.0, range_config=_range_config(), maturity=maturity,
        observation_records=[RangeAccrualObservationRecord(observation_time=t, observed_in_range=outcomes.get(i))
                             for i, t in enumerate(times)],
    )


def asian_dated(prices: Optional[Dict[int, float]] = None) -> AsianOption:
    prices = prices or {}
    return AsianOption(
        strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC, exercise_date=START + timedelta(days=365),
        observation_records=[AsianObservationRecord(observation_date=d, observed_price=prices.get(i))
                             for i, d in enumerate(fixing_dates())],
    )


def asian_numeric(prices: Optional[Dict[int, float]] = None) -> AsianOption:
    prices = prices or {}
    return AsianOption(
        strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC, maturity=1.0,
        observation_records=[AsianObservationRecord(observation_time=day_of(d) / 365.0, observed_price=prices.get(i))
                             for i, d in enumerate(fixing_dates())],
    )


MAKERS = {("ra", "dated"): ra_dated, ("ra", "numeric"): ra_numeric,
          ("asian", "dated"): asian_dated, ("asian", "numeric"): asian_numeric}


def eq_env(day: int, *, spot: Optional[float] = None, date: Optional[datetime] = None, **kw) -> PricingEnvironment:
    return PricingEnvironment(
        valuation_date=date if date is not None else START + timedelta(days=day),
        spot_quote=SpotQuote(spot=spot if spot is not None else spot_on(day)),
        vol_surface=FlatVolSurface(volatility=VOL), rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV), **kw,
    )


def hand_outcomes(kind: str, last_day: int, spot_of: Callable[[int], float] = spot_on) -> Dict[int, object]:
    """Outcomes of every fixing dated on/before run day ``last_day``, decided at ITS OWN day's spot."""
    passed = {i: day_of(d) for i, d in enumerate(fixing_dates()) if day_of(d) <= last_day}
    if kind == "ra":
        prod = ra_dated()
        recs = prod.get_observation_records()
        return {i: prod.is_in_range(spot_of(k), i, recs[i]) for i, k in passed.items()}
    return {i: spot_of(k) for i, k in passed.items()}


def by_hand(kind: str, last_day: int, form: str = "dated", spot_of: Callable[[int], float] = spot_on):
    """The booked contract with every fixing dated on/before ``last_day`` recorded by hand at its own day's spot."""
    return MAKERS[(kind, form)](hand_outcomes(kind, last_day, spot_of))


def quiet_price(engine, product, env) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        return float(engine.price(product, env))


def count_missing(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, sum(issubclass(w.category, MissingFixingWarning) for w in caught)


def eq_book(*products, underlying: str = UNDERLYING, env: Optional[PricingEnvironment] = None) -> EquityPortfolio:
    pf = EquityPortfolio(portfolio_name="w6", pricing_environments={underlying: env or eq_env(0)},
                         creation_date=START)
    for p in products:
        kind = "asian" if isinstance(p, AsianOption) else "ra"
        pf.add_position(product=p, quantity=1.0, entry_price=0.0, underlying=underlying, engine=ENGINES[kind](),
                        entry_timestamp=START)
    return pf


def move(pf, day: int, spot: Optional[float] = None, underlying: str = UNDERLYING,
         date: Optional[datetime] = None) -> None:
    """Apply run day ``day``'s market to ``pf`` in place, as the run engines do (env mutated, date then spot)."""
    env = pf.pricing_environments[underlying]
    env.valuation_date = date if date is not None else START + timedelta(days=day)
    env.spot_quote = SpotQuote(spot=spot if spot is not None else spot_on(day))


class DailyAdapter:
    """Market data adapter over START .. at ``spot_of`` (``freq`` any pandas frequency, e.g. ``W-MON``, ``h``)."""

    def __init__(self, n: int, freq: str = "D", spot_of: Callable[[int], float] = spot_on):
        self.dates = pd.date_range(START, periods=n, freq=freq)
        self.days = [(d.to_pydatetime() - START).days for d in self.dates]
        self.spot_of = spot_of

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        idx = self.dates
        frame = lambda col, vals: TimeSeriesData(pd.DataFrame({col: vals}, index=idx), asset_name, col)  # noqa: E731
        return MarketDataSet(
            spot_data=frame("spot", [self.spot_of(k) for k in self.days]),
            vol_data=frame("volatility", [VOL] * len(idx)), rate_data=frame("rate", [RATE] * len(idx)),
            div_yield_data=frame("div_yield", [DIV] * len(idx)),
        )
```

- [ ] **Step 2: Write the failing tests**

Create `test/test_run_fixings.py`:

```python
"""Run fixings (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md): product protocol and recorder."""
import copy
import warnings
from datetime import timedelta

import pytest

import run_fixings_fixtures as F
from quantark.asset.equity.product.option import RangeAccrualConfig, RangeAccrualObservationRecord, RangeAccrualOption
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning, snap_time
from quantark.util.numerical import is_close

DAY = 1.0 / 365.0


def _aged(product, days):
    for _ in range(days):
        product = product.with_time_elapsed(DAY)
    return product


# ---------------------------------------------------------------- snap_time
def test_snap_time():
    assert snap_time(4 / 365 - 4 * (1 / 365)) == 0.0 and snap_time(-DAY) == -DAY


# ---------------------------------------------------------------- equity protocol
def test_has_run_fixings():
    assert F.ra_dated().has_run_fixings and F.ra_numeric().has_run_fixings
    assert F.asian_dated().has_run_fixings and F.asian_numeric().has_run_fixings
    legacy = AsianOption(strike=100.0, maturity=1.0, num_observations=12)
    assert legacy.has_run_fixings is False


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_dated_contract_does_not_age(kind):
    prod = F.MAKERS[(kind, "dated")]()
    assert prod.with_time_elapsed(3 * DAY) is prod


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_numeric_ageing_is_a_copy_that_records_nothing(kind):
    prod = F.MAKERS[(kind, "numeric")]()
    before = copy.deepcopy(prod)
    aged = prod.with_time_elapsed(5 * DAY)
    assert aged is not prod and type(aged) is type(prod)
    assert prod.observation_records == before.observation_records and prod.maturity == before.maturity
    assert aged.maturity == 1.0 - 5 * DAY
    assert [r.observation_time for r in aged.observation_records] == [
        snap_time(r.observation_time - 5 * DAY) for r in prod.observation_records]
    outcome = "observed_in_range" if kind == "ra" else "observed_price"
    assert all(getattr(r, outcome) is None for r in aged.observation_records)
    assert prod.with_time_elapsed(0.0) is prod


def test_range_accrual_ageing_banks_the_elapsed_time():
    aged = F.ra_numeric().with_time_elapsed(5 * DAY)
    assert aged.accrued_offset == 5 * DAY
    assert aged.get_year_fraction(F.eq_env(5)) == pytest.approx(1.0, abs=1e-15)


def test_a_fixing_lands_on_exactly_zero():
    aged = _aged(F.ra_numeric(), F.FIX_DAY)
    assert aged.observation_records[0].observation_time == 0.0           # not an ulp either side
    assert aged.open_fixings(F.eq_env(F.FIX_DAY))[0] == (0.0, "#0 t=0")


def test_num_observations_schedule_is_written_out_before_the_maturity_moves():
    prod = RangeAccrualOption(initial_price=100.0, maturity=1.0, num_observations=4,
                              range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0,
                                                              accrual_rate=0.10))
    aged = prod.with_time_elapsed(0.1)
    assert aged.num_observations is None and aged.observation_times is None
    assert [r.observation_time for r in aged.observation_records] == [0.25 - 0.1, 0.5 - 0.1, 0.75 - 0.1, 1.0 - 0.1]


def test_legacy_asian_cannot_age():
    with pytest.raises(ValidationError, match="legacy"):
        AsianOption(strike=100.0, maturity=1.0, num_observations=12).with_time_elapsed(DAY)


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_open_fixings_lists_unrecorded_records_with_time_and_label(kind):
    env = F.eq_env(F.FIX_DAY)
    dated = F.MAKERS[(kind, "dated")]({0: True} if kind == "ra" else {0: 105.0})
    got = dated.open_fixings(env)
    assert 0 not in got and sorted(got) == [1, 2, 3, 4]
    assert got[1] == (dated.observation_records[1].resolve_time(env), "#1 2024-01-19")


@pytest.mark.parametrize("kind", ["ra", "asian"])
@pytest.mark.parametrize("day", [3, 4, 5, 10, 24])
def test_numeric_twin_aged_prices_as_the_dated_contract(kind, day):
    """R6: a numeric twin aged day by day equals its dated twin (same hand-recorded fixings), relative 1e-12."""
    engine = F.ENGINES[kind]()
    outcomes = F.hand_outcomes(kind, day - 1)
    dated = F.MAKERS[(kind, "dated")](outcomes)
    numeric = _aged(F.MAKERS[(kind, "numeric")](outcomes), day)
    env = F.eq_env(day)
    assert is_close(F.quiet_price(engine, numeric, env), F.quiet_price(engine, dated, env), rel_tol=1e-12)


def test_ageing_to_maturity_snaps_and_prices_as_the_dated_twin():
    """Review Focus 2: a numeric contract aged to exactly its maturity."""
    short = [d / 365.0 for d in (4, 10)]
    numeric = _aged(F.ra_numeric({0: True, 1: True}, times=short, maturity=10 / 365.0), 10)
    assert numeric.maturity == 0.0 and numeric.observation_records[-1].observation_time == 0.0
    dated = RangeAccrualOption(
        initial_price=100.0, range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0,
                                                             accrual_rate=0.10),
        exercise_date=F.START + timedelta(days=10), initial_date=F.START,
        observation_records=[RangeAccrualObservationRecord(observation_date=F.START + timedelta(days=d),
                                                            observed_in_range=True) for d in (4, 10)])
    engine = F.ENGINES["ra"]()
    assert is_close(F.quiet_price(engine, numeric, F.eq_env(10)), F.quiet_price(engine, dated, F.eq_env(10)),
                    rel_tol=1e-12)


# ---------------------------------------------------------------- X4: only history warns
@pytest.mark.parametrize("kind", ["ra", "asian"])
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_a_t0_fill_is_silent_and_a_past_fill_warns(kind, form):
    prod = F.MAKERS[(kind, form)]()
    if form == "numeric":
        prod = _aged(prod, F.FIX_DAY)                                     # record #0 at t = 0
    engine = F.ENGINES[kind]()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        engine.price(prod, F.eq_env(F.FIX_DAY))                          # t = 0: live, decided silently
    past = prod.with_time_elapsed(DAY) if form == "numeric" else prod
    with pytest.warns(MissingFixingWarning, match="1 past fixing") as rec:
        engine.price(past, F.eq_env(F.FIX_DAY + 1))
    assert "#0" in str(rec[0].message)
    assert prod.missing_fixing_fills(F.eq_env(F.FIX_DAY)) == {}          # w5b regression: t = 0 is not history
```

- [ ] **Step 3: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -n 2 -q`
Expected: collection error `ImportError: cannot import name 'snap_time'`.

- [ ] **Step 4: Implement**

`quantark/util/fixings.py` — add after `is_history`:

```python
def snap_time(t: float) -> float:
    """``t`` with a time indistinguishable from zero (``is_zero``) made exactly ``0.0``: an aged numeric fixing lands
    on t = 0 on its own day, not an ulp either side (spec 2026-10-09 run fixings §3.2)."""
    return 0.0 if is_zero(t) else t
```

`range_accrual_option.py` — import `Dict, Mapping, Optional, Tuple` (already) and `snap_time` from `quantark.util.fixings`. In `resolve_observations` replace the fill bookkeeping (lines 531-534):

```python
                if in_range is None:
                    # A past fixing without its outcome is decided at the current spot (user ruling 2026-10-09). At
                    # t = 0 it is today's live fixing (D1), not a missing one: only history is named in the warning.
                    in_range = self.is_in_range(float(pricing_env.spot), idx, rec)
                    if is_history(t):
                        filled.append(fixing_label(idx, rec))
```

Update the docstring sentence "A past observation (t <= 0) without ``observed_in_range`` is decided ... and one ``MissingFixingWarning`` names every observation so filled" to "... and one ``MissingFixingWarning`` names every STRICTLY past observation so filled (one at t = 0 is today's live fixing)".

Add after `with_missing_fixings_filled`:

```python
    @property
    def has_run_fixings(self) -> bool:
        """Takes part in run fixings (``quantark.util.run_fixings``): every schedule form becomes records."""
        return True

    def open_fixings(self, pricing_env: "PricingEnvironment") -> Dict[int, Tuple[float, str]]:
        """``{index: (time, label)}`` for every observation without ``observed_in_range``: its time resolved at
        ``pricing_env`` (negative when past) and its W5b label."""
        return {
            idx: (rec.resolve_time(pricing_env), fixing_label(idx, rec))
            for idx, rec in enumerate(self.get_observation_records())
            if rec.observed_in_range is None
        }

    def with_time_elapsed(self, dt: float) -> "RangeAccrualOption":
        """The run clock's step (spec 2026-10-09 run fixings §3.2): a COPY of a numeric contract with every numeric
        observation time and the maturity moved by ``dt`` (each snapped to exactly 0.0 when ``is_zero``) and ``dt``
        banked in ``accrued_offset``, as ``time_shift`` banks it. Records NOTHING (unlike ``time_shift``, which fixes
        a landing fixing at the held spot for theta): a fixing landing on t = 0 stays live. A legacy schedule is
        written out as records first. A dated contract, or ``dt`` of zero, returns ``self``."""
        if self.exercise_date is not None or self.maturity is None or is_zero(dt):
            return self
        records = [replace(rec) for rec in self.get_observation_records()]
        for rec in records:
            if rec.observation_time is not None:
                rec.observation_time = snap_time(rec.observation_time - dt)
        aged = copy.deepcopy(self)
        aged.observation_records = records
        aged.observation_times = None
        aged.num_observations = None
        aged.maturity = snap_time(self.maturity - dt)
        aged.accrued_offset = self.accrued_offset + dt
        return aged
```

`asian_option.py` — import `snap_time`. In `_resolve_from_records` (lines 412-415):

```python
                if price is None:
                    # A past fixing without its price takes the current spot (user ruling 2026-10-09); at t = 0 it is
                    # today's live fixing (D1), so only history is named in the warning.
                    price = float(pricing_env.spot)
                    if is_history(t):
                        filled.append(fixing_label(idx, rec))
```

Add after `with_missing_fixings_filled`:

```python
    @property
    def has_run_fixings(self) -> bool:
        """A record schedule takes part in run fixings; a legacy schedule has no past fixing to record (spec §5.2)."""
        return self.observation_records is not None

    def open_fixings(self, pricing_env: "PricingEnvironment") -> Dict[int, Tuple[float, str]]:
        """``{index: (time, label)}`` for every record without ``observed_price`` (empty for a legacy schedule)."""
        if self.observation_records is None:
            return {}
        return {
            idx: (rec.resolve_time(pricing_env), fixing_label(idx, rec))
            for idx, rec in enumerate(self.observation_records)
            if rec.observed_price is None
        }

    def with_time_elapsed(self, dt: float) -> "AsianOption":
        """The run clock's step: a COPY of a numeric record contract with every numeric record time and the maturity
        moved by ``dt`` (snapped to 0.0 when ``is_zero``). Records nothing. A dated contract, or ``dt`` of zero,
        returns ``self``. A legacy schedule has no records to move: refused."""
        if self.observation_records is None:
            raise ValidationError(
                "a legacy Asian schedule (no observation_records) cannot age by the run clock; book records "
                "(spec 2026-10-09 run fixings §5.2)"
            )
        if self.exercise_date is not None or self.maturity is None or is_zero(dt):
            return self
        aged = copy.deepcopy(self)
        aged.observation_records = [
            replace(rec, observation_time=snap_time(rec.observation_time - dt))
            if rec.observation_time is not None else rec
            for rec in aged.observation_records
        ]
        aged.maturity = snap_time(self.maturity - dt)
        return aged
```

Move the existing w5b test's expectation (`test/test_missing_fixings.py::TestEquityRangeAccrualFill::test_near_expiry_fills`, line 198): `assert len(caught) == 1` -> `assert len(caught) == 0  # both records are at t = 0: live, not history (run fixings §3.2)`.

Create `docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt`:

```text
# Run fixings (W6): existing expected values that moved. Format: test :: what : old -> new  [reason]
test/test_missing_fixings.py::TestEquityRangeAccrualFill::test_near_expiry_fills[analytical,mc] :: MissingFixingWarning count : 1 -> 0  [spec §3.2/§8: a t = 0 fill is live, not missing; price unchanged]
```

CHANGELOG `### Changed`:

```markdown
- A fixing at exactly t = 0 (today's, unprinted) that pricing decides at the current spot no longer counts as
  missing: `MissingFixingWarning` names only strictly past fills (`RangeAccrualOption`, `AsianOption`; spec
  2026-10-09 run fixings §3.2). Prices are unchanged.
```

CHANGELOG `### Added`:

```markdown
- Run-fixings protocol on `RangeAccrualOption` and record-schedule `AsianOption`: `has_run_fixings`,
  `open_fixings(env)` and `with_time_elapsed(dt)` (the run clock's step: a numeric contract's record times and
  maturity move by `dt`, snapped to exactly 0.0 when zero; records nothing; a dated contract returns itself).
  `quantark.util.fixings.snap_time`.
```

- [ ] **Step 5: Run to verify they pass, then the neighbours**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py test/test_missing_fixings.py test/test_missing_fixings_followup.py test/test_missing_fixings_unbumped.py test/test_missing_fixings_scenarios.py test/test_range_accrual_option.py test/test_asian_option.py -n 2 -q`
Expected: all pass. (Check `ls test/test_asian_option.py`; if absent use `test/ -k asian` restricted to files that exist.) Any other count assertion that moves: add its line to `moved_values.txt` only if the reason is the t = 0 silence; else STOP.

- [ ] **Step 6: Commit**

```bash
git -C <WT> add quantark/util/fixings.py quantark/asset/equity/product/option/range_accrual_option.py quantark/asset/equity/product/option/asian_option.py test/run_fixings_fixtures.py test/test_run_fixings.py test/test_missing_fixings.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt
git -C <WT> commit -m "feat(run-fixings): equity range accrual / Asian age by the run clock; a t = 0 fill is silent

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `AsianOption.time_shift` records the fixings its theta bump steps over (Ruling 3)

Spec §3.4 (theta predicts the run), X1c.

**Files:**
- Modify: `quantark/asset/equity/product/option/asian_option.py` (new method after `with_time_elapsed`), `CHANGELOG.md`
- Test: `test/test_run_fixings.py`

**Interfaces:**
- Consumes: Task 1 (`is_history` import already in `asian_option.py`).
- Produces: `AsianOption.time_shift(time_bump, bumped_date, pricing_env, *, keep_landing=False) -> None` — calls `BaseEquityOption.time_shift` first (which moves `maturity` or sets `pricing_env.valuation_date = bumped_date`), then sets `observed_price = float(pricing_env.spot)` on every record that is `is_history` at `pricing_env` and has no price.

- [ ] **Step 1: Write the failing tests** (append to `test/test_run_fixings.py`)

```python
# ---------------------------------------------------------------- Task 2: Asian theta records what it steps over
def test_asian_theta_step_records_the_stepped_over_fixing_at_the_held_spot():
    from quantark.asset.equity.riskmeasures.greeks.theta_shift import shift_for_theta

    prod = F.asian_dated()
    env = F.eq_env(F.FIX_DAY, spot=107.0)
    bumped = env.valuation_date + timedelta(days=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        shift = shift_for_theta(prod, env, DAY, bumped)
        theta_price = F.ENGINES["asian"]().price(shift.product, shift.env)
    assert shift.product.observation_records[0].observed_price == 107.0
    assert prod.observation_records[0].observed_price is None                 # the booked contract is untouched
    env_t1 = F.eq_env(F.FIX_DAY + 1, spot=107.0)
    assert theta_price == F.quiet_price(F.ENGINES["asian"](), prod, env_t1)   # value-neutral: pricing filled 107 too


def test_asian_greeks_on_a_fixing_day_are_silent():
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        GreeksCalculator().calculate(F.asian_dated(), F.eq_env(F.FIX_DAY), F.ENGINES["asian"]())
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -k "asian_theta or asian_greeks" -n 2 -q`
Expected: FAIL — `MissingFixingWarning` raised as error (and `observed_price is None`).

- [ ] **Step 3: Implement** (in `AsianOption`)

```python
    def time_shift(
        self, time_bump: float, bumped_date: datetime, pricing_env, *, keep_landing: bool = False
    ) -> None:
        """Age for theta (``BaseEquityOption.time_shift``), then record at the held spot every record the bump left
        strictly past without a price: the theta bump holds the spot, so this is the fill pricing would make at the
        bumped instant (value-neutral), recorded so it is not reported as missing (spec 2026-10-09 run fixings §3.4,
        the range accrual's theta rule). A record landing on the bumped instant (t = 0) stays live."""
        super().time_shift(time_bump, bumped_date, pricing_env, keep_landing=keep_landing)
        if self.observation_records is None:
            return
        spot = float(pricing_env.spot)
        self.observation_records = [
            replace(rec, observed_price=spot)
            if rec.observed_price is None and is_history(rec.resolve_time(pricing_env)) else rec
            for rec in self.observation_records
        ]
```

CHANGELOG `### Changed`:

```markdown
- `AsianOption.time_shift` (theta) records, at the held spot, a dated record its bump steps over; the bumped price is
  unchanged (pricing filled the same spot) and theta on a fixing day no longer warns `MissingFixingWarning`.
```

- [ ] **Step 4: Run to verify they pass, plus theta neighbours**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py test/test_missing_fixings.py test/test_missing_fixings_followup.py test/test_missing_fixings_unbumped.py -n 2 -q` and `... -m pytest test -k "asian and theta" -n 2 -q`
Expected: all pass, values unchanged (any moved theta value is NOT allowed: STOP and report).

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/equity/product/option/asian_option.py test/test_run_fixings.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): Asian theta records the fixing its bump steps over at the held spot

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: FX range accrual protocol (domestic, foreign, quanto)

Spec §3.2, §5.1, X4, R1, R6.

**Files:**
- Modify: `quantark/asset/fx/product/option/fx_range_accrual_option.py:250-430`, `test/run_fixings_fixtures.py` (FX part), `CHANGELOG.md`
- Test: `test/test_run_fixings.py`

**Interfaces:**
- Consumes: Task 1 `snap_time`.
- Produces: on `FxRangeAccrualOption` (inherited by `FxForeignRangeAccrualOption`, `FxQuantoRangeAccrualOption`): `has_run_fixings` (True), `open_fixings(fx_env)`, `with_time_elapsed(dt)` (numeric = `maturity is not None`, as `time_shift`: records, `maturity`, `delivery` when set, `accrued_offset += dt`; snap; copy; dated or zero `dt` -> `self`); `resolve_observations` names only `is_history` fills.
- Fixtures (appended to `test/run_fixings_fixtures.py`): `PAIR`, `fx_spot(day)`, `fx_env(day, *, spot=None, quanto=False, date=None, **kw)`, `FX_CLASSES`, `fx_ra(kind, form, outcomes=None, dates=None)`, `fx_hand_outcomes(kind, last_day, spot_of=fx_spot)`, `fx_book(*products, env=None)`, `fx_path(index, spot_of=fx_spot)`.

- [ ] **Step 1: Append the FX fixtures** to `test/run_fixings_fixtures.py`

```python
# ---------------------------------------------------------------- FX (EURUSD range [1.10, 1.30], notional 1e6)
import numpy as np  # noqa: E402

from quantark.asset.fx.engine.analytical import (  # noqa: E402
    FxQuantoRangeAccrualAnalyticalEngine, FxRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.engine.analytical.fx_foreign_range_accrual_engine import (  # noqa: E402
    FxForeignRangeAccrualAnalyticalEngine,
)
from quantark.asset.fx.product import CurrencyPair  # noqa: E402
from quantark.asset.fx.product.option import (  # noqa: E402
    FxQuantoRangeAccrualOption, FxRangeAccrualConfig, FxRangeAccrualObservationRecord, FxRangeAccrualOption,
)
from quantark.asset.fx.product.option.fx_foreign_range_accrual_option import FxForeignRangeAccrualOption  # noqa: E402
from quantark.portfolio.fx import FXPortfolio  # noqa: E402
from quantark.priceenv import FxPricingEnvironment  # noqa: E402
from quantark.priceenv.fx_pricing_environment import FxQuantoMarketData, QuantoConversionOrientation  # noqa: E402

PAIR = CurrencyPair("EUR", "USD")
FX_CLASSES = {
    "domestic": (FxRangeAccrualOption, FxRangeAccrualAnalyticalEngine),
    "foreign": (FxForeignRangeAccrualOption, FxForeignRangeAccrualAnalyticalEngine),
    "quanto": (FxQuantoRangeAccrualOption, FxQuantoRangeAccrualAnalyticalEngine),
}


def fx_spot(day: int) -> float:
    return 1.25 if day <= FIX_DAY else 1.25 + 0.01 * (day - FIX_DAY)


def fx_env(day: int, *, spot: Optional[float] = None, quanto: bool = False, date: Optional[datetime] = None,
           **kw) -> FxPricingEnvironment:
    q = FxQuantoMarketData(settlement_curve=FlatRateCurve(rate=0.01), quanto_vol=0.08, correlation=0.3,
                           conversion_orientation=QuantoConversionOrientation.SETTLEMENT_PER_DOMESTIC)
    return FxPricingEnvironment(
        valuation_date=date if date is not None else START + timedelta(days=day),
        spot_quote=SpotQuote(spot=fx_spot(day) if spot is None else spot),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10), quanto=q if quanto else None, **kw,
    )


def fx_ra(kind: str, form: str, outcomes: Optional[Dict[int, bool]] = None,
          dates: Optional[List[datetime]] = None, times: Optional[List[float]] = None):
    outcomes = outcomes or {}
    cls = FX_CLASSES[kind][0]
    extra = {"quanto_fx_rate": 110.0} if kind == "quanto" else {}
    cfg = FxRangeAccrualConfig(upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04)
    if form == "dated":
        recs = [FxRangeAccrualObservationRecord(observation_date=d, observed_in_range=outcomes.get(i))
                for i, d in enumerate(dates or fixing_dates())]
        return cls(notional=1_000_000.0, range_config=cfg, currency_pair=PAIR, expiry_date=START + timedelta(days=365),
                   initial_date=START, observation_records=recs, **extra)
    times = times if times is not None else [day_of(d) / 365.0 for d in fixing_dates()]
    recs = [FxRangeAccrualObservationRecord(observation_time=t, observed_in_range=outcomes.get(i))
            for i, t in enumerate(times)]
    return cls(notional=1_000_000.0, range_config=cfg, currency_pair=PAIR, maturity=1.0, observation_records=recs,
               **extra)


def fx_hand_outcomes(kind: str, last_day: int, spot_of: Callable[[int], float] = fx_spot) -> Dict[int, bool]:
    prod = fx_ra(kind, "dated")
    recs = prod.get_observation_records()
    return {i: prod.is_in_range(spot_of(day_of(d)), i, recs[i])
            for i, d in enumerate(fixing_dates()) if day_of(d) <= last_day}


def fx_book(*products, env: Optional[FxPricingEnvironment] = None) -> FXPortfolio:
    pf = FXPortfolio(portfolio_name="w6fx", pricing_environments={"EURUSD": env or fx_env(0)})
    for p in products:
        kind = next(k for k, (cls, _) in FX_CLASSES.items() if type(p) is cls)
        pf.add_position(product=p, quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=FX_CLASSES[kind][1]())
    return pf


def fx_path(index, spot_of: Callable[[int], float] = fx_spot) -> pd.DataFrame:
    days = [(pd.Timestamp(t).to_pydatetime() - START).days for t in index]
    n = len(index)
    return pd.DataFrame({"EURUSD_spot": [spot_of(k) for k in days], "EURUSD_vol": np.full(n, 0.10),
                         "EURUSD_dom_rate": np.full(n, 0.05), "EURUSD_for_rate": np.full(n, 0.03)}, index=index)
```

- [ ] **Step 2: Write the failing tests** (append to `test/test_run_fixings.py`)

```python
# ---------------------------------------------------------------- Task 3: FX protocol
FX_KINDS = ["domestic", "foreign", "quanto"]


@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_protocol_and_ageing(kind):
    dated, numeric = F.fx_ra(kind, "dated"), F.fx_ra(kind, "numeric")
    assert dated.has_run_fixings and dated.with_time_elapsed(DAY) is dated
    before = copy.deepcopy(numeric)
    aged = _aged(numeric, F.FIX_DAY)
    assert type(aged) is type(numeric) and numeric.observation_records == before.observation_records
    assert aged.observation_records[0].observation_time == 0.0
    assert is_close(aged.accrued_offset, F.FIX_DAY * DAY, rel_tol=1e-15)
    assert aged.open_fixings(F.fx_env(F.FIX_DAY))[0] == (0.0, "#0 t=0")
    assert dated.open_fixings(F.fx_env(F.FIX_DAY))[0] == (0.0, "#0 2024-01-05")


def test_fx_ageing_moves_a_numeric_delivery():
    prod = F.fx_ra("domestic", "numeric")
    prod.delivery = 1.0 + 2 * DAY
    assert prod.with_time_elapsed(DAY).delivery == 1.0 + 2 * DAY - DAY


@pytest.mark.parametrize("kind", FX_KINDS)
@pytest.mark.parametrize("day", [3, 4, 5, 10, 24])
def test_fx_numeric_twin_aged_prices_as_the_dated_contract(kind, day):
    engine = F.FX_CLASSES[kind][1]()
    outcomes = F.fx_hand_outcomes(kind, day - 1)
    env = F.fx_env(day, quanto=(kind == "quanto"))
    dated = F.fx_ra(kind, "dated", outcomes)
    numeric = _aged(F.fx_ra(kind, "numeric", outcomes), day)
    assert is_close(F.quiet_price(engine, numeric, env), F.quiet_price(engine, dated, env), rel_tol=1e-12)


@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_t0_fill_is_silent_and_a_past_fill_warns(kind):
    engine = F.FX_CLASSES[kind][1]()
    q = kind == "quanto"
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        engine.price(F.fx_ra(kind, "dated"), F.fx_env(F.FIX_DAY, quanto=q))
    with pytest.warns(MissingFixingWarning, match="1 past fixing"):
        engine.price(F.fx_ra(kind, "dated"), F.fx_env(F.FIX_DAY + 1, quanto=q))
```

- [ ] **Step 3: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -k fx -n 2 -q`
Expected: FAIL — `AttributeError: 'FxRangeAccrualOption' object has no attribute 'has_run_fixings'`.

- [ ] **Step 4: Implement** (in `FxRangeAccrualOption`; import `snap_time` from `quantark.util.fixings`)

In `resolve_observations` (lines 272-275):

```python
                if in_range is None:
                    # A past fixing without its outcome is decided at the current spot (user ruling 2026-10-09); at
                    # t = 0 it is today's live fixing (D1), so only history is named in the warning.
                    in_range = self.is_in_range(fx_env.spot, idx, rec)
                    if is_history(t):
                        filled.append(fixing_label(idx, rec))
```

Add after `with_missing_fixings_filled`:

```python
    @property
    def has_run_fixings(self) -> bool:
        """Takes part in run fixings (``quantark.util.run_fixings``), with the foreign and quanto variants."""
        return True

    def open_fixings(self, fx_env: "FxPricingEnvironment") -> Dict[int, Tuple[float, str]]:
        """``{index: (time, label)}`` for every fixing without ``observed_in_range`` (time resolved at ``fx_env``)."""
        return {
            idx: (rec.resolve_time(fx_env), fixing_label(idx, rec))
            for idx, rec in enumerate(self.get_observation_records())
            if rec.observed_in_range is None
        }

    def with_time_elapsed(self, dt: float) -> "FxRangeAccrualOption":
        """The run clock's step (spec 2026-10-09 run fixings §3.2): a COPY of a numeric contract with its fixing
        times, ``maturity`` and ``delivery`` moved by ``dt`` (snapped to exactly 0.0 when ``is_zero``) and ``dt``
        banked in ``accrued_offset``, as ``time_shift`` does, but without mutating and without recording anything.
        A dated contract, or ``dt`` of zero, returns ``self``."""
        if self.maturity is None or is_zero(dt):
            return self
        records = [replace(rec) for rec in self.get_observation_records()]
        for rec in records:
            if rec.observation_time is not None:
                rec.observation_time = snap_time(rec.observation_time - dt)
        aged = copy.deepcopy(self)
        aged._set_records(records)
        aged.maturity = snap_time(self.maturity - dt)
        if self.delivery is not None:
            aged.delivery = snap_time(self.delivery - dt)
        aged.accrued_offset = self.accrued_offset + dt
        return aged
```

Update the `resolve_observations` docstring like Task 1. CHANGELOG: extend Task 1's two bullets to name `FxRangeAccrualOption` (and its foreign / quanto subclasses).

- [ ] **Step 5: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py test/test_fx_range_accrual.py test/test_fx_quanto_range_accrual.py test/test_missing_fixings.py test/test_missing_fixings_unbumped.py -n 2 -q`
Expected: all pass (moved counts only for the t = 0 reason -> `moved_values.txt`).

- [ ] **Step 6: Commit**

```bash
git -C <WT> add quantark/asset/fx/product/option/fx_range_accrual_option.py test/run_fixings_fixtures.py test/test_run_fixings.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt
git -C <WT> commit -m "feat(run-fixings): FX range accruals age by the run clock; a t = 0 fill is silent

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `quantark/util/run_fixings.py` — recorder and stateless core

Spec §3.1, §3.3 R1–R10, §3.4, X3c–X3f (unit level), Review Focus 1 and 3.

**Files:**
- Create: `quantark/util/run_fixings.py`
- Test: `test/test_run_fixings.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: Tasks 1, 3 protocol; W5b `freeze_missing_fixings`.
- Produces (exact):
  ```python
  RUN_FIXINGS_PROTOCOL = ("missing_fixing_fills", "with_fixings_recorded", "with_time_elapsed", "open_fixings")
  def participates(product) -> bool
  def run_step_dt(prev_date, env) -> float          # 0.0 on the same date; ValidationError when env is earlier
  def at_spot(env, spot: float)                     # dataclasses.replace(env, spot_quote=replace(env.spot_quote, spot=spot))
  def require_date_instant(instant) -> None         # R8: ValidationError unless midnight
  def step_fixings(product, env, *, prev_date, prev_spot, run_fills, live_spots=None) -> Tuple[Any, Dict[int, Any]]
  def update_live(open_fixings, date, spot, before=None, exclude=()) -> Dict[int, Tuple[Any, float]]   # Ruling 4
  @dataclass(frozen=True) class RecordedFixing: position_id: str; index: int; label: str; outcome: Any; spot: float; gap: bool
  @dataclass(frozen=True) class RunFixingDay: date: Optional[datetime]; fixings: Tuple[RecordedFixing, ...];
                                               product_events: Tuple[Tuple[str, float, Any], ...] = (); pending_pv: float = 0.0
  class RunFixingRecorder:
      def register(self, portfolio) -> RunFixingDay
      def advance(self, portfolio) -> RunFixingDay
      days: Tuple[RunFixingDay, ...]        # property, every day returned so far
      recorded: Tuple[RecordedFixing, ...]  # property, flattened
  ```
  `step_fixings` does NOT age: the caller passes the product already aged to `env`'s date. It returns `(product with run_fills and the new fills recorded, new_fills)` where `new_fills = product_with_run_fills.missing_fixing_fills(at_spot(env, prev_spot))`, except that a new fill whose index is in `live_spots` (index -> spot of the instant it first went live, Ruling 4) is decided at that spot; it never mutates `run_fills`. `update_live` returns `{index: (date, spot)}` for the records of `open_fixings` that are live (t = 0) and not in `exclude`, keeping the entry of `before` for a record already there and `(date, spot)` for a newly live one.

- [ ] **Step 1: Write the failing tests** (append to `test/test_run_fixings.py`)

```python
# ---------------------------------------------------------------- Task 4: recorder
from quantark.util.run_fixings import (  # noqa: E402
    RecordedFixing, RunFixingRecorder, participates, run_step_dt, step_fixings,
)


def _rows(rec, pid):
    return [(f.index, f.outcome, f.spot, f.gap) for f in rec.recorded if f.position_id == pid]


def _drive(pf, rec, days, spot_of=F.spot_on):
    for k in days:
        F.move(pf, k, spot_of(k))
        rec.advance(pf)


def test_participates():
    assert participates(F.ra_dated()) and participates(F.asian_numeric()) and participates(F.fx_ra("quanto", "dated"))
    assert not participates(AsianOption(strike=100.0, maturity=1.0, num_observations=12))
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType
    assert not participates(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0))


def test_run_step_dt():
    env = F.eq_env(5)
    assert run_step_dt(F.START + timedelta(days=4), env) == 1 / 365.0
    assert run_step_dt(env.valuation_date, env) == 0.0
    with pytest.raises(ValidationError, match="back"):
        run_step_dt(F.START + timedelta(days=6), env)


def test_step_fixings_is_stateless_and_skips_run_fills():
    env = F.eq_env(5)
    fills = {}
    prod, new = step_fixings(F.ra_dated(), env, prev_date=F.START + timedelta(days=4), prev_spot=105.0, run_fills=fills)
    assert new == {0: True} and prod.observation_records[0].observed_in_range is True and fills == {}
    again, new2 = step_fixings(F.ra_dated(), env, prev_date=env.valuation_date, prev_spot=200.0, run_fills={0: True})
    assert new2 == {} and again.observation_records[0].observed_in_range is True


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_d2_the_clock_fixes_a_fixing_at_the_spot_of_the_day_it_was_live(kind):
    dated, numeric = F.MAKERS[(kind, "dated")](), F.MAKERS[(kind, "numeric")]()
    pf = F.eq_book(dated, numeric)
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        rec.register(pf)
        _drive(pf, rec, range(0, 12))
    want_out = True if kind == "ra" else 105.0
    assert _rows(rec, pid_d) == [(0, want_out, 105.0, False)]
    assert _rows(rec, pid_d) == _rows(rec, pid_n)                         # X2 bitwise
    assert pf.positions[pid_d].product is not dated and dated.observation_records[0] == F.MAKERS[(kind, "dated")]().observation_records[0]
    day5 = next(d for d in rec.days if d.fixings)
    assert day5.date == F.START + timedelta(days=5)


def test_r4_a_gap_fixing_takes_the_previous_instants_spot():
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=[4.5 / 365.0, 18 / 365.0, 120 / 365.0, 240 / 365.0, 1.0]))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        _drive(pf, rec, [0, 1, 2, 3, 6], spot_of=lambda k: 100.0 + k)     # day 4 and 5 skipped
    assert _rows(rec, pid_d) == [(0, True, 103.0, True)]
    assert _rows(rec, pid_n) == [(0, True, 103.0, True)]


def test_r5_history_at_the_first_valuation_warns_once_per_position_and_the_first_day_is_live():
    dates = [F.START - timedelta(days=3), F.START] + F.fixing_dates()[1:]
    pf = F.eq_book(F.ra_dated(dates=dates, initial_date=F.START - timedelta(days=10)),
                   F.ra_dated(dates=dates, initial_date=F.START - timedelta(days=10)), env=F.eq_env(0, spot=100.0))
    rec = RunFixingRecorder()
    with pytest.warns(MissingFixingWarning) as caught:
        day0 = rec.register(pf)
    assert len([w for w in caught if issubclass(w.category, MissingFixingWarning)]) == 2 and day0.fixings == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        F.move(pf, 0, 105.0)
        rec.advance(pf)                                                   # same date: refresh day 0's spot (Ruling 9)
        F.move(pf, 1, 120.0)
        rec.advance(pf)
    pid = list(pf.positions)[0]
    assert _rows(rec, pid) == [(1, True, 105.0, False)]                  # the START record, at day 0's market spot
    assert pf.positions[pid].product.observation_records[0].observed_in_range is True   # R5 fill at 100


def test_r8_a_time_of_day_instant_is_refused():
    pf = F.eq_book(F.ra_dated())
    rec = RunFixingRecorder()
    rec.register(pf)
    F.move(pf, 1, date=F.START + timedelta(days=1, hours=15))
    with pytest.raises(ValidationError, match="time of day"):
        rec.advance(pf)


def test_r8_r9_do_not_apply_without_a_participant():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.engine.analytical import BlackScholesEngine
    from quantark.util.enum import OptionType
    pf = F.eq_book()
    pf.add_position(product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
                    quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING, engine=BlackScholesEngine())
    rec = RunFixingRecorder()
    rec.register(pf)
    F.move(pf, 1, date=F.START + timedelta(days=1, hours=15))
    assert rec.advance(pf).fixings == ()


def test_r9_the_clock_cannot_go_back():
    pf = F.eq_book(F.ra_dated())
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, [2])
    F.move(pf, 1)
    with pytest.raises(ValidationError, match="back"):
        rec.advance(pf)


def test_non_participants_are_untouched():
    legacy = AsianOption(strike=100.0, maturity=1.0, num_observations=12)
    pf = F.eq_book(legacy)
    pid = list(pf.positions)[0]
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, range(1, 8))
    assert pf.positions[pid].product is legacy and rec.recorded == ()


def test_each_underlying_records_at_its_own_previous_spot():
    """Review Focus 1."""
    pf = F.eq_book(F.ra_dated(), underlying="IDX_A")
    pf.pricing_environments["IDX_B"] = F.eq_env(0)
    pf.add_position(product=F.ra_dated(), quantity=1.0, entry_price=0.0, underlying="IDX_B",
                    engine=F.ENGINES["ra"]())
    pid_a, pid_b = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    for k in range(1, 6):
        F.move(pf, k, 105.0, underlying="IDX_A")
        F.move(pf, k, 120.0 if k == 4 else 105.0, underlying="IDX_B")
        rec.advance(pf)
    assert _rows(rec, pid_a) == [(0, True, 105.0, False)]
    assert _rows(rec, pid_b) == [(0, False, 120.0, False)]


def test_a_fixing_live_for_several_instants_takes_the_first_live_instants_spot():
    """Review Focus 4 / Ruling 4 (user-approved): on a business-day clock with weekend rows Friday's fixing is t = 0
    on Friday, Saturday and Sunday and is decided on Monday at FRIDAY's spot (a same-date refresh on Friday counts),
    dated and numeric twins identical."""
    from quantark.util.calendar import CalendarType, DayCountConvention, calculate_year_fraction, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    kw = dict(day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal)
    times = [calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal)
             for d in F.fixing_dates()]
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=times), env=F.eq_env(0, spot=105.0, **kw))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    spots = {1: 105.0, 2: 105.0, 3: 105.0, 4: 120.0, 5: 120.0, 6: 120.0, 7: 120.0}   # day 4 = Friday
    for k in range(1, 8):
        F.move(pf, k, spots[k])
        rec.advance(pf)
        if k == 4:                                          # same-date refresh: Friday's market is 105 after all
            F.move(pf, 4, 105.0)
            rec.advance(pf)
    assert _rows(rec, pid_d) == _rows(rec, pid_n) == [(0, True, 105.0, False)]   # Friday's spot, not Sunday's 120
    assert next(d for d in rec.days if d.fixings).date == F.START + timedelta(days=7)   # recorded on Monday


def test_a_position_first_seen_at_advance_is_registered_there():
    """Review Focus 3 (Ruling 8)."""
    pf = F.eq_book(F.ra_dated())
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, range(1, 6))
    late = F.ra_dated(dates=[F.START + timedelta(days=2)] + F.fixing_dates()[1:])
    pf.add_position(product=late, quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING, engine=F.ENGINES["ra"]())
    pid = list(pf.positions)[-1]
    F.move(pf, 6)
    with pytest.warns(MissingFixingWarning):
        rec.advance(pf)                                                   # its first valuation: day 2 is history
    assert pf.positions[pid].product.observation_records[0].observed_in_range is True    # filled at 107 (in range)
    _drive(pf, rec, range(7, 20))
    assert _rows(rec, pid) == [(1, False, 119.0, False)]                 # day 18 at day 18's spot


def test_reverse_mode_is_decided_by_the_products_own_test():
    prod = F.ra_dated()
    prod.range_config = RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10, is_reverse=True)
    pf = F.eq_book(prod)
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, range(1, 6))
    assert _rows(rec, list(pf.positions)[0]) == [(0, False, 105.0, False)]


def test_business_day_clock_twins_age_identically():
    """X2 on BUSINESS_DAYS at recorder level (drift 3)."""
    from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    kw = dict(day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal)
    from quantark.util.calendar import calculate_year_fraction
    times = [calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal)
             for d in F.fixing_dates()]
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=times), env=F.eq_env(0, **kw))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    days = [k for k in range(0, 26) if (F.START + timedelta(days=k)).weekday() < 5]
    _drive(pf, rec, days)
    assert _rows(rec, pid_d) == _rows(rec, pid_n) and len(_rows(rec, pid_d)) == 2
```

Check `RangeAccrualConfig` accepts `is_reverse` as a constructor argument (it is a frozen dataclass; the attribute `range_config.is_reverse` is read by `is_in_range`); if it is a different keyword, use that keyword.

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -n 2 -q`
Expected: collection error `ModuleNotFoundError: No module named 'quantark.util.run_fixings'`.

- [ ] **Step 3: Implement** `quantark/util/run_fixings.py`

```python
"""Run fixings: a fixing that passes during a multi-day run is recorded once, at its own run day's spot.

Spec: docs/superpowers/specs/2026-10-09-run-fixings-design.md (W6). Asset-neutral and duck-typed like
``PortfolioLifecycleManager`` (never imports ``quantark.portfolio``): it acts on any object with ``positions``
(each with ``product``, ``quantity``, ``underlying``) and ``pricing_environments``.

A product takes part when ``has_run_fixings`` is true and it has the protocol methods ``missing_fixing_fills``,
``with_fixings_recorded`` (W5b), ``with_time_elapsed`` and ``open_fixings``. Rules (spec §3.3): R1 the run's quoted
spot; R2 t = 0 is live, t < 0 is history; R3 when the clock moves from path instant P to P', every record strictly
past at P' without an outcome is recorded at P's spot; R4 a record no instant fell on takes the previous instant's
spot, ``gap=True``; R5 the first valuation is W5b's freeze; R6 a numeric contract ages by the environment's year
fraction; R7 an undated run never advances; R8 time-of-day instants are refused; R9 the clock never goes back; R10
the run's working positions get copies, the booked product is never mutated.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import freeze_missing_fixings
from quantark.util.numerical import is_zero

RUN_FIXINGS_PROTOCOL = ("missing_fixing_fills", "with_fixings_recorded", "with_time_elapsed", "open_fixings")


@dataclass(frozen=True)
class RecordedFixing:
    """One fixing the run's clock recorded (R3/R4)."""

    position_id: str
    index: int            # record index (records are never dropped by these products)
    label: str            # W5b ``fixing_label`` at the step's instant
    outcome: Any          # bool (range accrual) or price / rate (Asian, TARN, TARF)
    spot: float           # the spot it was fixed at: the previous path instant's
    gap: bool             # True when it was not live (t = 0) at that instant (R4)


@dataclass(frozen=True)
class RunFixingDay:
    """What one ``register`` / ``advance`` did."""

    date: Optional[datetime]
    fixings: Tuple[RecordedFixing, ...]
    product_events: Tuple[Tuple[str, float, Any], ...] = ()   # (position_id, quantity, event): W4b (spec §6)
    pending_pv: float = 0.0                                   # Σ quantity × pending_pv(env): W4b (spec §6)


def participates(product: Any) -> bool:
    """True for a product that takes part in run fixings (every other position is left exactly as it is)."""
    return bool(getattr(product, "has_run_fixings", False)) and all(
        callable(getattr(product, name, None)) for name in RUN_FIXINGS_PROTOCOL
    )


def require_date_instant(instant: Any) -> None:
    """R8 / D6: a run instant holding a participating product must be a date (midnight)."""
    stamp = pd.Timestamp(instant)
    if stamp != stamp.normalize():
        # TODO(run-fixings): intraday fixing instants — a fixing-time term on the records (spec §10 follow-up)
        raise ValidationError(
            f"run instant {stamp} carries a time of day: a dated fixing has no fixing time, so a run holding a "
            "range accrual / Asian / FX range accrual must step on dates (spec 2026-10-09 run fixings R8)"
        )


def run_step_dt(prev_date: Any, env: Any) -> float:
    """R6: the environment's year fraction from ``prev_date`` to ``env.valuation_date`` (0.0 on the same date)."""
    new = env.valuation_date
    if pd.Timestamp(new) < pd.Timestamp(prev_date):
        raise ValidationError(f"the run clock cannot go back: from {prev_date} to {new} (R9)")
    if pd.Timestamp(new) == pd.Timestamp(prev_date):
        return 0.0
    return calculate_year_fraction(prev_date, new, env.day_count_convention, env.bus_days_in_year,
                                   calendar=getattr(env, "calendar", None))


def at_spot(env: Any, spot: float) -> Any:
    """A copy of ``env`` quoting ``spot`` (everything else shared)."""
    return dataclasses.replace(env, spot_quote=dataclasses.replace(env.spot_quote, spot=float(spot)))


def step_fixings(product: Any, env: Any, *, prev_date: Any, prev_spot: float,
                 run_fills: Mapping[int, Any],
                 live_spots: Optional[Mapping[int, float]] = None) -> Tuple[Any, Dict[int, Any]]:
    """R3/R4 for one product, stateless (the recorder and ``ProductReplay`` share it).

    ``product`` is already aged to ``env``'s date. Returns ``(recorded, new_fills)``: ``recorded`` carries
    ``run_fills`` (decisions of earlier steps) and ``new_fills``, the records strictly past at ``env`` and still open.
    A record in ``live_spots`` was live at an earlier instant and is decided at the spot of the first instant it was
    live (Ruling 4: the last path day on or before its date); any other (a gap) at ``prev_spot``. ``run_fills`` is
    not mutated.
    """
    require_date_instant(prev_date)
    require_date_instant(env.valuation_date)
    if pd.Timestamp(env.valuation_date) < pd.Timestamp(prev_date):
        raise ValidationError(f"the run clock cannot go back: from {prev_date} to {env.valuation_date} (R9)")
    if run_fills:
        product = product.with_fixings_recorded(dict(run_fills))
    new_fills = dict(product.missing_fixing_fills(at_spot(env, prev_spot)))
    by_spot: Dict[float, List[int]] = {}
    for index in new_fills:
        if live_spots and index in live_spots:
            by_spot.setdefault(float(live_spots[index]), []).append(index)
    for spot, indices in by_spot.items():          # grouped by the stored value: one decision per distinct spot
        at_live = product.missing_fixing_fills(at_spot(env, spot))
        for index in indices:
            new_fills[index] = at_live[index]
    return (product.with_fixings_recorded(new_fills) if new_fills else product), new_fills


def update_live(open_fixings: Mapping[int, Tuple[float, str]], date: Any, spot: float,
                before: Optional[Mapping[int, Tuple[Any, float]]] = None,
                exclude=()) -> Dict[int, Tuple[Any, float]]:
    """``{index: (date, spot)}`` of the instant each live (t = 0) record first went live (Ruling 4)."""
    before = before or {}
    return {i: before.get(i, (date, float(spot)))
            for i, (t, _label) in open_fixings.items() if is_zero(t) and i not in exclude}


class RunFixingRecorder:
    """Registers a run's participating positions at its first valuation and records their fixings as the clock moves.

    Call ``register(portfolio)`` once, after the first valuation's market is applied (it replaces W5b's
    ``freeze_portfolio_fixings`` in the run engines), and ``advance(portfolio)`` after every later path instant's
    market and date are applied, before anything is priced.
    """

    def __init__(self) -> None:
        self._last: Dict[str, Tuple[Any, float]] = {}      # underlying -> (date, quoted spot) of the last instant
        self._live: Dict[str, Dict[int, Tuple[Any, float]]] = {}   # position id -> index -> (date, spot) first live
        self._days: List[RunFixingDay] = []

    @property
    def days(self) -> Tuple[RunFixingDay, ...]:
        return tuple(self._days)

    @property
    def recorded(self) -> Tuple[RecordedFixing, ...]:
        return tuple(f for day in self._days for f in day.fixings)

    @staticmethod
    def _participants(portfolio) -> List[Tuple[str, Any]]:
        positions = getattr(portfolio, "positions", None) or {}
        return [(pid, pos) for pid, pos in list(positions.items()) if participates(getattr(pos, "product", None))]

    def _register_position(self, pid: str, position: Any, env: Any) -> None:
        """R5 for one position: W5b's freeze (one ``MissingFixingWarning``) and its live set at this instant."""
        frozen = freeze_missing_fixings(position.product, env)
        if frozen is not position.product:
            position.product = frozen
        self._live[pid] = update_live(frozen.open_fixings(env), env.valuation_date, env.spot)

    def _day(self, portfolio, participants, fixings) -> RunFixingDay:
        dates = [portfolio.pricing_environments[pos.underlying].valuation_date for _, pos in participants]
        day = RunFixingDay(date=max(dates) if dates else None, fixings=tuple(fixings))
        self._days.append(day)
        return day

    def register(self, portfolio) -> RunFixingDay:
        """The run's first valuation (R5): fill strictly past open records once at its spot; remember the instant."""
        participants = self._participants(portfolio)
        envs = portfolio.pricing_environments
        for pid, position in participants:
            env = envs[position.underlying]
            self._register_position(pid, position, env)
            self._last.setdefault(position.underlying, (env.valuation_date, float(env.spot)))
        return self._day(portfolio, participants, ())

    def advance(self, portfolio) -> RunFixingDay:
        """One path instant later (R3/R4/R6/R8/R9); a position first seen here is registered here (Ruling 8)."""
        participants = self._participants(portfolio)
        envs = portfolio.pricing_environments
        fixings: List[RecordedFixing] = []
        for underlying in sorted({pos.underlying for _, pos in participants}):
            env = envs[underlying]
            if underlying in self._last:
                prev_date = self._last[underlying][0]
                require_date_instant(prev_date)
                require_date_instant(env.valuation_date)
                if pd.Timestamp(env.valuation_date) < pd.Timestamp(prev_date):
                    raise ValidationError(
                        f"the run clock cannot go back: {underlying} from {prev_date} to {env.valuation_date} (R9)")
        for pid, position in participants:
            env = envs[position.underlying]
            last = self._last.get(position.underlying)
            if pid not in self._live or last is None:
                self._register_position(pid, position, env)
                continue
            prev_date, prev_spot = last
            live = self._live[pid]
            if pd.Timestamp(env.valuation_date) == pd.Timestamp(prev_date):
                # same instant: only its spot is refreshed (Ruling 9), including records that went live on it
                day = pd.Timestamp(env.valuation_date)
                self._live[pid] = {i: ((d, float(env.spot)) if pd.Timestamp(d) == day else (d, s))
                                   for i, (d, s) in live.items()}
                continue
            aged = position.product.with_time_elapsed(run_step_dt(prev_date, env))
            labels = aged.open_fixings(env)
            recorded, new = step_fixings(aged, env, prev_date=prev_date, prev_spot=prev_spot, run_fills={},
                                         live_spots={i: s for i, (_d, s) in live.items()})
            for index in sorted(new):
                fixings.append(RecordedFixing(position_id=pid, index=index, label=labels[index][1],
                                              outcome=new[index],
                                              spot=float(live[index][1]) if index in live else float(prev_spot),
                                              gap=index not in live))
            self._live[pid] = update_live(labels, env.valuation_date, env.spot, before=live, exclude=new)
            if recorded is not position.product:
                position.product = recorded
        for _, position in participants:
            env = envs[position.underlying]
            self._last[position.underlying] = (env.valuation_date, float(env.spot))
        live_ids = {pid for pid, _ in participants}
        for pid in [p for p in self._live if p not in live_ids]:
            del self._live[pid]
        return self._day(portfolio, participants, fixings)
```

Note on the live set after a step: `labels` was taken on the aged product BEFORE recording, so its t = 0 records minus the ones just recorded are exactly the records live at P' (the new fills are all `t < 0`, so `exclude=new` is defensive); a record already live keeps its first-live `(date, spot)` (Ruling 4).

CHANGELOG `### Added`:

```markdown
- `quantark.util.run_fixings` (spec `docs/superpowers/specs/2026-10-09-run-fixings-design.md`): `RunFixingRecorder`
  (`register` at a run's first valuation = W5b's freeze; `advance` at every later path instant records each fixing
  the clock moved past at the previous instant's spot, `gap=True` when no instant fell on it; numeric contracts age
  by the environment's year fraction; time-of-day instants and a backwards clock are refused), the stateless
  `step_fixings`, `participates`, `run_step_dt`, `RecordedFixing`, `RunFixingDay`.
```

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/util/run_fixings.py test/test_run_fixings.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): RunFixingRecorder and the stateless step_fixings core

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Recorder hooks for product-held run state (W4b's interface)

Spec §6 (Ruling 11). Generic; stub-tested; no TARN code.

**Files:**
- Modify: `quantark/util/run_fixings.py`, `CHANGELOG.md`
- Test: `test/test_run_fixings.py`

**Interfaces:**
- Consumes: Task 4.
- Produces (optional product hooks, each used only when present):
  - `matured_before(env) -> bool` — at `register` (and first-sight registration): True -> `MaturedBeforeRunWarning` (lazy import from `quantark.asset.equity.lifecycle.autocallable`), `portfolio.remove_position(pid)`, the position is not registered.
  - `lifecycle_events: Sequence[event]` attribute — at `advance`, `RunFixingDay.product_events` gets `(pid, position.quantity, event)` for every event the step's final product holds beyond the product at the previous instant (`events[len(prev.lifecycle_events):]`).
  - `pending_pv(env) -> float` — `RunFixingDay.pending_pv = Σ quantity × product.pending_pv(env)` over participants exposing it, at `register` and `advance`.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ---------------------------------------------------------------- Task 5: hooks for product-held state (spec §6)
class _Stub:
    """A participating product holding run state, like W4b's TARN: events appended when a fixing is recorded."""

    has_run_fixings = True

    def __init__(self, matured=False, events=(), recorded=()):
        self.matured, self.lifecycle_events, self.recorded = matured, list(events), dict(recorded)

    def matured_before(self, env):
        return self.matured

    def missing_fixing_fills(self, env):
        return {0: float(env.spot)} if 0 not in self.recorded and env.valuation_date > F.START + timedelta(days=4) else {}

    def open_fixings(self, env):
        return {} if 0 in self.recorded else {0: (0.0 if env.valuation_date == F.START + timedelta(days=4) else 1.0, "#0")}

    def with_fixings_recorded(self, fills):
        if not fills or 0 in self.recorded:
            return self
        return _Stub(self.matured, self.lifecycle_events + [("FIXING", fills[0])], {**self.recorded, **fills})

    def with_time_elapsed(self, dt):
        return self

    def with_missing_fixings_filled(self, env, *, warn=True):
        return self

    def pending_pv(self, env):
        return 2.0 * len(self.lifecycle_events)


class _DuckBook:
    """The duck-typed portfolio the recorder needs (positions, pricing_environments, remove_position)."""

    def __init__(self, **products):
        from types import SimpleNamespace
        self.pricing_environments = {F.UNDERLYING: F.eq_env(0)}
        self.positions = {pid: SimpleNamespace(product=p, quantity=q, underlying=F.UNDERLYING)
                          for pid, (p, q) in products.items()}

    def remove_position(self, pid):
        return self.positions.pop(pid, None)


def test_hooks_matured_events_pending_pv():
    from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
    pf = _DuckBook(keep=(_Stub(), -3.0), gone=(_Stub(matured=True), 1.0))
    rec = RunFixingRecorder()
    with pytest.warns(MaturedBeforeRunWarning):
        day0 = rec.register(pf)
    assert "gone" not in pf.positions and day0.pending_pv == 0.0
    days = []
    for k in range(1, 7):
        F.move(pf, k)
        days.append(rec.advance(pf))
    assert days[4].product_events == (("keep", -3.0, ("FIXING", 105.0)),)  # the step to day 5
    assert days[4].pending_pv == -3.0 * 2.0 and days[5].product_events == ()
```

- [ ] **Step 2: Run to verify it fails**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -k hooks -n 2 -q`
Expected: FAIL (`DID NOT WARN MaturedBeforeRunWarning`).

- [ ] **Step 3: Implement** in `run_fixings.py`

In `_register_position`, return a bool and check maturity first:

```python
    def _register_position(self, portfolio, pid: str, position: Any, env: Any) -> bool:
        """R5 for one position; False when it matured before the run and left the working portfolio (spec §6)."""
        matured_before = getattr(position.product, "matured_before", None)
        if callable(matured_before) and matured_before(env):
            from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning  # lazy: util -> asset
            warnings.warn(MaturedBeforeRunWarning(
                f"position {pid} ({type(position.product).__name__}) matured before the run start "
                f"{env.valuation_date}: it is not settled in the run, books no cash and is valued at zero"),
                stacklevel=3)
            portfolio.remove_position(pid)
            return False
        frozen = freeze_missing_fixings(position.product, env)
        if frozen is not position.product:
            position.product = frozen
        self._live[pid] = update_live(frozen.open_fixings(env), env.valuation_date, env.spot)
        return True
```

(`import warnings` at the top; update both call sites: `register` skips `_last.setdefault` when it returns False and re-reads `participants` before `_day`; `advance` passes `portfolio`.) In `advance`, keep `before = position.product` before ageing and after recording:

```python
            prev_events = getattr(before, "lifecycle_events", None)
            new_events = getattr(recorded, "lifecycle_events", None)
            if prev_events is not None and new_events is not None:
                events.extend((pid, float(position.quantity), ev) for ev in list(new_events)[len(prev_events):])
```

and `_day(portfolio, participants, fixings, events)` builds `RunFixingDay(..., product_events=tuple(events), pending_pv=...)` with

```python
        pending = 0.0
        for pid, pos in participants:
            hook = getattr(pos.product, "pending_pv", None)
            if pid in portfolio.positions and callable(hook):
                pending += float(pos.quantity) * float(hook(portfolio.pricing_environments[pos.underlying]))
```

CHANGELOG: append to the Task 4 bullet: "Optional product hooks `matured_before(env)`, `lifecycle_events` and `pending_pv(env)` feed `RunFixingDay.product_events` / `pending_pv` (for FX TARN / TARF, W4b)."

- [ ] **Step 4: Run to verify**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/util/run_fixings.py test/test_run_fixings.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): recorder hooks for product-held run state (matured_before, events, pending PV)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Equity backtest records run fixings

Spec §3.5 row 1, X1a–c, X2, X3b (weekly), X3e (`H`), X3g, X6; §8 "no roll rule" warning.

**Files:**
- Modify: `quantark/backtest/equity/engine.py:23,95-97,252-260`, `quantark/asset/equity/lifecycle/manager.py:181-198`, `CHANGELOG.md`
- Create: `test/test_run_fixings_runs.py`

**Interfaces:**
- Consumes: `RunFixingRecorder`, `participates` (Task 4).
- Produces: `BacktestEngine.run_fixings: RunFixingRecorder` (fresh in `_initialize`, read after `run()`); `PortfolioLifecycleManager._roll_untracked` returns `None` silently for participants.

- [ ] **Step 1: Write the failing tests** — create `test/test_run_fixings_runs.py`

```python
"""Run fixings through every run path (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md §7 X1-X3, X6)."""
import copy
import warnings
from datetime import timedelta

import pytest

import run_fixings_fixtures as F
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning
from quantark.util.numerical import is_close


def _rows(recorder, pid=None):
    return [(f.index, f.outcome, f.spot, f.gap) for f in recorder.recorded if pid is None or f.position_id == pid]


def _hand_rows(kind, spot_of=F.spot_on):
    """A DAYS-day run's recordings: every fixing dated on/before day DAYS - 2 (the last day's is still live), at its
    own day's spot, never a gap."""
    outs = F.hand_outcomes(kind, F.DAYS - 2, spot_of)
    return [(i, outs[i], spot_of(F.day_of(F.fixing_dates()[i])), False) for i in sorted(outs)]


# ================================================================ equity backtest
def _backtest(product, kind, adapter, tmp_path, frequency="D"):
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.portfolio import Position

    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=F.START,
        end_date=adapter.dates[-1].to_pydatetime(), underlying=F.UNDERLYING,
        initial_positions=[Position(product=product, quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING,
                                    engine=F.ENGINES[kind](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
        frequency=frequency, results_path=str(tmp_path),
    )
    engine = BacktestEngine(cfg)
    return engine, engine.run()


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_backtest_x1_x2_x6(kind, tmp_path):
    booked = F.MAKERS[(kind, "dated")]()
    before = copy.deepcopy(booked)
    (eng_d, res_d), warns = F.count_missing(_backtest, booked, kind, F.DailyAdapter(F.DAYS), tmp_path)
    eng_n, res_n = _backtest(F.MAKERS[(kind, "numeric")](), kind, F.DailyAdapter(F.DAYS), tmp_path)
    values_d, values_n = res_d.get_value_series().tolist(), res_n.get_value_series().tolist()
    engine = F.ENGINES[kind]()
    for k in range(F.DAYS):                                                      # X1a: bitwise vs by hand
        ref = F.quiet_price(engine, F.by_hand(kind, k - 1), F.eq_env(k))
        assert values_d[k] == ref, (k, values_d[k], ref)
        assert is_close(values_n[k], values_d[k], rel_tol=1e-12)                 # X2 values
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _hand_rows(kind)  # X2 sequences, X1b spot
    pid = list(eng_d.portfolio.positions)[0]                                     # X6
    assert eng_d.portfolio.positions[pid].product.observation_records == F.by_hand(kind, F.DAYS - 2).observation_records
    assert booked.observation_records == before.observation_records             # X3g


def test_equity_backtest_x1b_the_fixing_does_not_move_later(tmp_path):
    other = lambda k: F.spot_on(k) if k <= 5 else 80.0                          # noqa: E731  differs from day 6 on
    a, _ = _backtest(F.ra_dated(), "ra", F.DailyAdapter(F.DAYS), tmp_path)
    b, _ = _backtest(F.ra_dated(), "ra", F.DailyAdapter(F.DAYS, spot_of=other), tmp_path)
    assert _rows(a.run_fixings)[0] == _rows(b.run_fixings)[0] == (0, True, 105.0, False)


def test_equity_backtest_x3b_weekly_gap_takes_the_previous_instant(tmp_path):
    adapter = F.DailyAdapter(5, freq="W-MON")                                  # days 0, 7, 14, 21, 28
    (eng, res), warns = F.count_missing(_backtest, F.ra_dated(), "ra", adapter, tmp_path, "W")
    assert warns == 0
    assert _rows(eng.run_fixings) == [(0, True, 105.0, True), (1, False, 115.0, True)]
    assert res.get_value_series().tolist()[3] == F.quiet_price(F.ENGINES["ra"](), F.ra_dated({0: True, 1: False}),
                                                               F.eq_env(21))


def test_equity_backtest_x3e_hourly_refuses_a_participant(tmp_path):
    with pytest.raises(ValidationError, match="time of day"):
        _backtest(F.ra_dated(), "ra", F.DailyAdapter(3, freq="h"), tmp_path, "H")


def test_equity_backtest_numeric_participant_no_longer_warns_constant_maturity(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _backtest(F.ra_numeric(), "ra", F.DailyAdapter(6), tmp_path)
    assert not [w for w in caught if "no roll rule" in str(w.message)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py -n 2 -q`
Expected: FAIL — `AttributeError: 'BacktestEngine' object has no attribute 'run_fixings'` (and X1a mismatches from day 10, §2.2).

- [ ] **Step 3: Implement**

`quantark/backtest/equity/engine.py`: replace the import `from quantark.util.fixings import freeze_portfolio_fixings` with `from quantark.util.run_fixings import RunFixingRecorder`; in `__init__` replace `self._past_fixings_frozen: bool = False` with `self.run_fixings: RunFixingRecorder = RunFixingRecorder()` and `self._fixings_registered: bool = False`; in `_initialize` (first line) reset both (`self.run_fixings = RunFixingRecorder(); self._fixings_registered = False`). Replace lines 254-260:

```python
        # Run fixings (spec 2026-10-09 run fixings §3.5): the first market valuation is W5b's freeze (a fixing already
        # missing is filled once at its spot, R5); every later step records the fixings the clock moved past at the
        # previous step's spot (R3/R4). Before the explain recorder captures today's contracts.
        if not self._fixings_registered:
            self.run_fixings.register(self.portfolio)
            self._fixings_registered = True
        else:
            self.run_fixings.advance(self.portfolio)
```

(`grep -rn "_past_fixings_frozen" <WT>/quantark <WT>/test` must return nothing afterwards.)

`quantark/asset/equity/lifecycle/manager.py`: `from quantark.util.run_fixings import participates`; in `_roll_untracked`, first lines after `product = position.product`:

```python
        if participates(product):
            return None          # aged and fixed by the run's RunFixingRecorder (spec 2026-10-09 run fixings §3.5)
```

Update the module docstring sentence "Untracked products with a float maturity and no roll rule (schedule-bearing contracts) are repriced with a constant maturity and warn once" to add "; run-fixings participants (range accruals, record Asians) are aged by ``quantark.util.run_fixings`` instead".

CHANGELOG `### Changed`:

```markdown
- The equity `BacktestEngine` records a range accrual's / record Asian's fixing once, at the spot of the run day it
  was live, when the clock moves past it (`BacktestEngine.run_fixings`), instead of re-deciding it at every later
  day's spot; numeric contracts age with their records (no more constant-maturity "no roll rule" warning for them).
  A run whose instants carry a time of day (`frequency="H"`) and holds such a product is refused. Measured on the
  spec fixture: a dated range accrual on day 10, 2.744374 -> 4.649441 (= the contract with the day-4 fixing recorded).
```

- [ ] **Step 4: Run to verify they pass, plus neighbours**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py test/test_missing_fixings_scenarios.py test/test_float_maturity_roll.py test/test_backtest_lifecycle.py test/test_pnlexplain_backtest_equity.py -n 2 -q`
Expected: all pass. If a value in `test_missing_fixings_scenarios.py::test_equity_backtest_fills_once_at_the_first_valuation` moves it is not asserted (both arms age identically); if any assertion there fails, STOP and report.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/backtest/equity/engine.py quantark/asset/equity/lifecycle/manager.py test/test_run_fixings_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): the equity backtest records a fixing on the run day it passes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Equity dynamic scenario records run fixings

Spec §3.5 row 3, X1, X2, X3a, X3b (numeric gap), X3c, X3d, X3e, X3g, X3h, X6.

**Files:**
- Modify: `quantark/dynamicscenario/engine.py:49,114-119,140-160,184-188`, `CHANGELOG.md`
- Test: `test/test_run_fixings_runs.py`

**Interfaces:**
- Consumes: `RunFixingRecorder`.
- Produces: `DynamicScenarioEngine.run_fixings: RunFixingRecorder` (the last run's).

- [ ] **Step 1: Write the failing tests** (append)

```python
# ================================================================ equity dynamic scenario
def _eq_dynamic(products, spot_of=F.spot_on, n=F.DAYS, start=F.START, env=None):
    import contextlib
    import io

    from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine
    from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
    from quantark.stresstest.stress.stress_types import StressType

    pf = F.eq_book(*products, env=env)
    steps = [DayStep(day_index=k, changes=[ParameterChange("spot", StressType.VALUE, spot_of(k))]) for k in range(n)]
    engine = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=True, handle_lifecycle_events=False))
    with contextlib.redirect_stdout(io.StringIO()):
        res = engine.run(pf, DayPath(name="w6", steps=steps, start_date=start))
    return engine, res, pf


def _greeks(product, env):
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    from quantark.dynamicscenario import DynamicScenarioConfig
    use_analytical = DynamicScenarioConfig().greeks_method == "analytical"      # the run's own route
    return F.eq_book(product, env=env).get_portfolio_greeks(GreeksCalculator(), use_analytical=use_analytical)


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_equity_dynamic_x1_x2_x3g_x6(kind):
    booked = F.MAKERS[(kind, "dated")]()
    before = copy.deepcopy(booked)
    (eng_d, res_d, pf), warns = F.count_missing(_eq_dynamic, [booked])
    eng_n, res_n, _ = _eq_dynamic([F.MAKERS[(kind, "numeric")]()])
    engine = F.ENGINES[kind]()
    for k in range(F.DAYS):
        ref = F.quiet_price(engine, F.by_hand(kind, k - 1), F.eq_env(k))
        assert res_d.day_results[k].portfolio_value == ref, k                    # X1a
        assert is_close(res_n.day_results[k].portfolio_value, ref, rel_tol=1e-12)  # X2
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _hand_rows(kind)
    assert pf.positions[list(pf.positions)[0]].product is booked                 # X3g: input book untouched
    assert booked.observation_records == before.observation_records


def test_equity_dynamic_x3h_greeks_equal_the_hand_recorded_contract():
    _, res, _ = _eq_dynamic([F.ra_dated()])
    for k in (5, 10, 14):
        got, want = res.day_results[k].greeks, _greeks(F.by_hand("ra", k - 1), F.eq_env(k))
        assert {g: got[g] for g in ("delta", "gamma", "vega", "theta")} == {g: want[g] for g in ("delta", "gamma", "vega", "theta")}


def test_equity_dynamic_x3a_todays_fixing_is_live_then_history_at_its_day():
    spot_of = lambda k: 109.9 if k == F.FIX_DAY else F.spot_on(k)               # noqa: E731
    eng, res, _ = _eq_dynamic([F.ra_dated()], spot_of=spot_of, n=8)
    live = _greeks(F.ra_dated(), F.eq_env(F.FIX_DAY, spot=109.9))
    assert res.day_results[F.FIX_DAY].greeks["delta"] == live["delta"]
    from quantark.asset.equity.riskmeasures import GreeksCalculator
    g1 = GreeksCalculator().calculate(F.ra_dated(), F.eq_env(F.FIX_DAY, spot=109.9), F.ENGINES["ra"]())
    assert is_close(g1["delta"], -1.057904, rel_tol=0.0, abs_tol=5e-7)          # the spec's G1 number
    assert _rows(eng.run_fixings)[0] == (0, True, 109.9, False)


def test_equity_dynamic_x3b_a_numeric_fixing_between_days_takes_the_previous_day():
    times = [4.5 / 365.0, 18 / 365.0, 120 / 365.0, 240 / 365.0, 1.0]
    eng, _, _ = _eq_dynamic([F.ra_numeric(times=times)], n=7, spot_of=lambda k: 100.0 + k)
    assert _rows(eng.run_fixings) == [(0, True, 104.0, True)]


def test_equity_dynamic_x3c_history_warns_once_and_the_first_day_is_live():
    dates = [F.START - timedelta(days=3), F.START] + F.fixing_dates()[1:]
    prod = F.ra_dated(dates=dates, initial_date=F.START - timedelta(days=10))
    (eng, _, _), warns = F.count_missing(_eq_dynamic, [prod], n=3)
    assert warns == 1
    assert _rows(eng.run_fixings) == [(1, True, 105.0, False)]


def test_equity_dynamic_x3d_undated_path_records_nothing_and_values_as_frozen_at_start():
    eng, res, _ = _eq_dynamic([F.ra_dated()], start=None)
    assert eng.run_fixings.recorded == ()
    for k in range(F.DAYS):
        assert res.day_results[k].portfolio_value == F.quiet_price(F.ENGINES["ra"](), F.ra_dated(),
                                                                   F.eq_env(0, spot=F.spot_on(k)))


def test_equity_dynamic_x3e_time_stamped_dates_refuse_a_participant():
    with pytest.raises(ValidationError, match="time of day"):
        _eq_dynamic([F.ra_dated()], start=F.START + timedelta(hours=15), n=3)
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py -k equity_dynamic -n 2 -q`
Expected: FAIL — `AttributeError: ... 'run_fixings'`.

- [ ] **Step 3: Implement** in `quantark/dynamicscenario/engine.py`

- Import: replace `from quantark.util.fixings import freeze_portfolio_fixings` with `from quantark.util.run_fixings import RunFixingRecorder`; in `__init__` add `self.run_fixings: RunFixingRecorder = RunFixingRecorder()`.
- Delete lines 114-119 (the TODO comment and `portfolio = freeze_portfolio_fixings(portfolio)`).
- Immediately before `baseline_value = working_portfolio.get_portfolio_value()` (line 159):

```python
        # Run fixings (spec 2026-10-09 run fixings §3.5): registered on the WORKING copy at the first valuation (the
        # input portfolio is never touched); a fixing already missing is filled once at the baseline spot (R5).
        self.run_fixings = RunFixingRecorder()
        self.run_fixings.register(working_portfolio)
```

- After the `if day_date: for env in ...: env.valuation_date = day_date` block (line 187), before lifecycle processing:

```python
            if day_date:
                # the clock moved: record the fixings it passed at the previous instant's spot (R3/R4); an undated
                # path never moves the date, so nothing passes (R7)
                self.run_fixings.advance(working_portfolio)
```

CHANGELOG `### Changed`: "`DynamicScenarioEngine` (equity): the same run-fixings recording on a dated path (`DynamicScenarioEngine.run_fixings`); registered on the working copy, so the input portfolio is never touched; numeric range accruals / record Asians now age on a dated path (they were frozen)."

- [ ] **Step 4: Run to verify**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py test/test_missing_fixings_scenarios.py test/test_dynamic_scenario*.py -n 2 -q` (check the glob matches files: `ls test/test_dynamic_scenario*.py`)
Expected: all pass. `test_equity_dynamic_scenario_fills_once_at_the_first_valuation` keeps `count == 3`; if the count moves, STOP and report.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/dynamicscenario/engine.py test/test_run_fixings_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): the equity dynamic scenario records a fixing on the day it passes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: FX backtest and FX dynamic scenario record run fixings

Spec §3.5 rows 2 and 4, X1, X2 (incl. `BUSINESS_DAYS`), X3b (Saturday record), X3d, X3g, X6; Rulings 2 and 4; Review Focus 4.

**Files:**
- Modify: `quantark/backtest/fx/engine.py:20,42-50,63-64`, `quantark/dynamicscenario/fx/engine.py:29,66-71,84-96`, `CHANGELOG.md`
- Test: `test/test_run_fixings_runs.py`

**Interfaces:**
- Consumes: `RunFixingRecorder`.
- Produces: `FXBacktestEngine.run_fixings`, `FXDynamicScenarioEngine.run_fixings` (W4b Tasks 10–11 read `RunFixingDay` from them).

- [ ] **Step 1: Write the failing tests** (append)

```python
# ================================================================ FX backtest + FX dynamic scenario
import pandas as pd  # noqa: E402

FX_KINDS = ["domestic", "foreign", "quanto"]


def _fx_backtest(products, index=None, env=None, spot_of=F.fx_spot):
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy
    from quantark.backtest.transaction_costs import ZeroCostModel

    index = index if index is not None else pd.date_range(F.START, periods=F.DAYS, freq="D")
    cfg = FXBacktestConfig(portfolio=F.fx_book(*products, env=env), market_path=F.fx_path(index, spot_of),
                           strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
                           transaction_cost_model=ZeroCostModel())
    engine = FXBacktestEngine(cfg)
    return engine, [r["portfolio_value"] for r in engine.run().rows], cfg.portfolio


def _fx_dynamic(products, start=F.START, env=None):
    from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
    from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
    from quantark.stresstest.stress.stress_types import StressType

    steps = [DayStep(day_index=k, changes=[ParameterChange("spot", StressType.VALUE, F.fx_spot(k))])
             for k in range(F.DAYS)]
    engine = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=True))
    pf = F.fx_book(*products, env=env)
    res = engine.run(pf, DayPath(name="w6fx", steps=steps, start_date=start))
    return engine, [d.portfolio_value for d in res.day_results], pf


def _fx_hand_rows(kind):
    outs = F.fx_hand_outcomes(kind, F.DAYS - 2)
    return [(i, outs[i], F.fx_spot(F.day_of(F.fixing_dates()[i])), False) for i in sorted(outs)]


@pytest.mark.parametrize("run", [_fx_backtest, _fx_dynamic], ids=["backtest", "dynamic"])
@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_runs_x1_x2_x3g_x6(run, kind):
    q = kind == "quanto"
    booked = F.fx_ra(kind, "dated")
    before = copy.deepcopy(booked)
    (eng_d, values_d, pf), warns = F.count_missing(run, [booked], env=F.fx_env(0, quanto=q))
    eng_n, values_n, _ = run([F.fx_ra(kind, "numeric")], env=F.fx_env(0, quanto=q))
    engine = F.FX_CLASSES[kind][1]()
    for k in range(F.DAYS):
        ref = F.quiet_price(engine, F.fx_ra(kind, "dated", F.fx_hand_outcomes(kind, k - 1)), F.fx_env(k, quanto=q))
        assert values_d[k] == ref, (k, values_d[k], ref)                        # X1a
        assert is_close(values_n[k], ref, rel_tol=1e-12)                         # X2
    assert warns == 0                                                            # X1c
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) == _fx_hand_rows(kind)
    assert pf.positions[list(pf.positions)[0]].product is booked
    assert booked.observation_records == before.observation_records             # X3g


def test_fx_backtest_x1b_the_fixing_does_not_move_later():
    other = lambda k: F.fx_spot(k) if k <= 5 else 1.0                           # noqa: E731
    a, _, _ = _fx_backtest([F.fx_ra("domestic", "dated")])
    b, _, _ = _fx_backtest([F.fx_ra("domestic", "dated")], spot_of=other)
    assert _rows(a.run_fixings)[0] == _rows(b.run_fixings)[0] == (0, True, 1.25, False)


def test_fx_backtest_x3b_a_saturday_fixing_takes_fridays_spot_with_gap():
    sat = F.START + timedelta(days=5)                                           # 2024-01-06
    prod = F.fx_ra("domestic", "dated", dates=[sat] + F.fixing_dates()[1:])
    index = pd.bdate_range(F.START, periods=10)
    (eng, _, _), warns = F.count_missing(_fx_backtest, [prod], index=index, spot_of=lambda k: 1.20 + 0.01 * k)
    assert warns == 0
    assert _rows(eng.run_fixings)[0] == (0, True, 1.24, True)                  # Friday (day 4) spot, on Monday


def _business_env(day=0):
    from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    return F.fx_env(day, day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal), cal


def _business_twin(dates):
    from quantark.util.calendar import DayCountConvention, calculate_year_fraction
    _, cal = _business_env()
    times = [calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal) for d in dates]
    return F.fx_ra("domestic", "numeric", times=times), F.fx_ra("domestic", "dated", dates=dates)


def test_fx_backtest_x2_on_a_business_day_clock():
    numeric, dated = _business_twin(F.fixing_dates())
    env, _ = _business_env()
    eng_d, values_d, _ = _fx_backtest([dated], index=pd.bdate_range(F.START, periods=18), env=env)
    eng_n, values_n, _ = _fx_backtest([numeric], index=pd.bdate_range(F.START, periods=18), env=env)
    assert _rows(eng_d.run_fixings) == _rows(eng_n.run_fixings) and _rows(eng_d.run_fixings)
    assert all(is_close(a, b, rel_tol=1e-12) for a, b in zip(values_d, values_n))


def test_business_day_clock_on_a_calendar_path_records_at_the_last_live_instant():
    """Review Focus 4 / Ruling 4: Friday's fixing is t = 0 through the weekend on a business-day clock."""
    numeric, dated = _business_twin(F.fixing_dates())
    env, _ = _business_env()
    spot_of = lambda k: 1.25 + 0.001 * k                                        # noqa: E731
    eng_d, _, _ = _fx_backtest([dated], env=env, spot_of=spot_of, index=pd.date_range(F.START, periods=10))
    eng_n, _, _ = _fx_backtest([numeric], env=env, spot_of=spot_of, index=pd.date_range(F.START, periods=10))
    assert _rows(eng_d.run_fixings)[0] == _rows(eng_n.run_fixings)[0] == (0, True, spot_of(4), False)   # Friday's


def test_fx_dynamic_x3d_undated_path_records_nothing():
    eng, values, _ = _fx_dynamic([F.fx_ra("domestic", "dated")], start=None)
    assert eng.run_fixings.recorded == ()
    assert values[10] == F.quiet_price(F.FX_CLASSES["domestic"][1](), F.fx_ra("domestic", "dated"),
                                       F.fx_env(0, spot=F.fx_spot(10)))
```

`_fx_backtest` / `_fx_dynamic` take an `env` keyword (add `env=None` to `_fx_dynamic`'s parameters as shown; `F.count_missing(run, [booked], env=...)` forwards it).

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py -k fx -n 2 -q`
Expected: FAIL — `AttributeError: ... 'run_fixings'`.

- [ ] **Step 3: Implement**

`quantark/backtest/fx/engine.py`: import `RunFixingRecorder` instead of `freeze_portfolio_fixings`; `self.run_fixings = RunFixingRecorder()` in `__init__`; replace lines 46-50 (comment + freeze) with

```python
        # Run fixings (spec 2026-10-09 run fixings §3.5): the first row is W5b's freeze (R5); each row's ``advance``
        # records the fixings the clock moved past at the previous row's quoted spot (the first row repeats the date
        # and only refreshes it).
        self.run_fixings = RunFixingRecorder()
        self.run_fixings.register(portfolio)
```

and after `self._apply_row(portfolio, pairs, row, ts)` in the loop (line 64): `self.run_fixings.advance(portfolio)`.

`quantark/dynamicscenario/fx/engine.py`: same import; delete lines 66-71 (comment + freeze); before `baseline_value = working.get_portfolio_value()`:

```python
        self.run_fixings = RunFixingRecorder()
        self.run_fixings.register(working)        # the working copy: the input portfolio is never touched
```

and after the `env.valuation_date = day_date` loop: `if day_date is not None: self.run_fixings.advance(working)`. Add `self.run_fixings = RunFixingRecorder()` in `__init__`.

CHANGELOG `### Changed`: "`FXBacktestEngine` and `FXDynamicScenarioEngine` record FX range accrual fixings (domestic, foreign, quanto) on the run day they pass, at the quoted spot (`run_fixings`); numeric FX range accruals age in FX runs (they were frozen). FX range accrual on the spec fixture, day 10: 11,385.19 -> 19,005.46."

- [ ] **Step 4: Run to verify**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py test/test_missing_fixings_scenarios.py test/test_fx_backtest*.py test/test_fx_dynamic*.py -n 2 -q` (check globs with `ls`)
Expected: all pass; `test_fx_dynamic_scenario_fills_once_at_the_first_valuation` keeps `count == 2` (else STOP).

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/backtest/fx/engine.py quantark/dynamicscenario/fx/engine.py test/test_run_fixings_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): FX backtest and FX dynamic scenario record a fixing on the day it passes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Replay records run fixings

Spec §3.5 row 5, §3.6, X1 (Asian, drift 1), X2, X3g, X6.

**Files:**
- Modify: `quantark/asset/equity/lifecycle/autocallable.py:242-250`, `quantark/backtest/replay/product_replay.py:36,190-225,535-552`, `quantark/backtest/replay/engine.py:43-55`, `CHANGELOG.md`
- Test: `test/test_run_fixings_runs.py`

**Interfaces:**
- Consumes: `participates`, `step_fixings` (Task 4); `with_time_elapsed` (Task 1).
- Produces: `ProductReplay._last: Optional[Tuple[pd.Timestamp, float]]` and `ProductReplay._live: Dict[int, Tuple]` (Ruling 4, via Task 4 `update_live`) beside `_run_fills`; `product_for_date` steps when a later date is first seen.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ================================================================ replay (the Asian: no range-accrual route, drift 1)
def _replay(product, n=F.DAYS, first=0, spot_of=F.spot_on, explain=None):
    from replay_golden import fixtures

    from quantark.backtest.replay.config import AutocallableEngineConfig
    from quantark.backtest.replay.engine import ReplayBacktestEngine
    from quantark.backtest.replay.market import AutocallableMarketDataSet
    from quantark.util.enum.engine_enums import EngineType

    dates = pd.date_range(F.START + timedelta(days=first), periods=n, freq="D")
    spots = [spot_of(first + k) for k in range(n)]
    fut = [{"date": d, "contract": "IF2403", "futures_price": s * 1.004, "expiry_date": pd.Timestamp("2024-03-15"),
            "multiplier": 300.0} for d, s in zip(dates, spots)]
    cfg = fixtures.make_book_config()
    cfg.market_data = AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": spots}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [F.VOL] * n}),
        rate_data=pd.DataFrame({"date": dates, "rate": [F.RATE] * n}), futures_data=pd.DataFrame(fut))
    cfg.products = [fixtures.BookProduct(product=product, quantity=1.0, position_id=1, has_lifecycle=False)]
    cfg.calculate_event_probabilities = False
    cfg.engine_config = AutocallableEngineConfig(pricing_engine_type=EngineType.ANALYTICAL)
    if explain is not None:
        cfg.pnl_explain = explain
    engine = ReplayBacktestEngine(cfg)
    res = engine.run()
    return engine, res, res.states_df()["product_mtm"].tolist()


def test_replay_x1_x2_x3g_x6():
    booked = F.asian_dated()
    before = copy.deepcopy(booked)
    (eng_d, _, mtm_d), warns = F.count_missing(_replay, booked)
    eng_n, _, mtm_n = _replay(F.asian_numeric())
    for k in (0, 3, 4, 5, 10, 14, 18, 20, 24):
        _, _, window = _replay(F.by_hand("asian", k - 1), n=2, first=k)          # X1a: a window starting at k
        assert is_close(mtm_d[k], window[0], rel_tol=1e-12), k
        assert is_close(mtm_n[k], mtm_d[k], rel_tol=1e-12), k                    # X2
    assert warns == 0                                                            # X1c
    fills_d, fills_n = eng_d._replays[0]._run_fills, eng_n._replays[0]._run_fills
    assert fills_d == fills_n == F.hand_outcomes("asian", F.DAYS - 2)            # X1b / X2 / X6
    assert booked.observation_records == before.observation_records             # X3g


def test_replay_x1b_the_fixing_does_not_move_later():
    other = lambda k: F.spot_on(k) if k <= 5 else 80.0                          # noqa: E731
    a, _, _ = _replay(F.asian_dated(), n=10)
    b, _, _ = _replay(F.asian_dated(), n=10, spot_of=other)
    assert a._replays[0]._run_fills[0] == b._replays[0]._run_fills[0] == 105.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py -k replay -n 2 -q`
Expected: FAIL (day 10: run − window ≈ +0.946551, §2.2 E3; numeric twin 7.807415 vs 7.686469 on day 4).

- [ ] **Step 3: Implement**

`autocallable.py` `product_for_pricing` (lines 242-249), replace the maturity block:

```python
        if (
            getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity", None) is not None
            and origin is not None
        ):
            if participates(product):
                # a run-fixings product's records move with its maturity (spec 2026-10-09 run fixings §3.5); the
                # trackers' floor is kept (Ruling 12)
                product = product.with_time_elapsed(elapsed)
                product.maturity = max(float(product.maturity), 1e-8)
            else:
                product.maturity = max(float(product.maturity) - elapsed, 1e-8)
```

(`from quantark.util.run_fixings import participates`. `with_time_elapsed(0.0)` returns the tracker's own deep copy, which this method owns.)

`product_replay.py`: import `from quantark.util.run_fixings import participates, step_fixings, update_live`; in `__init__` after `self._run_fills` add `self._last: Optional[tuple] = None  # (date, spot) of the last date product_for_date first saw` and `self._live: Dict[int, tuple] = {}  # index -> (date, spot) first live (Ruling 4)`. Replace `product_for_date`:

```python
    def product_for_date(self, date: pd.Timestamp, pricing_env: PricingEnvironment, *, initial: bool = False):
        """The pricing copy for ``date``; ``initial`` before the run's first close is observed.

        The first call is the run's first valuation: a fixing already missing is decided once at its spot (W5b,
        R5). Each later date seen for the first time records the fixings that became history since the last date at
        that date's spot (spec 2026-10-09 run fixings §3.6, R3/R4); every pricing copy carries every decision.
        """
        product = self._tracker.product_for_pricing(date, pricing_env, initial=initial)
        day = pd.Timestamp(date).normalize()
        if self._run_fills is None:
            fills_of = getattr(product, "missing_fixing_fills", None)
            self._run_fills = dict(fills_of(pricing_env)) if fills_of is not None else {}
            self._last = (day, float(pricing_env.spot))
            frozen = freeze_missing_fixings(product, pricing_env) if self._run_fills else product
            if participates(frozen):
                self._live = update_live(frozen.open_fixings(pricing_env), day, pricing_env.spot)
            return frozen
        if participates(product) and day != self._last[0]:
            seen = product.with_fixings_recorded(self._run_fills) if self._run_fills else product
            labels = seen.open_fixings(pricing_env)
            product, new = step_fixings(product, pricing_env, prev_date=self._last[0], prev_spot=self._last[1],
                                        run_fills=self._run_fills,
                                        live_spots={i: s for i, (_d, s) in self._live.items()})
            self._run_fills.update(new)
            self._live = update_live(labels, day, pricing_env.spot, before=self._live, exclude=new)
            self._last = (day, float(pricing_env.spot))
            return product
        if self._run_fills:
            product = product.with_fixings_recorded(self._run_fills)
        return product
```

`carry_price_callback` (lines 546-550): replace the comment with

```python
        # The day's contract carries every fixing the run has recorded (product_for_date); this freeze finds nothing
        # strictly past, and a fixing dated today stays live under the carry bumps (D1).
```

`replay/engine.py` `_book_price_callback` (lines 50-52): same comment; drop the TODO line.

CHANGELOG `### Changed`: "The replay book records a fixing on the date it passes (at that date's spot) and ages a numeric record Asian's records with its maturity (spec fixture day 4: numeric 7.807415 -> 7.686469 = the dated twin)."

- [ ] **Step 4: Run to verify**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings_runs.py test/test_missing_fixings_scenarios.py test/test_missing_fixings_followup.py test/test_pnlexplain_backtest_replay.py test/test_backtest_replay*.py -n 2 -q` (check globs) and the replay golden gate `... -m pytest test/replay_golden -n 2 -q` if it is collected there (see `test/replay_golden/README.md`).
Expected: all pass; replay goldens unchanged (no participant in them; any golden diff: STOP).

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/asset/equity/lifecycle/autocallable.py quantark/backtest/replay/product_replay.py quantark/backtest/replay/engine.py test/test_run_fixings_runs.py CHANGELOG.md
git -C <WT> commit -m "feat(run-fixings): the replay records a fixing on the date it passes; numeric Asians age their records

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: PnL explain — fixings inside the step (E2, E4, E5) and the run recorders (X3i)

Spec §4, X3i, X5; Rulings 5, 6; Review Focus 5; drift 7, 8.

**Files:**
- Modify: `quantark/pnlexplain/equity/explain.py:22,42-80`, `quantark/pnlexplain/equity/fingerprints.py:28-35,312-368`, `quantark/pnlexplain/equity/lifecycle.py:111-170,227-238`, `CHANGELOG.md`
- Test: `test/test_missing_fixings_followup.py`, `test/test_run_fixings_runs.py`

**Interfaces:**
- Consumes: `participates`, `run_step_dt`, `at_spot` (Task 4); `with_time_elapsed` (Tasks 1, 3); Tasks 6, 9 runs.
- Produces:
  - `check_contract_roll(product_t0, product_alive_t1, calendar_days, *, expected_decrement=None, pricing_env_t1=None) -> None` — participant route (Ruling 5); `OUTCOME_FIELDS = frozenset({"observed_in_range", "observed_price"})`.
  - `resolve_transition` passes `pricing_env_t1=snap1.pricing_env` to both checks and, for a participant rolling the full step, `expected_decrement=run_step_dt(snap0.pricing_env.valuation_date, snap1.pricing_env)`.
  - `_freeze_pair_fixings` builds the pair contract (Ruling 6); the re-fill row is emitted whenever t0 fills or step fills exist; its label is `"fixing decided by the pair at t0's spot, decided differently at t1"`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_missing_fixings_followup.py`:

```python
# =============================================================================
# 3. Run fixings in PnL explain (spec 2026-10-09 run fixings §4, X5): a fixing that passes inside the step
# =============================================================================
import run_fixings_fixtures as RF  # noqa: E402
from quantark.util.numerical import is_close  # noqa: E402


def _rf_explain(product_t0, day_t0, product_t1, day_t1):
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.base import ExplainMethod
    from quantark.pnlexplain.config import PnLExplainConfig
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    engine = RangeAccrualAnalyticalEngine()
    e0, e1 = RF.eq_env(day_t0), RF.eq_env(day_t1)
    s0 = ValuationSnapshot(product_t0, engine, e0, date=e0.valuation_date, quantity=1.0)
    s1 = ValuationSnapshot(product_t1, engine, e1, date=e1.valuation_date, quantity=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        return explain(s0, s1, config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)))


def _close6(a, b):
    return is_close(a, b, rel_tol=0.0, abs_tol=5e-7)


@pytest.mark.parametrize("t1_outcomes, pv_t1, refill", [
    ({}, 2.744374, -1.905067),          # P1
    ({0: True}, 4.649441, 0.0),         # P2
    ({0: False}, 2.744374, -1.905067),  # P2b
], ids=["P1", "P2", "P2b"])
def test_x5_a_fixing_dated_t0_is_fixed_at_s0_for_the_pair(t1_outcomes, pv_t1, refill):
    from quantark.pnlexplain.base import ExplainMethod, Factor

    res = _rf_explain(RF.ra_dated(), 4, RF.ra_dated(t1_outcomes), 10)
    alive = RF.quiet_price(RangeAccrualAnalyticalEngine(), RF.ra_dated({0: True}), RF.eq_env(10))
    assert is_close(res.pv_alive_t1.total, alive, rel_tol=0.0, abs_tol=1e-12) and _close6(alive, 4.649441)
    assert _close6(res.pv_t1.total, pv_t1) and _close6(res.pv_t0.total, 5.707299)
    row = _row(res, Factor.FIXING_REFILL)
    assert row is not None and _close6(row.pnl, refill)
    assert is_close(row.pnl, res.pv_t1.total - res.pv_alive_t1.total - _row(res, Factor.LIFECYCLE_EVENT).pnl,
                    rel_tol=0.0, abs_tol=1e-12)
    for m in (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR):
        assert res.reconcile(m) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("t1_outcomes", [{}, {0: True}], ids=["P3", "P3b"])
def test_x5_a_fixing_strictly_inside_the_step_is_fixed_at_s0(t1_outcomes):
    from quantark.pnlexplain.base import Factor

    res = _rf_explain(RF.ra_dated(), 3, RF.ra_dated(t1_outcomes), 5)
    assert _close6(res.pv_t1.total, 5.592668) and _row(res, Factor.FIXING_REFILL).pnl == 0.0


def test_x5_p4_missing_at_t0_is_unchanged():
    from quantark.pnlexplain import explain
    from quantark.pnlexplain.base import ExplainMethod, Factor
    from quantark.pnlexplain.config import PnLExplainConfig
    from quantark.pnlexplain.equity.snapshot import ValuationSnapshot

    def prod():
        return RangeAccrualOption(
            initial_price=100.0, range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0,
                                                                 accrual_rate=0.10),
            exercise_date=RF.START + timedelta(days=365), initial_date=RF.START - timedelta(days=10),
            observation_records=[RangeAccrualObservationRecord(observation_date=RF.START - timedelta(days=3))]
            + [RangeAccrualObservationRecord(observation_date=d) for d in RF.fixing_dates()[2:]])
    engine = RangeAccrualAnalyticalEngine()
    e0, e1 = RF.eq_env(0, spot=105.0), RF.eq_env(6, spot=111.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        res = explain(ValuationSnapshot(prod(), engine, e0, date=e0.valuation_date, quantity=1.0),
                      ValuationSnapshot(prod(), engine, e1, date=e1.valuation_date, quantity=1.0),
                      config=PnLExplainConfig(methods=(ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)))
    assert _close6(_row(res, Factor.FIXING_REFILL).pnl, -2.380029)


def test_x5_changing_a_recorded_fixing_is_still_refused():
    with pytest.raises(ValidationError, match="contract replacement is not a time step"):
        _rf_explain(RF.ra_dated({0: True}), 5, RF.ra_dated({0: False}), 10)


def test_x5_an_outcome_on_a_future_record_is_still_refused():
    with pytest.raises(ValidationError):
        _rf_explain(RF.ra_dated(), 4, RF.ra_dated({1: True}), 10)               # #1 is dated day 18


def test_a_fixing_dated_t1_stays_live_in_the_pair():
    """Review Focus 5."""
    from quantark.pnlexplain.base import Factor

    res = _rf_explain(RF.ra_dated(), 3, RF.ra_dated(), 4)
    assert _row(res, Factor.FIXING_REFILL) is None
    assert res.pv_alive_t1.total == pytest.approx(res.pv_t1.total, abs=1e-12)


def test_x5_numeric_participants_roll_through_the_check():
    from quantark.pnlexplain.base import Factor

    day = 1.0 / 365.0
    t0 = RF.ra_numeric()
    for _ in range(3):
        t0 = t0.with_time_elapsed(day)
    t1 = t0
    for _ in range(2):
        t1 = t1.with_time_elapsed(day)
    res = _rf_explain(t0, 3, t1, 5)                                             # #0 (day 4) passes inside
    assert is_close(_row(res, Factor.FIXING_REFILL).pnl, 0.0, rel_tol=0.0, abs_tol=1e-12)
```

Append to `test/test_run_fixings_runs.py` (X3i):

```python
# ================================================================ X3i: the run explain recorders
def _refill_and_chain(explain_df, recon_df, values):
    refill = explain_df[explain_df["factor"] == "fixing_refill"]
    assert len(refill) > 0 and (refill["pnl"] == 0.0).all()
    pos = recon_df[(recon_df["level"] == "position") & (recon_df["method"] == "waterfall")]
    assert is_close(pos["expected"].sum(), values[-1] - values[0], rel_tol=0.0, abs_tol=1e-9)   # pv_t1(N) == pv_t0(N+1)


@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_x3i_equity_backtest_explain(form, tmp_path):
    from quantark.backtest import BacktestConfig, BacktestEngine, ZeroCostModel
    from quantark.backtest.strategy import DeltaNeutralStrategy
    from quantark.pnlexplain import ExplainMethod, PnLExplainConfig
    from quantark.portfolio import Position

    adapter = F.DailyAdapter(12)
    cfg = BacktestConfig(
        strategy=DeltaNeutralStrategy(delta_threshold=1e12), start_date=F.START,
        end_date=adapter.dates[-1].to_pydatetime(), underlying=F.UNDERLYING,
        initial_positions=[Position(product=F.MAKERS[("ra", form)](), quantity=1.0, entry_price=0.0,
                                    underlying=F.UNDERLYING, engine=F.ENGINES["ra"](), entry_timestamp=F.START)],
        market_data_adapter=adapter, transaction_cost_model=ZeroCostModel(), handle_lifecycle_events=False,
        results_path=str(tmp_path), pnl_explain=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,)),
    )
    res = BacktestEngine(cfg).run()
    _refill_and_chain(res.explain_df, res.explain_reconciliation_df, res.get_value_series().tolist())


def test_x3i_replay_explain():
    from quantark.pnlexplain import ExplainMethod, PnLExplainConfig

    _, res, mtm = _replay(F.asian_dated(), n=12, explain=PnLExplainConfig(methods=(ExplainMethod.WATERFALL,)))
    _refill_and_chain(res.explain_df(), res.explain_reconciliation_df(), mtm)   # book results: methods
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_missing_fixings_followup.py test/test_run_fixings_runs.py -k "x5 or x3i or dated_t1" -n 2 -q`
Expected: FAIL — P2/P2b/P3b raise `contract replacement is not a time step`; P1 has no re-fill row; `test_x3i_equity_backtest_explain[numeric]` raises the same.

- [ ] **Step 3: Implement**

`fingerprints.py` — import `from quantark.util.run_fixings import participates`; add after `MATURITY_FLOOR`:

```python
OUTCOME_FIELDS = frozenset({"observed_in_range", "observed_price"})   # a fixing's recorded outcome (run fixings E5)
_CLOCK_LEAVES = frozenset({"observation_time", "accrued_offset"})     # numeric clock terms the run clock moves


def _clock_equal(a: Any, b: Any) -> bool:
    xa, xb = _float_leaf(a), _float_leaf(b)
    if xa is None or xb is None:
        return a == b
    return is_close(xa, xb, rel_tol=0.0, abs_tol=ROLL_TOL)


def _fixing_records_equal(a: Any, b: Any, allowed: frozenset) -> bool:
    """Normalised ``observation_records``: equal except clock leaves within ROLL_TOL and an outcome that is open in
    ``a`` and recorded in ``b`` on a record index in ``allowed`` (spec 2026-10-09 run fixings E5)."""
    if a is None or b is None or not (isinstance(a, tuple) and a[:1] == ("seq",) and b[:1] == ("seq",)):
        return a == b
    if len(a[1]) != len(b[1]):
        return False
    for i, (ra, rb) in enumerate(zip(a[1], b[1])):
        if ra[0] != rb[0] or len(ra[1]) != len(rb[1]):
            return False
        for (na, va), (nb, vb) in zip(ra[1], rb[1]):
            if na != nb:
                return False
            if na in OUTCOME_FIELDS:
                if va != vb and not (va is None and vb is not None and i in allowed):
                    return False
            elif na in _CLOCK_LEAVES:
                if not _clock_equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


def _run_fixings_roll_equal(expected: Any, alive: Any, allowed: frozenset) -> bool:
    fa, fb = contract_fingerprint(expected), contract_fingerprint(alive)
    if fa[0] != fb[0]:
        return False
    ia, ib = dict(fa[1]), dict(fb[1])
    if set(ia) != set(ib):
        return False
    for name, va in ia.items():
        vb = ib[name]
        if name == "observation_records":
            if not _fixing_records_equal(va, vb, allowed):
                return False
        elif name in _CLOCK_LEAVES:
            if not _clock_equal(va, vb):
                return False
        elif va != vb:
            return False
    return True
```

In `check_contract_roll` add the keyword `pricing_env_t1: Any = None`, document it ("the t1 environment: a run-fixings product's outcome recorded at t1 on a record strictly past there is a fixing, not a replacement (E5)"), and replace the first two lines with:

```python
    if participates(product_t0):
        days0 = int(calendar_days) == 0
        decrement = 0.0 if days0 else (calendar_days / 365.0 if expected_decrement is None else float(expected_decrement))
        expected = product_t0.with_time_elapsed(decrement)
        allowed = frozenset(expected.missing_fixing_fills(pricing_env_t1)) if pricing_env_t1 is not None else frozenset()
        same = _run_fixings_roll_equal(expected, product_alive_t1, allowed)
    else:
        fp0, fp1 = contract_fingerprint(product_t0), contract_fingerprint(product_alive_t1)
        same = fp0 == fp1 if int(calendar_days) == 0 else _rolled_equal(fp0, fp1)
```

`lifecycle.py` `resolve_transition` — import `participates`, `run_step_dt`; add helper:

```python
def _roll_decrement(snap0: ValuationSnapshot, snap1: ValuationSnapshot, roll_days: int, calendar_days: int):
    """A run-fixings product ages on the run clock (R6) over a full step; every other contract keeps
    ``float_maturity_decrement``."""
    if participates(snap0.product) and roll_days == calendar_days:
        return run_step_dt(snap0.pricing_env.valuation_date, snap1.pricing_env)
    return float_maturity_decrement(snap0.pricing_env, snap0.date, roll_days)
```

and change the three `check_contract_roll` calls to pass `expected_decrement=_roll_decrement(snap0, snap1, <days>, calendar_days)` (the first two) and `pricing_env_t1=snap1.pricing_env` (all three).

`fixing_refill_row`: docstring -> "A fixing the pair decided at t0's spot — one missing at t0 (W5b) or one that passed inside the step (run fixings E2) — that the t1 snapshot decides differently (user decisions D4, D7). ``value_t1 − value_t1_carried``; zero when t1 agrees." and `metadata={"label": "fixing decided by the pair at t0's spot, decided differently at t1"}`. Check `grep -rn "missing fixing re-filled" <WT>/test <WT>/quantark` returns nothing after the edit.

`explain.py` — import `from quantark.util.run_fixings import at_spot, participates, run_step_dt`; replace `_freeze_pair_fixings`:

```python
def _freeze_pair_fixings(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    transition: LifecycleTransition,
) -> Tuple[ValuationSnapshot, LifecycleTransition, Optional[ValuationSnapshot]]:
    """The pair's fixing decisions (spec 2026-10-09 run fixings §4.3 E2; W5b; user decisions D2, D3, D4, D7).

    t0's missing past fixings are recorded at t0's spot on the t0 contract (history; every t0-dated state prices
    it, a fixing at t0 itself stays live). For a run-fixings product, every record strictly past at t1 and open in
    the t0 contract — dated t0 or strictly inside the step — is decided ONCE at t0's spot (S0), never by a shocked
    state's spot; the pair contract (Ruling 6: the t0 contract aged by the run clock, with every decision recorded)
    is the alive-at-t1 contract and the carried t1 snapshot's. ``snapshot_t1`` itself is unchanged: ``pv_t1`` is its
    own value, and ``FIXING_REFILL`` measures any disagreement with the pair. Returns ``(snapshot_t0, transition,
    snapshot_t1_carried)``; the last is None when the pair decided nothing.
    """
    product_t0 = snapshot_t0.product
    env0, env1 = snapshot_t0.pricing_env, snapshot_t1.pricing_env
    fills_of = getattr(product_t0, "missing_fixing_fills", None)
    t0_fills = fills_of(env0) if fills_of is not None else {}
    pair = None
    step_fills: Dict[int, Any] = {}
    if participates(product_t0):
        aged = product_t0.with_fixings_recorded(t0_fills).with_time_elapsed(
            run_step_dt(env0.valuation_date, env1))
        step_fills = aged.missing_fixing_fills(at_spot(env1, float(env0.spot)))
        pair = aged.with_fixings_recorded(step_fills)
    if not t0_fills and not step_fills:
        return snapshot_t0, transition, None
    if t0_fills:
        snapshot_t0 = dataclasses.replace(snapshot_t0, product=freeze_missing_fixings(product_t0, env0))
    alive = pair if pair is not None else _carry_fills(transition.product_alive_t1, t0_fills)
    carried = pair if pair is not None else _carry_fills(snapshot_t1.product, t0_fills)
    transition = dataclasses.replace(transition, product_alive_t1=alive)
    return snapshot_t0, transition, dataclasses.replace(snapshot_t1, product=carried)
```

CHANGELOG `### Changed`: "PnL explain: a range accrual / Asian fixing that passes inside the step (dated t0, or strictly between t0 and t1) is decided once at t0's spot for every revaluation of the pair; the t1 snapshot's own decision (its record, or its fill at t1's spot) shows in the `fixing_refill` row, now emitted whenever the pair decided a fixing (label: 'fixing decided by the pair at t0's spot, decided differently at t1'). A t1 contract that records a fixing the t0 contract left open is no longer refused (a t0-recorded outcome that changes still is), and an aged numeric contract passes the roll check. Spec probe P1: pv_alive_t1 2.744374 -> 4.649441, `fixing_refill` −1.905067, total unchanged −2.962925."

Add to `moved_values.txt`:

```text
(spec probe, no existing test) PnL explain P1 (t0 day 4, t1 day 10, dated range accrual) :: pv_alive_t1 : 2.744374 -> 4.649441 ; fixing_refill row : absent -> -1.905067 ; total unchanged  [spec §4.4/§8, D2/D7]
(spec probe, no existing test) PnL explain P2 / P2b / P3b :: raises ValidationError -> explains (P2 re-fill 0, P2b -1.905067, P3b 0)  [spec §4.3 E5]
```

- [ ] **Step 4: Run to verify**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_missing_fixings_followup.py test/test_missing_fixings_scenarios.py test/test_run_fixings_runs.py test/test_pnlexplain_*.py -n 2 -q`
Expected: all pass. Existing pnlexplain tests must not move (none holds a participant with a fixing inside its step); any move: STOP.

- [ ] **Step 5: Commit**

```bash
git -C <WT> add quantark/pnlexplain/equity/explain.py quantark/pnlexplain/equity/fingerprints.py quantark/pnlexplain/equity/lifecycle.py test/test_missing_fixings_followup.py test/test_run_fixings_runs.py CHANGELOG.md
git -C <WT> add -f docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt
git -C <WT> commit -m "feat(run-fixings): PnL explain fixes a fixing passing inside the step at t0's spot; the re-fill row reports t1's decision

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Final gate — TODO sweep, CHANGELOG consolidation, moved values, full suite (controller)

**Files:**
- Modify: `CHANGELOG.md` (the W5b paragraph), `docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt`

- [ ] **Step 1: TODO sweep**

Run: `grep -rn "TODO" <WT>/quantark | grep -i fix`
Expected: exactly one line, `quantark/util/run_fixings.py: ... TODO(run-fixings): intraday fixing instants ...`. Any other W5b `TODO(lifecycle)` line left at the §0 sites: fix it in the owning file (comment only) and re-run.

- [ ] **Step 2: CHANGELOG consolidation**

In the W5b `### Changed` paragraph replace "a fixing that passes during a run is still decided at each day's spot (TODO, a lifecycle matter)." with "a fixing that passes during a run is recorded once at the spot of the day it was live (run fixings, below)." Read the whole `[Unreleased]` section once: one bullet per change, no duplicates.

- [ ] **Step 3: Moved values for sign-off**

Append the spec §8 rows that moved in this branch and are not asserted by any test (numeric aging in runs, the "no roll rule" warning, the `H`-frequency refusal), each with "[spec §8]". Re-read the file: every line old -> new with a reason.

- [ ] **Step 4: Targeted suites (implementer, `-n 2`)**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_run_fixings.py test/test_run_fixings_runs.py test/test_missing_fixings.py test/test_missing_fixings_followup.py test/test_missing_fixings_scenarios.py test/test_missing_fixings_unbumped.py test/test_float_maturity_roll.py test/test_backtest_lifecycle.py test/test_pnlexplain_backtest_equity.py test/test_pnlexplain_backtest_replay.py -n 2 -q`
Expected: all pass.

- [ ] **Step 5: Full suite (CONTROLLER, `-n 6`)**

Run: `env PYTHONPATH=<WT>:<WT>/test /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n 6 -q` then `git -C <WT> checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html`.
Expected: only the known pre-existing main failures (memory: 4 mo_volmodels/otc tests since 0.7.0); compare against a `main` baseline run made in the same window. Any new failure: STOP.

- [ ] **Step 6: Commit**

```bash
git -C <WT> add CHANGELOG.md
git -C <WT> add -f docs/superpowers/specs/2026-10-09-run-fixings-evidence/moved_values.txt
git -C <WT> commit -m "docs(run-fixings): CHANGELOG consolidation and the moved-values list for sign-off

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Landing waits for the user's sign-off of `moved_values.txt` and the rulings above. Never push or merge.

---

## Spec coverage check (done while writing)

| Spec item | Task |
|---|---|
| §3.1 module, names | 4 (+ `pending_pv`, drift 5) |
| §3.2 protocol, t = 0 warning | 1, 3 (+ `open_fixings`, `has_run_fixings`, Ruling 1) |
| §3.3 R1–R10 | 4 (unit), 6–9 (paths) |
| §3.4 Greeks / theta in runs | 2 (Asian), 7 (X3h), 4 (no warnings after register) |
| §3.5 call sites | 6, 7, 8, 9 |
| §3.6 replay stateless | 9 |
| §4 explain E1–E6 | 10 |
| §5.2 legacy Asian not covered | 1 (`has_run_fixings` False, `with_time_elapsed` refuses), 4 (untouched) |
| §6 TARN hooks | 5; hand-off below |
| §7 X1a–c | 6, 7, 8, 9 |
| §7 X2 (+ BUSINESS_DAYS) | 4, 6, 7, 8, 9 |
| §7 X3a–i | 7 (a, b-numeric, c, d, e, h), 6 (b-weekly, e-H), 8 (b-Saturday, d), 4 (c, e, f), 6–9 (g), 10 (i) |
| §7 X4 | 1, 3 |
| §7 X5 | 10 |
| §7 X6 | 6, 7, 8, 9 |
| §8 moved values | 1, 10, 11 |

---

## Hand-off to W4b (FX TARN / TARF, re-plan of `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle.md` Tasks 9–11)

W4b's Tasks 9–11 build NO lifecycle class (`quantark/asset/fx/lifecycle/` is dropped, spec §6). They make `TargetRedemptionProduct` a run-fixings participant and read the recorder's output. Exact interfaces after W6 lands:

**Product protocol `TargetRedemptionProduct` must implement** (consumed by `quantark.util.run_fixings`):

| Member | Signature | Contract |
|---|---|---|
| `has_run_fixings` | property -> `bool` | `True` |
| `missing_fixing_fills` | `(fx_env) -> Dict[int, float]` | `{i: fx_env.spot}` for every unbanked fixing with `is_history(t)` (t = 0 excluded) |
| `with_fixings_recorded` | `(fills: Mapping[int, float]) -> copy` | banks the listed fixings IN INDEX ORDER at the given rates through `record_passed_fixings` arithmetic (clip, redemption, dead fixings, PAYMENT events appended to `lifecycle_events`); a banked fixing keeps its rate; `self` when nothing changes |
| `with_missing_fixings_filled` | `(fx_env, *, warn=True) -> copy` | W5b: `missing_fixing_fills` + one `MissingFixingWarning` (`is_history` labels only) + `with_fixings_recorded` |
| `with_time_elapsed` | `(dt: float) -> copy` | numeric: copy + `time_shift(dt)` + `snap_time` on fixing/pay times and maturity; records/banks NOTHING; dated or `is_zero(dt)`: `self` |
| `open_fixings` | `(fx_env) -> Dict[int, Tuple[float, str]]` | `{i: (resolve_time, label)}` for every unbanked fixing |
| `matured_before` (optional hook) | `(fx_env) -> bool` | recorder warns `MaturedBeforeRunWarning` and removes the position at registration |
| `lifecycle_events` (optional hook) | `Sequence[TargetRedemptionEvent]` | append-only; the recorder reports `events[len(before):]` of each step |
| `pending_pv` (optional hook) | `(fx_env) -> float` | summed `× quantity` into `RunFixingDay.pending_pv` |

Also: the built Task 6/7 `resolve()` banks an unbanked t = 0 fixing on the copy at the quoted (possibly bumped) spot but names only `is_history` fixings in its `MissingFixingWarning` (D1).

**Recorder outputs W4b reads** (`quantark.util.run_fixings`):

```python
class RunFixingRecorder:
    def register(self, portfolio) -> RunFixingDay      # FX engines call it on the working copy (W6 Task 8)
    def advance(self, portfolio) -> RunFixingDay       # after each row / dated day (W6 Task 8)
    days: Tuple[RunFixingDay, ...]; recorded: Tuple[RecordedFixing, ...]
@dataclass(frozen=True) class RunFixingDay:
    date: Optional[datetime]; fixings: Tuple[RecordedFixing, ...]
    product_events: Tuple[Tuple[str, float, Any], ...] = ()    # (position_id, quantity, TargetRedemptionEvent)
    pending_pv: float = 0.0
@dataclass(frozen=True) class RecordedFixing:
    position_id: str; index: int; label: str; outcome: Any; spot: float; gap: bool
FXBacktestEngine.run_fixings / FXDynamicScenarioEngine.run_fixings   # the last run's recorder
```

Task 10 (FX backtest) derives `paid_cash` from `product_events` PAYMENT entries (`quantity × event.amount`) and `pending_pv` from `RunFixingDay.pending_pv`; Task 11 (FX dynamic) keeps the Q8 refusal of undated paths BEFORE `register`, and its FIXING snapshots use `RecordedFixing.spot` (the previous path day's spot). Test changes per spec §6 (origin-day fixing banks on the first later `advance`, not at `register`; gap fixings at the previous day's spot with `gap=True`; `test_clock_cannot_go_back` / environment-date checks move to the recorder's R9).
