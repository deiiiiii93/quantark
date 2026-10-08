# KO-reset Lifecycle Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track `KnockOutResetSnowballOption` through the lifecycle core, the portfolio manager, single-product replay and the simulated-path backtest, with snapped numeric observation times shared by every autocallable.

**Architecture:** A `snap_contract` helper snaps every numeric observation time to the day the tracker observes it; every autocallable tracker holds the snapped contract and exposes it as `lifecycle_contract`. A `KOResetLifecycleTracker` subclass applies R1–R4 (schedule switch at the knock-in, schedule-1 maturity with V0, knocked-in maturity with V1, knock-out wins a tie) and converts a knocked-in `REBASED` contract to the `ABSOLUTE` contract it has become, so every engine prices an ordinary knocked-in contract. The simulation gets a KO-reset calendar schedule and vector lifecycle, a `knock_in_day` state dimension and a valuation-phase flag.

**Tech Stack:** Python 3.11, NumPy, pandas, pytest (xdist, at most `-n 6`).

**Spec:** `docs/superpowers/specs/2026-10-08-ko-reset-lifecycle-design.md` (commit d274abfe).

## Global Constraints

- R1: a knock-in during schedule 1 replaces schedule 1 by schedule 2 for good; schedule 2 is in force from the knock-in observation, that observation included.
- R2: a contract not knocked in when schedule 1 ends has matured there: it pays V0 on schedule 1's contract length; a knock-in is never tested after schedule 1 ends.
- R3: a schedule-1 knock-out and a knock-in at the same observation: the knock-out wins.
- R4: a carried knock-in is an event that happened; every pricer takes it as such.
- Snapping: numeric time → `round(time * 365) / 365`; a time with `time * 365` within `Tolerance.PRECISION` of an integer keeps its exact float; settlement lags kept as offsets from the snapped determination; dated records untouched.
- Snowball and Phoenix results byte-identical except §4a's snapped off-day numeric schedules (every moved row listed).
- `disable_ko_after_ki`: no schedule-2 knock-out at all, fresh or carried knock-in.
- Un-knocked `REBASED`: MC pricing engine required (manager, replay, simulation exact mode); simulation surface mode refuses; intraday unchanged. Refusals at configuration, naming product, mode and engine.
- The spec's `knock_in_date` is the state's existing `ki_date` (set by `mark_ki` to the knock-in close); no second field. The product carries it as `_otc_lifecycle_knock_in_date`.
- A schedule-2 record is consumed by the knock-in iff it was already due at the PREVIOUS observed close (no previous close: none consumed) — the vector form is "its first-due position is before the knock-in day".
- Numerical comparisons via `quantark.util.numerical`; exceptions `ValidationError`; canonical `quantark.*` imports.
- RAM: full suite with `-n 6` at most, RAM guard (background task bxr5unvrx) running; `git checkout --` the two mo sample files after any full suite.
- `docs/` is gitignored: `git add -f` for the plan/spec; commit by pathspec.

## Review Focus

1. A KO-reset whose schedule-1 last observation and a knock-in fall on a holiday — both must fire on the next close in the right order (KO1, KI, schedule-2 on the same close), not V0 maturity first. Pinned in Task 5 (`test_holiday_last_observation_knock_in_switches`).
2. A replay that starts mid-life with records already in the past — they are observed on the first close, never dropped, and no schedule-2 record is consumed on the first close. Pinned in Task 5 (`test_first_close_consumes_nothing`).
3. Annualized off-day numeric knock-out cash: snapping moves the accrual end by under half a day; the tracker's cash must equal the engines' decided cash for the same snapped contract. Pinned in Task 1 (`test_snapped_knock_out_cash_matches_the_engine`).
4. A simulated path that knocks in on the last day of the calendar under `REBASED` — its final maturity lies past the calendar; it must count as a data-end path (`allow_data_end`), never an index error. Pinned in Task 7 (`test_rebased_knock_in_past_calendar_is_data_end`).
5. A carried knock-in whose date lies before `start_date` (`REBASED`) — schedule 2 lands partly in the past; past records are dropped from the pricing copy and never tested. Pinned in Task 5 (`test_carried_rebased_knock_in_before_start`).

---

## File Structure

- Create `quantark/asset/equity/lifecycle/snapping.py` — `snap_time`, `snap_contract`.
- Create `quantark/asset/equity/lifecycle/ko_reset.py` — `KOResetLifecycleTracker`, `KOResetCalendarSchedule`, `SchedulePlacement`.
- Modify `quantark/asset/equity/lifecycle/autocallable.py` — snapped contract at construction, `lifecycle_contract`, `product_for_pricing(..., initial=False)`, hooks `_contract_for_pricing` / `_age_extra`, record-dict helper split from `_scheduled_records`.
- Modify `quantark/asset/equity/lifecycle/state.py` — `observed_post_ko_indices`.
- Modify `quantark/asset/equity/lifecycle/__init__.py` — exports.
- Modify `quantark/asset/equity/engine/settlement_support.py` — `resolve_terminal_timing_at`.
- Modify `quantark/asset/equity/product/option/ko_reset_snowball_option.py` — `get_maturity_payoff_v1(contract_tenor=)`, `with_absolute_schedule_2`.
- Modify `quantark/asset/equity/engine/mc/snowball_mc_engine.py` — REBASED accrued offset, V1 node per path, V1 tenor per path.
- Modify `quantark/asset/equity/lifecycle/manager.py` — KO-reset tracker + refusal.
- Modify `quantark/backtest/replay/{product_replay,engine,engine_factory,config}.py` — tracker class, initial phase, QUAD route, refusals.
- Modify `quantark/backtest/simulation/{lifecycle,engine,config}.py`, `pricing/{base,repricing,surface}.py` — KO-reset schedule/lifecycle, `knock_in_day`, phase, snapped contract.
- Tests: `test/lifecycle_ko_reset/` (new package), plus the existing suites.

---

### Task 1: Snapped numeric times and the valuation phase in the base tracker

**Files:**
- Create: `quantark/asset/equity/lifecycle/snapping.py`
- Modify: `quantark/asset/equity/lifecycle/autocallable.py:58-142`
- Modify: `quantark/asset/equity/lifecycle/__init__.py`
- Test: `test/lifecycle_ko_reset/__init__.py`, `test/lifecycle_ko_reset/conftest.py`, `test/lifecycle_ko_reset/test_snapped_times.py`

**Interfaces:**
- Produces: `snap_time(time: Optional[float]) -> Optional[float]`; `is_off_day(time: float) -> bool`; `snap_contract(product) -> product` (deep copy; snaps `barrier_config`, `post_barrier_config` when present, numeric `maturity`).
- Produces: `AutocallableLifecycleTracker.lifecycle_contract` (property, the snapped contract); `product_for_pricing(date, pricing_env, *, initial: bool = False)`; hooks `_contract_for_pricing(self) -> product` and `_age_extra(self, product, elapsed, date, pricing_env, keep_landing) -> None`.
- Produces: `_record_dicts(self, product, resolved, payoffs=None) -> List[dict]` (the dict building of `_scheduled_records`, reused by Task 5).

- [ ] **Step 1: Write the failing tests**

`test/lifecycle_ko_reset/__init__.py`: empty. `test/lifecycle_ko_reset/conftest.py`:

```python
from datetime import datetime

import pandas as pd

from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.option_enums import ObservationType

START = pd.Timestamp("2026-01-05")


def env(spot=100.0, date=START):
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(0.25),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=pd.Timestamp(date).to_pydatetime())


def schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def snowball(ko_times, ki_times, maturity, *, annualized=False, ko=103.0, ki=75.0):
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity,
        barrier_config=BarrierConfig(ko_barrier=ko, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule(ko_times, ko), ki_barrier=ki,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=schedule(ki_times, ki)),
        accrual_config=AccrualConfig(is_annualized=annualized),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))
```

`test/lifecycle_ko_reset/test_snapped_times.py`:

```python
import pandas as pd
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.snapping import is_off_day, snap_contract, snap_time
from quantark.asset.equity.param import PDEParams
from quantark.backtest.simulation.pricing.repricing import product_fingerprint

from .conftest import START, env, snowball

MONTHS = [k / 12.0 for k in range(1, 13)]


def test_snap_time_moves_an_off_day_time_to_its_rounded_day():
    assert snap_time(1.4 / 365.0) == 1 / 365.0
    assert snap_time(1 / 12.0) == 30 / 365.0
    assert snap_time(None) is None
    assert not is_off_day(30 / 365.0) and is_off_day(1 / 12.0)


def test_a_whole_day_time_keeps_its_exact_float():
    t = 0.0821917808219178          # 30 / 365 spelled another way
    assert snap_time(t) == t


def test_a_whole_day_contract_is_bit_identical():
    product = snowball([30 / 365, 60 / 365], [30 / 365], 60 / 365)
    assert product_fingerprint(snap_contract(product)) == product_fingerprint(product)


def test_the_tracker_holds_the_snapped_contract():
    tracker = AutocallableLifecycleTracker(product=snowball(MONTHS, MONTHS, 1.0), quantity=1.0, start_date=START)
    times = [r.observation_time for r in tracker.lifecycle_contract.barrier_config.ko_observation_schedule.records]
    assert times == [round(t * 365) / 365 for t in MONTHS]


def test_the_observed_record_leaves_the_post_close_copy_on_its_day():
    tracker = AutocallableLifecycleTracker(product=snowball(MONTHS, MONTHS, 1.0), quantity=1.0, start_date=START)
    day = START + pd.Timedelta(days=30)             # 1/12 year rounds to day 30
    post = tracker.product_for_pricing(day, env(date=day))
    initial = tracker.product_for_pricing(day, env(date=day), initial=True)
    assert len(post.barrier_config.ko_observation_schedule.records) == 11
    assert len(initial.barrier_config.ko_observation_schedule.records) == 12
    assert initial.barrier_config.ko_observation_schedule.records[0].observation_time == 0.0


def test_a_phoenix_coupon_schedule_snaps_with_its_knock_out_schedule():
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    phoenix = create_standard_phoenix(initial_price=100.0, maturity=1.0, observation_frequency=12)
    tracker = AutocallableLifecycleTracker(product=phoenix, quantity=1.0, start_date=START)
    day = START + pd.Timedelta(days=30)
    post = tracker.product_for_pricing(day, env(date=day))
    assert len(post.barrier_config.ko_observation_schedule.records) == 11


def test_snapped_knock_out_cash_matches_the_engine():
    """Review focus 3: annualized off-day cash is the snapped contract's, booked and priced alike."""
    tracker = AutocallableLifecycleTracker(product=snowball(MONTHS, MONTHS, 1.0, annualized=True), quantity=1.0,
                                           start_date=START)
    day = START + pd.Timedelta(days=30)
    events = tracker.observe(day, tracker.product_for_lifecycle(), env(spot=110.0, date=day), 110.0)
    assert events and events[0].event_type.name == "KNOCK_OUT"
    decided = SnowballPDESolver(PDEParams()).price(tracker.product_for_pricing(START + pd.Timedelta(days=30),
                                                                               env(110.0, day), initial=True),
                                                   env(110.0, day))
    assert events[0].cashflow == pytest.approx(decided, rel=1e-9)
```

Before writing, check the factory signature: `grep -n "def create_standard_phoenix" -A20 quantark/asset/equity/product/option/phoenix_helpers.py` and use its real keyword for a monthly schedule (the call above names the intent; adapt the keyword, keep the assertion).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_snapped_times.py -q`
Expected: FAIL — `ModuleNotFoundError: quantark.asset.equity.lifecycle.snapping`.

- [ ] **Step 3: Write `snapping.py`**

```python
"""Snapped numeric observation times (spec 2026-10-08 KO-reset lifecycle, section 4a).

The tracker observes a numeric record on the close ``round(time * 365)`` days after its start; a
pricing copy, a surface or an engine that kept the fractional time would see the record a fraction of
a day away -- still ahead on the close that observed it. Snapping the contract's numeric times to that
day makes every consumer agree with the tracker. A time already on a whole day keeps its exact float,
so whole-day contracts are bit-identical.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import List, Optional

from quantark.util.numerical import Tolerance

DAYS_PER_YEAR = 365.0

__all__ = ["is_off_day", "snap_contract", "snap_time"]


def is_off_day(time: float) -> bool:
    """Whether ``time`` (years) lies off a whole day by more than ``Tolerance.PRECISION`` days."""
    days = float(time) * DAYS_PER_YEAR
    return abs(days - round(days)) > Tolerance.PRECISION


def snap_time(time: Optional[float]) -> Optional[float]:
    """``round(time * 365) / 365`` (the rounding ``_record_is_due`` uses); a whole-day time unchanged."""
    if time is None or not is_off_day(time):
        return time
    return round(float(time) * DAYS_PER_YEAR) / DAYS_PER_YEAR


def _snap_record(record):
    time = record.observation_time
    if record.observation_date is not None or time is None or not is_off_day(time):
        return record
    snapped = snap_time(time)
    settlement = record.settlement_time
    if settlement is not None:
        settlement = snapped + (float(settlement) - float(time))   # the lag is kept
    return replace(record, observation_time=snapped, settlement_time=settlement)


def _snap_schedule(schedule):
    if schedule is None:
        return None
    records = [_snap_record(r) for r in schedule.records]
    if all(a is b for a, b in zip(records, schedule.records)):
        return schedule
    return replace(schedule, records=records)


def _snap_dates(dates: Optional[List[float]]):
    if dates is None or not any(is_off_day(t) for t in dates):
        return dates
    return [snap_time(t) for t in dates]


def _snap_config(config):
    if config is None:
        return None
    changes = {
        "ko_observation_schedule": _snap_schedule(config.ko_observation_schedule),
        "ki_observation_schedule": _snap_schedule(config.ki_observation_schedule),
        "ko_observation_dates": _snap_dates(config.ko_observation_dates),
        "ki_observation_dates": _snap_dates(config.ki_observation_dates),
    }
    if all(changes[name] is getattr(config, name) for name in changes):
        return config
    return replace(config, **changes)


def snap_contract(product):
    """A deep copy of ``product`` with every numeric observation time and a numeric maturity snapped."""
    contract = deepcopy(product)
    if getattr(contract, "barrier_config", None) is not None:
        contract.barrier_config = _snap_config(contract.barrier_config)
    if getattr(contract, "post_barrier_config", None) is not None:
        contract.post_barrier_config = _snap_config(contract.post_barrier_config)
    if getattr(contract, "exercise_date", None) is None and getattr(contract, "maturity", None) is not None:
        contract.maturity = snap_time(float(contract.maturity))
    return contract
```

- [ ] **Step 4: Wire the tracker**

In `autocallable.py`: import `snap_contract`; in `__init__` replace `self.product = product` by `self.product = snap_contract(product)`; add

```python
    @property
    def lifecycle_contract(self) -> Any:
        """The snapped contract every consumer builds from (spec section 4a)."""
        return self.product

    def _contract_for_pricing(self) -> Any:
        """The contract the pricing copy ages; a subclass may convert it first (KO-reset REBASED)."""
        product = deepcopy(self.product)
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        return product

    def _age_extra(self, product, elapsed, date, pricing_env, keep_landing) -> None:
        """Age schedules beyond ``barrier_config`` (none for snowball and Phoenix)."""
```

Rewrite `product_for_pricing(self, date, pricing_env, *, initial: bool = False)`: start from `product = self._contract_for_pricing()`, pass `keep_landing=initial` to `barrier_config.time_shift(...)`, and call `self._age_extra(product, elapsed, pd.Timestamp(date).to_pydatetime(), pricing_env, initial)` before `return`. Docstring: "``initial`` is the run's first valuation, before its first close is observed: a record due that day is kept at time 0. Otherwise the day's records have been applied and are gone."

Split `_scheduled_records`: keep its resolution (`resolve_ko_observations` / `resolve_ki_observations`) and move the loop body into `_record_dicts(self, product, resolved, payoffs=None)`; `payoffs` overrides `float(rec.payoff)` per record when given. `_scheduled_records` returns `self._record_dicts(product, resolved)`.

Export `snap_contract`, `snap_time` from `quantark/asset/equity/lifecycle/__init__.py`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_snapped_times.py -q`
Expected: 7 passed.

- [ ] **Step 6: Run the lifecycle and replay suites**

Run: `.venv/bin/python -m pytest -n 6 test/test_equity_lifecycle_trackers.py test/test_lifecycle_calendar_schedule.py test/test_backtest_lifecycle.py test/test_dynamic_scenario_lifecycle.py test/test_replay_goldens.py test/test_snowball_lifecycle_ki.py test/test_pnlexplain_lifecycle.py test/test_pnlexplain_lifecycle_days.py -q > "$WS/t1.log" 2>&1; tail -5 "$WS/t1.log"`
Expected: all pass. A failing golden is an off-day numeric product: list the row and its record in the ledger (`Ruling: golden row <id> moved by snapped record <time>`), regenerate it per `test/replay_golden` instructions, and confirm event rows (dates, cashflows of non-annualized products) did not move.

- [ ] **Step 7: Commit**

```bash
git add quantark/asset/equity/lifecycle/snapping.py quantark/asset/equity/lifecycle/autocallable.py quantark/asset/equity/lifecycle/__init__.py test/lifecycle_ko_reset/
git commit -m "feat(lifecycle): snapped numeric observation times and the initial valuation phase"
```

---

### Task 2: The snapped contract and the phase in the replay and the simulation

**Files:**
- Modify: `quantark/backtest/replay/product_replay.py:199-200`, `quantark/backtest/replay/engine.py:386-397`
- Modify: `quantark/backtest/simulation/pricing/base.py` (`DayStates`, `state_row`, `StateKey`)
- Modify: `quantark/backtest/simulation/pricing/repricing.py:142-230,425-445`
- Modify: `quantark/backtest/simulation/pricing/surface.py:194-262`
- Modify: `quantark/backtest/simulation/engine.py` (`_states_for`, day-0 gate, `_initial_book_value`)
- Test: `test/lifecycle_ko_reset/test_simulation_snapping.py`

**Interfaces:**
- Consumes: `snap_contract`, `product_for_pricing(..., initial=)` (Task 1).
- Produces: `ProductReplay.product_for_date(date, env, *, initial=False)`; `DayStates.initial: bool = False` and `DayStates.knock_in_day: Optional[np.ndarray] = None` (NamedTuple fields with defaults, appended last); `StateKey.initial: bool = False`, `StateKey.knock_in_day: int = -1` (digest unchanged when both are default); `RepricingPricer.aged_product(date, *, knocked_in, initial=False, knock_in_day=-1)`.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd

from quantark.asset.equity.lifecycle.snapping import snap_contract
from quantark.backtest.simulation.pricing.base import DayStates, StateKey
from quantark.backtest.simulation.pricing.repricing import product_fingerprint

from test.simulation.conftest import RATE, SPOT, VOL, ensemble_config, pde_engine_config, short_snowball, surface_pricing


def _off_day_snowball():
    """The review's control: a knock-out at day 1.4, observed on day 1's close."""
    from dataclasses import replace
    product = short_snowball(maturity_days=6, ko_days=(2, 5), ki_days=(1, 3))
    sched = product.barrier_config.ko_observation_schedule
    records = [replace(sched.records[0], observation_time=1.4 / 365.0)] + list(sched.records[1:])
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=replace(sched, records=records))
    return product


def _pricers(product):
    from quantark.backtest.simulation.pricing.cache import StateCache
    from quantark.backtest.simulation.config import CacheConfig
    from quantark.backtest.simulation.pricing.repricing import RepricingPricer
    from quantark.backtest.simulation.pricing.surface import LifeSurfacePricer
    from quantark.backtest.simulation.paths.market_path import trading_calendar
    from datetime import date

    dates = trading_calendar(date(2024, 1, 2), 8)
    cfg, pricing = pde_engine_config(), surface_pricing()
    cache = StateCache(CacheConfig(memory_bytes=8_000_000))
    exact = RepricingPricer(product, engine_config=cfg, start_date=dates[0], underlying="equity_index", cache=cache)
    surface = LifeSurfacePricer(product, engine_config=cfg, start_date=dates[0], dates=dates,
                                underlying="equity_index", vol_step=pricing.vol_step, q_step=pricing.q_step,
                                surface_cache_bytes=pricing.surface_cache_bytes, gate=pricing.gate, cache=cache)
    return dates, exact, surface


def test_the_pricers_fingerprint_the_snapped_contract():
    product = _off_day_snowball()
    _, exact, surface = _pricers(product)
    assert exact.product_fingerprint == product_fingerprint(snap_contract(product))
    assert surface.product_fingerprint == exact.product_fingerprint


def test_the_surface_and_the_exact_copy_agree_after_the_observation():
    """0.4381 (corrected copy) vs 4.9183 (surface) before snapping."""
    from quantark.backtest.simulation.pricing.base import GateScale, state_row
    from quantark.param import ContinuousDividendYield

    product = _off_day_snowball()
    dates, exact, surface = _pricers(product)
    row = DayStates(day_index=1, date=dates[1], path_index=np.array([0]), spot=np.array([SPOT * 0.99]),
                    vol=np.array([VOL]), rate=np.array([RATE]), q_T=np.array([0.0]),
                    div_yield=(ContinuousDividendYield(0.0),), basis_yield=np.array([0.0]),
                    env_key=np.array([7], dtype=np.int64), knocked_in=np.array([False]))
    report = surface.verify([row], surface_pricing().gate, GateScale(unit_notional=SPOT, hands_per_unit_delta=1.0))
    assert report.passed, report.as_dict()


def test_default_phase_and_knock_in_day_leave_the_key_digest_unchanged():
    base = dict(product_fingerprint="p", day_index=3, knocked_in=False, spot_key=1, vol_key=2, env_key=3,
                engine_fingerprint="e")
    assert StateKey(**base).digest() == StateKey(**base, initial=False, knock_in_day=-1).digest()
    assert StateKey(**base, initial=True).digest() != StateKey(**base).digest()
    assert StateKey(**base, knock_in_day=4).digest() != StateKey(**base).digest()


def test_the_initial_copy_keeps_a_record_landing_on_day_zero():
    from dataclasses import replace
    product = short_snowball()
    sched = product.barrier_config.ko_observation_schedule
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=replace(
        sched, records=[replace(sched.records[0], observation_time=0.0)] + list(sched.records[1:])))
    dates, exact, _ = _pricers(product)
    initial = exact.aged_product(dates[0], knocked_in=False, initial=True)
    post = exact.aged_product(dates[0], knocked_in=False)
    assert (len(initial.barrier_config.ko_observation_schedule.records)
            == len(post.barrier_config.ko_observation_schedule.records) + 1)
```

Run the existing helper names through `grep -n "def short_snowball\|def surface_pricing\|SPOT =\|RATE =\|VOL =" test/simulation/conftest.py` first; `test.simulation.conftest` imports as a package (`test/simulation/__init__.py` exists). If `from test.simulation...` collides with the stdlib `test` (memory: golden capture import), use `from simulation.conftest import ...` as `test/intraday` tests do (`from intraday.conftest import ...`).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_simulation_snapping.py -q`
Expected: FAIL — fingerprint mismatch, gate fails (~74 bp), `StateKey` has no `initial`, `aged_product` has no `initial`.

- [ ] **Step 3: Implement**

`base.py`: append to `DayStates` (after `knocked_in`):

```python
    initial: bool = False                     # the run's first valuation, before its first close (spec 4a)
    knock_in_day: Optional[np.ndarray] = None  # (m,) int64, -1 unless a knocked-in REBASED path (spec 6)
```

`state_row` copies `initial=states.initial` and `knock_in_day=None if states.knock_in_day is None else states.knock_in_day[sl].copy()`. `StateKey` gains `initial: bool = False` and `knock_in_day: int = -1` (last, with defaults); `digest` appends `b"\x01initial"` when `initial` and `np.int64(knock_in_day).tobytes()` prefixed by `b"\x02ki"` when `knock_in_day >= 0`, so default keys hash exactly as before.

`repricing.py`: `self.product = snap_contract(product)` first in `__init__` and use `self.product` for `_x_ref`, `_product_fp`, `create_pricing_engine`, `_greeks_carry_price`. `aged_product(self, date, *, knocked_in, initial=False, knock_in_day=-1)`: key `(date, knocked_in, initial, knock_in_day)`; the tracker is built by `self._tracker_for(knocked_in, knock_in_day)` (Task 6 makes it KO-reset aware; here it returns the existing `AutocallableLifecycleTracker` with `AutocallableLifecycleState(knocked_in=...)`); call `tracker.product_for_pricing(key[0], env, initial=initial)`. Add

```python
    def _phase_initial(self, date, knocked_in: bool, knock_in_day: int) -> bool:
        """``initial`` only where the initial copy differs from the post-close one (spec 4a: never share)."""
        a = self.aged_product(date, knocked_in=knocked_in, initial=True, knock_in_day=knock_in_day)
        b = self.aged_product(date, knocked_in=knocked_in, knock_in_day=knock_in_day)
        return product_fingerprint(a) != product_fingerprint(b)
```

and memoize it per `(date, knocked_in, knock_in_day)`. In `_exact_keys`, `_ladder_layout` keys and `_price_one`/`price_exact`, pass `initial = states.initial and self._phase_initial(...)` and `knock_in_day = -1 if states.knock_in_day is None else int(states.knock_in_day[n])` into both the `StateKey` and `aged_product`.

`surface.py`: `self.product = snap_contract(product)` first; build `self._exact` and the engine from `self.product`; `self._maturity = float(self.product.maturity)`. `price_day` raises `ValidationError("the life surface prices the post-close contract; an initial valuation with a record landing today needs repricing")` when `states.initial` and `self._exact._phase_initial(...)` is true for the day (never true in the fixtures).

`engine.py`: `_states_for(..., initial=False, knock_in_day=None)` passes them into `DayStates`; the day-0 gate rows and `_initial_book_value` pass `initial=True`.

`product_replay.py`: `product_for_date(self, date, pricing_env, *, initial=False)` → `self._tracker.product_for_pricing(date, pricing_env, initial=initial)`. `engine.py:394`: `replay.product_for_date(date, env, initial=True)`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_simulation_snapping.py -q`
Expected: 4 passed.

- [ ] **Step 5: Run the simulation and replay suites**

Run: `.venv/bin/python -m pytest -n 6 test/simulation test/test_replay_goldens.py test/test_replay_engine_unified.py test/test_snowball_simulated_paths_study.py -q > "$WS/t2.log" 2>&1; tail -5 "$WS/t2.log"`
Expected: all pass, conformance gaps 0.0 (whole-day fixtures are bit-identical; the keys of day-0 states are unchanged because no fixture has a record at time 0).

- [ ] **Step 6: Commit**

```bash
git add quantark/backtest/replay/product_replay.py quantark/backtest/replay/engine.py quantark/backtest/simulation/ test/lifecycle_ko_reset/test_simulation_snapping.py
git commit -m "feat(simulation): price the snapped contract; key the initial valuation phase"
```

---

### Task 3: KO-reset product inputs — explicit V1 tenor and the ABSOLUTE conversion

**Files:**
- Modify: `quantark/asset/equity/product/option/ko_reset_snowball_option.py:513-570` and a new method after `time_shift`
- Modify: `quantark/asset/equity/lifecycle/state.py:30-60`
- Test: `test/lifecycle_ko_reset/test_ko_reset_product.py`

**Interfaces:**
- Produces: `KnockOutResetSnowballOption.get_maturity_payoff_v1(spot, pricing_env=None, *, contract_tenor: Optional[float] = None)`.
- Produces: `KnockOutResetSnowballOption.with_absolute_schedule_2(knock_in_time: float) -> KnockOutResetSnowballOption` — `REBASED` only; schedule-2 records at `knock_in_time + offset` (settlement times shifted the same), `post_ko_mode = ABSOLUTE`, `maturity = knock_in_time + max(offsets)`; `knock_in_time` on the product's own clock (years from its numeric origin; negative allowed).
- Produces: `AutocallableLifecycleState.observed_post_ko_indices: set[int]`.

- [ ] **Step 1: Write the failing tests**

```python
from dataclasses import replace

import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .conftest import env, schedule

D = 1 / 365.0


def ko_reset(mode, post, *, ki_times=(91 * D, 182 * D), annualized=False, maturity=None, accrued_offset=0.0):
    pre = [91 * D, 182 * D]
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=maturity or (max(ki_times) + max(post)),
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=schedule(pre, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=schedule(list(ki_times), 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=schedule(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=annualized, accrued_offset=accrued_offset),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)


def test_an_explicit_v1_tenor_is_the_resolved_one_on_absolute():
    product = ko_reset(PostKOScheduleMode.ABSOLUTE, [273 * D, 365 * D], annualized=True)
    tenor = product._resolve_post_contract_tenor(env())
    assert product.get_maturity_payoff_v1(60.0, env(), contract_tenor=tenor) == product.get_maturity_payoff_v1(60.0, env())


def test_a_rebased_annualized_v1_with_its_tenor_prices():
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], annualized=True)
    with pytest.raises(ValidationError, match="ki_time is required"):
        product.get_maturity_payoff_v1(60.0, env())
    converted = product.with_absolute_schedule_2(91 * D)
    assert product.get_maturity_payoff_v1(60.0, env(), contract_tenor=91 * D + 365 * D) == pytest.approx(
        converted.get_maturity_payoff_v1(60.0, env()), rel=1e-15)


def test_the_conversion_places_schedule_two_after_the_knock_in():
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D])
    converted = product.with_absolute_schedule_2(182 * D)
    times = [r.observation_time for r in converted.post_barrier_config.ko_observation_schedule.records]
    assert times == [182 * D + 91 * D, 182 * D + 365 * D]
    assert converted.post_ko_mode == PostKOScheduleMode.ABSOLUTE
    assert converted.maturity == 182 * D + 365 * D
    assert product.post_ko_mode == PostKOScheduleMode.REBASED          # the original is untouched
    with pytest.raises(ValidationError):
        ko_reset(PostKOScheduleMode.ABSOLUTE, [273 * D]).with_absolute_schedule_2(0.1)


@pytest.mark.parametrize("engine", [
    lambda: KOResetSnowballPDESolver(PDEParams()),
    lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
    lambda: KOResetSnowballQuadEngineV2(),
    lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=8192)),
])
def test_a_converted_knocked_in_contract_prices_as_the_absolute_contract(engine):
    rebased = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D]).with_absolute_schedule_2(91 * D)
    direct = ko_reset(PostKOScheduleMode.ABSOLUTE, [182 * D, 456 * D], maturity=456 * D)
    for product in (rebased, direct):
        setattr(product, "_otc_lifecycle_knocked_in", True)
    assert engine().price(rebased, env(90.0)) == pytest.approx(engine().price(direct, env(90.0)), rel=1e-12)


def test_the_state_records_observed_schedule_two_indices():
    from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
    assert AutocallableLifecycleState().observed_post_ko_indices == set()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_ko_reset_product.py -q`
Expected: FAIL — `unexpected keyword argument 'contract_tenor'`, `no attribute 'with_absolute_schedule_2'`, `no attribute 'observed_post_ko_indices'`.

- [ ] **Step 3: Implement**

`get_maturity_payoff_v1(self, spot, pricing_env=None, *, contract_tenor: Optional[float] = None)`; in the annualized branch `contract_tenor = self._resolve_post_contract_tenor(pricing_env) if contract_tenor is None else contract_tenor`. Docstring: "``contract_tenor`` is the knocked-in contract's tenor resolved by a caller that knows the knock-in time (a REBASED contract cannot resolve it alone); ``None`` resolves it here."

```python
    def with_absolute_schedule_2(self, knock_in_time: float) -> "KnockOutResetSnowballOption":
        """The ABSOLUTE contract a REBASED one becomes once knocked in at ``knock_in_time``.

        ``knock_in_time`` is on this product's own clock (years from its numeric origin; negative for a
        knock-in before it). Every schedule-2 record moves to ``knock_in_time + offset`` keeping its
        barrier, rate and accrual; the contract ends at ``knock_in_time + max(offsets)``, the REBASED
        maturity (``get_post_maturity_time(ki_time)``), so V1 is paid and discounted when it ends.
        """
        if self.post_ko_mode != PostKOScheduleMode.REBASED:
            raise ValidationError("with_absolute_schedule_2 converts a REBASED contract only")
        schedule = self.post_barrier_config.ko_observation_schedule
        if schedule is None or not schedule.records:
            raise ValidationError("a REBASED contract needs its schedule-2 offsets to be placed")
        shift = float(knock_in_time)
        records = [
            replace(r, observation_time=shift + float(r.observation_time),
                    settlement_time=None if r.settlement_time is None else shift + float(r.settlement_time))
            for r in schedule.records
        ]
        product = deepcopy(self)
        product.post_barrier_config = replace(
            self.post_barrier_config, ko_observation_schedule=replace(schedule, records=records),
            ko_observation_dates=None,
        )
        product.post_ko_mode = PostKOScheduleMode.ABSOLUTE
        product.maturity = shift + max(float(r.observation_time) for r in schedule.records)
        return product
```

(Import `deepcopy`. A REBASED record's `settlement_time` is an offset like its observation time — the MC REBASED route passes it to the resolver beside `ki_time + offset`; ledger `Ruling:` if the measured engines disagree.)

`state.py`: add `observed_post_ko_indices: set[int] = field(default_factory=set)` after `observed_coupon_indices` with a comment "KO-reset schedule-2 records already observed (or consumed by the knock-in)".

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_ko_reset_product.py -q`
Expected: 8 passed. If MC differs beyond 1e-12 because the two grids differ (the converted contract keeps its schedule-1 KI records), compare MC with `rel=1e-9` only after checking `_build_time_grid_ko_reset` produces the same `all_times`; a mismatch there is a `Ruling:` line, not a widened tolerance.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/product/option/ko_reset_snowball_option.py quantark/asset/equity/lifecycle/state.py test/lifecycle_ko_reset/test_ko_reset_product.py
git commit -m "feat(ko-reset): explicit V1 tenor and the ABSOLUTE contract a knocked-in REBASED one becomes"
```

---

### Task 4: MC `REBASED` route — accrued offset, V1 at each path's maturity, V1 tenor per path

**Files:**
- Modify: `quantark/asset/equity/engine/mc/snowball_mc_engine.py:2140-2215` (REBASED knock-out accrual, V1 block)
- Test: `test/lifecycle_ko_reset/test_mc_rebased.py`

**Interfaces:**
- Consumes: `get_maturity_payoff_v1(..., contract_tenor=)` (Task 3).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.util.enum.option_enums import PostKOScheduleMode

from .conftest import env
from .test_ko_reset_product import D, ko_reset


def _mc():
    return SnowballMCEngine(params=MCParams(seed=11, num_paths=16384))


def test_an_aged_rebased_knock_out_keeps_the_accrued_offset():
    """One knock-in date, so the REBASED contract IS the ABSOLUTE one at 0.25 + offsets: 7.5, not 6.5."""
    rebased = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], ki_times=(91 * D,), annualized=True,
                       accrued_offset=0.1)
    absolute = ko_reset(PostKOScheduleMode.ABSOLUTE, [182 * D, 456 * D], ki_times=(91 * D,), annualized=True,
                        accrued_offset=0.1, maturity=456 * D)
    assert _mc().price(rebased, env(90.0)) == pytest.approx(_mc().price(absolute, env(90.0)), rel=1e-12)


def test_an_annualized_rebased_v1_prices():
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], annualized=True)
    assert np.isfinite(_mc().price(product, env(80.0)))


def test_v1_is_read_at_each_paths_own_maturity():
    """Paths knocked in at 91 days mature at 91 + 365; the grid runs on to 182 + 365."""
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D])
    engine = _mc()
    pricing_env = env(80.0)
    grid = engine._build_time_grid_ko_reset(product, pricing_env, product.get_max_maturity_time(pricing_env))
    times = grid["all_times"]
    paths = np.full((1, times.size + 1), 70.0)               # knocked in at 91 days, never above 95
    maturity_node = int(np.searchsorted(times, 91 * D + 365 * D))
    paths[0, maturity_node + 1] = 60.0
    paths[0, -1] = 40.0                                       # the grid's last node: another time
    payoffs, _, stats, _ = engine._compute_payoffs_ko_reset(product, pricing_env, paths, grid, 0.03,
                                                            float(times[-1]), 0.25, 1)
    assert stats["v1_count"] == 1
    assert payoffs[0] == pytest.approx(product.get_maturity_payoff_v1(60.0, pricing_env, contract_tenor=456 * D))
```

Adapt the `_compute_payoffs_ko_reset` positional arguments to its signature (`product, pricing_env, paths, grid, r, T, sigma, rng_seed`).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_mc_rebased.py -q`
Expected: FAIL — 6.5-vs-7.5 price gap; `ki_time is required for rebased post-KO maturity`; V1 read at 40.0.

- [ ] **Step 3: Implement**

In the REBASED knock-out block replace `else: initial_to_valuation = 0.0` by

```python
                elif annualized_ko:
                    # no initial date: the accrual already elapsed is banked here, as
                    # compute_ko_accrual_factor reads it (accrued_offset + observation time)
                    initial_to_valuation = float(product.accrual_config.accrued_offset)
                else:
                    initial_to_valuation = 0.0
```

Replace the V1 block's spots and payoffs:

```python
        if is_v1.any():
            v1_paths = np.flatnonzero(is_v1)
            tenors = None
            if post_mode == PostKOScheduleMode.ABSOLUTE or len(post_times) == 0:
                terminal_spots = paths[is_v1, -1]
            else:
                # each path matures at its own ki_time + max(offset): read V1 there, on that tenor
                last = int(np.argmax(post_times))
                nodes = np.array([post_indices_by_ki[int(first_ki_idx[i])][last] for i in v1_paths], dtype=int)
                terminal_spots = paths[v1_paths, nodes + 1]
                tenors = ki_time[v1_paths] + post_max_offset + product._elapsed_since_origin()
            v1_payoffs = np.array(
                [
                    product.get_maturity_payoff_v1(
                        float(spot), pricing_env, contract_tenor=None if tenors is None else float(tenors[n])
                    )
                    for n, spot in enumerate(terminal_spots)
                ],
                dtype=float,
            )
            payoffs[is_v1] = v1_payoffs
```

(keep the settlement-time code below unchanged). The ABSOLUTE branch passes `contract_tenor=None` and reads `paths[is_v1, -1]`: bit-identical.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_mc_rebased.py -q`
Expected: 3 passed.

- [ ] **Step 5: Run the KO-reset engine suites**

Run: `.venv/bin/python -m pytest -n 6 test/test_ko_reset_contract_rules.py test/test_autocallable_same_observation_tie.py test/test_autocallable_observation_at_valuation.py test/theta_boundary -q > "$WS/t4.log" 2>&1; tail -5 "$WS/t4.log"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/engine/mc/snowball_mc_engine.py test/lifecycle_ko_reset/test_mc_rebased.py
git commit -m "fix(mc): REBASED KO-reset keeps the accrued offset and reads V1 at each path's maturity on its tenor"
```

---

### Task 5: `KOResetLifecycleTracker`

**Files:**
- Create: `quantark/asset/equity/lifecycle/ko_reset.py`
- Modify: `quantark/asset/equity/engine/settlement_support.py` (add `resolve_terminal_timing_at`)
- Modify: `quantark/asset/equity/lifecycle/__init__.py`
- Test: `test/lifecycle_ko_reset/test_tracker_rules.py`

**Interfaces:**
- Consumes: Task 1 hooks, Task 3 product methods and state field.
- Produces: `KOResetLifecycleTracker(product, quantity, lifecycle=None, start_date=None, date_resolver=None, has_lifecycle=True)` with `observe`, `settle_maturity_if_due`, `product_for_lifecycle`, `product_for_pricing(date, env, *, initial=False)`, `knock_in_time() -> Optional[float]`, `contract() -> product` (the contract in force: the snapped contract, or its ABSOLUTE conversion once a REBASED contract is knocked in), `pre_contract_tenor(env) -> float`, `final_contract_tenor(env) -> float`.
- Produces: `resolve_terminal_timing_at(product, pricing_env, *, determination_time=None, determination_date=None, cashflow_id="terminal") -> ResolvedPaymentTiming`.
- Cashflow ids: `knock-out:ko1:{i}`, `knock-out:ko2:{i}`, `maturity`; KNOCK_OUT/KNOCK_IN metadata carry `"schedule": 1 | 2`.

- [ ] **Step 1: Write the failing tests**

A shared helper drives the tracker the way `PortfolioLifecycleManager._process_autocallable` does:

```python
import pandas as pd
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, LifecycleEventType as E
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env
from .test_ko_reset_product import D, ko_reset


def day(n):
    return START + pd.Timedelta(days=n)


def run(tracker, closes):
    """closes: {day number: spot}; returns [(day, event type, cashflow id)]."""
    out = []
    for n, spot in sorted(closes.items()):
        e = env(spot, day(n))
        product = tracker.product_for_lifecycle()
        events = tracker.observe(day(n), product, e, spot)
        maturity = tracker.settle_maturity_if_due(day(n), product, e, spot)
        events += [maturity] if maturity is not None else []
        out += [(n, ev.event_type, getattr(ev.realized_cashflow, "cashflow_id", None)) for ev in events]
        if not tracker.lifecycle.alive:
            break
    return out


def tracker(product, **kw):
    return KOResetLifecycleTracker(product=product, quantity=1.0, start_date=START, **kw)


ABS = lambda **kw: ko_reset(M.ABSOLUTE, [273 * D, 365 * D], maturity=365 * D, **kw)   # noqa: E731
REB = lambda **kw: ko_reset(M.REBASED, [91 * D, 365 * D], **kw)                        # noqa: E731


def test_a_schedule_one_knock_out_ends_the_contract():
    assert run(tracker(ABS()), {91: 104.0}) == [(91, E.KNOCK_OUT, "knock-out:ko1:0")]


def test_knock_out_wins_a_tie():
    product = ABS()
    from dataclasses import replace
    product.barrier_config = replace(product.barrier_config, ki_barrier=105.0)   # one spot breaches both
    product.barrier_config.ki_observation_schedule.records[0].barrier = 105.0
    t = tracker(product)
    assert run(t, {91: 104.0}) == [(91, E.KNOCK_OUT, "knock-out:ko1:0")] and not t.lifecycle.knocked_in


def test_a_knock_in_switches_and_tests_schedule_two_the_same_close():
    product = ko_reset(M.ABSOLUTE, [91 * D, 365 * D], maturity=365 * D)
    product.post_barrier_config.ko_observation_schedule.records[0].barrier = 70.0
    assert run(tracker(product), {91: 72.0}) == [(91, E.KNOCK_IN, None), (91, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_a_knock_in_on_schedule_ones_last_observation_switches_instead_of_maturing():
    t = tracker(ABS())
    assert run(t, {91: 90.0, 182: 70.0}) == [(182, E.KNOCK_IN, None)]
    assert t.lifecycle.alive and t.lifecycle.knocked_in


@pytest.mark.parametrize("engine", [
    lambda: KOResetSnowballPDESolver(PDEParams()), lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=2001)),
    lambda: KOResetSnowballQuadEngineV2(), lambda: SnowballMCEngine(params=MCParams(seed=7, num_paths=4096)),
])
def test_unknocked_at_schedule_ones_end_matures_with_v0_equal_to_the_engines(engine):
    t = tracker(ABS(annualized=True))
    events = run(t, {91: 90.0, 182: 90.0})
    assert events == [(182, E.MATURITY, "maturity")]
    cash = t.lifecycle.ledger.cashflows[0].amount
    assert cash == pytest.approx(0.05 * 100.0 * 182 * D, rel=1e-12)
    fresh = tracker(ABS(annualized=True))
    run(fresh, {91: 90.0})
    decided = engine().price(fresh.product_for_pricing(day(182), env(90.0, day(182)), initial=True), env(90.0, day(182)))
    assert decided == pytest.approx(cash, rel=1e-9)


def test_a_schedule_one_maturity_settles_with_the_terminal_lag():
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
    product = ABS()
    product.settlement_convention = SettlementConvention(lag=2 / 365.0, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    product.validate()
    t = tracker(product)
    run(t, {91: 90.0, 182: 90.0})
    cf = t.lifecycle.ledger.cashflows[0]
    assert cf.payment_time == pytest.approx(cf.determination_time + 2 / 365.0)


def test_continuous_knock_in_is_tested_on_schedule_ones_last_close():
    from dataclasses import replace
    product = ABS()
    product.barrier_config = replace(product.barrier_config, ki_continuous=True, ki_observation_schedule=None)
    product.validate()
    assert run(tracker(product), {182: 70.0}) == [(182, E.KNOCK_IN, None)]


def test_rebased_schedule_two_follows_the_knock_in_date():
    t = tracker(REB())
    assert run(t, {91: 70.0, 182: 96.0}) == [(91, E.KNOCK_IN, None), (182, E.KNOCK_OUT, "knock-out:ko2:0")]


@pytest.mark.parametrize("ki_day, maturity_day", [(91, 456), (182, 547)])
def test_a_knocked_in_rebased_contract_matures_on_its_own_tenor(ki_day, maturity_day):
    t = tracker(REB())
    closes = {ki_day: 70.0, maturity_day - 1: 80.0, maturity_day: 60.0}
    if ki_day == 182:
        closes[91] = 90.0
    events = run(t, closes)
    assert events[-1] == (maturity_day, E.MATURITY, "maturity")
    product = t.contract()
    assert t.lifecycle.ledger.cashflows[-1].amount == pytest.approx(
        product.get_maturity_payoff_v1(60.0, env(60.0), contract_tenor=maturity_day * D))


@pytest.mark.parametrize("carried", [False, True])
def test_disable_ko_after_ki_tests_no_schedule_two_knock_out(carried):
    from dataclasses import replace
    product = ko_reset(M.ABSOLUTE, [91 * D, 273 * D], maturity=273 * D)
    product.barrier_config = replace(product.barrier_config, disable_ko_after_ki=True)
    product.post_barrier_config.ko_observation_schedule.records[0].barrier = 70.0
    if carried:
        setattr(product, "_otc_lifecycle_knocked_in", True)
        assert run(tracker(product), {91: 99.0, 273: 99.0})[-1][1] == E.MATURITY
    else:
        assert run(tracker(product), {91: 72.0, 273: 99.0}) == [(91, E.KNOCK_IN, None), (273, E.MATURITY, "maturity")]


def test_a_due_day_on_a_holiday_fires_on_the_next_close():
    assert run(tracker(ABS()), {90: 99.0, 93: 104.0}) == [(93, E.KNOCK_OUT, "knock-out:ko1:0")]


def test_holiday_last_observation_knock_in_switches():
    """Review focus 1: schedule 1's last record and a knock-in on a holiday: KI, not V0 maturity."""
    t = tracker(ABS())
    assert run(t, {91: 90.0, 181: 90.0, 184: 70.0}) == [(184, E.KNOCK_IN, None)]
    assert t.lifecycle.alive


def test_first_close_consumes_nothing():
    """Review focus 2: a run starting after records are due observes them all on its first close."""
    product = ko_reset(M.ABSOLUTE, [91 * D, 365 * D], maturity=365 * D)
    product.post_barrier_config.ko_observation_schedule.records[0].barrier = 70.0
    assert run(tracker(product), {95: 72.0}) == [(95, E.KNOCK_IN, None), (95, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_schedule_two_records_before_the_knock_in_close_are_consumed():
    product = ko_reset(M.ABSOLUTE, [73 * D, 365 * D], maturity=365 * D)    # 0.2 years
    product.post_barrier_config.ko_observation_schedule.records[0].barrier = 70.0
    t = tracker(product)
    assert run(t, {73: 90.0, 91: 90.0, 182: 72.0}) == [(182, E.KNOCK_IN, None)]
    assert 0 in t.lifecycle.observed_post_ko_indices


def test_a_schedule_two_record_delayed_by_a_missing_close_is_still_tested():
    t = tracker(ko_reset(M.ABSOLUTE, [273 * D, 365 * D], maturity=365 * D))
    assert run(t, {91: 70.0, 272: 99.0, 275: 96.0}) == [(91, E.KNOCK_IN, None), (275, E.KNOCK_OUT, "knock-out:ko2:0")]


def test_a_carried_knock_in_is_kept():
    product = ABS()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    t = tracker(product)
    assert t.lifecycle.knocked_in and t.product_for_lifecycle()._otc_lifecycle_knocked_in


def test_a_carried_rebased_knock_in_without_a_date_is_refused():
    product = REB()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    with pytest.raises(ValidationError, match="knock-in date"):
        tracker(product)


def test_carried_rebased_knock_in_before_start():
    """Review focus 5: knocked in 100 days before the run: schedule 2 at -100 + (91, 365) days."""
    product = REB()
    setattr(product, "_otc_lifecycle_knocked_in", True)
    setattr(product, "_otc_lifecycle_knock_in_date", (START - pd.Timedelta(days=100)).to_pydatetime())
    t = tracker(product)
    copy = t.product_for_pricing(day(1), env(90.0, day(1)))
    assert [r.observation_time for r in copy.post_barrier_config.ko_observation_schedule.records] == [
        pytest.approx(264 * D)]
    assert run(t, {1: 99.0, 265: 80.0})[-1] == (265, E.MATURITY, "maturity")


def test_a_knocked_in_rebased_pricing_copy_is_the_absolute_contract():
    t = tracker(REB())
    run(t, {91: 70.0})
    copy = t.product_for_pricing(day(100), env(90.0, day(100)))
    assert copy.post_ko_mode == M.ABSOLUTE and copy._otc_lifecycle_knocked_in
    assert copy.maturity == pytest.approx((91 + 365 - 100) * D)
    assert copy._otc_lifecycle_knock_in_date == day(91).to_pydatetime()
```

Plus the intraday structural check (spec §7 "agrees with the intraday twin"): in `test/lifecycle_ko_reset/test_tracker_twin.py`, reuse `test/intraday/conftest.py`'s `dated_ko_reset`, `sse_calendar`, `sse_sessions`, `flat_env`, and `test_ko_reset_timeline._checkpoint`'s pattern: at checkpoint 2026-09-14 (knocked in and not), `resolve_context(...).numerical.product` has as many schedule-1 and schedule-2 records left as `KOResetLifecycleTracker(product=dated_ko_reset(...), quantity=1.0, start_date=T0, lifecycle=state).product_for_pricing(2026-09-14, env)` and the same `_otc_lifecycle_knocked_in`.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_tracker_rules.py test/lifecycle_ko_reset/test_tracker_twin.py -q`
Expected: FAIL — `ImportError: cannot import name 'KOResetLifecycleTracker'`.

- [ ] **Step 3: Add `resolve_terminal_timing_at`**

```python
def resolve_terminal_timing_at(
    product,
    pricing_env: "PricingEnvironment",
    *,
    determination_time: Optional[float] = None,
    determination_date: Optional[datetime] = None,
    cashflow_id: str = "terminal",
) -> ResolvedPaymentTiming:
    """A terminal determination/payment pair at a given determination (a KO-reset's schedule-1 or
    knocked-in maturity, which is not the product's own maturity)."""
    if (determination_time is None) == (determination_date is None):
        raise ValidationError("give exactly one of determination_time and determination_date")
    request = SettlementRequest(kind=CashflowKind.TERMINAL, determination_date=determination_date,
                                determination_time=determination_time, cashflow_id=cashflow_id)
    return SettlementResolver.resolve_contingent(product, request, pricing_env)
```

- [ ] **Step 4: Write `ko_reset.py`**

```python
"""Lifecycle tracking for the knock-out-reset snowball (spec 2026-10-08 KO-reset lifecycle).

R1 a knock-in during schedule 1 replaces it by schedule 2 for good, from that observation (inclusive);
R2 not knocked in when schedule 1 ends: matured there with V0 on schedule 1's tenor;
R3 a schedule-1 knock-out beats a knock-in at the same observation;
R4 a carried knock-in happened.
A knocked-in REBASED contract is the ABSOLUTE contract ``with_absolute_schedule_2`` builds at the
knock-in time, so every engine prices an ordinary knocked-in contract.
"""
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing_at
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .autocallable import AutocallableLifecycleTracker
from .events import LifecycleEvent, LifecycleEventType
from .state import AutocallableLifecycleState

DAYS_PER_YEAR = 365.0


class KOResetLifecycleTracker(AutocallableLifecycleTracker):
    """Tracks one ``KnockOutResetSnowballOption`` (rules R1-R4 above)."""

    def __init__(self, *, product, quantity, lifecycle=None, start_date=None, date_resolver=None,
                 has_lifecycle=True) -> None:
        if lifecycle is None:
            knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
            lifecycle = AutocallableLifecycleState(knocked_in=knocked_in)
            if knocked_in:
                lifecycle.ki_date = getattr(product, "_otc_lifecycle_knock_in_date", None)
        if (lifecycle.knocked_in and lifecycle.ki_date is None
                and product.post_ko_mode == PostKOScheduleMode.REBASED):
            raise ValidationError(
                "KO-reset snowball: a carried knock-in on a REBASED contract needs its knock-in date "
                "(_otc_lifecycle_knock_in_date): schedule 2 cannot be placed without it"
            )
        super().__init__(product=product, quantity=quantity, lifecycle=lifecycle, start_date=start_date,
                         date_resolver=date_resolver, has_lifecycle=has_lifecycle)
        self._previous_close: Optional[pd.Timestamp] = None

    # -- the contract in force ---------------------------------------------

    def knock_in_time(self) -> Optional[float]:
        """Years from ``start_date`` to the knock-in close, on the tracker's numeric clock."""
        if not self.lifecycle.knocked_in or self.lifecycle.ki_date is None or self.start_date is None:
            return None
        days = (pd.Timestamp(self.lifecycle.ki_date).normalize() - pd.Timestamp(self.start_date).normalize()).days
        return days / DAYS_PER_YEAR

    def contract(self) -> Any:
        """The snapped contract, or the ABSOLUTE contract a knocked-in REBASED one has become."""
        if self.lifecycle.knocked_in and self.product.post_ko_mode == PostKOScheduleMode.REBASED:
            return self.product.with_absolute_schedule_2(self.knock_in_time())
        return self.product

    def product_for_lifecycle(self) -> Any:
        product = super().product_for_lifecycle()
        setattr(product, "_otc_lifecycle_knock_in_date", self.lifecycle.ki_date)
        return product

    def _contract_for_pricing(self) -> Any:
        from copy import deepcopy
        product = deepcopy(self.contract())
        setattr(product, "_otc_lifecycle_knocked_in", self.lifecycle.knocked_in)
        setattr(product, "_otc_lifecycle_knock_in_date", self.lifecycle.ki_date)
        return product

    def _age_extra(self, product, elapsed, date, pricing_env, keep_landing) -> None:
        if product.post_ko_mode == PostKOScheduleMode.ABSOLUTE:
            product.post_barrier_config, _ = product.post_barrier_config.time_shift(
                elapsed, date, pricing_env, keep_landing=keep_landing)

    # -- records ------------------------------------------------------------

    def _schedule_records(self, product, env, config) -> List[dict]:
        """One schedule's knock-out records, cash = the product's own ``_ko_cash`` (schedule 1 and 2 alike)."""
        schedule_env = self._schedule_resolution_env(product, env)
        resolved, rates, sources = product._resolve_ko_schedule(config, schedule_env)
        payoffs = [product._ko_cash(r.observation_time, rate, src, schedule_env)
                   for r, rate, src in zip(resolved, rates, sources)]
        return self._record_dicts(product, resolved, payoffs)

    def pre_contract_tenor(self, env) -> float:
        return float(self.product._resolve_pre_contract_tenor(self._schedule_resolution_env(self.product, env)))

    def final_contract_tenor(self, env) -> float:
        contract = self.contract()
        return float(contract._resolve_post_contract_tenor(self._schedule_resolution_env(contract, env)))

    def _pre_timing(self, product, env):
        records = self._schedule_records(product, env, product.barrier_config)
        last = records[-1]
        schedule_env = self._schedule_resolution_env(product, env)
        if last["date"] is not None and last["determination_date"] is not None and self._uses_date_timing(product):
            return resolve_terminal_timing_at(product, schedule_env, determination_date=last["determination_date"],
                                              cashflow_id="ko_reset_v0_terminal"), last
        return resolve_terminal_timing_at(product, schedule_env, determination_time=last["time"],
                                          cashflow_id="ko_reset_v0_terminal"), last

    def _final_timing(self, env):
        contract = self.contract()
        schedule_env = self._schedule_resolution_env(contract, env)
        end_date = contract._resolve_schedule_end_date(contract.post_barrier_config.ko_observation_schedule)
        if end_date is not None and self._uses_date_timing(contract):
            return resolve_terminal_timing_at(contract, schedule_env, determination_date=end_date,
                                              cashflow_id="ko_reset_v1_terminal")
        return resolve_terminal_timing_at(contract, schedule_env,
                                          determination_time=contract.get_post_maturity_time(schedule_env),
                                          cashflow_id="ko_reset_v1_terminal")

    # -- the day's close ------------------------------------------------------

    def observe(self, date, product, env, spot) -> List[LifecycleEvent]:
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive or not self.has_lifecycle:
            return []
        events: List[LifecycleEvent] = []
        try:
            if not self.lifecycle.knocked_in:
                knock_out = self._test_knock_outs(date, product, env, spot, valuation_point, schedule=1)
                if knock_out is not None:
                    return [knock_out]
                knock_in = self._test_knock_in(date, product, env, spot, valuation_point)
                if knock_in is None:
                    return events
                events.append(knock_in)
                self._consume_schedule_two(env)
            if not self.contract().barrier_config.disable_ko_after_ki:
                knock_out = self._test_knock_outs(date, self.contract(), env, spot, valuation_point, schedule=2)
                if knock_out is not None:
                    events.append(knock_out)
            return events
        finally:
            self._previous_close = pd.Timestamp(date).normalize()
```

`_test_knock_outs(date, product, env, spot, valuation_point, *, schedule)`: records = `_schedule_records(product, env, product.barrier_config if schedule == 1 else product.post_barrier_config)`; observed set = `observed_ko_indices` (1) or `observed_post_ko_indices` (2); for each unobserved due record (`_record_is_due`) mark observed; on `_barrier_hit(..., is_ko=True)` build the KNOCK_OUT exactly as the base tracker's block does (`_record_cashflow(KNOCK_OUT, f"knock-out:ko{schedule}:{idx}", ...)`, `mark_ko`, event metadata `{"payoff", "settlement_date", "schedule": schedule}`) and return it.

`_test_knock_in(date, product, env, spot, valuation_point)`: R2 — return `None` when schedule 1's last record was due at the previous close (`self._previous_close is not None and self._record_is_due(self._previous_close, self._valuation_point(self._previous_close, product), last)`); otherwise the base tracker's continuous / discrete knock-in test (copy the block; `observation_index` and `metadata={"schedule": 1, ...}`), returning the KNOCK_IN event or `None`.

`_consume_schedule_two(env)`: when `self._previous_close` is not `None`, every schedule-2 record of `self.contract()` already due at the previous close joins `observed_post_ko_indices` (never tested).

```python
    def settle_maturity_if_due(self, date, product, env, spot) -> Optional[LifecycleEvent]:
        valuation_point = self._valuation_point(date, product)
        self.lifecycle.valuation_point = valuation_point
        if not self.lifecycle.alive:
            return None
        if self.lifecycle.knocked_in:
            timing = self._final_timing(env)
            if not self._timing_is_due(date, valuation_point, timing):
                return None
            payoff = float(self.contract().get_maturity_payoff_v1(
                spot, self._schedule_resolution_env(self.contract(), env),
                contract_tenor=self.final_contract_tenor(env)))
        else:
            timing, last = self._pre_timing(product, env)
            if not self._record_is_due(date, valuation_point, last):
                return None
            payoff = float(product.get_maturity_payoff_v0(
                spot, self._schedule_resolution_env(product, env), contract_tenor=self.pre_contract_tenor(env)))
        return self._book_maturity(date, spot, payoff, timing, valuation_point)
```

`_book_maturity` is the tail of the base `settle_maturity_if_due` (snapshot, `_timing_cashflow(MATURITY, "maturity", ...)`, `mark_maturity`, event); extract it in the base as `_book_maturity(self, date, spot, payoff, timing, valuation_point)` and call it from both.

Export `KOResetLifecycleTracker` from `__init__.py`.

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_tracker_rules.py test/lifecycle_ko_reset/test_tracker_twin.py -q`
Expected: all pass. A rule test that fails on numbers (not on structure) is debugged against the spec, never by loosening the assertion.

- [ ] **Step 6: Run the lifecycle suites**

Run: `.venv/bin/python -m pytest -n 6 test/lifecycle_ko_reset test/test_equity_lifecycle_trackers.py test/test_lifecycle_calendar_schedule.py test/test_replay_goldens.py -q > "$WS/t5.log" 2>&1; tail -5 "$WS/t5.log"`
Expected: all pass (the base `_book_maturity` extraction is behaviour-neutral).

- [ ] **Step 7: Commit**

```bash
git add quantark/asset/equity/lifecycle/ quantark/asset/equity/engine/settlement_support.py test/lifecycle_ko_reset/
git commit -m "feat(lifecycle): KO-reset snowball tracker (R1-R4, REBASED as the ABSOLUTE contract it becomes)"
```

---

### Task 6: Manager, replay and the provider matrix

**Files:**
- Modify: `quantark/asset/equity/lifecycle/manager.py:118-150`
- Modify: `quantark/backtest/replay/product_replay.py:179-186`, `quantark/backtest/replay/engine_factory.py:128-137`, `quantark/backtest/replay/config.py:636-676`
- Modify: `quantark/backtest/simulation/pricing/repricing.py` (`_tracker_for`)
- Create: `quantark/asset/equity/lifecycle/providers.py` — `require_rebased_provider`
- Test: `test/lifecycle_ko_reset/test_consumers.py`; update `test/test_dynamic_scenario_lifecycle.py::test_ko_reset_snowball_warns_and_is_untracked`

**Interfaces:**
- Consumes: `KOResetLifecycleTracker` (Task 5).
- Produces: `tracker_for(product, **kwargs) -> AutocallableLifecycleTracker` (in `ko_reset.py`: KO-reset → `KOResetLifecycleTracker`, else the base class); `require_rebased_provider(product, *, engines: Mapping[str, Any], consumer: str, position) -> None` raising `ValidationError` for an un-knocked REBASED contract unless every named engine is MC (a `SnowballMCEngine` instance or `EngineType.MONTE_CARLO`).

- [ ] **Step 1: Measure the vol-model engines on a knocked-in KO-reset**

Write `scratchpad/ko_reset_vol_models.py`: build an `ABSOLUTE` knocked-in KO-reset and the knocked-in `SnowballOption` on its schedule 2 (the identity from `test/test_ko_reset_contract_rules.py::knocked_in_snowball`), price both with each of `LocalVolSnowballPDESolver`, `LocalVolSnowballMCEngine`, `HestonSnowballPDESolver`, `HestonSnowballMCEngine`, `HestonSLVSnowballPDESolver`, `HestonSLVQESnowballMCEngine` (constructors as in `engine_factory.create_vol_model_engine`, a flat LV surface / small Heston params), and print the relative gap. Run it with `.venv/bin/python` (guarded `if __name__ == "__main__":`, no heredoc — the stdin spawn trap). Expected: the PDE solvers price schedule 1 (gap ≫ 1e-10); the MC engines inherit `SnowballMCEngine`'s KO-reset route. Ledger the table: `Task 6: Ruling: vol-model routes allowed for KO-reset = {engines with gap ≤ 1e-10 (PDE) / within 3 standard errors on a shared seed (MC)}; others refused`.

- [ ] **Step 2: Write the failing tests**

```python
import warnings

import pandas as pd
import pytest

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.lifecycle import KOResetLifecycleTracker, PortfolioLifecycleManager
from quantark.backtest.replay.config import AutocallableEngineConfig
from quantark.backtest.replay.engine_factory import create_autocallable_engine
from quantark.util.enum.engine_enums import EngineType
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import START, env
from .test_ko_reset_product import D, ko_reset


def _portfolio(product, engine):
    from quantark.portfolio.equity import EquityPortfolio   # adapt to test/test_dynamic_scenario_lifecycle.py's builder
    ...


def test_the_manager_tracks_a_ko_reset_and_closes_it_at_schedule_ones_end():
    """Before: a warning, untracked, and the day after schedule 1's end every engine raised."""
    product = ko_reset(M.ABSOLUTE, [273 * D, 365 * D], maturity=365 * D)
    portfolio = _portfolio(product, engine=None)
    manager = PortfolioLifecycleManager(START.to_pydatetime())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        manager.register_positions(portfolio)
    assert manager.num_tracked == 1
    for n, spot in ((91, 90.0), (182, 90.0)):
        portfolio.pricing_environments["equity_index"] = env(spot, START + pd.Timedelta(days=n))
        events = manager.process_day(portfolio, n, (START + pd.Timedelta(days=n)).to_pydatetime())
    assert [e.event.event_type.name for e in events] == ["MATURITY"] and not portfolio.positions
    manager.process_day(portfolio, 183, (START + pd.Timedelta(days=183)).to_pydatetime())   # does not raise


def test_the_manager_refuses_an_unknocked_rebased_contract_off_mc():
    from quantark.asset.equity.engine.pde_engine import PDEEngine
    portfolio = _portfolio(ko_reset(M.REBASED, [91 * D, 365 * D]), engine=PDEEngine())
    with pytest.raises(ValidationError, match="REBASED.*Monte Carlo"):
        PortfolioLifecycleManager(START.to_pydatetime()).register_positions(portfolio)


def test_replay_quad_prices_a_ko_reset_on_its_own_engine():
    """4.195 (SnowballQuadEngine, schedule 2 ignored) vs -1.744 (KOResetSnowballQuadEngine)."""
    product = ko_reset(M.ABSOLUTE, [273 * D, 365 * D], maturity=365 * D)
    engine = create_autocallable_engine(product=product, engine_type=EngineType.QUADRATURE,
                                        config=AutocallableEngineConfig(pricing_engine_type=EngineType.QUADRATURE))
    assert isinstance(engine, KOResetSnowballQuadEngine)


def test_replay_selects_the_ko_reset_tracker():
    from quantark.backtest.replay.product_replay import ProductReplay
    assert isinstance(_replay(ko_reset(M.ABSOLUTE, [273 * D], maturity=273 * D))._tracker, KOResetLifecycleTracker)


@pytest.mark.parametrize("engine_type", [EngineType.PDE, EngineType.QUADRATURE])
def test_the_replay_config_refuses_an_unknocked_rebased_contract_off_mc(engine_type):
    with pytest.raises(ValidationError, match="REBASED"):
        _replay_config(ko_reset(M.REBASED, [91 * D, 365 * D]), engine_type)


def test_an_mc_replay_of_a_rebased_contract_runs_through_its_knock_in():
    ...   # MC pricing + MC event stats/surface engines; a short dataset whose spot falls through 75 on day 91
```

Fill `_portfolio`, `_replay`, `_replay_config` and the MC-replay case from the existing builders: `grep -n "def _make_portfolio\|EquityPortfolio(\|add_position" test/test_dynamic_scenario_lifecycle.py` and `grep -n "def _config\|ReplayBacktestConfig(\|AutocallableMarketDataSet" test/test_replay_engine_unified.py`, copying their construction verbatim with the KO-reset product. The MC replay case asserts the knock-in event is booked, and that a knocked-in day's price on a PDE engine of the same pricing copy (`replay.product_for_date(day, env)`) is finite (§7: "its knocked-in days price on PDE and QUAD too").

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_consumers.py -q`
Expected: FAIL — the manager warns and tracks nothing; QUAD returns `SnowballQuadEngine`; replay builds the base tracker; no refusals.

- [ ] **Step 4: Implement**

`ko_reset.py`:

```python
def tracker_for(product, **kwargs) -> AutocallableLifecycleTracker:
    """The tracker class a product's rules need."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    cls = KOResetLifecycleTracker if isinstance(product, KnockOutResetSnowballOption) else AutocallableLifecycleTracker
    return cls(product=product, **kwargs)
```

`providers.py`:

```python
"""Who prices an un-knocked REBASED KO-reset (spec section 6a): Monte Carlo only."""
from typing import Any, Mapping

from quantark.util.enum.engine_enums import EngineType
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError


def _is_mc(engine: Any) -> bool:
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    return engine == EngineType.MONTE_CARLO or isinstance(engine, SnowballMCEngine)


def require_rebased_provider(product, *, engines: Mapping[str, Any], consumer: str, position: Any) -> None:
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    if not isinstance(product, KnockOutResetSnowballOption) or product.post_ko_mode != PostKOScheduleMode.REBASED:
        return
    if getattr(product, "_otc_lifecycle_knocked_in", False):
        return
    off = {role: engine for role, engine in engines.items() if not _is_mc(engine)}
    if off:
        raise ValidationError(
            f"{consumer}: position {position} is an un-knocked KO-reset snowball with a REBASED schedule 2, "
            "which only Monte Carlo prices (SnowballMCEngine places schedule 2 after each simulated knock-in); "
            f"got {', '.join(f'{role}={engine!r}' for role, engine in off.items())}"
        )
```

`manager.register_positions`: delete the KO-reset warning branch; for `SnowballOption`/`PhoenixOption` (KO-reset included) call `require_rebased_provider(product, engines={"pricing": position.engine}, consumer="PortfolioLifecycleManager", position=position_id)` then `tracker = tracker_for(product, quantity=position.quantity, start_date=self.base_date)`.

`product_replay.py`: `self._tracker = tracker_for(product, quantity=product_quantity, lifecycle=lifecycle, start_date=start_date, date_resolver=self._next_available_market_date, has_lifecycle=has_lifecycle)`. A KO-reset replay passes `lifecycle=None` when the engine built a fresh default state, so the tracker builds it from the product (§4): in `replay/engine.py:185` pass `lifecycle=None` for a KO-reset and read `replay._tracker.lifecycle` back into the engine's state slot.

`engine_factory.py` QUAD branch, before the `SnowballOption` test:

```python
        if isinstance(product, KnockOutResetSnowballOption):
            return KOResetSnowballQuadEngine(params=quad_params)
```

`config.py` `ReplayBacktestConfig.__post_init__`: for each product call `require_rebased_provider(bp.product, engines={"pricing": ec.pricing_engine_type, **({"surface": ec.resolve_surface_engine_type()} if self.calculate_surfaces else {}), **({"event_stats": ec.resolve_event_stats_engine_type()} if self.calculate_event_probabilities else {})}, consumer="ReplayBacktestConfig", position=bp.position_id)`; in the `vol_model != "bsm"` block refuse a `KnockOutResetSnowballOption` on every route Step 1 ruled out, naming the measured gap.

`repricing.py` `_tracker_for(knocked_in, knock_in_day)`: `tracker_for(self.product, quantity=1.0, has_lifecycle=True, lifecycle=AutocallableLifecycleState(knocked_in=knocked_in, ki_date=None if knock_in_day < 0 else self._dates[knock_in_day].to_pydatetime()), start_date=self.start_date)` — `RepricingPricer` gains an optional `dates` argument (the engine passes the path calendar) used only for `knock_in_day`.

Update `test_ko_reset_snowball_warns_and_is_untracked` to assert the KO-reset is tracked (rename `test_ko_reset_snowball_is_tracked`).

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_consumers.py test/test_dynamic_scenario_lifecycle.py -q`
Expected: all pass.

- [ ] **Step 6: Run the consumer suites**

Run: `.venv/bin/python -m pytest -n 6 test/test_backtest_lifecycle.py test/test_replay_engine_unified.py test/test_replay_goldens.py test/test_pnlexplain_backtest_replay.py test/test_float_maturity_roll.py test/test_sacva_ko_reset_exposure.py test/simulation -q > "$WS/t6.log" 2>&1; tail -5 "$WS/t6.log"`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add quantark/asset/equity/lifecycle/ quantark/backtest/replay/ quantark/backtest/simulation/pricing/repricing.py test/lifecycle_ko_reset/test_consumers.py test/test_dynamic_scenario_lifecycle.py
git commit -m "feat(lifecycle): track KO-reset in the manager and replay; QUAD routes to its own engine; REBASED provider matrix"
```

---

### Task 7: Simulated-path backtest for KO-reset

**Files:**
- Modify: `quantark/asset/equity/lifecycle/ko_reset.py` (add `SchedulePlacement`, `KOResetCalendarSchedule`, `resolve_calendar_schedule`)
- Modify: `quantark/backtest/simulation/lifecycle.py` (`step` split into `_step_product`; `knock_in_day`; `KOResetVectorLifecycle`)
- Modify: `quantark/backtest/simulation/engine.py` (tracker, lifecycle class, `_price_book`, `_payoffs`, `_check_calendar`)
- Modify: `quantark/backtest/simulation/config.py:180-226`
- Test: `test/lifecycle_ko_reset/test_simulation_ko_reset.py`

**Interfaces:**
- Consumes: Tasks 2, 5, 6.
- Produces:

```python
@dataclass(frozen=True)
class SchedulePlacement:
    """Schedule 2 and the knocked-in maturity on the run calendar, for one knock-in day (or ABSOLUTE)."""
    ko_due_day: np.ndarray; ko_barrier: np.ndarray; ko_payoff: np.ndarray
    ko_settlement_day: np.ndarray; ko_settlement_delay: np.ndarray
    final_due_day: int; final_settlement_day: int; final_settlement_delay: float; final_tenor: float

@dataclass(frozen=True)
class KOResetCalendarSchedule(CalendarSchedule):
    pre_due_day: int; pre_settlement_day: int; pre_settlement_delay: float; pre_tenor: float
    rebased: bool
    absolute: Optional[SchedulePlacement]          # ABSOLUTE
    place: Optional[Callable[[int], SchedulePlacement]]   # REBASED: knock-in day -> placement (memoised)
```

  `CalendarSchedule.ko_*` = schedule 1; `terminal_*` = the latest terminal possible (ABSOLUTE final; REBASED: the placement of the last day a knock-in can be tested), so `_check_calendar` keeps working. `KOResetVectorLifecycle.knock_in_day: (n_products, n_paths) int64`, -1 while not knocked in. Payoff callback `payoff_fn(p, idx, knocked_in, tenors)`; `tenors` per path (V0: `pre_tenor`; V1: the placement's `final_tenor`).

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from quantark.backtest.replay import ReplayProduct
from quantark.backtest.simulation.conformance import run_oracle
from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from simulation.conftest import SPOT, ensemble_config, make_market_path, surface_pricing


def short_ko_reset(mode, *, pre=(2, 5), ki=(1, 3), post=(4, 7), maturity_days=8):
    """Day-scale KO-reset: schedule 1 on days `pre`, KI on `ki`, schedule 2 on `post` (offsets if REBASED)."""
    ...   # built like simulation.conftest.short_snowball, with KnockOutResetSnowballOption and SPOT levels


def _paths(kind, n_days=12):
    """Deterministic spots: 'ko1' knocks out on schedule 1, 'v0' matures un-knocked, 'ki_ko2' knocks in then out
    on schedule 2, 'ki_v1' knocks in then matures."""
    path = make_market_path(n_paths=1, n_days=n_days)
    levels = {"ko1": {2: 1.05}, "v0": {}, "ki_ko2": {1: 0.70, 4: 1.0}, "ki_v1": {1: 0.70}}[kind]
    spot = np.full(n_days, SPOT * 0.98)
    for d, k in levels.items():
        spot[d] = SPOT * k
    path.spot[0, :] = spot
    return path


@pytest.mark.parametrize("mode", [M.ABSOLUTE, M.REBASED])
@pytest.mark.parametrize("kind", ["ko1", "v0", "ki_ko2", "ki_v1"])
def test_exact_mode_matches_the_replay_to_the_last_bit(mode, kind):
    engine = mc_engine_config() if mode == M.REBASED else None
    cfg = ensemble_config(products=[ReplayProduct(product=short_ko_reset(mode), quantity=-1000.0, position_id=1,
                                                  has_lifecycle=True)],
                          **({"engine_config": engine} if engine else {}))
    report = run_oracle(cfg, _paths(kind), 0)
    assert report.passed, report.summary()
    assert report.max_pv_gap == 0.0 and report.max_total_pnl_gap == 0.0


def test_surface_mode_prices_an_absolute_ko_reset_within_the_gate():
    cfg = ensemble_config(products=[ReplayProduct(product=short_ko_reset(M.ABSOLUTE), quantity=-1000.0,
                                                  position_id=1, has_lifecycle=True)], pricing=surface_pricing())
    results = EnsembleBacktestEngine(cfg).run(make_market_path(n_paths=3, n_days=12))
    assert results.manifest["gate"]["passed"]


def test_surface_mode_refuses_a_rebased_ko_reset_at_configuration():
    with pytest.raises(ValidationError, match="REBASED"):
        ensemble_config(products=[ReplayProduct(product=short_ko_reset(M.REBASED), quantity=-1000.0,
                                                position_id=1, has_lifecycle=True)], pricing=surface_pricing())


def test_rebased_knock_in_past_calendar_is_data_end():
    """Review focus 4."""
    cfg = ensemble_config(products=[ReplayProduct(product=short_ko_reset(M.REBASED, post=(4, 30)), quantity=-1000.0,
                                                  position_id=1, has_lifecycle=True)],
                          engine_config=mc_engine_config(), allow_data_end=True)
    results = EnsembleBacktestEngine(cfg).run(_paths("ki_v1", n_days=12))
    assert results.manifest["data_end_paths"] == 1
```

`mc_engine_config()`: `AutocallableEngineConfig(pricing_engine_type=EngineType.MONTE_CARLO, mc_params=MCParams(seed=3, num_paths=2048))`. Before writing `short_ko_reset`, read `simulation/conftest.py::short_snowball` (lines 75-104) and copy its construction (factory, settlement convention handling) with `KnockOutResetSnowballOption`; `post` offsets/days in `D` units.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_simulation_ko_reset.py -q`
Expected: FAIL — `EnsembleConfig` refuses the KO-reset ("prices standard snowballs only").

- [ ] **Step 3: Implement the calendar schedule**

`KOResetLifecycleTracker.resolve_calendar_schedule(dates, env)`: build the base fields for schedule 1 via `super().resolve_calendar_schedule(dates, env)` (its `ko_*` come from `_scheduled_records`; replace them with `_schedule_records(product, env, product.barrier_config)` so the cash is `_ko_cash`), the KI fields unchanged. `pre_*` from `_pre_timing` with the base helpers `first_due` / `settlement_position` (lift both to private methods `_first_due(dates, is_due)` and `_settlement_position(dates, settlement_date, due_day)` in the base, behaviour unchanged). `_placement(dates, env, contract)` resolves a contract's schedule 2 (`_schedule_records(contract, env, contract.post_barrier_config)`) and `_final_timing` on `dates` into a `SchedulePlacement`. ABSOLUTE: `absolute = _placement(dates, env, self.product)`. REBASED: `place(d)` builds a scratch tracker with `ki_date = dates[d]` and returns its `_placement(dates, env, scratch.contract())`, memoised in a dict.

- [ ] **Step 4: Implement the vector lifecycle**

Split `VectorLifecycle.step`'s loop body into `_step_product(self, p, schedule, d, spot, payoff_fn)` (same four blocks, same order). `KOResetVectorLifecycle(VectorLifecycle)` adds `self.knock_in_day = np.full(shape, -1, int64)`, per-path placements for REBASED, `observed_post` per product, and overrides `_step_product` for a `KOResetCalendarSchedule`:

1. schedule-1 knock-out on `alive & ~knocked_in` (the base `_knock_out` with that eligibility);
2. knock-in on `alive & ~knocked_in & ~today_ko`, only while `d <= schedule.pre_due_day` (R2): set `knock_in_day[p][hit] = d`; REBASED paths take `schedule.place(d)`;
3. consume: for newly knocked-in paths, schedule-2 records with due day `< d` are marked observed (never tested);
4. schedule-2 knock-out on `alive & knocked_in` (fresh included) unless `disable_ko_after_ki`: records with due day `== d` and not observed, per path's placement;
5. schedule-1 maturity: `d == schedule.pre_due_day` on `alive & ~knocked_in` → `payoff_fn(p, idx, False, tenors=pre_tenor)`, settled with `pre_settlement_day/delay`;
6. final maturity of `alive & knocked_in` paths whose placement's `final_due_day == d` → `payoff_fn(p, idx, True, tenors=final_tenor per path)`, settled with the placement's day/delay; a placement with `final_due_day == -1` (past the calendar) leaves the path alive → counted by `data_end_paths`;
7. the base `_settle`.

The base class keeps calling `payoff_fn(p, idx, ki)`; the KO-reset class calls `payoff_fn(p, idx, ki, tenors)`.

- [ ] **Step 5: Wire the engine and the config**

`engine.py`: `_tracker` → `tracker_for(bp.product, quantity=..., has_lifecycle=True, lifecycle=AutocallableLifecycleState(), start_date=start_date)`; lifecycle class `KOResetVectorLifecycle` when any schedule is a `KOResetCalendarSchedule`, else `VectorLifecycle`; `_price_book` passes `knock_in_day=lifecycle.knock_in_day[p][idx]` for a REBASED product (else `None`); `_payoffs(..., tenors=None)`: for a KO-reset product `pricer.product.get_maturity_payoff_v1(spot, env, contract_tenor=t)` if knocked in else `get_maturity_payoff_v0(spot, env, contract_tenor=t)`; a snowball keeps `get_payoff`. `_make_pricers` passes `dates` to `RepricingPricer`.

`config.py`: accept `KnockOutResetSnowballOption` (a `SnowballOption`); for a REBASED KO-reset: `provider == "life_surface"` → `ValidationError("surface mode prices the post-close contract off one surface per bucket; a REBASED KO-reset needs a knocked-in surface per knock-in day: use provider='repricing' (spec 6a)")`; otherwise `require_rebased_provider(..., engines={"pricing": self.engine_config.pricing_engine_type}, consumer="EnsembleConfig", position=bp.position_id)`. Ledger: `Task 7: Ruling: no vol-model gate in EnsembleConfig — vol_source must be 'scalar' there, which forces vol_model='bsm' (ReplayBacktestConfig) — cost if wrong: none, a vol-model run cannot be configured`.

- [ ] **Step 6: Run to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/lifecycle_ko_reset/test_simulation_ko_reset.py -q`
Expected: 11 passed. A conformance mismatch is debugged column by column with `report.summary()`; the oracle is the authority.

- [ ] **Step 7: Run the simulation suite**

Run: `.venv/bin/python -m pytest -n 6 test/simulation test/lifecycle_ko_reset -q > "$WS/t7.log" 2>&1; tail -5 "$WS/t7.log"`
Expected: all pass; snowball simulation goldens unchanged.

- [ ] **Step 8: Commit**

```bash
git add quantark/asset/equity/lifecycle/ko_reset.py quantark/asset/equity/lifecycle/autocallable.py quantark/backtest/simulation/ test/lifecycle_ko_reset/test_simulation_ko_reset.py
git commit -m "feat(simulation): KO-reset snowballs on simulated paths (schedule switch, per-path REBASED placement)"
```

---

### Task 8: Goldens, docs, full suite

**Files:**
- Modify: `CHANGELOG.md` ([Unreleased])
- Modify: `quantark/asset/equity/CLAUDE.md` (lifecycle line) — untracked file, edit only
- Modify: `quantark/backtest/CLAUDE.md` if it lists the simulation's product restriction
- Test: full suite

- [ ] **Step 1: Golden diff**

Run: `.venv/bin/python -m pytest -n 6 test/test_replay_goldens.py test/simulation test/test_snowball_simulated_paths_study.py -q > "$WS/t8_goldens.log" 2>&1; tail -5 "$WS/t8_goldens.log"`
Expected: pass. Any row regenerated in Tasks 1-7 is listed in the ledger with its record; event rows unchanged.

- [ ] **Step 2: CHANGELOG**

Under `## [Unreleased]`, `### Added`: KO-reset lifecycle tracking (manager, replay, simulation; both schedule-2 modes; provider matrix). `### Changed`: snapped numeric observation times for every autocallable tracker (off-day numeric schedules price as observed; list the moved goldens); replay QUAD routes KO-reset to `KOResetSnowballQuadEngine`. `### Fixed`: MC REBASED accrued offset (6.5 → 7.5 control), V1 at each path's own maturity, annualized REBASED V1 tenor.

- [ ] **Step 3: Full suite under the RAM guard**

Run: `.venv/bin/python -m pytest -n 6 -q > "$WS/full.log" 2>&1; tail -15 "$WS/full.log"; git checkout -- $(git diff --name-only -- example/ | tr '\n' ' ')`
Expected: all pass except the 4 known main-checkout failures (`reference_main_preexisting_failures`); `git status` clean apart from the untracked study files listed at session start.

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): KO-reset lifecycle tracking and snapped observation times"
```
