# Simulated-Path Backtest — Plan 1 of 4: Market Paths, Generators, Carry and Chain

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data layer of the simulated-path backtest: the `MarketPath` batch type, the constant-maturity carry curve and its conversion to a listed IM chain and to the pricer's dividend input, the history builder, and the three path generators (block bootstrap, GBM, designed), each persisted and convertible to a replay-engine dataset.

**Architecture:** A new package `quantark/backtest/simulation/` whose only dependencies are numpy, pandas and existing QuantArk curve objects. Every day of every path is a full market snapshot (spot, ATM vol, rate, carry curve on a fixed tenor grid). The chain builder reads the curve at each listed contract's actual tenor, so a contract's basis converges to zero at expiry without being modelled. One shared function (`term_dividend_yield`, extracted from the replay engine) turns a chain into the dividend object, so a model name means the same thing on real and simulated paths.

**Tech Stack:** Python 3.10–3.13, numpy, pandas, pytest (`-n0` while iterating; the parallel default for the final run).

**Spec:** `docs/superpowers/specs/2026-09-08-simulated-path-backtest-design.md`, sections 4.1, 5, 6 and the persistence part of 9. Later plans: plan 2 (repricing pricer, cache, vectorised lifecycle/hedge/engine, exact conformance), plan 3 (ladder, disk cache, gate, PDE life surface), plan 4 (results, distributions, example study).

## Global Constraints

- Work on branch `feat/simulated-path-backtest` (already exists, holds the spec). If you work in a worktree, create it with the native `EnterWorktree` tool from that branch and verify `git log -1` shows `38fe719e` or later. Run tests as `/Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 <files>` (add `PYTHONPATH=$PWD` in a worktree so its source shadows the editable install).
- Commits: every message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add example/` wholesale (two `example/mo_volmodels/data/` sample files churn under test runs; `git checkout --` them before staging). Files under `docs/` need `git add -f`. `CLAUDE.md` files are never added.
- No hidden defaults for generator parameters (spec §5): block length, drift, vol, tenor grid, calendar and start state are explicit keyword-only arguments; a missing one is a `TypeError`, a bad one a `ValidationError`.
- Fail closed (spec §12): raise `quantark.util.exceptions.ValidationError`; never substitute a value.
- Use `quantark.util.numerical` helpers (`is_zero`, `is_close`) instead of raw float comparisons in library code.
- Byte-identical guarantee for the extraction in Task 5: `test/test_replay_dividend_source.py`, `test/test_snowball_q_term_structure_study.py` and `test/replay_golden` must stay green with no test edits.
- Style: PEP 8, dataclasses with type hints, docstrings on public APIs, `from __future__ import annotations` at the top of every new module.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `quantark/backtest/simulation/__init__.py` (create) | public exports | 1, 11 |
| `quantark/backtest/simulation/paths/__init__.py` (create) | subpackage exports | 1 |
| `quantark/backtest/simulation/paths/market_path.py` (create) | `MarketPath`, `StartState`, `trading_calendar`, validation, fingerprint, npz persistence | 1, 10 |
| `quantark/backtest/simulation/carry.py` (create) | `carry_at`, `curve_from_chain`, IM contract calendar, `DayChain`, `day_chain` | 2, 3, 4 |
| `quantark/backtest/replay/dividend_source.py` (create) | `term_dividend_yield` shared by engine, study and simulation | 5 |
| `quantark/backtest/replay/product_replay.py` (modify) | `_term_dividend` delegates to `term_dividend_yield` | 5 |
| `example/snowball_q_term_structure/_common.py` (modify) | `dividend_for` delegates to `term_dividend_yield` | 5 |
| `quantark/backtest/simulation/dividends.py` (create) | `CurveTailPillars`, `dividend_yield_for_day` | 5 |
| `quantark/backtest/simulation/paths/history.py` (create) | `PathHistory` | 6 |
| `quantark/backtest/simulation/paths/bootstrap.py` (create) | `StationaryBlockBootstrap` | 7 |
| `quantark/backtest/simulation/paths/gbm.py` (create) | `GBMPaths`, `ConstantVol`, `StickyRealisedVol` | 8 |
| `quantark/backtest/simulation/paths/designed.py` (create) | `market_path_from_day_path`, `SnowballStressLibrary`, `stress_set` | 9 |
| `quantark/backtest/simulation/dataset.py` (create) | `to_market_dataset` (one path → `AutocallableMarketDataSet`) | 10 |
| `quantark/backtest/simulation/README.md` (create) | module guide | 11 |
| `test/simulation/__init__.py`, `test/simulation/conftest.py` (create) | shared synthetic fixtures | 1 |
| `test/simulation/test_market_path.py` (create) | Task 1, 10 tests | 1, 10 |
| `test/simulation/test_carry.py` (create) | Tasks 2–4 tests | 2, 3, 4 |
| `test/simulation/test_dividends.py` (create) | Task 5 tests | 5 |
| `test/test_replay_dividend_source.py` (modify: add one test) | extraction pin | 5 |
| `test/simulation/test_history.py` (create) | Task 6 tests | 6 |
| `test/simulation/test_bootstrap.py` (create) | Task 7 tests | 7 |
| `test/simulation/test_gbm.py` (create) | Task 8 tests | 8 |
| `test/simulation/test_designed.py` (create) | Task 9 tests | 9 |
| `test/simulation/test_dataset.py` (create) | Task 10 tests | 10 |

Conventions used by every task:

- `n_paths`, `n_days`, `n_tenors` are the axes of `MarketPath` arrays in that order.
- Carry is cumulative log carry `B(T) = ln(F(T)/S)`; negative for an index futures discount.
- Tenors are ACT/365 year fractions from the day; contract expiry tenor `T = (expiry − day).days / 365`.
- `FUTURES_MULTIPLIER = 200.0`, contract codes `IM` + `YYMM`.

---

### Task 1: Package skeleton, `MarketPath`, `StartState`, trading calendar

**Files:**
- Create: `quantark/backtest/simulation/__init__.py`, `quantark/backtest/simulation/paths/__init__.py`, `quantark/backtest/simulation/paths/market_path.py`
- Create: `test/simulation/__init__.py`, `test/simulation/conftest.py`, `test/simulation/test_market_path.py`

**Interfaces:**
- Produces:
  - `MarketPath(dates: pd.DatetimeIndex, spot: np.ndarray, atm_vol: np.ndarray, rate: np.ndarray, carry: np.ndarray, tenor_grid: np.ndarray, meta: dict)` frozen dataclass; properties `n_paths`, `n_days`, `n_tenors`; `path(i) -> MarketPath`; `fingerprint() -> str` (sha256 hex).
  - `StartState(spot: float, atm_vol: float, rate: float, carry: np.ndarray)` frozen dataclass, `carry` shape `(n_tenors,)`.
  - `trading_calendar(start: date, n_days: int, *, holidays: Sequence[date] = ()) -> pd.DatetimeIndex` — `n_days` Mon–Fri days from `start` inclusive (if `start` is a trading day), skipping `holidays`.
  - `DEFAULT_TENOR_GRID = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5])`.

- [ ] **Step 1: Write the failing tests**

`test/simulation/__init__.py` is empty. `test/simulation/conftest.py`:

```python
"""Shared synthetic fixtures for the simulation tests."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.market_path import (
    DEFAULT_TENOR_GRID,
    MarketPath,
    StartState,
    trading_calendar,
)

SPOT = 6000.0
VOL = 0.22
RATE = 0.02


def flat_carry(tenor_grid: np.ndarray, annual_carry: float = -0.10) -> np.ndarray:
    """B(T) = annual_carry * T: a flat 10% discount curve."""
    return annual_carry * np.asarray(tenor_grid, dtype=float)


@pytest.fixture()
def start_state() -> StartState:
    return StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))


@pytest.fixture()
def calendar() -> pd.DatetimeIndex:
    return trading_calendar(date(2024, 1, 2), 30)


def make_market_path(n_paths: int = 3, n_days: int = 30, start: date = date(2024, 1, 2)) -> MarketPath:
    """Deterministic flat-vol, drifting-spot batch for structural tests."""
    dates = trading_calendar(start, n_days)
    days = np.arange(n_days, dtype=float)
    spot = SPOT * (1.0 + 0.001 * days)[None, :] * (1.0 + 0.01 * np.arange(n_paths))[:, None]
    atm_vol = np.full((n_paths, n_days), VOL)
    rate = np.full((n_paths, n_days), RATE)
    carry = np.broadcast_to(flat_carry(DEFAULT_TENOR_GRID), (n_paths, n_days, DEFAULT_TENOR_GRID.size)).copy()
    return MarketPath(dates=dates, spot=spot, atm_vol=atm_vol, rate=rate, carry=carry,
                      tenor_grid=DEFAULT_TENOR_GRID.copy(), meta={"generator": "fixture"})
```

`test/simulation/test_market_path.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.market_path import (
    DEFAULT_TENOR_GRID,
    MarketPath,
    StartState,
    trading_calendar,
)
from quantark.util.exceptions import ValidationError

from .conftest import SPOT, make_market_path


def test_trading_calendar_skips_weekends_and_holidays():
    cal = trading_calendar(date(2024, 1, 4), 5, holidays=[date(2024, 1, 8)])
    # Thu 4, Fri 5, (Sat, Sun), Mon 8 holiday, Tue 9, Wed 10, Thu 11
    assert list(cal.date) == [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11)]


def test_trading_calendar_starts_on_the_next_trading_day_when_start_is_a_weekend():
    cal = trading_calendar(date(2024, 1, 6), 2)  # Saturday
    assert list(cal.date) == [date(2024, 1, 8), date(2024, 1, 9)]


def test_market_path_shapes_and_properties():
    mp = make_market_path(n_paths=3, n_days=30)
    assert (mp.n_paths, mp.n_days, mp.n_tenors) == (3, 30, DEFAULT_TENOR_GRID.size)
    one = mp.path(1)
    assert one.n_paths == 1
    assert one.spot[0, 0] == pytest.approx(SPOT * 1.01)
    assert one.dates.equals(mp.dates)


@pytest.mark.parametrize(
    "field, bad",
    [
        ("spot", lambda a: -a),                       # non-positive spot
        ("atm_vol", lambda a: np.zeros_like(a)),      # non-positive vol
        ("spot", lambda a: np.where(np.arange(a.size).reshape(a.shape) == 0, np.nan, a)),  # NaN
    ],
)
def test_market_path_rejects_invalid_arrays(field, bad):
    mp = make_market_path()
    kwargs = {k: getattr(mp, k) for k in ("dates", "spot", "atm_vol", "rate", "carry", "tenor_grid", "meta")}
    kwargs[field] = bad(kwargs[field])
    with pytest.raises(ValidationError):
        MarketPath(**kwargs)


def test_market_path_rejects_shape_mismatch_and_bad_tenor_grid():
    mp = make_market_path()
    with pytest.raises(ValidationError):
        MarketPath(dates=mp.dates, spot=mp.spot[:, :-1], atm_vol=mp.atm_vol, rate=mp.rate,
                   carry=mp.carry, tenor_grid=mp.tenor_grid, meta={})
    with pytest.raises(ValidationError):
        MarketPath(dates=mp.dates, spot=mp.spot, atm_vol=mp.atm_vol, rate=mp.rate,
                   carry=mp.carry, tenor_grid=np.array([0.5, 0.25, 1.0]), meta={})
    with pytest.raises(ValidationError):
        MarketPath(dates=mp.dates[::-1], spot=mp.spot, atm_vol=mp.atm_vol, rate=mp.rate,
                   carry=mp.carry, tenor_grid=mp.tenor_grid, meta={})


def test_fingerprint_changes_with_data_and_not_with_meta():
    a = make_market_path()
    b = MarketPath(dates=a.dates, spot=a.spot, atm_vol=a.atm_vol, rate=a.rate, carry=a.carry,
                   tenor_grid=a.tenor_grid, meta={"generator": "other"})
    c = MarketPath(dates=a.dates, spot=a.spot * 1.0001, atm_vol=a.atm_vol, rate=a.rate, carry=a.carry,
                   tenor_grid=a.tenor_grid, meta=a.meta)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert len(a.fingerprint()) == 64


def test_start_state_validates():
    with pytest.raises(ValidationError):
        StartState(spot=0.0, atm_vol=0.2, rate=0.02, carry=np.zeros(3))
    with pytest.raises(ValidationError):
        StartState(spot=100.0, atm_vol=0.2, rate=0.02, carry=np.array([[0.0]]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/__init__.py`:

```python
"""Simulated-path backtest for autocallable books (spec 2026-09-08).

Plan 1 ships the data layer: ``MarketPath`` batches, carry-curve to
listed-chain conversion, the shared dividend rule, the history builder and
the three path generators.
"""
from __future__ import annotations

from .paths.market_path import DEFAULT_TENOR_GRID, MarketPath, StartState, trading_calendar

__all__ = ["DEFAULT_TENOR_GRID", "MarketPath", "StartState", "trading_calendar"]
```

`quantark/backtest/simulation/paths/__init__.py`:

```python
from __future__ import annotations

from .market_path import DEFAULT_TENOR_GRID, MarketPath, StartState, trading_calendar

__all__ = ["DEFAULT_TENOR_GRID", "MarketPath", "StartState", "trading_calendar"]
```

`quantark/backtest/simulation/paths/market_path.py`:

```python
"""``MarketPath``: a batch of simulated joint market paths (spec §4.1).

Every day of every path is a full market snapshot: spot, the scalar ATM vol
the pricer receives, a flat continuously-compounded rate, and the
constant-maturity carry curve ``B(T_k) = ln(F(T_k)/S)`` on a fixed tenor
grid measured from that day.  ``B(0) = 0`` is implicit.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

DEFAULT_TENOR_GRID = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5])


def trading_calendar(start: date, n_days: int, *, holidays: Sequence[date] = ()) -> pd.DatetimeIndex:
    """``n_days`` Monday–Friday days from ``start`` (inclusive when it trades), skipping ``holidays``."""
    if n_days < 1:
        raise ValidationError("n_days must be at least 1")
    closed = {pd.Timestamp(h).date() for h in holidays}
    out = []
    cur = pd.Timestamp(start).date()
    while len(out) < n_days:
        if cur.weekday() < 5 and cur not in closed:
            out.append(pd.Timestamp(cur))
        cur += timedelta(days=1)
    return pd.DatetimeIndex(out)


def _validate_tenor_grid(tenor_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(tenor_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValidationError("tenor_grid must be a non-empty 1-D array")
    if not np.all(np.isfinite(grid)) or grid[0] <= 0.0 or np.any(np.diff(grid) <= 0.0):
        raise ValidationError("tenor_grid must be positive, finite and strictly increasing")
    return grid


@dataclass(frozen=True)
class StartState:
    """The inception snapshot a generator integrates from."""

    spot: float
    atm_vol: float
    rate: float
    carry: np.ndarray  # (n_tenors,)

    def __post_init__(self) -> None:
        carry = np.asarray(self.carry, dtype=float)
        if carry.ndim != 1 or not np.all(np.isfinite(carry)):
            raise ValidationError("StartState.carry must be a finite 1-D array")
        object.__setattr__(self, "carry", carry)
        if not (self.spot > 0.0 and self.atm_vol > 0.0):
            raise ValidationError("StartState needs positive spot and atm_vol")
        if not np.isfinite(self.rate):
            raise ValidationError("StartState.rate must be finite")


@dataclass(frozen=True)
class MarketPath:
    """A batch of paths on one trading calendar (spec §4.1)."""

    dates: pd.DatetimeIndex
    spot: np.ndarray       # (n_paths, n_days)
    atm_vol: np.ndarray    # (n_paths, n_days)
    rate: np.ndarray       # (n_paths, n_days)
    carry: np.ndarray      # (n_paths, n_days, n_tenors)
    tenor_grid: np.ndarray # (n_tenors,)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dates = pd.DatetimeIndex(self.dates)
        if len(dates) == 0 or not dates.is_monotonic_increasing or not dates.is_unique:
            raise ValidationError("dates must be a non-empty, strictly increasing DatetimeIndex")
        grid = _validate_tenor_grid(self.tenor_grid)
        spot = np.asarray(self.spot, dtype=float)
        vol = np.asarray(self.atm_vol, dtype=float)
        rate = np.asarray(self.rate, dtype=float)
        carry = np.asarray(self.carry, dtype=float)
        n_days = len(dates)
        if spot.ndim != 2 or spot.shape[1] != n_days:
            raise ValidationError(f"spot must have shape (n_paths, {n_days}), got {spot.shape}")
        n_paths = spot.shape[0]
        for name, arr in (("atm_vol", vol), ("rate", rate)):
            if arr.shape != (n_paths, n_days):
                raise ValidationError(f"{name} must have shape {(n_paths, n_days)}, got {arr.shape}")
        if carry.shape != (n_paths, n_days, grid.size):
            raise ValidationError(f"carry must have shape {(n_paths, n_days, grid.size)}, got {carry.shape}")
        for name, arr in (("spot", spot), ("atm_vol", vol), ("rate", rate), ("carry", carry)):
            if not np.all(np.isfinite(arr)):
                raise ValidationError(f"{name} contains non-finite values")
        if np.any(spot <= 0.0) or np.any(vol <= 0.0):
            raise ValidationError("spot and atm_vol must be positive")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "atm_vol", vol)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "carry", carry)
        object.__setattr__(self, "tenor_grid", grid)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def n_paths(self) -> int:
        return int(self.spot.shape[0])

    @property
    def n_days(self) -> int:
        return int(self.spot.shape[1])

    @property
    def n_tenors(self) -> int:
        return int(self.tenor_grid.size)

    def path(self, i: int) -> "MarketPath":
        """The single-path batch for path ``i`` (meta carries ``path_index``)."""
        if not 0 <= i < self.n_paths:
            raise ValidationError(f"path index {i} out of range for {self.n_paths} paths")
        return MarketPath(
            dates=self.dates, spot=self.spot[i : i + 1], atm_vol=self.atm_vol[i : i + 1],
            rate=self.rate[i : i + 1], carry=self.carry[i : i + 1], tenor_grid=self.tenor_grid,
            meta={**self.meta, "path_index": int(i)},
        )

    def fingerprint(self) -> str:
        """sha256 of the arrays, dates and tenor grid; ``meta`` is excluded."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.dates.asi8).tobytes())
        for arr in (self.tenor_grid, self.spot, self.atm_vol, self.rate, self.carry):
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation test/simulation
git commit -m "feat(backtest/simulation): MarketPath batch type, StartState and trading calendar

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Carry-curve arithmetic — `carry_at` and `curve_from_chain`

**Files:**
- Create: `quantark/backtest/simulation/carry.py`
- Create: `test/simulation/test_carry.py`

**Interfaces:**
- Consumes: `ForwardCarryCurve` (`quantark.param.div.forward_carry_curve`): `ForwardCarryCurve(nodes: list[tuple[T, B]])`, `.carry(T) -> float` (piecewise-linear in `B`, `B(0)=0`, last slope continued).
- Produces:
  - `carry_at(carry: np.ndarray, tenor_grid: np.ndarray, tenors: np.ndarray) -> np.ndarray` — `carry` shape `(..., n_tenors)`, `tenors` shape `(m,)`, returns `(..., m)`: piecewise-linear `B` in `T` between grid nodes with `B(0)=0`, last segment's slope continued beyond the last node; `tenors == 0` gives `0`; negative tenors raise.
  - `curve_from_chain(spot: float, tenors: Sequence[float], prices: Sequence[float], tenor_grid: np.ndarray) -> np.ndarray` — the history-side inverse: `B_i = ln(F_i/spot)` at the listed tenors, sampled on the grid with the `ForwardCarryCurve` rules; contracts with `tenor <= 0` are dropped; fewer than one remaining raises.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_carry.py` (first block; Tasks 3 and 4 append to this file):

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.carry import carry_at, curve_from_chain
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID
from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

GRID = np.array([0.25, 0.5, 1.0])
B = np.array([-0.02, -0.045, -0.10])  # B(T) at the grid


def test_carry_at_matches_forward_carry_curve_inside_and_beyond_the_grid():
    ref = ForwardCarryCurve(list(zip(GRID, B)))
    tenors = np.array([0.1, 0.25, 0.375, 0.5, 0.8, 1.0, 1.5, 2.0])
    out = carry_at(B, GRID, tenors)
    assert out.shape == (8,)
    for t, b in zip(tenors, out):
        assert b == pytest.approx(ref.carry(float(t)), abs=1e-15)


def test_carry_at_is_zero_at_zero_tenor_and_rejects_negative():
    assert carry_at(B, GRID, np.array([0.0]))[0] == 0.0
    with pytest.raises(ValidationError):
        carry_at(B, GRID, np.array([-0.01]))


def test_carry_at_broadcasts_over_paths_and_days():
    carry = np.stack([B, 2.0 * B])[:, None, :].repeat(4, axis=1)  # (2 paths, 4 days, 3 tenors)
    out = carry_at(carry, GRID, np.array([0.375, 1.5]))
    assert out.shape == (2, 4, 2)
    assert out[1, 3, 0] == pytest.approx(2.0 * (-0.02 - 0.045) / 2.0)
    assert out[0, 0, 1] == pytest.approx(-0.10 + (-0.10 + 0.045) / 0.5 * 0.5)


def test_curve_from_chain_round_trips_a_curve_and_drops_expired_contracts():
    spot = 6000.0
    listed_tenors = [0.0, 0.05, 0.3, 0.55, 0.9]         # first one expires today
    prices = [spot * np.exp(carry_at(B, GRID, np.array([t]))[0]) for t in listed_tenors]
    grid_b = curve_from_chain(spot, listed_tenors, prices, DEFAULT_TENOR_GRID)
    assert grid_b.shape == DEFAULT_TENOR_GRID.shape
    ref = ForwardCarryCurve([(t, np.log(p / spot)) for t, p in zip(listed_tenors[1:], prices[1:])])
    for t, b in zip(DEFAULT_TENOR_GRID, grid_b):
        assert b == pytest.approx(ref.carry(float(t)), abs=1e-15)
    with pytest.raises(ValidationError):
        curve_from_chain(spot, [0.0], [spot], DEFAULT_TENOR_GRID)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_carry.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.carry'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/carry.py`:

```python
"""Constant-maturity carry curves and the listed IM chain (spec §4.1, §6).

``B(T) = ln(F(T)/S)`` on a fixed tenor grid, piecewise-linear in ``T``
with ``B(0) = 0`` and the last segment's slope continued beyond the last
node -- the ``ForwardCarryCurve`` rules, vectorised over paths and days.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

from .paths.market_path import MarketPath


def carry_at(carry: np.ndarray, tenor_grid: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """Read ``B`` at ``tenors`` from curves given on ``tenor_grid``.

    ``carry`` has shape ``(..., n_tenors)``; the result has shape
    ``(..., len(tenors))``.  Interpolation is linear in ``B`` against ``T``
    (constant forward carry between nodes); beyond the last node the last
    segment continues; ``B(0) = 0``.
    """
    grid = np.concatenate([[0.0], np.asarray(tenor_grid, dtype=float)])
    b = np.asarray(carry, dtype=float)
    b = np.concatenate([np.zeros(b.shape[:-1] + (1,)), b], axis=-1)
    t = np.asarray(tenors, dtype=float)
    if np.any(t < 0.0):
        raise ValidationError("tenors must be non-negative")
    idx = np.searchsorted(grid, t, side="right") - 1
    idx = np.clip(idx, 0, grid.size - 2)
    t0, t1 = grid[idx], grid[idx + 1]
    w = (t - t0) / (t1 - t0)                       # w > 1 beyond the last node: slope continues
    b0, b1 = b[..., idx], b[..., idx + 1]
    return b0 + w * (b1 - b0)


def curve_from_chain(
    spot: float, tenors: Sequence[float], prices: Sequence[float], tenor_grid: np.ndarray
) -> np.ndarray:
    """The constant-maturity curve implied by a listed chain (history side).

    Contracts with a non-positive tenor (expiring today or earlier) are
    dropped; the remaining ``(T_i, ln(F_i/spot))`` nodes form a
    ``ForwardCarryCurve`` sampled at ``tenor_grid``.
    """
    if spot <= 0.0:
        raise ValidationError("spot must be positive")
    nodes = [
        (float(t), float(np.log(float(p) / spot)))
        for t, p in zip(tenors, prices)
        if float(t) > 0.0
    ]
    if not nodes:
        raise ValidationError("curve_from_chain needs at least one contract with a positive tenor")
    nodes.sort()
    curve = ForwardCarryCurve(nodes)
    return np.array([curve.carry(float(t)) for t in np.asarray(tenor_grid, dtype=float)])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_carry.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/carry.py test/simulation/test_carry.py
git commit -m "feat(backtest/simulation): constant-maturity carry read-out and chain inverse

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: CFFEX IM contract calendar

**Files:**
- Modify: `quantark/backtest/simulation/carry.py` (append)
- Modify: `test/simulation/test_carry.py` (append)

**Interfaces:**
- Produces:
  - `third_friday(year: int, month: int) -> date`.
  - `im_expiry(year: int, month: int, calendar: pd.DatetimeIndex) -> pd.Timestamp` — the third Friday, moved to the next day in `calendar` if the Friday is not in it (raises if the calendar ends before that).
  - `listed_im_contracts(day: pd.Timestamp, calendar: pd.DatetimeIndex) -> list[tuple[str, pd.Timestamp]]` — four `(code, expiry)` pairs: current month (its expiry `>= day`), the following month, then the next two March/June/September/December months after the second contract's month; codes `IM` + `YYMM`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_carry.py`:

```python
from quantark.backtest.simulation.carry import im_expiry, listed_im_contracts, third_friday
from quantark.backtest.simulation.paths.market_path import trading_calendar


def test_third_friday():
    assert third_friday(2024, 1) == date(2024, 1, 19)
    assert third_friday(2024, 2) == date(2024, 2, 16)
    assert third_friday(2024, 3) == date(2024, 3, 15)
    assert third_friday(2024, 6) == date(2024, 6, 21)
    assert third_friday(2024, 9) == date(2024, 9, 20)


def test_im_expiry_rolls_a_holiday_friday_to_the_next_trading_day():
    cal = trading_calendar(date(2024, 1, 2), 300, holidays=[date(2024, 2, 16)])
    assert im_expiry(2024, 1, cal) == pd.Timestamp("2024-01-19")
    assert im_expiry(2024, 2, cal) == pd.Timestamp("2024-02-19")  # Monday after the holiday Friday


def test_listed_im_contracts_follow_the_cffex_cycle():
    cal = trading_calendar(date(2024, 1, 2), 400)
    listed = listed_im_contracts(pd.Timestamp("2024-01-02"), cal)
    assert [c for c, _ in listed] == ["IM2401", "IM2402", "IM2403", "IM2406"]
    assert [e.date() for _, e in listed] == [date(2024, 1, 19), date(2024, 2, 16), date(2024, 3, 15), date(2024, 6, 21)]
    # the expiring contract is still listed on its expiry day ...
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-01-19"), cal)][0] == "IM2401"
    # ... and gone the next trading day, when a new quarterly is listed
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-01-22"), cal)] == ["IM2402", "IM2403", "IM2406", "IM2409"]
    # after the March expiry the two nearest months are April and May
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-03-18"), cal)] == ["IM2404", "IM2405", "IM2406", "IM2409"]
    # December wraps the year
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-11-11"), cal)] == ["IM2411", "IM2412", "IM2503", "IM2506"]
    # after the November expiry (15th) the December contract is nearest and January follows
    assert [c for c, _ in listed_im_contracts(pd.Timestamp("2024-11-18"), cal)] == ["IM2412", "IM2501", "IM2503", "IM2506"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_carry.py -q -k "friday or expiry or listed"`
Expected: `ImportError: cannot import name 'im_expiry'`

- [ ] **Step 3: Write the implementation**

Append to `quantark/backtest/simulation/carry.py` (the import block from Task 2 already has everything this needs):

```python
_QUARTERLY = (3, 6, 9, 12)


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7      # Friday is weekday 4
    return first + timedelta(days=offset + 14)


def im_expiry(year: int, month: int, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """Third Friday of the month, or the next trading day in ``calendar`` if it does not trade."""
    friday = pd.Timestamp(third_friday(year, month))
    if friday in calendar:
        return friday
    later = calendar[calendar > friday]
    if len(later) == 0:
        raise ValidationError(f"calendar ends before the IM{year % 100:02d}{month:02d} expiry {friday.date()}")
    return pd.Timestamp(later[0])


def _next_month(year: int, month: int) -> Tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def listed_im_contracts(day: pd.Timestamp, calendar: pd.DatetimeIndex) -> List[Tuple[str, pd.Timestamp]]:
    """The four IM contracts CFFEX lists on ``day``: current month, next month, next two quarterlies."""
    day = pd.Timestamp(day).normalize()
    year, month = day.year, day.month
    if im_expiry(year, month, calendar) < day:       # this month's contract has expired
        year, month = _next_month(year, month)
    months = [(year, month)]
    months.append(_next_month(year, month))
    y, m = months[-1]
    while len(months) < 4:
        y, m = _next_month(y, m)
        if m in _QUARTERLY:
            months.append((y, m))
    return [(f"IM{y % 100:02d}{m:02d}", im_expiry(y, m, calendar)) for y, m in months]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_carry.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/carry.py test/simulation/test_carry.py
git commit -m "feat(backtest/simulation): CFFEX IM contract calendar (third Friday, quarterly cycle)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `DayChain` — the listed chain priced off each path's curve

**Files:**
- Modify: `quantark/backtest/simulation/carry.py` (append)
- Modify: `test/simulation/test_carry.py` (append)

**Interfaces:**
- Consumes: `MarketPath` (Task 1), `carry_at`, `listed_im_contracts` (Tasks 2–3).
- Produces:
  - `FUTURES_MULTIPLIER = 200.0`.
  - `DayChain(date: pd.Timestamp, contracts: tuple[str, ...], expiries: tuple[pd.Timestamp, ...], tenors: np.ndarray, prices: np.ndarray, multiplier: float)` frozen dataclass; `prices` shape `(n_paths, n_contracts)`; `frame(path_index: int) -> pd.DataFrame` with the replay futures columns `date, contract, futures_price, expiry_date, multiplier`.
  - `day_chain(path: MarketPath, day_index: int, *, multiplier: float = FUTURES_MULTIPLIER) -> DayChain`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_carry.py`:

```python
from quantark.backtest.simulation.carry import FUTURES_MULTIPLIER, DayChain, day_chain
from .conftest import SPOT, make_market_path


def test_day_chain_prices_each_path_off_its_own_curve():
    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    chain = day_chain(mp, 0)
    assert isinstance(chain, DayChain)
    assert chain.contracts == ("IM2401", "IM2402", "IM2403", "IM2406")
    assert chain.prices.shape == (2, 4)
    assert chain.multiplier == FUTURES_MULTIPLIER
    for i in range(2):
        for j, t in enumerate(chain.tenors):
            expected = mp.spot[i, 0] * np.exp(carry_at(mp.carry[i, 0], mp.tenor_grid, np.array([t]))[0])
            assert chain.prices[i, j] == pytest.approx(expected, rel=1e-14)
    assert chain.tenors[0] == pytest.approx((date(2024, 1, 19) - date(2024, 1, 2)).days / 365.0)


def test_day_chain_frame_has_the_replay_columns():
    mp = make_market_path(n_paths=2, n_days=30)
    frame = day_chain(mp, 3).frame(1)
    assert list(frame.columns) == ["date", "contract", "futures_price", "expiry_date", "multiplier"]
    assert len(frame) == 4
    assert (frame["date"] == mp.dates[3]).all()
    assert frame["futures_price"].iloc[2] == pytest.approx(day_chain(mp, 3).prices[1, 2])


def test_expiring_contract_prices_at_spot_on_its_expiry_day():
    mp = make_market_path(n_paths=1, n_days=30, start=date(2024, 1, 2))
    day = list(mp.dates).index(pd.Timestamp("2024-01-19"))
    chain = day_chain(mp, day)
    assert chain.contracts[0] == "IM2401"
    assert chain.tenors[0] == 0.0
    assert chain.prices[0, 0] == pytest.approx(mp.spot[0, day])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_carry.py -q -k day_chain`
Expected: `ImportError: cannot import name 'DayChain'`

- [ ] **Step 3: Write the implementation**

Append to `quantark/backtest/simulation/carry.py`:

```python
FUTURES_MULTIPLIER = 200.0


def _extended_calendar(dates: pd.DatetimeIndex, *, years_beyond: float = 1.5) -> pd.DatetimeIndex:
    """``dates`` followed by plain weekdays for ``years_beyond`` years.

    ``listed_im_contracts`` needs the calendar to reach the far contract's
    expiry, up to nine months past the day.  Holidays after the path's own
    calendar are unknown and treated as trading days; that only moves the
    expiry date of contracts that expire after the path ends, which never
    trade in the run.
    """
    last = pd.Timestamp(dates[-1])
    tail = pd.bdate_range(last + pd.Timedelta(days=1), last + pd.Timedelta(days=int(365 * years_beyond)))
    return dates.append(tail)


@dataclass(frozen=True)
class DayChain:
    """The listed IM chain on one day, priced for every path (spec §6)."""

    date: pd.Timestamp
    contracts: Tuple[str, ...]
    expiries: Tuple[pd.Timestamp, ...]
    tenors: np.ndarray      # (n_contracts,) ACT/365 from ``date``
    prices: np.ndarray      # (n_paths, n_contracts)
    multiplier: float

    def frame(self, path_index: int) -> pd.DataFrame:
        """One path's chain in the replay engine's futures-frame layout."""
        return pd.DataFrame(
            {
                "date": [self.date] * len(self.contracts),
                "contract": list(self.contracts),
                "futures_price": self.prices[path_index].astype(float).tolist(),
                "expiry_date": list(self.expiries),
                "multiplier": [float(self.multiplier)] * len(self.contracts),
            }
        )


def day_chain(path: MarketPath, day_index: int, *, multiplier: float = FUTURES_MULTIPLIER) -> DayChain:
    """Price the four listed contracts on ``path.dates[day_index]`` for every path.

    ``F_i = S * exp(B(T_i))`` with ``T_i`` the contract's remaining tenor read
    off each path's constant-maturity curve; a contract expiring today has
    ``T = 0`` and prices at spot.
    """
    if not 0 <= day_index < path.n_days:
        raise ValidationError(f"day_index {day_index} out of range for {path.n_days} days")
    if multiplier <= 0.0:
        raise ValidationError("multiplier must be positive")
    day = pd.Timestamp(path.dates[day_index])
    listed = listed_im_contracts(day, _extended_calendar(path.dates))
    contracts = tuple(code for code, _ in listed)
    expiries = tuple(exp for _, exp in listed)
    tenors = np.array([(exp - day).days / 365.0 for exp in expiries])
    b = carry_at(path.carry[:, day_index, :], path.tenor_grid, tenors)   # (n_paths, n_contracts)
    prices = path.spot[:, day_index][:, None] * np.exp(b)
    return DayChain(date=day, contracts=contracts, expiries=expiries, tenors=tenors,
                    prices=prices, multiplier=float(multiplier))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_carry.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/carry.py test/simulation/test_carry.py
git commit -m "feat(backtest/simulation): DayChain prices the listed IM chain off each path's carry curve

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: One dividend rule for engine, study and simulation

**Files:**
- Create: `quantark/backtest/replay/dividend_source.py`
- Modify: `quantark/backtest/replay/product_replay.py` (`_term_dividend`, lines ~221–290)
- Modify: `example/snowball_q_term_structure/_common.py` (`dividend_for`, the `futures_curve` branch)
- Create: `quantark/backtest/simulation/dividends.py`
- Modify: `test/test_replay_dividend_source.py` (append one test)
- Create: `test/simulation/test_dividends.py`

**Interfaces:**
- Consumes: `IndexFuturesQuote`, `IndexFuturesCurve` (`quantark.asset.equity.market`), `ForwardCarryCurve`, `ContinuousDividendYield` (`quantark.param.div`), `surface_tail_carry_yield` (`quantark.backtest.replay.product_replay`), `derive_implied_dividend_yield` and `SignedDividendYield` (`quantark.backtest.replay.market`), `DayChain` (Task 4), `AutocallableEngineConfig` (`quantark.backtest.replay`).
- Produces:
  - `term_dividend_yield(quotes: Sequence[IndexFuturesQuote], *, spot: float, rate_curve, extrapolation: str, underlying: str = "index", artifact=None)` in `quantark.backtest.replay.dividend_source` — exactly the branches of today's `_term_dividend` after the quotes are built: `surface_forward_carry` (needs `artifact`, else `ValidationError`), one-node limit (`ContinuousDividendYield`), `flat_forward_carry`, `flat_q`; unknown extrapolation raises.
  - `CurveTailPillars(tenors: np.ndarray, carry: np.ndarray)` in `quantark.backtest.simulation.dividends` with `implied_q_pillars(rate: float) -> (times, yields)`; it plays the artifact's role for `surface_forward_carry` on a simulated path (spec §6).
  - `dividend_yield_for_day(chain: DayChain, path_index: int, *, spot: float, rate: float, engine_config: AutocallableEngineConfig, active_contract: str, curve_tenors: np.ndarray, curve_carry: np.ndarray) -> DividendYield` — the dividend object the pricer receives under `engine_config.dividend_source` for that path and day.
  - `legacy_implied_q(spot: float, futures_price: float, tenor: float, rate: float) -> float` — the floored, simple-compounded active-contract yield (`derive_implied_dividend_yield`'s `implied_q`).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_replay_dividend_source.py`:

```python
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.asset.equity.market import IndexFuturesQuote
from quantark.param import FlatRateCurve


class TestSharedDividendRule:
    def test_term_dividend_yield_equals_the_engine_for_every_extrapolation(self, history_dir):
        dataset = _market_data(surface_history=VolSurfaceHistory(history_dir))
        d = DATES[0]
        quotes = [
            IndexFuturesQuote(contract=c, maturity=_ttm(e, d), price=SPOT * math.exp((RATE - q) * _ttm(e, d)),
                              multiplier=200.0, expiry_date=e.to_pydatetime())
            for c, e, q in CHAIN
        ]
        artifact = VolSurfaceHistory(history_dir).surface_for(d)
        for extrapolation in ("flat_q", "flat_forward_carry", "surface_forward_carry"):
            replay = _replay(dataset, AutocallableEngineConfig(
                dividend_source="futures_curve", futures_curve_extrapolation=extrapolation,
                futures_curve_min_tenor_days=1))
            env, *_ = _build_env(replay, dataset, d)
            shared = term_dividend_yield(quotes, spot=SPOT, rate_curve=FlatRateCurve(rate=RATE),
                                         extrapolation=extrapolation, underlying="CSI1000", artifact=artifact)
            for t in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
                assert shared.get_yield(t) == pytest.approx(env.div_yield.get_yield(t), abs=1e-15)

    def test_surface_forward_carry_without_an_artifact_fails_closed(self):
        quotes = [IndexFuturesQuote(contract="IM2403", maturity=0.2, price=SPOT * 0.99, multiplier=200.0)]
        with pytest.raises(ValidationError):
            term_dividend_yield(quotes, spot=SPOT, rate_curve=FlatRateCurve(rate=RATE),
                                extrapolation="surface_forward_carry")
        with pytest.raises(ValidationError):
            term_dividend_yield(quotes, spot=SPOT, rate_curve=FlatRateCurve(rate=RATE), extrapolation="cubic")
```

`test/simulation/test_dividends.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay import AutocallableEngineConfig
from quantark.backtest.replay.market import derive_implied_dividend_yield
from quantark.backtest.simulation.carry import carry_at, day_chain
from quantark.backtest.simulation.dividends import CurveTailPillars, dividend_yield_for_day, legacy_implied_q
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, make_market_path


def _chain_and_path():
    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    return mp, day_chain(mp, 0)


def test_legacy_implied_q_is_the_engine_channel():
    _, implied = derive_implied_dividend_yield(rate=RATE, spot=SPOT, futures_price=SPOT * 0.995, time_to_maturity=0.05)
    assert legacy_implied_q(SPOT, SPOT * 0.995, 0.05, RATE) == pytest.approx(implied)


def test_flat_active_source_returns_the_active_contract_yield():
    mp, chain = _chain_and_path()
    cfg = AutocallableEngineConfig()  # dividend_source None = legacy
    div = dividend_yield_for_day(chain, 1, spot=mp.spot[1, 0], rate=RATE, engine_config=cfg,
                                 active_contract="IM2402", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[1, 0])
    j = chain.contracts.index("IM2402")
    expected = legacy_implied_q(mp.spot[1, 0], chain.prices[1, j], chain.tenors[j], RATE)
    assert div.get_yield(0.5) == pytest.approx(expected)
    assert div.get_yield(2.0) == pytest.approx(expected)


def test_term_sources_reprice_the_listed_contracts_beyond_the_minimum_tenor():
    mp, chain = _chain_and_path()
    for extrapolation in ("flat_q", "flat_forward_carry", "surface_forward_carry"):
        cfg = AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_extrapolation=extrapolation,
                                       futures_curve_min_tenor_days=7)
        div = dividend_yield_for_day(chain, 0, spot=mp.spot[0, 0], rate=RATE, engine_config=cfg,
                                     active_contract="IM2401", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[0, 0])
        for j, t in enumerate(chain.tenors):
            if t * 365.0 >= 7:
                model_forward = mp.spot[0, 0] * np.exp((RATE - div.get_yield(t)) * t)
                assert model_forward == pytest.approx(chain.prices[0, j], rel=1e-12)


def test_surface_forward_carry_tail_follows_the_simulated_curve():
    mp, chain = _chain_and_path()
    cfg = AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_extrapolation="surface_forward_carry",
                                   futures_curve_min_tenor_days=1)
    div = dividend_yield_for_day(chain, 0, spot=mp.spot[0, 0], rate=RATE, engine_config=cfg,
                                 active_contract="IM2401", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[0, 0])
    # the fixture curve is flat -10% carry: beyond the last contract the yield is r + 10%
    t_last = float(chain.tenors[-1])
    for t in (1.0, 1.5, 2.5):
        b_model = (RATE - div.get_yield(t)) * t
        b_curve = carry_at(mp.carry[0, 0], mp.tenor_grid, np.array([t]))[0]
        assert b_model == pytest.approx(b_curve, abs=1e-12)
    assert t_last < 1.0


def test_curve_tail_pillars_expose_only_tenors_beyond_the_last_contract():
    grid = np.array([0.25, 0.5, 1.0, 1.5])
    pillars = CurveTailPillars(tenors=grid, carry=-0.1 * grid)
    times, yields = pillars.implied_q_pillars(rate=RATE)
    assert times == [0.25, 0.5, 1.0, 1.5]
    assert yields == pytest.approx([RATE + 0.1] * 4)


def test_no_eligible_contract_fails_closed():
    mp, chain = _chain_and_path()
    cfg = AutocallableEngineConfig(dividend_source="futures_curve", futures_curve_min_tenor_days=400)
    with pytest.raises(ValidationError):
        dividend_yield_for_day(chain, 0, spot=mp.spot[0, 0], rate=RATE, engine_config=cfg,
                               active_contract="IM2401", curve_tenors=mp.tenor_grid, curve_carry=mp.carry[0, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/test_replay_dividend_source.py -k Shared test/simulation/test_dividends.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.replay.dividend_source'`

- [ ] **Step 3: Write the shared rule and delegate to it**

`quantark/backtest/replay/dividend_source.py`:

```python
"""The term-structure dividend rule shared by the replay engine, the
q term-structure study and the simulated-path backtest.

One function, so a ``dividend_source`` / ``futures_curve_extrapolation``
pair means the same object wherever a chain is turned into the pricer's
dividend input.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

from quantark.asset.equity.market import IndexFuturesCurve, IndexFuturesQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.div.forward_carry_curve import ForwardCarryCurve
from quantark.util.exceptions import ValidationError

EXTRAPOLATIONS = ("flat_q", "flat_forward_carry", "surface_forward_carry")


def term_dividend_yield(
    quotes: Sequence[IndexFuturesQuote],
    *,
    spot: float,
    rate_curve: Any,
    extrapolation: str,
    underlying: str = "index",
    artifact: Optional[Any] = None,
):
    """Dividend object for the ``futures_curve`` source from eligible quotes.

    ``surface_forward_carry``: the chain in log-forward space continued past
    its last node with ``artifact.implied_q_pillars``' forward carry.  One
    quote is the flat one-node limit of the other two conventions;
    ``flat_forward_carry`` continues the last segment's forward carry;
    ``flat_q`` holds the endpoint zero yield.  No quotes, an unknown
    convention or a missing artifact fail closed.
    """
    if extrapolation not in EXTRAPOLATIONS:
        raise ValidationError(
            f"futures_curve_extrapolation must be one of {EXTRAPOLATIONS}, got {extrapolation!r}"
        )
    if not quotes:
        raise ValidationError("term_dividend_yield needs at least one eligible futures quote")
    if extrapolation == "surface_forward_carry":
        if artifact is None:
            raise ValidationError("'surface_forward_carry' requires an artifact with implied_q_pillars")
        from quantark.backtest.replay.product_replay import surface_tail_carry_yield

        return surface_tail_carry_yield(
            spot=float(spot),
            forward_nodes=[(q.maturity, q.price) for q in quotes],
            artifact=artifact,
            rate_curve=rate_curve,
        )
    if len(quotes) == 1:
        q = quotes[0]
        return ContinuousDividendYield(
            float(rate_curve.get_rate(q.maturity)) - math.log(q.price / float(spot)) / q.maturity
        )
    curve = IndexFuturesCurve(underlying=underlying or "index", spot=float(spot), quotes=list(quotes))
    if extrapolation == "flat_forward_carry":
        return ForwardCarryCurve.from_index_futures(curve, rate_curve).to_dividend_yield(rate_curve)
    return curve.to_dividend_yield_curve(rate_curve)
```

In `quantark/backtest/replay/product_replay.py`, replace the body of `_term_dividend` from `if not quotes:` to the end of the `futures_curve` branch with:

```python
            if not quotes:
                raise ValidationError(
                    f"dividend_source='futures_curve' found no contract with at "
                    f"least {min_days} days to expiry on {date.date()}"
                )
            extrapolation = getattr(
                self.engine_config, "futures_curve_extrapolation", "flat_q"
            )
            artifact = (
                self._surface_artifact(date, "surface_forward_carry")
                if extrapolation == "surface_forward_carry"
                else None
            )
            return term_dividend_yield(
                quotes,
                spot=float(market["spot"]),
                rate_curve=rate_curve,
                extrapolation=extrapolation,
                underlying=self.underlying or "index",
                artifact=artifact,
            )
```

and add `from .dividend_source import term_dividend_yield` to its imports (the `dividend_source` module imports `surface_tail_carry_yield` lazily inside the function to avoid a cycle). Remove the now-unused `IndexFuturesCurve` import from `product_replay.py` if nothing else uses it (keep `IndexFuturesQuote`).

In `example/snowball_q_term_structure/_common.py`, replace the `futures_curve` branch of `dividend_for` (from `if model.extrapolation == "surface_forward_carry":` to `return curve.to_dividend_yield_curve(rate_curve)`) with:

```python
        return term_dividend_yield(
            quotes,
            spot=float(spot),
            rate_curve=rate_curve,
            extrapolation=model.extrapolation,
            underlying=UNDERLYING_NAME,
            artifact=artifact,
        )
```

importing `from quantark.backtest.replay.dividend_source import term_dividend_yield`; drop the imports that become unused (`surface_tail_carry_yield`, `ForwardCarryCurve`, `ContinuousDividendYield`, `futures_curve` if only used there — check with `grep -n` before removing). Note the shared function raises for a missing artifact, matching the study's previous message intent.

`quantark/backtest/simulation/dividends.py`:

```python
"""The pricer's dividend input on a simulated path (spec §6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from quantark.asset.equity.market import IndexFuturesQuote
from quantark.backtest.replay.config import AutocallableEngineConfig
from quantark.backtest.replay.dividend_source import term_dividend_yield
from quantark.backtest.replay.market import SignedDividendYield, derive_implied_dividend_yield
from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError

from .carry import DayChain


def legacy_implied_q(spot: float, futures_price: float, tenor: float, rate: float) -> float:
    """The engine's historical active-contract channel: simple compounding, floored at zero."""
    _, implied_q = derive_implied_dividend_yield(
        rate=float(rate), spot=float(spot), futures_price=float(futures_price), time_to_maturity=float(tenor)
    )
    return float(implied_q)


@dataclass(frozen=True)
class CurveTailPillars:
    """A simulated carry curve standing in for the IV artifact's parity forwards.

    ``implied_q_pillars(rate)`` returns ``q_k = rate - B(T_k)/T_k`` at the
    curve's tenors, which is all ``surface_tail_carry_yield`` reads; the
    level cancels there and only the forward carry beyond the last listed
    contract survives (spec §6).
    """

    tenors: np.ndarray
    carry: np.ndarray

    def implied_q_pillars(self, rate: float) -> Tuple[List[float], List[float]]:
        t = np.asarray(self.tenors, dtype=float)
        b = np.asarray(self.carry, dtype=float)
        return [float(x) for x in t], [float(rate) - float(bk) / float(tk) for tk, bk in zip(t, b)]


def dividend_yield_for_day(
    chain: DayChain,
    path_index: int,
    *,
    spot: float,
    rate: float,
    engine_config: AutocallableEngineConfig,
    active_contract: str,
    curve_tenors: np.ndarray,
    curve_carry: np.ndarray,
):
    """The dividend object the pricer receives under ``engine_config.dividend_source``.

    ``None`` / ``"active_contract"``: the floored simple-compounded yield of
    ``active_contract`` (``SignedDividendYield``).  ``"futures_curve"``:
    every listed contract with at least ``futures_curve_min_tenor_days`` to
    expiry through the shared ``term_dividend_yield``; for
    ``surface_forward_carry`` the path's own curve supplies the tail.
    ``"surface_forwards"`` has no meaning on a simulated path and raises.
    """
    source = engine_config.dividend_source
    prices = chain.prices[path_index]
    if source in (None, "active_contract"):
        if active_contract not in chain.contracts:
            raise ValidationError(f"active contract {active_contract!r} is not listed on {chain.date.date()}")
        j = chain.contracts.index(active_contract)
        return SignedDividendYield(legacy_implied_q(spot, prices[j], chain.tenors[j], rate))
    if source == "futures_curve":
        min_days = int(engine_config.futures_curve_min_tenor_days)
        quotes = [
            IndexFuturesQuote(
                contract=c, maturity=float(t), price=float(p), multiplier=float(chain.multiplier),
                expiry_date=e.to_pydatetime(),
            )
            for c, e, t, p in zip(chain.contracts, chain.expiries, chain.tenors, prices)
            if round(float(t) * 365.0) >= min_days
        ]
        if not quotes:
            raise ValidationError(
                f"dividend_source='futures_curve' found no contract with at least "
                f"{min_days} days to expiry on {chain.date.date()}"
            )
        return term_dividend_yield(
            quotes,
            spot=float(spot),
            rate_curve=FlatRateCurve(rate=float(rate)),
            extrapolation=engine_config.futures_curve_extrapolation,
            underlying="index",
            artifact=CurveTailPillars(tenors=curve_tenors, carry=curve_carry),
        )
    raise ValidationError(f"dividend_source {source!r} is not available on a simulated path")
```

- [ ] **Step 4: Run the tests and the byte-identical regression**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_dividends.py test/test_replay_dividend_source.py test/test_snowball_q_term_structure_study.py test/replay_golden -q`
Expected: all pass, no test edits outside the two new blocks.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/replay/dividend_source.py quantark/backtest/replay/product_replay.py \
        example/snowball_q_term_structure/_common.py quantark/backtest/simulation/dividends.py \
        test/test_replay_dividend_source.py test/simulation/test_dividends.py
git commit -m "refactor(backtest/replay): one term-dividend rule shared by engine, study and simulation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `PathHistory` — the realised history as a daily state series

**Files:**
- Create: `quantark/backtest/simulation/paths/history.py`
- Create: `test/simulation/test_history.py`

**Interfaces:**
- Consumes: `curve_from_chain` (Task 2), `StartState` (Task 1).
- Produces:
  - `PathHistory(dates: pd.DatetimeIndex, spot: np.ndarray, atm_vol: np.ndarray, rate: np.ndarray, carry: np.ndarray, tenor_grid: np.ndarray, source_fingerprint: str)` frozen dataclass, arrays `(n_days,)` and `(n_days, n_tenors)`.
  - `PathHistory.from_frames(*, spot: pd.DataFrame, vol: pd.DataFrame, futures: pd.DataFrame, rate: float | pd.DataFrame, tenor_grid: np.ndarray) -> PathHistory` — frames as in the study (`spot`: `date, spot`; `vol`: `date, volatility`; `futures`: `date, contract, futures_price, expiry_date, multiplier`; `rate` a float or a `date, rate` frame). Days present in all inputs are kept (intersection); a day whose chain has no contract with a positive tenor raises.
  - `changes() -> np.ndarray` shape `(n_days − 1, 3 + n_tenors)`: columns `[Δln S, Δvol, Δrate, ΔB_1 … ΔB_K]`.
  - `levels() -> np.ndarray` shape `(n_days, 3 + n_tenors)`: `[ln S, vol, rate, B_1 … B_K]`.
  - `snapshot(day_index: int = -1) -> StartState`.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_history.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.carry import carry_at
from quantark.backtest.simulation.paths.history import PathHistory
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.util.exceptions import ValidationError

RATE = 0.02


def _frames(n_days: int = 10):
    dates = trading_calendar(date(2024, 1, 2), n_days)
    spot = pd.DataFrame({"date": dates, "spot": 6000.0 + 10.0 * np.arange(n_days)})
    vol = pd.DataFrame({"date": dates, "volatility": 0.20 + 0.001 * np.arange(n_days)})
    rows = []
    chain = [("IM2401", pd.Timestamp("2024-01-19")), ("IM2402", pd.Timestamp("2024-02-16")),
             ("IM2403", pd.Timestamp("2024-03-15")), ("IM2406", pd.Timestamp("2024-06-21"))]
    for k, d in enumerate(dates):
        s = float(spot["spot"].iloc[k])
        for c, e in chain:
            t = (e - d).days / 365.0
            rows.append({"date": d, "contract": c, "futures_price": s * np.exp(-0.10 * t),
                         "expiry_date": e, "multiplier": 200.0})
    return spot, vol, pd.DataFrame(rows)


def test_from_frames_builds_levels_changes_and_snapshot():
    spot, vol, futures = _frames()
    h = PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
    assert h.spot.shape == (10,) and h.carry.shape == (10, DEFAULT_TENOR_GRID.size)
    assert h.levels().shape == (10, 3 + DEFAULT_TENOR_GRID.size)
    ch = h.changes()
    assert ch.shape == (9, 3 + DEFAULT_TENOR_GRID.size)
    assert ch[0, 0] == pytest.approx(np.log(6010.0 / 6000.0))
    assert ch[0, 1] == pytest.approx(0.001)
    assert ch[:, 2] == pytest.approx(0.0)
    # a flat -10% carry chain gives a flat -10% * T curve on every day
    for k in range(10):
        assert h.carry[k] == pytest.approx(-0.10 * DEFAULT_TENOR_GRID, abs=1e-12)
    snap = h.snapshot()
    assert isinstance(snap, StartState)
    assert snap.spot == pytest.approx(6090.0) and snap.atm_vol == pytest.approx(0.209)
    assert len(h.source_fingerprint) == 64


def test_from_frames_intersects_dates_and_accepts_a_rate_frame():
    spot, vol, futures = _frames()
    vol = vol.iloc[2:]                       # vol history starts later
    rate = pd.DataFrame({"date": spot["date"], "rate": [RATE] * len(spot)})
    h = PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=rate, tenor_grid=DEFAULT_TENOR_GRID)
    assert h.dates[0] == vol["date"].iloc[0] and len(h.dates) == 8


def test_from_frames_rejects_a_day_without_a_live_contract():
    spot, vol, futures = _frames()
    day3 = spot["date"].iloc[3]
    futures.loc[futures["date"] == day3, "expiry_date"] = day3 - pd.Timedelta(days=1)  # all expired that day
    with pytest.raises(ValidationError):
        PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_history.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.paths.history'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/paths/history.py`:

```python
"""``PathHistory``: the realised joint market history as a daily state series (spec §5.1)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from ..carry import curve_from_chain
from .market_path import StartState, _validate_tenor_grid


@dataclass(frozen=True)
class PathHistory:
    dates: pd.DatetimeIndex
    spot: np.ndarray      # (n_days,)
    atm_vol: np.ndarray   # (n_days,)
    rate: np.ndarray      # (n_days,)
    carry: np.ndarray     # (n_days, n_tenors)
    tenor_grid: np.ndarray
    source_fingerprint: str

    @property
    def n_days(self) -> int:
        return int(self.spot.size)

    @classmethod
    def from_frames(
        cls,
        *,
        spot: pd.DataFrame,
        vol: pd.DataFrame,
        futures: pd.DataFrame,
        rate: Union[float, pd.DataFrame],
        tenor_grid: np.ndarray,
    ) -> "PathHistory":
        grid = _validate_tenor_grid(tenor_grid)
        s = spot[["date", "spot"]].assign(date=pd.to_datetime(spot["date"]).dt.normalize()).set_index("date")["spot"]
        v = vol[["date", "volatility"]].assign(date=pd.to_datetime(vol["date"]).dt.normalize()).set_index("date")["volatility"]
        if isinstance(rate, pd.DataFrame):
            r = rate[["date", "rate"]].assign(date=pd.to_datetime(rate["date"]).dt.normalize()).set_index("date")["rate"]
        else:
            r = pd.Series(float(rate), index=s.index)
        fut = futures.assign(date=pd.to_datetime(futures["date"]).dt.normalize(),
                             expiry_date=pd.to_datetime(futures["expiry_date"]).dt.normalize())
        common = s.index.intersection(v.index).intersection(r.index).intersection(pd.DatetimeIndex(fut["date"].unique()))
        common = common.sort_values()
        if len(common) < 2:
            raise ValidationError("PathHistory needs at least two common days across spot, vol, rate and futures")
        carry = np.empty((len(common), grid.size))
        by_day = {d: g for d, g in fut.groupby("date")}
        for k, d in enumerate(common):
            g = by_day[d]
            tenors = ((g["expiry_date"] - d).dt.days / 365.0).to_numpy()
            try:
                carry[k] = curve_from_chain(float(s[d]), tenors, g["futures_price"].to_numpy(), grid)
            except ValidationError as exc:
                raise ValidationError(f"{d.date()}: {exc}") from exc
        h = hashlib.sha256()
        for arr in (common.asi8, s[common].to_numpy(float), v[common].to_numpy(float), r[common].to_numpy(float), carry):
            h.update(np.ascontiguousarray(arr).tobytes())
        return cls(
            dates=pd.DatetimeIndex(common), spot=s[common].to_numpy(float), atm_vol=v[common].to_numpy(float),
            rate=r[common].to_numpy(float), carry=carry, tenor_grid=grid, source_fingerprint=h.hexdigest(),
        )

    def levels(self) -> np.ndarray:
        return np.column_stack([np.log(self.spot), self.atm_vol, self.rate, self.carry])

    def changes(self) -> np.ndarray:
        return np.diff(self.levels(), axis=0)

    def snapshot(self, day_index: int = -1) -> StartState:
        return StartState(spot=float(self.spot[day_index]), atm_vol=float(self.atm_vol[day_index]),
                          rate=float(self.rate[day_index]), carry=self.carry[day_index].copy())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_history.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/paths/history.py test/simulation/test_history.py
git commit -m "feat(backtest/simulation): PathHistory turns the realised history into a daily joint state series

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Stationary block bootstrap generator

**Files:**
- Create: `quantark/backtest/simulation/paths/bootstrap.py`
- Create: `test/simulation/test_bootstrap.py`

**Interfaces:**
- Consumes: `PathHistory` (Task 6), `MarketPath`, `StartState` (Task 1).
- Produces: `StationaryBlockBootstrap(history: PathHistory, *, mean_block_days: int, demean_returns: bool, annual_drift: float, vol_floor: float, carry_mode: Literal["changes", "levels"], start: StartState, calendar: pd.DatetimeIndex)` with `generate(n_paths: int, n_days: int, *, seed: int) -> MarketPath`. `meta` carries `generator="stationary_block_bootstrap"`, every parameter, `seed`, `history_fingerprint`, `vol_floor_hits` (count of floored path-days).
  - `stationary_block_indices(rng: np.random.Generator, n_source: int, n_paths: int, n_days: int, mean_block_days: int) -> np.ndarray` (int array `(n_paths, n_days)` into the source rows, circular wrap).

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_bootstrap.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.bootstrap import StationaryBlockBootstrap, stationary_block_indices
from quantark.backtest.simulation.paths.history import PathHistory
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.util.exceptions import ValidationError

from .test_history import _frames, RATE


@pytest.fixture()
def history() -> PathHistory:
    spot, vol, futures = _frames(n_days=60)
    rng = np.random.default_rng(0)
    spot["spot"] = 6000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(spot))))
    vol["volatility"] = 0.2 + np.cumsum(rng.normal(0.0, 0.002, len(vol)))
    return PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)


def _gen(history, **overrides):
    kwargs = dict(mean_block_days=5, demean_returns=False, annual_drift=0.0, vol_floor=0.05,
                  carry_mode="changes", start=history.snapshot(), calendar=trading_calendar(date(2025, 1, 2), 40))
    kwargs.update(overrides)
    return StationaryBlockBootstrap(history, **kwargs)


def test_block_indices_are_circular_runs_with_the_requested_mean_length():
    rng = np.random.default_rng(1)
    idx = stationary_block_indices(rng, n_source=50, n_paths=200, n_days=100, mean_block_days=10)
    assert idx.shape == (200, 100) and idx.min() >= 0 and idx.max() < 50
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 50).mean()
    assert continues == pytest.approx(0.9, abs=0.02)     # p(new block) = 1/10
    idx1 = stationary_block_indices(rng, n_source=50, n_paths=200, n_days=100, mean_block_days=1)
    assert (idx1[:, 1:] == (idx1[:, :-1] + 1) % 50).mean() == pytest.approx(0.02, abs=0.02)  # iid apart from chance


def test_generate_is_seed_reproducible_and_records_meta(history):
    gen = _gen(history)
    a = gen.generate(4, 40, seed=7)
    b = gen.generate(4, 40, seed=7)
    c = gen.generate(4, 40, seed=8)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    assert a.meta["generator"] == "stationary_block_bootstrap" and a.meta["seed"] == 7
    assert a.meta["mean_block_days"] == 5 and a.meta["history_fingerprint"] == history.source_fingerprint
    assert a.n_paths == 4 and a.n_days == 40 and a.dates.equals(trading_calendar(date(2025, 1, 2), 40))


def test_paths_start_from_the_start_state_and_integrate_resampled_changes(history):
    gen = _gen(history, mean_block_days=10_000)      # one block: the history replayed in order from a random start
    mp = gen.generate(1, 20, seed=3)
    changes = history.changes()
    # find the start row from the first spot change
    first = np.log(mp.spot[0, 1] / mp.spot[0, 0])
    k = int(np.argmin(np.abs(changes[:, 0] - first)))
    assert mp.spot[0, 0] == pytest.approx(history.snapshot().spot)
    for j in range(1, 20):
        row = changes[(k + j - 1) % len(changes)]
        assert np.log(mp.spot[0, j] / mp.spot[0, j - 1]) == pytest.approx(row[0], abs=1e-12)
        assert mp.atm_vol[0, j] - mp.atm_vol[0, j - 1] == pytest.approx(row[1], abs=1e-12)
        assert mp.carry[0, j] - mp.carry[0, j - 1] == pytest.approx(row[3:], abs=1e-12)


def test_demean_and_drift_set_the_mean_log_return(history):
    gen = _gen(history, demean_returns=True, annual_drift=0.0504, mean_block_days=1)
    mp = gen.generate(400, 60, seed=11)
    mean_daily = np.log(mp.spot[:, -1] / mp.spot[:, 0]).mean() / 59
    assert mean_daily == pytest.approx(0.0504 / 252, abs=4e-4)


def test_vol_floor_binds_and_is_counted(history):
    gen = _gen(history, vol_floor=0.30)                # above the whole history: floor binds everywhere
    mp = gen.generate(3, 10, seed=5)
    assert np.all(mp.atm_vol >= 0.30)
    assert mp.meta["vol_floor_hits"] > 0


def test_levels_mode_resamples_carry_levels(history):
    gen = _gen(history, carry_mode="levels", mean_block_days=1)
    mp = gen.generate(2, 15, seed=2)
    rows = {tuple(np.round(r, 12)) for r in history.carry}
    for i in range(2):
        for j in range(1, 15):
            assert tuple(np.round(mp.carry[i, j], 12)) in rows


def test_parameters_are_validated(history):
    with pytest.raises(ValidationError):
        _gen(history, mean_block_days=0)
    with pytest.raises(ValidationError):
        _gen(history, carry_mode="random")
    with pytest.raises(ValidationError):
        _gen(history).generate(0, 10, seed=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_bootstrap.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.paths.bootstrap'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/paths/bootstrap.py`:

```python
"""Politis–Romano stationary block bootstrap of the joint daily state (spec §5.1)."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from .history import PathHistory
from .market_path import MarketPath, StartState

TRADING_DAYS_PER_YEAR = 252.0


def stationary_block_indices(
    rng: np.random.Generator, n_source: int, n_paths: int, n_days: int, mean_block_days: int
) -> np.ndarray:
    """Row indices into the source: geometric blocks (mean ``mean_block_days``), circular."""
    if n_source < 1 or n_paths < 1 or n_days < 1 or mean_block_days < 1:
        raise ValidationError("n_source, n_paths, n_days and mean_block_days must be positive")
    p_new = 1.0 / float(mean_block_days)
    starts = rng.integers(0, n_source, size=(n_paths, n_days))
    new_block = rng.random((n_paths, n_days)) < p_new
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for d in range(1, n_days):
        idx[:, d] = np.where(new_block[:, d], starts[:, d], (idx[:, d - 1] + 1) % n_source)
    return idx


class StationaryBlockBootstrap:
    def __init__(
        self,
        history: PathHistory,
        *,
        mean_block_days: int,
        demean_returns: bool,
        annual_drift: float,
        vol_floor: float,
        carry_mode: Literal["changes", "levels"],
        start: StartState,
        calendar: pd.DatetimeIndex,
    ) -> None:
        if int(mean_block_days) < 1:
            raise ValidationError("mean_block_days must be at least 1")
        if carry_mode not in ("changes", "levels"):
            raise ValidationError("carry_mode must be 'changes' or 'levels'")
        if not np.isfinite(annual_drift) or not np.isfinite(vol_floor) or vol_floor <= 0.0:
            raise ValidationError("annual_drift must be finite and vol_floor positive")
        if start.carry.shape != history.tenor_grid.shape:
            raise ValidationError("start.carry must be on the history's tenor grid")
        self.history = history
        self.mean_block_days = int(mean_block_days)
        self.demean_returns = bool(demean_returns)
        self.annual_drift = float(annual_drift)
        self.vol_floor = float(vol_floor)
        self.carry_mode = carry_mode
        self.start = start
        self.calendar = pd.DatetimeIndex(calendar)

    def generate(self, n_paths: int, n_days: int, *, seed: int) -> MarketPath:
        if n_paths < 1 or n_days < 1:
            raise ValidationError("n_paths and n_days must be positive")
        if n_days > len(self.calendar):
            raise ValidationError(f"calendar has {len(self.calendar)} days, {n_days} requested")
        rng = np.random.default_rng(int(seed))
        changes = self.history.changes()                     # (m, 3 + K)
        returns = changes[:, 0]
        if self.demean_returns:
            returns = returns - returns.mean()
        returns = returns + self.annual_drift / TRADING_DAYS_PER_YEAR
        n_tenors = self.history.tenor_grid.size
        idx = stationary_block_indices(rng, changes.shape[0], n_paths, n_days - 1, self.mean_block_days) \
            if n_days > 1 else np.empty((n_paths, 0), dtype=np.int64)

        log_spot = np.empty((n_paths, n_days)); log_spot[:, 0] = np.log(self.start.spot)
        vol = np.empty((n_paths, n_days)); vol[:, 0] = self.start.atm_vol
        rate = np.empty((n_paths, n_days)); rate[:, 0] = self.start.rate
        carry = np.empty((n_paths, n_days, n_tenors)); carry[:, 0, :] = self.start.carry
        floor_hits = 0
        for d in range(1, n_days):
            rows = idx[:, d - 1]
            log_spot[:, d] = log_spot[:, d - 1] + returns[rows]
            raw_vol = vol[:, d - 1] + changes[rows, 1]
            floor_hits += int(np.count_nonzero(raw_vol < self.vol_floor))
            vol[:, d] = np.maximum(raw_vol, self.vol_floor)
            rate[:, d] = rate[:, d - 1] + changes[rows, 2]
            if self.carry_mode == "changes":
                carry[:, d, :] = carry[:, d - 1, :] + changes[rows, 3:]
            else:
                carry[:, d, :] = self.history.carry[rows + 1]   # the level of the resampled day
        meta = {
            "generator": "stationary_block_bootstrap", "seed": int(seed),
            "mean_block_days": self.mean_block_days, "demean_returns": self.demean_returns,
            "annual_drift": self.annual_drift, "vol_floor": self.vol_floor, "carry_mode": self.carry_mode,
            "history_fingerprint": self.history.source_fingerprint, "vol_floor_hits": floor_hits,
            "start": {"spot": self.start.spot, "atm_vol": self.start.atm_vol, "rate": self.start.rate},
        }
        return MarketPath(dates=self.calendar[:n_days], spot=np.exp(log_spot), atm_vol=vol, rate=rate,
                          carry=carry, tenor_grid=self.history.tenor_grid.copy(), meta=meta)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_bootstrap.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/paths/bootstrap.py test/simulation/test_bootstrap.py
git commit -m "feat(backtest/simulation): stationary block bootstrap of the joint spot/vol/carry history

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: GBM generator with vol rules

**Files:**
- Create: `quantark/backtest/simulation/paths/gbm.py`
- Create: `test/simulation/test_gbm.py`

**Interfaces:**
- Consumes: `MarketPath`, `StartState` (Task 1).
- Produces:
  - `ConstantVol(level: float)` and `StickyRealisedVol(a: float, b: float, window: int)` frozen dataclasses with `apply(log_returns: np.ndarray, start_vol: float) -> np.ndarray` (`log_returns` shape `(n_paths, n_days − 1)`; returns `(n_paths, n_days)`; the sticky rule uses the trailing realised vol over `window` returns annualised by `sqrt(252)`, with the start vol until `window` returns exist).
  - `GBMPaths(*, start: StartState, calendar: pd.DatetimeIndex, mu: float, sigma: float, vol_rule: ConstantVol | StickyRealisedVol, carry_schedule: np.ndarray | None, rate: float, tenor_grid: np.ndarray)` with `generate(n_paths, n_days, *, seed) -> MarketPath`; `carry_schedule` is `(n_days, n_tenors)` or `None` (hold `start.carry`).

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_gbm.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths, StickyRealisedVol
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.util.exceptions import ValidationError

from .conftest import flat_carry


def _start():
    return StartState(spot=6000.0, atm_vol=0.22, rate=0.02, carry=flat_carry(DEFAULT_TENOR_GRID))


def _gen(**overrides):
    kwargs = dict(start=_start(), calendar=trading_calendar(date(2025, 1, 2), 260), mu=0.05, sigma=0.25,
                  vol_rule=ConstantVol(0.22), carry_schedule=None, rate=0.02, tenor_grid=DEFAULT_TENOR_GRID)
    kwargs.update(overrides)
    return GBMPaths(**kwargs)


def test_log_return_moments_match_mu_and_sigma():
    mp = _gen().generate(4000, 253, seed=1)
    r = np.log(mp.spot[:, 1:] / mp.spot[:, :-1])
    assert r.mean() * 252 == pytest.approx(0.05 - 0.5 * 0.25**2, abs=0.01)
    assert r.std() * np.sqrt(252) == pytest.approx(0.25, abs=0.01)
    assert mp.spot[:, 0] == pytest.approx(6000.0)
    assert mp.meta["generator"] == "gbm" and mp.meta["seed"] == 1


def test_constant_vol_and_held_carry():
    mp = _gen().generate(3, 20, seed=2)
    assert np.all(mp.atm_vol == 0.22)
    assert mp.carry[1, 7] == pytest.approx(flat_carry(DEFAULT_TENOR_GRID))
    assert np.all(mp.rate == 0.02)


def test_sticky_realised_vol_rule():
    rule = StickyRealisedVol(a=0.05, b=1.0, window=5)
    r = np.full((1, 9), 0.01)
    out = rule.apply(r, start_vol=0.3)
    assert out.shape == (1, 10)
    assert np.all(out[0, :5] == 0.3)                       # not enough returns yet
    assert out[0, 5] == pytest.approx(0.05 + 1.0 * 0.0)     # identical returns: realised vol 0
    mp = _gen(vol_rule=rule).generate(2, 30, seed=3)
    assert np.all(mp.atm_vol > 0.0)


def test_carry_schedule_is_used_when_given():
    schedule = np.outer(np.linspace(-0.05, -0.15, 20), DEFAULT_TENOR_GRID)
    mp = _gen(carry_schedule=schedule).generate(2, 20, seed=4)
    assert mp.carry[0] == pytest.approx(schedule) and mp.carry[1] == pytest.approx(schedule)
    with pytest.raises(ValidationError):
        _gen(carry_schedule=schedule).generate(2, 25, seed=4)


def test_parameters_are_validated():
    with pytest.raises(ValidationError):
        _gen(sigma=-0.1)
    with pytest.raises(ValidationError):
        ConstantVol(0.0)
    with pytest.raises(ValidationError):
        StickyRealisedVol(a=0.05, b=1.0, window=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_gbm.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.paths.gbm'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/paths/gbm.py`:

```python
"""GBM spot with a real-world drift and a vol rule (spec §5.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from .market_path import MarketPath, StartState

TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class ConstantVol:
    level: float

    def __post_init__(self) -> None:
        if not self.level > 0.0:
            raise ValidationError("ConstantVol.level must be positive")

    def apply(self, log_returns: np.ndarray, start_vol: float) -> np.ndarray:
        n_paths, n_ret = log_returns.shape
        return np.full((n_paths, n_ret + 1), self.level)


@dataclass(frozen=True)
class StickyRealisedVol:
    """``atm_vol_d = a + b * realised_vol(window)`` once ``window`` returns exist, ``start_vol`` before."""

    a: float
    b: float
    window: int

    def __post_init__(self) -> None:
        if int(self.window) < 1:
            raise ValidationError("StickyRealisedVol.window must be at least 1")
        if not (np.isfinite(self.a) and np.isfinite(self.b)):
            raise ValidationError("StickyRealisedVol coefficients must be finite")

    def apply(self, log_returns: np.ndarray, start_vol: float) -> np.ndarray:
        n_paths, n_ret = log_returns.shape
        out = np.full((n_paths, n_ret + 1), float(start_vol))
        w = int(self.window)
        for d in range(w, n_ret + 1):
            realised = log_returns[:, d - w : d].std(axis=1, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
            out[:, d] = self.a + self.b * realised
        if np.any(out <= 0.0):
            raise ValidationError("StickyRealisedVol produced a non-positive vol; raise a or b")
        return out


class GBMPaths:
    def __init__(
        self,
        *,
        start: StartState,
        calendar: pd.DatetimeIndex,
        mu: float,
        sigma: float,
        vol_rule: Union[ConstantVol, StickyRealisedVol],
        carry_schedule: Optional[np.ndarray],
        rate: float,
        tenor_grid: np.ndarray,
    ) -> None:
        if not np.isfinite(mu) or not sigma > 0.0 or not np.isfinite(rate):
            raise ValidationError("mu and rate must be finite and sigma positive")
        grid = _validate_tenor_grid(tenor_grid)
        if grid.size != start.carry.size:
            raise ValidationError("start.carry must be on tenor_grid")
        self.tenor_grid = grid
        if carry_schedule is not None:
            carry_schedule = np.asarray(carry_schedule, dtype=float)
            if carry_schedule.ndim != 2 or carry_schedule.shape[1] != start.carry.size:
                raise ValidationError("carry_schedule must have shape (n_days, n_tenors)")
        self.start, self.calendar = start, pd.DatetimeIndex(calendar)
        self.mu, self.sigma, self.vol_rule, self.rate = float(mu), float(sigma), vol_rule, float(rate)
        self.carry_schedule = carry_schedule

    def generate(self, n_paths: int, n_days: int, *, seed: int) -> MarketPath:
        if n_paths < 1 or n_days < 1:
            raise ValidationError("n_paths and n_days must be positive")
        if n_days > len(self.calendar):
            raise ValidationError(f"calendar has {len(self.calendar)} days, {n_days} requested")
        if self.carry_schedule is not None and self.carry_schedule.shape[0] < n_days:
            raise ValidationError("carry_schedule is shorter than n_days")
        rng = np.random.default_rng(int(seed))
        dt = 1.0 / TRADING_DAYS_PER_YEAR
        z = rng.standard_normal((n_paths, n_days - 1))
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
        log_spot = np.log(self.start.spot) + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1)
        vol = self.vol_rule.apply(log_returns, self.start.atm_vol)
        n_tenors = self.start.carry.size
        if self.carry_schedule is None:
            carry = np.broadcast_to(self.start.carry, (n_paths, n_days, n_tenors)).copy()
        else:
            carry = np.broadcast_to(self.carry_schedule[:n_days], (n_paths, n_days, n_tenors)).copy()
        meta = {"generator": "gbm", "seed": int(seed), "mu": self.mu, "sigma": self.sigma,
                "vol_rule": {"type": type(self.vol_rule).__name__, **self.vol_rule.__dict__},
                "carry_schedule": self.carry_schedule is not None, "rate": self.rate,
                "start": {"spot": self.start.spot, "atm_vol": self.start.atm_vol, "rate": self.start.rate}}
        return MarketPath(dates=self.calendar[:n_days], spot=np.exp(log_spot), atm_vol=vol,
                          rate=np.full((n_paths, n_days), self.rate), carry=carry,
                          tenor_grid=self.tenor_grid.copy(), meta=meta)
```

(`from .market_path import MarketPath, StartState, _validate_tenor_grid` at the top of the module.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_gbm.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/paths/gbm.py test/simulation/test_gbm.py
git commit -m "feat(backtest/simulation): GBM path generator with constant and sticky-realised vol rules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Designed paths and the snowball stress library

**Files:**
- Create: `quantark/backtest/simulation/paths/designed.py`
- Create: `test/simulation/test_designed.py`

**Interfaces:**
- Consumes: `DayPath`, `DayStep`, `ParameterChange` (`quantark.dynamicscenario.path.day_path`; `ParameterChange.apply(current) -> float` applies `StressType` PERCENTAGE/ABSOLUTE/VALUE), `StressType` (`quantark.stresstest.stress.stress_types`), `MarketPath`, `StartState` (Task 1).
- Produces:
  - `market_path_from_day_path(day_path: DayPath, *, start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray) -> MarketPath` — one path; day `d` of the path is the state after applying `DayStep d`'s changes to the previous day's state (day 0 applies to the start state). Parameters: `spot`, `volatility`, `rate` via `ParameterChange.apply`; `basis` shifts the annualised carry: ABSOLUTE `v` → `B_k += v · T_k`, VALUE `v` → `B_k = v · T_k`, PERCENTAGE → `ValidationError`. Unknown parameters raise.
  - `SnowballStressLibrary` with static methods returning `DayPath`: `crash_into_ki(depth: float, days: int, total_days: int)`, `v_shape(depth: float, days_down: int, days_up: int, total_days: int)`, `vol_spike(vol_up: float, decay_days: int, total_days: int)`, `basis_blowout(carry_shift: float, days: int, total_days: int)`, `grind_up_to_ko(pct_per_day: float, days: int, total_days: int)`; every step after the move holds the state (a `DayStep` with no changes).
  - `stress_set(paths: Sequence[DayPath], *, start, calendar, tenor_grid) -> MarketPath` — stacks equal-length designed paths into one batch (`meta["scenario_names"]`); unequal lengths raise.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_designed.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.simulation.paths.designed import SnowballStressLibrary, market_path_from_day_path, stress_set
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.exceptions import ValidationError

from .conftest import flat_carry

START = StartState(spot=6000.0, atm_vol=0.22, rate=0.02, carry=flat_carry(DEFAULT_TENOR_GRID))
CAL = trading_calendar(date(2025, 1, 2), 60)


def test_day_path_changes_are_applied_cumulatively():
    dp = DayPath(name="t", steps=[
        DayStep(0, [ParameterChange("spot", StressType.PERCENTAGE, -0.10)]),
        DayStep(1, [ParameterChange("volatility", StressType.ABSOLUTE, 0.05), ParameterChange("basis", StressType.ABSOLUTE, -0.02)]),
        DayStep(2, [ParameterChange("rate", StressType.VALUE, 0.03)]),
        DayStep(3, []),
    ])
    mp = market_path_from_day_path(dp, start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert mp.n_paths == 1 and mp.n_days == 4
    assert mp.spot[0] == pytest.approx([5400.0, 5400.0, 5400.0, 5400.0])
    assert mp.atm_vol[0] == pytest.approx([0.22, 0.27, 0.27, 0.27])
    assert mp.rate[0] == pytest.approx([0.02, 0.02, 0.03, 0.03])
    assert mp.carry[0, 0] == pytest.approx(START.carry)
    assert mp.carry[0, 1] == pytest.approx(START.carry - 0.02 * DEFAULT_TENOR_GRID)
    assert mp.meta["generator"] == "designed" and mp.meta["scenario_name"] == "t"


def test_unknown_parameter_and_percentage_basis_are_rejected():
    bad = DayPath(name="x", steps=[DayStep(0, [ParameterChange("dividend", StressType.ABSOLUTE, 0.01)])])
    with pytest.raises(ValidationError):
        market_path_from_day_path(bad, start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    bad = DayPath(name="y", steps=[DayStep(0, [ParameterChange("basis", StressType.PERCENTAGE, 0.1)])])
    with pytest.raises(ValidationError):
        market_path_from_day_path(bad, start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)


def test_stress_library_shapes():
    crash = market_path_from_day_path(SnowballStressLibrary.crash_into_ki(depth=0.30, days=10, total_days=40),
                                      start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert crash.n_days == 40
    assert crash.spot[0, 9] == pytest.approx(6000.0 * 0.70, rel=1e-9)
    assert crash.spot[0, 39] == pytest.approx(6000.0 * 0.70, rel=1e-9)
    v = market_path_from_day_path(SnowballStressLibrary.v_shape(depth=0.30, days_down=10, days_up=10, total_days=40),
                                  start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert v.spot[0, 9] == pytest.approx(6000.0 * 0.70, rel=1e-9)
    assert v.spot[0, 19] == pytest.approx(6000.0, rel=1e-9)
    spike = market_path_from_day_path(SnowballStressLibrary.vol_spike(vol_up=0.15, decay_days=5, total_days=20),
                                      start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert spike.atm_vol[0, 0] == pytest.approx(0.37) and spike.atm_vol[0, 5] == pytest.approx(0.22, abs=1e-9)
    blow = market_path_from_day_path(SnowballStressLibrary.basis_blowout(carry_shift=-0.10, days=5, total_days=20),
                                     start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert blow.carry[0, 4] == pytest.approx(START.carry - 0.10 * DEFAULT_TENOR_GRID)
    grind = market_path_from_day_path(SnowballStressLibrary.grind_up_to_ko(pct_per_day=0.005, days=10, total_days=20),
                                      start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert grind.spot[0, 9] == pytest.approx(6000.0 * 1.005**10)


def test_stress_set_stacks_equal_length_paths():
    a = SnowballStressLibrary.crash_into_ki(depth=0.3, days=10, total_days=30)
    b = SnowballStressLibrary.vol_spike(vol_up=0.1, decay_days=5, total_days=30)
    mp = stress_set([a, b], start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
    assert mp.n_paths == 2 and mp.n_days == 30
    assert mp.meta["scenario_names"] == [a.name, b.name]
    with pytest.raises(ValidationError):
        stress_set([a, SnowballStressLibrary.vol_spike(vol_up=0.1, decay_days=5, total_days=31)],
                   start=START, calendar=CAL, tenor_grid=DEFAULT_TENOR_GRID)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_designed.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.paths.designed'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/paths/designed.py`:

```python
"""Designed paths from the dynamic-scenario ``PathBuilder`` (spec §5.3)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.exceptions import ValidationError

from .market_path import MarketPath, StartState, _validate_tenor_grid

_PARAMS = ("spot", "volatility", "rate", "basis")


def market_path_from_day_path(
    day_path: DayPath, *, start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray
) -> MarketPath:
    """One path: day ``d`` is the state after ``DayStep d``'s changes (day 0 applies to ``start``)."""
    grid = _validate_tenor_grid(tenor_grid)
    if start.carry.size != grid.size:
        raise ValidationError("start.carry must be on tenor_grid")
    steps = sorted(day_path.steps, key=lambda s: s.day_index)
    n_days = len(steps)
    if n_days < 1:
        raise ValidationError("a designed path needs at least one DayStep")
    if n_days > len(calendar):
        raise ValidationError(f"calendar has {len(calendar)} days, the path has {n_days}")
    spot, vol, rate = float(start.spot), float(start.atm_vol), float(start.rate)
    carry = start.carry.astype(float).copy()
    out_spot, out_vol, out_rate, out_carry = [], [], [], []
    for step in steps:
        for change in step.changes:
            if change.parameter not in _PARAMS:
                raise ValidationError(f"designed paths support {_PARAMS}, got {change.parameter!r}")
            if change.parameter == "spot":
                spot = float(change.apply(spot))
            elif change.parameter == "volatility":
                vol = float(change.apply(vol))
            elif change.parameter == "rate":
                rate = float(change.apply(rate))
            else:  # basis: a change in the annualised carry
                if change.stress_type == StressType.ABSOLUTE:
                    carry = carry + float(change.stress_value) * grid
                elif change.stress_type == StressType.VALUE:
                    carry = float(change.stress_value) * grid
                else:
                    raise ValidationError("basis changes must be ABSOLUTE or VALUE (annualised carry)")
        out_spot.append(spot); out_vol.append(vol); out_rate.append(rate); out_carry.append(carry.copy())
    meta = {"generator": "designed", "scenario_name": day_path.name, "description": day_path.description,
            "start": {"spot": start.spot, "atm_vol": start.atm_vol, "rate": start.rate}}
    return MarketPath(dates=pd.DatetimeIndex(calendar)[:n_days], spot=np.array([out_spot]),
                      atm_vol=np.array([out_vol]), rate=np.array([out_rate]), carry=np.array([out_carry]),
                      tenor_grid=grid.copy(), meta=meta)


def _hold(n: int, start_index: int) -> List[DayStep]:
    return [DayStep(i, []) for i in range(start_index, n)]


class SnowballStressLibrary:
    """Named adverse paths for a snowball; each returns a ``DayPath`` of ``total_days`` steps."""

    @staticmethod
    def crash_into_ki(depth: float, days: int, total_days: int) -> DayPath:
        _check(days, total_days)
        daily = (1.0 - depth) ** (1.0 / days) - 1.0
        steps = [DayStep(i, [ParameterChange("spot", StressType.PERCENTAGE, daily)]) for i in range(days)]
        return DayPath(name=f"crash_into_ki_{depth:.0%}_{days}d", steps=steps + _hold(total_days, days),
                       description=f"spot falls {depth:.0%} over {days} days, then holds")

    @staticmethod
    def v_shape(depth: float, days_down: int, days_up: int, total_days: int) -> DayPath:
        _check(days_down + days_up, total_days)
        down = (1.0 - depth) ** (1.0 / days_down) - 1.0
        up = (1.0 / (1.0 - depth)) ** (1.0 / days_up) - 1.0
        steps = [DayStep(i, [ParameterChange("spot", StressType.PERCENTAGE, down)]) for i in range(days_down)]
        steps += [DayStep(days_down + i, [ParameterChange("spot", StressType.PERCENTAGE, up)]) for i in range(days_up)]
        return DayPath(name=f"v_shape_{depth:.0%}_{days_down}d_{days_up}d", steps=steps + _hold(total_days, days_down + days_up),
                       description="spot falls then fully recovers")

    @staticmethod
    def vol_spike(vol_up: float, decay_days: int, total_days: int) -> DayPath:
        _check(decay_days + 1, total_days)
        steps = [DayStep(0, [ParameterChange("volatility", StressType.ABSOLUTE, vol_up)])]
        steps += [DayStep(i, [ParameterChange("volatility", StressType.ABSOLUTE, -vol_up / decay_days)]) for i in range(1, decay_days + 1)]
        return DayPath(name=f"vol_spike_{vol_up:.0%}_{decay_days}d", steps=steps + _hold(total_days, decay_days + 1),
                       description="vol jumps then decays linearly back")

    @staticmethod
    def basis_blowout(carry_shift: float, days: int, total_days: int) -> DayPath:
        _check(days, total_days)
        steps = [DayStep(i, [ParameterChange("basis", StressType.ABSOLUTE, carry_shift / days)]) for i in range(days)]
        return DayPath(name=f"basis_blowout_{carry_shift:+.0%}_{days}d", steps=steps + _hold(total_days, days),
                       description="annualised carry shifts in parallel, then holds")

    @staticmethod
    def grind_up_to_ko(pct_per_day: float, days: int, total_days: int) -> DayPath:
        _check(days, total_days)
        steps = [DayStep(i, [ParameterChange("spot", StressType.PERCENTAGE, pct_per_day)]) for i in range(days)]
        return DayPath(name=f"grind_up_{pct_per_day:.2%}_{days}d", steps=steps + _hold(total_days, days),
                       description="steady rally into the KO barrier")


def _check(move_days: int, total_days: int) -> None:
    if move_days < 1 or total_days < move_days:
        raise ValidationError("total_days must be at least the number of moving days, both positive")


def stress_set(paths: Sequence[DayPath], *, start: StartState, calendar: pd.DatetimeIndex, tenor_grid: np.ndarray) -> MarketPath:
    """Stack equal-length designed paths into one batch."""
    if not paths:
        raise ValidationError("stress_set needs at least one path")
    singles = [market_path_from_day_path(p, start=start, calendar=calendar, tenor_grid=tenor_grid) for p in paths]
    n_days = {s.n_days for s in singles}
    if len(n_days) != 1:
        raise ValidationError(f"designed paths must have equal length, got {sorted(n_days)}")
    first = singles[0]
    return MarketPath(
        dates=first.dates, spot=np.vstack([s.spot for s in singles]), atm_vol=np.vstack([s.atm_vol for s in singles]),
        rate=np.vstack([s.rate for s in singles]), carry=np.vstack([s.carry for s in singles]),
        tenor_grid=first.tenor_grid, meta={"generator": "designed", "scenario_names": [p.name for p in paths],
                                           "start": first.meta["start"]},
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_designed.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/paths/designed.py test/simulation/test_designed.py
git commit -m "feat(backtest/simulation): designed paths from DayPath and a snowball stress library

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Persistence and the replay-engine dataset for one path

**Files:**
- Modify: `quantark/backtest/simulation/paths/market_path.py` (add `to_npz` / `from_npz`)
- Create: `quantark/backtest/simulation/dataset.py`
- Modify: `test/simulation/test_market_path.py` (append)
- Create: `test/simulation/test_dataset.py`

**Interfaces:**
- Consumes: `MarketPath` (Task 1), `day_chain` (Task 4), `AutocallableMarketDataSet.from_dataframes(spot_data, vol_data, rate_data, futures_data, surface_history=None)` (`quantark.backtest.replay`).
- Produces:
  - `MarketPath.to_npz(path: str | Path) -> None` and `MarketPath.from_npz(path) -> MarketPath` (arrays plus `dates` as int64 ns and `meta` as a JSON string; the fingerprint survives the round trip).
  - `to_market_dataset(path: MarketPath, path_index: int, *, multiplier: float = 200.0) -> AutocallableMarketDataSet` — the replay engine's dataset for one path: spot/vol/rate frames and the futures frame of every day's `DayChain`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_market_path.py`:

```python
def test_npz_round_trip_keeps_arrays_dates_meta_and_fingerprint(tmp_path):
    mp = make_market_path()
    file = tmp_path / "paths.npz"
    mp.to_npz(file)
    back = MarketPath.from_npz(file)
    assert back.fingerprint() == mp.fingerprint()
    assert back.dates.equals(mp.dates) and back.meta == mp.meta
    assert np.array_equal(back.carry, mp.carry)
```

`test/simulation/test_dataset.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay import AutocallableMarketDataSet, FuturesRollPolicy
from quantark.backtest.simulation.carry import day_chain
from quantark.backtest.simulation.dataset import to_market_dataset

from .conftest import make_market_path


def test_dataset_frames_follow_the_path():
    mp = make_market_path(n_paths=2, n_days=30, start=date(2024, 1, 2))
    ds = to_market_dataset(mp, 1)
    assert isinstance(ds, AutocallableMarketDataSet)
    assert list(ds.dates) == list(mp.dates)
    row = ds.get_market_row(mp.dates[5])
    assert row["spot"] == pytest.approx(mp.spot[1, 5])
    assert row["volatility"] == pytest.approx(mp.atm_vol[1, 5])
    assert row["rate"] == pytest.approx(mp.rate[1, 5])
    chain = ds.get_futures_slice(mp.dates[5])
    assert list(chain["contract"]) == list(day_chain(mp, 5).contracts)
    assert chain["futures_price"].to_numpy() == pytest.approx(day_chain(mp, 5).prices[1])
    selected = FuturesRollPolicy().select_contract(chain, mp.dates[5], None)
    assert selected["contract"] == "IM2401"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -k npz test/simulation/test_dataset.py -q`
Expected: `AttributeError: 'MarketPath' object has no attribute 'to_npz'` and `ModuleNotFoundError ... dataset`

- [ ] **Step 3: Write the implementation**

Add to `MarketPath` in `market_path.py` (with `import json` and `from pathlib import Path`):

```python
    def to_npz(self, path) -> None:
        np.savez_compressed(
            Path(path), dates=self.dates.asi8, spot=self.spot, atm_vol=self.atm_vol, rate=self.rate,
            carry=self.carry, tenor_grid=self.tenor_grid, meta=np.array(json.dumps(self.meta, default=str)),
        )

    @classmethod
    def from_npz(cls, path) -> "MarketPath":
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                dates=pd.DatetimeIndex(pd.to_datetime(z["dates"])), spot=z["spot"], atm_vol=z["atm_vol"],
                rate=z["rate"], carry=z["carry"], tenor_grid=z["tenor_grid"], meta=json.loads(str(z["meta"])),
            )
```

`quantark/backtest/simulation/dataset.py`:

```python
"""One simulated path as the replay engine's ``AutocallableMarketDataSet`` (spec §6, §10)."""
from __future__ import annotations

import pandas as pd

from quantark.backtest.replay import AutocallableMarketDataSet

from .carry import FUTURES_MULTIPLIER, day_chain
from .paths.market_path import MarketPath


def to_market_dataset(path: MarketPath, path_index: int, *, multiplier: float = FUTURES_MULTIPLIER) -> AutocallableMarketDataSet:
    single = path.path(path_index)
    dates = list(single.dates)
    spot = pd.DataFrame({"date": dates, "spot": single.spot[0]})
    vol = pd.DataFrame({"date": dates, "volatility": single.atm_vol[0]})
    rate = pd.DataFrame({"date": dates, "rate": single.rate[0]})
    futures = pd.concat([day_chain(single, d, multiplier=multiplier).frame(0) for d in range(single.n_days)],
                        ignore_index=True)
    return AutocallableMarketDataSet.from_dataframes(spot_data=spot, vol_data=vol, rate_data=rate, futures_data=futures)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/paths/market_path.py quantark/backtest/simulation/dataset.py \
        test/simulation/test_market_path.py test/simulation/test_dataset.py
git commit -m "feat(backtest/simulation): MarketPath npz persistence and a replay dataset per path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Public exports, module README, full regression

**Files:**
- Modify: `quantark/backtest/simulation/__init__.py`, `quantark/backtest/simulation/paths/__init__.py`
- Create: `quantark/backtest/simulation/README.md`

**Interfaces:**
- Produces: `from quantark.backtest.simulation import MarketPath, StartState, trading_calendar, DEFAULT_TENOR_GRID, PathHistory, StationaryBlockBootstrap, GBMPaths, ConstantVol, StickyRealisedVol, market_path_from_day_path, SnowballStressLibrary, stress_set, DayChain, day_chain, carry_at, curve_from_chain, listed_im_contracts, dividend_yield_for_day, to_market_dataset`.

- [ ] **Step 1: Write the failing test**

Append to `test/simulation/test_market_path.py`:

```python
def test_public_api_is_exported():
    import quantark.backtest.simulation as sim
    for name in ("MarketPath", "StartState", "trading_calendar", "DEFAULT_TENOR_GRID", "PathHistory",
                 "StationaryBlockBootstrap", "GBMPaths", "ConstantVol", "StickyRealisedVol",
                 "market_path_from_day_path", "SnowballStressLibrary", "stress_set", "DayChain", "day_chain",
                 "carry_at", "curve_from_chain", "listed_im_contracts", "dividend_yield_for_day", "to_market_dataset"):
        assert hasattr(sim, name), name
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -k public_api -q`
Expected: `AssertionError: PathHistory`

- [ ] **Step 3: Export and document**

`quantark/backtest/simulation/__init__.py`:

```python
"""Simulated-path backtest for autocallable books (spec 2026-09-08).

Plan 1: the data layer -- ``MarketPath`` batches, carry-curve to
listed-chain conversion, the shared dividend rule, the history builder and
the three path generators.  See README.md in this package.
"""
from __future__ import annotations

from .carry import DayChain, carry_at, curve_from_chain, day_chain, listed_im_contracts
from .dataset import to_market_dataset
from .dividends import dividend_yield_for_day
from .paths import (
    DEFAULT_TENOR_GRID, ConstantVol, GBMPaths, MarketPath, PathHistory, SnowballStressLibrary,
    StartState, StationaryBlockBootstrap, StickyRealisedVol, market_path_from_day_path, stress_set,
    trading_calendar,
)

__all__ = [
    "DEFAULT_TENOR_GRID", "MarketPath", "StartState", "trading_calendar", "PathHistory",
    "StationaryBlockBootstrap", "GBMPaths", "ConstantVol", "StickyRealisedVol",
    "market_path_from_day_path", "SnowballStressLibrary", "stress_set", "DayChain", "day_chain",
    "carry_at", "curve_from_chain", "listed_im_contracts", "dividend_yield_for_day", "to_market_dataset",
]
```

`quantark/backtest/simulation/paths/__init__.py` re-exports the same path-layer names from `market_path`, `history`, `bootstrap`, `gbm`, `designed`.

`quantark/backtest/simulation/README.md` — write these sections, each two to six sentences, from the spec and the code: *What it is* (spec §1, plan 1 scope), *MarketPath* (the table of §4.1 and the constant-maturity curve paragraph), *Generators* (bootstrap state vector and options, GBM and vol rules, designed paths and the stress library, seeds and `meta`), *From curve to chain to q(T)* (CFFEX calendar, `F_i = S·exp(B(T_i))`, `dividend_yield_for_day` and the curve-as-artifact tail), *Using a path with the replay engine* (`to_market_dataset`), *What comes next* (plans 2–4 in one line each). Include one runnable snippet:

```python
from datetime import date
from quantark.backtest.simulation import (DEFAULT_TENOR_GRID, PathHistory, StationaryBlockBootstrap,
                                          trading_calendar, to_market_dataset)
history = PathHistory.from_frames(spot=spot_df, vol=vol_df, futures=futures_df, rate=0.02, tenor_grid=DEFAULT_TENOR_GRID)
gen = StationaryBlockBootstrap(history, mean_block_days=20, demean_returns=True, annual_drift=0.0, vol_floor=0.08,
                               carry_mode="changes", start=history.snapshot(), calendar=trading_calendar(date(2026, 9, 8), 260))
paths = gen.generate(2000, 260, seed=1)
dataset = to_market_dataset(paths, 17)   # one path for ReplayBacktestEngine
```

- [ ] **Step 4: Run the full regression**

Run: `.venv/bin/python -m pytest test/simulation test/test_replay_dividend_source.py test/test_snowball_q_term_structure_study.py test/replay_golden test/test_forward_carry_curve.py -q`
Expected: all pass (parallel default).

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation test/simulation
git commit -m "feat(backtest/simulation): public API and module guide for the path layer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- §4.1 `MarketPath` and invariants → Task 1; constant-maturity paragraph → Tasks 2, 6; persistence (§9 `to_dir` for paths) → Task 10.
- §5 generators: bootstrap options (`mean_block_days`, demean, drift, `vol_floor` with counted hits, `carry_mode`) → Task 7; GBM with `constant` / `sticky_realised` rules and `carry_schedule` → Task 8; designed paths, stress library, sets → Task 9; seeds and `meta` → all three.
- §6 contract calendar, prices, `q(T)` rule shared with the engine and the study, curve-as-artifact tail, chain builder exported for the oracle → Tasks 3, 4, 5, 10.
- §12 fail-closed → every task's validation tests.
- Not in this plan (by design): §7 providers and cache, §8 loop, §9 results beyond persistence, §10 oracle — plans 2–4.
- Type consistency check: `MarketPath` field names (`spot`, `atm_vol`, `rate`, `carry`, `tenor_grid`, `meta`) are used identically in Tasks 4–10; `DayChain.prices` is `(n_paths, n_contracts)` everywhere; `StartState.carry` is `(n_tenors,)`; `dividend_yield_for_day` keyword names match between Task 5's implementation and tests; `GBMPaths` takes `tenor_grid` (Task 8 note) and the test's `_gen` must pass it.
