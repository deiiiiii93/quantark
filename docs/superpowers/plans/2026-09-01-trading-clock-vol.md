# Trading-Clock Volatility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Price with vol quoted per trading year (244 CFFEX/CSI/SSE, 252 CFETS) while r/q stay calendar-quoted, on either engine time axis, without changing any engine.

**Architecture:** One shared `BusinessTimeMap` (calendar↔trading time; `to_trading` places variance with exact holiday plateaus, `to_calendar` places carry continuously between trading-date knots). A variance-preserving `TradingClockVolSurface` serves the default calendar axis; `TradingClockRateCurve`/`TradingClockDividendYield` serve the opt-in trading axis. Degenerate zero-variance steps get exact treatment in QUAD (deterministic shift) and PDE (upwind + θ=1).

**Tech Stack:** Python 3.10+, numpy, existing `quantark` layers (`util/calendar`, `param`, `priceenv`, equity engines). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md` — read it first; every task cites its sections.

## Global Constraints

- **Run tests from the worktree with the source shadowed**: the editable install resolves `quantark` to the main repo, so ALWAYS use
  `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest <target> -n0`
  (`-n0` for targeted runs; plain `python -m pytest` would test the WRONG tree).
- **Exact semantics, no floors** (spec D6): never introduce a vol floor or tolerance to dodge a degenerate case; zero variance is a correct value with exact handling.
- **Degenerate triggers key on `Δw == 0.0` exactly** (spec §4.2): bitwise plateau equality is a hard contract, not a tolerance.
- **Denominators**: 244 (CFFEX/CSI/SSE), 252 (CFETS); the denominator always travels inside a `TradingClock`, never as a bare int parameter.
- **Numerical utilities**: use `quantark.util.numerical` (`is_zero`, `safe_sqrt`, `safe_divide`, `Tolerance`) — never raw float comparisons or hardcoded tolerances, EXCEPT the exact `== 0.0` trigger above, which is deliberate and documented.
- **Existing behavior is bitwise-frozen**: surfaces without `total_variance`, PDE runs with no zero-vol steps, and QUAD runs with all-positive vols must produce byte-identical results before/after every task (regression steps below enforce this).
- **Commit after every task** with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` footer; `git add -f` is required for new files under `docs/`.
- All library dates are `datetime` (the `Calendar` class normalizes); year fractions are ACT/365 on the calendar axis (spec §4.2 contract).

---

## Phase 1 — calendar axis (delivers trading-clock vol)

### Task 1: `TradingClock` + `BusinessTimeMap`

**Files:**
- Create: `quantark/util/calendar/trading_clock.py`
- Modify: `quantark/util/calendar/__init__.py` (add exports)
- Test: `test/test_trading_clock_map.py`

**Interfaces:**
- Consumes: `quantark.util.calendar.business_calendar.Calendar` (`is_business_day(datetime) -> bool`), `quantark.util.exceptions.ValidationError`.
- Produces (later tasks rely on these exact names):
  - `TradingClock(calendar: Calendar, days_per_year: int)` — frozen dataclass.
  - `BusinessTimeMap(clock, anchor_date: datetime, horizon_date: datetime, extend_weekdays: bool = False)` with
    `to_trading(tau_cal: float | np.ndarray) -> float | np.ndarray`,
    `to_calendar(tau_td: float | np.ndarray) -> float | np.ndarray`,
    `initial_slope() -> float`, and attribute `clock`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_trading_clock_map.py
"""BusinessTimeMap property tests (spec 2026-09-01-trading-clock-vol §4.1, §7.1)."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.util.calendar import Calendar, CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError

ANCHOR = datetime(2026, 2, 9)   # Monday; CNY 2026 week: 2/16(Mon)-2/20(Fri) holidays
HORIZON = datetime(2027, 2, 9)


def _cn_calendar():
    return create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def _map(days_per_year=244):
    clock = TradingClock(calendar=_cn_calendar(), days_per_year=days_per_year)
    return BusinessTimeMap(clock, ANCHOR, HORIZON)


def test_to_trading_monotone_and_bounded():
    m = _map()
    taus = np.linspace(0.0, 0.9, 4001)
    out = m.to_trading(taus)
    assert np.all(np.diff(out) >= 0.0)
    assert out[0] == 0.0


def test_holiday_plateau_is_bitwise_flat():
    """Spec §4.2(a): inside a plateau to_trading returns the stored knot VALUE."""
    m = _map()
    cal = _cn_calendar()
    # find the first holiday weekday after anchor
    d = ANCHOR
    while cal.is_business_day(d) or d.weekday() >= 5:
        d += timedelta(days=1)
    t0 = (d - ANCHOR).days / 365.0
    a = m.to_trading(t0 + 0.10 / 365.0)
    b = m.to_trading(t0 + 0.90 / 365.0)
    assert a == b                      # bitwise, not approx


def test_trading_day_slope():
    m = _map()
    cal = _cn_calendar()
    d = ANCHOR
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    t0 = (d - ANCHOR).days / 365.0
    lo, hi = m.to_trading(t0), m.to_trading(t0 + 1.0 / 365.0)
    assert (hi - lo) == pytest.approx(1.0 / 244.0, abs=1e-15)


def test_knot_round_trip_exact():
    """to_trading and to_calendar agree exactly at every trading-date knot."""
    m = _map()
    cal = _cn_calendar()
    k, d = 0, ANCHOR
    while d < ANCHOR + timedelta(days=120):
        if cal.is_business_day(d):
            k += 1
            u = k / 244.0
            c = m.to_calendar(u)
            assert m.to_trading(c) == pytest.approx(u, abs=1e-15)
        d += timedelta(days=1)


def test_to_calendar_is_continuous_and_strictly_increasing():
    """Spec §4.1 (review iter-2 P2a): no plateau-start jumps in the cash clock."""
    m = _map()
    us = np.linspace(0.0, 200.0 / 244.0, 20001)
    cs = m.to_calendar(us)
    diffs = np.diff(cs)
    assert np.all(diffs > 0.0)                       # strictly increasing
    assert float(np.max(diffs)) < 15.0 / 365.0 / 40  # no jump anywhere near a holiday span


def test_horizon_fail_closed():
    m = _map()
    with pytest.raises(ValidationError):
        m.to_trading(1.5)          # past horizon
    with pytest.raises(ValidationError):
        m.to_trading(-0.1)


def test_cfets_denominator():
    m = _map(days_per_year=252)
    cal = _cn_calendar()
    d = ANCHOR
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    t0 = (d - ANCHOR).days / 365.0
    assert (m.to_trading(t0 + 1.0 / 365.0) - m.to_trading(t0)) == pytest.approx(1.0 / 252.0, abs=1e-15)


def test_ndarray_and_scalar_agree():
    m = _map()
    taus = np.array([0.01, 0.1, 0.3])
    arr = m.to_trading(taus)
    for i, t in enumerate(taus):
        assert arr[i] == m.to_trading(float(t))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_map.py -n0 -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` for `trading_clock`.

- [ ] **Step 3: Implement `trading_clock.py`**

```python
"""Trading-clock ↔ calendar-clock time maps.

Spec: docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md §4.1.
`to_trading` places VARIANCE: piecewise linear, slope (365/D) across a
trading day, exactly flat (stored knot value, no arithmetic) across
holidays. `to_calendar` places CARRY: the continuous piecewise-linear map
between consecutive trading-date knots — deliberately NOT the pointwise
inverse of `to_trading`; the two agree exactly at trading-date knots.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from quantark.util.calendar.business_calendar import Calendar
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class TradingClock:
    """A calendar paired with its annualization denominator (244 SSE, 252 CFETS)."""

    calendar: Calendar
    days_per_year: int

    def __post_init__(self) -> None:
        if self.days_per_year <= 0:
            raise ValidationError(
                f"days_per_year must be positive, got {self.days_per_year}"
            )


class BusinessTimeMap:
    """Precomputed daily-knot map anchored at the valuation date."""

    def __init__(
        self,
        clock: TradingClock,
        anchor_date: datetime,
        horizon_date: datetime,
        extend_weekdays: bool = False,
    ) -> None:
        if horizon_date <= anchor_date:
            raise ValidationError("horizon_date must be after anchor_date")
        self.clock = clock
        self.anchor_date = anchor_date
        self.extend_weekdays = bool(extend_weekdays)
        n_days = (horizon_date - anchor_date).days
        inv_d = 1.0 / float(clock.days_per_year)

        # td_start[i] = trading time at the START of calendar day i;
        # is_td[i] = day i is a trading day. td_start has n_days+1 entries.
        is_td = np.zeros(n_days, dtype=bool)
        day = anchor_date
        for i in range(n_days):
            is_td[i] = clock.calendar.is_business_day(day)
            day += timedelta(days=1)
        td_start = np.concatenate(([0.0], np.cumsum(np.where(is_td, inv_d, 0.0))))
        self._is_td = is_td
        self._td_start = td_start
        self._n_days = n_days
        self._inv_d = inv_d
        # knots for to_calendar: (u_k, c_k) at the END of each trading day
        td_idx = np.nonzero(is_td)[0]
        self._u_knots = np.concatenate(([0.0], td_start[td_idx + 1]))
        self._c_knots = np.concatenate(([0.0], (td_idx + 1) / 365.0))

    def _day_frac(self, tau_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = tau_cal * 365.0
        i = np.floor(f).astype(int)
        return i, f - i

    def to_trading(self, tau_cal):
        arr = np.asarray(tau_cal, dtype=float)
        scalar = arr.ndim == 0
        a = np.atleast_1d(arr)
        if np.any(a < 0.0):
            raise ValidationError("tau_cal must be non-negative")
        i, frac = self._day_frac(a)
        over = i >= self._n_days
        if np.any(over):
            if not self.extend_weekdays:
                raise ValidationError(
                    "time beyond BusinessTimeMap horizon; extend the horizon or "
                    "construct with extend_weekdays=True"
                )
            raise ValidationError(
                "extend_weekdays horizon extension not yet implemented for this "
                "query range; extend horizon_date"
            )
        # trading day: knot + frac/D; holiday: the stored knot VALUE, no
        # arithmetic — plateau bitwise-equality contract (spec §4.2a).
        out = np.where(
            self._is_td[i],
            self._td_start[i] + frac * self._inv_d,
            self._td_start[i],
        )
        return float(out[0]) if scalar else out

    def to_calendar(self, tau_td):
        arr = np.asarray(tau_td, dtype=float)
        scalar = arr.ndim == 0
        a = np.atleast_1d(arr)
        if np.any(a < 0.0):
            raise ValidationError("tau_td must be non-negative")
        if np.any(a > self._u_knots[-1]):
            raise ValidationError("tau_td beyond BusinessTimeMap horizon")
        out = np.interp(a, self._u_knots, self._c_knots)
        return float(out[0]) if scalar else out

    def initial_slope(self) -> float:
        """d(to_trading)/d(tau_cal) at 0+ — 365/D on a trading day, else 0."""
        if self._n_days == 0 or not self._is_td[0]:
            return 0.0
        return 365.0 * self._inv_d

    def __repr__(self) -> str:
        return (
            f"BusinessTimeMap(D={self.clock.days_per_year}, "
            f"anchor={self.anchor_date.date()}, days={self._n_days}, "
            f"extend_weekdays={self.extend_weekdays})"
        )
```

Note the `extend_weekdays=True` branch raises for now: Phase 1 never
queries past horizon (constructors size the horizon from the product), and
inventing weekday extension without a consumer violates YAGNI; the flag
exists so the constructor contract is stable. Leave it exactly as written.

- [ ] **Step 4: Add exports**

In `quantark/util/calendar/__init__.py`, add to the existing import block and `__all__`:

```python
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_map.py -n0 -v`
Expected: all PASS. If `create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))` rejects the kwarg, read `business_calendar.py:447` and adapt the test helper to the actual factory signature — do not change the factory.

- [ ] **Step 6: Commit**

```bash
git add quantark/util/calendar/trading_clock.py quantark/util/calendar/__init__.py test/test_trading_clock_map.py
git commit -m "feat(clock): TradingClock + BusinessTimeMap with exact holiday plateaus"
```

### Task 2: `TradingClockVolSurface`

**Files:**
- Create: `quantark/param/vol/trading_clock_surface.py`
- Modify: `quantark/param/vol/__init__.py` (export)
- Test: `test/test_trading_clock_surface.py`

**Interfaces:**
- Consumes: `BusinessTimeMap.to_trading`, `initial_slope` (Task 1); `BlackImpliedVolSurface` ABC (`get_vol(strike, time_to_maturity, spot)`), `TermStructureVolSurface` for tests.
- Produces: `TradingClockVolSurface(inner: BlackImpliedVolSurface, time_map: BusinessTimeMap)` with `get_vol(strike, tau_cal, spot)` and `total_variance(strike, tau_cal, spot)` (full query signature — spec review iter-2 P2c). `is_smile` mirrors `inner`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_trading_clock_surface.py
"""TradingClockVolSurface: variance preservation (spec §4.2, §7)."""
from datetime import datetime

import numpy as np
import pytest

from quantark.param.vol.vol_surface import FlatVolSurface, TermStructureVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

ANCHOR = datetime(2026, 2, 9)
HORIZON = datetime(2027, 2, 9)


def _map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return BusinessTimeMap(TradingClock(cal, 244), ANCHOR, HORIZON)


def test_total_variance_is_preserved_flat():
    """sigma_td flat: w_cal(tau) == sigma_td^2 * tau_td(tau) for any tau."""
    m = _map()
    s = TradingClockVolSurface(inner=FlatVolSurface(0.20), time_map=m)
    for tau in (0.05, 0.11, 0.37, 0.80):
        tau_td = m.to_trading(tau)
        sigma_cal = s.get_vol(100.0, tau, 100.0)
        assert sigma_cal**2 * tau == pytest.approx(0.04 * tau_td, rel=1e-14)


def test_total_variance_method_matches_get_vol_reconstruction():
    m = _map()
    inner = TermStructureVolSurface(times=[10 / 244, 60 / 244, 200 / 244],
                                    vols=[0.25, 0.22, 0.20])
    s = TradingClockVolSurface(inner=inner, time_map=m)
    tau = 0.33
    w = s.total_variance(100.0, tau, 100.0)
    sigma = s.get_vol(100.0, tau, 100.0)
    assert w == pytest.approx(sigma * sigma * tau, rel=1e-13)


def test_is_smile_passthrough():
    m = _map()
    assert TradingClockVolSurface(FlatVolSurface(0.2), m).is_smile is False


def test_zero_time_limit_uses_initial_slope():
    m = _map()
    s = TradingClockVolSurface(FlatVolSurface(0.20), m)
    v = s.get_vol(100.0, 0.0, 100.0)
    assert v == pytest.approx(0.20 * np.sqrt(m.initial_slope()), rel=1e-14)


def test_total_variance_vectorized_over_time():
    m = _map()
    s = TradingClockVolSurface(FlatVolSurface(0.20), m)
    taus = np.array([0.05, 0.11, 0.37])
    w = s.total_variance(100.0, taus, 100.0)
    assert w.shape == taus.shape
    for i, t in enumerate(taus):
        assert w[i] == s.total_variance(100.0, float(t), 100.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_surface.py -n0 -v`
Expected: FAIL — ImportError.

- [ ] **Step 3: Implement**

```python
# quantark/param/vol/trading_clock_surface.py
"""Wrap a trading-time-quoted implied surface for calendar-axis engines.

Spec §4.2: get_vol(K, tau_cal) = sigma_inner(K, tau_td) * sqrt(tau_td/tau_cal)
preserves total variance exactly: w_cal(tau) = w_td(to_trading(tau)).
`total_variance` carries the FULL get_vol query signature so smile surfaces
reproduce the identical inner query, and computes w once from tau_td so
holiday plateaus difference to Dw == 0.0 exactly downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.param.vol.vol_surface import BlackImpliedVolSurface
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class TradingClockVolSurface(BlackImpliedVolSurface):
    inner: BlackImpliedVolSurface
    time_map: BusinessTimeMap

    def __post_init__(self) -> None:
        if not isinstance(self.inner, BlackImpliedVolSurface):
            raise ValidationError("inner must be a BlackImpliedVolSurface")
        # class-level passthrough is impossible on a frozen dataclass; the
        # instance attribute shadows the ClassVar, which every consumer of
        # is_smile reads via the instance.
        object.__setattr__(self, "is_smile", bool(self.inner.is_smile))

    def total_variance(self, strike: float, time_to_maturity, spot: float):
        """w_cal(tau_cal) = sigma_inner(K, tau_td)^2 * tau_td, vectorized."""
        tau_td = self.time_map.to_trading(time_to_maturity)
        arr = np.atleast_1d(np.asarray(tau_td, dtype=float))
        out = np.empty_like(arr)
        for j, u in enumerate(arr):
            if u <= 0.0:
                out[j] = 0.0
            else:
                v = float(self.inner.get_vol(float(strike), float(u), float(spot)))
                out[j] = v * v * u
        if np.ndim(tau_td) == 0:
            return float(out[0])
        return out

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        tau_cal = float(time_to_maturity)
        if tau_cal <= 0.0:
            slope = self.time_map.initial_slope()
            u = 1.0 / (365.0 * 10.0)  # one-tenth day in trading units for the inner lookup
            v = float(self.inner.get_vol(float(strike), u, float(spot)))
            return v * float(np.sqrt(slope))
        tau_td = float(self.time_map.to_trading(tau_cal))
        if tau_td <= 0.0:
            return 0.0  # pure-holiday horizon: zero accrued variance (documented)
        v = float(self.inner.get_vol(float(strike), tau_td, float(spot)))
        return v * float(np.sqrt(tau_td / tau_cal))

    def __repr__(self) -> str:
        return f"TradingClockVolSurface({self.inner!r}, {self.time_map!r})"
```

- [ ] **Step 4: Export** — add `TradingClockVolSurface` to `quantark/param/vol/__init__.py` mirroring the existing export style.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_surface.py -n0 -v`
Expected: PASS. If the `is_smile` instance-shadow trips the ABC (it is a ClassVar on the base), switch to overriding a `@property is_smile` on this class returning `self.inner.is_smile` and drop the `object.__setattr__` — whichever satisfies both the dataclass and consumers reading `surface.is_smile`.

- [ ] **Step 6: Commit**

```bash
git add quantark/param/vol/trading_clock_surface.py quantark/param/vol/__init__.py test/test_trading_clock_surface.py
git commit -m "feat(clock): TradingClockVolSurface — variance-preserving calendar-axis wrapper"
```

### Task 3: `total_variance` fast path in term sampling

**Files:**
- Modify: `quantark/param/term_sampling.py` (`step_vols_on_grid`)
- Modify: `quantark/priceenv/term_sampling.py` (`TermCoefficients.from_env`)
- Test: `test/test_term_sampling_total_variance.py`

**Interfaces:**
- Consumes: `TradingClockVolSurface.total_variance(strike, t, spot)` (Task 2).
- Produces: `step_vols_on_grid(get_vol, ref_strike, t_grid, total_variance=None)` — new optional keyword, a callable `(strike, t_array) -> w_array`. `TermCoefficients.from_env` detects `total_variance` on `pricing_env.vol_surface` via getattr and passes a bound lambda closing over `pricing_env.spot`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_term_sampling_total_variance.py
"""Plateau exactness through the full step-vol path (spec §4.2, §7.7)."""
from datetime import datetime

import numpy as np

from quantark.param.term_sampling import step_vols_on_grid
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

ANCHOR = datetime(2026, 2, 9)


def _surface():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 2, 9))
    return TradingClockVolSurface(FlatVolSurface(0.20), m), m


def test_holiday_interval_step_vol_is_exactly_zero():
    surface, m = _surface()
    # dense grid: 8 steps per calendar day across 40 days (covers CNY week)
    t = np.linspace(0.0, 40.0 / 365.0, 40 * 8 + 1)
    tv = lambda k, ts: surface.total_variance(k, ts, 100.0)
    out = step_vols_on_grid(
        lambda k, ts: surface.get_vol(k, ts, 100.0), 100.0, t, total_variance=tv
    )
    tau_td = m.to_trading(t)
    holiday_steps = np.diff(tau_td) == 0.0
    assert holiday_steps.any()                       # CNY is in range
    assert np.all(out[holiday_steps] == 0.0)         # BITWISE zero (spec trigger)
    assert np.all(out[~holiday_steps] > 0.0)


def test_without_total_variance_existing_path_is_bitwise_unchanged():
    t = np.linspace(0.01, 1.0, 250)
    get_vol = lambda k, ts: 0.20
    a = step_vols_on_grid(get_vol, 100.0, t)
    b = step_vols_on_grid(get_vol, 100.0, t, total_variance=None)
    assert np.array_equal(a, b)


def test_fast_path_rejects_decreasing_w():
    import pytest
    from quantark.util.exceptions import NumericalError
    t = np.array([0.0, 0.1, 0.2])
    bad_tv = lambda k, ts: np.where(np.asarray(ts) > 0.15, 0.001, 0.004)
    with pytest.raises(NumericalError):
        step_vols_on_grid(lambda k, ts: 0.2, 100.0, t, total_variance=bad_tv)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_term_sampling_total_variance.py -n0 -v`
Expected: FAIL — `TypeError: step_vols_on_grid() got an unexpected keyword argument`.

- [ ] **Step 3: Implement the fast path**

In `quantark/param/term_sampling.py`, change the signature of `step_vols_on_grid` and insert the branch BEFORE the existing `get_vol` sampling (existing body stays byte-identical for `total_variance=None`):

```python
def step_vols_on_grid(
    get_vol: Callable[[float, float], float],
    ref_strike: float,
    t_grid: np.ndarray,
    total_variance: Callable[[float, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Per-interval vols from total-variance differencing at a reference strike.

    w(t) = get_vol(ref_strike, t)^2 * t;  step vol = sqrt((w1 - w0) / dt).
    When ``total_variance`` is provided (surfaces exposing an exact w, e.g.
    TradingClockVolSurface), w is taken from it directly — no sigma^2*t
    reconstruction — so holiday plateaus difference to EXACTLY 0.0 (the
    degenerate-branch trigger; spec 2026-09-01 trading-clock-vol §4.2).
    Raises NumericalError if total variance decreases beyond tolerance
    (calendar arbitrage in the input surface).
    """
    t = _validate_grid(t_grid)
    if total_variance is not None:
        w = np.asarray(total_variance(float(ref_strike), t), dtype=float)
        if w.shape != t.shape or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise NumericalError("total_variance produced an invalid w array")
        dw = np.diff(w)
        if np.any(dw < -1e-12):
            raise NumericalError(
                "total variance is decreasing on the grid (calendar arbitrage)"
            )
        out = np.sqrt(np.maximum(dw, 0.0) / np.diff(t))
        if not np.all(np.isfinite(out)):
            raise NumericalError("total_variance produced non-finite step vols")
        return out
    # ... existing body unchanged from here ...
```

In `quantark/priceenv/term_sampling.py`, inside `TermCoefficients.from_env`, replace the `step_vols` line:

```python
        surface = getattr(pricing_env, "vol_surface", None)
        tv_method = getattr(surface, "total_variance", None)
        tv = None
        if tv_method is not None:
            spot = pricing_env.spot
            tv = lambda k, ts: tv_method(k, ts, spot)
        return cls(
            t_grid=t,
            fwd_rates=fwd_rates,
            fwd_carry=forward_carry_on_grid(pricing_env.get_div_yield, t),
            step_vols=step_vols_on_grid(
                pricing_env.get_vol, ref_strike, t, total_variance=tv
            ),
            node_dfs=node_dfs,
            step_dfs=node_dfs[1:] / node_dfs[:-1],
        )
```

- [ ] **Step 4: Run new tests + regression**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_term_sampling_total_variance.py test/test_param_term_sampling.py test/test_pde_term_coefficients.py test/test_mc_term_inputs.py -n0 -v`
Expected: all PASS (the last three prove existing surfaces bitwise-unchanged).

- [ ] **Step 5: Commit**

```bash
git add quantark/param/term_sampling.py quantark/priceenv/term_sampling.py test/test_term_sampling_total_variance.py
git commit -m "feat(clock): exact total-variance fast path in step-vol sampling"
```

### Task 4: QUAD deterministic-shift step for zero variance

**Files:**
- Modify: `quantark/asset/equity/engine/quad/quad_core.py`
- Test: `test/test_quad_zero_vol_step.py`

**Interfaces:**
- Consumes: `QuadratureCore(grid_x, spot, observation_times, rate, div, vol)`; per-step arrays `self.vol`, `self.tau`, `self.dt`, `self.r`, `self.q`; `_calculate_values_in_process` (step transition), `_factor_value_at_m` (analytic sub-period factors).
- Produces: a `QuadratureCore` that accepts `vol[m] == 0.0` on interior steps and prices them as an exact deterministic shift. No API change.

- [ ] **Step 1: Write the failing test**

```python
# test/test_quad_zero_vol_step.py
"""Zero-variance QUAD interval == deterministic shift (spec §4.5, §7.6)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.quad.quad_core import QuadCoreInputs, QuadratureCore


def _price(times, vols, r=0.02, q=0.01, grid_x=2001, spot=100.0):
    core = QuadratureCore(
        grid_x=grid_x, spot=spot, observation_times=list(times),
        rate=r, div=q, vol=list(vols),
    )
    n = len(times)
    inputs = QuadCoreInputs(
        observation_times=list(times),
        k_plus=[np.inf] * n, k_minus=[0.0] * n,          # no barriers: European
        # remaining QuadCoreInputs fields: copy the European construction
        # used in test files importing QuadCoreInputs (grep
        # "QuadCoreInputs(" under test/) — the payoff is a vanilla call K=100.
    )
    return core.price(inputs)


def test_zero_vol_middle_interval_equals_collapsed_grid():
    """[0.3, 0.35, 0.7] with vol=(0.2, 0.0, 0.2): the middle interval carries
    only drift/discount, so the price equals the two-interval product with the
    same per-interval variances and the same total carry."""
    p_split = _price(times=(0.3, 0.35, 0.7), vols=(0.2, 0.0, 0.2))
    # collapsed control: same total variance and same DF/carry to each obs
    # date; interval [0.3, 0.7] gets the vol that reproduces
    # w = 0.2^2*0.3 (nothing added in [0.3,0.35]) + 0.2^2*0.35 over 0.4y
    vol_23 = np.sqrt((0.2**2 * 0.35) / 0.4)
    p_ctrl = _price(times=(0.3, 0.7), vols=(0.2, vol_23))
    assert p_split == pytest.approx(p_ctrl, rel=5e-5)


def test_all_positive_vols_bitwise_unchanged():
    """Regression: with no zero step the refactor must not change a single bit."""
    a = _price(times=(0.3, 0.7), vols=(0.2, 0.21))
    # This value is asserted against itself pre/post change by running the test
    # once on the parent commit and freezing the printed repr below.
    print(repr(a))  # freeze on first run, then assert equality
```

Before implementing, run the second test on the parent commit and freeze
the printed value into an `assert a == <frozen>` line. Also complete the
`QuadCoreInputs` construction from an existing usage (grep
`QuadCoreInputs(` in `quantark/asset/equity/engine/quad/quad_adapters.py`)
— the adapter shows the exact field list for a European payoff.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_quad_zero_vol_step.py -n0 -v`
Expected: FAIL — `ValidationError: volatility must be positive.`

- [ ] **Step 3: Implement**

Three edits in `quad_core.py`:

(3a) `__init__` validation — allow exact zeros, still reject negatives; guard the α/β division; keep the too-small check for POSITIVE-vol steps only:

```python
        if np.any(self.vol[1:] < 0.0):
            raise ValidationError("volatility must be non-negative.")
        ...
        self.tau = 0.5 * self.vol * self.vol * self.dt
        pos = self.vol > 0.0
        if np.any((self.tau[1:] <= 0.0) & pos[1:]):
            raise ValidationError("time step too small for quadrature solver.")

        with np.errstate(divide="ignore", invalid="ignore"):
            v2 = np.where(pos, self.vol * self.vol, 1.0)
            self.alpha = np.where(
                pos, (self.r - self.q - 0.5 * self.vol * self.vol) / v2, 0.0
            )
            self.beta = np.where(
                pos,
                (self.r - self.q - 0.5 * self.vol * self.vol) ** 2 / v2**2
                + 2.0 * self.r / v2,
                0.0,
            )
        ...
        vol_pos = self.vol[1:][self.vol[1:] > 0.0]
        if vol_pos.size == 0:
            raise ValidationError("at least one interval must carry variance.")
        vol_max = float(np.max(vol_pos))
```

(3b) `_factor_value_at_m` σ→0 limit — after `vol = float(self.vol[step_index])`, insert the deterministic branch (same sign-aware pattern as `quantark/param/vol/vannavolga/bs_fx.py::_d1_d2`):

```python
        if vol == 0.0:
            spot_array = np.asarray(spot, dtype=float)
            forward = spot_array * math.exp((r - q) * dt)
            itm = np.where(forward >= strike, 1.0, 0.0)
            ind = itm if epsilon >= 0 else 1.0 - itm
            if kind == "a":
                return spot_array * math.exp(-q * dt) * ind
            return math.exp(-r * dt) * ind
```

(place it after the special-strike blocks so `strike <= 0` / `inf` keep
their existing exact handling).

(3c) `_calculate_values_in_process` deterministic transition — at the top:

```python
        if float(self.tau[m]) == 0.0:
            return self._deterministic_step(
                m, y_array, bound_upper, bound_lower, factors,
                prev_values, prev_points, prev_boundaries,
            )
```

and add the method:

```python
    def _deterministic_step(
        self, m, y_array, bound_upper, bound_lower, factors,
        prev_values, prev_points, prev_boundaries,
    ) -> StepState:
        """sigma -> 0 limit of the transition: y_{m-1}(x) =
        e^{-r dt} * y_m(x + (r - q) dt), linear interpolation on the log
        grid (the only numerical error the Gaussian step already carries).
        Spec 2026-09-01 trading-clock-vol §4.5."""
        dt = float(self.dt[m])
        r, q = float(self.r[m]), float(self.q[m])
        shift = (r - q) * dt
        df = math.exp(-r * dt)

        bound_lr_m_n1 = float(bound_lower[m - 1])
        bound_ur_m_n1 = float(bound_upper[m - 1])
        p_lr, p_ur, p0 = self._select_simpson_indices(bound_lr_m_n1, bound_ur_m_n1)
        xee_lr = 0.5 * (self.grid[p_lr] + bound_lr_m_n1)
        xee_ur = 0.5 * (self.grid[p_ur + p0] + bound_ur_m_n1)
        x_m_n1 = np.array([bound_lr_m_n1, bound_ur_m_n1, xee_lr, xee_ur])

        y_shifted = df * np.interp(self.grid + shift, self.grid, y_array)
        v_quad = df * np.interp(x_m_n1 + shift, self.grid, y_array)

        spot_pts = self.spot * np.exp(x_m_n1)
        values = v_quad + self._barrier_payoff(spot_pts, factors, m)
        y_new = self._add_barrier_payoff(
            y_shifted, factors, self.spot * np.exp(self.grid), m
        )
        u_new = self._calculate_integral_simpson(y_new, p_lr, p_ur, p0)
        return StepState(
            y_array=y_new, u_array=u_new,
            values=tuple(values),
            points=(p_lr, p_ur, p0),
            boundaries=(bound_lr_m_n1, bound_ur_m_n1, xee_lr, xee_ur),
        )
```

Also short-circuit the two callers that build ω before the branch can fire:
in `_backward_recursion`, move `omega_array = self._omega(...)` /
`prefactor = self._prefactor(m)` INSIDE an `if float(self.tau[m]) > 0.0:`
and pass `None` otherwise (the deterministic branch never reads them). Keep
`_omega`/`_prefactor` untouched.

- [ ] **Step 4: Run tests + QUAD regression**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_quad_zero_vol_step.py -n0 -v`
Then the QUAD family: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/ -k "quad" -n0 -q`
Expected: all PASS; the frozen bitwise value in test 2 must match exactly.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/quad_core.py test/test_quad_zero_vol_step.py
git commit -m "feat(clock): exact deterministic-shift QUAD step for zero-variance intervals"
```

### Task 5: PDE upwind + θ=1 on zero-diffusion steps

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py` (`StepCoefficients`, `_build_step_coefficients`, the sweep site at ~line 1245)
- Test: `test/test_pde_zero_diffusion_step.py`

**Interfaces:**
- Consumes: `StepCoefficients` NamedTuple (`lcu_sets`, `set_index`, `n_unique` at base_pde_solver.py:85); the vectorized coefficient build (lines 1105-1141); the sweep's `theta = float(theta_by_step[j])` (line ~1250).
- Produces: `StepCoefficients` gains `zero_diffusion_sets: frozenset` (default `frozenset()` — every existing constructor call stays valid); zero-σ unique sets get upwind (l, c, u); the sweep forces `theta = 1.0` on those steps.

- [ ] **Step 1: Write the failing test**

```python
# test/test_pde_zero_diffusion_step.py
"""Zero-diffusion steps: upwind operator + theta=1 monotonicity (spec §4.5)."""
import numpy as np

from quantark.asset.equity.engine.pde.base_pde_solver import StepCoefficients


def test_step_coefficients_zero_diffusion_default_empty():
    sc = StepCoefficients(lcu_sets=[(np.zeros(3),) * 3],
                          set_index=np.zeros(1, dtype=int), n_unique=1)
    assert sc.zero_diffusion_sets == frozenset()


def test_upwind_coefficients_signs_positive_drift():
    """Build coefficients on a grid whose step vols include an exact zero and
    check the zero-sigma set is upwind: l == 0, u == mu/dx >= 0, c <= 0."""
    from test.helpers_pde_zero_diffusion import build_coeffs_with_zero_step
    sc, dx, r, q = build_coeffs_with_zero_step(mu_sign=+1)
    k = next(iter(sc.zero_diffusion_sets))
    l, c, u = sc.lcu_sets[k]
    mu = r - q
    assert np.all(l[1:-1] == 0.0)
    assert np.allclose(u[1:-1], mu / dx)
    assert np.all(c[1:-1] <= 0.0)


def test_monotone_no_new_extrema_on_kinked_profile():
    """A pure-advection theta=1 upwind step of a kinked payoff creates no new
    extrema (M-matrix property)."""
    from test.helpers_pde_zero_diffusion import advance_zero_diffusion_step
    x = np.linspace(-0.5, 0.5, 401)
    v0 = np.maximum(0.0, 1.0 - np.abs(x) * 4.0)          # kink at 0 and +-0.25
    v1 = advance_zero_diffusion_step(v0, x, r=0.02, q=0.01, dt=9.0 / 365.0)
    assert float(np.min(v1)) >= -1e-15
    assert float(np.max(v1)) <= float(np.max(v0)) + 1e-15
```

Write `test/helpers_pde_zero_diffusion.py` alongside: `build_coeffs_with_zero_step`
constructs a minimal env (flat r=0.02, q=0.01) with a `TradingClockVolSurface`
over a CNY-straddling window (reuse the Task 3 fixture pattern), calls the
solver's `_build_step_coefficients` on a grid whose holiday steps produce
σ_step == 0, and returns `(StepCoefficients, dx, r, q)`.
`advance_zero_diffusion_step` assembles the θ=1 tridiagonal
`(I − dt·A)V₁ = V₀` from the upwind (l, c, u) and solves with
`scipy.linalg.solve_banded` — 15 lines, no engine internals.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_pde_zero_diffusion_step.py -n0 -v`
Expected: FAIL — `AttributeError: zero_diffusion_sets` / helper ImportError.

- [ ] **Step 3: Implement**

(3a) `StepCoefficients` — append a defaulted field (NamedTuple keeps all existing constructor calls valid):

```python
class StepCoefficients(NamedTuple):
    ...
    lcu_sets: list
    set_index: np.ndarray
    n_unique: int
    zero_diffusion_sets: frozenset = frozenset()
```

(3b) `_build_step_coefficients` — after `uniq, set_index = np.unique(...)` compute the zero-σ mask, and branch the coefficient build per unique set. Replace the vectorized block's final assembly with:

```python
        sig_zero = uniq[:, 2] == 0.0            # exact: the §4.2 trigger
        mu_full = r_u - q_u                     # -0.5*sig^2 term vanishes at sig=0
        ...
        # existing central-difference fill for ALL rows stays as-is, then
        # overwrite the zero-diffusion rows with first-order upwind selected
        # by the sign of mu (monotone M-matrix; spec §4.5):
        for k in np.nonzero(sig_zero)[0]:
            mu_k = float(mu_full[k, 0])
            r_k = float(uniq[k, 0])
            if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
                dxs_p = dxs_m = float(dx_vec[0]) * np.ones(num_x)
            else:
                dxs_m = np.concatenate(([dx_vec[0]], dx_vec))       # h_minus per node
                dxs_p = np.concatenate((dx_vec, [dx_vec[-1]]))      # h_plus per node
            if mu_k >= 0.0:
                l[k, :] = 0.0
                u[k, :] = mu_k / dxs_p
                c[k, :] = -mu_k / dxs_p - r_k
            else:
                l[k, :] = -mu_k / dxs_m
                u[k, :] = 0.0
                c[k, :] = mu_k / dxs_m - r_k
        ...
        result = StepCoefficients(
            lcu_sets=lcu_sets,
            set_index=np.asarray(set_index, dtype=int).reshape(n_steps),
            n_unique=n_unique,
            zero_diffusion_sets=frozenset(np.nonzero(sig_zero)[0].tolist()),
        )
```

(Match the uniform/non-uniform handling to how the surrounding block builds
`dx_vec` slices — read lines 1119-1135 before editing; boundary rows keep
the existing copy-from-neighbor convention at indices 0 and −1.)

(3c) Sweep site (~line 1250) — force θ=1 on zero-diffusion steps:

```python
            theta = float(theta_by_step[j])
            if step_coeffs is not None:
                k = int(step_coeffs.set_index[j])
                if k in step_coeffs.zero_diffusion_sets:
                    theta = 1.0   # upwind advection is monotone only fully implicit (spec §4.5)
                l, _c_j, u = step_coeffs.lcu_sets[k]
                ...
```

- [ ] **Step 4: Run tests + PDE regression**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_pde_zero_diffusion_step.py -n0 -v`
Then: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_pde_term_coefficients.py test/test_pde_convergence_gate.py -n0 -q` (plus any snowball PDE goldens under `test/` matching `-k "snowball and pde"`).
Expected: all PASS — with no zero-vol step, `zero_diffusion_sets` is empty and behavior is bitwise-identical.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/base_pde_solver.py test/test_pde_zero_diffusion_step.py test/helpers_pde_zero_diffusion.py
git commit -m "feat(clock): upwind + fully-implicit zero-diffusion PDE steps"
```

### Task 6: MC zero-vol step test

**Files:**
- Test: `test/test_mc_zero_vol_step.py` (no library change expected; the test proves it)

**Interfaces:**
- Consumes: `quantark/asset/equity/engine/mc/term_inputs.py` (builds `TermCoefficients` and passes `vol=tc.step_vols` to path generation); the snowball MC engine construction pattern from `test/` (grep `SnowballMCEngine(` for the smallest fixture).

- [ ] **Step 1: Write the test**

```python
# test/test_mc_zero_vol_step.py
"""MC with sigma_step == 0 on holiday steps: drift-only advance, finite PV."""
from datetime import datetime

import numpy as np

from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
# env + engine imports: copy the smallest SnowballMCEngine fixture from an
# existing test (grep "SnowballMCEngine(" under test/), swapping only the
# vol_surface for the wrapped one below.

ANCHOR = datetime(2026, 2, 9)


def test_snowball_mc_finite_over_cny_with_trading_clock_surface():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 6, 9))
    surface = TradingClockVolSurface(FlatVolSurface(0.20), m)
    # build env(valuation=ANCHOR, spot=100, r=0.02, q=0.01, vol_surface=surface)
    # and a 6-month monthly-KO snowball whose obs times are ACT/365 fractions
    # of real trading dates spanning CNY (compute with cal, as in the Task 7
    # parity test); price with SnowballMCEngine(MCParams(num_paths=20000,
    # seed=7)).
    price = ...  # engine.price(product, env)
    assert np.isfinite(price)


def test_european_mc_zero_vol_matches_deterministic_forward():
    """All-holiday horizon control: sigma==0 for every step -> the terminal
    spot is the deterministic forward, so a vanilla call prices to
    DF * max(F - K, 0) exactly (up to float noise, no MC error)."""
    # Use EuropeanMCEngine with a FlatVolSurface(1e-12)? NO — build the same
    # wrapped surface with an anchor placed on the FIRST DAY of the CNY
    # holiday block and maturity inside the block (tau_td == 0 throughout).
    # Then price = DF(T) * max(S*exp((r-q)T) - K, 0) with zero variance.
    ...
```

Complete both bodies from existing fixtures (the engines and env builders
are used in dozens of tests — copy the smallest, do not invent new
scaffolding). If path generation divides by σ anywhere, this test surfaces
it — fix at the division site with an exact `is_zero` branch (drift-only
step), never a floor, and add that file to the commit.

- [ ] **Step 2: Run**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_zero_vol_step.py -n0 -v`
Expected: PASS with no library change (GBM log-step `exp((r−q−½σ²)dt + σ√dt·Z)` is well-defined at σ=0). If it fails, fix per Step 1's note.

- [ ] **Step 3: Commit**

```bash
git add test/test_mc_zero_vol_step.py
git commit -m "test(clock): MC zero-vol holiday steps are drift-only and finite"
```

### Task 7: Phase-1 validation bundle — European parity, node marginals, contract docs

**Files:**
- Test: `test/test_trading_clock_parity.py`
- Modify: `quantark/asset/equity/CLAUDE.md` (input-contract bullet)

**Interfaces:**
- Consumes: everything from Tasks 1-3; `BlackScholesEngine`, `EuropeanVanillaOption`, `PricingEnvironment`, `FlatRateCurve`, `ContinuousDividendYield`, `SpotQuote` (construction patterns: `test/test_european_option.py`).

- [ ] **Step 1: Write the tests**

```python
# test/test_trading_clock_parity.py
"""Spec §7.2 European parity + §7.5 node marginals on the calendar axis."""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.param.vol.trading_clock_surface import TradingClockVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.enum import OptionType

ANCHOR = datetime(2026, 2, 9)
R, Q, SIGMA_TD, SPOT, K = 0.02, 0.01, 0.20, 100.0, 100.0


def _env_and_map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 6, 9))
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(R), valuation_date=ANCHOR,
        spot_quote=SpotQuote(SPOT),
        vol_surface=TradingClockVolSurface(FlatVolSurface(SIGMA_TD), m),
        div_yield=ContinuousDividendYield(Q),
    )
    return env, m, cal


def _bs_with_total_variance(S, K, df, carry_df, w):
    """Closed-form BS in (DF, total-variance) form — the parity oracle."""
    if w <= 0.0:
        f = S * carry_df / df
        return df * max(f - K, 0.0)
    sw = math.sqrt(w)
    f = S * carry_df / df
    d1 = (math.log(f / K) + 0.5 * w) / sw
    return df * (f * stats.norm.cdf(d1) - K * stats.norm.cdf(d1 - sw))


def test_european_parity_across_cny():
    env, m, cal = _env_and_map()
    # expiry = a trading date ~3 months out (past CNY)
    d = ANCHOR + timedelta(days=90)
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    tau_cal = (d - ANCHOR).days / 365.0
    tau_td = m.to_trading(tau_cal)
    opt = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=tau_cal)
    price = BlackScholesEngine().price(opt, env)
    oracle = _bs_with_total_variance(
        SPOT, K, math.exp(-R * tau_cal), math.exp(-Q * tau_cal),
        SIGMA_TD**2 * tau_td,
    )
    assert price == pytest.approx(oracle, rel=1e-12)


def test_node_marginals_variance_is_trading_and_carry_is_calendar():
    """Spec §7.5 via TermCoefficients: accumulated step variance to each node
    equals w_td(node); accumulated (r - q)·dt equals the calendar carry."""
    from quantark.priceenv.term_sampling import TermCoefficients
    env, m, cal = _env_and_map()
    t = np.linspace(0.0, 60.0 / 365.0, 60 * 4 + 1)
    tc = TermCoefficients.from_env(env, t, ref_strike=K)
    w_acc = np.cumsum(tc.step_vols**2 * np.diff(t))
    w_expected = SIGMA_TD**2 * np.asarray(m.to_trading(t[1:]))
    np.testing.assert_allclose(w_acc, w_expected, rtol=0, atol=1e-14)
    carry_acc = np.cumsum((tc.fwd_rates - tc.fwd_carry) * np.diff(t))
    np.testing.assert_allclose(carry_acc, (R - Q) * t[1:], rtol=0, atol=1e-13)
```

- [ ] **Step 2: Run**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_parity.py -n0 -v`
Expected: PASS (Tasks 1-3 make these identities hold; a failure here is a real bug in them — debug, don't loosen tolerances).

- [ ] **Step 3: Add the contract line to `quantark/asset/equity/CLAUDE.md`**

Append to Key Conventions:

```markdown
- **Trading-clock vol**: `TradingClockVolSurface` (spec
  `docs/superpowers/specs/2026-09-01-trading-clock-vol-design.md`) wraps
  trading-time-quoted surfaces for calendar-axis engines. Contract: engine
  times must be ACT/365 calendar fractions of real dates, and the map's
  `anchor_date` must equal the env's `valuation_date`. Holiday intervals
  legitimately carry sigma_step == 0 (exact); QUAD/PDE handle them via
  deterministic-shift / upwind theta=1 branches.
```

- [ ] **Step 4: Full-suite sanity + commit**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/ -q -x --ignore=test/mo_volmodels` (full run; use default parallelism here).
Then `git checkout -- example/` if any mo sample files churned.

```bash
git add test/test_trading_clock_parity.py quantark/asset/equity/CLAUDE.md
git commit -m "test(clock): European parity + node-marginal invariants; document the contract"
```

---

## Phase 2 — trading axis (opt-in) + hardening

### Task 8: `TradingClockRateCurve` + `TradingClockDividendYield`

**Files:**
- Create: `quantark/param/rrf/trading_clock_curve.py`
- Create: `quantark/param/div/trading_clock_yield.py`
- Modify: `quantark/param/rrf/__init__.py`, `quantark/param/div/__init__.py` (exports)
- Test: `test/test_trading_clock_curves.py`

**Interfaces:**
- Consumes: `BusinessTimeMap.to_calendar` (Task 1); `RateCurve` ABC (`get_rate`, `get_discount_factor`, inherited `get_forward_rate`); `DividendYield` ABC (`get_yield`).
- Produces: `TradingClockRateCurve(inner: RateCurve, time_map: BusinessTimeMap)`, `TradingClockDividendYield(inner: DividendYield, time_map: BusinessTimeMap)`; both expose `.clock` (= `time_map.clock`) for Task 9's validation.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_trading_clock_curves.py
"""Trading-axis cash wrappers (spec §4.3, review iter-1 P1 + iter-2 P2a)."""
from datetime import datetime, timedelta

import math
import numpy as np
import pytest

from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.div.trading_clock_yield import TradingClockDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock

ANCHOR = datetime(2026, 2, 9)
R, Q = 0.02, 0.01


def _map():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    return BusinessTimeMap(TradingClock(cal, 244), ANCHOR, datetime(2027, 6, 9)), cal


def test_df_invariance_at_every_trading_knot():
    """Spec §7.3 / D5: DF_td(u_k) equals DF_cal(c_k) at every trading date."""
    m, cal = _map()
    curve = TradingClockRateCurve(inner=FlatRateCurve(R), time_map=m)
    k, d = 0, ANCHOR
    while d < ANCHOR + timedelta(days=200):
        if cal.is_business_day(d):
            k += 1
            u = k / 244.0
            c = (d + timedelta(days=1) - ANCHOR).days / 365.0  # end-of-day knot
            assert curve.get_discount_factor(u) == pytest.approx(
                math.exp(-R * c), rel=1e-15
            )
        d += timedelta(days=1)


def test_df_is_continuous_in_trading_time():
    """Review iter-2 P2a: no holiday-carry jump between adjacent substeps."""
    m, _ = _map()
    curve = TradingClockRateCurve(inner=FlatRateCurve(R), time_map=m)
    us = np.linspace(1e-6, 200.0 / 244.0, 40001)
    dfs = np.array([curve.get_discount_factor(float(u)) for u in us])
    rel_jumps = np.abs(np.diff(dfs)) / dfs[:-1]
    assert float(np.max(rel_jumps)) < R * (15.0 / 365.0) / 40  # spread, not spiked


def test_dividend_wrapper_preserves_cumulative_yield():
    """Review iter-1 P1: q_td(u)*u == q_cal(c)*c, NOT q_cal(c)*u."""
    m, _ = _map()
    y = TradingClockDividendYield(inner=ContinuousDividendYield(Q), time_map=m)
    for u in (5 / 244.0, 30 / 244.0, 130 / 244.0):
        c = m.to_calendar(u)
        assert y.get_yield(u) * u == pytest.approx(Q * c, rel=1e-14)


def test_forward_rates_telescope_exactly():
    m, _ = _map()
    curve = TradingClockRateCurve(inner=FlatRateCurve(R), time_map=m)
    u = np.linspace(1 / 244.0, 100 / 244.0, 100)
    dfs = np.array([curve.get_discount_factor(float(x)) for x in u])
    total = -math.log(dfs[-1] / dfs[0])
    parts = -np.log(dfs[1:] / dfs[:-1])
    assert float(np.sum(parts)) == pytest.approx(total, rel=1e-13)
```

- [ ] **Step 2: Run to verify failure** — ImportError expected.

- [ ] **Step 3: Implement**

```python
# quantark/param/rrf/trading_clock_curve.py
"""Calendar-quoted curve re-expressed in trading time (spec §4.3).

DF_td(u) = DF_cal(to_calendar(u)). to_calendar is the CONTINUOUS
knot-to-knot map, so a holiday's carry spreads across its adjacent trading
tick instead of appearing as a jump (review iter-2 P2a). DF ratios
telescope exactly; the D5 invariant (same DF for the same date) holds at
every trading-date knot.
"""
from __future__ import annotations

import math

from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


class TradingClockRateCurve(RateCurve):
    def __init__(self, inner: RateCurve, time_map: BusinessTimeMap) -> None:
        if not isinstance(inner, RateCurve):
            raise ValidationError("inner must be a RateCurve")
        self.inner = inner
        self.time_map = time_map
        self.clock = time_map.clock

    def get_discount_factor(self, time_to_maturity: float) -> float:
        if time_to_maturity < 0:
            raise ValidationError(
                f"Time to maturity must be non-negative, got {time_to_maturity}"
            )
        if time_to_maturity == 0.0:
            return 1.0
        c = float(self.time_map.to_calendar(float(time_to_maturity)))
        return self.inner.get_discount_factor(c)

    def get_rate(self, time_to_maturity: float) -> float:
        if time_to_maturity <= 0.0:
            raise ValidationError(
                f"Time to maturity must be positive, got {time_to_maturity}"
            )
        df = self.get_discount_factor(time_to_maturity)
        return -math.log(df) / float(time_to_maturity)
```

```python
# quantark/param/div/trading_clock_yield.py
"""Cumulative-yield-preserving dividend wrapper (spec §4.3, review iter-1 P1).

forward_carry_on_grid differences q(t)*t, so an argument-only remap would
accrue q_cal(c)*u instead of the true q_cal(c)*c. Define
q_td(u) = q_cal(c)*c/u so that q_td(u)*u == q_cal(c)*c for u > 0; 0 at u=0
(the forward_carry_on_grid convention at t=0).
"""
from __future__ import annotations

from quantark.param.div.dividend_yield import DividendYield
from quantark.util.calendar.trading_clock import BusinessTimeMap
from quantark.util.exceptions import ValidationError


class TradingClockDividendYield(DividendYield):
    def __init__(self, inner: DividendYield, time_map: BusinessTimeMap) -> None:
        if not isinstance(inner, DividendYield):
            raise ValidationError("inner must be a DividendYield")
        self.inner = inner
        self.time_map = time_map
        self.clock = time_map.clock

    def get_yield(self, time_to_maturity: float) -> float:
        u = float(time_to_maturity)
        if u <= 0.0:
            return 0.0
        c = float(self.time_map.to_calendar(u))
        return self.inner.get_yield(c) * c / u
```

- [ ] **Step 4: Exports + run**

Add both to their packages' `__init__.py`. Run:
`PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_curves.py -n0 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/param/rrf/trading_clock_curve.py quantark/param/div/trading_clock_yield.py quantark/param/rrf/__init__.py quantark/param/div/__init__.py test/test_trading_clock_curves.py
git commit -m "feat(clock): trading-axis cash wrappers — continuous carry, cumulative-yield exact"
```

### Task 9: `BUSINESS_DAYS` hardening + one-clock validation

**Files:**
- Modify: `quantark/util/calendar/day_counter.py:102-125`
- Create: `quantark/priceenv/clock_validation.py`
- Modify: `quantark/priceenv/pricing_environment.py` (`__post_init__` hook)
- Modify: `test/test_european_option.py:440-460` (the silent-identity victim)
- Test: `test/test_business_days_hardening.py`

**Interfaces:**
- Consumes: wrapper `.clock` attributes (Task 8); `calculate_year_fraction` (day_counter.py:44).
- Produces: `validate_trading_clock_configuration(env) -> None` in `quantark/priceenv/clock_validation.py`, called from `PricingEnvironment.__post_init__` when `day_count_convention == DayCountConvention.BUSINESS_DAYS`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_business_days_hardening.py
"""BUSINESS_DAYS without a calendar raises; one clock per env (spec §4.3/4.4)."""
from datetime import datetime

import pytest

from quantark.param import SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.rrf.trading_clock_curve import TradingClockRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.calendar.trading_clock import BusinessTimeMap, TradingClock
from quantark.util.exceptions import ValidationError


def test_business_days_without_calendar_raises():
    """The old fallback was EXACTLY act/365 (num_days*(D/365)/D) — a silent lie."""
    with pytest.raises(ValidationError, match="calendar"):
        calculate_year_fraction(
            datetime(2026, 1, 1), datetime(2026, 7, 1),
            DayCountConvention.BUSINESS_DAYS, 252,
        )


def test_env_clock_mismatch_raises():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2027, 2, 9))
    wrapped = TradingClockRateCurve(FlatRateCurve(0.02), m)
    with pytest.raises(ValidationError, match="clock"):
        PricingEnvironment(
            rate_curve=wrapped, valuation_date=datetime(2026, 2, 9),
            spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
            day_count_convention=DayCountConvention.BUSINESS_DAYS,
            bus_days_in_year=252,          # != wrapper's 244 -> must raise
            calendar=cal,
        )


def test_env_clock_match_passes():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    m = BusinessTimeMap(TradingClock(cal, 244), datetime(2026, 2, 9), datetime(2027, 2, 9))
    wrapped = TradingClockRateCurve(FlatRateCurve(0.02), m)
    PricingEnvironment(
        rate_curve=wrapped, valuation_date=datetime(2026, 2, 9),
        spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=244, calendar=cal,
    )
```

- [ ] **Step 2: Run to verify failure** — the first test fails today because the fallback silently returns `num_days/365`.

- [ ] **Step 3: Implement**

(3a) `day_counter.py` — replace the no-calendar fallback (lines 122-125):

```python
        # A calendar is REQUIRED: the old fallback num_days*(D/365)/D
        # cancels to exactly num_days/365 for every D — silently identical
        # to ACT/365, never an approximation of business days.
        raise ValidationError(
            "BUSINESS_DAYS day count requires a business-day calendar; "
            "pass calendar=... (the no-calendar fallback was exactly ACT/365)"
        )
```

(3b) `quantark/priceenv/clock_validation.py`:

```python
"""One trading clock per environment (spec §4.3, review iter-2 P2b)."""
from quantark.util.exceptions import ValidationError


def validate_trading_clock_configuration(env) -> None:
    """Every wrapped curve's TradingClock must match the env resolver."""
    for name in ("rate_curve", "div_yield", "vol_surface"):
        obj = getattr(env, name, None)
        clock = getattr(obj, "clock", None)
        if clock is None:
            continue
        if clock.days_per_year != env.bus_days_in_year:
            raise ValidationError(
                f"{name} carries a TradingClock with days_per_year="
                f"{clock.days_per_year} but the environment resolves times "
                f"with bus_days_in_year={env.bus_days_in_year}; one clock "
                "per configuration"
            )
        if env.calendar is not None and clock.calendar is not env.calendar:
            raise ValidationError(
                f"{name} carries a TradingClock whose calendar is not the "
                "environment's calendar object; one clock per configuration"
            )
```

(3c) `pricing_environment.py` `__post_init__` — append:

```python
        if self.day_count_convention == DayCountConvention.BUSINESS_DAYS:
            from quantark.priceenv.clock_validation import (
                validate_trading_clock_configuration,
            )
            validate_trading_clock_configuration(self)
```

(3d) Fix `test/test_european_option.py:443-451`: add
`calendar=create_calendar(CalendarType.CHINA_SSE)` (import at top of file)
to that `PricingEnvironment(...)` and update the test's expected T to the
genuine business-day fraction (compute it in the test from
`calendar.count_business_days(valuation_date, exercise_date, include_start=False, include_end=True) / 252`). Read the surrounding assertions first; keep the test's intent (business-day pricing), fix its arithmetic.

- [ ] **Step 4: Run + regression**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_business_days_hardening.py test/test_european_option.py test/test_greeks_theta_schedule.py -n0 -v`
Also grep for other no-calendar BUSINESS_DAYS callers that now raise:
`grep -rn "BUSINESS_DAYS" test/ example/ quantark/ | grep -v calendar=` and fix each the same way (pass the calendar the test/example already implies).
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/util/calendar/day_counter.py quantark/priceenv/clock_validation.py quantark/priceenv/pricing_environment.py test/test_business_days_hardening.py test/test_european_option.py
git commit -m "fix(clock): BUSINESS_DAYS requires a calendar; one validated clock per env"
```

### Task 10: Axis-equivalence flagship test

**Files:**
- Test: `test/test_trading_clock_axis_equivalence.py`

**Interfaces:**
- Consumes: everything from Tasks 1-9. Product/env/engine construction patterns: `test/test_pde_convergence_gate.py` (date-derived snowball), `SnowballPDESolver` / `SnowballQuadEngine` / `SnowballMCEngine` fixtures under `test/`.

- [ ] **Step 1: Write the test**

```python
# test/test_trading_clock_axis_equivalence.py
"""Spec §7.4: the same CNY-straddling product priced on both axes agrees.

Calendar axis: ACT/365 times of real trading dates + TradingClockVolSurface.
Trading axis:  n_td/244 times of the SAME dates + native trading surface +
               TradingClockRateCurve / TradingClockDividendYield.
Both integrate the same (Dtau_cal, Dn_td/244) per date-step, so prices agree
to discretization tolerance. The European leg is exact (analytical), the
snowball legs are the engine gates.
"""
from datetime import datetime, timedelta

import math
import pytest

# imports per fixtures; see step notes

ANCHOR = datetime(2026, 1, 15)      # ~1 month before CNY 2026
R, Q, SIGMA_TD = 0.02, 0.01, 0.20


def _dates_and_times(cal, months=6):
    """Monthly KO dates snapped to next trading day; both time coordinates."""
    # build ko_dates as in test_pde_convergence_gate (add_months + next
    # trading day via cal), then:
    #   t_cal[i] = (d_i - ANCHOR).days / 365.0
    #   t_td[i]  = count_business_days(ANCHOR, d_i, include_start=False,
    #              include_end=True) / 244.0
    ...


def test_european_axis_equivalence_exact():
    """Analytical BS both ways — agreement to 1e-12 relative (no grids)."""
    # calendar axis: env A (FlatRateCurve, ContinuousDividendYield, wrapped
    #   vol surface), option maturity = t_cal[-1]
    # trading axis: env B (TradingClockRateCurve, TradingClockDividendYield,
    #   FlatVolSurface(SIGMA_TD) native), option maturity = t_td[-1]
    # assert price_A == pytest.approx(price_B, rel=1e-12)
    ...


def test_snowball_axis_equivalence_pde_quad_mc():
    """Each engine family prices the same term sheet both ways.

    Tolerances: PDE and QUAD 1e-3 relative (discretization differs between
    the two axes' grids); MC uses a fixed seed and 3 standard errors."""
    ...
```

Complete the three bodies from the named fixtures. The tolerance choices are
the spec's "discretization tolerance": the two axes discretize differently
(the SAME dates, different coordinates), so agreement is engine-level, not
bitwise. If PDE disagrees beyond 1e-3, first check the trading-axis grid is
being fed `t_td` (not `t_cal`) for observation times AND that the env's
`bus_days_in_year=244` matches the map — the Task 9 validator should have
caught a mismatch.

- [ ] **Step 2: Run**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_trading_clock_axis_equivalence.py -n0 -v`
Expected: PASS. This is the design's flagship gate — investigate any failure fully (systematic-debugging skill) before touching tolerances.

- [ ] **Step 3: Commit**

```bash
git add test/test_trading_clock_axis_equivalence.py
git commit -m "test(clock): axis-equivalence flagship — both clocks price the same dates alike"
```

### Task 11: Docs and pointers

**Files:**
- Create: `docs/trading-clock.md` (unit-conversion table from spec §5 + usage examples for both axes)
- Modify: `quantark/asset/equity/engine/pde/grid/config.py` (docstring only: `day_count` is a step-density constant in axis units, not a day-count convention)
- Modify: `quantark/priceenv/pricing_environment.py` (docstring: float-based products bypass `day_count_convention`; `get_maturity`/`resolve_time` honor it)
- Modify: `example/phoenix_external_case_compare.py` (top-of-file comment pointing at `TradingClockRateCurve`/`TradingClockDividendYield` as the correct trading-axis construction)

- [ ] **Step 1: Write `docs/trading-clock.md`** — contents: the §1 SDE, the two-axis table (which wrapper on which side), the §5 unit-conversion table verbatim from the spec, one worked example per axis (copy the constructions from Tasks 7 and 10 tests), and the anchor/valuation-date contract.

- [ ] **Step 2: Apply the three docstring/comment edits.** Each is 2-5 lines; cite the spec path in each.

- [ ] **Step 3: Full suite + commit**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/ -q --ignore=test/mo_volmodels`
Then `git checkout -- example/` for mo sample-file churn if any.

```bash
git add -f docs/trading-clock.md
git add quantark/asset/equity/engine/pde/grid/config.py quantark/priceenv/pricing_environment.py example/phoenix_external_case_compare.py
git commit -m "docs(clock): usage guide, unit conversions, and convention docstrings"
```

---

## Self-review record

- Spec coverage: §4.1→T1, §4.2→T2+T3, §4.3→T8+T9, §4.4→T9, §4.5→T4+T5+T6, §4.6→no code (relationship note lives in the spec; T7 CLAUDE.md line names the contract), §5→T11, §6→T1/T2/T8/T9 error tests, §7.1→T1, §7.2→T7, §7.3→T8, §7.4→T10, §7.5→T7, §7.6→T4/T5/T6, §7.7→T3, §8 phasing→task order, §9→T11.
- Types: `TradingClock(calendar, days_per_year)`, `BusinessTimeMap(clock, anchor_date, horizon_date, extend_weekdays)`, `to_trading`/`to_calendar`/`initial_slope`, `total_variance(strike, t, spot)`, `step_vols_on_grid(..., total_variance=None)`, `zero_diffusion_sets` — names checked consistent across Tasks 1-10.
- Known executor-completion points (deliberate, marked in-task): QuadCoreInputs field list (T4, from quad_adapters), MC fixtures (T6, T10), snowball fixture bodies (T10) — each names the exact file to copy from; everything else is fully specified.
