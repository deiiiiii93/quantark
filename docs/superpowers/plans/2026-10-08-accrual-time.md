# Accrual Time Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every autocallable record carries its own immutable accrual period, filled once at construction, and every
product, engine and intraday path reads accrual through one resolver, so moving observation timing never moves cash.

**Architecture:** A new module `quantark/asset/equity/product/option/accrual.py` holds the data model
(`AccrualKind`, `AccrualBasis`, `KnockInAnchor`, `AccrualPeriod`) and the resolver (`ko_accrual_factor`,
`coupon_period_fraction`, `contract_tenor`). A second module, `accrual_build.py`, detaches a product's schedules at
construction and fills each record's `accrual` and the product's terminal periods (`_terminal_accrual`). The
products' existing accrual methods (`resolve_ko_observations`, `compute_ko_accrual_factor`, `coupon_period_fraction`,
`get_contract_tenor`, `_resolve_pre/post_contract_tenor`) become thin delegates, so most engines need no change. Monte
Carlo's REBASED branch, QUAD V2's Phoenix adapter, the intraday Phoenix fractions and the intraday twin are rewired to
the resolver. The lifecycle state gains a persisted `knock_in_anchor`.

**Tech Stack:** Python 3.10–3.13, NumPy, pytest (`-n auto --dist worksteal`; full suite with `-n 6` under the RAM guard).

**Spec:** `docs/superpowers/specs/2026-10-08-accrual-time-design.md` (approved 2026-10-08 after review rounds 1–3).

## Global Constraints

- Branch `accrual-time`, cut from `main` (e3021668) with the four spec commits cherry-picked; built in the main checkout.
- Numeric accrual is the quoted fraction (R1): a numeric record's period end is `accrued_offset + observation_time` **at construction**, never recomputed.
- `AccrualEndType ∈ {OBSERVATION, SETTLEMENT}`, default `OBSERVATION` (R2): existing contracts do not move.
- A dated contract whose annualized cash needs an accrual start but has no `initial_date` is refused at construction (R5). A contract with no annualized leg gets `NONE` periods.
- `BUSINESS_DAYS` on an annualized cumulative leg needs `AccrualConfig.accrual_calendar` and `accrual_bus_days_in_year`, or construction refuses with "BUSINESS_DAYS accrual needs its calendar and denominator on the contract" (§4.7).
- No accrual reads the pricing environment (§4.7). `env` stays in signatures.
- REBASED offsets are quoted fractions, on a dated anchor too; nothing converts them to days (§4.3, round 3 P1-1).
- Zero-length periods (`end == start` on the coupon leg, `end == origin` on the cumulative leg) give exactly `0.0` without calling a year-fraction helper.
- Dated contracts with `initial_date` and unaged numeric contracts are bit-identical (§7). Aged numeric moves at ulp level; numeric-with-`initial_date` moves to the quoted fraction. Every moved golden row is listed with its magnitude.
- Never invent fallback semantics: a term the code cannot resolve raises `ValidationError` (memory: no stupid fallbacks).
- Canonical `quantark.*` imports only; `quantark.util.numerical` for comparisons; `ValidationError` for refusals.
- New docs files need `git add -f`; commit by pathspec; every commit ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Full suite at most `-n 6`, RAM guard running; afterwards `git checkout -- ` the two mo sample files if a mo test rewrote them.

### Scope rulings carried from the spec (decided while planning)

- **P-1. §5.2's conversion (`with_absolute_schedule_2`) and the KO-reset tracker do not exist on `main`.** They live on `handoff-small-fixes` and adopt this work after the rebase (spec §7). Here, the "converted copy" controls are pinned on the resolver with an **embedded** anchor, the period a conversion produces. The MC route is pinned end to end. A carried REBASED knock-in stays refused by MC on `main` (it has no knock-in time), as today.
- **P-2. `NONE` is the kind with no endpoints.** Overrides may accompany it: the spec's "`cumulative_fraction` only" row is `NONE` plus `cumulative_fraction`. A bare `NONE` (no overrides) is the spec's non-accruing row.
- **P-3. The coupon leg keeps `calculate_day_count_fraction`** (today's Phoenix call, which never read the environment), so §4.7's refusal applies to the cumulative leg only, the one that called `calculate_year_fraction` with the environment's calendar.
- **P-4. Terminal periods live on the product as `_terminal_accrual: Dict[str, AccrualPeriod]`** (keys `contract`, `pre`, `post`). It is a plain attribute, not a dataclass field: `deepcopy` and pickle carry it, and a rebuilt product (`dataclasses.replace`) derives it again. Like a record's period, an existing `_terminal_accrual` is kept on re-validation.
- **P-5. Stale-term guard.** A period copied onto a product whose basis terms differ (a changed `annualization_day_count`, coupon day count, or accrual calendar) is refused at construction. A `DATE` read whose period basis differs from the product's current day count is refused at resolution. Without the guard, a contract term changed after construction would leave the cash silently stale.
- **P-6. V0's fixed-rebate branch stops resolving the tenor when the rebate is not annualized.** Today it resolves it and throws it away, which would now raise on a `NONE` terminal.
- **P-7. `get_tenor`'s dated path gains the basis's calendar** through the terminal period. A `BUSINESS_DAYS` annualized contract now carries one (§4.7). Spec item 8 is otherwise untouched: base `get_tenor` is unchanged for every other product.

## Review Focus

1. A product built, then changed in place (`p.coupon_config = replace(...)`, `p.annualization_day_count = ...`): a reasonable user expects the new terms or an error, never the old cash. Pinned in Task 4 (`test_changed_coupon_day_count_after_construction_raises`).
2. Two products built from one shared `ObservationSchedule`, or a schedule edited after the product was built: each product's cash is its own. Pinned in Task 3 (`test_shared_schedule_is_detached`).
3. A contract aged by `product_for_pricing` and then rebuilt with `dataclasses.replace`: its records keep their periods and its numeric terminal differs at most at ulp level. Pinned in Task 4 (`test_replace_after_ageing_keeps_record_cash`).
4. A Phoenix whose valuation date equals an observation date, with QUAD V2 `event_phase="after"`: the adapter's active set and its fractions stay aligned by source index. Pinned in Task 6 (`test_quad_v2_fractions_follow_source_index_after_phase`).
5. A REBASED contract with a non-annualized knock-out and a `NONE` knock-in period (dated, no `initial_date`): MC pays the flat rate and never asks for an anchor. Pinned in Task 5 (`test_unannualized_rebased_needs_no_anchor`).

---

## File Structure

| File | Responsibility |
|---|---|
| `quantark/asset/equity/product/option/accrual.py` (new) | Data model + the only accrual math (resolver) |
| `quantark/asset/equity/product/option/accrual_build.py` (new) | Construction: detach schedules, fill record periods and terminals, refusals |
| `quantark/util/enum/option_enums.py`, `quantark/util/enum/__init__.py` | `AccrualEndType` |
| `quantark/asset/equity/product/option/snowball_config.py` | `AccrualConfig`: `accrual_end`, `post_accrual_factors`, `accrual_calendar`, `accrual_bus_days_in_year`; `accrual_dates` removed |
| `quantark/asset/equity/product/option/observation_schedule.py` | `ObservationRecord.accrual` |
| `snowball_option.py`, `phoenix_option.py`, `ko_reset_snowball_option.py` | Call the builder; accrual methods delegate to the resolver |
| `quantark/asset/equity/engine/mc/snowball_mc_engine.py` | REBASED per-path anchors and V1 tenor |
| `quantark/asset/equity/engine/quad/v2/adapters.py` | Phoenix fractions from the resolver |
| `quantark/intraday/events.py`, `quantark/intraday/twin.py` | Fractions from the resolver; twin copies periods |
| `quantark/asset/equity/lifecycle/state.py`, `autocallable.py` | Persisted `knock_in_anchor` |
| `test/accrual_time/` (new package) | All new tests |

---

### Task 1: The accrual data model and its math

**Files:**
- Create: `quantark/asset/equity/product/option/accrual.py`
- Create: `test/accrual_time/__init__.py` (empty), `test/accrual_time/test_period.py`

**Interfaces:**
- Produces: `AccrualKind {TIME, DATE, NONE}`; `AccrualBasis(day_count, calendar=None, bus_days_in_year=None)` with `.cumulative(start, end) -> float`, `.period(start, end) -> float`, `.is_complete -> bool`; `KnockInAnchor(source_index, fixing, origin, basis)` with `.accrual() -> float`; `AccrualPeriod(kind, origin=None, start=None, end=None, basis=None, period_basis=None, cumulative_fraction=None, period_fraction=None, relative=False, anchor=None)`; `cumulative_accrual(period, *, knock_in=None) -> float`; `coupon_accrual(period) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
"""AccrualPeriod: the representations, the refusals and the math (spec §4.1, §5)."""
from datetime import datetime

import pytest

from quantark.asset.equity.product.option.accrual import (
    AccrualBasis, AccrualKind as K, AccrualPeriod, KnockInAnchor, coupon_accrual, cumulative_accrual,
)
from quantark.util.calendar import CalendarType, create_calendar, calculate_year_fraction
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.exceptions import ValidationError

A365 = AccrualBasis(DC.ACT_365)
T0, T1, T2 = datetime(2026, 1, 5), datetime(2026, 4, 6), datetime(2026, 7, 6)


def test_time_period_reads_the_quoted_fraction():
    p = AccrualPeriod(K.TIME, origin=0.0, start=1 / 12, end=2 / 12)
    assert cumulative_accrual(p) == 2 / 12
    assert coupon_accrual(p) == 2 / 12 - 1 / 12


def test_date_period_is_todays_year_fraction_call():
    p = AccrualPeriod(K.DATE, origin=T0, start=T1, end=T2, basis=A365)
    assert cumulative_accrual(p) == calculate_year_fraction(T0, T2, DC.ACT_365)
    assert coupon_accrual(p) == (T2 - T1).days / 365.0


def test_overrides_win_on_their_own_leg():
    p = AccrualPeriod(K.TIME, origin=0.0, start=0.0, end=0.25, cumulative_fraction=0.4, period_fraction=0.1)
    assert (cumulative_accrual(p), coupon_accrual(p)) == (0.4, 0.1)


def test_cumulative_fraction_only_serves_the_cumulative_leg():
    p = AccrualPeriod(K.NONE, cumulative_fraction=0.75)
    assert cumulative_accrual(p) == 0.75
    with pytest.raises(ValidationError, match="no accrual terms"):
        coupon_accrual(p)


def test_end_only_first_period_refuses_the_coupon_leg():
    p = AccrualPeriod(K.TIME, origin=0.0, end=0.3)
    assert cumulative_accrual(p) == 0.3
    with pytest.raises(ValidationError, match="period start unknown"):
        coupon_accrual(p)


def test_none_raises_when_asked():
    with pytest.raises(ValidationError, match="no accrual terms"):
        cumulative_accrual(AccrualPeriod(K.NONE))


@pytest.mark.parametrize("kw", [
    dict(kind=K.NONE, end=0.25),                                         # NONE carries no endpoints
    dict(kind=K.TIME, origin=0.0),                                       # no end
    dict(kind=K.TIME, origin=0.0, start=0.3, end=0.2),                   # end < start
    dict(kind=K.TIME, origin=0.5, end=0.2),                              # end < origin
    dict(kind=K.TIME, origin=0.0, end=T1),                               # mixed TIME / DATE
    dict(kind=K.DATE, origin=T0, end=T1),                                # DATE without basis
    dict(kind=K.TIME, start=0.0, end=0.25, relative=True, origin=0.0),   # relative has no origin
    dict(kind=K.DATE, start=T0, end=T1, relative=True, basis=A365),      # relative is TIME
    dict(kind=K.TIME, origin=0.0, end=0.25, cumulative_fraction=-0.1),   # negative override
])
def test_refused_representations(kw):
    with pytest.raises(ValidationError):
        AccrualPeriod(**kw)


def test_zero_length_periods_are_zero_without_the_helper():
    p = AccrualPeriod(K.DATE, origin=T1, start=T1, end=T1, basis=A365)
    assert cumulative_accrual(p) == 0.0 and coupon_accrual(p) == 0.0


def test_relative_reads_the_anchor_plus_the_quoted_offset():
    anchor = KnockInAnchor(source_index=0, fixing=0.5, origin=0.0, basis=A365)
    p = AccrualPeriod(K.TIME, start=0.0, end=0.25, relative=True)
    assert cumulative_accrual(p, knock_in=anchor) == 0.75
    assert cumulative_accrual(AccrualPeriod(K.TIME, start=0.0, end=0.25, relative=True, anchor=anchor)) == 0.75
    with pytest.raises(ValidationError, match="anchor"):
        cumulative_accrual(p)


def test_dated_anchor_plus_quoted_offset_is_not_91_days():
    """Review round 3, P1-1: offset 0.25 on a dated knock-in adds 0.25, not YF(91 days)."""
    anchor = KnockInAnchor(source_index=0, fixing=T1, origin=T0, basis=A365)
    p = AccrualPeriod(K.TIME, start=0.0, end=0.25, relative=True, anchor=anchor)
    assert cumulative_accrual(p) == calculate_year_fraction(T0, T1, DC.ACT_365) + 0.25
    assert cumulative_accrual(p) != calculate_year_fraction(T0, datetime(2026, 7, 6), DC.ACT_365)


def test_business_days_reads_its_own_calendar_and_denominator():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    basis = AccrualBasis(DC.BUSINESS_DAYS, calendar=cal, bus_days_in_year=244.0)
    p = AccrualPeriod(K.DATE, origin=T0, end=T1, basis=basis)
    assert cumulative_accrual(p) == calculate_year_fraction(T0, T1, DC.BUSINESS_DAYS, 244.0, calendar=cal)
    with pytest.raises(ValidationError, match="BUSINESS_DAYS accrual needs its calendar and denominator"):
        cumulative_accrual(AccrualPeriod(K.DATE, origin=T0, end=T1, basis=AccrualBasis(DC.BUSINESS_DAYS)))
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_period.py -q`
Expected: collection error, `ModuleNotFoundError: ... accrual`.

- [ ] **Step 3: Write `accrual.py`**

```python
"""Accrual time, separate from observation time (spec ``2026-10-08-accrual-time-design.md``).

An observation is a trading-day quantity: when the market is checked. Accrual is a calendar quantity in the contract's
interest basis: how much the contract owes. Each record carries its own immutable ``AccrualPeriod``, filled once at
construction (``accrual_build``), so ageing, snapping or rebasing a schedule moves observation timing and never cash.
The functions at the bottom of this module are the only place accrual is computed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Optional, Union

from quantark.util.calendar import calculate_day_count_fraction, calculate_year_fraction
from quantark.util.calendar.business_calendar import Calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.exceptions import ValidationError

Instant = Union[float, datetime]

BUSINESS_DAYS_TERMS = "BUSINESS_DAYS accrual needs its calendar and denominator on the contract"


class AccrualKind(Enum):
    """What a period's endpoints are: contract-clock years, calendar dates, or none at all."""

    TIME = "time"
    DATE = "date"
    NONE = "none"


@dataclass(frozen=True)
class AccrualBasis:
    """Every term a year fraction reads (spec §4.7): the day count, and for ``BUSINESS_DAYS`` its calendar and
    denominator, frozen with the contract so no accrual reads the pricing environment."""

    day_count: DayCountConvention
    calendar: Optional[Calendar] = None
    bus_days_in_year: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.day_count, DayCountConvention):
            raise ValidationError(f"accrual day count must be DayCountConvention, got {self.day_count!r}")
        if self.calendar is not None and not isinstance(self.calendar, Calendar):
            raise ValidationError(f"accrual calendar must be Calendar, got {type(self.calendar).__name__}")
        if self.bus_days_in_year is not None:
            value = float(self.bus_days_in_year)
            if not isfinite(value) or value <= 0.0:
                raise ValidationError(f"accrual bus_days_in_year must be positive, got {self.bus_days_in_year!r}")
            object.__setattr__(self, "bus_days_in_year", value)

    @property
    def is_complete(self) -> bool:
        """Whether every term this basis's cumulative fraction reads is present."""
        return self.day_count is not DayCountConvention.BUSINESS_DAYS or (
            self.calendar is not None and self.bus_days_in_year is not None)

    def same_terms(self, other: "AccrualBasis") -> bool:
        """Equal day count, denominator and calendar identity (a calendar's ``==`` is object identity)."""
        def ident(calendar):
            return None if calendar is None else calendar.identity()
        return (self.day_count is other.day_count and self.bus_days_in_year == other.bus_days_in_year
                and ident(self.calendar) == ident(other.calendar))

    def cumulative(self, start: datetime, end: datetime) -> float:
        """The cumulative leg's fraction: today's ``calculate_year_fraction`` call, with this basis's terms."""
        if not self.is_complete:
            raise ValidationError(BUSINESS_DAYS_TERMS)
        bus_days = 252 if self.bus_days_in_year is None else self.bus_days_in_year
        return float(calculate_year_fraction(start, end, self.day_count, bus_days, calendar=self.calendar))

    def period(self, start: datetime, end: datetime) -> float:
        """The coupon leg's fraction: today's Phoenix ``calculate_day_count_fraction`` call (plan ruling P-3)."""
        return float(calculate_day_count_fraction(start, end, self.day_count))


def _check_instant(kind: AccrualKind, value, name: str) -> None:
    if value is None:
        return
    if kind is AccrualKind.DATE:
        if not isinstance(value, datetime):
            raise ValidationError(f"{name} of a DATE accrual period must be a datetime, got {value!r}")
        return
    if isinstance(value, (bool, datetime)) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValidationError(f"{name} of a TIME accrual period must be a finite year fraction, got {value!r}")


def _check_fraction(value, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0:
        raise ValidationError(f"{name} must be a finite non-negative fraction, got {value!r}")


@dataclass(frozen=True)
class KnockInAnchor:
    """The knock-in a REBASED schedule hangs from (spec §5.1): resolved once, when the knock-in happens.

    ``fixing`` is the knock-in record's contractual fixing (its accrual ``end``), not the close it was observed on.
    """

    source_index: int
    fixing: Instant
    origin: Instant
    basis: AccrualBasis

    def __post_init__(self):
        kind = AccrualKind.DATE if isinstance(self.fixing, datetime) else AccrualKind.TIME
        _check_instant(kind, self.fixing, "knock-in fixing")
        _check_instant(kind, self.origin, "knock-in origin")
        if not isinstance(self.basis, AccrualBasis):
            raise ValidationError("a knock-in anchor needs its AccrualBasis")

    def accrual(self) -> float:
        """The contract's accrual from its origin to the knock-in fixing."""
        if self.fixing == self.origin:
            return 0.0
        if isinstance(self.fixing, datetime):
            return self.basis.cumulative(self.origin, self.fixing)
        return float(self.fixing) - float(self.origin)


@dataclass(frozen=True)
class AccrualPeriod:
    """One record's accrual, self-contained (spec §4.1).

    Representations (every other combination raises):
      * ``NONE``: no endpoints; with ``cumulative_fraction`` / ``period_fraction`` overrides, or none at all
        (a non-accruing record).
      * ``TIME`` / ``DATE`` with ``origin`` + ``end``, ``start`` optional (unknown).
      * ``relative`` (``TIME``): ``start`` + ``end`` are quoted offsets from a knock-in anchor, embedded or passed.
    """

    kind: AccrualKind
    origin: Optional[Instant] = None
    start: Optional[Instant] = None
    end: Optional[Instant] = None
    basis: Optional[AccrualBasis] = None
    period_basis: Optional[AccrualBasis] = None
    cumulative_fraction: Optional[float] = None
    period_fraction: Optional[float] = None
    relative: bool = False
    anchor: Optional[KnockInAnchor] = None

    def __post_init__(self):
        if not isinstance(self.kind, AccrualKind):
            raise ValidationError(f"accrual kind must be AccrualKind, got {self.kind!r}")
        _check_fraction(self.cumulative_fraction, "cumulative_fraction")
        _check_fraction(self.period_fraction, "period_fraction")
        for name in ("origin", "start", "end"):
            _check_instant(self.kind, getattr(self, name), name)
        if self.anchor is not None and not isinstance(self.anchor, KnockInAnchor):
            raise ValidationError("an accrual anchor must be a KnockInAnchor")
        if self.basis is not None and not isinstance(self.basis, AccrualBasis):
            raise ValidationError("an accrual basis must be an AccrualBasis")
        if self.period_basis is None and self.basis is not None:
            object.__setattr__(self, "period_basis", self.basis)
        if self.kind is AccrualKind.NONE:
            if any(v is not None for v in (self.origin, self.start, self.end)) or self.relative or self.anchor:
                raise ValidationError("a NONE accrual period carries no endpoints")
            return
        if self.relative:
            if self.kind is not AccrualKind.TIME or self.origin is not None:
                raise ValidationError("a relative accrual period is TIME offsets with no origin")
            if self.start is None or self.end is None:
                raise ValidationError("a relative accrual period needs start and end offsets")
        else:
            if self.anchor is not None:
                raise ValidationError("only a relative accrual period carries an anchor")
            if self.origin is None or self.end is None:
                raise ValidationError("an accrual period needs its origin and end")
            if self.end < self.origin:
                raise ValidationError(f"accrual end {self.end!r} is before its origin {self.origin!r}")
            if self.start is not None and self.start < self.origin:
                raise ValidationError(f"accrual start {self.start!r} is before its origin {self.origin!r}")
        if self.start is not None and self.end < self.start:
            raise ValidationError(f"accrual end {self.end!r} is before its start {self.start!r}")
        if self.kind is AccrualKind.DATE and self.basis is None:
            raise ValidationError("a DATE accrual period needs its basis")


def cumulative_accrual(period: AccrualPeriod, *, knock_in: Optional[KnockInAnchor] = None) -> float:
    """The cumulative leg (knock-out accrual, terminal tenor) of one period."""
    if period.cumulative_fraction is not None:
        return float(period.cumulative_fraction)
    if period.relative:
        anchor = period.anchor if period.anchor is not None else knock_in
        if anchor is None:
            raise ValidationError("a REBASED accrual period needs its knock-in anchor, embedded or passed")
        return anchor.accrual() + float(period.end)
    if period.kind is AccrualKind.NONE:
        raise ValidationError("this record has no accrual terms (a non-accruing contract)")
    if period.end == period.origin:
        return 0.0
    if period.kind is AccrualKind.DATE:
        return period.basis.cumulative(period.origin, period.end)
    return float(period.end) - float(period.origin)


def coupon_accrual(period: AccrualPeriod) -> float:
    """The coupon leg (Phoenix period fraction) of one period."""
    if period.period_fraction is not None:
        return float(period.period_fraction)
    if period.kind is AccrualKind.NONE:
        raise ValidationError("this record has no accrual terms (a non-accruing contract)")
    if period.start is None:
        raise ValidationError("coupon period start unknown: give the contract a fixed_coupon_year_fraction or "
                              "accrual_factors")
    if period.end == period.start:
        return 0.0
    if period.kind is AccrualKind.DATE:
        return period.period_basis.period(period.start, period.end)
    return float(period.end) - float(period.start)
```

Check `quantark.util.calendar` exports `calculate_day_count_fraction` (it is imported that way in `observation_schedule.py`).

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_period.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/product/option/accrual.py test/accrual_time/__init__.py test/accrual_time/test_period.py
git commit -m "feat(accrual): AccrualPeriod, AccrualBasis, KnockInAnchor and the accrual math" -- quantark/asset/equity/product/option/accrual.py test/accrual_time
```

---

### Task 2: Contract terms: `AccrualEndType`, `AccrualConfig` fields, `ObservationRecord.accrual`

**Files:**
- Modify: `quantark/util/enum/option_enums.py` (after `TenorEnd`), `quantark/util/enum/__init__.py`
- Modify: `quantark/asset/equity/product/option/snowball_config.py` (`AccrualConfig`)
- Modify: `quantark/asset/equity/product/option/observation_schedule.py` (`ObservationRecord`)
- Modify: `quantark/intraday/twin.py:134-135,197` (drop the `accrual_dates=None` kwargs)
- Test: `test/accrual_time/test_config.py`

**Interfaces:**
- Consumes: `AccrualPeriod` (Task 1).
- Produces: `AccrualEndType {OBSERVATION, SETTLEMENT}` from `quantark.util.enum`; `AccrualConfig.accrual_end`, `.post_accrual_factors: Optional[List[float]]`, `.accrual_calendar: Optional[Calendar]`, `.accrual_bus_days_in_year: Optional[float]`; `AccrualConfig.shifted(..., dropped_post=0)` slices `post_accrual_factors`; `ObservationRecord.accrual: Optional[AccrualPeriod] = None`.

- [ ] **Step 1: Write the failing tests**

```python
"""Contract-level accrual terms (spec §4, §4.7)."""
import pytest

from quantark.asset.equity.product.option.accrual import AccrualKind, AccrualPeriod
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import AccrualEndType


def test_defaults_do_not_move_existing_contracts():
    c = AccrualConfig()
    assert c.accrual_end is AccrualEndType.OBSERVATION
    assert c.post_accrual_factors is None and c.accrual_calendar is None and c.accrual_bus_days_in_year is None
    assert not hasattr(c, "accrual_dates")


@pytest.mark.parametrize("kw", [
    dict(accrual_end="settlement"), dict(post_accrual_factors=(0.1,)), dict(post_accrual_factors=[-0.1]),
    dict(accrual_calendar="SSE"), dict(accrual_bus_days_in_year=0.0),
])
def test_bad_terms_are_refused(kw):
    with pytest.raises(ValueError):
        AccrualConfig(**kw)


def test_shifted_slices_both_positional_lists():
    c = AccrualConfig(accrual_factors=[0.1, 0.2, 0.3], post_accrual_factors=[1.0, 2.0])
    s = c.shifted(0.1, dropped_observations=1, dropped_post=1)
    assert s.accrual_factors == [0.2, 0.3] and s.post_accrual_factors == [2.0]


def test_terms_carry_a_calendar():
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2027))
    assert AccrualConfig(accrual_calendar=cal, accrual_bus_days_in_year=244).accrual_bus_days_in_year == 244


def test_record_carries_an_accrual_period():
    p = AccrualPeriod(AccrualKind.TIME, origin=0.0, end=0.25)
    assert ObservationRecord(observation_time=0.25, accrual=p).accrual is p
    assert ObservationRecord(observation_time=0.25).accrual is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_config.py -q`
Expected: `ImportError: cannot import name 'AccrualEndType'`.

- [ ] **Step 3: Implement**

`option_enums.py`, after `TenorEnd`:

```python
class AccrualEndType(Enum):
    """Where a record's accrual ends (spec 2026-10-08 accrual time, R2): its fixing or its payment."""

    OBSERVATION = auto()  # the observation (fixing): today's behaviour, the default
    SETTLEMENT = auto()   # the record's authoritative settlement endpoint

    def __str__(self):
        return self.name.capitalize()
```

Export it from `quantark/util/enum/__init__.py` (import list and `__all__`, beside `TenorEnd`).

`AccrualConfig`: delete `accrual_dates` (field and docstring line); add after `first_period_start`:

```python
    #: Where every record's accrual ends (R2): its observation, or its authoritative settlement endpoint.
    accrual_end: AccrualEndType = AccrualEndType.OBSERVATION
    #: KO-reset schedule 2's explicit factors, positionally (spec §4.3). ``accrual_factors`` is schedule 1's only.
    post_accrual_factors: Optional[List[float]] = None
    #: The accrual basis's calendar and denominator for a ``BUSINESS_DAYS`` annualized leg (spec §4.7). Frozen
    #: with the contract: no accrual reads the pricing environment's.
    accrual_calendar: Optional[Calendar] = None
    accrual_bus_days_in_year: Optional[float] = None
```

(`from quantark.util.calendar.business_calendar import Calendar`; `AccrualEndType` from `quantark.util.enum`.)

In `__post_init__`, after the `accrual_factors` loop, factor the loop into `_check_factor_list(name, values)` and call it for both lists, then:

```python
        if not isinstance(self.accrual_end, AccrualEndType):
            raise ValueError(f"accrual_end must be AccrualEndType, got {self.accrual_end!r}")
        if self.accrual_calendar is not None and not isinstance(self.accrual_calendar, Calendar):
            raise ValueError(f"accrual_calendar must be Calendar, got {type(self.accrual_calendar).__name__}")
        if self.accrual_bus_days_in_year is not None:
            days = float(self.accrual_bus_days_in_year)
            if not isfinite(days) or days <= 0.0:
                raise ValueError(f"accrual_bus_days_in_year must be positive, got {self.accrual_bus_days_in_year}")
```

`shifted(self, time_bump, dropped_observations=0, schedule=None, dropped_post=0)`: also slice
`post_accrual_factors = list(self.post_accrual_factors[dropped_post:])` when it is set and `dropped_post > 0`.
(Factors only keep `validate()`'s length check true on a rebuilt aged product; periods already hold them.)

`ObservationRecord`: add the last field

```python
    #: The record's accrual period (spec 2026-10-08 accrual time), filled once at product construction and never
    #: recomputed; observation timing (``observation_time``) may move, this does not.
    accrual: Optional["AccrualPeriod"] = None
```

with `from .accrual import AccrualPeriod` under `TYPE_CHECKING`, and in `validate()`:

```python
        if self.accrual is not None:
            from .accrual import AccrualPeriod
            if not isinstance(self.accrual, AccrualPeriod):
                raise ValidationError(f"ObservationRecord.accrual must be AccrualPeriod, got {type(self.accrual).__name__}")
```

`twin.py`: remove `accrual_dates=None` from both `replace(twin.accrual_config, ...)` calls (Task 7 rewrites them; this keeps them importable).

- [ ] **Step 4: Run them to see them pass**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time -q && .venv/bin/python -m pytest -q test/test_snowball_option.py test/test_phoenix_option.py test/test_aged_accrual.py`
Expected: all pass (no consumer reads the new fields yet).

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(accrual): AccrualEndType, schedule-2 factors, frozen basis terms, ObservationRecord.accrual" -- quantark/util/enum quantark/asset/equity/product/option/snowball_config.py quantark/asset/equity/product/option/observation_schedule.py quantark/intraday/twin.py test/accrual_time/test_config.py
```

---

### Task 3: Filling periods at construction (`accrual_build.py`)

**Files:**
- Create: `quantark/asset/equity/product/option/accrual_build.py`
- Modify: `snowball_option.py` (`validate`: call `build_contract_accrual(self)` last), `phoenix_option.py` (same), `ko_reset_snowball_option.py` (`validate`: `_validate_post_ko_mode()` before `super().validate()`)
- Create: `test/accrual_time/conftest.py`, `test/accrual_time/test_build.py`

**Interfaces:**
- Consumes: Task 1 types; Task 2 config fields.
- Produces: `build_contract_accrual(product) -> None`. After construction, every record of `barrier_config.ko_observation_schedule`, `.ki_observation_schedule` and (KO-reset) `post_barrier_config.ko_observation_schedule` carries `accrual`; `product._terminal_accrual` holds `contract` (all), plus `pre` and `post` (KO-reset). Schedules are detached copies.

- [ ] **Step 1: Write the fixtures**

`test/accrual_time/conftest.py`:

```python
"""Small autocallables on the quoted clock (k/12) and on dates, for the accrual-time tests."""
from datetime import datetime, timedelta

from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import PostKOScheduleMode

T0 = datetime(2026, 1, 5)


def times(*ts, level=103.0):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in ts])


def dates(*days, level=103.0, start=T0):
    return ObservationSchedule(records=[ObservationRecord(observation_date=start + timedelta(days=d), barrier=level)
                                        for d in days])


def env(valuation=T0, spot=100.0, **kw):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.0), valuation_date=valuation, **kw)


def barriers(ko, ki=None, rate=0.12):
    return BarrierConfig(ko_barrier=103.0, ko_rate=rate, ko_observation_schedule=ko,
                         ki_barrier=None if ki is None else 75.0, ki_observation_schedule=ki)


def snowball(ko=None, ki=None, accrual=None, rate=0.12, maturity=0.25, **kw):
    ko = ko if ko is not None else times(1 / 12, 2 / 12, 3 / 12)
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barriers(ko, ki, rate),
                          payoff_config=kw.pop("payoff_config", PayoffConfig(rebate_rate=0.05)),
                          accrual_config=accrual or AccrualConfig(), maturity=maturity, **kw)


def phoenix(ko=None, ki=None, accrual=None, coupon=None, maturity=0.25, **kw):
    ko = ko if ko is not None else times(1 / 12, 2 / 12, 3 / 12)
    return PhoenixOption(initial_price=100.0, strike=100.0, barrier_config=barriers(ko, ki),
                         coupon_config=coupon or CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12),
                         accrual_config=accrual or AccrualConfig(), maturity=maturity, **kw)


def ko_reset(mode=PostKOScheduleMode.ABSOLUTE, pre=None, ki=None, post=None, accrual=None, maturity=None, **kw):
    pre = pre if pre is not None else times(0.25, 0.5)
    ki = ki if ki is not None else times(0.25, 0.5, level=75.0)
    post = post if post is not None else times(0.25, 0.5, 0.75, level=100.0)
    if maturity is None:
        maturity = 1.25 if mode is PostKOScheduleMode.REBASED else 0.75
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=barriers(pre, ki),
        post_barrier_config=BarrierConfig(ko_barrier=100.0, ko_rate=0.05, ko_observation_schedule=post),
        payoff_config=kw.pop("payoff_config", PayoffConfig(rebate_rate=0.05)), accrual_config=accrual or AccrualConfig(),
        maturity=maturity, post_ko_mode=mode, **kw)


def periods(schedule):
    return [r.accrual for r in schedule.records]
```

- [ ] **Step 2: Write the failing tests** (`test/accrual_time/test_build.py`)

```python
"""Construction fills every record's period and the terminals (spec §4.2–4.7)."""
from dataclasses import replace
from datetime import timedelta

import pytest

from quantark.asset.equity.product.option.accrual import AccrualKind as K
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit as U
from quantark.util.calendar import BusinessDayConvention, CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import AccrualEndType, CouponPayType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, ko_reset, periods, phoenix, snowball, times

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def test_numeric_records_end_at_the_quoted_fraction():
    p = snowball()
    assert [(a.kind, a.origin, a.start, a.end) for a in periods(p.barrier_config.ko_observation_schedule)] == [
        (K.TIME, 0.0, 0.0, 1 / 12), (K.TIME, 0.0, 1 / 12, 2 / 12), (K.TIME, 0.0, 2 / 12, 3 / 12)]


def test_aged_legacy_contract_banks_its_offset():
    p = snowball(ko=times(0.2), accrual=AccrualConfig(accrued_offset=0.3, first_period_start=0.25))
    a = p.barrier_config.ko_observation_schedule.records[0].accrual
    assert (a.start, a.end) == (0.25, 0.3 + 0.2)


def test_unknown_first_period_start_stays_unknown():
    p = snowball(ko=times(0.2), accrual=AccrualConfig(accrued_offset=0.3, first_period_start=None))
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.start is None


def test_dated_records_chain_from_initial_date():
    p = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60))
    a0, a1 = periods(p.barrier_config.ko_observation_schedule)
    assert (a0.kind, a0.origin, a0.start, a0.end) == (K.DATE, T0, T0, T0 + timedelta(days=30))
    assert a1.start == a0.end and a1.basis.day_count is DC.ACT_365


def test_dated_annualized_without_initial_date_is_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        snowball(ko=dates(30, 60), maturity=None, exercise_date=T0 + timedelta(days=60))


def test_dated_non_annualized_without_initial_date_is_none():
    p = snowball(ko=dates(30, 60), maturity=None, exercise_date=T0 + timedelta(days=60),
                 accrual=AccrualConfig(is_annualized=False))
    assert {a.kind for a in periods(p.barrier_config.ko_observation_schedule)} == {K.NONE}


def test_accrual_factors_set_both_legs_on_schedule_one():
    p = phoenix(accrual=AccrualConfig(accrual_factors=[0.1, 0.2, 0.3]))
    assert [(a.cumulative_fraction, a.period_fraction) for a in periods(p.barrier_config.ko_observation_schedule)] == [
        (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]


def test_fixed_coupon_fraction_sets_the_coupon_leg_only():
    p = phoenix(coupon=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, fixed_coupon_year_fraction=1 / 12))
    a = p.barrier_config.ko_observation_schedule.records[2].accrual
    assert (a.cumulative_fraction, a.period_fraction, a.end) == (None, 1 / 12, 3 / 12)


def test_schedule_specific_factors():
    ok = ko_reset(accrual=AccrualConfig(accrual_factors=[0.3, 0.6]))
    assert [a.cumulative_fraction for a in periods(ok.post_barrier_config.ko_observation_schedule)] == [None] * 3
    with pytest.raises(ValidationError, match="post_accrual_factors"):
        ko_reset(accrual=AccrualConfig(post_accrual_factors=[0.1, 0.2]))
    with pytest.raises(ValidationError, match="accrual_factors"):
        ko_reset(accrual=AccrualConfig(accrual_factors=[0.1, 0.2, 0.3]))


def test_rebased_schedule_two_is_relative_offsets():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(accrued_offset=0.3))
    a = periods(p.post_barrier_config.ko_observation_schedule)
    assert [(x.relative, x.start, x.end) for x in a] == [(True, 0.0, 0.25), (True, 0.25, 0.5), (True, 0.5, 0.75)]


def test_rebased_post_factor_is_the_offsets_accrual():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(post_accrual_factors=[0.2, 0.4, 0.6]))
    a = periods(p.post_barrier_config.ko_observation_schedule)
    assert [(x.relative, x.end, x.cumulative_fraction) for x in a] == [(True, 0.2, None), (True, 0.4, None),
                                                                       (True, 0.6, None)]


def test_knock_in_records_accrue_to_their_fixing_even_under_settlement():
    conv = SettlementConvention(lag=0.01, lag_unit=U.YEAR_FRACTION)
    p = ko_reset(M.REBASED, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT), settlement_convention=conv)
    assert [a.end for a in periods(p.barrier_config.ki_observation_schedule)] == [0.25, 0.5]
    assert periods(p.barrier_config.ko_observation_schedule)[0].end == 0.25 + 0.01


def test_rebased_settlement_endpoint_is_the_explicit_offset_payment():
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    p = ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT))
    assert periods(p.post_barrier_config.ko_observation_schedule)[1].end == 0.52


@pytest.mark.parametrize("conv, sched, match", [
    (SettlementConvention(lag=2, lag_unit=U.BUSINESS_DAYS), "dates", "calendar"),
    (SettlementConvention(lag=0.01, lag_unit=U.YEAR_FRACTION), "dates", "YEAR_FRACTION"),
    (SettlementConvention(lag=2, lag_unit=U.BUSINESS_DAYS, calendar=CAL), "times", "BUSINESS_DAYS"),
])
def test_unresolvable_settlement_endpoints_are_refused(conv, sched, match):
    ko = dates(30, 60) if sched == "dates" else times(0.25)
    kw = dict(initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60)) if sched == "dates" else {}
    with pytest.raises(ValidationError, match=match):
        snowball(ko=ko, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT), settlement_convention=conv, **kw)


def test_business_day_settlement_endpoint_uses_the_conventions_calendar():
    conv = SettlementConvention(lag=2, lag_unit=U.BUSINESS_DAYS, calendar=CAL,
                                business_day_convention=BusinessDayConvention.FOLLOWING)
    p = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60),
                 accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT), settlement_convention=conv)
    fixing = T0 + timedelta(days=30)
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.end == CAL.adjust_date(
        CAL.add_business_days(fixing, 2), BusinessDayConvention.FOLLOWING)


def test_expiry_coupons_under_settlement_are_refused():
    with pytest.raises(ValidationError, match="EXPIRY"):
        snowball(accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT, coupon_pay_type=CouponPayType.EXPIRY))


def test_business_days_annualized_needs_its_terms():
    kw = dict(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60),
              annualization_day_count=DC.BUSINESS_DAYS)
    with pytest.raises(ValidationError, match="BUSINESS_DAYS accrual needs its calendar and denominator"):
        snowball(**kw)
    p = snowball(accrual=AccrualConfig(accrual_calendar=CAL, accrual_bus_days_in_year=244.0), **kw)
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.basis.calendar is CAL


def test_shared_schedule_is_detached():
    shared = times(1 / 12)
    a = snowball(ko=shared, maturity=1 / 12, accrual=AccrualConfig(accrual_factors=[1 / 12]))
    b = snowball(ko=shared, maturity=1 / 12, accrual=AccrualConfig(accrual_factors=[0.1]))
    assert shared.records[0].accrual is None
    assert a.barrier_config.ko_observation_schedule.records[0].accrual.cumulative_fraction == 1 / 12
    assert b.barrier_config.ko_observation_schedule.records[0].accrual.cumulative_fraction == 0.1


def test_a_record_carrying_a_period_keeps_it():
    p = snowball()
    q = replace(p, accrual_config=AccrualConfig(accrued_offset=0.5))
    assert periods(q.barrier_config.ko_observation_schedule) == periods(p.barrier_config.ko_observation_schedule)


def test_a_kept_period_with_other_basis_terms_is_refused():
    p = snowball(ko=dates(30), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=30))
    with pytest.raises(ValidationError, match="basis"):
        replace(p, annualization_day_count=DC.ACT_360)


def test_terminals():
    s = snowball(tenor=1.0)
    assert s._terminal_accrual["contract"].cumulative_fraction == 1.0
    n = snowball(accrual=AccrualConfig(accrued_offset=0.1))
    assert (n._terminal_accrual["contract"].kind, n._terminal_accrual["contract"].end) == (K.TIME, 0.1 + 0.25)
    d = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60))
    assert d._terminal_accrual["contract"].end == T0 + timedelta(days=60)
    r = ko_reset(M.REBASED)
    assert (r._terminal_accrual["pre"].end, r._terminal_accrual["post"].relative, r._terminal_accrual["post"].end) == (
        0.5, True, 0.75)
    a = ko_reset()
    assert a._terminal_accrual["post"].end == 0.75


def test_rebased_terminal_ignores_the_payment_offset():
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    p = ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT))
    assert p._terminal_accrual["post"].end == 0.5


def test_annualized_rebate_without_dated_origin_is_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        snowball(ko=dates(30), maturity=None, exercise_date=T0 + timedelta(days=30),
                 accrual=AccrualConfig(is_annualized=False, is_annualized_rebate=True),
                 payoff_config=PayoffConfig(rebate_rate=0.05))
```

- [ ] **Step 3: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_build.py -q`
Expected: failures (records carry no `accrual`, `_terminal_accrual` missing).

- [ ] **Step 4: Write `accrual_build.py`**

```python
"""Turning a contract's accrual terms into periods, once, at construction (spec §4.2–4.7)."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar import BusinessDayConvention
from quantark.util.enum import AccrualEndType, CouponPayType, PostKOScheduleMode
from quantark.util.exceptions import ValidationError

from .accrual import BUSINESS_DAYS_TERMS, AccrualBasis, AccrualKind, AccrualPeriod
from .observation_schedule import ObservationRecord, ObservationSchedule

R5 = ("a dated contract whose annualized cash needs an accrual start needs initial_date (accrual time spec R5): "
      "its accrual would otherwise start 'today'")


def _detach(schedule: Optional[ObservationSchedule]) -> Optional[ObservationSchedule]:
    """A copy whose records are fresh copies: periods derived for this product never reach another (§4.2)."""
    if schedule is None:
        return None
    return ObservationSchedule(records=[replace(r) for r in schedule.records],
                               aggregation_mode=schedule.aggregation_mode, frequency=schedule.frequency)


class _Terms:
    """The contract terms every record's period is derived from."""

    def __init__(self, product):
        ac = product.accrual_config
        self.product = product
        self.accrued_offset = float(ac.accrued_offset)
        self.first_period_start = ac.first_period_start
        self.settlement = ac.accrual_end is AccrualEndType.SETTLEMENT
        self.initial_date = product.initial_date
        self.basis = AccrualBasis(product.annualization_day_count, ac.accrual_calendar, ac.accrual_bus_days_in_year)
        coupon = getattr(product, "coupon_config", None)
        self.period_basis = (self.basis if coupon is None else
                             AccrualBasis(coupon.day_count_convention, ac.accrual_calendar, ac.accrual_bus_days_in_year))
        self.ko_annualized = bool(product._effective_annualized_flag(ac.is_annualized_ko))
        self.coupon_annualized = coupon is not None and bool(product.is_coupon_rate_annualized)
        self.fixed = None if coupon is None else coupon.fixed_coupon_year_fraction
        if self.settlement and (ac.coupon_pay_type is CouponPayType.EXPIRY
                                or getattr(coupon, "coupon_pay_type", None) is CouponPayType.EXPIRY):
            raise ValidationError("an EXPIRY coupon under ACCRUAL_END_TYPE=SETTLEMENT: its accrual end is a "
                                  "term-sheet choice the contract must state (accrual time spec §4.5)")

    # -- endpoints -----------------------------------------------------------------------------------------
    def fixing(self, rec: ObservationRecord):
        """(kind, origin, end) of a record's contractual fixing."""
        if rec.observation_date is not None:
            if self.initial_date is None:
                return AccrualKind.NONE, None, None
            return AccrualKind.DATE, self.initial_date, rec.observation_date
        return AccrualKind.TIME, 0.0, self.accrued_offset + float(rec.observation_time)

    def payment(self, rec: ObservationRecord):
        """(kind, origin, end) of a record's authoritative settlement endpoint (§4.5), own terms only."""
        kind, origin, end = self.fixing(rec)
        if kind is AccrualKind.NONE:
            return kind, origin, end
        raw = settlement_endpoint(self.product.settlement_convention, rec)
        return kind, origin, raw if kind is AccrualKind.DATE else self.accrued_offset + raw

    def end_of(self, rec: ObservationRecord):
        return self.payment(rec) if self.settlement else self.fixing(rec)

    # -- refusals ------------------------------------------------------------------------------------------
    def require(self, kind: AccrualKind, cumulative_needed: bool, coupon_needed: bool = False) -> None:
        if kind is AccrualKind.NONE and (cumulative_needed or coupon_needed):
            raise ValidationError(R5)
        if kind is AccrualKind.DATE and cumulative_needed and not self.basis.is_complete:
            raise ValidationError(BUSINESS_DAYS_TERMS)


def settlement_endpoint(convention: Optional[SettlementConvention], rec: ObservationRecord):
    """The settlement resolver's precedence, from the product's own terms (no environment), §4.5."""
    if rec.observation_date is not None:
        if rec.settlement_date is not None:
            return rec.settlement_date
        if rec.settlement_time is not None:
            raise ValidationError("a dated record's SETTLEMENT accrual needs a settlement_date, not a settlement_time")
        if convention is None or convention.lag == 0.0:
            return rec.observation_date
        unit, bdc, cal = convention.lag_unit, convention.business_day_convention, convention.calendar
        if unit is SettlementLagUnit.BUSINESS_DAYS:
            if cal is None:
                raise ValidationError("SETTLEMENT accrual on a business-day lag needs the settlement convention's own "
                                      "calendar (construction has no environment)")
            return cal.adjust_date(cal.add_business_days(rec.observation_date, int(convention.lag)), bdc)
        if unit is SettlementLagUnit.CALENDAR_DAYS:
            day = rec.observation_date + timedelta(days=int(convention.lag))
            if bdc is BusinessDayConvention.UNADJUSTED:
                return day
            if cal is None:
                raise ValidationError("SETTLEMENT accrual on an adjusted calendar-day lag needs the settlement "
                                      "convention's own calendar")
            return cal.adjust_date(day, bdc)
        raise ValidationError("a YEAR_FRACTION settlement lag cannot end a dated record's accrual")
    if rec.settlement_date is not None:
        raise ValidationError("a numeric record's SETTLEMENT accrual needs a settlement_time, not a settlement_date")
    if rec.settlement_time is not None:
        return float(rec.settlement_time)
    if convention is None or convention.lag == 0.0:
        return float(rec.observation_time)
    if convention.lag_unit is SettlementLagUnit.YEAR_FRACTION:
        return float(rec.observation_time) + float(convention.lag)
    raise ValidationError(f"a {convention.lag_unit.name} settlement lag cannot end a numeric record's accrual")


def _keep(rec: ObservationRecord, terms: _Terms, coupon: bool) -> bool:
    """A record already carrying a period keeps it (§4.2), unless its basis terms are another contract's (P-5)."""
    period = rec.accrual
    if period is None:
        return False
    if period.basis is not None and not period.basis.same_terms(terms.basis):
        raise ValidationError("a record's accrual period carries another basis than this contract's "
                              "(annualization day count or accrual calendar changed after construction)")
    if coupon and period.period_basis is not None and not period.period_basis.same_terms(terms.period_basis):
        raise ValidationError("a record's accrual period carries another coupon basis than this contract's")
    return True


def _fill_ko(records: List[ObservationRecord], terms: _Terms, factors, name: str, *, coupon: bool,
             factor_sets_coupon: bool) -> None:
    if factors is not None and len(factors) != len(records):
        raise ValidationError(f"{name} length ({len(factors)}) must match its schedule's {len(records)} records")
    previous = None
    for i, rec in enumerate(records):
        if _keep(rec, terms, coupon):
            previous = rec.accrual
            continue
        kind, origin, end = terms.end_of(rec)
        cumulative = None if factors is None else float(factors[i])
        period_fraction = cumulative if factor_sets_coupon else None
        if coupon and period_fraction is None:
            period_fraction = terms.fixed
        start = None
        if kind is not AccrualKind.NONE:
            if previous is None:
                start = (terms.initial_date if kind is AccrualKind.DATE else
                         (None if terms.first_period_start is None else float(terms.first_period_start)))
            elif previous.kind is kind and not previous.relative:
                start = previous.end
        terms.require(kind, terms.ko_annualized and cumulative is None,
                      coupon and terms.coupon_annualized and period_fraction is None)
        rec.accrual = AccrualPeriod(kind, origin=origin, start=start, end=end,
                                    basis=None if kind is AccrualKind.NONE else terms.basis,
                                    period_basis=None if kind is AccrualKind.NONE else terms.period_basis,
                                    cumulative_fraction=cumulative, period_fraction=period_fraction)
        previous = rec.accrual


def _fill_ki(records: List[ObservationRecord], terms: _Terms, anchor_needed: bool) -> None:
    """Knock-in records accrue to their contractual fixing, always (§4.3): the end is a REBASED anchor only."""
    for rec in records:
        if _keep(rec, terms, False):
            continue
        kind, origin, end = terms.fixing(rec)
        terms.require(kind, anchor_needed)
        rec.accrual = AccrualPeriod(kind, origin=origin, end=end,
                                    basis=None if kind is AccrualKind.NONE else terms.basis)


def _fill_rebased(records: List[ObservationRecord], terms: _Terms, factors) -> None:
    """REBASED schedule 2: quoted offsets from the knock-in, its own rule (§4.3)."""
    if factors is not None and len(factors) != len(records):
        raise ValidationError(f"post_accrual_factors length ({len(factors)}) must match schedule 2's "
                              f"{len(records)} records")
    previous_end = 0.0
    for i, rec in enumerate(records):
        if _keep(rec, terms, False):
            previous_end = rec.accrual.end
            continue
        if factors is not None:
            end = float(factors[i])
        elif terms.settlement:
            end = settlement_endpoint(terms.product.settlement_convention, rec)
        else:
            end = float(rec.observation_time)
        rec.accrual = AccrualPeriod(AccrualKind.TIME, start=min(previous_end, end), end=end, relative=True)
        previous_end = end


def _schedule_terminal(records: List[ObservationRecord], terms: _Terms) -> Optional[AccrualPeriod]:
    """Origin to a schedule's last contractual fixing (KO-reset `pre`, ABSOLUTE `post`), §4.4."""
    if not records:
        return None
    dated = [r.observation_date for r in records if r.observation_date is not None]
    if dated:
        if terms.initial_date is None:
            return AccrualPeriod(AccrualKind.NONE)
        return AccrualPeriod(AccrualKind.DATE, origin=terms.initial_date, end=max(dated), basis=terms.basis)
    return AccrualPeriod(AccrualKind.TIME, origin=0.0,
                         end=terms.accrued_offset + max(float(r.observation_time) for r in records))


def _contract_terminal(product, terms: _Terms) -> AccrualPeriod:
    """``BaseEquityOption.get_tenor``'s rule, frozen (§4.4)."""
    if product.tenor is not None:
        return AccrualPeriod(AccrualKind.NONE, cumulative_fraction=float(product.tenor))
    if product.initial_date is not None and product.get_tenor_end_date() is not None:
        return AccrualPeriod(AccrualKind.DATE, origin=product.initial_date, end=product.get_tenor_end_date(),
                             basis=terms.basis)
    if product.maturity is not None:
        return AccrualPeriod(AccrualKind.TIME, origin=0.0, end=terms.accrued_offset + float(product.maturity))
    return AccrualPeriod(AccrualKind.NONE)


def _terminals(product, terms: _Terms) -> Dict[str, AccrualPeriod]:
    ac = product.accrual_config
    out = {"contract": _contract_terminal(product, terms)}
    post_config = getattr(product, "post_barrier_config", None)
    if post_config is not None:
        pre_records = product.barrier_config.ko_observation_schedule.records
        out["pre"] = _schedule_terminal(pre_records, terms) or out["contract"]
        post_records = post_config.ko_observation_schedule.records
        if product.post_ko_mode is PostKOScheduleMode.REBASED:
            final = max(float(r.observation_time) for r in post_records) if post_records else 0.0
            out["post"] = AccrualPeriod(AccrualKind.TIME, start=0.0, end=final, relative=True)
        else:
            out["post"] = _schedule_terminal(post_records, terms) or out["pre"]
    v0 = "pre" if post_config is not None else "contract"
    v1 = "post" if post_config is not None else "contract"
    for leg, needed in ((v0, bool(ac.is_annualized_rebate)), (v1, bool(ac.is_annualized_ki))):
        period = out[leg]
        if period.cumulative_fraction is None and not period.relative:
            terms.require(period.kind, needed)
    return out


def build_contract_accrual(product) -> None:
    """Detach the product's schedules and give every record without one its period, and the product its
    terminal periods (spec §4.2). Called last in ``validate()``."""
    terms = _Terms(product)
    ac = product.accrual_config
    bc = product.barrier_config
    ko, ki = _detach(bc.ko_observation_schedule), _detach(bc.ki_observation_schedule)
    phoenix = hasattr(product, "coupon_config")
    if ko is not None:
        _fill_ko(ko.records, terms, ac.accrual_factors, "accrual_factors", coupon=phoenix, factor_sets_coupon=True)
    post_config = getattr(product, "post_barrier_config", None)
    rebased = post_config is not None and product.post_ko_mode is PostKOScheduleMode.REBASED
    if ki is not None:
        _fill_ki(ki.records, terms, anchor_needed=rebased and terms.ko_annualized and ac.post_accrual_factors is None)
    product.barrier_config = replace(bc, ko_observation_schedule=ko, ki_observation_schedule=ki)
    if post_config is not None:
        post = _detach(post_config.ko_observation_schedule)
        if post is not None:
            if rebased:
                _fill_rebased(post.records, terms, ac.post_accrual_factors)
            else:
                _fill_ko(post.records, terms, ac.post_accrual_factors, "post_accrual_factors", coupon=False,
                         factor_sets_coupon=False)
        product.post_barrier_config = replace(post_config, ko_observation_schedule=post)
    elif ac.post_accrual_factors is not None:
        raise ValidationError("post_accrual_factors belong to a KO-reset contract's schedule 2")
    if getattr(product, "_terminal_accrual", None) is None:
        product._terminal_accrual = _terminals(product, terms)
```

Notes for the implementer:
- In `_fill_rebased`, a factor makes `end = factor`, as the spec says ("relative with end = factor"). `start = min(previous_end, end)` keeps `start <= end` when a later factor is smaller. The coupon leg of schedule 2 is never read (KO-reset is not a Phoenix).
- `replace(bc, ...)` re-runs `BarrierConfig.__post_init__`; that is fine. `ki_observation_schedule=None` stays `None`.
- Wire the call: `SnowballOption.validate()` and `PhoenixOption.validate()` end with `build_contract_accrual(self)` after `self._build_observation_schedules()`. `KnockOutResetSnowballOption.validate()` becomes `self._validate_post_ko_mode(); super().validate()`, so that its builder pass sees a checked mode.
- Snowball's and Phoenix's `_validate_accrual_parameters` already check `accrual_factors` length against the KO count; leave them (the builder's check is the same message family).

- [ ] **Step 5: Run them to see them pass**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/product/option/accrual_build.py test/accrual_time/conftest.py test/accrual_time/test_build.py
git commit -m "feat(accrual): fill record and terminal periods at construction; detach schedules; R5 and basis refusals" -- quantark/asset/equity/product/option test/accrual_time
```

---

### Task 4: One resolver for the products

**Files:**
- Modify: `accrual.py` (add the product-level resolver and `knock_in_anchor_for`)
- Modify: `snowball_option.py` (`resolve_ko_observations` accrual block; `get_contract_tenor`; V0 fixed-rebate branch, P-6)
- Modify: `phoenix_option.py` (`resolve_ko_observations`, `coupon_period_fraction`, `get_coupon_payoff`, `get_ko_payoff`, `get_contract_tenor`, V0)
- Modify: `ko_reset_snowball_option.py` (`compute_ko_accrual_factor`, `_resolve_pre_contract_tenor`, `_resolve_post_contract_tenor`, `get_contract_tenor`, `get_maturity_payoff_v1(..., *, contract_tenor=None)`)
- Test: `test/accrual_time/test_products.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `ko_accrual_factor(product, record, env=None, *, knock_in=None) -> float`
  - `coupon_period_fraction(product, record, env=None) -> float`
  - `contract_tenor(product, env=None, leg="contract", *, knock_in=None) -> float`
  - `knock_in_anchor_for(record, source_index) -> Optional[KnockInAnchor]`, which returns `None` for a `NONE` period
  - `KnockOutResetSnowballOption.compute_ko_accrual_factor(observation_time, schedule_record, pricing_env, *, knock_in=None)`, whose `observation_time` is now unused and kept only for callers
  - `KnockOutResetSnowballOption.get_maturity_payoff_v1(spot, pricing_env=None, *, contract_tenor=None)`

- [ ] **Step 1: Write the failing tests**

```python
"""Cash reads periods only: invariance, precedence, the two legs, §3 regressions (spec §5, §8)."""
from dataclasses import replace
from datetime import timedelta

import pytest

from quantark.asset.equity.product.option.accrual import (
    AccrualPeriod, contract_tenor, coupon_period_fraction, knock_in_anchor_for, ko_accrual_factor,
)
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, env, ko_reset, phoenix, snowball, times


def ko_cash(p, e=None):
    return [r.payoff for r in p.resolve_ko_observations(e or env())]


@pytest.mark.parametrize("step", [12, 52, 252])
@pytest.mark.parametrize("make", [snowball, phoenix], ids=["snowball", "phoenix"])
def test_cash_is_invariant_under_ageing_and_snapping(make, step):
    ts = [k / step for k in range(1, 4)]
    p = make(ko=times(*ts), maturity=ts[-1])
    base = ko_cash(p)
    snapped = make(ko=times(*ts), maturity=ts[-1])
    for r in snapped.barrier_config.ko_observation_schedule.records:
        r.observation_time = round(r.observation_time * 365) / 365
    assert ko_cash(snapped) == base
    aged = make(ko=times(*ts), maturity=ts[-1])
    aged.time_shift(0.5 / step, T0 + timedelta(days=1), env())
    assert ko_cash(aged) == base[-len(ko_cash(aged)):]


def test_explicit_factor_wins_over_the_flag():
    p = snowball(ko=times(0.25), accrual=AccrualConfig(accrual_factors=[0.4], is_annualized_ko=False))
    assert ko_cash(p) == [100.0 + 100.0 * 0.12 * 0.4]


def test_phoenix_two_legs():
    p = phoenix(coupon=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, fixed_coupon_year_fraction=1 / 12))
    recs = p.barrier_config.ko_observation_schedule.records
    assert [100 * 0.12 * ko_accrual_factor(p, r) for r in recs] == pytest.approx([1.0, 2.0, 3.0])
    assert [p.get_coupon_payoff(i) for i in range(3)] == pytest.approx([1.0, 1.0, 1.0])


def test_numeric_with_initial_date_accrues_the_quoted_fraction():
    p = snowball(ko=times(0.25), maturity=0.25, initial_date=T0)
    assert ko_cash(p, env(T0 + timedelta(days=30))) == [100.0 + 100.0 * 0.12 * 0.25]


def test_phoenix_get_ko_payoff_keeps_accrued_offset():
    p = phoenix(ko=times(0.2), maturity=0.2, accrual=AccrualConfig(accrued_offset=0.3, first_period_start=0.25))
    assert p.get_ko_payoff(80.0, 0) == pytest.approx(100.0 + 100.0 * 0.12 * 0.5 + 100.0 * 0.12 * 0.25)


def test_aged_dated_phoenix_first_remaining_period_starts_at_the_previous_record():
    p = phoenix(ko=dates(30, 60, 90), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=90))
    p.time_shift(35 / 365, T0 + timedelta(days=35), env())
    assert coupon_period_fraction(p, p.barrier_config.ko_observation_schedule.records[0]) == 30 / 365


def test_tenors():
    assert contract_tenor(snowball(tenor=1.0)) == 1.0
    p = ko_reset(M.REBASED, post=times(0.25, 0.5, level=100.0))   # final offset 0.5 (spec round 1 control)
    with pytest.raises(ValidationError, match="anchor"):
        contract_tenor(p, None, "post")
    ki = p.barrier_config.ki_observation_schedule.records
    assert [contract_tenor(p, None, "post", knock_in=knock_in_anchor_for(r, i)) for i, r in enumerate(ki)] == [
        0.75, 1.0]


def test_rebased_offset_on_an_aged_contract():
    """Review round 2: elapsed 0.3, remaining knock-in 0.2 (anchor 0.5), offset 0.25 -> 0.75."""
    p = ko_reset(M.REBASED, pre=times(0.2), ki=times(0.2, level=75.0), post=times(0.25, level=100.0),
                 accrual=AccrualConfig(accrued_offset=0.3), maturity=0.45)
    anchor = knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0)
    post = p.post_barrier_config.ko_observation_schedule.records[0]
    assert ko_accrual_factor(p, post, knock_in=anchor) == pytest.approx(0.75, abs=0.0)


def test_embedded_anchor_terminal_matches_the_mc_route():
    """Round 3 P1-3: knock-in 0.25, final observation offset 0.5, payment offset 0.52 under SETTLEMENT -> 0.75."""
    from quantark.util.enum import AccrualEndType
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    p = ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT))
    anchor = knock_in_anchor_for(p.barrier_config.ki_observation_schedule.records[0], 0)
    embedded = replace(p._terminal_accrual["post"], anchor=anchor)
    assert contract_tenor(p, None, "post", knock_in=anchor) == 0.75
    from quantark.asset.equity.product.option.accrual import cumulative_accrual
    assert cumulative_accrual(embedded) == 0.75
    paid = p.post_barrier_config.ko_observation_schedule.records[1]
    assert ko_accrual_factor(p, paid, knock_in=anchor) == pytest.approx(0.77, abs=1e-15)


def test_non_accruing_dated_contract_pays_as_today():
    p = snowball(ko=dates(30), maturity=None, exercise_date=T0 + timedelta(days=30),
                 accrual=AccrualConfig(is_annualized=False))
    assert ko_cash(p) == [100.0 + 12.0]
    with pytest.raises(ValidationError, match="no accrual terms"):
        from quantark.asset.equity.product.option.accrual import cumulative_accrual
        cumulative_accrual(p.barrier_config.ko_observation_schedule.records[0].accrual)


def test_changed_coupon_day_count_after_construction_raises():
    p = phoenix(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60))
    p.coupon_config = replace(p.coupon_config, day_count_convention=DC.ACT_360)
    with pytest.raises(ValidationError, match="coupon basis"):
        p.get_coupon_payoff(0)


def test_replace_after_ageing_keeps_record_cash():
    p = snowball()
    before = ko_cash(p)
    p.time_shift(1 / 24, T0 + timedelta(days=15), env())
    q = replace(p)
    assert ko_cash(q) == before[-len(ko_cash(q)):]
    assert contract_tenor(q) == pytest.approx(contract_tenor(snowball()), rel=0, abs=1e-15)


def test_v0_unannualized_rebate_never_reads_the_tenor():
    p = snowball(ko=dates(30), maturity=None, exercise_date=T0 + timedelta(days=30),
                 accrual=AccrualConfig(is_annualized=False), payoff_config=PayoffConfig(rebate_rate=0.05))
    assert p.get_maturity_payoff_v0(100.0) == 105.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_products.py -q`
Expected: `ImportError: cannot import name 'contract_tenor'`.

- [ ] **Step 3: Add the product-level resolver to `accrual.py`**

```python
def _period_of(record) -> AccrualPeriod:
    period = getattr(record, "accrual", None)
    if period is None:
        raise ValidationError("this observation record carries no accrual period (built outside a product?)")
    return period


def _guard_basis(period: AccrualPeriod, product, coupon: bool) -> None:
    """P-5: a DATE read under day counts the product no longer has is stale, never silently paid."""
    if period.kind is not AccrualKind.DATE:
        return
    if coupon:
        expected = getattr(getattr(product, "coupon_config", None), "day_count_convention", None)
        if expected is not None and period.period_basis.day_count is not expected:
            raise ValidationError("the record's coupon basis is not the contract's coupon day count (changed after "
                                  "construction): rebuild the product")
    elif period.basis.day_count is not product.annualization_day_count:
        raise ValidationError("the record's accrual basis is not the contract's annualization day count (changed "
                              "after construction): rebuild the product")


def ko_accrual_factor(product, record, env=None, *, knock_in: Optional[KnockInAnchor] = None) -> float:
    """The knock-out leg, in today's precedence: explicit factor, then the annualization flag, then the period."""
    period = _period_of(record)
    if period.cumulative_fraction is not None:
        return float(period.cumulative_fraction)
    if not product._effective_annualized_flag(product.accrual_config.is_annualized_ko):
        return 1.0
    _guard_basis(period, product, coupon=False)
    return cumulative_accrual(period, knock_in=knock_in)


def coupon_period_fraction(product, record, env=None) -> float:
    """The Phoenix coupon leg: 1 for a per-period rate, else the period's fraction."""
    if not product.is_coupon_rate_annualized:
        return 1.0
    period = _period_of(record)
    if period.period_fraction is None:
        _guard_basis(period, product, coupon=True)
    return coupon_accrual(period)


def contract_tenor(product, env=None, leg: str = "contract", *, knock_in: Optional[KnockInAnchor] = None) -> float:
    """The cumulative leg of a terminal period (spec §4.4): ``contract``, or KO-reset's ``pre`` / ``post``."""
    terminals = getattr(product, "_terminal_accrual", None)
    if not terminals or leg not in terminals:
        raise ValidationError(f"{type(product).__name__} carries no {leg!r} terminal accrual period")
    period = terminals[leg]
    if period.cumulative_fraction is None:
        _guard_basis(period, product, coupon=False)
    return cumulative_accrual(period, knock_in=knock_in)


def knock_in_anchor_for(record, source_index: int) -> Optional[KnockInAnchor]:
    """The anchor a knock-in on ``record`` fixes (spec §5.1); ``None`` for a non-accruing record."""
    period = _period_of(record)
    if period.kind is AccrualKind.NONE:
        return None
    return KnockInAnchor(source_index=int(source_index), fixing=period.end, origin=period.origin, basis=period.basis)
```

`knock_in_anchor_for` on a `TIME` period gives a basis of `None`. Give `TIME` periods the contract basis too: in
`accrual_build._fill_ki` / `_fill_ko`, pass `basis=terms.basis` for `TIME` as well, not only for `DATE`. Then
`KnockInAnchor.__post_init__` holds. Task 3's `test_build` stays green, because it never asserts that a `TIME` basis is `None`.

- [ ] **Step 4: Rewire the products**

`SnowballOption.resolve_ko_observations` and `PhoenixOption.resolve_ko_observations`: delete the block from `if accrual_factors is not None:`
to `accrual_factor = 1.0` (and the now-unused `annualized_ko`, `accrual_factors` and `bus_days_in_year` locals). Put in its place

```python
            accrual_factor = ko_accrual_factor(self, source_record, pricing_env)
```

`SnowballOption.get_contract_tenor` and `PhoenixOption.get_contract_tenor` become `return contract_tenor(self, pricing_env)`.

V0 fixed-rebate branch (Snowball, Phoenix, and KO-reset's override) (P-6): resolve the tenor only when it is read:

```python
        else:
            rebate = self.payoff_config.rebate_rate * self.initial_price * self.contract_multiplier
            if self.accrual_config.is_annualized_rebate:
                rebate *= contract_tenor or self.get_contract_tenor(pricing_env)   # KO-reset: _resolve_pre_contract_tenor
```

Order of multiplication: today it is `rate * price * multiplier * tenor` in one expression. Keep that order exactly, so the result stays bit-identical:

```python
            if self.accrual_config.is_annualized_rebate:
                tenor = contract_tenor or self.get_contract_tenor(pricing_env)
                rebate = (self.payoff_config.rebate_rate * self.initial_price * self.contract_multiplier * tenor)
            else:
                rebate = self.payoff_config.rebate_rate * self.initial_price * self.contract_multiplier
```

(The local named `contract_tenor` in `get_maturity_payoff_v0` shadows the imported function. Import the
module function as `from .accrual import contract_tenor as _contract_tenor` in `snowball_option.py`,
`phoenix_option.py` and `ko_reset_snowball_option.py`.)

`PhoenixOption.coupon_period_fraction(observation_idx)`:

```python
        records = self.barrier_config.ko_observation_schedule.records
        if not 0 <= observation_idx < len(records):
            raise ValidationError(f"coupon {observation_idx} has no knock-out record")
        return coupon_period_fraction(self, records[observation_idx])
```

`get_coupon_payoff`: unchanged except that the no-argument branch already calls `self.coupon_period_fraction`.
`get_ko_payoff`: replace the accrual block with
`accrual_factor = ko_accrual_factor(self, self.barrier_config.ko_observation_schedule.records[observation_idx], pricing_env)`.
Replace the coupon with `current_coupon = self.get_coupon_payoff(observation_idx)` when triggered.

KO-reset:

```python
    def compute_ko_accrual_factor(self, observation_time, schedule_record, pricing_env, *, knock_in=None) -> float:
        """The knock-out leg of ``schedule_record`` (``observation_time`` is unused: accrual is the record's own)."""
        return ko_accrual_factor(self, schedule_record, pricing_env, knock_in=knock_in)

    def _resolve_pre_contract_tenor(self, pricing_env: PricingEnv) -> float:
        return _contract_tenor(self, pricing_env, "pre")

    def _resolve_post_contract_tenor(self, pricing_env: PricingEnv, *, knock_in=None) -> float:
        return _contract_tenor(self, pricing_env, "post", knock_in=knock_in)

    def get_contract_tenor(self, pricing_env: PricingEnv = None) -> float:
        return _contract_tenor(self, pricing_env, "contract" if pricing_env is None else "pre")
```

`get_maturity_payoff_v1(self, spot, pricing_env=None, *, contract_tenor=None)`: the annualized-KI line becomes
`downside *= contract_tenor if contract_tenor is not None else self._resolve_post_contract_tenor(pricing_env, knock_in=getattr(self, "_otc_lifecycle_knock_in_anchor", None))`.
Delete `_resolve_schedule_end_date` once nothing reads it (`grep -rn _resolve_schedule_end_date quantark test`).

- [ ] **Step 5: Run the task's tests and the product suites**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time -q`
Expected: all pass.
Run: `.venv/bin/python -m pytest -n 6 -q test/test_snowball_option.py test/test_phoenix_option.py test/test_phoenix_coupon_accrual.py test/test_aged_accrual.py test/test_observation_accrual.py test/test_ko_reset_contract_rules.py > $WS/t4.log 2>&1; tail -30 $WS/t4.log`
Expected: failures only in the categories of §7. Each is recorded in the ledger as a sweep item for Task 9, not
fixed here: R5 refusal, a numeric-with-`initial_date` move, a §3.1–5 regression fix, or a test that mutates terms
after construction. Any other failure is a defect: fix it now.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(accrual): products read accrual through one resolver" -- quantark/asset/equity/product/option test/accrual_time/test_products.py
```

---

### Task 5: Monte Carlo REBASED: per-path anchors and the V1 tenor

**Files:**
- Modify: `quantark/asset/equity/engine/mc/snowball_mc_engine.py` (KO-reset grid builder ~1150–1172: add `ki_records`; `_compute_payoffs_ko_reset` REBASED KO accrual ~2093–2141 and V1 ~2180–2192)
- Test: `test/accrual_time/test_mc_rebased.py`

**Interfaces:**
- Consumes: `knock_in_anchor_for`, `ko_accrual_factor` / `product.compute_ko_accrual_factor(..., knock_in=)`, `get_maturity_payoff_v1(..., contract_tenor=)`.
- Produces: grid key `"ki_records"` (source KI records aligned with `ki_times`).

- [ ] **Step 1: Write the failing tests**

```python
"""MC's REBASED route pays the resolver's cash, per path (spec §5, §6)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.accrual import contract_tenor, knock_in_anchor_for, ko_accrual_factor
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.util.enum import PostKOScheduleMode as M

from .conftest import env, ko_reset, times


def _price_parts(product):
    engine = SnowballMCEngine(params=MCParams(seed=7, num_paths=4096))
    e = env(spot=100.0)
    engine.price(product, e)
    return engine, e


def test_rebased_mc_ko_cash_is_the_resolvers(monkeypatch):
    p = ko_reset(M.REBASED, accrual=AccrualConfig(accrued_offset=0.3))
    seen = []
    real = type(p).compute_ko_accrual_factor
    def spy(self, t, rec, env_, *, knock_in=None):
        value = real(self, t, rec, env_, knock_in=knock_in)
        seen.append((rec.accrual.end, None if knock_in is None else knock_in.fixing, value))
        return value
    monkeypatch.setattr(type(p), "compute_ko_accrual_factor", spy)
    _price_parts(p)
    rebased = [s for s in seen if s[1] is not None]
    assert rebased and all(v == f + end for end, f, v in rebased)


def test_rebased_v1_tenor_is_per_path_anchor():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(is_annualized_ki=True), payoff_config=PayoffConfig(rebate_rate=0.05))
    ki = p.barrier_config.ki_observation_schedule.records
    tenors = {contract_tenor(p, None, "post", knock_in=knock_in_anchor_for(r, i)) for i, r in enumerate(ki)}
    assert tenors == {1.0, 1.25}  # final offset 0.75 after fixings 0.25 and 0.5
    _price_parts(p)   # used to raise "ki_time is required for rebased post-KO maturity"


def test_unannualized_rebased_needs_no_anchor():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(is_annualized=False))
    post = p.post_barrier_config.ko_observation_schedule.records[0]
    assert ko_accrual_factor(p, post) == 1.0
    _price_parts(p)
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_mc_rebased.py -q`
Expected: the first test fails (the REBASED branch computes accrual inline; the spy sees no anchored calls); the second raises `ki_time is required`.

- [ ] **Step 3: Implement**

Grid builder (non-carried, discrete branch), next to `ki_times`:

```python
            ki_schedule = product.barrier_config.ki_observation_schedule
            ki_records = [ki_schedule.records[index] for index in in_horizon]
```

with `ki_records: List = []` initialised beside `ki_times`, and `"ki_records": ki_records` in the returned grid.

In `_compute_payoffs_ko_reset`, REBASED post-KO payoff: replace everything from `annualized_ko = ...` to `accrual = 1.0` with

```python
                ki_records = grid["ki_records"]
                anchors = [knock_in_anchor_for(rec, i) for i, rec in enumerate(ki_records)]
                accrual_table = np.array([
                    [float(product.compute_ko_accrual_factor(None, rec, pricing_env, knock_in=anchor))
                     for rec in post_records]
                    for anchor in anchors
                ], dtype=float)
                idx = np.flatnonzero(is_post_ko)
                local_idx = first_post_idx[idx]
                offsets = post_times[local_idx]
                rates = post_rates[local_idx]
                accrual = accrual_table[first_ki_idx[idx], local_idx]
```

The coupon line follows unchanged. Then remove the `calculate_year_fraction` import if it is unused.

V1 for REBASED:

```python
        if is_v1.any():
            terminal_spots = paths[is_v1, -1]
            if post_mode == PostKOScheduleMode.REBASED and product.accrual_config.is_annualized_ki:
                anchors = [knock_in_anchor_for(rec, i) for i, rec in enumerate(grid["ki_records"])]
                tenors = np.array([product._resolve_post_contract_tenor(pricing_env, knock_in=a) for a in anchors])
                path_tenor = tenors[first_ki_idx[is_v1]]
                v1_payoffs = np.array([product.get_maturity_payoff_v1(s, pricing_env, contract_tenor=t)
                                       for s, t in zip(terminal_spots, path_tenor)], dtype=float)
            else:
                v1_payoffs = np.array([product.get_maturity_payoff_v1(s, pricing_env) for s in terminal_spots],
                                      dtype=float)
```

(`first_ki_idx >= 0` on every V1 path of a non-carried REBASED contract: V1 means knocked in. A continuous KI is refused for REBASED.)

- [ ] **Step 4: Run them to see them pass, plus the KO-reset MC suite**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_mc_rebased.py -q && .venv/bin/python -m pytest -n 6 -q test/test_ko_reset_snowball_mc_engine.py`
Expected: pass. A moved REBASED price that has `initial_date` or `accrued_offset` is a §3.2 fix: list it for Task 9.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(accrual): MC REBASED reads each path's knock-in anchor for knock-out cash and the V1 tenor" -- quantark/asset/equity/engine/mc/snowball_mc_engine.py test/accrual_time/test_mc_rebased.py
```

---

### Task 6: One answer everywhere: QUAD V2 and intraday Phoenix fractions

**Files:**
- Modify: `quantark/asset/equity/engine/quad/v2/adapters.py` (Phoenix fractions block ~364–418)
- Modify: `quantark/intraday/events.py` (`phoenix_coupon_fractions`)
- Test: `test/accrual_time/test_one_answer.py`

**Interfaces:**
- Consumes: `PhoenixOption.coupon_period_fraction(i)`, `get_coupon_payoff(i)`.
- Produces: `phoenix_coupon_fractions(product, ko_times)`: same signature; `ko_times` is no longer read.

- [ ] **Step 1: Write the failing tests**

```python
"""The same accrual in the product, every engine and the intraday path (spec §8 'One answer everywhere')."""
from datetime import timedelta

import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.accrual import ko_accrual_factor
from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.intraday.events import phoenix_coupon_fractions

from .conftest import T0, dates, env, phoenix, snowball, times


def test_knock_out_at_the_valuation_instant_pays_the_resolvers_cash():
    """An aged contract (offset 0.25) whose first record is at time zero, spot above the barrier."""
    p = snowball(ko=times(0.0, 0.25), maturity=0.25, accrual=AccrualConfig(accrued_offset=0.25))
    cash = 100.0 + 100.0 * 0.12 * ko_accrual_factor(p, p.barrier_config.ko_observation_schedule.records[0])
    e = env(spot=110.0)
    for engine in (SnowballMCEngine(params=MCParams(seed=1, num_paths=512)), SnowballPDESolver(PDEParams()),
                   SnowballQuadEngine(params=QuadParams(grid_points=401))):
        assert engine.price(p, e) == pytest.approx(cash, rel=0, abs=1e-12), type(engine).__name__


@pytest.mark.parametrize("dated", [False, True])
def test_phoenix_fractions_agree(dated):
    kw = dict(ko=dates(30, 61, 91), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=91)) if dated \
        else dict(ko=times(1 / 12, 2 / 12, 3 / 12), accrual=AccrualConfig(accrued_offset=0.1, first_period_start=0.05))
    p = phoenix(**kw)
    product = p.get_coupon_period_year_fractions(env())
    intraday = phoenix_coupon_fractions(p, [None] * 3)
    assert product == intraday == [p.coupon_period_fraction(i) for i in range(3)]
    assert intraday[0] == (30 / 365 if dated else 0.1 + 1 / 12 - 0.05)
```

Add a QUAD V2 check. Price the same Phoenix with `PhoenixQuadEngineV2`, and assert that the compiled contract's
coupon amounts equal `[p.get_coupon_payoff(i) for i in active]`. Use `quantark.asset.equity.engine.quad.v2.adapters.compile_contract`:
read its signature first, and assert on the `coupon_prefix` differences it returns. Also add
`test_quad_v2_fractions_follow_source_index_after_phase` (Review Focus 4). It values the dated Phoenix on its second
observation date with `event_phase="after"`, and checks that the compiled coupons are those of records 2 and later
by source index.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_one_answer.py -q`
Expected: the numeric Phoenix case fails (intraday's first period is the raw `ko_times[0]`, §3.5).

- [ ] **Step 3: Implement**

`events.py`:

```python
def phoenix_coupon_fractions(product, ko_times: List[float]) -> List[float]:
    """Contractual per-period coupon fractions over the FULL KO schedule, from each record's accrual period
    (spec 2026-10-08 accrual time §6): the same answer as the product and QUAD V2. ``ko_times`` is unused."""
    n = len(product.barrier_config.ko_observation_schedule.records)
    return [float(product.coupon_period_fraction(i)) for i in range(n)]
```

`adapters.py`: replace the `factors = ...` block through `fractions = product.get_coupon_period_year_fractions(env)[:len(pre)]` with

```python
        fractions = [product.coupon_period_fraction(i) for i in active]
```

Then replace the dated `historic` branch, `elif (active and active[0] >= initial_memory and product.initial_date ...)`,
with one that reads the retained periods:

```python
            elif active and active[0] >= initial_memory:
                historic = [product.get_coupon_payoff(i) for i in range(active[0] - initial_memory, active[0])]
```

Keep the `fixed` branch first and the `NotImplementedError` last, unchanged.

- [ ] **Step 4: Run them to see them pass, plus the Phoenix and intraday suites**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time -q && .venv/bin/python -m pytest -n 6 -q test/test_phoenix_quad.py test/test_phoenix_coupon_accrual.py test/intraday`
Expected: `test/accrual_time` passes. In the other suites, any first-period move (§3.4–3.5) is listed for Task 9.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(accrual): QUAD V2 and intraday Phoenix fractions read the record periods" -- quantark/asset/equity/engine/quad/v2/adapters.py quantark/intraday/events.py test/accrual_time/test_one_answer.py
```

---

### Task 7: The intraday twin copies periods

**Files:**
- Modify: `quantark/intraday/twin.py` (`_autocallable_twin`, `_ko_reset_twin`)
- Test: `test/accrual_time/test_twin.py`

**Interfaces:**
- Consumes: record periods; `knock_in_anchor` on the state (Task 8 adds the field; read with `getattr(state, "knock_in_anchor", None)`).
- Produces: twin records carry their source record's `accrual` (by source index); twins no longer back-solve `accrual_factors` nor bank one `accrued_offset`.

- [ ] **Step 1: Write the failing tests**

First read how `test/intraday/test_twin*.py` builds a twin: the calendar, `ContractTimeline` and `twin_for(...)`
entry point. Reuse its fixtures (`intraday.conftest.dated_phoenix`, `flat_env`), then assert:

```python
def test_twin_records_carry_the_source_periods(...):
    twin = <build twin of dated_phoenix at a mid-life timestamp>
    src = product.barrier_config.ko_observation_schedule.records
    remaining = <indices of remaining KO events>
    assert [r.accrual for r in twin.barrier_config.ko_observation_schedule.records] == [src[i].accrual for i in remaining]


def test_dated_twin_resolves_the_source_cash_exactly(...):
    # initial_date is cleared on the twin; cash and coupons equal the source contract's to the bit
    assert [r.payoff for r in twin.resolve_ko_observations(cash_env)] == [e.cash for e in ko_events]
    assert [twin.get_coupon_payoff(k) for k in range(n)] == [product.get_coupon_payoff(i) for i in remaining]


def test_business_days_twin_needs_no_environment_calendar(...):
    # a BUSINESS_DAYS-annualized dated snowball with accrual_calendar/accrual_bus_days_in_year: its twin's cash,
    # verified under the calendar-free _cash_env, equals the source's
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_twin.py -q`
Expected: twin records carry `accrual=None` (the twin builds fresh records).

- [ ] **Step 3: Implement**

`_autocallable_twin`:
- Build each KO record with `accrual=src[e.index].accrual`, and each KI record with `accrual=ki_src[e.index].accrual`, where `ki_src = product.barrier_config.ki_observation_schedule.records`.
- Delete the `ko_factors` and `coupon_factors` back-solve and the Phoenix "one positional list" `CapabilityError`.
- Set `accrual_factors=_pick(product.accrual_config.accrual_factors, indices)` (positional consistency only; `None` stays `None`), and keep `accrued_offset` unchanged.
- Memory arrears are unchanged.

`_ko_reset_twin`:
- Delete the `offset` computation and its `CapabilityError`.
- Copy each record's `accrual` from `src[e.index]` in `schedule(...)`, and the KI records' likewise.
- Set `accrual_factors=_pick(..., pre_indices)` and `post_accrual_factors=_pick(..., post_indices)`; leave `accrued_offset` as is.
- `setattr(twin, "_otc_lifecycle_knock_in_anchor", getattr(state, "knock_in_anchor", None))`.
- The existing cash and tenor verification loops stay. They now compare periods with themselves, which turns them into equalities.

- [ ] **Step 4: Run them to see them pass, plus the intraday suite**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_twin.py -q && .venv/bin/python -m pytest -n 6 -q test/intraday`
Expected: pass; any moved intraday golden is listed for Task 9.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(accrual): the intraday twin carries its source records' periods" -- quantark/intraday/twin.py test/accrual_time/test_twin.py
```

---

### Task 8: The persisted knock-in anchor

**Files:**
- Modify: `quantark/asset/equity/lifecycle/state.py` (`AutocallableLifecycleState.knock_in_anchor`; `mark_ki(timestamp, anchor=None)`)
- Modify: `quantark/asset/equity/lifecycle/autocallable.py` (discrete KI branch builds the anchor; `product_for_lifecycle` / `product_for_pricing` set `_otc_lifecycle_knock_in_anchor`)
- Test: `test/accrual_time/test_anchor.py`

**Interfaces:**
- Consumes: `knock_in_anchor_for(record, source_index)`.
- Produces: `AutocallableLifecycleState.knock_in_anchor: Optional[KnockInAnchor] = None`; product attribute `_otc_lifecycle_knock_in_anchor`.

- [ ] **Step 1: Write the failing tests**

Model them on `test/test_snowball_lifecycle_ki.py`: read it for the tracker and run helpers. Then pin:
- **Persisted anchor.** A snowball with knock-in records due on days 91 and 93, observed first at close 95 with the spot below the barrier. The state's anchor is `source_index == 0` with `fixing == 91/365`, its period's end. `state.ki_date` is day 95.
- **It survives.** The anchor survives `product_for_pricing` past both records (on the copy as `_otc_lifecycle_knock_in_anchor`), `deepcopy(state)` and `pickle` round trips.
- **Fingerprint.** `lifecycle_fingerprint` renders a state carrying an anchor without raising, and differs from a state without one. The anchor is not bookkeeping: it changes at an event, like `ki_date`.

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_anchor.py -q`
Expected: `AttributeError: ... knock_in_anchor`.

- [ ] **Step 3: Implement**

`state.py`:

```python
    #: The knock-in a REBASED schedule hangs from (accrual time spec §5.1): resolved once, from the record whose
    #: test fired, and persisted, never looked up again. ``None`` before a knock-in, for a continuously
    #: monitored one, or for a non-accruing record.
    knock_in_anchor: Optional["KnockInAnchor"] = None

    def mark_ki(self, timestamp: datetime, anchor: Optional["KnockInAnchor"] = None) -> bool:
        if self.knocked_in or self.knocked_out:
            return False
        self.knocked_in = True
        self.ki_date = timestamp
        self.knock_in_anchor = anchor
        return True
```

`autocallable.py`, discrete KI branch:

```python
                    anchor = knock_in_anchor_for(product.barrier_config.ki_observation_schedule.records[idx], idx)
                    if self.lifecycle.mark_ki(pd.Timestamp(date).to_pydatetime(), anchor):
```

(`idx` indexes `_scheduled_records(product, env, "ki")`, which resolves the unaged `product_for_lifecycle()` copy's whole
schedule, so it is the original source index. Check that by reading `_scheduled_records` before relying on it.)
Both `product_for_*` add `setattr(product, "_otc_lifecycle_knock_in_anchor", self.lifecycle.knock_in_anchor)`.

If `fingerprints._normalize` cannot render a frozen dataclass holding a `Calendar`, extend `_normalize` with a dataclass branch: name plus fields, normalized recursively. `Calendar` already has its own branch.

- [ ] **Step 4: Run them to see them pass, plus lifecycle, pnl-explain and replay suites**

Run: `.venv/bin/python -m pytest -n0 test/accrual_time/test_anchor.py -q && .venv/bin/python -m pytest -n 6 -q test/test_snowball_lifecycle_ki.py test/test_equity_lifecycle_trackers.py test/pnlexplain`
Expected: pass. (A path that does not exist voids the run: check each path exists first.)

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(lifecycle): persist the knock-in anchor in the state and the pricing copy" -- quantark/asset/equity/lifecycle quantark/pnlexplain test/accrual_time/test_anchor.py
```

---

### Task 9: Compatibility sweep, goldens, CHANGELOG

**Files:**
- Modify: every test, example and golden the full suite shows to be in one of §7's categories (listed in the ledger, one line each)
- Modify: `CHANGELOG.md` (an `## [Unreleased]` section), `README.md` (a short "Accrual time" note under the autocallable section)

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -n 6 -q -p no:cacheprovider > $WS/full1.log 2>&1; tail -60 $WS/full1.log`
Expected: failures. Sort each one into exactly one bucket, and write one ledger line per test:

| bucket | action |
|---|---|
| R5: dated annualized without `initial_date` | give the fixture an `initial_date` (the contract's trade date, the fixture's valuation date when it has none), or `is_annualized=False` when the test is about something else; list it |
| §4.7: `BUSINESS_DAYS` annualized without terms | add `accrual_calendar` (the fixture env's calendar) and `accrual_bus_days_in_year` (its `bus_days_in_year`); list it |
| terms changed after construction (P-5) | rebuild the product with the term instead of assigning it; list it |
| moved value, §3.1–5 fix or numeric-with-`initial_date` | re-pin with the measured old→new and magnitude in the ledger and in the CHANGELOG |
| aged numeric ulp move | loosen nothing; re-pin the exact value, record the magnitude (≤ 1e-13 relative) |
| fingerprint-pinned golden | regenerate with the golden's documented capture command; list the file |
| known main failures (memory: 4 mo_volmodels/otc tests) | leave them |
| anything else | a defect: debug it (superpowers:systematic-debugging) and fix it in code; if the right answer is a semantic call, **stop and ask the user** |

A moved value outside these buckets is never re-pinned without the user (memory: a `test_book_backtest` anchor re-pin done without asking).

- [ ] **Step 2: Fix, re-run until green**

Run: `.venv/bin/python -m pytest -n 6 -q -p no:cacheprovider > $WS/full2.log 2>&1; tail -15 $WS/full2.log`
Expected: everything passes except the known main failures. Then `git checkout -- ` the two mo sample files if they changed (`git status --short example/`).

- [ ] **Step 3: CHANGELOG and README**

Add `## [Unreleased]` above `[0.10.0]`, in the house style: a paragraph naming the spec, then `### Added` (AccrualPeriod
/ resolver / `AccrualEndType` / `post_accrual_factors` / basis terms / persisted anchor), `### Changed` (R5 and
§4.7 refusals; numeric-with-`initial_date` accrues the quoted fraction; terms are frozen at construction), `### Fixed`
(§3 items 1–5 and MC REBASED's aged accrual and V1 tenor), `### Removed` (`AccrualConfig.accrual_dates`), and the
moved-value list from Step 1 with magnitudes.

- [ ] **Step 4: Commit**

```bash
git commit -m "test(accrual): compatibility sweep - fixtures given their accrual terms, moved values re-pinned and listed" -- <each listed path>
git commit -m "docs(changelog): accrual time separate from observation time" -- CHANGELOG.md README.md
```

---

## Self-review notes

- Spec coverage:
  - §4 data model: Tasks 1–2.
  - §4.2–4.7 construction: Task 3.
  - §5 resolver: Task 4.
  - §5.1 anchor: Tasks 4, 5 and 8.
  - §5.2 conversion: scope ruling P-1, with resolver controls in Task 4.
  - §6 consumers: products in Task 4, MC in 5, QUAD V2 and intraday events in 6, twin in 7.
  - §7 compatibility: Task 9.
  - §8 tests: spread across Tasks 1–8.
- The round-1 "carried knock-in date that matches no record is refused" control maps onto P-1. On `main`, a carried REBASED knock-in is refused outright, which covers it; the KO-reset branch refines this after the rebase.
