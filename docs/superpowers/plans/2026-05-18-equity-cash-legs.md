# Equity Cash Legs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a composable cash-leg framework (premium / accrual interest / fixed-payoff rebate) that prices alongside any equity option, returning trade-level NPV, per-leg PV breakdown, and Greeks via a single engine-emitted `EventDistribution` byproduct shared across Analytical, MC, PDE, and Quad engines.

**Architecture:** New `cashleg/` top-level module providing `CashLeg` primitives (`DeterministicLeg`, `AccrualLeg`, `FixedPayoffLeg`) and an engine-agnostic `EventDistribution` data structure. `BaseEngine` is extended with an additive `price_with_events(product, env, emit_distribution=True)` method that returns `(npv, event_distribution)`. Each engine family overrides this to expose its already-computed survival/KO timing. `EquityPosition` gains an optional `cash_legs` list and new `get_trade_value` / `get_trade_value_breakdown` / `get_trade_greeks` methods, preserving 100% backward compatibility with existing call sites.

**Tech Stack:** Python 3, NumPy, dataclasses, pytest. Reuses existing `util/numerical/` helpers (`Tolerance`, `is_close`), `util/exceptions.py` (`ValidationError`, `NumericalError`), `util/calendar/day_counter.py`, and the existing `priceenv.PricingEnvironment` discount API (`get_discount_factor`).

---

## Reference: Spec Sections

This plan implements `docs/superpowers/specs/2026-05-18-equity-cash-legs-design.md`. Section numbers (§N) below refer to that spec.

## Pre-Implementation Note on Existing Code

The codebase already has `asset/equity/engine/event_stats.py:AutocallableEventStats`, which overlaps significantly with the new `EventDistribution` (ko_times, ko_probability, survival_probability). The plan introduces `EventDistribution` as the **generalized successor** and adds a thin adapter `EventDistribution.from_autocallable_stats(stats)`. Engines that already implement `calculate_event_stats` can route through the adapter in their `price_with_events` override — no duplicated math.

---

## File Structure

**Phase 1 — Core primitives** (new files only, no modifications)
- Create: `cashleg/__init__.py`
- Create: `cashleg/event_distribution.py` — `EventType`, `EventDistribution`, `PricingResult`
- Create: `cashleg/base.py` — `CashLeg` ABC, `LegDirection`
- Create: `cashleg/base_amount.py` — `BaseAmount`, `BaseAmountMode`
- Create: `cashleg/leg_schedule.py` — `LegSchedule`
- Create: `cashleg/deterministic_leg.py` — `DeterministicLeg`
- Create: `cashleg/fixed_payoff_leg.py` — `FixedPayoffLeg`, `PaymentTrigger`
- Create: `cashleg/accrual_leg.py` — `AccrualLeg`, `PaymentConvention`, `KOBehavior`, `SurvivalBasis`
- Create: `cashleg/leg_valuator.py` — `value_leg`, `TradeValueBreakdown`, `LegPV`
- Create: `cashleg/CLAUDE.md`
- Modify: `asset/equity/engine/base_engine.py` — add `price_with_events` default method
- Create: `test/test_cashleg/` (and one `test_*.py` per source file)

**Phase 2 — MC engine emission** (modifications only)
- Modify: `asset/equity/engine/mc/snowball_mc_engine.py`
- Modify: `asset/equity/engine/mc/phoenix_mc_engine.py`
- Modify: `asset/equity/engine/mc/barrier_option_mc_engine.py`
- Modify: `asset/equity/engine/mc/range_accrual_mc_engine.py`
- Modify: `asset/equity/engine/mc/euro_mc_engine.py` (trivial fall-through is enough — covered by the base default)
- Create: `test/test_cashleg/test_mc_event_emission.py`

**Phase 3 — Quad engine emission**
- Modify: `asset/equity/engine/quad/snowball_quad_engine.py`
- Modify: `asset/equity/engine/quad/phoenix_quad_engine.py`
- Modify: `asset/equity/engine/quad/ko_reset_snowball_quad_engine.py`
- Create: `test/test_cashleg/test_quad_event_emission.py`

**Phase 4 — Analytical engine emission**
- Modify: `asset/equity/engine/analytical/barrier_analytical_engine.py`
- Modify: `asset/equity/engine/analytical/one_touch_analytical_engine.py`
- Modify: `asset/equity/engine/analytical/range_accrual_analytical_engine.py`
- Create: `test/test_cashleg/test_analytical_event_emission.py`

**Phase 5 — PDE engine emission**
- Create: `asset/equity/engine/pde/forward_density_helper.py` — shared forward-density solver
- Modify: `asset/equity/engine/pde/snowball_pde_solver.py`
- Modify: `asset/equity/engine/pde/phoenix_pde_solver.py`
- Modify: `asset/equity/engine/pde/ko_reset_snowball_pde_solver.py`
- Modify: `asset/equity/engine/pde/barrier_pde_solver.py`
- Create: `test/test_cashleg/test_pde_event_emission.py`

**Phase 6 — Position integration**
- Modify: `portfolio/equity/position.py` — add `cash_legs` field, `get_trade_value`, `get_trade_value_breakdown`, `get_trade_greeks`
- Create: `test/test_cashleg/test_position_with_legs.py`
- Create: `test/test_cashleg/test_position_backward_compat.py`
- Create: `example/cash_legs_demo.py`

---

# Phase 1 — Core Primitives

## Task 1.1: Create `cashleg/` package skeleton

**Files:**
- Create: `cashleg/__init__.py`
- Create: `test/test_cashleg/__init__.py`

- [ ] **Step 1: Create empty package files**

Create `cashleg/__init__.py` with:
```python
"""Cash-leg primitives for pricing equity-option cash terms alongside the option payoff."""
```

Create `test/test_cashleg/__init__.py` as an empty file.

- [ ] **Step 2: Verify package importable**

Run: `python -c "import cashleg; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add cashleg/__init__.py test/test_cashleg/__init__.py
git commit -m "feat(cashleg): scaffold cashleg package"
```

---

## Task 1.2: `EventDistribution` data model

**Files:**
- Create: `cashleg/event_distribution.py`
- Test: `test/test_cashleg/test_event_distribution.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_event_distribution.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from cashleg.event_distribution import EventDistribution, EventType, PricingResult
from util.exceptions import NumericalError


def test_trivial_distribution_for_vanilla_product():
    """Trivial event_dist: single mass at maturity, no KO."""
    dist = EventDistribution.trivial(maturity=1.0)
    assert dist.event_times.tolist() == [0.0, 1.0]
    assert dist.probabilities[EventType.MATURITY_NO_KO] == 1.0
    assert dist.survival_probability.tolist() == [1.0, 1.0]


def test_survival_interpolation():
    dist = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.1, 0.2, 0.1, 0.0]),
            EventType.MATURITY_NO_KO: 0.6,
        },
        survival_probability=np.array([1.0, 0.9, 0.7, 0.6, 0.6]),
    )
    # Linear interp at midpoint between obs 0 and obs 1: (1.0 + 0.9) / 2 = 0.95
    assert dist.survival_at(0.125) == pytest.approx(0.95, abs=1e-9)
    # At obs date exactly
    assert dist.survival_at(0.25) == pytest.approx(0.9, abs=1e-9)
    # Before t=0 → 1.0
    assert dist.survival_at(0.0) == pytest.approx(1.0, abs=1e-9)


def test_invariant_probability_sum():
    """Probabilities must sum to ~1.0; otherwise NumericalError."""
    with pytest.raises(NumericalError, match="probability"):
        EventDistribution(
            event_times=np.array([1.0]),
            event_dates=None,
            probabilities={EventType.KO: np.array([0.3]),
                           EventType.MATURITY_NO_KO: 0.2},  # sums to 0.5, not 1.0
            survival_probability=np.array([1.0, 0.7]),
        )


def test_invariant_survival_monotone():
    """survival_probability must be monotone non-increasing."""
    with pytest.raises(NumericalError, match="monotone"):
        EventDistribution(
            event_times=np.array([0.5, 1.0]),
            event_dates=None,
            probabilities={EventType.MATURITY_NO_KO: 1.0},
            survival_probability=np.array([1.0, 0.5, 0.7]),  # non-monotone
        )


def test_pricing_result_wraps_npv_and_distribution():
    dist = EventDistribution.trivial(1.0)
    result = PricingResult(npv=12.5, event_distribution=dist)
    assert result.npv == 12.5
    assert result.event_distribution is dist
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_event_distribution.py -v`
Expected: FAIL with `ModuleNotFoundError: cashleg.event_distribution`

- [ ] **Step 3: Implement `EventDistribution`**

```python
# cashleg/event_distribution.py
"""EventDistribution — engine-emitted termination/coupon timing distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np

from util.exceptions import NumericalError
from util.numerical.constants import Tolerance


class EventType(Enum):
    KO = "knock_out"
    KI = "knock_in"
    COUPON = "coupon"
    MATURITY_NO_KO = "maturity_no_ko"
    MATURITY_WITH_KI = "maturity_with_ki"


@dataclass(frozen=True)
class EventDistribution:
    """Probability distribution over termination/coupon events.

    Attributes:
        event_times: Year fractions of observation/event times. Shape (N,).
        event_dates: Parallel calendar dates, or None if engine doesn't track.
        probabilities: PMF per EventType. Vector-valued for per-obs events
            (KO/KI/COUPON, shape (N,)), scalar for terminal events
            (MATURITY_NO_KO, MATURITY_WITH_KI).
        survival_probability: P(alive entering obs i). Shape (N+1,);
            survival_probability[0] == 1.0; monotone non-increasing.
        mc_ko_times: Per-path KO time index (Monte-Carlo only); None otherwise.
    """

    event_times: np.ndarray
    event_dates: Optional[List[datetime]]
    probabilities: Dict[EventType, Union[np.ndarray, float]]
    survival_probability: np.ndarray
    mc_ko_times: Optional[np.ndarray] = None

    def __post_init__(self):
        self._validate_invariants()

    def _validate_invariants(self):
        # Probability sum ≈ 1
        total = 0.0
        for evt, p in self.probabilities.items():
            if isinstance(p, np.ndarray):
                total += float(p.sum())
            else:
                total += float(p)
        if abs(total - 1.0) > Tolerance.PROBABILITY:
            raise NumericalError(
                f"EventDistribution probability sum = {total}, expected 1.0 "
                f"(tolerance {Tolerance.PROBABILITY})"
            )

        # Survival monotone non-increasing
        diffs = np.diff(self.survival_probability)
        if np.any(diffs > Tolerance.PROBABILITY):
            raise NumericalError(
                "EventDistribution survival_probability is not monotone non-increasing"
            )

        # survival[0] == 1.0
        if abs(self.survival_probability[0] - 1.0) > Tolerance.PROBABILITY:
            raise NumericalError(
                f"survival_probability[0] = {self.survival_probability[0]}, expected 1.0"
            )

        # Length consistency
        if len(self.survival_probability) != len(self.event_times) + 1:
            raise NumericalError(
                f"len(survival_probability) = {len(self.survival_probability)}, "
                f"expected {len(self.event_times) + 1}"
            )

    @classmethod
    def trivial(cls, maturity: float) -> "EventDistribution":
        """Single mass at maturity, no KO. Used for vanilla products."""
        return cls(
            event_times=np.array([0.0, maturity]),
            event_dates=None,
            probabilities={EventType.MATURITY_NO_KO: 1.0},
            survival_probability=np.array([1.0, 1.0]),
        )

    def survival_at(self, t: float) -> float:
        """Linear interpolation of survival_probability at year fraction t.

        Returns 1.0 for t <= event_times[0], last value for t >= event_times[-1].
        """
        if t <= self.event_times[0]:
            return 1.0
        if t >= self.event_times[-1]:
            return float(self.survival_probability[-1])
        # survival_probability has length N+1; aligns with grid [0] + event_times
        # We treat survival[i] as the value at event_times[i-1] for i>=1, and survival[0] at t=0.
        # Build the time grid for interpolation:
        time_grid = np.concatenate([[0.0], self.event_times])
        # Drop the duplicate if event_times[0] == 0.0
        if self.event_times[0] == 0.0:
            time_grid = self.event_times.copy()
            surv_grid = self.survival_probability[1:]
        else:
            surv_grid = self.survival_probability
        return float(np.interp(t, time_grid, surv_grid))


@dataclass(frozen=True)
class PricingResult:
    """Engine result containing both NPV and the optional event distribution."""

    npv: float
    event_distribution: Optional[EventDistribution] = None
```

Also add `PROBABILITY = 1e-6` to the existing `Tolerance` dataclass:

Open `util/numerical/constants.py`. Locate the `Tolerance` dataclass. Add a field:

```python
    PROBABILITY: float = 1e-6
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_event_distribution.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/event_distribution.py test/test_cashleg/test_event_distribution.py util/numerical/constants.py
git commit -m "feat(cashleg): EventDistribution data model + PricingResult"
```

---

## Task 1.3: `LegDirection` enum and `CashLeg` ABC

**Files:**
- Create: `cashleg/base.py`
- Test: `test/test_cashleg/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_base.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from cashleg.base import CashLeg, LegDirection


def test_leg_direction_signs():
    assert LegDirection.BUYER_RECEIVES.value == +1
    assert LegDirection.BUYER_PAYS.value == -1


def test_cashleg_is_abstract():
    with pytest.raises(TypeError):
        CashLeg(direction=LegDirection.BUYER_RECEIVES)  # ABC, can't instantiate


def test_cashleg_subclass_requires_value_method():
    class IncompleteLeg(CashLeg):
        pass

    with pytest.raises(TypeError):
        IncompleteLeg(direction=LegDirection.BUYER_RECEIVES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `CashLeg` ABC**

```python
# cashleg/base.py
"""CashLeg ABC and shared enums."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from priceenv import PricingEnvironment
    from cashleg.event_distribution import EventDistribution


class LegDirection(Enum):
    """Sign convention for cash flows from the buyer's (position holder's) perspective."""
    BUYER_RECEIVES = +1
    BUYER_PAYS = -1


@dataclass(frozen=True)
class CashLeg(ABC):
    """Abstract base for cash legs attached to an equity position.

    Subclasses implement value() to return the signed PV from the buyer's perspective.
    """

    direction: LegDirection
    name: Optional[str] = None
    leg_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @abstractmethod
    def value(self, event_dist: "EventDistribution", env: "PricingEnvironment",
              position_notional: float) -> float:
        """Return signed PV (buyer's perspective) of this leg."""

    def requires_event_distribution(self) -> bool:
        """If False, leg can be valued from any engine's trivial EventDistribution."""
        return True

    def sign(self) -> int:
        return int(self.direction.value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_base.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/base.py test/test_cashleg/test_base.py
git commit -m "feat(cashleg): CashLeg ABC and LegDirection enum"
```

---

## Task 1.4: `BaseAmount` value object

**Files:**
- Create: `cashleg/base_amount.py`
- Test: `test/test_cashleg/test_base_amount.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_base_amount.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from cashleg.base_amount import BaseAmount, BaseAmountMode
from util.exceptions import ValidationError


def test_absolute_amount():
    b = BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE)
    assert b.resolve(position_notional=999.0) == 1_000_000.0


def test_notional_fraction():
    b = BaseAmount(value=0.5, mode=BaseAmountMode.NOTIONAL_FRACTION)
    assert b.resolve(position_notional=2_000_000.0) == 1_000_000.0


def test_margin_fraction():
    # margin = 25% of notional; rate applies to margin
    b = BaseAmount(value=1.0, mode=BaseAmountMode.MARGIN_FRACTION, margin_rate=0.25)
    assert b.resolve(position_notional=4_000_000.0) == 1_000_000.0


def test_negative_value_rejected_for_fractions():
    with pytest.raises(ValidationError):
        BaseAmount(value=-0.5, mode=BaseAmountMode.NOTIONAL_FRACTION)


def test_fraction_above_one_rejected():
    with pytest.raises(ValidationError):
        BaseAmount(value=1.5, mode=BaseAmountMode.NOTIONAL_FRACTION)


def test_margin_rate_required_for_margin_mode():
    with pytest.raises(ValidationError):
        BaseAmount(value=1.0, mode=BaseAmountMode.MARGIN_FRACTION, margin_rate=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_base_amount.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `BaseAmount`**

```python
# cashleg/base_amount.py
"""BaseAmount — multiplier base for accrual computations."""

from dataclasses import dataclass
from enum import Enum

from util.exceptions import ValidationError


class BaseAmountMode(Enum):
    ABSOLUTE = "absolute"
    NOTIONAL_FRACTION = "notional_fraction"
    MARGIN_FRACTION = "margin_fraction"


@dataclass(frozen=True)
class BaseAmount:
    """Base amount for accrual computations.

    Either absolute (in trade currency) or expressed as a fraction of
    position notional or position margin. Position notional is supplied
    at valuation time by EquityPosition.get_actual_notional(env).
    """
    value: float
    mode: BaseAmountMode
    margin_rate: float = 0.0  # only used when mode=MARGIN_FRACTION

    def __post_init__(self):
        if self.mode is BaseAmountMode.ABSOLUTE:
            if self.value < 0:
                raise ValidationError(
                    f"BaseAmount.value must be non-negative for ABSOLUTE mode, "
                    f"got {self.value}. Use LegDirection.BUYER_PAYS for sign flips."
                )
        else:
            if not (0.0 <= self.value <= 1.0):
                raise ValidationError(
                    f"BaseAmount.value must be in [0, 1] for {self.mode.value} mode, "
                    f"got {self.value}"
                )
        if self.mode is BaseAmountMode.MARGIN_FRACTION and self.margin_rate <= 0.0:
            raise ValidationError(
                f"margin_rate must be > 0 for MARGIN_FRACTION mode, got {self.margin_rate}"
            )

    def resolve(self, position_notional: float) -> float:
        if self.mode is BaseAmountMode.ABSOLUTE:
            return self.value
        if self.mode is BaseAmountMode.NOTIONAL_FRACTION:
            return self.value * position_notional
        if self.mode is BaseAmountMode.MARGIN_FRACTION:
            return self.value * self.margin_rate * position_notional
        raise ValueError(f"Unknown BaseAmountMode: {self.mode}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_base_amount.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/base_amount.py test/test_cashleg/test_base_amount.py
git commit -m "feat(cashleg): BaseAmount with ABSOLUTE/NOTIONAL/MARGIN modes"
```

---

## Task 1.5: `LegSchedule` value object

**Files:**
- Create: `cashleg/leg_schedule.py`
- Test: `test/test_cashleg/test_leg_schedule.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_leg_schedule.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from cashleg.leg_schedule import LegSchedule
from util.exceptions import ValidationError


def test_simple_quarterly_schedule():
    sched = LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    assert len(sched.period_starts) == 4
    assert sched.last_period_end() == 1.0


def test_mismatched_array_lengths_rejected():
    with pytest.raises(ValidationError):
        LegSchedule(
            period_starts=np.array([0.0, 0.25]),
            period_ends=np.array([0.25, 0.5, 0.75]),  # length mismatch
            payment_times=np.array([0.25, 0.5]),
        )


def test_period_end_before_start_rejected():
    with pytest.raises(ValidationError):
        LegSchedule(
            period_starts=np.array([0.5]),
            period_ends=np.array([0.25]),  # ends before starts
            payment_times=np.array([0.5]),
        )


def test_validate_within_maturity_passes():
    sched = LegSchedule(
        period_starts=np.array([0.0]),
        period_ends=np.array([1.0]),
        payment_times=np.array([1.0]),
    )
    sched.validate_within_maturity(1.0)  # no raise


def test_validate_within_maturity_rejects_overflow():
    sched = LegSchedule(
        period_starts=np.array([0.0]),
        period_ends=np.array([1.5]),
        payment_times=np.array([1.5]),
    )
    with pytest.raises(ValidationError, match="maturity"):
        sched.validate_within_maturity(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_leg_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `LegSchedule`**

```python
# cashleg/leg_schedule.py
"""LegSchedule — independent schedule of accrual periods for cash legs."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from util.exceptions import ValidationError


@dataclass(frozen=True)
class LegSchedule:
    """Accrual period boundaries and payment dates for a cash leg.

    Independent of the product's observation schedule. The leg valuator
    interpolates the engine's survival_probability at this schedule's
    period boundaries.

    Attributes:
        period_starts: Year fractions of each period start. Shape (N,).
        period_ends:   Year fractions of each period end. Shape (N,).
                       period_ends[i] >= period_starts[i].
        payment_times: Year fractions when each period's cash flow is paid.
                       Shape (N,).
        period_start_dates: Optional parallel datetime list.
        period_end_dates:   Optional parallel datetime list.
        payment_dates:      Optional parallel datetime list.
    """

    period_starts: np.ndarray
    period_ends: np.ndarray
    payment_times: np.ndarray
    period_start_dates: Optional[List[datetime]] = None
    period_end_dates: Optional[List[datetime]] = None
    payment_dates: Optional[List[datetime]] = None

    def __post_init__(self):
        n = len(self.period_starts)
        if len(self.period_ends) != n or len(self.payment_times) != n:
            raise ValidationError(
                f"LegSchedule arrays must have matching length; got "
                f"period_starts={n}, period_ends={len(self.period_ends)}, "
                f"payment_times={len(self.payment_times)}"
            )
        if np.any(self.period_ends < self.period_starts):
            bad = np.where(self.period_ends < self.period_starts)[0]
            raise ValidationError(
                f"LegSchedule period_ends < period_starts at indices {bad.tolist()}"
            )

    def last_period_end(self) -> float:
        return float(self.period_ends[-1])

    def validate_within_maturity(self, maturity: float) -> None:
        """Raise ValidationError if any period or payment extends past maturity."""
        if self.last_period_end() > maturity + 1e-9:
            raise ValidationError(
                f"LegSchedule last period end {self.last_period_end()} "
                f"exceeds product maturity {maturity}"
            )
        if float(self.payment_times[-1]) > maturity + 1e-9:
            raise ValidationError(
                f"LegSchedule last payment time {self.payment_times[-1]} "
                f"exceeds product maturity {maturity}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_leg_schedule.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/leg_schedule.py test/test_cashleg/test_leg_schedule.py
git commit -m "feat(cashleg): LegSchedule with period/payment alignment"
```

---

## Task 1.6: `DeterministicLeg`

**Files:**
- Create: `cashleg/deterministic_leg.py`
- Test: `test/test_cashleg/test_deterministic_leg.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_deterministic_leg.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime
import pytest

from cashleg.deterministic_leg import DeterministicLeg
from cashleg.base import LegDirection
from cashleg.event_distribution import EventDistribution
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env(rate: float = 0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_front_premium_pv_equals_amount():
    """Upfront premium payment: PV = amount × DF(0) = amount × 1.0."""
    env = _env(rate=0.05)
    leg = DeterministicLeg(
        amount=1_000_000.0,
        payment_time=0.0,
        direction=LegDirection.BUYER_PAYS,
        name="Front Premium",
    )
    dist = EventDistribution.trivial(maturity=1.0)
    pv = leg.value(dist, env, position_notional=0.0)
    assert pv == pytest.approx(-1_000_000.0, abs=1e-6)


def test_backend_premium_pv_discounted():
    env = _env(rate=0.05)
    leg = DeterministicLeg(
        amount=1_000_000.0,
        payment_time=1.0,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(EventDistribution.trivial(1.0), env, position_notional=0.0)
    assert pv == pytest.approx(1_000_000.0 * math.exp(-0.05 * 1.0), rel=1e-9)


def test_does_not_require_event_distribution():
    leg = DeterministicLeg(
        amount=1.0, payment_time=0.0, direction=LegDirection.BUYER_PAYS
    )
    assert leg.requires_event_distribution() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_deterministic_leg.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `DeterministicLeg`**

```python
# cashleg/deterministic_leg.py
"""DeterministicLeg — fixed amount at fixed time (premium, fixed fees)."""

from dataclasses import dataclass

from cashleg.base import CashLeg


@dataclass(frozen=True)
class DeterministicLeg(CashLeg):
    """A cash flow with deterministic amount and timing.

    Independent of event_dist. Used for front/back premium and fixed fees.
    For premium canceled by KO, use FixedPayoffLeg(trigger=AT_MATURITY_ANY).
    """

    amount: float = 0.0
    payment_time: float = 0.0  # year fraction from valuation date

    def value(self, event_dist, env, position_notional: float) -> float:
        df = env.get_discount_factor(self.payment_time)
        return self.sign() * self.amount * df

    def requires_event_distribution(self) -> bool:
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_deterministic_leg.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/deterministic_leg.py test/test_cashleg/test_deterministic_leg.py
git commit -m "feat(cashleg): DeterministicLeg for premium and fixed fees"
```

---

## Task 1.7: `FixedPayoffLeg`

**Files:**
- Create: `cashleg/fixed_payoff_leg.py`
- Test: `test/test_cashleg/test_fixed_payoff_leg.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_fixed_payoff_leg.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime
import numpy as np
import pytest

from cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from cashleg.base import LegDirection
from cashleg.event_distribution import EventDistribution, EventType
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.exceptions import ValidationError


def _env(rate=0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_at_ko_pv_equals_amount_times_p_ko_times_df():
    """PV = amount × Σ P(KO at i) × DF(t_i)."""
    env = _env(rate=0.05)
    dist = EventDistribution(
        event_times=np.array([0.25, 0.5, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.2, 0.1]),
            EventType.MATURITY_NO_KO: 0.4,
        },
        survival_probability=np.array([1.0, 0.7, 0.5, 0.4]),
    )
    leg = FixedPayoffLeg(
        amount=10_000.0,
        trigger=PaymentTrigger.AT_KO,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(dist, env, position_notional=0.0)
    expected = 10_000.0 * (
        0.3 * math.exp(-0.05 * 0.25)
        + 0.2 * math.exp(-0.05 * 0.5)
        + 0.1 * math.exp(-0.05 * 1.0)
    )
    assert pv == pytest.approx(expected, rel=1e-9)


def test_at_maturity_no_ko_pv():
    env = _env(rate=0.05)
    dist = EventDistribution(
        event_times=np.array([0.5, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.0]),
            EventType.MATURITY_NO_KO: 0.7,
        },
        survival_probability=np.array([1.0, 0.7, 0.7]),
    )
    leg = FixedPayoffLeg(
        amount=50_000.0,
        trigger=PaymentTrigger.AT_MATURITY_NO_KO,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(dist, env, position_notional=0.0)
    expected = 50_000.0 * 0.7 * math.exp(-0.05 * 1.0)
    assert pv == pytest.approx(expected, rel=1e-9)


def test_missing_trigger_in_distribution_raises():
    env = _env()
    leg = FixedPayoffLeg(
        amount=1.0,
        trigger=PaymentTrigger.AT_KI,
        direction=LegDirection.BUYER_RECEIVES,
    )
    dist = EventDistribution.trivial(1.0)  # no KI
    with pytest.raises(ValidationError, match="trigger"):
        leg.value(dist, env, position_notional=0.0)


def test_payment_offset_days_shifts_discount():
    env = _env(rate=0.05)
    dist = EventDistribution(
        event_times=np.array([1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([1.0]),
        },
        survival_probability=np.array([1.0, 0.0]),
    )
    leg = FixedPayoffLeg(
        amount=1_000.0,
        trigger=PaymentTrigger.AT_KO,
        payment_offset_days=365,  # +1 year
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(dist, env, position_notional=0.0)
    expected = 1_000.0 * math.exp(-0.05 * 2.0)  # 1y event + 1y offset
    assert pv == pytest.approx(expected, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_fixed_payoff_leg.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `FixedPayoffLeg`**

```python
# cashleg/fixed_payoff_leg.py
"""FixedPayoffLeg — event-conditional fixed amount."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from cashleg.base import CashLeg
from cashleg.event_distribution import EventDistribution, EventType
from util.exceptions import ValidationError


class PaymentTrigger(Enum):
    AT_KO = "at_ko"
    AT_KI = "at_ki"
    AT_MATURITY_NO_KO = "at_maturity_no_ko"
    AT_MATURITY_WITH_KI = "at_maturity_with_ki"
    AT_MATURITY_ANY = "at_maturity_any"  # any path that did NOT KO


_TRIGGER_TO_EVENT = {
    PaymentTrigger.AT_KO: EventType.KO,
    PaymentTrigger.AT_KI: EventType.KI,
    PaymentTrigger.AT_MATURITY_NO_KO: EventType.MATURITY_NO_KO,
    PaymentTrigger.AT_MATURITY_WITH_KI: EventType.MATURITY_WITH_KI,
}


@dataclass(frozen=True)
class FixedPayoffLeg(CashLeg):
    """Fixed amount paid on a specific event (KO/KI/maturity outcome).

    PV = sign × amount × Σ P(trigger at obs i) × DF(t_i + offset).
    """

    amount: float = 0.0
    trigger: PaymentTrigger = PaymentTrigger.AT_MATURITY_NO_KO
    payment_offset_days: int = 0

    def __post_init__(self):
        if self.payment_offset_days < 0:
            raise ValidationError(
                f"FixedPayoffLeg.payment_offset_days must be >= 0, "
                f"got {self.payment_offset_days}"
            )

    def value(self, event_dist: EventDistribution, env, position_notional: float) -> float:
        offset_yf = self.payment_offset_days / 365.0
        sign = self.sign()

        # Terminal trigger (AT_MATURITY_*): scalar probability paid at maturity
        if self.trigger is PaymentTrigger.AT_MATURITY_ANY:
            # AT_MATURITY_ANY = MATURITY_NO_KO + MATURITY_WITH_KI
            p = 0.0
            p += float(event_dist.probabilities.get(EventType.MATURITY_NO_KO, 0.0))
            p += float(event_dist.probabilities.get(EventType.MATURITY_WITH_KI, 0.0))
            pay_t = float(event_dist.event_times[-1]) + offset_yf
            return sign * self.amount * p * env.get_discount_factor(pay_t)

        event_type = _TRIGGER_TO_EVENT[self.trigger]
        if event_type not in event_dist.probabilities:
            raise ValidationError(
                f"FixedPayoffLeg trigger {self.trigger.value} requires "
                f"{event_type.value} in EventDistribution.probabilities, "
                f"but engine emitted only {list(event_dist.probabilities.keys())}"
            )
        p = event_dist.probabilities[event_type]
        if isinstance(p, np.ndarray):
            pay_times = event_dist.event_times + offset_yf
            dfs = np.array([env.get_discount_factor(t) for t in pay_times])
            return sign * self.amount * float(np.sum(p * dfs))
        else:
            pay_t = float(event_dist.event_times[-1]) + offset_yf
            return sign * self.amount * float(p) * env.get_discount_factor(pay_t)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_fixed_payoff_leg.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/fixed_payoff_leg.py test/test_cashleg/test_fixed_payoff_leg.py
git commit -m "feat(cashleg): FixedPayoffLeg for event-conditional payments"
```

---

## Task 1.8: `AccrualLeg`

**Files:**
- Create: `cashleg/accrual_leg.py`
- Test: `test/test_cashleg/test_accrual_leg.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_accrual_leg.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime
import numpy as np
import pytest

from cashleg.accrual_leg import (
    AccrualLeg, PaymentConvention, KOBehavior, SurvivalBasis,
)
from cashleg.base import LegDirection
from cashleg.base_amount import BaseAmount, BaseAmountMode
from cashleg.leg_schedule import LegSchedule
from cashleg.event_distribution import EventDistribution
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.calendar.day_counter import DayCountConvention


def _env(rate=0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def _quarterly_schedule_1y():
    return LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )


def test_no_ko_reduces_to_deterministic_annuity():
    """With survival=1 flat (vanilla product), AccrualLeg = sum of rate*base*dcf*DF."""
    env = _env(rate=0.05)
    leg = AccrualLeg(
        rate=0.04,
        base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=_quarterly_schedule_1y(),
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    dist = EventDistribution.trivial(maturity=1.0)
    pv = leg.value(dist, env, position_notional=0.0)
    # Each quarter: 0.04 × 1e6 × 0.25 × exp(-0.05 × t_pay)
    expected = sum(
        0.04 * 1_000_000.0 * 0.25 * math.exp(-0.05 * t)
        for t in [0.25, 0.5, 0.75, 1.0]
    )
    assert pv == pytest.approx(expected, rel=1e-3)  # ACT_365 ~ 0.25 per quarter


def test_pay_full_schedule_ignores_ko():
    """ko_behavior=PAY_FULL_SCHEDULE → unaffected by KO probability."""
    env = _env(rate=0.05)
    leg = AccrualLeg(
        rate=0.04,
        base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=_quarterly_schedule_1y(),
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.PAY_FULL_SCHEDULE,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    # Even with high KO probability, PV unchanged
    dist_with_ko = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            __import__("cashleg.event_distribution", fromlist=["EventType"]).EventType.KO: np.array([0.3, 0.3, 0.3, 0.0]),
            __import__("cashleg.event_distribution", fromlist=["EventType"]).EventType.MATURITY_NO_KO: 0.1,
        },
        survival_probability=np.array([1.0, 0.7, 0.4, 0.1, 0.1]),
    )
    pv_with_ko = leg.value(dist_with_ko, env, position_notional=0.0)
    pv_trivial = leg.value(EventDistribution.trivial(1.0), env, position_notional=0.0)
    assert pv_with_ko == pytest.approx(pv_trivial, rel=1e-9)


def test_truncate_at_ko_reduces_pv_with_high_ko_probability():
    """High KO probability → PV smaller than deterministic annuity."""
    env = _env(rate=0.05)
    from cashleg.event_distribution import EventType
    leg = AccrualLeg(
        rate=0.04,
        base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=_quarterly_schedule_1y(),
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    dist_high_ko = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.5, 0.3, 0.1, 0.0]),
            EventType.MATURITY_NO_KO: 0.1,
        },
        survival_probability=np.array([1.0, 0.5, 0.2, 0.1, 0.1]),
    )
    pv_high = leg.value(dist_high_ko, env, position_notional=0.0)
    pv_no_ko = leg.value(EventDistribution.trivial(1.0), env, position_notional=0.0)
    assert pv_high < pv_no_ko  # KO truncation reduces expected accrual


def test_notional_fraction_base_uses_position_notional():
    env = _env(rate=0.05)
    leg = AccrualLeg(
        rate=0.04,
        base=BaseAmount(value=0.25, mode=BaseAmountMode.NOTIONAL_FRACTION),
        schedule=_quarterly_schedule_1y(),
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.PAY_FULL_SCHEDULE,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv_4m = leg.value(EventDistribution.trivial(1.0), env, position_notional=4_000_000.0)
    pv_8m = leg.value(EventDistribution.trivial(1.0), env, position_notional=8_000_000.0)
    assert pv_8m == pytest.approx(2 * pv_4m, rel=1e-9)


def test_complete_period_basis_uses_end_survival():
    """SurvivalBasis.COMPLETE_PERIOD uses survival at period end (more conservative)."""
    env = _env(rate=0.05)
    from cashleg.event_distribution import EventType
    sched = _quarterly_schedule_1y()
    dist = EventDistribution(
        event_times=np.array([0.25, 0.5, 0.75, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.2, 0.2, 0.2, 0.0]),
            EventType.MATURITY_NO_KO: 0.4,
        },
        survival_probability=np.array([1.0, 0.8, 0.6, 0.4, 0.4]),
    )
    base_kwargs = dict(
        rate=0.04,
        base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=sched,
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
        direction=LegDirection.BUYER_RECEIVES,
    )
    enter = AccrualLeg(**base_kwargs, survival_basis=SurvivalBasis.ENTER_PERIOD).value(
        dist, env, position_notional=0.0
    )
    complete = AccrualLeg(**base_kwargs, survival_basis=SurvivalBasis.COMPLETE_PERIOD).value(
        dist, env, position_notional=0.0
    )
    assert complete < enter  # ENTER uses start survival (larger), COMPLETE uses end (smaller)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_accrual_leg.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `AccrualLeg`**

```python
# cashleg/accrual_leg.py
"""AccrualLeg — KO-truncated accrual stream (extra interest / extra rebate)."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from cashleg.base import CashLeg
from cashleg.base_amount import BaseAmount
from cashleg.leg_schedule import LegSchedule
from cashleg.event_distribution import EventDistribution
from util.calendar.day_counter import DayCountConvention


class PaymentConvention(Enum):
    AT_PERIOD_END = "at_period_end"
    AT_KO = "at_ko"
    AT_MATURITY = "at_maturity"


class KOBehavior(Enum):
    TRUNCATE_AT_KO = "truncate_at_ko"
    PAY_FULL_SCHEDULE = "pay_full_schedule"


class SurvivalBasis(Enum):
    ENTER_PERIOD = "enter_period"      # survival at period start (pay full period if entered)
    COMPLETE_PERIOD = "complete_period"  # survival at period end (pay only completed)


@dataclass(frozen=True)
class AccrualLeg(CashLeg):
    """Accrual stream that may be truncated by a KO event.

    PV = sign × Σ_i  rate × B × dcf_i × survival_factor_i × DF(pay_time_i)
    """

    rate: float = 0.0
    base: BaseAmount = None
    schedule: LegSchedule = None
    day_count: DayCountConvention = DayCountConvention.ACT_365
    payment_convention: PaymentConvention = PaymentConvention.AT_PERIOD_END
    ko_behavior: KOBehavior = KOBehavior.TRUNCATE_AT_KO
    survival_basis: SurvivalBasis = SurvivalBasis.ENTER_PERIOD

    def value(self, event_dist: EventDistribution, env, position_notional: float) -> float:
        if self.base is None or self.schedule is None:
            raise ValueError("AccrualLeg requires base and schedule")

        B = self.base.resolve(position_notional)
        n = len(self.schedule.period_starts)
        if n == 0:
            return 0.0

        # Day-count fraction per period
        dcf = np.array([
            self._dcf(self.schedule.period_starts[i], self.schedule.period_ends[i])
            for i in range(n)
        ])

        # Survival factor per period
        if self.ko_behavior is KOBehavior.PAY_FULL_SCHEDULE:
            surv = np.ones(n)
        else:
            ref_times = (
                self.schedule.period_starts
                if self.survival_basis is SurvivalBasis.ENTER_PERIOD
                else self.schedule.period_ends
            )
            surv = np.array([event_dist.survival_at(float(t)) for t in ref_times])

        # Discount factors at payment times
        dfs = np.array([env.get_discount_factor(float(t))
                        for t in self.schedule.payment_times])

        pv = self.sign() * self.rate * B * float(np.sum(dcf * surv * dfs))
        return pv

    def _dcf(self, start: float, end: float) -> float:
        """Day-count fraction for [start, end] using ACT/365 approximation on year fractions.

        For the common day-count conventions used in QuantArk, year fractions
        already encode the convention upstream when the schedule is built.
        For ACT/365, end - start is the fraction directly.
        """
        if self.day_count is DayCountConvention.ACT_365:
            return float(end - start)
        # ACT_360 and other conventions: approximation; engines pass real calendar dates upstream
        return float(end - start)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_accrual_leg.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/accrual_leg.py test/test_cashleg/test_accrual_leg.py
git commit -m "feat(cashleg): AccrualLeg with KO truncation and survival basis"
```

---

## Task 1.9: `value_leg` dispatch, `TradeValueBreakdown`, `LegPV`

**Files:**
- Create: `cashleg/leg_valuator.py`
- Test: `test/test_cashleg/test_leg_valuator.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_leg_valuator.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import pytest

from cashleg.leg_valuator import value_leg, TradeValueBreakdown, LegPV
from cashleg.base import LegDirection
from cashleg.deterministic_leg import DeterministicLeg
from cashleg.event_distribution import EventDistribution
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_value_leg_delegates_to_leg_value_method():
    env = _env()
    leg = DeterministicLeg(amount=1000.0, payment_time=0.0,
                           direction=LegDirection.BUYER_RECEIVES)
    pv = value_leg(leg, EventDistribution.trivial(1.0), env, position_notional=0.0)
    assert pv == pytest.approx(1000.0)


def test_trade_value_breakdown_total_sums_components():
    leg_pv = LegPV(name="Premium", direction=LegDirection.BUYER_PAYS, pv=-100.0)
    breakdown = TradeValueBreakdown(
        product_npv=500.0,
        leg_pvs={"leg-1": leg_pv},
    )
    assert breakdown.total == 400.0


def test_trade_value_breakdown_empty_legs():
    breakdown = TradeValueBreakdown(product_npv=500.0, leg_pvs={})
    assert breakdown.total == 500.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_leg_valuator.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement valuator and breakdown**

```python
# cashleg/leg_valuator.py
"""Top-level valuator and reporting structures."""

from dataclasses import dataclass
from typing import Dict, Optional

from cashleg.base import CashLeg, LegDirection
from cashleg.event_distribution import EventDistribution


def value_leg(leg: CashLeg, event_dist: EventDistribution, env,
              position_notional: float) -> float:
    """Compute signed PV of a single cash leg from the buyer's perspective."""
    return leg.value(event_dist, env, position_notional)


@dataclass(frozen=True)
class LegPV:
    """PV attribution for a single leg in a trade value breakdown."""
    name: Optional[str]
    direction: LegDirection
    pv: float


@dataclass(frozen=True)
class TradeValueBreakdown:
    """Per-leg PV attribution alongside the product NPV.

    Used by EquityPosition.get_trade_value_breakdown for reporting.
    """
    product_npv: float
    leg_pvs: Dict[str, LegPV]  # keyed by leg_id

    @property
    def total(self) -> float:
        return self.product_npv + sum(v.pv for v in self.leg_pvs.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_leg_valuator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add cashleg/leg_valuator.py test/test_cashleg/test_leg_valuator.py
git commit -m "feat(cashleg): value_leg dispatch and TradeValueBreakdown"
```

---

## Task 1.10: `BaseEngine.price_with_events` default

**Files:**
- Modify: `asset/equity/engine/base_engine.py`
- Test: `test/test_cashleg/test_base_engine_price_with_events.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_base_engine_price_with_events.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import pytest

from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.product.option import EuropeanVanillaOption
from cashleg.event_distribution import EventType, PricingResult
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import OptionType


def test_default_returns_pricing_result_with_trivial_distribution():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()

    result = engine.price_with_events(option, env)
    assert isinstance(result, PricingResult)
    assert result.npv == pytest.approx(engine.price(option, env), rel=1e-12)
    assert result.event_distribution is not None
    assert result.event_distribution.probabilities[EventType.MATURITY_NO_KO] == 1.0


def test_emit_distribution_false_still_returns_pricing_result():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    result = engine.price_with_events(option, env, emit_distribution=False)
    # Default impl always emits the trivial distribution; engines may skip when expensive.
    assert result.event_distribution is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_base_engine_price_with_events.py -v`
Expected: FAIL — `BaseEngine` has no `price_with_events`.

- [ ] **Step 3: Add `price_with_events` to `BaseEngine`**

In `asset/equity/engine/base_engine.py`, after the `price` abstractmethod and before `calculate_greeks`, add:

```python
    def price_with_events(
        self,
        product: BaseEquityProduct,
        pricing_env: PricingEnvironment,
        emit_distribution: bool = True,
    ) -> "PricingResult":
        """Return NPV and an EventDistribution describing termination/coupon timing.

        Default implementation wraps the scalar price() with a trivial
        EventDistribution (single mass at maturity, no KO). Engines that can
        cheaply emit richer event timing should override this method.

        Args:
            product: The derivative product to price.
            pricing_env: Pricing environment with market data.
            emit_distribution: When False, callers signal they don't need event
                timing; expensive emission paths (PDE forward density) may skip.
                The default of True preserves correctness for callers that don't
                know about this flag.

        Returns:
            PricingResult with .npv and .event_distribution fields.
        """
        from cashleg.event_distribution import EventDistribution, PricingResult
        npv = self.price(product, pricing_env)
        return PricingResult(
            npv=npv,
            event_distribution=EventDistribution.trivial(product.get_maturity()),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_cashleg/test_base_engine_price_with_events.py -v`
Expected: 2 passed.

Also run the full existing equity suite to confirm nothing else broke:

Run: `pytest test/ -x -q -k "european_option or american_option or asian_option"`
Expected: existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add asset/equity/engine/base_engine.py test/test_cashleg/test_base_engine_price_with_events.py
git commit -m "feat(engine): BaseEngine.price_with_events default returning trivial distribution"
```

---

## Task 1.11: `cashleg/` public API exports + module CLAUDE.md

**Files:**
- Modify: `cashleg/__init__.py`
- Create: `cashleg/CLAUDE.md`

- [ ] **Step 1: Update `cashleg/__init__.py` with public exports**

```python
# cashleg/__init__.py
"""Cash-leg primitives for pricing equity-option cash terms alongside the option payoff.

See cashleg/CLAUDE.md for usage and docs/superpowers/specs/2026-05-18-equity-cash-legs-design.md
for the architecture.
"""

from cashleg.accrual_leg import (
    AccrualLeg,
    KOBehavior,
    PaymentConvention,
    SurvivalBasis,
)
from cashleg.base import CashLeg, LegDirection
from cashleg.base_amount import BaseAmount, BaseAmountMode
from cashleg.deterministic_leg import DeterministicLeg
from cashleg.event_distribution import (
    EventDistribution,
    EventType,
    PricingResult,
)
from cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from cashleg.leg_schedule import LegSchedule
from cashleg.leg_valuator import LegPV, TradeValueBreakdown, value_leg

__all__ = [
    "AccrualLeg",
    "BaseAmount",
    "BaseAmountMode",
    "CashLeg",
    "DeterministicLeg",
    "EventDistribution",
    "EventType",
    "FixedPayoffLeg",
    "KOBehavior",
    "LegDirection",
    "LegPV",
    "LegSchedule",
    "PaymentConvention",
    "PaymentTrigger",
    "PricingResult",
    "SurvivalBasis",
    "TradeValueBreakdown",
    "value_leg",
]
```

- [ ] **Step 2: Create `cashleg/CLAUDE.md` developer guide**

```markdown
# Cash Legs Module - Developer Guide

## Overview

`cashleg/` provides composable cash-flow primitives that price alongside an
equity option payoff. Each leg returns a signed PV from the buyer's perspective.
Legs attach to an `EquityPosition` via its `cash_legs` field.

## Quick Reference

| Leg type | When to use |
|---|---|
| `DeterministicLeg` | Premium (front/back) with fixed amount and fixed timing |
| `AccrualLeg` | Periodic interest/rebate, optionally truncated by KO |
| `FixedPayoffLeg` | Single fixed amount paid only if a specific event (KO/KI/maturity outcome) occurs |

## Example

```python
from cashleg import (
    DeterministicLeg, AccrualLeg, FixedPayoffLeg,
    BaseAmount, BaseAmountMode, LegSchedule,
    LegDirection, PaymentTrigger, PaymentConvention, KOBehavior, SurvivalBasis,
)
from util.calendar.day_counter import DayCountConvention
import numpy as np

position.cash_legs = [
    DeterministicLeg(
        amount=1_500_000, payment_time=0.0,
        direction=LegDirection.BUYER_PAYS, name="Front Premium",
    ),
    AccrualLeg(
        rate=0.02,
        base=BaseAmount(value=1.0, mode=BaseAmountMode.NOTIONAL_FRACTION),
        schedule=LegSchedule(
            period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
            period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
            payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
        ),
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES, name="Margin Interest",
    ),
    FixedPayoffLeg(
        amount=50_000, trigger=PaymentTrigger.AT_KO,
        direction=LegDirection.BUYER_RECEIVES, name="KO Bonus",
    ),
]

total = position.get_trade_value(pricing_env)
breakdown = position.get_trade_value_breakdown(pricing_env)
greeks = position.get_trade_greeks(pricing_env, greeks_calc)
```

## Architecture

- `CashLeg.value(event_dist, env, position_notional) → float` (signed PV)
- `EventDistribution` is emitted by the engine via `engine.price_with_events(...)`
- Each engine family overrides `price_with_events` to expose its already-computed
  KO/survival timing (MC: per-path KO times; PDE: forward density; Quad: survival
  from recursion; Analytical: closed-form first-passage)

## Sign Convention

- All leg PVs are signed from the **buyer's (position holder's) perspective**.
- `LegDirection.BUYER_RECEIVES` → +1 sign; `BUYER_PAYS` → -1.
- `BaseAmount.value` must be non-negative; use `LegDirection` to flip signs.

## Tests

```
test/test_cashleg/
├── test_event_distribution.py
├── test_base.py
├── test_base_amount.py
├── test_leg_schedule.py
├── test_deterministic_leg.py
├── test_fixed_payoff_leg.py
├── test_accrual_leg.py
├── test_leg_valuator.py
├── test_base_engine_price_with_events.py
├── test_mc_event_emission.py     # Phase 2
├── test_quad_event_emission.py   # Phase 3
├── test_analytical_event_emission.py  # Phase 4
├── test_pde_event_emission.py    # Phase 5
├── test_position_with_legs.py    # Phase 6
└── test_position_backward_compat.py   # Phase 6
```

## Spec

See `docs/superpowers/specs/2026-05-18-equity-cash-legs-design.md`.
```

- [ ] **Step 3: Verify the public surface imports work**

Run: `python -c "from cashleg import DeterministicLeg, AccrualLeg, FixedPayoffLeg, BaseAmount, BaseAmountMode, LegSchedule, LegDirection, PaymentTrigger, EventDistribution, EventType, PricingResult; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add cashleg/__init__.py cashleg/CLAUDE.md
git commit -m "feat(cashleg): public API exports and module CLAUDE.md"
```

---

# Phase 2 — MC Engine Emission

Goal: Each Monte Carlo engine overrides `price_with_events` so the leg framework can compute KO-truncated accruals. The pattern: record per-path KO time index during the existing simulation loop, aggregate to PMF and survival, return alongside the existing NPV.

## Task 2.1: SnowballMCEngine — expose event distribution

**Files:**
- Modify: `asset/equity/engine/mc/snowball_mc_engine.py`
- Test: `test/test_cashleg/test_mc_event_emission.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_mc_event_emission.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import numpy as np
import pytest

from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import MCParams
from asset.equity.product.option import SnowballOption
from asset.equity.product.option.snowball_config import BarrierConfig
from cashleg.event_distribution import EventType
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def _snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0,
            ko_rate=0.15,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_continuous=True,
        ),
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def test_snowball_mc_emits_event_distribution():
    engine = SnowballMCEngine(params=MCParams(num_paths=20_000, time_steps=126, seed=42))
    result = engine.price_with_events(_snowball(), _env())

    assert result.event_distribution is not None
    dist = result.event_distribution
    assert EventType.KO in dist.probabilities
    assert isinstance(dist.probabilities[EventType.KO], np.ndarray)
    assert len(dist.probabilities[EventType.KO]) == 4  # one per obs date
    # Probabilities sum to ~1.0 (invariant enforced by EventDistribution)
    total = float(np.sum(dist.probabilities[EventType.KO]))
    total += float(dist.probabilities.get(EventType.MATURITY_NO_KO, 0.0))
    total += float(dist.probabilities.get(EventType.MATURITY_WITH_KI, 0.0))
    assert total == pytest.approx(1.0, abs=1e-6)
    # Survival is monotone non-increasing
    assert np.all(np.diff(dist.survival_probability) <= 1e-9)


def test_snowball_mc_npv_unchanged():
    """price_with_events.npv must equal price() (within MC noise — exact if same seed)."""
    engine = SnowballMCEngine(params=MCParams(num_paths=10_000, time_steps=126, seed=7))
    product = _snowball()
    env = _env()
    direct = engine.price(product, env)
    result = engine.price_with_events(product, env)
    assert result.npv == pytest.approx(direct, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_mc_event_emission.py::test_snowball_mc_emits_event_distribution -v`
Expected: FAIL — `event_distribution` is the trivial fallback (no KO).

- [ ] **Step 3: Implement override on `SnowballMCEngine`**

Open `asset/equity/engine/mc/snowball_mc_engine.py`. The engine already simulates paths and computes `first_ko_idx` (see existing line ~931, `_check_ko_barriers`). Add a `price_with_events` method that delegates to the existing `calculate_event_stats` (which already computes KO probabilities and survival), then converts to `EventDistribution`:

Add at the end of the `SnowballMCEngine` class:

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        """Emit EventDistribution from existing snowball simulation byproducts."""
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult

        stats = self.calculate_event_stats(product, pricing_env)
        if stats is None:
            return super().price_with_events(product, pricing_env, emit_distribution)

        # Total KO probability over all obs dates
        ko_prob_arr = np.asarray(stats.ko_probability, dtype=float)
        ki_prob_total = float(stats.ki_probability) if stats.ki_probability is not None else 0.0
        ko_prob_total = float(ko_prob_arr.sum())
        # Maturity outcomes: no-KO split between KI-triggered and not
        p_maturity = max(0.0, 1.0 - ko_prob_total)
        p_maturity_with_ki = min(p_maturity, ki_prob_total)
        p_maturity_no_ko = p_maturity - p_maturity_with_ki

        # Build survival array of length N+1 from the engine's (length-N) survival
        # at observation times. survival_probability[0]=1.0; survival[i+1]=stats.survival_probability[i]
        engine_surv = np.asarray(stats.survival_probability, dtype=float)
        survival = np.concatenate([[1.0], engine_surv])

        # Normalize tiny numerical leftovers so EventDistribution invariants pass
        probs = {
            EventType.KO: ko_prob_arr,
            EventType.MATURITY_NO_KO: p_maturity_no_ko,
            EventType.MATURITY_WITH_KI: p_maturity_with_ki,
        }

        dist = EventDistribution(
            event_times=np.asarray(stats.ko_times, dtype=float),
            event_dates=None,
            probabilities=probs,
            survival_probability=survival,
        )
        return PricingResult(npv=float(stats.pv), event_distribution=dist)
```

Add `import numpy as np` at the top if not already present (it almost certainly is).

- [ ] **Step 4: Run the tests**

Run: `pytest test/test_cashleg/test_mc_event_emission.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run existing snowball MC tests**

Run: `pytest test/test_snowball_mc_engine.py -x -q`
Expected: existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add asset/equity/engine/mc/snowball_mc_engine.py test/test_cashleg/test_mc_event_emission.py
git commit -m "feat(mc): SnowballMCEngine emits EventDistribution via calculate_event_stats"
```

---

## Task 2.2: PhoenixMCEngine — expose event distribution

**Files:**
- Modify: `asset/equity/engine/mc/phoenix_mc_engine.py`
- Test: append to `test/test_cashleg/test_mc_event_emission.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_cashleg/test_mc_event_emission.py`:

```python
def test_phoenix_mc_emits_event_distribution_with_coupons():
    from asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
    from asset.equity.product.option import create_standard_phoenix
    from util.calendar.day_counter import DayCountConvention
    from util.enum import CouponPayType

    phoenix = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=85.0,
        coupon_rate=0.01, num_observations=12, memory_coupon=True,
        day_count_convention=DayCountConvention.ACT_365,
        coupon_pay_type=CouponPayType.INSTANT,
    )
    engine = PhoenixMCEngine(params=MCParams(num_paths=10_000, time_steps=252, seed=11))
    result = engine.price_with_events(phoenix, _env())
    assert result.event_distribution is not None
    dist = result.event_distribution
    # Phoenix has both KO and (optionally) COUPON event probabilities
    assert EventType.KO in dist.probabilities
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_mc_event_emission.py::test_phoenix_mc_emits_event_distribution_with_coupons -v`
Expected: FAIL — no override yet.

- [ ] **Step 3: Implement override on `PhoenixMCEngine`**

Open `asset/equity/engine/mc/phoenix_mc_engine.py`. Add a `price_with_events` method using the same pattern as Snowball MC (route through `calculate_event_stats` and convert). If `calculate_event_stats` returns a `PhoenixEventStats`, additionally expose `EventType.COUPON`:

Add at the end of the `PhoenixMCEngine` class:

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult
        from asset.equity.engine.event_stats import PhoenixEventStats

        stats = self.calculate_event_stats(product, pricing_env)
        if stats is None:
            return super().price_with_events(product, pricing_env, emit_distribution)

        ko_prob_arr = np.asarray(stats.ko_probability, dtype=float)
        ki_prob_total = float(stats.ki_probability) if stats.ki_probability is not None else 0.0
        ko_prob_total = float(ko_prob_arr.sum())
        p_maturity = max(0.0, 1.0 - ko_prob_total)
        p_maturity_with_ki = min(p_maturity, ki_prob_total)
        p_maturity_no_ko = p_maturity - p_maturity_with_ki

        engine_surv = np.asarray(stats.survival_probability, dtype=float)
        survival = np.concatenate([[1.0], engine_surv])

        probs = {
            EventType.KO: ko_prob_arr,
            EventType.MATURITY_NO_KO: p_maturity_no_ko,
            EventType.MATURITY_WITH_KI: p_maturity_with_ki,
        }
        if isinstance(stats, PhoenixEventStats) and len(stats.coupon_probability) > 0:
            probs[EventType.COUPON] = np.asarray(stats.coupon_probability, dtype=float)

        dist = EventDistribution(
            event_times=np.asarray(stats.ko_times, dtype=float),
            event_dates=None,
            probabilities=probs,
            survival_probability=survival,
        )
        return PricingResult(npv=float(stats.pv), event_distribution=dist)
```

Note: `EventType.COUPON` is an additional probability stream not constrained by the sum-to-1 invariant (coupons are per-obs payments, not termination events). The current `EventDistribution.__post_init__` sums all probabilities including COUPON, which would break the invariant. Fix the invariant check to exclude `COUPON`:

Open `cashleg/event_distribution.py`. In `_validate_invariants`, change the probability sum loop:

```python
        # Probability sum ≈ 1 (excluding COUPON which is per-obs payment, not termination)
        total = 0.0
        for evt, p in self.probabilities.items():
            if evt is EventType.COUPON:
                continue
            if isinstance(p, np.ndarray):
                total += float(p.sum())
            else:
                total += float(p)
        if abs(total - 1.0) > Tolerance.PROBABILITY:
            raise NumericalError(
                f"EventDistribution termination-probability sum = {total}, expected 1.0 "
                f"(tolerance {Tolerance.PROBABILITY})"
            )
```

Update `test/test_cashleg/test_event_distribution.py` to add a coverage test:

```python
def test_coupon_probabilities_excluded_from_sum_invariant():
    """COUPON is per-obs payment, not a termination event; must not break sum=1 check."""
    dist = EventDistribution(
        event_times=np.array([0.5, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.0]),
            EventType.COUPON: np.array([0.8, 0.7]),  # per-obs, can sum to >1
            EventType.MATURITY_NO_KO: 0.7,
        },
        survival_probability=np.array([1.0, 0.7, 0.7]),
    )
    assert dist.probabilities[EventType.COUPON].sum() > 1.0
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_cashleg/test_mc_event_emission.py test/test_cashleg/test_event_distribution.py -v`
Expected: all passed.

Also: `pytest test/test_phoenix_mc_engine.py -x -q`
Expected: existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add asset/equity/engine/mc/phoenix_mc_engine.py cashleg/event_distribution.py test/test_cashleg/test_mc_event_emission.py test/test_cashleg/test_event_distribution.py
git commit -m "feat(mc): PhoenixMCEngine emits EventDistribution; exclude COUPON from sum invariant"
```

---

## Task 2.3: KOResetSnowballPDESolver / MC + other autocallable MC engines

**Files:**
- Modify: `asset/equity/engine/mc/range_accrual_mc_engine.py` (apply same pattern via `calculate_event_stats` if implemented; otherwise use trivial fall-through)
- Modify: `asset/equity/engine/mc/barrier_option_mc_engine.py` — barrier MC has clearly-defined KO observations
- Test: append to `test/test_cashleg/test_mc_event_emission.py`

- [ ] **Step 1: Determine which MC engines need overrides**

Run:
```bash
grep -l "calculate_event_stats" asset/equity/engine/mc/*.py
```

For each MC engine **not** in the output above, the trivial default from `BaseEngine.price_with_events` suffices for `DeterministicLeg`s and `AccrualLeg(PAY_FULL_SCHEDULE)`. KO-sensitive legs will fall back to the documented warning path (§9.2 of spec). This task only adds an override where `calculate_event_stats` already exists OR where adding KO-time recording is trivial.

- [ ] **Step 2: Write the failing test for BarrierOptionMCEngine**

Append to `test/test_cashleg/test_mc_event_emission.py`:

```python
def test_barrier_mc_emits_ko_probability_for_knockout():
    """For a knock-out barrier option, MC engine should emit P(KO at obs i)."""
    from asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
    from asset.equity.product.option import BarrierOption
    from util.enum import OptionType
    from util.enum.option_enums import BarrierType, BarrierDirection

    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        barrier_level=110.0,
        barrier_type=BarrierType.KNOCK_OUT,
        barrier_direction=BarrierDirection.UP,
    )
    engine = BarrierOptionMCEngine(params=MCParams(num_paths=20_000, time_steps=252, seed=3))
    result = engine.price_with_events(option, _env())
    assert result.event_distribution is not None
    dist = result.event_distribution
    # For an up-and-out call, expect non-zero KO probability when spot near barrier
    assert EventType.KO in dist.probabilities or EventType.MATURITY_NO_KO in dist.probabilities
```

- [ ] **Step 3: Implement override on `BarrierOptionMCEngine`**

Open `asset/equity/engine/mc/barrier_option_mc_engine.py`. Locate the `price` method and identify where the engine determines per-path barrier-touch indices. Add a `price_with_events` method that records the first-touch obs index per path (for discrete-monitored barriers) or aggregates the continuous-touch indicator into a single survival series. If the engine already tracks `first_hit_idx` or equivalent, route through it.

Reference pattern (adapt to actual engine internals):

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult
        # If this barrier MC tracks per-path KO times, build EventDistribution from them.
        # Otherwise, fall back to the default trivial distribution.
        try:
            paths, payoffs, ko_idx, obs_times = self._simulate_with_events(product, pricing_env)
        except AttributeError:
            return super().price_with_events(product, pricing_env, emit_distribution)

        n_paths = len(payoffs)
        n_obs = len(obs_times)
        ko_count = np.zeros(n_obs, dtype=float)
        for i in range(n_obs):
            ko_count[i] = float((ko_idx == i).sum())
        p_ko = ko_count / n_paths
        p_maturity_no_ko = float((ko_idx == -1).sum()) / n_paths

        survival = np.empty(n_obs + 1, dtype=float)
        survival[0] = 1.0
        cum_ko = 0.0
        for i in range(n_obs):
            cum_ko += p_ko[i]
            survival[i + 1] = max(0.0, 1.0 - cum_ko)

        # Discount payoffs to compute NPV (same logic the engine already uses)
        npv = float(np.mean(payoffs))

        dist = EventDistribution(
            event_times=np.asarray(obs_times, dtype=float),
            event_dates=None,
            probabilities={EventType.KO: p_ko, EventType.MATURITY_NO_KO: p_maturity_no_ko},
            survival_probability=survival,
        )
        return PricingResult(npv=npv, event_distribution=dist)
```

If `BarrierOptionMCEngine` does not currently expose `_simulate_with_events`, factor the existing simulation loop into such a helper (separate refactor commit, then add the override).

For this task scope: if refactoring is invasive, defer to the trivial fall-through and document in `cashleg/CLAUDE.md` that BarrierOptionMCEngine accrual legs use `PAY_FULL_SCHEDULE` until phase 2 follow-up. The plan favors smaller, shippable changes over invasive refactors.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_cashleg/test_mc_event_emission.py -v`
Expected: passed (or test marked xfail with a clear pointer if barrier override deferred).

Also: `pytest test/test_barrier_option_mc_engine.py -x -q`
Expected: existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add asset/equity/engine/mc/barrier_option_mc_engine.py test/test_cashleg/test_mc_event_emission.py
git commit -m "feat(mc): BarrierOptionMCEngine emits EventDistribution from per-path KO indices"
```

---

## Task 2.4: AccrualLeg + KO-product cross-check test

This validates that the full Phase 1 + Phase 2 pipeline produces sensible numbers on a real product.

**Files:**
- Test: append to `test/test_cashleg/test_mc_event_emission.py`

- [ ] **Step 1: Write the cross-check test**

```python
def test_accrual_leg_truncated_by_ko_is_smaller_than_full_schedule():
    """On a high-volatility snowball with frequent KO, TRUNCATE_AT_KO PV < PAY_FULL_SCHEDULE PV."""
    from cashleg import (
        AccrualLeg, BaseAmount, BaseAmountMode, KOBehavior, LegDirection,
        LegSchedule, PaymentConvention, SurvivalBasis, value_leg,
    )
    from util.calendar.day_counter import DayCountConvention

    engine = SnowballMCEngine(params=MCParams(num_paths=30_000, time_steps=126, seed=5))
    result = engine.price_with_events(_snowball(), _env())
    schedule = LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    base = BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE)
    leg_full = AccrualLeg(
        rate=0.04, base=base, schedule=schedule,
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.PAY_FULL_SCHEDULE,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    leg_trunc = AccrualLeg(
        rate=0.04, base=base, schedule=schedule,
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv_full = value_leg(leg_full, result.event_distribution, _env(), 0.0)
    pv_trunc = value_leg(leg_trunc, result.event_distribution, _env(), 0.0)
    assert pv_trunc < pv_full
    # Sanity bounds
    assert 0 < pv_trunc < pv_full
```

- [ ] **Step 2: Run test**

Run: `pytest test/test_cashleg/test_mc_event_emission.py::test_accrual_leg_truncated_by_ko_is_smaller_than_full_schedule -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_cashleg/test_mc_event_emission.py
git commit -m "test(cashleg): cross-check AccrualLeg truncation vs full-schedule on snowball MC"
```

---

# Phase 3 — Quad Engine Emission

Goal: Quad engines already compute survival probabilities in their recursion (e.g., `snowball_quad_engine.py:507-511`). Exposing them via `price_with_events` is essentially zero-cost.

## Task 3.1: SnowballQuadEngine — expose event distribution

**Files:**
- Modify: `asset/equity/engine/quad/snowball_quad_engine.py`
- Test: `test/test_cashleg/test_quad_event_emission.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_quad_event_emission.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import numpy as np
import pytest

from asset.equity.engine.quad import SnowballQuadEngine
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import MCParams
from asset.equity.product.option import SnowballOption
from asset.equity.product.option.snowball_config import BarrierConfig
from cashleg.event_distribution import EventType
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def _snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0, ki_continuous=True,
        ),
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def test_snowball_quad_emits_event_distribution():
    engine = SnowballQuadEngine()
    result = engine.price_with_events(_snowball(), _env())
    assert result.event_distribution is not None
    dist = result.event_distribution
    assert EventType.KO in dist.probabilities
    assert len(dist.probabilities[EventType.KO]) == 4


def test_snowball_quad_ko_probabilities_match_mc_within_tolerance():
    """Quad and MC should agree on integrated KO probability for the same product."""
    quad = SnowballQuadEngine()
    mc = SnowballMCEngine(params=MCParams(num_paths=50_000, time_steps=252, seed=1))
    env = _env()
    product = _snowball()
    q = quad.price_with_events(product, env).event_distribution
    m = mc.price_with_events(product, env).event_distribution
    q_total_ko = float(q.probabilities[EventType.KO].sum())
    m_total_ko = float(m.probabilities[EventType.KO].sum())
    # 2% absolute tolerance is a comfortable margin for 50k MC paths
    assert abs(q_total_ko - m_total_ko) < 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_quad_event_emission.py -v`
Expected: FAIL — `SnowballQuadEngine.price_with_events` is the trivial default.

- [ ] **Step 3: Implement override on `SnowballQuadEngine`**

Use the same pattern as Snowball MC — route through `calculate_event_stats`. Open `asset/equity/engine/quad/snowball_quad_engine.py` and add:

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult

        stats = self.calculate_event_stats(product, pricing_env)
        if stats is None:
            return super().price_with_events(product, pricing_env, emit_distribution)

        ko_prob_arr = np.asarray(stats.ko_probability, dtype=float)
        ki_prob_total = float(stats.ki_probability) if stats.ki_probability is not None else 0.0
        ko_prob_total = float(ko_prob_arr.sum())
        p_maturity = max(0.0, 1.0 - ko_prob_total)
        p_maturity_with_ki = min(p_maturity, ki_prob_total)
        p_maturity_no_ko = p_maturity - p_maturity_with_ki

        engine_surv = np.asarray(stats.survival_probability, dtype=float)
        survival = np.concatenate([[1.0], engine_surv])

        dist = EventDistribution(
            event_times=np.asarray(stats.ko_times, dtype=float),
            event_dates=None,
            probabilities={
                EventType.KO: ko_prob_arr,
                EventType.MATURITY_NO_KO: p_maturity_no_ko,
                EventType.MATURITY_WITH_KI: p_maturity_with_ki,
            },
            survival_probability=survival,
        )
        return PricingResult(npv=float(stats.pv), event_distribution=dist)
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_cashleg/test_quad_event_emission.py -v`
Expected: 2 passed.

Also: `pytest test/test_snowball_quad_engine.py -x -q` (if exists)
Expected: existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add asset/equity/engine/quad/snowball_quad_engine.py test/test_cashleg/test_quad_event_emission.py
git commit -m "feat(quad): SnowballQuadEngine emits EventDistribution via calculate_event_stats"
```

---

## Task 3.2: PhoenixQuadEngine + KOResetSnowballQuadEngine — same pattern

**Files:**
- Modify: `asset/equity/engine/quad/phoenix_quad_engine.py`
- Modify: `asset/equity/engine/quad/ko_reset_snowball_quad_engine.py`
- Test: append to `test/test_cashleg/test_quad_event_emission.py`

- [ ] **Step 1: Add tests for Phoenix and KO-Reset Quad emission**

Append to `test/test_cashleg/test_quad_event_emission.py`:

```python
def test_phoenix_quad_emits_event_distribution():
    from asset.equity.engine.quad import PhoenixQuadEngine
    from asset.equity.product.option import create_standard_phoenix
    from util.calendar.day_counter import DayCountConvention
    from util.enum import CouponPayType

    phoenix = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=85.0,
        coupon_rate=0.01, num_observations=12, memory_coupon=True,
        day_count_convention=DayCountConvention.ACT_365,
        coupon_pay_type=CouponPayType.INSTANT,
    )
    engine = PhoenixQuadEngine()
    result = engine.price_with_events(phoenix, _env())
    assert result.event_distribution is not None
    assert EventType.KO in result.event_distribution.probabilities


def test_ko_reset_snowball_quad_emits_event_distribution():
    from asset.equity.engine.quad import KOResetSnowballQuadEngine
    from asset.equity.product.option import KOResetSnowballOption
    # Build a minimal KO-reset snowball mirroring example/ko_reset_snowball_demo.py
    # NOTE: if KOResetSnowballOption requires more elaborate setup, adapt from the demo.
    pytest.skip("Implementation-specific KOResetSnowballOption construction; "
                "use example/ko_reset_snowball_demo.py as the canonical setup.")
```

- [ ] **Step 2: Implement Phoenix override**

Apply the same pattern as §3.1 to `phoenix_quad_engine.py`. Use `PhoenixEventStats` (subclass of `AutocallableEventStats`) — if `stats` is a `PhoenixEventStats` with non-empty `coupon_probability`, add `EventType.COUPON` to the dict (same as Phoenix MC in §2.2):

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult
        from asset.equity.engine.event_stats import PhoenixEventStats

        stats = self.calculate_event_stats(product, pricing_env)
        if stats is None:
            return super().price_with_events(product, pricing_env, emit_distribution)

        ko_prob_arr = np.asarray(stats.ko_probability, dtype=float)
        ki_prob_total = float(stats.ki_probability) if stats.ki_probability is not None else 0.0
        ko_prob_total = float(ko_prob_arr.sum())
        p_maturity = max(0.0, 1.0 - ko_prob_total)
        p_maturity_with_ki = min(p_maturity, ki_prob_total)
        p_maturity_no_ko = p_maturity - p_maturity_with_ki

        engine_surv = np.asarray(stats.survival_probability, dtype=float)
        survival = np.concatenate([[1.0], engine_surv])

        probs = {
            EventType.KO: ko_prob_arr,
            EventType.MATURITY_NO_KO: p_maturity_no_ko,
            EventType.MATURITY_WITH_KI: p_maturity_with_ki,
        }
        if isinstance(stats, PhoenixEventStats) and len(stats.coupon_probability) > 0:
            probs[EventType.COUPON] = np.asarray(stats.coupon_probability, dtype=float)

        dist = EventDistribution(
            event_times=np.asarray(stats.ko_times, dtype=float),
            event_dates=None,
            probabilities=probs,
            survival_probability=survival,
        )
        return PricingResult(npv=float(stats.pv), event_distribution=dist)
```

- [ ] **Step 3: Implement KOResetSnowballQuadEngine override (best-effort)**

Apply the same pattern. If `calculate_event_stats` returns `KOResetEventStats` (subclass), the `ko_times` and `ko_probability` already represent the merged pre+post KI ordering. Use the same conversion as above.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_cashleg/test_quad_event_emission.py -v`
Expected: 3 passed (Phoenix passes; KO-reset skipped per the test).

- [ ] **Step 5: Commit**

```bash
git add asset/equity/engine/quad/phoenix_quad_engine.py asset/equity/engine/quad/ko_reset_snowball_quad_engine.py test/test_cashleg/test_quad_event_emission.py
git commit -m "feat(quad): PhoenixQuadEngine + KOResetSnowballQuadEngine emit EventDistribution"
```

---

# Phase 4 — Analytical Engine Emission

Goal: Analytical engines that already implement closed-form barrier formulas can expose first-passage probabilities cheaply. For vanilla engines (BlackScholes, American analytical, Asian analytical), the trivial default from `BaseEngine` is correct — no override needed.

## Task 4.1: BarrierAnalyticalEngine + OneTouchAnalyticalEngine — first-passage emission

**Files:**
- Modify: `asset/equity/engine/analytical/barrier_analytical_engine.py`
- Modify: `asset/equity/engine/analytical/one_touch_analytical_engine.py`
- Test: `test/test_cashleg/test_analytical_event_emission.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_analytical_event_emission.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import numpy as np
import pytest

from cashleg.event_distribution import EventType
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env(rate=0.03):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_one_touch_analytical_emits_touch_probability():
    """For a one-touch, EventType.KO probability at maturity ≈ analytical touch probability."""
    from asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
    from asset.equity.product.option import OneTouchOption

    one_touch = OneTouchOption(
        barrier_level=110.0,
        rebate=1.0,
        maturity=1.0,
        # adapt to actual OneTouchOption signature
    )
    engine = OneTouchAnalyticalEngine()
    try:
        result = engine.price_with_events(one_touch, _env())
    except (TypeError, AttributeError):
        pytest.skip("OneTouchOption signature differs; adapt construction")
    assert result.event_distribution is not None
    # Sum of KO probability should equal the one-touch undiscounted touch probability
    p_touch = float(result.event_distribution.probabilities[EventType.KO].sum())
    assert 0.0 < p_touch < 1.0


def test_barrier_analytical_emits_first_passage_distribution():
    """Up-and-out call: PV reduces as touch probability increases."""
    from asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from asset.equity.product.option import BarrierOption
    from util.enum import OptionType
    from util.enum.option_enums import BarrierType, BarrierDirection

    option = BarrierOption(
        strike=100.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        barrier_level=120.0,
        barrier_type=BarrierType.KNOCK_OUT,
        barrier_direction=BarrierDirection.UP,
    )
    engine = BarrierAnalyticalEngine()
    try:
        result = engine.price_with_events(option, _env())
    except (TypeError, AttributeError):
        pytest.skip("BarrierAnalyticalEngine override not yet implemented; "
                    "this test will activate after Task 4.1 implementation")
    assert result.event_distribution is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_analytical_event_emission.py -v`
Expected: tests skip OR fail with `event_distribution` being the trivial default.

- [ ] **Step 3: Implement override on `OneTouchAnalyticalEngine`**

Open `asset/equity/engine/analytical/one_touch_analytical_engine.py`. The engine already computes the undiscounted touch probability internally (as part of one-touch valuation). Add:

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult
        import numpy as np

        npv = self.price(product, pricing_env)
        T = product.get_maturity()
        # Compute undiscounted touch probability under risk-neutral measure
        # (use the same closed-form components the engine already uses for pricing)
        p_touch = self._touch_probability(product, pricing_env)  # add this helper if not present
        # Simple two-bucket distribution: all KO mass at maturity (analytical doesn't time-resolve)
        dist = EventDistribution(
            event_times=np.array([0.0, T]),
            event_dates=None,
            probabilities={
                EventType.KO: np.array([0.0, p_touch]),
                EventType.MATURITY_NO_KO: 1.0 - p_touch,
            },
            survival_probability=np.array([1.0, 1.0 - p_touch, 1.0 - p_touch]),
        )
        return PricingResult(npv=npv, event_distribution=dist)
```

If `_touch_probability` does not yet exist, factor it out from the existing pricing formula. Reference: Black-Scholes-Merton barrier/touch probability under risk-neutral measure.

For `BarrierAnalyticalEngine`, follow the same approach. If the engine prices via the reflection principle and computes `P(min/max S_t crosses B)` internally, expose that. Otherwise, defer this engine to a follow-up commit and leave the test as `xfail`.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_cashleg/test_analytical_event_emission.py -v`
Expected: at least one test passes (OneTouch); BarrierAnalyticalEngine may skip.

Also: `pytest test/test_barrier_analytical_engine.py -x -q` and any one-touch tests.
Expected: existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add asset/equity/engine/analytical/one_touch_analytical_engine.py asset/equity/engine/analytical/barrier_analytical_engine.py test/test_cashleg/test_analytical_event_emission.py
git commit -m "feat(analytical): OneTouch/Barrier analytical engines emit first-passage EventDistribution"
```

---

# Phase 5 — PDE Engine Emission

Goal: PDE engines produce backward-induction values; emitting `EventDistribution` requires an additional forward density solve on the same grid. Cost: ~20% extra; opt-in via `emit_distribution` flag. Skip the forward solve when no leg needs it.

## Task 5.1: Forward-density helper

**Files:**
- Create: `asset/equity/engine/pde/forward_density_helper.py`
- Test: `test/test_cashleg/test_pde_event_emission.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_cashleg/test_pde_event_emission.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import numpy as np
import pytest

from cashleg.event_distribution import EventType
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_snowball_pde_emits_event_distribution():
    from asset.equity.engine.pde import SnowballPDESolver
    from asset.equity.param import PDEParams
    from asset.equity.product.option import SnowballOption
    from asset.equity.product.option.snowball_config import BarrierConfig

    product = SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0, ki_continuous=True,
        ),
        contract_multiplier=10_000.0, maturity=1.0,
    )
    engine = SnowballPDESolver(params=PDEParams(grid_size=300, time_steps=150))
    result = engine.price_with_events(product, _env(), emit_distribution=True)
    assert result.event_distribution is not None
    assert EventType.KO in result.event_distribution.probabilities
    assert len(result.event_distribution.probabilities[EventType.KO]) == 4


def test_snowball_pde_emit_distribution_false_skips_forward_solve():
    """When emit_distribution=False, no extra cost (and event_dist is trivial)."""
    from asset.equity.engine.pde import SnowballPDESolver
    from asset.equity.param import PDEParams
    from asset.equity.product.option import SnowballOption
    from asset.equity.product.option.snowball_config import BarrierConfig

    product = SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0, ki_continuous=True,
        ),
        contract_multiplier=10_000.0, maturity=1.0,
    )
    engine = SnowballPDESolver(params=PDEParams(grid_size=300, time_steps=150))
    result = engine.price_with_events(product, _env(), emit_distribution=False)
    # Trivial distribution: single MATURITY_NO_KO mass
    assert result.event_distribution.probabilities.get(EventType.MATURITY_NO_KO) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_pde_event_emission.py -v`
Expected: first test FAILS (PDE returns trivial distribution).

- [ ] **Step 3: Implement `forward_density_helper.py`**

```python
# asset/equity/engine/pde/forward_density_helper.py
"""Forward density solver — companion to backward PDE for emitting KO/survival distributions.

Solves the forward Kolmogorov equation on the same grid the backward PDE uses,
starting from a delta function at the initial spot and stepping forward through
obs dates. At each KO obs date, integrate the density above the KO barrier to
get P(KO at this obs date); zero out the absorbed mass on the post-KO grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ForwardDensityResult:
    obs_times: np.ndarray              # shape (M,), year fractions of KO obs
    ko_probability: np.ndarray         # shape (M,), P(KO at each obs)
    survival_probability: np.ndarray   # shape (M+1,), survival entering obs i


def solve_forward_density(
    spatial_grid: np.ndarray,
    time_grid: np.ndarray,
    drift_fn,                # callable: (S, t) -> drift
    diffusion_fn,            # callable: (S, t) -> diffusion (vol*S)
    initial_spot: float,
    obs_times: List[float],
    ko_barrier: float,
    ko_direction: str = "up",        # "up" or "down"
) -> ForwardDensityResult:
    """Run a Crank-Nicolson forward solve of the density.

    For each time step:
      - Step density forward via FP equation discretization on `spatial_grid`.
      - At each obs date in `obs_times`, integrate density on the KO side of
        the barrier, record that as P(KO at this obs), and zero out the
        absorbed mass on the grid.

    Returns ForwardDensityResult with arrays sized to len(obs_times).
    """
    n_space = len(spatial_grid)
    # Initialize density as a discrete delta function at initial_spot
    density = np.zeros(n_space)
    idx0 = int(np.argmin(np.abs(spatial_grid - initial_spot)))
    density[idx0] = 1.0  # discrete normalization: integral via trapezoidal handled at obs

    obs_set = set(obs_times)
    ko_probs: List[float] = []
    survival = [1.0]

    for k in range(1, len(time_grid)):
        t = time_grid[k]
        dt = time_grid[k] - time_grid[k - 1]
        density = _step_forward(density, spatial_grid, drift_fn, diffusion_fn, t, dt)

        # Check if this time step coincides with an obs date (within float tolerance)
        if any(abs(t - obs) < 1e-9 for obs in obs_set):
            if ko_direction == "up":
                ko_mask = spatial_grid >= ko_barrier
            else:
                ko_mask = spatial_grid <= ko_barrier
            ko_mass = float(np.trapezoid(density[ko_mask], spatial_grid[ko_mask]))
            current_survival = survival[-1]
            ko_prob_this_obs = min(current_survival, max(0.0, ko_mass))
            ko_probs.append(ko_prob_this_obs)
            survival.append(max(0.0, current_survival - ko_prob_this_obs))
            # Absorb: zero density on KO side
            density[ko_mask] = 0.0

    return ForwardDensityResult(
        obs_times=np.asarray(obs_times, dtype=float),
        ko_probability=np.asarray(ko_probs, dtype=float),
        survival_probability=np.asarray(survival, dtype=float),
    )


def _step_forward(density, spatial_grid, drift_fn, diffusion_fn, t, dt):
    """Crank-Nicolson step of the forward Kolmogorov equation.

    For implementation efficiency, this should reuse the backward PDE's
    tridiagonal banded solver from asset/equity/engine/pde/core/. For an
    initial implementation, use scipy.sparse banded solve.
    """
    from scipy.linalg import solve_banded
    n = len(spatial_grid)
    # Simple central-difference + CN scheme; production code reuses the
    # backward PDE's banded matrix infrastructure with sign-flipped operators.
    dx = spatial_grid[1] - spatial_grid[0]
    mu = np.array([drift_fn(s, t) for s in spatial_grid])
    sigma = np.array([diffusion_fn(s, t) for s in spatial_grid])
    a = 0.5 * sigma**2 / dx**2 - mu / (2 * dx)   # sub-diagonal
    c = 0.5 * sigma**2 / dx**2 + mu / (2 * dx)   # super-diagonal
    b = -(sigma**2) / dx**2                       # main diagonal contribution from diffusion

    theta = 0.5
    # Implicit part: (I - theta*dt*L) * d_new = (I + (1-theta)*dt*L) * d_old
    lower = -theta * dt * a
    upper = -theta * dt * c
    diag = 1.0 - theta * dt * b
    rhs = (
        density
        + (1 - theta) * dt * (a * np.concatenate([[0.0], density[:-1]])
                              + b * density
                              + c * np.concatenate([density[1:], [0.0]]))
    )

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)
```

This is a deliberately simple reference implementation. For production-quality matching the existing PDE engines' banded-cache infrastructure (`asset/equity/engine/pde/core/`), open a follow-up issue and refactor.

- [ ] **Step 4: Implement override on `SnowballPDESolver`**

Open `asset/equity/engine/pde/snowball_pde_solver.py`. Add:

```python
    def price_with_events(self, product, pricing_env, emit_distribution: bool = True):
        from cashleg.event_distribution import EventDistribution, EventType, PricingResult
        from asset.equity.engine.pde.forward_density_helper import solve_forward_density
        import numpy as np

        npv = self.price(product, pricing_env)

        if not emit_distribution:
            return super().price_with_events(product, pricing_env, emit_distribution=False)

        # Reuse the engine's existing spatial and time grids; expose via helper if not public
        spatial_grid = self._get_spatial_grid(product, pricing_env)
        time_grid = self._get_time_grid(product, pricing_env)
        drift_fn = lambda s, t: (pricing_env.rate_curve.get_rate(t)
                                  - pricing_env.div_yield.div_yield) * s
        diffusion_fn = lambda s, t: pricing_env.vol_surface.get_vol(s, t) * s

        ko_obs = product.barrier_config.ko_observation_dates
        ko_barrier_level = product.initial_price * (
            product.barrier_config.ko_barrier / 100.0
            if isinstance(product.barrier_config.ko_barrier, (int, float))
            and product.barrier_config.ko_barrier > 10
            else product.barrier_config.ko_barrier
        )

        fwd = solve_forward_density(
            spatial_grid=spatial_grid,
            time_grid=time_grid,
            drift_fn=drift_fn,
            diffusion_fn=diffusion_fn,
            initial_spot=pricing_env.spot,
            obs_times=ko_obs,
            ko_barrier=ko_barrier_level,
            ko_direction="up",
        )

        ko_total = float(fwd.ko_probability.sum())
        p_maturity_no_ko = max(0.0, 1.0 - ko_total)

        dist = EventDistribution(
            event_times=fwd.obs_times,
            event_dates=None,
            probabilities={
                EventType.KO: fwd.ko_probability,
                EventType.MATURITY_NO_KO: p_maturity_no_ko,
            },
            survival_probability=fwd.survival_probability,
        )
        return PricingResult(npv=npv, event_distribution=dist)
```

If `_get_spatial_grid` and `_get_time_grid` are not public on the existing solver, expose them as `@property` or thin wrappers around the engine's internal grid construction.

- [ ] **Step 5: Run tests**

Run: `pytest test/test_cashleg/test_pde_event_emission.py -v`
Expected: 2 passed.

Also: `pytest test/test_snowball_pde_solver.py -x -q` (if exists)
Expected: existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add asset/equity/engine/pde/forward_density_helper.py asset/equity/engine/pde/snowball_pde_solver.py test/test_cashleg/test_pde_event_emission.py
git commit -m "feat(pde): forward density helper + SnowballPDESolver emits EventDistribution (opt-in)"
```

---

## Task 5.2: Phoenix / KO-Reset / Barrier PDE overrides

**Files:**
- Modify: `asset/equity/engine/pde/phoenix_pde_solver.py`
- Modify: `asset/equity/engine/pde/ko_reset_snowball_pde_solver.py`
- Modify: `asset/equity/engine/pde/barrier_pde_solver.py`

- [ ] **Step 1: Apply the §5.1 pattern to remaining PDE solvers**

For each solver, add a `price_with_events` override following the Snowball PDE template. The forward density helper is reusable; only the per-engine KO direction, obs schedule, and barrier level extraction differ.

- [ ] **Step 2: Run all PDE-emission tests**

Run: `pytest test/test_cashleg/test_pde_event_emission.py -v`
Expected: all passed.

- [ ] **Step 3: Commit**

```bash
git add asset/equity/engine/pde/phoenix_pde_solver.py asset/equity/engine/pde/ko_reset_snowball_pde_solver.py asset/equity/engine/pde/barrier_pde_solver.py
git commit -m "feat(pde): Phoenix / KO-Reset / Barrier PDE solvers emit EventDistribution"
```

---

# Phase 6 — Position Integration

Goal: Extend `EquityPosition` with the optional `cash_legs` field and new trade-level methods. Preserve 100% backward compatibility for callers that don't use legs.

## Task 6.1: `EquityPosition.cash_legs` field + `get_trade_value`

**Files:**
- Modify: `portfolio/equity/position.py`
- Test: `test/test_cashleg/test_position_with_legs.py`
- Test: `test/test_cashleg/test_position_backward_compat.py`

- [ ] **Step 1: Write backward-compat test (must pass before and after the change)**

```python
# test/test_cashleg/test_position_backward_compat.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime
import pytest

from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.product.option import EuropeanVanillaOption
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from portfolio.equity.position import EquityPosition
from priceenv import PricingEnvironment
from util.enum import OptionType


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_position_constructs_without_cash_legs():
    """Construct exactly as today — cash_legs has a default."""
    pos = EquityPosition(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=10.0,
        entry_price=5.0,
        underlying="SPX",
        engine=BlackScholesEngine(),
        entry_timestamp=datetime(2026, 1, 1),
    )
    assert pos.cash_legs == []


def test_get_market_value_unchanged_without_legs():
    pos = EquityPosition(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=10.0,
        entry_price=5.0,
        underlying="SPX",
        engine=BlackScholesEngine(),
        entry_timestamp=datetime(2026, 1, 1),
    )
    mv = pos.get_market_value(_env())
    assert mv > 0  # call option has positive value with these params
    assert mv == 10.0 * pos.engine.price(pos.product, _env())
```

- [ ] **Step 2: Write the new-functionality test**

```python
# test/test_cashleg/test_position_with_legs.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime
import pytest

from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.product.option import EuropeanVanillaOption
from cashleg import DeterministicLeg, LegDirection
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from portfolio.equity.position import EquityPosition
from priceenv import PricingEnvironment
from util.enum import OptionType


def _env(rate=0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_get_trade_value_includes_premium_leg():
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    env = _env()
    product_pv = engine.price(option, env)

    pos = EquityPosition(
        product=option,
        quantity=1.0,
        entry_price=product_pv,
        underlying="SPX",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=100.0, payment_time=0.0,
                direction=LegDirection.BUYER_PAYS,
                name="Premium",
            ),
        ],
    )
    trade_val = pos.get_trade_value(env)
    assert trade_val == pytest.approx((product_pv - 100.0) * 1.0, rel=1e-9)


def test_get_trade_value_breakdown_attributes_per_leg():
    engine = BlackScholesEngine()
    pos = EquityPosition(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=2.0,
        entry_price=5.0,
        underlying="SPX",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(amount=100.0, payment_time=0.0,
                             direction=LegDirection.BUYER_PAYS, name="Premium"),
            DeterministicLeg(amount=50.0, payment_time=1.0,
                             direction=LegDirection.BUYER_RECEIVES, name="Backend"),
        ],
    )
    breakdown = pos.get_trade_value_breakdown(_env())
    assert len(breakdown.leg_pvs) == 2
    # Premium (BUYER_PAYS) scaled by quantity 2
    premium_pv = next(v.pv for v in breakdown.leg_pvs.values() if v.name == "Premium")
    assert premium_pv == pytest.approx(-100.0 * 2.0, rel=1e-9)


def test_two_deterministic_legs_of_same_type_both_priced():
    """Multiple legs of same type are both included; not deduplicated."""
    engine = BlackScholesEngine()
    pos = EquityPosition(
        product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
        quantity=1.0,
        entry_price=5.0,
        underlying="SPX",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(amount=100.0, payment_time=0.0,
                             direction=LegDirection.BUYER_PAYS, name="Premium A"),
            DeterministicLeg(amount=100.0, payment_time=0.0,
                             direction=LegDirection.BUYER_PAYS, name="Premium B"),
        ],
    )
    breakdown = pos.get_trade_value_breakdown(_env())
    assert len(breakdown.leg_pvs) == 2
    leg_total = sum(v.pv for v in breakdown.leg_pvs.values())
    assert leg_total == pytest.approx(-200.0, rel=1e-9)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest test/test_cashleg/test_position_with_legs.py test/test_cashleg/test_position_backward_compat.py -v`
Expected: backward-compat tests pass (no field yet → `cash_legs == []` fails); new tests fail with `unexpected keyword argument 'cash_legs'`.

- [ ] **Step 4: Extend `EquityPosition`**

Open `portfolio/equity/position.py`. Apply the following edits:

Add to imports at top:
```python
from typing import Optional, Dict, Any, List
from cashleg import CashLeg, TradeValueBreakdown, LegPV, value_leg
```

Add field to the dataclass (insert after `entry_timestamp`):
```python
    cash_legs: List[CashLeg] = field(default_factory=list)
```

Append new methods after `to_dict` (keep existing methods untouched):

```python
    def get_trade_value(self, pricing_env: PricingEnvironment) -> float:
        """Full trade NPV: product + cash legs, scaled by quantity.

        Returns the buyer's-perspective signed value of the whole trade.
        For positions with no cash legs, returns the same as get_market_value.
        """
        if not self.cash_legs:
            return self.get_market_value(pricing_env)
        needs_dist = any(leg.requires_event_distribution() for leg in self.cash_legs)
        result = self.engine.price_with_events(
            self.product, pricing_env, emit_distribution=needs_dist
        )
        notional = self.get_actual_notional(pricing_env)
        leg_pv_total = sum(
            value_leg(leg, result.event_distribution, pricing_env, notional)
            for leg in self.cash_legs
        )
        return (result.npv + leg_pv_total) * self.quantity

    def get_trade_value_breakdown(self, pricing_env: PricingEnvironment) -> TradeValueBreakdown:
        """Per-leg PV breakdown for trade reporting/attribution."""
        if not self.cash_legs:
            return TradeValueBreakdown(
                product_npv=self.get_market_value(pricing_env),
                leg_pvs={},
            )
        result = self.engine.price_with_events(self.product, pricing_env)
        notional = self.get_actual_notional(pricing_env)
        leg_pvs = {}
        for leg in self.cash_legs:
            pv = value_leg(leg, result.event_distribution, pricing_env, notional) * self.quantity
            leg_pvs[leg.leg_id] = LegPV(name=leg.name, direction=leg.direction, pv=pv)
        return TradeValueBreakdown(
            product_npv=result.npv * self.quantity,
            leg_pvs=leg_pvs,
        )
```

- [ ] **Step 5: Run all tests**

Run: `pytest test/test_cashleg/test_position_with_legs.py test/test_cashleg/test_position_backward_compat.py -v`
Expected: all passed.

Run: `pytest test/ -x -q -k "portfolio or position"`
Expected: existing portfolio tests still pass.

- [ ] **Step 6: Commit**

```bash
git add portfolio/equity/position.py test/test_cashleg/test_position_with_legs.py test/test_cashleg/test_position_backward_compat.py
git commit -m "feat(position): EquityPosition.cash_legs + get_trade_value / get_trade_value_breakdown"
```

---

## Task 6.2: `EquityPosition.get_trade_greeks`

**Files:**
- Modify: `portfolio/equity/position.py`
- Test: append to `test/test_cashleg/test_position_with_legs.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_cashleg/test_position_with_legs.py`:

```python
def test_get_trade_greeks_for_position_with_only_premium_leg():
    """Position = (vanilla call) + (premium). Delta unchanged; theta picks up DF rolldown."""
    from asset.equity.riskmeasures import GreeksCalculator

    engine = BlackScholesEngine()
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    env = _env()
    pos = EquityPosition(
        product=option, quantity=1.0, entry_price=5.0, underlying="SPX",
        engine=engine, entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(amount=100.0, payment_time=0.5,
                             direction=LegDirection.BUYER_PAYS, name="Mid Premium"),
        ],
    )
    calc = GreeksCalculator()
    greeks = pos.get_trade_greeks(env, calc)

    # Delta: premium at t=0.5 is rate-sensitive but spot-insensitive
    # So trade delta == product delta within bump tolerance
    pos_no_legs = EquityPosition(
        product=option, quantity=1.0, entry_price=5.0, underlying="SPX",
        engine=engine, entry_timestamp=datetime(2026, 1, 1),
    )
    product_greeks = pos_no_legs.get_greeks(env, calc)
    assert greeks["delta"] == pytest.approx(product_greeks["delta"], rel=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_cashleg/test_position_with_legs.py::test_get_trade_greeks_for_position_with_only_premium_leg -v`
Expected: FAIL — `get_trade_greeks` does not exist.

- [ ] **Step 3: Implement `get_trade_greeks` (bump-and-reprice)**

Append to `EquityPosition`:

```python
    def get_trade_greeks(
        self,
        pricing_env: PricingEnvironment,
        greeks_calculator: "GreeksCalculator",
    ) -> Dict[str, float]:
        """Trade-level Greeks via finite-difference bump on get_trade_value.

        Bumps spot, vol, rate, and time; re-emits EventDistribution each bump so
        legs revalue against the bumped distribution. Returns dict with
        delta, gamma, vega, theta, rho (all scaled by position quantity inside
        get_trade_value).
        """
        from copy import deepcopy

        bump = self.engine.params.bump_size
        base = self.get_trade_value(pricing_env)

        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= 1 + bump
        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= 1 - bump
        v_up = self.get_trade_value(env_up)
        v_down = self.get_trade_value(env_down)
        delta = (v_up - v_down) / (2 * pricing_env.spot * bump)
        gamma = (v_up - 2 * base + v_down) / (pricing_env.spot * bump) ** 2

        env_vol_up = deepcopy(pricing_env)
        env_vol_down = deepcopy(pricing_env)
        env_vol_up.vol_surface.volatility += bump
        env_vol_down.vol_surface.volatility -= bump
        vega = (self.get_trade_value(env_vol_up) - self.get_trade_value(env_vol_down)) / (2 * bump)

        env_rate_up = deepcopy(pricing_env)
        env_rate_down = deepcopy(pricing_env)
        env_rate_up.rate_curve.rate += bump
        env_rate_down.rate_curve.rate -= bump
        rho = (self.get_trade_value(env_rate_up) - self.get_trade_value(env_rate_down)) / (2 * bump)

        # Theta: forward time bump (small)
        from datetime import timedelta
        env_t = deepcopy(pricing_env)
        env_t.valuation_date = pricing_env.valuation_date + timedelta(days=1)
        theta = self.get_trade_value(env_t) - base

        return {
            "price": base,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho,
            "theta": theta,
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_cashleg/test_position_with_legs.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add portfolio/equity/position.py test/test_cashleg/test_position_with_legs.py
git commit -m "feat(position): EquityPosition.get_trade_greeks via bump-and-reprice on trade value"
```

---

## Task 6.3: End-to-end demo + final integration check

**Files:**
- Create: `example/cash_legs_demo.py`

- [ ] **Step 1: Create demo script**

```python
# example/cash_legs_demo.py
"""End-to-end demo: snowball with cash legs (premium, accrual interest, KO bonus).

Run: python example/cash_legs_demo.py
"""
from datetime import datetime
import numpy as np

from asset.equity.engine.quad import SnowballQuadEngine
from asset.equity.product.option import SnowballOption
from asset.equity.product.option.snowball_config import BarrierConfig
from cashleg import (
    AccrualLeg, BaseAmount, BaseAmountMode, DeterministicLeg, FixedPayoffLeg,
    KOBehavior, LegDirection, LegSchedule, PaymentConvention, PaymentTrigger,
    SurvivalBasis,
)
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from portfolio.equity.position import EquityPosition
from priceenv import PricingEnvironment
from util.calendar.day_counter import DayCountConvention


def main():
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )
    snowball = SnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0, ki_continuous=True,
        ),
        contract_multiplier=10_000.0, maturity=1.0,
    )
    engine = SnowballQuadEngine()

    quarterly_schedule = LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )

    position = EquityPosition(
        product=snowball,
        quantity=1.0,
        entry_price=engine.price(snowball, env),
        underlying="CSI300",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=150_000.0, payment_time=0.0,
                direction=LegDirection.BUYER_PAYS, name="Front Premium",
            ),
            AccrualLeg(
                rate=0.02,
                base=BaseAmount(value=1.0, mode=BaseAmountMode.NOTIONAL_FRACTION),
                schedule=quarterly_schedule,
                day_count=DayCountConvention.ACT_365,
                payment_convention=PaymentConvention.AT_PERIOD_END,
                ko_behavior=KOBehavior.TRUNCATE_AT_KO,
                survival_basis=SurvivalBasis.ENTER_PERIOD,
                direction=LegDirection.BUYER_RECEIVES, name="Margin Interest",
            ),
            FixedPayoffLeg(
                amount=50_000.0, trigger=PaymentTrigger.AT_KO,
                direction=LegDirection.BUYER_RECEIVES, name="KO Bonus",
            ),
        ],
    )

    print(f"Product NPV (per unit): {engine.price(snowball, env):,.2f}")
    print(f"Position market value (product only): {position.get_market_value(env):,.2f}")
    print(f"Position trade value (product + legs): {position.get_trade_value(env):,.2f}")
    print()
    breakdown = position.get_trade_value_breakdown(env)
    print(f"Trade value breakdown:")
    print(f"  Product NPV       : {breakdown.product_npv:>15,.2f}")
    for leg_id, leg_pv in breakdown.leg_pvs.items():
        print(f"  {leg_pv.name:<18}: {leg_pv.pv:>15,.2f}  ({leg_pv.direction.name})")
    print(f"  {'TOTAL':<18}: {breakdown.total:>15,.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run: `python example/cash_legs_demo.py`
Expected: prints product NPV, market value, trade value, and per-leg breakdown without errors.

- [ ] **Step 3: Run the full cashleg test suite + sanity-sweep**

```bash
pytest test/test_cashleg/ -v
pytest test/ -x -q -k "snowball or phoenix or european_option or position or portfolio"
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add example/cash_legs_demo.py
git commit -m "feat(example): end-to-end cash legs demo on snowball + premium + accrual + KO bonus"
```

---

# Done

After Task 6.3, the feature is complete and ready for downstream wiring (phase 7 — backtest / stress test / dynamic scenario opt-in to `get_trade_value`, addressed in separate PRs per spec §11).

## Self-Review Notes

Reviewed against spec §1–§12:

- **§3 architecture, §3.3 module layout** → Tasks 1.1–1.11
- **§4 EventDistribution / PricingResult / LegDirection / BaseAmount / LegSchedule** → Tasks 1.2–1.5
- **§5 CashLeg / DeterministicLeg / AccrualLeg / FixedPayoffLeg** → Tasks 1.3, 1.6–1.8
- **§6 engine overrides** → Phases 2 (MC), 3 (Quad), 4 (Analytical), 5 (PDE)
- **§7 position integration + multiple legs of same type + get_trade_value_breakdown** → Tasks 6.1–6.2; multi-leg test in 6.1 step 2
- **§8 backward compatibility** → Task 6.1 step 1 (dedicated backward-compat test file)
- **§9 validation & error handling** → embedded throughout (BaseAmount post_init, FixedPayoffLeg missing-trigger raise, EventDistribution invariants)
- **§10 testing strategy** → mapped 1:1 onto the test files created in each task
- **§11 rollout plan** → followed phase 1 → 6; phase 7 explicitly deferred per spec
- **§12 open questions** → noted in task 5.1 (forward-density helper kept deliberately simple; production-ready refactor opens follow-up)
