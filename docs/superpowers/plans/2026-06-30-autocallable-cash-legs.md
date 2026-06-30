# Autocallable-Driven Cash Legs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `AutocallableCashLeg` that prices Snowball/Phoenix-driven cash legs (margin/prepayment, backend premium, backend interest, rebate, minimum return) from the parent autocallable's `EventDistribution`, and give Phoenix PDE/QUAD the native event stats this requires.

**Architecture:** A new `CashLeg` subclass consumes the parent's per-observation KO (or coupon) probabilities + terminal buckets and applies a per-observation accrual schedule with explicit workbook-sourced settlement times; Greeks flow free through the existing `EquityPosition.get_trade_greeks` bump loop. To make every Snowball/Phoenix × MC/PDE/QUAD combination method-consistent, native `calculate_event_stats` is added to `PhoenixPDESolver` and `PhoenixQuadEngine` (mirroring the Snowball implementations + coupon indicator surfaces; no MC).

**Tech Stack:** Python 3.11+, NumPy, dataclasses; pytest (pytest-xdist). Library code under `quantark.*`.

**Spec:** `docs/superpowers/specs/2026-06-30-autocallable-cash-legs-design.md`

## Global Constraints

- Canonical imports only: `quantark.*` (never the legacy flat names).
- Numerical ops in **library code** (`quantark/…`) go via `quantark.util.numerical` — no hardcoded tolerances, no raw float `==`. Use `Tolerance`, `almost_equal`, `is_close`, `safe_*`, `validate_*`. (Test assertions are exempt: `np.testing.assert_allclose(atol=…)`, `==` for exact algebraic identities, and explicit MC-band tolerances are expected in tests and used throughout this plan.)
- Exceptions: raise `ValidationError` for bad inputs (from `quantark.util.exceptions`). Fail loud — no padding, truncation, silent realignment, or fallback semantics.
- **No MC inside PDE/QUAD:** the native Phoenix PDE/QUAD event-stats code must not import or instantiate any `*MCEngine`.
- All dataclasses `frozen=True`; subclass leg fields must carry defaults (dataclass inheritance requirement — see Task 1).
- Tests live under `test/test_cashleg/`, named `test_*.py`. Run with `.venv/bin/python -m pytest`.
- Sign convention: `CashLeg.sign()` = `+1` (`BUYER_RECEIVES`) / `−1` (`BUYER_PAYS`). Leg PV is signed, from the buyer's perspective.
- Leg `notional` is absolute per-unit; the position multiplies leg PV by `quantity`. The leg's `value()` uses `self.notional`, not the `position_notional` argument.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `quantark/cashleg/autocallable_leg.py` | `AutocallableCashLeg` + `AutocallableLegType` / `PvFormula` / `AccrualBasis` enums + valuation | Create |
| `quantark/cashleg/__init__.py` | Public exports | Modify |
| `quantark/asset/equity/engine/pde/snowball_pde_solver.py` | Refactor `calculate_event_stats` into reusable `_compute_event_stats` + `_event_stats_product_type()` + `_make_event_stats()` hooks | Modify |
| `quantark/asset/equity/engine/pde/phoenix_pde_solver.py` | Native `calculate_event_stats` (KO/KI reuse + coupon surfaces) | Modify |
| `quantark/asset/equity/engine/quad/snowball_quad_engine.py` | Same reusable-hook refactor as the PDE | Modify |
| `quantark/asset/equity/engine/quad/phoenix_quad_engine.py` | Native `calculate_event_stats` (replace MC delegation) | Modify |
| `quantark/cashleg/CLAUDE.md` | Document `AutocallableCashLeg`; fix stale module list | Modify |
| `test/test_cashleg/_autocallable_helpers.py` | Shared builders: env / Snowball / Phoenix / engines / leg factories | Create |
| `test/test_cashleg/test_autocallable_leg.py` | Unit tests vs independent re-implementation, synthetic `EventDistribution` | Create |
| `test/test_cashleg/test_autocallable_leg_position.py` | Position PV + Greeks + quantity contract | Create |
| `test/test_cashleg/test_autocallable_leg_distribution.py` | `price_with_events` → `EventType.COUPON` mapping + COUPON-basis position valuation | Create |
| `test/test_cashleg/test_autocallable_leg_golden.py` | Required golden-case parity (PV + per-leg delta/gamma) across engines | Create |
| `test/test_cashleg/fixtures/autocallable_golden_case.json` | Sanitized golden inputs + frozen targets | Create |
| `test/test_cashleg/test_phoenix_pde_event_stats.py` | Phoenix PDE event stats vs Phoenix MC; no-MC-import assertion | Create |
| `test/test_cashleg/test_phoenix_quad_event_stats.py` | Phoenix QUAD event stats vs Phoenix MC; no-MC-import assertion | Create |

The shared helper module (`_autocallable_helpers.py`) is created in Task 6 and
imported by every engine/position/golden test, so no test imports a non-existent
fixture module.

---

## Task 0: Reconnaissance & test-harness baseline

**Files:**
- Read only.

- [ ] **Step 1: Confirm cashleg exports and test layout**

Run:
```bash
cd /Users/fuxinyao/quant-ark
sed -n '1,40p' quantark/cashleg/__init__.py
ls test/test_cashleg/ 2>/dev/null
.venv/bin/python -c "from quantark.util.numerical import Tolerance; print([a for a in dir(Tolerance) if not a.startswith('_')])"
# The builder patterns Task 6 copies into the shared helper module:
sed -n '25,120p' test/test_snowball_mc_engine.py   # create_pricing_env / create_basic_barrier_config / create_standard_snowball
sed -n '1,70p'   test/test_phoenix_quad.py          # create_pricing_env / create_phoenix
```
Expected: see the current `__all__` (DeterministicLeg/AccrualLeg/FixedPayoffLeg + enums), the `test/test_cashleg/` dir, the available `Tolerance` attributes (later tasks use `Tolerance.PROBABILITY` for the schedule-identity tolerance; if a more specific time tolerance exists, prefer it), and the verbatim builder helpers that Task 6 ports into `test/test_cashleg/_autocallable_helpers.py`.

- [ ] **Step 2: Confirm the EventDistribution construction contract**

Run:
```bash
.venv/bin/python - <<'PY'
import numpy as np
from quantark.cashleg.event_distribution import EventDistribution, EventType
ed = EventDistribution(
    event_times=np.array([0.5, 1.0]),
    event_dates=None,
    probabilities={EventType.KO: np.array([0.3, 0.2]),
                   EventType.MATURITY_NO_KO: 0.4,
                   EventType.MATURITY_WITH_KI: 0.1},
    survival_probability=np.array([1.0, 0.7, 0.5]),
)
print("ok", ed.survival_at(0.75), ed.probabilities[EventType.KO])
PY
```
Expected: prints `ok ...` with no exception (this is the synthetic-distribution helper shape the unit tests reuse). If it raises, read `quantark/cashleg/event_distribution.py` `_validate_invariants` and adjust the helper in Task 2 accordingly.

This task has no commit (read-only baseline).

---

## Task 1: `AutocallableCashLeg` skeleton — enums, fields, validation

**Files:**
- Create: `quantark/cashleg/autocallable_leg.py`
- Test: `test/test_cashleg/test_autocallable_leg.py`

**Interfaces:**
- Consumes: `CashLeg`, `LegDirection` (`quantark.cashleg.base`); `EventType` (`quantark.cashleg.event_distribution`); `ValidationError` (`quantark.util.exceptions`).
- Produces:
  - `class AutocallableLegType(Enum)`: `MARGIN`, `BACKEND_PREMIUM`, `BACKEND_INTEREST`, `REBATE`, `MINIMUM_RETURN`.
  - `class PvFormula(Enum)`: `NORMAL`, `NOTIONAL_MINUS_PAYOFF`.
  - `class AccrualBasis(Enum)`: `KO_MATURITY`, `COUPON`.
  - `@dataclass(frozen=True) class AutocallableCashLeg(CashLeg)` with fields: `leg_type: Optional[AutocallableLegType]=None`, `notional: float=0.0`, `rate: float=0.0`, `observation_schedule: Sequence[float]=()`, `accrual_factors: Sequence[float]=()`, `settlement_schedule: Sequence[float]=()`, `terminal_accrual_factor: float=0.0`, `terminal_settlement_time: float=0.0`, `pv_formula: PvFormula=PvFormula.NORMAL`, `accrual_basis: AccrualBasis=AccrualBasis.KO_MATURITY`, `terminal_events: frozenset=frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})`, `notional_settlement_time: Optional[float]=None`.
  - `value(self, event_dist, env, position_notional) -> float` (filled in Task 3).
  - `requires_event_distribution(self) -> bool` returns `True`.

- [ ] **Step 1: Write the failing validation tests**

```python
# test/test_cashleg/test_autocallable_leg.py
import pytest

from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg, AutocallableLegType, PvFormula, AccrualBasis,
)
from quantark.util.exceptions import ValidationError


def _leg(**kw):
    base = dict(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.MARGIN,
        notional=1_000_000.0,
        rate=0.05,
        observation_schedule=(0.5, 1.0),
        accrual_factors=(0.5, 1.0),
        settlement_schedule=(0.51, 1.01),
        terminal_accrual_factor=1.0,
        terminal_settlement_time=1.01,
    )
    base.update(kw)
    return AutocallableCashLeg(**base)


def test_constructs_with_defaults():
    leg = _leg()
    assert leg.pv_formula is PvFormula.NORMAL
    assert leg.accrual_basis is AccrualBasis.KO_MATURITY
    assert leg.terminal_events == frozenset(
        {EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}
    )
    assert leg.requires_event_distribution() is True
    assert leg.sign() == 1


def test_missing_leg_type_rejected():
    with pytest.raises(ValidationError):
        _leg(leg_type=None)


def test_unequal_schedule_lengths_rejected():
    with pytest.raises(ValidationError):
        _leg(accrual_factors=(0.5,))  # len 1 vs obs len 2


def test_future_dated_notional_settlement_rejected_for_margin_formula():
    with pytest.raises(ValidationError):
        _leg(pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF,
             notional_settlement_time=0.25)


def test_terminal_events_must_be_maturity_buckets():
    with pytest.raises(ValidationError):
        _leg(terminal_events=frozenset({EventType.KO}))


def test_nan_accrual_factor_rejected():
    with pytest.raises(ValidationError):
        _leg(accrual_factors=(0.5, float("nan")))


def test_negative_observation_time_rejected():
    with pytest.raises(ValidationError):
        _leg(observation_schedule=(-0.1, 1.0), settlement_schedule=(-0.1, 1.0))


def test_negative_notional_rejected():
    with pytest.raises(ValidationError):
        _leg(notional=-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py -x -q`
Expected: FAIL with `ModuleNotFoundError: quantark.cashleg.autocallable_leg`.

- [ ] **Step 3: Write the module skeleton with validation**

```python
# quantark/cashleg/autocallable_leg.py
"""Autocallable-driven cash legs (margin/backend/rebate/minimum-return)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from quantark.cashleg.base import CashLeg
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, almost_equal, is_valid_number, validate_positive


class AutocallableLegType(Enum):
    """Label for an autocallable adjustment cash leg (no hidden behaviour)."""

    MARGIN = "margin"
    BACKEND_PREMIUM = "backend_premium"
    BACKEND_INTEREST = "backend_interest"
    REBATE = "rebate"
    MINIMUM_RETURN = "minimum_return"


class PvFormula(Enum):
    """How the discounted contingent return R maps to leg PV."""

    NORMAL = "normal"                          # PV = sign * R
    NOTIONAL_MINUS_PAYOFF = "notional_minus_payoff"  # PV = sign * (notional - R)


class AccrualBasis(Enum):
    """Which probability stream drives the contingent accruals."""

    KO_MATURITY = "ko_maturity"   # KO observation dates + terminal branch
    COUPON = "coupon"             # Phoenix coupon observations + terminal branch


_TERMINAL_BUCKETS = frozenset(
    {EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}
)
_BASIS_EVENT = {
    AccrualBasis.KO_MATURITY: EventType.KO,
    AccrualBasis.COUPON: EventType.COUPON,
}


@dataclass(frozen=True)
class AutocallableCashLeg(CashLeg):
    """KO/coupon-contingent return leg priced off a parent EventDistribution."""

    leg_type: Optional[AutocallableLegType] = None
    notional: float = 0.0
    rate: float = 0.0
    observation_schedule: Sequence[float] = ()
    accrual_factors: Sequence[float] = ()
    settlement_schedule: Sequence[float] = ()
    terminal_accrual_factor: float = 0.0
    terminal_settlement_time: float = 0.0
    pv_formula: PvFormula = PvFormula.NORMAL
    accrual_basis: AccrualBasis = AccrualBasis.KO_MATURITY
    terminal_events: frozenset = field(default_factory=lambda: _TERMINAL_BUCKETS)
    notional_settlement_time: Optional[float] = None

    def __post_init__(self) -> None:
        if self.leg_type is None or not isinstance(self.leg_type, AutocallableLegType):
            raise ValidationError("AutocallableCashLeg requires a valid leg_type")
        if not isinstance(self.pv_formula, PvFormula):
            raise ValidationError(f"Invalid PvFormula: {self.pv_formula}")
        if not isinstance(self.accrual_basis, AccrualBasis):
            raise ValidationError(f"Invalid AccrualBasis: {self.accrual_basis}")
        # notional is absolute per-unit ⇒ finite and non-negative (uses the
        # project numerical validator; see Global Constraints).
        validate_positive(self.notional, "notional", allow_zero=True)
        if not is_valid_number(self.rate):
            raise ValidationError(f"rate must be finite, got {self.rate}")
        if not np.isfinite(float(self.terminal_accrual_factor)):
            raise ValidationError("terminal_accrual_factor must be finite")
        if not np.isfinite(float(self.terminal_settlement_time)) or self.terminal_settlement_time < 0.0:
            raise ValidationError("terminal_settlement_time must be finite and >= 0")

        n = len(self.observation_schedule)
        if not (len(self.accrual_factors) == n and len(self.settlement_schedule) == n):
            raise ValidationError(
                "observation_schedule, accrual_factors, settlement_schedule "
                f"must share one length; got {n}, {len(self.accrual_factors)}, "
                f"{len(self.settlement_schedule)}"
            )

        for label, seq in (
            ("observation_schedule", self.observation_schedule),
            ("accrual_factors", self.accrual_factors),
            ("settlement_schedule", self.settlement_schedule),
        ):
            arr = np.asarray(seq, dtype=float)
            if arr.size and not np.all(np.isfinite(arr)):
                raise ValidationError(f"{label} must be all-finite, got {seq!r}")
        obs_arr = np.asarray(self.observation_schedule, dtype=float)
        ss_arr = np.asarray(self.settlement_schedule, dtype=float)
        if np.any(obs_arr < 0.0) or np.any(ss_arr < 0.0):
            raise ValidationError(
                "observation_schedule and settlement_schedule times must be >= 0"
            )

        if not self.terminal_events or not self.terminal_events.issubset(_TERMINAL_BUCKETS):
            raise ValidationError(
                "terminal_events must be a non-empty subset of "
                "{MATURITY_NO_KO, MATURITY_WITH_KI}"
            )

        if self.pv_formula is PvFormula.NOTIONAL_MINUS_PAYOFF:
            if self.notional_settlement_time is not None:
                t0 = float(self.notional_settlement_time)
                if not np.isfinite(t0) or t0 > Tolerance.PROBABILITY:
                    raise ValidationError(
                        "NOTIONAL_MINUS_PAYOFF requires the notional to be an "
                        "outstanding claim as of valuation (notional_settlement_time "
                        f"<= 0); got {self.notional_settlement_time}. Use a "
                        "DeterministicLeg for a future-dated notional exchange."
                    )

    def requires_event_distribution(self) -> bool:
        return True

    def value(
        self, event_dist: EventDistribution, env, position_notional: float
    ) -> float:
        raise NotImplementedError  # implemented in Task 3
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py -x -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add quantark/cashleg/autocallable_leg.py test/test_cashleg/test_autocallable_leg.py
git commit -m "feat(cashleg): AutocallableCashLeg skeleton with field validation"
```

---

## Task 2: Synthetic-distribution test helper + independent re-implementation

**Files:**
- Test: `test/test_cashleg/test_autocallable_leg.py` (extend)

**Interfaces:**
- Produces (test-local): `make_distribution(...)` building a valid `EventDistribution`; `expected_R(...)` an independent re-implementation of the §5 sum used as the oracle in Task 3.

- [ ] **Step 1: Add the helper and a flat-curve env stub**

```python
# append to test/test_cashleg/test_autocallable_leg.py
import numpy as np
from quantark.cashleg.event_distribution import EventDistribution


class FlatEnv:
    """Minimal pricing-env stub: continuously-compounded flat rate."""

    def __init__(self, rate: float = 0.02):
        self._r = rate

    def get_discount_factor(self, t: float) -> float:
        return float(np.exp(-self._r * float(t)))


def make_distribution(event_times, ko_probs, mat_no_ko, mat_with_ki,
                      coupon_probs=None):
    event_times = np.asarray(event_times, dtype=float)
    ko_probs = np.asarray(ko_probs, dtype=float)
    # survival[0]=1; survival[i+1]=survival[i]-ko_probs[i] (one leading value)
    survival = np.concatenate([[1.0], 1.0 - np.cumsum(ko_probs)])
    probs = {
        EventType.KO: ko_probs,
        EventType.MATURITY_NO_KO: float(mat_no_ko),
        EventType.MATURITY_WITH_KI: float(mat_with_ki),
    }
    if coupon_probs is not None:
        probs[EventType.COUPON] = np.asarray(coupon_probs, dtype=float)
    return EventDistribution(
        event_times=event_times,
        event_dates=None,
        probabilities=probs,
        survival_probability=survival,
    )


def expected_R(notional, rate, accrual_factors, settlement_schedule,
               event_probs, terminal_accrual_factor, p_term,
               terminal_settlement_time, env):
    af = np.asarray(accrual_factors, float)
    ss = np.asarray(settlement_schedule, float)
    ep = np.asarray(event_probs, float)
    df_obs = np.array([env.get_discount_factor(t) for t in ss])
    df_term = env.get_discount_factor(terminal_settlement_time)
    return notional * rate * (
        float(np.sum(af * ep * df_obs))
        + terminal_accrual_factor * p_term * df_term
    )
```

- [ ] **Step 2: Verify the helper builds a valid distribution**

```python
def test_helper_builds_valid_distribution():
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    assert ed.probabilities[EventType.KO].tolist() == [0.3, 0.2]
    assert abs(ed.survival_at(1.0) - 0.5) < 1e-12
```

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py::test_helper_builds_valid_distribution -x -q`
Expected: PASS. (If `EventDistribution.__post_init__` rejects the shape, reconcile against Task 0 Step 2 output before proceeding.)

- [ ] **Step 3: Commit**

```bash
git add test/test_cashleg/test_autocallable_leg.py
git commit -m "test(cashleg): synthetic EventDistribution helper + R oracle"
```

---

## Task 3: `value()` — KO_MATURITY valuation, both PV formulas, fail-loud guards

**Files:**
- Modify: `quantark/cashleg/autocallable_leg.py` (replace the `value` stub)
- Test: `test/test_cashleg/test_autocallable_leg.py` (extend)

**Interfaces:**
- Consumes: `EventDistribution.probabilities`, `EventDistribution.event_times`, `env.get_discount_factor`.
- Produces: signed `float` PV per spec §5.

- [ ] **Step 1: Write failing valuation + guard tests**

```python
def test_normal_ko_maturity_value_matches_oracle():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(
        pv_formula=PvFormula.NORMAL, rate=0.05, notional=1_000_000.0,
        observation_schedule=(0.5, 1.0), accrual_factors=(0.5, 1.0),
        settlement_schedule=(0.5, 1.0), terminal_accrual_factor=1.0,
        terminal_settlement_time=1.0,
    )
    p_term = 0.4 + 0.1
    R = expected_R(1_000_000.0, 0.05, (0.5, 1.0), (0.5, 1.0),
                   [0.3, 0.2], 1.0, p_term, 1.0, env)
    assert abs(leg.value(ed, env, 0.0) - R) < 1e-6


def test_margin_formula_is_notional_minus_R():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(
        pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF, rate=0.05,
        notional=1_000_000.0, observation_schedule=(0.5, 1.0),
        accrual_factors=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
        terminal_accrual_factor=1.0, terminal_settlement_time=1.0,
    )
    R = expected_R(1_000_000.0, 0.05, (0.5, 1.0), (0.5, 1.0),
                   [0.3, 0.2], 1.0, 0.5, 1.0, env)
    assert abs(leg.value(ed, env, 0.0) - (1_000_000.0 - R)) < 1e-6


def test_buyer_pays_flips_sign():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    recv = _leg(direction=LegDirection.BUYER_RECEIVES)
    pays = _leg(direction=LegDirection.BUYER_PAYS)
    assert abs(recv.value(ed, env, 0.0) + pays.value(ed, env, 0.0)) < 1e-9


def test_missing_ko_stream_raises():
    env = FlatEnv(0.02)
    trivial = EventDistribution.trivial(1.0)  # only MATURITY_NO_KO float
    leg = _leg()
    with pytest.raises(ValidationError):
        leg.value(trivial, env, 0.0)


def test_shifted_observation_schedule_raises():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(observation_schedule=(0.49, 1.0))  # same length, shifted
    with pytest.raises(ValidationError):
        leg.value(ed, env, 0.0)


def test_missing_terminal_bucket_raises():
    # Distribution lacks MATURITY_WITH_KI, which the default terminal_events needs.
    env = FlatEnv(0.02)
    ed = EventDistribution(
        event_times=np.array([0.5, 1.0]),
        event_dates=None,
        probabilities={EventType.KO: np.array([0.3, 0.2]),
                       EventType.MATURITY_NO_KO: 0.5},
        survival_probability=np.array([1.0, 0.7, 0.5]),
    )
    leg = _leg(observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    with pytest.raises(ValidationError):
        leg.value(ed, env, 0.0)


@pytest.mark.parametrize("lt", list(AutocallableLegType))
def test_all_five_leg_types_value_finite(lt):
    # Requirement #6: every leg_type prices through the same path.
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    formula = (PvFormula.NOTIONAL_MINUS_PAYOFF
               if lt is AutocallableLegType.MARGIN else PvFormula.NORMAL)
    leg = _leg(leg_type=lt, pv_formula=formula,
               observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    assert np.isfinite(leg.value(ed, env, 0.0))


def test_value_ignores_position_notional_argument():
    # Contract: leg uses self.notional; the position applies quantity scaling.
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    assert leg.value(ed, env, 0.0) == leg.value(ed, env, 999_999.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py -x -q -k "value or formula or sign or raises"`
Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement `value()`**

Replace the `value` stub in `quantark/cashleg/autocallable_leg.py` with:

```python
    def value(
        self, event_dist: EventDistribution, env, position_notional: float
    ) -> float:
        af = np.asarray(self.accrual_factors, dtype=float)
        contingent = 0.0
        if af.size:
            event_type = _BASIS_EVENT[self.accrual_basis]
            prob = event_dist.probabilities.get(event_type)
            if not isinstance(prob, np.ndarray):
                raise ValidationError(
                    f"AutocallableCashLeg ({self.accrual_basis.value}) requires an "
                    f"array {event_type.value} stream in the parent EventDistribution; "
                    "the parent engine emitted none (trivial/unsupported distribution)."
                )
            obs = np.asarray(self.observation_schedule, dtype=float)
            ss = np.asarray(self.settlement_schedule, dtype=float)
            ev_times = np.asarray(event_dist.event_times, dtype=float)
            if prob.shape != obs.shape:
                raise ValidationError(
                    f"{event_type.value} length {prob.shape} != observation_schedule "
                    f"length {obs.shape}; align the leg to the parent's future grid."
                )
            if obs.shape != ev_times.shape or not np.all(
                [almost_equal(float(a), float(b), tol=Tolerance.PROBABILITY)
                 for a, b in zip(obs, ev_times)]
            ):
                raise ValidationError(
                    "observation_schedule does not match the parent's filtered "
                    "event_times (shifted/dropped observation); refusing to realign."
                )
            df_obs = np.array([env.get_discount_factor(float(t)) for t in ss])
            contingent = float(np.sum(af * prob * df_obs))

        for e in self.terminal_events:
            prob_e = event_dist.probabilities.get(e)
            if prob_e is None or not is_valid_number(prob_e):
                raise ValidationError(
                    f"terminal event {e.value} missing or non-scalar in the parent "
                    "EventDistribution; cannot value the terminal branch."
                )
        p_term = sum(float(event_dist.probabilities[e]) for e in self.terminal_events)
        df_term = env.get_discount_factor(float(self.terminal_settlement_time))
        contingent += float(self.terminal_accrual_factor) * p_term * float(df_term)

        R = float(self.notional) * float(self.rate) * contingent
        if self.pv_formula is PvFormula.NORMAL:
            return self.sign() * R
        return self.sign() * (float(self.notional) - R)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py -x -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add quantark/cashleg/autocallable_leg.py test/test_cashleg/test_autocallable_leg.py
git commit -m "feat(cashleg): AutocallableCashLeg KO_MATURITY valuation + fail-loud guards"
```

---

## Task 4: COUPON basis + `value_standalone` helper

**Files:**
- Modify: `quantark/cashleg/autocallable_leg.py` (add `value_standalone`)
- Test: `test/test_cashleg/test_autocallable_leg.py` (extend)

**Interfaces:**
- Consumes: `engine.price_with_events(product, env, emit_distribution=True) -> PricingResult` with `.event_distribution`.
- Produces: `value_standalone(self, parent_product, engine, env) -> float`.

- [ ] **Step 1: Write failing tests (COUPON basis + standalone)**

```python
def test_coupon_basis_uses_coupon_stream():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.1, 0.1], 0.4, 0.4,
                           coupon_probs=[0.7, 0.5])
    leg = _leg(accrual_basis=AccrualBasis.COUPON, rate=0.03,
               observation_schedule=(0.5, 1.0), accrual_factors=(0.5, 0.5),
               settlement_schedule=(0.5, 1.0), terminal_accrual_factor=0.0,
               terminal_settlement_time=1.0)
    R = expected_R(1_000_000.0, 0.03, (0.5, 0.5), (0.5, 1.0),
                   [0.7, 0.5], 0.0, 0.8, 1.0, env)
    assert abs(leg.value(ed, env, 0.0) - R) < 1e-6


def test_coupon_basis_missing_stream_raises():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.1, 0.1], 0.4, 0.4)  # no coupon
    leg = _leg(accrual_basis=AccrualBasis.COUPON)
    with pytest.raises(ValidationError):
        leg.value(ed, env, 0.0)


class _StubEngine:
    def __init__(self, ed):
        self._ed = ed
    def price_with_events(self, product, env, emit_distribution=True):
        from quantark.cashleg.event_distribution import PricingResult
        return PricingResult(npv=0.0, event_distribution=self._ed)


def test_value_standalone_routes_through_price_with_events():
    env = FlatEnv(0.02)
    ed = make_distribution([0.5, 1.0], [0.3, 0.2], 0.4, 0.1)
    leg = _leg(observation_schedule=(0.5, 1.0), settlement_schedule=(0.5, 1.0),
               terminal_settlement_time=1.0)
    direct = leg.value(ed, env, 0.0)
    via = leg.value_standalone(parent_product=object(), engine=_StubEngine(ed), env=env)
    assert abs(direct - via) < 1e-12
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py -x -q -k "coupon or standalone"`
Expected: FAIL (`AttributeError: value_standalone`).

- [ ] **Step 3: Implement `value_standalone`**

Add to `AutocallableCashLeg` (COUPON basis already works via `_BASIS_EVENT`):

```python
    def value_standalone(self, parent_product, engine, env) -> float:
        """Price this leg directly against a parent product + engine."""
        result = engine.price_with_events(parent_product, env, emit_distribution=True)
        return self.value(result.event_distribution, env, float(self.notional))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/cashleg/autocallable_leg.py test/test_cashleg/test_autocallable_leg.py
git commit -m "feat(cashleg): COUPON accrual basis + value_standalone helper"
```

---

## Task 5: Public exports

**Files:**
- Modify: `quantark/cashleg/__init__.py`
- Test: `test/test_cashleg/test_autocallable_leg.py` (extend)

**Interfaces:**
- Produces: `from quantark.cashleg import AutocallableCashLeg, AutocallableLegType, PvFormula, AccrualBasis`.

- [ ] **Step 1: Write the failing export test**

```python
def test_public_exports():
    import quantark.cashleg as cl
    for name in ("AutocallableCashLeg", "AutocallableLegType",
                 "PvFormula", "AccrualBasis"):
        assert hasattr(cl, name), name
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py::test_public_exports -x -q`
Expected: FAIL.

- [ ] **Step 3: Add exports**

Edit `quantark/cashleg/__init__.py`: add the import and extend `__all__` (match the existing style observed in Task 0 Step 1):

```python
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg,
    AutocallableLegType,
    PvFormula,
    AccrualBasis,
)
```
Append `"AutocallableCashLeg"`, `"AutocallableLegType"`, `"PvFormula"`, `"AccrualBasis"` to `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg.py::test_public_exports -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/cashleg/__init__.py test/test_cashleg/test_autocallable_leg.py
git commit -m "feat(cashleg): export AutocallableCashLeg and enums"
```

---

## Task 6: Shared test builders (`_autocallable_helpers.py`)

**Files:**
- Create: `test/test_cashleg/_autocallable_helpers.py`
- Test: `test/test_cashleg/test_helpers_smoke.py`

**Interfaces:**
- Produces: `make_env`, `make_snowball`, `make_phoenix`, `make_engine(kind, asset)`,
  `future_event_times(product, engine, env)`, `make_margin_leg(obs)`. Ported from
  the verbatim builders in `test/test_snowball_mc_engine.py:25-117` and
  `test/test_phoenix_quad.py:25-68` (confirmed in Task 0). No phantom imports —
  every engine/position/golden test imports this real module.
- **Import convention (record here):** confirm how the suite imports sibling test
  modules — `grep -rn "^from test\.\|^import test\." test/test_cashleg | head`.
  If the suite uses `from test.test_cashleg.X import …`, ensure `test/__init__.py`
  and `test/test_cashleg/__init__.py` exist (create empty if missing) so the
  package import resolves; if it uses bare `from X import …` with a `conftest.py`
  on `sys.path`, follow that instead. Use the chosen style verbatim in **all**
  new test files in this plan (replace the illustrative `from test.test_cashleg.
  _autocallable_helpers import …` lines if the repo convention differs).

- [ ] **Step 1: Create the helper module**

```python
# test/test_cashleg/_autocallable_helpers.py
from datetime import datetime
import numpy as np

from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.cashleg.base import LegDirection
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg, AutocallableLegType, PvFormula,
)


def make_env(spot=100.0, vol=0.20, rate=0.03, div_yield=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def make_snowball(ko_dates=(0.5, 1.0), ko_barrier=103.0, ko_rate=0.15,
                  ki_barrier=75.0, maturity=1.0):
    barrier = BarrierConfig(
        ko_barrier=ko_barrier, ko_rate=ko_rate,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier,
        contract_multiplier=1.0, maturity=maturity, is_reverse=False,
    )


def make_phoenix(ko_dates=(0.5, 1.0), ko_barrier=105.0,
                 coupon_barrier=(80.0, 80.0), memory=False,
                 coupon_pay=CouponPayType.INSTANT, maturity=1.0):
    barrier = BarrierConfig(
        ko_barrier=ko_barrier, ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(ko_dates),
        ki_barrier=None,
    )
    coupon = CouponBarrierConfig(
        coupon_barrier=list(coupon_barrier), coupon_rate=0.02,
        coupon_pay_type=coupon_pay,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=memory,
    )
    return PhoenixOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier,
        coupon_config=coupon,
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        contract_multiplier=1.0, maturity=maturity,
    )


_ENGINES = {
    ("snowball", "mc"): lambda: SnowballMCEngine(params=MCParams(num_paths=60_000, seed=7)),
    ("snowball", "pde"): lambda: SnowballPDESolver(params=PDEParams(grid_size=400, time_steps=400)),
    ("snowball", "quad"): lambda: SnowballQuadEngine(params=QuadParams(grid_points=1001)),
    ("phoenix", "mc"): lambda: PhoenixMCEngine(params=MCParams(num_paths=60_000, seed=7)),
    ("phoenix", "pde"): lambda: PhoenixPDESolver(params=PDEParams(grid_size=400, time_steps=400)),
    ("phoenix", "quad"): lambda: PhoenixQuadEngine(params=QuadParams(grid_points=1001)),
}


def make_engine(kind, asset="snowball"):
    return _ENGINES[(asset, kind)]()


def future_event_times(product, engine, env):
    """Parent's filtered future observation grid the leg must align to."""
    result = engine.price_with_events(product, env, emit_distribution=True)
    return np.asarray(result.event_distribution.event_times, dtype=float)


def make_margin_leg(obs, notional=1_000_000.0, rate=0.04,
                    direction=LegDirection.BUYER_RECEIVES):
    obs = [float(t) for t in obs]
    n = len(obs)
    return AutocallableCashLeg(
        direction=direction, leg_type=AutocallableLegType.MARGIN,
        notional=notional, rate=rate,
        observation_schedule=tuple(obs),
        accrual_factors=tuple(np.linspace(0.25, 1.0, n)),
        settlement_schedule=tuple(obs),
        terminal_accrual_factor=1.0, terminal_settlement_time=obs[-1],
        pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF,
    )
```

- [ ] **Step 2: Smoke test**

```python
# test/test_cashleg/test_helpers_smoke.py
import numpy as np
from test.test_cashleg._autocallable_helpers import (
    make_env, make_snowball, make_phoenix, make_engine, future_event_times,
)


def test_builders_and_future_times():
    env = make_env()
    sb = make_snowball(); ph = make_phoenix()
    assert make_engine("pde", "snowball").price(sb, env) > 0
    et = future_event_times(sb, make_engine("mc", "snowball"), env)
    assert et.ndim == 1 and et.size >= 1 and np.all(np.diff(et) > 0)
```

Run: `.venv/bin/python -m pytest test/test_cashleg/test_helpers_smoke.py -x -q`
Expected: PASS. If any import path is wrong, fix from the Task 0 output before continuing.

- [ ] **Step 3: Commit**

```bash
git add test/test_cashleg/_autocallable_helpers.py test/test_cashleg/test_helpers_smoke.py
git commit -m "test(cashleg): shared autocallable test builders"
```

## Phase B — Snowball PDE refactor → native Phoenix PDE event stats

**Design note (Tasks 7–12):** the leg consumes `ko_probability`,
`survival_probability`, `coupon_probability`, and the terminal buckets — all
**indicator expectations, independent of memory-coupon accumulation** (memory
changes coupon *amounts*, not the probability a coupon condition is met while
alive). So native event stats add **stacked KO + coupon-trigger indicator
surfaces**, mirroring Snowball — not the memory vector-state pricer.
`PhoenixMCEngine.calculate_event_stats` is the oracle.

**Coupon/KO simultaneity convention (verified, `phoenix_mc_engine.py:744-757`):**
`coupon_hit = coupon_barrier_met & alive_before`, where
`alive_before = (~is_ko) | (first_ko_idx >= obs_idx)` — i.e. a path that **knocks
out at this same observation still counts toward `coupon_probability[i]`**. In
the backward indicator recursion: **apply the KO jump first** (it zeros the KO
region across KO columns and *future* coupon columns — KO kills future coupons),
**then set this observation's coupon-trigger column on the coupon-pay mask**, so
the coupon-i indicator is populated even inside the KO region (counting the
simultaneous-KO coupon, matching MC).

### Task 7: Behavior-preserving Snowball PDE event-stats refactor

**Files:**
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py`

**Interfaces:**
- Produces (overridable hooks): `_event_stats_product_type() -> type` (default `SnowballOption`); `_make_event_stats(**fields) -> AutocallableEventStats`; `_compute_event_stats(product, env)` holding the existing body. `calculate_event_stats` becomes `guard + delegate`. Snowball behaviour byte-for-byte unchanged.

- [ ] **Step 1: Capture the Snowball baseline (characterization test)**

```python
# test/test_cashleg/test_phoenix_pde_event_stats.py
import pathlib
import numpy as np
import pytest
from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from test.test_cashleg._autocallable_helpers import make_env, make_snowball, make_phoenix, make_engine


def test_snowball_pde_event_stats_unchanged_after_refactor():
    env = make_env(); sb = make_snowball()
    s = make_engine("pde", "snowball").calculate_event_stats(sb, env)
    assert s is not None
    assert s.ko_probability.shape == s.ko_times.shape
    assert 0.0 <= float(np.sum(s.ko_probability)) <= 1.0 + 1e-9
```

Run before refactor: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py::test_snowball_pde_event_stats_unchanged_after_refactor -x -q`
Expected: PASS (baseline on current code).

- [ ] **Step 2: Refactor into hooks (no behaviour change)**

In `snowball_pde_solver.py`, replace the `calculate_event_stats` method:

```python
def calculate_event_stats(self, product, pricing_env):
    if not isinstance(product, self._event_stats_product_type()):
        return None
    if pricing_env is None:
        return None
    return self._compute_event_stats(product, pricing_env)

def _event_stats_product_type(self):
    return SnowballOption

def _make_event_stats(self, **fields):
    return AutocallableEventStats(**fields)

def _compute_event_stats(self, product, pricing_env):
    # ... the existing body that followed the old isinstance/None guards
    #     (lines ~330-612), unchanged, EXCEPT the final
    #     `return AutocallableEventStats(...)` becomes
    #     `return self._make_event_stats(...)` with the identical keyword fields.
```

- [ ] **Step 3: Run the refactor guard + Snowball PDE regression**

Run:
```bash
.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py::test_snowball_pde_event_stats_unchanged_after_refactor -x -q
.venv/bin/python -m pytest test/ -k "snowball and pde" -q
```
Expected: PASS (Snowball behaviour preserved).

- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/pde/snowball_pde_solver.py \
        test/test_cashleg/test_phoenix_pde_event_stats.py
git commit -m "refactor(pde): extract reusable Snowball event-stats hooks (no behaviour change)"
```

### Task 8: Phoenix PDE — native KO/KI event stats (coupon empty)

**Files:**
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py`
- Test: `test/test_cashleg/test_phoenix_pde_event_stats.py` (extend)

**Interfaces:**
- Produces: `PhoenixPDESolver.calculate_event_stats -> PhoenixEventStats` with correct KO/survival/ki (coupon arrays empty for now), constructing **no** MC engine.

- [ ] **Step 1: Write failing KO/survival + no-MC tests**

```python
def test_phoenix_pde_ko_survival_match_mc():
    env = make_env(); ph = make_phoenix()
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert isinstance(s_pde, PhoenixEventStats)
    np.testing.assert_allclose(s_pde.ko_times, s_mc.ko_times, atol=1e-9)
    np.testing.assert_allclose(s_pde.ko_probability, s_mc.ko_probability, atol=5e-3)
    np.testing.assert_allclose(s_pde.survival_probability, s_mc.survival_probability, atol=5e-3)


def test_phoenix_pde_module_has_no_mc_import():
    import quantark.asset.equity.engine.pde.phoenix_pde_solver as mod
    assert "MCEngine" not in pathlib.Path(mod.__file__).read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py::test_phoenix_pde_ko_survival_match_mc -x -q`
Expected: FAIL — inherited hook returns `None` for Phoenix (`s_pde is None`).

- [ ] **Step 3: Override the hooks in `PhoenixPDESolver`**

```python
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.engine.event_stats import PhoenixEventStats

class PhoenixPDESolver(SnowballPDESolver):
    def _event_stats_product_type(self):
        return PhoenixOption

    def _make_event_stats(self, **fields):
        return PhoenixEventStats(**fields)   # coupon arrays default empty; Task 9 fills them
```

The inherited `_compute_event_stats` already propagates KO/KI surfaces for any
product exposing `resolve_ko_observations`/barriers (Phoenix does). No MC.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py -x -q`
Expected: PASS (KO/survival within band; no-MC-import holds).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/phoenix_pde_solver.py \
        test/test_cashleg/test_phoenix_pde_event_stats.py
git commit -m "feat(pde): native Phoenix KO/KI event stats via reusable hooks"
```

### Task 9: Phoenix PDE — coupon indicator surfaces → coupon_probability

**Files:**
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py`
- Test: `test/test_cashleg/test_phoenix_pde_event_stats.py` (extend)

**Interfaces:**
- Consumes: `self._coupon_barriers`, `self._coupon_observation_indices` (populated by `PhoenixPDESolver._build_grids:290-344`); `_get_barrier_mask`; `_cashflow_value_at_time`.
- Produces: `PhoenixEventStats.coupon_probability[i] = P(coupon condition met at obs i AND alive entering i)`, matching MC; `expected_discounted_coupon_cashflow[i] = coupon_amounts[i] * ed_coupon_unit[i]`.

- [ ] **Step 1: Write failing coupon tests (incl. overlapping KO/coupon)**

```python
def test_phoenix_pde_coupon_prob_match_mc():
    env = make_env(); ph = make_phoenix()
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert s_pde.coupon_probability.shape == s_mc.coupon_probability.shape
    np.testing.assert_allclose(s_pde.coupon_probability, s_mc.coupon_probability, atol=5e-3)


def test_phoenix_pde_coupon_at_simultaneous_ko_matches_mc():
    # KO barrier below coupon barrier ⇒ KO region ⊂ coupon-pay region: every KO
    # observation is also a coupon trigger. Locks the "coupon counts at KO" rule.
    env = make_env()
    ph = make_phoenix(ko_barrier=90.0, coupon_barrier=(80.0, 80.0))
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    np.testing.assert_allclose(s_pde.coupon_probability, s_mc.coupon_probability, atol=6e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py -x -q -k coupon`
Expected: FAIL — `coupon_probability` empty (shape mismatch).

- [ ] **Step 3a: Widen the surface with coupon columns (propagate-only)**

In `PhoenixPDESolver`, override `_compute_event_stats` by copying the base body
(the surface width changes, so `super()._compute_event_stats` cannot be reused;
the base stays the Snowball reference) and widening the stacked surface from
`n_ko + 1` to `2*n_ko + 1` columns — layout `[KO_0..KO_{n-1}, COUP_0..COUP_{n-1},
KI]`. Propagate the new coupon columns through diffusion, the KO-region zeroing,
and the KI transition **exactly like the KO columns**, but do **not** set them
yet (they stay zero). Repoint the KI column index from `n_ko` to `2*n_ko`.

Checkpoint — KO/KI behaviour is unchanged because the coupon columns are zero:

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py -x -q -k "ko_survival or no_mc"`
Expected: PASS (Task 8's KO/survival/no-MC tests still hold).

- [ ] **Step 3b: Set coupon columns at observations + extract coupon stats**

At each coupon-observation time-index `j`
(`obs_idx = self._coupon_observation_indices.get(j)`), **after** the KO jump that
zeros `[mask_ko, :]` and sets the KO column, set the coupon column on the
coupon-pay mask (retaining a simultaneous-KO coupon — the verified convention):

```python
coupon_barrier = float(self._coupon_barriers[obs_idx])
pay_mask = self._get_barrier_mask(s_vec, coupon_barrier, product.is_reverse, is_up_barrier=True)
df_delay = self._cashflow_value_at_time(
    pricing_env=pricing_env, cashflow=1.0,
    current_time=float(t_vec[j]), settlement_time=rec.settlement_time,
)
coup_col = n_ko + obs_idx
v0_cur[pay_mask, coup_col] = df_delay
v1_cur[pay_mask, coup_col] = df_delay
```

Then extract and pass into `self._make_event_stats(...)`:

```python
ed_coup = np.array(
    [float(np.interp(spot_log, x_vec, initial_grid[:, n_ko + i])) for i in range(n_ko)],
    dtype=float,
)
coupon_probability = np.zeros(n_ko, dtype=float)
expected_discounted_coupon_cashflow = np.zeros(n_ko, dtype=float)
for i, rec in enumerate(ko_records):
    settle = float(rec.settlement_time if rec.settlement_time is not None else rec.observation_time)
    df0 = pricing_env.get_discount_factor(settle)
    if df0 > 0.0:
        coupon_probability[i] = float(ed_coup[i] / df0)
    expected_discounted_coupon_cashflow[i] = float(ed_coup[i] * float(self._coupon_amounts[i]))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_pde_event_stats.py -x -q`
Expected: PASS (KO, survival, coupon, and simultaneous-KO coupon within band).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/phoenix_pde_solver.py \
        test/test_cashleg/test_phoenix_pde_event_stats.py
git commit -m "feat(pde): native Phoenix coupon_probability via indicator surfaces"
```

## Phase C — Snowball QUAD refactor → native Phoenix QUAD event stats

### Task 10: Behavior-preserving Snowball QUAD event-stats refactor

**Files:**
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py`
- Test: `test/test_cashleg/test_phoenix_quad_event_stats.py` (Create)

**Interfaces:**
- Produces: the same `_event_stats_product_type()` / `_make_event_stats()` /
  `_compute_event_stats()` hooks as Task 7; Snowball behaviour unchanged.

- [ ] **Step 1: Characterization guard**

```python
# test/test_cashleg/test_phoenix_quad_event_stats.py
import pathlib
import numpy as np
from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from test.test_cashleg._autocallable_helpers import make_env, make_snowball, make_phoenix, make_engine


def test_snowball_quad_event_stats_unchanged_after_refactor():
    env = make_env(); sb = make_snowball()
    s = make_engine("quad", "snowball").calculate_event_stats(sb, env)
    assert s is not None and s.ko_probability.shape == s.ko_times.shape
```

Run (baseline): `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py::test_snowball_quad_event_stats_unchanged_after_refactor -x -q`
Expected: PASS.

- [ ] **Step 2: Apply the identical hook refactor**

In `snowball_quad_engine.py`, split `calculate_event_stats` into
`guard + _compute_event_stats`, add `_event_stats_product_type()` (default
`SnowballOption`) and `_make_event_stats(**fields)` (default
`AutocallableEventStats(**fields)`), and change the final
`return AutocallableEventStats(...)` (line ~691) to
`return self._make_event_stats(...)`.

- [ ] **Step 3: Run guard + Snowball QUAD regression**

Run:
```bash
.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py::test_snowball_quad_event_stats_unchanged_after_refactor -x -q
.venv/bin/python -m pytest test/ -k "snowball and quad" -q
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py \
        test/test_cashleg/test_phoenix_quad_event_stats.py
git commit -m "refactor(quad): extract reusable Snowball event-stats hooks (no behaviour change)"
```

### Task 11: Phoenix QUAD — native KO/KI event stats (remove MC delegation)

**Files:**
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py`
- Test: `test/test_cashleg/test_phoenix_quad_event_stats.py` (extend)

- [ ] **Step 1: Write failing KO/survival + no-MC tests**

```python
def test_phoenix_quad_ko_survival_match_mc():
    env = make_env(); ph = make_phoenix()
    s_q = make_engine("quad", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert isinstance(s_q, PhoenixEventStats)
    np.testing.assert_allclose(s_q.ko_probability, s_mc.ko_probability, atol=5e-3)
    np.testing.assert_allclose(s_q.survival_probability, s_mc.survival_probability, atol=5e-3)


def test_phoenix_quad_module_has_no_mc_import():
    import quantark.asset.equity.engine.quad.phoenix_quad_engine as mod
    assert "MCEngine" not in pathlib.Path(mod.__file__).read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py::test_phoenix_quad_module_has_no_mc_import -x -q`
Expected: FAIL — the module still imports `PhoenixMCEngine` (`phoenix_quad_engine.py:486-495`).

- [ ] **Step 3: Delete the MC delegation; override hooks**

Remove the `calculate_event_stats` body that builds `PhoenixMCEngine(MCParams())`
(and the now-unused `MCParams`/`PhoenixMCEngine` imports). Override:

```python
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.engine.event_stats import PhoenixEventStats

class PhoenixQuadEngine(SnowballQuadEngine):
    def _event_stats_product_type(self):
        return PhoenixOption

    def _make_event_stats(self, **fields):
        return PhoenixEventStats(**fields)
```

The inherited Snowball QUAD `_compute_event_stats` (KO indicator recursion + KI
sub-recursion) runs natively for Phoenix. No MC.

- [ ] **Step 4: Run to verify pass**

Run:
```bash
.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py -x -q
.venv/bin/python -m pytest test/ -k "phoenix and quad" -q
```
Expected: PASS (KO/survival within band; no-MC-import holds; existing Phoenix QUAD pricing tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/phoenix_quad_engine.py \
        test/test_cashleg/test_phoenix_quad_event_stats.py
git commit -m "feat(quad): native Phoenix KO/KI event stats; remove MC delegation"
```

### Task 12: Phoenix QUAD — coupon indicator surfaces → coupon_probability

**Files:**
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py`
- Test: `test/test_cashleg/test_phoenix_quad_event_stats.py` (extend)

- [ ] **Step 1: Write failing coupon tests (incl. overlapping KO/coupon)**

```python
def test_phoenix_quad_coupon_prob_match_mc():
    env = make_env(); ph = make_phoenix()
    s_q = make_engine("quad", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert s_q.coupon_probability.shape == s_mc.coupon_probability.shape
    np.testing.assert_allclose(s_q.coupon_probability, s_mc.coupon_probability, atol=5e-3)


def test_phoenix_quad_coupon_at_simultaneous_ko_matches_mc():
    env = make_env()
    ph = make_phoenix(ko_barrier=90.0, coupon_barrier=(80.0, 80.0))
    s_q = make_engine("quad", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    np.testing.assert_allclose(s_q.coupon_probability, s_mc.coupon_probability, atol=6e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py -x -q -k coupon`
Expected: FAIL — coupon arrays empty.

- [ ] **Step 3a: Add coupon-trigger rows (propagate-only)**

In `PhoenixQuadEngine`, override `_compute_event_stats` by copying the inherited
Snowball QUAD body and carrying `n_ko` extra **coupon-trigger** indicator rows
alongside the KO rows (e.g. a second `coup_surface` block of shape `(n_ko, grid)`).
Diffuse them with the same `_diffuse_fft`/`_diffuse_with_bridge` calls used for
the KO rows, and apply the KO-region zeroing/KI transition to them identically —
but do **not** set them yet (all zero).

Checkpoint — KO/KI unchanged:

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py -x -q -k "ko_survival or no_mc"`
Expected: PASS (Task 11's KO/survival/no-MC tests still hold).

- [ ] **Step 3b: Set coupon rows at observations + extract coupon stats**

At each coupon observation set the coupon row on the coupon-pay weight
(`_smooth_step_weight(..., trigger_is_down=product.is_reverse)`, hard-mask
fallback) **after** the KO update, so simultaneous-KO coupons are retained. Then
extract and pass both arrays into `self._make_event_stats(...)`:

```python
ed_coup = np.array(
    [math_utils.interpolate(coup_surface[i], x=0.0) for i in range(n_ko)],
    dtype=float,
)
coupon_probability = np.zeros(n_ko, dtype=float)
expected_discounted_coupon_cashflow = np.zeros(n_ko, dtype=float)
for i, rec in enumerate(ko_records):
    df_total = math.exp(-rate * float(rec.observation_time)) * float(
        self._ko_discount(rate, float(rec.observation_time), rec.settlement_time)
    )
    if df_total > 0:
        coupon_probability[i] = float(ed_coup[i] / df_total)
    expected_discounted_coupon_cashflow[i] = float(ed_coup[i] * float(coupon_amounts[i]))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_phoenix_quad_event_stats.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/phoenix_quad_engine.py \
        test/test_cashleg/test_phoenix_quad_event_stats.py
git commit -m "feat(quad): native Phoenix coupon_probability via indicator surfaces"
```

## Phase D — Integration & acceptance

### Task 13: EventDistribution path + COUPON-basis position valuation

**Files:**
- Test: `test/test_cashleg/test_autocallable_leg_distribution.py` (Create)
- (No source change.) The COUPON mapper **already exists**:
  `EventDistribution.from_autocallable_stats` (`event_distribution.py:114-117`)
  does `if isinstance(stats, PhoenixEventStats) and stats.coupon_probability.size > 0:
  probabilities[EventType.COUPON] = np.asarray(stats.coupon_probability, ...)`. So
  once Phases B/C make the engines return `PhoenixEventStats` with a populated
  `coupon_probability`, `price_with_events` emits `EventType.COUPON` with no
  adapter change. This task verifies that end-to-end.

- [ ] **Step 0: Confirm the existing mapper (reconnaissance)**

Run: `grep -n "EventType.COUPON\|coupon_probability" quantark/cashleg/event_distribution.py`
Expected: shows the `from_autocallable_stats` branch above. If it is absent or
gated differently, add a concrete source step here to populate `EventType.COUPON`
before writing the tests.

- [ ] **Step 1: Write the tests**

```python
# test/test_cashleg/test_autocallable_leg_distribution.py
from datetime import datetime
import numpy as np
import pytest

from quantark.portfolio import EquityPosition
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg, AutocallableLegType, AccrualBasis,
)
from test.test_cashleg._autocallable_helpers import (
    make_env, make_phoenix, make_engine, future_event_times,
)


@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_phoenix_price_with_events_emits_coupon_stream(kind):
    env = make_env(); ph = make_phoenix()
    dist = make_engine(kind, "phoenix").price_with_events(
        ph, env, emit_distribution=True
    ).event_distribution
    assert EventType.COUPON in dist.probabilities
    assert np.asarray(dist.probabilities[EventType.COUPON]).size == dist.event_times.size


@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_coupon_basis_leg_prices_in_position(kind):
    env = make_env(); ph = make_phoenix()
    engine = make_engine(kind, "phoenix")
    obs = future_event_times(ph, engine, env)
    leg = AutocallableCashLeg(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.BACKEND_INTEREST,
        notional=1_000_000.0, rate=0.03,
        observation_schedule=tuple(obs),
        accrual_factors=tuple(np.full(obs.size, 0.5)),
        settlement_schedule=tuple(obs),
        terminal_accrual_factor=0.0, terminal_settlement_time=float(obs[-1]),
        accrual_basis=AccrualBasis.COUPON,
    )
    pos = EquityPosition(product=ph, quantity=1.0, entry_price=0.0, underlying="UND",
                         engine=engine, entry_timestamp=datetime(2024, 1, 1),
                         cash_legs=[leg])
    pv = pos.get_trade_value(env)
    assert np.isfinite(pv) and pv != 0.0
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg_distribution.py -q`
Expected: PASS for mc/pde/quad (depends on Phases B/C).

- [ ] **Step 3: Commit**

```bash
git add test/test_cashleg/test_autocallable_leg_distribution.py
git commit -m "test(cashleg): EventDistribution COUPON mapping + COUPON-basis position"
```

### Task 14: Position PV + Greeks + quantity contract

**Files:**
- Test: `test/test_cashleg/test_autocallable_leg_position.py` (Create)
- (No library change — `EquityPosition` already handles legs and Greeks.)

- [ ] **Step 1: Write the tests**

```python
# test/test_cashleg/test_autocallable_leg_position.py
from datetime import datetime
import numpy as np
import pytest

from quantark.portfolio import EquityPosition
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.cashleg.event_distribution import EventDistribution
from quantark.util.exceptions import ValidationError
from test.test_cashleg._autocallable_helpers import (
    make_env, make_snowball, make_phoenix, make_engine, future_event_times, make_margin_leg,
)


def _pos(product, engine, legs, quantity=1.0):
    return EquityPosition(product=product, quantity=quantity, entry_price=0.0,
                          underlying="UND", engine=engine,
                          entry_timestamp=datetime(2024, 1, 1), cash_legs=legs)


@pytest.mark.parametrize("asset", ["snowball", "phoenix"])
@pytest.mark.parametrize("kind", ["mc", "pde", "quad"])
def test_margin_leg_pv_and_greeks_finite(asset, kind):
    env = make_env()
    product = make_snowball() if asset == "snowball" else make_phoenix()
    engine = make_engine(kind, asset)
    leg = make_margin_leg(future_event_times(product, engine, env))
    pos = _pos(product, engine, [leg])
    greeks = pos.get_trade_greeks(env, GreeksCalculator())
    assert np.isfinite(pos.get_trade_value(env))
    assert np.isfinite(greeks["delta"]) and abs(greeks["delta"]) > 0.0
    assert np.isfinite(greeks["gamma"])


def test_quantity_scales_trade_value_linearly():
    env = make_env(); product = make_snowball(); engine = make_engine("pde", "snowball")
    obs = future_event_times(product, engine, env)
    v1 = _pos(product, engine, [make_margin_leg(obs)], quantity=1.0).get_trade_value(env)
    v3 = _pos(product, engine, [make_margin_leg(obs)], quantity=3.0).get_trade_value(env)
    assert abs(v3 - 3.0 * v1) <= 1e-6 * max(1.0, abs(v1))


def test_fail_loud_when_engine_emits_no_ko_stream():
    env = make_env(); product = make_snowball(); engine = make_engine("pde", "snowball")
    leg = make_margin_leg(future_event_times(product, engine, env))
    with pytest.raises(ValidationError):
        leg.value(EventDistribution.trivial(float(leg.terminal_settlement_time)), env, 0.0)
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg_position.py -q`
Expected: PASS. If `delta == 0`, confirm the leg's `observation_schedule` equals
`future_event_times(...)` exactly (else the schedule-identity guard raises).

- [ ] **Step 3: Commit**

```bash
git add test/test_cashleg/test_autocallable_leg_position.py
git commit -m "test(cashleg): position PV/Greeks across engines + quantity contract"
```

### Task 15: Required golden-case parity (PV + per-leg delta/gamma)

**Files:**
- Create: `test/test_cashleg/fixtures/autocallable_golden_case.json`
- Test: `test/test_cashleg/test_autocallable_leg_golden.py`

**Interfaces:**
- Uses `EquityPosition.get_trade_value_breakdown(env) -> TradeValueBreakdown`
  (exists, `position.py:116-148`; `leg_pvs` keyed by `leg.leg_id`,
  quantity-scaled) for per-leg PV, and central-difference bumps for per-leg
  delta/gamma.

- [ ] **Step 1: Create the fixture (complete schema)**

`test/test_cashleg/fixtures/autocallable_golden_case.json` — neutral keys, no
external identifier:

```json
{
  "asset": "snowball",
  "env": {"spot": 100.0, "vol": 0.22, "rate": 0.02, "div_yield": 0.0},
  "product": {"ko_dates": [0.5, 1.0], "ko_barrier": 100.0, "ko_rate": 0.10,
              "ki_barrier": 80.0, "maturity": 1.0},
  "legs": {
    "pv_margin":   {"leg_type": "MARGIN", "pv_formula": "NOTIONAL_MINUS_PAYOFF",
                    "direction": "BUYER_RECEIVES", "notional": 20004513.86,
                    "rate": 0.4965986394557823, "accrual_factors": [0.5, 1.0],
                    "terminal_accrual_factor": 2.0136986301369864},
    "pv_interest": {"leg_type": "BACKEND_INTEREST", "pv_formula": "NORMAL",
                    "direction": "BUYER_PAYS", "notional": 20004513.86,
                    "rate": 0.001, "accrual_factors": [0.5, 1.0],
                    "terminal_accrual_factor": 1.0},
    "pv_rebate":   {"leg_type": "REBATE", "pv_formula": "NORMAL",
                    "direction": "BUYER_PAYS", "notional": 20004513.86,
                    "rate": 0.02, "accrual_factors": [0.5, 1.0],
                    "terminal_accrual_factor": 1.0}
  },
  "tolerances": {"pv_rel": 5e-3, "delta_rel": 1e-2, "gamma_rel": 5e-2,
                 "mc_pv_rel": 2e-2, "mc_delta_rel": 5e-2}
}
```

The frozen `targets` block (per-leg `{pv, delta, gamma}`) is produced once by
running the native legs under PDE (Step 3) and committed alongside the fixture —
a regression lock + cross-engine parity check. If/when the literal vendor confirm
is authorized, replace `product`/`legs` with its sanitized parameters and freeze
the vendor targets; the literal `207,475.74 / −3,417.10 / −409,798.69` parity
test then lives in the adapter repo where the synthetic-`SnowballOption`
workaround exists (spec §7 input dependency).

- [ ] **Step 2: Write the parity test (`_build_case(engine_kind)` concrete; PV + delta + gamma)**

```python
# test/test_cashleg/test_autocallable_leg_golden.py
import json, pathlib
from datetime import datetime
from copy import deepcopy
import numpy as np
import pytest

from quantark.portfolio import EquityPosition
from quantark.cashleg.base import LegDirection
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg, AutocallableLegType, PvFormula,
)
from test.test_cashleg._autocallable_helpers import (
    make_env, make_snowball, make_engine, future_event_times,
)

FIX = pathlib.Path(__file__).parent / "fixtures" / "autocallable_golden_case.json"


def _build_case(engine_kind):
    data = json.loads(FIX.read_text())
    env = make_env(**data["env"])
    product = make_snowball(**data["product"])
    engine = make_engine(engine_kind, data["asset"])
    obs = future_event_times(product, engine, env)
    legs = {}
    for name, spec in data["legs"].items():
        legs[name] = AutocallableCashLeg(
            direction=LegDirection[spec["direction"]],
            leg_type=AutocallableLegType[spec["leg_type"]],
            pv_formula=PvFormula[spec["pv_formula"]],
            notional=spec["notional"], rate=spec["rate"],
            observation_schedule=tuple(obs),
            accrual_factors=tuple(spec["accrual_factors"]),
            settlement_schedule=tuple(obs),
            terminal_accrual_factor=spec["terminal_accrual_factor"],
            terminal_settlement_time=float(obs[-1]),
        )
    pos = EquityPosition(product=product, quantity=1.0, entry_price=0.0,
                         underlying="UND", engine=engine,
                         entry_timestamp=datetime(2024, 1, 1),
                         cash_legs=list(legs.values()))
    return env, pos, legs, data


def _leg_pv(pos, env, leg_id):
    return pos.get_trade_value_breakdown(env).leg_pvs[leg_id].pv


def _leg_delta_gamma(pos, env, leg_id, h=1e-3):
    s = env.spot_quote.spot
    up, dn = deepcopy(env), deepcopy(env)
    up.spot_quote.spot = s * (1 + h); dn.spot_quote.spot = s * (1 - h)
    p0 = _leg_pv(pos, env, leg_id)
    pu = _leg_pv(pos, up, leg_id); pd = _leg_pv(pos, dn, leg_id)
    return (pu - pd) / (2 * s * h), (pu - 2 * p0 + pd) / (s * h) ** 2


@pytest.mark.parametrize("engine_kind", ["pde", "quad", "mc"])
def test_golden_case_pv_and_greeks(engine_kind):
    env, pos, legs, data = _build_case(engine_kind)
    tol = data["tolerances"]
    pv_rel = tol["mc_pv_rel"] if engine_kind == "mc" else tol["pv_rel"]
    d_rel = tol["mc_delta_rel"] if engine_kind == "mc" else tol["delta_rel"]
    targets = data["targets"]      # frozen native-PDE values (Step 3)
    for name, leg in legs.items():
        pv = _leg_pv(pos, env, leg.leg_id)
        delta, gamma = _leg_delta_gamma(pos, env, leg.leg_id)
        t = targets[name]
        assert abs(pv - t["pv"]) <= pv_rel * max(1.0, abs(t["pv"])), (name, "pv", pv)
        assert abs(delta - t["delta"]) <= d_rel * max(1.0, abs(t["delta"])), (name, "delta", delta)
        if engine_kind != "mc":
            assert abs(gamma - t["gamma"]) <= tol["gamma_rel"] * max(1.0, abs(t["gamma"]))
```

- [ ] **Step 3: Write the one-shot target generator**

Create `test/test_cashleg/fixtures/_generate_golden_targets.py`. It imports the
case builder by file path (so it does not depend on the test-package import
mode), computes per-leg PV/delta/gamma from the **PDE** engine, and writes the
`"targets"` block back into the JSON:

```python
# test/test_cashleg/fixtures/_generate_golden_targets.py
"""One-shot: freeze native-PDE golden targets into autocallable_golden_case.json.
Re-run only to intentionally re-baseline. Run: .venv/bin/python <this file>."""
import importlib.util, json, pathlib

_here = pathlib.Path(__file__).resolve()
_test_file = _here.parents[1] / "test_autocallable_leg_golden.py"
_spec = importlib.util.spec_from_file_location("golden_mod", _test_file)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


def main():
    env, pos, legs, data = golden._build_case("pde")
    data["targets"] = {}
    for name, leg in legs.items():
        pv = golden._leg_pv(pos, env, leg.leg_id)
        delta, gamma = golden._leg_delta_gamma(pos, env, leg.leg_id)
        data["targets"][name] = {"pv": pv, "delta": delta, "gamma": gamma}
    golden.FIX.write_text(json.dumps(data, indent=2))
    print(json.dumps(data["targets"], indent=2))


if __name__ == "__main__":
    main()
```

(Loading the test file by path also pulls in `test_autocallable_leg_golden`'s own
imports of `quantark.*` and `_autocallable_helpers`; if `_autocallable_helpers`
cannot be imported in script context, mirror the repo's intra-test import
convention recorded in Task 6.)

- [ ] **Step 4: Generate, then lock**

Run the generator (writes `targets`), then the parity test:

```bash
.venv/bin/python test/test_cashleg/fixtures/_generate_golden_targets.py
.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg_golden.py -q
```
Expected: the generator prints a `targets` block and writes it into the fixture;
the parity test then PASSES for pde/quad (tight) and mc (wider band). PDE↔QUAD
agreement is the cross-engine correctness signal; MC-within-band confirms it.

- [ ] **Step 5: Commit**

```bash
git add test/test_cashleg/fixtures/autocallable_golden_case.json \
        test/test_cashleg/fixtures/_generate_golden_targets.py \
        test/test_cashleg/test_autocallable_leg_golden.py
git commit -m "test(cashleg): required golden-case PV+delta+gamma parity across engines"
```

### Task 16: Full-suite green + docs

**Files:**
- Modify: `quantark/cashleg/CLAUDE.md`

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Fix any regression before committing.

- [ ] **Step 2: Update `cashleg/CLAUDE.md`**

Add an `AutocallableCashLeg` row to the leg-types table (KO/coupon-contingent
return leg; margin via `NOTIONAL_MINUS_PAYOFF`) and reconcile the "Module
Structure" list with the actual files (add `autocallable_leg.py`; remove modules
that do not exist).

- [ ] **Step 3: Commit**

```bash
git add quantark/cashleg/CLAUDE.md
git commit -m "docs(cashleg): document AutocallableCashLeg; fix module list"
```

---

## Self-Review

**Task map (16 tasks, 4 phases):**

| Phase | Tasks | Deliverable |
|-------|-------|-------------|
| A — leg core | 0–5 | dataclass, valuation, guards, COUPON basis, standalone, exports |
| (bridge) | 6 | shared real test builders (`_autocallable_helpers.py`) |
| B — Phoenix PDE | 7–9 | Snowball refactor → Phoenix KO/KI → coupon surfaces |
| C — Phoenix QUAD | 10–12 | Snowball refactor → Phoenix KO/KI (drop MC) → coupon surfaces |
| D — integration | 13–16 | EventDistribution path + COUPON position, position Greeks + quantity, golden parity, full-suite + docs |

**Spec coverage:** §3 architecture/standalone → Tasks 1,3,4; §3a fail-loud →
Tasks 3,14; §3b Phoenix PDE → 7–9, Phoenix QUAD → 10–12; §4 dataclass → 1; §5
valuation/alignment/`NOTIONAL_MINUS_PAYOFF` → 1,3,4; §6 Greeks/exports → 5,14;
§7 per-type/invariants/regression → 3,4,14, Phoenix-vs-MC → 8,9,11,12, required
golden → 15; §8/§9 → 7–15.

**Review findings (Stage-4) resolution:**
- P1 fixture imports → Task 6 creates the real `_autocallable_helpers.py`; every test imports it (no phantom modules).
- P1 golden placeholder → Task 15 has a complete JSON schema, concrete `_build_case(engine_kind)`, per-leg PV+delta+gamma via existing `get_trade_value_breakdown` + bumps, frozen targets; vendor-literal parity documented as the adapter-side test (spec §7 input dependency).
- P2 Snowball files / oversized refactors → split into dedicated refactor tasks (7, 10) with Snowball regression, then Phoenix enablement (8, 11), then coupons (9, 12); each task's Files list names the Snowball file it edits.
- P2 simultaneous coupon/KO → convention verified from `phoenix_mc_engine.py:744-757` and stated in the Phase B note; overlapping-barrier tests in Tasks 9, 12.
- P2 EventDistribution path → Task 13 tests `price_with_events` COUPON mapping + a COUPON-basis position across engines.
- P2 no-MC-inside enforcement → `test_*_module_has_no_mc_import` static assertions (Tasks 8, 11) plus removal of the MC import (Task 11).
- P3 schedule validation → Task 1 validates finite/non-negative schedule values (+ NaN/negative tests).
- P3 quantity/notional contract → Task 14 `test_quantity_scales_trade_value_linearly` + Task 3 `test_value_ignores_position_notional_argument`.

**Placeholder scan:** the only intentional run-time fill is Task 15's `targets`
block (frozen from a Step-3 PDE run — a normal golden-lock, not a logic gap) and
the documented external-data dependency for vendor-literal numbers. No `...` in
logic paths.

**Type consistency:** the `_event_stats_product_type()` / `_make_event_stats(**fields)`
/ `_compute_event_stats(product, env)` hook trio is defined once per engine family
(Tasks 7, 10) and reused identically (8, 9, 11, 12); `make_*` builder signatures
match every call site; `EquityPosition(...)` is always constructed with the full
field set (`product, quantity, entry_price, underlying, engine, entry_timestamp,
cash_legs`) confirmed from `position.py`; `get_trade_value_breakdown` keys on
`leg.leg_id` as the source defines.
