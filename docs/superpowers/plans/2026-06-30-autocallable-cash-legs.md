# Autocallable-Driven Cash Legs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native `AutocallableCashLeg` that prices Snowball/Phoenix-driven cash legs (margin/prepayment, backend premium, backend interest, rebate, minimum return) from the parent autocallable's `EventDistribution`, and give Phoenix PDE/QUAD the native event stats this requires.

**Architecture:** A new `CashLeg` subclass consumes the parent's per-observation KO (or coupon) probabilities + terminal buckets and applies a per-observation accrual schedule with explicit workbook-sourced settlement times; Greeks flow free through the existing `EquityPosition.get_trade_greeks` bump loop. To make every Snowball/Phoenix × MC/PDE/QUAD combination method-consistent, native `calculate_event_stats` is added to `PhoenixPDESolver` and `PhoenixQuadEngine` (mirroring the Snowball implementations + coupon indicator surfaces; no MC).

**Tech Stack:** Python 3.11+, NumPy, dataclasses; pytest (pytest-xdist). Library code under `quantark.*`.

**Spec:** `docs/superpowers/specs/2026-06-30-autocallable-cash-legs-design.md`

## Global Constraints

- Canonical imports only: `quantark.*` (never the legacy flat names).
- Numerical ops via `quantark.util.numerical` — no hardcoded tolerances, no raw float `==`. Use `Tolerance`, `almost_equal`, `is_close`, `safe_*`, `validate_*`.
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
| `quantark/asset/equity/engine/pde/phoenix_pde_solver.py` | Native `calculate_event_stats` (KO/KI + coupon surfaces) | Modify |
| `quantark/asset/equity/engine/quad/phoenix_quad_engine.py` | Native `calculate_event_stats` (replace MC delegation) | Modify |
| `test/test_cashleg/test_autocallable_leg.py` | Unit tests vs independent re-implementation, synthetic `EventDistribution` | Create |
| `test/test_cashleg/test_autocallable_leg_position.py` | Position PV + Greeks integration | Create |
| `test/test_cashleg/test_autocallable_leg_golden.py` | Required golden-case parity (PV + Greeks) across engines | Create |
| `test/test_cashleg/fixtures/autocallable_golden_case.json` | Sanitized golden inputs + frozen targets | Create |
| `test/test_equity/test_phoenix_pde_event_stats.py` | Phoenix PDE event stats vs Phoenix MC | Create |
| `test/test_equity/test_phoenix_quad_event_stats.py` | Phoenix QUAD event stats vs Phoenix MC; assert no MC engine constructed | Create |

(Exact engine test directory confirmed in Task 0.)

---

## Task 0: Reconnaissance & test-harness baseline

**Files:**
- Read only.

- [ ] **Step 1: Confirm cashleg exports and test layout**

Run:
```bash
cd /Users/fuxinyao/quant-ark
sed -n '1,40p' quantark/cashleg/__init__.py
ls test/test_cashleg/ 2>/dev/null; ls test/ | grep -i equity
.venv/bin/python -c "from quantark.util.numerical import Tolerance; print([a for a in dir(Tolerance) if not a.startswith('_')])"
```
Expected: see the current `__all__` (DeterministicLeg/AccrualLeg/FixedPayoffLeg + enums), the `test/test_cashleg/` dir, the equity test dir name, and the available `Tolerance` attributes. Note the real names — later tasks reference `Tolerance.PROBABILITY` for the schedule-identity tolerance; if a more specific time tolerance exists, prefer it.

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
from quantark.util.numerical import Tolerance, almost_equal


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
        if not np.isfinite(float(self.notional)):
            raise ValidationError(f"notional must be finite, got {self.notional}")
        if not np.isfinite(float(self.rate)):
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
Expected: PASS (5 passed).

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

        p_term = sum(
            float(event_dist.probabilities.get(e, 0.0)) for e in self.terminal_events
        )
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

## Phase B — Native Phoenix PDE event stats

**Design note (applies to Tasks 6–9):** The leg consumes only `ko_probability`,
`survival_probability`, `coupon_probability`, and the terminal buckets. All four
are **indicator expectations** and are **independent of memory-coupon
accumulation** (memory changes coupon *amounts*, not the probability a coupon
condition is met while alive). So the native event stats are built by
propagating **stacked KO + coupon-trigger indicator surfaces**, mirroring the
Snowball template — **not** the memory vector-state pricer. `PhoenixMCEngine.
calculate_event_stats` (`phoenix_mc_engine.py:163-317`) is the reference: the
PDE/QUAD ports must reproduce its `coupon_probability` semantics. The native
code must construct **no** `*MCEngine` (enforced by a test).

### Task 6: Phoenix PDE — native KO/KI event stats (no coupon yet)

**Files:**
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py`
- Test: `test/test_equity/test_phoenix_pde_event_stats.py` (Create)

**Interfaces:**
- Consumes: the Snowball template `SnowballPDESolver.calculate_event_stats` (`snowball_pde_solver.py:323-612`) and its helpers (`_filter_observations_by_tau`, `_build_grids`, `_get_barrier_mask`, `_cashflow_value_at_time`, `_resolve_ki_barrier_at_tidx`).
- Produces: `PhoenixPDESolver.calculate_event_stats(product, pricing_env) -> Optional[PhoenixEventStats]` returning correct `ko_times`/`ko_probability`/`survival_probability`/`ki_probability`/`expected_discounted_*` for a Phoenix product (coupon fields empty for now).

- [ ] **Step 1: Write the failing MC cross-check test (KO/survival only)**

```python
# test/test_equity/test_phoenix_pde_event_stats.py
import numpy as np
import pytest

from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.param.engine_params import PDEParams, MCParams
from quantark.asset.equity.engine.event_stats import PhoenixEventStats
# NOTE (Task 0 follow-up): import the shared Phoenix test-product builder used by
# the existing Phoenix engine tests. Confirm its location with:
#   grep -rl "PhoenixOption(" test/ | head
# and reuse that fixture/helper here as `make_phoenix_product(pricing_env)`.
from test.test_equity._phoenix_fixtures import make_phoenix_product, make_env  # adjust import to the real helper


@pytest.fixture
def phoenix_case():
    env = make_env()
    product = make_phoenix_product(env)   # non-memory Phoenix, discrete KO/coupon
    return product, env


def test_phoenix_pde_ko_survival_match_mc(phoenix_case):
    product, env = phoenix_case
    pde = PhoenixPDESolver(params=PDEParams(grid_size=600, time_steps=600))
    mc = PhoenixMCEngine(params=MCParams(num_paths=200_000, time_steps=252, seed=7))

    s_pde = pde.calculate_event_stats(product, env)
    s_mc = mc.calculate_event_stats(product, env)

    assert isinstance(s_pde, PhoenixEventStats)
    np.testing.assert_allclose(s_pde.ko_times, s_mc.ko_times, atol=1e-9)
    # 3 sigma MC band; KO/survival are the tight ones
    np.testing.assert_allclose(s_pde.ko_probability, s_mc.ko_probability, atol=5e-3)
    np.testing.assert_allclose(
        s_pde.survival_probability, s_mc.survival_probability, atol=5e-3
    )
    assert abs(s_pde.ki_probability - s_mc.ki_probability) < 5e-3


def test_phoenix_pde_builds_no_mc_engine(phoenix_case, monkeypatch):
    product, env = phoenix_case
    import quantark.asset.equity.engine.mc.phoenix_mc_engine as mcmod

    def _boom(*a, **k):
        raise AssertionError("PhoenixPDESolver must not construct an MC engine")

    monkeypatch.setattr(mcmod, "PhoenixMCEngine", _boom)
    PhoenixPDESolver(params=PDEParams(grid_size=400, time_steps=400)).calculate_event_stats(
        product, env
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_pde_event_stats.py::test_phoenix_pde_ko_survival_match_mc -x -q`
Expected: FAIL — the inherited method returns `None` (Phoenix is not a `SnowballOption`), so `s_pde` is `None` and `isinstance(... PhoenixEventStats)` fails.

- [ ] **Step 3: Override `calculate_event_stats` in `PhoenixPDESolver` (KO/KI port)**

Add a `calculate_event_stats` override that reuses the Snowball algorithm but
keys on `PhoenixOption` and returns a `PhoenixEventStats`. The simplest faithful
port: call the Snowball KO/KI machinery by temporarily relaxing the product-type
guard. Concretely, refactor the Snowball method so the body is reusable, then
call it from Phoenix. In `snowball_pde_solver.py`, extract the body of
`calculate_event_stats` after the `isinstance` guard into a helper:

```python
# snowball_pde_solver.py — replace the guarded body with a guard + delegation
def calculate_event_stats(self, product, pricing_env):
    if not isinstance(product, self._event_stats_product_type()):
        return None
    if pricing_env is None:
        return None
    return self._compute_event_stats(product, pricing_env)

def _event_stats_product_type(self):
    """Product type accepted by calculate_event_stats (overridable)."""
    return SnowballOption

def _compute_event_stats(self, product, pricing_env):
    # ... existing body (lines ~330-612), unchanged, but returning via
    #     self._finalize_event_stats(...) so subclasses can swap the dataclass ...
```

Then in `phoenix_pde_solver.py`:

```python
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption

class PhoenixPDESolver(SnowballPDESolver):
    def _event_stats_product_type(self):
        return PhoenixOption
```

For Task 6, return a `PhoenixEventStats` with empty coupon arrays. The cleanest
seam: have `_compute_event_stats` build the field dict and call a
`_make_event_stats(**fields)` factory that the base implements as
`AutocallableEventStats(**fields)` and Phoenix overrides as
`PhoenixEventStats(**fields)`:

```python
# snowball_pde_solver.py
def _make_event_stats(self, **fields):
    return AutocallableEventStats(**fields)

# phoenix_pde_solver.py
def _make_event_stats(self, **fields):
    return PhoenixEventStats(**fields)
```

Replace the literal `return AutocallableEventStats(...)` at the end of
`_compute_event_stats` with `return self._make_event_stats(...)` using the same
keyword fields (verbatim list: `pv, ko_times, ko_probability,
survival_probability, expected_discounted_ko_cashflow, ki_probability,
expected_discounted_maturity_cashflow, reconciliation_error, ki_times,
ki_event_probability, ki_survival_probability`).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_pde_event_stats.py -x -q`
Expected: PASS (KO/survival/ki within band; no-MC guard passes). Also run the
Snowball PDE event-stats regression to prove the refactor is behaviour-preserving:
`.venv/bin/python -m pytest test/ -k "snowball and event_stats and pde" -q`
Expected: PASS (unchanged).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/snowball_pde_solver.py \
        quantark/asset/equity/engine/pde/phoenix_pde_solver.py \
        test/test_equity/test_phoenix_pde_event_stats.py
git commit -m "feat(pde): native Phoenix KO/KI event stats via reusable Snowball core"
```

### Task 7: Phoenix PDE — coupon-trigger indicator surfaces → coupon_probability

**Files:**
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py`
- Test: `test/test_equity/test_phoenix_pde_event_stats.py` (extend)

**Interfaces:**
- Consumes: `self._coupon_barriers`, `self._coupon_observation_indices` (already populated by `PhoenixPDESolver._build_grids`, lines 290-344); `_get_barrier_mask`; `_cashflow_value_at_time`.
- Produces: `PhoenixEventStats.coupon_probability[i] = P(coupon condition met at obs i AND alive)`, matching `PhoenixMCEngine` semantics; `expected_discounted_coupon_cashflow[i] = coupon_amount[i] * ed_coupon_unit[i]` for the non-memory case.

- [ ] **Step 1: Write the failing coupon cross-check test**

```python
def test_phoenix_pde_coupon_prob_matches_mc(phoenix_case):
    product, env = phoenix_case
    pde = PhoenixPDESolver(params=PDEParams(grid_size=600, time_steps=600))
    mc = PhoenixMCEngine(params=MCParams(num_paths=200_000, time_steps=252, seed=7))
    s_pde = pde.calculate_event_stats(product, env)
    s_mc = mc.calculate_event_stats(product, env)
    assert s_pde.coupon_probability.shape == s_mc.coupon_probability.shape
    np.testing.assert_allclose(
        s_pde.coupon_probability, s_mc.coupon_probability, atol=5e-3
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_pde_event_stats.py::test_phoenix_pde_coupon_prob_matches_mc -x -q`
Expected: FAIL — `coupon_probability` is empty (shape mismatch).

- [ ] **Step 3: Add coupon indicator columns to `_compute_event_stats`**

In `PhoenixPDESolver`, override `_compute_event_stats` to extend the stacked
surface with `n_ko` extra **coupon-trigger** columns alongside the KO columns
(layout `[KO_0..KO_{n-1}, COUP_0..COUP_{n-1}, KI]`). Mirror the KO jump exactly,
but using the **coupon barrier** and **without terminating** the other surfaces
(a coupon does not knock the note out):

```python
# At each observation time-index j that is a coupon observation
# (obs_idx = self._coupon_observation_indices.get(j)):
coupon_barrier = float(self._coupon_barriers[obs_idx])
pay_mask = self._get_barrier_mask(
    s_vec, coupon_barrier, product.is_reverse, is_up_barrier=True
)
df_delay = self._cashflow_value_at_time(
    pricing_env=pricing_env, cashflow=1.0,
    current_time=float(t_vec[j]), settlement_time=rec.settlement_time,
)
# coupon indicator for this obs: set on pay_mask (alive paths only — KO jump
# below zeros the KO region across ALL columns, so coupon columns are absorbed
# by KO automatically, matching "coupon paid only if not KO'd")
coup_col = n_ko + obs_idx
v0_cur[pay_mask, coup_col] = df_delay
v1_cur[pay_mask, coup_col] = df_delay
```

Order the jumps so the KO jump (which zeros `[mask_ko, :]`) runs **after** the
coupon set if the product pays the coupon at KO, or **before** if it does not —
choose the order that reproduces `PhoenixMCEngine.coupon_probability` (inspect
`phoenix_mc_engine.py:163-317` to confirm whether a coupon at a simultaneous KO
counts). Extract coupon probabilities exactly like KO:

```python
ed_coup = np.array(
    [float(np.interp(spot_log, x_vec, initial_grid[:, n_ko + i])) for i in range(n_ko)],
    dtype=float,
)
coupon_probability = np.zeros(n_ko, dtype=float)
expected_discounted_coupon_cashflow = np.zeros(n_ko, dtype=float)
for i, rec in enumerate(ko_records):
    settle = float(rec.settlement_time if rec.settlement_time is not None
                   else rec.observation_time)
    df0 = pricing_env.get_discount_factor(settle)
    if df0 > 0.0:
        coupon_probability[i] = float(ed_coup[i] / df0)
    expected_discounted_coupon_cashflow[i] = float(
        ed_coup[i] * float(self._coupon_amounts[i])
    )
```

Pass `coupon_probability` and `expected_discounted_coupon_cashflow` into
`self._make_event_stats(...)` (Phoenix factory accepts them; the Snowball factory
ignores them since the base method never supplies them).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_pde_event_stats.py -x -q`
Expected: PASS (KO, survival, ki, and coupon all within band).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/phoenix_pde_solver.py \
        test/test_equity/test_phoenix_pde_event_stats.py
git commit -m "feat(pde): native Phoenix coupon_probability via indicator surfaces"
```

## Phase C — Native Phoenix QUAD event stats (replace MC delegation)

### Task 8: Phoenix QUAD — native KO/KI event stats

**Files:**
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py`
- Test: `test/test_equity/test_phoenix_quad_event_stats.py` (Create)

**Interfaces:**
- Consumes: the Snowball QUAD template `SnowballQuadEngine.calculate_event_stats` (`snowball_quad_engine.py:366-715`) and inherited helpers (`_match_record`, `_merge_times`, `_build_dt`, `_ko_discount`, `_diffuse_fft`, `_diffuse_with_bridge`, `_smooth_step_weight`, `_is_knocked_in_at_valuation`).
- Produces: native `PhoenixQuadEngine.calculate_event_stats -> PhoenixEventStats` with KO/survival/ki fields, **no `PhoenixMCEngine` construction**.

- [ ] **Step 1: Write the failing tests (KO/survival vs MC; no-MC guard)**

```python
# test/test_equity/test_phoenix_quad_event_stats.py
import numpy as np
import pytest

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.param.engine_params import QuadParams, MCParams
from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from test.test_equity._phoenix_fixtures import make_phoenix_product, make_env  # adjust to real helper


@pytest.fixture
def phoenix_case():
    env = make_env()
    return make_phoenix_product(env), env


def test_phoenix_quad_ko_survival_match_mc(phoenix_case):
    product, env = phoenix_case
    q = PhoenixQuadEngine(params=QuadParams(grid_points=2001))
    mc = PhoenixMCEngine(params=MCParams(num_paths=200_000, time_steps=252, seed=11))
    s_q = q.calculate_event_stats(product, env)
    s_mc = mc.calculate_event_stats(product, env)
    assert isinstance(s_q, PhoenixEventStats)
    np.testing.assert_allclose(s_q.ko_probability, s_mc.ko_probability, atol=5e-3)
    np.testing.assert_allclose(s_q.survival_probability, s_mc.survival_probability, atol=5e-3)
    assert abs(s_q.ki_probability - s_mc.ki_probability) < 5e-3


def test_phoenix_quad_builds_no_mc_engine(phoenix_case, monkeypatch):
    product, env = phoenix_case
    import quantark.asset.equity.engine.mc.phoenix_mc_engine as mcmod
    monkeypatch.setattr(mcmod, "PhoenixMCEngine",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("QUAD event stats must not build MC")))
    PhoenixQuadEngine(params=QuadParams(grid_points=1001)).calculate_event_stats(product, env)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_quad_event_stats.py::test_phoenix_quad_builds_no_mc_engine -x -q`
Expected: FAIL — current body constructs `PhoenixMCEngine(MCParams())` (`phoenix_quad_engine.py:486-495`), tripping the monkeypatched guard.

- [ ] **Step 3: Replace the MC delegation with a native KO/KI recursion**

Refactor `SnowballQuadEngine.calculate_event_stats` the same way as the PDE
(Task 6): keep the `isinstance` guard overridable via `_event_stats_product_type()`
and route construction through `_make_event_stats(**fields)`; base returns
`AutocallableEventStats`, Phoenix returns `PhoenixEventStats`. Then in
`PhoenixQuadEngine`, delete the MC delegation (lines 486-495) and override
`_event_stats_product_type()` to return `PhoenixOption`, reusing the inherited
KO indicator recursion (`snowball_quad_engine.py:482-600`) and KI sub-recursion
(`605-689`) verbatim. Snowball QUAD already propagates stacked KO surfaces and
extracts `ko_probability[i] = ed_unit[i] / df_total` (line 596) and
`survival = 1 - cumsum(ko_prob)` (line 600) — unchanged for Phoenix.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_quad_event_stats.py -x -q`
Then the Snowball QUAD regression:
`.venv/bin/python -m pytest test/ -k "snowball and event_stats and quad" -q`
Expected: PASS both (Phoenix KO/survival/ki within band; Snowball unchanged).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py \
        quantark/asset/equity/engine/quad/phoenix_quad_engine.py \
        test/test_equity/test_phoenix_quad_event_stats.py
git commit -m "feat(quad): native Phoenix KO/KI event stats; remove MC delegation"
```

### Task 9: Phoenix QUAD — coupon-trigger indicators → coupon_probability

**Files:**
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py`
- Test: `test/test_equity/test_phoenix_quad_event_stats.py` (extend)

**Interfaces:**
- Consumes: coupon barriers built as in `PhoenixQuadEngine.price()` (`phoenix_quad_engine.py:98-106`); `_smooth_step_weight`; `_diffuse_fft`/`_diffuse_with_bridge`; `_ko_discount`.
- Produces: `coupon_probability` / `expected_discounted_coupon_cashflow` on the returned `PhoenixEventStats`, matching `PhoenixMCEngine`.

- [ ] **Step 1: Write the failing coupon cross-check test**

```python
def test_phoenix_quad_coupon_prob_matches_mc(phoenix_case):
    product, env = phoenix_case
    q = PhoenixQuadEngine(params=QuadParams(grid_points=2001))
    mc = PhoenixMCEngine(params=MCParams(num_paths=200_000, time_steps=252, seed=11))
    s_q = q.calculate_event_stats(product, env)
    s_mc = mc.calculate_event_stats(product, env)
    assert s_q.coupon_probability.shape == s_mc.coupon_probability.shape
    np.testing.assert_allclose(s_q.coupon_probability, s_mc.coupon_probability, atol=5e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_quad_event_stats.py::test_phoenix_quad_coupon_prob_matches_mc -x -q`
Expected: FAIL — `coupon_probability` empty.

- [ ] **Step 3: Propagate coupon-trigger indicator surfaces**

In `PhoenixQuadEngine`, override the recursion to carry `n_ko` extra
coupon-trigger indicator rows alongside the KO indicator rows (stacked array
shape `(2*n_ko, grid)` or a second `v_out_coup` block). At each coupon
observation set the coupon row on the coupon-pay weight (use `_smooth_step_weight`
with the coupon barrier, falling back to a hard mask) to the discounted
indicator; diffuse with the same `_diffuse_fft`/`_diffuse_with_bridge` calls used
for the KO rows. Extract like KO:

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

Match the coupon-vs-simultaneous-KO convention to `PhoenixMCEngine` (Task 7 note).
Pass both arrays into `self._make_event_stats(...)`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest test/test_equity/test_phoenix_quad_event_stats.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/phoenix_quad_engine.py \
        test/test_equity/test_phoenix_quad_event_stats.py
git commit -m "feat(quad): native Phoenix coupon_probability via indicator surfaces"
```

## Phase D — Position integration & required golden parity

### Task 10: Position PV + Greeks across all engines

**Files:**
- Test: `test/test_cashleg/test_autocallable_leg_position.py` (Create)
- (No library change expected — `EquityPosition.get_trade_value`/`get_trade_greeks` already handle legs.)

**Interfaces:**
- Consumes: `EquityPosition(product, engine, quantity, cash_legs=[...])`, `.get_trade_value(env)`, `.get_trade_greeks(env, GreeksCalculator())`.

- [ ] **Step 1: Write the integration tests**

```python
# test/test_cashleg/test_autocallable_leg_position.py
import numpy as np
import pytest

from quantark.portfolio import EquityPosition
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
from quantark.cashleg.base import LegDirection
from quantark.cashleg.autocallable_leg import (
    AutocallableCashLeg, AutocallableLegType, PvFormula,
)
# Reuse the Snowball product/engine/env builders from existing cashleg tests:
#   grep -rn "SnowballOption(" test/test_cashleg | head
from test.test_cashleg._autocall_fixtures import (  # adjust import to real helper
    make_snowball, make_engine, make_env, future_ko_year_fractions,
)


def _margin_leg(env, product):
    obs = future_ko_year_fractions(product, env)   # == event_times of price_with_events
    n = len(obs)
    return AutocallableCashLeg(
        direction=LegDirection.BUYER_RECEIVES,
        leg_type=AutocallableLegType.MARGIN,
        notional=1_000_000.0, rate=0.04,
        observation_schedule=tuple(obs),
        accrual_factors=tuple(np.linspace(0.25, 1.0, n)),
        settlement_schedule=tuple(obs),
        terminal_accrual_factor=1.0,
        terminal_settlement_time=float(obs[-1]),
        pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF,
    )


@pytest.mark.parametrize("engine_kind", ["mc", "pde", "quad"])
def test_margin_leg_pv_and_greeks_finite(engine_kind):
    env = make_env()
    product = make_snowball(env)
    engine = make_engine(engine_kind)
    leg = _margin_leg(env, product)
    pos = EquityPosition(product=product, engine=engine, quantity=1.0,
                         underlying="UND", cash_legs=[leg])
    base = pos.get_trade_value(env)
    greeks = pos.get_trade_greeks(env, GreeksCalculator())
    assert np.isfinite(base)
    assert np.isfinite(greeks["delta"]) and abs(greeks["delta"]) > 0.0
    assert np.isfinite(greeks["gamma"])


def test_fail_loud_when_engine_emits_no_ko_stream():
    # A vanilla (non-autocallable) engine yields a trivial distribution → leg raises.
    env = make_env()
    product = make_snowball(env)
    leg = _margin_leg(env, product)
    from quantark.cashleg.event_distribution import EventDistribution
    from quantark.util.exceptions import ValidationError
    with pytest.raises(ValidationError):
        leg.value(EventDistribution.trivial(float(leg.terminal_settlement_time)), env, 0.0)
```

- [ ] **Step 2: Run to verify (expect pass once fixtures resolve)**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg_position.py -q`
Expected: PASS for all three engines (Phoenix variants gain PDE/QUAD support from Phases B/C; this task uses Snowball). If `delta == 0`, confirm the leg's `observation_schedule` exactly equals `price_with_events(...).event_distribution.event_times` (the schedule-identity guard would otherwise raise).

- [ ] **Step 3: Commit**

```bash
git add test/test_cashleg/test_autocallable_leg_position.py
git commit -m "test(cashleg): position PV + Greeks for AutocallableCashLeg across engines"
```

### Task 11: Required golden-case parity fixture (PV + Greeks)

**Files:**
- Create: `test/test_cashleg/fixtures/autocallable_golden_case.json`
- Test: `test/test_cashleg/test_autocallable_leg_golden.py`

**Interfaces:**
- Consumes: the frozen golden inputs + targets (sanitized; **no proprietary trade identifier**).
- Produces: a CI-enforced parity test (native legs vs frozen workaround targets) across supported engines.

- [ ] **Step 1: Create the sanitized fixture**

`test/test_cashleg/fixtures/autocallable_golden_case.json` holds the parent
autocallable parameters, the per-future-observation accrual + settlement factors,
the discount curve, and the frozen target PV/delta/gamma for `pv_margin`,
`pv_interest`, `pv_rebate`. Use neutral keys; do **not** include any external
trade identifier or vendor name. Frozen targets (model values):

```json
{
  "legs": {
    "pv_margin":   {"pv": 207475.74,  "delta": -1183832.92, "gamma": 98566.13},
    "pv_interest": {"pv": -3417.10,   "delta": 17000.95,    "gamma": -1393.67},
    "pv_rebate":   {"pv": -409798.69, "delta": -24505.34,   "gamma": 2040.32}
  },
  "total_delta": -13990506.81,
  "tolerances": {"pv": 1.0, "delta": 1000.0, "gamma": 50.0}
}
```

If the literal confirm cannot be committed, generate the parent params + factors
so the legacy synthetic-`SnowballOption` workaround reproduces these numbers, and
freeze the workaround's own outputs as the targets (native-vs-workaround parity
either way).

- [ ] **Step 2: Write the parity test**

```python
# test/test_cashleg/test_autocallable_leg_golden.py
import json, pathlib
import numpy as np
import pytest

FIX = pathlib.Path(__file__).parent / "fixtures" / "autocallable_golden_case.json"


def _build_case():
    """Build the parent product, engine, env, and the three legs from the fixture.
    Implement using the same builders referenced in Task 10; map JSON params to
    SnowballOption/PricingEnvironment fields and AutocallableCashLeg constructors.
    Returns (env, position, expected_dict)."""
    data = json.loads(FIX.read_text())
    ...  # construct from data — concrete builder lives alongside Task 10 fixtures
    return env, position, data


@pytest.mark.parametrize("engine_kind", ["mc", "pde", "quad"])
def test_golden_case_pv_and_greeks(engine_kind):
    env, position, data = _build_case()  # _build_case selects engine by engine_kind
    tol = data["tolerances"]
    breakdown = position.trade_value_breakdown(env)   # per-leg PV attribution
    for name, tgt in data["legs"].items():
        assert abs(breakdown.leg_pvs[name].pv - tgt["pv"]) <= tol["pv"], name
    from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator
    greeks = position.get_trade_greeks(env, GreeksCalculator())
    assert abs(greeks["delta"] - data["total_delta"]) <= tol["delta"]
```

(If a per-leg `trade_value_breakdown` accessor does not yet exist on
`EquityPosition`, add a thin method returning `TradeValueBreakdown` from
`quantark/cashleg/leg_valuator.py` — it already defines `LegPV` and
`TradeValueBreakdown`. Wire it as its own sub-step with a focused test before
using it here.)

- [ ] **Step 3: Run the parity test**

Run: `.venv/bin/python -m pytest test/test_cashleg/test_autocallable_leg_golden.py -q`
Expected: PASS across mc/pde/quad within the fixture tolerances. MC uses a fixed
seed and a tolerance band wide enough for its standard error; PDE/QUAD are tight.

- [ ] **Step 4: Commit**

```bash
git add test/test_cashleg/fixtures/autocallable_golden_case.json \
        test/test_cashleg/test_autocallable_leg_golden.py \
        quantark/portfolio/equity/position.py
git commit -m "test(cashleg): required golden-case PV+Greeks parity across engines"
```

### Task 12: Full-suite green + docs touch

**Files:**
- Modify: `quantark/cashleg/CLAUDE.md` (note the new leg type; the file currently lists stale modules — add `autocallable_leg.py` and the `AutocallableCashLeg` row, and correct the module list to match the real directory).

- [ ] **Step 1: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (no regressions). Investigate and fix any failure before committing.

- [ ] **Step 2: Update `cashleg/CLAUDE.md`**

Add an `AutocallableCashLeg` row to the leg-types table (KO/coupon-contingent
return leg; margin via `NOTIONAL_MINUS_PAYOFF`) and reconcile the "Module
Structure" list with the actual files (`autocallable_leg.py`, and remove the
modules that do not exist).

- [ ] **Step 3: Commit**

```bash
git add quantark/cashleg/CLAUDE.md
git commit -m "docs(cashleg): document AutocallableCashLeg; fix module list"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Plan task(s) |
|--------------|--------------|
| §3 architecture + `value_standalone` | Tasks 1, 3, 4 |
| §3a fail-loud guard | Tasks 3, 10 |
| §3b native Phoenix PDE event stats | Tasks 6, 7 |
| §3b native Phoenix QUAD event stats | Tasks 8, 9 |
| §4 dataclass (explicit settlement/terminal/schedule) | Task 1 |
| §5 valuation (R, both PV formulas, COUPON basis) | Tasks 3, 4 |
| §5 alignment contract (length + shifted-slice) | Tasks 1, 3 |
| §5 `NOTIONAL_MINUS_PAYOFF` validity domain | Tasks 1, 3 |
| §6 Greeks via existing bump loop | Task 10 |
| §6 exports/registration | Task 5 |
| §7 per-leg-type / invariants / regression | Tasks 3, 4, 10 |
| §7 Phoenix PDE/QUAD-vs-MC cross-check | Tasks 6–9 |
| §7 required golden fixture | Task 11 |
| §8/§9 | Tasks 6–11 scope |

All five leg types are exercised: margin (Task 11 `pv_margin` + Task 3),
backend_interest (Task 11 `pv_interest`), rebate (Task 11 `pv_rebate`); backend_premium
and minimum_return share the `NORMAL`/`KO_MATURITY` path proven in Task 3 — add a
parametrized unit case over all five `AutocallableLegType` values to Task 3's file
so requirement #6's "cover all 5 leg types" is explicit.

**Placeholder scan:** the only deliberately-deferred specifics are the project's
existing test fixtures/builders (`_phoenix_fixtures`, `_autocall_fixtures`),
flagged with the exact `grep` to locate them in Task 0/6/10, and the JSON→objects
mapping in Task 11's `_build_case` (depends on those builders). These are
codebase-discovery steps, not logic placeholders.

**Type consistency:** `_make_event_stats(**fields)` / `_event_stats_product_type()`
seams are introduced once (Task 6) and reused identically (Tasks 7–9);
`AutocallableCashLeg.value(event_dist, env, position_notional)` and
`value_standalone(parent_product, engine, env)` signatures are stable across
Tasks 3, 4, 10, 11; `PhoenixEventStats` coupon field names
(`coupon_probability`, `expected_discounted_coupon_cashflow`) match
`event_stats.py:48-61`.

**Added during review:** the parametrized all-five-leg-types unit case (Task 3),
and the `trade_value_breakdown` accessor sub-step (Task 11) if absent.
