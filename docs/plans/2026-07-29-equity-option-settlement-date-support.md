# Equity Option Settlement-Date Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every supported cash-settled equity-option valuation path distinguish payoff determination from payment, discount each cashflow to its own payment date/time, and preserve an append-only pending-receivable lifecycle through payment.

**Architecture:** Add one engine-independent settlement resolver and one immutable lifecycle cashflow ledger. Products describe contractual settlement terms, engines consume normalized `ResolvedPaymentTiming`, and portfolio/lifecycle/risk layers propagate the same state. Stochastic dynamics always stop at determination; delayed payment affects only discounting, except that American exercise compares continuation with a node-specific delayed-exercise obstacle.

**Tech Stack:** Python 3, dataclasses, NumPy/SciPy, QuantArk calendar and curve abstractions, pytest, analytical/Monte Carlo/PDE/quadrature equity engines.

---

## Working contract

Implement against the approved design:

- `docs/superpowers/specs/2026-07-29-equity-option-settlement-date-support-design.md`

Use this isolated worktree and branch:

```text
worktree: /private/tmp/quant-ark-equity-option-settlement-date
branch:   codex/equity-option-settlement-date
base:     f26f2bbfe5d374625d2319556287ec22ee33a14c
```

The main checkout contains unrelated user work, including a separate
constructor/date-normalization stream. Do not copy files from that dirty
checkout. Before Task 2, integrate the committed result of that stream into
this branch, resolve only semantic overlaps, and rerun its constructor
contract tests.

Use this focused test command throughout:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider <test paths>
```

For each task:

1. add the named failing test;
2. run the narrow test and confirm it fails for the stated reason;
3. implement the smallest complete slice;
4. run the narrow and adjacent regression tests;
5. inspect `git diff --check`;
6. commit only the named slice.

Do not weaken assertions merely to accommodate numerical noise. Use exact
curve identities where possible and the repository's centralized numerical
tolerance helpers where approximation is intrinsic.

## Non-negotiable invariants

- No settlement terms means payment at determination and price identity with
  the pre-feature implementation.
- `settlement_date` is a terminal-cashflow override only.
- Event cashflows use record-level payment terms first, then the product
  convention; they never inherit terminal `settlement_date`.
- Explicit dates/times beat derived conventions.
- Date-based lags require an authoritative determination date.
- Numeric-time products require `explicit_payment_time` or a
  `YEAR_FRACTION` lag; never synthesize a date from `time * 365`.
- `payment_time >= determination_time`.
- Dynamics end at determination. Payment timing contributes only a
  curve-exact discount ratio.
- Mixed-payment products are decomposed and discounted leg by leg.
- On or after determination, contingent value may be replaced only by an
  authoritative realized cashflow in lifecycle state.
- On payment, derivative and pending-receivable PV are zero; the paid cash
  account books the amount once.
- Unsupported engine/formula combinations raise the existing
  `quantark.execution.errors.CapabilityError`. The design's generic name
  `EngineCapabilityError` maps to this existing framework type.

## Repository impact map

| Layer | Primary files | Required result |
|---|---|---|
| Contract kernel | `quantark/asset/equity/settlement.py` | One precedence rule and one curve-exact timing result |
| Products | `quantark/asset/equity/product/option/*.py` | Optional convention propagated without breaking constructors |
| Schedules | `observation_schedule.py`, DCN schedule/grid | Explicit date/time retained and resolved fail-closed |
| Lifecycle | `quantark/asset/equity/lifecycle/*.py` | Immutable pending/paid ledger, no immediate cash booking |
| Engine API | `base_engine.py`, execution adapters, positions | Optional lifecycle state propagated everywhere |
| Engines | analytical, MC, PDE, QUAD packages | Per-cashflow settlement, or explicit capability rejection |
| Risk | `greeks_calculator.py`, position Greeks, bucketed Greeks | Repricing preserves settlement and lifecycle state |
| Attribution | `event_stats.py`, `cashleg/event_distribution.py` | Determination and payment timing both visible and PV-reconciled |

## Task 0: Integrate the concurrent constructor contract and record a clean baseline

**Files:**

- Integrate: the committed branch/commit from the separate
  constructor/date-normalization session
- Verify: `quantark/asset/equity/product/option/base_equity_option.py`
- Verify: every concrete file in
  `quantark/asset/equity/product/option/*.py`
- Test: `test/test_base_equity_option_contract.py`
- Test: `test/test_european_option.py`
- Test: `test/test_american_option_analytical.py`
- Test: `test/test_barrier_option_mc_engine.py`
- Test: `test/test_snowball_option.py`
- Test: `test/test_phoenix_option.py`
- Test: `test/test_dcn_option.py`

**Step 1: Verify the planning worktree is clean**

Run:

```bash
git status --short --branch
git log -1 --oneline
```

Expected: branch `codex/equity-option-settlement-date`, no uncommitted files
other than this plan before it is committed.

**Step 2: Integrate only committed constructor work**

After the other session supplies a commit:

```bash
git merge --no-ff <constructor-normalization-commit>
```

If the result is already on a shared ancestor, use:

```bash
git rebase <constructor-normalization-branch>
```

Resolve overlaps around shared lifecycle/date fields, not unrelated style.
Preserve its decision about ownership of `initial_price`, constructor
keyword-only boundaries, and date validation.

**Step 3: Run the constructor baseline**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_base_equity_option_contract.py \
  test/test_european_option.py \
  test/test_american_option_analytical.py \
  test/test_barrier_option_mc_engine.py \
  test/test_snowball_option.py \
  test/test_phoenix_option.py \
  test/test_dcn_option.py
```

Expected: PASS before settlement implementation begins. If the constructor
session has no dedicated test file, add its committed test path to this list
rather than recreating its work.

**Step 4: Record the baseline**

Run:

```bash
git status --short
git rev-parse HEAD
```

Do not create a synthetic baseline commit if the merge/rebase already
recorded the integration.

## Task 1: Build the fail-closed settlement timing kernel

**Files:**

- Create: `quantark/asset/equity/settlement.py`
- Modify: `quantark/asset/equity/__init__.py`
- Create: `test/test_equity_settlement_resolver.py`

**Step 1: Write resolver contract tests**

Cover all precedence and validation paths:

```python
from datetime import datetime

import pytest

from quantark.asset.equity.settlement import (
    CashflowKind,
    SettlementConvention,
    SettlementLagUnit,
    SettlementRequest,
    SettlementResolver,
)
from quantark.param import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import (
    BusinessDayConvention,
    Calendar,
    DayCountConvention,
)
from quantark.util.exceptions import ValidationError


def test_zero_lag_is_identity(date_option, pricing_env):
    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=date_option.exercise_date,
        ),
        pricing_env,
    )
    assert timing.payment_date == timing.determination_date
    assert timing.payment_time == pytest.approx(timing.determination_time)
    assert timing.delay_df == pytest.approx(1.0)


def test_explicit_event_payment_beats_product_convention(
    date_option, pricing_env
):
    date_option.settlement_convention = SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
        calendar=pricing_env.calendar,
    )
    explicit = datetime(2026, 8, 7)
    timing = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.COUPON,
            determination_date=datetime(2026, 8, 3),
            explicit_payment_date=explicit,
        ),
        pricing_env,
    )
    assert timing.payment_date == explicit


def test_terminal_override_does_not_apply_to_event(
    date_option, pricing_env
):
    terminal_payment = datetime(2026, 9, 10)
    date_option.settlement_date = terminal_payment
    event = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.COUPON,
            determination_date=datetime(2026, 8, 3),
        ),
        pricing_env,
    )
    terminal = SettlementResolver.resolve_contingent(
        date_option,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=date_option.exercise_date,
        ),
        pricing_env,
    )
    assert event.payment_date == event.determination_date
    assert terminal.payment_date == terminal_payment


def test_time_only_determination_rejects_business_day_lag(
    time_option, pricing_env
):
    time_option.settlement_convention = SettlementConvention(
        lag=2,
        lag_unit=SettlementLagUnit.BUSINESS_DAYS,
        calendar=pricing_env.calendar,
    )
    with pytest.raises(ValidationError, match="authoritative determination date"):
        SettlementResolver.resolve_contingent(
            time_option,
            SettlementRequest(
                kind=CashflowKind.TERMINAL,
                determination_time=1.0,
            ),
            pricing_env,
        )


@pytest.mark.parametrize("bad_lag", [-1.0, float("nan"), float("inf")])
def test_invalid_lag_rejected(bad_lag):
    with pytest.raises(ValidationError):
        SettlementConvention(lag=bad_lag)
```

Also test:

- T+2 business days across a weekend and explicit holiday;
- calendar-day lag plus `MODIFIED_FOLLOWING`;
- fractional `YEAR_FRACTION` lag;
- explicit date and explicit time consistency;
- determination date and time consistency;
- explicit payment before determination;
- missing business calendar;
- terminal override before expiry;
- non-positive/non-finite discount factors using a test curve;
- clear error text including product type, cashflow kind, and requested terms.

**Step 2: Run the tests to verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_equity_settlement_resolver.py
```

Expected: FAIL because `quantark.asset.equity.settlement` does not exist.

**Step 3: Implement the value types and resolver**

Use this public shape:

```python
class SettlementLagUnit(Enum):
    BUSINESS_DAYS = "business_days"
    CALENDAR_DAYS = "calendar_days"
    YEAR_FRACTION = "year_fraction"


@dataclass(frozen=True)
class SettlementConvention:
    lag: float = 0.0
    lag_unit: SettlementLagUnit = SettlementLagUnit.BUSINESS_DAYS
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.FOLLOWING
    )
    calendar: Optional[Calendar] = None

    def __post_init__(self) -> None:
        if not isfinite(self.lag) or self.lag < 0.0:
            raise ValidationError("settlement lag must be finite and non-negative")
        if (
            self.lag_unit
            in {SettlementLagUnit.BUSINESS_DAYS, SettlementLagUnit.CALENDAR_DAYS}
            and not float(self.lag).is_integer()
        ):
            raise ValidationError("day-based settlement lag must be integral")


class CashflowKind(Enum):
    TERMINAL = "terminal"
    EXERCISE = "exercise"
    HIT = "hit"
    OBSERVATION = "observation"
    COUPON = "coupon"
    REDEMPTION = "redemption"
    REBATE = "rebate"


@dataclass(frozen=True)
class SettlementRequest:
    kind: CashflowKind
    determination_date: Optional[datetime] = None
    determination_time: Optional[float] = None
    explicit_payment_date: Optional[datetime] = None
    explicit_payment_time: Optional[float] = None
    cashflow_id: Optional[str] = None


@dataclass(frozen=True)
class ResolvedPaymentTiming:
    kind: CashflowKind
    determination_date: Optional[datetime]
    determination_time: float
    payment_date: Optional[datetime]
    payment_time: float
    determination_df: float
    payment_df: float
    delay_df: float
```

Implement:

```python
class SettlementResolver:
    @classmethod
    def resolve_contingent(
        cls,
        product,
        request: SettlementRequest,
        pricing_env: PricingEnvironment,
    ) -> ResolvedPaymentTiming:
        determination_date, determination_time = cls._resolve_determination(
            request, pricing_env
        )
        payment_date, payment_time = cls._resolve_payment(
            product,
            request,
            determination_date,
            determination_time,
            pricing_env,
        )
        cls._validate_ordering(
            product, request, determination_time, payment_time
        )
        determination_df = cls._checked_df(
            pricing_env, determination_time, "determination"
        )
        payment_df = cls._checked_df(pricing_env, payment_time, "payment")
        return ResolvedPaymentTiming(
            kind=request.kind,
            determination_date=determination_date,
            determination_time=determination_time,
            payment_date=payment_date,
            payment_time=payment_time,
            determination_df=determination_df,
            payment_df=payment_df,
            delay_df=payment_df / determination_df,
        )
```

Resolution must use this order:

1. request explicit payment date/time;
2. product `settlement_date` only for `TERMINAL`;
3. product `settlement_convention`;
4. determination identity.

For date conversion use:

```python
calculate_year_fraction(
    pricing_env.valuation_date,
    target_date,
    pricing_env.day_count_convention,
    pricing_env.bus_days_in_year,
    calendar=pricing_env.calendar,
)
```

For calendar-day lag, add `timedelta(days=int(lag))` and then call the
selected calendar's `adjust_date`. For business-day lag, call
`calendar.add_business_days`. Explicit payment dates are contractual and are
not adjusted again.

Do not import pricing engines from this module. Put `PricingEnvironment`
behind `TYPE_CHECKING` if needed to keep imports acyclic.

Export the settlement module or its public values from
`quantark/asset/equity/__init__.py`.

**Step 4: Run the resolver tests**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_equity_settlement_resolver.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add quantark/asset/equity/settlement.py \
  quantark/asset/equity/__init__.py \
  test/test_equity_settlement_resolver.py
git commit -m "feat(settlement): add equity payment timing resolver"
```

## Task 2: Propagate settlement terms through products and schedules

**Files:**

- Modify: `quantark/asset/equity/product/option/base_equity_option.py`
- Modify: every concrete constructor under
  `quantark/asset/equity/product/option/*.py`
- Modify: `quantark/asset/equity/product/option/observation_schedule.py`
- Modify: `quantark/asset/equity/product/option/dcn_schedule.py`
- Modify: `quantark/asset/equity/product/option/dcn_grid.py`
- Modify: `quantark/asset/equity/product/option/dcn_option.py`
- Modify: `quantark/asset/equity/product/option/__init__.py`
- Create: `test/test_equity_option_settlement_contract.py`
- Modify: `test/test_asian_observation_record.py`
- Modify: `test/test_dcn_schedule.py`

**Step 1: Write product and schedule contract tests**

Add an introspection test that instantiates every option family through its
public constructor and asserts that a passed `SettlementConvention` is
preserved. Include:

- European, American, digital, Asian;
- single/double barrier, one/double touch;
- single/double sharkfin;
- accumulator, range accrual;
- Snowball, Phoenix, KO-reset Snowball, DCN.

Add schedule precedence tests:

```python
def test_record_explicit_settlement_time_survives_resolution(env):
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_time=0.5,
                settlement_time=0.55,
                barrier=110.0,
            )
        ]
    )
    [record] = schedule.resolve(env, require_single=True)
    assert record.observation_time == pytest.approx(0.5)
    assert record.settlement_time == pytest.approx(0.55)


def test_invalid_record_settlement_date_fails_closed(env):
    schedule = ObservationSchedule(
        records=[
            ObservationRecord(
                observation_date=datetime(2026, 9, 1),
                settlement_date=datetime(2026, 8, 31),
                barrier=110.0,
            )
        ]
    )
    with pytest.raises(ValidationError, match="before"):
        schedule.resolve(env, require_single=True)
```

Test that a product terminal `settlement_date` does not change an event
record's payment time and that a record without explicit payment uses the
product convention when the product is supplied to `resolve`.

**Step 2: Run the tests to verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_equity_option_settlement_contract.py \
  test/test_asian_observation_record.py \
  test/test_dcn_schedule.py
```

Expected: FAIL because constructors do not expose the convention and
`ObservationRecord` has no `settlement_time`.

**Step 3: Add the product field without reopening constructor ownership**

Add to the normalized base constructor:

```python
settlement_convention: Optional[SettlementConvention] = None
```

Store and validate the type. Forward the keyword unchanged from each concrete
product constructor. Do not change the concurrent session's decisions about
`initial_price`, `maturity_date`, `tenor_end`, or keyword-only placement.

Standalone products that do not inherit `BaseEquityOption` must expose the
same optional field directly.

**Step 4: Make observation settlement resolution use the shared kernel**

Add:

```python
settlement_time: Optional[float] = None
```

to `ObservationRecord`. Change `ObservationSchedule.resolve` to accept the
owning `product` or a `settlement_convention` context. Build a
`SettlementRequest(kind=CashflowKind.OBSERVATION, ...)` for each record and
delegate to `SettlementResolver`.

Remove the broad:

```python
except Exception:
    settlement_t = t
```

Invalid dates, missing calendars, and inconsistent date/time pairs must
propagate as `ValidationError`. Preserve both observation and settlement
dates in `ResolvedObservationRecord` when authoritative dates exist so later
event statistics need not reconstruct them.

For DCN, retain its explicit contractual payment dates and feed them as
`explicit_payment_date`; do not derive replacement dates.

**Step 5: Run product and schedule regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_equity_option_settlement_contract.py \
  test/test_asian_observation_record.py \
  test/test_asian_option.py \
  test/test_snowball_option.py \
  test/test_phoenix_option.py \
  test/test_dcn_schedule.py \
  test/test_dcn_option.py
```

Expected: PASS.

**Step 6: Commit**

Stage only product/schedule files and their tests:

```bash
git add quantark/asset/equity/product/option \
  test/test_equity_option_settlement_contract.py \
  test/test_asian_observation_record.py \
  test/test_dcn_schedule.py
git commit -m "feat(settlement): propagate option payment terms"
```

## Task 3: Add the immutable realized-cashflow ledger

**Files:**

- Create: `quantark/asset/equity/lifecycle/cashflows.py`
- Modify: `quantark/asset/equity/lifecycle/state.py`
- Modify: `quantark/asset/equity/lifecycle/__init__.py`
- Create: `test/test_lifecycle_cashflow_ledger.py`
- Modify: `test/test_equity_lifecycle_trackers.py`

**Step 1: Write ledger tests**

Cover:

- registering a new cashflow returns `True`;
- registering an identical payload is idempotent and returns `False`;
- the same ID with conflicting amount, payment date, or payment time raises;
- pending/paid partition at dates and numeric times;
- pending PV uses `DF(valuation, payment)` and not the original
  determination DF;
- `realized_cashflows` compatibility property includes paid cash only;
- state snapshots/deepcopies preserve ledger contents.

Use the public types:

```python
cashflow = RealizedCashflow(
    cashflow_id="trade-1:coupon:3",
    event_type=LifecycleEventType.COUPON,
    amount=12.5,
    determination_date=datetime(2026, 8, 3),
    payment_date=datetime(2026, 8, 5),
)
ledger = LifecycleCashflowLedger()
assert ledger.register(cashflow)
assert not ledger.register(cashflow)
assert ledger.pending(ValuationPoint(date=datetime(2026, 8, 4))) == (cashflow,)
assert ledger.paid(ValuationPoint(date=datetime(2026, 8, 5))) == (cashflow,)
```

**Step 2: Run the tests to verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_lifecycle_cashflow_ledger.py
```

Expected: FAIL because `cashflows.py` does not exist.

**Step 3: Implement the immutable ledger**

Use:

```python
@dataclass(frozen=True)
class ValuationPoint:
    date: Optional[datetime] = None
    time: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.date is None) == (self.time is None):
            raise ValidationError("valuation point requires exactly one representation")


@dataclass(frozen=True)
class RealizedCashflow:
    cashflow_id: str
    event_type: LifecycleEventType
    amount: float
    determination_date: Optional[datetime] = None
    determination_time: Optional[float] = None
    payment_date: Optional[datetime] = None
    payment_time: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        # Validate finite amount, matching representations, and payment ordering.
```

The ledger owns a private dict keyed by `cashflow_id` and exposes immutable,
deterministically ordered tuples. Its `pending_pv` calls
`SettlementResolver.resolve_pending`; it never mutates status.

Add `resolve_pending(realized_cashflow, pricing_env)` to the resolver. This
path accepts past determination, requires future payment for pending value,
and computes the payment time relative to the current valuation environment.

Compose `LifecycleCashflowLedger` into
`AutocallableLifecycleState` and `BarrierLifecycleState`. Replace the mutable
scalar field with:

```python
ledger: LifecycleCashflowLedger = field(default_factory=LifecycleCashflowLedger)

@property
def realized_cashflows(self) -> float:
    return self.ledger.paid_total(self.valuation_point)
```

Provide explicit `valuation_point` ownership in state/tracker code; do not
guess dates from numeric times.

Define and export a structural protocol in `state.py`:

```python
class EquityOptionLifecycleState(Protocol):
    valuation_point: Optional[ValuationPoint]
    ledger: LifecycleCashflowLedger
```

The concrete state classes may retain their product-specific alive/KO/KI
flags. Before the first tracker observation, `valuation_point` may be `None`
and the paid-cash compatibility property returns zero.

**Step 4: Run ledger and lifecycle state tests**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_lifecycle_cashflow_ledger.py \
  test/test_equity_lifecycle_trackers.py \
  test/test_snowball_lifecycle_ki.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add quantark/asset/equity/lifecycle \
  quantark/asset/equity/settlement.py \
  test/test_lifecycle_cashflow_ledger.py \
  test/test_equity_lifecycle_trackers.py
git commit -m "feat(lifecycle): add pending cashflow ledger"
```

## Task 4: Add shared engine settlement helpers and explicit capabilities

**Files:**

- Modify: `quantark/asset/equity/engine/base_engine.py`
- Modify: `quantark/asset/equity/engine/capabilities.py`
- Create: `quantark/asset/equity/engine/settlement_support.py`
- Modify: concrete `price` signatures under
  `quantark/asset/equity/engine/{analytical,mc,pde,quad}/`
- Modify: `quantark/execution/legacy_adapter.py`
- Modify:
  `quantark/asset/equity/engine/mc/autocallable_execution_adapters.py`
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Modify: `quantark/asset/equity/engine/pde/pde_execution_adapters.py`
- Create: `test/test_engine_settlement_capabilities.py`
- Create: `test/test_engine_lifecycle_signature.py`

**Step 1: Write API and capability tests**

Introspect every concrete equity option engine:

```python
signature = inspect.signature(engine.price)
assert signature.parameters["lifecycle_state"].kind is inspect.Parameter.KEYWORD_ONLY
assert signature.parameters["lifecycle_state"].default is None
```

Test the existing BSM/LV/Heston/SLV by MC/PDE/QUAD matrix:

- supported cells keep their current support;
- QUAD remains unsupported for LV/Heston/SLV;
- a requested settlement form must have a declared support level;
- a path that has not opted in raises `CapabilityError` before simulation or
  grid construction.

**Step 2: Run the tests to verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_engine_settlement_capabilities.py \
  test/test_engine_lifecycle_signature.py
```

Expected: FAIL because signatures and settlement capabilities are absent.

**Step 3: Define support levels and shared helpers**

Use an explicit support enum:

```python
class SettlementSupport(Enum):
    NONE = "none"
    TERMINAL_ONLY = "terminal_only"
    EVENT_AND_TERMINAL = "event_and_terminal"
    AMERICAN_EXERCISE = "american_exercise"
```

Extend `EngineCapability` with a backward-compatible default, and add a
validator that accepts product, engine, and requested cashflow timings. Reuse
`quantark.execution.errors.CapabilityError` for a semantically unsupported
request. Continue using `ValidationError` for malformed settlement data.

In `settlement_support.py`, add small shared operations:

```python
def resolve_terminal_timing(product, env) -> ResolvedPaymentTiming:
    return SettlementResolver.resolve_contingent(
        product,
        SettlementRequest(
            kind=CashflowKind.TERMINAL,
            determination_date=getattr(product, "exercise_date", None),
            determination_time=product.get_maturity(env),
        ),
        env,
    )


def apply_determination_to_payment(value_at_determination, timing):
    return value_at_determination * timing.delay_df


def pending_receivable_pv(lifecycle_state, env) -> float:
    if lifecycle_state is None:
        return 0.0
    return lifecycle_state.ledger.pending_pv(
        ValuationPoint(date=env.valuation_date), env
    )
```

Only pass both determination date and time when they are independently
authoritative and consistent. For numeric products pass only time.

**Step 4: Normalize public engine signatures**

Change the base contract to:

```python
def price(
    self,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    *,
    lifecycle_state: Optional[EquityOptionLifecycleState] = None,
) -> float:
```

Apply the keyword to every concrete equity option engine. At this stage,
engines not yet settlement-enabled must call the capability guard and reject
non-zero settlement; zero-lag behavior stays unchanged.

Propagate the keyword through `price_with_events`,
`calculate_greeks`, execution legacy/batch adapters, and bump contexts.
Do not add it to unrelated FX, credit, or TRS engines.

**Step 5: Run API, capability, and execution regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_engine_settlement_capabilities.py \
  test/test_engine_lifecycle_signature.py \
  test/execution/test_greek_bump_cells.py \
  test/execution/test_dcn_batch_adapter.py \
  test/test_greeks_mode_and_engine_type.py
```

Expected: PASS. No expensive engine should run before an unsupported request
is rejected.

**Step 6: Commit**

```bash
git add quantark/asset/equity/engine \
  quantark/execution/legacy_adapter.py \
  test/test_engine_settlement_capabilities.py \
  test/test_engine_lifecycle_signature.py \
  test/execution/test_greek_bump_cells.py
git commit -m "feat(settlement): declare engine timing capabilities"
```

## Task 5: Implement the European BSM reference slice across analytical, MC, PDE, and QUAD

**Files:**

- Modify: `quantark/asset/equity/engine/analytical/black_scholes_engine.py`
- Modify: `quantark/asset/equity/engine/mc/euro_mc_engine.py`
- Modify: `quantark/asset/equity/engine/pde/european_pde_solver.py`
- Modify: `quantark/asset/equity/engine/quad/european_quad_engine.py`
- Modify: `test/test_european_option.py`
- Modify: `test/test_euro_mc_engine.py`
- Modify: `test/test_european_quad_engine.py`
- Create: `test/test_european_settlement_matrix.py`

**Step 1: Write exact curve-identity tests**

For a terminal payoff determined at `Td` and paid at `Tp`, assert:

```python
immediate = engine.price(immediate_product, env)
delayed = engine.price(delayed_product, env)
expected = immediate * (
    env.get_discount_factor(Tp) / env.get_discount_factor(Td)
)
assert delayed == pytest.approx(expected, rel=engine_tolerance)
```

Use:

- a flat curve for an intuitive `exp(-r * (Tp - Td))` check;
- an interpolated curve to prove the engine uses curve DFs, not a flat rate;
- date-based explicit terminal settlement;
- numeric `YEAR_FRACTION` lag;
- zero-lag bitwise or tight-tolerance identity;
- delayed put-call parity:

```text
C - P = DF(Tp) * (F(Td) - K)
```

where the underlying distribution remains at `Td`.

For MC, reuse identical random draws/seed so the ratio test is deterministic.
For PDE and QUAD, compare the same grid/config with tolerances justified by
their existing convergence behavior.

**Step 2: Run the matrix test to verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_european_settlement_matrix.py
```

Expected: FAIL because engines still discount to determination.

**Step 3: Apply the same normalized timing in all four engines**

Analytical:

```python
timing = resolve_terminal_timing(product, pricing_env)
value_at_determination_discounting = existing_price
price = value_at_determination_discounting * timing.delay_df
```

Apply the factor before lower-bound validation, and update the delayed lower
bound:

```text
call >= max(0, S0 * carry_to_Td * DF(Tp)/DF(Td) - K * DF(Tp))
put  >= max(0, K * DF(Tp) - S0 * carry_to_Td * DF(Tp)/DF(Td))
```

MC:

- simulate paths only to `Td`;
- compute terminal payoff unchanged;
- multiply by `timing.payment_df`, not the maturity DF;
- do not append payment time to the stochastic path grid.

PDE:

- retain the PDE time grid through `Td`;
- multiply the terminal payoff vector by `timing.delay_df` when injecting it
  at determination;
- do not extend the PDE grid to `Tp`.

QUAD:

- retain transition recursion through `Td`;
- replace terminal DF use with `timing.payment_df`, or multiply the
  determination-discounted result by `timing.delay_df`;
- do not change transition variance or carry.

Add lifecycle handling: a live product returns contingent PV plus earlier
pending ledger PV; a terminally determined product requires an authoritative
realized cashflow and returns only its pending PV.

**Step 4: Run focused regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_european_settlement_matrix.py \
  test/test_european_option.py \
  test/test_euro_mc_engine.py \
  test/test_european_quad_engine.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add quantark/asset/equity/engine/analytical/black_scholes_engine.py \
  quantark/asset/equity/engine/mc/euro_mc_engine.py \
  quantark/asset/equity/engine/pde/european_pde_solver.py \
  quantark/asset/equity/engine/quad/european_quad_engine.py \
  test/test_european_option.py \
  test/test_euro_mc_engine.py \
  test/test_european_quad_engine.py \
  test/test_european_settlement_matrix.py
git commit -m "feat(settlement): price delayed European payoffs"
```

## Task 6: Extend terminal settlement to supported LV, Heston, and SLV cells

**Files:**

- Modify: `quantark/asset/equity/engine/analytical/heston_analytical_engine.py`
- Modify: `quantark/asset/equity/engine/mc/heston_mc_engine.py`
- Modify: `quantark/asset/equity/engine/mc/local_vol_mc_engine.py`
- Modify: `quantark/asset/equity/engine/mc/heston_slv_mc_engine.py`
- Modify: `quantark/asset/equity/engine/pde/heston_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/local_vol_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/heston_slv_pde_solver.py`
- Modify: `quantark/asset/equity/engine/capabilities.py`
- Create: `test/test_vol_model_terminal_settlement.py`

**Step 1: Write model/engine matrix tests**

Parameterize over supported engine fixtures:

```python
@pytest.mark.parametrize(
    "dynamics,engine_name",
    [
        ("local_vol", "mc"),
        ("local_vol", "pde"),
        ("heston", "analytical"),
        ("heston", "mc"),
        ("heston", "pde"),
        ("slv", "mc"),
        ("slv", "pde"),
    ],
)
def test_terminal_delay_scales_value_without_changing_dynamics(...):
    ...
```

For each deterministic terminal cashflow, assert the same
`DF(Tp) / DF(Td)` identity. For stochastic payoff engines, freeze seeds or
prepared grids and verify:

- path/grid stopping time remains `Td`;
- delayed and immediate prices differ only by the curve ratio;
- zero lag reproduces the prior price.

Add explicit rejection tests for LV/Heston/SLV QUAD requests using
`CapabilityError`.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_vol_model_terminal_settlement.py
```

Expected: supported engines ignore or reject settlement before this task.

**Step 3: Reuse the reference-slice integration points**

For analytical Heston, apply the terminal delay ratio to the existing
expiry-determined formula. For MC engines, replace terminal payoff
discounting with `payment_df`. For PDE engines, inject `delay_df * payoff` at
the determination boundary.

Do not:

- evolve variance/local volatility/leverage through `Tp`;
- request volatility at `Tp`;
- change time-step counts because payment is delayed;
- add QUAD support for dynamics that the current capability matrix rejects.

**Step 4: Run focused model regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_vol_model_terminal_settlement.py \
  test/volmodels/test_barrier_heston_mc.py \
  test/volmodels/test_barrier_heston_pde.py \
  test/volmodels/test_barrier_lv_mc.py \
  test/volmodels/test_barrier_lv_pde.py \
  test/volmodels/test_barrier_slv_mc.py \
  test/volmodels/test_barrier_slv_pde.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add quantark/asset/equity/engine/analytical/heston_analytical_engine.py \
  quantark/asset/equity/engine/mc/heston_mc_engine.py \
  quantark/asset/equity/engine/mc/local_vol_mc_engine.py \
  quantark/asset/equity/engine/mc/heston_slv_mc_engine.py \
  quantark/asset/equity/engine/pde/heston_pde_solver.py \
  quantark/asset/equity/engine/pde/local_vol_pde_solver.py \
  quantark/asset/equity/engine/pde/heston_slv_pde_solver.py \
  quantark/asset/equity/engine/capabilities.py \
  test/test_vol_model_terminal_settlement.py
git commit -m "feat(settlement): support terminal vol-model payoffs"
```

## Task 7: Cover digital, Asian, range-accrual, and accumulator cashflows

**Files:**

- Modify: `quantark/asset/equity/engine/analytical/digital_option_engine.py`
- Modify: `quantark/asset/equity/engine/mc/digital_option_mc_engine.py`
- Modify: `quantark/asset/equity/engine/analytical/asian_option_analytical_engine.py`
- Modify: `quantark/asset/equity/engine/mc/asian_option_mc_engine.py`
- Modify: `quantark/asset/equity/engine/analytical/range_accrual_analytical_engine.py`
- Modify: `quantark/asset/equity/engine/mc/range_accrual_mc_engine.py`
- Modify: `quantark/asset/equity/engine/analytical/accumulator_analytical_engine.py`
- Modify: `quantark/asset/equity/engine/mc/accumulator_mc_engine.py`
- Create: `test/test_terminal_and_leg_settlement.py`
- Modify: existing digital, Asian, range, and accumulator engine tests

**Step 1: Write single- and multi-cashflow tests**

Test:

- digital and Asian terminal payoff scale by the terminal curve ratio;
- terminal range-accrual amount uses terminal payment timing;
- each accumulator fixing/settlement leg uses its own record payment timing;
- an accumulator with two equal deterministic legs at `Tp1` and `Tp2` equals
  `amount1 * DF(Tp1) + amount2 * DF(Tp2)`;
- terminal settlement override does not replace accumulator event settlement;
- analytical/MC results reconcile for deterministic or high-precision cases.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_terminal_and_leg_settlement.py
```

Expected: at least one engine discounts mixed legs with one maturity DF.

**Step 3: Implement per-leg resolution**

Use terminal timing for one-payoff products. For the accumulator, resolve
each fixing's contractual payment request once before pricing and carry an
aligned `payment_times`/`payment_dfs` array through analytical and MC payoff
code. A pathwise cashflow must be multiplied by its own DF before summing.

If an analytical approximation cannot decompose its payoff into the
contractual legs, declare only `TERMINAL_ONLY` support and raise
`CapabilityError` for non-uniform event settlement.

**Step 4: Run adjacent regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_terminal_and_leg_settlement.py \
  test/test_digital_option_analytical.py \
  test/test_digital_option_mc_engine.py \
  test/test_asian_option_analytical.py \
  test/test_asian_mc_weighted.py \
  test/test_range_accrual_analytical_engine.py \
  test/test_range_accrual_mc_engine.py \
  test/test_accumulator_analytical_engine.py \
  test/test_accumulator_mc_engine.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add quantark/asset/equity/engine/analytical \
  quantark/asset/equity/engine/mc \
  test/test_terminal_and_leg_settlement.py \
  test/test_digital_option_analytical.py \
  test/test_digital_option_mc_engine.py \
  test/test_asian_option_analytical.py \
  test/test_asian_mc_weighted.py \
  test/test_range_accrual_analytical_engine.py \
  test/test_range_accrual_mc_engine.py \
  test/test_accumulator_analytical_engine.py \
  test/test_accumulator_mc_engine.py
git commit -m "feat(settlement): discount terminal and scheduled legs"
```

Before committing, inspect the staged list and remove any unrelated
analytical/MC files accidentally included by the broad directory add.

## Task 8: Implement barrier, touch, and sharkfin settlement in analytical and MC engines

**Files:**

- Modify: `quantark/asset/equity/engine/analytical/barrier_analytical_engine.py`
- Modify:
  `quantark/asset/equity/engine/analytical/double_barrier_option_engine.py`
- Modify:
  `quantark/asset/equity/engine/analytical/one_touch_analytical_engine.py`
- Modify:
  `quantark/asset/equity/engine/analytical/single_sharkfin_option_analytical_engine.py`
- Modify:
  `quantark/asset/equity/engine/analytical/double_sharkfin_option_analytical_engine.py`
- Modify: `quantark/asset/equity/engine/mc/barrier_option_mc_engine.py`
- Modify: `quantark/asset/equity/engine/mc/barrier_vol_mc_engines.py`
- Modify:
  `quantark/asset/equity/engine/mc/single_sharkfin_option_mc_engine.py`
- Modify:
  `quantark/asset/equity/engine/mc/double_sharkfin_option_mc_engine.py`
- Create: `test/test_barrier_family_settlement.py`
- Modify: existing barrier/touch/sharkfin tests

**Step 1: Write payoff-timing truth-table tests**

Parameterize:

| Cashflow behavior | Determination | Payment resolution |
|---|---|---|
| `pay_at_hit=True` / `payment_at_hit=True` | hit node | explicit record or convention from hit |
| expiry-paid rebate | hit determines state, expiry determines amount payment | terminal resolution |
| no-touch terminal payout | expiry | terminal resolution |
| sharkfin terminal participation | expiry | terminal resolution |

Test terminal override isolation, T+N event delay, zero-lag identity, and a
curve with different forward rates across hit dates.

For a first-hit analytical formula that cannot represent a node-dependent
business-day lag, assert early `CapabilityError` rather than an approximate
average delay.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_barrier_family_settlement.py
```

Expected: failure from maturity-wide discounting or missing capability gate.

**Step 3: Implement leg-specific pricing**

For analytical engines:

- terminal legs use terminal timing;
- known discrete hit dates may be summed with their own payment DFs;
- continuous first-hit formulas accept only settlement forms proven
  compatible with the derivation;
- reject arbitrary node-dependent date conventions before formula work.

For MC:

- retain hit index/time per path;
- map each hit index to a pre-resolved event payment DF;
- discount expiry-paid and terminal legs with terminal payment DF;
- sum discounted cashflows path by path.

Do not multiply the completed barrier price by one blanket factor when hit
and terminal legs coexist.

**Step 4: Run regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_barrier_family_settlement.py \
  test/test_barrier_analytical_engine.py \
  test/test_double_barrier_option_engine.py \
  test/test_one_touch_analytical_engine.py \
  test/test_barrier_option_mc_engine.py \
  test/test_single_sharkfin_analytical_engine.py \
  test/test_single_sharkfin_mc_engine.py \
  test/test_double_sharkfin_analytical_engine.py \
  test/test_double_sharkfin_mc_engine.py \
  test/test_barrier_vol_mc_engines.py
```

Expected: PASS.

**Step 5: Commit**

Stage the exact listed engine and test files, then:

```bash
git commit -m "feat(settlement): price barrier-family payment legs"
```

## Task 9: Implement barrier/touch settlement in PDE and QUAD

**Files:**

- Modify: `quantark/asset/equity/engine/pde/barrier_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/double_barrier_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/one_touch_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/double_one_touch_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/barrier_vol_pde_solvers.py`
- Modify: `quantark/asset/equity/engine/quad/quad_adapters.py`
- Modify: `quantark/asset/equity/engine/quad/discrete_quad_engine.py`
- Create: `test/test_barrier_numerical_settlement.py`
- Modify: existing barrier/touch PDE/QUAD tests

**Step 1: Write node-injection tests**

Use small deterministic fixtures to prove:

- a hit rebate inserted at PDE/QUAD node `ti` is multiplied by
  `DF(Tpay_i) / DF(ti)`;
- an expiry-paid rebate is inserted with terminal delay;
- stochastic grids stop at contractual determination times;
- MC/PDE/QUAD agree within existing numerical tolerances.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_barrier_numerical_settlement.py
```

Expected: node value uses determination payment or terminal blanket DF.

**Step 3: Centralize node delay arrays**

Resolve all event timings before grid construction and carry:

```python
event_delay_dfs[i] = event_payment_df[i] / determination_df[i]
terminal_delay_df = terminal_payment_df / terminal_determination_df
```

Use `event_delay_dfs[i] * rebate` when applying an event boundary or
quadrature projection. Reuse `quad_adapters` as the single QUAD integration
seam; remove duplicate local settlement calculations when equivalent.

**Step 4: Run numerical regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_barrier_numerical_settlement.py \
  test/test_barrier_quad_engine.py \
  test/test_one_touch_quad_engine.py \
  test/test_barrier_vol_pde_engines.py \
  test/volmodels/test_barrier_core.py
```

Expected: PASS.

**Step 5: Commit**

Stage exact files and commit:

```bash
git commit -m "feat(settlement): apply barrier node payment delays"
```

## Task 10: Normalize Snowball, Phoenix, KO-reset, and DCN Monte Carlo cashflows

**Files:**

- Modify: `quantark/asset/equity/engine/mc/snowball_mc_engine.py`
- Modify: `quantark/asset/equity/engine/mc/phoenix_mc_engine.py`
- Modify: `quantark/asset/equity/engine/mc/dcn_mc_engine.py`
- Modify: `quantark/asset/equity/engine/mc/snowball_vol_mc_engines.py`
- Modify: `quantark/asset/equity/engine/mc/phoenix_vol_mc_engines.py`
- Modify: `quantark/asset/equity/engine/mc/dcn_vol_mc_engines.py`
- Modify:
  `quantark/asset/equity/engine/mc/autocallable_execution_adapters.py`
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Create: `test/test_autocallable_mc_settlement.py`
- Modify: existing Snowball/Phoenix/KO-reset/DCN MC tests

**Step 1: Write cashflow decomposition tests**

Test:

- Snowball KO redemption uses the triggering observation's payment timing;
- Phoenix coupons can remain pending while the note stays live;
- Phoenix coupon, KO redemption, and terminal loss use different payment DFs;
- KO-reset pre/post-KI KO cashflows preserve their actual event timing;
- DCN explicit coupon/KO/loss payment dates remain authoritative;
- changing payment dates does not change KO/KI/coupon probabilities or path
  evolution;
- discounted leg sum equals reported MC PV under a fixed seed.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_autocallable_mc_settlement.py
```

Expected: at least one existing settlement array is locally derived or one
leg uses an observation/maturity DF instead of payment DF.

**Step 3: Build one timing bundle per valuation**

Resolve observation records once and pass aligned arrays into payoff
kernels:

```python
@dataclass(frozen=True)
class AutocallablePaymentTimings:
    observation_times: np.ndarray
    observation_payment_times: np.ndarray
    observation_payment_dfs: np.ndarray
    terminal: ResolvedPaymentTiming
```

Extend it for Phoenix coupon timing and DCN explicit schedules as required.
Existing `settlement_times` seams should become consumers of this bundle
rather than alternative resolution implementations.

Every payoff kernel should return undiscounted leg amounts plus event indices
or directly discounted leg arrays. Do not apply a final maturity DF after
mixed cashflows have been summed.

**Step 4: Run MC regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_autocallable_mc_settlement.py \
  test/test_snowball_mc_engine.py \
  test/test_phoenix_mc.py \
  test/test_ko_reset_snowball_mc_engine.py \
  test/test_dcn_mc_engine.py \
  test/test_dcn_payoff.py \
  test/test_dcn_vol_mc_engines.py \
  test/test_phoenix_vol_model_engines.py \
  test/test_snowball_vol_model_engines.py
```

Expected: PASS.

**Step 5: Commit**

Stage exact MC, adapter, and test files, then:

```bash
git commit -m "feat(settlement): normalize autocallable MC cashflows"
```

## Task 11: Normalize structured-product PDE and QUAD cashflows

**Files:**

- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py`
- Modify:
  `quantark/asset/equity/engine/pde/ko_reset_snowball_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/dcn_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py`
- Modify: `quantark/asset/equity/engine/pde/phoenix_vol_pde_solvers.py`
- Modify: `quantark/asset/equity/engine/pde/dcn_vol_pde_solvers.py`
- Modify: `quantark/asset/equity/engine/pde/pde_execution_adapters.py`
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py`
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py`
- Modify:
  `quantark/asset/equity/engine/quad/ko_reset_snowball_quad_engine.py`
- Modify: `quantark/asset/equity/engine/quad/quad_adapters.py`
- Create: `test/test_structured_numerical_settlement.py`
- Modify: `test/test_snowball_pde.py`
- Modify: `test/test_snowball_quad_engine.py`
- Modify: `test/test_phoenix_pde.py`
- Modify: `test/test_phoenix_quad.py`
- Modify: `test/test_ko_reset_snowball_pde.py`
- Modify: `test/test_ko_reset_snowball_quad_engine.py`
- Modify: `test/test_dcn_pde_solver.py`
- Modify: `test/test_dcn_vol_pde_solvers.py`

**Step 1: Write cross-engine reconciliation tests**

Use compact grids and products with two distinct payment dates. Assert:

- each event injection uses its node-specific delay ratio;
- terminal loss/redemption uses terminal timing;
- QUAD's inserted settlement times do not extend stochastic transition
  dynamics;
- probabilities are unchanged when only payment timing changes;
- MC/PDE/QUAD cashflow PVs reconcile within their existing tolerances;
- unsupported LV/Heston/SLV QUAD paths raise `CapabilityError`.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_structured_numerical_settlement.py
```

Expected: failure from local timing logic or maturity-wide discounting.

**Step 3: Apply timing bundles at every cashflow injection**

PDE:

- keep the state grid through determination only;
- multiply KO/coupon/redemption/loss values by their delay ratio at the node;
- preserve existing probability-mass propagation.

QUAD:

- resolve all determination/payment arrays in `quad_adapters`;
- use the arrays in Snowball, Phoenix, and KO-reset recursions;
- remove equivalent per-engine `_ko_discount`/settlement calculations only
  after their existing tests are ported to the shared helper.

**Step 4: Run structured numerical regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_structured_numerical_settlement.py \
  test/test_snowball_pde.py \
  test/test_snowball_quad_engine.py \
  test/test_phoenix_pde.py \
  test/test_phoenix_quad.py \
  test/test_ko_reset_snowball_pde.py \
  test/test_ko_reset_snowball_quad_engine.py \
  test/test_dcn_pde_solver.py \
  test/test_dcn_vol_pde_solvers.py
```

Expected: PASS.

**Step 5: Commit**

Stage exact PDE/QUAD and test files, then:

```bash
git commit -m "feat(settlement): apply structured node payment delays"
```

## Task 12: Implement node-specific delayed settlement for American exercise

**Files:**

- Modify: `quantark/asset/equity/engine/mc/american_option_mc_engine.py`
- Modify: `quantark/asset/equity/engine/pde/american_pde_solver.py`
- Modify: `quantark/asset/equity/engine/analytical/american_option_engine.py`
- Create: `test/test_american_option_settlement.py`
- Modify: `test/test_american_option_mc_engine.py`
- Modify: `test/test_american_option_analytical.py`

**Step 1: Write exercise-obstacle tests**

Test:

- zero lag reproduces current American values;
- for each candidate exercise node `ti`, obstacle is
  `intrinsic(S_i) * DF(Tpay_i) / DF(ti)`;
- PDE and LSMC use node-specific delay, not a terminal factor;
- delayed settlement changes early-exercise boundary in the expected
  direction for a deep-in-the-money put;
- date-based business-day settlement uses actual exercise-node dates;
- numeric American products accept only year-fraction settlement;
- the analytical approximation rejects non-zero settlement unless its
  derivation is explicitly implemented.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_american_option_settlement.py
```

Expected: American engines compare continuation against immediate intrinsic
or silently use a terminal payment factor.

**Step 3: Build an exercise-node delay vector**

For LSMC:

```python
exercise_delay_dfs = resolve_exercise_node_delay_dfs(
    product, exercise_times, exercise_dates, pricing_env
)
exercise_values = intrinsic_values * exercise_delay_dfs[:, None]
```

Regress continuation as before; only replace the exercise payoff in the
comparison and selected cashflow. Discount selected delayed-settlement
exercise values consistently to valuation.

For PDE, use the same delay vector when applying
`max(continuation, exercise_value)` at each backward node. The terminal
payoff is the terminal node's delayed exercise value.

For date-based rules, derive node dates from an authoritative calendar grid
anchored to real exercise dates. Do not construct dates with `ti * 365`.

**Step 4: Run American regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_american_option_settlement.py \
  test/test_american_option_mc_engine.py \
  test/test_american_option_analytical.py \
  quantark/asset/equity/engine/validation/script/boundary_check_american_pde.py
```

If the validation script is not a pytest file, execute it separately with
the repository venv and record its numerical summary.

**Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/american_option_mc_engine.py \
  quantark/asset/equity/engine/pde/american_pde_solver.py \
  quantark/asset/equity/engine/analytical/american_option_engine.py \
  test/test_american_option_settlement.py \
  test/test_american_option_mc_engine.py \
  test/test_american_option_analytical.py
git commit -m "feat(settlement): delay American exercise payments"
```

## Task 13: Make event statistics settlement-aware and PV-reconciling

**Files:**

- Modify: `quantark/asset/equity/engine/event_stats.py`
- Modify: all engine `calculate_event_stats` implementations
- Modify: `quantark/cashleg/event_distribution.py`
- Modify: `quantark/cashleg/` consumers that discount events
- Modify: `test/test_event_stats_api.py`
- Modify: `test/test_quad_pde_event_stats.py`
- Modify: `test/test_cashleg/test_phoenix_pde_event_stats.py`
- Modify: `test/test_cashleg/test_phoenix_quad_event_stats.py`
- Create: `test/test_settlement_event_stats_reconciliation.py`

**Step 1: Write schema and reconciliation tests**

Require aligned outputs:

```python
assert len(stats.determination_times) == len(stats.payment_times)
assert len(stats.payment_times) == len(stats.expected_undiscounted_cashflows)
assert len(stats.expected_discounted_cashflows) == len(stats.payment_times)
assert stats.pv == pytest.approx(
    stats.expected_discounted_cashflows.sum(),
    abs=reconciliation_tolerance,
)
```

Test that expected life and event probabilities remain functions of
determination times while cashflow attribution uses payment times. Cover
Snowball, Phoenix, KO-reset, and DCN.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_settlement_event_stats_reconciliation.py
```

Expected: new aligned fields are absent.

**Step 3: Extend result objects additively**

Add defaulted arrays to preserve existing constructors:

```python
determination_times: np.ndarray = field(default_factory=lambda: np.array([]))
payment_times: np.ndarray = field(default_factory=lambda: np.array([]))
expected_undiscounted_cashflows: np.ndarray = field(
    default_factory=lambda: np.array([])
)
expected_discounted_cashflows: np.ndarray = field(
    default_factory=lambda: np.array([])
)
determination_dates: Optional[tuple[datetime, ...]] = None
payment_dates: Optional[tuple[datetime, ...]] = None
```

Keep legacy KO/coupon fields during migration, but calculate them from the
same leg arrays. Update `EventDistribution` to carry payment timing where
cash-leg consumers need it; `event_times` remains determination timing.

**Step 4: Run event-stat regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_settlement_event_stats_reconciliation.py \
  test/test_event_stats_api.py \
  test/test_event_stats_pruning.py \
  test/test_quad_pde_event_stats.py \
  test/test_cashleg/test_phoenix_pde_event_stats.py \
  test/test_cashleg/test_phoenix_quad_event_stats.py
```

Expected: PASS.

**Step 5: Commit**

Stage exact event-stat, consumer, engine, and test files, then:

```bash
git commit -m "feat(settlement): expose payment-aware event stats"
```

## Task 14: Separate lifecycle determination, pending payment, and paid cash

**Files:**

- Modify: `quantark/asset/equity/lifecycle/autocallable.py`
- Modify: `quantark/asset/equity/lifecycle/barrier.py`
- Modify: `quantark/asset/equity/lifecycle/manager.py`
- Modify: `quantark/asset/equity/lifecycle/events.py`
- Modify: `quantark/backtest/otc/state.py`
- Modify: `quantark/backtest/otc/_replay.py`
- Modify: `quantark/backtest/otc/engine.py`
- Modify: `quantark/backtest/otc/book_engine.py`
- Modify: `quantark/backtest/equity/engine.py`
- Modify: `quantark/backtest/equity/state.py`
- Modify: `quantark/dynamicscenario/lifecycle_manager.py`
- Modify: `quantark/dynamicscenario/engine.py`
- Modify: `quantark/portfolio/equity/position.py`
- Create: `test/test_settlement_lifecycle_timeline.py`
- Modify: `test/test_backtest_lifecycle.py`
- Modify: `test/test_dynamic_scenario_lifecycle.py`

**Step 1: Write the full timeline test**

For a KO/coupon determined on day D and paid on D+2:

1. before D: position has contingent PV;
2. on D after authoritative fixing: event registers immutable cashflow;
3. D < valuation < D+2: contingent leg is gone, ledger pending PV remains;
4. on D+2: pending derivative PV is zero and paid cash increases once;
5. after D+2: no derivative/pending value; paid cash remains once.

Also test a Phoenix with a pending coupon and live continuation value, and
duplicate replay of the same event.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_settlement_lifecycle_timeline.py
```

Expected: the current manager books realized cash immediately and removes a
terminated position before payment.

**Step 3: Register cashflows at determination and book only when paid**

Trackers must:

- resolve the event's payment timing;
- emit/register a `RealizedCashflow` with stable ID;
- mark the contingent claim determined;
- retain pending ledger state through payment.

Remove date fabrication from `_scheduled_records` and
`_maturity_settlement_date`. Date products use authoritative schedule dates;
numeric products use numeric `ValuationPoint` and time-shift behavior.

Change `PortfolioLifecycleManager` from one `realized_cash` accumulator to a
ledger-backed split:

```python
pending_receivable_pv = ledger.pending_pv(as_of, env)
paid_cash = ledger.paid_total(as_of)
portfolio_value = live_positions_mtm + pending_receivable_pv + paid_cash
```

Do not keep a paid cashflow in both derivative PV and cash balance.

**Step 4: Run lifecycle/consumer regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_settlement_lifecycle_timeline.py \
  test/test_backtest_lifecycle.py \
  test/test_dynamic_scenario_lifecycle.py \
  test/test_equity_lifecycle_trackers.py \
  test/test_snowball_lifecycle_ki.py \
  test/test_equity_position_trade_greeks.py
```

Expected: PASS with continuous portfolio value across determination and
payment.

**Step 5: Commit**

Stage exact lifecycle, backtest, scenario, position, and test files, then:

```bash
git commit -m "feat(lifecycle): carry receivables through payment"
```

## Task 15: Propagate lifecycle state through execution, portfolio, and repricing

**Files:**

- Modify: `quantark/execution/greeks.py`
- Modify: `quantark/execution/legacy_adapter.py`
- Modify:
  `quantark/asset/equity/engine/mc/autocallable_execution_adapters.py`
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Modify: `quantark/asset/equity/engine/pde/pde_execution_adapters.py`
- Modify: `quantark/portfolio/equity/position.py`
- Modify: `quantark/backtest/otc/_replay.py`
- Modify: `quantark/backtest/otc/engine.py`
- Modify: `quantark/backtest/equity/engine.py`
- Modify: `quantark/dynamicscenario/engine.py`
- Modify: `quantark/asset/equity/engine/base_engine.py`
- Create: `test/execution/test_settlement_state_propagation.py`
- Modify: `test/test_equity_position_trade_greeks.py`
- Modify: `test/execution/test_greek_bump_cells.py`

**Step 1: Write propagation spies**

Use a spy engine and immutable state object to assert the exact same lifecycle
state reaches:

- direct `Position.price`;
- `price_with_events`;
- execution kernel/legacy adapter;
- batch valuation;
- spot/vol/rate bump repricing;
- theta/time-shift repricing;
- scenario/backtest repricing.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/execution/test_settlement_state_propagation.py
```

Expected: at least one adapter drops the keyword.

**Step 3: Thread state without hidden mutation**

Add lifecycle state to request/context structures only where an equity option
valuation can carry it. Preserve object identity for an immutable snapshot;
do not let bump code mutate the original ledger. Ensure `price_with_events`
uses the state both for NPV and event distribution.

**Step 4: Run execution regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/execution/test_settlement_state_propagation.py \
  test/execution/test_greek_bump_cells.py \
  test/execution/test_dcn_batch_adapter.py \
  test/test_equity_position_trade_greeks.py
```

Expected: PASS.

**Step 5: Commit**

Stage exact execution/position/repricer files and tests, then:

```bash
git commit -m "feat(settlement): propagate lifecycle valuation state"
```

## Task 16: Make all Greeks and risk attribution settlement-aware

**Files:**

- Modify: `quantark/asset/equity/riskmeasures/greeks_calculator.py`
- Modify: native Greek methods in affected analytical engines
- Modify: `quantark/asset/equity/riskmeasures/bucketed_greeks.py`
- Modify: `quantark/asset/equity/engine/localvol_greeks.py`
- Modify: `quantark/portfolio/equity/position.py`
- Create: `test/test_settlement_greeks.py`
- Modify: `test/test_greeks_theta_schedule.py`
- Modify: `test/test_bucketed_greeks_api.py`
- Modify: `test/test_point_greeks.py`

**Step 1: Write risk identity tests**

For a deterministic settlement factor independent of spot and vol:

```python
assert delayed["delta"] == pytest.approx(immediate["delta"] * delay_df)
assert delayed["gamma"] == pytest.approx(immediate["gamma"] * delay_df)
assert delayed["vega"] == pytest.approx(immediate["vega"] * delay_df)
```

Also assert:

- analytical Greeks match bump-and-reprice;
- rho includes both determination and payment curve exposure;
- bucketed rate risk includes settlement-delay curve pillars;
- date theta keeps contractual dates fixed while valuation date advances;
- numeric theta reduces determination and payment times together;
- pending known cash has zero delta/gamma/vega/dividend risk;
- pending known cash retains rate risk and calendar theta;
- theta crossing payment transfers value to paid cash without a P&L jump.

**Step 2: Verify the red state**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_settlement_greeks.py
```

Expected: native or numerical risk drops settlement state or bumps the wrong
time.

**Step 3: Route every reprice through settlement resolution**

Native analytical Greeks may multiply determination-based spot/vol Greeks by
`delay_df`. Rho and theta must differentiate/reprice the full curve-exact
payment timing; do not apply the spot/vol shortcut.

Numerical bump code must preserve product payment terms and lifecycle state.
For date theta, advance `pricing_env.valuation_date` but leave contractual
dates unchanged. For numeric theta, use existing `time_shift` on both
determination and payment times.

**Step 4: Run risk regressions**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_settlement_greeks.py \
  test/test_greeks_theta_schedule.py \
  test/test_bucketed_greeks_api.py \
  test/test_point_greeks.py \
  test/test_equity_position_trade_greeks.py \
  test/test_spot_greeks_curve.py
```

Expected: PASS.

**Step 5: Commit**

Stage exact risk, engine, position, and test files, then:

```bash
git commit -m "feat(settlement): include payment timing in equity risk"
```

## Task 17: Run the complete matrix, performance gates, and publish usage documentation

**Files:**

- Create: `docs/equity-option-settlement.md`
- Modify: package/API documentation where option constructor examples live
- Modify: `CHANGELOG.md`
- Create: `test/test_equity_settlement_full_matrix.py`
- Modify: any test manifest or marker configuration required by the repo

**Step 1: Add the final support-matrix test**

Parameterize the current supported product/model/engine cells. For each cell,
require exactly one of:

1. zero/non-zero settlement prices correctly and passes its family identity;
2. the request fails before numerical work with `CapabilityError` and an
   actionable message.

The matrix must explicitly include:

- dynamics: BSM, Local Vol, Heston, SLV;
- engines: analytical where present, MC, PDE, QUAD;
- product families: European, American, digital, Asian, barrier/touch,
  sharkfin, accumulator/range, Snowball, Phoenix, KO-reset, DCN;
- timing forms: zero lag, explicit date, explicit time, business-day lag,
  calendar-day lag, year-fraction lag;
- lifecycle states: live, determined/pending, paid.

**Step 2: Verify the full-matrix test**

Run:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_equity_settlement_full_matrix.py
```

Expected: PASS.

**Step 3: Write user-facing documentation**

Document:

- terminal `settlement_date`;
- `SettlementConvention` with date and numeric examples;
- per-observation `settlement_date` and `settlement_time`;
- precedence;
- lifecycle-state use between determination and payment;
- capability errors;
- zero-lag backward compatibility.

Include examples:

```python
option = EuropeanVanillaOption(
    strike=100.0,
    exercise_date=datetime(2027, 6, 18),
    settlement_date=datetime(2027, 6, 22),
    option_type=OptionType.CALL,
)
```

and:

```python
option = EuropeanVanillaOption(
    strike=100.0,
    maturity=1.0,
    settlement_convention=SettlementConvention(
        lag=2 / 365,
        lag_unit=SettlementLagUnit.YEAR_FRACTION,
    ),
    option_type=OptionType.CALL,
)
```

Explain why a time-only product cannot use a business-day lag.

**Step 4: Run broad regressions in bounded groups**

Run the settlement and product groups first:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_equity_settlement_resolver.py \
  test/test_equity_option_settlement_contract.py \
  test/test_lifecycle_cashflow_ledger.py \
  test/test_engine_settlement_capabilities.py \
  test/test_european_settlement_matrix.py \
  test/test_vol_model_terminal_settlement.py \
  test/test_terminal_and_leg_settlement.py \
  test/test_barrier_family_settlement.py \
  test/test_barrier_numerical_settlement.py \
  test/test_autocallable_mc_settlement.py \
  test/test_structured_numerical_settlement.py \
  test/test_american_option_settlement.py \
  test/test_settlement_event_stats_reconciliation.py \
  test/test_settlement_lifecycle_timeline.py \
  test/test_settlement_greeks.py \
  test/test_equity_settlement_full_matrix.py
```

Then run the existing equity option suite:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider \
  test/test_*option*.py \
  test/test_*barrier*.py \
  test/test_*snowball*.py \
  test/test_*phoenix*.py \
  test/test_*dcn*.py \
  test/test_*event_stats*.py \
  test/test_*greek*.py
```

Finally run the repository's CI commands:

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider -m "not slow"
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  -m pytest -q -o addopts='' -p no:cacheprovider
```

If the all-tests command exceeds the local time budget, run the same
directory/file shards with the identical marker configuration and record
every shard.

**Step 5: Run performance comparisons**

Benchmark representative zero-lag and delayed cases for:

- European MC/PDE/QUAD;
- barrier MC/PDE;
- Snowball MC/PDE/QUAD;
- Phoenix MC/PDE/QUAD.

Acceptance:

- zero-lag resolver overhead is outside inner path/grid loops;
- payment timing arrays are resolved once per price call;
- no material path/grid size increase;
- runtime regression is within the repository's established benchmark
  tolerance, or is explained and approved.

**Step 6: Audit forbidden fallbacks**

Run:

```bash
rg -n "maturity \\* 365|observation_time \\* 365|except Exception" \
  quantark/asset/equity quantark/backtest quantark/dynamicscenario
rg -n "settlement_date|settlement_time|settlement_convention" \
  quantark/asset/equity/engine
git diff --check
```

Inspect every match. There must be no fabricated date, blanket exception
fallback, or engine-local silent zero-lag fallback.

**Step 7: Commit documentation and the final matrix**

```bash
git add docs/equity-option-settlement.md \
  test/test_equity_settlement_full_matrix.py
git add CHANGELOG.md  # only if updated
git commit -m "docs(settlement): publish equity option payment support"
```

## Final acceptance checklist

- [ ] Zero lag reproduces all existing supported engine prices.
- [ ] European delayed payoff matches the curve-exact identity.
- [ ] No stochastic process evolves beyond determination.
- [ ] Every mixed cashflow is discounted at its own payment time.
- [ ] American MC/PDE use a node-specific delayed exercise obstacle.
- [ ] Pending receivables are valued until payment and booked once.
- [ ] Native and numerical Greeks include settlement timing.
- [ ] Event-stat discounted cashflows reconcile to NPV.
- [ ] Every current BSM/LV/Heston/SLV by MC/PDE/QUAD cell either works or
      raises an explicit capability error.
- [ ] No date is fabricated from a numeric year fraction.
- [ ] No broad exception silently falls back to determination payment.
- [ ] Full focused and repository regression commands are recorded.
- [ ] Only settlement-related files are committed from this worktree.
