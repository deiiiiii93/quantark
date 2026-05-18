# Equity Options Cash Legs — Design Spec

**Date:** 2026-05-18
**Status:** Approved design, pending implementation plan
**Scope:** All equity options (vanilla, exotic, autocallable)

---

## 1. Motivation

In practical OTC equity-option trading, contracts carry cash terms beyond the option payoff itself:

- **Premium** — upfront or backend lump-sum cash paid at trade inception or maturity
- **Extra interest on notional or cash margin** — periodic interest paid by one counterparty on the trade's notional or on posted cash collateral
- **Extra rebate on notional or cash margin** — additional rebate accruing alongside the trade

Some of these cash terms are deterministic in both amount and timing (premium). Others are time-dependent: an accruing rebate is shortened if the product knocks out early; a fixed-amount rebate may or may not be paid depending on which event terminates the contract.

To expose the **full risk profile of the entire trade** — not just the option payoff — these cash terms must be priced together with the option, and Greeks must reflect the combined exposure.

Today the codebase has partial machinery for this inside autocallable products (`AccrualConfig`, KO coupons, `is_annualized_*` flags in `asset/equity/product/option/snowball_config.py`), but it is bundled into each product class. There is no general, composable cash-leg abstraction usable across all option types and all engine families.

---

## 2. Goals & Non-Goals

### Goals

- Composable cash-leg primitives usable on **any equity product** and **any engine family** (Analytical, MC, PDE, Quad).
- **Near-zero pricing-time overhead** — cash legs valued from a byproduct of the existing pricing pass, not a second pass.
- **Backward compatibility**: positions without cash legs behave exactly as today. Existing engine API (`engine.price(product, env) → float`) is unchanged.
- Clean integration with the existing `EquityPosition` / `EquityPortfolio` model, so backtest, stress test, and dynamic scenario can opt in incrementally.
- Trade-level NPV and Greeks include both product and cash legs.

### Non-Goals (explicitly out of scope)

- **MTM-dependent margin** (margin balance that varies with trade MTM — recursive valuation problem)
- **Cross-currency cash legs** (single trade currency assumed)
- **Path-dependent cash legs beyond KO-truncation** (e.g., amount depending on average spot)
- **Fixed-income equivalent** (`portfolio/fi/position.py`) — same pattern, future work item
- **Analytical leg Greeks** — v1 uses bump-and-reprice; analytical leg-level Greeks deferred until profiling shows a need

---

## 3. Architecture Overview

### 3.1 Approach: Engine-emitted EventDistribution + standalone CashLeg valuator

Each pricing engine, as a byproduct of its main pricing pass, emits a small standardized `EventDistribution` object describing the probability and timing of termination/coupon events (KO, KI, maturity outcomes). A separate, engine-agnostic `cashleg/` module values cash legs purely from this distribution plus the `PricingEnvironment`. Orchestration lives on the existing `EquityPosition`.

This decouples:

- **Engine math** (unchanged for product NPV; adds a small standardized byproduct)
- **Leg math** (knows only `EventDistribution` and discounting; engine-agnostic)
- **Orchestration** (existing `EquityPosition` — no new wrapper type)

### 3.2 Why this architecture

- **Compatibility across engine families**: every engine type can naturally produce the event distribution as a byproduct (MC: per-path KO times; PDE: forward density; Quad: survival from recursion; Analytical: closed-form first-passage). The interface is the contract; the implementation varies.
- **Near-zero overhead**: in MC/Quad/Analytical, the byproduct is free or near-free. PDE pays ~20% extra for an opt-in forward density pass, gated on whether any leg actually needs it.
- **No parallel `Trade` type**: `EquityPosition` already plays this role today; extending it preserves backward compatibility for `portfolio/`, `backtest/`, `stresstest/`, `dynamicscenario/`.
- **Adding new leg types requires zero engine changes** — only a new subclass of `CashLeg`.

### 3.3 Module layout

```
cashleg/
├── __init__.py
├── base.py                  # CashLeg ABC, LegDirection enum
├── deterministic_leg.py     # DeterministicLeg (front/back premium)
├── accrual_leg.py           # AccrualLeg (KO-truncated streams)
├── fixed_payoff_leg.py      # FixedPayoffLeg (event-conditional fixed amounts)
├── base_amount.py           # BaseAmount + BaseAmountMode (ABSOLUTE | NOTIONAL_FRACTION | MARGIN_FRACTION)
├── leg_schedule.py          # LegSchedule (period boundaries, payment dates)
├── event_distribution.py    # EventDistribution dataclass + EventType enum + PricingResult
├── leg_valuator.py          # value_leg(leg, event_dist, env, notional) → signed PV
└── CLAUDE.md
```

Engine API extension in `asset/equity/engine/base_engine.py`:

```python
@dataclass(frozen=True)
class PricingResult:
    npv: float
    event_distribution: Optional[EventDistribution]

class BaseEngine:
    # existing: def price(self, product, env) -> float  (unchanged)

    def price_with_events(self, product, env,
                          emit_distribution: bool = True) -> PricingResult:
        """Default: wrap price() with EventDistribution.trivial(maturity).
        Engines that can cheaply emit richer event timing override this.

        emit_distribution: when False, callers signal they don't need event timing
        (e.g., no legs require it); expensive emission paths (PDE forward density)
        may skip. The default of True preserves correctness for callers that don't
        know about this flag."""
        npv = self.price(product, env)
        return PricingResult(npv=npv,
                             event_distribution=EventDistribution.trivial(product.get_maturity()))
```

Position integration in `portfolio/equity/position.py`:

```python
@dataclass
class EquityPosition:
    # existing fields unchanged
    product: BaseEquityProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: BaseEngine
    entry_timestamp: datetime
    cash_legs: list[CashLeg] = field(default_factory=list)   # NEW — defaults empty
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # existing methods (get_market_value, get_greeks, ...) unchanged

    def get_trade_value(self, env) -> float: ...
    def get_trade_value_breakdown(self, env) -> TradeValueBreakdown: ...
    def get_trade_greeks(self, env, calc) -> dict[str, float]: ...
```

---

## 4. Core Data Model

### 4.1 `EventDistribution`

```python
class EventType(Enum):
    KO = "knock_out"
    KI = "knock_in"
    COUPON = "coupon"                  # per-observation coupon event (Phoenix-style)
    MATURITY_NO_KO = "maturity_no_ko"  # reached maturity without KO
    MATURITY_WITH_KI = "maturity_with_ki"  # reached maturity, KI had triggered

@dataclass(frozen=True)
class EventDistribution:
    event_times: np.ndarray                         # shape (N,), year fractions
    event_dates: Optional[list[datetime]]           # parallel calendar dates if engine tracks
    probabilities: dict[EventType, np.ndarray | float]
                                                    # per-time PMF for KO/KI/COUPON, scalar for MATURITY_*
    survival_probability: np.ndarray                # shape (N+1,); survival[i] = P(alive entering obs i); [0]=1.0
    mc_ko_times: Optional[np.ndarray] = None        # per-path KO time index, MC-only

    @classmethod
    def trivial(cls, maturity: float) -> "EventDistribution":
        """Single-event distribution: probability mass at maturity, no KO. Used for vanilla products."""

    def survival_at(self, t: float) -> float:
        """Linear interpolation of survival_probability at arbitrary year fraction."""
```

**Invariants** (engine post-conditions, checked in `__post_init__`):

- `sum(probabilities[KO]) + sum(probabilities[KI]) + probabilities.get(MATURITY_NO_KO, 0) + probabilities.get(MATURITY_WITH_KI, 0) ≈ 1.0` (tolerance `Tolerance.PROBABILITY = 1e-6`)
- `survival_probability` is monotone non-increasing
- `survival_probability[0] == 1.0`
- `len(survival_probability) == len(event_times) + 1`

Violations raise `NumericalError`.

### 4.2 `PricingResult`

```python
@dataclass(frozen=True)
class PricingResult:
    npv: float
    event_distribution: Optional[EventDistribution]
```

Engines without an `EventDistribution`-capable override return `PricingResult(npv, EventDistribution.trivial(T))`.

### 4.3 `LegDirection`

```python
class LegDirection(Enum):
    BUYER_RECEIVES = +1
    BUYER_PAYS = -1
```

All leg PVs are returned signed from the buyer's (position holder's) perspective. Combined with `EquityPosition.quantity` (positive long, negative short), this yields correct portfolio-level signs.

### 4.4 `BaseAmount`

```python
class BaseAmountMode(Enum):
    ABSOLUTE = "absolute"
    NOTIONAL_FRACTION = "notional_fraction"
    MARGIN_FRACTION = "margin_fraction"

@dataclass(frozen=True)
class BaseAmount:
    value: float                       # absolute amount, or fraction (in [0, 1])
    mode: BaseAmountMode
    margin_rate: float = 0.0           # only used when mode=MARGIN_FRACTION

    def resolve(self, position_notional: float) -> float:
        if self.mode is BaseAmountMode.ABSOLUTE:
            return self.value
        if self.mode is BaseAmountMode.NOTIONAL_FRACTION:
            return self.value * position_notional
        if self.mode is BaseAmountMode.MARGIN_FRACTION:
            return self.value * self.margin_rate * position_notional
```

`position_notional` is supplied by `EquityPosition.get_actual_notional(env)`, the existing method.

### 4.5 `LegSchedule`

```python
@dataclass(frozen=True)
class LegSchedule:
    period_starts: np.ndarray          # shape (N,), year fractions
    period_ends:   np.ndarray          # shape (N,), year fractions; period_ends[i] >= period_starts[i]
    payment_times: np.ndarray          # shape (N,); when each period's cash flow is paid
    period_start_dates: Optional[list[datetime]]
    period_end_dates:   Optional[list[datetime]]
    payment_dates:      Optional[list[datetime]]
```

The schedule is **independent of the product's observation schedule**. The leg valuator interpolates `event_distribution.survival_probability` at the leg's period boundaries.

---

## 5. CashLeg Type Hierarchy

### 5.1 `CashLeg` ABC

```python
@dataclass(frozen=True)
class CashLeg(ABC):
    direction: LegDirection
    name: Optional[str] = None              # human-readable label, e.g. "Front Premium"
    leg_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @abstractmethod
    def value(self, event_dist: EventDistribution, env: PricingEnvironment,
              position_notional: float) -> float:
        """Return signed PV from buyer's perspective."""

    def requires_event_distribution(self) -> bool:
        """If False, leg can be valued from any engine's trivial EventDistribution."""
        return True
```

### 5.2 `DeterministicLeg`

```python
@dataclass(frozen=True)
class DeterministicLeg(CashLeg):
    amount: float                            # absolute, in trade currency
    payment_time: float                      # year fraction (0.0 = upfront)
```

**Valuation:**

```
PV = sign(direction) × amount × DF(payment_time)
```

- Independent of `event_dist`.
- Use for front premium (`payment_time=0`), backend premium paid unconditionally, fixed fees.
- For backend premium that is canceled by KO, use `FixedPayoffLeg(trigger=AT_MATURITY_ANY)` instead.

`requires_event_distribution()` returns `False` — works on every engine.

### 5.3 `AccrualLeg`

```python
class PaymentConvention(Enum):
    AT_PERIOD_END = "at_period_end"          # pay each period's accrual at period end
    AT_KO = "at_ko"                          # accumulate; pay full accrued amount at KO (or maturity)
    AT_MATURITY = "at_maturity"              # accumulate all; pay at maturity

class KOBehavior(Enum):
    TRUNCATE_AT_KO = "truncate_at_ko"        # cease accruing at KO
    PAY_FULL_SCHEDULE = "pay_full_schedule"  # KO does not affect accrual

class SurvivalBasis(Enum):
    ENTER_PERIOD = "enter_period"            # pay full period if alive entering it
    COMPLETE_PERIOD = "complete_period"      # pay only fully completed periods (conservative)

@dataclass(frozen=True)
class AccrualLeg(CashLeg):
    rate: float                              # annualized rate (e.g., 0.02 for 2%)
    base: BaseAmount
    schedule: LegSchedule
    day_count: DayCountConvention
    payment_convention: PaymentConvention = PaymentConvention.AT_PERIOD_END
    ko_behavior: KOBehavior = KOBehavior.TRUNCATE_AT_KO
    survival_basis: SurvivalBasis = SurvivalBasis.ENTER_PERIOD
```

**Valuation (`TRUNCATE_AT_KO`, `AT_PERIOD_END`, `ENTER_PERIOD` — the common case):**

```
B = base.resolve(position_notional)
PV = sign × Σ_i  rate × B × dcf_i × survival_factor_i × DF(payment_time_i)

where
  dcf_i             = day_count.fraction(period_starts[i], period_ends[i])
  survival_factor_i = event_dist.survival_at(period_starts[i])     # ENTER_PERIOD
                    = event_dist.survival_at(period_ends[i])       # COMPLETE_PERIOD
  payment_time_i    = schedule.payment_times[i]
```

**Variants:**

- `payment_convention=AT_KO`: amounts accumulate; PV uses `event_dist.probabilities[KO]` per obs date and pays accrued sum at the KO time.
- `ko_behavior=PAY_FULL_SCHEDULE`: drop `survival_factor_i`; PV reduces to a deterministic annuity. Works on any engine.

### 5.4 `FixedPayoffLeg`

```python
class PaymentTrigger(Enum):
    AT_KO = "at_ko"                          # paid at the KO event date on KO paths
    AT_KI = "at_ki"                          # paid at the KI event date on KI paths
    AT_MATURITY_NO_KO = "at_maturity_no_ko"  # paid at maturity on paths that survived without KI
    AT_MATURITY_WITH_KI = "at_maturity_with_ki"  # paid at maturity on paths that reached maturity with KI triggered
    AT_MATURITY_ANY = "at_maturity_any"      # paid at maturity on any path that did NOT KO
                                             # (equivalent to AT_MATURITY_NO_KO + AT_MATURITY_WITH_KI)

@dataclass(frozen=True)
class FixedPayoffLeg(CashLeg):
    amount: float                            # absolute fixed amount
    trigger: PaymentTrigger
    payment_offset_days: int = 0             # T+N settlement delay from trigger event
```

**Trigger semantics — "backend premium that is canceled by KO"** corresponds to `trigger=AT_MATURITY_ANY`: paid only on paths that reach maturity (i.e., never KO'd), regardless of whether KI triggered. For a backend premium that is also canceled by KI, use `AT_MATURITY_NO_KO`.

**Valuation:**

```
PV = sign × amount × Σ_i  P(trigger at obs i) × DF(t_i + offset_year_fraction)
```

Where `P(trigger at obs i)` is read from `event_dist.probabilities[trigger_event_type]`.

`AT_MATURITY_NO_KO`, `AT_MATURITY_WITH_KI`, `AT_MATURITY_ANY` are scalars over the maturity event(s).

---

## 6. Engine Extensions

### 6.1 Default behavior (`BaseEngine.price_with_events`)

Any engine that does not override returns:

```python
def price_with_events(self, product, env, emit_distribution: bool = True):
    return PricingResult(
        npv=self.price(product, env),
        event_distribution=EventDistribution.trivial(product.get_maturity()),
    )
```

This means: existing engines work immediately for `DeterministicLeg`s and for `AccrualLeg`s with `ko_behavior=PAY_FULL_SCHEDULE`. KO-sensitive legs (`AccrualLeg` with `TRUNCATE_AT_KO`, `FixedPayoffLeg`) emit a warning when paired with a non-overriding engine on a KO-capable product (see §9.2).

### 6.2 Per-engine override summary

| Engine family | Override needed? | Emission mechanism | Cost overhead |
|---|---|---|---|
| Analytical — vanilla (BlackScholes, American, Asian) | No | Trivial default | 0% |
| Analytical — barrier / one-touch / range accrual | Yes | Closed-form first-passage CDF over obs dates | <1% |
| Monte Carlo (all) | Yes | Record per-path KO time during existing simulation; aggregate to PMF + survival | <1% (one int array per path) |
| PDE (Snowball / Phoenix / KO-Reset / Barrier) | Yes | Additional forward-density pass on same grid; emits density at each obs date | ~20% — opt-in via `emit_distribution` flag |
| Quadrature (Snowball / Phoenix / KO-Reset / European) | Yes | Survival probabilities are already computed in the recursion; expose them | 0% |

### 6.3 Cost-control flag

```python
def price_with_events(self, product, env, emit_distribution: bool = True) -> PricingResult:
    ...
```

`EquityPosition.get_trade_value` passes `emit_distribution=any(leg.requires_event_distribution() for leg in self.cash_legs)`. If no leg needs the distribution, the PDE engine skips its forward pass — zero overhead.

### 6.4 Engine post-conditions

Engines overriding `price_with_events` must satisfy the invariants in §4.1. The `EventDistribution.__post_init__` performs the checks; failures raise `NumericalError`.

---

## 7. Orchestration & Position Integration

### 7.1 `EquityPosition` extension

```python
# portfolio/equity/position.py
@dataclass
class EquityPosition:
    product: BaseEquityProduct
    quantity: float
    entry_price: float
    underlying: str
    engine: BaseEngine
    entry_timestamp: datetime
    cash_legs: list[CashLeg] = field(default_factory=list)   # NEW
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # --- existing methods unchanged ---
    def get_market_value(self, env) -> float:
        """Product-only market value × quantity. Unchanged for backward compatibility."""
        return self.engine.price(self.product, env) * self.quantity

    def get_greeks(self, env, calc, use_analytical=True) -> dict[str, float]:
        """Product-only Greeks × quantity. Unchanged."""
        ...

    # --- new methods ---
    def get_trade_value(self, env) -> float:
        """Full trade NPV: product + cash legs, scaled by quantity. Buyer's perspective."""
        needs_dist = any(leg.requires_event_distribution() for leg in self.cash_legs)
        if self.cash_legs:
            result = self.engine.price_with_events(self.product, env,
                                                   emit_distribution=needs_dist)
        else:
            return self.get_market_value(env)
        notional = self.get_actual_notional(env)
        leg_pv = sum(leg.value(result.event_distribution, env, notional)
                     for leg in self.cash_legs)
        return (result.npv + leg_pv) * self.quantity

    def get_trade_value_breakdown(self, env) -> "TradeValueBreakdown":
        """Per-leg PV attribution for reporting."""
        result = self.engine.price_with_events(self.product, env)
        notional = self.get_actual_notional(env)
        return TradeValueBreakdown(
            product_npv=result.npv * self.quantity,
            leg_pvs={leg.leg_id: LegPV(name=leg.name,
                                       direction=leg.direction,
                                       pv=leg.value(result.event_distribution, env, notional)
                                          * self.quantity)
                     for leg in self.cash_legs},
        )

    def get_trade_greeks(self, env, calc: GreeksCalculator) -> dict[str, float]:
        """Trade-level Greeks (product + legs) via finite-difference bump on get_trade_value."""
        ...
```

```python
@dataclass(frozen=True)
class LegPV:
    name: Optional[str]
    direction: LegDirection
    pv: float

@dataclass(frozen=True)
class TradeValueBreakdown:
    product_npv: float
    leg_pvs: dict[str, LegPV]            # keyed by leg_id

    @property
    def total(self) -> float:
        return self.product_npv + sum(v.pv for v in self.leg_pvs.values())
```

### 7.2 Multiple legs of the same type

The `cash_legs: list[CashLeg]` field naturally supports any number of legs of any combination of types. Example:

```python
position = EquityPosition(
    product=snowball, quantity=1, engine=quad_engine, ...,
    cash_legs=[
        DeterministicLeg(amount=1_500_000, payment_time=0.0,
                         direction=LegDirection.BUYER_PAYS,
                         name="Front Premium"),
        DeterministicLeg(amount=200_000, payment_time=1.0,
                         direction=LegDirection.BUYER_RECEIVES,
                         name="Backend Rebate"),
        AccrualLeg(rate=0.02, base=NotionalBase(...), schedule=monthly_schedule,
                   direction=LegDirection.BUYER_RECEIVES,
                   name="Margin Interest"),
        AccrualLeg(rate=0.005, base=NotionalBase(...), schedule=quarterly_schedule,
                   direction=LegDirection.BUYER_PAYS,
                   name="Funding Cost"),
        FixedPayoffLeg(amount=50_000, trigger=PaymentTrigger.AT_KO,
                       direction=LegDirection.BUYER_RECEIVES,
                       name="KO Bonus"),
    ],
)
```

The framework does not deduplicate, collapse, or warn on duplicates — multiple legs of the same type are valid and common (e.g., two interest legs from different counterparties). Each leg's contribution is preserved separately via `leg_id` in `get_trade_value_breakdown`.

### 7.3 Greeks (v1: bump-and-reprice)

```python
def get_trade_greeks(self, env, calc) -> dict[str, float]:
    bumps = calc.get_bump_config()
    base = self.get_trade_value(env)
    return {
        'delta': (self.get_trade_value(env.bump_spot(+bumps.spot)) -
                  self.get_trade_value(env.bump_spot(-bumps.spot))) / (2 * bumps.spot),
        'gamma': (self.get_trade_value(env.bump_spot(+bumps.spot))
                  - 2 * base
                  + self.get_trade_value(env.bump_spot(-bumps.spot))) / (bumps.spot ** 2),
        'vega':  (self.get_trade_value(env.bump_vol(+bumps.vol)) -
                  self.get_trade_value(env.bump_vol(-bumps.vol))) / (2 * bumps.vol),
        'theta': (self.get_trade_value(env.bump_time(+bumps.time)) - base) / bumps.time,
        'rho':   (self.get_trade_value(env.bump_rate(+bumps.rate)) -
                  self.get_trade_value(env.bump_rate(-bumps.rate))) / (2 * bumps.rate),
    }
```

**Correctness rationale:** a bump in spot/vol/time/rate changes both the product NPV and the engine's emitted `EventDistribution`. Legs automatically re-value against the bumped distribution. No per-leg analytical work is needed for end-to-end correct Greeks.

**v2 optimization (deferred):** analytical leg Greeks computed from a single engine call plus closed-form derivatives w.r.t. `EventDistribution` entries. Implement only if profiling shows position-level Greeks are a bottleneck.

---

## 8. Backward Compatibility

| Scenario | Behavior |
|---|---|
| Existing `EquityPosition` with no `cash_legs` | Identical to today. Same constructor, same `get_market_value`, same `get_greeks`. |
| Existing engine call sites (`engine.price(product, env)`) | Unchanged. `price_with_events` is additive. |
| Existing backtest / stress test / dynamic scenario modules | Unchanged. They opt in to `get_trade_value` / `get_trade_greeks` in separate, isolated PRs. |
| Existing tests in `test/` | All continue to pass. New tests live under `test/test_cashleg/`. |
| Existing `portfolio/fi/position.py` | Unchanged in v1. Same pattern applied later as separate work item. |

---

## 9. Validation & Error Handling

### 9.1 Construction-time validation

- `LegSchedule.period_ends[-1] ≤ product.get_maturity()` — leg schedule cannot extend past product maturity. Raises `ValidationError`.
- `BaseAmount.value ≥ 0` when `mode == ABSOLUTE`; `value ∈ [0, 1]` when `mode ∈ {NOTIONAL_FRACTION, MARGIN_FRACTION}`. Negative legs are represented by `direction=BUYER_PAYS`, not by negative amounts.
- `FixedPayoffLeg.payment_offset_days ≥ 0`.
- `AccrualLeg.rate` may be negative (funding cost paid by counterparty modeled with positive direction).
- `LegSchedule` array lengths must match across `period_starts`, `period_ends`, `payment_times`.

### 9.2 Runtime validation

- `EventDistribution.__post_init__` enforces probability-sum and survival-monotonicity invariants. Failures raise `NumericalError`.
- `AccrualLeg.value` checks that `event_dist.survival_probability` is non-trivial when `ko_behavior=TRUNCATE_AT_KO`. If trivial (i.e., engine did not override `price_with_events`), the valuator emits a warning via the project logger and proceeds (the result equals `PAY_FULL_SCHEDULE`); strict mode (env flag) escalates to `ValidationError`.
- `FixedPayoffLeg.value` raises `ValidationError` if `event_dist.probabilities` does not contain the requested `trigger` event type.

### 9.3 Numerical tolerances

All comparisons and probability sums use `util/numerical/comparison.py` utilities. No hardcoded tolerances — `Tolerance.PROBABILITY` (1e-6) for probability sum invariants.

---

## 10. Testing Strategy

```
test/test_cashleg/
├── test_deterministic_leg.py      # PV = amount × DF; sign conventions; payment timing
├── test_accrual_leg.py            # vs. closed-form annuity (no KO); vs. snowball coupon (with KO)
├── test_fixed_payoff_leg.py       # vs. P(touch) × amount on barrier products
├── test_base_amount.py            # ABSOLUTE / NOTIONAL_FRACTION / MARGIN_FRACTION resolution
├── test_event_distribution.py     # invariants (sum=1, monotone survival), trivial constructor
├── test_position_with_legs.py     # backward-compat (no legs); get_trade_value vs get_market_value
└── test_engine_event_emission.py  # each engine type emits valid EventDistribution
```

### 10.1 Key correctness tests

| Test | Method |
|---|---|
| **DeterministicLeg PV** | Front premium → exactly `amount × 1.0`; backend → `amount × exp(-rT)` |
| **AccrualLeg matches snowball KO coupon** | Build an AccrualLeg replicating snowball's `coupon_rate` semantics; price (snowball with `coupon_rate=0`) + (AccrualLeg) and assert equals snowball with `coupon_rate=X` and no leg, within MC tolerance |
| **AccrualLeg on European option** | Survival flat at 1; PV reduces to `rate × base × T × DF(T)` (single-period closed form) |
| **FixedPayoffLeg at KO matches one-touch** | `FixedPayoffLeg(amount=1, trigger=AT_KO)` on a single-barrier product should equal the one-touch price |
| **Engine cross-consistency** | Same product + same leg priced via MC / PDE / Quad: leg PVs agree within engine convergence tolerance |
| **Sign conventions** | Long position, `BUYER_PAYS` premium → negative contribution; flip both → positive |
| **Trade Greeks: leg-only position** | Position with only a `DeterministicLeg` → delta=0, gamma=0, vega=0, theta=∂(amount × DF)/∂t |
| **Trade Greeks: vanilla + premium** | `delta_trade == delta_product` (premium at t=0 has no spot sensitivity) |
| **Trade Greeks: snowball + accrual leg** | delta/vega differ from product-only by leg's KO-sensitivity; signs match economic intuition |
| **Multiple legs of same type** | Two `DeterministicLeg`s with opposite directions and equal amounts net to zero in `get_trade_value`, but appear separately in `get_trade_value_breakdown` |
| **Backward compatibility** | Existing test suite passes unchanged; `EquityPosition(...)` without `cash_legs` produces identical `get_market_value` / `get_greeks` to today |

---

## 11. Rollout Plan

Each phase is independently shippable.

| Phase | Deliverable | Scope |
|---|---|---|
| **1 — Core primitives** | `cashleg/` module: `EventDistribution`, `CashLeg` ABC, three leg types, `BaseAmount`, `LegSchedule`, `leg_valuator`. `BaseEngine.price_with_events` default. Tests for legs in isolation against trivial event_dist. | No engine changes yet. `DeterministicLeg`s work everywhere. |
| **2 — MC engine emission** | Override `price_with_events` on all equity MC engines (Snowball, Phoenix, KO-Reset, Range Accrual, Barrier MC, European MC, etc.). Tests for engine-emitted event_dist; AccrualLeg + KO product cross-checks. | Unlocks full feature on MC, the most common engine for autocallables. |
| **3 — Quad engine emission** | Override on Snowball/Phoenix/KO-Reset Quad engines. Cross-validation tests against MC engine results from phase 2. | Zero-cost emission; ideal for production cash-leg pricing. |
| **4 — Analytical engine emission** | Override on barrier, one-touch, range-accrual analytical engines using closed-form first-passage. | Analytical-speed pricing for vanilla-ish products with cash legs. |
| **5 — PDE engine emission** | Forward density pass added to Phoenix / Snowball / KO-Reset / Barrier PDE solvers. Flag-gated to avoid 20% cost when no accrual legs. | Completes engine coverage. |
| **6 — Position integration** | `EquityPosition.cash_legs`, `get_trade_value`, `get_trade_value_breakdown`, `get_trade_greeks`. Backward-compat tests. | Makes the feature usable from portfolio. |
| **7 — Downstream wiring** (separate PRs, out of scope here) | Backtest, stress test, dynamic scenario opt in to `get_trade_value`. Each module adopts independently. | Future work. |

---

## 12. Open Questions

None blocking implementation. Items to revisit during phase 5 (PDE) and phase 6 (Position):

- For PDE engines, whether to share the forward-density solver code across all PDE engine subclasses via a mixin or via a free function on the grid representation. Resolve during implementation based on grid-code commonality.
- Whether `get_trade_greeks` should reuse a single engine call across all bumps via vectorized bump (parallel bumps with cached grid state). Defer to v2 unless trade-level Greeks become a profiling hotspot.
