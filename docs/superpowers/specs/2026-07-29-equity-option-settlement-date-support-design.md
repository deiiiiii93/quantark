# Equity Option Settlement-Date Support - Design

**Date:** 2026-07-29
**Status:** Approved
**Area:** `quantark/asset/equity/`, `quantark/priceenv/`,
`quantark/execution/`, `quantark/portfolio/equity/`
**Scope:** All cash-settled equity option products and every registered pricing
engine that supports them

## Summary

Add settlement-date-aware payoff discounting to the complete equity option
pricing stack.

The product's stochastic exposure ends when a payoff is determined: at
exercise, fixing, barrier hit, coupon observation, knock-out, or maturity.
Cash may be paid later. Engines must therefore use two separate clocks:

- the **determination clock** controls underlying dynamics, volatility,
  monitoring, averaging, and optimal exercise;
- the **payment clock** controls cash discounting.

Settlement behavior is centralized in one shared resolver. Products describe
contractual settlement terms; engines consume normalized determination and
payment timing. The same resolver is used by analytical, MC, PDE, and QUAD
engines, lifecycle trackers, event statistics, and Greeks.

The design covers:

- explicit terminal `settlement_date`;
- per-event `settlement_date` and `settlement_time`;
- derived payment dates from settlement lag, calendar, and business-day
  adjustment;
- date-based and numeric-time products without fabricating calendar dates;
- terminal, hit-paid, expiry-paid, coupon, knock-out, accumulator, and
  early-exercise cashflows;
- live, determined-but-unpaid, and paid lifecycle states;
- PV, native and numerical Greeks, expected cashflow attribution, and
  event statistics;
- BSM, Local Vol, Heston, and SLV model families across their supported
  analytical, MC, PDE, and QUAD engines.

When no settlement terms are supplied, payment occurs at determination. This
zero-lag rule preserves current prices and public behavior.

## Goals

1. Make payment timing an explicit, shared valuation contract rather than an
   engine-specific convention.
2. Keep determination and payment times separate throughout every engine.
3. Discount every cashflow with the rate curve at its own payment time.
4. Preserve exact zero-lag behavior for all existing products and tests.
5. Support multiple payment dates within one product.
6. Change the American exercise obstacle to reflect delayed cash receipt.
7. Support valuation after payoff determination and before payment.
8. Make PV, Greeks, event statistics, and lifecycle accounting reconcile.
9. Fail closed when a requested settlement convention cannot be represented
   correctly.
10. Keep the implementation centralized and avoid parallel settlement-aware
    engine classes.

## Non-goals

- Physical delivery of stock against strike cash.
- Settlement currency conversion, quanto settlement, or cross-currency
  discounting.
- Stochastic interest rates or correlation between rates and the payoff.
- Credit or funding valuation adjustments during the settlement delay.
- Legal close-out, failed settlement, or operational payment rescheduling.
- Automatic market defaults such as universal T+2 or product-family T+N.
- Fabricating dates from numeric maturities.
- Replacing product-specific observation or payoff rules.
- Adding new pricing model families.

## Normative decisions

| Topic | Decision |
|---|---|
| Product scope | Every cash-settled equity option and every supported engine |
| Default | Zero lag: pay on determination |
| Architecture | One central settlement and cashflow timing kernel |
| Explicit vs derived | Explicit cashflow payment terms override derived terms |
| Terminal date | Existing product `settlement_date` is the terminal-payment override |
| Event settlement | Per-record date/time first; otherwise derive from the product convention |
| Numeric products | Support explicit `settlement_time` or numeric lag; never infer a date |
| Dynamics | Stop at determination; do not evolve spot or volatility through settlement delay |
| Discounting | Curve-exact discount factors at payment time |
| Lifecycle | Pending realized cashflows live in shared lifecycle state, not products or market data |
| Post-determination valuation | Value known pending cash; fail if the required fixing/payoff is absent |
| Payment date | On the payment date, the derivative cashflow is paid and its option PV is zero |
| Outputs | PV, Greeks, event statistics, and cashflow attribution must agree |
| Unsupported formulas | Raise a capability error; never silently ignore settlement |

## Terminology

### Determination

The event at which a cashflow amount becomes known, or at which a contingent
claim is exercised:

- expiry fixing for a European option;
- averaging completion for an Asian option;
- an American exercise decision;
- barrier hit for a hit-paid rebate;
- observation for a digital coupon or knock-out redemption;
- maturity for a surviving autocallable.

### Payment

The contractual date or time at which determined cash is transferred.
Payment may equal determination.

### Settlement delay factor

For deterministic rate curves, the value at determination of one unit paid at
payment is:

\[
A(T_d,T_p)
=
\frac{DF(0,T_p)}{DF(0,T_d)}
\]

where \(T_d\) is determination time and \(T_p\) is payment time.

### Terminal cashflow

A cashflow determined at the product's final fixing, exercise, or maturity.
Only terminal cashflows use the product-level `settlement_date` override.

### Event cashflow

A coupon, knock-out redemption, rebate, or other cashflow determined at an
intermediate observation or hit. Event cashflows use record-level payment
terms or the product settlement convention. They do not inherit the product's
terminal `settlement_date`.

## Current state

The repository already contains most of the product-side seams:

- `BaseEquityOption` owns `exercise_date`, `settlement_date`,
  `maturity_date`, and shared lifecycle normalization.
- Concrete equity option constructors are being normalized to forward those
  shared lifecycle fields.
- `ObservationRecord` already supports `observation_date`,
  `observation_time`, `settlement_date`, and resolves to
  `ResolvedObservationRecord.settlement_time`.
- Barrier-like products distinguish hit-paid and expiry-paid rebates through
  fields such as `pay_at_hit` or `payment_at_hit`.
- Autocallables distinguish instant and expiry payment through
  `CouponPayType`.
- DCN schedules already carry explicit payment dates independently for
  coupon, knock-out, and loss cashflows.
- Shared equity lifecycle trackers exist for autocallables and barrier
  products.
- MC, PDE, and QUAD event-stat paths already expose discounted expected
  cashflow concepts for some structured products.
- `PricingEnvironment` exposes curve discount factors and an optional
  business calendar.

The missing contract is consistency. Most engines still use exercise or
maturity as both determination and payment. Some structured-product paths
already use per-observation settlement times, but the behavior is not shared
across product and engine families. Lifecycle state aggregates realized cash
without a general pending-receivable ledger.

This change completes and centralizes those existing seams. It must not add
another product-date abstraction or duplicate discounting rules inside each
engine.

## Architecture

### Package placement

Add one shared module:

```text
quantark/asset/equity/settlement.py
```

It contains:

- `SettlementLagUnit`;
- `SettlementConvention`;
- `CashflowKind`;
- `SettlementRequest`;
- `ResolvedPaymentTiming`;
- `SettlementResolver`;
- shared settlement validation.

The module may import calendar utilities and use `PricingEnvironment` under
`TYPE_CHECKING`; it must not import pricing engines. Products, engines,
lifecycle trackers, execution, and portfolio code may all depend on it
without a cycle.

Extend the existing lifecycle package:

```text
quantark/asset/equity/lifecycle/
├── cashflows.py       # immutable realized cashflow + ledger
├── state.py           # states compose the ledger
└── ...
```

### Settlement convention

```python
class SettlementLagUnit(Enum):
    BUSINESS_DAYS = auto()
    CALENDAR_DAYS = auto()
    YEAR_FRACTION = auto()


@dataclass(frozen=True)
class SettlementConvention:
    lag: float = 0.0
    lag_unit: SettlementLagUnit = SettlementLagUnit.BUSINESS_DAYS
    business_day_convention: BusinessDayConvention = (
        BusinessDayConvention.FOLLOWING
    )
    calendar: Optional[Calendar] = None
```

Rules:

- `lag >= 0` and finite.
- Business-day and calendar-day lag must be integer-valued.
- Year-fraction lag may be fractional.
- A zero lag is an identity and needs no calendar.
- Business-day lag requires `convention.calendar` or
  `pricing_env.calendar`.
- Calendar-day lag adds calendar days, then applies the business-day
  convention when it is not `UNADJUSTED`.
- Business-day lag uses `Calendar.add_business_days`; the business-day
  adjustment is then a no-op in the normal case.
- Explicit payment dates are already contractual dates and are not adjusted
  again.
- The settlement calendar adjusts contractual dates. Conversion of resolved
  dates to rate-curve time uses the pricing environment's day-count basis,
  not `BaseEquityOption.annualization_day_count`.

The product-level convention is optional. Absence means zero lag.

### Product contract

`BaseEquityOption` gains:

```python
settlement_convention: Optional[SettlementConvention] = None
```

All concrete option constructors that expose shared lifecycle fields must
forward it. Product classes outside the base hierarchy, such as standalone
touch products, add the same field directly.

The meanings of existing fields remain:

- `exercise_date`: determination/expiry date;
- `settlement_date`: explicit terminal payment date;
- `maturity_date`: contract lifecycle metadata;
- `tenor_end`: product accrual endpoint selection;
- `annualization_day_count`: product accrual convention.

Settlement discounting must not reuse `tenor_end` or
`annualization_day_count`. Accrual and payment timing are related contractual
concepts but distinct valuation responsibilities.

### Settlement request

```python
class CashflowKind(Enum):
    TERMINAL = auto()
    EXERCISE = auto()
    HIT = auto()
    OBSERVATION = auto()
    COUPON = auto()
    REDEMPTION = auto()
    REBATE = auto()


@dataclass(frozen=True)
class SettlementRequest:
    kind: CashflowKind
    determination_date: Optional[datetime] = None
    determination_time: Optional[float] = None
    explicit_payment_date: Optional[datetime] = None
    explicit_payment_time: Optional[float] = None
```

Exactly one determination representation is required, unless both date and
time are supplied and resolve consistently under the pricing environment.
The same rule applies to explicit payment representations.

`CashflowKind.TERMINAL` is the only kind eligible for the product-level
`settlement_date` override. `EXERCISE` is deliberately separate: an American
exercise at an unknown future node settles from the actual exercise node
under the convention, not at the fixed terminal settlement date.

### Resolved payment timing

```python
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

Invariants:

- `determination_time >= 0` for live contingent valuation;
- `payment_time >= determination_time`;
- `determination_df = DF(0, determination_time)`;
- `payment_df = DF(0, payment_time)`;
- `delay_df = payment_df / determination_df`;
- all discount factors are finite and strictly positive.

Pending-cashflow valuation after determination uses a separate resolver path:
determination may be in the past, but payment must still be in the future.
The amount is already known and PV is `amount * DF(valuation, payment)`.

### Resolution precedence

For every cashflow:

1. Use `SettlementRequest.explicit_payment_date` or
   `explicit_payment_time` when supplied.
2. For `CashflowKind.TERMINAL` only, use the product's explicit
   `settlement_date`.
3. Apply the product's `settlement_convention` to determination.
4. If no settlement terms exist, set payment equal to determination.

No later rule may override an earlier one.

The resolver must expose two explicit entry points:

```python
SettlementResolver.resolve_contingent(product, request, pricing_env)
SettlementResolver.resolve_pending(realized_cashflow, pricing_env)
```

This prevents negative determination times from leaking into live-engine
code.

### No fabricated dates

The following is prohibited:

```python
exercise_date = valuation_date + timedelta(days=round(maturity * 365))
```

A business-day or calendar-day settlement rule requires a real determination
date. A numeric product must provide:

- an explicit `settlement_time`; or
- a `SettlementConvention` with `YEAR_FRACTION` lag.

If a date-based convention is requested for a time-only determination, raise
`ValidationError`.

## Pricing mathematics

### General deterministic-rate identity

For cashflow \(X_i\) determined at \(T_{d,i}\) and paid at \(T_{p,i}\):

\[
PV
=
\sum_i
\mathbb{E}^{Q}
\left[
DF(0,T_{p,i}) X_i
\right]
\]

The equity process and event probabilities are simulated or solved only to
\(T_{d,i}\). Under the deterministic-rate contract already used by the
equity engines:

\[
V(T_{d,i})
=
X_i
\frac{DF(0,T_{p,i})}{DF(0,T_{d,i})}
\]

This `delay_df` is the value injected into a PDE or QUAD surface at the
determination node.

### European terminal payoff

If an existing engine returns the correctly discounted expiry-settled value
`V_expiry`, delayed payment is:

\[
V_{\text{delayed}}
=
V_{\text{expiry}}
\frac{DF(0,T_p)}{DF(0,T_d)}
\]

The full expiry-settled value is scaled. Scaling only the strike component is
incorrect.

For BSM, \(d_1\), \(d_2\), forward carry, and volatility use \(T_d\). The
settlement delay does not change the distribution of \(S_{T_d}\).

The delayed European lower bound and put-call parity are the expiry-settled
relations multiplied by the same positive delay factor.

### Multiple cashflows

A product with cashflows at different payment times must never use a blanket
terminal multiplier:

\[
PV
=
\sum_i DF(0,T_{p,i}) \mathbb{E}^{Q}[X_i]
\]

This applies to accumulators, range accruals, Phoenix coupons, knock-out
redemptions, DCN legs, and event-paid rebates.

### American exercise

At candidate exercise time \(t\), the exercise obstacle becomes:

\[
E(S,t)
=
\text{intrinsic}(S)
\frac{DF(0,T_p(t))}{DF(0,t)}
\]

The engine compares this delayed cash value with continuation:

\[
V(S,t)=\max(C(S,t),E(S,t))
\]

Delayed settlement can change the optimal exercise boundary. Applying one
terminal scaling factor after solving the immediate-settlement problem is
incorrect.

For date-based settlement, every candidate exercise node must have an
authoritative date. A date-based American PDE/LSMC grid must therefore carry
node dates. A numeric-time American product may use only numeric settlement
lag.

### Hit-paid and expiry-paid cashflows

- `pay_at_hit=True`, `payment_at_hit=True`, or
  `CouponPayType.INSTANT` means determination is the hit/observation node and
  settlement convention is applied from that node.
- `pay_at_hit=False`, `payment_at_hit=False`, or
  `CouponPayType.EXPIRY` means the cashflow uses terminal payment resolution.
- "Instant" means zero contractual deferral before applying the shared
  convention. With a T+N convention it pays N settlement days after the
  event, not literally at the event timestamp.

## Engine integration

### Shared rule

Every engine must resolve all payment timing before its expensive numerical
loop. Resolution must not occur once per Monte Carlo path or once per spatial
node.

All registered engines must do one of the following:

1. price the requested settlement terms correctly;
2. raise a precise capability error before numerical work begins.

No engine may accept non-zero settlement terms and silently price at
determination.

### Analytical engines

#### Terminal-only claims

BSM, European digital, Heston European, terminal Asian approximations, and
other terminal-only formulas:

1. calculate the existing determination-date value;
2. multiply by `delay_df`;
3. apply contract multiplier exactly once;
4. update native Greeks and sanity bounds consistently.

#### Mixed-event formulas

Analytical barrier, touch, double-barrier, and sharkfin formulas must split
terminal and hit-paid legs when their payment dates differ.

An analytical first-hit formula may support non-zero hit settlement only when
the delay factor is representable by that formula. For example, a constant
numeric lag under the formula's flat-rate assumptions may admit exact
scaling. A business-day lag under a term curve generally makes the delay
factor hit-time-dependent.

When exact representation is unavailable, the analytical engine raises
`EngineCapabilityError` directing the caller to MC, PDE, or QUAD. It must not
use an average hit date or a terminal scaling approximation.

#### American analytical approximations

BAW/BS93-style approximations assume an immediate exercise payoff and cannot
in general reproduce a date-dependent delayed exercise obstacle. They may
support only settlement forms proven compatible with their derivation.
Otherwise they fail capability checking and direct the caller to LSMC or PDE.

### Monte Carlo engines

MC engines:

1. build determination and payment arrays once;
2. simulate spot only through determination nodes;
3. compute pathwise cashflow amounts at their determination nodes;
4. multiply each pathwise cashflow by its own `payment_df`;
5. aggregate discounted and undiscounted cashflows separately.

There is no single terminal discount for a multi-cashflow payoff.

LSMC uses curve-exact discounting between regression nodes. At every exercise
node, the immediate exercise value uses that node's `delay_df`. The exercise
policy, exercise probabilities, and event statistics therefore reflect
settlement lag.

BSM, Local Vol, Heston, and SLV path generators share the same payoff and
settlement arrays. Settlement logic must not be copied into each model
variant.

### PDE engines

PDE stochastic grids end at the latest determination time, not the latest
payment time.

- Terminal condition: multiply the terminal payoff by terminal `delay_df`.
- Coupon/event injection: multiply event amount by event `delay_df`.
- Knock-out overwrite: use redemption value at the KO observation node,
  discounted from that node to its payment.
- Expiry-paid rebate: use terminal payment timing.
- American obstacle: use node-specific delayed exercise value.

Term-structured rates use exact curve ratios. Do not approximate the delay as
`exp(-get_rate(T_d) * (T_p - T_d))`.

Model-specific PDE cores - BSM, Local Vol, Heston, and SLV - consume the same
resolved event values. Settlement changes boundary/event values, not the
underlying model coefficients before determination.

### QUAD engines

QUAD transition intervals remain determination-to-determination intervals.
At each observation:

- cashflow injection uses its event `delay_df`;
- terminal payoff uses terminal `delay_df`;
- expected cashflow attribution stores both undiscounted and discounted
  values.

`quad_adapters` already has record-level settlement seams. The shared resolver
replaces local fallback rules so European, barrier adapters, Snowball,
Phoenix, and KO-reset engines agree.

### Engine/model coverage matrix

The settlement contract applies to every existing supported cell, not only
one reference engine:

| Model family | Analytical | MC | PDE | QUAD |
|---|---:|---:|---:|---:|
| BSM / constant vol | Required | Required | Required | Required |
| Local Vol | n/a unless an approximation already exists | Required | Required | Required where currently supported |
| Heston | Required for existing European formula | Required | Required | Required where currently supported |
| SLV | n/a | Required | Required | Required where currently supported |

"Required where currently supported" does not create a new product/model
capability. It means every existing engine/product pairing must honor
settlement.

## Product-family semantics

| Product family | Determination | Payment resolution |
|---|---|---|
| European vanilla | Expiry fixing | terminal explicit date, convention, or zero lag |
| Digital | Expiry fixing | terminal resolution |
| Asian | Final averaging/fixing | terminal resolution |
| Range accrual | Each paid accrual or final aggregate, according to existing contract | per-record explicit timing, convention, or terminal resolution |
| Accumulator | Each contractual settlement leg; KO rebate at its contractual event | per-record timing or convention |
| American | Actual exercise node | exercise node plus convention; never fixed terminal date unless exercise occurs at terminal |
| Barrier / double barrier | Terminal payoff at expiry; hit rebate at hit when configured | split terminal and event resolution |
| One-touch / double one-touch | Hit for touch payment; expiry for no-touch or expiry-paid amount | event or terminal resolution |
| Sharkfin | Barrier rebate event and surviving terminal payoff | resolve legs separately |
| Snowball / KO reset | KO observation or maturity | per-record timing; terminal resolution for survival |
| Phoenix | Coupon observation, KO observation, or maturity | each coupon/KO record separately; terminal resolution for maturity |
| DCN | Existing coupon, KO, and loss determination dates | preserve explicit schedule payment dates |

If a product currently aggregates economically distinct cashflows before
discounting, its payoff result must be decomposed before settlement support is
considered complete.

## Lifecycle and pending receivables

### Immutable realized cashflow

```python
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
```

The cashflow ID is stable for the economic event, for example:

```text
<trade-id>:coupon:<observation-index>
<trade-id>:ko:<observation-index>
<trade-id>:terminal
<trade-id>:exercise:<exercise-date>
```

Registering the same ID and identical payload is idempotent. Registering the
same ID with a conflicting amount or payment date raises `ValidationError`.

Cashflow status is derived, not mutated:

- pending when valuation is before payment;
- paid when valuation is on or after payment.

This keeps the ledger append-only.

### Cashflow ledger

```python
class LifecycleCashflowLedger:
    def register(self, cashflow: RealizedCashflow) -> bool: ...
    def pending(self, as_of: ValuationPoint) -> tuple[RealizedCashflow, ...]: ...
    def paid(self, as_of: ValuationPoint) -> tuple[RealizedCashflow, ...]: ...
    def pending_pv(
        self,
        as_of: ValuationPoint,
        pricing_env: PricingEnvironment,
    ) -> float: ...
    def paid_total(self, as_of: ValuationPoint) -> float: ...
```

`ValuationPoint` carries either an authoritative date or a numeric time
origin. Numeric times decay through the existing `time_shift` machinery; they
are not converted into dates.

### Shared lifecycle states

`AutocallableLifecycleState` and `BarrierLifecycleState` compose the ledger.
A small shared lifecycle protocol exposes:

- current valuation point;
- product alive/determined flags;
- product-specific path state such as KI/KO;
- realized cashflow ledger.

The legacy `realized_cashflows` surface remains temporarily as a derived
compatibility property for paid cash only. It must not include pending
receivables. Backtest, scenario, and portfolio consumers migrate to:

- option PV for remaining contingent value;
- ledger pending PV for determined-but-unpaid cash;
- ledger paid total for cash-account value.

### Three valuation states

#### Live

Price all remaining contingent cashflows plus any earlier pending cashflows.
A Phoenix may simultaneously have a live continuation value and pending
coupons.

#### Determined but unpaid

The determined cashflow has no delta, gamma, vega, or dividend risk. Its value
is:

\[
PV_{\text{pending}}
=
\text{known amount}
\times
DF(\text{valuation},\text{payment})
\]

It retains rate and calendar theta risk.

If the fixing, exercise result, or realized amount is missing, fail closed.
Never use current spot as a historical fixing.

#### Paid

On the payment date, the derivative value of that cashflow is zero. Portfolio
cash accounting receives the payment. The option engine must not retain the
cashflow and the cash account simultaneously.

### Engine API propagation

Add an optional lifecycle keyword to the common public valuation path:

```python
price(
    product,
    pricing_env,
    *,
    lifecycle_state: Optional[EquityOptionLifecycleState] = None,
) -> float
```

`price_with_events`, execution adapters, positions, portfolio valuation,
Greek calculators, and scenario/backtest repricers propagate the same state.

Before determination, lifecycle state remains optional. On or after a known
determination event, state or authoritative fixing data is required.

## Greeks and risk

### Native analytical Greeks

For a terminal claim whose expiry-settled value is \(V_d\):

\[
V_p = A(T_d,T_p)V_d
\]

When the settlement factor is independent of spot and volatility:

- delta, gamma, vega, and dividend sensitivity scale by \(A\);
- theta must differentiate/revalue both clocks;
- rho must include curve sensitivity of both the determination-date formula
  and the delay factor.

Native formulas must be tested against bump-and-reprice results. It is not
sufficient to scale price and leave the existing Greek methods unchanged.

### Numerical Greeks

All numerical Greeks reprice through the settlement resolver.

- Spot and vol bumps preserve payment terms.
- Curve bumps recompute `determination_df`, `payment_df`, and `delay_df`.
- Calendar theta advances valuation date but keeps absolute contractual
  determination and payment dates fixed.
- Numeric theta shifts both determination and payment times by the same
  elapsed time; it does not change the contractual lag.
- Theta crossing determination requires authoritative fixing/lifecycle state.
- Theta crossing payment transfers value from derivative PV to paid cash at
  the portfolio layer.

Pending cashflows have zero spot/vol risk and non-zero rate/theta risk until
payment.

### Event statistics

Event result objects expose aligned arrays:

```python
determination_times
payment_times
expected_undiscounted_cashflows
expected_discounted_cashflows
```

Date-based results may additionally expose determination/payment dates.

Required invariants:

\[
PV
=
\sum_i \text{expected discounted cashflow}_i
\]

subject only to clearly named legs excluded from a particular result type.

Event probabilities and expected life use determination/observation times,
not payment times. Cashflow PV attribution uses payment times.

## Validation and failure behavior

Raise `ValidationError`, `PricingError`, or `EngineCapabilityError` as
appropriate for:

- negative or non-finite lag;
- non-integral business-day or calendar-day lag;
- payment before determination;
- missing calendar for a business-day rule;
- insufficient holiday-calendar coverage;
- date-based settlement requested for a time-only determination;
- both date/time supplied but inconsistent;
- terminal `settlement_date` before terminal determination;
- record-level settlement before its observation;
- non-finite or non-positive discount factor;
- realized event without authoritative payoff amount;
- missing historical fixing after determination;
- duplicate cashflow ID with conflicting payload;
- analytical formula that cannot represent the requested delay;
- an engine/product path that has not declared settlement capability.

Error messages must identify:

- product type;
- engine type;
- cashflow/event identifier;
- determination timing;
- requested settlement terms;
- exact unsupported or invalid condition.

No error path may fall back to:

- exercise-date discounting;
- maturity-date discounting;
- current spot as historical fixing;
- weekend-only calendar behavior outside known calendar coverage;
- an average payment date;
- flat-rate `exp(-r * delay)` when curve DFs are available.

## Backward compatibility

### Price identity

With no explicit payment terms and no settlement convention:

- payment equals determination;
- `delay_df == 1`;
- all prices, Greeks, event statistics, and exercise boundaries remain
  unchanged.

This identity is the compatibility mechanism. Do not add a feature flag.

### API compatibility

- Existing `settlement_date` remains accepted.
- Existing observation `settlement_date` and `settlement_time` remain
  accepted.
- New `settlement_convention` is optional and keyword-only where constructor
  compatibility requires.
- Existing result fields remain during a deprecation window; new timing and
  cashflow fields are additive.
- Existing engine classes and enums remain. No `SettlementAware*Engine`
  variants are introduced.

### Lifecycle compatibility

Existing lifecycle booleans and event types remain. Consumers of aggregated
`realized_cashflows` migrate to the ledger interfaces before non-zero
settlement is enabled in those workflows.

## Testing strategy

### Resolver unit tests

1. Zero-lag identity for date and time representations.
2. Explicit record date/time overrides every fallback.
3. Terminal product `settlement_date` applies only to terminal cashflow.
4. Event cashflow ignores terminal `settlement_date`.
5. Convention derives business-day payment across weekends and holidays.
6. Calendar-day lag plus each business-day adjustment convention.
7. Year-fraction lag for numeric products.
8. Business-day convention on a time-only event fails.
9. Payment before determination fails.
10. Inconsistent date/time pair fails.
11. Calendar coverage failure is loud.
12. Curve-exact `delay_df` on flat and non-flat curves.

### Exact single-cashflow benchmarks

For European call, put, and digital:

- compare explicit delayed-settlement formula against analytical engine;
- verify put-call parity with delayed payment;
- verify delayed lower bound;
- test flat, upward-sloping, downward-sloping, and kinked rate curves;
- verify determination-time volatility and forward inputs are unchanged;
- verify zero-rate and zero-delay identities.

Run the same terminal-payoff contract through:

- BSM analytical, MC, PDE, and QUAD;
- Local Vol MC/PDE and QUAD where supported;
- Heston analytical/MC/PDE and QUAD where supported;
- SLV MC/PDE and QUAD where supported.

MC comparisons use confidence intervals; PDE/QUAD use convergence tolerances.

### Multi-cashflow benchmarks

1. Two deterministic cashflows with different payment dates reproduce the
   exact sum of curve DFs.
2. Accumulator legs use individual settlement times.
3. Barrier hit rebate and surviving terminal payoff use different timing.
4. One-touch hit-paid vs expiry-paid values differ by the expected delay.
5. Snowball KO redemption uses observation-specific payment time.
6. Phoenix coupons, KO redemption, and terminal payoff reconcile separately.
7. DCN retains its existing explicit coupon/KO/loss payment dates.
8. MC, PDE, and QUAD expected discounted cashflow vectors sum to their PVs.

### American benchmarks

1. Zero-lag American price and boundary remain unchanged.
2. Delayed-settlement PDE obstacle uses node-specific delay.
3. LSMC and PDE agree within numerical tolerance under the same convention.
4. Delayed settlement changes the early-exercise boundary in the expected
   direction on controlled put/dividend cases.
5. Post-exercise pending value is known cash only.
6. Unsupported analytical approximation raises capability error.
7. Date-based lag on a numeric-only exercise grid fails rather than inventing
   dates.

### Lifecycle benchmarks

Test the full timeline:

1. before determination: live option PV;
2. at determination with authoritative fixing: contingent value becomes a
   realized pending cashflow;
3. between determination and payment: pending PV only for terminated claim;
4. on payment: derivative PV becomes zero and paid cash increases once;
5. after payment: no derivative or pending value;
6. repeated event processing: no duplicate cash;
7. missing fixing: fail closed;
8. Phoenix: pending coupon coexists with live continuation;
9. KO: live product terminates but redemption remains pending;
10. portfolio total value remains continuous across the derivative-to-cash
    transfer, apart from market moves.

### Greeks

- Native vs numerical delta, gamma, vega, theta, rho, and dividend rho.
- Pending receivable delta/gamma/vega equal zero.
- Pending receivable rho and theta match curve bump/revaluation.
- Theta before, at, between, and on determination/payment boundaries.
- Bucketed rate risk includes the settlement-delay portion of the curve.
- Cash Greeks and position scaling apply contract multiplier and quantity
  exactly once.

### Regression and performance

- Entire existing suite passes unchanged under zero lag.
- Dedicated golden tests for representative analytical, MC, PDE, and QUAD
  products.
- Resolver work is outside MC path and PDE/QUAD node loops.
- Representative zero-lag MC/PDE/QUAD runtime regression is no more than 5%
  after noise-controlled benchmarking.
- Memory overhead for event arrays is linear in number of contractual
  cashflows, not paths times cashflows beyond arrays already required by MC.

## Implementation sequence

The feature is one contract delivered in independently gated phases.

| Phase | Content | Required gate |
|---|---|---|
| 0 | Shared settlement types/resolver; product constructor propagation; observation resolver integration; lifecycle cashflow ledger | Resolver tests; constructor-contract tests; zero-lag identity |
| 1 | Terminal single-cashflow engines and native Greeks across BSM/LV/Heston/SLV supported analytical/MC/PDE/QUAD cells | Exact European/digital benchmarks; cross-engine matrix |
| 2 | Barrier/touch/sharkfin, accumulator/range accrual, Snowball/KO-reset/Phoenix/DCN multi-cashflow paths and event stats | Mixed-cashflow reconciliation; MC/PDE/QUAD agreement |
| 3 | American LSMC/PDE obstacle; analytical capability checks; pending-receivable valuation | American boundary tests; full lifecycle timeline |
| 4 | Execution, position, portfolio, scenario, backtest, and risk propagation; compatibility cleanup | Full suite; portfolio cash continuity; performance gates |

Each phase must update all affected outputs - price, Greeks, and event
statistics - for the product/engine slice it enables. A phase is not complete
if price uses settlement but its native risk or cashflow attribution does not.

## Acceptance criteria

The feature is complete only when:

1. Zero settlement lag reproduces every existing supported engine result.
2. A delayed European payoff matches the curve-exact analytical identity.
3. Determination-time dynamics remain unchanged by settlement delay.
4. Every multi-cashflow product discounts each leg at its own payment time.
5. American delayed settlement changes the exercise obstacle, not merely the
   final reported PV.
6. Pending realized cash is valued through payment and never double-counted
   with paid cash.
7. Native Greeks agree with bump-and-reprice Greeks.
8. Event-stat cashflow PVs reconcile to engine PV.
9. The complete existing BSM/LV/Heston/SLV and MC/PDE/QUAD support matrix
   either prices correctly or rejects unsupported formulas explicitly.
10. No code path fabricates dates, silently ignores settlement, or uses a
    maturity-wide discount for mixed payment dates.

## Follow-on work

After this design is implemented, the same settlement kernel may be reused
for FX, rates, credit, and physical-settlement work. Those integrations
require separate designs because their currency, delivery, and discounting
contracts differ.
